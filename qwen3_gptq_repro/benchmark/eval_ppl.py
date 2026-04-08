#!/usr/bin/env python3
"""
WikiText-2 Perplexity Benchmark Script (single-model granularity)

Usage examples:
  # Evaluate FP16 baseline (no activation quantization)
  python benchmark/eval_ppl.py --model-dir /path/to/Qwen3-4B --label fp16_baseline

  # Evaluate with per-token INT8 activation quantization
  python benchmark/eval_ppl.py --model-dir /path/to/Qwen3-4B \
      --quant-weights output/gptq_from_raw/qwen3-4b-instruct-2507-gptq-4bit.pt \
      --act-quant int8 --label gptq_4bit_raw_A8

  # Evaluate with per-group INT4 activation quantization (group_size=128)
  python benchmark/eval_ppl.py --model-dir /path/to/Qwen3-4B \
      --act-quant int4-g128 --label fp16_A4g128

  # Mixed: A4-g128 globally, but A8 for down_proj layers
  python benchmark/eval_ppl.py --model-dir /path/to/Qwen3-4B \
      --act-quant int4-g128 --act-quant-override down_proj:int8 \
      --label fp16_A4g128_downA8
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# 0. Path resolution anchor
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)

# ---------------------------------------------------------------------------
# 1. CLI argument parsing
# ---------------------------------------------------------------------------

def _validate_act_quant(value: str) -> str:
    """Validate --act-quant argument: accepts 'none', 'int8', or 'int4-g<N>'."""
    if value in ("none", "int8"):
        return value
    m = re.fullmatch(r"int4-g(\d+)", value)
    if m:
        gs = int(m.group(1))
        if gs <= 0:
            raise argparse.ArgumentTypeError(f"group_size must be positive, got {gs}")
        return value
    raise argparse.ArgumentTypeError(
        f"Invalid act-quant format: '{value}'. "
        f"Accepted: 'none', 'int8', 'int4-g<N>' (e.g., 'int4-g128')"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="WikiText-2 Perplexity Benchmark (single-model granularity)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="HuggingFace model directory (original pretrained weights).",
    )
    parser.add_argument(
        "--quant-weights",
        type=str,
        default="",
        help="Path to quantized state_dict (.pt). If not specified, evaluate FP16 original model.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="unnamed",
        help="Label for this evaluation run (default: 'unnamed').",
    )
    parser.add_argument(
        "--seqlen",
        type=int,
        default=2048,
        help="Sliding window size (default: 2048).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(_PROJECT_DIR, "output", "benchmark"),
        help="Results output directory.",
    )
    parser.add_argument(
        "--local-wikitext2-dir",
        type=str,
        default="",
        help="Local WikiText-2 dataset directory (optional, for offline use).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model loading precision (default: float16).",
    )
    # --- Activation quantization arguments ---
    parser.add_argument(
        "--act-quant",
        type=_validate_act_quant,
        default="none",
        help=(
            "Activation quantization format. Accepted values: "
            "'none' (FP16 pass-through, default), "
            "'int8' (per-token INT8 symmetric), "
            "'int4-g<N>' (per-group INT4 symmetric, e.g., 'int4-g128')."
        ),
    )
    parser.add_argument(
        "--act-quant-override",
        type=str,
        default="",
        help=(
            "Per-layer activation quantization overrides. "
            "Comma-separated key:value pairs, e.g., 'down_proj:int8,o_proj:int8'. "
            "Matching layers use the override format instead of the global --act-quant."
        ),
    )
    return parser.parse_args()

# ---------------------------------------------------------------------------
# 2. Model loading
# ---------------------------------------------------------------------------

def dtype_from_str(name: str) -> torch.dtype:
    """Convert string to torch.dtype."""
    mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    return mapping.get(name, torch.float32)


def load_model(model_dir: str, quant_weights: str, dtype: torch.dtype):
    """
    Load model.

    - If quant_weights is empty, load FP16 original model directly.
    - If quant_weights is provided, load model structure first, then replace state_dict.
    """
    print(f"[load] Loading model structure: {model_dir}")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype)
    except TypeError:
        # Compatibility with older transformers versions
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype)

    if quant_weights:
        print(f"[load] Loading quantized weights: {quant_weights}")
        state_dict = torch.load(quant_weights, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"[warn] missing_keys: {len(missing_keys)}")
            for k in missing_keys[:5]:
                print(f"       - {k}")
            if len(missing_keys) > 5:
                print(f"       ... and {len(missing_keys) - 5} more")
        if unexpected_keys:
            print(f"[warn] unexpected_keys: {len(unexpected_keys)}")
            for k in unexpected_keys[:5]:
                print(f"       - {k}")
            if len(unexpected_keys) > 5:
                print(f"       ... and {len(unexpected_keys) - 5} more")
    else:
        print("[load] No quantized weights specified, using FP16 original model.")

    # Set seqlen, consistent with qwen3_gptq.py get_qwen3()
    max_seq = getattr(model.config, "max_position_embeddings", 2048)
    model.seqlen = min(2048, int(max_seq))

    return model

# ---------------------------------------------------------------------------
# 3. Multi-format activation fake quantization functions
# ---------------------------------------------------------------------------

def fake_quantize_activation_int8(x: torch.Tensor) -> torch.Tensor:
    """
    Per-token INT8 symmetric fake quantization.

    INT8 symmetric range: [-127, 127] (exclude -128 for symmetry).
    Per-token scale: scale = max(|x|, dim=-1) / 127
    """
    x_abs_max = x.abs().amax(dim=-1, keepdim=True)  # [..., 1]
    scale = x_abs_max / 127.0
    scale = scale.clamp(min=1e-10)

    x_q = (x / scale).round().clamp(-127, 127)
    x_dq = x_q * scale
    return x_dq


def fake_quantize_activation_int4_group(x: torch.Tensor, group_size: int) -> torch.Tensor:
    """
    Per-group INT4 symmetric fake quantization.

    INT4 symmetric range: [-7, 7] (4-bit signed, exclude -8 for symmetry).
    Each contiguous group of `group_size` elements along the hidden dimension
    gets its own scale: scale = max(|x_group|) / 7.

    If hidden_dim is not evenly divisible by group_size, the last group is
    zero-padded before quantization and unpadded after dequantization.
    """
    orig_shape = x.shape
    hidden_dim = orig_shape[-1]

    # Flatten to 2D: [N, hidden_dim]
    x_2d = x.reshape(-1, hidden_dim)
    N = x_2d.shape[0]

    # Pad if necessary
    remainder = hidden_dim % group_size
    if remainder != 0:
        pad_size = group_size - remainder
        x_2d = F.pad(x_2d, (0, pad_size), value=0.0)
        padded_hidden = x_2d.shape[-1]
    else:
        pad_size = 0
        padded_hidden = hidden_dim

    n_groups = padded_hidden // group_size

    # Reshape to [N, n_groups, group_size]
    x_grouped = x_2d.reshape(N, n_groups, group_size)

    # Per-group scale
    x_abs_max = x_grouped.abs().amax(dim=-1, keepdim=True)  # [N, n_groups, 1]
    scale = x_abs_max / 7.0
    scale = scale.clamp(min=1e-10)

    # Quantize and dequantize
    x_q = (x_grouped / scale).round().clamp(-7, 7)
    x_dq = x_q * scale

    # Reshape back and remove padding
    x_dq = x_dq.reshape(N, padded_hidden)
    if pad_size > 0:
        x_dq = x_dq[:, :hidden_dim]

    return x_dq.reshape(orig_shape)


def fake_quantize_activation(x: torch.Tensor, fmt: str, layer_name: str) -> torch.Tensor:
    """
    Dispatcher: apply fake quantization based on format string.

    Args:
        x: Input activation tensor, shape [..., hidden_dim]
        fmt: Format string — 'none', 'int8', or 'int4-g<N>'
        layer_name: Layer name for warning messages

    Returns:
        Fake-quantized activation tensor (same shape as input), or original x if fmt='none'
    """
    if fmt == "none":
        return x

    if fmt == "int8":
        x_dq = fake_quantize_activation_int8(x)
    else:
        # Parse int4-g<N>
        m = re.fullmatch(r"int4-g(\d+)", fmt)
        if m is None:
            print(f"[warn] Unknown act-quant format '{fmt}' for layer {layer_name}, skipping.")
            return x
        group_size = int(m.group(1))
        x_dq = fake_quantize_activation_int4_group(x, group_size)

    # NaN/Inf safety check
    if torch.isnan(x_dq).any() or torch.isinf(x_dq).any():
        print(f"[warn] NaN/Inf detected after activation quantization, "
              f"falling back to FP16 for layer: {layer_name}")
        return x

    return x_dq

# ---------------------------------------------------------------------------
# 4. Per-layer activation quantization override mechanism
# ---------------------------------------------------------------------------

def parse_act_quant_override(override_str: str) -> dict:
    """
    Parse comma-separated key:value override string.

    Example: "down_proj:int8,o_proj:int8" -> {"down_proj": "int8", "o_proj": "int8"}
    """
    if not override_str or not override_str.strip():
        return {}

    overrides = {}
    for pair in override_str.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            print(f"[warn] Invalid override format '{pair}', expected 'key:value'. Skipping.")
            continue
        key, value = pair.split(":", 1)
        key = key.strip()
        value = value.strip()
        # Validate the value
        try:
            _validate_act_quant(value)
        except argparse.ArgumentTypeError as e:
            print(f"[warn] Invalid override value for '{key}': {e}. Skipping.")
            continue
        overrides[key] = value

    return overrides


def resolve_act_quant_format(layer_name: str, global_fmt: str, overrides: dict) -> str:
    """
    Determine the activation quantization format for a given layer.

    If any override key is a substring of layer_name, return the override format.
    Otherwise return the global format.
    """
    for key, fmt in overrides.items():
        if key in layer_name:
            return fmt
    return global_fmt

# ---------------------------------------------------------------------------
# 5. Hook injection: multi-format activation quantization
# ---------------------------------------------------------------------------

def inject_activation_quant(model, global_fmt: str, overrides: dict) -> list:
    """
    Traverse all nn.Linear layers and register forward pre-hooks for activation quantization.

    Layers whose resolved format is 'none' are skipped entirely.
    After injection, prints a summary table showing each layer type and its assigned format.

    Args:
        model: PyTorch model
        global_fmt: Global activation quantization format string
        overrides: Per-layer override dict from parse_act_quant_override()

    Returns:
        handles: List of registered hook handles for later removal
    """
    handles = []
    n_injected = 0
    n_skipped = 0

    # Track format assignment per layer type for summary
    layer_type_formats = defaultdict(set)

    def _make_pre_hook(layer_name: str, fmt: str):
        """Create a pre-hook closure for the specified layer and format."""
        def _pre_hook(module, args):
            if len(args) == 0:
                return args
            x = args[0]
            x_dq = fake_quantize_activation(x, fmt, layer_name)
            return (x_dq,) + args[1:]
        return _pre_hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            fmt = resolve_act_quant_format(name, global_fmt, overrides)

            # Extract layer type suffix (e.g., "q_proj", "down_proj")
            parts = name.split(".")
            layer_type = parts[-1] if parts else name
            layer_type_formats[layer_type].add(fmt)

            if fmt == "none":
                n_skipped += 1
                continue

            h = module.register_forward_pre_hook(_make_pre_hook(name, fmt))
            handles.append(h)
            n_injected += 1

    # Print summary
    print(f"\n[act_quant] Activation quantization hook injection summary:")
    print(f"  Injected: {n_injected} layers | Skipped (none): {n_skipped} layers")
    print(f"  Global format: {global_fmt}")
    if overrides:
        print(f"  Overrides: {overrides}")
    print(f"  {'Layer Type':<20s}  {'Assigned Format'}")
    print(f"  {'-'*20}  {'-'*20}")
    for lt in sorted(layer_type_formats.keys()):
        fmts = ", ".join(sorted(layer_type_formats[lt]))
        print(f"  {lt:<20s}  {fmts}")
    print()

    return handles


def remove_activation_quant(handles: list):
    """Remove all activation quantization hooks."""
    for h in handles:
        h.remove()
    print(f"[act_quant] Removed {len(handles)} activation quantization hooks.")

# ---------------------------------------------------------------------------
# 6. WikiText-2 test set data loading
# ---------------------------------------------------------------------------

def load_wikitext2_test(tokenizer, local_wikitext2_dir: str = ""):
    """
    Load WikiText-2 test set and encode to input_ids.

    Tries local loading first, falls back to online loading.

    Returns:
        input_ids: torch.Tensor, shape [1, total_tokens]
        source: str, data source description
    """
    # Try local loading
    if local_wikitext2_dir:
        try:
            source = f"local: {local_wikitext2_dir}"
            print(f"[data] Attempting to load WikiText-2 test set from local: {local_wikitext2_dir}")
            input_ids = _load_local_wikitext2_test(tokenizer, local_wikitext2_dir)
            print(f"[data] Local loading successful, total {input_ids.shape[1]} tokens")
            return input_ids, source
        except Exception as exc:
            print(f"[data] Local loading failed: {exc}")
            print("[data] Attempting online loading...")

    # Try online loading
    try:
        from datasets import load_dataset

        print("[data] Loading WikiText-2 test set from HuggingFace online...")
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        joined_text = "\n\n".join(testdata["text"])
        encodings = tokenizer(joined_text, return_tensors="pt")
        input_ids = encodings.input_ids
        source = "online: huggingface wikitext-2-raw-v1 test"
        print(f"[data] Online loading successful, total {input_ids.shape[1]} tokens")
        return input_ids, source
    except Exception as exc:
        online_err = str(exc)

    # Both methods failed
    print(f"[error] WikiText-2 test set loading failed!")
    print(f"  Online error: {online_err}")
    if local_wikitext2_dir:
        print(f"  Local directory: {local_wikitext2_dir}")
    else:
        print(f"  --local-wikitext2-dir not specified, cannot attempt local loading.")
    print(f"  Please ensure network is available, or specify local dataset via --local-wikitext2-dir.")
    sys.exit(1)


def _load_local_wikitext2_test(tokenizer, local_dir: str):
    """Load WikiText-2 test split from local directory."""
    from datasets import load_dataset, load_from_disk

    local_path = Path(local_dir).resolve()

    # Method 1: arrow file
    test_arrow = local_path / "wikitext-test.arrow"
    if test_arrow.exists():
        ds = load_dataset("arrow", data_files={"test": str(test_arrow)}, split="test")
        text_col = "text" if "text" in ds.column_names else ds.column_names[0]
        joined_text = "\n\n".join(ds[text_col])
        encodings = tokenizer(joined_text, return_tensors="pt")
        return encodings.input_ids

    # Method 2: load_from_disk (HuggingFace DatasetDict format)
    loaded = load_from_disk(str(local_path))
    if hasattr(loaded, "keys") and "test" in loaded:
        test_ds = loaded["test"]
    elif hasattr(loaded, "keys") and "train" in loaded:
        print("[warn] No test split in local dataset, using train split (not recommended).")
        test_ds = loaded["train"]
    else:
        test_ds = loaded

    text_col = "text" if "text" in test_ds.column_names else test_ds.column_names[0]
    joined_text = "\n\n".join(test_ds[text_col])
    encodings = tokenizer(joined_text, return_tensors="pt")
    return encodings.input_ids

# ---------------------------------------------------------------------------
# 7. Sliding window Perplexity computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_ppl(model, input_ids: torch.Tensor, seqlen: int, device: torch.device):
    """
    Compute Perplexity using non-overlapping sliding windows.

    Splits input_ids into non-overlapping windows of size seqlen, runs forward pass
    on each window, accumulates cross-entropy loss, and computes PPL = exp(total_nll / total_tokens).

    Returns:
        ppl: float, perplexity value
        total_tokens: int, total tokens used in computation
    """
    model.eval()
    model.to(device)

    total_tokens = input_ids.shape[1]
    n_windows = total_tokens // seqlen
    if n_windows == 0:
        raise ValueError(
            f"Test set token count ({total_tokens}) is less than seqlen ({seqlen}), cannot evaluate."
        )

    total_nll = 0.0
    total_count = 0

    print(f"[eval] Starting evaluation: {n_windows} windows, seqlen={seqlen}, total tokens={total_tokens}")

    for i in range(n_windows):
        start = i * seqlen
        end = start + seqlen
        window_ids = input_ids[:, start:end].to(device)  # [1, seqlen]

        outputs = model(window_ids)
        logits = outputs.logits  # [1, seqlen, vocab_size]

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = window_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )

        total_nll += loss.item()
        total_count += shift_labels.numel()

        if (i + 1) % 10 == 0 or (i + 1) == n_windows:
            current_ppl = math.exp(total_nll / total_count)
            print(f"  Window {i + 1}/{n_windows}  cumulative PPL = {current_ppl:.4f}")

    ppl = math.exp(total_nll / total_count)
    return ppl, total_count

# ---------------------------------------------------------------------------
# 8. Result output and saving
# ---------------------------------------------------------------------------

def build_act_quant_descriptor(global_fmt: str, overrides: dict) -> str:
    """
    Build a compact activation quantization descriptor string.

    Examples:
        - "none"
        - "int8"
        - "int4-g128"
        - "int4-g128+down_proj:int8"
        - "int4-g128+down_proj:int8,o_proj:int8"
    """
    if not overrides:
        return global_fmt
    override_parts = [f"{k}:{v}" for k, v in sorted(overrides.items())]
    return f"{global_fmt}+{','.join(override_parts)}"


def print_results(label: str, ppl: float, total_tokens: int, elapsed: float, act_quant_desc: str):
    """Print evaluation results to terminal."""
    print()
    print("=" * 60)
    print(f"  Evaluation Results")
    print("=" * 60)
    print(f"  Label            : {label}")
    print(f"  Perplexity (PPL) : {ppl:.4f}")
    print(f"  ActQ Format      : {act_quant_desc}")
    print(f"  Eval Tokens      : {total_tokens}")
    print(f"  Elapsed          : {elapsed:.2f} s")
    print("=" * 60)
    print()


def save_results_json(
    output_dir: str,
    label: str,
    ppl: float,
    total_tokens: int,
    elapsed: float,
    model_dir: str,
    quant_weights: str,
    seqlen: int,
    seed: int,
    dtype: str,
    data_source: str,
    act_quant: str = "none",
    act_quant_overrides: dict = None,
):
    """Save evaluation results as JSON file."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    result = {
        "label": label,
        "ppl": round(ppl, 4),
        "total_tokens": total_tokens,
        "elapsed_seconds": round(elapsed, 2),
        "model_dir": str(Path(model_dir).resolve()),
        "quant_weights": str(Path(quant_weights).resolve()) if quant_weights else "",
        "seqlen": seqlen,
        "seed": seed,
        "dtype": dtype,
        "act_quant": act_quant,
        "act_quant_overrides": act_quant_overrides if act_quant_overrides else None,
        "data_source": data_source,
        "timestamp": datetime.now().isoformat(),
    }

    json_file = out_path / f"ppl_{label}.json"
    json_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] JSON result saved: {json_file}")
    return result


def append_results_txt(output_dir: str, result: dict):
    """
    Append evaluation result to results.txt summary file.

    Each evaluation appends one record for easy cross-variant PPL comparison.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    txt_file = out_path / "results.txt"

    # Header (written only when file does not exist or is empty)
    header = (
        f"{'Label':<45s}  {'PPL':>10s}  {'ActQ':<30s}  {'Tokens':>10s}  "
        f"{'Time(s)':>10s}  {'Dtype':<10s}  {'Timestamp':<26s}  {'Quant Weights'}"
    )
    separator = "-" * len(header)

    write_header = not txt_file.exists() or txt_file.stat().st_size == 0

    quant_w = result.get("quant_weights", "")
    if not quant_w:
        quant_w = "(FP16 original)"

    act_q = result.get("act_quant", "none")

    line = (
        f"{result['label']:<45s}  {result['ppl']:>10.4f}  {act_q:<30s}  {result['total_tokens']:>10d}  "
        f"{result['elapsed_seconds']:>10.2f}  {result['dtype']:<10s}  "
        f"{result['timestamp']:<26s}  {quant_w}"
    )

    with open(txt_file, "a", encoding="utf-8") as f:
        if write_header:
            f.write("WikiText-2 Perplexity Benchmark Results (V2 - Multi-Format ActQ)\n")
            f.write(f"{'=' * len(header)}\n")
            f.write(header + "\n")
            f.write(separator + "\n")
        f.write(line + "\n")

    print(f"[save] Result appended to: {txt_file}")

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Fix random seed
    torch.manual_seed(args.seed)

    # Device detection
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"[env] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[env] CUDA not available, using CPU (will be slow).")

    dtype = dtype_from_str(args.dtype)

    # Load tokenizer
    print(f"[load] Loading tokenizer: {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)

    # Load model
    model = load_model(args.model_dir, args.quant_weights, dtype)

    # Adjust seqlen
    seqlen = min(args.seqlen, model.seqlen)
    if seqlen != args.seqlen:
        print(f"[warn] --seqlen {args.seqlen} > model.seqlen {model.seqlen}, using {seqlen}")

    # Parse activation quantization settings
    global_fmt = args.act_quant
    overrides = parse_act_quant_override(args.act_quant_override)
    act_quant_desc = build_act_quant_descriptor(global_fmt, overrides)

    # Inject activation quantization hooks
    act_quant_handles = []
    if global_fmt != "none" or overrides:
        act_quant_handles = inject_activation_quant(model, global_fmt, overrides)
    else:
        print("[act_quant] Activation quantization disabled (--act-quant none, no overrides).")

    # Load WikiText-2 test set
    input_ids, data_source = load_wikitext2_test(tokenizer, args.local_wikitext2_dir)

    # Evaluate
    tick = time.time()
    ppl, total_tokens = evaluate_ppl(model, input_ids, seqlen, device)
    elapsed = time.time() - tick

    # Remove activation quantization hooks
    if act_quant_handles:
        remove_activation_quant(act_quant_handles)

    # Print results to terminal
    print_results(args.label, ppl, total_tokens, elapsed, act_quant_desc)

    # Save JSON (single result)
    result = save_results_json(
        output_dir=args.output_dir,
        label=args.label,
        ppl=ppl,
        total_tokens=total_tokens,
        elapsed=elapsed,
        model_dir=args.model_dir,
        quant_weights=args.quant_weights,
        seqlen=seqlen,
        seed=args.seed,
        dtype=args.dtype,
        data_source=data_source,
        act_quant=act_quant_desc,
        act_quant_overrides=overrides if overrides else None,
    )

    # Append to results.txt (summary file)
    append_results_txt(args.output_dir, result)


if __name__ == "__main__":
    main()
