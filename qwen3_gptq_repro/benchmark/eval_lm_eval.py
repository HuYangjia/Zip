#!/usr/bin/env python3
"""
LM-Eval Harness Benchmark Script

Usage examples:
  # 1) FP16 baseline on HellaSwag
  python benchmark/eval_lm_eval.py \
      --model-dir /path/to/Qwen3-4B-Instruct-2507 \
      --tasks hellaswag \
      --apply-chat-template \
      --label fp16_hellaswag

  # 2) Evaluate quantized weights using the same loading style as eval_ppl.py
  python benchmark/eval_lm_eval.py \
      --model-dir /path/to/Qwen3-4B-Instruct-2507 \
      --quant-weights output/exp_submatrix_mixed/b128x128_r10_qe/qwen3-4b-instruct-2507-gptq-4bit.pt \
      --tasks hellaswag,arc_easy \
      --batch-size auto \
      --apply-chat-template \
      --label v9_b128x128_r10_qe

  # 3) Evaluate with activation fake quantization enabled
  python benchmark/eval_lm_eval.py \
      --model-dir /path/to/Qwen3-4B-Instruct-2507 \
      --quant-weights output/exp_submatrix_mixed/b128x128_r10_qe/qwen3-4b-instruct-2507-gptq-4bit.pt \
      --tasks piqa \
      --act-quant int4-g128 \
      --act-quant-override down_proj:int8 \
      --apply-chat-template \
      --label v9_piqa_a4g128_downa8
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from smooth_block_quant import fake_quant_activation_int4_group_symmetric


def _validate_act_quant(value: str) -> str:
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
        "Accepted: 'none', 'int8', 'int4-g<N>' (for example 'int4-g128')"
    )


def _parse_batch_size(value: str):
    if value == "auto":
        return value
    try:
        ivalue = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid --batch-size '{value}'. Use an integer or 'auto'."
        ) from exc
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("batch-size must be positive")
    return ivalue


def _parse_limit(value: str):
    if value is None or value == "":
        return None
    try:
        if "." in value:
            fvalue = float(value)
            if fvalue <= 0:
                raise ValueError
            return fvalue
        ivalue = int(value)
        if ivalue <= 0:
            raise ValueError
        return ivalue
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid --limit '{value}'. Use a positive int or float."
        ) from exc


def _parse_gen_kwargs(value: str):
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            "--gen-kwargs must be a JSON object string, "
            'for example \'{"temperature": 0.0, "top_p": 1.0}\''
        ) from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("--gen-kwargs must decode to a JSON object")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="LM-Eval Harness benchmark with the same quantized model loading path as eval_ppl.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="HuggingFace model directory (original pretrained weights / tokenizer).",
    )
    parser.add_argument(
        "--quant-weights",
        type=str,
        default="",
        help="Path to quantized state_dict (.pt). If omitted, evaluates the original model.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        required=True,
        help="Comma-separated lm-eval task names, for example 'hellaswag,arc_easy,piqa'.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="unnamed",
        help="Label for this run (used in saved filenames).",
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=0,
        help="Few-shot example count for lm-eval tasks.",
    )
    parser.add_argument(
        "--batch-size",
        type=_parse_batch_size,
        default="auto",
        help="Evaluation batch size. Use an integer or 'auto'.",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=64,
        help="Maximum auto batch size to probe when --batch-size auto is used.",
    )
    parser.add_argument(
        "--limit",
        type=_parse_limit,
        default=None,
        help="Optional task sample limit. Accepts an int count or float fraction.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(_PROJECT_DIR, "output", "lm_eval"),
        help="Directory for lm-eval JSON outputs and summary text.",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="results_lm_eval.txt",
        help="Summary filename created inside --output-dir.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Requested model precision when loading from HuggingFace.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for Python / NumPy / Torch / few-shot sampling.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Evaluation device, for example 'auto', 'cuda:0', or 'cpu'.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading tokenizer / model wrappers.",
    )
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="Apply the model chat template. Recommended for instruct/chat checkpoints.",
    )
    parser.add_argument(
        "--fewshot-as-multiturn",
        action="store_true",
        help="Format few-shot examples as multi-turn chats when chat template is enabled.",
    )
    parser.add_argument(
        "--system-instruction",
        type=str,
        default="",
        help="Optional system instruction forwarded to lm-eval.",
    )
    parser.add_argument(
        "--gen-kwargs",
        type=_parse_gen_kwargs,
        default=None,
        help='Optional JSON object string passed as lm-eval gen_kwargs, e.g. \'{"temperature": 0.0}\'.',
    )
    parser.add_argument(
        "--log-samples",
        action="store_true",
        help="Ask lm-eval to store per-sample model outputs in the JSON result.",
    )
    parser.add_argument(
        "--include-path",
        type=str,
        default="",
        help="Optional directory containing custom lm-eval tasks.",
    )
    parser.add_argument(
        "--hf-home",
        type=str,
        default="",
        help=(
            "Optional HuggingFace cache root. If set, HF_HOME will point here, "
            "so hub and datasets caches can live under one directory."
        ),
    )
    parser.add_argument(
        "--dataset-cache-dir",
        type=str,
        default="",
        help=(
            "Optional datasets cache directory used by zero-shot tasks. "
            "Downloaded task datasets will be stored here and reused from here."
        ),
    )
    parser.add_argument(
        "--hub-cache-dir",
        type=str,
        default="",
        help=(
            "Optional HuggingFace Hub cache directory for dataset/model artifacts. "
            "Useful when you want to separate hub downloads from HF_HOME."
        ),
    )
    parser.add_argument(
        "--datasets-offline",
        action="store_true",
        help=(
            "Load zero-shot datasets from local cache only. "
            "If required dataset files are missing from --dataset-cache-dir / HF cache, evaluation will fail."
        ),
    )
    parser.add_argument(
        "--dataset-cache-alias",
        action="append",
        default=[],
        help=(
            "Optional offline cache alias in the form 'hub_id=local_dir_or_name'. "
            "Example: --dataset-cache-alias allenai/winogrande=winogrande. "
            "When used with --dataset-cache-dir, the script creates the expected HuggingFace cache alias."
        ),
    )
    parser.add_argument(
        "--act-quant",
        type=_validate_act_quant,
        default="none",
        help="Activation fake quant format: 'none', 'int8', or 'int4-g<N>'.",
    )
    parser.add_argument(
        "--act-quant-override",
        type=str,
        default="",
        help="Comma-separated per-layer overrides like 'down_proj:int8,o_proj:int8'.",
    )
    return parser.parse_args()


def dtype_from_str(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


def load_model(model_dir: str, quant_weights: str, dtype: torch.dtype, trust_remote_code: bool):
    print(f"[load] Loading model structure: {model_dir}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    if quant_weights:
        print(f"[load] Loading quantized weights: {quant_weights}")
        state_dict = torch.load(quant_weights, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"[warn] missing_keys: {len(missing_keys)}")
            for key in missing_keys[:5]:
                print(f"       - {key}")
            if len(missing_keys) > 5:
                print(f"       ... and {len(missing_keys) - 5} more")
        if unexpected_keys:
            print(f"[warn] unexpected_keys: {len(unexpected_keys)}")
            for key in unexpected_keys[:5]:
                print(f"       - {key}")
            if len(unexpected_keys) > 5:
                print(f"       ... and {len(unexpected_keys) - 5} more")
    else:
        print("[load] No quantized weights specified, using original model weights.")

    max_seq = getattr(model.config, "max_position_embeddings", 2048)
    model.seqlen = min(2048, int(max_seq))
    return model


def fake_quantize_activation_int8(x: torch.Tensor) -> torch.Tensor:
    x_abs_max = x.abs().amax(dim=-1, keepdim=True)
    scale = (x_abs_max / 127.0).clamp(min=1e-10)
    x_q = (x / scale).round().clamp(-127, 127)
    return x_q * scale


def fake_quantize_activation_int4_group(x: torch.Tensor, group_size: int) -> torch.Tensor:
    return fake_quant_activation_int4_group_symmetric(x, group_size)


def fake_quantize_activation(x: torch.Tensor, fmt: str, layer_name: str) -> torch.Tensor:
    if fmt == "none":
        return x
    if fmt == "int8":
        x_dq = fake_quantize_activation_int8(x)
    else:
        m = re.fullmatch(r"int4-g(\d+)", fmt)
        if m is None:
            print(f"[warn] Unknown act-quant format '{fmt}' for layer {layer_name}, skipping.")
            return x
        x_dq = fake_quantize_activation_int4_group(x, int(m.group(1)))

    if torch.isnan(x_dq).any() or torch.isinf(x_dq).any():
        print(f"[warn] NaN/Inf after activation quantization in layer {layer_name}, falling back to FP.")
        return x
    return x_dq


def parse_act_quant_override(override_str: str) -> dict:
    if not override_str or not override_str.strip():
        return {}

    overrides = {}
    for pair in override_str.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            print(f"[warn] Invalid override format '{pair}', expected key:value. Skipping.")
            continue
        key, value = pair.split(":", 1)
        key = key.strip()
        value = value.strip()
        try:
            _validate_act_quant(value)
        except argparse.ArgumentTypeError as exc:
            print(f"[warn] Invalid override for '{key}': {exc}. Skipping.")
            continue
        overrides[key] = value
    return overrides


def resolve_act_quant_format(layer_name: str, global_fmt: str, overrides: dict) -> str:
    for key, fmt in overrides.items():
        if key in layer_name:
            return fmt
    return global_fmt


def inject_activation_quant(model, global_fmt: str, overrides: dict) -> list:
    handles = []
    n_injected = 0
    n_skipped = 0
    layer_type_formats = defaultdict(set)

    def _make_pre_hook(layer_name: str, fmt: str):
        def _pre_hook(module, args):
            if not args:
                return args
            x = args[0]
            x_dq = fake_quantize_activation(x, fmt, layer_name)
            return (x_dq,) + args[1:]

        return _pre_hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            fmt = resolve_act_quant_format(name, global_fmt, overrides)
            layer_type = name.split(".")[-1] if name else name
            layer_type_formats[layer_type].add(fmt)

            if fmt == "none":
                n_skipped += 1
                continue

            handles.append(module.register_forward_pre_hook(_make_pre_hook(name, fmt)))
            n_injected += 1

    print("\n[act_quant] Activation quantization hook injection summary:")
    print(f"  Injected: {n_injected} layers | Skipped (none): {n_skipped} layers")
    print(f"  Global format: {global_fmt}")
    if overrides:
        print(f"  Overrides: {overrides}")
    print(f"  {'Layer Type':<20s}  {'Assigned Format'}")
    print(f"  {'-' * 20}  {'-' * 20}")
    for layer_type in sorted(layer_type_formats):
        formats = ", ".join(sorted(layer_type_formats[layer_type]))
        print(f"  {layer_type:<20s}  {formats}")
    print()
    return handles


def remove_activation_quant(handles: list):
    for handle in handles:
        handle.remove()
    print(f"[act_quant] Removed {len(handles)} activation quantization hooks.")


def build_act_quant_descriptor(global_fmt: str, overrides: dict) -> str:
    if not overrides:
        return global_fmt
    override_parts = [f"{key}:{value}" for key, value in sorted(overrides.items())]
    return f"{global_fmt}+{','.join(override_parts)}"


def choose_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def configure_hf_dataset_env(args):
    configured = {}

    if args.hf_home:
        hf_home = str(Path(args.hf_home).resolve())
        os.environ["HF_HOME"] = hf_home
        configured["HF_HOME"] = hf_home

    if args.dataset_cache_dir:
        dataset_cache_dir = str(Path(args.dataset_cache_dir).resolve())
        os.environ["HF_DATASETS_CACHE"] = dataset_cache_dir
        configured["HF_DATASETS_CACHE"] = dataset_cache_dir

    if args.hub_cache_dir:
        hub_cache_dir = str(Path(args.hub_cache_dir).resolve())
        os.environ["HF_HUB_CACHE"] = hub_cache_dir
        configured["HF_HUB_CACHE"] = hub_cache_dir

    if args.datasets_offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        configured["HF_DATASETS_OFFLINE"] = "1"

    return configured


def _hf_cache_key_from_hub_id(hub_id: str) -> str:
    return hub_id.replace("/", "___")


def _default_dataset_aliases(tasks: list) -> dict:
    aliases = {}
    if "winogrande" in tasks:
        aliases["allenai/winogrande"] = "winogrande"
    return aliases


def _parse_dataset_alias_args(alias_args: list) -> dict:
    aliases = {}
    for item in alias_args:
        if "=" not in item:
            print(f"[warn] Invalid --dataset-cache-alias '{item}', expected hub_id=local_dir_or_name. Skipping.")
            continue
        hub_id, local_ref = item.split("=", 1)
        hub_id = hub_id.strip()
        local_ref = local_ref.strip()
        if not hub_id or not local_ref:
            print(f"[warn] Invalid --dataset-cache-alias '{item}', empty hub_id or local path. Skipping.")
            continue
        aliases[hub_id] = local_ref
    return aliases


def ensure_dataset_cache_aliases(dataset_cache_dir: str, tasks: list, alias_args: list):
    if not dataset_cache_dir:
        return []

    cache_root = Path(dataset_cache_dir).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    aliases = _default_dataset_aliases(tasks)
    aliases.update(_parse_dataset_alias_args(alias_args))

    created = []
    for hub_id, local_ref in aliases.items():
        expected_dir = cache_root / _hf_cache_key_from_hub_id(hub_id)
        local_path = Path(local_ref)
        if not local_path.is_absolute():
            local_path = cache_root / local_ref
        local_path = local_path.resolve()

        if expected_dir.exists():
            continue
        if not local_path.exists():
            print(f"[warn] Dataset cache alias source does not exist: {local_path} (for {hub_id})")
            continue

        try:
            os.symlink(local_path, expected_dir, target_is_directory=True)
            created.append((hub_id, str(expected_dir), str(local_path)))
        except FileExistsError:
            continue

    return created


def import_lm_eval():
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.tasks import TaskManager
        from lm_eval.utils import handle_non_serializable
    except ImportError as exc:
        raise SystemExit(
            "lm_eval is not installed. Please install the official HuggingFace backend first:\n"
            '  pip install "lm_eval[hf]"\n'
            f"Original import error: {exc}"
        ) from exc

    return lm_eval, HFLM, TaskManager, handle_non_serializable


def flatten_metrics(results: dict):
    flattened = []
    for task_name, metric_dict in results.get("results", {}).items():
        for metric_name, value in metric_dict.items():
            if metric_name.endswith("_stderr") or metric_name.endswith(",stderr"):
                continue
            if isinstance(value, (dict, list, tuple)):
                continue

            stderr = None
            stderr_key_variants = [
                f"{metric_name}_stderr",
                f"{metric_name},stderr",
            ]
            for stderr_key in stderr_key_variants:
                if stderr_key in metric_dict:
                    stderr = metric_dict[stderr_key]
                    break

            flattened.append(
                {
                    "task": task_name,
                    "metric": metric_name,
                    "value": value,
                    "stderr": stderr,
                }
            )
    return flattened


def _is_number(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _format_metric_value(value, width: int = 12) -> str:
    if value is None:
        return "".rjust(width)
    if _is_number(value):
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return f"{numeric}".rjust(width)
        return f"{numeric:.6f}".rjust(width)

    text = str(value)
    if len(text) > width:
        text = text[: width - 1] + "…"
    return text.rjust(width)


def print_results(label: str, flattened_metrics: list, elapsed: float, act_quant_desc: str):
    print()
    print("=" * 80)
    print("  LM-Eval Results")
    print("=" * 80)
    print(f"  Label       : {label}")
    print(f"  ActQ Format : {act_quant_desc}")
    print(f"  Elapsed     : {elapsed:.2f} s")
    print("-" * 80)
    if not flattened_metrics:
        print("  [warn] No task metrics were returned.")
    else:
        print(f"  {'Task':<24s} {'Metric':<24s} {'Value':>12s} {'Stderr':>12s}")
        print(f"  {'-' * 24} {'-' * 24} {'-' * 12} {'-' * 12}")
        for item in flattened_metrics:
            value_text = _format_metric_value(item["value"])
            stderr_text = _format_metric_value(item["stderr"])
            print(
                f"  {item['task']:<24.24s} "
                f"{item['metric']:<24.24s} "
                f"{value_text} "
                f"{stderr_text}"
            )
    print("=" * 80)
    print()


def save_results_json(output_dir: str, label: str, payload: dict, handle_non_serializable):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_file = out_path / f"lm_eval_{label}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=handle_non_serializable)
    print(f"[save] JSON result saved: {json_file}")
    return json_file


def append_results_txt(output_dir: str, label: str, run_meta: dict, flattened_metrics: list, results_file: str):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    txt_file = out_path / results_file

    header = (
        f"{'Label':<28s}  {'Task':<24s}  {'Metric':<24s}  {'Value':>12s}  "
        f"{'Stderr':>12s}  {'Fewshot':>7s}  {'ActQ':<24s}  {'Timestamp':<26s}"
    )
    separator = "-" * len(header)
    write_header = not txt_file.exists() or txt_file.stat().st_size == 0

    with open(txt_file, "a", encoding="utf-8") as f:
        if write_header:
            f.write("LM-Eval Harness Results\n")
            f.write(f"{'=' * len(header)}\n")
            f.write(header + "\n")
            f.write(separator + "\n")

        for item in flattened_metrics:
            value_text = _format_metric_value(item["value"])
            stderr_text = _format_metric_value(item["stderr"])
            line = (
                f"{label:<28.28s}  "
                f"{item['task']:<24.24s}  "
                f"{item['metric']:<24.24s}  "
                f"{value_text}  "
                f"{stderr_text}  "
                f"{run_meta['num_fewshot']:>7d}  "
                f"{run_meta['act_quant']:<24.24s}  "
                f"{run_meta['timestamp']:<26s}"
            )
            f.write(line + "\n")

    print(f"[save] Summary appended to: {txt_file}")


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = choose_device(args.device)
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if not tasks:
        raise SystemExit("No valid tasks provided in --tasks")

    hf_env = configure_hf_dataset_env(args)
    alias_links = ensure_dataset_cache_aliases(args.dataset_cache_dir, tasks, args.dataset_cache_alias)
    lm_eval, HFLM, TaskManager, handle_non_serializable = import_lm_eval()

    if device.type == "cuda":
        print(f"[env] Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        print(f"[env] Using device: {device}")
    if hf_env:
        print("[env] HuggingFace dataset/cache configuration:")
        for key, value in hf_env.items():
            print(f"  {key}={value}")
    if alias_links:
        print("[env] Created dataset cache aliases:")
        for hub_id, expected_dir, local_path in alias_links:
            print(f"  {hub_id} -> {expected_dir} -> {local_path}")

    dtype = dtype_from_str(args.dtype)

    print(f"[load] Loading tokenizer: {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        use_fast=False,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args.model_dir, args.quant_weights, dtype, args.trust_remote_code)
    model.eval()
    model.to(device)

    overrides = parse_act_quant_override(args.act_quant_override)
    act_quant_desc = build_act_quant_descriptor(args.act_quant, overrides)

    act_quant_handles = []
    if args.act_quant != "none" or overrides:
        act_quant_handles = inject_activation_quant(model, args.act_quant, overrides)
    else:
        print("[act_quant] Activation quantization disabled (--act-quant none, no overrides).")

    print(f"[eval] Selected tasks: {tasks}")
    print(f"[eval] batch_size={args.batch_size}, max_batch_size={args.max_batch_size}, fewshot={args.num_fewshot}")

    task_manager = TaskManager(include_path=args.include_path) if args.include_path else None
    run_timestamp = datetime.now().isoformat()

    try:
        tick = time.time()
        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            backend="causal",
            batch_size=args.batch_size,
            max_batch_size=args.max_batch_size,
            trust_remote_code=args.trust_remote_code,
        )

        results = lm_eval.simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            max_batch_size=args.max_batch_size,
            device=str(device),
            limit=args.limit,
            log_samples=args.log_samples,
            task_manager=task_manager,
            gen_kwargs=args.gen_kwargs,
            apply_chat_template=args.apply_chat_template,
            system_instruction=args.system_instruction or None,
            fewshot_as_multiturn=args.fewshot_as_multiturn,
            random_seed=args.seed,
            numpy_random_seed=args.seed,
            torch_random_seed=args.seed,
            fewshot_random_seed=args.seed,
        )
        elapsed = time.time() - tick
    finally:
        if act_quant_handles:
            remove_activation_quant(act_quant_handles)

    flattened_metrics = flatten_metrics(results)
    payload = {
        "label": args.label,
        "timestamp": run_timestamp,
        "elapsed_seconds": round(elapsed, 2),
        "model_dir": str(Path(args.model_dir).resolve()),
        "quant_weights": str(Path(args.quant_weights).resolve()) if args.quant_weights else "",
        "tasks": tasks,
        "num_fewshot": args.num_fewshot,
        "batch_size": args.batch_size,
        "max_batch_size": args.max_batch_size,
        "limit": args.limit,
        "dtype": args.dtype,
        "device": str(device),
        "apply_chat_template": args.apply_chat_template,
        "fewshot_as_multiturn": args.fewshot_as_multiturn,
        "system_instruction": args.system_instruction or None,
        "gen_kwargs": args.gen_kwargs,
        "hf_cache_env": hf_env if hf_env else None,
        "act_quant": act_quant_desc,
        "act_quant_overrides": overrides if overrides else None,
        "results": results,
        "flattened_metrics": flattened_metrics,
    }
    save_results_json(args.output_dir, args.label, payload, handle_non_serializable)

    append_results_txt(
        output_dir=args.output_dir,
        label=args.label,
        run_meta={
            "num_fewshot": args.num_fewshot,
            "act_quant": act_quant_desc,
            "timestamp": run_timestamp,
        },
        flattened_metrics=flattened_metrics,
        results_file=args.results_file,
    )

    print_results(args.label, flattened_metrics, elapsed, act_quant_desc)


if __name__ == "__main__":
    main()
