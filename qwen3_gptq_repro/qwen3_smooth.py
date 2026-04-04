import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from qwen3_gptq import (
    dtype_from_str,
    get_qwen3,
    get_wikitext2_or_fallback_loader,
    load_custom_model_class,
    register_custom_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3 SmoothQuant preprocessing (alpha=1.0 only).")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507",
        help="Local model directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output/smooth",
        help="Directory to save smooth artifacts.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Load dtype for model weights.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="SmoothQuant alpha. This script currently supports alpha=1.0 only.",
    )
    parser.add_argument("--eps", type=float, default=1e-5, help="Numerical floor for smooth scale.")
    parser.add_argument(
        "--custom-modeling-file",
        type=str,
        default="",
        help="Optional custom modeling file path, e.g. modeling_qwn3_update.py.",
    )
    parser.add_argument(
        "--local-wikitext2-dir",
        type=str,
        default="",
        help="Optional local WikiText2 directory (contains wikitext-train.arrow or load_from_disk output).",
    )
    return parser.parse_args()


@torch.no_grad()
def collect_rmsnorm_activation_max(model, dataloader, dev: torch.device):
    layers = model.model.layers
    tracked = {}
    stats = {}
    handles = []

    for i, layer in enumerate(layers):
        in_name = f"model.layers.{i}.input_layernorm"
        post_name = f"model.layers.{i}.post_attention_layernorm"
        tracked[in_name] = layer.input_layernorm
        tracked[post_name] = layer.post_attention_layernorm
        stats[in_name] = None
        stats[post_name] = None

    def make_hook(name):
        def hook(_, __, out):
            tensor = out[0] if isinstance(out, (tuple, list)) else out
            if tensor.ndim == 3:
                cur = tensor.detach().abs().amax(dim=(0, 1))
            elif tensor.ndim == 2:
                cur = tensor.detach().abs().amax(dim=0)
            else:
                raise RuntimeError(f"unexpected activation ndim for {name}: {tensor.ndim}")
            prev = stats[name]
            stats[name] = cur if prev is None else torch.maximum(prev, cur)

        return hook

    for name, mod in tracked.items():
        handles.append(mod.register_forward_hook(make_hook(name)))

    for batch in dataloader:
        model(batch[0].to(dev))

    for h in handles:
        h.remove()

    missing = [name for name, value in stats.items() if value is None]
    if missing:
        raise RuntimeError(f"failed to collect activation stats for: {missing}")
    return {k: v.float().cpu() for k, v in stats.items()}


@torch.no_grad()
def smooth_rmsnorm_and_linears(norm, linears, act_absmax: torch.Tensor, alpha: float, eps: float):
    if alpha != 1.0:
        raise ValueError("Only alpha=1.0 is supported in this script.")
    scale = torch.clamp(act_absmax, min=eps)
    scale_on_dev = scale.to(device=norm.weight.device, dtype=norm.weight.dtype)

    before_norm = norm.weight.detach().float().cpu()
    norm.weight.data.div_(scale_on_dev)
    after_norm = norm.weight.detach().float().cpu()

    for linear in linears:
        if linear.weight.shape[1] != scale_on_dev.numel():
            raise ValueError(
                f"Linear in_features mismatch: expected {scale_on_dev.numel()}, got {linear.weight.shape[1]}"
            )
        linear.weight.data.mul_(scale_on_dev.view(1, -1))
    return scale.float().cpu(), before_norm, after_norm


@torch.no_grad()
def apply_smoothquant(model, activation_stats: dict, alpha: float, eps: float):
    layers = model.model.layers
    smooth_data = {}

    for i, layer in enumerate(layers):
        in_norm_name = f"model.layers.{i}.input_layernorm"
        attn_key = f"model.layers.{i}.self_attn"
        attn_scale, attn_before, attn_after = smooth_rmsnorm_and_linears(
            norm=layer.input_layernorm,
            linears=[layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
            act_absmax=activation_stats[in_norm_name],
            alpha=alpha,
            eps=eps,
        )
        smooth_data[attn_key] = {
            "norm_name": in_norm_name,
            "target_linears": [
                f"model.layers.{i}.self_attn.q_proj",
                f"model.layers.{i}.self_attn.k_proj",
                f"model.layers.{i}.self_attn.v_proj",
            ],
            "act_absmax": attn_scale,
            "smooth_scale": attn_scale,
            "rmsnorm_weight_before": attn_before,
            "rmsnorm_weight_after": attn_after,
        }

        post_norm_name = f"model.layers.{i}.post_attention_layernorm"
        mlp_key = f"model.layers.{i}.mlp"
        mlp_scale, mlp_before, mlp_after = smooth_rmsnorm_and_linears(
            norm=layer.post_attention_layernorm,
            linears=[layer.mlp.gate_proj, layer.mlp.up_proj],
            act_absmax=activation_stats[post_norm_name],
            alpha=alpha,
            eps=eps,
        )
        smooth_data[mlp_key] = {
            "norm_name": post_norm_name,
            "target_linears": [
                f"model.layers.{i}.mlp.gate_proj",
                f"model.layers.{i}.mlp.up_proj",
            ],
            "act_absmax": mlp_scale,
            "smooth_scale": mlp_scale,
            "rmsnorm_weight_before": mlp_before,
            "rmsnorm_weight_after": mlp_after,
        }
    return smooth_data


def main():
    args = parse_args()
    if abs(args.alpha - 1.0) > 1e-8:
        raise ValueError("This workflow currently fixes SmoothQuant alpha to 1.0.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.custom_modeling_file:
        custom_model_cls = load_custom_model_class(args.custom_modeling_file)
        register_custom_model(custom_model_cls)
        print("registered custom class:", custom_model_cls)

    dtype = dtype_from_str(args.dtype)
    model = get_qwen3(args.model_dir, dtype=dtype)
    model.eval()

    if args.seqlen > model.seqlen:
        print(f"[warn] --seqlen {args.seqlen} > model.seqlen {model.seqlen}; use model.seqlen.")
        args.seqlen = model.seqlen
    model.seqlen = args.seqlen

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    trainloader, source_info = get_wikitext2_or_fallback_loader(
        tokenizer=tokenizer,
        model_dir=args.model_dir,
        nsamples=args.nsamples,
        seqlen=model.seqlen,
        seed=args.seed,
        output_dir=output_dir,
        local_wikitext2_dir=args.local_wikitext2_dir,
    )
    print("calibration source:", source_info["calib_source"])
    if source_info["fallback_used"]:
        print("[fallback] local/wiki load failed, temporary generated text is used.")
        print("[fallback] reason:", source_info["reason"])
        print("[fallback] text file:", source_info["temp_text_file"])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this SmoothQuant script.")
    dev = torch.device("cuda:0")
    model = model.to(dev)

    tick = time.time()
    activation_stats = collect_rmsnorm_activation_max(model=model, dataloader=trainloader, dev=dev)
    smooth_data = apply_smoothquant(model=model, activation_stats=activation_stats, alpha=args.alpha, eps=args.eps)
    elapsed = time.time() - tick
    print(f"smoothquant time: {elapsed:.2f}s")

    model = model.cpu()
    torch.cuda.empty_cache()

    smoothed_weights_path = output_dir / "smoothed_model_state_dict.pt"
    smooth_scales_path = output_dir / "smooth_scales.pt"
    smooth_meta_path = output_dir / "smooth_metadata.json"

    torch.save(model.state_dict(), smoothed_weights_path)
    torch.save(
        {
            "alpha": args.alpha,
            "eps": args.eps,
            "smooth_data": smooth_data,
        },
        smooth_scales_path,
    )

    metadata = {
        "model_dir": str(Path(args.model_dir).resolve()),
        "alpha": args.alpha,
        "eps": args.eps,
        "nsamples": args.nsamples,
        "seqlen": model.seqlen,
        "dtype": args.dtype,
        "calibration": source_info,
        "num_smoothed_groups": len(smooth_data),
        "smoothed_weights_path": str(smoothed_weights_path),
        "smooth_scales_path": str(smooth_scales_path),
        "elapsed_seconds": elapsed,
    }
    smooth_meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("saved smoothed weights:", smoothed_weights_path)
    print("saved smooth scales   :", smooth_scales_path)
    print("saved smooth meta     :", smooth_meta_path)


if __name__ == "__main__":
    main()
