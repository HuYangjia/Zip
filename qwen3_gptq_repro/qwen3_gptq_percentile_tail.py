import argparse
import json
import sys
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


REPO_ROOT = Path(__file__).resolve().parents[1]
GPTQ_DIR = REPO_ROOT / "gptq"
if str(GPTQ_DIR) not in sys.path:
    sys.path.insert(0, str(GPTQ_DIR))

from gptq import GPTQ  # noqa: E402
from modelutils import find_layers  # noqa: E402
from quant import Quantizer  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Percentile-tail prototype: main absorb first, spill to tail later."
    )
    parser.add_argument("--model-dir", type=str, default="/home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output/exp_percentile_tail/proto",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=32)
    parser.add_argument("--seqlen", type=int, default=1024)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--groupsize", type=int, default=128)
    parser.add_argument("--sym", action="store_true")
    parser.add_argument("--act-order", action="store_true")
    parser.add_argument("--true-sequential", action="store_true")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--custom-modeling-file", type=str, default="")
    parser.add_argument("--local-wikitext2-dir", type=str, default="")
    parser.add_argument("--init-state-dict", type=str, default="")

    parser.add_argument("--enable-percentile-tail", action="store_true")
    parser.add_argument("--tail-ratio", type=float, default=0.05)
    parser.add_argument("--tail-rank", type=int, default=0)
    parser.add_argument("--percentile-k", type=float, default=75.0)
    parser.add_argument("--main-wbits", type=int, default=4)
    parser.add_argument("--tail-quant", type=str, default="int8", choices=["int8", "int4_fp4"])
    parser.add_argument("--lambda-reg", type=float, default=1e-4)
    parser.add_argument("--use-gptq-main", action="store_true")

    parser.add_argument("--enable-main-absorb", action="store_true")
    parser.add_argument("--main-absorb-mode", type=str, default="budget_clamped")
    parser.add_argument("--main-absorb-budget-rule", type=str, default="int4_boundary")
    parser.add_argument("--constraint-eps", type=float, default=1e-5)
    parser.add_argument("--disable-tail-compensation", action="store_true")
    parser.add_argument("--stat-samples", type=int, default=8)
    return parser.parse_args()


def _resolve_tail_cols(in_features: int, tail_ratio: float, tail_rank: int):
    if tail_rank > 0:
        tail_cols = min(max(1, int(tail_rank)), max(1, in_features - 1))
    else:
        tail_cols = max(1, int(round(in_features * tail_ratio)))
        tail_cols = min(max(1, in_features - 1), tail_cols)
    return in_features - tail_cols, tail_cols


def compute_percentile_scale(w_main: torch.Tensor, percentile_k: float, q_max: float, eps: float = 1e-8):
    abs_main = w_main.abs()
    q = percentile_k / 100.0
    m = torch.quantile(abs_main, q=q, dim=1)
    scale = torch.clamp(m / q_max, min=eps)
    return m, scale


def quantize_main_uniform(w_main: torch.Tensor, scale: torch.Tensor, q_max: float):
    scale_col = scale.unsqueeze(1)
    q = torch.round(w_main / scale_col).clamp(-q_max, q_max)
    return q * scale_col


def apply_main_constrained_absorb(
    w_main: torch.Tensor,
    w_main_q0: torch.Tensor,
    q_max: float,
    enable_main_absorb: bool,
    boundary: torch.Tensor,
    constraint_eps: float,
):
    residual_before = w_main - w_main_q0
    budget = torch.clamp(boundary - w_main_q0.abs(), min=0.0)

    if enable_main_absorb:
        delta = torch.sign(residual_before) * torch.minimum(residual_before.abs(), budget)
        w_main_q1 = w_main_q0 + delta
    else:
        delta = torch.zeros_like(w_main_q0)
        w_main_q1 = w_main_q0

    residual_after = w_main - w_main_q1
    violate_mask = w_main_q1.abs() > (boundary + constraint_eps)
    violation_count = int(violate_mask.sum().item())
    if violation_count > 0:
        overflow = (w_main_q1.abs() - boundary).clamp(min=0.0)
        idx = torch.nonzero(violate_mask, as_tuple=False)
        preview_n = min(16, idx.shape[0])
        violation_preview = [
            [int(idx[i, 0].item()), int(idx[i, 1].item())]
            for i in range(preview_n)
        ]
        violation_overflow_max = float(overflow.max().item())
    else:
        violation_preview = []
        violation_overflow_max = 0.0

    budget_mean = float(budget.mean().item())
    budget_min = float(budget.min().item())
    budget_max = float(budget.max().item())
    util = (delta.abs() / (budget + 1e-8)).clamp(max=1.0)
    util = util[budget > 0]
    utilization = float(util.mean().item()) if util.numel() else 0.0

    return {
        "w_main_q1": w_main_q1,
        "residual_before": residual_before,
        "residual_after": residual_after,
        "budget_mean": budget_mean,
        "budget_min": budget_min,
        "budget_max": budget_max,
        "budget_utilization": utilization,
        "constraint_violations": violation_count,
        "constraint_violation_positions_preview": violation_preview,
        "constraint_violation_overflow_max": violation_overflow_max,
    }


def compute_tail_projection(
    spill_residual: torch.Tensor,
    x_main: torch.Tensor,
    x_tail: torch.Tensor,
    w_tail: torch.Tensor,
    lambda_reg: float,
    disable_tail_compensation: bool,
):
    # spill_residual: [d_out, d_main], x_main: [n_stat, d_main], x_tail: [n_stat, r]
    h_tt = x_tail.transpose(0, 1) @ x_tail
    htt_cond = float(torch.linalg.cond(h_tt + torch.eye(h_tt.shape[0], dtype=h_tt.dtype)).item())

    lam = float(lambda_reg)
    a = h_tt + lam * torch.eye(h_tt.shape[0], dtype=h_tt.dtype)
    while not torch.isfinite(torch.linalg.cond(a)) or torch.linalg.cond(a) > 1e6:
        lam *= 10.0
        a = h_tt + lam * torch.eye(h_tt.shape[0], dtype=h_tt.dtype)
        if lam > 1.0:
            break

    # e_spill * H_mt = (E * X_m^T) * X_t
    spill_hmt = (spill_residual @ x_main.transpose(0, 1)) @ x_tail
    if disable_tail_compensation:
        e_tail_star = torch.zeros_like(w_tail)
    else:
        e_tail_star = -torch.linalg.solve(a.transpose(0, 1), spill_hmt.transpose(0, 1)).transpose(0, 1)
    w_tail_target = w_tail - e_tail_star
    return w_tail_target, htt_cond, lam, torch.norm(-e_tail_star, dim=1)


def quantize_tail_high_precision(w_tail_target: torch.Tensor, tail_quant: str):
    if tail_quant == "int4_fp4":
        raise NotImplementedError("tail quant 'int4_fp4' is reserved; current prototype supports int8 only.")
    scales = torch.clamp(w_tail_target.abs().amax(dim=1, keepdim=True) / 127.0, min=1e-8)
    q = torch.round(w_tail_target / scales).clamp(-127, 127)
    w_tail_q = q * scales
    tail_quant_error = torch.norm(w_tail_target - w_tail_q, dim=1)
    tail_sat = (q.abs() >= 127).float().mean(dim=1)
    return w_tail_q, tail_quant_error, tail_sat


def collect_linear_inputs_summary(captured_inputs, in_features, stat_samples):
    if not captured_inputs:
        return torch.zeros((1, in_features), dtype=torch.float32)
    x = torch.stack(captured_inputs, dim=0).float()  # [nsamples, in_features]
    n = min(int(stat_samples), x.shape[0])
    return x[:n]


def collect_layer_percentile_tail_stats(stats_records, layer_idx, name, record):
    stats_records[f"model.layers.{layer_idx}.{name}"] = record


@torch.no_grad()
def apply_percentile_tail_on_linear(
    linear: nn.Linear,
    x_summary: torch.Tensor,
    args,
    q0_weight_for_mode_b: torch.Tensor = None,
    original_weight: torch.Tensor = None,
):
    orig = original_weight if original_weight is not None else linear.weight.data.float().cpu()
    d_out, d_in = orig.shape
    main_cols, tail_cols = _resolve_tail_cols(d_in, args.tail_ratio, args.tail_rank)

    w_main = orig[:, :main_cols]
    w_tail = orig[:, main_cols:]

    q_max = float((2 ** (args.main_wbits - 1)) - 1)
    m, s_main = compute_percentile_scale(w_main, args.percentile_k, q_max=q_max)

    if q0_weight_for_mode_b is not None:
        w_main_q0 = q0_weight_for_mode_b[:, :main_cols].float().cpu()
    else:
        w_main_q0 = quantize_main_uniform(w_main, s_main, q_max=q_max)

    if q0_weight_for_mode_b is not None and args.use_gptq_main:
        # In mode B, use GPTQ-main implied per-row bound to avoid mismatched percentile boundary.
        s_from_q0 = torch.clamp(w_main_q0.abs().amax(dim=1) / q_max, min=1e-8)
        boundary = s_from_q0.unsqueeze(1) * q_max
        boundary_source = "gptq_q0_rowmax"
    else:
        boundary = s_main.unsqueeze(1) * q_max
        boundary_source = "percentile_scale"

    absorb = apply_main_constrained_absorb(
        w_main=w_main,
        w_main_q0=w_main_q0,
        q_max=q_max,
        enable_main_absorb=args.enable_main_absorb,
        boundary=boundary,
        constraint_eps=args.constraint_eps,
    )
    w_main_q1 = absorb["w_main_q1"]
    spill = absorb["residual_after"]

    x_main = x_summary[:, :main_cols]
    x_tail = x_summary[:, main_cols:]
    w_tail_target, htt_cond, lam_used, tail_corr_norm = compute_tail_projection(
        spill_residual=spill,
        x_main=x_main,
        x_tail=x_tail,
        w_tail=w_tail,
        lambda_reg=args.lambda_reg,
        disable_tail_compensation=args.disable_tail_compensation,
    )
    w_tail_q, tail_q_err, tail_sat_ratio = quantize_tail_high_precision(w_tail_target, args.tail_quant)

    merged = torch.cat([w_main_q1, w_tail_q], dim=1)
    final_residual = orig - merged

    spill_norm = torch.norm(spill, dim=1)
    main_before_norm = torch.norm(absorb["residual_before"], dim=1)
    main_after_norm = torch.norm(spill, dim=1)
    final_norm = torch.norm(final_residual, dim=1)
    spill_to_tail = spill_norm / (torch.norm(w_tail, dim=1) + 1e-8)

    stats = {
        "tail_start": int(main_cols),
        "tail_cols": int(tail_cols),
        "main_boundary_source": boundary_source,
        "M_mean": float(m.mean().item()),
        "M_min": float(m.min().item()),
        "M_max": float(m.max().item()),
        "main_residual_before_absorb_norm": float(main_before_norm.mean().item()),
        "main_residual_after_absorb_norm": float(main_after_norm.mean().item()),
        "spill_residual_norm": float(spill_norm.mean().item()),
        "spill_ratio_to_tail": float(spill_to_tail.mean().item()),
        "main_absorb_budget_mean": absorb["budget_mean"],
        "main_absorb_budget_min": absorb["budget_min"],
        "main_absorb_budget_max": absorb["budget_max"],
        "main_absorb_budget_utilization": absorb["budget_utilization"],
        "tail_correction_norm": float(tail_corr_norm.mean().item()),
        "tail_quant_error": float(tail_q_err.mean().item()),
        "tail_saturation_ratio": float(tail_sat_ratio.mean().item()),
        "tail_pre_quant_max": float(w_tail_target.abs().max().item()),
        "tail_post_correction_max": float(w_tail_target.abs().max().item()),
        "htt_condition_number_est": htt_cond,
        "lambda_used": lam_used,
        "main_constraint_violations": int(absorb["constraint_violations"]),
        "main_constraint_violation_positions_preview": absorb["constraint_violation_positions_preview"],
        "main_constraint_violation_overflow_max": absorb["constraint_violation_overflow_max"],
        "main_constraint_eps": float(args.constraint_eps),
        "final_residual_after_tail_quant_norm": float(final_norm.mean().item()),
    }
    return merged, stats


@torch.no_grad()
def qwen3_sequential_percentile_tail(model, dataloader, dev: torch.device, args):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None, "position_ids": None, "cache_position": None, "position_embeddings": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.attention_type = getattr(module, "attention_type", "full_attention")

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask")
            cache["position_ids"] = kwargs.get("position_ids")
            cache["cache_position"] = kwargs.get("cache_position")
            cache["position_embeddings"] = kwargs.get("position_embeddings")
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    cache_position = cache["cache_position"]
    position_embeddings = cache["position_embeddings"]

    quantizers = {}
    layer_stats = {}

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)
        if args.true_sequential:
            sequential = [
                ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
                ["self_attn.o_proj"],
                ["mlp.up_proj", "mlp.gate_proj"],
                ["mlp.down_proj"],
            ]
        else:
            sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names if n in full}
            if not subset:
                continue

            gptq = {}
            captured = {name: [] for name in subset}
            for name in subset:
                if args.use_gptq_main:
                    gptq[name] = GPTQ(subset[name])
                    gptq[name].quantizer = Quantizer()
                    gptq[name].quantizer.configure(args.main_wbits, perchannel=True, sym=args.sym, mse=False)

            def add_batch(name):
                def tmp(_, inp, out):
                    x = inp[0].detach()
                    if x.ndim == 3:
                        x_sum = x.mean(dim=1).squeeze(0).float().cpu()
                    else:
                        x_sum = x.squeeze(0).float().cpu()
                    captured[name].append(x_sum)
                    if args.use_gptq_main:
                        gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = [subset[name].register_forward_hook(add_batch(name)) for name in subset]
            for j in range(args.nsamples):
                layer_out = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
                outs[j] = layer_out[0] if isinstance(layer_out, tuple) else layer_out
            for h in handles:
                h.remove()

            for name in subset:
                print(f"Layer {i} -> {name}")
                linear = subset[name]
                q0 = None
                orig_before_gptq = linear.weight.data.float().cpu().clone()
                if args.use_gptq_main:
                    gptq[name].fasterquant(
                        percdamp=args.percdamp,
                        groupsize=args.groupsize,
                        actorder=args.act_order,
                        static_groups=False,
                    )
                    q0 = linear.weight.data.float().cpu().clone()
                    gptq[name].free()

                x_summary = collect_linear_inputs_summary(
                    captured_inputs=captured[name],
                    in_features=linear.weight.shape[1],
                    stat_samples=args.stat_samples,
                )
                merged, stats = apply_percentile_tail_on_linear(
                    linear=linear,
                    x_summary=x_summary,
                    args=args,
                    q0_weight_for_mode_b=q0,
                    original_weight=orig_before_gptq,
                )
                linear.weight.data.copy_(merged.to(device=linear.weight.device, dtype=linear.weight.dtype))

                collect_layer_percentile_tail_stats(layer_stats, i, name, stats)
                quantizers[f"model.layers.{i}.{name}"] = {
                    "bits_main": args.main_wbits,
                    "tail_quant": args.tail_quant,
                    "tail_cols": stats["tail_cols"],
                }

        for j in range(args.nsamples):
            layer_out = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            outs[j] = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    return quantizers, layer_stats


def main():
    args = parse_args()
    if args.wbits != 4:
        raise ValueError("This prototype currently expects --wbits 4 as base setting.")
    if not args.enable_percentile_tail:
        raise ValueError("Set --enable-percentile-tail to run this prototype.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.custom_modeling_file:
        custom_model_cls = load_custom_model_class(args.custom_modeling_file)
        register_custom_model(custom_model_cls)
        print("registered custom class:", custom_model_cls)

    dtype = dtype_from_str(args.dtype)
    model = get_qwen3(args.model_dir, dtype=dtype)
    model.eval()

    init_state_dict_path = ""
    if args.init_state_dict:
        init_state_dict_path = str(Path(args.init_state_dict).resolve())
        loaded = torch.load(init_state_dict_path, map_location="cpu")
        if isinstance(loaded, dict) and "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
            loaded = loaded["state_dict"]
        model.load_state_dict(loaded, strict=True)
        print("initialized model from state_dict:", init_state_dict_path)

    if args.seqlen > model.seqlen:
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
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    dev = torch.device("cuda:0")

    tick = time.time()
    quantizers, layer_stats = qwen3_sequential_percentile_tail(model=model, dataloader=trainloader, dev=dev, args=args)
    total_sec = time.time() - tick

    weights_path = output_dir / "qwen3-4b-instruct-2507-percentile-tail.pt"
    torch.save(model.state_dict(), weights_path)

    q_max = int((2 ** (args.main_wbits - 1)) - 1)
    metadata = {
        "model_dir": str(Path(args.model_dir).resolve()),
        "enable_percentile_tail": args.enable_percentile_tail,
        "tail_ratio": args.tail_ratio,
        "tail_rank": args.tail_rank,
        "percentile_k": args.percentile_k,
        "q_max": q_max,
        "main_quant_type": "gptq" if args.use_gptq_main else "percentile_uniform",
        "tail_quant_type": args.tail_quant,
        "lambda_reg": args.lambda_reg,
        "enable_main_absorb": args.enable_main_absorb,
        "main_absorb_mode": args.main_absorb_mode,
        "main_absorb_budget_rule": args.main_absorb_budget_rule,
        "constraint_eps": args.constraint_eps,
        "use_gptq_main": args.use_gptq_main,
        "disable_tail_compensation": args.disable_tail_compensation,
        "nsamples": args.nsamples,
        "seqlen": model.seqlen,
        "groupsize": args.groupsize,
        "percdamp": args.percdamp,
        "act_order": args.act_order,
        "true_sequential": args.true_sequential,
        "dtype": args.dtype,
        "init_state_dict": init_state_dict_path,
        "calibration": source_info,
        "num_quantized_linear_layers": len(quantizers),
        "elapsed_seconds": total_sec,
        "weights_path": str(weights_path),
        "layer_stats": layer_stats,
    }
    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved weights:", weights_path)
    print("saved meta   :", meta_path)


if __name__ == "__main__":
    main()
