"""
Tail Absorb 主脚本 —— GPTQ Tail 列参与 INT8 量化 + FakeQuant 验证

核心流程：
1. 加载模型（可选加载 smooth 后的权重）
2. 逐层处理：
   a. 用 Catcher 捕获校准输入
   b. 对每个 Linear 层：
      - 创建 GPTQTailAbsorb 实例，配置 PercentileQuantizer
      - 调用 add_batch 收集 Hessian
      - 调用 fasterquant(tail_rank=...)：
        main 列 4-bit FakeQuant，tail 列 INT8 FakeQuant
        两种列都正常传播误差，GPTQ 补偿链完整
      - fasterquant 返回后，layer.weight.data 已是 FakeQuant 浮点权重
3. 保存 FakeQuant 浮点权重和 metadata
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer

# 从 qwen3_gptq.py 导入公共工具函数
from qwen3_gptq import (
    dtype_from_str,
    get_qwen3,
    get_wikitext2_or_fallback_loader,
    load_custom_model_class,
    register_custom_model,
)

# 导入 Tail Absorb 核心类
from gptq_tail_absorb import (
    GPTQTailAbsorb,
    PercentileQuantizer,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GPTQ_DIR = REPO_ROOT / "gptq"
if str(GPTQ_DIR) not in sys.path:
    sys.path.insert(0, str(GPTQ_DIR))

from modelutils import find_layers  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tail Absorb: GPTQ with tail columns quantized as INT8 (FakeQuant). "
        "Tail columns are determined by actorder sorting (last tail_rank columns)."
    )
    # 基础参数（与现有脚本兼容）
    parser.add_argument("--model-dir", type=str,
                        default="/home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--output-dir", type=str,
                        default="/home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output/exp_tail_absorb")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=32)
    parser.add_argument("--seqlen", type=int, default=1024)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--groupsize", type=int, default=128)
    parser.add_argument("--sym", action="store_true", help="Enable symmetric quantization.")
    # act-order 默认启用（因为 tail 列定义依赖 Hessian 排序）
    parser.add_argument("--no-act-order", action="store_true",
                        help="Disable activation-order GPTQ heuristic (default: enabled).")
    parser.add_argument("--true-sequential", action="store_true",
                        help="Enable true sequential group quantization.")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--custom-modeling-file", type=str, default="")
    parser.add_argument("--local-wikitext2-dir", type=str, default="")
    parser.add_argument("--init-state-dict", type=str, default="",
                        help="Optional state_dict path (e.g. SmoothQuant preprocessed weights).")

    # Tail Absorb 相关参数
    parser.add_argument("--tail-rank", type=int, default=16,
                        help="Number of tail columns (absolute). "
                        "After actorder sorting, the last tail_rank columns use INT8.")
    parser.add_argument("--percentile-k", type=float, default=75.0,
                        help="Percentile (0-100) for determining quantization scale of main columns.")
    parser.add_argument("--use-standard-quantizer", action="store_true",
                        help="Use standard min/max Quantizer instead of PercentileQuantizer (for ablation).")
    parser.add_argument("--head-absorb", action="store_true",
                        help="Enable Head Absorb mode (V8): INT8 quantize the most important columns "
                        "(first tail_rank columns after actorder sorting) instead of the least important ones.")
    return parser.parse_args()


@torch.no_grad()
def qwen3_sequential_tail_absorb(model, dataloader, dev: torch.device, args):
    """
    逐层量化主流程：Tail Absorb — GPTQ with INT8 tail columns。

    对每个 Linear 层：
    1. 收集 Hessian 信息
    2. 调用 GPTQTailAbsorb.fasterquant：
       - main 列 4-bit FakeQuant
       - tail 列 INT8 FakeQuant
       - 两种列都正常传播误差
    3. fasterquant 返回后，layer.weight.data 已是 FakeQuant 浮点权重（列顺序已还原）
    """
    # act_order 默认启用，除非用户显式禁用
    act_order = not args.no_act_order

    head_absorb = getattr(args, 'head_absorb', False)
    mode_str = "Head Absorb (V8)" if head_absorb else "Tail Absorb (V7)"

    print("=" * 60)
    print(f"Starting {mode_str} (GPTQ + INT8 FakeQuant) ...")
    print(f"  percentile_k={args.percentile_k}, tail_rank={args.tail_rank}")
    print(f"  head_absorb={head_absorb}")
    print(f"  groupsize={args.groupsize}, percdamp={args.percdamp}, "
          f"act_order={act_order}")
    print(f"  use_standard_quantizer={args.use_standard_quantizer}")
    print("=" * 60)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # 将 embedding 和 norm 移到 GPU
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size),
        dtype=dtype, device=dev,
    )
    cache = {
        "i": 0,
        "attention_mask": None,
        "position_ids": None,
        "cache_position": None,
        "position_embeddings": None,
    }

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

    print("Calibration capture ready.")

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

            # 创建 GPTQTailAbsorb 实例
            gptq = {}
            for name in subset:
                gptq[name] = GPTQTailAbsorb(subset[name])
                if args.use_standard_quantizer:
                    from quant import Quantizer  # noqa: E402
                    gptq[name].quantizer = Quantizer()
                else:
                    gptq[name].quantizer = PercentileQuantizer(
                        percentile_k=args.percentile_k
                    )
                gptq[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = [
                subset[name].register_forward_hook(add_batch(name))
                for name in subset
            ]
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
                linear = subset[name]
                d_out, d_in = linear.weight.shape

                # 保存原始权重用于诊断
                w_orig = linear.weight.data.float().cpu().clone()

                print(f"\nLayer {i} -> {name} (d_in={d_in}, d_out={d_out}, "
                      f"tail_rank={args.tail_rank})")

                # 调用 Tail/Head Absorb 版 fasterquant
                # INT8 量化已在 fasterquant 内部完成，无需外部后处理
                absorb_stats = gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    groupsize=args.groupsize,
                    actorder=act_order,
                    static_groups=False,
                    tail_rank=args.tail_rank,
                    head_absorb=head_absorb,
                )

                # 获取 clip ratio（如果使用 PercentileQuantizer）
                clip_ratio = 0.0
                if hasattr(gptq[name].quantizer, 'clip_ratio'):
                    clip_ratio = gptq[name].quantizer.clip_ratio

                # fasterquant 返回后，layer.weight.data 已是 FakeQuant 浮点权重
                w_fakequant = linear.weight.data.float().cpu().clone()

                # 计算整体残差
                final_residual_norm = float(
                    (w_fakequant - w_orig).norm().item()
                )

                # 收集诊断统计
                stats = {
                    "d_in": d_in,
                    "d_out": d_out,
                    "tail_rank": args.tail_rank,
                    "percentile_k": args.percentile_k,
                    "percentile_clip_ratio": clip_ratio,
                    "final_residual_norm": final_residual_norm,
                    "gptq_loss": absorb_stats["gptq_loss"],
                    "n_main_quantized": absorb_stats["n_main_quantized"],
                    "n_tail_int8": absorb_stats["n_tail_int8"],
                    "elapsed_seconds": absorb_stats["elapsed_seconds"],
                }

                print(f"  final_residual_norm={final_residual_norm:.4f}, "
                      f"gptq_loss={absorb_stats['gptq_loss']:.4f}, "
                      f"clip_ratio={clip_ratio:.4f}")

                layer_stats[f"model.layers.{i}.{name}"] = stats
                quantizers[f"model.layers.{i}.{name}"] = {
                    "bits_main": args.wbits,
                    "tail_quant": "int8",
                    "tail_rank": args.tail_rank,
                    "percentile_k": args.percentile_k,
                }
                gptq[name].free()

        # 用 FakeQuant 后的权重重新计算 outs，确保下一层校准数据准确
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
        raise ValueError("Only 4-bit quantization is supported. Please set --wbits 4.")

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
        try:
            model.load_state_dict(loaded, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to load --init-state-dict {init_state_dict_path}: {exc}"
            ) from exc
        print("initialized model from state_dict:", init_state_dict_path)

    if args.seqlen > model.seqlen:
        print(f"[warn] --seqlen {args.seqlen} > model.seqlen {model.seqlen}; using model.seqlen.")
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
        print("[fallback] loading failed, using temporary generated text.")
        print("[fallback] reason:", source_info["reason"])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA environment required.")
    dev = torch.device("cuda:0")

    tick = time.time()
    quantizers, layer_stats = qwen3_sequential_tail_absorb(
        model=model,
        dataloader=trainloader,
        dev=dev,
        args=args,
    )
    total_sec = time.time() - tick
    print(f"\nTotal quantization time: {total_sec:.2f}s")

    # 保存 FakeQuant 浮点权重
    weights_path = output_dir / "qwen3-4b-instruct-2507-gptq-4bit.pt"
    torch.save(model.state_dict(), weights_path)

    # act_order 实际值
    act_order = not args.no_act_order

    # 保存 metadata
    head_absorb = getattr(args, 'head_absorb', False)
    if head_absorb:
        method_name = "head_absorb"
        method_desc = ("Head Absorb (V8): GPTQ with main columns 4-bit FakeQuant "
                       "and head columns INT8 FakeQuant. "
                       "Head columns are the first tail_rank columns after actorder sorting "
                       "(highest Hessian diagonal = most important activations). "
                       "Both main and head columns propagate quantization error normally.")
    else:
        method_name = "tail_absorb"
        method_desc = ("Tail Absorb (V7): GPTQ with main columns 4-bit FakeQuant "
                       "and tail columns INT8 FakeQuant. "
                       "Tail columns are the last tail_rank columns after actorder sorting "
                       "(lowest Hessian diagonal = least important activations). "
                       "Both main and tail columns propagate quantization error normally.")
    metadata = {
        "method": method_name,
        "description": method_desc,
        "model_dir": str(Path(args.model_dir).resolve()),
        "wbits": args.wbits,
        "percentile_k": args.percentile_k,
        "use_standard_quantizer": args.use_standard_quantizer,
        "head_absorb": head_absorb,
        "tail_rank": args.tail_rank,
        "nsamples": args.nsamples,
        "seqlen": model.seqlen,
        "percdamp": args.percdamp,
        "groupsize": args.groupsize,
        "sym": args.sym,
        "act_order": act_order,
        "true_sequential": args.true_sequential,
        "dtype": args.dtype,
        "init_state_dict": {
            "used": bool(args.init_state_dict),
            "path": init_state_dict_path,
        },
        "calibration": source_info,
        "num_quantized_linear_layers": len(quantizers),
        "elapsed_seconds": total_sec,
        "weights_path": str(weights_path),
        "layer_stats": layer_stats,
    }
    meta_path = output_dir / "metadata.json"
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nsaved weights: {weights_path}")
    print(f"saved meta   : {meta_path}")


if __name__ == "__main__":
    main()
