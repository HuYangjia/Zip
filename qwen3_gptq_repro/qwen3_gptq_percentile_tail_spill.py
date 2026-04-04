"""
改进版 Mode A 实验脚本 —— Percentile-GPTQ with Tail Spill

核心流程：
1. 加载模型（可选加载 smooth 后的权重）
2. 逐层处理：
   a. 用 Catcher 捕获校准输入
   b. 对每个 Linear 层：
      - 计算 tail_start = d_in - tail_rank
      - 创建 GPTQTailSpill 实例，配置 PercentileQuantizer
      - 调用 add_batch 收集 Hessian
      - 调用 fasterquant(tail_start=tail_start)：main 列 4-bit 量化，tail 列跳过
      - 提取 tail 列（已吸收误差），做 INT8 量化
      - 合并 main + tail 权重
3. 保存量化后的权重和 metadata
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

# 导入修改版 GPTQ 核心类
from gptq_tail_spill import (
    GPTQTailSpill,
    PercentileQuantizer,
    quantize_tail_int8,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GPTQ_DIR = REPO_ROOT / "gptq"
if str(GPTQ_DIR) not in sys.path:
    sys.path.insert(0, str(GPTQ_DIR))

from modelutils import find_layers  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="改进版 Mode A: Percentile-GPTQ with Tail Spill — "
        "在 GPTQ 逐列迭代中让 main 列量化误差自然溢出到 tail 列。"
    )
    # 基础参数（与现有脚本兼容）
    parser.add_argument("--model-dir", type=str,
                        default="/home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--output-dir", type=str,
                        default="/home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output/exp_percentile_tail_spill")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=32)
    parser.add_argument("--seqlen", type=int, default=1024)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--groupsize", type=int, default=128)
    parser.add_argument("--sym", action="store_true", help="启用对称量化。")
    parser.add_argument("--act-order", action="store_true", help="启用 activation-order GPTQ 启发式。")
    parser.add_argument("--true-sequential", action="store_true", help="启用 true sequential 分组量化。")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--custom-modeling-file", type=str, default="")
    parser.add_argument("--local-wikitext2-dir", type=str, default="")
    parser.add_argument("--init-state-dict", type=str, default="",
                        help="可选的 state_dict 路径（如 smooth 后的权重）。")

    # 新增参数：Tail Spill 相关
    parser.add_argument("--tail-rank", type=int, default=16,
                        help="Tail 列数（绝对值）。优先于 --tail-ratio。")
    parser.add_argument("--tail-ratio", type=float, default=0.05,
                        help="Tail 列比例，仅在 --tail-rank=0 时使用。")
    parser.add_argument("--percentile-k", type=float, default=75.0,
                        help="Percentile 百分位数（0-100），用于确定量化 scale。")
    parser.add_argument("--use-standard-quantizer", action="store_true",
                        help="使用标准 min/max Quantizer 而非 PercentileQuantizer（用于对比实验）。")
    return parser.parse_args()


def _resolve_tail_cols(in_features: int, tail_ratio: float, tail_rank: int):
    """计算 tail 列数和 tail_start 位置。"""
    if tail_rank > 0:
        tail_cols = min(max(1, int(tail_rank)), max(1, in_features - 1))
    else:
        tail_cols = max(1, int(round(in_features * tail_ratio)))
        tail_cols = min(max(1, in_features - 1), tail_cols)
    return in_features - tail_cols, tail_cols


@torch.no_grad()
def qwen3_sequential_percentile_tail_spill(model, dataloader, dev: torch.device, args):
    """
    逐层量化主流程：Percentile-GPTQ with Tail Spill。

    对每个 Linear 层：
    1. 收集 Hessian 信息
    2. 调用修改版 fasterquant，main 列 4-bit 量化，tail 列跳过
    3. 提取 tail 列（已吸收误差），做 INT8 量化
    4. 合并 main + tail 权重
    """
    print("=" * 60)
    print("Starting Percentile-GPTQ with Tail Spill ...")
    print(f"  percentile_k={args.percentile_k}, tail_rank={args.tail_rank}, "
          f"tail_ratio={args.tail_ratio}")
    print(f"  groupsize={args.groupsize}, percdamp={args.percdamp}, "
          f"act_order={args.act_order}")
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

            # 创建 GPTQTailSpill 实例
            gptq = {}
            for name in subset:
                gptq[name] = GPTQTailSpill(subset[name])
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
                tail_start, tail_cols = _resolve_tail_cols(
                    d_in, args.tail_ratio, args.tail_rank
                )

                # 保存原始权重用于诊断
                w_orig = linear.weight.data.float().cpu().clone()

                print(f"\nLayer {i} -> {name} (d_in={d_in}, tail_start={tail_start}, "
                      f"tail_cols={tail_cols})")

                # 获取 percentile clip ratio（如果使用 PercentileQuantizer）
                clip_ratio = 0.0

                # 调用修改版 fasterquant
                spill_stats = gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    groupsize=args.groupsize,
                    actorder=args.act_order,
                    static_groups=False,
                    tail_start=tail_start,
                )

                # 获取 clip ratio
                if hasattr(gptq[name].quantizer, 'clip_ratio'):
                    clip_ratio = gptq[name].quantizer.clip_ratio

                # 此时 linear.weight.data 中：
                # - main 列（[:, :tail_start]）已被 4-bit 量化
                # - tail 列（[:, tail_start:]）保留浮点值（已吸收 main 误差）
                w_after_gptq = linear.weight.data.float().cpu().clone()

                # 计算 main 区域量化误差
                main_orig = w_orig[:, :tail_start]
                main_quantized = w_after_gptq[:, :tail_start]
                main_error_norm = float(
                    (main_quantized - main_orig).norm().item()
                )

                # 计算 tail 区域权重变化量（吸收了多少 main 误差）
                tail_orig = w_orig[:, tail_start:]
                tail_after_gptq = w_after_gptq[:, tail_start:]
                tail_absorbed = float(
                    (tail_after_gptq - tail_orig).norm().item()
                )
                tail_absorbed_per_row = (tail_after_gptq - tail_orig).norm(dim=1)

                # 对 tail 列做 INT8 量化
                w_tail_q, tail_int8_stats = quantize_tail_int8(
                    tail_after_gptq.to(dev)
                )

                # 合并 main + tail 权重
                merged = w_after_gptq.clone()
                merged[:, tail_start:] = w_tail_q.cpu()

                # 计算最终整体残差
                final_residual_norm = float(
                    (merged - w_orig).norm().item()
                )

                # 写回 Linear 层
                linear.weight.data.copy_(
                    merged.to(device=linear.weight.device, dtype=linear.weight.dtype)
                )

                # 收集诊断统计
                stats = {
                    "d_in": d_in,
                    "d_out": d_out,
                    "tail_start": tail_start,
                    "tail_cols": tail_cols,
                    "percentile_k": args.percentile_k,
                    "percentile_clip_ratio": clip_ratio,
                    "main_error_norm": main_error_norm,
                    "tail_absorbed_norm": tail_absorbed,
                    "tail_absorbed_per_row_mean": float(tail_absorbed_per_row.mean().item()),
                    "tail_absorbed_per_row_max": float(tail_absorbed_per_row.max().item()),
                    "tail_int8": tail_int8_stats,
                    "final_residual_norm": final_residual_norm,
                    "gptq_loss": spill_stats["gptq_loss"],
                    "n_main_quantized": spill_stats["n_main_quantized"],
                    "n_tail_skipped": spill_stats["n_tail_skipped"],
                }

                print(f"  main_error_norm={main_error_norm:.4f}, "
                      f"tail_absorbed={tail_absorbed:.4f}, "
                      f"tail_int8_err={tail_int8_stats['quant_error_norm_mean']:.6f}")
                print(f"  final_residual_norm={final_residual_norm:.4f}, "
                      f"clip_ratio={clip_ratio:.4f}")

                layer_stats[f"model.layers.{i}.{name}"] = stats
                quantizers[f"model.layers.{i}.{name}"] = {
                    "bits_main": args.wbits,
                    "tail_quant": "int8",
                    "tail_cols": tail_cols,
                    "percentile_k": args.percentile_k,
                }
                gptq[name].free()

        # 用量化后的权重重新计算 outs，确保下一层校准数据准确
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
        raise ValueError("当前仅支持 4-bit 量化。请设置 --wbits 4。")

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
                f"加载 --init-state-dict {init_state_dict_path} 失败: {exc}"
            ) from exc
        print("initialized model from state_dict:", init_state_dict_path)

    if args.seqlen > model.seqlen:
        print(f"[warn] --seqlen {args.seqlen} > model.seqlen {model.seqlen}; 使用 model.seqlen。")
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
        print("[fallback] 加载失败，使用临时生成文本。")
        print("[fallback] reason:", source_info["reason"])

    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 环境。")
    dev = torch.device("cuda:0")

    tick = time.time()
    quantizers, layer_stats = qwen3_sequential_percentile_tail_spill(
        model=model,
        dataloader=trainloader,
        dev=dev,
        args=args,
    )
    total_sec = time.time() - tick
    print(f"\n量化总耗时: {total_sec:.2f}s")

    # 保存权重
    weights_path = output_dir / "qwen3-4b-instruct-2507-gptq-4bit.pt"
    torch.save(model.state_dict(), weights_path)

    # 保存 metadata
    metadata = {
        "method": "percentile_gptq_tail_spill",
        "description": "改进版 Mode A: GPTQ 逐列迭代中 main 列 4-bit 量化，"
                       "tail 列跳过量化（吸收误差），最后 tail 做 INT8。",
        "model_dir": str(Path(args.model_dir).resolve()),
        "wbits": args.wbits,
        "percentile_k": args.percentile_k,
        "use_standard_quantizer": args.use_standard_quantizer,
        "tail_rank": args.tail_rank,
        "tail_ratio": args.tail_ratio,
        "nsamples": args.nsamples,
        "seqlen": model.seqlen,
        "percdamp": args.percdamp,
        "groupsize": args.groupsize,
        "sym": args.sym,
        "act_order": args.act_order,
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
