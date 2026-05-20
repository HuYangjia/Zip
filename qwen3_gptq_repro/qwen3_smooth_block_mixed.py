"""Qwen3 Smooth Block Mixed Quantization 入口脚本。

这个脚本是可直接运行的 Qwen3 smooth+quant pipeline。 调用  qwen3_gptq.py 中的
模型/校准数据加载工具。真正的 smooth+block 搜索和最终量化逻辑在
smooth_block_quant.calibrate_smooth_block_mixed_gptq(...) 中实现。

输入:
  1. Qwen3 模型目录, 由 --model-dir 指定, 必填。
  2. WikiText2 校准文本, 默认通过 qwen3_gptq.py 的加载逻辑获取; 如果已有
     本地数据, 可用 --local-wikitext2-dir 指定。
  3. 可选初始权重, 由 --init-state-dict 指定, 常用于从已有 state_dict 继续
     做 smooth+quant。
  4. 可选自定义 modeling 文件, 由 --custom-modeling-file 指定。
  5. 可选 JSON 覆盖文件:
       --alpha-grid-json 覆盖各组 smooth alpha 搜索网格。
       --module-groups-json 覆盖模块分组, 例如 attn_qkv / ffn_up_gate。

主要参数:
  --output-dir:
    保存量化权重、smooth groups、metadata 和可选诊断结果的目录。
  --nsamples / --seqlen / --seed:
    校准样本数量、序列长度和随机种子。
  --block-rows / --block-cols:
    block mask 的形状。当前脚本要求 --block-cols 固定为 128。
  --budget-ratio:
    block mask/second path 的预算比例, 范围 [0, 1]。
  --groupsize:
    权重/激活量化 group size, 只能是 32 / 64 / 128。
  --weight-bits / --act-bits:
    权重和激活 bit 数, 默认 W4A4。
  --second-path:
    第二路径模式, 支持 residual_int4 或 int8。
  --act-order:
    是否启用 GPTQ activation-order; 默认关闭, 显式加该参数才开启。
  --max-search-batches:
    smooth alpha 搜索、mask 搜索和 A4 recalibration 使用的最大校准 batch 数。
  --search-eval-batch-size / --gptq-batch-size / --output-error-batch-size:
    分别控制搜索评估、GPTQ Hessian 累积和 output error 统计的 batch chunk。
  --save-full-results:
    除 metadata 摘要外, 额外保存完整 results pt 文件。
  --save-layer-output-errors / --layer-output-error-path:
    是否保存每个 Linear 的 output error 诊断及其路径。

运行示例:
  cd /root/autodl-tmp/Zip/qwen3_gptq_repro
  python qwen3_smooth_block_mixed.py \
    --model-dir /root/autodl-tmp/model/Qwen3-4B-Instruct-2507 \
    --output-dir output/smooth_block_mixed_example \
    --nsamples 128 \
    --seqlen 1024 \
    --block-rows 16 \
    --block-cols 128 \
    --groupsize 64 \
    --budget-ratio 0.2 \
    --second-path residual_int4 \
    --act-order \
    --percdamp 0.01 \
    --max-search-batches 128 \
    --save-layer-output-errors

输出:
  output-dir/qwen3-smooth-block-mixed-state_dict.pt:
    smooth+quant 后的 FakeQuant 浮点权重 state_dict。
  output-dir/smooth_groups.pt:
    每组模块搜索得到的 smooth scale/alpha 等信息, 后续 Hessian 诊断脚本会读取它。
  output-dir/qwen3_smooth_block_mixed_metadata.json:
    运行参数、搜索摘要、最终量化摘要、smooth groups 摘要和校准数据来源。
  output-dir/layer_output_errors.json:
    如果指定 --save-layer-output-errors, 保存 layer/module 级 output error。
  output-dir/qwen3-smooth-block-mixed-results.pt:
    如果指定 --save-full-results, 保存完整 torch results dict。

与 qwen3_gptq_submatrix_mixed.py 的区别:
  qwen3_smooth_block_mixed.py:
    先搜索 SmoothQuant alpha, 对部分组 fuse smooth scale, 再做 block mask 选择
    和 mixed GPTQ/residual INT4 或 INT8 second path。主要产物包含 smooth_groups.pt。
  qwen3_gptq_submatrix_mixed.py:
    不做 smooth alpha 搜索, 而是直接在 GPTQ 中按子矩阵敏感度选择 INT8 blocks,
    其核心算法来自 gptq_submatrix_mixed.GPTQSubmatrixMixed。
"""

import argparse
import inspect
import json
import time
from pathlib import Path

import torch


DEFAULT_MODULE_GROUPS = {
    "attn_qkv": ["q_proj", "k_proj", "v_proj"],
    "attn_o": ["o_proj"],
    "ffn_up_gate": ["up_proj", "gate_proj"],
    "ffn_down": ["down_proj"],
}

DEFAULT_ALPHA_GRIDS = {
    "attn_qkv": [
        0.55,
        0.575,
        0.60,
        0.625,
        0.65,
        0.675,
        0.70,
        0.725,
        0.75,
        0.8,
        0.85,
        0.90,
        0.95,
        0.99,
    ],
    "ffn_up_gate": [
        0.60,
        0.65,
        0.70,
        0.725,
        0.75,
        0.775,
        0.80,
        0.825,
        0.85,
        0.875,
        0.90,
        0.95,
        0.99,
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smooth alpha search + block-mask selection + mixed GPTQ/residual INT4 for Qwen3."
    )
    parser.add_argument("--model-dir", type=str, required=True, help="Local model directory.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/root/autodl-tmp/Zip/qwen3_gptq_repro/output/smooth_block_mixed",
        help="Directory to save quantized artifacts.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=32)
    parser.add_argument("--seqlen", type=int, default=1024)
    parser.add_argument("--block-rows", type=int, default=128)
    parser.add_argument("--block-cols", type=int, default=128)
    parser.add_argument("--budget-ratio", type=float, default=0.05)
    parser.add_argument(
        "--max-search-batches",
        type=int,
        default=8,
        help="Maximum calibration batches to use for smooth/mask search and A4 recalibration.",
    )
    parser.add_argument("--groupsize", type=int, default=128, choices=[32, 64, 128])
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Load dtype for model weights.",
    )
    parser.add_argument("--weight-bits", type=int, default=4)
    parser.add_argument("--act-bits", type=int, default=4)
    parser.add_argument(
        "--second-path",
        type=str,
        default="residual_int4",
        choices=["residual_int4", "int8"],
        help="Second-path mode for final quantization.",
    )
    parser.add_argument(
        "--act-order",
        action="store_true",
        help="Enable activation-order GPTQ for smooth/mask search and final quantization.",
    )
    parser.add_argument("--custom-modeling-file", type=str, default="")
    parser.add_argument("--local-wikitext2-dir", type=str, default="")
    parser.add_argument(
        "--init-state-dict",
        type=str,
        default="",
        help="Optional state_dict path to load before smooth+quant.",
    )
    parser.add_argument(
        "--alpha-grid-json",
        type=str,
        default="",
        help="Optional JSON file overriding alpha grids. Shape: {group_name: [0.45, 0.5, ...]}",
    )
    parser.add_argument(
        "--module-groups-json",
        type=str,
        default="",
        help="Optional JSON file overriding module groups. Shape: {group_name: [suffix1, suffix2, ...]}",
    )
    parser.add_argument(
        "--save-full-results",
        action="store_true",
        help="Save the full torch results dict in addition to metadata summary.",
    )
    parser.add_argument(
        "--save-layer-output-errors",
        action="store_true",
        help="Save per-Linear output error stats measured during final quantization.",
    )
    parser.add_argument(
        "--layer-output-error-path",
        type=str,
        default="",
        help="Optional JSON path for --save-layer-output-errors. Defaults to output-dir/layer_output_errors.json.",
    )
    parser.add_argument(
        "--output-error-batch-size",
        type=int,
        default=1,
        help="Number of cached calibration samples per chunk when measuring layer output errors.",
    )
    parser.add_argument(
        "--search-eval-batch-size",
        type=int,
        default=1,
        help="Number of cached calibration samples per chunk during smooth/mask search proxy evaluation.",
    )
    parser.add_argument(
        "--gptq-batch-size",
        type=int,
        default=1,
        help="Number of cached calibration samples per chunk when accumulating GPTQ Hessians.",
    )
    args = parser.parse_args()

    if args.block_rows <= 0 or args.block_cols <= 0:
        parser.error("--block-rows and --block-cols must be positive.")
    if args.block_cols != 128:
        parser.error("--block-cols is fixed at 128; use --groupsize 32/64/128 to change quantization group size.")
    if not (0.0 <= args.budget_ratio <= 1.0):
        parser.error("--budget-ratio must be in [0.0, 1.0].")
    if args.output_error_batch_size <= 0:
        parser.error("--output-error-batch-size must be positive.")
    if args.search_eval_batch_size <= 0:
        parser.error("--search-eval-batch-size must be positive.")
    if args.gptq_batch_size <= 0:
        parser.error("--gptq-batch-size must be positive.")
    return args


def load_json_override(path_str: str, default_value: dict) -> dict:
    if not path_str:
        return default_value
    path = Path(path_str).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON override must be a dict: {path}")
    return data


def summarize_results(results: dict) -> dict:
    from smooth_block_quant import summarize_output_errors

    activation_quant = {}
    for module_name, stats in results.get("activation_quant", {}).items():
        activation_quant[module_name] = {
            "bits": stats["bits"],
            "group_size": stats["group_size"],
            "sym": stats.get("sym", True),
            "scale_min": stats["scale_min"],
            "scale_max": stats["scale_max"],
            "scale_mean": stats["scale_mean"],
        }
    smooth_groups = {}
    for group_key, stats in results.get("smooth_groups", {}).items():
        scale = stats.get("smooth_scale")
        smooth_groups[group_key] = {
            "group": stats.get("group"),
            "prefix": stats.get("prefix"),
            "norm_name": stats.get("norm_name"),
            "target_linears": stats.get("target_linears", []),
            "alpha": stats.get("alpha"),
            "fused_into_rmsnorm": bool(stats.get("fused_into_rmsnorm", False)),
            "scale_min": float(scale.min().item()) if torch.is_tensor(scale) else None,
            "scale_max": float(scale.max().item()) if torch.is_tensor(scale) else None,
            "scale_mean": float(scale.float().mean().item()) if torch.is_tensor(scale) else None,
        }
    return {
        "search": results.get("search", {}),
        "final": results.get("final", {}),
        "activation_quant": activation_quant,
        "smooth_groups": smooth_groups,
        "output_error": summarize_output_errors(results.get("output_error", {})),
    }


def main():
    args = parse_args()
    from transformers import AutoTokenizer

    from qwen3_gptq import (
        dtype_from_str,
        get_qwen3,
        get_wikitext2_or_fallback_loader,
        load_custom_model_class,
        register_custom_model,
    )
    from smooth_block_quant import (
        calibrate_smooth_block_mixed_gptq,
        save_output_error_report,
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.custom_modeling_file:
        custom_model_cls = load_custom_model_class(args.custom_modeling_file)
        register_custom_model(custom_model_cls)
        print("registered custom class:", custom_model_cls)
        print("registered from file  :", inspect.getfile(custom_model_cls))

    module_groups = load_json_override(args.module_groups_json, DEFAULT_MODULE_GROUPS)
    alpha_grids = load_json_override(args.alpha_grid_json, DEFAULT_ALPHA_GRIDS)

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
        raise RuntimeError("CUDA is required for this smooth+quant CLI.")
    dev = torch.device("cuda:0")
    model = model.to(dev)

    print("=" * 60)
    print("Starting Smooth Block Mixed Quantization ...")
    print(f"  block_shape=({args.block_rows}, {args.block_cols})")
    print(f"  groupsize={args.groupsize}")
    print(f"  budget_ratio={args.budget_ratio}")
    print(f"  second_path={args.second_path}")
    print(f"  max_search_batches={args.max_search_batches}")
    print(f"  search_eval_batch_size={args.search_eval_batch_size}")
    print(f"  gptq_batch_size={args.gptq_batch_size}")
    print(f"  nsamples={args.nsamples}, seqlen={args.seqlen}, dtype={args.dtype}")
    print("=" * 60)

    tick = time.time()
    results = calibrate_smooth_block_mixed_gptq(
        model=model,
        calibration_loader=trainloader,
        module_groups=module_groups,
        alpha_grids=alpha_grids,
        block_shape=(args.block_rows, args.block_cols),
        budget_ratio=args.budget_ratio,
        groupsize=args.groupsize,
        act_bits=args.act_bits,
        weight_bits=args.weight_bits,
        use_residual_second_path=(args.second_path == "residual_int4"),
        percdamp=args.percdamp,
        actorder=args.act_order,
        max_batches=args.max_search_batches,
        collect_output_errors=args.save_layer_output_errors,
        output_error_batch_size=args.output_error_batch_size,
        search_eval_batch_size=args.search_eval_batch_size,
        gptq_batch_size=args.gptq_batch_size,
    )
    total_sec = time.time() - tick
    print(f"smooth+quant time: {total_sec:.2f}s")

    model = model.cpu()
    torch.cuda.empty_cache()

    weights_path = output_dir / "qwen3-smooth-block-mixed-state_dict.pt"
    torch.save(model.state_dict(), weights_path)

    smooth_groups_path = output_dir / "smooth_groups.pt"
    torch.save(results.get("smooth_groups", {}), smooth_groups_path)

    results_path = output_dir / "qwen3-smooth-block-mixed-results.pt"
    if args.save_full_results:
        torch.save(results, results_path)

    output_error_path = (
        Path(args.layer_output_error_path).resolve()
        if args.layer_output_error_path
        else output_dir / "layer_output_errors.json"
    )
    if args.save_layer_output_errors:
        save_output_error_report(output_error_path, results.get("output_error", {}))

    metadata = {
        "model_dir": str(Path(args.model_dir).resolve()),
        "dtype": args.dtype,
        "seed": args.seed,
        "nsamples": args.nsamples,
        "seqlen": model.seqlen,
        "block_shape": [args.block_rows, args.block_cols],
        "groupsize": args.groupsize,
        "residual_groupsize": args.groupsize if args.second_path == "residual_int4" else None,
        "budget_ratio": args.budget_ratio,
        "max_search_batches": args.max_search_batches,
        "search_eval_batch_size": args.search_eval_batch_size,
        "gptq_batch_size": args.gptq_batch_size,
        "percdamp": args.percdamp,
        "weight_bits": args.weight_bits,
        "act_bits": args.act_bits,
        "second_path": args.second_path,
        "act_order": args.act_order,
        "smooth_granularity": "group_shared_rmsnorm_fused",
        "smooth_enabled_groups": ["attn_qkv", "ffn_up_gate"],
        "smooth_disabled_groups": ["attn_o", "ffn_down"],
        "activation_quant_runtime": {
            "format": f"int4-g{args.groupsize}-symmetric",
            "group_size": args.groupsize,
            "per_token": True,
            "sym": True,
        },
        "save_layer_output_errors": args.save_layer_output_errors,
        "layer_output_error_path": str(output_error_path) if args.save_layer_output_errors else "",
        "output_error_batch_size": args.output_error_batch_size,
        "module_groups": module_groups,
        "alpha_grids": alpha_grids,
        "init_state_dict": {
            "used": bool(args.init_state_dict),
            "path": init_state_dict_path,
        },
        "calibration": source_info,
        "elapsed_seconds": total_sec,
        "weights_path": str(weights_path),
        "smooth_groups_path": str(smooth_groups_path),
        "results_path": str(results_path) if args.save_full_results else "",
        "results_summary": summarize_results(results),
    }
    meta_path = output_dir / "qwen3_smooth_block_mixed_metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("saved weights:", weights_path)
    print("saved smooth groups:", smooth_groups_path)
    if args.save_full_results:
        print("saved results:", results_path)
    if args.save_layer_output_errors:
        print("saved layer output errors:", output_error_path)
    print("saved meta   :", meta_path)


if __name__ == "__main__":
    main()
