#!/usr/bin/env python3
"""
WikiText-2 Perplexity 评测脚本（单模型粒度）

用法示例：
  # 评测 FP16 基线
  python benchmark/eval_ppl.py --model-dir /path/to/Qwen3-4B --label fp16_baseline

  # 评测量化变体
  python benchmark/eval_ppl.py --model-dir /path/to/Qwen3-4B \
      --quant-weights output/gptq_from_raw/qwen3-4b-instruct-2507-gptq-4bit.pt \
      --label gptq_4bit

  # 使用本地 WikiText-2 数据
  python benchmark/eval_ppl.py --model-dir /path/to/Qwen3-4B \
      --local-wikitext2-dir /path/to/wikitext2_local \
      --label fp16_baseline
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# 1. 命令行参数解析
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="WikiText-2 Perplexity 评测（单模型粒度）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="HuggingFace 模型目录（原始 pretrained 权重）。",
    )
    parser.add_argument(
        "--quant-weights",
        type=str,
        default="",
        help="量化后的 state_dict (.pt) 路径。不指定则评测 FP16 原始模型。",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="unnamed",
        help="本次评测的标签，用于标识结果（默认 'unnamed'）。",
    )
    parser.add_argument(
        "--seqlen",
        type=int,
        default=2048,
        help="滑动窗口大小（默认 2048）。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/benchmark",
        help="结果保存目录（默认 ./output/benchmark）。",
    )
    parser.add_argument(
        "--local-wikitext2-dir",
        type=str,
        default="",
        help="本地 WikiText-2 数据集目录（可选，离线环境使用）。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="随机种子（默认 0）。",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="模型加载精度（默认 float16）。",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 2. 模型加载
# ---------------------------------------------------------------------------

def dtype_from_str(name: str) -> torch.dtype:
    """将字符串转换为 torch.dtype。"""
    mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    return mapping.get(name, torch.float32)


def load_model(model_dir: str, quant_weights: str, dtype: torch.dtype):
    """
    加载模型。

    - 若 quant_weights 为空，直接加载 FP16 原始模型。
    - 若 quant_weights 非空，先加载原始模型结构，再用量化权重替换 state_dict。
    """
    print(f"[load] 加载模型结构: {model_dir}")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype)
    except TypeError:
        # 兼容旧版 transformers
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype)

    if quant_weights:
        print(f"[load] 加载量化权重: {quant_weights}")
        state_dict = torch.load(quant_weights, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"[warn] missing_keys: {len(missing_keys)} 个")
            for k in missing_keys[:5]:
                print(f"       - {k}")
            if len(missing_keys) > 5:
                print(f"       ... 及其他 {len(missing_keys) - 5} 个")
        if unexpected_keys:
            print(f"[warn] unexpected_keys: {len(unexpected_keys)} 个")
            for k in unexpected_keys[:5]:
                print(f"       - {k}")
            if len(unexpected_keys) > 5:
                print(f"       ... 及其他 {len(unexpected_keys) - 5} 个")
    else:
        print("[load] 未指定量化权重，使用 FP16 原始模型。")

    # 设置 seqlen，与 qwen3_gptq.py 中 get_qwen3() 保持一致
    max_seq = getattr(model.config, "max_position_embeddings", 2048)
    model.seqlen = min(2048, int(max_seq))

    return model


# ---------------------------------------------------------------------------
# 3. WikiText-2 test set 数据加载
# ---------------------------------------------------------------------------

def load_wikitext2_test(tokenizer, local_wikitext2_dir: str = ""):
    """
    加载 WikiText-2 test set 并编码为 input_ids。

    优先本地加载，若失败则在线加载。
    评测不使用 fallback 生成文本（与校准不同）。

    Returns:
        input_ids: torch.Tensor, shape [1, total_tokens]
        source: str, 数据来源描述
    """
    # 尝试从本地加载
    if local_wikitext2_dir:
        try:
            source = f"local: {local_wikitext2_dir}"
            print(f"[data] 尝试从本地加载 WikiText-2 test set: {local_wikitext2_dir}")
            input_ids = _load_local_wikitext2_test(tokenizer, local_wikitext2_dir)
            print(f"[data] 本地加载成功，共 {input_ids.shape[1]} 个 token")
            return input_ids, source
        except Exception as exc:
            print(f"[data] 本地加载失败: {exc}")
            print("[data] 尝试在线加载...")

    # 尝试在线加载
    try:
        from datasets import load_dataset

        print("[data] 从 HuggingFace 在线加载 WikiText-2 test set...")
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        joined_text = "\n\n".join(testdata["text"])
        encodings = tokenizer(joined_text, return_tensors="pt")
        input_ids = encodings.input_ids
        source = "online: huggingface wikitext-2-raw-v1 test"
        print(f"[data] 在线加载成功，共 {input_ids.shape[1]} 个 token")
        return input_ids, source
    except Exception as exc:
        online_err = str(exc)

    # 两种方式均失败
    print(f"[error] WikiText-2 test set 加载失败！")
    print(f"  在线加载错误: {online_err}")
    if local_wikitext2_dir:
        print(f"  本地目录: {local_wikitext2_dir}")
    else:
        print(f"  未指定 --local-wikitext2-dir，无法尝试本地加载。")
    print(f"  请确保网络可用，或通过 --local-wikitext2-dir 指定本地数据集目录。")
    sys.exit(1)


def _load_local_wikitext2_test(tokenizer, local_dir: str):
    """从本地目录加载 WikiText-2 test split。"""
    from datasets import load_dataset, load_from_disk

    local_path = Path(local_dir).resolve()

    # 方式 1：arrow 文件
    test_arrow = local_path / "wikitext-test.arrow"
    if test_arrow.exists():
        ds = load_dataset("arrow", data_files={"test": str(test_arrow)}, split="test")
        text_col = "text" if "text" in ds.column_names else ds.column_names[0]
        joined_text = "\n\n".join(ds[text_col])
        encodings = tokenizer(joined_text, return_tensors="pt")
        return encodings.input_ids

    # 方式 2：load_from_disk（HuggingFace DatasetDict 格式）
    loaded = load_from_disk(str(local_path))
    if hasattr(loaded, "keys") and "test" in loaded:
        test_ds = loaded["test"]
    elif hasattr(loaded, "keys") and "train" in loaded:
        # 如果只有 train split，退而求其次（不推荐）
        print("[warn] 本地数据集无 test split，使用 train split（不推荐）。")
        test_ds = loaded["train"]
    else:
        test_ds = loaded

    text_col = "text" if "text" in test_ds.column_names else test_ds.column_names[0]
    joined_text = "\n\n".join(test_ds[text_col])
    encodings = tokenizer(joined_text, return_tensors="pt")
    return encodings.input_ids


# ---------------------------------------------------------------------------
# 4. 滑动窗口 Perplexity 计算
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_ppl(model, input_ids: torch.Tensor, seqlen: int, device: torch.device):
    """
    使用非重叠滑动窗口计算 Perplexity。

    将 input_ids 按 seqlen 切分为不重叠的窗口，对每个窗口做前向传播，
    累加 cross-entropy loss，最终 PPL = exp(total_nll / total_tokens)。

    Args:
        model: 已加载的 CausalLM 模型
        input_ids: [1, total_tokens] 的 token 张量
        seqlen: 滑动窗口大小
        device: 计算设备

    Returns:
        ppl: float, perplexity 值
        total_tokens: int, 参与计算的 token 总数
    """
    model.eval()
    model.to(device)

    total_tokens = input_ids.shape[1]
    # 计算完整窗口数量（丢弃末尾不足一个窗口的 token）
    n_windows = total_tokens // seqlen
    if n_windows == 0:
        raise ValueError(
            f"test set token 数 ({total_tokens}) 小于 seqlen ({seqlen})，无法评测。"
        )

    total_nll = 0.0
    total_count = 0

    print(f"[eval] 开始评测: {n_windows} 个窗口, seqlen={seqlen}, 总 token={total_tokens}")

    for i in range(n_windows):
        start = i * seqlen
        end = start + seqlen
        window_ids = input_ids[:, start:end].to(device)  # [1, seqlen]

        # 前向传播
        outputs = model(window_ids)
        logits = outputs.logits  # [1, seqlen, vocab_size]

        # 计算 cross-entropy loss
        # logits[:, :-1, :] 预测 targets[:, 1:]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = window_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )

        total_nll += loss.item()
        total_count += shift_labels.numel()

        # 进度打印
        if (i + 1) % 10 == 0 or (i + 1) == n_windows:
            current_ppl = math.exp(total_nll / total_count)
            print(f"  窗口 {i + 1}/{n_windows}  累计 PPL = {current_ppl:.4f}")

    ppl = math.exp(total_nll / total_count)
    return ppl, total_count


# ---------------------------------------------------------------------------
# 5. 结果输出与保存
# ---------------------------------------------------------------------------

def print_results(label: str, ppl: float, total_tokens: int, elapsed: float):
    """格式化输出评测结果到终端。"""
    print()
    print("=" * 60)
    print(f"  评测结果")
    print("=" * 60)
    print(f"  标签 (label)     : {label}")
    print(f"  Perplexity (PPL) : {ppl:.4f}")
    print(f"  评测 token 数    : {total_tokens}")
    print(f"  耗时             : {elapsed:.2f} 秒")
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
):
    """将评测结果保存为 JSON 文件。"""
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
        "data_source": data_source,
        "timestamp": datetime.now().isoformat(),
    }

    json_file = out_path / f"ppl_{label}.json"
    json_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] JSON 结果已保存: {json_file}")
    return result


def append_results_txt(output_dir: str, result: dict):
    """
    将评测结果追加写入 results.txt 汇总文件。

    每次评测追加一条记录，方便在一个文件中纵览所有变体的 PPL 对比。
    文件格式：表头 + 对齐的文本行。
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    txt_file = out_path / "results.txt"

    # 表头（仅在文件不存在或为空时写入）
    header = (
        f"{'Label':<30s}  {'PPL':>10s}  {'Tokens':>10s}  "
        f"{'Time(s)':>10s}  {'Dtype':<10s}  {'Timestamp':<26s}  {'Quant Weights'}"
    )
    separator = "-" * len(header)

    write_header = not txt_file.exists() or txt_file.stat().st_size == 0

    quant_w = result.get("quant_weights", "")
    if not quant_w:
        quant_w = "(FP16 原始模型)"

    line = (
        f"{result['label']:<30s}  {result['ppl']:>10.4f}  {result['total_tokens']:>10d}  "
        f"{result['elapsed_seconds']:>10.2f}  {result['dtype']:<10s}  "
        f"{result['timestamp']:<26s}  {quant_w}"
    )

    with open(txt_file, "a", encoding="utf-8") as f:
        if write_header:
            f.write("WikiText-2 Perplexity Benchmark Results\n")
            f.write(f"{'=' * len(header)}\n")
            f.write(header + "\n")
            f.write(separator + "\n")
        f.write(line + "\n")

    print(f"[save] 结果已追加到: {txt_file}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # 固定随机种子
    torch.manual_seed(args.seed)

    # 设备检测
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"[env] 使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[env] CUDA 不可用，使用 CPU（速度会很慢）。")

    dtype = dtype_from_str(args.dtype)

    # 加载 tokenizer
    print(f"[load] 加载 tokenizer: {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)

    # 加载模型
    model = load_model(args.model_dir, args.quant_weights, dtype)

    # 调整 seqlen
    seqlen = min(args.seqlen, model.seqlen)
    if seqlen != args.seqlen:
        print(f"[warn] --seqlen {args.seqlen} > model.seqlen {model.seqlen}，使用 {seqlen}")

    # 加载 WikiText-2 test set
    input_ids, data_source = load_wikitext2_test(tokenizer, args.local_wikitext2_dir)

    # 评测
    tick = time.time()
    ppl, total_tokens = evaluate_ppl(model, input_ids, seqlen, device)
    elapsed = time.time() - tick

    # 输出结果到终端
    print_results(args.label, ppl, total_tokens, elapsed)

    # 保存 JSON（单次结果）
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
    )

    # 追加写入 results.txt（汇总文件）
    append_results_txt(args.output_dir, result)


if __name__ == "__main__":
    main()
