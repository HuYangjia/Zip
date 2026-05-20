#!/usr/bin/env python3
"""Diagnose whether W4A4 error concentrates in Hessian-sensitive blocks.

输入:
  1. 预训练模型目录, 由 --model-dir 控制, 默认读取
     /root/autodl-tmp/model/Qwen3-4B-Instruct-2507。
  2. SmoothQuant 分组/scale 文件, 由 --smooth-groups 控制, 默认读取
     qwen3_gptq_repro/output/smooth_v12_b32/smooth_groups.pt。
  3. 校准文本, 默认通过项目内 WikiText2 加载逻辑采样; 如果已有本地数据,
     可以用 --local-wikitext2-dir 指定。

主要参数:
  --module-filter / --layer-filter 控制分析哪些 Linear 模块和层。
  --block-rows / --block-cols 控制 block 形状, 默认 16 x 128。
  --groupsize / --weight-bits / --act-order 控制 GPTQ 风格权重量化。
  --top-mass-fracs / --repair-fracs 控制统计和 repair curve 的横轴采样点。
  --max-plot-modules / --heatmap-scale 控制保存多少模块 heatmap 以及使用 raw
  还是 log10 色阶。

运行示例:
  cd /root/autodl-tmp/Zip/qwen3_gptq_repro
  python observation/prove_hessian_sensitive_blocks.py \
    --smooth-groups output/smooth_v16_b32/smooth_groups.pt \
    --module-filter q_proj,v_proj \
    --layer-filter 24,34 \
    --block-rows 16 \
    --block-cols 128 \
    --groupsize 32 \
    --output-dir observation/output/hessian_sensitive_blocks_example

画出的图像:
  output-dir/plots/concentration_curves.png:
    top-k blocks 捕获 exact output error mass 的集中度曲线。
  output-dir/plots/repair_curves.png:
    按 Hessian/Frobenius/channel/delta/random 选择 block 后剩余误差比例。
  output-dir/plots/score_correlation.png:
    Hessian/Frobenius/channel 分数与 exact output score 的 log-log 散点图。
  output-dir/plots/heatmaps/*.png:
    每个高误差模块的 exact output score 与 Hessian diag score block heatmap。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import struct
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn


SCRIPT_DIR = Path(__file__).resolve().parent
REPRO_DIR = SCRIPT_DIR.parent
GPTQ_DIR = REPRO_DIR.parent / "gptq"
for _path in (REPRO_DIR, GPTQ_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)


DEFAULT_MODEL_DIR = Path("/root/autodl-tmp/model/Qwen3-4B-Instruct-2507")
DEFAULT_SMOOTH_GROUPS = REPRO_DIR / "output" / "smooth_v12_b32" / "smooth_groups.pt"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "hessian_sensitive_blocks"

DEFAULT_MODULE_GROUPS = {
    "attn_qkv": ["q_proj", "k_proj", "v_proj"],
    "attn_o": ["o_proj"],
    "ffn_up_gate": ["up_proj", "gate_proj"],
    "ffn_down": ["down_proj"],
}
SMOOTH_ENABLED_GROUPS = {"attn_qkv", "ffn_up_gate"}
GROUP_ORDER = {
    "attn_qkv": 0,
    "attn_o": 1,
    "ffn_up_gate": 2,
    "ffn_down": 3,
}
LAYER_RE = re.compile(r"model\.layers\.(?P<layer_id>\d+)\.")


@dataclass
class SmoothInfo:
    scale: torch.Tensor | None
    alpha: float | None
    group_key: str
    smooth_used: bool


@dataclass
class ActivationContext:
    h_diag: torch.Tensor
    h_blocks: list[torch.Tensor]
    perm: torch.Tensor | None
    token_count: int
    input_dim: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute W_s, Q0, R, Q4A(X_s), and block-wise Hessian/output "
            "scores to test Hessian-sensitive block concentration."
        )
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--smooth-groups", type=Path, default=DEFAULT_SMOOTH_GROUPS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--local-wikitext2-dir", type=str, default="")
    parser.add_argument("--custom-modeling-file", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=1024)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Model load dtype.",
    )
    parser.add_argument(
        "--cache-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="CPU dtype used for cached calibration activations.",
    )
    parser.add_argument(
        "--analysis-device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device for scoring matrices.",
    )
    parser.add_argument("--block-rows", type=int, default=16)
    parser.add_argument("--block-cols", type=int, default=128)
    parser.add_argument("--groupsize", type=int, default=32)
    parser.add_argument("--weight-bits", type=int, default=4)
    parser.add_argument("--act-order", dest="act_order", action="store_true", default=True)
    parser.add_argument("--no-act-order", dest="act_order", action="store_false")
    parser.add_argument(
        "--score-max-tokens",
        type=int,
        default=0,
        help="Optional token subsample for Hessian/output scoring. 0 means all cached tokens.",
    )
    parser.add_argument(
        "--module-filter",
        type=str,
        default="",
        help="Comma-separated substrings, e.g. up_proj,down_proj.",
    )
    parser.add_argument(
        "--layer-filter",
        type=str,
        default="",
        help="Layer ids or ranges, e.g. 0,3,8-11.",
    )
    parser.add_argument(
        "--collect-granularity",
        choices=["layer", "group"],
        default="layer",
        help="Collect all source inputs in a layer at once, or one group at a time.",
    )
    parser.add_argument(
        "--top-mass-fracs",
        type=str,
        default="0.01,0.05,0.10,0.20",
        help="Fractions used for concentration summaries.",
    )
    parser.add_argument(
        "--repair-fracs",
        type=str,
        default="0.01,0.05,0.10,0.20",
        help="Fractions used for repair curves.",
    )
    parser.add_argument("--random-trials", type=int, default=8)
    parser.add_argument("--max-plot-modules", type=int, default=24)
    parser.add_argument("--scatter-sample", type=int, default=50000)
    parser.add_argument("--save-per-block", action="store_true", default=True)
    parser.add_argument("--no-save-per-block", dest="save_per_block", action="store_false")
    parser.add_argument(
        "--heatmap-scale",
        choices=["log10", "raw"],
        default="log10",
        help="Scale used for per-module heatmaps. Default keeps the historical log10 plots.",
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if args.block_rows <= 0 or args.block_cols <= 0:
        parser.error("--block-rows and --block-cols must be positive.")
    if args.groupsize <= 0:
        parser.error("--groupsize must be positive.")
    if args.nsamples <= 0 or args.seqlen <= 0:
        parser.error("--nsamples and --seqlen must be positive.")
    if args.max_batches < 0:
        parser.error("--max-batches must be >= 0.")
    if args.score_max_tokens < 0:
        parser.error("--score-max-tokens must be >= 0.")
    return args


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def parse_float_list(text: str) -> list[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value < 0:
            raise ValueError(f"Fractions must be non-negative: {text}")
        values.append(value)
    return values


def parse_layer_filter(text: str) -> set[int] | None:
    text = text.strip()
    if not text:
        return None
    layers: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start, end = int(left), int(right)
            if end < start:
                raise ValueError(f"Invalid layer range: {part}")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    return layers


def module_type(module_name: str) -> str:
    return module_name.rsplit(".", 1)[-1]


def layer_id_from_name(module_name: str) -> int | None:
    match = LAYER_RE.search(module_name)
    return int(match.group("layer_id")) if match is not None else None


def finite_float(value: float | np.floating | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def resolve_module_groups(model: nn.Module) -> dict[str, dict[str, list[str]]]:
    groups: dict[str, dict[str, list[str]]] = {name: {} for name in DEFAULT_MODULE_GROUPS}
    linear_names = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    for group_name, suffixes in DEFAULT_MODULE_GROUPS.items():
        for name in linear_names:
            for suffix in suffixes:
                token = suffix if suffix.startswith(".") else f".{suffix}"
                if name.endswith(token):
                    prefix = name[: -len(token)]
                    groups[group_name].setdefault(prefix, []).append(name)
        for prefix in groups[group_name]:
            groups[group_name][prefix].sort()
    return groups


def build_layer_work_items(
    resolved_groups: dict[str, dict[str, list[str]]],
    layer_filter: set[int] | None,
    module_filters: list[str],
) -> list[tuple[str, list[tuple[str, str, list[str]]]]]:
    grouped: dict[str, list[tuple[str, str, list[str]]]] = {}
    for group_name, prefix_map in resolved_groups.items():
        for group_prefix, module_names in prefix_map.items():
            filtered = []
            for name in module_names:
                lid = layer_id_from_name(name)
                if layer_filter is not None and lid not in layer_filter:
                    continue
                if module_filters and not any(token in name for token in module_filters):
                    continue
                filtered.append(name)
            if not filtered:
                continue
            layer_prefix = group_prefix
            for marker in (".self_attn", ".mlp"):
                idx = group_prefix.rfind(marker)
                if idx >= 0:
                    layer_prefix = group_prefix[:idx]
                    break
            grouped.setdefault(layer_prefix, []).append((group_name, group_prefix, filtered))

    def layer_sort_key(item: tuple[str, list[tuple[str, str, list[str]]]]) -> tuple[int, str]:
        prefix = item[0]
        match = LAYER_RE.search(prefix + ".")
        return (int(match.group("layer_id")) if match is not None else 10**9, prefix)

    work_items = []
    for prefix, group_items in grouped.items():
        group_items.sort(key=lambda item: GROUP_ORDER.get(item[0], 100))
        work_items.append((prefix, group_items))
    work_items.sort(key=layer_sort_key)
    return work_items


def load_smooth_info(path: Path) -> dict[str, SmoothInfo]:
    if not path or not path.exists():
        print(f"[warn] smooth_groups not found, using identity smooth scale: {path}", flush=True)
        return {}
    data = torch.load(path, map_location="cpu")
    module_to_info: dict[str, SmoothInfo] = {}
    for key, payload in data.items():
        if not isinstance(payload, dict):
            continue
        scale = payload.get("smooth_scale")
        if not torch.is_tensor(scale):
            continue
        for module_name in payload.get("target_linears", []):
            module_to_info[module_name] = SmoothInfo(
                scale=scale.detach().float().cpu(),
                alpha=float(payload["alpha"]) if payload.get("alpha") is not None else None,
                group_key=str(key),
                smooth_used=True,
            )
    print(f"[smooth] loaded scales for {len(module_to_info)} modules from {path}", flush=True)
    return module_to_info


def smooth_info_for_module(
    module_name: str,
    module: nn.Linear,
    module_to_smooth: dict[str, SmoothInfo],
) -> SmoothInfo:
    found = module_to_smooth.get(module_name)
    if found is not None:
        return found
    return SmoothInfo(
        scale=torch.ones(module.in_features, dtype=torch.float32),
        alpha=None,
        group_key="identity",
        smooth_used=False,
    )


def move_batch_to_device(batch, device: torch.device):
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_batch_to_device(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(value, device) for value in batch)
    return batch


@torch.no_grad()
def collect_input_cache(
    model: nn.Module,
    calibration_loader: Iterable,
    target_module_names: list[str],
    cache_dtype: torch.dtype,
    max_batches: int | None,
) -> dict[str, torch.Tensor]:
    model_device = next(model.parameters()).device
    selected = set(target_module_names)
    caches = {name: [] for name in selected}
    handles = []

    for name, module in model.named_modules():
        if name not in selected:
            continue

        def hook(_module, inp, _out, module_name=name):
            x = inp[0].detach().to(device="cpu", dtype=cache_dtype)
            caches[module_name].append(x)

        handles.append(module.register_forward_hook(hook))

    old_use_cache = getattr(getattr(model, "config", None), "use_cache", None)
    if old_use_cache is not None:
        model.config.use_cache = False
    try:
        for batch_idx, batch in enumerate(calibration_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = move_batch_to_device(batch, model_device)
            if isinstance(batch, (list, tuple)):
                model(batch[0])
            elif isinstance(batch, dict):
                model(**batch)
            else:
                model(batch)
    finally:
        for handle in handles:
            handle.remove()
        if old_use_cache is not None:
            model.config.use_cache = old_use_cache

    merged = {}
    for name, values in caches.items():
        if not values:
            continue
        merged[name] = values[0] if len(values) == 1 else torch.cat(values, dim=0)
    return merged


def compute_h_diag(x2d: torch.Tensor, chunk_tokens: int = 8192) -> torch.Tensor:
    h_diag = torch.zeros(x2d.shape[1], device=x2d.device, dtype=torch.float32)
    token_count = max(int(x2d.shape[0]), 1)
    for start in range(0, x2d.shape[0], chunk_tokens):
        chunk = x2d[start : start + chunk_tokens].float()
        h_diag += chunk.pow(2).sum(dim=0)
    return 2.0 * h_diag / float(token_count)


@torch.no_grad()
def build_activation_context(
    x_cache: torch.Tensor,
    smooth_scale: torch.Tensor,
    groupsize: int,
    block_cols: int,
    act_order: bool,
    score_max_tokens: int,
    device: torch.device,
    seed: int,
) -> ActivationContext:
    from smooth_block_quant import fake_quant_activation_int4_group_symmetric, smooth_input

    x = x_cache.to(device=device, non_blocking=True)
    scale = smooth_scale.to(device=device, dtype=torch.float32)
    x_s = smooth_input(x.float(), scale)
    x_q = fake_quant_activation_int4_group_symmetric(x_s, group_size=groupsize)
    x2d = x_q.reshape(-1, x_q.shape[-1])
    if score_max_tokens > 0 and x2d.shape[0] > score_max_tokens:
        rng = np.random.default_rng(seed)
        indices_np = np.sort(rng.choice(x2d.shape[0], size=score_max_tokens, replace=False))
        indices = torch.as_tensor(indices_np, device=device, dtype=torch.long)
        x2d = x2d.index_select(0, indices)

    token_count = int(x2d.shape[0])
    input_dim = int(x2d.shape[1])
    h_diag_unordered = compute_h_diag(x2d)
    perm = torch.argsort(h_diag_unordered, descending=True) if act_order else None
    h_diag = h_diag_unordered.index_select(0, perm) if perm is not None else h_diag_unordered

    h_blocks: list[torch.Tensor] = []
    for c0 in range(0, input_dim, block_cols):
        c1 = min(c0 + block_cols, input_dim)
        if perm is None:
            x_block = x2d[:, c0:c1].float()
        else:
            col_idx = perm[c0:c1]
            x_block = x2d.index_select(1, col_idx).float()
        h_block = 2.0 * x_block.t().matmul(x_block) / float(max(token_count, 1))
        h_blocks.append(h_block)
        del x_block

    del x, x_s, x_q, x2d, h_diag_unordered
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ActivationContext(
        h_diag=h_diag,
        h_blocks=h_blocks,
        perm=perm,
        token_count=token_count,
        input_dim=input_dim,
    )


def aggregate_rows(row_values: torch.Tensor, block_rows: int, nrow: int) -> torch.Tensor:
    total_rows = nrow * block_rows
    if row_values.numel() < total_rows:
        pad = torch.zeros(total_rows - row_values.numel(), device=row_values.device, dtype=row_values.dtype)
        row_values = torch.cat([row_values, pad], dim=0)
    return row_values.view(nrow, block_rows).sum(dim=1)


def tensor_matrix_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def gini(values: np.ndarray) -> float | None:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    total = float(values.sum())
    if total <= 0:
        return 0.0
    sorted_values = np.sort(values)
    n = sorted_values.size
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * sorted_values) / (n * total)) - ((n + 1.0) / n))


def effective_count(values: np.ndarray) -> float | None:
    values = values[np.isfinite(values)]
    total = float(values.sum())
    denom = float(np.square(values).sum())
    if values.size == 0 or total <= 0 or denom <= 0:
        return None
    return float(total * total / denom)


def top_mass(values: np.ndarray, fracs: list[float]) -> dict[str, float | None]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {f"top_{frac:g}_mass": None for frac in fracs}
    total = float(values.sum())
    if total <= 0:
        return {f"top_{frac:g}_mass": 0.0 for frac in fracs}
    sorted_values = np.sort(values)[::-1]
    out: dict[str, float | None] = {}
    for frac in fracs:
        k = min(values.size, max(1, int(round(values.size * frac)))) if frac > 0 else 0
        captured = float(sorted_values[:k].sum()) if k > 0 else 0.0
        out[f"top_{frac:g}_mass"] = captured / total
    return out


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask].astype(np.float64)
    y = y[mask].astype(np.float64)
    if x.size < 2:
        return None
    x = x - x.mean()
    y = y - y.mean()
    denom = math.sqrt(float(np.square(x).sum() * np.square(y).sum()))
    if denom <= 0:
        return None
    return float(np.dot(x, y) / denom)


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask].astype(np.float64)
    y = y[mask].astype(np.float64)
    if x.size < 2:
        return None
    return pearson(average_ranks(x), average_ranks(y))


def correlations(exact: np.ndarray, score: np.ndarray) -> tuple[float | None, float | None]:
    return pearson(score, exact), spearman(score, exact)


def repair_curve(
    exact: np.ndarray,
    rankings: dict[str, np.ndarray],
    fracs: list[float],
    random_trials: int,
    seed: int,
) -> list[dict[str, float | int | str]]:
    exact = exact.reshape(-1).astype(np.float64)
    total = float(exact[np.isfinite(exact)].sum())
    n = exact.size
    rows: list[dict[str, float | int | str]] = []
    rng = np.random.default_rng(seed)
    for method, score in rankings.items():
        score = score.reshape(-1).astype(np.float64)
        order = np.argsort(-np.nan_to_num(score, nan=-np.inf), kind="mergesort")
        for frac in fracs:
            k = min(n, max(1, int(round(n * frac)))) if frac > 0 else 0
            captured = float(exact[order[:k]].sum()) if k > 0 else 0.0
            rows.append(
                {
                    "method": method,
                    "fraction": float(frac),
                    "selected_blocks": int(k),
                    "captured_mass": captured / total if total > 0 else None,
                    "remaining_ratio": 1.0 - captured / total if total > 0 else None,
                }
            )

    for frac in fracs:
        k = min(n, max(1, int(round(n * frac)))) if frac > 0 else 0
        if k <= 0 or n == 0:
            captured_ratio = 0.0
        else:
            captured_values = []
            for _ in range(max(1, random_trials)):
                idx = rng.choice(n, size=k, replace=False)
                captured_values.append(float(exact[idx].sum()))
            captured_ratio = float(np.mean(captured_values)) / total if total > 0 else 0.0
        rows.append(
            {
                "method": "random",
                "fraction": float(frac),
                "selected_blocks": int(k),
                "captured_mass": captured_ratio if total > 0 else None,
                "remaining_ratio": 1.0 - captured_ratio if total > 0 else None,
            }
        )
    return rows


def make_quantizer(bits: int):
    from smooth_block_quant.calibrate import make_quantizer as _make_quantizer

    return _make_quantizer(bits=bits, sym=False)


@torch.no_grad()
def analyze_module(
    module_name: str,
    module: nn.Linear,
    smooth_info: SmoothInfo,
    context: ActivationContext,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict, list[dict], dict[str, np.ndarray], list[dict]]:
    from smooth_block_quant.block_mask import compute_delta_hessian_scores
    from smooth_block_quant.residual import fake_quantize_weight

    block_rows = args.block_rows
    block_cols = args.block_cols
    weight = module.weight.detach().to(device=device, dtype=torch.float32)
    scale = smooth_info.scale
    if scale is None:
        scale = torch.ones(weight.shape[1], dtype=torch.float32)
    scale = scale.to(device=device, dtype=torch.float32)
    w_s = weight * scale.view(1, -1)
    if context.perm is not None:
        w_for = w_s.index_select(1, context.perm)
    else:
        w_for = w_s

    d_out, d_in = w_for.shape
    nrow = math.ceil(d_out / block_rows)
    ncol = math.ceil(d_in / block_cols)
    quantizer = make_quantizer(args.weight_bits)
    q0 = fake_quantize_weight(w_for, quantizer, groupsize=args.groupsize)
    residual = w_for - q0

    frob_scores = torch.zeros(nrow, ncol, device=device, dtype=torch.float32)
    hdiag_scores = torch.zeros_like(frob_scores)
    exact_hessian_scores = torch.zeros_like(frob_scores)
    exact_output_scores = torch.zeros_like(frob_scores)
    channel_scores = torch.zeros_like(frob_scores)

    channel_energy = residual.pow(2).mul(context.h_diag.view(1, -1)).sum(dim=0)
    output_scale = float(context.token_count) / 2.0
    for bc in range(ncol):
        c0 = bc * block_cols
        c1 = min(c0 + block_cols, d_in)
        r_col = residual[:, c0:c1].float()
        h_diag_col = context.h_diag[c0:c1].float()
        h_block = context.h_blocks[bc].to(device=device, dtype=torch.float32)

        frob_row = r_col.pow(2).sum(dim=1)
        hdiag_row = r_col.pow(2).mul(h_diag_col.view(1, -1)).sum(dim=1)
        exact_row = r_col.matmul(h_block).mul(r_col).sum(dim=1).clamp_min(0.0)
        channel_value = channel_energy[c0:c1].sum()

        frob_scores[:, bc] = aggregate_rows(frob_row, block_rows, nrow)
        hdiag_scores[:, bc] = aggregate_rows(hdiag_row, block_rows, nrow)
        exact_hessian_scores[:, bc] = aggregate_rows(exact_row, block_rows, nrow)
        exact_output_scores[:, bc] = exact_hessian_scores[:, bc] * output_scale
        channel_scores[:, bc] = channel_value

    delta_scores = compute_delta_hessian_scores(
        W=w_for,
        block_shape=(block_rows, block_cols),
        quantizer=quantizer,
        H_diag=context.h_diag,
        groupsize=args.groupsize,
    )

    exact_np = tensor_matrix_to_numpy(exact_output_scores)
    hdiag_np = tensor_matrix_to_numpy(hdiag_scores)
    frob_np = tensor_matrix_to_numpy(frob_scores)
    channel_np = tensor_matrix_to_numpy(channel_scores)
    delta_np = tensor_matrix_to_numpy(delta_scores)

    h_pearson, h_spearman = correlations(exact_np.reshape(-1), hdiag_np.reshape(-1))
    f_pearson, f_spearman = correlations(exact_np.reshape(-1), frob_np.reshape(-1))
    c_pearson, c_spearman = correlations(exact_np.reshape(-1), channel_np.reshape(-1))
    d_pearson, d_spearman = correlations(exact_np.reshape(-1), delta_np.reshape(-1))
    top_mass_values = top_mass(exact_np.reshape(-1), parse_float_list(args.top_mass_fracs))
    eff = effective_count(exact_np.reshape(-1))

    module_repair = repair_curve(
        exact=exact_np,
        rankings={
            "hessian_diag": hdiag_np,
            "frob": frob_np,
            "channel_only": channel_np,
            "delta_hessian": delta_np,
        },
        fracs=parse_float_list(args.repair_fracs),
        random_trials=args.random_trials,
        seed=args.seed + (layer_id_from_name(module_name) or 0) * 1009 + len(module_name),
    )

    summary = {
        "module": module_name,
        "layer_id": layer_id_from_name(module_name),
        "module_type": module_type(module_name),
        "smooth_used": smooth_info.smooth_used,
        "smooth_alpha": smooth_info.alpha,
        "smooth_group_key": smooth_info.group_key,
        "out_features": int(d_out),
        "in_features": int(d_in),
        "n_blocks": int(nrow * ncol),
        "grid_rows": int(nrow),
        "grid_cols": int(ncol),
        "score_tokens": int(context.token_count),
        "total_exact_output_score": finite_float(float(exact_np.sum())),
        "total_hdiag_score": finite_float(float(hdiag_np.sum())),
        "total_frob_score": finite_float(float(frob_np.sum())),
        "gini_exact_output": finite_float(gini(exact_np.reshape(-1))),
        "effective_block_count": finite_float(eff),
        "effective_block_ratio": finite_float(eff / float(nrow * ncol)) if eff is not None else None,
        "corr_hessian_pearson": finite_float(h_pearson),
        "corr_hessian_spearman": finite_float(h_spearman),
        "corr_frob_pearson": finite_float(f_pearson),
        "corr_frob_spearman": finite_float(f_spearman),
        "corr_channel_pearson": finite_float(c_pearson),
        "corr_channel_spearman": finite_float(c_spearman),
        "corr_delta_pearson": finite_float(d_pearson),
        "corr_delta_spearman": finite_float(d_spearman),
    }
    summary.update({key: finite_float(value) for key, value in top_mass_values.items()})
    for row in module_repair:
        if row["method"] == "hessian_diag":
            key = f"hessian_remaining_at_{row['fraction']:g}"
            summary[key] = finite_float(row.get("remaining_ratio"))
        if row["method"] == "random":
            key = f"random_remaining_at_{row['fraction']:g}"
            summary[key] = finite_float(row.get("remaining_ratio"))

    block_rows_json = []
    for br in range(nrow):
        r0 = br * block_rows
        r1 = min((br + 1) * block_rows, d_out)
        for bc in range(ncol):
            c0 = bc * block_cols
            c1 = min((bc + 1) * block_cols, d_in)
            block_rows_json.append(
                {
                    "module": module_name,
                    "block_row": br,
                    "block_col": bc,
                    "row_start": r0,
                    "row_end": r1,
                    "analysis_col_start": c0,
                    "analysis_col_end": c1,
                    "frob_score": finite_float(frob_np[br, bc]),
                    "hdiag_score": finite_float(hdiag_np[br, bc]),
                    "exact_output_score": finite_float(exact_np[br, bc]),
                    "channel_score": finite_float(channel_np[br, bc]),
                    "delta_hessian": finite_float(delta_np[br, bc]),
                }
            )

    matrices = {
        "exact": exact_np,
        "hessian": hdiag_np,
        "frob": frob_np,
        "channel": channel_np,
        "delta": delta_np,
    }

    del weight, scale, w_s, w_for, q0, residual, frob_scores, hdiag_scores
    del exact_hessian_scores, exact_output_scores, channel_scores, delta_scores
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, block_rows_json, matrices, module_repair


def write_module_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base_fields = [
        "module",
        "layer_id",
        "module_type",
        "smooth_used",
        "smooth_alpha",
        "smooth_group_key",
        "out_features",
        "in_features",
        "n_blocks",
        "grid_rows",
        "grid_cols",
        "score_tokens",
        "total_exact_output_score",
        "total_hdiag_score",
        "total_frob_score",
        "gini_exact_output",
        "effective_block_count",
        "effective_block_ratio",
        "corr_hessian_pearson",
        "corr_hessian_spearman",
        "corr_frob_pearson",
        "corr_frob_spearman",
        "corr_channel_pearson",
        "corr_channel_spearman",
        "corr_delta_pearson",
        "corr_delta_spearman",
    ]
    extra_fields = sorted({key for row in rows for key in row if key not in base_fields})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=base_fields + extra_fields)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Iterable[dict], mode: str = "a") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def summarize_global(
    module_rows: list[dict],
    global_arrays: dict[str, list[np.ndarray]],
    repair_fracs: list[float],
    top_mass_fracs: list[float],
    random_trials: int,
    seed: int,
    elapsed_seconds: float,
    args: argparse.Namespace,
) -> dict:
    arrays = {key: np.concatenate(values).reshape(-1) for key, values in global_arrays.items() if values}
    exact = arrays["exact"]
    global_repair = repair_curve(
        exact=exact,
        rankings={
            "hessian_diag": arrays["hessian"],
            "frob": arrays["frob"],
            "channel_only": arrays["channel"],
            "delta_hessian": arrays["delta"],
        },
        fracs=repair_fracs,
        random_trials=random_trials,
        seed=seed,
    )
    corr = {}
    for key in ("hessian", "frob", "channel", "delta"):
        p, s = correlations(exact, arrays[key])
        corr[f"{key}_pearson"] = finite_float(p)
        corr[f"{key}_spearman"] = finite_float(s)
    eff = effective_count(exact)
    return {
        "created_at_unix": time.time(),
        "elapsed_seconds": elapsed_seconds,
        "model_dir": str(args.model_dir.resolve()),
        "smooth_groups": str(args.smooth_groups.resolve()) if args.smooth_groups else "",
        "nsamples": args.nsamples,
        "seqlen": args.seqlen,
        "max_batches": args.max_batches,
        "score_max_tokens": args.score_max_tokens,
        "block_shape": [args.block_rows, args.block_cols],
        "groupsize": args.groupsize,
        "act_order": bool(args.act_order),
        "module_count": len(module_rows),
        "total_blocks": int(exact.size),
        "total_exact_output_score": finite_float(float(exact.sum())),
        "gini_exact_output": finite_float(gini(exact)),
        "effective_block_count": finite_float(eff),
        "effective_block_ratio": finite_float(eff / float(exact.size)) if eff is not None else None,
        "top_mass": {key: finite_float(value) for key, value in top_mass(exact, top_mass_fracs).items()},
        "correlations": corr,
        "repair_curve": global_repair,
        "top_modules_by_exact_output": sorted(
            (
                {
                    "module": row["module"],
                    "total_exact_output_score": row["total_exact_output_score"],
                    "gini_exact_output": row["gini_exact_output"],
                    "corr_hessian_spearman": row["corr_hessian_spearman"],
                    "corr_frob_spearman": row["corr_frob_spearman"],
                    "corr_channel_spearman": row["corr_channel_spearman"],
                }
                for row in module_rows
            ),
            key=lambda row: row["total_exact_output_score"] or 0.0,
            reverse=True,
        )[:25],
    }


def setup_matplotlib():
    try:
        import matplotlib
    except ModuleNotFoundError:
        return None

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_png(path: Path, image: np.ndarray) -> None:
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB uint8 image, got shape {image.shape}")
    height, width, _ = image.shape

    def chunk(kind: bytes, data: bytes) -> bytes:
        payload = kind + data
        return (
            struct.pack(">I", len(data))
            + payload
            + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=6))
        + chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def draw_line(image: np.ndarray, points: list[tuple[int, int]], color: tuple[int, int, int]) -> None:
    if len(points) < 2:
        return
    height, width, _ = image.shape
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0
        while True:
            if 0 <= x < width and 0 <= y < height:
                image[y, x] = color
                if y + 1 < height:
                    image[y + 1, x] = color
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy


def draw_axes(image: np.ndarray, margin: int = 48) -> tuple[int, int, int, int]:
    height, width, _ = image.shape
    left, right = margin, width - margin
    top, bottom = margin, height - margin
    image[bottom : bottom + 2, left:right] = (0, 0, 0)
    image[top:bottom, left : left + 2] = (0, 0, 0)
    for i in range(6):
        x = left + int((right - left) * i / 5)
        y = bottom - int((bottom - top) * i / 5)
        image[bottom - 4 : bottom + 5, x : x + 1] = (0, 0, 0)
        image[y : y + 1, left - 4 : left + 5] = (0, 0, 0)
    return left, right, top, bottom


def plot_lines_fallback(
    path: Path,
    series: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int]]],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    size: tuple[int, int] = (900, 650),
) -> None:
    width, height = size
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    left, right, top, bottom = draw_axes(image)
    xmin, xmax = xlim
    ymin, ymax = ylim
    xspan = max(xmax - xmin, 1e-30)
    yspan = max(ymax - ymin, 1e-30)
    for xs, ys, color in series:
        mask = np.isfinite(xs) & np.isfinite(ys)
        xs = xs[mask]
        ys = ys[mask]
        if xs.size == 0:
            continue
        px = left + np.clip(((xs - xmin) / xspan * (right - left)).astype(int), 0, right - left)
        py = bottom - np.clip(((ys - ymin) / yspan * (bottom - top)).astype(int), 0, bottom - top)
        draw_line(image, list(zip(px.tolist(), py.tolist())), color)
    save_png(path, image)


def plot_scatter_fallback(
    path: Path,
    panels: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int]]],
    size: tuple[int, int] = (1500, 500),
) -> None:
    width, height = size
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    panel_w = width // len(panels)
    for panel_idx, (xs, ys, color) in enumerate(panels):
        x = np.log10(np.maximum(xs.astype(np.float64), 1e-30))
        y = np.log10(np.maximum(ys.astype(np.float64), 1e-30))
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        x0 = panel_idx * panel_w
        sub = image[:, x0 : x0 + panel_w]
        left, right, top, bottom = draw_axes(sub, margin=40)
        if x.size == 0:
            continue
        xmin, xmax = np.percentile(x, [1, 99])
        ymin, ymax = np.percentile(y, [1, 99])
        if xmax <= xmin:
            xmax = xmin + 1.0
        if ymax <= ymin:
            ymax = ymin + 1.0
        px = left + np.clip(((x - xmin) / (xmax - xmin) * (right - left)).astype(int), 0, right - left)
        py = bottom - np.clip(((y - ymin) / (ymax - ymin) * (bottom - top)).astype(int), 0, bottom - top)
        sub[py, px] = color
    save_png(path, image)


def heat_color(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        norm = np.zeros_like(values)
    else:
        lo, hi = np.percentile(values[finite], [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        norm = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    r = np.clip(255 * np.maximum(0.0, 1.5 * norm - 0.25), 0, 255)
    g = np.clip(255 * (1.0 - np.abs(norm - 0.55) * 1.8), 0, 255)
    b = np.clip(255 * (1.0 - norm * 1.2), 0, 255)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def resize_nearest(image: np.ndarray, height: int, width: int) -> np.ndarray:
    y_idx = np.linspace(0, image.shape[0] - 1, height).astype(int)
    x_idx = np.linspace(0, image.shape[1] - 1, width).astype(int)
    return image[y_idx][:, x_idx]


def plot_global_fallback(
    output_dir: Path,
    arrays: dict[str, np.ndarray],
    repair_rows: list[dict],
    scatter_sample: int,
    seed: int,
) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    legend = {
        "concentration_curves.png": {
            "blue": "exact sorted blocks",
            "gray": "uniform baseline",
        },
        "repair_curves.png": {
            "blue": "hessian_diag",
            "red": "frob",
            "green": "channel_only",
            "purple": "delta_hessian",
            "gray": "random",
        },
        "score_correlation.png": {
            "left_blue": "hessian score vs exact output",
            "middle_red": "frob score vs exact output",
            "right_green": "channel-only score vs exact output",
        },
    }
    (plots_dir / "plot_legend.json").write_text(json.dumps(legend, indent=2), encoding="utf-8")

    exact = arrays["exact"].astype(np.float64)
    total = float(exact.sum())
    sorted_exact = np.sort(exact)[::-1]
    if total > 0 and sorted_exact.size:
        cumulative = np.cumsum(sorted_exact) / total
        grid = np.linspace(0.0, 1.0, 400)
        idx = np.clip((grid * (sorted_exact.size - 1)).astype(int), 0, sorted_exact.size - 1)
        plot_lines_fallback(
            plots_dir / "concentration_curves.png",
            [
                (grid, cumulative[idx], (36, 99, 235)),
                (np.array([0.0, 1.0]), np.array([0.0, 1.0]), (150, 150, 150)),
            ],
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
        )

    colors = {
        "hessian_diag": (36, 99, 235),
        "frob": (220, 60, 60),
        "channel_only": (30, 150, 90),
        "delta_hessian": (130, 75, 200),
        "random": (130, 130, 130),
    }
    series = []
    for method, color in colors.items():
        rows = sorted((row for row in repair_rows if row["method"] == method), key=lambda row: row["fraction"])
        if not rows:
            continue
        series.append(
            (
                np.array([row["fraction"] for row in rows], dtype=np.float64),
                np.array([row["remaining_ratio"] for row in rows], dtype=np.float64),
                color,
            )
        )
    plot_lines_fallback(
        plots_dir / "repair_curves.png",
        series,
        xlim=(0.0, max(float(row["fraction"]) for row in repair_rows) if repair_rows else 1.0),
        ylim=(0.0, 1.0),
    )

    rng = np.random.default_rng(seed)
    n = exact.size
    if n > scatter_sample > 0:
        idx = rng.choice(n, size=scatter_sample, replace=False)
    else:
        idx = np.arange(n)
    plot_scatter_fallback(
        plots_dir / "score_correlation.png",
        [
            (arrays["hessian"][idx], exact[idx], (36, 99, 235)),
            (arrays["frob"][idx], exact[idx], (220, 60, 60)),
            (arrays["channel"][idx], exact[idx], (30, 150, 90)),
        ],
    )


def plot_heatmaps_fallback(
    output_dir: Path,
    heatmap_items: list[tuple[str, dict[str, np.ndarray]]],
    max_items: int,
    scale: str,
) -> None:
    if max_items <= 0:
        return
    heatmap_dir = output_dir / "plots" / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    for module_name, matrices in heatmap_items[:max_items]:
        if scale == "raw":
            exact_data = matrices["exact"]
            hessian_data = matrices["hessian"]
        else:
            exact_data = np.log10(np.maximum(matrices["exact"], 1e-30))
            hessian_data = np.log10(np.maximum(matrices["hessian"], 1e-30))
        exact = heat_color(exact_data)
        hessian = heat_color(hessian_data)
        exact = resize_nearest(exact, 420, 520)
        hessian = resize_nearest(hessian, 420, 520)
        gap = np.full((420, 24, 3), 255, dtype=np.uint8)
        image = np.concatenate([exact, gap, hessian], axis=1)
        safe_name = module_name.replace(".", "_")
        save_png(heatmap_dir / f"{safe_name}.png", image)


def plot_global(
    output_dir: Path,
    arrays: dict[str, np.ndarray],
    repair_rows: list[dict],
    scatter_sample: int,
    seed: int,
) -> None:
    plt = setup_matplotlib()
    if plt is None:
        plot_global_fallback(output_dir, arrays, repair_rows, scatter_sample, seed)
        return
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    exact = arrays["exact"].astype(np.float64)
    total = float(exact.sum())
    sorted_exact = np.sort(exact)[::-1]
    if total > 0 and sorted_exact.size:
        cumulative = np.cumsum(sorted_exact) / total
        grid = np.linspace(0.0, 1.0, 200)
        idx = np.clip((grid * (sorted_exact.size - 1)).astype(int), 0, sorted_exact.size - 1)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(grid, cumulative[idx], label="exact sorted blocks")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="uniform")
        ax.set_xlabel("Top fraction of blocks")
        ax.set_ylabel("Captured exact output error mass")
        ax.set_title("Block Error Concentration")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "concentration_curves.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    methods = sorted({row["method"] for row in repair_rows})
    for method in methods:
        rows = sorted((row for row in repair_rows if row["method"] == method), key=lambda row: row["fraction"])
        ax.plot(
            [row["fraction"] for row in rows],
            [row["remaining_ratio"] for row in rows],
            marker="o",
            label=method,
        )
    ax.set_xlabel("Selected block fraction")
    ax.set_ylabel("Remaining exact output error ratio")
    ax.set_title("Repair Curves")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "repair_curves.png", dpi=180)
    plt.close(fig)

    rng = np.random.default_rng(seed)
    n = exact.size
    if n > scatter_sample > 0:
        idx = rng.choice(n, size=scatter_sample, replace=False)
    else:
        idx = np.arange(n)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [
        ("hessian", "Hessian diag score"),
        ("frob", "Frobenius score"),
        ("channel", "Channel-only score"),
    ]
    for ax, (key, title) in zip(axes, panels):
        x = np.maximum(arrays[key][idx].astype(np.float64), 1e-30)
        y = np.maximum(exact[idx], 1e-30)
        ax.scatter(x, y, s=3, alpha=0.15)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(title)
        ax.set_ylabel("Exact output score")
        ax.grid(True, alpha=0.25)
    fig.suptitle("Block Score vs Exact Local Output Error")
    fig.tight_layout()
    fig.savefig(plots_dir / "score_correlation.png", dpi=180)
    plt.close(fig)


def plot_heatmaps(
    output_dir: Path,
    heatmap_items: list[tuple[str, dict[str, np.ndarray]]],
    max_items: int,
    scale: str,
) -> None:
    if max_items <= 0:
        return
    plt = setup_matplotlib()
    if plt is None:
        plot_heatmaps_fallback(output_dir, heatmap_items, max_items, scale)
        return
    heatmap_dir = output_dir / "plots" / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    for module_name, matrices in heatmap_items[:max_items]:
        if scale == "raw":
            exact = matrices["exact"]
            hessian = matrices["hessian"]
            exact_title = "raw exact output score"
            hessian_title = "raw Hessian diag score"
        else:
            exact = np.log10(np.maximum(matrices["exact"], 1e-30))
            hessian = np.log10(np.maximum(matrices["hessian"], 1e-30))
            exact_title = "log10 exact output score"
            hessian_title = "log10 Hessian diag score"
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, data, title in (
            (axes[0], exact, exact_title),
            (axes[1], hessian, hessian_title),
        ):
            image = ax.imshow(data, aspect="auto", interpolation="nearest")
            ax.set_title(title)
            ax.set_xlabel("Block col")
            ax.set_ylabel("Block row")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(module_name)
        fig.tight_layout()
        safe_name = module_name.replace(".", "_")
        fig.savefig(heatmap_dir / f"{safe_name}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from transformers import AutoTokenizer

    from qwen3_gptq import (
        dtype_from_str,
        get_qwen3,
        get_wikitext2_or_fallback_loader,
        load_custom_model_class,
        register_custom_model,
    )

    if args.custom_modeling_file:
        custom_model_cls = load_custom_model_class(args.custom_modeling_file)
        register_custom_model(custom_model_cls)
        print(f"[model] registered custom class from {args.custom_modeling_file}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    dtype = dtype_from_str(args.dtype)
    print(f"[model] loading {args.model_dir} dtype={args.dtype}", flush=True)
    model = get_qwen3(str(args.model_dir), dtype=dtype)
    model.eval()
    if args.seqlen > model.seqlen:
        print(f"[warn] --seqlen {args.seqlen} > model.seqlen {model.seqlen}; using model.seqlen.", flush=True)
        args.seqlen = model.seqlen
    model.seqlen = args.seqlen

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), use_fast=False)
    trainloader, source_info = get_wikitext2_or_fallback_loader(
        tokenizer=tokenizer,
        model_dir=str(args.model_dir),
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        seed=args.seed,
        output_dir=args.output_dir,
        local_wikitext2_dir=args.local_wikitext2_dir,
    )
    print(f"[calib] source={source_info['calib_source']} fallback={source_info['fallback_used']}", flush=True)
    max_batches = args.max_batches if args.max_batches > 0 else None

    if args.analysis_device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:0" if args.analysis_device == "cuda" else "cpu")
    model = model.to(device)
    print(f"[device] model/scoring device={device}", flush=True)

    modules_by_name = dict(model.named_modules())
    resolved = resolve_module_groups(model)
    layer_filter = parse_layer_filter(args.layer_filter)
    module_filters = [item.strip() for item in args.module_filter.split(",") if item.strip()]
    layer_work_items = build_layer_work_items(resolved, layer_filter, module_filters)
    target_module_count = sum(len(names) for _, group_items in layer_work_items for _, _, names in group_items)
    print(f"[plan] layers={len(layer_work_items)} target_modules={target_module_count}", flush=True)
    if target_module_count == 0:
        raise RuntimeError("No target modules matched the filters.")

    module_to_smooth = load_smooth_info(args.smooth_groups)
    cache_dtype = dtype_from_name(args.cache_dtype)
    top_mass_fracs = parse_float_list(args.top_mass_fracs)
    repair_fracs = parse_float_list(args.repair_fracs)

    module_summary_rows: list[dict] = []
    global_arrays: dict[str, list[np.ndarray]] = {
        "exact": [],
        "hessian": [],
        "frob": [],
        "channel": [],
        "delta": [],
    }
    heatmap_candidates: list[tuple[str, float, dict[str, np.ndarray]]] = []
    per_block_path = args.output_dir / "per_block_metrics.jsonl"
    if args.save_per_block:
        per_block_path.write_text("", encoding="utf-8")
    repair_rows_path = args.output_dir / "repair_curves_by_module.jsonl"
    repair_rows_path.write_text("", encoding="utf-8")

    start_time = time.time()
    processed = 0
    for layer_index, (layer_prefix, group_items) in enumerate(layer_work_items, start=1):
        print(f"[layer {layer_index}/{len(layer_work_items)}] {layer_prefix}", flush=True)
        if args.collect_granularity == "layer":
            source_names = [names[0] for _, _, names in group_items]
            input_cache = collect_input_cache(
                model=model,
                calibration_loader=trainloader,
                target_module_names=source_names,
                cache_dtype=cache_dtype,
                max_batches=max_batches,
            )
            group_batches = [(group_items, input_cache)]
        else:
            group_batches = []
            for item in group_items:
                source_name = item[2][0]
                cache = collect_input_cache(
                    model=model,
                    calibration_loader=trainloader,
                    target_module_names=[source_name],
                    cache_dtype=cache_dtype,
                    max_batches=max_batches,
                )
                group_batches.append(([item], cache))

        for cur_group_items, input_cache in group_batches:
            for group_name, _group_prefix, module_names in cur_group_items:
                source_name = module_names[0]
                if source_name not in input_cache:
                    raise RuntimeError(f"Missing input cache for {source_name}")
                source_module = modules_by_name[source_name]
                source_smooth = smooth_info_for_module(source_name, source_module, module_to_smooth)
                print(
                    f"  [group] {group_name} source={source_name} modules={len(module_names)} "
                    f"smooth={source_smooth.smooth_used}",
                    flush=True,
                )
                context = build_activation_context(
                    x_cache=input_cache[source_name],
                    smooth_scale=source_smooth.scale
                    if source_smooth.scale is not None
                    else torch.ones(source_module.in_features),
                    groupsize=args.groupsize,
                    block_cols=args.block_cols,
                    act_order=args.act_order,
                    score_max_tokens=args.score_max_tokens,
                    device=device,
                    seed=args.seed + processed,
                )
                for module_name in module_names:
                    module = modules_by_name[module_name]
                    smooth_info = smooth_info_for_module(module_name, module, module_to_smooth)
                    summary, block_rows_json, matrices, module_repair = analyze_module(
                        module_name=module_name,
                        module=module,
                        smooth_info=smooth_info,
                        context=context,
                        args=args,
                        device=device,
                    )
                    summary["group"] = group_name
                    module_summary_rows.append(summary)
                    global_arrays["exact"].append(matrices["exact"].reshape(-1))
                    global_arrays["hessian"].append(matrices["hessian"].reshape(-1))
                    global_arrays["frob"].append(matrices["frob"].reshape(-1))
                    global_arrays["channel"].append(matrices["channel"].reshape(-1))
                    global_arrays["delta"].append(matrices["delta"].reshape(-1))
                    total_exact = float(summary["total_exact_output_score"] or 0.0)
                    heatmap_candidates.append((module_name, total_exact, matrices))
                    if args.save_per_block:
                        write_jsonl(per_block_path, block_rows_json, mode="a")
                    for row in module_repair:
                        row = dict(row)
                        row["module"] = module_name
                        row["group"] = group_name
                        row["layer_id"] = summary["layer_id"]
                        row["module_type"] = summary["module_type"]
                        write_jsonl(repair_rows_path, [row], mode="a")
                    processed += 1
                    print(
                        "    [module] "
                        f"{module_name} exact={summary['total_exact_output_score']:.6g} "
                        f"top5={summary.get('top_0.05_mass')} "
                        f"rho_h={summary.get('corr_hessian_spearman')}",
                        flush=True,
                    )
                del context
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del input_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    write_module_summary(args.output_dir / "module_summary.csv", module_summary_rows)
    global_summary = summarize_global(
        module_rows=module_summary_rows,
        global_arrays=global_arrays,
        repair_fracs=repair_fracs,
        top_mass_fracs=top_mass_fracs,
        random_trials=args.random_trials,
        seed=args.seed,
        elapsed_seconds=elapsed,
        args=args,
    )
    global_summary["calibration"] = source_info
    global_summary_path = args.output_dir / "global_summary.json"
    global_summary_path.write_text(json.dumps(global_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.no_plots:
        arrays_for_plot = {key: np.concatenate(values).reshape(-1) for key, values in global_arrays.items() if values}
        plot_global(
            output_dir=args.output_dir,
            arrays=arrays_for_plot,
            repair_rows=global_summary["repair_curve"],
            scatter_sample=args.scatter_sample,
            seed=args.seed,
        )
        heatmap_candidates.sort(key=lambda item: item[1], reverse=True)
        plot_heatmaps(
            output_dir=args.output_dir,
            heatmap_items=[(name, matrices) for name, _, matrices in heatmap_candidates],
            max_items=args.max_plot_modules,
            scale=args.heatmap_scale,
        )

    print("[done] wrote:", args.output_dir, flush=True)
    print("[done] module summary:", args.output_dir / "module_summary.csv", flush=True)
    print("[done] global summary:", global_summary_path, flush=True)
    if args.save_per_block:
        print("[done] per-block metrics:", per_block_path, flush=True)


if __name__ == "__main__":
    main()
