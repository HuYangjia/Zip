"""
子矩阵级混合精度 GPTQ 核心类（V9 Submatrix Mixed Precision）

核心思想：
- 将权重矩阵分割为 (brow × bcol) 的子矩阵网格
- Phase 1：对每个子矩阵做 INT4 fake-quant，计算量化误差，选出 top-k% 作为 INT8 区域
- Phase 2：在 GPTQ 逐列迭代中，根据 high_precision_mask 决定每个行段使用 INT4 还是 INT8
- GPTQ 误差传播机制完全不变

与 V7 GPTQTailAbsorb 的唯一区别：
  V7: is_int8_col = (col_idx >= tail_start)  — 固定列范围
  V9: high_precision_mask[block_row, block_col] == True  — 自适应子矩阵位置
"""

import copy
import logging
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path

# 确保能导入原始 GPTQ 库
REPO_ROOT = Path(__file__).resolve().parents[1]
GPTQ_DIR = REPO_ROOT / "gptq"
if str(GPTQ_DIR) not in sys.path:
    sys.path.insert(0, str(GPTQ_DIR))

from gptq import GPTQ  # noqa: E402
from quant import Quantizer, quantize  # noqa: E402

# 从 gptq_tail_absorb.py 导入复用组件（不重复实现）
from gptq_tail_absorb import PercentileQuantizer, _int8_fakequant_column  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 1 辅助函数：向量化 per-block INT4 fake-quant
# ---------------------------------------------------------------------------

def _vectorized_int4_fakequant_blocks(
    blocks: torch.Tensor,
    maxq: int,
    sym: bool,
) -> tuple:
    """
    对 4D 块视图一次性执行向量化的 INT4 fake-quant。

    参数:
        blocks: (nrow, ncol, brow, bcol) 4D 块视图
        maxq: 量化最大值（如 4-bit 为 15）
        sym: 是否对称量化

    返回:
        blocks_q: (nrow, ncol, brow, bcol) 量化后的块
        errors: (nrow, ncol) 每个块的 Frobenius 范数误差
    """
    # 计算每个块的 min/max，形状 (nrow, ncol)
    # 先 reshape 为 (nrow, ncol, brow*bcol) 以便沿最后一维取 min/max
    nrow, ncol, brow, bcol = blocks.shape
    blocks_flat = blocks.reshape(nrow, ncol, -1)  # (nrow, ncol, brow*bcol)

    tmp_zero = torch.zeros(nrow, ncol, device=blocks.device, dtype=blocks.dtype)
    block_min = torch.minimum(blocks_flat.amin(dim=-1), tmp_zero)  # (nrow, ncol)
    block_max = torch.maximum(blocks_flat.amax(dim=-1), tmp_zero)  # (nrow, ncol)

    if sym:
        block_max = torch.maximum(torch.abs(block_min), block_max)
        tmp_neg = block_min < 0
        block_min[tmp_neg] = -block_max[tmp_neg]

    # 处理全零块
    all_zero = (block_min == 0) & (block_max == 0)
    block_min[all_zero] = -1
    block_max[all_zero] = +1

    # 计算 per-block scale 和 zero，形状 (nrow, ncol)
    scale = (block_max - block_min) / maxq
    if sym:
        zero = torch.full_like(scale, (maxq + 1) / 2)
    else:
        zero = torch.round(-block_min / scale)

    # 扩展 scale/zero 到 (nrow, ncol, 1, 1) 以便广播
    scale_4d = scale.unsqueeze(-1).unsqueeze(-1)  # (nrow, ncol, 1, 1)
    zero_4d = zero.unsqueeze(-1).unsqueeze(-1)    # (nrow, ncol, 1, 1)

    # 向量化 quantize: q = clamp(round(x / scale) + zero, 0, maxq)
    # dequant: x_q = scale * (q - zero)
    q = torch.clamp(torch.round(blocks / scale_4d) + zero_4d, 0, maxq)
    blocks_q = scale_4d * (q - zero_4d)

    # 计算每个块的 Frobenius 范数误差
    diff = blocks - blocks_q
    errors = torch.norm(diff.reshape(nrow, ncol, -1), p=2, dim=-1)  # (nrow, ncol)

    return blocks_q, errors


def _pad_and_reshape_to_blocks(
    W: torch.Tensor,
    brow: int,
    bcol: int,
    nrow: int,
    ncol: int,
) -> torch.Tensor:
    """
    将权重矩阵 W 进行 zero-padding 并 reshape 为 4D 块视图。

    参数:
        W: (d_out, d_in) 权重矩阵
        brow, bcol: 块尺寸
        nrow, ncol: 网格尺寸

    返回:
        blocks: (nrow, ncol, brow, bcol) 4D 块视图
    """
    d_out, d_in = W.shape
    pad_rows = nrow * brow - d_out
    pad_cols = ncol * bcol - d_in

    if pad_rows > 0 or pad_cols > 0:
        # F.pad 参数顺序: (left, right, top, bottom)
        W_padded = F.pad(W, (0, pad_cols, 0, pad_rows), value=0.0)
    else:
        W_padded = W

    # reshape: (nrow*brow, ncol*bcol) -> (nrow, brow, ncol, bcol) -> (nrow, ncol, brow, bcol)
    blocks = W_padded.reshape(nrow, brow, ncol, bcol).permute(0, 2, 1, 3).contiguous()
    return blocks


# ---------------------------------------------------------------------------
# Phase 1：子矩阵敏感度评分
# ---------------------------------------------------------------------------

def compute_block_sensitivity(
    W: torch.Tensor,
    block_shape: tuple,
    budget_ratio: float,
    metric: str = "quant_error",
    quantizer: "Quantizer | None" = None,
    H_diag: "torch.Tensor | None" = None,
) -> tuple:
    """
    计算每个子矩阵块的敏感度分数，选出 top-k% 作为 INT8 高精度区域。

    参数:
        W: (d_out, d_in) 浮点权重张量（已处理 dead columns、已 act-order 排列）
        block_shape: (brow, bcol) 子矩阵尺寸
        budget_ratio: INT8 预算比例，范围 [0.0, 1.0]
        metric: 敏感度评分方法，"quant_error" / "weight_norm" / "hessian_weighted"
        quantizer: 当 metric 为 quant_error 或 hessian_weighted 时需要（已配置为 4-bit）
        H_diag: 当 metric 为 hessian_weighted 时需要，Hessian 对角线向量 (d_in,)

    返回:
        scores: (nrow, ncol) 每个块的敏感度分数
        high_precision_mask: (nrow, ncol) 布尔张量，True 表示该块使用 INT8
    """
    d_out, d_in = W.shape
    brow, bcol = block_shape
    dev = W.device

    # 计算网格尺寸（向上取整，处理边界块）
    nrow = math.ceil(d_out / brow)
    ncol = math.ceil(d_in / bcol)
    n_total = nrow * ncol

    # 计算 INT8 块数量
    n_high = max(1, round(n_total * budget_ratio))
    n_high = min(n_high, n_total)  # 不超过总块数

    if metric == "weight_norm":
        # 向量化：pad + reshape + per-block Frobenius 范数
        blocks = _pad_and_reshape_to_blocks(W, brow, bcol, nrow, ncol)
        # blocks: (nrow, ncol, brow, bcol)
        scores = torch.norm(blocks.reshape(nrow, ncol, -1), p=2, dim=-1)  # (nrow, ncol)

    elif metric == "quant_error":
        if quantizer is None:
            raise ValueError("quantizer 参数在 quant_error 模式下不能为 None")
        # 向量化：pad + reshape + per-block INT4 fake-quant + error norm
        blocks = _pad_and_reshape_to_blocks(W, brow, bcol, nrow, ncol)
        maxq = int(quantizer.maxq.item())
        sym = quantizer.sym
        _, scores = _vectorized_int4_fakequant_blocks(blocks, maxq, sym)

    elif metric == "hessian_weighted":
        if quantizer is None:
            raise ValueError("quantizer 参数在 hessian_weighted 模式下不能为 None")
        if H_diag is None:
            raise ValueError("H_diag 参数在 hessian_weighted 模式下不能为 None")
        # 向量化：pad + reshape + per-block INT4 fake-quant + Hessian 加权误差
        blocks = _pad_and_reshape_to_blocks(W, brow, bcol, nrow, ncol)
        maxq = int(quantizer.maxq.item())
        sym = quantizer.sym
        blocks_q, _ = _vectorized_int4_fakequant_blocks(blocks, maxq, sym)

        # 将 H_diag pad 到 ncol*bcol 并 reshape 为 (1, ncol, 1, bcol)
        pad_cols = ncol * bcol - d_in
        if pad_cols > 0:
            H_diag_padded = F.pad(H_diag, (0, pad_cols), value=0.0)
        else:
            H_diag_padded = H_diag
        h_diag_view = H_diag_padded.reshape(ncol, bcol).unsqueeze(0).unsqueeze(2)
        # h_diag_view: (1, ncol, 1, bcol)

        # S = sum((blocks - blocks_q)^2 * h_diag_view) per block
        err_sq = (blocks - blocks_q) ** 2  # (nrow, ncol, brow, bcol)
        scores = (err_sq * h_diag_view).sum(dim=(-2, -1))  # (nrow, ncol)
    else:
        raise ValueError(
            f"未知的 sensitivity_metric: {metric}，"
            f"支持: quant_error, weight_norm, hessian_weighted"
        )

    # 选出 top-k 块
    scores_flat = scores.flatten()
    _, topk_indices = torch.topk(scores_flat, k=n_high)
    high_precision_mask = torch.zeros(n_total, dtype=torch.bool, device=dev)
    high_precision_mask[topk_indices] = True
    high_precision_mask = high_precision_mask.reshape(nrow, ncol)

    # 打印诊断日志
    top5_k = min(5, n_total)
    top5_vals, _ = torch.topk(scores_flat, k=top5_k)
    top5_list = [f"{v:.4f}" for v in top5_vals.tolist()]
    logger.info(
        f"[Phase1] 网格 {nrow}×{ncol} = {n_total} 块, "
        f"INT8 块数: {n_high}, metric: {metric}, "
        f"top-5 scores: [{', '.join(top5_list)}]"
    )
    print(
        f"  [Phase1] grid={nrow}x{ncol} ({n_total} blocks), "
        f"INT8={n_high}, metric={metric}, "
        f"top5=[{', '.join(top5_list)}]"
    )

    return scores, high_precision_mask


# ---------------------------------------------------------------------------
# GPTQSubmatrixMixed: 子矩阵级混合精度 GPTQ
# ---------------------------------------------------------------------------


class GPTQSubmatrixMixed(GPTQ):
    """
    继承标准 GPTQ 类，修改 fasterquant 方法：
    - Phase 1：计算子矩阵敏感度评分，选出 INT8 高精度区域
    - Phase 2：逐列迭代中，根据 high_precision_mask 对每个行段选择 INT4 或 INT8 量化
    - GPTQ 误差传播完全不变
    """

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
        block_shape=(128, 128),
        budget_ratio=0.05,
        sensitivity_metric="quant_error",
    ):
        """
        子矩阵级混合精度 fasterquant。

        参数:
            blocksize: GPTQ 块大小（逐列循环的外层分块）
            percdamp: Hessian 阻尼系数
            groupsize: per-group 量化的 group 大小，-1 表示 per-channel
            actorder: 是否启用 act-order 排序
            static_groups: 是否预计算 group quantizer
            block_shape: (brow, bcol) 子矩阵尺寸
            budget_ratio: INT8 预算比例，0.0 = 纯 INT4，1.0 = 纯 INT8
            sensitivity_metric: 敏感度评分方法

        返回:
            stats: dict，包含诊断统计信息
        """
        import transformers

        brow, bcol = block_shape

        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        d_out, d_in = W.shape
        nrow = math.ceil(d_out / brow)
        ncol = math.ceil(d_in / bcol)

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        # ---- act-order 排序 ----
        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)
        else:
            perm = None
            invperm = None

        # ================================================================
        # Phase 1：子矩阵敏感度评分，确定 high_precision_mask
        # 时机：act-order 排列之后、逐列循环之前
        # ================================================================
        if budget_ratio <= 0.0:
            # 退化为标准 GPTQ：全部 INT4
            high_precision_mask = torch.zeros(nrow, ncol, dtype=torch.bool, device=self.dev)
            block_scores = torch.zeros(nrow, ncol, device=self.dev)
            print(f"  [Phase1] budget_ratio=0, 跳过评分, 全部 INT4")
        elif budget_ratio >= 1.0:
            # 全部 INT8
            high_precision_mask = torch.ones(nrow, ncol, dtype=torch.bool, device=self.dev)
            block_scores = torch.zeros(nrow, ncol, device=self.dev)
            print(f"  [Phase1] budget_ratio=1.0, 跳过评分, 全部 INT8")
        else:
            # 正常评分
            H_diag = torch.diag(H) if sensitivity_metric == "hessian_weighted" else None
            block_scores, high_precision_mask = compute_block_sensitivity(
                W=W,
                block_shape=block_shape,
                budget_ratio=budget_ratio,
                metric=sensitivity_metric,
                quantizer=self.quantizer,
                H_diag=H_diag,
            )

        n_int8_blocks = int(high_precision_mask.sum().item())
        n_total_blocks = nrow * ncol

        # ---- static_groups 预计算 ----
        # 仅为存在 INT4 行段的列范围生成 group quantizer
        if static_groups:
            groups = []
            for i in range(0, self.columns, groupsize):
                bc = i // bcol
                # 检查该列是否存在至少一个 INT4 行段
                if bc < ncol and not high_precision_mask[:, bc].all():
                    quantizer = copy.deepcopy(self.quantizer)
                    group_end = min(i + groupsize, self.columns)
                    quantizer.find_params(W[:, i:group_end], weight=True)
                    groups.append(quantizer)
                else:
                    # 该列全部是 INT8，不需要 group scale，放一个占位
                    groups.append(None)

        # ---- Hessian 逆矩阵准备 ----
        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        # ================================================================
        # Phase 2：逐列量化循环（子矩阵级混合精度）
        # ================================================================
        n_int4_segments = 0
        n_int8_segments = 0

        # 预计算列类型查找表：避免逐列循环中重复判断
        # col_is_all_int4[bc] = True 表示该 block_col 的所有行块均为 INT4
        # col_is_all_int8[bc] = True 表示该 block_col 的所有行块均为 INT8
        col_has_any_int8 = high_precision_mask.any(dim=0)   # (ncol,)
        col_is_all_int8 = high_precision_mask.all(dim=0)    # (ncol,)
        col_is_all_int4 = ~col_has_any_int8                 # (ncol,)
        # col_is_mixed = col_has_any_int8 & ~col_is_all_int8

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                col_idx = i1 + i
                block_col = col_idx // bcol

                # ---- group scale 更新（与 V7 main 列逻辑相同） ----
                if groupsize != -1:
                    if not static_groups:
                        if col_idx % groupsize == 0:
                            group_end = min(col_idx + groupsize, self.columns)
                            self.quantizer.find_params(
                                W[:, col_idx:group_end], weight=True
                            )
                    else:
                        idx = col_idx
                        if actorder:
                            idx = perm[idx]
                        grp_idx = idx // groupsize
                        if grp_idx < len(groups) and groups[grp_idx] is not None:
                            self.quantizer = groups[grp_idx]

                # ---- 逐行段混合精度量化（快速路径优化） ----
                if block_col < ncol and col_is_all_int4[block_col]:
                    # 快速路径：该列所有行块均为 INT4，直接一次量化
                    q = quantize(
                        w.unsqueeze(1),
                        self.quantizer.scale,
                        self.quantizer.zero,
                        self.quantizer.maxq,
                    ).flatten()
                    n_int4_segments += nrow

                elif block_col < ncol and col_is_all_int8[block_col]:
                    # 快速路径：该列所有行块均为 INT8，直接一次量化
                    q = _int8_fakequant_column(w)
                    n_int8_segments += nrow

                else:
                    # 混合列：先做整列 INT4，再对 INT8 行段覆盖
                    q = quantize(
                        w.unsqueeze(1),
                        self.quantizer.scale,
                        self.quantizer.zero,
                        self.quantizer.maxq,
                    ).flatten()
                    n_int4_count = 0
                    n_int8_count = 0
                    for br in range(nrow):
                        if block_col < ncol and high_precision_mask[br, block_col]:
                            r0 = br * brow
                            r1 = min(r0 + brow, d_out)
                            q[r0:r1] = _int8_fakequant_column(w[r0:r1])
                            n_int8_count += 1
                        else:
                            n_int4_count += 1
                    n_int4_segments += n_int4_count
                    n_int8_segments += n_int8_count

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                # 误差传播：全列，不区分行段精度（与 V7 完全相同）
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            # 将 block 内的误差传播到后续所有列
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        elapsed = time.time() - tick
        gptq_loss = float(torch.sum(Losses).item())

        print(f"time {elapsed:.2f}")
        print(f"error {gptq_loss}")
        print(
            f"mode: submatrix_mixed, grid={nrow}x{ncol}, "
            f"INT8 blocks: {n_int8_blocks}/{n_total_blocks}, "
            f"INT4 segments: {n_int4_segments}, INT8 segments: {n_int8_segments}"
        )

        # ---- 列顺序还原 ----
        if actorder:
            Q = Q[:, invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()

        # 将 FakeQuant 浮点权重写回 layer
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

        # ---- top-5 敏感度分数 ----
        top5_k = min(5, n_total_blocks)
        if block_scores.numel() > 0:
            top5_vals, _ = torch.topk(block_scores.flatten(), k=top5_k)
            top5_list = top5_vals.tolist()
        else:
            top5_list = []

        # 返回诊断统计信息
        stats = {
            "gptq_loss": gptq_loss,
            "elapsed_seconds": elapsed,
            "n_int4_segments": n_int4_segments,
            "n_int8_segments": n_int8_segments,
            "grid_shape": [nrow, ncol],
            "n_int8_blocks": n_int8_blocks,
            "n_total_blocks": n_total_blocks,
            "top5_sensitivity_scores": top5_list,
            "block_shape": list(block_shape),
            "budget_ratio": budget_ratio,
            "sensitivity_metric": sensitivity_metric,
        }

        return stats
