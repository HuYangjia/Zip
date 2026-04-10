"""
Tail/Head Absorb 版 GPTQ 核心类 —— 指定列参与 INT8 量化 + FakeQuant

核心思想：
- GPTQTailAbsorb: 在 GPTQ 的 fasterquant 逐列迭代中，
  支持两种模式：
  · Tail Absorb（V7 默认）：actorder 排序后的最后 tail_rank 列（最不重要）使用 INT8 量化
  · Head Absorb（V8）：actorder 排序后的最前面 tail_rank 列（最重要）使用 INT8 量化
  其余 main 列使用 4-bit 量化。
  两种列都正常计算 Err1 并向右传播，保持 GPTQ 补偿链完整。
- PercentileQuantizer: 用第 k 百分位数确定量化 scale，替代默认的 min/max。
- FakeQuant: 量化后立即反量化，Q 矩阵存储的是浮点值。
"""

import copy
import math
import time

import torch
import torch.nn as nn

import sys
from pathlib import Path

# 确保能导入原始 GPTQ 库
REPO_ROOT = Path(__file__).resolve().parents[1]
GPTQ_DIR = REPO_ROOT / "gptq"
if str(GPTQ_DIR) not in sys.path:
    sys.path.insert(0, str(GPTQ_DIR))

from gptq import GPTQ  # noqa: E402
from quant import Quantizer, quantize  # noqa: E402


# ---------------------------------------------------------------------------
# PercentileQuantizer: 用 percentile 替代 min/max 计算量化 scale
# （从 gptq_tail_spill.py 复制，保持独立，不跨文件导入）
# ---------------------------------------------------------------------------

class PercentileQuantizer(Quantizer):
    """
    继承标准 Quantizer，重写 find_params 方法：
    使用第 k 百分位数（默认 75）来确定量化范围，而非 min/max。
    这样可以截断 outlier，避免它们拉大 scale。
    """

    def __init__(self, shape=1, percentile_k=75.0):
        super().__init__(shape)
        self.percentile_k = percentile_k
        self.clip_ratio = 0.0  # 记录被截断的权重比例

    def find_params(self, x, weight=False):
        """
        用 percentile 确定量化范围。
        对于 sym=True 的情况：
          xmax = percentile(|x|, k/100) per row
          scale = xmax / ((maxq+1)/2 - 1)  即 xmax / q_max
        """
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        # 用 percentile 替代 min/max
        q_frac = self.percentile_k / 100.0
        abs_x = x.abs()

        # torch.quantile 需要 float32/float64
        abs_x_f = abs_x.float()
        xmax = torch.quantile(abs_x_f, q=q_frac, dim=1).to(dev)

        # 记录被截断的比例（超出 percentile 范围的元素）
        clip_mask = abs_x > xmax.unsqueeze(1)
        total_elements = abs_x.numel()
        clipped_elements = clip_mask.sum().item()
        self.clip_ratio = clipped_elements / max(total_elements, 1)

        if self.sym:
            # 对称量化：xmin = -xmax
            xmin = -xmax
        else:
            # 非对称：也用 percentile 确定 min
            xmin = -xmax  # 简化处理，对称截断

        # 处理全零行
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
            self.scale = xmax
            self.zero = xmin
        else:
            self.scale = (xmax - xmin) / self.maxq
            if self.sym:
                self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
            else:
                self.zero = torch.round(-xmin / self.scale)

        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            reshape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(reshape)
            self.zero = self.zero.reshape(reshape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)


# ---------------------------------------------------------------------------
# INT8 per-column 对称量化辅助函数（FakeQuant：量化后立即反量化）
# ---------------------------------------------------------------------------

def _int8_fakequant_column(w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    对单列权重向量执行 INT8 对称量化 + 反量化（FakeQuant）。

    参数:
        w: [rows] 一维浮点向量
        eps: 防止除零的小常数

    返回:
        q_dequant: [rows] 反量化后的浮点向量
    """
    max_abs = w.abs().max()
    scale = torch.clamp(max_abs / 127.0, min=eps)
    q_int = torch.clamp(torch.round(w / scale), -128, 127)
    q_dequant = q_int * scale
    return q_dequant


# ---------------------------------------------------------------------------
# GPTQTailAbsorb: 修改版 GPTQ，tail 列使用 INT8 量化并正常传播误差
# ---------------------------------------------------------------------------

class GPTQTailAbsorb(GPTQ):
    """
    继承标准 GPTQ 类，修改 fasterquant 方法：
    - 接受 tail_rank 参数
    - actorder 排序后，最后 tail_rank 列作为 tail 列
    - tail 列使用 INT8 per-column 对称量化（FakeQuant），正常计算 Err1 并向右传播
    - main 列使用 4-bit Percentile 量化（FakeQuant）
    - GPTQ 补偿链保持完整
    - 量化完成后使用 invperm 还原列顺序
    """

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
        tail_rank=0,
        head_absorb=False,
    ):
        """
        Tail/Head Absorb 版 fasterquant。

        参数:
            tail_rank: INT8 列数量（绝对值）。
                       head_absorb=False（默认，V7 Tail Absorb）：
                         排序后的最后 tail_rank 列使用 INT8 量化。
                       head_absorb=True（V8 Head Absorb）：
                         排序后的最前面 tail_rank 列使用 INT8 量化。
                       0 表示不使用 INT8 列（退化为标准 GPTQ）。
            head_absorb: 是否启用 Head Absorb 模式。
                         True = 最重要的列（排序后最前面）使用 INT8。
                         False = 最不重要的列（排序后最后面）使用 INT8。

        返回:
            stats: dict，包含诊断统计信息
        """
        import transformers

        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        # 提前计算 INT8 列范围，供 static_groups 预计算和逐列判断使用
        # tail_rank=0 时所有列都是 main，退化为标准 GPTQ
        tail_rank = max(0, min(tail_rank, self.columns - 1))

        if head_absorb:
            # Head Absorb（V8）：排序后最前面 tail_rank 列为 INT8 列
            # INT8 列范围：[0, head_end)，main 列范围：[head_end, columns)
            head_end = tail_rank
            # main 列的起始位置（用于 static_groups 和动态 group）
            main_start = head_end
            main_end = self.columns
        else:
            # Tail Absorb（V7 默认）：排序后最后 tail_rank 列为 INT8 列
            # main 列范围：[0, tail_start)，INT8 列范围：[tail_start, columns)
            tail_start = self.columns - tail_rank
            main_start = 0
            main_end = tail_start

        if static_groups:
            # 仅为 main 列范围生成预计算 quantizer
            # INT8 列使用独立的 INT8 量化，不需要 group scale
            groups = []
            for i in range(main_start, main_end, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                # 最后一个 group 截断到 main_end，排除 INT8 列数据
                group_end = min(i + groupsize, main_end)
                quantizer.find_params(W[:, i:group_end], weight=True)
                groups.append(quantizer)

        # act-order 排序：按 Hessian 对角线降序排列
        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)
        else:
            perm = None
            invperm = None

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        # 统计信息
        n_main_quantized = 0
        n_tail_int8 = 0

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
                if head_absorb:
                    is_int8_col = (col_idx < head_end)
                else:
                    is_int8_col = (col_idx >= tail_start)

                if is_int8_col:
                    # ---- INT8 列：per-column 对称量化（FakeQuant） ----
                    # 忽略 groupsize 的 group scale，使用独立的 INT8 scale
                    q = _int8_fakequant_column(w)
                    Q1[:, i] = q
                    Losses1[:, i] = (w - q) ** 2 / d ** 2

                    # 正常计算误差并向右传播（这是与 Tail Spill 的核心区别）
                    err1 = (w - q) / d
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1
                    n_tail_int8 += 1
                else:
                    # ---- Main 列：4-bit 量化（FakeQuant） ----
                    if groupsize != -1:
                        if not static_groups:
                            if (col_idx - main_start) % groupsize == 0:
                                # group 取值范围截断到 main_end，
                                # 排除 INT8 列数据对 main 列 group scale 的污染。
                                group_end = min(col_idx + groupsize, main_end)
                                if group_end > col_idx:
                                    self.quantizer.find_params(
                                        W[:, col_idx:group_end], weight=True
                                    )
                        else:
                            idx = col_idx
                            if actorder:
                                idx = perm[idx]
                            self.quantizer = groups[(idx - main_start) // groupsize]

                    q = quantize(
                        w.unsqueeze(1),
                        self.quantizer.scale,
                        self.quantizer.zero,
                        self.quantizer.maxq,
                    ).flatten()
                    Q1[:, i] = q
                    Losses1[:, i] = (w - q) ** 2 / d ** 2

                    err1 = (w - q) / d
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1
                    n_main_quantized += 1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            # 将 block 内的误差传播到后续所有列
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        elapsed = time.time() - tick
        print('time %.2f' % elapsed)
        print('error', torch.sum(Losses).item())
        mode_str = 'head_absorb' if head_absorb else 'tail_absorb'
        print(f'mode: {mode_str}, main quantized (4-bit): {n_main_quantized}, '
              f'int8 quantized: {n_tail_int8}')

        # 列顺序还原：使用 invperm 将排序后的 Q 还原回原始列顺序
        if actorder:
            Q = Q[:, invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()

        # 将 FakeQuant 浮点权重写回 layer（列顺序已还原）
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

        # 返回诊断统计信息
        stats = {
            "n_main_quantized": n_main_quantized,
            "n_tail_int8": n_tail_int8,
            "tail_rank": tail_rank,
            "head_absorb": head_absorb,
            "main_start": main_start,
            "main_end": main_end,
            "gptq_loss": float(torch.sum(Losses).item()),
            "elapsed_seconds": elapsed,
        }

        return stats
