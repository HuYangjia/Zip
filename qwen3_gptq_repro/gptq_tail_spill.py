"""
修改版 GPTQ 核心类 —— 支持 Percentile Scale + Tail Spill

核心思想：
- GPTQTailSpill: 在 GPTQ 的 fasterquant 逐列迭代中，当列索引 >= tail_start 时
  跳过 4-bit 量化（保留浮点值），但 Hessian 误差补偿仍然正常传播。
  这样 main 列的量化误差会自然"溢出"到 tail 列。
- PercentileQuantizer: 用第 k 百分位数确定量化 scale，替代默认的 min/max。
- quantize_tail_int8: 对 GPTQ 处理后的 tail 列做 per-row INT8 对称量化。
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
# GPTQTailSpill: 修改版 GPTQ，支持 tail 列跳过量化
# ---------------------------------------------------------------------------

class GPTQTailSpill(GPTQ):
    """
    继承标准 GPTQ 类，修改 fasterquant 方法：
    - 接受 tail_start 参数
    - 当列索引 >= tail_start 时，跳过 4-bit 量化（不调用 quantize）
    - Hessian 误差补偿仍然正常传播到后续列
    - tail 列保留浮点值（已吸收 main 列的量化误差）
    """

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
        tail_start=-1,
    ):
        """
        修改版 fasterquant，支持 tail 列跳过。

        参数:
            tail_start: 列索引 >= tail_start 的列跳过量化。
                        -1 表示不跳过（等同于标准 GPTQ）。
                        注意：这是在排序前的原始列索引。

        返回:
            tail_spill_stats: dict，包含 tail 相关的诊断信息
        """
        import transformers

        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        # 保存原始权重用于诊断
        W_orig = W.clone()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)], weight=True)
                groups.append(quantizer)

        # act-order 排序
        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

            # 将原始列索引的 tail_start 映射到排序后的索引
            # tail 列 = 原始索引 >= tail_start 的列
            if tail_start >= 0:
                # 创建 mask：原始索引 >= tail_start 的列在排序后的位置
                orig_is_tail = torch.zeros(self.columns, dtype=torch.bool, device=self.dev)
                orig_is_tail[tail_start:] = True
                # 排序后的 is_tail mask
                sorted_is_tail = orig_is_tail[perm]
            else:
                sorted_is_tail = None
        else:
            perm = None
            invperm = None
            if tail_start >= 0:
                sorted_is_tail = torch.zeros(self.columns, dtype=torch.bool, device=self.dev)
                sorted_is_tail[tail_start:] = True
            else:
                sorted_is_tail = None

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
        n_tail_skipped = 0

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
                is_tail_col = False
                if sorted_is_tail is not None and sorted_is_tail[col_idx]:
                    is_tail_col = True

                if is_tail_col:
                    # Tail 列：跳过量化，保留浮点值
                    # Q1 记录的是"量化后"的值，对 tail 列就是原始浮点值
                    Q1[:, i] = w
                    # 没有量化误差，所以 err = 0，不传播误差
                    # 但 Losses 记录为 0
                    Losses1[:, i] = 0
                    # Err1 保持 0，不影响后续列
                    n_tail_skipped += 1
                else:
                    # Main 列：正常 GPTQ 4-bit 量化
                    if groupsize != -1:
                        if not static_groups:
                            if (i1 + i) % groupsize == 0:
                                self.quantizer.find_params(
                                    W[:, (i1 + i):(i1 + i + groupsize)], weight=True
                                )
                        else:
                            idx = i1 + i
                            if actorder:
                                idx = perm[idx]
                            self.quantizer = groups[idx // groupsize]

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
        print(f'main quantized: {n_main_quantized}, tail skipped: {n_tail_skipped}')

        if actorder:
            Q = Q[:, invperm]
            W_orig_sorted = W_orig  # W_orig 是排序前的

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()

        # 将结果写回 layer
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

        # 收集诊断统计
        Q_final = Q.reshape(self.layer.weight.shape).float()
        if isinstance(self.layer, nn.Conv2d):
            Q_final = Q_final.flatten(1)

        tail_spill_stats = {
            "n_main_quantized": n_main_quantized,
            "n_tail_skipped": n_tail_skipped,
            "gptq_loss": float(torch.sum(Losses).item()),
            "elapsed_seconds": elapsed,
        }

        return tail_spill_stats


# ---------------------------------------------------------------------------
# quantize_tail_int8: 对 tail 列做 per-row INT8 对称量化
# ---------------------------------------------------------------------------

def quantize_tail_int8(w_tail: torch.Tensor, eps: float = 1e-8):
    """
    对 tail 列做 per-row INT8 对称量化。

    参数:
        w_tail: [d_out, tail_cols] 浮点权重（已吸收 main 误差）
        eps: 防止除零的小常数

    返回:
        w_tail_q: [d_out, tail_cols] 量化后反量化的权重
        stats: dict，包含量化误差、饱和率等统计信息
    """
    w_tail_f = w_tail.float()
    max_abs = w_tail_f.abs().amax(dim=1, keepdim=True)
    scales = torch.clamp(max_abs / 127.0, min=eps)

    q = torch.clamp(torch.round(w_tail_f / scales), -127, 127)
    w_tail_q = q * scales

    # 统计信息
    quant_error = (w_tail_q - w_tail_f).norm(dim=1)  # per-row L2 误差
    sat_mask = q.abs() >= 127
    sat_ratio = sat_mask.float().mean(dim=1)  # per-row 饱和率

    stats = {
        "quant_error_norm_mean": float(quant_error.mean().item()),
        "quant_error_norm_max": float(quant_error.max().item()),
        "saturation_ratio_mean": float(sat_ratio.mean().item()),
        "saturation_ratio_max": float(sat_ratio.max().item()),
        "scale_min": float(scales.min().item()),
        "scale_max": float(scales.max().item()),
        "tail_abs_max": float(max_abs.max().item()),
    }

    return w_tail_q, stats
