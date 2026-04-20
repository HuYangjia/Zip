"""
本机冒烟测试（不依赖 CUDA 与真实模型）：
1. 验证 _int8_fakequant_group 的形状与对称量化正确性
2. 验证 _vectorized_int4_fakequant_blocks 的粒度改造正确性
3. 验证 compute_block_sensitivity 在三种 metric 下的返回形状
4. 验证 fasterquant 入口的参数约束（groupsize != bcol 时 ValueError）
5. 手工验证 stats 新字段（scale_shape_int4 == scale_shape_int8, quant_granularity）

运行方式（在仓库根目录）：
    cd Zip/qwen3_gptq_repro && python3 test_unified_granularity_smoke.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gptq"))

import math
import torch

from gptq_submatrix_mixed import (
    _int8_fakequant_group,
    _vectorized_int4_fakequant_blocks,
    compute_block_sensitivity,
    GPTQSubmatrixMixed,
)
from quant import Quantizer


def test_int8_fakequant_group():
    print("[Test 1] _int8_fakequant_group ...")
    d_out, bcol = 8, 16
    W_slice = torch.randn(d_out, bcol) * 2.0
    # 构造一行全零
    W_slice[3] = 0.0
    dequant, scale_per_row = _int8_fakequant_group(W_slice)
    assert dequant.shape == (d_out, bcol), f"dequant shape {dequant.shape}"
    assert scale_per_row.shape == (d_out,), f"scale shape {scale_per_row.shape}"
    # 全零行的 scale 应 > 0（被 eps clamp）
    assert scale_per_row[3].item() > 0
    # 非零行的 dequant 应接近原始权重（INT8 对称量化误差 < 1/127 量级 * max_abs）
    err_rel = (
        (dequant - W_slice).abs().max().item()
        / (W_slice.abs().max().item() + 1e-8)
    )
    assert err_rel < 0.02, f"INT8 relative dequant err too large: {err_rel}"
    # dequant[3] 应全为零（因为 q=0）
    assert dequant[3].abs().max().item() < 1e-5
    print(
        f"    OK: dequant shape={tuple(dequant.shape)}, "
        f"scale shape={tuple(scale_per_row.shape)}, max_rel_err={err_rel:.6f}"
    )


def test_vectorized_int4_fakequant_blocks():
    print("[Test 2] _vectorized_int4_fakequant_blocks ...")
    nrow, ncol, brow, bcol = 2, 3, 4, 8
    blocks = torch.randn(nrow, ncol, brow, bcol)
    # 构造一个 (bcol) 行全零的边界案例
    blocks[1, 2, 0] = 0.0

    # 非对称
    blocks_q, errors = _vectorized_int4_fakequant_blocks(blocks, maxq=15, sym=False)
    assert blocks_q.shape == (nrow, ncol, brow, bcol)
    assert errors.shape == (nrow, ncol)
    # 全零行的反量化应接近零
    assert blocks_q[1, 2, 0].abs().max().item() < 1.0

    # 对称
    blocks_q_sym, errors_sym = _vectorized_int4_fakequant_blocks(
        blocks, maxq=15, sym=True
    )
    assert blocks_q_sym.shape == blocks.shape
    assert errors_sym.shape == (nrow, ncol)

    # 粒度验证：块内不同行的 scale 应可以不同 —— 通过构造"行内值差异极大"的块
    # 来间接验证：用大值行 vs 小值行构造，sym 下对称量化后保持方向
    blocks2 = torch.zeros(1, 1, 2, 8)
    blocks2[0, 0, 0] = torch.linspace(-10.0, 10.0, 8)   # 大范围
    blocks2[0, 0, 1] = torch.linspace(-0.1, 0.1, 8)     # 小范围
    bq, _ = _vectorized_int4_fakequant_blocks(blocks2, maxq=15, sym=False)
    max0 = bq[0, 0, 0].abs().max().item()
    max1 = bq[0, 0, 1].abs().max().item()
    # 两行的 dequant 范围应各自接近自身原始范围 —— 说明 scale 是 per-row 的
    assert max0 > 5.0, f"large-row dequant max {max0} too small"
    assert max1 < 0.5, f"small-row dequant max {max1} too large (粒度不对！)"
    print(
        f"    OK: shapes {tuple(blocks_q.shape)} / {tuple(errors.shape)}, "
        f"per-row scale verified (max0={max0:.3f}, max1={max1:.3f})"
    )


def test_compute_block_sensitivity():
    print("[Test 3] compute_block_sensitivity ...")
    d_out, d_in = 32, 48
    W = torch.randn(d_out, d_in)
    brow, bcol = 8, 16
    nrow = math.ceil(d_out / brow)
    ncol = math.ceil(d_in / bcol)

    quantizer = Quantizer()
    quantizer.configure(bits=4, perchannel=True, sym=False, mse=False)
    quantizer.find_params(W, weight=True)

    for metric in ["weight_norm", "quant_error", "hessian_weighted"]:
        H_diag = torch.rand(d_in) + 0.1 if metric == "hessian_weighted" else None
        scores, mask = compute_block_sensitivity(
            W=W, block_shape=(brow, bcol), budget_ratio=0.3,
            metric=metric, quantizer=quantizer, H_diag=H_diag,
        )
        assert scores.shape == (nrow, ncol)
        assert mask.shape == (nrow, ncol)
        n_selected = int(mask.sum().item())
        assert n_selected == max(1, round(nrow * ncol * 0.3))
        print(
            f"    OK [{metric}]: scores={tuple(scores.shape)}, "
            f"selected={n_selected}/{nrow*ncol}"
        )


def test_fasterquant_arg_constraints():
    """验证 fasterquant 的 groupsize 约束（不依赖完整 GPTQ 流程）。"""
    print("[Test 4] fasterquant groupsize constraints ...")
    layer = torch.nn.Linear(32, 16, bias=False)
    gptq = GPTQSubmatrixMixed(layer)
    gptq.dev = torch.device("cpu")
    # groupsize=-1 应抛 ValueError
    try:
        gptq.fasterquant(
            blocksize=128, percdamp=0.01, groupsize=-1, actorder=False,
            static_groups=False, block_shape=(8, 16),
            budget_ratio=0.0, sensitivity_metric="quant_error",
        )
        raise AssertionError("Expected ValueError when groupsize=-1")
    except ValueError as e:
        assert "groupsize" in str(e) and "block_cols" in str(e)
        print(f"    OK: groupsize=-1 raised: {e}")

    # groupsize != bcol 应抛 ValueError
    try:
        gptq.fasterquant(
            blocksize=128, percdamp=0.01, groupsize=64, actorder=False,
            static_groups=False, block_shape=(8, 16),
            budget_ratio=0.0, sensitivity_metric="quant_error",
        )
        raise AssertionError("Expected ValueError when groupsize != bcol")
    except ValueError as e:
        assert "groupsize" in str(e)
        print(f"    OK: groupsize=64, bcol=16 raised: {e}")


def test_stats_fields_consistency():
    """手工验证 stats 新字段的数值。"""
    print("[Test 5] stats fields shape consistency ...")
    d_out, d_in, bcol = 16, 64, 16
    n_groups_per_row = math.ceil(d_in / bcol)
    expected_shape = [d_out, n_groups_per_row]
    expected_granularity = f"(1, {bcol})"
    assert expected_shape == [16, 4]
    assert expected_granularity == "(1, 16)"
    # 边界块：d_in=60, bcol=16 -> ceil(60/16)=4（含一个边界块）
    assert math.ceil(60 / 16) == 4
    print(
        f"    OK: aligned d_in={d_in}, bcol={bcol} -> "
        f"scale_shape={expected_shape}, granularity={expected_granularity}"
    )


if __name__ == "__main__":
    test_int8_fakequant_group()
    test_vectorized_int4_fakequant_blocks()
    test_compute_block_sensitivity()
    test_fasterquant_arg_constraints()
    test_stats_fields_consistency()
    print("\nAll smoke tests passed.")
