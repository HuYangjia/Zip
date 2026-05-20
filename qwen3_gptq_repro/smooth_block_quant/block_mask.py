import copy
import logging
import math

import torch
import torch.nn.functional as F

from .compat import (
    _pad_and_reshape_to_blocks,
    _vectorized_int4_fakequant_blocks,
    legacy_compute_block_sensitivity,
    quantize,
)

logger = logging.getLogger(__name__)


def compute_hessian_diag(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x2d = x.detach().float().reshape(-1, x.shape[-1])
    num_tokens = max(int(x2d.shape[0]), 1)
    h_diag = 2.0 * x2d.pow(2).sum(dim=0) / float(num_tokens)
    return h_diag.clamp(min=eps)


def compute_delta_hessian_scores(
    W: torch.Tensor,
    block_shape: tuple[int, int],
    quantizer,
    H_diag: torch.Tensor | None = None,
    groupsize: int | None = None,
) -> torch.Tensor:
    d_out, d_in = W.shape
    brow, bcol = block_shape
    nrow = math.ceil(d_out / brow)
    ncol = math.ceil(d_in / bcol)
    if groupsize is not None:
        q0 = _fake_quantize_column_groups(W, quantizer, groupsize)
        residual = W - q0
        q1 = _fake_quantize_column_groups(residual, quantizer, groupsize)
        err_before = _pad_and_reshape_to_blocks(W - q0, brow, bcol, nrow, ncol)
        err_after = _pad_and_reshape_to_blocks(W - (q0 + q1), brow, bcol, nrow, ncol)
    else:
        blocks = _pad_and_reshape_to_blocks(W, brow, bcol, nrow, ncol)
        maxq = int(quantizer.maxq.item())
        sym = quantizer.sym
        q0, _ = _vectorized_int4_fakequant_blocks(blocks, maxq=maxq, sym=sym)
        residual = blocks - q0
        q1, _ = _vectorized_int4_fakequant_blocks(residual, maxq=maxq, sym=sym)
        err_before = blocks - q0
        err_after = blocks - (q0 + q1)
    if H_diag is not None:
        pad_cols = ncol * bcol - d_in
        if pad_cols > 0:
            H_diag = F.pad(H_diag, (0, pad_cols), value=0.0)
        h_view = H_diag.reshape(ncol, bcol).unsqueeze(0).unsqueeze(2)
        before = (err_before.pow(2) * h_view).sum(dim=(-2, -1))
        after = (err_after.pow(2) * h_view).sum(dim=(-2, -1))
    else:
        before = err_before.pow(2).sum(dim=(-2, -1))
        after = err_after.pow(2).sum(dim=(-2, -1))
    return (before - after).clamp_min(0.0)


def _fake_quantize_column_groups(
    W: torch.Tensor,
    quantizer,
    groupsize: int,
) -> torch.Tensor:
    if groupsize <= 0:
        raise ValueError(f"groupsize must be positive, got {groupsize}")
    Q = torch.zeros_like(W)
    for c0 in range(0, W.shape[1], groupsize):
        c1 = min(c0 + groupsize, W.shape[1])
        qtz = copy.deepcopy(quantizer)
        qtz.find_params(W[:, c0:c1], weight=True)
        Q[:, c0:c1] = quantize(W[:, c0:c1], qtz.scale, qtz.zero, qtz.maxq)
    return Q


def compute_block_sensitivity(
    W: torch.Tensor,
    block_shape: tuple[int, int],
    budget_ratio: float,
    metric: str = "quant_error",
    quantizer=None,
    H_diag: torch.Tensor | None = None,
    groupsize: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if metric != "delta_hessian":
        return legacy_compute_block_sensitivity(
            W=W,
            block_shape=block_shape,
            budget_ratio=budget_ratio,
            metric=metric,
            quantizer=quantizer,
            H_diag=H_diag,
        )

    if quantizer is None:
        raise ValueError("quantizer 参数在 delta_hessian 模式下不能为 None")

    d_out, d_in = W.shape
    brow, bcol = block_shape
    nrow = math.ceil(d_out / brow)
    ncol = math.ceil(d_in / bcol)
    n_total = nrow * ncol
    scores = compute_delta_hessian_scores(
        W,
        block_shape,
        quantizer,
        H_diag=H_diag,
        groupsize=groupsize,
    )
    if budget_ratio <= 0:
        return scores, torch.zeros(nrow, ncol, dtype=torch.bool, device=W.device)
    if budget_ratio >= 1:
        return scores, torch.ones(nrow, ncol, dtype=torch.bool, device=W.device)

    n_high = max(1, round(n_total * budget_ratio))
    _, topk = torch.topk(scores.flatten(), k=n_high)
    mask = torch.zeros(n_total, dtype=torch.bool, device=W.device)
    mask[topk] = True
    mask = mask.view(nrow, ncol)

    top5_k = min(5, n_total)
    top5_vals, _ = torch.topk(scores.flatten(), k=top5_k)
    top5_list = [f"{v:.4f}" for v in top5_vals.tolist()]
    logger.info(
        "[BlockMask] metric=%s grid=%dx%d budget=%.4f selected=%d top5=[%s]",
        metric,
        nrow,
        ncol,
        budget_ratio,
        int(mask.sum().item()),
        ", ".join(top5_list),
    )
    return scores, mask
