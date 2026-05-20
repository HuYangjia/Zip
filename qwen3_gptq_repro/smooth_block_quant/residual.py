import copy

import torch

from .compat import quantize


def fake_quantize_weight(
    W: torch.Tensor,
    quantizer,
    groupsize: int = -1,
) -> torch.Tensor:
    if groupsize == -1:
        qtz = copy.deepcopy(quantizer)
        qtz.find_params(W, weight=True)
        return quantize(W, qtz.scale, qtz.zero, qtz.maxq)
    if groupsize <= 0:
        raise ValueError(f"groupsize must be positive or -1, got {groupsize}")

    Q = torch.zeros_like(W)
    for c0 in range(0, W.shape[1], groupsize):
        c1 = min(c0 + groupsize, W.shape[1])
        qtz = copy.deepcopy(quantizer)
        qtz.find_params(W[:, c0:c1], weight=True)
        Q[:, c0:c1] = quantize(W[:, c0:c1], qtz.scale, qtz.zero, qtz.maxq)
    return Q


def fit_selected_block_residual_int4(
    W_smooth: torch.Tensor,
    Q0_dense_int4: torch.Tensor,
    high_precision_mask: torch.Tensor,
    block_shape: tuple[int, int],
    residual_quantizer,
    groupsize: int = -1,
) -> torch.Tensor:
    brow, bcol = block_shape
    d_out, d_in = W_smooth.shape
    nrow, ncol = high_precision_mask.shape
    Q1 = torch.zeros_like(W_smooth)
    residual = W_smooth - Q0_dense_int4

    for br in range(nrow):
        for bc in range(ncol):
            if not bool(high_precision_mask[br, bc]):
                continue
            r0, r1 = br * brow, min((br + 1) * brow, d_out)
            c0, c1 = bc * bcol, min((bc + 1) * bcol, d_in)
            Q1[r0:r1, c0:c1] = fake_quantize_weight(
                residual[r0:r1, c0:c1],
                residual_quantizer,
                groupsize=groupsize,
            )
    return Q1


def build_w4_plus_residual_proxy(
    W_smooth: torch.Tensor,
    block_shape: tuple[int, int],
    high_precision_mask: torch.Tensor,
    base_quantizer,
    residual_quantizer=None,
    groupsize: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if residual_quantizer is None:
        residual_quantizer = base_quantizer
    Q0 = fake_quantize_weight(W_smooth, base_quantizer, groupsize=groupsize)
    Q1 = fit_selected_block_residual_int4(
        W_smooth,
        Q0,
        high_precision_mask,
        block_shape,
        residual_quantizer=residual_quantizer,
        groupsize=groupsize,
    )
    return Q0, Q1, Q0 + Q1


def pack_residual_block_metadata(
    high_precision_mask: torch.Tensor,
    block_shape: tuple[int, int],
    smooth_scale: torch.Tensor | None = None,
) -> dict:
    selected = high_precision_mask.nonzero(as_tuple=False).tolist()
    return {
        "block_shape": list(block_shape),
        "selected_block_indices": selected,
        "selected_block_count": len(selected),
        "grid_shape": list(high_precision_mask.shape),
        "smooth_scale": None if smooth_scale is None else smooth_scale.detach().cpu(),
    }
