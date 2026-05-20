from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass
class ActivationQuantCalib:
    scale: torch.Tensor
    zero: torch.Tensor
    bits: int
    group_size: int
    sym: bool = False

    @property
    def maxq(self) -> int:
        return 2**self.bits - 1


def _reshape_groups(x: torch.Tensor, group_size: int) -> tuple[torch.Tensor, int, int]:
    x2d = x.reshape(-1, x.shape[-1]).float()
    last_dim = x2d.shape[-1]
    if group_size <= 0:
        group_size = last_dim
    num_groups = math.ceil(last_dim / group_size)
    pad = num_groups * group_size - last_dim
    if pad > 0:
        x2d = F.pad(x2d, (0, pad), value=0.0)
    return x2d.view(x2d.shape[0], num_groups, group_size), last_dim, group_size


def _compute_qparams(
    groups: torch.Tensor,
    bits: int,
    sym: bool,
    per_token: bool,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    maxq = float(2**bits - 1)
    reduce_dims = (-1,) if per_token else (0, -1)
    xmin = groups.amin(dim=reduce_dims, keepdim=True)
    xmax = groups.amax(dim=reduce_dims, keepdim=True)
    xmin = torch.minimum(xmin, torch.zeros_like(xmin))
    xmax = torch.maximum(xmax, torch.zeros_like(xmax))
    if sym:
        xmax = torch.maximum(xmax, xmin.abs())
        xmin = -xmax
    dead = (xmin == 0) & (xmax == 0)
    xmin = torch.where(dead, torch.full_like(xmin, -1.0), xmin)
    xmax = torch.where(dead, torch.full_like(xmax, 1.0), xmax)
    scale = ((xmax - xmin) / maxq).clamp(min=eps)
    if sym:
        zero = torch.full_like(scale, (maxq + 1) / 2.0)
    else:
        zero = torch.round(-xmin / scale)
    return scale, zero


def fake_quant_activation_a4(
    x: torch.Tensor,
    group_size: int = -1,
    per_token: bool = True,
    sym: bool = False,
    eps: float = 1e-8,
    return_qparams: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    groups, last_dim, group_size = _reshape_groups(x, group_size)
    scale, zero = _compute_qparams(groups, bits=4, sym=sym, per_token=per_token, eps=eps)
    q = torch.clamp(torch.round(groups / scale) + zero, 0, 15)
    dequant = scale * (q - zero)
    dequant = dequant.view(-1, groups.shape[1] * group_size)[..., :last_dim].view_as(x)
    dequant = dequant.to(dtype=x.dtype)
    if return_qparams:
        return dequant, scale.squeeze(0).squeeze(-1), zero.squeeze(0).squeeze(-1)
    return dequant


def fake_quant_activation_int4_group_symmetric(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Per-token, per-group symmetric INT4 fake quantization used by benchmarks."""
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")

    orig_shape = x.shape
    hidden_dim = orig_shape[-1]
    x2d = x.reshape(-1, hidden_dim)

    remainder = hidden_dim % group_size
    if remainder:
        pad = group_size - remainder
        x2d = F.pad(x2d, (0, pad), value=0.0)
    else:
        pad = 0

    padded_hidden = x2d.shape[-1]
    groups = x2d.reshape(-1, padded_hidden // group_size, group_size)
    scale = (groups.abs().amax(dim=-1, keepdim=True) / 7.0).clamp(min=eps)
    q = torch.round(groups / scale).clamp(-7, 7)
    dequant = (q * scale).reshape(-1, padded_hidden)
    if pad:
        dequant = dequant[:, :hidden_dim]
    return dequant.reshape(orig_shape).to(dtype=x.dtype)


def calibrate_activation_quantizer(
    inputs: torch.Tensor,
    bits: int = 4,
    group_size: int = -1,
    sym: bool = False,
    eps: float = 1e-8,
) -> ActivationQuantCalib:
    groups, _, group_size = _reshape_groups(inputs, group_size)
    scale, zero = _compute_qparams(groups, bits=bits, sym=sym, per_token=False, eps=eps)
    return ActivationQuantCalib(
        scale=scale.squeeze(0).squeeze(-1).detach().cpu(),
        zero=zero.squeeze(0).squeeze(-1).detach().cpu(),
        bits=bits,
        group_size=group_size,
        sym=sym,
    )
