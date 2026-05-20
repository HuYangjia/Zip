import torch


def get_input_absmax(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().abs().reshape(-1, x.shape[-1]).amax(dim=0)


def get_weight_absmax(weight: torch.Tensor) -> torch.Tensor:
    return weight.detach().float().abs().amax(dim=0)


def compute_smooth_scale(
    x_absmax: torch.Tensor,
    w_absmax: torch.Tensor,
    alpha: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    x_absmax = x_absmax.clamp(min=eps)
    w_absmax = w_absmax.clamp(min=eps)
    scale = x_absmax.pow(alpha) / w_absmax.pow(1.0 - alpha)
    return scale.clamp(min=eps)


def smooth_weight(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return weight * scale.view(1, -1).to(device=weight.device, dtype=weight.dtype)


def smooth_input(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    view_shape = [1] * x.ndim
    view_shape[-1] = -1
    return x / scale.view(view_shape).to(device=x.device, dtype=x.dtype)


def apply_smooth_to_linear_weight(linear, scale: torch.Tensor) -> None:
    linear.weight.data.mul_(scale.view(1, -1).to(device=linear.weight.device, dtype=linear.weight.dtype))
