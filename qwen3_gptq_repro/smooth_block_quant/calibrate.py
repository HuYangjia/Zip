from dataclasses import dataclass
import gc
import logging
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation_quant import (
    calibrate_activation_quantizer,
    fake_quant_activation_int4_group_symmetric,
)
from .block_mask import compute_block_sensitivity
from .compat import Quantizer
from .mixed_gptq import GPTQSubmatrixMixedV2
from .output_error import compute_linear_output_error
from .residual import build_w4_plus_residual_proxy, fake_quantize_weight
from .smooth import (
    compute_smooth_scale,
    get_weight_absmax,
    smooth_input,
    smooth_weight,
)

logger = logging.getLogger(__name__)


@dataclass
class ModuleSearchResult:
    module_name: str
    alpha: float
    loss: float
    loss_before_residual: float
    smooth_scale: torch.Tensor
    block_scores: torch.Tensor
    block_mask: torch.Tensor
    act_order_perm: torch.Tensor | None = None


@dataclass
class GroupSearchResult:
    group_key: str
    alpha: float
    loss: float
    module_results: dict[str, ModuleSearchResult]


@dataclass
class SharedSmoothGroupSearchResult:
    group_key: str
    alpha: float
    loss: float
    smooth_scale: torch.Tensor
    module_results: dict[str, ModuleSearchResult]


def make_quantizer(bits: int = 4, sym: bool = False) -> Quantizer:
    quantizer = Quantizer()
    quantizer.configure(bits=bits, perchannel=True, sym=sym, mse=False)
    return quantizer


def _apply_act_order_for_search(
    W: torch.Tensor,
    H_diag: torch.Tensor,
    actorder: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not actorder:
        return W, H_diag, None
    perm = torch.argsort(H_diag, descending=True)
    return W[:, perm], H_diag[perm], perm


def _restore_columns(
    W: torch.Tensor,
    perm: torch.Tensor | None,
) -> torch.Tensor:
    if perm is None:
        return W
    invperm = torch.argsort(perm)
    return W[:, invperm]


def _validate_chunk_size(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def input_absmax_proxy(
    x: torch.Tensor,
    batch_size: int = 1,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Compute input absmax without materializing the whole calibration cache on GPU."""
    batch_size = _validate_chunk_size("batch_size", batch_size)
    device = torch.device(device) if device is not None else x.device
    result = None
    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        x_chunk = x[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
        cur = x_chunk.detach().abs().reshape(-1, x_chunk.shape[-1]).amax(dim=0)
        result = cur if result is None else torch.maximum(result, cur)
        del x_chunk, cur
    if result is None:
        raise ValueError("cannot compute input absmax from an empty tensor")
    return result


def hessian_diag_proxy(
    x: torch.Tensor,
    smooth_scale: torch.Tensor,
    batch_size: int = 1,
    device: torch.device | str | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute the diagonal Hessian proxy in chunks to avoid large fp32 activations."""
    batch_size = _validate_chunk_size("batch_size", batch_size)
    device = torch.device(device) if device is not None else smooth_scale.device
    scale = smooth_scale.to(device=device, dtype=torch.float32)
    h_sum = None
    num_tokens = 0
    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        x_chunk = x[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
        x_s = smooth_input(x_chunk, scale)
        x2d = x_s.detach().reshape(-1, x_s.shape[-1])
        cur = x2d.pow(2).sum(dim=0)
        h_sum = cur if h_sum is None else h_sum + cur
        num_tokens += int(x2d.shape[0])
        del x_chunk, x_s, x2d, cur
    if h_sum is None:
        raise ValueError("cannot compute Hessian diagonal from an empty tensor")
    return (2.0 * h_sum / float(max(num_tokens, 1))).clamp(min=eps)


def layer_output_mse_proxy(
    x: torch.Tensor,
    y_fp: torch.Tensor,
    W_hat: torch.Tensor,
    bias: torch.Tensor | None,
    smooth_scale: torch.Tensor,
    act_group_size: int,
    act_sym: bool = True,
    batch_size: int = 1,
) -> float:
    batch_size = _validate_chunk_size("batch_size", batch_size)
    if x.shape[0] != y_fp.shape[0]:
        raise ValueError(f"input/output batch mismatch: {tuple(x.shape)} vs {tuple(y_fp.shape)}")

    device = W_hat.device
    W_eval = W_hat.float()
    bias_eval = None if bias is None else bias.to(device=device, dtype=torch.float32)
    scale = smooth_scale.to(device=device, dtype=torch.float32)
    total_sq_error = 0.0
    total_numel = 0

    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        x_chunk = x[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
        y_chunk = y_fp[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
        x_s = smooth_input(x_chunk, scale)
        if act_sym:
            x_q = fake_quant_activation_int4_group_symmetric(
                x_s,
                group_size=act_group_size,
            )
        else:
            from .activation_quant import fake_quant_activation_a4

            x_q = fake_quant_activation_a4(
                x_s,
                group_size=act_group_size,
                per_token=True,
                sym=act_sym,
            )
        out_q = F.linear(x_q, W_eval, bias_eval)
        if out_q.shape != y_chunk.shape:
            raise ValueError(f"output shape mismatch: quantized {tuple(out_q.shape)} vs fp {tuple(y_chunk.shape)}")
        diff = out_q - y_chunk
        total_sq_error += float(diff.pow(2).sum().item())
        total_numel += int(diff.numel())
        del x_chunk, y_chunk, x_s, x_q, out_q, diff

    if total_numel == 0:
        raise ValueError("cannot compute MSE from empty tensors")
    return total_sq_error / total_numel


def _add_gptq_batches_streaming(
    gptq: GPTQSubmatrixMixedV2,
    inputs: torch.Tensor,
    smooth_scale: torch.Tensor | None,
    groupsize: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    batch_size = _validate_chunk_size("batch_size", batch_size)
    smooth_scale_dev = (
        smooth_scale.to(device=device)
        if smooth_scale is not None
        else None
    )
    for start in range(0, inputs.shape[0], batch_size):
        end = min(start + batch_size, inputs.shape[0])
        x = inputs[start:end].to(device=device, dtype=dtype, non_blocking=True)
        if smooth_scale_dev is not None:
            x = smooth_input(x, smooth_scale_dev)
        x = fake_quant_activation_int4_group_symmetric(x, group_size=groupsize)
        gptq.add_batch(x, None)
        del x



def resolve_module_groups(model: nn.Module, module_groups: dict[str, list[str]]) -> dict[str, dict[str, list[str]]]:
    groups: dict[str, dict[str, list[str]]] = {}
    linear_names = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    }
    for group_name, suffixes in module_groups.items():
        groups[group_name] = {}
        for name in linear_names:
            for suffix in suffixes:
                suffix_token = suffix if suffix.startswith(".") else f".{suffix}"
                if name.endswith(suffix_token):
                    prefix = name[: -len(suffix_token)]
                    groups[group_name].setdefault(prefix, []).append(name)
        for prefix in list(groups[group_name].keys()):
            groups[group_name][prefix].sort()
    return groups


def _move_batch_to_device(batch, device: torch.device):
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: _move_batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [_move_batch_to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(_move_batch_to_device(v, device) for v in batch)
    return batch


@torch.no_grad()
def collect_linear_module_io(
    model: nn.Module,
    calibration_loader: Iterable,
    target_module_names: Iterable[str] | None = None,
    max_batches: int | None = None,
    cache_device: str | torch.device = "cpu",
    cache_dtype: torch.dtype | None = torch.float16,
) -> dict[str, dict[str, torch.Tensor]]:
    model_device = next(model.parameters()).device
    selected = set(target_module_names or [])
    if not selected:
        selected = {
            name for name, module in model.named_modules() if isinstance(module, nn.Linear)
        }
    caches = {name: {"inputs": [], "outputs": []} for name in selected}
    handles = []

    for name, module in model.named_modules():
        if name not in selected:
            continue

        def hook(_module, inp, out, module_name=name):
            x = inp[0].detach()
            y = out.detach()
            if cache_dtype is None:
                x = x.to(device=cache_device)
                y = y.to(device=cache_device)
            else:
                x = x.to(device=cache_device, dtype=cache_dtype)
                y = y.to(device=cache_device, dtype=cache_dtype)
            caches[module_name]["inputs"].append(x)
            caches[module_name]["outputs"].append(y)

        handles.append(module.register_forward_hook(hook))

    old_use_cache = getattr(getattr(model, "config", None), "use_cache", None)
    if old_use_cache is not None:
        model.config.use_cache = False
    try:
        for batch_idx, batch in enumerate(calibration_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = _move_batch_to_device(batch, model_device)
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
    for name, cache in caches.items():
        if not cache["inputs"]:
            continue
        inputs = cache["inputs"][0] if len(cache["inputs"]) == 1 else torch.cat(cache["inputs"], dim=0)
        outputs = cache["outputs"][0] if len(cache["outputs"]) == 1 else torch.cat(cache["outputs"], dim=0)
        merged[name] = {
            "inputs": inputs,
            "outputs": outputs,
        }
    del caches
    return merged


def _evaluate_alpha_candidate(
    module_name: str,
    module: nn.Linear,
    x: torch.Tensor,
    y_fp: torch.Tensor,
    alpha: float,
    block_shape: tuple[int, int],
    budget_ratio: float,
    groupsize: int,
    act_group_size: int,
    weight_quantizer,
    residual_quantizer,
    second_path: str,
    actorder: bool = False,
    search_eval_batch_size: int = 1,
) -> ModuleSearchResult:
    x_absmax = input_absmax_proxy(
        x,
        batch_size=search_eval_batch_size,
        device=module.weight.device,
    )
    w_absmax = get_weight_absmax(module.weight.data)
    smooth_scale = compute_smooth_scale(x_absmax, w_absmax, alpha)
    W_smooth = smooth_weight(module.weight.data.float(), smooth_scale)
    H_diag = hessian_diag_proxy(
        x,
        smooth_scale=smooth_scale,
        batch_size=search_eval_batch_size,
        device=module.weight.device,
    )
    W_for_mask, H_diag_for_mask, act_order_perm = _apply_act_order_for_search(
        W_smooth,
        H_diag,
        actorder,
    )
    scores, mask = compute_block_sensitivity(
        W=W_for_mask,
        block_shape=block_shape,
        budget_ratio=budget_ratio,
        metric="delta_hessian",
        quantizer=weight_quantizer,
        H_diag=H_diag_for_mask,
        groupsize=groupsize,
    )
    Q0_for_mask = fake_quantize_weight(W_for_mask, weight_quantizer, groupsize=groupsize)
    Q0 = _restore_columns(Q0_for_mask, act_order_perm)
    loss_before = layer_output_mse_proxy(
        x=x,
        y_fp=y_fp,
        W_hat=Q0,
        bias=module.bias,
        smooth_scale=smooth_scale,
        act_group_size=act_group_size,
        batch_size=search_eval_batch_size,
    )
    if second_path == "residual_int4":
        _, _, W_hat = build_w4_plus_residual_proxy(
            W_for_mask,
            block_shape=block_shape,
            high_precision_mask=mask,
            base_quantizer=weight_quantizer,
            residual_quantizer=residual_quantizer,
            groupsize=groupsize,
        )
        W_hat = _restore_columns(W_hat, act_order_perm)
    else:
        W_hat = Q0
    loss = layer_output_mse_proxy(
        x=x,
        y_fp=y_fp,
        W_hat=W_hat,
        bias=module.bias,
        smooth_scale=smooth_scale,
        act_group_size=act_group_size,
        batch_size=search_eval_batch_size,
    )
    return ModuleSearchResult(
        module_name=module_name,
        alpha=alpha,
        loss=loss,
        loss_before_residual=loss_before,
        smooth_scale=smooth_scale.detach().cpu(),
        block_scores=scores.detach().cpu(),
        block_mask=mask.detach().cpu(),
        act_order_perm=None if act_order_perm is None else act_order_perm.detach().cpu(),
    )


def _evaluate_alpha_candidate_with_scale(
    module_name: str,
    module: nn.Linear,
    x: torch.Tensor,
    y_fp: torch.Tensor,
    alpha: float,
    smooth_scale: torch.Tensor,
    block_shape: tuple[int, int],
    budget_ratio: float,
    groupsize: int,
    act_group_size: int,
    weight_quantizer,
    residual_quantizer,
    second_path: str,
    actorder: bool = False,
    search_eval_batch_size: int = 1,
) -> ModuleSearchResult:
    W_smooth = smooth_weight(module.weight.data.float(), smooth_scale)
    H_diag = hessian_diag_proxy(
        x,
        smooth_scale=smooth_scale,
        batch_size=search_eval_batch_size,
        device=module.weight.device,
    )
    W_for_mask, H_diag_for_mask, act_order_perm = _apply_act_order_for_search(
        W_smooth,
        H_diag,
        actorder,
    )
    scores, mask = compute_block_sensitivity(
        W=W_for_mask,
        block_shape=block_shape,
        budget_ratio=budget_ratio,
        metric="delta_hessian",
        quantizer=weight_quantizer,
        H_diag=H_diag_for_mask,
        groupsize=groupsize,
    )
    Q0_for_mask = fake_quantize_weight(W_for_mask, weight_quantizer, groupsize=groupsize)
    Q0 = _restore_columns(Q0_for_mask, act_order_perm)
    loss_before = layer_output_mse_proxy(
        x=x,
        y_fp=y_fp,
        W_hat=Q0,
        bias=module.bias,
        smooth_scale=smooth_scale,
        act_group_size=act_group_size,
        act_sym=True,
        batch_size=search_eval_batch_size,
    )
    if second_path == "residual_int4":
        _, _, W_hat = build_w4_plus_residual_proxy(
            W_for_mask,
            block_shape=block_shape,
            high_precision_mask=mask,
            base_quantizer=weight_quantizer,
            residual_quantizer=residual_quantizer,
            groupsize=groupsize,
        )
        W_hat = _restore_columns(W_hat, act_order_perm)
    else:
        W_hat = Q0
    loss = layer_output_mse_proxy(
        x=x,
        y_fp=y_fp,
        W_hat=W_hat,
        bias=module.bias,
        smooth_scale=smooth_scale,
        act_group_size=act_group_size,
        act_sym=True,
        batch_size=search_eval_batch_size,
    )
    return ModuleSearchResult(
        module_name=module_name,
        alpha=alpha,
        loss=loss,
        loss_before_residual=loss_before,
        smooth_scale=smooth_scale.detach().cpu(),
        block_scores=scores.detach().cpu(),
        block_mask=mask.detach().cpu(),
        act_order_perm=None if act_order_perm is None else act_order_perm.detach().cpu(),
    )


def search_smooth_alpha_and_block_mask(
    module_map: dict[str, nn.Linear],
    io_cache: dict[str, dict[str, torch.Tensor]],
    alpha_candidates: list[float],
    block_shape: tuple[int, int] = (128, 128),
    budget_ratio: float = 0.05,
    groupsize: int = 128,
    act_group_size: int = 128,
    weight_bits: int = 4,
    second_path: str = "residual_int4",
    actorder: bool = False,
    search_eval_batch_size: int = 1,
) -> GroupSearchResult:
    weight_quantizer = make_quantizer(bits=weight_bits, sym=False)
    residual_quantizer = make_quantizer(bits=weight_bits, sym=False)
    best_result = None

    for alpha in alpha_candidates:
        module_results = {}
        total_loss = 0.0
        selected_blocks = 0
        total_blocks = 0
        for module_name, module in module_map.items():
            cache = io_cache[module_name]
            module_result = _evaluate_alpha_candidate(
                module_name=module_name,
                module=module,
                x=cache["inputs"],
                y_fp=cache["outputs"],
                alpha=alpha,
                block_shape=block_shape,
                budget_ratio=budget_ratio,
                groupsize=groupsize,
                act_group_size=act_group_size,
                weight_quantizer=weight_quantizer,
                residual_quantizer=residual_quantizer,
                second_path=second_path,
                actorder=actorder,
                search_eval_batch_size=search_eval_batch_size,
            )
            module_results[module_name] = module_result
            total_loss += module_result.loss
            selected_blocks += int(module_result.block_mask.sum().item())
            total_blocks += int(module_result.block_mask.numel())
        logger.info(
            "[SmoothSearch] group=%s alpha=%.4f loss=%.6f selected_blocks=%d/%d",
            ",".join(module_map.keys()),
            alpha,
            total_loss,
            selected_blocks,
            total_blocks,
        )
        if best_result is None or total_loss < best_result.loss:
            best_result = GroupSearchResult(
                group_key=",".join(module_map.keys()),
                alpha=alpha,
                loss=total_loss,
                module_results=module_results,
            )

    assert best_result is not None
    return best_result


def search_shared_smooth_alpha_and_block_mask(
    module_map: dict[str, nn.Linear],
    io_cache: dict[str, dict[str, torch.Tensor]],
    alpha_candidates: list[float],
    block_shape: tuple[int, int] = (128, 128),
    budget_ratio: float = 0.05,
    groupsize: int = 128,
    act_group_size: int = 128,
    weight_bits: int = 4,
    second_path: str = "residual_int4",
    actorder: bool = False,
    search_eval_batch_size: int = 1,
) -> SharedSmoothGroupSearchResult:
    weight_quantizer = make_quantizer(bits=weight_bits, sym=False)
    residual_quantizer = make_quantizer(bits=weight_bits, sym=False)
    best_result = None
    first_module_name = next(iter(module_map))
    first_module = module_map[first_module_name]
    shared_inputs = io_cache[first_module_name]["inputs"]
    x_absmax = input_absmax_proxy(
        shared_inputs,
        batch_size=search_eval_batch_size,
        device=first_module.weight.device,
    )
    shared_w_absmax = None
    for module in module_map.values():
        cur = get_weight_absmax(module.weight.data)
        shared_w_absmax = cur if shared_w_absmax is None else torch.maximum(shared_w_absmax, cur)
    assert shared_w_absmax is not None

    for alpha in alpha_candidates:
        smooth_scale = compute_smooth_scale(x_absmax, shared_w_absmax, alpha)
        module_results = {}
        total_loss = 0.0
        selected_blocks = 0
        total_blocks = 0
        for module_name, module in module_map.items():
            cache = io_cache[module_name]
            module_result = _evaluate_alpha_candidate_with_scale(
                module_name=module_name,
                module=module,
                x=cache["inputs"],
                y_fp=cache["outputs"],
                alpha=alpha,
                smooth_scale=smooth_scale.to(device=module.weight.device),
                block_shape=block_shape,
                budget_ratio=budget_ratio,
                groupsize=groupsize,
                act_group_size=act_group_size,
                weight_quantizer=weight_quantizer,
                residual_quantizer=residual_quantizer,
                second_path=second_path,
                actorder=actorder,
                search_eval_batch_size=search_eval_batch_size,
            )
            module_results[module_name] = module_result
            total_loss += module_result.loss
            selected_blocks += int(module_result.block_mask.sum().item())
            total_blocks += int(module_result.block_mask.numel())
        logger.info(
            "[SharedSmoothSearch] group=%s alpha=%.4f loss=%.6f selected_blocks=%d/%d",
            ",".join(module_map.keys()),
            alpha,
            total_loss,
            selected_blocks,
            total_blocks,
        )
        if best_result is None or total_loss < best_result.loss:
            best_result = SharedSmoothGroupSearchResult(
                group_key=",".join(module_map.keys()),
                alpha=alpha,
                loss=total_loss,
                smooth_scale=smooth_scale.detach().cpu(),
                module_results=module_results,
            )

    assert best_result is not None
    return best_result


@torch.no_grad()
def recalibrate_activation_quantizers(
    model: nn.Module,
    calibration_loader: Iterable,
    target_module_names: list[str],
    group_size: int = 128,
    max_batches: int | None = None,
) -> dict[str, dict]:
    io_cache = collect_linear_module_io(
        model,
        calibration_loader,
        target_module_names=target_module_names,
        max_batches=max_batches,
    )
    activation_stats = {}
    for module_name, cache in io_cache.items():
        calib = calibrate_activation_quantizer(cache["inputs"], bits=4, group_size=group_size, sym=False)
        activation_stats[module_name] = {
            "scale": calib.scale,
            "zero": calib.zero,
            "bits": calib.bits,
            "group_size": calib.group_size,
            "scale_min": float(calib.scale.min().item()),
            "scale_max": float(calib.scale.max().item()),
            "scale_mean": float(calib.scale.mean().item()),
        }
        logger.info(
            "[A4Calib] module=%s scale_min=%.6f scale_max=%.6f scale_mean=%.6f",
            module_name,
            activation_stats[module_name]["scale_min"],
            activation_stats[module_name]["scale_max"],
            activation_stats[module_name]["scale_mean"],
        )
    return activation_stats


@torch.no_grad()
def recalibrate_activation_quantizers_streaming(
    model: nn.Module,
    calibration_loader: Iterable,
    target_module_names: Iterable[str],
    group_size: int = 128,
    bits: int = 4,
    sym: bool = False,
    max_batches: int | None = None,
    eps: float = 1e-8,
) -> dict[str, dict]:
    model_device = next(model.parameters()).device
    selected = set(target_module_names)
    stats: dict[str, dict] = {}
    handles = []

    def update_stats(module_name: str, x: torch.Tensor) -> None:
        x2d = x.detach().float().reshape(-1, x.shape[-1])
        last_dim = x2d.shape[-1]
        actual_group_size = group_size if group_size > 0 else last_dim
        num_groups = (last_dim + actual_group_size - 1) // actual_group_size
        pad = num_groups * actual_group_size - last_dim
        if pad > 0:
            x2d = F.pad(x2d, (0, pad), value=0.0)
        groups = x2d.view(x2d.shape[0], num_groups, actual_group_size)
        xmin = groups.amin(dim=(0, 2))
        xmax = groups.amax(dim=(0, 2))

        prev = stats.get(module_name)
        if prev is None:
            stats[module_name] = {
                "xmin": xmin.cpu(),
                "xmax": xmax.cpu(),
                "group_size": actual_group_size,
            }
            return
        prev["xmin"] = torch.minimum(prev["xmin"], xmin.cpu())
        prev["xmax"] = torch.maximum(prev["xmax"], xmax.cpu())

    for name, module in model.named_modules():
        if name not in selected:
            continue

        def hook(_module, inp, _out, module_name=name):
            update_stats(module_name, inp[0])

        handles.append(module.register_forward_hook(hook))

    old_use_cache = getattr(getattr(model, "config", None), "use_cache", None)
    if old_use_cache is not None:
        model.config.use_cache = False
    try:
        for batch_idx, batch in enumerate(calibration_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = _move_batch_to_device(batch, model_device)
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

    activation_stats = {}
    maxq = float(2**bits - 1)
    for module_name, module_stats in stats.items():
        xmin = torch.minimum(module_stats["xmin"], torch.zeros_like(module_stats["xmin"]))
        xmax = torch.maximum(module_stats["xmax"], torch.zeros_like(module_stats["xmax"]))
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
        activation_stats[module_name] = {
            "scale": scale.detach().cpu(),
            "zero": zero.detach().cpu(),
            "bits": bits,
            "group_size": module_stats["group_size"],
            "sym": sym,
            "scale_min": float(scale.min().item()),
            "scale_max": float(scale.max().item()),
            "scale_mean": float(scale.mean().item()),
        }
        logger.info(
            "[A4Calib] module=%s scale_min=%.6f scale_max=%.6f scale_mean=%.6f",
            module_name,
            activation_stats[module_name]["scale_min"],
            activation_stats[module_name]["scale_max"],
            activation_stats[module_name]["scale_mean"],
        )
    return activation_stats


def _prefix_work_items(
    resolved_groups: dict[str, dict[str, list[str]]],
) -> list[tuple[str, list[tuple[str, str, list[str]]], list[str]]]:
    def work_key(prefix: str) -> str:
        for marker in (".self_attn", ".mlp"):
            marker_index = prefix.rfind(marker)
            if marker_index >= 0:
                return prefix[:marker_index]
        return prefix

    grouped: dict[str, list[tuple[str, str, list[str]]]] = {}
    for group_name, prefix_map in resolved_groups.items():
        for prefix, module_names in prefix_map.items():
            grouped.setdefault(work_key(prefix), []).append((group_name, prefix, module_names))

    work_items = []
    for prefix, group_items in grouped.items():
        target_names = []
        seen = set()
        for _, _, module_names in group_items:
            for module_name in module_names:
                if module_name in seen:
                    continue
                seen.add(module_name)
                target_names.append(module_name)
        work_items.append((prefix, group_items, target_names))
    return work_items


def _final_quantize_module(
    module: nn.Linear,
    cache: dict[str, torch.Tensor],
    block_shape: tuple[int, int],
    budget_ratio: float,
    groupsize: int,
    weight_bits: int,
    percdamp: float,
    actorder: bool,
    second_path: str,
    module_result: ModuleSearchResult | None = None,
    smooth_scale: torch.Tensor | None = None,
    gptq_batch_size: int = 1,
) -> dict:
    gptq = GPTQSubmatrixMixedV2(module)
    gptq.quantizer = make_quantizer(bits=weight_bits, sym=False)

    if smooth_scale is not None:
        smooth_scale_dev = smooth_scale.to(module.weight.device)
        smoothed_weight = (
            module.weight.data.float() * smooth_scale_dev.float().view(1, -1)
        ).to(dtype=module.weight.dtype)
        module.weight.data.copy_(smoothed_weight)
        if module_result is None:
            raise ValueError("module_result is required when smooth_scale is provided")
        precomputed_mask = module_result.block_mask.to(module.weight.device)
        precomputed_scores = module_result.block_scores.to(module.weight.device)
        precomputed_perm = (
            module_result.act_order_perm.to(module.weight.device)
            if module_result.act_order_perm is not None
            else None
        )
    else:
        smooth_scale_dev = None
        precomputed_mask = None
        precomputed_scores = None
        precomputed_perm = None

    # Make GPTQ Hessian collection see the same A4 fake-quantized activations
    # that are later used during benchmark evaluation.
    _add_gptq_batches_streaming(
        gptq=gptq,
        inputs=cache["inputs"],
        smooth_scale=smooth_scale_dev,
        groupsize=groupsize,
        batch_size=gptq_batch_size,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    final_stats = gptq.fasterquant(
        percdamp=percdamp,
        groupsize=groupsize,
        actorder=actorder,
        static_groups=False,
        block_shape=block_shape,
        budget_ratio=budget_ratio,
        sensitivity_metric="delta_hessian",
        precomputed_mask=precomputed_mask,
        precomputed_scores=precomputed_scores,
        precomputed_perm=precomputed_perm,
        smooth_scale=smooth_scale_dev,
        second_path=second_path,
        residual_quantizer=make_quantizer(bits=weight_bits, sym=False),
    )
    if not torch.isfinite(module.weight.data).all():
        raise FloatingPointError("non-finite quantized weight detected")
    gptq.free()
    del gptq
    return final_stats


@torch.no_grad()
def calibrate_smooth_block_mixed_gptq(
    model: nn.Module,
    calibration_loader: Iterable,
    module_groups: dict[str, list[str]],
    alpha_grids: dict[str, list[float]],
    block_shape: tuple[int, int] = (128, 128),
    budget_ratio: float = 0.05,
    groupsize: int = 128,
    act_bits: int = 4,
    weight_bits: int = 4,
    use_residual_second_path: bool = True,
    percdamp: float = 0.01,
    actorder: bool = False,
    max_batches: int | None = None,
    collect_output_errors: bool = False,
    output_error_batch_size: int = 1,
    search_eval_batch_size: int = 1,
    gptq_batch_size: int = 1,
) -> dict:
    del act_bits
    _validate_chunk_size("search_eval_batch_size", search_eval_batch_size)
    _validate_chunk_size("gptq_batch_size", gptq_batch_size)
    resolved_groups = resolve_module_groups(model, module_groups)
    modules_by_name = dict(model.named_modules())
    work_items = _prefix_work_items(resolved_groups)
    target_names = [
        name
        for _, _, names in work_items
        for name in names
    ]
    smooth_group_to_norm = {
        "attn_qkv": "input_layernorm",
        "ffn_up_gate": "post_attention_layernorm",
    }
    group_order = {
        "attn_qkv": 0,
        "attn_o": 1,
        "ffn_up_gate": 2,
        "ffn_down": 3,
    }

    all_results = {
        "search": {},
        "final": {},
        "activation_quant": {},
        "output_error": {},
        "smooth_groups": {},
    }
    for prefix, group_items, prefix_target_names in work_items:
        print(
            f"[SmoothBlockMixed] prefix={prefix} modules={len(prefix_target_names)}",
            flush=True,
        )
        for group_name, group_prefix, module_names in sorted(
            group_items,
            key=lambda item: group_order.get(item[0], 100),
        ):
            module_map = {name: modules_by_name[name] for name in module_names}
            smooth_enabled = group_name in smooth_group_to_norm
            print(
                f"[SmoothBlockMixed] group={group_name} prefix={group_prefix} "
                f"modules={len(module_names)} smooth={smooth_enabled}",
                flush=True,
            )
            logger.info(
                "[SmoothBlockMixed] collecting IO prefix=%s group=%s modules=%d",
                prefix,
                group_name,
                len(module_names),
            )
            io_cache = collect_linear_module_io(
                model,
                calibration_loader,
                target_module_names=module_names,
                max_batches=max_batches,
            )

            if smooth_enabled:
                alpha_candidates = alpha_grids.get(group_name, [1.0])
                search_result = search_shared_smooth_alpha_and_block_mask(
                    module_map=module_map,
                    io_cache=io_cache,
                    alpha_candidates=alpha_candidates,
                    block_shape=block_shape,
                    budget_ratio=budget_ratio,
                    groupsize=groupsize,
                    act_group_size=groupsize,
                    weight_bits=weight_bits,
                    second_path="residual_int4" if use_residual_second_path else "int8",
                    actorder=actorder,
                    search_eval_batch_size=search_eval_batch_size,
                )
                smooth_scale_cpu = search_result.smooth_scale
                all_results["search"][f"{group_name}:{group_prefix}"] = {
                    "best_alpha": search_result.alpha,
                    "loss": search_result.loss,
                    "shared_smooth_scale": True,
                    "act_order": bool(actorder),
                    "modules": {
                        name: {
                            "loss": result.loss,
                            "loss_before_residual": result.loss_before_residual,
                            "selected_blocks": int(result.block_mask.sum().item()),
                            "total_blocks": int(result.block_mask.numel()),
                            "act_order_perm_saved": result.act_order_perm is not None,
                        }
                        for name, result in search_result.module_results.items()
                    },
                }
                norm_name = f"{prefix}.{smooth_group_to_norm[group_name]}"
                norm = modules_by_name[norm_name]
                norm_before = norm.weight.detach().float().cpu()
            else:
                search_result = None
                smooth_scale_cpu = None
                norm_name = ""
                norm = None
                norm_before = None
                all_results["search"][f"{group_name}:{group_prefix}"] = {
                    "best_alpha": None,
                    "loss": None,
                    "shared_smooth_scale": False,
                    "smooth_applied": False,
                    "act_order": bool(actorder),
                    "modules": {},
                }

            module_results = search_result.module_results if search_result is not None else {}
            for module_name, module in module_map.items():
                module_result = module_results.get(module_name)
                module = module_map[module_name]
                cache = io_cache[module_name]
                final_stats = _final_quantize_module(
                    module=module,
                    cache=cache,
                    block_shape=block_shape,
                    budget_ratio=budget_ratio,
                    groupsize=groupsize,
                    weight_bits=weight_bits,
                    percdamp=percdamp,
                    actorder=actorder,
                    second_path="residual_int4" if use_residual_second_path else "int8",
                    module_result=module_result,
                    smooth_scale=smooth_scale_cpu,
                    gptq_batch_size=gptq_batch_size,
                )
                final_stats.update(
                    {
                        "smooth_group_shared": bool(smooth_enabled),
                        "smooth_folded_into_rmsnorm": bool(smooth_enabled),
                        "norm_name": norm_name,
                    }
                )
                all_results["final"][module_name] = final_stats
                selected_blocks = int(final_stats.get("selected_block_count", final_stats.get("n_int8_blocks", 0)))
                total_blocks = int(final_stats.get("n_total_blocks", 0))
                if not smooth_enabled:
                    all_results["search"][f"{group_name}:{group_prefix}"]["modules"][module_name] = {
                        "loss": None,
                        "loss_before_residual": None,
                        "selected_blocks": selected_blocks,
                        "total_blocks": total_blocks,
                    }
                if collect_output_errors:
                    error_inputs = cache["inputs"]
                    if smooth_enabled:
                        error_inputs = smooth_input(error_inputs.float(), smooth_scale_cpu.float())
                    output_error = compute_linear_output_error(
                        linear=module,
                        inputs=error_inputs,
                        reference_outputs=cache["outputs"],
                        batch_size=output_error_batch_size,
                        device=module.weight.device,
                    )
                    output_error.update(
                        {
                            "group": group_name,
                            "prefix": group_prefix,
                            "alpha": float(module_result.alpha) if module_result is not None else None,
                            "search_loss": float(module_result.loss) if module_result is not None else None,
                            "search_loss_before_residual": (
                                float(module_result.loss_before_residual)
                                if module_result is not None
                                else None
                            ),
                            "selected_blocks": (
                                int(module_result.block_mask.sum().item())
                                if module_result is not None
                                else selected_blocks
                            ),
                            "total_blocks": (
                                int(module_result.block_mask.numel())
                                if module_result is not None
                                else total_blocks
                            ),
                            "second_path": final_stats.get("second_path", ""),
                            "gptq_loss": float(final_stats.get("gptq_loss", 0.0)),
                            "smooth_group_shared": bool(smooth_enabled),
                            "norm_name": norm_name,
                        }
                    )
                    all_results["output_error"][module_name] = output_error
                    logger.info(
                        "[OutputError] module=%s mse=%s relative_l2=%s max_abs=%s",
                        module_name,
                        output_error.get("mse"),
                        output_error.get("relative_l2"),
                        output_error.get("max_abs"),
                    )
                logger.info(
                    "[FinalGPTQ] module=%s alpha=%s final_loss=%.6f second_path=%s smooth=%s",
                    module_name,
                    f"{module_result.alpha:.4f}" if module_result is not None else "n/a",
                    float(final_stats["gptq_loss"]),
                    final_stats["second_path"],
                    smooth_enabled,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if smooth_enabled:
                norm_scale = smooth_scale_cpu.to(device=norm.weight.device, dtype=norm.weight.dtype)
                norm.weight.data.div_(norm_scale)
                norm_after = norm.weight.detach().float().cpu()
                all_results["smooth_groups"][f"{group_name}:{prefix}"] = {
                    "group": group_name,
                    "prefix": prefix,
                    "norm_name": norm_name,
                    "target_linears": list(module_names),
                    "alpha": float(search_result.alpha),
                    "smooth_scale": smooth_scale_cpu,
                    "rmsnorm_weight_before": norm_before,
                    "rmsnorm_weight_after": norm_after,
                    "fused_into_rmsnorm": True,
                }
            del io_cache, search_result
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(
        f"[SmoothBlockMixed:A4Calib] streaming modules={len(target_names)}",
        flush=True,
    )
    all_results["activation_quant"] = recalibrate_activation_quantizers_streaming(
        model,
        calibration_loader,
        target_module_names=target_names,
        group_size=groupsize,
        bits=4,
        sym=True,
        max_batches=max_batches,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_results
