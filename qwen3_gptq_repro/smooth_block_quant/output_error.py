import json
import math
from pathlib import Path

import torch
import torch.nn as nn


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


@torch.no_grad()
def compute_linear_output_error(
    linear: nn.Linear,
    inputs: torch.Tensor,
    reference_outputs: torch.Tensor,
    batch_size: int = 1,
    device: torch.device | str | None = None,
) -> dict:
    """Compute summary error stats between a quantized Linear and cached FP outputs."""
    if inputs.shape[0] != reference_outputs.shape[0]:
        raise ValueError(
            f"inputs/output batch mismatch: {tuple(inputs.shape)} vs {tuple(reference_outputs.shape)}"
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    device = torch.device(device) if device is not None else linear.weight.device
    original_training = linear.training
    linear.eval()

    total_numel = 0
    sum_sq_error = 0.0
    sum_abs_error = 0.0
    max_abs_error = 0.0
    ref_sum_sq = 0.0
    quant_sum_sq = 0.0
    dot_sum = 0.0
    finite = True

    try:
        for start in range(0, inputs.shape[0], batch_size):
            end = min(start + batch_size, inputs.shape[0])
            x = inputs[start:end].to(device=device, dtype=linear.weight.dtype, non_blocking=True)
            y_ref = reference_outputs[start:end].to(device=device, non_blocking=True)
            y_quant = linear(x)

            if y_quant.shape != y_ref.shape:
                raise ValueError(
                    f"output shape mismatch: quantized {tuple(y_quant.shape)} vs reference {tuple(y_ref.shape)}"
                )

            y_quant_f = y_quant.float()
            y_ref_f = y_ref.float()
            diff = y_quant_f - y_ref_f
            chunk_finite = bool(torch.isfinite(y_quant_f).all().item()) and bool(torch.isfinite(diff).all().item())
            finite = finite and chunk_finite
            if not chunk_finite:
                continue

            total_numel += int(diff.numel())
            abs_diff = diff.abs()
            sum_sq_error += float(diff.pow(2).sum().item())
            sum_abs_error += float(abs_diff.sum().item())
            max_abs_error = max(max_abs_error, float(abs_diff.max().item()) if abs_diff.numel() else 0.0)
            ref_sum_sq += float(y_ref_f.pow(2).sum().item())
            quant_sum_sq += float(y_quant_f.pow(2).sum().item())
            dot_sum += float((y_ref_f * y_quant_f).sum().item())
    finally:
        linear.train(original_training)

    if total_numel == 0:
        return {
            "shape": list(reference_outputs.shape),
            "numel": int(reference_outputs.numel()),
            "finite": False,
            "mse": None,
            "rmse": None,
            "mae": None,
            "max_abs": None,
            "relative_l2": None,
            "cosine": None,
            "reference_l2": None,
            "quantized_l2": None,
        }

    mse = sum_sq_error / total_numel
    rmse = math.sqrt(mse)
    mae = sum_abs_error / total_numel
    reference_l2 = math.sqrt(ref_sum_sq)
    quantized_l2 = math.sqrt(quant_sum_sq)
    relative_l2 = math.sqrt(sum_sq_error / ref_sum_sq) if ref_sum_sq > 0 else None
    cosine = dot_sum / math.sqrt(ref_sum_sq * quant_sum_sq) if ref_sum_sq > 0 and quant_sum_sq > 0 else None

    return {
        "shape": list(reference_outputs.shape),
        "numel": total_numel,
        "finite": finite,
        "mse": _finite_or_none(mse),
        "rmse": _finite_or_none(rmse),
        "mae": _finite_or_none(mae),
        "max_abs": _finite_or_none(max_abs_error),
        "relative_l2": _finite_or_none(relative_l2) if relative_l2 is not None else None,
        "cosine": _finite_or_none(cosine) if cosine is not None else None,
        "reference_l2": _finite_or_none(reference_l2),
        "quantized_l2": _finite_or_none(quantized_l2),
    }


def summarize_output_errors(output_errors: dict[str, dict], topk: int = 10) -> dict:
    rows = [
        {
            "module": module_name,
            "mse": stats.get("mse"),
            "relative_l2": stats.get("relative_l2"),
            "max_abs": stats.get("max_abs"),
        }
        for module_name, stats in output_errors.items()
    ]
    finite_rows = [row for row in rows if row["mse"] is not None]

    def top_by(key: str) -> list[dict]:
        return sorted(
            (row for row in finite_rows if row.get(key) is not None),
            key=lambda row: row[key],
            reverse=True,
        )[:topk]

    return {
        "module_count": len(output_errors),
        "finite_module_count": len(finite_rows),
        "top_mse": top_by("mse"),
        "top_relative_l2": top_by("relative_l2"),
        "top_max_abs": top_by("max_abs"),
    }


def save_output_error_report(path: str | Path, output_errors: dict[str, dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output_errors, ensure_ascii=False, indent=2), encoding="utf-8")
