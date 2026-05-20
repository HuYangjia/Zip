import copy
import logging
import math
import time

import torch
import torch.nn as nn
import transformers

from .block_mask import compute_block_sensitivity
from .compat import LegacyGPTQSubmatrixMixed, _int8_fakequant_group, quantize

logger = logging.getLogger(__name__)


def _safe_hessian_inverse_cholesky(
    H: torch.Tensor,
    columns: int,
    percdamp: float,
    max_tries: int = 8,
) -> tuple[torch.Tensor, float, int]:
    H = H.float()
    device = H.device
    H = (H + H.t()) * 0.5
    H = torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)

    diag_idx = torch.arange(columns, device=device)
    h_diag = torch.diag(H)
    positive_diag = h_diag[h_diag > 0]
    diag_mean = positive_diag.mean() if positive_diag.numel() else H.new_tensor(1.0)
    base_damp = torch.clamp(percdamp * diag_mean, min=1e-6)

    eye = torch.eye(columns, dtype=H.dtype, device=device)
    last_info = None
    for attempt in range(max_tries):
        damp = base_damp * (10.0 ** attempt)
        H_try = H.clone()
        bad_diag = torch.diag(H_try) <= 0
        if bool(bad_diag.any()):
            H_try[diag_idx[bad_diag], diag_idx[bad_diag]] = diag_mean
        H_try = H_try + eye * damp

        chol, info = torch.linalg.cholesky_ex(H_try)
        last_info = int(info.item())
        if last_info == 0:
            H_inv = torch.cholesky_inverse(chol)
            H_inv = (H_inv + H_inv.t()) * 0.5
            chol_inv, inv_info = torch.linalg.cholesky_ex(H_inv, upper=True)
            if int(inv_info.item()) == 0:
                return chol_inv, float(damp.item()), attempt
            last_info = int(inv_info.item())

    raise RuntimeError(
        f"Hessian Cholesky failed after {max_tries} damping attempts; last info={last_info}"
    )


def _check_positive_groupsize(groupsize: int) -> None:
    if groupsize <= 0:
        raise ValueError(f"groupsize must be positive, got {groupsize}")


def _build_residual_scale_cache(
    residual_block: torch.Tensor,
    high_precision_mask: torch.Tensor,
    block_col: int,
    brow: int,
    groupsize: int,
    residual_quantizer,
    first_group_key: int = 0,
) -> dict[int, dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    _check_positive_groupsize(groupsize)
    d_out = residual_block.shape[0]
    nrow = high_precision_mask.shape[0]
    block_cache = {}
    for br in range(nrow):
        if not bool(high_precision_mask[br, block_col]):
            continue
        r0 = br * brow
        r1 = min(r0 + brow, d_out)
        row_cache = {}
        for c0 in range(0, residual_block.shape[1], groupsize):
            c1 = min(c0 + groupsize, residual_block.shape[1])
            qtz = copy.deepcopy(residual_quantizer)
            qtz.find_params(residual_block[r0:r1, c0:c1], weight=True)
            row_cache[first_group_key + c0 // groupsize] = (qtz.scale, qtz.zero, qtz.maxq)
        block_cache[br] = row_cache
    return block_cache


class GPTQSubmatrixMixedV2(LegacyGPTQSubmatrixMixed):
    def _prepare_inputs(self, groupsize: int, block_shape: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, int, int]:
        _check_positive_groupsize(groupsize)
        brow, bcol = block_shape

        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        H = self.H.float()
        del self.H
        H = (H + H.t()) * 0.5
        H = torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
        dead = torch.diag(H) <= 0
        H[dead, dead] = 1
        W[:, dead] = 0
        return W, H, None, None, brow, bcol

    def _run_mixed_gptq(
        self,
        W: torch.Tensor,
        H: torch.Tensor,
        high_precision_mask: torch.Tensor,
        block_scores: torch.Tensor,
        blocksize: int,
        percdamp: float,
        groupsize: int,
        static_groups: bool,
        block_shape: tuple[int, int],
        invperm: torch.Tensor | None = None,
        second_path: str = "int8",
        residual_quantizer=None,
    ) -> tuple[torch.Tensor, dict]:
        if second_path not in {"int8", "residual_int4"}:
            raise ValueError(f"不支持的 second_path: {second_path}")
        brow, bcol = block_shape
        d_out, d_in = W.shape
        nrow = math.ceil(d_out / brow)
        ncol = math.ceil(d_in / bcol)
        if second_path == "residual_int4" and residual_quantizer is None:
            residual_quantizer = self.quantizer
        if static_groups:
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i : min(i + groupsize, self.columns)], weight=True)
                groups.append(quantizer)
        else:
            groups = None

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)
        Q_base = torch.zeros_like(W) if second_path == "residual_int4" else None
        W_initial = W.clone() if second_path == "residual_int4" else None

        Hinv, used_damp, damp_attempts = _safe_hessian_inverse_cholesky(
            H=H,
            columns=self.columns,
            percdamp=percdamp,
        )
        if not torch.isfinite(Hinv).all():
            raise FloatingPointError("non-finite H inverse Cholesky factor in GPTQ")
        if damp_attempts:
            logger.warning(
                "[GPTQ] Hessian needed extra damping: percdamp=%.6f used_damp=%.6e attempts=%d",
                percdamp,
                used_damp,
                damp_attempts + 1,
            )

        n_int4_segments = 0
        n_int8_segments = 0
        col_has_any_int8 = high_precision_mask.any(dim=0)
        col_is_all_int8 = high_precision_mask.all(dim=0)
        col_is_all_int4 = ~col_has_any_int8
        int8_scale_cache: dict[int, torch.Tensor] = {}
        residual_scale_cache: dict[int, dict[int, dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]] = {}
        n_residual_int4_segments = 0

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
                if not torch.isfinite(d) or float(d.abs().item()) <= 1e-12:
                    raise FloatingPointError(
                        f"invalid GPTQ diagonal d at column {i1 + i}: {float(d.item())}"
                    )
                col_idx = i1 + i
                block_col = col_idx // bcol

                if groupsize != -1:
                    if not static_groups:
                        if col_idx % groupsize == 0:
                            group_end = min(col_idx + groupsize, self.columns)
                            self.quantizer.find_params(W[:, col_idx:group_end], weight=True)
                    else:
                        self.quantizer = groups[col_idx // groupsize]

                if block_col < ncol and bool(col_has_any_int8[block_col]):
                    if second_path == "int8" and col_idx % bcol == 0:
                        group_end_int8 = min(col_idx + bcol, self.columns)
                        group_slice = W[:, col_idx:group_end_int8]
                        _, scale_row_cached = _int8_fakequant_group(group_slice)
                        int8_scale_cache[block_col] = scale_row_cached
                    elif second_path == "residual_int4" and col_idx % groupsize == 0:
                        block_end = min((block_col + 1) * bcol, self.columns)
                        group_end = min(col_idx + groupsize, block_end)
                        group_slice = W[:, col_idx:group_end]
                        q0_group = quantize(
                            group_slice,
                            self.quantizer.scale,
                            self.quantizer.zero,
                            self.quantizer.maxq,
                        )
                        residual_group = group_slice - q0_group
                        group_key = (col_idx - block_col * bcol) // groupsize
                        group_cache = _build_residual_scale_cache(
                            residual_group,
                            high_precision_mask,
                            block_col,
                            brow,
                            groupsize,
                            residual_quantizer,
                            first_group_key=group_key,
                        )
                        block_cache = residual_scale_cache.setdefault(block_col, {})
                        for br, row_cache in group_cache.items():
                            block_cache.setdefault(br, {}).update(row_cache)

                q0 = quantize(
                    w.unsqueeze(1),
                    self.quantizer.scale,
                    self.quantizer.zero,
                    self.quantizer.maxq,
                ).flatten()

                if second_path == "residual_int4":
                    q = q0.clone()
                    n_int4_segments += nrow
                    if block_col < ncol and bool(col_has_any_int8[block_col]):
                        block_cache = residual_scale_cache.get(block_col)
                        if block_cache is None:
                            raise RuntimeError(f"missing residual INT4 cache for block_col={block_col}")
                        for br in range(nrow):
                            if not bool(high_precision_mask[br, block_col]):
                                continue
                            r0 = br * brow
                            r1 = min(r0 + brow, d_out)
                            group_key = (col_idx - block_col * bcol) // groupsize
                            scale, zero, maxq = block_cache[br][group_key]
                            residual = (w[r0:r1] - q0[r0:r1]).unsqueeze(1)
                            q1 = quantize(residual, scale, zero, maxq).flatten()
                            q[r0:r1] = q0[r0:r1] + q1
                            n_residual_int4_segments += 1
                    if Q_base is not None:
                        Q_base[:, col_idx] = q0
                else:
                    if block_col < ncol and bool(col_is_all_int4[block_col]):
                        q = q0
                        n_int4_segments += nrow
                    elif block_col < ncol and bool(col_is_all_int8[block_col]):
                        scale_row = int8_scale_cache[block_col]
                        q = torch.clamp(torch.round(w / scale_row), -128, 127) * scale_row
                        n_int8_segments += nrow
                    else:
                        q = q0.clone()
                        scale_row = int8_scale_cache.get(block_col)
                        for br in range(nrow):
                            r0 = br * brow
                            r1 = min(r0 + brow, d_out)
                            if block_col < ncol and bool(high_precision_mask[br, block_col]):
                                s_seg = scale_row[r0:r1]
                                q[r0:r1] = torch.clamp(torch.round(w[r0:r1] / s_seg), -128, 127) * s_seg
                                n_int8_segments += 1
                            else:
                                n_int4_segments += 1

                if not torch.isfinite(q).all():
                    raise FloatingPointError(f"non-finite quantized column at column {col_idx}")
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2
                err1 = (w - q) / d
                if not torch.isfinite(err1).all():
                    raise FloatingPointError(f"non-finite GPTQ error at column {col_idx}")
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        residual_norm_before = None
        residual_norm_after = None
        if second_path == "residual_int4":
            assert Q_base is not None and W_initial is not None
            residual_norm_before = float((W_initial - Q_base).norm().item())
            residual_norm_after = float((W_initial - Q).norm().item())

        if invperm is not None:
            Q = Q[:, invperm]
        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()

        top5_vals = []
        if block_scores.numel() > 0:
            top5_vals, _ = torch.topk(block_scores.flatten(), k=min(5, block_scores.numel()))
            top5_vals = top5_vals.tolist()
        stats = {
            "gptq_loss": float(torch.sum(Losses).item()),
            "n_int4_segments": n_int4_segments,
            "n_int8_segments": n_int8_segments,
            "grid_shape": [nrow, ncol],
            "n_int8_blocks": int(high_precision_mask.sum().item()),
            "n_total_blocks": int(high_precision_mask.numel()),
            "top5_sensitivity_scores": top5_vals,
            "block_shape": list(block_shape),
            "groupsize": groupsize,
            "dense_groupsize": groupsize,
            "hessian_damp": used_damp,
            "hessian_damp_attempts": damp_attempts,
            "act_order": invperm is not None,
        }
        if second_path == "residual_int4":
            stats.update(
                {
                    "residual_in_gptq": True,
                    "residual_groupsize": groupsize,
                    "n_residual_int4_segments": n_residual_int4_segments,
                    "residual_norm_before": residual_norm_before,
                    "residual_norm_after": residual_norm_after,
                }
            )
        quant_weight = Q.reshape(self.layer.weight.shape).to(dtype=self.layer.weight.dtype)
        if not torch.isfinite(quant_weight).all():
            raise FloatingPointError("non-finite quantized weight produced by GPTQ")
        return quant_weight, stats

    def fasterquant(
        self,
        blocksize: int = 128,
        percdamp: float = 0.01,
        groupsize: int = -1,
        actorder: bool = False,
        static_groups: bool = False,
        block_shape: tuple[int, int] = (128, 128),
        budget_ratio: float = 0.05,
        sensitivity_metric: str = "quant_error",
        precomputed_mask: torch.Tensor | None = None,
        precomputed_scores: torch.Tensor | None = None,
        precomputed_perm: torch.Tensor | None = None,
        smooth_scale: torch.Tensor | None = None,
        second_path: str = "int8",
        residual_quantizer=None,
    ) -> dict:
        if second_path not in {"int8", "residual_int4"}:
            raise ValueError(f"不支持的 second_path: {second_path}")
        tick = time.time()
        if not self.quantizer.ready():
            W_init = self.layer.weight.data.float()
            if isinstance(self.layer, nn.Conv2d):
                W_init = W_init.flatten(1)
            if isinstance(self.layer, transformers.Conv1D):
                W_init = W_init.t()
            self.quantizer.find_params(W_init, weight=True)

        W_reference, H, _, _, _, _ = self._prepare_inputs(groupsize, block_shape)
        d_out, d_in = W_reference.shape
        brow, bcol = block_shape
        nrow = math.ceil(d_out / brow)
        ncol = math.ceil(d_in / bcol)

        perm = None
        invperm = None
        W_quant = W_reference
        H_quant = H
        if actorder:
            if precomputed_perm is None:
                perm = torch.argsort(torch.diag(H_quant), descending=True)
            else:
                perm = precomputed_perm.to(device=W_reference.device, dtype=torch.long)
                if perm.numel() != self.columns:
                    raise ValueError(
                        f"precomputed_perm length mismatch: expected {self.columns}, got {perm.numel()}"
                    )
                sorted_perm = torch.sort(perm).values
                expected = torch.arange(self.columns, device=W_reference.device)
                if not bool(torch.equal(sorted_perm, expected)):
                    raise ValueError("precomputed_perm must be a permutation of all input columns")
            invperm = torch.argsort(perm)
            W_quant = W_reference[:, perm]
            H_quant = H[perm][:, perm]
        elif precomputed_perm is not None:
            raise ValueError("precomputed_perm requires actorder=True")

        if precomputed_mask is not None:
            high_precision_mask = precomputed_mask.to(device=W_quant.device, dtype=torch.bool)
            if tuple(high_precision_mask.shape) != (nrow, ncol):
                raise ValueError(
                    f"precomputed_mask shape mismatch: expected {(nrow, ncol)}, got {tuple(high_precision_mask.shape)}"
                )
            block_scores = (
                precomputed_scores.to(device=W_quant.device, dtype=W_quant.dtype)
                if precomputed_scores is not None
                else torch.zeros_like(high_precision_mask, dtype=W_quant.dtype)
            )
            if tuple(block_scores.shape) != (nrow, ncol):
                raise ValueError(
                    f"precomputed_scores shape mismatch: expected {(nrow, ncol)}, got {tuple(block_scores.shape)}"
                )
        elif budget_ratio <= 0:
            high_precision_mask = torch.zeros(nrow, ncol, dtype=torch.bool, device=W_quant.device)
            block_scores = torch.zeros(nrow, ncol, dtype=W_quant.dtype, device=W_quant.device)
        elif budget_ratio >= 1 and second_path == "int8":
            high_precision_mask = torch.ones(nrow, ncol, dtype=torch.bool, device=W_quant.device)
            block_scores = torch.zeros(nrow, ncol, dtype=W_quant.dtype, device=W_quant.device)
        else:
            H_diag = torch.diag(H_quant) if sensitivity_metric in {"hessian_weighted", "delta_hessian"} else None
            block_scores, high_precision_mask = compute_block_sensitivity(
                W=W_quant,
                block_shape=block_shape,
                budget_ratio=budget_ratio,
                metric=sensitivity_metric,
                quantizer=self.quantizer,
                H_diag=H_diag,
                groupsize=groupsize,
            )

        quant_weight, stats = self._run_mixed_gptq(
            W_quant,
            H_quant,
            high_precision_mask,
            block_scores,
            blocksize,
            percdamp,
            groupsize,
            static_groups=static_groups,
            block_shape=block_shape,
            invperm=invperm,
            second_path=second_path,
            residual_quantizer=residual_quantizer,
        )
        if not torch.isfinite(quant_weight).all():
            raise FloatingPointError("non-finite quantized weight detected before layer copy")
        self.layer.weight.data.copy_(quant_weight)
        elapsed = time.time() - tick
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        stats.update(
            {
                "elapsed_seconds": elapsed,
                "budget_ratio": budget_ratio,
                "sensitivity_metric": sensitivity_metric,
                "second_path": second_path,
                "selected_block_count": int(high_precision_mask.sum().item()),
                "smooth_applied": smooth_scale is not None,
            }
        )
        return stats
