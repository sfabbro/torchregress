"""Normalized small-block outlier mixtures for one shared Gaussian latent."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor

from .multisource import MarginalizedMultiSourceGaussianNLLLoss, PackedObservations


class _WeightedLogSumExp(torch.autograd.Function):
    """Stable log mixture with one-sided weight derivatives at zero weights."""

    @staticmethod
    def forward(ctx, log_density: Tensor, weights: Tensor) -> Tensor:
        result = torch.logsumexp(log_density + weights.log(), dim=0)
        ctx.save_for_backward(log_density, weights, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        log_density, weights, result = ctx.saved_tensors
        density_gradient = (log_density + weights.log() - result).exp()
        weight_gradient = (log_density - result).exp()
        return grad_output * density_gradient, grad_output * weight_gradient


class RobustMultiSourceGaussianNLLLoss(MarginalizedMultiSourceGaussianNLLLoss):
    r"""Marginalize coherent inlier/outlier assignments of measurement blocks.

    For block indicators ``z_b ~ Bernoulli(p_b)``, the observation noise block
    is ``R_b`` when inlying and ``s_b**2 R_b`` when outlying. Every assignment
    has the *same single* latent covariance contribution ``H V H.T``. Its
    normalized Gaussian density is mixed with the normalized product Bernoulli
    probabilities. Full within-block covariance is preserved. Cross-block
    measurement covariance is rejected, since independent block inflation
    would otherwise require a different, explicitly specified noise model.

    ``observations.block_ids`` must contain nonnegative integer IDs for observed
    scalars. IDs are local to each star and need not be contiguous. Missing
    blocks integrate to one and are not enumerated. Optional call arguments
    ``outlier_probabilities`` and ``outlier_scales`` are scalars or ``[B,K]``;
    entries must be constant within each observed block. Their gradients are
    retained, including probability endpoints. Default parameters are fixed
    floats, not learned parameters. Scales are standard-deviation multipliers
    of at least one. This model inflates *measurement* noise, not latent spread.

    Enumeration is exponential: both ``max_blocks`` and ``max_assignments``
    bound work explicitly. Use a small, scientifically justified block model.
    The Gaussian linearization and covariance-jitter caveats of the base class
    apply. No pseudo-Huber transformation or unnormalized density is used.
    """

    def __init__(
        self,
        reduction: Literal["none", "sum", "mean"] = "mean",
        method: Literal["auto", "direct", "information"] = "auto",
        jitter: float = 0.0,
        max_jitter: float = 0.0,
        *,
        outlier_probability: float = 0.05,
        outlier_scale: float = 10.0,
        max_blocks: int = 8,
        max_assignments: int = 256,
    ) -> None:
        super().__init__(reduction, method, jitter, max_jitter)
        if not math.isfinite(outlier_probability) or not 0 <= outlier_probability <= 1:
            raise ValueError("Outlier probability must be finite in [0,1]")
        if not math.isfinite(outlier_scale) or outlier_scale < 1:
            raise ValueError("Outlier standard-deviation scale must be finite and >= 1")
        if (
            isinstance(max_blocks, bool)
            or not isinstance(max_blocks, int)
            or max_blocks < 1
            or isinstance(max_assignments, bool)
            or not isinstance(max_assignments, int)
            or max_assignments < 1
        ):
            raise ValueError("Positive integer enumeration bounds required")
        self.outlier_probability = float(outlier_probability)
        self.outlier_scale = float(outlier_scale)
        self.max_blocks = max_blocks
        self.max_assignments = max_assignments

    @staticmethod
    def _expand(value: Tensor | float, obs: PackedObservations, name: str) -> Tensor:
        result = torch.as_tensor(
            value,
            device=obs.values.device,
            dtype=(
                None
                if isinstance(value, Tensor)
                else torch.promote_types(obs.values.dtype, torch.float32)
            ),
        )
        if not result.is_floating_point():
            result = result.to(torch.promote_types(obs.values.dtype, torch.float32))
        if result.ndim == 0:
            return result.expand_as(obs.values)
        if result.shape != obs.values.shape:
            raise ValueError(f"{name} must be scalar or [B,K]")
        return result

    def per_star(
        self,
        mean: Tensor,
        scale_tril: Tensor,
        obs: PackedObservations,
        *,
        outlier_probabilities: Tensor | float | None = None,
        outlier_scales: Tensor | float | None = None,
    ) -> Tensor:
        """Return per-star normalized mixture NLLs, zero for empty stars."""
        # Validate all base shapes and selected values before inspecting blocks.
        inlier = super().per_star(mean, scale_tril, obs)
        base_diagnostics = dict(self.last_diagnostics)
        if obs.block_ids is None:
            raise ValueError("Robust likelihood requires explicit block_ids")
        probabilities = self._expand(
            self.outlier_probability if outlier_probabilities is None else outlier_probabilities,
            obs,
            "outlier_probabilities",
        )
        scales = self._expand(
            self.outlier_scale if outlier_scales is None else outlier_scales,
            obs,
            "outlier_scales",
        )
        selected_p, selected_s = probabilities[obs.mask], scales[obs.mask]
        if not torch.isfinite(selected_p).all() or ((selected_p < 0) | (selected_p > 1)).any():
            raise ValueError("Observed outlier probabilities must be finite in [0,1]")
        if not torch.isfinite(selected_s).all() or (selected_s < 1).any():
            raise ValueError("Observed outlier scales must be finite and >= 1")
        work_dtype = torch.promote_types(
            inlier.dtype, torch.promote_types(probabilities.dtype, scales.dtype)
        )
        probabilities, scales = probabilities.to(work_dtype), scales.to(work_dtype)
        results = []
        assignment_counts = []
        for row in range(len(mean)):
            active = obs.mask[row].nonzero(as_tuple=True)[0]
            if active.numel() == 0:
                results.append(inlier[row])
                assignment_counts.append(1)
                continue
            ids = obs.block_ids[row, active]
            if (ids < 0).any():
                raise ValueError("Observed block IDs must be nonnegative")
            blocks, inverse = torch.unique(ids, sorted=True, return_inverse=True)
            count = blocks.numel()
            if count > self.max_blocks or 2**count > self.max_assignments:
                raise ValueError("Robust assignment enumeration exceeds configured bound")
            r = obs.measurement_covariance[row][active][:, active].to(work_dtype)
            if (r[inverse[:, None] != inverse[None, :]] != 0).any():
                raise ValueError("Cross-block measurement covariance is unsupported")
            block_p, block_s = [], []
            for index in range(count):
                local = inverse == index
                p, s = probabilities[row, active[local]], scales[row, active[local]]
                if (p != p[0]).any() or (s != s[0]).any():
                    raise ValueError("Outlier parameters must be constant within each block")
                # Mean preserves gradients to all repeated scalar entries.
                block_p.append(p.mean())
                block_s.append(s.mean())
            p, s = torch.stack(block_p), torch.stack(block_s)
            densities, mixture_weights = [], []
            for code in range(2**count):
                outlier = torch.tensor(
                    [bool(code & (1 << index)) for index in range(count)],
                    device=mean.device,
                )
                mixture_weights.append(torch.where(outlier, p, 1 - p).prod())
                if code == 0:
                    densities.append(-inlier[row])
                    continue
                multipliers = torch.where(outlier, s, torch.ones_like(s))[inverse]
                noise = r * multipliers[:, None] * multipliers[None, :]
                local_obs = PackedObservations(
                    obs.values[row : row + 1, active],
                    obs.mapped_mean[row : row + 1, active],
                    obs.jacobian[row : row + 1, active],
                    noise[None],
                    torch.ones((1, len(active)), dtype=torch.bool, device=mean.device),
                )
                densities.append(
                    -super().per_star(mean[row : row + 1], scale_tril[row : row + 1], local_obs)[0]
                )
            log_density = torch.stack(densities)
            weights = torch.stack(mixture_weights).to(log_density.dtype)
            results.append(-_WeightedLogSumExp.apply(log_density, weights))
            assignment_counts.append(2**count)
        self.last_diagnostics = {
            "inlier": base_diagnostics,
            "assignment_counts": assignment_counts,
            "mixture": "independent measurement-block indicators; one shared Gaussian latent",
        }
        return torch.stack(results) if results else inlier

    def forward(
        self,
        mean: Tensor,
        scale_tril: Tensor,
        observations: PackedObservations,
        weights: Tensor | None = None,
        *,
        outlier_probabilities: Tensor | float | None = None,
        outlier_scales: Tensor | float | None = None,
    ) -> Tensor:
        """Use the base loss's nonempty-star weighted reduction convention."""
        per_object = self.per_star(
            mean,
            scale_tril,
            observations,
            outlier_probabilities=outlier_probabilities,
            outlier_scales=outlier_scales,
        )
        active = observations.mask.any(-1)
        if weights is None:
            weights = torch.ones_like(per_object)
        elif (
            weights.shape != per_object.shape
            or weights.device != per_object.device
            or not torch.isfinite(weights).all()
            or (weights < 0).any()
        ):
            raise ValueError("weights must be finite nonnegative [B] on the same device")
        effective = weights.to(per_object.dtype) * active
        weighted = per_object * effective
        if self.reduction == "none":
            return weighted
        if self.reduction == "sum":
            return weighted.sum()
        if effective.sum().item() == 0:
            return mean.sum() * 0
        return weighted.sum() / effective.sum()
