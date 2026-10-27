"""Joint Gaussian likelihood for sparse observations of one shared latent vector.

The likelihood is exact for a first-order linear observation model; it is not
an exact marginalization of a nonlinear observation function. Multiple sources
and anchors must be stacked for each object before calling the loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class PackedObservations:
    """Padded observations and their linearization at a latent predictive mean.

    ``values`` and ``mapped_mean`` are ``[B,K]``, ``jacobian`` is ``[B,K,D]``,
    ``measurement_covariance`` is ``[B,K,K]``, and ``mask`` is boolean ``[B,K]``.
    Unobserved entries may contain NaNs; only the selected subproblem is read.
    The covariance may include cross-source or anchor correlations. Duplicate
    measurements with unknown dependencies must be resolved by the caller.
    """

    values: Tensor
    mapped_mean: Tensor
    jacobian: Tensor
    measurement_covariance: Tensor
    mask: Tensor
    block_ids: Tensor | None = None


def _cholesky(matrix: Tensor, jitter: float, max_jitter: float) -> tuple[Tensor, float]:
    """Factor an SPD matrix with explicitly bounded diagonal regularization."""
    eye = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    amount = jitter
    while True:
        factor, info = torch.linalg.cholesky_ex(matrix + amount * eye)
        if int(info.item()) == 0:
            return factor, amount
        if max_jitter <= amount:
            raise ValueError("Observed measurement covariance is not positive definite")
        amount = min(max_jitter, max(1e-12, amount * 10))


class MarginalizedMultiSourceGaussianNLLLoss(nn.Module):
    """Normalized NLL of all observed measurements sharing one Gaussian latent.

    With latent covariance ``L @ L.T`` and the supplied local Jacobian ``H``,
    each observed stacked vector has covariance ``R + H @ L @ L.T @ H.T``.
    The information backend whitens by the Cholesky factor of ``R`` and uses a
    residual-plus-ridge quadratic rather than subtracting large quadratic forms.
    No explicit matrix inverse is formed. Measurement covariance must be SPD;
    optional jitter is bounded and recorded in ``last_diagnostics``.

    Args:
        reduction: ``mean`` averages over nonempty stars, ``sum`` sums them,
            and ``none`` returns one loss per star (zero for an empty star).
        method: ``direct``, ``information``, or ``auto`` (information if K > D).
        jitter: Initial nonnegative measurement-covariance diagonal addition.
        max_jitter: Maximum allowed addition; defaults to no adaptive retry.
    """

    def __init__(
        self,
        reduction: Literal["none", "sum", "mean"] = "mean",
        method: Literal["auto", "direct", "information"] = "auto",
        jitter: float = 0.0,
        max_jitter: float = 0.0,
    ) -> None:
        super().__init__()
        if reduction not in {"none", "sum", "mean"}:
            raise ValueError("Unknown reduction")
        if method not in {"auto", "direct", "information"}:
            raise ValueError("Unknown factorization method")
        if not 0 <= jitter <= max_jitter or not math.isfinite(max_jitter):
            raise ValueError("Require 0 <= jitter <= finite max_jitter")
        self.reduction = reduction
        self.method = method
        self.jitter = jitter
        self.max_jitter = max_jitter
        self.last_diagnostics: dict[str, object] = {}

    def per_star(self, mean: Tensor, scale_tril: Tensor, obs: PackedObservations) -> Tensor:
        """Return normalized per-object NLLs, including the Gaussian constant."""
        # An outer AMP context must not demote the explicit fp32/fp64 solves.
        with torch.autocast(device_type=mean.device.type, enabled=False):
            return self._per_star_no_autocast(mean, scale_tril, obs)

    def _per_star_no_autocast(
        self, mean: Tensor, scale_tril: Tensor, obs: PackedObservations
    ) -> Tensor:
        if mean.ndim != 2:
            raise ValueError("mean must be [B,D]")
        batch, dims = mean.shape
        if dims < 1 or scale_tril.shape != (batch, dims, dims):
            raise ValueError("scale_tril must be [B,D,D]")
        if obs.values.ndim != 2 or obs.values.shape[0] != batch:
            raise ValueError("values must be [B,K]")
        k = obs.values.shape[1]
        if (
            obs.mapped_mean.shape != (batch, k)
            or obs.jacobian.shape != (batch, k, dims)
            or obs.measurement_covariance.shape != (batch, k, k)
            or obs.mask.shape != (batch, k)
            or obs.mask.dtype != torch.bool
        ):
            raise ValueError("Observation shapes or mask dtype differ")
        if obs.block_ids is not None and (
            obs.block_ids.shape != (batch, k)
            or obs.block_ids.device != mean.device
            or obs.block_ids.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("block_ids must be integer [B,K] on the same device")
        tensors = (
            scale_tril,
            obs.values,
            obs.mapped_mean,
            obs.jacobian,
            obs.measurement_covariance,
        )
        if not mean.is_floating_point() or any(not x.is_floating_point() for x in tensors):
            raise ValueError("All values and covariances must be floating point")
        if any(x.device != mean.device for x in (*tensors, obs.mask)):
            raise ValueError("All tensors must be on the same device")
        if not torch.isfinite(mean).all() or not torch.isfinite(scale_tril).all():
            raise ValueError("Latent parameters must be finite")
        if (torch.diagonal(scale_tril, dim1=-2, dim2=-1) <= 0).any():
            raise ValueError("Latent scale must have a positive diagonal")
        if torch.count_nonzero(torch.triu(scale_tril, diagonal=1)):
            raise ValueError("Latent scale must be lower triangular")
        work_dtype = torch.promote_types(mean.dtype, scale_tril.dtype)
        for x in tensors[1:]:
            work_dtype = torch.promote_types(work_dtype, x.dtype)
        if work_dtype in (torch.float16, torch.bfloat16):
            work_dtype = torch.float32
        out: list[Tensor] = []
        jitters: list[float] = []
        methods: list[str] = []
        for i in range(batch):
            active = obs.mask[i].nonzero(as_tuple=True)[0]
            n = active.numel()
            if n == 0:
                out.append(mean[i].to(work_dtype).sum() * 0)
                jitters.append(0.0)
                methods.append("empty")
                continue
            residual = obs.values[i, active].to(work_dtype) - obs.mapped_mean[i, active].to(
                work_dtype
            )
            h = obs.jacobian[i, active].to(work_dtype)
            r = obs.measurement_covariance[i][active][:, active].to(work_dtype)
            lower = scale_tril[i].to(work_dtype)
            if not all(torch.isfinite(x).all() for x in (residual, h, r)):
                raise ValueError("Observed values, mappings, and covariance must be finite")
            tolerance = 1e-6 if work_dtype == torch.float32 else 1e-12
            if not torch.allclose(r, r.T, rtol=tolerance, atol=tolerance):
                raise ValueError("Observed measurement covariance must be symmetric")
            r = (r + r.T) * 0.5
            lr, added = _cholesky(r, self.jitter, self.max_jitter)
            method = (
                self.method if self.method != "auto" else ("information" if n > dims else "direct")
            )
            if method == "direct":
                hl = h @ lower
                cov = r + added * torch.eye(n, dtype=r.dtype, device=r.device) + hl @ hl.T
                factor = torch.linalg.cholesky(cov)
                whitened = torch.linalg.solve_triangular(factor, residual[:, None], upper=False)
                quadratic = whitened.square().sum()
                logdet = 2 * factor.diagonal().log().sum()
            else:
                z = torch.linalg.solve_triangular(lr, residual[:, None], upper=False)
                a = torch.linalg.solve_triangular(lr, h @ lower, upper=False)
                c = torch.eye(dims, dtype=a.dtype, device=a.device) + a.T @ a
                lc = torch.linalg.cholesky(c)
                solution = torch.cholesky_solve(a.T @ z, lc)
                quadratic = (z - a @ solution).square().sum() + solution.square().sum()
                logdet = 2 * (lr.diagonal().log().sum() + lc.diagonal().log().sum())
            out.append(0.5 * (n * math.log(2 * math.pi) + quadratic + logdet))
            jitters.append(added)
            methods.append(method)
        self.last_diagnostics = {
            "jitter": jitters,
            "method": methods,
            "observed_counts": obs.mask.sum(-1).tolist(),
        }
        return torch.stack(out) if out else mean.new_empty((0,), dtype=work_dtype)

    def forward(
        self,
        mean: Tensor,
        scale_tril: Tensor,
        observations: PackedObservations,
        weights: Tensor | None = None,
    ) -> Tensor:
        per_object = self.per_star(mean, scale_tril, observations)
        active = observations.mask.any(-1)
        if weights is None:
            weights = torch.ones_like(per_object)
        else:
            if weights.shape != per_object.shape or weights.device != per_object.device:
                raise ValueError("weights must match the batch and device")
            if not torch.isfinite(weights).all() or (weights < 0).any():
                raise ValueError("weights must be finite and nonnegative")
            weights = weights.to(per_object.dtype)
        effective = weights * active.to(weights.dtype)
        weighted = per_object * effective
        if self.reduction == "none":
            return weighted
        if self.reduction == "sum":
            return weighted.sum()
        denominator = effective.sum()
        if denominator.item() == 0:
            return mean.sum() * 0
        return weighted.sum() / denominator
