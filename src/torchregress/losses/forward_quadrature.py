"""Scalar Gaussian-latent forward likelihoods by Gauss–Hermite quadrature."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn


class GaussianLatentForwardObservationNLLLoss(nn.Module):
    r"""Approximate ``-log E_z Normal(y; observation_map(z), error)``.

    The latent is scalar ``z ~ Normal(latent_mean, latent_std)``. Inputs have
    shape ``[B]`` and ``observation_map`` receives nodes of shape ``[N,Q]`` for
    the N observed, positive-weight rows. It must return the same shape, on
    the same device. Negative observed values are valid. Missing entries may
    contain NaNs and never enter the map or likelihood. Errors must be finite
    and strictly positive; latent standard deviations may also be zero for
    the deterministic limit. All errors are standard deviations, not variances.

    For Hermite nodes x and weights w, the expectation is approximated by
    ``sum_q w_q / sqrt(pi) * p(y | mean + sqrt(2)*std*x_q)``. This is a
    normalized Gaussian observation likelihood and a finite-order numerical
    approximation to the latent integral, not an exact nonlinear marginal.
    Narrow likelihoods can require more nodes; compare quadrature orders
    before using results. Multiple dependent observations of a shared latent
    must be integrated jointly, rather than multiplying calls to this loss.

    Computation uses at least fp32 with autocast disabled; fp64 inputs retain
    fp64 arithmetic. The map must itself preserve autograd. ``mean`` reduction
    averages over positive-weight observed rows, ``sum`` sums weighted NLLs,
    and ``none`` returns [B], zero at missing or zero-weight entries.
    """

    nodes: Tensor
    log_weights: Tensor

    def __init__(
        self,
        quadrature_points: int = 32,
        reduction: Literal["none", "sum", "mean"] = "mean",
    ) -> None:
        super().__init__()
        if (
            isinstance(quadrature_points, bool)
            or not isinstance(quadrature_points, int)
            or not 1 <= quadrature_points <= 256
        ):
            raise ValueError("quadrature_points must be an integer in [1,256]")
        if reduction not in {"none", "sum", "mean"}:
            raise ValueError("Unknown reduction")
        nodes, weights = np.polynomial.hermite.hermgauss(quadrature_points)
        self.register_buffer("nodes", torch.from_numpy(nodes))
        self.register_buffer(
            "log_weights", torch.from_numpy(np.log(weights) - 0.5 * math.log(math.pi))
        )
        self.quadrature_points = quadrature_points
        self.reduction = reduction

    def forward(
        self,
        latent_mean: Tensor,
        latent_std: Tensor,
        observed: Tensor,
        observation_std: Tensor,
        observation_map: Callable[[Tensor], Tensor],
        mask: Tensor | None = None,
        weights: Tensor | None = None,
    ) -> Tensor:
        """Evaluate the observed scalar rows without detaching latent or map parameters."""
        inputs = (latent_mean, latent_std, observed, observation_std)
        if latent_mean.ndim != 1 or any(t.shape != latent_mean.shape for t in inputs):
            raise ValueError("All scalar inputs must have matching [B] shapes")
        if any(not t.is_floating_point() or t.device != latent_mean.device for t in inputs):
            raise ValueError("Scalar inputs must be floating point on the same device")
        if mask is None:
            mask = torch.ones_like(latent_mean, dtype=torch.bool)
        if (
            mask.shape != latent_mean.shape
            or mask.dtype != torch.bool
            or mask.device != latent_mean.device
        ):
            raise ValueError("mask must be boolean [B] on the input device")
        if weights is None:
            weights = torch.ones_like(latent_mean)
        if (
            weights.shape != latent_mean.shape
            or weights.device != latent_mean.device
            or not weights.is_floating_point()
            or not torch.isfinite(weights).all()
            or (weights < 0).any()
        ):
            raise ValueError("weights must be finite nonnegative floating [B] on the input device")
        dtype = torch.float32
        for tensor in (*inputs, weights):
            dtype = torch.promote_types(dtype, tensor.dtype)
        with torch.autocast(device_type=latent_mean.device.type, enabled=False):
            active = mask & (weights > 0)
            # Keep empty results differentiable without reading missing NaNs.
            safe_mean = torch.where(active, latent_mean.to(dtype), 0.0)
            safe_std = torch.where(active, latent_std.to(dtype), 0.0)
            result = (safe_mean + safe_std) * 0.0
            if active.any():
                mean, std, y, error = (t[active].to(dtype) for t in inputs)
                if not all(torch.isfinite(t).all() for t in (mean, std, y, error)):
                    raise ValueError("Observed inputs must be finite")
                if (std < 0).any() or (error <= 0).any():
                    raise ValueError("Latent std must be nonnegative and observation std positive")
                nodes = mean[:, None] + math.sqrt(2) * std[:, None] * self.nodes.to(
                    device=mean.device, dtype=dtype
                )
                mapped = observation_map(nodes)
                if (
                    not isinstance(mapped, Tensor)
                    or mapped.shape != nodes.shape
                    or not mapped.is_floating_point()
                    or mapped.device != nodes.device
                    or not torch.isfinite(mapped).all()
                ):
                    raise ValueError(
                        "observation_map must return finite floating [N,Q] on the input device"
                    )
                # Preserve higher-precision output from a user-provided map.
                dtype = torch.promote_types(dtype, mapped.dtype)
                mapped, y, error = mapped.to(dtype), y.to(dtype), error.to(dtype)
                residual = (y[:, None] - mapped) / error[:, None]
                log_likelihood = (
                    -0.5 * residual.square() - error[:, None].log() - 0.5 * math.log(2 * math.pi)
                )
                log_integral = torch.logsumexp(
                    log_likelihood + self.log_weights.to(device=mean.device, dtype=dtype), dim=-1
                )
                if not torch.isfinite(log_integral).all():
                    raise ValueError(
                        "Nonfinite quadrature integral; use higher precision or rescale inputs"
                    )
                result = result.to(dtype).masked_scatter(active, -log_integral)
            effective = torch.where(active, weights.to(result.dtype), 0.0)
            result = result * effective
            if self.reduction == "none":
                return result
            if self.reduction == "sum":
                return result.sum()
            denominator = effective.sum()
            return result.sum() / denominator if active.any() else result.sum()
