"""Neyman-orthogonal (double/debiased machine learning) estimating equations.

The canonical partially linear model is

    y = theta * x + eta(z) + eps,    E[eps | x, z] = 0,

where ``theta`` is the finite-dimensional target and ``eta`` an unknown
nuisance function. The orthogonal (Robinson) moment

    psi(theta) = (y - E[y|z]) - theta * (x - E[x|z])

identifies ``theta`` as the ratio of the two residual projections,

    theta_hat = sum_i v_i u_i / sum_i v_i x_i,   v = x - E[x|z],  u = y - E[y|z],

with the nuisances ``E[y|z]`` and ``E[x|z]`` estimated by **cross-fitting**
(``folds >= 2``): first-order nuisance errors are ignorable, so the estimator
stays consistent even when the nuisance learner is a flexible nonparametric
model that converges slower than ``sqrt(n)``. The influence-function variance
gives asymptotically valid confidence intervals.

This is the estimator class behind the cosmodist implicit-nuisance spine
(``docs/design/2026-09-16-operator-parameterization.md``, workstream W-M):
selection, population, instrument, and lensing kernels are learned from the
same real data, while cosmology is estimated from an orthogonal moment.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    "OrthogonalEstimate",
    "naive_linear_estimate",
    "orthogonal_partially_linear",
]


@dataclass(frozen=True)
class OrthogonalEstimate:
    """Point estimate and influence-function inference for ``theta``."""

    theta: float
    sigma: float
    ci_low: float
    ci_high: float
    n: int
    folds: int
    nuisance_r2_x: float
    nuisance_r2_y: float
    cross_fitted: bool

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "theta": self.theta,
            "sigma": self.sigma,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n": self.n,
            "folds": self.folds,
            "nuisance_r2_x": self.nuisance_r2_x,
            "nuisance_r2_y": self.nuisance_r2_y,
            "cross_fitted": self.cross_fitted,
        }


def _to_1d(name: str, value: Tensor | list[float]) -> Tensor:
    tensor = value.detach() if isinstance(value, Tensor) else torch.as_tensor(value)
    tensor = tensor.to(dtype=torch.float64).reshape(-1)
    if tensor.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite")
    return tensor


def _poly_features(z: Tensor, degree: int) -> Tensor:
    columns = [torch.ones_like(z[:, :1])]
    for power in range(1, degree + 1):
        columns.append(z**power)
    return torch.cat(columns, dim=1)


def _ridge_fit_predict(
    features: Tensor,
    target: Tensor,
    *,
    ridge: float,
    train: Tensor,
    test: Tensor,
) -> tuple[Tensor, float]:
    x_train = features[train]
    gram = x_train.T @ x_train + ridge * torch.eye(
        x_train.shape[1], dtype=x_train.dtype, device=x_train.device
    )
    weights = torch.linalg.solve(gram, x_train.T @ target[train])
    predictions = features[test] @ weights
    residual = target[test] - predictions
    total = target[test] - target[test].mean()
    variance = float((total**2).sum())
    r2 = 1.0 - float((residual**2).sum()) / variance if variance > 0.0 else 0.0
    return predictions, r2


def _cross_fitted_residuals(
    features: Tensor,
    target: Tensor,
    *,
    folds: int,
    ridge: float,
    generator: torch.Generator,
) -> tuple[Tensor, float]:
    n_samples = features.shape[0]
    if folds < 2:
        train = torch.arange(n_samples)
        predictions, r2 = _ridge_fit_predict(features, target, ridge=ridge, train=train, test=train)
        return target - predictions, r2
    order = torch.randperm(n_samples, generator=generator)
    residual = torch.empty_like(target)
    r2_folds: list[float] = []
    for fold in range(folds):
        test = order[fold::folds]
        mask = torch.ones(n_samples, dtype=torch.bool)
        mask[test] = False
        train = mask.nonzero().flatten()
        predictions, r2 = _ridge_fit_predict(features, target, ridge=ridge, train=train, test=test)
        residual[test] = target[test] - predictions
        r2_folds.append(r2)
    return residual, sum(r2_folds) / len(r2_folds)


def orthogonal_partially_linear(
    y: Tensor | list[float],
    x: Tensor | list[float],
    z: Tensor | list[float],
    *,
    folds: int = 5,
    ridge: float = 1.0e-6,
    nuisance_degree: int = 3,
    nuisance_features: Callable[[Tensor], Tensor] | None = None,
    confidence: float = 0.95,
    seed: int = 0,
) -> OrthogonalEstimate:
    """Cross-fitted orthogonal estimate of ``theta`` in ``y = theta x + eta(z) + eps``.

    Args:
        y: Outcome vector ``[N]``.
        x: Treatment/feature vector ``[N]``.
        z: Nuisance covariate(s): ``[N]`` or ``[N, K]``.
        folds: Number of cross-fitting folds (``>= 2`` recommended; ``1`` gives
            the non-cross-fitted variant for diagnostics only).
        ridge: Ridge penalty for the nuisance regressions.
        nuisance_degree: Degree of the polynomial nuisance basis on ``z``
            (ignored when ``nuisance_features`` is given).
        nuisance_features: Optional callable mapping ``z`` ``[N, K]`` to a
            feature matrix ``[N, P]``; use for categorical/per-method
            nuisances (one-hot) or richer bases (splines, RBFs).
        confidence: Two-sided confidence level for the reported interval.
        seed: Seed for the fold assignment.

    Returns:
        :class:`OrthogonalEstimate` with the point estimate, influence-function
        standard error, and interval.
    """
    y_vec = _to_1d("y", y)
    x_vec = _to_1d("x", x)
    if y_vec.shape != x_vec.shape:
        raise ValueError("y and x must have the same length")
    z_tensor = z.detach() if isinstance(z, Tensor) else torch.as_tensor(z)
    z_tensor = z_tensor.to(dtype=torch.float64)
    if z_tensor.ndim == 1:
        z_tensor = z_tensor.reshape(-1, 1)
    if z_tensor.ndim != 2 or z_tensor.shape[0] != y_vec.numel():
        raise ValueError("z must have shape [N] or [N,K] matching y")
    if not bool(torch.isfinite(z_tensor).all()):
        raise ValueError("z must be finite")
    if folds < 1:
        raise ValueError("folds must be at least 1")
    if nuisance_degree < 0:
        raise ValueError("nuisance_degree must be non-negative")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")

    generator = torch.Generator().manual_seed(seed)
    if nuisance_features is not None:
        features = nuisance_features(z_tensor)
        if not torch.is_tensor(features):
            features = torch.as_tensor(features)
        features = features.to(dtype=torch.float64)
        if features.ndim != 2 or features.shape[0] != y_vec.numel():
            raise ValueError(
                f"nuisance_features must return shape [N,P] matching y, got {tuple(features.shape)}"
            )
        if features.shape[1] < 1:
            raise ValueError("nuisance_features must return at least one column")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("nuisance_features must be finite")
    else:
        features = _poly_features(z_tensor, nuisance_degree)
    residual_x, r2_x = _cross_fitted_residuals(
        features, x_vec, folds=folds, ridge=ridge, generator=generator
    )
    residual_y, r2_y = _cross_fitted_residuals(
        features, y_vec, folds=folds, ridge=ridge, generator=generator
    )

    denominator = float((residual_x * residual_x).sum())
    treatment_scale = float((x_vec * x_vec).sum())
    if denominator <= 0.0 or denominator <= 1.0e-8 * treatment_scale:
        raise ValueError("x is perfectly explained by z; theta is not identified")
    theta = float((residual_x * residual_y).sum()) / denominator

    scores = residual_x * (residual_y - theta * residual_x)
    variance = float((scores**2).sum()) / (denominator**2) * y_vec.numel()
    sigma = variance**0.5
    z_value = _normal_quantile(0.5 + 0.5 * confidence)
    return OrthogonalEstimate(
        theta=theta,
        sigma=sigma,
        ci_low=theta - z_value * sigma,
        ci_high=theta + z_value * sigma,
        n=int(y_vec.numel()),
        folds=int(folds),
        nuisance_r2_x=float(r2_x),
        nuisance_r2_y=float(r2_y),
        cross_fitted=folds >= 2,
    )


def naive_linear_estimate(
    y: Tensor | list[float],
    x: Tensor | list[float],
    *,
    confidence: float = 0.95,
) -> OrthogonalEstimate:
    """Ordinary least squares of ``y`` on ``x`` ignoring the nuisance ``z``.

    Provided as the biased baseline that the orthogonal estimator must beat on
    confounded problems.
    """
    y_vec = _to_1d("y", y)
    x_vec = _to_1d("x", x)
    if y_vec.shape != x_vec.shape:
        raise ValueError("y and x must have the same length")
    design = torch.stack([torch.ones_like(x_vec), x_vec], dim=1)
    gram = design.T @ design
    weights = torch.linalg.solve(gram, design.T @ y_vec)
    theta = float(weights[1])
    residual = y_vec - design @ weights
    sigma = float(residual.std(unbiased=True)) / float(
        torch.sqrt((x_vec**2).sum() - x_vec.numel() * x_vec.mean() ** 2)
    )
    z_value = _normal_quantile(0.5 + 0.5 * confidence)
    return OrthogonalEstimate(
        theta=theta,
        sigma=sigma,
        ci_low=theta - z_value * sigma,
        ci_high=theta + z_value * sigma,
        n=int(y_vec.numel()),
        folds=1,
        nuisance_r2_x=0.0,
        nuisance_r2_y=0.0,
        cross_fitted=False,
    )


def _normal_quantile(probability: float) -> float:
    return math.sqrt(2.0) * float(torch.erfinv(torch.tensor(2.0 * probability - 1.0)).item())
