"""Tests for Neyman-orthogonal (double/debiased ML) partially linear estimation."""

from __future__ import annotations

import math

import pytest
import torch

from torchregress.inference.orthogonal import (
    OrthogonalEstimate,
    naive_linear_estimate,
    orthogonal_partially_linear,
)


def _confounded_sample(
    n: int = 4000, theta: float = 1.0, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    z = torch.rand(n, generator=generator, dtype=torch.float64)
    eta = torch.sin(2.0 * math.pi * z) + z**2
    x = 2.0 * eta + 0.5 * torch.randn(n, generator=generator, dtype=torch.float64)
    y = theta * x + eta + 0.5 * torch.randn(n, generator=generator, dtype=torch.float64)
    return y, x, z


def test_orthogonal_estimator_beats_the_confounded_naive_estimator():
    y, x, z = _confounded_sample(theta=1.0, seed=0)
    naive = naive_linear_estimate(y, x)
    orthogonal = orthogonal_partially_linear(y, x, z, folds=5, seed=0)
    assert abs(naive.theta - 1.0) > 0.2
    assert abs(orthogonal.theta - 1.0) < 3.0 * orthogonal.sigma
    assert orthogonal.nuisance_r2_x > 0.5
    assert orthogonal.nuisance_r2_y > 0.5
    assert orthogonal.cross_fitted


def test_orthogonal_interval_covers_truth_across_seeds():
    covered = 0
    reps = 40
    for seed in range(reps):
        y, x, z = _confounded_sample(n=1500, theta=0.7, seed=seed)
        estimate = orthogonal_partially_linear(y, x, z, folds=5, seed=seed)
        if estimate.ci_low <= 0.7 <= estimate.ci_high:
            covered += 1
    assert covered >= 0.85 * reps


def test_orthogonal_estimator_handles_multidimensional_nuisance():
    generator = torch.Generator().manual_seed(1)
    n = 3000
    z = torch.rand(n, 2, generator=generator, dtype=torch.float64)
    eta = torch.sin(2.0 * math.pi * z[:, 0]) + 0.5 * z[:, 1] ** 2
    x = 1.5 * eta + 0.4 * torch.randn(n, generator=generator, dtype=torch.float64)
    y = 0.5 * x + eta + 0.4 * torch.randn(n, generator=generator, dtype=torch.float64)
    estimate = orthogonal_partially_linear(y, x, z, folds=5, seed=3)
    assert abs(estimate.theta - 0.5) < 3.0 * estimate.sigma


def test_non_cross_fitted_variant_is_flagged():
    y, x, z = _confounded_sample(n=500, theta=1.0, seed=2)
    estimate = orthogonal_partially_linear(y, x, z, folds=1, seed=2)
    assert not estimate.cross_fitted
    assert estimate.folds == 1


def test_estimate_serialization_and_validation():
    y, x, z = _confounded_sample(n=200, seed=4)
    estimate = orthogonal_partially_linear(y, x, z, folds=3, seed=4)
    payload = estimate.to_dict()
    assert isinstance(estimate, OrthogonalEstimate)
    assert payload["n"] == 200
    assert payload["folds"] == 3
    assert payload["ci_low"] < payload["theta"] < payload["ci_high"]
    with pytest.raises(ValueError, match="same length"):
        orthogonal_partially_linear(y[:-1], x, z)
    with pytest.raises(ValueError, match="finite"):
        orthogonal_partially_linear(torch.full((4,), float("nan")), x[:4], z[:4])
    with pytest.raises(ValueError, match="shape"):
        orthogonal_partially_linear(y, x, z.reshape(-1, 1, 1))
    with pytest.raises(ValueError, match="folds"):
        orthogonal_partially_linear(y, x, z, folds=0)
    with pytest.raises(ValueError, match="confidence"):
        orthogonal_partially_linear(y, x, z, confidence=1.0)


def test_degenerate_treatment_is_rejected():
    z = torch.linspace(0.0, 1.0, 200, dtype=torch.float64)
    x = z.clone()
    y = 2.0 * x + torch.randn(200, dtype=torch.float64) * 0.01
    with pytest.raises(ValueError, match="not identified"):
        orthogonal_partially_linear(y, x, z, folds=4)


def test_naive_estimator_is_unbiased_without_confounding():
    generator = torch.Generator().manual_seed(5)
    n = 4000
    x = torch.randn(n, generator=generator, dtype=torch.float64)
    y = 0.3 * x + 0.5 * torch.randn(n, generator=generator, dtype=torch.float64)
    naive = naive_linear_estimate(y, x)
    assert abs(naive.theta - 0.3) < 3.0 * naive.sigma
    assert not naive.cross_fitted
