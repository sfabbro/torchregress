"""The joint package reuses the library's primitives: one behaviour, one implementation."""

from __future__ import annotations

import math
import warnings

import pytest
import torch
from torch import nn

import torchregress.joint as joint
from torchregress.ensemble import DensityEnsemble
from torchregress.ensemble.density import _em_mixture_weights
from torchregress.joint import (
    ConditionalModelJoint,
    DensityScore,
    GaussianJoint,
    HPDScore,
    JointConformal,
    MahalanobisScore,
    SampleBallScore,
    nees_consistency,
)
from torchregress.joint._common import masked_gaussian_quadratic, train
from torchregress.joint.estimator import _stacking_weights
from torchregress.losses import create_flow_loss, known_covariance_gaussian_nll
from torchregress.metrics import sample_hpd_level

F64 = torch.float64


def _gaussian(n: int, d: int, seed: int) -> GaussianJoint:
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(n, d, d, generator=gen, dtype=F64)
    cov = a @ a.mT + 0.5 * torch.eye(d, dtype=F64)
    return GaussianJoint.from_covariance(torch.randn(n, d, generator=gen, dtype=F64), cov)


# ═══════════════════════════════════════════════════════════════════════════════
# shared primitives
# ═══════════════════════════════════════════════════════════════════════════════


def test_hpd_level_is_the_metrics_estimator() -> None:
    pred = _gaussian(40, 3, seed=0)
    y = pred.sample(1)[:, 0]
    torch.manual_seed(1)
    level = pred.hpd_level(y, n_samples=128)
    torch.manual_seed(1)
    expected = sample_hpd_level(pred.log_prob(y), pred.log_prob(pred.sample(128)))
    assert torch.equal(level, expected)


def test_masked_gaussian_density_is_known_covariance_gaussian_nll() -> None:
    pred = _gaussian(60, 4, seed=2)
    gen = torch.Generator().manual_seed(3)
    mask = torch.rand(60, 4, generator=gen) > 0.4
    mask[0] = False  # nothing observed: density 0
    y = torch.randn(60, 4, generator=gen, dtype=F64)
    ref = -known_covariance_gaussian_nll(y - pred.loc, pred.covariance(), mask=mask).nll
    torch.testing.assert_close(pred.log_prob(y, mask), ref, rtol=0, atol=1e-10)
    assert float(pred.log_prob(y, mask)[0]) == 0.0
    # [n, S, d] candidates: each one equals the [n, d] evaluation
    y3 = torch.randn(60, 5, 4, generator=gen, dtype=F64)
    lp3 = pred.log_prob(y3, mask)
    for s in range(5):
        torch.testing.assert_close(lp3[:, s], pred.log_prob(y3[:, s], mask), rtol=0, atol=1e-10)
    # gradients reach the mean and the factor (the estimator trains through this path)
    loc = pred.loc.clone().requires_grad_(True)
    tril = pred.scale_tril.clone().requires_grad_(True)
    GaussianJoint(loc, tril).log_prob(y, mask).sum().backward()
    assert loc.grad is not None and tril.grad is not None
    assert bool(torch.isfinite(loc.grad).all()) and float(loc.grad.abs().sum()) > 0
    assert float(tril.grad.abs().sum()) > 0


def test_masked_mahalanobis_and_nees_share_the_quadratic_form() -> None:
    pred = _gaussian(50, 3, seed=4)
    gen = torch.Generator().manual_seed(5)
    mask = torch.rand(50, 3, generator=gen) > 0.3
    y = torch.randn(50, 3, generator=gen, dtype=F64)
    quad, _, dof = masked_gaussian_quadratic(y - pred.loc, pred.covariance(), mask)
    nees = nees_consistency(pred.loc, pred.covariance(), y, mask=mask)
    torch.testing.assert_close(nees.nees, quad, rtol=0, atol=1e-10)
    score = MahalanobisScore().score(pred, y, mask)
    expected = torch.special.gammainc((0.5 * dof).clamp(min=0.5), 0.5 * quad)
    expected = torch.where(dof > 0, expected, torch.zeros_like(expected))
    torch.testing.assert_close(score, expected, rtol=0, atol=1e-12)


def test_stacking_and_density_ensemble_share_one_em() -> None:
    gen = torch.Generator().manual_seed(6)
    log_density = torch.randn(300, 3, generator=gen, dtype=F64) - torch.tensor(
        [0.0, 0.4, 1.0], dtype=F64
    )
    stacked = _stacking_weights(log_density)
    fitted = _em_mixture_weights(log_density.T, max_iter=5000, loglik_tol=1e-15)
    torch.testing.assert_close(stacked, fitted, rtol=0, atol=1e-6)
    assert math.isclose(float(stacked.sum()), 1.0, rel_tol=1e-12)


def test_trainer_counts_non_finite_batches() -> None:
    net = nn.Linear(2, 1)
    x = torch.randn(32, 2)

    def nan_loss(idx: torch.Tensor) -> torch.Tensor:
        return net(x[idx]).sum() * math.nan

    with pytest.raises(RuntimeError, match="not finite on any"):
        train(
            net,
            nan_loss,
            32,
            lambda: net(x).square().mean(),
            max_epochs=2,
            patience=2,
            batch_size=8,
            lr=1e-2,
            generator=torch.Generator().manual_seed(0),
            skip_nonfinite=True,
        )
    calls = {"n": 0}

    def sometimes_nan(idx: torch.Tensor) -> torch.Tensor:
        calls["n"] += 1
        loss = net(x[idx]).square().mean()
        return loss * math.nan if calls["n"] % 2 else loss

    with pytest.warns(RuntimeWarning, match="skipped"):
        result = train(
            net,
            sometimes_nan,
            32,
            lambda: net(x).square().mean(),
            max_epochs=2,
            patience=2,
            batch_size=8,
            lr=1e-2,
            generator=torch.Generator().manual_seed(0),
            skip_nonfinite=True,
        )
    assert result.n_skipped == 4 and result.n_steps == 4


def test_result_containers_are_exported() -> None:
    for name in (
        "PreRankHistograms",
        "NEESConsistency",
        "HPDCalibration",
        "CoverageByMask",
        "WorstSlabCoverage",
        "CoverageGap",
        "DependenceError",
        "ConditionalModelJoint",
    ):
        assert name in joint.__all__ and hasattr(joint, name)
    for name in ("PRERANKS", "holm_adjust", "standard_normal_cdf"):
        assert name not in joint.__all__
    res = joint.nees_consistency(
        torch.zeros(5, 2), torch.eye(2, dtype=torch.float32), torch.ones(5, 2)
    )
    assert isinstance(res, joint.NEESConsistency)


# ═══════════════════════════════════════════════════════════════════════════════
# adapter for density heads with the flow-head API
# ═══════════════════════════════════════════════════════════════════════════════


def _flow(seed: int, context_dim: int = 3) -> nn.Module:
    torch.manual_seed(seed)
    return create_flow_loss(n_features=2, context_dim=context_dim, flow_type="maf", n_transforms=2)


def _heads() -> dict[str, nn.Module]:
    return {"flow": _flow(0), "ensemble": DensityEnsemble([_flow(1), _flow(2)])}


@pytest.mark.parametrize("kind", ["flow", "ensemble"])
def test_conditional_model_joint_layouts_and_densities(kind: str) -> None:
    model = _heads()[kind]
    context = torch.randn(9, 3, generator=torch.Generator().manual_seed(7))
    pred = ConditionalModelJoint(model, context)
    assert (pred.batch_size, pred.event_dim) == (9, 2)
    assert pred.has_density and not pred.has_latent and not pred.exact_marginals
    draws = pred.sample(6)
    assert draws.shape == (9, 6, 2)
    assert pred.sample(1).shape == (9, 1, 2)
    y = torch.randn(9, 2)
    with torch.no_grad():
        torch.testing.assert_close(pred.log_prob(y), model.log_prob(context, y))
        lp3 = pred.log_prob(draws)
        assert lp3.shape == (9, 6)
        for s in range(6):
            torch.testing.assert_close(lp3[:, s], model.log_prob(context, draws[:, s]))
    assert pred.marginal(1).sample(4).shape == (9, 4, 1)
    with pytest.raises(NotImplementedError):
        pred.log_prob(y, torch.ones(9, 2, dtype=torch.bool))


@pytest.mark.parametrize("kind", ["flow", "ensemble"])
def test_conformal_region_on_a_wrapped_density_head(kind: str) -> None:
    torch.manual_seed(8)
    model = _heads()[kind]
    gen = torch.Generator().manual_seed(9)
    c_cal, c_test = torch.randn(300, 3, generator=gen), torch.randn(40, 3, generator=gen)
    with torch.no_grad():
        y_cal = ConditionalModelJoint(model, c_cal).sample(1)[:, 0]
    for score in (DensityScore(), HPDScore(n_samples=32), SampleBallScore(n_samples=16)):
        conformal = JointConformal(score, alpha=0.2).calibrate(
            ConditionalModelJoint(model, c_cal), y_cal
        )
        assert bool(torch.isfinite(conformal.thresholds_).all())
        region = conformal.region(ConditionalModelJoint(model, c_test))
        assert region.contains(torch.zeros(40, 2)).shape == (40,)
    # the density score of the wrapper is the model's negative log density
    with torch.no_grad():
        scores = DensityScore().score(ConditionalModelJoint(model, c_cal), y_cal)
        torch.testing.assert_close(scores, -model.log_prob(c_cal, y_cal))


def test_conditional_model_joint_validates_the_model() -> None:
    with pytest.raises(TypeError, match="log_prob"):
        ConditionalModelJoint(nn.Linear(2, 2), torch.randn(3, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = ConditionalModelJoint(_flow(3), torch.randn(4, 3), event_dim=2)
    assert pred.event_dim == 2
