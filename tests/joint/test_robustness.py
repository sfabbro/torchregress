"""Regression tests for failure modes of the joint estimator and its parts."""

from __future__ import annotations

import math
import warnings

import pytest
import torch

from torchregress.joint import (
    GaussianCopula,
    GaussianJoint,
    IndependenceDependence,
    JointRegressor,
    LatentScore,
    MahalanobisScore,
    SampleJoint,
    ScaleRecalibrator,
    prerank_histograms,
    select_dependence,
)
from torchregress.joint.dependence import FlowDependence
from torchregress.joint.estimator import _stack, _stacking_weights

F64 = torch.float64
SMALL = dict(max_epochs=5, patience=3, hidden=(16, 16), batch_size=64, device="cpu")


def _mean_abs_offdiag_corr(cov: torch.Tensor) -> float:
    sd = cov.diagonal().sqrt()
    corr = cov / sd[:, None] / sd[None]
    d = corr.shape[0]
    return float((corr - torch.eye(d, dtype=corr.dtype)).abs().sum()) / (d * (d - 1))


# 1 ──────────────────────────────────────────────────────────────────────────────
def test_low_rank_gaussian_head_learns_correlation() -> None:
    """d = 40 (rank-8 head by default) with correlations near 0.99."""
    gen = torch.Generator().manual_seed(0)
    n = 2000
    x = torch.randn(n, 2, generator=gen)
    common = torch.randn(n, 1, generator=gen)
    y = common + 0.1 * torch.randn(n, 40, generator=gen)
    model = JointRegressor(
        "gaussian", max_epochs=30, patience=5, recalibrate=False, device="cpu"
    ).fit(x, y)
    cov = model.predict(x[:3]).covariance()[0]
    assert _mean_abs_offdiag_corr(cov) > 0.8


# 2 ──────────────────────────────────────────────────────────────────────────────
class _Misspecified(JointRegressor):
    """Predicts N(0, I) for every row; the data have variance 2.56 (d = 6)."""

    def predict(self, X):  # type: ignore[no-untyped-def]
        n = torch.as_tensor(X).shape[0]
        return GaussianJoint(torch.zeros(n, 6, dtype=F64), torch.eye(6, dtype=F64))


def _misspecified_model(n_cal: int, gen: torch.Generator) -> _Misspecified:
    model = _Misspecified(device="cpu")
    y = 1.6 * torch.randn(n_cal, 6, generator=gen, dtype=F64)
    mask = torch.rand(n_cal, 6, generator=gen) < 0.5
    model._conformal_rows = (torch.zeros(n_cal, 1), torch.where(mask, y, 0.0), mask)
    return model


def test_region_covers_complete_rows_when_conformal_rows_have_missing_targets() -> None:
    gen = torch.Generator().manual_seed(1)
    n_cal, n_test, alpha = 20000, 20000, 0.1
    model = _misspecified_model(n_cal, gen)
    n_complete = int(model._conformal_rows[2].all(1).sum())  # about n_cal / 64
    region = model.region(torch.zeros(n_test, 1), alpha)
    y_test = 1.6 * torch.randn(n_test, 6, generator=gen, dtype=F64)
    v = alpha * (1 - alpha)
    # complete test rows, no mask
    coverage = region.contains(y_test).double().mean().item()
    assert abs(coverage - (1 - alpha)) <= 4 * math.sqrt(v / n_test + v / (n_complete + 2))
    # same missingness mechanism, with the mask
    mask = torch.rand(n_test, 6, generator=gen) < 0.5
    observed = mask.any(1)
    inside = region.contains(y_test, mask=mask)
    masked = inside[observed].double().mean().item()
    assert abs(masked - (1 - alpha)) <= 4 * math.sqrt(v / int(observed.sum()) + v / (n_cal + 2))


def test_region_refuses_too_few_complete_conformal_rows() -> None:
    model = _misspecified_model(200, torch.Generator().manual_seed(2))  # ~3 complete rows
    with pytest.raises(ValueError, match="complete"):
        model.region(torch.zeros(5, 1), 0.1)


# 3 ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("method", ["auto", "gaussian", "mixture", "flow", "sklar"])
def test_single_target(method: str) -> None:
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(200, 3, generator=gen)
    y = x[:, :1] + 0.3 * torch.randn(200, 1, generator=gen)
    model = JointRegressor(method, **SMALL).fit(x, y)
    dist = model.predict(x[:10])
    assert dist.event_dim == 1 and dist.sample(4).shape == (10, 4, 1)
    assert bool(torch.isfinite(dist.log_prob(y[:10])).all())
    region = model.region(x[:10], 0.1)
    assert region.contains(y[:10]).shape == (10,)
    if method in ("auto", "sklar"):
        assert model.dependence_ == "independence"


def test_flow_dependence_identity_initialisation_in_one_dimension() -> None:
    z = torch.randn(100, 1, dtype=F64)
    reference = -0.5 * z.square().sum(-1) - 0.5 * math.log(2 * math.pi)
    for x in (None, torch.randn(100, 2, dtype=F64)):
        dep = FlowDependence(max_epochs=0).fit(z, x)
        torch.testing.assert_close(dep.log_prob(z, x), reference, rtol=0, atol=1e-10)


# 4 ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("method", ["gaussian", "sklar"])
def test_constant_target_column(method: str) -> None:
    gen = torch.Generator().manual_seed(4)
    x = torch.randn(150, 3, generator=gen)
    y = torch.cat([torch.randn(150, 1, generator=gen), torch.full((150, 1), 3.0)], 1)
    model = JointRegressor(method, **SMALL).fit(x, y)
    assert model.scale_ is not None and bool(torch.isfinite(model.scale_).all())
    assert bool(torch.isfinite(model.predict(x[:5]).sample(3)).all())


@pytest.mark.parametrize("objective", ["variance", "nll"])
def test_scale_recalibrator_keeps_a_unit_factor_without_residual_spread(objective: str) -> None:
    mean = torch.zeros(50, 2, dtype=F64)
    pred = GaussianJoint(mean, torch.eye(2, dtype=F64))
    y = torch.cat([2.0 * torch.randn(50, 1, dtype=F64), torch.zeros(50, 1, dtype=F64)], 1)
    scale = ScaleRecalibrator(objective).fit(pred, y).scale_
    assert scale is not None and float(scale[1]) == 1.0 and float(scale[0]) > 1.3


# 5 ──────────────────────────────────────────────────────────────────────────────
def test_stacking_keeps_rows_where_a_member_has_zero_density() -> None:
    n = 1000
    a = torch.full((n,), -1.5, dtype=F64)
    b = torch.full((n,), -1.0, dtype=F64)
    b[:100] = -math.inf  # better on 90 % of the rows, impossible on 10 %
    stacked = torch.stack([a, b], 1)
    weights = _stack(stacked)
    assert 0.2 < float(weights[0]) < 0.3  # all rows: about 0.254
    torch.testing.assert_close(weights, _stacking_weights(stacked), rtol=0, atol=1e-8)
    score = torch.logsumexp(stacked + weights.log(), 1).mean()
    assert math.isfinite(float(score))
    # NaN members are dropped, NaN rows ignored, nothing usable -> uniform
    with_nan = torch.cat([stacked, torch.full((n, 1), math.nan, dtype=F64)], 1)
    w_nan = _stack(with_nan)
    assert float(w_nan[2]) == 0.0
    torch.testing.assert_close(w_nan[:2], weights, rtol=0, atol=1e-12)
    all_bad = torch.full((5, 3), -math.inf, dtype=F64)
    torch.testing.assert_close(_stack(all_bad), torch.full((3,), 1 / 3, dtype=F64))


# 6 ──────────────────────────────────────────────────────────────────────────────
class _BrokenCopula(GaussianCopula):
    name = "broken"

    def log_prob(self, z, x=None):  # type: ignore[no-untyped-def]
        out = super().log_prob(z, x)
        out[0] = math.nan
        return out


def test_select_dependence_never_picks_a_nan_score() -> None:
    z = torch.randn(200, 3, generator=torch.Generator().manual_seed(5), dtype=F64)
    chosen, table = select_dependence([IndependenceDependence().fit(z), _BrokenCopula().fit(z)], z)
    assert chosen.name == "independence"
    assert math.isnan(table["broken"])


# 7 ──────────────────────────────────────────────────────────────────────────────
def test_prerank_defaults_with_one_target() -> None:
    res = prerank_histograms(torch.randn(20, 9, 1), torch.randn(20, 1))
    assert "dependence" not in res.names and len(res.names) == 5


# 9 is in test_integration.py (trainer), 8 in test_regions.py (NaN scores).


# 10 ─────────────────────────────────────────────────────────────────────────────
def test_scale_recalibration_of_a_sampled_mean_is_reproducible() -> None:
    gen = torch.Generator().manual_seed(6)
    draws = torch.randn(30, 40, 2, generator=gen)
    pred = SampleJoint(draws)  # mean() resamples: Monte Carlo
    y = 2.0 * torch.randn(30, 2, generator=gen)
    recal = ScaleRecalibrator(n_samples=64).fit(pred, y)
    state = torch.get_rng_state()
    first = recal.transform(pred)
    second = recal.transform(pred)
    assert torch.equal(torch.get_rng_state(), state)
    assert torch.equal(first.loc, second.loc)


# 11 ─────────────────────────────────────────────────────────────────────────────
def test_fit_leaves_the_caller_rng_alone_and_is_deterministic() -> None:
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(120, 2, generator=gen)
    y = torch.randn(120, 2, generator=gen)
    torch.manual_seed(123)
    state = torch.get_rng_state()
    first = JointRegressor("sklar", seed=3, **SMALL).fit(x, y)
    assert torch.equal(torch.get_rng_state(), state)
    second = JointRegressor("sklar", seed=3, **SMALL).fit(x, y)
    assert first.dependence_scores_ == second.dependence_scores_
    torch.testing.assert_close(
        first.predict(x[:4]).log_prob(y[:4]), second.predict(x[:4]).log_prob(y[:4])
    )


# 12 ─────────────────────────────────────────────────────────────────────────────
def test_gaussian_copula_with_a_constant_score_column() -> None:
    z = torch.randn(100, 3, generator=torch.Generator().manual_seed(8), dtype=F64)
    z[:, 2] = 0.4
    for shrinkage in ("auto", 0.0):
        cop = GaussianCopula(shrinkage=shrinkage).fit(z)
        assert cop.correlation_ is not None
        torch.testing.assert_close(cop.correlation_.diagonal(), torch.ones(3, dtype=F64))
        assert bool(torch.isfinite(cop.log_prob(z)).all())


# 13 ─────────────────────────────────────────────────────────────────────────────
def test_invalid_settings_and_inputs_raise() -> None:
    with pytest.raises(ValueError, match="n_folds"):
        JointRegressor(n_folds=1)
    gen = torch.Generator().manual_seed(9)
    x, y = torch.randn(60, 2, generator=gen), torch.randn(60, 2, generator=gen)
    with pytest.raises(ValueError, match="n_folds"):
        JointRegressor("sklar", n_folds=40, **SMALL).fit(x, y)  # ~37 training rows
    y_inf = y.clone()
    y_inf[0, 0] = math.inf
    with pytest.raises(ValueError, match="inf"):
        JointRegressor("gaussian", **SMALL).fit(x, y_inf)


def test_scale_reports_the_applied_factors_and_the_region_cache_follows_the_score() -> None:
    gen = torch.Generator().manual_seed(10)
    x = torch.randn(300, 2, generator=gen)
    y = x + 0.5 * torch.randn(300, 2, generator=gen)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = JointRegressor("gaussian", **SMALL).fit(x, y)
    assert model.scale_ is not None
    if model._recalibrator is None:
        assert torch.equal(model.scale_, torch.ones(2))
    else:
        assert torch.equal(model.scale_, model._recalibrator.scale_.cpu())
    first = model.region(x[:5], 0.1)
    assert isinstance(first.score, LatentScore)
    model.score = MahalanobisScore()
    assert isinstance(model.region(x[:5], 0.1).score, MahalanobisScore)


def test_masked_gaussian_paths_accept_covariances_symmetric_only_to_float32_rounding() -> None:
    """A rescaled float32 covariance is asymmetric at 1e-7; NEES upcasts it to float64."""
    from torchregress.joint import AffineJoint, GaussianJoint, nees_consistency

    torch.manual_seed(0)
    tril = torch.randn(500, 3, 3).tril() + 3 * torch.eye(3).to(dtype=torch.float32)
    pred = AffineJoint(
        GaussianJoint(torch.zeros(500, 3), tril), torch.zeros(3), torch.tensor([0.3, 1.7, 2.9])
    )
    cov = pred.covariance()
    assert float((cov - cov.mT).abs().max()) > 0  # the premise: not bitwise symmetric
    y = pred.sample(1)[:, 0]
    result = nees_consistency(pred.mean(), cov, y)
    assert result.verdict == "consistent"
    mask = torch.rand(500, 3) > 0.3
    masked = nees_consistency(pred.mean(), cov, y, mask=mask)
    assert bool(torch.isfinite(masked.statistic))
