"""JointRegressor end to end: heads, stacking, recalibration, conformal regions."""

from __future__ import annotations

import math

import pytest
import torch

from torchregress.joint import JointPredictive, JointRegressor, QuantileNetworkEngine

BUDGET = dict(max_epochs=60, patience=10, hidden=(48, 48), batch_size=128)


def _data(n: int, seed: int, d: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """Nonlinear mean, correlation that depends on x, one skewed target."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, generator=gen)
    rho = torch.tanh(x[:, 0])
    e = torch.randn(n, d, generator=gen)
    y = torch.empty(n, d)
    y[:, 0] = torch.sin(x[:, 1]) + 0.5 * e[:, 0]
    y[:, 1] = x[:, 2] ** 2 + 0.5 * (rho * e[:, 0] + (1 - rho**2).sqrt() * e[:, 1])
    y[:, 2] = 0.3 * x[:, 3] + 0.3 * torch.exp(0.6 * e[:, 2])
    return x, y


def _coverage_tolerance(n: int, level: float, n_cal: int) -> float:
    # Test binomial noise plus the calibration-set fluctuation of the realised coverage.
    return 4.0 * math.sqrt(level * (1 - level) * (1 / n + 1 / n_cal))


@pytest.mark.parametrize("method", ["gaussian", "mixture", "flow", "sklar"])
def test_each_head_fits_and_predicts(method: str) -> None:
    x, y = _data(900, seed=0)
    model = JointRegressor(method, seed=0, **BUDGET).fit(x, y)
    dist = model.predict(x[:7])
    assert isinstance(dist, JointPredictive)
    assert (dist.batch_size, dist.event_dim) == (7, 3)
    draws = dist.sample(11)
    assert draws.shape == (7, 11, 3) and bool(torch.isfinite(draws).all())
    assert dist.has_density
    lp = dist.log_prob(y[:7])
    assert lp.shape == (7,) and bool(torch.isfinite(lp).all())
    assert dist.log_prob(draws).shape == (7, 11)
    assert model.members_ == [method]
    assert model.scale_ is not None and model.scale_.shape == (3,)
    if method == "sklar":
        assert set(model.dependence_scores_) >= {"independence", "gaussian_copula"}


def test_fitted_models_beat_the_unconditional_gaussian() -> None:
    x, y = _data(2500, seed=1)
    x_te, y_te = _data(1500, seed=2)
    mean, cov = y.mean(0), torch.cov(y.T)
    baseline = torch.distributions.MultivariateNormal(mean, cov).log_prob(y_te).mean()
    for method in ("gaussian", "mixture"):
        model = JointRegressor(method, seed=0, **BUDGET).fit(x, y)
        score = model.predict(x_te).log_prob(y_te).mean()
        assert float(score) > float(baseline) + 0.3, method


def test_auto_stacks_heads_and_regions_cover() -> None:
    x, y = _data(2400, seed=3)
    model = JointRegressor(seed=0, **BUDGET).fit(x, y)
    assert set(model.members_) == {"gaussian", "mixture", "sklar", "flow"}
    assert model.weights_ is not None
    assert float(model.weights_.sum()) == pytest.approx(1.0, abs=1e-5)
    assert set(model.validation_log_score_) == set(model.members_)

    x_te, y_te = _data(2000, seed=4)
    dist = model.predict(x_te)
    stacked = float(dist.log_prob(y_te).mean())
    best_single = max(model.validation_log_score_.values())
    assert math.isfinite(stacked) and math.isfinite(best_single)

    n_cal = round(0.15 * 2400)
    for alpha in (0.1, 0.32):
        covered = model.region(x_te, alpha=alpha).contains(y_te.to(model.device))
        assert covered.shape == (2000,)
        cov = float(covered.float().mean())
        assert abs(cov - (1 - alpha)) < _coverage_tolerance(2000, 1 - alpha, n_cal), (alpha, cov)
    multi = model.region(x_te[:5], alpha=[0.1, 0.32]).contains(y_te[:5].to(model.device))
    assert multi.shape == (5, 2)
    assert bool((multi[:, 0] | ~multi[:, 1]).all())  # the 90% region contains the 68% one


def test_region_for_targets_of_interest() -> None:
    x, y = _data(1500, seed=5)
    model = JointRegressor("gaussian", seed=0, **BUDGET).fit(x, y)
    x_te, y_te = _data(2000, seed=6)
    region = model.region(x_te, alpha=0.1, interest=[1])
    assert region.predictive.event_dim == 1
    covered = region.contains(y_te[:, [1]].to(model.device))
    n_cal = round(0.15 * 1500)
    assert abs(float(covered.float().mean()) - 0.9) < _coverage_tolerance(2000, 0.9, n_cal)
    # Dropping the nuisance targets makes the region for the target of interest tighter
    # than the projection of the joint region.
    joint_box = model.region(x_te[:200], alpha=0.1).bounding_box()
    lower, upper = region.bounding_box()
    width_interest = float((upper[:200, 0] - lower[:200, 0]).mean())
    width_joint = float((joint_box[1][:, 1] - joint_box[0][:, 1]).mean())
    assert width_interest < width_joint


def test_units_do_not_matter() -> None:
    x, y = _data(900, seed=7)
    scale = torch.tensor([1000.0, 0.01, 5.0])
    shift = torch.tensor([5000.0, -3.0, 0.0])
    a = JointRegressor("gaussian", seed=0, **BUDGET).fit(x, y)
    b = JointRegressor("gaussian", seed=0, **BUDGET).fit(x, shift + scale * y)
    lp_a = a.predict(x[:50]).log_prob(y[:50])
    lp_b = b.predict(x[:50]).log_prob(shift + scale * y[:50])
    torch.testing.assert_close(lp_b + scale.log().sum(), lp_a, atol=5e-2, rtol=0)
    torch.testing.assert_close(
        b.predict(x[:50]).mean(), shift + scale * a.predict(x[:50]).mean(), atol=1e-2, rtol=1e-3
    )


def test_missing_targets_are_integrated_out() -> None:
    x, y = _data(1800, seed=8)
    gen = torch.Generator().manual_seed(0)
    missing = torch.rand(y.shape, generator=gen) < 0.25
    y_obs = torch.where(missing, torch.nan, y)
    model = JointRegressor(seed=0, **BUDGET).fit(x, y_obs)
    assert set(model.members_) == {"gaussian", "mixture"}
    x_te, y_te = _data(1500, seed=9)
    dist = model.predict(x_te)
    mask = torch.rand(y_te.shape, generator=gen) > 0.25
    lp = dist.log_prob(torch.where(mask, y_te, torch.nan).to(model.device), mask.to(model.device))
    assert bool(torch.isfinite(lp).all())
    region = model.region(x_te, alpha=0.1)
    covered = region.contains(y_te.to(model.device), mask.to(model.device))
    n_cal = round(0.15 * 1800)
    assert abs(float(covered.float().mean()) - 0.9) < _coverage_tolerance(1500, 0.9, n_cal)


def test_custom_marginal_engine_and_validation() -> None:
    x, y = _data(700, seed=10)
    engine = QuantileNetworkEngine(n_quantiles=15, hidden=(32,), max_epochs=40, patience=8)
    model = JointRegressor("sklar", marginals=engine, seed=0, **BUDGET).fit(x, y)
    assert model.predict(x[:3]).sample(4).shape == (3, 4, 3)
    with pytest.raises(ValueError, match="unknown method"):
        JointRegressor("forest")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fractions"):
        JointRegressor(val_fraction=0.5, recal_fraction=0.3, conformal_fraction=0.3)
    with pytest.raises(RuntimeError, match="fit"):
        JointRegressor().region(x)
    with pytest.raises(ValueError, match="at least 50"):
        JointRegressor().fit(x[:20], y[:20])
    with pytest.raises(ValueError, match="observed value"):
        JointRegressor().fit(x, torch.cat([y[:, :2], torch.full((700, 1), torch.nan)], 1))
