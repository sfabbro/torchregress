"""Tests for Neyman confidence sets with learned critical values (LF2I)."""

from __future__ import annotations

import math

import pytest
import torch
from scipy.stats import chi2

from torchregress.inference import (
    BinnedCoverage,
    CriticalValueRegressor,
    NeymanConfidenceSet,
    SmoothCoverage,
    binned_conditional_coverage,
    neyman,
    smooth_conditional_coverage,
)
from torchregress.losses import WaldoConformal
from torchregress.losses.conformal import finite_sample_quantile

# ═══════════════════════════════════════════════════════════════════════════════
# data
# ═══════════════════════════════════════════════════════════════════════════════


def _gaussian_mean(n: int, p: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """X ~ N(theta, I_P), theta ~ U[-2, 2]^P; lambda = exact -log LR ~ chi2_P / 2."""
    g = torch.Generator().manual_seed(seed)
    theta = torch.rand(n, p, generator=g, dtype=torch.float64) * 4 - 2
    x = theta + torch.randn(n, p, generator=g, dtype=torch.float64)
    stat = 0.5 * ((x - theta) ** 2).sum(-1)
    return theta, x, stat


def _heteroscedastic(n: int, p: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Skewed population, noise growing with theta, unit-variance Wald statistic.

    theta_j = 3 * Beta(1, 4) (rare at large theta), X ~ N(theta, s(theta)^2 I) with
    s = 0.3 + 0.5 * sum(theta) / sqrt(P), and lambda = |x - theta|^2 (a statistic
    whose distribution, s^2 chi2_P, depends on theta).
    """
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(n, p, generator=g, dtype=torch.float64)
    theta = 3.0 * (1.0 - (1.0 - u) ** 0.25)
    scale = 0.3 + 0.5 * theta.sum(-1, keepdim=True) / math.sqrt(p)
    x = theta + scale * torch.randn(n, p, generator=g, dtype=torch.float64)
    return theta, ((x - theta) ** 2).sum(-1)


def _region_key(theta: torch.Tensor) -> torch.Tensor:
    return theta.sum(-1) / math.sqrt(theta.shape[-1])


# ═══════════════════════════════════════════════════════════════════════════════
# regular model: the statistic is pivotal (chi2_P / 2 at every theta)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("method", ["mlp", "knn"])
def test_gaussian_mean_matches_chi_square_quantile(p: int, method: str) -> None:
    alphas = [0.32, 0.1]
    theta, _, stat = _gaussian_mean(6000, p, seed=p)
    reg = CriticalValueRegressor(alphas, method=method).fit(theta, stat)

    g = torch.Generator().manual_seed(10 + p)
    grid = torch.rand(500, p, generator=g, dtype=torch.float64) * 3.2 - 1.6
    pred = reg.predict(grid)
    assert pred.shape == (500, 2)
    exact = torch.tensor(chi2.ppf([1 - a for a in alphas], p) / 2, dtype=torch.float64)
    assert float((pred / exact - 1).mean(0).abs().max()) < 0.08
    # Judge the estimate where it matters, in coverage: P(lambda <= C_hat(theta) | theta).
    # The chi2 tail density is small, so a 10 % quantile error moves coverage by ~1-2 %.
    levels = torch.tensor([1 - a for a in alphas], dtype=torch.float64)
    error = torch.from_numpy(chi2.cdf(2 * pred.numpy(), p)) - levels
    # Mean over theta: the fit's in-sample coverage has a binomial standard error
    # of ~0.007 at level 0.68 on the 4800 training rows; allow ~3 of them.
    assert float(error.mean(0).abs().max()) < 0.02
    if method == "mlp":
        assert float(error.abs().max()) < 0.05, float(error.abs().max())
    else:
        # k-NN: the order statistic of k ~ 200-600 neighbours is noisy pointwise.
        assert float(error.abs().median()) < 0.03
        assert float(error.abs().max()) < 0.12, float(error.abs().max())

    # Held-out coverage is nominal at each level.
    theta_t, _, stat_t = _gaussian_mean(6000, p, seed=100 + p)
    covered = NeymanConfidenceSet(reg).contains(stat_t, theta_t)
    assert covered.shape == (6000, 2)
    coverage = covered.double().mean(0)
    for c, a in zip(coverage.tolist(), alphas):
        assert abs(c - (1 - a)) < 0.03


def _boundary_cdf(c: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """P(lambda <= c | theta) for X ~ N(theta, 1), theta > 0, MLE max(X, 0).

    lambda = (x - theta)^2 for x >= 0 and theta^2 - 2 x theta for x < 0, so the
    acceptance set is x in [max(0, theta - sqrt(c)), theta + sqrt(c)] together
    with x in [min(0, (theta^2 - c) / (2 theta)), 0).
    """
    root = c.clamp_min(0).sqrt()
    normal = torch.distributions.Normal(theta, torch.ones_like(theta))
    upper = normal.cdf(theta + root) - normal.cdf(torch.clamp(theta - root, min=0.0))
    lower = normal.cdf(torch.zeros_like(theta)) - normal.cdf(
        torch.clamp((theta**2 - c) / (2 * theta), max=0.0)
    )
    return upper + lower


@pytest.mark.parametrize("method", ["mlp", "knn"])
def test_boundary_constrained_mean_non_regular(method: str) -> None:
    """Gaussian mean with theta >= 0 (non-regular at the boundary).

    lambda(theta; x) = (x - theta)^2 - (x - max(x, 0))^2 is 0 with probability
    1/2 at theta = 0, so its 0.9 quantile rises from chi2_1(0.8) = 1.64 at the
    boundary to chi2_1(0.9) = 2.71 away from it: no constant threshold has
    nominal coverage everywhere, the learned one does.
    """
    g = torch.Generator().manual_seed(3)
    theta = torch.rand(6000, 1, generator=g, dtype=torch.float64) * 4.0
    x = theta + torch.randn(6000, 1, generator=g, dtype=torch.float64)
    stat = ((x - theta) ** 2 - (x - x.clamp_min(0.0)) ** 2).squeeze(-1)
    grid = torch.linspace(0.05, 3.8, 40, dtype=torch.float64)

    reg = CriticalValueRegressor(0.1, method=method).fit(theta, stat)
    learned = _boundary_cdf(reg.predict(grid.unsqueeze(-1)), grid)
    constant = _boundary_cdf(finite_sample_quantile(stat, 0.1).expand(40), grid)
    tol = 0.035 if method == "mlp" else 0.045
    assert float((learned - 0.9).abs().max()) < tol, float((learned - 0.9).abs().max())
    # The marginal threshold over-covers at the boundary, under-covers away from it.
    assert float(constant[0]) > 0.94 and float(constant[-1]) < 0.9


# ═══════════════════════════════════════════════════════════════════════════════
# heteroscedastic model on a skewed population
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(("p", "method"), [(1, "knn"), (1, "mlp"), (2, "mlp")])
def test_learned_critical_value_restores_coverage_in_rare_region(p: int, method: str) -> None:
    alpha = 0.32
    nominal = 1 - alpha
    theta, stat = _heteroscedastic(10000, p, seed=20 + p)
    theta_t, stat_t = _heteroscedastic(4000, p, seed=40 + p)
    # Judge coverage where the calibration set has support (key <= 1.6 holds
    # ~95 % of the population and leaves ~500 calibration rows above 1.2).  Further
    # out the local calibration sample is too small for any estimate of C(theta)
    # to be accurate, and the diagnostics rightly flag it.
    keep = _region_key(theta_t) <= 1.6
    theta_t, stat_t = theta_t[keep], stat_t[keep]

    constant = stat_t <= finite_sample_quantile(stat, alpha)
    cs = NeymanConfidenceSet.calibrate(theta, stat, alpha=alpha, method=method)
    calibrated = cs.contains(stat_t, theta_t)

    key = _region_key(theta_t)
    regions = [key < 0.4, (key >= 0.4) & (key < 0.8), (key >= 0.8) & (key < 1.2), key >= 1.2]
    const_cov = [float(constant[r].double().mean()) for r in regions]
    cal_cov = [float(calibrated[r].double().mean()) for r in regions]
    # The constant threshold over-covers the dense region and under-covers the
    # rare one, whereas the Neyman set is nominal everywhere.
    assert const_cov[0] > nominal + 0.08
    assert const_cov[-1] < nominal - 0.15
    # Tolerance: binomial error of the ~250 held-out rows in the rare region
    # (0.03) plus the estimation error of C(theta) there; k-NN neighbourhoods
    # are one-sided at the sparse end of the support and lean towards the
    # denser, lower-noise side.
    tol = 0.07 if method == "mlp" else 0.1
    for c in cal_cov:
        assert abs(c - nominal) < tol, cal_cov

    # The conditional-coverage diagnostics (family-wise 1 %) flag the constant
    # threshold only.
    for covered, is_constant in ((constant, True), (calibrated, False)):
        binned = binned_conditional_coverage(
            theta_t, covered, nominal=nominal, bins=4, confidence=0.99
        )
        smooth = smooth_conditional_coverage(theta_t, covered, nominal=nominal, confidence=0.99)
        flagged_bins = [bool(b.under_covered.any()) for b in binned]
        flagged_frac = float(smooth.under_covered.double().mean())
        if is_constant:
            # Flagged where the population is rare (large theta) ...
            assert all(bool(b.under_covered[2:].any()) for b in binned)
            assert not any(bool(b.under_covered[0]) for b in binned)
            assert flagged_frac > 0.1
            assert float(key[smooth.under_covered].min()) > 0.4
            # ... and over-coverage where it is dense.
            assert bool(smooth.over_covered[key < 0.2].all())
        else:
            assert flagged_frac < 0.05
            # (k-NN's sparse-end bias can be large enough for a bin to be
            # flagged, correctly; the MLP's must not be.)
            if method == "mlp":
                assert not any(flagged_bins), [b.coverage.tolist() for b in binned]


# ═══════════════════════════════════════════════════════════════════════════════
# regressor contracts
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("method", ["mlp", "knn"])
def test_levels_do_not_cross_and_follow_the_given_order(method: str) -> None:
    theta, stat = _heteroscedastic(1500, 2, seed=5)
    alphas = [0.05, 0.32, 0.1, 0.5]
    reg = CriticalValueRegressor(alphas, method=method, max_epochs=40).fit(theta, stat)
    assert reg.levels == pytest.approx([0.95, 0.68, 0.9, 0.5])
    grid = torch.rand(300, 2, dtype=torch.float64) * 6 - 1.5  # includes extrapolation
    pred = reg.predict(grid)
    assert pred.shape == (300, 4)
    ordered = pred[:, [3, 1, 2, 0]]  # levels 0.5 < 0.68 < 0.9 < 0.95
    assert bool((ordered[:, 1:] >= ordered[:, :-1]).all())
    # Each column is the single-level fit of the same level (k-NN: exactly).
    if method == "knn":
        single = CriticalValueRegressor(0.1, method="knn", n_neighbors=reg.n_neighbors_)
        torch.testing.assert_close(single.fit(theta, stat).predict(grid), pred[:, 2])


def test_knn_is_the_finite_sample_quantile_of_the_neighbours() -> None:
    theta, stat = _heteroscedastic(400, 2, seed=6)
    reg = CriticalValueRegressor([0.1, 0.32], method="knn", n_neighbors=37).fit(theta, stat)
    query = theta[:5] + 0.01
    mean, std = theta.mean(0), theta.std(0, unbiased=False)
    dist = torch.cdist((query - mean) / std, (theta - mean) / std)
    nearest = dist.topk(37, largest=False).indices
    expected = torch.stack(
        [
            torch.stack([finite_sample_quantile(stat[idx], a) for a in (0.1, 0.32)])
            for idx in nearest
        ]
    )
    torch.testing.assert_close(reg.predict(query), expected)
    # alpha < 1/(k+1): no finite order statistic carries the level.
    tiny = CriticalValueRegressor(0.01, method="knn", n_neighbors=20).fit(theta, stat)
    assert bool(torch.isinf(tiny.predict(query)).all())


def test_mlp_is_deterministic_for_a_seed() -> None:
    theta, stat = _heteroscedastic(1000, 2, seed=7)
    kwargs = dict(max_epochs=15, seed=3)
    a = CriticalValueRegressor([0.1, 0.32], **kwargs).fit(theta, stat)  # type: ignore[arg-type]
    b = CriticalValueRegressor([0.1, 0.32], **kwargs).fit(theta, stat)  # type: ignore[arg-type]
    c = CriticalValueRegressor([0.1, 0.32], max_epochs=15, seed=4).fit(theta, stat)
    grid = theta[:50]
    assert torch.equal(a.predict(grid), b.predict(grid))
    assert not torch.equal(a.predict(grid), c.predict(grid))
    assert a.history_["train"] == b.history_["train"]
    assert a.best_epoch_ is not None and len(a.history_["val"]) >= a.best_epoch_ + 1


@pytest.mark.parametrize("method", ["mlp", "knn"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_float32_and_float64(method: str, dtype: torch.dtype, device: str) -> None:
    theta, stat = _heteroscedastic(1200, 2, seed=8)
    theta, stat = theta.to(device, dtype), stat.to(device, dtype)
    reg = CriticalValueRegressor(0.1, method=method, max_epochs=20).fit(theta, stat)
    pred = reg.predict(theta[:20])
    assert pred.dtype == dtype and pred.device == theta.device and pred.shape == (20,)
    assert bool(torch.isfinite(pred).all()) and bool((pred > 0).all())
    # Queries in another dtype come back in that dtype.
    other = torch.float64 if dtype == torch.float32 else torch.float32
    assert reg.predict(theta[:3].to(other)).dtype == other
    # float32 and float64 fits agree to optimisation tolerance.
    ref = CriticalValueRegressor(0.1, method=method, max_epochs=20).fit(
        theta.double(), stat.double()
    )
    torch.testing.assert_close(
        pred.double(), ref.predict(theta[:20].double()), rtol=0.05, atol=0.02
    )


def test_statistic_transform() -> None:
    """The log reparametrisation is used for non-negative statistics only."""
    theta, stat = _heteroscedastic(3000, 1, seed=19)
    auto = CriticalValueRegressor(0.1, batch_size=1024).fit(theta, stat)
    assert auto.transform_ == "log"
    none = CriticalValueRegressor(0.1, batch_size=1024, transform="none").fit(theta, stat)
    assert none.transform_ == "none"
    # Both estimate the same quantile function, s(theta)^2 chi2_1(0.9).
    grid = torch.linspace(0.05, 1.0, 20, dtype=torch.float64)
    scale2 = (0.3 + 0.5 * grid) ** 2
    for reg in (auto, none):
        pred = reg.predict(grid.unsqueeze(-1))
        coverage = torch.from_numpy(chi2.cdf((pred / scale2).numpy(), 1))
        assert float((coverage - 0.9).abs().max()) < 0.04
    shifted = CriticalValueRegressor(0.1, max_epochs=30).fit(theta, stat - 1.0)
    assert shifted.transform_ == "none"
    with pytest.raises(ValueError, match="non-negative"):
        CriticalValueRegressor(0.1, transform="log").fit(theta, stat - 1.0)
    with pytest.raises(ValueError, match="transform"):
        CriticalValueRegressor(transform="sqrt")  # type: ignore[arg-type]


def test_one_dimensional_theta_and_constant_column() -> None:
    theta, _, stat = _gaussian_mean(800, 1, seed=9)
    reg = CriticalValueRegressor(0.1, method="knn").fit(theta[:, 0], stat)
    assert reg.predict(theta[:4, 0]).shape == (4,)
    # A constant parameter column is not rescaled to nan.
    padded = torch.cat([theta, torch.ones_like(theta)], dim=1)
    pred = CriticalValueRegressor(0.1, method="knn").fit(padded, stat).predict(padded[:4])
    assert bool(torch.isfinite(pred).all())


def test_regressor_validation() -> None:
    theta, _, stat = _gaussian_mean(50, 2, seed=11)
    with pytest.raises(ValueError, match="alpha"):
        CriticalValueRegressor(1.2)
    with pytest.raises(ValueError, match="distinct"):
        CriticalValueRegressor([0.1, 0.1])
    with pytest.raises(ValueError, match="method"):
        CriticalValueRegressor(method="gbm")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="val_fraction"):
        CriticalValueRegressor(val_fraction=1.0)
    with pytest.raises(RuntimeError, match="fit"):
        CriticalValueRegressor().predict(theta)
    with pytest.raises(ValueError, match="rows"):
        CriticalValueRegressor().fit(theta, stat[:-1])
    with pytest.raises(ValueError, match="finite"):
        CriticalValueRegressor().fit(theta, torch.where(stat > 1, math.inf, stat))
    reg = CriticalValueRegressor(0.1, method="knn").fit(theta, stat)
    with pytest.raises(ValueError, match="columns"):
        reg.predict(theta[:, :1])
    multi = CriticalValueRegressor([0.1, 0.2], method="knn").fit(theta, stat)
    with pytest.raises(ValueError, match="single alpha"):
        multi.predict_critical(theta)


# ═══════════════════════════════════════════════════════════════════════════════
# confidence sets
# ═══════════════════════════════════════════════════════════════════════════════


def _gaussian_statistic(theta_bank: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """-log LR of N(theta, I): [K, P] or [n, K, P] bank, [n, P] data -> [n, K]."""
    bank = theta_bank if theta_bank.dim() == 3 else theta_bank.unsqueeze(0)
    return 0.5 * ((x.unsqueeze(1) - bank) ** 2).sum(-1)


def test_region_matches_contains_and_covers_nominally() -> None:
    theta, _, stat = _gaussian_mean(3000, 2, seed=12)
    cs = NeymanConfidenceSet.calibrate(theta, stat, alpha=[0.32, 0.1], method="knn")
    theta_t, x_t, stat_t = _gaussian_mean(400, 2, seed=13)

    axis = torch.linspace(-4, 4, 41, dtype=torch.float64)
    bank = torch.cartesian_prod(axis, axis)  # [K, 2]
    masks = cs.region(_gaussian_statistic, bank, x_t)
    assert masks.shape == (400, bank.shape[0], 2) and masks.dtype == torch.bool
    # Identical to contains() on the evaluated statistic, and chunk-invariant.
    direct = cs.contains(_gaussian_statistic(bank, x_t[:20]), bank.expand(20, -1, -1))
    assert torch.equal(masks[:20], direct)
    assert torch.equal(cs.region(_gaussian_statistic, bank, x_t, chunk_size=37), masks)
    # Nested levels: the 68 % set is inside the 90 % set.
    assert bool((masks[..., 0] <= masks[..., 1]).all())

    # Per-row bank holding the true parameter: its membership is the coverage.
    per_row = torch.cat([theta_t.unsqueeze(1), bank.expand(400, -1, -1)[:, :10]], dim=1)
    covered = cs.region(_gaussian_statistic, per_row, x_t, chunk_size=64)[:, 0]
    torch.testing.assert_close(covered, cs.contains(stat_t, theta_t))
    coverage = covered.double().mean(0)
    assert abs(float(coverage[0]) - 0.68) < 0.07 and abs(float(coverage[1]) - 0.9) < 0.05


def test_callable_critical_value_and_validation() -> None:
    _, x, stat = _gaussian_mean(100, 1, seed=14)
    theta = torch.zeros(100, 1, dtype=torch.float64)
    cs = NeymanConfidenceSet(lambda t: torch.full((t.shape[0],), 1.0, dtype=t.dtype))
    torch.testing.assert_close(cs.contains(stat, theta), stat <= 1.0)
    # P = 1 parameters may be given with the statistic's shape.
    torch.testing.assert_close(cs.contains(stat, theta[:, 0]), stat <= 1.0)
    assert cs.critical_value(torch.zeros(3, 4, 1)).shape == (3, 4)
    with pytest.raises(ValueError, match="leading shape"):
        cs.contains(stat, torch.zeros(7, 1))
    with pytest.raises(ValueError, match="fitted"):
        NeymanConfidenceSet(CriticalValueRegressor())
    with pytest.raises(TypeError):
        NeymanConfidenceSet(1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="statistic_fn"):
        cs.region(lambda b, xx: torch.zeros(xx.shape[0], 2), torch.zeros(3, 1), x)


def test_waldo_conformal_accepts_the_regressor() -> None:
    """WaldoConformal uses the learned C(y) through its regressor interface."""
    theta, _ = _heteroscedastic(4000, 1, seed=15)
    g = torch.Generator().manual_seed(16)
    # The parameter is the regression target; the estimate's error grows with it,
    # but the score assumes unit variance.
    target = theta
    mean = theta + (0.3 + 0.5 * theta) * torch.randn(theta.shape, generator=g, dtype=torch.float64)
    reg = CriticalValueRegressor(0.1, method="knn")
    cp = WaldoConformal(alpha=0.1, critical_value_regressor=reg)
    cp.calibrate(mean, target)
    torch.testing.assert_close(cp.critical_value(theta[:5]), reg.predict(theta[:5]))
    covered = cp.contains(mean, target)
    rare = theta[:, 0] > 1.5
    assert abs(float(covered[rare].double().mean()) - 0.9) < 0.05
    # The constant WALDO threshold under-covers the same region.
    constant = WaldoConformal(alpha=0.1)
    constant.calibrate(mean, target)
    assert float(constant.contains(mean, target)[rare].double().mean()) < 0.8


# ═══════════════════════════════════════════════════════════════════════════════
# diagnostics
# ═══════════════════════════════════════════════════════════════════════════════


def test_binned_coverage_wilson_matches_statsmodels() -> None:
    from statsmodels.stats.proportion import proportion_confint

    g = torch.Generator().manual_seed(17)
    theta = torch.rand(500, 2, generator=g, dtype=torch.float64)
    covered = torch.rand(500, generator=g, dtype=torch.float64) < 0.7
    out = binned_conditional_coverage(
        theta, covered, nominal=0.9, bins=4, confidence=0.9, bonferroni=False
    )
    assert len(out) == 2 and isinstance(out[0], BinnedCoverage)
    b = out[1]
    assert int(b.count.sum()) == 500
    for i in range(4):
        n, k = int(b.count[i]), round(float(b.coverage[i]) * int(b.count[i]))
        lo, hi = proportion_confint(k, n, alpha=0.1, method="wilson")
        assert float(b.lower[i]) == pytest.approx(lo, abs=1e-10)
        assert float(b.upper[i]) == pytest.approx(hi, abs=1e-10)
    # Coverage 0.7 against a 0.9 target is flagged in every bin.
    assert bool(b.under_covered.all()) and not bool(b.over_covered.any())


def test_binned_coverage_bins_and_empty_bins() -> None:
    theta = torch.tensor([0.0, 0.1, 0.2, 0.9, 1.0], dtype=torch.float64)
    covered = torch.tensor([1, 1, 0, 1, 1])
    (b,) = binned_conditional_coverage(theta, covered, nominal=0.5, bins=4)
    assert b.count.tolist() == [3, 0, 0, 2]
    assert math.isnan(float(b.coverage[1])) and not bool(b.under_covered[1])
    (q,) = binned_conditional_coverage(theta, covered, nominal=0.5, bins=2, binning="quantile")
    assert int(q.count.sum()) == 5
    edges = [torch.tensor([0.0, 0.5, 1.0])]
    (e,) = binned_conditional_coverage(theta, covered, nominal=0.5, bins=edges)
    assert e.count.tolist() == [3, 2]
    with pytest.raises(ValueError, match="increasing"):
        binned_conditional_coverage(theta, covered, nominal=0.5, bins=[torch.tensor([1.0, 0.0])])
    with pytest.raises(ValueError, match="nominal"):
        binned_conditional_coverage(theta, covered, nominal=1.0)


def test_smooth_coverage_recovers_a_logistic_curve_and_is_calibrated() -> None:
    g = torch.Generator().manual_seed(18)
    theta = torch.rand(6000, 1, generator=g, dtype=torch.float64) * 4 - 2
    true_p = torch.sigmoid(1.2 + 0.8 * theta[:, 0])
    covered = torch.rand(6000, generator=g, dtype=torch.float64) < true_p
    grid = torch.linspace(-1.8, 1.8, 50, dtype=torch.float64).unsqueeze(-1)
    out = smooth_conditional_coverage(theta, covered, nominal=0.8, evaluate_at=grid)
    assert isinstance(out, SmoothCoverage) and out.probability.shape == (50,)
    truth = torch.sigmoid(1.2 + 0.8 * grid[:, 0])
    assert float((out.probability - truth).abs().max()) < 0.05
    assert bool(((out.lower <= truth) & (truth <= out.upper)).all())
    # The band is wider pointwise-simultaneous than pointwise.
    pointwise = smooth_conditional_coverage(
        theta, covered, nominal=0.8, evaluate_at=grid, simultaneous=False
    )
    assert bool((pointwise.upper - pointwise.lower < out.upper - out.lower).all())
    # Flags: below 0.8 on the left, above on the right.
    assert bool(out.under_covered[0]) and bool(out.over_covered[-1])


def test_smooth_coverage_handles_perfect_coverage() -> None:
    theta = torch.randn(200, 2, dtype=torch.float64)
    out = smooth_conditional_coverage(theta, torch.ones(200, dtype=torch.bool), nominal=0.9)
    assert bool(torch.isfinite(out.probability).all())
    assert not bool(out.under_covered.any())


def test_module_exports() -> None:
    assert set(neyman.__all__) == {
        "BinnedCoverage",
        "CriticalValueRegressor",
        "NeymanConfidenceSet",
        "SmoothCoverage",
        "binned_conditional_coverage",
        "smooth_conditional_coverage",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# re-scored neighbours and confidence-adjusted ranks (k-NN)
# ═══════════════════════════════════════════════════════════════════════════════

_BACKGROUND = 3.0


def _poisson_statistic(theta_bank: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """-2 log LR for n ~ Poisson(mu + b), mu >= 0: [K, 1] or [n, K, 1] bank -> [n, K]."""
    bank = theta_bank if theta_bank.dim() == 3 else theta_bank.unsqueeze(0)
    rate = bank[..., 0] + _BACKGROUND
    n = x.reshape(-1, 1)
    best = n.clamp_min(_BACKGROUND)
    return (2 * ((rate - best) + torch.xlogy(n, best / rate))).clamp_min(0)


def _poisson_exact_coverage(reg: CriticalValueRegressor, grid: torch.Tensor) -> torch.Tensor:
    counts = torch.arange(0, 121, dtype=torch.float64)
    rate = grid + _BACKGROUND
    pmf = torch.exp(counts * torch.log(rate) - rate - torch.lgamma(counts + 1))  # [G, C]
    accepted = _poisson_statistic(grid, counts).T.unsqueeze(-1) <= reg.predict(grid).unsqueeze(1)
    return (pmf.unsqueeze(-1) * accepted).sum(1)


def _poisson_calibration(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    mu = torch.rand(n, 1, generator=gen, dtype=torch.float64) * 16
    x = torch.poisson(mu + _BACKGROUND, generator=gen)
    stat = _poisson_statistic(mu.unsqueeze(1), x)[:, 0]
    return mu, x, stat


def test_confidence_index_is_the_binomial_upper_bound() -> None:
    from scipy.stats import binom

    from torchregress.inference.neyman import _confidence_index

    alphas = torch.tensor([0.3173, 0.1, 0.05], dtype=torch.float64)
    for k in (40, 345, 1380):
        for conf in (0.5, 0.9, 0.95):
            got = _confidence_index(k, alphas, conf).tolist()
            want = [int(binom.ppf(conf, k, 1 - a)) for a in alphas.tolist()]
            assert got == want
    # Too few neighbours to certify a 99.9% quantile: no order statistic qualifies.
    assert int(_confidence_index(20, torch.tensor([0.001], dtype=torch.float64), 0.95)) == 20


def test_rescored_neighbours_use_the_statistic_at_the_query() -> None:
    mu, x, stat = _poisson_calibration(600, seed=3)
    reg = CriticalValueRegressor(
        [0.1, 0.32], method="knn", n_neighbors=41, statistic_fn=_poisson_statistic
    ).fit(mu, stat, x)
    query = torch.tensor([[0.0], [2.5], [9.0]], dtype=torch.float64)
    got = reg.predict(query)
    for row, q in enumerate(query):
        idx = torch.topk((mu[:, 0] - q[0]).abs(), 41, largest=False).indices
        values = torch.sort(_poisson_statistic(q.reshape(1, 1), x[idx])[:, 0]).values
        for col, alpha in enumerate([0.1, 0.32]):
            assert got[row, col] == values[math.ceil(42 * (1 - alpha)) - 1]


def test_rescored_knn_covers_count_data_where_pooled_estimates_do_not() -> None:
    """Exact coverage by pmf sums: the saw-tooth critical value of a count statistic."""
    mu, x, stat = _poisson_calibration(20_000, seed=0)
    grid = torch.linspace(0, 15, 601, dtype=torch.float64).unsqueeze(-1)
    nominal = torch.tensor([1 - 0.3173, 0.9], dtype=torch.float64)

    pooled = CriticalValueRegressor([0.3173, 0.1], method="knn").fit(mu, stat)
    pooled_cov = _poisson_exact_coverage(pooled, grid)
    assert float((nominal - pooled_cov).max()) > 0.04  # drops whole atoms

    rescored = CriticalValueRegressor(
        [0.3173, 0.1], method="knn", statistic_fn=_poisson_statistic
    ).fit(mu, stat, x)
    rescored_cov = _poisson_exact_coverage(rescored, grid)
    assert float((nominal - rescored_cov).max()) < 0.02

    certified = CriticalValueRegressor(
        [0.3173, 0.1], method="knn", statistic_fn=_poisson_statistic, confidence=0.95
    ).fit(mu, stat, x)
    certified_cov = _poisson_exact_coverage(certified, grid)
    assert float((nominal - certified_cov).max()) < 0.005
    assert bool((certified.predict(grid) >= rescored.predict(grid)).all())


def test_calibrate_forwards_observations_and_validates() -> None:
    mu, x, stat = _poisson_calibration(500, seed=5)
    cs = NeymanConfidenceSet.calibrate(
        mu, stat, alpha=0.1, x=x, method="knn", statistic_fn=_poisson_statistic, confidence=0.9
    )
    bank = torch.linspace(0, 15, 31, dtype=torch.float64).unsqueeze(-1)
    assert cs.region(_poisson_statistic, bank, x[:7]).shape == (7, 31)
    with pytest.raises(ValueError, match="knn"):
        CriticalValueRegressor(0.1, confidence=0.9)
    with pytest.raises(ValueError, match="knn"):
        CriticalValueRegressor(0.1, statistic_fn=_poisson_statistic)
    with pytest.raises(ValueError, match="confidence"):
        CriticalValueRegressor(0.1, method="knn", confidence=1.0)
    with pytest.raises(ValueError, match="required with statistic_fn"):
        CriticalValueRegressor(0.1, method="knn", statistic_fn=_poisson_statistic).fit(mu, stat)
    with pytest.raises(ValueError, match="required with statistic_fn"):
        CriticalValueRegressor(0.1, method="knn").fit(mu, stat, x)
