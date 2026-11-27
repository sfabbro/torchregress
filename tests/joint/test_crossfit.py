"""Cross-fitting in ``JointRegressor`` (``cv=K``): folds, stacking, recalibration, regions."""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from torchregress.joint import (
    CrossConformalRegion,
    JointPredictive,
    JointRegion,
    JointRegressor,
    MixtureJoint,
)
from torchregress.viz import plot_corner_plot, plot_joint_diagnostics

TINY = dict(max_epochs=15, patience=5, hidden=(16,), device="cpu")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _data(n: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Two targets, Gaussian given x, with a mean, scale and correlation known in closed form."""
    x = torch.rand(n, 1, generator=gen) * 4 - 2
    e = torch.randn(n, 2, generator=gen)
    s = 0.3 + 0.2 * x[:, 0].abs()
    y1 = torch.sin(x[:, 0]) + s * e[:, 0]
    y2 = 0.5 * x[:, 0] + s * (0.6 * e[:, 0] + 0.8 * e[:, 1])
    return x, torch.stack([y1, y2], 1)


def _true_log_density(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    s = 0.3 + 0.2 * x[:, 0].abs()
    mean = torch.stack([torch.sin(x[:, 0]), 0.5 * x[:, 0]], 1)
    corr = torch.tensor([[1.0, 0.6], [0.6, 1.0]], dtype=torch.float32)
    return torch.distributions.MultivariateNormal(mean, s[:, None, None] ** 2 * corr).log_prob(y)


def _three_features(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, generator=gen)
    e = torch.randn(n, 2, generator=gen)
    y = torch.stack([x[:, 0] + 0.3 * e[:, 0], x[:, 0] * x[:, 1] + 0.5 * e[:, 1]], 1)
    return x, y


# ── cv=None is today's model ───────────────────────────────────────────────────


# Recorded on a clean checkout of 3c11d77 (before cross-fitting), same data and seed.
_RECORDED = {
    ("gaussian", False): [
        -2.0643906593322754,
        -2.3229081630706787,
        -2.231020450592041,
        -3.1147398948669434,
        -4.709880828857422,
        -2.126042604446411,
    ],
    ("gaussian", True): [
        -2.1055209636688232,
        -2.3167917728424072,
        -2.2689225673675537,
        -3.1756739616394043,
        -4.651389122009277,
        -2.1677470207214355,
    ],
    ("mixture", False): [
        -1.4592998027801514,
        -1.7823817729949951,
        -1.4429843425750732,
        -3.4035274982452393,
        -8.345921516418457,
        -1.4684250354766846,
    ],
}


@pytest.mark.parametrize(("method", "missing"), list(_RECORDED))
def test_cv_none_reproduces_the_recorded_fit(method: str, missing: bool) -> None:
    torch.manual_seed(0)
    n = 200
    x = torch.randn(n, 3)
    y = torch.stack([x[:, 0] + 0.3 * torch.randn(n), x[:, 0] * x[:, 1] + 0.5 * torch.randn(n)], 1)
    x_test = torch.randn(40, 3)
    y_test = torch.stack(
        [
            x_test[:, 0] + 0.3 * torch.randn(40),
            x_test[:, 0] * x_test[:, 1] + 0.5 * torch.randn(40),
        ],
        1,
    )
    if missing:
        y[::7, 1] = float("nan")
    model = JointRegressor(method, max_epochs=15, patience=5, hidden=(16,), seed=3).fit(x, y)
    assert model.cv is None and model.folds_ is None
    lp = model.predict(x_test).log_prob(y_test)[:6]
    recorded = torch.tensor(_RECORDED[(method, missing)])
    # The values were recorded on macOS arm64, where the fit is reproduced to the last
    # bit.  The same training on Linux x86-64 ends up to 6e-3 away (rounding differences
    # accumulated over the epochs, measured in the CANFAR gate), so the comparison allows
    # 2e-2.  The control below shows that this still tells fits apart: another seed (other
    # splits and initial weights) is more than five times further away (0.14 to 3.8 here).
    tolerance = 2e-2
    torch.testing.assert_close(lp, recorded, rtol=0, atol=tolerance)
    other = JointRegressor(method, max_epochs=15, patience=5, hidden=(16,), seed=4).fit(x, y)
    gap = (other.predict(x_test).log_prob(y_test)[:6] - recorded).abs().max()
    assert float(gap) > 5 * tolerance, float(gap)


# ── folds ──────────────────────────────────────────────────────────────────────


def test_folds_partition_the_rows_and_a_fold_model_never_sees_its_fold(monkeypatch) -> None:
    x, y = _three_features(150, seed=1)
    given: list[torch.Tensor] = []
    original = JointRegressor._fit

    def spy(self, X, Y):  # type: ignore[no-untyped-def]
        if not self._holdout:  # a fold model
            given.append(torch.as_tensor(X).clone())
        return original(self, X, Y)

    monkeypatch.setattr(JointRegressor, "_fit", spy)
    model = JointRegressor("gaussian", cv=4, seed=0, **TINY).fit(x, y)
    folds = model.folds_
    assert folds is not None and folds.shape == (150,)
    counts = torch.bincount(folds, minlength=4)
    assert counts.sum() == 150 and int(counts.max() - counts.min()) <= 1
    assert len(given) == 4
    for k, rows in enumerate(given):
        # Features are continuous random values, so a row is identified by its features.
        seen = (rows[:, None, :] == x[None, :, :]).all(-1).any(0)  # [N]
        assert bool((seen == (folds != k)).all()), f"fold model {k} was given the wrong rows"
        assert rows.shape[0] == int((folds != k).sum())
    assert model.n_recal_ == model.n_conformal_ == 150
    assert model.n_train_ + model.n_val_ == min(int((folds != k).sum()) for k in range(4))


# ── coverage of the cross-conformal region ────────────────────────────────────


def test_cross_conformal_coverage_and_its_spread_against_one_split() -> None:
    """200 data sets of N = 120 (d = 2), K = 5, alpha = 0.2, 200 fresh test points each.

    Measured (seed-fixed): cross-conformal mean coverage 0.807 (one fresh point per
    data set: 0.815 +- 0.027), sd across data sets 0.046; single split (12 conformal
    rows) 0.846 (one point: 0.855 +- 0.025), sd 0.107.  The binomial sd of 200 test
    points alone is 0.028.  Proven bound for the cross-conformal set: 0.536.
    """
    n_sets, n, k_folds, alpha = 200, 120, 5, 0.2
    budget = dict(max_epochs=20, patience=5, hidden=(8,), device="cpu")
    per_set: dict[str, list[float]] = {"cv": [], "split": []}
    one_point: dict[str, list[float]] = {"cv": [], "split": []}
    for j in range(n_sets):
        gen = torch.Generator().manual_seed(1000 + j)
        x, y = _data(n, gen)
        x_test, y_test = _data(200, gen)
        for name, cv in (("cv", k_folds), ("split", None)):
            model = JointRegressor("gaussian", cv=cv, seed=j, **budget).fit(x, y)
            inside = model.region(x_test, alpha=alpha).contains(y_test).float()
            per_set[name].append(float(inside.mean()))
            one_point[name].append(float(inside[0]))
    bound = (
        1
        - 2 * alpha
        - min(2 * (1 - 1 / k_folds) / (n / k_folds + 1), (1 - k_folds / n) / (k_folds + 1))
    )
    assert bound == pytest.approx(0.536, abs=1e-3)
    cv_cov, split_cov = torch.tensor(per_set["cv"]), torch.tensor(per_set["split"])
    cv_one = torch.tensor(one_point["cv"])
    se_one = math.sqrt(0.2 * 0.8 / n_sets)
    # The theorem: one fresh point per data set, coverage at least the bound.
    assert float(cv_one.mean()) >= bound - 3 * se_one
    # Observed, not proved: close to 1 - alpha (mean over 40,000 test points).
    assert abs(float(cv_cov.mean()) - (1 - alpha)) < 0.03
    # Split conformal: at least 1 - alpha (12 conformal rows: exactly 11/13 = 0.846).
    se_split = float(split_cov.std()) / math.sqrt(n_sets)
    assert float(split_cov.mean()) >= 1 - alpha - 3 * se_split
    # The point of the feature: much less variation from one data set to the next.
    assert float(cv_cov.std()) < 0.6 * float(split_cov.std())
    assert float(cv_cov.std()) < 0.06


def test_cross_conformal_region_matches_its_p_value_definition() -> None:
    x, y = _three_features(120, seed=2)
    model = JointRegressor("gaussian", cv=3, seed=0, **TINY).fit(x, y)
    x_test, y_test = _three_features(30, seed=3)
    alpha = 0.25
    region = model.region(x_test, alpha=alpha)
    assert isinstance(region, CrossConformalRegion) and isinstance(region, JointRegion)
    assert region.score.name == "cross-latent"
    # Recompute the p-value by brute force from the out-of-fold scores.
    folds = model.folds_
    assert folds is not None
    bases = region.score.bases  # type: ignore[attr-defined]
    p_count = torch.ones(30)
    for k, base in enumerate(bases):
        rows = (folds == k).nonzero().squeeze(1)
        cal = model._fold_base(k, x[rows], own=True)
        s = base.score(cal, y[rows])
        test = model._fold_base(k, x_test, own=True)
        t = base.score(test, y_test)
        p_count += (s[None, :] >= t[:, None]).sum(1).float()
    p_value = p_count / (120 + 1)
    assert torch.equal(region.contains(y_test), p_value > alpha)


# ── density quality on small data ─────────────────────────────────────────────


def test_density_gap_to_the_truth_on_small_data() -> None:
    """Mean test log-score gap to the true density, cv = 5 against one split, paired.

    Gaussian head, 60 epochs, 4 seeds (measured, nats): N = 150 gap 0.280 +- 0.039
    (cv) against 0.494 +- 0.135 (split), paired difference 0.21 +- 0.15; N = 400
    0.137 +- 0.008 against 0.183 +- 0.030, paired 0.047 +- 0.036.  The difference
    is positive on average at both sizes but within about 1.5 standard errors of
    zero; the full-budget ``JointRegressor()`` comparison is in the docs.
    """
    budget = dict(max_epochs=60, patience=10, hidden=(32, 32), device="cpu")
    for n, seeds in ((150, 4), (400, 4)):
        diffs = []
        for seed in range(seeds):
            gen = torch.Generator().manual_seed(77 + seed)
            x, y = _data(n, gen)
            x_test, y_test = _data(2000, gen)
            truth = _true_log_density(x_test, y_test)
            gaps = {}
            for name, cv in (("cv", 5), ("split", None)):
                model = JointRegressor("gaussian", cv=cv, seed=seed, **budget).fit(x, y)
                gaps[name] = float((truth - model.predict(x_test).log_prob(y_test)).mean())
            assert gaps["cv"] > -0.05  # the truth is the best density
            diffs.append(gaps["split"] - gaps["cv"])
        d = torch.tensor(diffs)
        # cv = 5 is closer to the truth on these small data sets (paired over seeds).
        assert float(d.mean()) > 0, f"N = {n}: split - cv gap {d.tolist()}"


# ── surface ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("method", ["gaussian", "mixture", "flow", "sklar", "energy", "auto"])
def test_every_method_under_cv(method: str) -> None:
    x, y = _three_features(120, seed=4)
    model = JointRegressor(method, cv=3, seed=0, **TINY).fit(x, y)
    pred = model.predict(x[:9])
    assert isinstance(pred, MixtureJoint)
    assert (pred.batch_size, pred.event_dim) == (9, 2)
    assert model.members_ == model._fold_models[0].members_
    assert all(m.n_members in (None, 1) for m in model._fold_models)
    if method == "energy":
        assert not pred.has_density
        assert set(model.validation_energy_score_) == {"energy"}
        assert model.validation_log_score_ == {}
    else:
        lp = pred.log_prob(y[:9])
        assert bool(torch.isfinite(lp).all())
        assert set(model.validation_log_score_) == set(model.members_)
        assert math.isclose(float(model.weights_.sum()), 1.0, rel_tol=1e-5)
    if method == "sklar":
        assert len(model.fold_choices_["dependence"]) == 3
        assert model.dependence_ in model.fold_choices_["dependence"]
        assert model.marginal_engine_ in model.fold_choices_["marginal_engine"]
    region = model.region(x[:5], alpha=0.2)
    assert region.contains(y[:5]).shape == (5,)
    names = {b.name for b in region.score.bases}  # type: ignore[attr-defined]
    if method == "auto":  # per fold: a fold whose own stack is one Gaussian gets "latent"
        assert names <= {"hpd", "latent"}
    else:
        expected = {"energy": "sample_ball", "gaussian": "latent"}.get(method, "hpd")
        assert names == {expected}
    log_volume, _ = region.log_volume(256)
    assert bool(torch.isfinite(log_volume).all())
    # Every fold model's parameters are counted.
    assert model.n_parameters_ == sum(m.n_parameters_ for m in model._fold_models)
    assert model.n_parameters_ > 0


def test_n_members_is_one_per_fold_unless_given() -> None:
    x, y = _three_features(100, seed=5)
    one = JointRegressor("gaussian", cv=2, seed=0, **TINY).fit(x, y)
    two = JointRegressor("gaussian", cv=2, n_members=2, seed=0, **TINY).fit(x, y)
    assert [len(m._heads["gaussian"]) for m in one._fold_models] == [1, 1]
    assert [len(m._heads["gaussian"]) for m in two._fold_models] == [2, 2]
    assert two.n_parameters_ == 2 * one.n_parameters_
    assert len(two.predict(x[:3]).members) == 4  # type: ignore[attr-defined]


def test_regions_levels_interest_and_candidate_draws() -> None:
    x, y = _three_features(150, seed=6)
    model = JointRegressor("mixture", cv=3, seed=0, **TINY).fit(x, y)
    x_test, y_test = _three_features(40, seed=7)
    region = model.region(x_test, alpha=[0.3, 0.1])
    inside = region.contains(y_test)
    assert inside.shape == (40, 2)
    assert bool((inside[:, 0] <= inside[:, 1]).all())  # nested levels
    cand = y_test[:, None, :] + 0.5 * torch.randn(
        40, 6, 2, generator=torch.Generator().manual_seed(0)
    )
    assert region.contains(cand).shape == (40, 6, 2)
    lo, hi = region.bounding_box(256)
    assert lo.shape == hi.shape == (40, 2, 2)
    draws = region.sample(16, level=1)
    assert draws.shape == (40, 16, 2)
    assert bool(region.contains(draws[:, :, :])[..., 1][torch.isfinite(draws).all(-1)].all())
    lv, se = region.log_volume(512)
    assert lv.shape == (40, 2) and bool((lv[:, 0] <= lv[:, 1] + 3 * (se[:, 0] + se[:, 1])).all())
    # One target of interest: the other is integrated out before calibration.
    sub = model.region(x_test, alpha=0.2, interest=[1])
    assert sub.predictive.event_dim == 1
    assert sub.contains(y_test[:, [1]]).shape == (40,)
    # The out-of-fold scores are cached per (interest, score).
    assert len(model._cross_cache) == 2


def test_missing_targets_masked_and_complete_row_regions() -> None:
    x, y = _three_features(200, seed=8)
    y_holes = y.clone()
    y_holes[::4, 0] = float("nan")
    model = JointRegressor("gaussian", cv=4, seed=0, **TINY).fit(x, y_holes)
    assert model.members_ == ["gaussian"]
    x_test, y_test = _three_features(60, seed=9)
    region = model.region(x_test, alpha=0.2)
    assert region.mask_region is not None
    assert isinstance(region.mask_region, CrossConformalRegion)
    assert region.score.name == "cross-latent"
    assert region.mask_region.score.name == "cross-mahalanobis"
    # Complete-row region from the 150 complete rows, masked region from all 200.
    assert region.n_calibration_ == 150
    assert region.mask_region.n_calibration_ == 200
    mask = torch.ones(60, 2, dtype=torch.bool)
    mask[::3, 0] = False
    assert region.contains(y_test).shape == (60,)
    assert region.contains(y_test, mask=mask).shape == (60,)
    none_observed = torch.zeros(60, 2, dtype=torch.bool)
    assert bool(region.contains(y_test, mask=none_observed).all())
    # The flow head needs complete rows: dropped with a warning, as without cv.
    with pytest.warns(UserWarning, match="complete rows"):
        flow = JointRegressor("flow", cv=2, seed=0, **TINY).fit(x, y_holes)
    assert bool(torch.isfinite(flow.predict(x_test).log_prob(y_test)).all())


def test_calibrate_replaces_the_cross_conformal_region() -> None:
    x, y = _three_features(150, seed=10)
    model = JointRegressor("gaussian", cv=3, seed=0, **TINY).fit(x, y)
    x_cal, y_cal = _three_features(80, seed=11)
    x_test, y_test = _three_features(20, seed=12)
    assert isinstance(model.region(x_test, alpha=0.2), CrossConformalRegion)
    model.calibrate(x_cal, y_cal)
    region = model.region(x_test, alpha=0.2)
    assert type(region) is JointRegion  # split conformal on the averaged predictive
    assert model.n_conformal_ == 80
    assert region.predictive.batch_size == 20
    assert region.contains(y_test).shape == (20,)


def test_numpy_inputs_and_determinism() -> None:
    x, y = _three_features(120, seed=13)
    a = JointRegressor("mixture", cv=3, seed=5, **TINY).fit(x.numpy(), y.numpy())
    b = JointRegressor("mixture", cv=3, seed=5, **TINY).fit(x, y)
    c = JointRegressor("mixture", cv=3, seed=6, **TINY).fit(x, y)
    x_test = np.asarray(x[:10])
    lp_a = a.predict(x_test).log_prob(y[:10])
    lp_b = b.predict(x_test).log_prob(y[:10])
    assert torch.equal(a.folds_, b.folds_)
    assert torch.equal(lp_a, lp_b)
    assert not torch.equal(lp_a, c.predict(x_test).log_prob(y[:10]))
    torch.manual_seed(0)
    r_a = a.region(x_test, alpha=0.2).contains(y[:10])
    torch.manual_seed(0)
    r_b = b.region(x_test, alpha=0.2).contains(y[:10])
    assert torch.equal(r_a, r_b)


def test_recalibration_is_shared_and_reported() -> None:
    x, y = _three_features(150, seed=14)
    model = JointRegressor("mixture", cv=3, seed=0, **TINY).fit(x, y)
    assert model.scale_ is not None and model.scale_.shape == (2,)
    # The out-of-fold fit moved the factors more than 2% for this data and seed.
    assert model._recalibrator is not None
    assert float(model.scale_.log().abs().max()) > 0.02
    base = [model._fold_base(k, x[:4]) for k in range(3)]
    folds = model._fold_predictives(x[:4])
    for b, f in zip(base, folds):
        # Each fold scaled about its own center by the same factors:
        # p'(y) = p(m + (y - m) / s) / prod(s).
        center = b.center()
        s = model.scale_
        expected = b.log_prob(center + (y[:4] - center) / s) - s.log().sum()
        torch.testing.assert_close(f.log_prob(y[:4]), expected, rtol=1e-5, atol=1e-5)
        assert isinstance(f, JointPredictive)
    # predict is the equal-weight average of the recalibrated fold predictives.
    avg = sum(f.log_prob(y[:4]).exp() for f in folds) / 3
    torch.testing.assert_close(model.predict(x[:4]).log_prob(y[:4]).exp(), avg)


def test_plots_accept_the_cross_conformal_region() -> None:
    x, y = _three_features(150, seed=15)
    model = JointRegressor("gaussian", cv=3, seed=0, **TINY).fit(x, y)
    x_test, y_test = _three_features(30, seed=16)
    pred = model.predict(x_test)
    region = model.region(x_test, alpha=[0.5, 0.1])
    fig = plot_corner_plot(pred, row=0, region=region, n_samples=400, return_figure=True)
    assert fig is not None
    fig, axes = plot_joint_diagnostics(
        pred, y_test, x=x_test, region=region, n_samples=64, return_axes=True
    )
    assert "coverage" in axes and "volume" in axes


def test_constructor_and_fit_validation() -> None:
    for bad in (1, 0, -3, 2.5, True):
        with pytest.raises(ValueError, match="cv must be"):
            JointRegressor(cv=bad)  # type: ignore[arg-type]
    x, y = _three_features(50, seed=17)
    with pytest.raises(ValueError, match="exceeds the 50 rows"):
        JointRegressor("gaussian", cv=51, **TINY).fit(x, y)
    with pytest.raises(ValueError, match="at least 50 rows"):
        JointRegressor("gaussian", cv=2, **TINY).fit(x[:40], y[:40])
