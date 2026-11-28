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


class TestScalingDefaults:
    """Large-N settings: capped hold-out splits and the automatic batch size."""

    @staticmethod
    def _data(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        gen = torch.Generator().manual_seed(7)
        x = torch.randn(n, 3, generator=gen)
        y = torch.stack([x[:, 0], x[:, 0] - x[:, 1]], dim=1) + 0.3 * torch.randn(
            n, 2, generator=gen
        )
        return x, y

    def test_holdout_splits_are_capped(self) -> None:
        x, y = self._data(4000)
        model = JointRegressor("gaussian", max_epochs=2, max_holdout=150, device="cpu").fit(x, y)
        assert (model.n_val_, model.n_recal_, model.n_conformal_) == (150, 150, 150)
        assert model.n_train_ == 4000 - 450
        # the conformal guarantee does not depend on the cap
        xt, yt = self._data(3000)
        cover = model.region(xt, alpha=0.2).contains(yt).float().mean()
        se = (0.2 * 0.8 / 3000 + 0.2 * 0.8 / 151) ** 0.5
        assert abs(float(cover) - 0.8) < 3.5 * se + 1 / 151

    def test_default_cap_does_not_bind_on_small_data(self) -> None:
        x, y = self._data(400)
        model = JointRegressor("gaussian", max_epochs=1, device="cpu").fit(x, y)
        assert (model.n_val_, model.n_recal_, model.n_conformal_) == (40, 20, 40)
        assert model.batch_size_ == 32  # 300 training rows: 10 steps per epoch

    @pytest.mark.parametrize(
        ("n_train", "expected"),
        [
            (40, 16),
            (127, 16),
            (128, 16),
            (255, 16),
            (256, 32),
            (308, 32),
            (1000, 64),
            (2047, 128),
            (2048, 256),
            (102_400, 256),
            (102_401, 512),
            (245_000, 1024),
            (5_000_000, 4096),
        ],
    )
    def test_auto_batch_size_keeps_an_epoch_between_8_and_400_steps(
        self, n_train: int, expected: int
    ) -> None:
        from torchregress.joint.estimator import _auto_batch_size

        batch = _auto_batch_size(n_train)
        assert batch == expected
        steps = n_train / batch
        assert steps >= 8 or batch == 16
        assert steps <= 400 or batch == 4096

    def test_auto_batch_size_grows_with_the_training_set(self) -> None:
        x, y = self._data(300_000)
        model = JointRegressor("gaussian", max_epochs=1, hidden=(16,), device="cpu").fit(x, y)
        # 10%, 5% and 10% of 300,000 are 30,000, 15,000 and 30,000: two splits hit the cap
        assert (model.n_val_, model.n_recal_, model.n_conformal_) == (20_000, 15_000, 20_000)
        assert model.n_train_ == 245_000
        assert model.batch_size_ == 1024  # 245,000 / 1024 = 240 steps, 245,000 / 512 > 400
        fixed = JointRegressor("gaussian", max_epochs=1, batch_size=64, device="cpu")
        assert fixed.fit(*self._data(400)).batch_size_ == 64

    def test_rejects_bad_settings(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            JointRegressor(batch_size="big")
        with pytest.raises(ValueError, match="batch_size"):
            JointRegressor(batch_size=0)
        with pytest.raises(ValueError, match="max_holdout"):
            JointRegressor(max_holdout=4)

    def test_flow_predict_needs_no_draws(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Scoring a recalibrated flow predictive must not sample (the center is exact)."""
        from torchregress.joint.predictive import FlowJoint

        x, y = self._data(1500)
        y = y + 0.5 * torch.sin(3 * x[:, :2])
        model = JointRegressor("flow", max_epochs=4, device="cpu").fit(x, y)
        calls = {"n": 0}
        original = FlowJoint.sample

        def counting(self: FlowJoint, n_samples: int) -> torch.Tensor:
            calls["n"] += 1
            return original(self, n_samples)

        monkeypatch.setattr(FlowJoint, "sample", counting)
        pred = model.predict(x[:50])
        assert torch.isfinite(pred.log_prob(y[:50])).all()
        assert calls["n"] == 0


class TestEnsembleCopies:
    """``n_members``: deep-ensemble copies of each network head."""

    @staticmethod
    def _data(n: int = 600) -> tuple[torch.Tensor, torch.Tensor]:
        gen = torch.Generator().manual_seed(11)
        x = torch.randn(n, 3, generator=gen)
        sign = torch.where(torch.rand(n, generator=gen) < 0.5, 1.0, -1.0)
        y = torch.stack([x[:, 0] + sign, x[:, 1] - sign], dim=1) + 0.3 * torch.randn(
            n, 2, generator=gen
        )
        return x, y

    def test_single_head_default_is_one_copy(self) -> None:
        from torchregress.joint import GaussianJoint

        x, y = self._data()
        model = JointRegressor("gaussian", max_epochs=3, device="cpu").fit(x, y)
        assert len(model._heads["gaussian"]) == 1
        assert isinstance(model.predict(x[:5]), GaussianJoint)

    def test_copies_are_averaged_with_equal_weights(self) -> None:
        from torchregress.joint import MixtureJoint

        x, y = self._data()
        model = JointRegressor(
            "mixture", n_members=3, max_epochs=6, recalibrate=False, device="cpu"
        ).fit(x, y)
        heads = model._heads["mixture"]
        assert len(heads) == 3
        # different initialisations: the copies are not the same network
        first = [p.detach().clone() for p in heads[0].parameters()]
        assert any(not torch.equal(a, b) for a, b in zip(first, heads[1].parameters()))
        pred = model.predict(x[:40])
        inner = pred.base  # type: ignore[attr-defined]
        assert isinstance(inner, MixtureJoint) and len(inner.members) == 3
        torch.testing.assert_close(inner.weights, torch.full((3,), 1 / 3))
        # density of the ensemble = mean of the copies' densities
        ys = (y[:40] - model._y_stats[0]) / model._y_stats[1]  # type: ignore[index]
        xs = model._standardised(x[:40])
        parts = torch.stack([h.predictive(xs).log_prob(ys) for h in heads])
        expected = torch.logsumexp(parts, 0) - math.log(3.0)
        jac = model._y_stats[1].log().sum()  # type: ignore[index]
        torch.testing.assert_close(pred.log_prob(y[:40]), expected - jac, rtol=1e-4, atol=1e-4)
        assert model.weights_ is not None and model.weights_.tolist() == [1.0]

    def test_auto_ensembles_the_network_heads_and_stacks_across_types(self) -> None:
        from torchregress.joint import MixtureJoint

        x, y = self._data()
        model = JointRegressor(max_epochs=3, device="cpu").fit(x, y)
        assert {k: len(v) for k, v in model._heads.items()} == {
            "gaussian": 3,
            "mixture": 3,
            "flow": 3,
        }
        assert model.members_ == ["gaussian", "mixture", "sklar", "flow"]
        assert model.weights_ is not None and model.weights_.shape == (4,)
        pred = model.predict(x[:20])
        inner = pred
        while hasattr(inner, "base"):  # unit change, and the scale recalibration when kept
            inner = inner.base
        assert isinstance(inner, MixtureJoint)
        assert not any(isinstance(m, MixtureJoint) for m in inner.members)  # one flat mixture
        torch.testing.assert_close(inner.weights.sum(), torch.tensor(1.0))
        assert torch.isfinite(pred.log_prob(y[:20])).all()
        assert pred.sample(7).shape == (20, 7, 2)
        one = JointRegressor(max_epochs=3, n_members=1, device="cpu").fit(x, y)
        assert all(len(v) == 1 for v in one._heads.values())

    def test_ensemble_handles_missing_targets_and_is_deterministic(self) -> None:
        x, y = self._data()
        y = y.clone()
        y[::7, 1] = float("nan")
        a = JointRegressor("gaussian", n_members=2, max_epochs=4, device="cpu").fit(x, y)
        b = JointRegressor("gaussian", n_members=2, max_epochs=4, device="cpu").fit(x, y)
        mask = torch.isfinite(y[:30])
        filled = torch.nan_to_num(y[:30])
        la = a.predict(x[:30]).log_prob(filled, mask)
        torch.testing.assert_close(la, b.predict(x[:30]).log_prob(filled, mask))
        assert torch.isfinite(la).all()
        region = a.region(x[:30], alpha=0.2)
        assert region.contains(filled, mask=mask).shape == (30,)

    def test_rejects_bad_n_members(self) -> None:
        with pytest.raises(ValueError, match="n_members"):
            JointRegressor(n_members=0)


def test_n_parameters_counts_every_copy_and_the_sklar_networks() -> None:
    gen = torch.Generator().manual_seed(2)
    x = torch.randn(300, 3, generator=gen)
    y = torch.stack([x[:, 0], x[:, 0] - x[:, 1]], dim=1) + 0.3 * torch.randn(300, 2, generator=gen)
    one = JointRegressor("gaussian", max_epochs=1, hidden=(8,), device="cpu").fit(x, y)
    # (3 -> 8) + (8 -> 2 means + 3 Cholesky entries)
    assert one.n_parameters_ == (3 * 8 + 8) + (8 * 5 + 5)
    three = JointRegressor("gaussian", n_members=3, max_epochs=1, hidden=(8,), device="cpu")
    assert three.fit(x, y).n_parameters_ == 3 * one.n_parameters_
    sklar = JointRegressor("sklar", max_epochs=1, hidden=(8,), device="cpu").fit(x, y)
    assert sklar.n_parameters_ > 0
    with pytest.raises(RuntimeError, match="fit"):
        _ = JointRegressor().n_parameters_


class TestFeatureClipping:
    """Features are clamped to the training range before they enter the networks."""

    @staticmethod
    def _data(n: int = 800) -> tuple[torch.Tensor, torch.Tensor]:
        gen = torch.Generator().manual_seed(21)
        x = torch.randn(n, 3, generator=gen)
        y = torch.stack([x[:, 0], x[:, 0] - x[:, 1]], dim=1) + 0.2 * torch.randn(
            n, 2, generator=gen
        )
        return x, y

    def test_a_glitched_feature_does_not_blow_up_the_density(self) -> None:
        x, y = self._data()
        model = JointRegressor("gaussian", max_epochs=40, device="cpu").fit(x, y)
        glitch = x[:5].clone()
        glitch[:, 2] = 500.0  # hundreds of standard deviations outside the training range
        at_edge = x[:5].clone()
        at_edge[:, 2] = x[:, 2].max()
        # the model's training rows are a subset of x, so the clamp is at most that maximum
        lp_glitch = model.predict(glitch).log_prob(y[:5])
        assert torch.isfinite(lp_glitch).all()
        clamped = glitch.clone()
        assert model._x_range is not None and model._x_stats is not None
        top = model._x_range[1][2] * model._x_stats[1][2] + model._x_stats[0][2]
        clamped[:, 2] = top
        torch.testing.assert_close(lp_glitch, model.predict(clamped).log_prob(y[:5]))
        assert float(top) <= float(x[:, 2].max()) + 1e-5
        # without the clamp the same rows are scored somewhere else entirely
        free = JointRegressor("gaussian", max_epochs=40, clip_features=False, device="cpu")
        lp_free = free.fit(x, y).predict(glitch).log_prob(y[:5])
        assert not torch.allclose(lp_free, lp_glitch)

    def test_rows_inside_the_range_are_untouched(self) -> None:
        x, y = self._data()
        model = JointRegressor("gaussian", max_epochs=5, device="cpu").fit(x, y)
        assert model._x_range is not None and model._x_stats is not None
        xs = (x - model._x_stats[0]) / model._x_stats[1]
        inside = ((xs >= model._x_range[0]) & (xs <= model._x_range[1])).all(1)
        assert int(inside.sum()) >= model.n_train_  # every training row, by construction
        clamped = model.predict(x[inside][:50]).log_prob(y[inside][:50])
        model.clip_features = False  # the same fitted networks, no clamp at predict time
        torch.testing.assert_close(clamped, model.predict(x[inside][:50]).log_prob(y[inside][:50]))


class TestAutoHeads:
    """Which heads ``method="auto"`` trains."""

    @staticmethod
    def _fit(d: int, missing: bool = False) -> JointRegressor:
        gen = torch.Generator().manual_seed(d)
        x = torch.randn(160, 3, generator=gen)
        y = x @ torch.randn(3, d, generator=gen) + 0.5 * torch.randn(160, d, generator=gen)
        if missing:
            y[::7, 0] = float("nan")
        model = JointRegressor(hidden=(8,), max_epochs=2, patience=1, n_members=1, seed=0)
        return model.fit(x, y)

    def test_flow_head_is_trained_up_to_32_targets(self) -> None:
        assert self._fit(32).members_ == ["gaussian", "mixture", "sklar", "flow"]

    def test_no_flow_head_beyond_32_targets(self) -> None:
        model = self._fit(33)
        assert model.members_ == ["gaussian", "mixture", "sklar"]
        assert model.predict(torch.zeros(2, 3)).sample(4).shape == (2, 4, 33)

    def test_missing_targets_leave_the_complete_row_heads_out(self) -> None:
        assert self._fit(4, missing=True).members_ == ["gaussian", "mixture"]


class TestComponentSpreadBound:
    """Mixture components are no wider than the range of the training targets."""

    @staticmethod
    def _data(n: int = 300, d: int = 3, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        gen = torch.Generator().manual_seed(seed)
        x = torch.randn(n, 4, generator=gen)
        y = x[:, :d] + 0.3 * torch.randn(n, d, generator=gen)
        return x, y

    @pytest.mark.parametrize(("d", "seed"), [(3, 0), (8, 0), (8, 1), (8, 2), (12, 0)])
    def test_mixture_head_draws_stay_in_range_far_outside_the_training_features(
        self, d: int, seed: int
    ) -> None:
        # d <= 8: full covariances in the precision form (a nearly singular factor, whose
        # inverse overflows float32 at eight targets); d = 12: diagonal components.
        gen = torch.Generator().manual_seed(seed)
        x = torch.randn(400, 4, generator=gen)
        y = x @ torch.randn(4, d, generator=gen) + 0.3 * torch.randn(400, d, generator=gen)
        model = JointRegressor(
            "mixture", hidden=(16,), max_epochs=30, patience=5, clip_features=False, seed=seed
        ).fit(x, y)
        far = 1e4 * torch.ones(8, 4)  # the network extrapolates linearly to absurd outputs
        torch.manual_seed(0)
        draws = model.predict(far).sample(2000)
        reach = ((draws - y.mean(0)) / y.std(0)).abs().amax()
        head = model._heads["mixture"][0]
        assert 5.0 <= head.bound <= 25.0
        # Means within the bound B, total standard deviation at most B sqrt(d): a draw
        # beyond B + 8 B sqrt(d) has probability about 1e-15 (the recalibration factors
        # stay within a few percent of one and are covered by the factor 1.5).
        assert bool(torch.isfinite(draws).all())
        assert float(reach) < 1.5 * head.bound * (1 + 8 * math.sqrt(d))

    def test_engines_saved_before_the_bound_existed_still_predict(self) -> None:
        from torchregress.joint import MixtureNetworkEngine

        x, y = self._data(d=2)
        engine = MixtureNetworkEngine(hidden=(8,), max_epochs=3, patience=1, seed=0).fit(x, y)
        del engine.__dict__["_mean_range"], engine._mdn.__dict__["max_std"]
        assert bool(torch.isfinite(engine.predict(x[:4]).scale).all())

    def test_mixture_marginal_engine_bounds_each_target_by_its_own_range(self) -> None:
        from torchregress.joint import MixtureNetworkEngine

        x, y = self._data(d=2)
        y = y * torch.tensor([1.0, 50.0])  # two targets on different scales
        engine = MixtureNetworkEngine(hidden=(16,), max_epochs=20, patience=5, seed=0).fit(x, y)
        bound = 1.25 * (y - y.mean(0)).abs().amax(0)
        far = engine.predict(1e4 * torch.ones(8, 4))
        scales = far.scale  # [n, d, K]
        assert bool((scales.amax(dim=(0, 2)) <= bound * (1 + 1e-6)).all())
        reach = (far.loc - y.mean(0).view(1, 2, 1)).abs().amax(dim=(0, 2))
        assert bool((reach <= bound * (1 + 1e-6)).all())
        assert float(scales[:, 1].amax()) > float(bound[0])  # each target has its own bound
        inside = engine.predict(x[:16]).scale
        assert bool((inside.amax(dim=(0, 2)) < bound).all())
