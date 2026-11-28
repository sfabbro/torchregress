"""``JointRegressor(location="mse")``: a cross-fitted point model, the heads on its residuals."""

from __future__ import annotations

import pytest
import torch

from torchregress.joint import JointRegressor
from torchregress.joint.estimator import _LOCATION_FOLDS

BUDGET = dict(max_epochs=60, patience=10, hidden=(32, 32), device="cpu")


def _data(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Three targets with a nonlinear mean and noise that grows with ``|x0|``."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 5, generator=gen)
    mean = torch.stack(
        [torch.sin(2 * x[:, 0]) + x[:, 1] ** 2, x[:, 2] * x[:, 3], 2 * torch.tanh(x[:, 4])], dim=1
    )
    noise = torch.randn(n, 3, generator=gen) * (0.2 + 0.2 * x[:, :1].abs())
    noise[:, 1] += 0.7 * noise[:, 0]
    return x, mean + noise, mean


def test_rejects_unknown_location() -> None:
    with pytest.raises(ValueError, match="location"):
        JointRegressor(location="median")


def test_training_rows_get_out_of_fold_predictions_and_other_rows_the_average() -> None:
    x, y, _ = _data(400, seed=0)
    model = JointRegressor("gaussian", location="mse", seed=0, **BUDGET)
    model.batch_size_, model.n_train_ = 32, 300
    xs = (x - x.mean(0)) / x.std(0)
    ys = (y - y.mean(0)) / y.std(0)
    mask = torch.ones_like(ys, dtype=torch.bool)
    train = torch.arange(300)
    targets, spread = model._fit_location(xs, ys, mask, train)
    assert model._location is not None and len(model._location) == _LOCATION_FOLDS
    with torch.no_grad():
        nets = torch.stack([net(xs) for net in model._location])  # [K, n, d]
    average = nets.mean(0)
    held_out = torch.arange(300, 400)
    # rows outside the training split: the residual of the average of the K networks
    torch.testing.assert_close(
        targets[held_out] * spread, (ys - average)[held_out], atol=1e-6, rtol=0
    )
    # training rows: the residual of exactly one network, the one that did not see the row
    residual = targets[train] * spread
    per_net = ys[train].unsqueeze(0) - nets[:, train]  # [K, 300, d]
    match = (per_net - residual.unsqueeze(0)).abs().amax(-1) < 1e-6  # [K, 300]
    assert bool((match.sum(0) == 1).all())
    counts = match.sum(1)
    assert int(counts.max() - counts.min()) <= 1 and int(counts.sum()) == 300
    # and that network fits the row worse than the others do on average: it is out of fold
    own = (per_net.square().sum(-1) * match).sum(0)
    rest = (per_net.square().sum(-1) * ~match).sum(0) / (_LOCATION_FOLDS - 1)
    assert float(own.mean()) > 1.1 * float(rest.mean())
    # the returned scale is the spread of those out-of-fold residuals
    torch.testing.assert_close(spread, residual.square().mean(0).sqrt(), rtol=1e-5, atol=0)


def test_predict_is_the_point_model_plus_the_residual_predictive() -> None:
    x, y, _ = _data(500, seed=1)
    model = JointRegressor("gaussian", location="mse", recalibrate=False, seed=0, **BUDGET).fit(
        x, y
    )
    x_new, y_new, _ = _data(40, seed=2)
    pred = model.predict(x_new)
    xs = model._standardised(x_new)
    mu, sd = model._y_stats
    head = model._heads["gaussian"][0].predictive(xs)
    # density: the head's density of the residual targets, with the Jacobian of the scale
    targets = model._head_targets(xs, y_new, torch.ones_like(y_new, dtype=torch.bool))
    torch.testing.assert_close(
        pred.log_prob(y_new), head.log_prob(targets) - sd.log().sum(), rtol=1e-4, atol=1e-4
    )
    # mean: the point model plus the head's (small) mean residual, in data units
    torch.testing.assert_close(
        pred.mean(), mu + sd * (model._location_at(xs) + head.mean()), rtol=1e-4, atol=1e-4
    )
    assert (
        model.n_parameters_ > JointRegressor("gaussian", seed=0, **BUDGET).fit(x, y).n_parameters_
    )


def test_centres_and_density_improve_on_a_problem_with_a_hard_mean() -> None:
    # Paired over seeds, small budget, flow head.  Measured with this budget (3 seeds): root
    # mean square error of the predictive mean to the true mean 0.235 -> 0.199 (gains 0.030,
    # 0.047, 0.029), test negative log density 1.604 -> 1.491 (gains 0.133, 0.171, 0.035).
    gains_rmse, gains_nll = [], []
    for seed in range(3):
        x, y, _ = _data(1500, seed=10 + seed)
        x_new, y_new, mean_new = _data(2000, seed=100 + seed)
        scores = {}
        for location in (None, "mse"):
            model = JointRegressor("flow", location=location, seed=seed, **BUDGET).fit(x, y)
            pred = model.predict(x_new)
            torch.manual_seed(0)
            with torch.no_grad():
                rmse = float((pred.sample(256).mean(1) - mean_new).square().mean().sqrt())
                nll = float(-pred.log_prob(y_new).mean())
            scores[location] = (rmse, nll)
        gains_rmse.append(scores[None][0] - scores["mse"][0])
        gains_nll.append(scores[None][1] - scores["mse"][1])
    assert sum(gains_rmse) / 3 > 0.015, gains_rmse
    assert sum(gains_nll) / 3 > 0.05, gains_nll


@pytest.mark.parametrize("method", ["gaussian", "auto"])
def test_regions_keep_their_coverage(method: str) -> None:
    # The conformal rows are scored with networks that never saw them, as new rows are.
    x, y, _ = _data(1200, seed=3)
    x_new, y_new, _ = _data(3000, seed=4)
    model = JointRegressor(method, location="mse", n_members=1, seed=0, **BUDGET).fit(x, y)
    alpha = 0.2
    cover = float(model.region(x_new, alpha=alpha).contains(y_new).float().mean())
    n_cal = model.n_conformal_
    se = (alpha * (1 - alpha) / 3000 + alpha * (1 - alpha) / (n_cal + 1)) ** 0.5
    assert abs(cover - (1 - alpha)) < 3.5 * se + 1 / (n_cal + 1), cover
    # and so are the per-target 90% intervals of the predictive itself, within a few points
    torch.manual_seed(0)
    draws = model.predict(x_new).sample(400)
    lo, hi = draws.quantile(0.05, dim=1), draws.quantile(0.95, dim=1)
    inside = float(((y_new >= lo) & (y_new <= hi)).float().mean())
    assert 0.84 < inside < 0.96, inside


@pytest.mark.parametrize("method", ["gaussian", "mixture", "flow", "sklar", "energy", "auto"])
def test_every_method_runs_with_a_location_model(method: str) -> None:
    x, y, _ = _data(240, seed=5)
    tiny = dict(max_epochs=4, patience=2, hidden=(8,), n_members=1, device="cpu")
    model = JointRegressor(method, location="mse", seed=0, **tiny).fit(x, y)
    pred = model.predict(x[:7])
    assert (pred.batch_size, pred.event_dim) == (7, 3)
    assert pred.sample(5).shape == (7, 5, 3)
    if method != "energy":
        assert bool(torch.isfinite(pred.log_prob(y[:7])).all())
    region = model.region(x[:7], alpha=0.2)
    assert region.contains(y[:7]).shape == (7,)


def test_the_point_model_ignores_whatever_sits_in_a_missing_slot() -> None:
    # The squared error is masked: the networks, and so the residuals of every observed
    # target, are bit for bit the same whether a missing slot holds 0 or 1000.
    x, y, _ = _data(400, seed=6)
    xs = (x - x.mean(0)) / x.std(0)
    ys = (y - y.mean(0)) / y.std(0)
    mask = torch.ones_like(ys, dtype=torch.bool)
    mask[::5, 1] = False
    train = torch.arange(300)
    fitted = []
    for fill in (0.0, 1000.0):
        model = JointRegressor("gaussian", location="mse", seed=0, **BUDGET)
        model.batch_size_, model.n_train_ = 32, 300
        filled = torch.where(mask, ys, torch.full_like(ys, fill))
        fitted.append(model._fit_location(xs, filled, mask, train))
    (targets, spread), (targets_big, spread_big) = fitted
    torch.testing.assert_close(targets, targets_big, rtol=0, atol=0)
    torch.testing.assert_close(spread, spread_big, rtol=0, atol=0)
    assert bool((targets[~mask] == 0).all())  # a missing target has no residual
    # the spread of the target with holes is that of its observed out-of-fold residuals
    seen = mask[train, 1]
    torch.testing.assert_close(
        spread[1], (targets[train, 1][seen] * spread[1]).square().mean().sqrt(), rtol=1e-5, atol=0
    )
    assert float(spread[1]) < 1.0  # in units of the target's own spread: the model explains it


def test_fit_passes_the_observed_pattern_to_the_point_model() -> None:
    x, y, _ = _data(400, seed=6)
    y_holes = y.clone()
    y_holes[::5, 1] = float("nan")
    seen: dict[str, torch.Tensor] = {}

    class Recording(JointRegressor):
        def _fit_location(
            self, xs: torch.Tensor, ys: torch.Tensor, mask: torch.Tensor, train_d: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            seen["mask"], seen["ys"] = mask.cpu(), ys.cpu()
            return super()._fit_location(xs, ys, mask, train_d)

    model = Recording("gaussian", location="mse", seed=0, **BUDGET).fit(x, y_holes)
    assert bool((seen["mask"] == ~torch.isnan(y_holes)).all())
    assert bool(torch.isfinite(seen["ys"]).all())
    x_new, y_new, _ = _data(50, seed=7)
    pred = model.predict(x_new)
    mask = torch.ones_like(y_new, dtype=torch.bool)
    mask[:, 1] = False
    assert bool(torch.isfinite(pred.log_prob(y_new, mask)).all())
    assert bool(torch.isfinite(pred.mean()).all())


def test_cross_fitting_passes_the_location_model_to_every_fold() -> None:
    x, y, _ = _data(200, seed=8)
    tiny = dict(max_epochs=4, patience=2, hidden=(8,), device="cpu")
    model = JointRegressor("gaussian", location="mse", cv=3, seed=0, **tiny).fit(x, y)
    assert all(fold._location is not None for fold in model._fold_models)
    # The out-of-fold scores are densities of the residual targets of each fold model:
    # the reported validation score is the data-unit density of the held-out rows.
    plain = JointRegressor("gaussian", location="mse", cv=3, recalibrate=False, seed=0, **tiny).fit(
        x, y
    )
    assert plain.folds_ is not None
    held_out = []
    for k in range(3):
        rows = (plain.folds_ == k).nonzero().squeeze(1)
        with torch.no_grad():
            held_out.append(plain._fold_base(k, x[rows]).log_prob(y[rows]))
    assert plain.validation_log_score_["gaussian"] == pytest.approx(
        float(torch.cat(held_out).mean()), abs=1e-4
    )
    pred = model.predict(x[:5])
    assert bool(torch.isfinite(pred.log_prob(y[:5])).all())
    assert model.region(x[:5], alpha=0.2).contains(y[:5]).shape == (5,)


def test_fewer_folds_on_tiny_training_sets_and_reproducible_for_a_seed() -> None:
    x, y, _ = _data(60, seed=9)
    tiny = dict(max_epochs=3, patience=2, hidden=(8,), device="cpu")
    first = JointRegressor("gaussian", location="mse", seed=3, **tiny).fit(x, y)
    again = JointRegressor("gaussian", location="mse", seed=3, **tiny).fit(x, y)
    assert first._location is not None
    assert len(first._location) == max(2, min(_LOCATION_FOLDS, first.n_train_ // 8))
    torch.testing.assert_close(
        first.predict(x[:6]).log_prob(y[:6]), again.predict(x[:6]).log_prob(y[:6]), rtol=0, atol=0
    )
