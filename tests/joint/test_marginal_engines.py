"""Built-in marginal engines of the Sklar route and the automatic choice between them.

Two problems with known conditional marginals:

* the problem of ``examples/joint_regression_comparison.py`` (three targets, a
  two-component Gaussian mixture given ``x``), which favours mixture marginals;
* a skewed and heavy-tailed one (a log-normal target and a Student-t target with
  3 degrees of freedom, both with an ``x``-dependent scale) on few rows, where the
  quantile engine is the better of the two.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from scipy.stats import kstest
from torch import Tensor
from torch.distributions import LogNormal, MixtureSameFamily, Normal, StudentT

from torchregress.joint import (
    JointRegressor,
    MixtureMarginals,
    MixtureNetworkEngine,
    QuantileNetworkEngine,
    as_joint,
)
from torchregress.losses import MixtureDensityLoss

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "joint_regression_comparison.py"
ENGINE_BUDGET = dict(max_epochs=200, patience=20)


def _example() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_joint_regression_comparison", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example() -> ModuleType:
    return _example()


def _standardise(
    x: Tensor, y: Tensor, x_test: Tensor, y_test: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor, float]:
    """Standardise with training statistics; also the log-Jacobian of the target scaling."""
    xm, xs, ym, ys = x.mean(0), x.std(0), y.mean(0), y.std(0)
    out = (x - xm) / xs, (y - ym) / ys, (x_test - xm) / xs, (y_test - ym) / ys
    return (*out, float(ys.log().sum()))


def _true_marginal_log_density(example: ModuleType, x: Tensor, y: Tensor) -> Tensor:
    """Sum over targets of the true marginal log densities ``[n]``."""
    truth = example.true_conditional(x)
    comp = truth.component_distribution
    var = comp.covariance_matrix.diagonal(dim1=-2, dim2=-1)
    total = torch.zeros(x.shape[0])
    for j in range(y.shape[1]):
        marginal = MixtureSameFamily(
            truth.mixture_distribution, Normal(comp.mean[..., j], var[..., j].sqrt())
        )
        total = total + marginal.log_prob(y[:, j])
    return total


# ═══════════════════════════════════════════════════════════════ example problem


@pytest.fixture(scope="module")
def example_fits(example: ModuleType) -> dict[str, object]:
    # 2,400 rows: what JointRegressor trains on from the example's 4,000.
    x, y = example.make_data(4000, 0)
    x_te, y_te = example.make_data(1000, 1000)
    xs, ys, xs_te, ys_te, log_jac = _standardise(x[:2400], y[:2400], x_te, y_te)
    truth = _true_marginal_log_density(example, x_te, y_te)
    out: dict[str, object] = {"truth": truth, "ys_te": ys_te}
    for name, engine in (
        ("quantile", QuantileNetworkEngine(**ENGINE_BUDGET)),
        ("mixture", MixtureNetworkEngine(**ENGINE_BUDGET)),
    ):
        marg = engine.fit(xs, ys).predict(xs_te)
        out[name] = marg
        out[f"{name}_lp"] = marg.log_prob(ys_te).sum(-1) - log_jac
    return out


class TestMixtureEngineOnTheExample:
    def test_marginal_gap_to_the_truth(self, example_fits: dict[str, object]) -> None:
        truth = example_fits["truth"]
        lp_mix, lp_q = example_fits["mixture_lp"], example_fits["quantile_lp"]
        assert isinstance(truth, Tensor) and isinstance(lp_mix, Tensor)
        assert isinstance(lp_q, Tensor)
        gap_mix = float((truth - lp_mix).mean())  # measured: +0.09 nats
        gap_q = float((truth - lp_q).mean())  # measured: +0.32 nats
        diff = lp_mix - lp_q
        se = float(diff.std()) / math.sqrt(diff.numel())  # about 0.03
        assert gap_mix < 0.15, gap_mix
        assert float(diff.mean()) > 4 * se, (gap_mix, gap_q, se)

    def test_pits_are_uniform(self, example_fits: dict[str, object]) -> None:
        marg, ys_te = example_fits["mixture"], example_fits["ys_te"]
        assert isinstance(marg, MixtureMarginals) and isinstance(ys_te, Tensor)
        pit = marg.cdf(ys_te)
        p_values = [kstest(pit[:, j].numpy(), "uniform").pvalue for j in range(3)]
        assert min(p_values) > 0.01 / 3, p_values  # measured: 0.40, 0.80, 0.50

    def test_deterministic_and_leaves_the_caller_rng_alone(self, example: ModuleType) -> None:
        x, y = example.make_data(400, 3)
        torch.manual_seed(123)
        before = torch.rand(3)
        torch.manual_seed(123)
        fits = [MixtureNetworkEngine(max_epochs=5, seed=7).fit(x, y) for _ in range(2)]
        assert torch.equal(torch.rand(3), before)
        a, b = (f.predict(x[:20]) for f in fits)
        for p, q in zip((a.log_weights, a.loc, a.scale), (b.log_weights, b.loc, b.scale)):
            assert torch.equal(p, q)

    def test_missing_targets_are_left_out_of_the_loss(self, example: ModuleType) -> None:
        x, y = example.make_data(400, 4)
        gen = torch.Generator().manual_seed(0)
        mask = torch.rand(y.shape, generator=gen) > 0.3
        filled = [torch.where(mask, y, torch.full_like(y, v)) for v in (0.0, 1e6)]
        fits = [MixtureNetworkEngine(max_epochs=5).fit(x, f, mask) for f in filled]
        a, b = (f.predict(x[:20]) for f in fits)
        torch.testing.assert_close(a.loc, b.loc)
        torch.testing.assert_close(a.scale, b.scale)
        assert bool(torch.isfinite(a.log_prob(y[:20])).all())

    def test_starts_at_a_unit_variance_mixture(self) -> None:
        engine = MixtureNetworkEngine(max_epochs=0).fit(torch.randn(50, 2), torch.randn(50, 3))
        marg = engine.predict(torch.randn(4, 2))
        w = marg.log_weights.exp()
        mean = (w * marg.loc).sum(-1)
        var = (w * (marg.scale.square() + marg.loc.square())).sum(-1) - mean.square()
        torch.testing.assert_close(mean, torch.zeros_like(mean), atol=1e-6, rtol=0)
        torch.testing.assert_close(var, torch.ones_like(var), atol=1e-5, rtol=0)
        assert (marg.loc.diff(dim=-1) > 0).all()  # distinct components

    def test_as_joint_of_a_per_target_mdn(self) -> None:
        loss = MixtureDensityLoss(n_components=4, n_features=1)
        out = torch.randn(6, 3, loss.expected_output_size, dtype=torch.float64)
        joint = as_joint(loss, out)
        y = torch.randn(6, 3, dtype=torch.float64)
        per_target = loss.distribution(out).log_prob(y.unsqueeze(-1))  # [6, 3]
        torch.testing.assert_close(joint.log_prob(y), per_target.sum(-1))
        with pytest.raises(ValueError):
            as_joint(MixtureDensityLoss(n_components=2, n_features=1, df=4.0), out[..., :6])


# ═══════════════════════════════════════════════════ skewed, heavy-tailed problem


def _skewed_truth(x: Tensor) -> tuple[LogNormal, StudentT]:
    """A log-normal target and a Student-t (3 dof) target, scales depending on ``x``."""
    lognormal = LogNormal(0.5 * x[:, 0], 0.4 * (1.0 + 0.5 * x[:, 1].abs()))
    student = StudentT(3.0, torch.sin(2.0 * x[:, 2]), 0.3 * (1.0 + x[:, 0].abs()))
    return lognormal, student


def _skewed_data(n: int, seed: int) -> tuple[Tensor, Tensor]:
    x = torch.randn(n, 3, generator=torch.Generator().manual_seed(seed))
    with torch.random.fork_rng():
        torch.manual_seed(seed + 1)
        y = torch.stack([d.sample() for d in _skewed_truth(x)], dim=1)
    return x, y


def _skewed_log_density(x: Tensor, y: Tensor) -> Tensor:
    lognormal, student = _skewed_truth(x)
    return lognormal.log_prob(y[:, 0]) + student.log_prob(y[:, 1])


def test_automatic_choice_picks_the_better_engine_on_fresh_rows() -> None:
    # 1,000 rows (600 train the engines, 150 choose): on this sample the quantile
    # engine has the better held-out marginal log score (measured on 4,000 fresh
    # rows: gap to the truth +0.37 nats for quantiles, +0.49 for mixtures; with 4,000
    # rows the mixture engine wins, +0.12 against +0.19, and is chosen).
    x, y = _skewed_data(1000, 0)
    x_te, y_te = _skewed_data(4000, 10_000)
    model = JointRegressor("sklar", max_epochs=200, patience=20, seed=0, device="cpu").fit(x, y)
    assert set(model.marginal_scores_) == {"quantile", "mixture"}
    assert model.marginal_engine_ == max(model.marginal_scores_, key=model.marginal_scores_.get)
    assert model._x_stats is not None and model._y_stats is not None
    xs_te = (x_te - model._x_stats[0]) / model._x_stats[1]
    ys_te = (y_te - model._y_stats[0]) / model._y_stats[1]
    log_jac = float(model._y_stats[1].log().sum())
    test = {
        name: engine.predict(xs_te).log_prob(ys_te).sum(-1) - log_jac
        for name, engine in model._marginal_candidates.items()
    }
    other = "quantile" if model.marginal_engine_ == "mixture" else "mixture"
    diff = test[model.marginal_engine_] - test[other]
    se = float(diff.std()) / math.sqrt(diff.numel())
    assert float(diff.mean()) > -2 * se, (model.marginal_scores_, float(diff.mean()), se)
    assert model.marginal_engine_ == "quantile"
    truth = _skewed_log_density(x_te, y_te)
    assert float((truth - test["quantile"]).mean()) < 0.5


# ═══════════════════════════════════════════════════════════════ end to end


def _coverage_tolerance(n: int, level: float, n_cal: int) -> float:
    return 4.0 * math.sqrt(level * (1 - level) * (1 / n + 1 / n_cal))


@pytest.mark.parametrize("method", ["sklar", "auto"])
def test_joint_regressor_on_the_example(example: ModuleType, method: str) -> None:
    x, y = example.make_data(4000, 0)
    x_te, y_te = example.make_data(1000, 1000)
    oracle = float(-example.true_conditional(x_te).log_prob(y_te).mean())
    model = JointRegressor(method, max_epochs=200, patience=20, seed=0, device="cpu").fit(x, y)
    assert model.marginal_engine_ == "mixture"
    assert model.marginal_scores_["mixture"] > model.marginal_scores_["quantile"]
    gap = float(-model.predict(x_te).log_prob(y_te).mean()) - oracle
    # Measured: Sklar 0.560 (0.569 with the quantile engine), auto 0.143 (0.146 before).
    assert gap < (0.60 if method == "sklar" else 0.17), gap
    torch.manual_seed(0)
    region = model.region(x_te, alpha=0.1)
    coverage = float(region.contains(y_te).float().mean())
    assert model.n_conformal_ is not None
    assert abs(coverage - 0.9) < _coverage_tolerance(1000, 0.9, model.n_conformal_), coverage


def test_user_engine_is_used_as_is() -> None:
    x, y = _skewed_data(300, 1)
    engine = QuantileNetworkEngine(max_epochs=3)
    model = JointRegressor("sklar", marginals=engine, max_epochs=3, seed=0).fit(x, y)
    assert model.marginal_engine_ == "QuantileNetworkEngine"
    assert list(model.marginal_scores_) == ["QuantileNetworkEngine"]
    assert engine._net is None  # the template is copied, never fitted in place
    other = JointRegressor("gaussian", max_epochs=3, seed=0).fit(x, y)
    assert other.marginal_engine_ is None and other.marginal_scores_ == {}
