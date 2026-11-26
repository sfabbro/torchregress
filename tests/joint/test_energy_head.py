"""The energy-score (engression) head of JointRegressor: loss, fit quality, regions, scale."""

from __future__ import annotations

import copy
import math
import time
import warnings

import numpy as np
import pytest
import torch
from scipy import stats

from torchregress.joint import ConditionalModelJoint, JointRegressor, as_joint
from torchregress.joint.estimator import _EnergyHead, _masked_energy_score
from torchregress.metrics import energy_score

# Budget of the quality tests: lr 1e-3 and patience 40, because energy-score training
# has long plateaus that the default patience of 30 at lr 2e-3 stops on.
BUDGET = dict(hidden=(128, 128), lr=1e-3, max_epochs=200, patience=40, device="cpu")


def _draw(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Heteroscedastic, skewed y0; y1 depends on y0; y2 on x and on y0's noise.

    The noise map is Lipschitz in the Gaussian noise (a skew by ``e + 0.6 |e|``,
    smoothed), the class a network of Gaussian noise represents: a Lipschitz map of
    Gaussian noise has sub-Gaussian tails, so heavier tails (``sinh(e)``) are outside it.
    """
    n = x.shape[0]
    e = torch.randn(n, 3, generator=gen)
    s = 0.4 * (1 + 0.5 * x[:, 1].abs())
    y0 = torch.sin(x[:, 0]) + s * (e[:, 0] + 0.6 * torch.sqrt(e[:, 0] ** 2 + 0.25))
    y1 = 0.8 * y0 + (0.3 * torch.sigmoid(2 * x[:, 0]) + 0.1) * e[:, 1]
    y2 = x[:, 1] ** 2 / 2 + 0.3 * e[:, 2] - 0.3 * e[:, 0]
    return torch.stack([y0, y1, y2], 1)


def _data(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=gen)
    return x, _draw(x, gen)


def _mean_correlation(draws: torch.Tensor) -> torch.Tensor:
    centred = draws - draws.mean(1, keepdim=True)
    cov = centred.mT @ centred / (draws.shape[1] - 1)
    sd = cov.diagonal(dim1=-2, dim2=-1).sqrt()
    return (cov / (sd.unsqueeze(-1) * sd.unsqueeze(-2))).mean(0)


def _coverage_tolerance(n: int, level: float, n_cal: int) -> float:
    # Test binomial noise plus the calibration-set fluctuation of the realised coverage.
    return 4.0 * math.sqrt(level * (1 - level) * (1 / n + 1 / n_cal))


@pytest.fixture(scope="module")
def fitted() -> JointRegressor:
    x, y = _data(20_000, seed=0)
    return JointRegressor("energy", seed=0, **BUDGET).fit(x, y)  # type: ignore[arg-type]


# ----------------------------------------------------------------------- the loss
def test_loss_is_the_energy_score() -> None:
    gen = torch.Generator().manual_seed(0)
    draws = torch.randn(40, 5, 3, generator=gen)  # [n, m, d]
    y = torch.randn(40, 3, generator=gen)
    ours = _masked_energy_score(draws, y, None)
    torch.testing.assert_close(ours, energy_score(draws.movedim(1, 0), y, reduction="none"))
    # The formula of Shen and Meinshausen 2024 (unbiased cross term over j != k).
    m = draws.shape[1]
    term1 = (draws - y.unsqueeze(1)).norm(dim=-1).mean(1)
    pair = (draws.unsqueeze(1) - draws.unsqueeze(2)).norm(dim=-1).sum((1, 2))
    torch.testing.assert_close(ours, term1 - pair / (2 * m * (m - 1)))


def test_masked_loss_is_the_energy_score_of_the_observed_subvector() -> None:
    gen = torch.Generator().manual_seed(1)
    draws = torch.randn(30, 4, 3, generator=gen)
    y = torch.randn(30, 3, generator=gen)
    mask = torch.rand(30, 3, generator=gen) > 0.4
    mask[0] = False  # nothing observed: score zero
    mask[1] = True
    ours = _masked_energy_score(draws, y, mask)
    assert float(ours[0]) == 0.0
    for i in range(1, 30):
        obs = mask[i]
        if not bool(obs.any()):
            continue
        ref = energy_score(draws[i : i + 1, :, obs].movedim(1, 0), y[i : i + 1, obs])
        assert float(ours[i]) == pytest.approx(float(ref), abs=1e-6)


def test_loss_has_finite_gradients_at_ties() -> None:
    # A draw equal to the observation (it happens in float32 training) or a row with
    # nothing observed must not give NaN gradients.
    head = _EnergyHead(2, 3, (8,), noise_dim=4, n_draws=2)
    x = torch.randn(6, 2)
    eps = head.noise(6, 2, x)
    with torch.no_grad():
        y = head.generate(x, eps)[:, 0].clone()
    mask = torch.ones(6, 3, dtype=torch.bool)
    mask[2] = False
    head.score(x, y, mask, eps).backward()
    assert all(bool(torch.isfinite(p.grad).all()) for p in head.parameters())


# ------------------------------------------------------------------- fit quality
def test_energy_head_against_the_oracle(fitted: JointRegressor) -> None:
    x_te, y_te = _data(2000, seed=1)
    pred = fitted.predict(x_te)
    assert not pred.has_density
    assert fitted.members_ == ["energy"] and fitted.validation_log_score_ == {}
    assert set(fitted.validation_energy_score_) == {"energy"}
    torch.manual_seed(1)
    model_draws = pred.sample(200)
    gen = torch.Generator().manual_seed(2)
    oracle_draws = torch.stack([_draw(x_te, gen) for _ in range(200)], 1)
    es_model = float(energy_score(model_draws.movedim(1, 0), y_te))
    es_oracle = float(energy_score(oracle_draws.movedim(1, 0), y_te))
    print(f"energy score: model {es_model:.4f}, oracle {es_oracle:.4f}")
    assert es_model < 1.03 * es_oracle
    # Marginal PITs (randomised over the 200 draws: fraction below the target).
    pit = (model_draws < y_te.unsqueeze(1)).double().mean(1)
    pvalues = [stats.kstest(pit[:, j].numpy(), "uniform").pvalue for j in range(3)]
    print(f"marginal PIT KS p-values: {[f'{p:.3g}' for p in pvalues]}")
    assert min(pvalues) > 1e-3
    corr_model, corr_oracle = _mean_correlation(model_draws), _mean_correlation(oracle_draws)
    print(
        f"mean correlation (0,1), (0,2): model {corr_model[0, 1]:.3f}, {corr_model[0, 2]:.3f}; "
        f"oracle {corr_oracle[0, 1]:.3f}, {corr_oracle[0, 2]:.3f}"
    )
    assert float((corr_model - corr_oracle).abs().max()) < 0.08


def test_regions_cover(fitted: JointRegressor) -> None:
    x_te, y_te = _data(2000, seed=3)
    n_cal = fitted.n_conformal_
    assert n_cal is not None
    region = fitted.region(x_te, alpha=0.1)
    assert region.score.name == "sample_ball"
    cover = float(region.contains(y_te).float().mean())
    sub = fitted.region(x_te, alpha=0.1, interest=[0, 2])
    assert sub.predictive.event_dim == 2
    cover_sub = float(sub.contains(y_te[:, [0, 2]]).float().mean())
    print(f"coverage at 90%: joint {cover:.3f}, interest [0, 2] {cover_sub:.3f}")
    for c in (cover, cover_sub):
        assert abs(c - 0.9) < _coverage_tolerance(2000, 0.9, n_cal)


def test_regions_cover_with_missing_calibration_targets(fitted: JointRegressor) -> None:
    x_cal, y_cal = _data(1500, seed=4)
    gen = torch.Generator().manual_seed(4)
    y_cal = torch.where(torch.rand(y_cal.shape, generator=gen) < 0.2, torch.nan, y_cal)
    x_te, y_te = _data(2000, seed=5)
    mask = torch.rand(y_te.shape, generator=gen) > 0.2
    model = copy.deepcopy(fitted).calibrate(x_cal, y_cal)  # leave the fixture's rows alone
    region = model.region(x_te, alpha=0.1)
    n_complete = int(torch.isfinite(y_cal).all(1).sum())
    cover = float(region.contains(y_te).float().mean())
    cover_masked = float(region.contains(y_te, mask).float().mean())
    print(
        f"coverage with missing calibration targets: complete {cover:.3f}, masked "
        f"{cover_masked:.3f}"
    )
    assert abs(cover - 0.9) < _coverage_tolerance(2000, 0.9, n_complete)
    assert abs(cover_masked - 0.9) < _coverage_tolerance(2000, 0.9, 1500)


# ---------------------------------------------------------------- missing targets
def test_missing_targets_train_on_every_row() -> None:
    x, y = _data(1200, seed=6)
    gen = torch.Generator().manual_seed(6)
    y_obs = torch.where(torch.rand(y.shape, generator=gen) < 0.3, torch.nan, y)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # no "rows are dropped" warning
        model = JointRegressor("energy", seed=0, max_epochs=5, device="cpu").fit(x, y_obs)
    assert model.n_train_ is not None
    assert math.isfinite(model.validation_energy_score_["energy"])
    assert model.predict(x[:4]).sample(3).shape == (4, 3, 3)


# ------------------------------------------------------------------------- scale
def test_many_targets_train_and_sample_in_bounded_time() -> None:
    gen = torch.Generator().manual_seed(7)
    n, d, r = 3000, 64, 4
    x = torch.randn(n, 3, generator=gen)
    loadings = torch.randn(d, r, generator=gen) / math.sqrt(r)
    factors = torch.tanh(x[:, :1]) + torch.randn(n, r, generator=gen) * (0.5 + x[:, 1:2].abs())
    y = factors @ loadings.T + 0.1 * torch.randn(n, d, generator=gen)
    start = time.perf_counter()
    model = JointRegressor("energy", seed=0, max_epochs=30, patience=10, device="cpu").fit(x, y)
    fit_time = time.perf_counter() - start
    pred = model.predict(torch.randn(1000, 3, generator=gen))
    start = time.perf_counter()
    draws = pred.sample(200)
    sample_time = time.perf_counter() - start
    print(f"d = 64: fit {fit_time:.1f} s, 1,000 x 200 draws {sample_time:.2f} s")
    assert draws.shape == (1000, 200, 64) and bool(torch.isfinite(draws).all())
    assert fit_time < 120.0 and sample_time < 10.0
    # The low-rank structure is learned: most of the predictive variance is in r directions.
    eig = torch.linalg.eigvalsh(pred.covariance(256)[:50])
    top = eig[:, -r:].sum(-1) / eig.sum(-1)
    assert float(top.mean()) > 0.8


# ----------------------------------------------------------------- determinism, io
def test_determinism_numpy_inputs_and_cpu_predict() -> None:
    x, y = _data(400, seed=8)
    kw = dict(seed=3, max_epochs=5, hidden=(32,), device="cpu")
    a = JointRegressor("energy", **kw).fit(x.numpy(), y.numpy().astype(np.float64))  # type: ignore[arg-type]
    b = JointRegressor("energy", **kw).fit(x, y)  # type: ignore[arg-type]
    assert a.validation_energy_score_ == b.validation_energy_score_
    pa, pb = a.predict(x[:5].numpy()), b.predict(x[:5])
    assert pa.device == torch.device("cpu")
    torch.manual_seed(0)
    da = pa.sample(6)
    torch.manual_seed(0)
    db = pb.sample(6)
    torch.testing.assert_close(da, db)
    with pytest.raises(NotImplementedError, match="no density"):
        pa.log_prob(y[:5])
    assert pa.marginal([1]).sample(4).shape == (5, 4, 1)
    assert pa.mean().shape == (5, 3) and pa.covariance().shape == (5, 3, 3)
    with pytest.raises(ValueError, match="energy_draws"):
        JointRegressor("energy", energy_draws=1)
    with pytest.raises(ValueError, match="noise_dim"):
        JointRegressor("energy", noise_dim=0)


def test_as_joint_accepts_a_user_generator() -> None:
    class Generator(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Linear(3, 2)

        def sample(self, context: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
            eps = torch.randn(context.shape[0], n_samples, 1)
            ctx = context.unsqueeze(1).expand(-1, n_samples, -1)
            return self.net(torch.cat([ctx, eps], -1))

    context = torch.randn(7, 2)
    pred = as_joint(Generator(), context)
    assert isinstance(pred, ConditionalModelJoint) and not pred.has_density
    assert pred.sample(5).shape == (7, 5, 2)
    with pytest.raises(NotImplementedError, match="no density"):
        pred.log_prob(torch.zeros(7, 2))
