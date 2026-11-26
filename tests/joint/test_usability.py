"""Inputs users pass, default scores, HPD of candidates, estimator conveniences."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from torchregress.joint import (
    AffineJoint,
    ConditionalModelJoint,
    DensityScore,
    DistributionJoint,
    FlowJoint,
    GaussianCopula,
    GaussianJoint,
    GaussianMarginals,
    HPDScore,
    JointConformal,
    JointRegressor,
    LatentRecalibrator,
    LatentScore,
    MahalanobisScore,
    MarginalRecalibrator,
    MixtureJoint,
    SampleBallScore,
    SampleJoint,
    SchaakeShuffle,
    SklarJoint,
)
from torchregress.joint.estimator import _is_gaussian
from torchregress.losses import MixtureDensityLoss, create_flow_loss
from torchregress.metrics import sample_hpd_level

F64 = torch.float64
SMALL = dict(max_epochs=4, patience=2, hidden=(16, 16), batch_size=64, device="cpu")


def _gaussian(n: int = 20, d: int = 2, dtype: torch.dtype = torch.float32) -> GaussianJoint:
    gen = torch.Generator().manual_seed(0)
    mean = torch.randn(n, d, generator=gen).to(dtype)
    return GaussianJoint(mean, torch.eye(d, dtype=dtype))


def _mdn(n: int = 20, d: int = 2) -> DistributionJoint:
    torch.manual_seed(1)
    mdn = MixtureDensityLoss(n_components=3, n_features=d, covariance_type="full")
    return DistributionJoint(mdn.distribution(torch.randn(n, mdn.expected_output_size)))


def _all_predictives() -> dict:
    g = _gaussian()
    torch.manual_seed(2)
    flow = create_flow_loss(n_features=2, context_dim=3, flow_type="maf", n_transforms=2)
    context = torch.randn(20, 3)
    z = torch.randn(200, 2)
    sklar = SklarJoint(GaussianMarginals(torch.zeros(20, 2), torch.ones(20, 2)), g)
    template = SchaakeShuffle().fit(z).predictive(20)
    y = g.sample(1)[:, 0]
    return {
        "gaussian": g,
        "mdn": _mdn(),
        "flow": FlowJoint(flow, context),
        "model": ConditionalModelJoint(flow, context),
        "sample": SampleJoint(g.sample(30)),
        "projected": SampleJoint(g.sample(30)).marginal([1]),
        "masked_marginal": _mdn().marginal([0]),
        "affine": AffineJoint(g, torch.zeros(2), torch.full((2,), 2.0)),
        "mixture": MixtureJoint([g, g]),
        "sklar": sklar,
        "template": template,
        "latent_recal": LatentRecalibrator().fit(g, y).transform(g),
        "marginal_recal": MarginalRecalibrator().fit(g, y).transform(g),
        "copula": GaussianCopula().fit(z).predictive(20),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# A. inputs of any dtype, arrays, device
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("name", list(_all_predictives()))
def test_every_predictive_declares_dtype_and_device_without_drawing(name: str) -> None:
    pred = _all_predictives()[name]
    state = torch.get_rng_state()
    dtype, device = pred.dtype, pred.device
    assert torch.equal(torch.get_rng_state(), state)
    assert dtype == torch.float32 and device.type == "cpu"
    # inputs of another dtype, and NumPy arrays, are cast to the predictive
    y64 = np.random.default_rng(0).normal(size=(pred.batch_size, pred.event_dim))
    assert pred.as_input(y64).dtype == torch.float32
    if pred.has_density:
        lp = pred.log_prob(y64)
        assert lp.dtype == torch.float32 and lp.shape == (pred.batch_size,)
        assert pred.log_prob(torch.as_tensor(y64)).dtype == torch.float32


def test_mixture_head_accepts_float64_targets_and_numpy_masks() -> None:
    pred = _mdn()
    y = torch.randn(20, 2, dtype=F64)
    lp = pred.log_prob(y)  # was: "expected scalar type Double but found Float"
    torch.testing.assert_close(lp, pred.log_prob(y.float()))
    mask = np.ones((20, 2), dtype=bool)
    mask[:5, 1] = False
    torch.testing.assert_close(
        pred.log_prob(y.numpy(), mask), pred.log_prob(y.float(), torch.as_tensor(mask))
    )


def test_latent_maps_hpd_and_regions_accept_arrays_of_any_dtype() -> None:
    g = _gaussian()
    y = np.random.default_rng(1).normal(size=(20, 2))  # float64 array
    torch.testing.assert_close(
        g.from_latent(g.to_latent(y)), torch.as_tensor(y, dtype=torch.float32)
    )
    torch.manual_seed(0)
    a = g.hpd_level(y, 64)
    torch.manual_seed(0)
    b = g.hpd_level(torch.as_tensor(y, dtype=torch.float32), 64)
    assert torch.equal(a, b)
    weights = np.full(20, 2.0)
    cp = JointConformal(DensityScore(), 0.2).calibrate(
        g, y, weights=weights, groups=np.zeros(20, dtype=np.int64)
    )
    region = cp.region(g, test_weights=weights, groups=np.zeros(20, dtype=np.int64))
    inside = region.contains(y)
    assert inside.shape == (20,) and inside.dtype == torch.bool
    masked = JointConformal(MahalanobisScore(), 0.2).calibrate(
        g, y, mask=np.ones((20, 2), dtype=bool)
    )
    assert masked.region(g).contains(y, mask=np.ones((20, 2), dtype=bool)).shape == (20,)


def test_estimator_predictive_accepts_float64_numpy_targets() -> None:
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(150, 2, generator=gen)
    y = (x + 0.3 * torch.randn(150, 2, generator=gen)).double().numpy()
    model = JointRegressor("mixture", **SMALL).fit(x.numpy(), y)
    dist = model.predict(x[:6].double().numpy())
    assert dist.log_prob(y[:6]).shape == (6,)
    assert model.region(x[:6].numpy(), 0.2).contains(y[:6]).shape == (6,)


@pytest.mark.skipif(
    not (torch.cuda.is_available() or torch.backends.mps.is_available()),
    reason="needs an accelerator",
)
def test_cpu_inputs_reach_a_predictive_on_an_accelerator() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    g = GaussianJoint(torch.zeros(5, 2, device=device), torch.eye(2, device=device))
    lp = g.log_prob(np.zeros((5, 2)))
    assert lp.device.type == device.type


# ═══════════════════════════════════════════════════════════════════════════════
# B. default score
# ═══════════════════════════════════════════════════════════════════════════════


def test_auto_score_uses_the_latent_ball_only_for_gaussians() -> None:
    g = _gaussian()
    assert _is_gaussian(g) and _is_gaussian(AffineJoint(g, torch.zeros(2), torch.ones(2)))
    model = JointRegressor()
    assert isinstance(model._score_for(g, False), LatentScore)
    assert isinstance(
        model._score_for(AffineJoint(g, torch.zeros(2), torch.ones(2)), False), LatentScore
    )
    preds = _all_predictives()
    for name in ("flow", "mdn", "mixture", "sklar", "latent_recal", "marginal_recal", "model"):
        assert isinstance(model._score_for(preds[name], False), HPDScore), name
    assert isinstance(model._score_for(preds["sample"], False), SampleBallScore)
    assert isinstance(model._score_for(g, True), MahalanobisScore)


@pytest.mark.parametrize(
    ("method", "name"), [("gaussian", "latent"), ("mixture", "hpd"), ("flow", "hpd")]
)
def test_region_reports_the_score_it_used(method: str, name: str) -> None:
    gen = torch.Generator().manual_seed(4)
    x = torch.randn(150, 2, generator=gen)
    y = x + 0.3 * torch.randn(150, 2, generator=gen)
    model = JointRegressor(method, **SMALL).fit(x, y)
    assert model.region(x[:5], 0.1).score.name == name


# ═══════════════════════════════════════════════════════════════════════════════
# C. HPD levels of candidates
# ═══════════════════════════════════════════════════════════════════════════════


def test_hpd_level_of_candidates_shares_reference_draws_per_row() -> None:
    g = _gaussian(30, 3, F64)
    y3 = g.sample(7)
    torch.manual_seed(11)
    levels = g.hpd_level(y3, n_samples=64)
    assert levels.shape == (30, 7)
    torch.manual_seed(11)
    draws = g.log_prob(g.sample(64))
    for s in range(7):
        torch.testing.assert_close(levels[:, s], sample_hpd_level(g.log_prob(y3[:, s]), draws))


# ═══════════════════════════════════════════════════════════════════════════════
# D. estimator conveniences
# ═══════════════════════════════════════════════════════════════════════════════


def _fitted(n: int = 200, **kw) -> tuple[JointRegressor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(5)
    x = torch.randn(n, 2, generator=gen)
    y = 3.0 * x + 2.0 * torch.randn(n, 2, generator=gen) + 10.0
    return JointRegressor("gaussian", **{**SMALL, **kw}).fit(x, y), x, y


def test_split_sizes_are_reported() -> None:
    model, _, _ = _fitted(200)
    assert (model.n_val_, model.n_recal_, model.n_conformal_) == (20, 10, 20)
    assert model.n_train_ == 200 - 20 - 10 - 20


def test_validation_log_score_is_in_data_units() -> None:
    model, x, y = _fitted(200, recalibrate=False)
    perm = torch.randperm(200, generator=torch.Generator().manual_seed(model.seed))
    val = perm[: model.n_val_]
    expected = float(model.predict(x[val]).log_prob(y[val]).mean())
    assert math.isclose(model.validation_log_score_["gaussian"], expected, rel_tol=0, abs_tol=1e-4)


def test_calibrate_replaces_the_conformal_rows() -> None:
    model, x, y = _fitted(200)
    before = model.region(x[:5], 0.2).threshold.clone()
    gen = torch.Generator().manual_seed(6)
    x_new = torch.randn(400, 2, generator=gen)
    y_new = 3.0 * x_new + 2.0 * torch.randn(400, 2, generator=gen) + 10.0
    assert model.calibrate(x_new.numpy(), y_new.numpy()) is model
    assert model.n_conformal_ == 400
    after = model.region(x[:5], 0.2)
    assert not torch.equal(after.threshold, before)
    expected = JointConformal(LatentScore(), 0.2).calibrate(model.predict(x_new), y_new)
    torch.testing.assert_close(after.threshold[0], expected.thresholds_.reshape(1))
    with pytest.raises(ValueError, match="columns"):
        model.calibrate(x_new[:, :1], y_new)
    y_bad = y_new.clone()
    y_bad[0, 0] = math.inf
    with pytest.raises(ValueError, match="inf"):
        model.calibrate(x_new, y_bad)
    with pytest.raises(RuntimeError, match="fit"):
        JointRegressor().calibrate(x_new, y_new)


def test_recalibrators_accept_numpy_targets() -> None:
    from torchregress.joint import (
        GaussianJoint,
        LatentRecalibrator,
        MarginalRecalibrator,
        ScaleRecalibrator,
        latent_calibration_error,
    )

    torch.manual_seed(0)
    mean = torch.zeros(400, 2)
    narrow = GaussianJoint(mean, 0.6 * torch.eye(2).to(dtype=torch.float32))
    y = torch.randn(400, 2).double().numpy()  # float64 NumPy, as a user would have it
    for recal in (ScaleRecalibrator(), LatentRecalibrator(), MarginalRecalibrator()):
        out = recal.fit(narrow, y).transform(narrow)
        assert out.sample(3).shape == (400, 3, 2)
    before = float(latent_calibration_error(narrow, y))
    after = float(
        latent_calibration_error(LatentRecalibrator().fit(narrow, y).transform(narrow), y)
    )
    assert after < before
    mask = (torch.rand(400, 2) > 0.2).numpy()
    assert ScaleRecalibrator().fit(narrow, y, mask=mask).scale_ is not None


def test_diagnostics_accept_numpy_arrays() -> None:
    """Every public diagnostic takes NumPy arrays and gives the tensor answer."""

    from torchregress.joint import (
        conditional_coverage_gap,
        coverage_by_mask,
        dependence_error,
        hpd_calibration,
        nees_consistency,
        prerank_histograms,
        worst_slab_coverage,
    )

    gen = torch.Generator().manual_seed(3)
    n, s, d = 240, 60, 3
    x = torch.randn(n, 4, generator=gen)
    mean = torch.randn(n, d, generator=gen, dtype=torch.float64)
    eye = torch.eye(d).to(device=mean.device, dtype=mean.dtype)
    cov = eye.expand(n, d, d).clone()
    y = mean + torch.randn(n, d, generator=gen, dtype=torch.float64)
    draws = mean.unsqueeze(1) + torch.randn(n, s, d, generator=gen, dtype=torch.float64)
    covered = torch.rand(n, generator=gen) < 0.9
    mask = torch.rand(n, d, generator=gen) < 0.8
    levels = torch.rand(n, generator=gen, dtype=torch.float64)

    a = prerank_histograms(draws.numpy(), y.numpy())
    b = prerank_histograms(draws, y)
    assert a.names == b.names
    torch.testing.assert_close(a.p_values, b.p_values)
    torch.testing.assert_close(
        nees_consistency(mean.numpy(), cov.numpy(), y.numpy(), mask=mask.numpy()).nees,
        nees_consistency(mean, cov, y, mask=mask).nees,
    )
    assert hpd_calibration(levels.numpy()).ks_statistic == hpd_calibration(levels).ks_statistic
    torch.testing.assert_close(
        coverage_by_mask(covered.numpy(), mask.numpy()).coverage,
        coverage_by_mask(covered, mask).coverage,
    )
    assert (
        worst_slab_coverage(x.numpy(), covered.numpy()).coverage
        == worst_slab_coverage(x, covered).coverage
    )
    assert (
        conditional_coverage_gap(x.numpy(), covered.numpy(), alpha=0.1).brier
        == conditional_coverage_gap(x, covered, alpha=0.1).brier
    )
    torch.testing.assert_close(
        dependence_error(draws.numpy(), y.numpy()).predicted,
        dependence_error(draws, y).predicted,
    )
