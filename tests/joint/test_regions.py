"""Joint conformal regions: thresholds, coverage, masks, volume, sampling."""

from __future__ import annotations

import math

import pytest
import torch
from scipy.stats import chi2
from torch.distributions import MultivariateNormal

from torchregress.joint.predictive import DistributionJoint, GaussianJoint, SampleJoint
from torchregress.joint.regions import (
    ConformalScore,
    DensityScore,
    HPDScore,
    JointConformal,
    LatentScore,
    MahalanobisScore,
    MaxResidualScore,
    SampleBallScore,
)
from torchregress.losses import MultivariateScoreConformal
from torchregress.losses.conformal import _weighted_conformal_thresholds, finite_sample_quantile

F64 = torch.float64
D = 3
# Correlated factor shared by all rows (scaled per row), so ellipsoids beat boxes.
BASE_TRIL = torch.tensor([[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [-0.5, 0.3, 0.4]], dtype=F64)


def _problem(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rows ``(mean, scale_tril, y)`` with ``y | x ~ N(mean(x), L(x) L(x)^T)``."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 1, generator=gen, dtype=F64)
    mean = torch.sin(3.0 * x) * torch.tensor([1.0, -0.5, 2.0], dtype=F64) + x
    tril = (0.5 + x).unsqueeze(-1) * BASE_TRIL
    eps = torch.randn(n, D, generator=gen, dtype=F64)
    y = mean + (eps.unsqueeze(1) @ tril.mT).squeeze(1)
    return mean, tril, y


def _predictive(mean: torch.Tensor, tril: torch.Tensor, case: str) -> GaussianJoint:
    if case == "well_specified":
        return GaussianJoint(mean, tril)
    # covariance shrunk by 4 and mean shifted
    return GaussianJoint(mean + 0.5, tril / 2.0)


SCORES: dict[str, type[ConformalScore]] = {
    "density": DensityScore,
    "hpd": HPDScore,
    "latent": LatentScore,
    "mahalanobis": MahalanobisScore,
    "max_residual": MaxResidualScore,
    "sample_ball": SampleBallScore,
}

N_CAL, N_TEST, ALPHA = 4000, 8000, 0.1


def _coverage_sd(n_cal: int, n_test: int, alpha: float) -> float:
    # Test-set binomial noise plus the spread of the calibration-conditional
    # coverage (Beta with variance about alpha(1 - alpha) / (n_cal + 2)).
    v = alpha * (1 - alpha)
    return math.sqrt(v / n_test + v / (n_cal + 2))


# ═══════════════════════════════════════════════════════════════════════════════
# coverage
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("case", ["well_specified", "misspecified"])
@pytest.mark.parametrize("name", list(SCORES))
def test_marginal_coverage(name: str, case: str) -> None:
    """n_cal = 4000, n_test = 8000, alpha = 0.1; tolerance 4 standard errors (about 0.023)."""
    torch.manual_seed(1)
    m_cal, l_cal, y_cal = _problem(N_CAL, seed=10)
    m_test, l_test, y_test = _problem(N_TEST, seed=11)
    cp = JointConformal(SCORES[name](), alpha=ALPHA)
    cp.calibrate(_predictive(m_cal, l_cal, case), y_cal)
    region = cp.region(_predictive(m_test, l_test, case))
    covered = region.contains(y_test)
    assert covered.shape == (N_TEST,) and covered.dtype == torch.bool
    coverage = covered.double().mean().item()
    assert abs(coverage - (1 - ALPHA)) <= 4 * _coverage_sd(N_CAL, N_TEST, ALPHA), coverage


# ═══════════════════════════════════════════════════════════════════════════════
# exact threshold rules
# ═══════════════════════════════════════════════════════════════════════════════


def test_threshold_is_the_conformal_order_statistic() -> None:
    mean, tril, y = _problem(50, seed=0)
    pred = GaussianJoint(mean, tril)
    alphas = (0.1, 0.3)
    cp = JointConformal(DensityScore(), alpha=alphas).calibrate(pred, y)
    scores = DensityScore().score(pred, y).sort().values
    for i, a in enumerate(alphas):
        k = math.ceil(51 * (1 - a))
        assert cp.thresholds_[i] == scores[k - 1]
    assert cp.thresholds_.shape == (2,)


def test_threshold_is_infinite_when_n_is_too_small() -> None:
    mean, tril, y = _problem(5, seed=0)  # ceil(6 * 0.9) = 6 > 5
    pred = GaussianJoint(mean, tril)
    cp = JointConformal(MahalanobisScore(), alpha=0.1).calibrate(pred, y)
    assert math.isinf(cp.thresholds_.item())
    region = cp.region(pred)
    assert bool(region.contains(y + 1e6).all())
    log_v, _ = region.log_volume()
    assert bool(torch.isinf(log_v).all() & (log_v > 0).all())


def test_levels_are_nested() -> None:
    mean, tril, y = _problem(400, seed=2)
    pred = GaussianJoint(mean, tril)
    cp = JointConformal(LatentScore(), alpha=[0.05, 0.1, 0.3]).calibrate(pred, y)
    assert bool((cp.thresholds_[:-1] >= cp.thresholds_[1:]).all())
    m2, l2, y2 = _problem(300, seed=3)
    inside = cp.region(GaussianJoint(m2, l2)).contains(y2)
    assert inside.shape == (300, 3)
    assert bool((inside[:, 0] >= inside[:, 1]).all() & (inside[:, 1] >= inside[:, 2]).all())


def test_equal_weights_reduce_to_the_unweighted_threshold() -> None:
    mean, tril, y = _problem(199, seed=4)
    pred = GaussianJoint(mean, tril)
    alphas = [0.05, 0.1, 0.25]
    plain = JointConformal(MaxResidualScore(), alpha=alphas).calibrate(pred, y)
    weighted = JointConformal(MaxResidualScore(), alpha=alphas).calibrate(
        pred, y, weights=torch.full((199,), 2.5, dtype=F64)
    )
    torch.testing.assert_close(weighted.thresholds_, plain.thresholds_, rtol=0, atol=0)
    m2, l2, _ = _problem(7, seed=5)
    test_pred = GaussianJoint(m2, l2)
    r_w = weighted.region(test_pred, test_weights=torch.full((7,), 2.5, dtype=F64))
    r_p = plain.region(test_pred)
    torch.testing.assert_close(r_w.threshold, r_p.threshold, rtol=0, atol=0)


def test_weighted_quantile_by_hand() -> None:
    # Scores |y| = 1, 2, 3, 4 with weights 1, 1, 1, 3.
    pred = GaussianJoint(torch.zeros(4, 1, dtype=F64), torch.ones(1, 1, dtype=F64))
    y = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=F64)
    weights = torch.tensor([1.0, 1.0, 1.0, 3.0], dtype=F64)
    cp = JointConformal(MaxResidualScore(), alpha=[0.5, 0.7, 0.2])
    cp.calibrate(pred, y, weights=weights)
    test_pred = GaussianJoint(torch.zeros(2, 1, dtype=F64), torch.ones(1, 1, dtype=F64))
    region = cp.region(test_pred, test_weights=torch.tensor([2.0, 0.0], dtype=F64))
    # Test weight 2: total 8, cumulative 1, 2, 3, 6; targets 4, 2.4, 6.4.
    # Test weight 0: total 6; targets 3, 1.8, 4.8.
    expected = torch.tensor([[4.0, 3.0, math.inf], [3.0, 2.0, 4.0]], dtype=F64)
    torch.testing.assert_close(region.threshold, expected, rtol=0, atol=0)


def test_mondrian_equals_per_group_calibration() -> None:
    mean, tril, y = _problem(300, seed=6)
    pred = GaussianJoint(mean, tril)
    groups = torch.randint(0, 3, (300,), generator=torch.Generator().manual_seed(0)) * 5
    cp = JointConformal(DensityScore(), alpha=[0.1, 0.2]).calibrate(pred, y, groups=groups)
    assert cp.thresholds_.shape == (3, 2)
    torch.testing.assert_close(cp.group_ids_, torch.tensor([0, 5, 10]))
    for i, gid in enumerate([0, 5, 10]):
        sel = groups == gid
        alone = JointConformal(DensityScore(), alpha=[0.1, 0.2]).calibrate(pred.select(sel), y[sel])
        torch.testing.assert_close(cp.thresholds_[i], alone.thresholds_, rtol=0, atol=0)
    m2, l2, _ = _problem(4, seed=7)
    region = cp.region(GaussianJoint(m2, l2), groups=torch.tensor([10, 0, 5, 99]))
    torch.testing.assert_close(region.threshold[:3], cp.thresholds_[[2, 0, 1]], rtol=0, atol=0)
    assert bool(torch.isinf(region.threshold[3]).all())  # unseen group


# ═══════════════════════════════════════════════════════════════════════════════
# volume
# ═══════════════════════════════════════════════════════════════════════════════


def _log_unit_ball(d: int) -> float:
    return 0.5 * d * math.log(math.pi) - math.lgamma(0.5 * d + 1)


def test_mahalanobis_closed_form_volume() -> None:
    mean, tril, y = _problem(500, seed=8)
    cp = JointConformal(MahalanobisScore(), alpha=0.1).calibrate(GaussianJoint(mean, tril), y)
    m2, l2, _ = _problem(10, seed=9)
    log_v, se = cp.region(GaussianJoint(m2, l2)).log_volume()
    r = cp.thresholds_.item()
    expected = _log_unit_ball(D) + D * math.log(r) + 0.5 * torch.logdet(l2 @ l2.mT)
    torch.testing.assert_close(log_v, expected)
    assert bool((se == 0).all())

    # calibrated with a mask: chi-square scale, radius^2 = F^{-1}_{chi2_d}(t)
    mask = torch.ones(500, D, dtype=torch.bool)
    cp_m = JointConformal(MahalanobisScore(), alpha=0.1)
    cp_m.calibrate(GaussianJoint(mean, tril), y, mask=mask)
    t = cp_m.thresholds_.item()
    torch.testing.assert_close(torch.tensor([math.sqrt(chi2.ppf(t, D))], dtype=F64), cp.thresholds_)
    log_v_m, _ = cp_m.region(GaussianJoint(m2, l2)).log_volume()
    torch.testing.assert_close(log_v_m, expected)


@pytest.mark.parametrize(
    ("name", "case"),
    [
        ("density", "well_specified"),
        ("latent", "well_specified"),
        ("density", "misspecified"),
        ("latent", "misspecified"),
        ("density_no_latent", "well_specified"),
    ],
)
def test_importance_sampling_volume_matches_the_ellipsoid(name: str, case: str) -> None:
    """20 rows, K = 4096 draws each; every row within 4 standard errors.

    Predictives with a latent map use the defensive latent-scale mixture; the
    ``density_no_latent`` case (a ``DistributionJoint``) samples the predictive only.
    """
    torch.manual_seed(3)
    mean, tril, y = _problem(500, seed=12)
    m2, l2, _ = _problem(20, seed=13)
    cal_pred, test_pred = _predictive(mean, tril, case), _predictive(m2, l2, case)
    if name == "density_no_latent":
        cal = DistributionJoint(MultivariateNormal(cal_pred.loc, scale_tril=cal_pred.scale_tril))
        test = DistributionJoint(MultivariateNormal(test_pred.loc, scale_tril=test_pred.scale_tril))
        assert not test.has_latent
        cp = JointConformal(DensityScore(), alpha=0.1).calibrate(cal, y)
        region = cp.region(test)
    else:
        cp = JointConformal(SCORES[name](), alpha=0.1).calibrate(cal_pred, y)
        region = cp.region(test_pred)
    log_v, se = region.log_volume(n_samples=4096)
    t = cp.thresholds_.item()
    half_log_det = test_pred.scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    if name.startswith("density"):  # -log p <= t  <=>  Q <= 2 (t - d/2 log 2 pi - 1/2 log det)
        r2 = 2.0 * (t - 0.5 * D * math.log(2 * math.pi) - half_log_det)
        log_r = 0.5 * r2.log()
    else:
        log_r = torch.full_like(half_log_det, math.log(t))
    expected = _log_unit_ball(D) + D * log_r + half_log_det
    assert bool((se > 0).all())
    assert bool(((log_v - expected).abs() <= 4 * se).all()), (log_v - expected) / se


def test_box_volume_is_exact() -> None:
    mean, tril, y = _problem(500, seed=14)
    cp = JointConformal(MaxResidualScore(), alpha=[0.1, 0.2]).calibrate(
        GaussianJoint(mean, tril), y
    )
    m2, l2, _ = _problem(6, seed=15)
    region = cp.region(GaussianJoint(m2, l2))
    log_v, se = region.log_volume()
    std = (l2 @ l2.mT).diagonal(dim1=-2, dim2=-1).sqrt()
    expected = (2 * cp.thresholds_.view(1, 2, 1) * std.unsqueeze(1)).log().sum(-1)
    torch.testing.assert_close(log_v, expected)
    assert log_v.shape == (6, 2) and bool((se == 0).all())
    lower, upper = region.bounding_box()
    assert lower.shape == (6, 2, D)
    torch.testing.assert_close((upper - lower).log().sum(-1), expected)


def test_sample_ball_volume_without_density_agrees_with_importance_sampling() -> None:
    """Box Monte Carlo (no density) and importance sampling agree within 4 combined SE."""
    torch.manual_seed(4)
    mean, tril, y = _problem(500, seed=16)
    cp = JointConformal(SampleBallScore(n_samples=32), alpha=0.2)
    cp.calibrate(GaussianJoint(mean, tril), y)
    m2, l2, _ = _problem(8, seed=17)
    region = cp.region(GaussianJoint(m2, l2))
    is_est, is_se = region._importance_log_volume(8192)
    box_est, box_se = region._box_mc_log_volume(8192)
    combined = (is_se.square() + box_se.square()).sqrt()
    assert bool(((is_est - box_est).abs() <= 4 * combined).all())

    sample_pred = SampleJoint(GaussianJoint(mean, tril).sample(200))
    cp_s = JointConformal(SampleBallScore(n_samples=32), alpha=0.2).calibrate(sample_pred, y)
    log_v, se = cp_s.region(sample_pred).log_volume(2048)
    assert bool(torch.isfinite(log_v).all() & torch.isfinite(se).all())


def test_ellipsoid_regions_are_smaller_than_the_box() -> None:
    torch.manual_seed(5)
    mean, tril, y = _problem(2000, seed=18)
    m2, l2, _ = _problem(200, seed=19)
    mean_log_v = {}
    for name in ["density", "latent", "mahalanobis", "max_residual"]:
        cp = JointConformal(SCORES[name](), alpha=0.1).calibrate(GaussianJoint(mean, tril), y)
        log_v, _ = cp.region(GaussianJoint(m2, l2)).log_volume(n_samples=2048)
        mean_log_v[name] = log_v.mean().item()
    for name in ["density", "latent", "mahalanobis"]:
        assert mean_log_v[name] < mean_log_v["max_residual"], mean_log_v


# ═══════════════════════════════════════════════════════════════════════════════
# masks
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("name", ["mahalanobis", "max_residual", "density"])
def test_missing_targets_coverage(name: str) -> None:
    """MCAR (30 % missing); n_cal = 4000, n_test = 8000, tolerance 4 SE on observed rows."""
    gen = torch.Generator().manual_seed(20)
    m_cal, l_cal, y_cal = _problem(N_CAL, seed=21)
    m_test, l_test, y_test = _problem(N_TEST, seed=22)
    mask_cal = torch.rand(N_CAL, D, generator=gen) > 0.3
    mask_test = torch.rand(N_TEST, D, generator=gen) > 0.3
    mask_cal[:5] = False  # all-missing rows are dropped
    y_cal = torch.where(mask_cal, y_cal, torch.full_like(y_cal, math.nan))
    cp = JointConformal(SCORES[name](), alpha=ALPHA)
    cp.calibrate(_predictive(m_cal, l_cal, "misspecified"), y_cal, mask=mask_cal)
    assert cp.n_calibration_ == int(mask_cal.any(-1).sum()) <= N_CAL - 5
    region = cp.region(_predictive(m_test, l_test, "misspecified"))
    inside = region.contains(y_test, mask=mask_test)
    observed = mask_test.any(-1)
    assert bool(inside[~observed].all())  # nothing observed: trivially inside
    coverage = inside[observed].double().mean().item()
    sd = _coverage_sd(cp.n_calibration_, int(observed.sum()), ALPHA)
    assert abs(coverage - (1 - ALPHA)) <= 4 * sd, coverage


def test_mask_capability_errors() -> None:
    mean, tril, y = _problem(20, seed=23)
    pred = GaussianJoint(mean, tril)
    mask = torch.ones(20, D, dtype=torch.bool)
    for score in [HPDScore(), LatentScore(), SampleBallScore()]:
        with pytest.raises(ValueError, match="mask"):
            JointConformal(score).calibrate(pred, y, mask=mask)
    cp = JointConformal(DensityScore()).calibrate(pred, y)
    with pytest.raises(ValueError, match="without a mask"):
        cp.region(pred).contains(y, mask=mask)


# ═══════════════════════════════════════════════════════════════════════════════
# sampling, boxes, shapes, capabilities
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("name", list(SCORES))
def test_sample_is_inside_and_in_the_box(name: str) -> None:
    torch.manual_seed(6)
    mean, tril, y = _problem(500, seed=24)
    cp = JointConformal(SCORES[name](), alpha=0.2).calibrate(GaussianJoint(mean, tril), y)
    m2, l2, _ = _problem(5, seed=25)
    region = cp.region(GaussianJoint(m2, l2))
    draws = region.sample(300)
    assert draws.shape == (5, 300, D)
    assert bool(region.contains(draws).all())
    lower, upper = region.bounding_box(n_samples=1024)
    assert lower.shape == upper.shape == (5, D)
    if name in ("mahalanobis", "max_residual"):  # exact box
        assert bool(((draws >= lower.unsqueeze(1)) & (draws <= upper.unsqueeze(1))).all())
    if name in ("density", "latent", "mahalanobis"):
        # each is the ellipsoid Q <= r^2 for a Gaussian: compare with its exact box
        t = cp.thresholds_.item()
        half_log_det = l2.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        if name == "density":
            r = (2.0 * (t - 0.5 * D * math.log(2 * math.pi) - half_log_det)).sqrt()
        else:
            r = torch.full_like(half_log_det, t)
        half = r.unsqueeze(-1) * (l2 @ l2.mT).diagonal(dim1=-2, dim2=-1).sqrt()
        e_lower, e_upper = m2 - half, m2 + half
        if name == "mahalanobis":
            torch.testing.assert_close(lower, e_lower)
            torch.testing.assert_close(upper, e_upper)
        else:  # from 1024 accepted draws: inside the exact box and covering most of it
            assert bool((lower >= e_lower).all() & (upper <= e_upper).all())
            assert bool(((upper - lower) >= 0.7 * (e_upper - e_lower)).all())


def test_sample_ball_region_is_deterministic() -> None:
    mean, tril, y = _problem(200, seed=26)
    cp = JointConformal(SampleBallScore(n_samples=16), alpha=0.1)
    cp.calibrate(GaussianJoint(mean, tril), y)
    m2, l2, y2 = _problem(50, seed=27)
    region = cp.region(GaussianJoint(m2, l2))
    probe = y2.unsqueeze(1) + torch.randn(50, 40, D, dtype=F64)
    assert torch.equal(region.contains(probe), region.contains(probe))
    assert torch.equal(region.contains(y2), region.contains(y2))


def test_shapes_for_scalar_and_multi_level_alpha() -> None:
    mean, tril, y = _problem(300, seed=28)
    pred = GaussianJoint(mean, tril)
    m2, l2, y2 = _problem(7, seed=29)
    test_pred = GaussianJoint(m2, l2)
    single = JointConformal(DensityScore(), alpha=0.1).calibrate(pred, y).region(test_pred)
    multi = JointConformal(DensityScore(), alpha=(0.1, 0.2)).calibrate(pred, y).region(test_pred)
    probe = test_pred.sample(11)
    assert single.threshold.shape == (7, 1) and multi.threshold.shape == (7, 2)
    assert single.contains(y2).shape == (7,) and multi.contains(y2).shape == (7, 2)
    assert single.contains(probe).shape == (7, 11) and multi.contains(probe).shape == (7, 11, 2)
    assert single.log_volume(256)[0].shape == (7,) and multi.log_volume(256)[1].shape == (7, 2)
    assert single.sample(5).shape == (7, 5, D)
    assert multi.sample(5, level=1).shape == (7, 5, D)
    with pytest.raises(ValueError, match="level"):
        multi.sample(5)
    assert multi.bounding_box(256)[0].shape == (7, 2, D)
    assert single.bounding_box(256)[1].shape == (7, D)


def test_capability_errors() -> None:
    mean, tril, y = _problem(30, seed=30)
    draws = SampleJoint(GaussianJoint(mean, tril).sample(50))
    for score in [LatentScore(), DensityScore(), HPDScore()]:
        with pytest.raises(ValueError, match="needs a predictive"):
            JointConformal(score).calibrate(draws, y)
    # moment-based scores and sample balls work on draws only
    for score in [MahalanobisScore(), MaxResidualScore(), SampleBallScore()]:
        region = JointConformal(score, alpha=0.2).calibrate(draws, y).region(draws)
        assert region.contains(y).shape == (30,)
    with pytest.raises(ValueError, match="alpha"):
        JointConformal(DensityScore(), alpha=1.5)
    with pytest.raises(RuntimeError, match="calibrate"):
        JointConformal(DensityScore()).region(GaussianJoint(mean, tril))


def test_marginal_region_for_targets_of_interest() -> None:
    """Regions for a subset of targets come from predictive.marginal(dims) and y[:, dims]."""
    mean, tril, y = _problem(N_CAL, seed=31)
    m2, l2, y2 = _problem(N_TEST, seed=32)
    dims = [0, 2]
    cp = JointConformal(LatentScore(), alpha=ALPHA)
    cp.calibrate(GaussianJoint(mean, tril).marginal(dims), y[:, dims])
    region = cp.region(GaussianJoint(m2, l2).marginal(dims))
    coverage = region.contains(y2[:, dims]).double().mean().item()
    assert abs(coverage - (1 - ALPHA)) <= 4 * _coverage_sd(N_CAL, N_TEST, ALPHA)


# ═══════════════════════════════════════════════════════════════════════════════
# one rank rule in the library
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("n", [9, 19, 50, 199])
def test_thresholds_equal_the_shipped_conformal_helpers(n: int) -> None:
    """Every level: the documented rank, finite_sample_quantile and MultivariateScoreConformal."""
    mean, tril, y = _problem(n, seed=40 + n)
    pred = GaussianJoint(mean, tril)
    alphas = [k / 40 for k in range(1, 40)] + [0.3, 0.7, 0.1]
    cp = JointConformal(DensityScore(), alpha=alphas).calibrate(pred, y)
    scores = DensityScore().score(pred, y)
    ordered = scores.sort().values
    msc_scores = None
    for i, a in enumerate(alphas):
        k = math.ceil((n + 1) * (1 - a))  # in floating point, as documented
        expected = ordered[k - 1] if k <= n else torch.tensor(math.inf, dtype=F64)
        assert cp.thresholds_[i] == expected, (n, a)
        assert cp.thresholds_[i] == finite_sample_quantile(scores, a)
        # MultivariateScoreConformal(score_fn="nll") scores are the same -log p
        msc = MultivariateScoreConformal(a, score_fn="nll").calibrate(mean, tril @ tril.mT, y)
        if msc_scores is None:
            msc_scores = msc._scores_cal
            torch.testing.assert_close(msc_scores, scores, rtol=0, atol=1e-10)
        if math.isinf(float(expected)):
            assert math.isinf(msc.region_radius())
        else:
            assert math.isclose(msc.region_radius(), float(cp.thresholds_[i]), abs_tol=1e-10)


def test_rank_above_an_integer_in_floating_point_takes_the_next_order_statistic() -> None:
    # (9 + 1)(1 - 0.7) = 3.0000000000000004 in floating point: rank 4, not 3.
    assert (9 + 1) * (1 - 0.7) > 3
    mean, tril, y = _problem(9, seed=50)
    pred = GaussianJoint(mean, tril)
    cp = JointConformal(DensityScore(), alpha=0.7).calibrate(pred, y)
    assert cp.thresholds_.item() == DensityScore().score(pred, y).sort().values[3].item()


def test_weighted_thresholds_equal_the_shipped_weighted_quantile() -> None:
    mean, tril, y = _problem(120, seed=51)
    pred = GaussianJoint(mean, tril)
    gen = torch.Generator().manual_seed(52)
    w = torch.rand(120, generator=gen, dtype=F64) + 0.05
    tw = torch.rand(7, generator=gen, dtype=F64) * 3
    alphas = [0.05, 0.1, 0.25]
    cp = JointConformal(MaxResidualScore(), alpha=alphas).calibrate(pred, y, weights=w)
    scores = MaxResidualScore().score(pred, y)
    for i, a in enumerate(alphas):
        assert cp.thresholds_[i] == _weighted_conformal_thresholds(scores, 1 - a, w)[0]
    m2, l2, _ = _problem(7, seed=53)
    region = cp.region(GaussianJoint(m2, l2), test_weights=tw)
    for i, a in enumerate(alphas):
        torch.testing.assert_close(
            region.threshold[:, i],
            _weighted_conformal_thresholds(scores, 1 - a, w, tw),
            rtol=0,
            atol=0,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class _NaNScore(MaxResidualScore):
    """Max residual, NaN on rows whose first target is negative."""

    def _evaluate(self, predictive, y, mask, state):  # type: ignore[no-untyped-def]
        out = super()._evaluate(predictive, y, mask, state)
        bad = (y[..., 0] < 0) if y.dim() == 3 else (y[:, 0] < 0)
        return torch.where(bad, torch.full_like(out, math.nan), out)


def test_nan_scores_count_as_infinite_in_calibration_and_membership() -> None:
    mean = torch.zeros(40, 1, dtype=F64)
    pred = GaussianJoint(mean, torch.ones(1, 1, dtype=F64))
    y = torch.linspace(-0.5, 3.0, 40, dtype=F64).unsqueeze(1)  # 6 negative -> NaN scores
    finite = JointConformal(_NaNScore(), alpha=0.5).calibrate(pred, y)
    assert math.isfinite(finite.thresholds_.item())
    region = finite.region(pred)
    probe = torch.full((40, 1), -1.0, dtype=F64)
    assert not bool(region.contains(probe).any())  # NaN score: outside a finite region
    infinite = JointConformal(_NaNScore(), alpha=0.01).calibrate(pred, y)  # 41 * 0.99 > 40
    assert math.isinf(infinite.thresholds_.item())
    assert bool(infinite.region(pred).contains(probe).all())  # inside an infinite one
    # a +inf score stays +inf (it is not clamped to the largest float)
    y_inf = y.clone()
    y_inf[-1] = math.inf
    top = JointConformal(MaxResidualScore(), alpha=0.01).calibrate(pred, y_inf)
    assert math.isinf(top.thresholds_.item())


def test_mondrian_with_every_calibration_row_dropped() -> None:
    mean, tril, y = _problem(30, seed=54)
    pred = GaussianJoint(mean, tril)
    mask = torch.zeros(30, D, dtype=torch.bool)
    groups = torch.arange(30) % 3
    cp = JointConformal(MahalanobisScore(), alpha=0.1).calibrate(pred, y, mask=mask, groups=groups)
    assert cp.n_calibration_ == 0 and cp.group_ids_.numel() == 0
    region = cp.region(pred, groups=groups)
    assert bool(torch.isinf(region.threshold).all())
    assert bool(region.contains(y).all())


# ═══════════════════════════════════════════════════════════════════════════════
# edges: thresholds calibrated with a mask, singular covariances, target count
# ═══════════════════════════════════════════════════════════════════════════════


def _half_missing(n: int, d: int, seed: int) -> tuple[GaussianJoint, torch.Tensor, torch.Tensor]:
    """A correct ``N(0, I_d)`` predictive, its targets, and a mask with half the targets missing."""
    gen = torch.Generator().manual_seed(seed)
    pred = GaussianJoint(torch.zeros(n, d, dtype=F64), torch.eye(d, dtype=F64).expand(n, d, d))
    y = torch.randn(n, d, generator=gen, dtype=F64)
    mask = torch.rand(n, d, generator=gen) < 0.5
    return pred, y, mask


@pytest.mark.parametrize("score", [DensityScore, MaxResidualScore])
def test_a_mask_calibrated_threshold_is_not_used_for_complete_rows(
    score: type[ConformalScore],
) -> None:
    # These scores grow with the number of observed targets.  Before the guard, the 90%
    # region below covered 33% (density) and 84% (max residual) of complete test rows.
    pred, y, mask = _half_missing(4000, 5, seed=30)
    test, y_test, mask_test = _half_missing(4000, 5, seed=31)
    assert not score.mask_comparable
    region = JointConformal(score(), alpha=0.1).calibrate(pred, y, mask=mask).region(test)
    assert not region.covers_complete_rows
    assert JointConformal(score(), alpha=0.1).calibrate(pred, y).region(test).covers_complete_rows
    for call in (
        lambda: region.contains(y_test),
        lambda: region.log_volume(),
        lambda: region.sample(4),
        lambda: region.bounding_box(),
    ):
        with pytest.raises(ValueError, match="calibrated on rows with missing targets"):
            call()
    # With the mask of the test rows the guarantee holds, as before.
    inside = region.contains(y_test, mask=mask_test)[mask_test.any(-1)]
    sd = _coverage_sd(int(mask.any(-1).sum()), int(inside.numel()), 0.1)
    assert abs(float(inside.double().mean()) - 0.9) <= 4 * sd
    # The threshold really is too small for complete rows: this is what the guard refuses.
    complete = torch.ones_like(mask_test)
    assert float(region.contains(y_test, mask=complete).double().mean()) < 0.86


def test_mahalanobis_keeps_one_scale_across_missingness_patterns() -> None:
    pred, y, mask = _half_missing(4000, 5, seed=32)
    test, y_test, _ = _half_missing(4000, 5, seed=33)
    assert MahalanobisScore.mask_comparable
    region = (
        JointConformal(MahalanobisScore(), alpha=0.1).calibrate(pred, y, mask=mask).region(test)
    )
    assert region.covers_complete_rows
    coverage = float(region.contains(y_test).double().mean())  # complete rows, no mask
    assert abs(coverage - 0.9) <= 4 * _coverage_sd(int(mask.any(-1).sum()), 4000, 0.1), coverage
    assert bool(torch.isfinite(region.log_volume()[0]).all())


def _constant_target() -> tuple[SampleJoint, torch.Tensor, torch.Tensor]:
    """Draws whose third target is the same in every draw: a singular covariance."""
    gen = torch.Generator().manual_seed(34)
    draws = torch.randn(50, 40, 3, generator=gen)
    draws[:, :, 2] = 1.0
    y = torch.randn(50, 3, generator=gen)
    y[:, 2] = 1.0
    mask = torch.ones(50, 3, dtype=torch.bool)
    mask[::2, 0] = False
    return SampleJoint(draws), y, mask


def test_mahalanobis_with_a_mask_handles_a_singular_covariance() -> None:
    # The unmasked path adds jitter to rows that do not factorise; the masked path used to
    # raise "total covariance must have a positive finite diagonal".
    pred, y, mask = _constant_target()
    plain = JointConformal(MahalanobisScore(), alpha=0.1).calibrate(pred, y)
    masked = JointConformal(MahalanobisScore(), alpha=0.1).calibrate(pred, y, mask=mask)
    assert bool(torch.isfinite(plain.thresholds_).all())
    assert bool(torch.isfinite(masked.thresholds_).all()) and 0 < float(masked.thresholds_) < 1
    region = masked.region(pred)
    assert region.contains(y, mask=mask).double().mean() > 0.8
    # a complete row is scored on the same jittered matrix by both paths
    complete = torch.ones_like(mask)
    radius = MahalanobisScore().score(pred, y)
    level = MahalanobisScore().score(pred, y, mask=complete)
    want = torch.special.gammainc(torch.tensor(1.5), 0.5 * radius.square())
    torch.testing.assert_close(level, want, rtol=1e-4, atol=1e-6)


def test_max_residual_ignores_a_target_with_zero_spread_and_zero_residual() -> None:
    # 0 / 0 was NaN, counted as +inf: every threshold infinite, the region all of R^d.
    pred, y, mask = _constant_target()
    for kwargs in ({}, {"mask": mask}):
        cp = JointConformal(MaxResidualScore(), alpha=0.1).calibrate(pred, y, **kwargs)
        assert bool(torch.isfinite(cp.thresholds_).all())
    # With one set of moments: the constant target adds nothing to the maximum.
    score = MaxResidualScore()
    state = score._prepare(pred)
    without_third = torch.ones_like(mask)
    without_third[:, 2] = False
    scores = score._evaluate(pred, y, None, state)
    assert bool(torch.isfinite(scores).all())
    torch.testing.assert_close(scores, score._evaluate(pred, y, without_third, state))
    off = y.clone()
    off[:3, 2] = 1.5  # a residual where no spread is predicted: infinitely atypical
    assert bool(torch.isposinf(score._evaluate(pred, off, None, state)[:3]).all())


def test_region_refuses_a_predictive_with_another_number_of_targets() -> None:
    mean, tril, y = _problem(200, seed=35)
    cp = JointConformal(DensityScore(), alpha=0.1).calibrate(GaussianJoint(mean, tril), y)
    fewer = GaussianJoint(mean[:, :2], tril[:, :2, :2])
    with pytest.raises(ValueError, match="calibrated on 3 targets, but the predictive has 2"):
        cp.region(fewer)
    assert cp.region(GaussianJoint(mean[:5], tril[:5])).contains(y[:5]).shape == (5,)


def test_an_infinite_test_weight_gives_an_infinite_threshold() -> None:
    mean, tril, y = _problem(300, seed=36)
    pred = GaussianJoint(mean, tril)
    weights = torch.ones(300, dtype=F64)
    cp = JointConformal(MaxResidualScore(), alpha=0.1).calibrate(pred, y, weights=weights)
    test = GaussianJoint(mean[:3], tril[:3])
    region = cp.region(test, test_weights=torch.tensor([1.0, math.inf, 1e12], dtype=F64))
    reference = cp.region(test, test_weights=torch.tensor([1.0, 1.0, 1.0], dtype=F64))
    assert float(region.threshold[0, 0]) == float(reference.threshold[0, 0])
    assert math.isinf(float(region.threshold[1, 0]))  # the limit of large finite weights
    assert math.isinf(float(region.threshold[2, 0]))
    with pytest.raises(ValueError, match="non-negative"):
        cp.region(test, test_weights=torch.tensor([1.0, math.nan, 1.0], dtype=F64))
    with pytest.raises(ValueError, match="non-negative"):
        cp.region(test, test_weights=torch.tensor([1.0, -1.0, 1.0], dtype=F64))
