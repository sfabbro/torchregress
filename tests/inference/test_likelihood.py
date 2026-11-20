"""Tests for torchregress.inference.likelihood (MLE and profile likelihood intervals)."""

from __future__ import annotations

import math
import os

import pytest
import torch
from scipy.optimize import brentq
from scipy.stats import chi2, norm

from torchregress.inference import (
    ProfileInterval,
    ScoreMaximum,
    flow_score,
    likelihood_ratio_statistic,
    maximize_score,
    profile_interval,
    ratio_classifier_score,
    wilks_threshold,
)

F64 = torch.float64
Z975 = float(norm.ppf(0.975))


# ─── Gaussian linear model ──────────────────────────────────────────────────


def _gaussian_linear(  # type: ignore[no-untyped-def]
    dtype: torch.dtype, n_rows: int = 48, seed: int = 0, device: torch.device | str = "cpu"
):
    """x = A theta + eps, eps ~ N(0, Sigma), P = 3, x of dimension 6."""
    g = torch.Generator().manual_seed(seed)
    p, d = 3, 6
    design = torch.randn(d, p, generator=g, dtype=F64)
    root = 0.3 * torch.randn(d, d, generator=g, dtype=F64) + torch.eye(d, dtype=F64)
    sigma = root @ root.T + 0.1 * torch.eye(d, dtype=F64)
    precision = torch.linalg.inv(sigma)
    theta = torch.randn(n_rows, p, generator=g, dtype=F64)
    noise = torch.randn(n_rows, d, generator=g, dtype=F64) @ torch.linalg.cholesky(sigma).T
    x = theta @ design.T + noise
    fisher = design.T @ precision @ design
    gls = torch.linalg.solve(fisher, (x @ precision @ design).T).T
    std = torch.linalg.inv(fisher).diagonal().sqrt()
    a_t, prec_t = design.to(device, dtype), precision.to(device, dtype)

    def score(theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        resid = x - theta @ a_t.T
        return -0.5 * ((resid @ prec_t) * resid).sum(-1)

    return score, x.to(device, dtype), gls, std


BOX3 = [[-20.0, 20.0]] * 3


def test_mle_equals_gls_float64() -> None:
    score, x, gls, _ = _gaussian_linear(F64)
    result = maximize_score(score, x, BOX3, generator=torch.Generator().manual_seed(0))
    assert isinstance(result, ScoreMaximum)
    theta_hat, s_max, at_bound = result
    torch.testing.assert_close(theta_hat, gls, rtol=0, atol=1e-10)
    torch.testing.assert_close(s_max, score(gls, x), rtol=0, atol=1e-10)
    assert not bool(at_bound.any())
    assert theta_hat.dtype == F64 and s_max.shape == (x.shape[0],)


@pytest.mark.parametrize(("threshold", "n_sigma"), [(0.5, 1.0), (0.5 * Z975**2, Z975)])
@pytest.mark.parametrize("index", [0, 1, -1])
def test_profile_interval_equals_gaussian_interval(
    threshold: float, n_sigma: float, index: int
) -> None:
    score, x, gls, std = _gaussian_linear(F64)
    theta_hat, s_max, _ = maximize_score(score, x, BOX3, generator=torch.Generator().manual_seed(0))
    interval = profile_interval(score, x, theta_hat, s_max, index, threshold, BOX3)
    assert isinstance(interval, ProfileInterval)
    j = index % 3
    half = n_sigma * std[j]
    torch.testing.assert_close(interval.lower, gls[:, j] - half, rtol=0, atol=1e-8)
    torch.testing.assert_close(interval.upper, gls[:, j] + half, rtol=0, atol=1e-8)
    assert not bool(interval.lower_at_bound.any() or interval.upper_at_bound.any())


def test_profile_interval_per_row_threshold() -> None:
    score, x, gls, std = _gaussian_linear(F64, n_rows=8)
    theta_hat, s_max, _ = maximize_score(score, x, BOX3, generator=torch.Generator().manual_seed(0))
    level = torch.linspace(0.6, 0.99, 8, dtype=F64)
    z = torch.as_tensor(norm.ppf(0.5 + 0.5 * level.numpy()), dtype=F64)
    threshold = torch.tensor([wilks_threshold(float(c)) for c in level], dtype=F64)
    lower, upper, _, _ = profile_interval(score, x, theta_hat, s_max, 2, threshold, BOX3)
    torch.testing.assert_close(upper - lower, 2.0 * z * std[2], rtol=0, atol=1e-7)


def test_float32_gaussian() -> None:
    score, x, gls, std = _gaussian_linear(torch.float32)
    theta_hat, s_max, _ = maximize_score(score, x, BOX3, generator=torch.Generator().manual_seed(0))
    assert theta_hat.dtype == torch.float32 and s_max.dtype == torch.float32
    torch.testing.assert_close(theta_hat.double(), gls, rtol=0, atol=2e-3)
    lower, upper, _, _ = profile_interval(score, x, theta_hat, s_max, 0, 0.5, BOX3)
    assert lower.dtype == torch.float32
    torch.testing.assert_close(lower.double(), gls[:, 0] - std[0], rtol=0, atol=5e-3)
    torch.testing.assert_close(upper.double(), gls[:, 0] + std[0], rtol=0, atol=5e-3)


def test_deterministic_given_generator() -> None:
    def bumpy(theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.cos(3.0 * (theta - x)).sum(-1) - 0.1 * ((theta - x) ** 2).sum(-1)

    x = torch.randn(32, 2, generator=torch.Generator().manual_seed(3), dtype=F64)
    box = [[-4.0, 4.0]] * 2
    first = maximize_score(bumpy, x, box, generator=torch.Generator().manual_seed(7), chunk_size=97)
    second = maximize_score(bumpy, x, box, generator=torch.Generator().manual_seed(7))
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)
    one = profile_interval(bumpy, x, first.theta, first.score, 1, 1.0, box)
    two = profile_interval(bumpy, x, first.theta, first.score, 1, 1.0, box, chunk_size=13)
    for a, b in zip(one, two, strict=True):
        assert torch.equal(a, b)


def test_init_joins_the_starts() -> None:
    score, x, gls, _ = _gaussian_linear(F64, n_rows=4)
    result = maximize_score(score, x, BOX3, n_candidates=1, n_starts=1, steps=0, init=gls)
    torch.testing.assert_close(result.theta, gls, rtol=0, atol=1e-12)


# ─── multimodality ──────────────────────────────────────────────────────────

# Narrow global mode at x - 2 (weight 0.2, width 0.05), broad local mode at
# x + 2 (weight 0.8, width 1): peak densities 1.60 vs 0.32.
_W, _M, _S = (0.2, 0.8), (-2.0, 2.0), (0.05, 1.0)


def _two_bump(theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    t = (theta - x).squeeze(-1) if theta.dim() == 2 else theta - x
    t = t.reshape(t.shape[0], -1)[:, 0]
    comps = [
        math.log(w) - math.log(s) - 0.5 * math.log(2 * math.pi) - 0.5 * ((t - m) / s) ** 2
        for w, m, s in zip(_W, _M, _S, strict=True)
    ]
    return torch.logsumexp(torch.stack(comps, dim=-1), dim=-1)


def _two_bump_np(t: float) -> float:
    return math.log(sum(w * norm.pdf(t, m, s) for w, m, s in zip(_W, _M, _S, strict=True)))


BOX1 = [[-6.0, 6.0]]


def test_multistart_finds_narrow_global_mode() -> None:
    x = torch.linspace(-1.0, 1.0, 21, dtype=F64).unsqueeze(1)
    theta_hat, s_max, at_bound = maximize_score(
        _two_bump, x, BOX1, generator=torch.Generator().manual_seed(0)
    )
    grid = torch.linspace(-6.0, 6.0, 200_001, dtype=F64)
    dense = torch.stack([_two_bump(grid.unsqueeze(1), xi.expand(grid.numel(), 1)) for xi in x])
    assert bool((s_max >= dense.amax(dim=1) - 1e-12).all())
    torch.testing.assert_close(
        theta_hat[:, 0] - x[:, 0], torch.full((21,), -2.0, dtype=F64), rtol=0, atol=1e-2
    )
    assert not bool(at_bound.any())


def test_multistart_finds_global_mode_in_two_dimensions() -> None:
    def bumps(theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        narrow = math.log(0.3) - 2 * math.log(0.3) - 0.5 * (((theta - x + 2.0) / 0.3) ** 2).sum(-1)
        broad = math.log(0.7) - 0.5 * (((theta - x - 1.5) / 1.0) ** 2).sum(-1)
        return torch.logaddexp(narrow, broad)

    x = torch.randn(64, 2, generator=torch.Generator().manual_seed(1), dtype=F64) * 0.5
    theta_hat, _, _ = maximize_score(
        bumps, x, [[-6.0, 6.0]] * 2, generator=torch.Generator().manual_seed(2)
    )
    torch.testing.assert_close(theta_hat - x, torch.full_like(x, -2.0), rtol=0, atol=1e-2)


def test_bimodal_profile_returns_outermost_crossing() -> None:
    """The interval spans the dip between the narrow and the broad mode."""
    x = torch.tensor([[0.0], [0.5]], dtype=F64)
    threshold = 2.0
    theta_hat, s_max, _ = maximize_score(
        _two_bump, x, BOX1, generator=torch.Generator().manual_seed(0)
    )
    lower, upper, lo_flag, hi_flag = profile_interval(
        _two_bump, x, theta_hat, s_max, 0, threshold, BOX1
    )
    for n, shift in enumerate([0.0, 0.5]):
        peak = s_max[n].item()

        def gap(t: float, peak: float = peak) -> float:
            return _two_bump_np(t) - (peak - threshold)

        t_hat = theta_hat[n, 0].item() - shift
        ref_lower = brentq(gap, -6.0, t_hat, xtol=1e-14) + shift
        ref_upper = brentq(gap, 2.0, 6.0, xtol=1e-14) + shift  # broad-mode crossing
        assert gap(0.0) < 0  # the set is disconnected: the dip lies below threshold
        assert lower[n].item() == pytest.approx(ref_lower, abs=1e-7)
        assert upper[n].item() == pytest.approx(ref_upper, abs=1e-7)
    assert not bool(lo_flag.any() or hi_flag.any())


# ─── bounds ─────────────────────────────────────────────────────────────────


def _quadratic(centre_scale: float = 1.0):  # type: ignore[no-untyped-def]
    def score(theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return -0.5 * (((theta - x) / centre_scale) ** 2).sum(-1)

    return score


def test_mle_on_bound_is_flagged_and_interval_clipped() -> None:
    score = _quadratic()
    box = [[-3.0, 2.0], [-5.0, 5.0]]
    x = torch.tensor([[3.0, 0.5], [0.0, 0.0]], dtype=F64)
    theta_hat, s_max, at_bound = maximize_score(score, x, box)
    torch.testing.assert_close(theta_hat, torch.tensor([[2.0, 0.5], [0.0, 0.0]], dtype=F64))
    assert at_bound.tolist() == [[True, False], [False, False]]
    assert theta_hat[0, 0].item() == 2.0  # exactly on the bound

    c = 0.5
    lower, upper, lo_flag, hi_flag = profile_interval(score, x, theta_hat, s_max, 0, c, box)
    # Row 0: (t - 3)^2 = 1 + 2c on the lower side; the upper side is the bound.
    assert lower[0].item() == pytest.approx(3.0 - math.sqrt(1.0 + 2.0 * c), abs=1e-9)
    assert upper[0].item() == 2.0 and bool(hi_flag[0]) and not bool(lo_flag[0])
    # Row 1: interior +-1, unflagged.
    assert lower[1].item() == pytest.approx(-1.0, abs=1e-9)
    assert upper[1].item() == pytest.approx(1.0, abs=1e-9)
    assert not bool(lo_flag[1] or hi_flag[1])


def test_interior_mle_with_interval_reaching_bound() -> None:
    score = _quadratic(0.5)
    x = torch.tensor([[1.8]], dtype=F64)
    theta_hat, s_max, at_bound = maximize_score(score, x, [[-1.0, 2.0]])
    assert not bool(at_bound.any())
    lower, upper, lo_flag, hi_flag = profile_interval(
        score, x, theta_hat, s_max, 0, 0.5, [[-1.0, 2.0]]
    )
    assert lower.item() == pytest.approx(1.3, abs=1e-9) and not bool(lo_flag)
    assert upper.item() == 2.0 and bool(hi_flag)


# ─── likelihood-ratio statistic and Wilks threshold ─────────────────────────


def test_likelihood_ratio_statistic_shapes_and_clamp() -> None:
    score = _quadratic()
    x = torch.tensor([[0.0, 0.0], [1.0, -1.0], [2.0, 2.0]], dtype=F64)
    s_max = torch.zeros(3, dtype=F64)
    theta0 = x + torch.tensor([1.0, 0.0], dtype=F64)
    torch.testing.assert_close(
        likelihood_ratio_statistic(score, x, theta0, s_max), torch.full((3,), 0.5, dtype=F64)
    )
    bank = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=F64)
    stat = likelihood_ratio_statistic(score, x, bank[None], s_max, chunk_size=2)
    assert stat.shape == (3, 2)
    torch.testing.assert_close(stat, 0.5 * ((bank[None] - x[:, None]) ** 2).sum(-1))
    raw = likelihood_ratio_statistic(score, x, x, s_max - 1.0, clamp=False)
    torch.testing.assert_close(raw, torch.full((3,), -1.0, dtype=F64))
    assert bool((likelihood_ratio_statistic(score, x, x, s_max - 1.0) == 0).all())


def test_wilks_threshold_matches_chi_square() -> None:
    assert wilks_threshold(0.6826894921370859) == pytest.approx(0.5, abs=1e-12)
    for confidence, df in [(0.95, 1), (0.9, 2), (0.99, 5)]:
        expected = 0.5 * chi2.ppf(confidence, df)
        assert wilks_threshold(confidence, df) == pytest.approx(expected, rel=1e-12)
    with pytest.raises(ValueError):
        wilks_threshold(1.0)
    with pytest.raises(ValueError):
        wilks_threshold(0.9, df=0)


def test_input_validation() -> None:
    score = _quadratic()
    x = torch.zeros(2, 1, dtype=F64)
    with pytest.raises(ValueError, match="lower < upper"):
        maximize_score(score, x, [[1.0, 0.0]])
    with pytest.raises(ValueError, match="n_starts"):
        maximize_score(score, x, [[0.0, 1.0]], n_candidates=2, n_starts=3)
    with pytest.raises(TypeError, match="floating"):
        maximize_score(score, torch.zeros(2, 1, dtype=torch.long), [[0.0, 1.0]])
    with pytest.raises(ValueError, match="one value per row"):
        maximize_score(lambda t, x: t, torch.zeros(2, 2, dtype=F64), [[0.0, 1.0]] * 2)
    theta_hat, s_max, _ = maximize_score(score, x, [[-1.0, 1.0]])
    with pytest.raises(ValueError, match="threshold"):
        profile_interval(score, x, theta_hat, s_max, 0, -1.0, [[-1.0, 1.0]])
    with pytest.raises(ValueError, match="index"):
        profile_interval(score, x, theta_hat, s_max, 1, 0.5, [[-1.0, 1.0]])


# ─── adapters ───────────────────────────────────────────────────────────────


def test_ratio_classifier_score_tracks_analytic_mle() -> None:
    """A small CNRE estimator on x = theta + N(0, 0.5^2) gives MLE ~ x."""
    from torchregress.losses import NeuralRatioLoss
    from torchregress.losses.nre import build_ratio_classifier

    torch.manual_seed(0)
    sigma = 0.5
    classifier = build_ratio_classifier(1, 1, hidden_dims=(32, 32))
    loss_fn = NeuralRatioLoss(classifier, criterion="cnre")
    optimiser = torch.optim.Adam(classifier.parameters(), lr=3e-3)
    for _ in range(400):
        theta = 4.0 * torch.rand(256, 1) - 2.0
        x = theta + sigma * torch.randn(256, 1)
        negatives = 4.0 * torch.rand(256, 4, 1) - 2.0
        loss = loss_fn(theta, x, negative_context=negatives)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    classifier.eval()

    x_test = torch.linspace(-1.0, 1.0, 64).unsqueeze(1)
    for module in (classifier, loss_fn):  # module or the loss holding it
        theta_hat, s_max, _ = maximize_score(
            ratio_classifier_score(module),
            x_test,
            [[-2.0, 2.0]],
            generator=torch.Generator().manual_seed(0),
        )
        assert torch.isfinite(s_max).all()
        corr = torch.corrcoef(torch.stack([theta_hat[:, 0], x_test[:, 0]]))[0, 1]
        assert corr > 0.9
        assert (theta_hat - x_test).abs().median() < 0.3
    lower, upper, _, _ = profile_interval(
        ratio_classifier_score(classifier), x_test, theta_hat, s_max, 0, 0.5, [[-2.0, 2.0]]
    )
    assert bool((lower <= theta_hat[:, 0]).all() and (upper >= theta_hat[:, 0]).all())


def test_flow_score_on_contrastive_flow_loss() -> None:
    nflows = pytest.importorskip("torchregress.losses.nflows")
    if not nflows.HAS_ZUKO:
        pytest.skip("zuko is not installed")
    torch.manual_seed(0)
    flow = nflows.create_flow_model(
        2, context_dim=2, flow_type="maf", n_transforms=2, hidden_features=16
    )
    loss = nflows.ContrastiveFlowLoss(flow)
    theta = torch.randn(5, 2)
    x = torch.randn(5, 2)
    expected = flow(theta).log_prob(x)
    torch.testing.assert_close(flow_score(loss)(theta, x), expected)
    torch.testing.assert_close(flow_score(flow)(theta, x), expected)
    box = [[-2.0, 2.0]] * 2
    theta_hat, s_max, _ = maximize_score(
        flow_score(loss), x, box, n_candidates=32, generator=torch.Generator().manual_seed(0)
    )
    assert torch.isfinite(s_max).all()
    lower, upper, _, _ = profile_interval(flow_score(loss), x, theta_hat, s_max, 0, 0.5, box)
    assert bool((lower <= theta_hat[:, 0] + 1e-5).all() and (upper >= theta_hat[:, 0] - 1e-5).all())


def test_non_finite_region_is_avoided() -> None:
    """A score that is NaN on part of the box (log of a negative rate)."""

    def poisson(theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return (x * torch.log(theta) - theta).sum(-1)

    x = torch.tensor([[0.5], [2.0], [3.5]], dtype=F64)
    box = [[-1.0, 6.0]]
    theta_hat, s_max, _ = maximize_score(
        poisson, x, box, generator=torch.Generator().manual_seed(0)
    )
    torch.testing.assert_close(theta_hat, x, rtol=0, atol=1e-9)
    lower, upper, lo_flag, hi_flag = profile_interval(poisson, x, theta_hat, s_max, 0, 0.5, box)
    for n, k in enumerate([0.5, 2.0, 3.5]):

        def gap(t: float, k: float = k) -> float:
            return k * math.log(t) - t - (k * math.log(k) - k - 0.5)

        assert lower[n].item() == pytest.approx(brentq(gap, 1e-12, k, xtol=1e-14), abs=1e-7)
        assert upper[n].item() == pytest.approx(brentq(gap, k, 6.0, xtol=1e-14), abs=1e-7)
    assert not bool(lo_flag.any() or hi_flag.any())


@pytest.mark.cuda
def test_accelerator_matches_cpu() -> None:
    """Same estimates on the accelerator (``TORCHREGRESS_PARITY_DEVICE``, default cuda)."""
    device = torch.device(os.environ.get("TORCHREGRESS_PARITY_DEVICE", "cuda"))
    box = torch.tensor(BOX3)
    results = []
    for target in (torch.device("cpu"), device):
        score, x, _, _ = _gaussian_linear(torch.float32, n_rows=16, device=target)
        best = maximize_score(score, x, box.to(target), generator=torch.Generator().manual_seed(0))
        iv = profile_interval(score, x, best.theta, best.score, 1, 0.5, box.to(target))
        assert best.theta.device.type == target.type and iv.lower.device.type == target.type
        results.append([best.theta.cpu(), iv.lower.cpu(), iv.upper.cpu()])
    for on_cpu, on_device in zip(*results, strict=True):
        torch.testing.assert_close(on_device, on_cpu, rtol=0, atol=5e-3)
