"""Per-target marginal adapters: CDF / inverse round trips, densities, tails, repairs."""

from __future__ import annotations

import math

import pytest
import torch
from torch.distributions import Normal

from torchregress.joint.marginals import (
    BinnedMarginals,
    GaussianMarginals,
    Marginals,
    QuantileMarginals,
    SampleMarginals,
)

F64 = torch.float64
N, D = 4, 3


def _gaussian() -> GaussianMarginals:
    gen = torch.Generator().manual_seed(0)
    return GaussianMarginals(
        torch.randn(N, D, generator=gen, dtype=F64),
        torch.rand(N, D, generator=gen, dtype=F64) + 0.3,
    )


def _quantile(crossing: bool = False, ties: bool = False) -> QuantileMarginals:
    gen = torch.Generator().manual_seed(1)
    levels = torch.linspace(0.05, 0.95, 19, dtype=F64)
    base = Normal(torch.zeros((), dtype=F64), torch.ones((), dtype=F64)).icdf(levels)
    loc = torch.randn(N, D, 1, generator=gen, dtype=F64)
    scale = torch.rand(N, D, 1, generator=gen, dtype=F64) + 0.5
    q = loc + scale * base**3 / 2 + scale * base  # skewed, heavier than normal
    if crossing:
        q = q + 0.4 * torch.randn(q.shape, generator=gen, dtype=F64)
    if ties:
        q[..., 5:9] = q[..., 5:6]
        q[..., -3:] = q[..., -3:-2]
    return QuantileMarginals(levels, q)


def _binned(shared_edges: bool = False) -> BinnedMarginals:
    gen = torch.Generator().manual_seed(2)
    n_bins = 12
    if shared_edges:
        edges = torch.linspace(-3.0, 4.0, n_bins + 1, dtype=F64)
    else:
        widths = torch.rand(N, D, n_bins, generator=gen, dtype=F64) + 0.1
        start = torch.randn(N, D, 1, generator=gen, dtype=F64) - 2.0
        edges = torch.cat([start, start + widths.cumsum(-1)], dim=-1)
    probs = torch.rand(N, D, n_bins, generator=gen, dtype=F64)
    probs[..., 3] = 0.0  # an empty bin
    probs = probs / probs.sum(-1, keepdim=True)
    return BinnedMarginals(edges, probs)


def _samples() -> SampleMarginals:
    gen = torch.Generator().manual_seed(3)
    draws = torch.distributions.StudentT(torch.tensor(4.0, dtype=F64)).sample((N, 300, D))
    draws = draws * (torch.rand(N, 1, D, generator=gen, dtype=F64) + 0.5)
    return SampleMarginals(torch.round(draws * 4) / 4)  # rounding creates many ties


ADAPTERS = {
    "gaussian": _gaussian,
    "quantile": _quantile,
    "quantile_crossing": lambda: _quantile(crossing=True),
    "quantile_ties": lambda: _quantile(ties=True),
    "binned": _binned,
    "binned_shared_edges": lambda: _binned(shared_edges=True),
    "samples": _samples,
}


def _knots(marg: Marginals) -> torch.Tensor | None:
    """Break points ``[N, D, K]`` of a piecewise adapter (``None`` for a smooth CDF)."""
    knots = getattr(marg, "knots", None)
    return knots if isinstance(knots, torch.Tensor) else None


@pytest.fixture(params=sorted(ADAPTERS))
def marg(request: pytest.FixtureRequest) -> Marginals:
    torch.manual_seed(0)
    return ADAPTERS[request.param]()


def _u_grid(m: int = 201) -> torch.Tensor:
    """[N, m, D] probabilities from 1e-30 (far left tail) to 1 - 1e-9 (far right tail)."""
    inner = torch.linspace(1e-4, 1 - 1e-4, m - 6, dtype=F64)
    tails = torch.tensor([1e-30, 1e-15, 1e-8, 1 - 1e-6, 1 - 1e-8, 1 - 1e-9], dtype=F64)
    u = torch.sort(torch.cat([inner, tails])).values
    return u.view(1, -1, 1).expand(N, -1, D).contiguous()


class TestMarginalProperties:
    def test_round_trip_u_to_y_to_u(self, marg: Marginals) -> None:
        u = _u_grid()
        y = marg.icdf(u)
        assert torch.isfinite(y).all()
        torch.testing.assert_close(marg.cdf(y), u, rtol=1e-9, atol=1e-12)

    def test_round_trip_y_to_u_to_y_including_far_tails(self, marg: Marginals) -> None:
        lo = marg.icdf(torch.full((N, D), 1e-12, dtype=F64))
        hi = marg.icdf(torch.full((N, D), 1 - 1e-7, dtype=F64))
        t = torch.linspace(0.0, 1.0, 401, dtype=F64).view(1, -1, 1)
        y = lo.unsqueeze(1) + t * (hi - lo).unsqueeze(1)
        torch.testing.assert_close(marg.icdf(marg.cdf(y)), y, rtol=1e-7, atol=1e-7)

    def test_cdf_strictly_increasing_and_density_positive(self, marg: Marginals) -> None:
        y = marg.icdf(_u_grid(2001))
        assert (y.diff(dim=1) > 0).all()
        assert (marg.cdf(y).diff(dim=1) > 0).all()
        far = torch.cat([y[:, :1] - 50.0, y[:, -1:] + 50.0], dim=1)
        assert torch.isfinite(marg.log_prob(far)).all()

    def test_log_prob_is_log_derivative_of_cdf(self, marg: Marginals) -> None:
        knots = _knots(marg)
        if knots is None:
            # smooth CDF: central differences on the bulk, step 1e-5
            y = marg.icdf(_u_grid(401)[:, 3:-3])
            h = torch.full_like(y, 1e-5)
            rtol = 1e-8
        else:
            # piecewise: segment mid-points and tail points, step 1e-3 of the local
            # width (the density is constant on a segment, exponential in a tail)
            mids = 0.5 * (knots[..., 1:] + knots[..., :-1])
            widths = knots.diff(dim=-1)
            left = knots[..., :1] - torch.tensor([0.5, 3.0, 20.0], dtype=F64) * widths[..., :1]
            right = knots[..., -1:] + torch.tensor([0.5, 3.0, 20.0], dtype=F64) * widths[..., -1:]
            y = torch.cat([left, mids, right], dim=-1).movedim(-1, 1)
            tail_h = torch.cat(
                [widths[..., :1].expand(-1, -1, 3), widths[..., -1:].expand(-1, -1, 3)], -1
            )
            h = 1e-3 * torch.cat([tail_h[..., :3], widths, tail_h[..., 3:]], dim=-1).movedim(-1, 1)
            rtol = 1e-5
        num = (marg.cdf(y + h) - marg.cdf(y - h)) / (2 * h)
        dens = marg.log_prob(y).exp()
        # float64 rounding of the CDF difference, relative to the step
        atol = 4 * torch.finfo(F64).eps / h
        assert ((dens - num).abs() <= rtol * num + atol).all()

    def test_density_integrates_to_one(self, marg: Marginals) -> None:
        lo = marg.icdf(torch.full((N, D), 1e-13, dtype=F64))
        hi = marg.icdf(torch.full((N, D), 1 - 1e-13, dtype=F64))
        t = torch.linspace(0.0, 1.0, 200_001, dtype=F64).view(1, 1, -1)
        grid = lo.unsqueeze(-1) + t * (hi - lo).unsqueeze(-1)  # [N, D, M]
        # quantile points resolve tails whatever their scale; break points make the
        # midpoint rule exact on the piecewise-constant part
        tail_u = torch.logspace(-13, -1, 4000, dtype=F64)
        u = torch.cat([tail_u, 1 - tail_u]).view(1, -1, 1).expand(N, -1, D)
        extra = [marg.icdf(u).movedim(1, -1)]
        knots = _knots(marg)
        if knots is not None:
            extra.append(knots)
        grid = torch.sort(torch.cat([grid, *extra], dim=-1), dim=-1).values
        mid = (0.5 * (grid[..., 1:] + grid[..., :-1])).movedim(-1, 1)
        mass = (marg.log_prob(mid).exp() * grid.diff(dim=-1).movedim(-1, 1)).sum(1)
        torch.testing.assert_close(mass, torch.ones(N, D, dtype=F64), rtol=0, atol=1e-6)

    def test_layouts_agree(self, marg: Marginals) -> None:
        y = marg.icdf(_u_grid(7))
        for fn in (marg.cdf, marg.log_prob, marg.normal_scores):
            full = fn(y)
            assert full.shape == y.shape
            torch.testing.assert_close(fn(y[:, 3]), full[:, 3])

    def test_normal_scores_are_clipped_and_invertible(self, marg: Marginals) -> None:
        y = marg.icdf(_u_grid(101))
        z = marg.normal_scores(y)
        assert torch.isfinite(z).all()
        if not isinstance(marg, GaussianMarginals):
            assert z.abs().max() <= -torch.special.ndtri(torch.tensor(1e-6, dtype=F64)) + 1e-12
        inner = (z.abs() < 4.0) & (y.abs() < 1e6)
        back = marg.from_normal_scores(z)
        torch.testing.assert_close(back[inner], y[inner], rtol=1e-7, atol=1e-7)

    def test_select_targets(self, marg: Marginals) -> None:
        y = marg.icdf(_u_grid(9))
        sub = marg.select_targets([2, 0])
        assert type(sub) is type(marg) and sub.event_dim == 2
        torch.testing.assert_close(sub.cdf(y[..., [2, 0]]), marg.cdf(y)[..., [2, 0]])
        torch.testing.assert_close(sub.log_prob(y[..., [2, 0]]), marg.log_prob(y)[..., [2, 0]])

    def test_median(self, marg: Marginals) -> None:
        torch.testing.assert_close(marg.cdf(marg.median()), torch.full((N, D), 0.5, dtype=F64))


class TestQuantileMarginals:
    def test_reproduces_known_cdf_between_knots(self) -> None:
        levels = torch.linspace(0.02, 0.98, 49, dtype=F64)
        loc = torch.tensor([[0.0, 2.0]], dtype=F64)
        scale = torch.tensor([[1.0, 0.5]], dtype=F64)
        ref = Normal(loc.unsqueeze(-1), scale.unsqueeze(-1))
        marg = QuantileMarginals(levels, ref.icdf(levels))
        # exact at the knots
        torch.testing.assert_close(
            marg.cdf(ref.icdf(levels)[0].mT.unsqueeze(0)), levels.view(1, -1, 1).expand(1, -1, 2)
        )
        # between the knots: linear interpolation error <= max|F''| h^2 / 8
        y = loc.unsqueeze(1) + scale.unsqueeze(1) * torch.linspace(-2.0, 2.0, 999, dtype=F64).view(
            1, -1, 1
        )
        exact = Normal(loc.unsqueeze(1), scale.unsqueeze(1)).cdf(y)
        h = (ref.icdf(levels).diff(dim=-1)).amax(-1)  # [1, 2] largest knot gap
        bound = math.exp(-0.5) / math.sqrt(2 * math.pi) / scale**2 * h**2 / 8
        err = (marg.cdf(y) - exact).abs().amax(1)
        assert (err <= bound + 1e-12).all(), (err, bound)

    def test_crossing_quantiles_are_sorted(self) -> None:
        levels = torch.tensor([0.1, 0.5, 0.9], dtype=F64)
        sorted_q = torch.tensor([[[-1.0, 0.0, 1.0]]], dtype=F64)
        crossed = torch.tensor([[[0.0, 1.0, -1.0]]], dtype=F64)
        y = torch.linspace(-3, 3, 13, dtype=F64).view(1, -1, 1)
        a = QuantileMarginals(levels, sorted_q, tail_knots=2)
        b = QuantileMarginals(levels, crossed, tail_knots=2)
        torch.testing.assert_close(a.cdf(y), b.cdf(y))

    def test_tied_quantiles_become_strictly_increasing(self) -> None:
        levels = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], dtype=F64)
        q = torch.tensor([[[1.0, 1.0, 1.0, 1.0, 1.0]]], dtype=F64)  # a point mass
        marg = QuantileMarginals(levels, q)
        assert (marg.knots.diff(dim=-1) > 0).all()
        u = torch.linspace(0.01, 0.99, 99, dtype=F64).view(1, -1, 1)
        y = marg.icdf(u)
        assert (y.diff(dim=1) > 0).all() and torch.isfinite(marg.log_prob(y)).all()
        torch.testing.assert_close(marg.cdf(y), u)

    def test_tail_matches_exponential_through_outer_knots(self) -> None:
        levels = torch.tensor([0.1, 0.2, 0.8, 0.9], dtype=F64)
        q = torch.tensor([[[-2.0, -1.0, 1.0, 3.0]]], dtype=F64)
        marg = QuantileMarginals(levels, q, tail_knots=2)
        b_left = 1.0 / math.log(2.0)
        b_right = 2.0 / math.log(2.0)
        y = torch.tensor([[[-5.0]], [[6.0]]], dtype=F64).view(1, 2, 1)
        expect = torch.tensor([0.1 * math.exp(-3.0 / b_left), 1 - 0.1 * math.exp(-3.0 / b_right)])
        torch.testing.assert_close(marg.cdf(y).flatten(), expect.to(F64))

    def test_rejects_bad_levels(self) -> None:
        with pytest.raises(ValueError):
            QuantileMarginals(torch.tensor([0.0, 0.5], dtype=F64), torch.zeros(1, 1, 2, dtype=F64))
        with pytest.raises(ValueError):
            QuantileMarginals(torch.tensor([0.5, 0.4], dtype=F64), torch.zeros(1, 1, 2, dtype=F64))


class TestBinnedMarginals:
    def test_bin_masses_and_tail_mass(self) -> None:
        edges = torch.tensor([0.0, 1.0, 3.0, 4.0], dtype=F64)
        probs = torch.tensor([[[0.2, 0.5, 0.3]]], dtype=F64)
        marg = BinnedMarginals(edges, probs, tail_mass=0.01, min_prob=0.0)
        at_edges = marg.cdf(edges.view(1, -1, 1))[0, :, 0]
        expect = 0.005 + 0.99 * torch.tensor([0.0, 0.2, 0.7, 1.0], dtype=F64)
        torch.testing.assert_close(at_edges, expect)
        mid = torch.tensor([[[2.0]]], dtype=F64)
        torch.testing.assert_close(marg.log_prob(mid).exp().item(), 0.99 * 0.5 / 2.0)

    def test_rejects_unsorted_edges(self) -> None:
        with pytest.raises(ValueError):
            BinnedMarginals(
                torch.tensor([0.0, 2.0, 1.0], dtype=F64), torch.full((1, 1, 2), 0.5, dtype=F64)
            )


class TestSampleMarginals:
    def test_matches_distribution_of_draws(self) -> None:
        torch.manual_seed(5)
        draws = torch.randn(2, 4000, 2, dtype=F64)
        marg = SampleMarginals(draws)
        y = torch.linspace(-2.5, 2.5, 51, dtype=F64).view(1, -1, 1).expand(2, -1, 2)
        exact = Normal(0.0, 1.0).cdf(y)
        # Dvoretzky-Kiefer-Wolfowitz: sup |F_S - F| <= 0.03 with probability > 0.999 at S = 4000
        assert (marg.cdf(y) - exact).abs().max() < 0.03

    def test_layout_and_levels(self) -> None:
        draws = torch.arange(10, dtype=F64).view(1, 10, 1)
        marg = SampleMarginals(draws, tail_knots=2)
        torch.testing.assert_close(
            marg.cdf(draws), (torch.arange(1, 11, dtype=F64) / 11).view(1, 10, 1)
        )
