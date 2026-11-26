"""Post-hoc recalibration of joint predictives: scale, latent and marginal recalibrators."""

from __future__ import annotations

import math
from typing import Literal

import pytest
import torch
from scipy.stats import kstest
from torch.distributions import MultivariateNormal

from torchregress.joint.calibration import (
    LatentRecalibratedJoint,
    LatentRecalibrator,
    MarginalRecalibratedJoint,
    MarginalRecalibrator,
    ScaleRecalibrator,
    latent_calibration_error,
)
from torchregress.joint.predictive import (
    AffineJoint,
    DistributionJoint,
    FlowJoint,
    GaussianJoint,
    JointPredictive,
    MixtureJoint,
    SampleJoint,
)
from torchregress.losses.nflows import NormalizingFlowLoss, create_flow_model

F64 = torch.float64
D = 3


def _chol() -> torch.Tensor:
    a = torch.randn(D, D, generator=torch.Generator().manual_seed(99), dtype=F64)
    return torch.linalg.cholesky(a @ a.T / D + torch.eye(D, dtype=F64))


def _data(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows with mean mu_i ~ N(0, I) and targets y_i ~ N(mu_i, L L^T)."""
    gen = torch.Generator().manual_seed(seed)
    mean = torch.randn(n, D, generator=gen, dtype=F64)
    return mean, mean + torch.randn(n, D, generator=gen, dtype=F64) @ _chol().T


def _nll(pred: JointPredictive, y: torch.Tensor) -> float:
    with torch.no_grad():
        return -float(pred.log_prob(y).mean())


# Over-confident (covariance shrunk by 0.6^2), calibrated, under-confident (1.6^2).
FACTORS = {"over": 0.6, "calibrated": 1.0, "under": 1.6}


class TestScaleRecalibrator:
    N = 4000

    def _per_target(self) -> tuple[GaussianJoint, torch.Tensor, GaussianJoint, torch.Tensor]:
        # Per-target miscalibration: predictive scale_tril diag(0.5, 1, 2) L.
        dmat = torch.diag(torch.tensor([0.5, 1.0, 2.0], dtype=F64)).to(dtype=F64)
        m_c, y_c = _data(self.N, 1)
        m_t, y_t = _data(self.N, 2)
        return GaussianJoint(m_c, dmat @ _chol()), y_c, GaussianJoint(m_t, dmat @ _chol()), y_t

    def test_variance_objective_restores_unit_standardised_variance(self) -> None:
        cal, y_c, test, y_t = self._per_target()
        rec = ScaleRecalibrator().fit(cal, y_c)
        # s_j^2 estimates 1 / D_jj^2 with relative standard error sqrt(2 / n).
        se = math.sqrt(2 / self.N)
        target = torch.tensor([2.0, 1.0, 0.5], dtype=F64)
        assert rec.scale_ is not None
        assert float((rec.scale_.square() / target.square() - 1).abs().max()) < 4 * se
        out = rec.transform(test)
        assert isinstance(out, AffineJoint)
        assert out.has_density and out.has_latent and out.exact_marginals
        var = out.covariance().diagonal(dim1=-2, dim2=-1)
        z2 = ((y_t - out.mean()).square() / var).mean(0)
        # Test-set mean of chi^2_1 has se sqrt(2/n); calibration adds the same again.
        assert float((z2 - 1).abs().max()) < 4 * math.sqrt(2 * 2 / self.N), z2
        assert _nll(out, y_t) < _nll(test, y_t) - 0.1

    def test_nll_objective_agrees_and_maximises_the_calibration_likelihood(self) -> None:
        cal, y_c, test, y_t = self._per_target()
        var = ScaleRecalibrator("variance").fit(cal, y_c)
        nll = ScaleRecalibrator("nll").fit(cal, y_c)
        assert var.scale_ is not None and nll.scale_ is not None
        assert float((nll.scale_ / var.scale_ - 1).abs().max()) < 0.05
        assert _nll(nll.transform(cal), y_c) <= _nll(var.transform(cal), y_c) + 1e-9
        assert _nll(nll.transform(test), y_t) < _nll(test, y_t) - 0.1

    @pytest.mark.parametrize("objective", ["variance", "nll"])
    def test_identity_on_a_calibrated_predictive(self, objective: str) -> None:
        m_c, y_c = _data(self.N, 3)
        rec = ScaleRecalibrator(objective).fit(GaussianJoint(m_c, _chol()), y_c)
        assert rec.scale_ is not None
        # s has standard error ~ sqrt(1 / (2 n)).
        assert float((rec.scale_ - 1).abs().max()) < 4 * math.sqrt(1 / (2 * self.N))

    def test_mask_and_masked_density(self) -> None:
        cal, y_c, test, y_t = self._per_target()
        gen = torch.Generator().manual_seed(5)
        mask = torch.rand(self.N, D, generator=gen) > 0.3
        y_nan = torch.where(mask, y_c, torch.full_like(y_c, float("nan")))
        full = ScaleRecalibrator().fit(cal, y_c)
        part = ScaleRecalibrator("nll").fit(cal, y_nan, mask)
        assert full.scale_ is not None and part.scale_ is not None
        assert float((part.scale_ / full.scale_ - 1).abs().max()) < 0.1
        out = part.transform(test)
        sub = out.marginal([0, 2])
        m = torch.tensor([[True, False, True]]).expand(self.N, D)
        torch.testing.assert_close(out.log_prob(y_t, m), sub.log_prob(y_t[:, [0, 2]]))


class TestLatentRecalibrator:
    N = 2000

    def _pair(
        self, factor: float
    ) -> tuple[GaussianJoint, torch.Tensor, GaussianJoint, torch.Tensor]:
        m_c, y_c = _data(self.N, 10)
        m_t, y_t = _data(self.N, 11)
        return GaussianJoint(m_c, factor * _chol()), y_c, GaussianJoint(m_t, factor * _chol()), y_t

    @pytest.mark.parametrize("case", ["over", "under"])
    def test_fixes_miscalibrated_gaussians(self, case: str) -> None:
        torch.manual_seed(0)
        cal, y_c, test, y_t = self._pair(FACTORS[case])
        out = LatentRecalibrator().fit(cal, y_c).transform(test)
        before, after = latent_calibration_error(test, y_t), latent_calibration_error(out, y_t)
        # A calibrated predictive has E[L-ECE] ~ 0.31 sqrt(1/n_test + 1/n_cal) ~ 0.01 here.
        assert float(before) > 0.15
        assert float(after) < 0.03
        assert _nll(out, y_t) < _nll(test, y_t) - 0.2
        # HPD levels of the recalibrated predictive are uniform (n = 2000, 512 draws).
        levels = out.hpd_level(y_t)
        assert kstest(levels.numpy(), "uniform").pvalue > 0.01
        assert kstest(test.hpd_level(y_t).numpy(), "uniform").pvalue < 1e-6

    def test_identity_on_a_calibrated_predictive(self) -> None:
        cal, y_c, test, y_t = self._pair(1.0)
        out = LatentRecalibrator().fit(cal, y_c).transform(test)
        # The fitted r' has relative noise ~ sqrt(K / n) = sqrt(6 / 2000) ~ 0.055.
        assert float((out.log_prob(y_t) - test.log_prob(y_t)).abs().mean()) < 0.1
        assert abs(_nll(out, y_t) - _nll(test, y_t)) < 0.02
        assert float(latent_calibration_error(out, y_t)) < 0.03

    def test_density_integrates_to_one(self) -> None:
        # Importance sampling with a proposal three times wider than the base
        # (which covers the recalibrated predictive of an over-confident base).
        torch.manual_seed(1)
        cal, y_c, _, _ = self._pair(0.6)
        rows = GaussianJoint(cal.loc[:4], cal.scale_tril[:4])
        out = LatentRecalibrator().fit(cal, y_c).transform(rows)
        proposal = GaussianJoint(rows.loc, 3.0 * rows.scale_tril)
        draws = proposal.sample(100_000)
        w = (out.log_prob(draws) - proposal.log_prob(draws)).exp()
        mass, se = w.mean(1), w.std(1) / math.sqrt(w.shape[1])
        assert bool(((mass - 1).abs() < 4 * se).all()), (mass, se)

    def test_log_prob_matches_autograd_change_of_variables(self) -> None:
        cal, y_c, _, _ = self._pair(1.6)
        rows = GaussianJoint(cal.loc[:3], cal.scale_tril[:3])
        out = LatentRecalibrator(n_knots=8).fit(cal, y_c).transform(rows)
        z = torch.randn(3, D, generator=torch.Generator().manual_seed(4), dtype=F64)
        z[0] *= 3.0  # one point in the linear tail of the spline
        z[1] *= 0.05  # and one near the origin
        jac = torch.autograd.functional.jacobian(
            lambda flat: out.from_latent(flat.reshape(3, D)).reshape(-1), z.reshape(-1)
        )
        blocks = torch.stack([jac[i * D : (i + 1) * D, i * D : (i + 1) * D] for i in range(3)])
        log_det = torch.linalg.slogdet(blocks).logabsdet
        ref = -0.5 * z.square().sum(-1) - 0.5 * D * math.log(2 * math.pi) - log_det
        torch.testing.assert_close(out.log_prob(out.from_latent(z)), ref)

    def test_latent_round_trip_and_flags(self) -> None:
        cal, y_c, test, _ = self._pair(0.6)
        out = LatentRecalibrator().fit(cal, y_c).transform(test)
        assert isinstance(out, LatentRecalibratedJoint)
        assert out.has_density and out.has_latent and not out.exact_marginals
        z = torch.randn(self.N, 5, D, dtype=F64)
        torch.testing.assert_close(out.to_latent(out.from_latent(z)), z)
        torch.testing.assert_close(out.to_latent(out.from_latent(z[:, 0])), z[:, 0])
        assert out.sample(7).shape == (self.N, 7, D)
        assert out.log_prob(out.sample(4)).shape == (self.N, 4)
        with pytest.raises(NotImplementedError):
            out.log_prob(z[:, 0], torch.ones(self.N, D, dtype=torch.bool))
        # A recalibrated predictive can be recalibrated again.
        again = LatentRecalibrator().fit(out, y_c[: self.N]).transform(out)
        assert again.has_latent and again.has_density

    def test_flow(self) -> None:
        # Untrained conditional flow, data from N(1, 3^2 I) instead of the flow.
        torch.manual_seed(0)
        flow = create_flow_model(n_features=2, context_dim=3, flow_type="maf", n_transforms=2)
        loss = NormalizingFlowLoss(flow=flow)
        gen = torch.Generator().manual_seed(7)
        n = 1000
        ctx_c, ctx_t = torch.randn(n, 3, generator=gen), torch.randn(n, 3, generator=gen)
        y_c = 1.0 + 3.0 * torch.randn(n, 2, generator=gen)
        y_t = 1.0 + 3.0 * torch.randn(n, 2, generator=gen)
        cal, test = FlowJoint(loss, ctx_c), FlowJoint(loss, ctx_t)
        out = LatentRecalibrator().fit(cal, y_c).transform(test)
        assert out.has_density and out.has_latent
        before, after = latent_calibration_error(test, y_t), latent_calibration_error(out, y_t)
        assert float(before) > 0.1
        assert float(after) < 0.04  # ~ 3x E[L-ECE] = 0.31 sqrt(2 / 1000) = 0.014
        with torch.no_grad():
            lp = out.log_prob(y_t)
        assert lp.dtype == torch.float32 and bool(lp.isfinite().all())
        assert float(lp.mean()) > float(test.log_prob(y_t).detach().mean())
        assert out.sample(3).shape == (n, 3, 2)


class TestMarginalRecalibrator:
    N = 2000

    def _mis_scaled(self, n: int, seed: int) -> tuple[GaussianJoint, torch.Tensor]:
        # Target scales 0.5, 2, 1 times the truth, and target 2 shifted by +0.5.
        mean, y = _data(n, seed)
        dmat = torch.diag(torch.tensor([0.5, 2.0, 1.0], dtype=F64)).to(dtype=F64)
        shift = torch.tensor([0.0, 0.0, 0.5], dtype=F64)
        return GaussianJoint(mean + shift, dmat @ _chol()), y

    @pytest.mark.parametrize("n_knots", [None, "all"])
    def test_gaussian_marginals_become_uniform(self, n_knots: Literal["all"] | None) -> None:
        cal, y_c = self._mis_scaled(self.N, 20)
        test, y_t = self._mis_scaled(self.N, 21)
        rec = MarginalRecalibrator(n_knots=n_knots).fit(cal, y_c)
        out = rec.transform(test)
        assert isinstance(out, MarginalRecalibratedJoint)
        assert out.has_density and out.has_latent and out.exact_marginals
        before, after = rec.pit(test, y_t), out.marginal_cdf(y_t)
        for j in range(D):  # Bonferroni over the 3 targets
            assert kstest(before[:, j].numpy(), "uniform").pvalue < 1e-6
            assert kstest(after[:, j].numpy(), "uniform").pvalue > 0.01 / D

    def test_gaussian_copula_is_kept(self) -> None:
        cal, y_c = self._mis_scaled(self.N, 20)
        out = MarginalRecalibrator().fit(cal, y_c).transform(cal)
        z = torch.randn(self.N, 50, D, dtype=F64)
        base, new = cal.from_latent(z), out.from_latent(z)
        assert torch.equal(base.argsort(1), new.argsort(1))  # identical ranks per target
        torch.testing.assert_close(out.to_latent(new), z)

    def test_gaussian_density_masks_and_marginals(self) -> None:
        cal, y_c = self._mis_scaled(self.N, 20)
        out = MarginalRecalibrator().fit(cal, y_c).transform(cal)
        m = torch.tensor([[True, False, True]]).expand(self.N, D)
        sub = out.marginal([0, 2])
        torch.testing.assert_close(out.log_prob(y_c, m), sub.log_prob(y_c[:, [0, 2]]))
        # One-dimensional marginal integrates to one (trapezoid on a wide grid).
        one = out.marginal(1)
        assert isinstance(one, MarginalRecalibratedJoint)
        rows = MarginalRecalibratedJoint(
            GaussianJoint(one.base.loc[:5], one.base.scale_tril[:5]), one.calibrators
        )
        grid = (one.base.loc[:5, 0] - 40.0).unsqueeze(1) + torch.linspace(
            0.0, 80.0, 40_001, dtype=F64
        )
        dens = rows.log_prob(grid.unsqueeze(-1)).exp()
        mass = torch.trapezoid(dens, grid, dim=1)
        torch.testing.assert_close(mass, torch.ones(5, dtype=F64), atol=1e-4, rtol=0)

    def test_default_held_out_nll_improves(self) -> None:
        cal, y_c = self._mis_scaled(self.N, 20)
        test, y_t = self._mis_scaled(self.N, 21)
        out = MarginalRecalibrator().fit(cal, y_c).transform(test)
        assert _nll(out, y_t) < _nll(test, y_t) - 0.2

    def test_default_density_stays_close_to_the_truth(self) -> None:
        # Calibrated Gaussian, m = 2000 (K = 22 knots), held-out n = 2000, d = 3.
        # Measured excess NLL over the true density: 0.026 nats in total (mean of 4
        # seeds); the margin 0.06 is about twice that.  With every rank as a knot the
        # excess is ~3 x 0.42 = 1.3 nats, which this margin excludes.
        mean_c, y_c = _data(self.N, 22)
        mean_t, y_t = _data(self.N, 23)
        cal, test = GaussianJoint(mean_c, _chol()), GaussianJoint(mean_t, _chol())
        out = MarginalRecalibrator().fit(cal, y_c).transform(test)
        assert abs(_nll(out, y_t) - _nll(test, y_t)) < 0.06
        rough = MarginalRecalibrator(n_knots="all").fit(cal, y_c).transform(test)
        assert _nll(rough, y_t) > _nll(test, y_t) + 0.6

    @pytest.mark.parametrize("n_knots", [None, "all"])
    def test_knot_ranks_are_mapped_to_their_levels(self, n_knots: Literal["all"] | None) -> None:
        # G_j(u_(k)) = k / (m + 1) exactly at the knot ranks k (every rank for "all").
        cal, y_c = self._mis_scaled(self.N, 20)
        rec = MarginalRecalibrator(n_knots=n_knots).fit(cal, y_c)
        out = rec.transform(cal)
        assert isinstance(out, MarginalRecalibratedJoint) and rec.calibrators_ is not None
        v = out.marginal_cdf(y_c).sort(0).values  # [m, d]: V of the k-th calibration PIT
        levels = torch.arange(1, self.N + 1, dtype=F64) / (self.N + 1)
        for j, spline in enumerate(rec.calibrators_):
            ranks = (torch.special.ndtr(spline.yk) * (self.N + 1)).round().long()
            assert len(ranks) == (self.N if n_knots == "all" else 22)
            torch.testing.assert_close(v[ranks - 1, j], levels[ranks - 1], atol=1e-10, rtol=0)

    def test_all_knots_finite_sample_property(self) -> None:
        # P(V <= k / (m + 1)) = k / (m + 1) for every k, over R = 2000 independent
        # calibration sets of m = 9 rows plus one test row (d = 1, predictive N(0, 0.5^2),
        # truth N(0, 1)).  Each frequency is checked within 4 binomial standard errors.
        m, reps = 9, 2000
        gen = torch.Generator().manual_seed(8)
        y = torch.randn(reps, m + 1, 1, generator=gen, dtype=F64)
        one = torch.full((1, 1), 0.5, dtype=F64)
        v = torch.empty(reps, dtype=F64)
        for r in range(reps):
            pred = GaussianJoint(torch.zeros(m, 1, dtype=F64), one)
            rec = MarginalRecalibrator(n_knots="all").fit(pred, y[r, :m])
            test = rec.transform(GaussianJoint(torch.zeros(1, 1, dtype=F64), one))
            assert isinstance(test, MarginalRecalibratedJoint)
            v[r] = test.marginal_cdf(y[r, m:])[0, 0]
        level = torch.arange(1, m + 1, dtype=F64) / (m + 1)
        freq = (v.unsqueeze(1) <= level + 1e-12).to(F64).mean(0)
        se = (level * (1 - level) / reps).sqrt()
        assert bool(((freq - level).abs() < 4 * se).all()), (freq, level)

    def test_identity_on_a_calibrated_predictive(self) -> None:
        mean, y = _data(self.N, 22)
        pred = GaussianJoint(mean, _chol())
        rec = MarginalRecalibrator(n_knots="all").fit(pred, y)
        assert rec.calibrators_ is not None
        u = torch.linspace(0.0005, 0.9995, 1000, dtype=F64)
        # sup |G_j - id| is the KS distance of the calibration PITs (up to 1 / (n + 1)):
        # below 1.63 / sqrt(n) with probability 0.99 per target.
        for spline in rec.calibrators_:
            g = torch.special.ndtr(spline.forward(torch.special.ndtri(u))[0])
            assert float((g - u).abs().max()) < 1.63 / math.sqrt(self.N)

    def test_draws_only_path(self) -> None:
        torch.manual_seed(0)
        cal_g, y_c = self._mis_scaled(self.N, 30)
        test_g, y_t = self._mis_scaled(self.N, 31)

        def as_distribution(g: GaussianJoint) -> DistributionJoint:
            return DistributionJoint(MultivariateNormal(g.loc, scale_tril=g.scale_tril))

        rec = MarginalRecalibrator(n_samples=500).fit(as_distribution(cal_g), y_c)
        out = rec.transform(as_distribution(test_g))
        assert isinstance(out, SampleJoint)
        draws = out.sample(500)
        # Randomised PIT of y among the recalibrated draws; uniform within Bonferroni.
        gen = torch.Generator().manual_seed(1)
        below = (draws < y_t.unsqueeze(1)).sum(1).to(F64)
        pit = (below + torch.rand(below.shape, generator=gen, dtype=F64)) / 501
        for j in range(D):
            assert kstest(pit[:, j].numpy(), "uniform").pvalue > 0.01 / D

    def test_draws_only_path_keeps_ranks(self) -> None:
        cal_g, y_c = self._mis_scaled(500, 32)
        fixed = SampleJoint(cal_g.sample(300))
        out = MarginalRecalibrator(n_samples=300).fit(fixed, y_c).transform(fixed)
        new = out.sample(300)
        assert torch.equal(fixed.samples.argsort(1), new.argsort(1))

    def test_mask_in_fit(self) -> None:
        cal, y_c = self._mis_scaled(self.N, 20)
        mask = torch.rand(self.N, D, generator=torch.Generator().manual_seed(3)) > 0.5
        y_nan = torch.where(mask, y_c, float("nan"))
        rec = MarginalRecalibrator(n_knots="all").fit(cal, y_nan, mask)
        assert rec.calibrators_ is not None
        assert [c.xk.shape[0] for c in rec.calibrators_] == mask.sum(0).tolist()
        auto = MarginalRecalibrator().fit(cal, y_nan, mask)
        assert auto.calibrators_ is not None
        expected = [min(m, max(10, round(0.5 * math.sqrt(m)))) for m in mask.sum(0).tolist()]
        assert [c.xk.shape[0] for c in auto.calibrators_] == expected


class TestCenter:
    """``JointPredictive.center`` and ``ScaleRecalibrator(about="center")``."""

    def test_center_of_gaussian_and_wrappers_is_the_mean(self) -> None:
        gen = torch.Generator().manual_seed(0)
        loc = torch.randn(6, 3, generator=gen, dtype=torch.float64)
        raw = torch.randn(6, 3, 3, generator=gen, dtype=torch.float64)
        base = GaussianJoint.from_covariance(
            loc, raw @ raw.mT + torch.eye(3).to(device=raw.device, dtype=raw.dtype)
        )
        torch.testing.assert_close(base.center(), loc)
        shift = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
        scale = torch.tensor([2.0, 0.5, 3.0], dtype=torch.float64)
        wrapped = AffineJoint(base, shift, scale)
        torch.testing.assert_close(wrapped.center(), wrapped.mean())
        torch.testing.assert_close(wrapped.marginal([2, 0]).center(), wrapped.mean()[:, [2, 0]])
        other = GaussianJoint(loc + 1.0, base.scale_tril)
        mix = MixtureJoint([base, other], torch.tensor([0.25, 0.75], dtype=torch.float64))
        torch.testing.assert_close(mix.center(), mix.mean())

    def test_about_center_equals_about_mean_for_a_gaussian(self) -> None:
        gen = torch.Generator().manual_seed(1)
        n, d = 800, 3
        loc = torch.randn(n, d, generator=gen, dtype=torch.float64)
        eye = torch.eye(d).to(device=loc.device, dtype=loc.dtype)
        base = GaussianJoint(loc, eye.expand(n, d, d))
        true = torch.tensor([0.6, 1.0, 1.7], dtype=torch.float64)
        y = loc + true * torch.randn(n, d, generator=gen, dtype=torch.float64)
        by_mean = ScaleRecalibrator("nll").fit(base, y)
        by_center = ScaleRecalibrator("nll", about="center").fit(base, y)
        assert by_mean.scale_ is not None and by_center.scale_ is not None
        # the Gaussian maximum-likelihood factor is the root mean squared residual
        exact = (y - loc).square().mean(0).sqrt()
        torch.testing.assert_close(by_mean.scale_, exact, rtol=2e-2, atol=0)
        torch.testing.assert_close(by_center.scale_, exact, rtol=2e-2, atol=0)
        out = by_center.transform(base)
        torch.testing.assert_close(out.mean(), loc)

    def test_rejects_unknown_about(self) -> None:
        with pytest.raises(ValueError, match="about"):
            ScaleRecalibrator(about="median")
