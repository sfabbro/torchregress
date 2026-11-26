"""Battle tests for :class:`torchregress.losses.StudentTMixture`.

Parity with scipy (multivariate_t / multivariate_normal / t), exact limits
(df -> inf, K = 1, identical components), explicit constructions of the
linear-observation and censored likelihoods, brute-force conditioning, Monte
Carlo quantiles and moments, gradchecks, dtype/device agreement, extreme
scales and degenerate inputs.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
import torch
from scipy import stats

from torchregress.losses import StudentTMixture
from torchregress.losses.student_t_mixture import _lgamma_ratio, _t_tail_bound

F64 = torch.float64
FORMS = ("scale_tril", "precision_tril", "scale_diag")


def _params(b=5, k=3, p=4, seed=0, dtype=F64, diag=False):
    g = torch.Generator().manual_seed(seed)
    loc = torch.randn(b, k, p, generator=g, dtype=dtype)
    if diag:
        s = torch.rand(b, k, p, generator=g, dtype=dtype) + 0.3
        cov = torch.diag_embed(s.square()).to(device=s.device, dtype=s.dtype)
    else:
        a = torch.randn(b, k, p, p, generator=g, dtype=dtype)
        cov = a @ a.mT / p + 0.3 * torch.eye(p, dtype=dtype)
    logits = torch.randn(b, k, generator=g, dtype=dtype)
    return logits, loc, cov


def _mixture(logits, loc, cov, df, form):
    if form == "scale_tril":
        return StudentTMixture(logits, loc, scale_tril=torch.linalg.cholesky(cov), df=df)
    if form == "precision_tril":
        prec = torch.linalg.cholesky(torch.linalg.inv(cov))
        return StudentTMixture(logits, loc, precision_tril=prec, df=df)
    diag = torch.diagonal(cov, dim1=-2, dim2=-1).sqrt()
    return StudentTMixture(logits, loc, scale_diag=diag, df=df)


def _scipy_component(mu, cov, df):
    if math.isinf(df):
        return stats.multivariate_normal(mu, cov)
    return stats.multivariate_t(mu, cov, df=df)


def _scipy_mixture_logpdf(logits, loc, cov, x, df, mask=None):
    w = torch.softmax(logits, -1).numpy()
    out = []
    for b in range(loc.shape[0]):
        o = np.ones(loc.shape[-1], bool) if mask is None else mask[b].numpy()
        if not o.any():
            out.append(0.0)
            continue
        dens = [
            _scipy_component(loc[b, k].numpy()[o], cov[b, k].numpy()[np.ix_(o, o)], df).logpdf(
                x[b].numpy()[o]
            )
            for k in range(loc.shape[1])
        ]
        out.append(float(np.logaddexp.reduce(np.log(w[b]) + np.asarray(dens))))
    return np.asarray(out)


# ---------------------------------------------------------------------------
# log densities
# ---------------------------------------------------------------------------


class TestLogProb:
    @pytest.mark.parametrize("form", FORMS)
    @pytest.mark.parametrize("df", [0.7, 3.0, 25.0, math.inf])
    def test_matches_scipy(self, form, df):
        logits, loc, cov = _params(diag=form == "scale_diag")
        x = torch.randn(5, 4, dtype=F64, generator=torch.Generator().manual_seed(9)) * 2
        mix = _mixture(logits, loc, cov, df, form)
        ref = _scipy_mixture_logpdf(logits, loc, cov, x, df)
        np.testing.assert_allclose(mix.log_prob(x).numpy(), ref, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("form", FORMS)
    @pytest.mark.parametrize("df", [2.5, math.inf])
    def test_masked_matches_scipy_marginal(self, form, df):
        logits, loc, cov = _params(b=8, diag=form == "scale_diag")
        g = torch.Generator().manual_seed(3)
        x = torch.randn(8, 4, dtype=F64, generator=g)
        mask = torch.rand(8, 4, generator=g) > 0.45
        mask[0] = False  # nothing observed
        mask[1] = True  # everything observed
        mix = _mixture(logits, loc, cov, df, form)
        x_nan = torch.where(mask, x, torch.nan)
        got = mix.log_prob(x_nan, mask=mask)
        ref = _scipy_mixture_logpdf(logits, loc, cov, x, df, mask=mask)
        np.testing.assert_allclose(got.numpy(), ref, rtol=1e-12, atol=1e-12)
        assert got[0].item() == 0.0

    def test_sample_shape_broadcasting(self):
        logits, loc, cov = _params()
        mix = _mixture(logits, loc, cov, 4.0, "scale_tril")
        x = torch.randn(7, 5, 4, dtype=F64)
        lp = mix.log_prob(x)
        assert lp.shape == (7, 5)
        np.testing.assert_allclose(lp[3].numpy(), mix.log_prob(x[3]).numpy(), rtol=1e-14)

    @pytest.mark.parametrize("form", FORMS)
    def test_large_df_tends_to_gaussian(self, form):
        logits, loc, cov = _params(diag=form == "scale_diag")
        x = torch.randn(5, 4, dtype=F64)
        gauss = _mixture(logits, loc, cov, math.inf, form).log_prob(x)
        # The gap is O(1/nu) with no numerical floor in the density itself:
        # nu * gap converges. The test's own subtraction of two log densities
        # costs about eps * |log p| each, i.e. nu * eps * |log p| after scaling
        # (2e-3 at nu = 1e12, log p ~ -8), so that bound is added to atol.
        dfs = (1e6, 1e9, 1e12)
        scaled = [df * (_mixture(logits, loc, cov, df, form).log_prob(x) - gauss) for df in dfs]
        floor = [4 * torch.finfo(F64).eps * df * gauss.abs() for df in dfs]
        for i, tol in ((1, 1e-3), (0, 1e-2)):
            err = (scaled[i] - scaled[2]).abs()
            assert torch.all(err <= tol + tol * scaled[2].abs() + floor[i] + floor[2])

    def test_single_component_equals_distribution(self):
        logits, loc, cov = _params(k=1)
        x = torch.randn(5, 4, dtype=F64)
        lmat = torch.linalg.cholesky(cov)
        mix = StudentTMixture(logits, loc, scale_tril=lmat, df=math.inf)
        ref = torch.distributions.MultivariateNormal(loc[:, 0], scale_tril=lmat[:, 0]).log_prob(x)
        torch.testing.assert_close(mix.log_prob(x), ref, rtol=1e-13, atol=1e-13)
        mix_t = StudentTMixture(logits, loc, scale_tril=lmat, df=6.0)
        ref_t = [
            stats.multivariate_t(loc[b, 0].numpy(), cov[b, 0].numpy(), df=6.0).logpdf(x[b].numpy())
            for b in range(5)
        ]
        np.testing.assert_allclose(mix_t.log_prob(x).numpy(), ref_t, rtol=1e-12)

    @pytest.mark.parametrize("form", FORMS)
    def test_identical_components_equal_one_component(self, form):
        logits, loc, cov = _params(k=1, diag=form == "scale_diag")
        x = torch.randn(5, 4, dtype=F64)
        one = _mixture(logits, loc, cov, 3.5, form).log_prob(x)
        many = _mixture(
            torch.randn(5, 4, dtype=F64),
            loc.expand(5, 4, 4),
            cov.expand(5, 4, 4, 4),
            3.5,
            form,
        ).log_prob(x)
        torch.testing.assert_close(many, one, rtol=1e-13, atol=1e-13)

    def test_per_component_and_per_row_df(self):
        logits, loc, cov = _params(b=3, k=2, p=2)
        df = torch.tensor([[2.0, math.inf], [5.0, 1.5], [math.inf, math.inf]], dtype=F64)
        x = torch.randn(3, 2, dtype=F64)
        mix = StudentTMixture(logits, loc, scale_tril=torch.linalg.cholesky(cov), df=df)
        w = torch.softmax(logits, -1).numpy()
        for b in range(3):
            dens = [
                _scipy_component(loc[b, k].numpy(), cov[b, k].numpy(), df[b, k].item()).logpdf(
                    x[b].numpy()
                )
                for k in range(2)
            ]
            ref = np.logaddexp.reduce(np.log(w[b]) + np.asarray(dens))
            assert mix.log_prob(x)[b].item() == pytest.approx(ref, rel=1e-12)

    def test_lgamma_ratio_matches_mpmath(self):
        mpmath = pytest.importorskip("mpmath")
        mpmath.mp.dps = 40
        xs = [0.05, 1.0, 10.0, 49.99, 50.0, 50.01, 300.0, 1e4, 1e9, 1e15]
        for a in (0.0, 0.5, 1.0, 2.5, 10.0):
            got = _lgamma_ratio(torch.tensor(xs, dtype=F64), torch.tensor(a, dtype=F64))
            for x, value in zip(xs, got.tolist(), strict=True):
                exact = float(
                    mpmath.loggamma(mpmath.mpf(x) + a)
                    - mpmath.loggamma(mpmath.mpf(x))
                    - a * mpmath.log(mpmath.mpf(x))
                )
                assert abs(value - exact) <= 5e-14 * max(1.0, abs(exact)), (x, a)

    def test_shape_validation(self):
        logits, loc, cov = _params()
        lmat = torch.linalg.cholesky(cov)
        with pytest.raises(ValueError, match="exactly one"):
            StudentTMixture(logits, loc)
        with pytest.raises(ValueError, match="exactly one"):
            StudentTMixture(logits, loc, scale_tril=lmat, precision_tril=lmat)
        with pytest.raises(ValueError, match="components"):
            StudentTMixture(logits[:, :2], loc, scale_tril=lmat)
        with pytest.raises(ValueError, match="df"):
            StudentTMixture(logits, loc, scale_tril=lmat, df=0.0)
        with pytest.raises(ValueError, match="positive diagonal"):
            StudentTMixture(logits, loc, scale_tril=-lmat, validate_args=True)


# ---------------------------------------------------------------------------
# linear observations and censoring
# ---------------------------------------------------------------------------


def _explicit_observation_logpdf(mix_args, design, offset, noise, z, valid, df):
    logits, loc, cov = mix_args
    b_n, k_n = loc.shape[:2]
    out = np.zeros((b_n, k_n))
    for b in range(b_n):
        v = valid[b].numpy()
        if not v.any():
            continue
        for k in range(k_n):
            a = design[b, min(k, design.shape[1] - 1)].numpy()[v]
            off = 0.0 if offset is None else offset[b, min(k, offset.shape[1] - 1)].numpy()[v]
            mean = a @ loc[b, k].numpy() + off
            c = a @ cov[b, k].numpy() @ a.T + np.diag(noise[b].numpy()[v])
            out[b, k] = _scipy_component(mean, c, df).logpdf(z[b].numpy()[v])
    return out


class TestObservation:
    @pytest.mark.parametrize("form", FORMS)
    @pytest.mark.parametrize("n_obs", [3, 4, 9])  # direct (D <= P) and information (D > P)
    @pytest.mark.parametrize("df", [3.0, math.inf])
    def test_matches_explicit_construction(self, form, n_obs, df):
        logits, loc, cov = _params(b=6, diag=form == "scale_diag")
        g = torch.Generator().manual_seed(11)
        design = torch.randn(6, 3, n_obs, 4, dtype=F64, generator=g)
        offset = torch.randn(6, 3, n_obs, dtype=F64, generator=g)
        noise = torch.rand(6, n_obs, dtype=F64, generator=g) + 0.05
        z = torch.randn(6, n_obs, dtype=F64, generator=g) * 2
        valid = torch.rand(6, n_obs, generator=g) > 0.3
        valid[0] = False
        valid[1] = True
        mix = _mixture(logits, loc, cov, df, form)
        got = mix.component_observation_log_prob(
            torch.where(valid, z, torch.nan), design, noise, offset=offset, valid=valid
        )
        ref = _explicit_observation_logpdf((logits, loc, cov), design, offset, noise, z, valid, df)
        np.testing.assert_allclose(got.numpy(), ref, rtol=1e-11, atol=1e-11)
        assert (got[0] == 0).all()
        assert mix.mixture_log_prob(got)[0].item() == 0.0

    def test_shared_design_without_offset(self):
        logits, loc, cov = _params(b=4)
        design = torch.randn(4, 1, 6, 4, dtype=F64)
        noise = torch.rand(4, 6, dtype=F64) + 0.1
        z = torch.randn(4, 6, dtype=F64)
        mix = _mixture(logits, loc, cov, 4.0, "scale_tril")
        got = mix.component_observation_log_prob(z, design, noise)
        ref = _explicit_observation_logpdf(
            (logits, loc, cov), design, None, noise, z, torch.ones(4, 6, dtype=torch.bool), 4.0
        )
        np.testing.assert_allclose(got.numpy(), ref, rtol=1e-11)

    def test_identity_design_with_tiny_noise_is_log_prob(self):
        logits, loc, cov = _params(b=4)
        mix = _mixture(logits, loc, cov, 5.0, "scale_tril")
        x = torch.randn(4, 4, dtype=F64)
        eye = torch.eye(4, dtype=F64).expand(4, 1, 4, 4)
        got = mix.component_observation_log_prob(x, eye, torch.full((4, 4), 1e-14, dtype=F64))
        torch.testing.assert_close(got, mix.component_log_prob(x), rtol=1e-9, atol=1e-9)

    def test_information_form_is_accurate_at_high_signal_to_noise(self):
        # D > P with noise 1e-10 of the projected scale: a subtracted Woodbury
        # quadratic would lose ~10 digits; the sum-of-squares form does not.
        logits, loc, cov = _params(b=3, k=2, p=2)
        design = torch.randn(3, 2, 5, 2, dtype=F64)
        noise = torch.full((3, 5), 1e-10, dtype=F64)
        x_true = loc[:, 0]
        z = (design[:, 0] @ x_true.unsqueeze(-1)).squeeze(-1) + 1e-5 * torch.randn(3, 5, dtype=F64)
        mix = _mixture(logits, loc, cov, math.inf, "scale_tril")
        got = mix.component_observation_log_prob(z, design, noise)
        mpmath = pytest.importorskip("mpmath")
        mpmath.mp.dps = 60
        ref = np.zeros((3, 2))
        for b in range(3):
            for k in range(2):
                a = mpmath.matrix(design[b, k].tolist())
                c = a * mpmath.matrix(cov[b, k].tolist()) * a.T
                for i in range(5):
                    c[i, i] += mpmath.mpf(noise[b, i].item())
                r = mpmath.matrix(z[b].tolist()) - a * mpmath.matrix(loc[b, k].tolist())
                quad = (r.T * mpmath.lu_solve(c, r))[0]
                ref[b, k] = float(
                    -0.5 * (quad + mpmath.log(mpmath.det(c)) + 5 * mpmath.log(2 * mpmath.pi))
                )
        np.testing.assert_allclose(got.numpy(), ref, rtol=1e-7)

    @pytest.mark.parametrize("df", [1.5, 7.0, math.inf])
    def test_censored_matches_scipy(self, df):
        logits, loc, cov = _params(b=5)
        g = torch.Generator().manual_seed(5)
        design = torch.randn(5, 3, 2, 4, dtype=F64, generator=g)
        offset = torch.randn(5, 3, 2, dtype=F64, generator=g)
        noise = torch.rand(5, 2, dtype=F64, generator=g)
        limit = torch.randn(5, 2, dtype=F64, generator=g) * 3
        upper = torch.tensor([[True, False]] * 5)
        valid = torch.ones(5, 2, dtype=torch.bool)
        valid[2, 1] = False
        mix = _mixture(logits, loc, cov, df, "scale_tril")
        got = mix.component_censored_log_prob(
            limit, design, noise, offset=offset, upper=upper, valid=valid
        )
        ref = np.zeros((5, 3))
        for b in range(5):
            for k in range(3):
                for m in range(2):
                    if not valid[b, m]:
                        continue
                    a = design[b, k, m].numpy()
                    mean = a @ loc[b, k].numpy() + offset[b, k, m].item()
                    sd = math.sqrt(a @ cov[b, k].numpy() @ a + noise[b, m].item())
                    zz = (limit[b, m].item() - mean) / sd
                    zz = zz if upper[b, m] else -zz
                    dist = stats.norm if math.isinf(df) else stats.t(df)
                    ref[b, k] += dist.logcdf(zz)
        np.testing.assert_allclose(got.numpy(), ref, rtol=1e-11, atol=1e-12)

    @pytest.mark.parametrize("df", [2.0, 5.0, math.inf])
    def test_censored_deep_tail_is_exact(self, df):
        loc = torch.zeros(1, 1, 1, dtype=F64)
        mix = StudentTMixture(
            torch.zeros(1, 1, dtype=F64), loc, scale_tril=torch.ones(1, 1, 1, 1, dtype=F64), df=df
        )
        design = torch.ones(1, 1, 1, 1, dtype=F64)
        for z in (-40.0, -8.0, 0.0, 8.0, 40.0):
            got = mix.component_censored_log_prob(
                torch.tensor([[z]], dtype=F64), design, torch.zeros(1, 1, dtype=F64)
            ).item()
            ref = (stats.norm if math.isinf(df) else stats.t(df)).logcdf(z)
            assert got == pytest.approx(ref, rel=1e-10, abs=1e-14)

    def test_scalar_upper_false_is_lower_limit(self):
        logits, loc, cov = _params(b=2, k=2, p=2)
        mix = _mixture(logits, loc, cov, 4.0, "scale_tril")
        design = torch.randn(2, 1, 1, 2, dtype=F64)
        noise = torch.full((2, 1), 0.1, dtype=F64)
        limit = torch.zeros(2, 1, dtype=F64)
        up = mix.component_censored_log_prob(limit, design, noise, upper=True)
        lo = mix.component_censored_log_prob(limit, design, noise, upper=False)
        torch.testing.assert_close(up.exp() + lo.exp(), torch.ones_like(up), rtol=1e-13, atol=0)


# ---------------------------------------------------------------------------
# conditioning
# ---------------------------------------------------------------------------


class TestCondition:
    @pytest.mark.parametrize("df", [3.0, math.inf])
    @pytest.mark.parametrize("j", [0, 1])
    def test_matches_brute_force_integration(self, df, j):
        # P = 2: condition on y = x_j + e and compare the conditional density on a
        # grid with the normalised joint density of (x, y), "noise inside the t".
        logits, loc, cov = _params(b=1, k=2, p=2, seed=4)
        noise = 0.3
        y = 0.4
        mix = _mixture(logits, loc, cov, df, "scale_tril")
        cond = mix.condition(j, torch.tensor([y], dtype=F64), torch.tensor([noise], dtype=F64))
        grid = torch.linspace(-30, 30, 1201, dtype=F64)
        gx, gy = torch.meshgrid(grid, grid, indexing="ij")
        pts = torch.stack([gx, gy], -1).reshape(-1, 2)
        w = torch.softmax(logits[0], -1).numpy()
        joint = np.zeros(len(pts))
        for k in range(2):
            m = np.append(loc[0, k].numpy(), loc[0, k, j].item())
            c = np.zeros((3, 3))
            c[:2, :2] = cov[0, k].numpy()
            c[:2, 2] = c[2, :2] = cov[0, k, :, j].numpy()
            c[2, 2] = cov[0, k, j, j].item() + noise
            joint += w[k] * _scipy_component(m, c, df).pdf(
                np.column_stack([pts.numpy(), np.full(len(pts), y)])
            )
        dx = (grid[1] - grid[0]).item()
        evidence = joint.sum() * dx * dx
        brute = joint / evidence
        got = cond.log_prob(pts.unsqueeze(1)).squeeze(1).exp().numpy()
        # Riemann sums on a fine grid: compare where the density is not negligible.
        sel = brute > 1e-3 * brute.max()
        np.testing.assert_allclose(got[sel], brute[sel], rtol=2e-3)
        # Predictive evidence: mixture of the per-component predictive t densities.
        pred = np.sum(
            [
                w[k]
                * (stats.norm if math.isinf(df) else stats.t(df)).pdf(
                    (y - loc[0, k, j].item()) / math.sqrt(cov[0, k, j, j].item() + noise)
                )
                / math.sqrt(cov[0, k, j, j].item() + noise)
                for k in range(2)
            ]
        )
        assert evidence == pytest.approx(pred, rel=2e-3)
        expected_df = df + 1 if math.isfinite(df) else df
        assert torch.all(cond.df == expected_df)

    @pytest.mark.parametrize("form", FORMS)
    def test_conditional_parameters_closed_form(self, form):
        logits, loc, cov = _params(b=4, k=3, p=4, diag=form == "scale_diag")
        mix = _mixture(logits, loc, cov, 4.0, form)
        y = torch.randn(4, dtype=F64)
        n = torch.rand(4, dtype=F64) + 1e-3
        j = 2
        cond = mix.condition(j, y, n)
        s = cov[..., j, j] + n[:, None]
        c = cov[..., :, j]
        r = y[:, None] - loc[..., j]
        q = r.square() / s
        exp_loc = loc + c * (r / s).unsqueeze(-1)
        exp_cov = (cov - c.unsqueeze(-1) * c.unsqueeze(-2) / s[..., None, None]) * (
            (4.0 + q) / 5.0
        )[..., None, None]
        torch.testing.assert_close(cond.loc, exp_loc, rtol=1e-12, atol=1e-12)
        lc = cond.scale_tril
        torch.testing.assert_close(lc @ lc.mT, exp_cov, rtol=1e-11, atol=1e-12)
        assert torch.all(lc.triu(1) == 0)
        pred = torch.tensor(stats.t(4.0).logpdf((r / s.sqrt()).numpy()), dtype=F64) - 0.5 * s.log()
        torch.testing.assert_close(
            cond.log_weights, torch.log_softmax(mix.log_weights + pred, -1), rtol=1e-12, atol=1e-12
        )

    def test_unconditioned_rows_unchanged(self):
        logits, loc, cov = _params(b=4)
        mix = _mixture(logits, loc, cov, 5.0, "scale_tril")
        y = torch.tensor([0.1, math.nan, 0.3, 0.2], dtype=F64)
        n = torch.tensor([0.1, 0.1, -1.0, math.inf], dtype=F64)
        cond = mix.condition(1, y, n)
        for b in (1, 2, 3):
            torch.testing.assert_close(cond.loc[b], mix.loc[b])
            torch.testing.assert_close(cond.scale_tril[b], mix.scale_tril[b])
            torch.testing.assert_close(cond.log_weights[b], mix.log_weights[b])
            assert torch.all(cond.df[b] == 5.0)
        assert torch.all(cond.df[0] == 6.0)

    def test_nearly_exact_observation_keeps_relative_accuracy(self):
        # Conditional variance of the observed coordinate is ~ noise: the tail-sum
        # factor gets it to full relative accuracy (forming Σ - c cᵀ/s would not).
        logits, loc, cov = _params(b=2, k=2, p=3)
        mix = _mixture(logits, loc, cov, math.inf, "scale_tril")
        n = torch.full((2,), 1e-20, dtype=F64)
        cond = mix.condition(1, loc[:, 0, 1], n)
        var_j = cond.component_scale[..., 1].square()
        expected = cov[..., 1, 1] * n[:, None] / (cov[..., 1, 1] + n[:, None])
        torch.testing.assert_close(var_j, expected, rtol=1e-10, atol=0)

    def test_matches_masked_log_prob_for_exact_coordinate(self):
        # p(x_rest | x_j) from conditioning (n -> 0) equals p(x) / p(x_j).
        logits, loc, cov = _params(b=3, k=2, p=3)
        mix = _mixture(logits, loc, cov, 3.0, "scale_tril")
        x = torch.randn(3, 3, dtype=F64)
        cond = mix.condition(0, x[:, 0], torch.zeros(3, dtype=F64))
        rest = torch.tensor([False, True, True]).expand(3, 3)
        obs = torch.tensor([True, False, False]).expand(3, 3)
        lhs = cond.log_prob(x, mask=rest)
        rhs = mix.log_prob(x) - mix.log_prob(x, mask=obs)
        torch.testing.assert_close(lhs, rhs, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# quantiles, CDF, sampling, moments
# ---------------------------------------------------------------------------


class TestQuantilesAndSampling:
    @pytest.mark.parametrize("df", [0.6, 1.0, 4.0, 1e6, math.inf])
    def test_single_component_quantile_matches_scipy(self, df):
        loc = torch.tensor([[[0.3, -2.0]]], dtype=F64)
        scale = torch.tensor([[[0.5, 3.0]]], dtype=F64)
        mix = StudentTMixture(torch.zeros(1, 1, dtype=F64), loc, scale_diag=scale, df=df)
        levels = [1e-6, 0.01, 0.16, 0.5, 0.84, 0.99, 1 - 1e-6]
        got = mix.marginal_quantile(levels)[0]
        dist = stats.norm if math.isinf(df) else stats.t(df)
        ref = loc[0, 0, :, None].numpy() + scale[0, 0, :, None].numpy() * dist.ppf(levels)
        np.testing.assert_allclose(got.numpy(), ref, rtol=1e-9, atol=1e-9)

    @pytest.mark.parametrize("df", [0.6, 1.0, 4.0, math.inf])
    def test_far_upper_tail_quantiles_are_relatively_accurate(self, df):
        # Upper levels are solved on the survival function: the quantile at
        # 1 - p matches scipy's isf(p) to relative precision (a CDF-space solve
        # is limited to an absolute error of about eps / f(x) there) and the
        # solution is symmetric with the lower tail.
        mix = StudentTMixture(
            torch.zeros(1, 1, dtype=F64),
            torch.zeros(1, 1, 1, dtype=F64),
            scale_diag=torch.ones(1, 1, 1, dtype=F64),
            df=df,
        )
        p = np.array([1e-6, 1e-9, 1e-12])
        dist = stats.norm if math.isinf(df) else stats.t(df)
        levels = 1.0 - p  # rounded: the exact upper-tail mass is 1 - levels
        upper = mix.marginal_quantile(levels.tolist())[0, 0].numpy()
        lower = mix.marginal_quantile(p.tolist())[0, 0].numpy()
        np.testing.assert_allclose(upper, dist.isf(1.0 - levels), rtol=1e-11)
        np.testing.assert_allclose(lower, dist.ppf(p), rtol=1e-11)

    @pytest.mark.parametrize("form", FORMS)
    def test_mixture_quantiles_invert_cdf(self, form):
        logits, loc, cov = _params(b=20, k=4, p=3, seed=8, diag=form == "scale_diag")
        loc = loc * 4  # well separated, multimodal marginals
        df = torch.tensor([0.8, 2.0, 5.0, math.inf], dtype=F64)
        mix = _mixture(logits, loc, cov, df, form)
        levels = torch.tensor([1e-4, 0.025, 0.16, 0.5, 0.84, 0.975, 1 - 1e-4], dtype=F64)
        qs = mix.marginal_quantile(levels)
        assert qs.shape == (20, 3, 7)
        cdf = mix.marginal_cdf(qs.permute(2, 0, 1))
        torch.testing.assert_close(cdf, levels[:, None, None].expand_as(cdf), rtol=1e-9, atol=1e-12)
        assert torch.all(qs.diff(dim=-1) > 0)
        scalar = mix.marginal_quantile(0.5)
        torch.testing.assert_close(scalar, qs[..., 3])

    def test_marginal_cdf_matches_scipy(self):
        logits, loc, cov = _params(b=3, k=2, p=2)
        mix = _mixture(logits, loc, cov, 3.0, "scale_tril")
        grid = torch.linspace(-6, 6, 25, dtype=F64)[:, None, None].expand(25, 3, 2)
        got = mix.marginal_cdf(grid)
        w = torch.softmax(logits, -1).numpy()
        sd = np.sqrt(np.diagonal(cov.numpy(), axis1=-2, axis2=-1))
        ref = (
            w[None, :, :, None]
            * stats.t(3.0).cdf((grid.numpy()[:, :, None, :] - loc.numpy()[None]) / sd[None])
        ).sum(2)
        np.testing.assert_allclose(got.numpy(), ref, rtol=1e-12, atol=1e-14)

    def test_quantiles_and_moments_vs_monte_carlo(self):
        torch.manual_seed(0)
        logits, loc, cov = _params(b=2, k=3, p=2, seed=6)
        df = torch.tensor([4.0, 9.0, math.inf], dtype=F64)
        for form in FORMS:
            c = (
                cov
                if form != "scale_diag"
                else torch.diag_embed(cov.diagonal(dim1=-2, dim2=-1)).to(
                    device=cov.device, dtype=cov.dtype
                )
            )
            mix = _mixture(logits, loc, c, df, form)
            n = 400_000
            draws = mix.sample((n,))
            assert draws.shape == (n, 2, 2)
            levels = torch.tensor([0.05, 0.3, 0.5, 0.7, 0.95], dtype=F64)
            qs = mix.marginal_quantile(levels)  # [2, 2, 5]
            frac = (draws.unsqueeze(-1) <= qs.unsqueeze(0)).double().mean(0)
            se = (levels * (1 - levels) / n).sqrt()
            assert torch.all((frac - levels).abs() < 5 * se), form
            torch.testing.assert_close(draws.mean(0), mix.mean, rtol=0, atol=0.03)
            emp_cov = torch.einsum("nbi,nbj->bij", draws - draws.mean(0), draws - draws.mean(0)) / (
                n - 1
            )
            torch.testing.assert_close(emp_cov, mix.covariance_matrix, rtol=0.05, atol=0.05)
            torch.testing.assert_close(
                mix.variance, mix.covariance_matrix.diagonal(dim1=-2, dim2=-1)
            )

    def test_moments_undefined_for_small_df(self):
        logits, loc, cov = _params(b=1, k=2, p=2)
        df = torch.tensor([1.5, 5.0], dtype=F64)
        mix = _mixture(logits, loc, cov, df, "scale_tril")
        assert torch.isfinite(mix.mean).all()
        assert torch.isinf(mix.variance).all()
        mix = _mixture(logits, loc, cov, torch.tensor([0.9, 5.0], dtype=F64), "scale_tril")
        assert torch.isnan(mix.mean).all()

    def test_tail_bound_brackets(self):
        alpha = torch.tensor([1e-12, 1e-6, 0.01, 0.3, 0.49], dtype=F64)
        for df in (0.3, 1.0, 3.0, 100.0, 1e8, math.inf):
            c = _t_tail_bound(alpha, torch.tensor(df, dtype=F64)).numpy()
            dist = stats.norm if math.isinf(df) else stats.t(df)
            assert np.all(dist.cdf(-c) <= alpha.numpy() * (1 + 1e-12)), df

    def test_point_mass_component_after_exact_conditioning(self):
        logits, loc, cov = _params(b=2, k=2, p=2)
        mix = _mixture(logits, loc, cov, 3.0, "scale_tril")
        cond = mix.condition(0, torch.tensor([0.5, -0.5], dtype=F64), torch.zeros(2, dtype=F64))
        q = cond.marginal_quantile([0.1, 0.5, 0.9])
        torch.testing.assert_close(q[:, 0], torch.tensor([[0.5] * 3, [-0.5] * 3], dtype=F64))
        assert torch.isfinite(q).all()


# ---------------------------------------------------------------------------
# gradients, dtypes, devices, scales, degenerate inputs
# ---------------------------------------------------------------------------


def _tril_param(p, k, b, seed):
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(b, k, p, p, dtype=F64, generator=g) * 0.3
    return raw.tril(-1) + torch.diag_embed(torch.rand(b, k, p, dtype=F64, generator=g) + 0.5).to(
        dtype=F64
    )


class TestGradients:
    def test_gradcheck_log_prob_all_parameters(self):
        b, k, p = 3, 2, 3
        logits = torch.randn(b, k, dtype=F64, requires_grad=True)
        loc = torch.randn(b, k, p, dtype=F64, requires_grad=True)
        lmat = _tril_param(p, k, b, 1).requires_grad_(True)
        df = torch.tensor([3.0, 40.0], dtype=F64, requires_grad=True)
        x = torch.randn(b, p, dtype=F64)
        mask = torch.tensor([[True, False, True], [True, True, True], [False, True, False]])

        def fn(logits, loc, lmat, df):
            mix = StudentTMixture(logits, loc, scale_tril=lmat.tril(), df=df)
            return mix.log_prob(x), mix.log_prob(x, mask=mask)

        assert torch.autograd.gradcheck(fn, (logits, loc, lmat, df))

    def test_gradcheck_precision_and_diag(self):
        b, k, p = 2, 2, 3
        loc = torch.randn(b, k, p, dtype=F64, requires_grad=True)
        tmat = _tril_param(p, k, b, 2).requires_grad_(True)
        s = (torch.rand(b, k, p, dtype=F64) + 0.5).requires_grad_(True)
        x = torch.randn(b, p, dtype=F64)
        assert torch.autograd.gradcheck(
            lambda loc, t: StudentTMixture(
                torch.zeros(b, k, dtype=F64), loc, precision_tril=t.tril(), df=2.5
            ).log_prob(x),
            (loc, tmat),
        )
        assert torch.autograd.gradcheck(
            lambda loc, s: StudentTMixture(
                torch.zeros(b, k, dtype=F64), loc, scale_diag=s, df=2.5
            ).log_prob(x),
            (loc, s),
        )

    @pytest.mark.parametrize("n_obs", [2, 5])
    def test_gradcheck_observation_and_censored(self, n_obs):
        b, k, p = 2, 2, 3
        loc = torch.randn(b, k, p, dtype=F64, requires_grad=True)
        lmat = _tril_param(p, k, b, 3).requires_grad_(True)
        design = torch.randn(b, k, n_obs, p, dtype=F64, requires_grad=True)
        noise = (torch.rand(b, n_obs, dtype=F64) + 0.2).requires_grad_(True)
        df = torch.tensor(4.0, dtype=F64, requires_grad=True)
        z = torch.randn(b, n_obs, dtype=F64)
        valid = torch.ones(b, n_obs, dtype=torch.bool)
        valid[1, 0] = False

        def fn(loc, lmat, design, noise, df):
            mix = StudentTMixture(torch.zeros(b, k, dtype=F64), loc, scale_tril=lmat.tril(), df=df)
            obs = mix.component_observation_log_prob(z, design, noise, valid=valid)
            cens = mix.component_censored_log_prob(z, design, noise, valid=valid)
            return mix.mixture_log_prob(obs + cens)

        assert torch.autograd.gradcheck(fn, (loc, lmat, design, noise, df))

    def test_gradcheck_condition(self):
        b, k, p = 2, 2, 3
        loc = torch.randn(b, k, p, dtype=F64, requires_grad=True)
        lmat = _tril_param(p, k, b, 4).requires_grad_(True)
        y = torch.randn(b, dtype=F64)
        x = torch.randn(b, p, dtype=F64)

        def fn(loc, lmat):
            mix = StudentTMixture(torch.zeros(b, k, dtype=F64), loc, scale_tril=lmat.tril(), df=3.0)
            cond = mix.condition(1, y, torch.full((b,), 0.2, dtype=F64))
            return cond.log_prob(x), cond.loc, cond.scale_tril

        assert torch.autograd.gradcheck(fn, (loc, lmat))

    @pytest.mark.parametrize("df0", [0.2, 1.0, 1e4, 1e8, 1e12])
    @pytest.mark.parametrize("dtype", [torch.float32, F64])
    def test_gradients_finite_at_extreme_df(self, df0, dtype):
        b, k, p = 4, 2, 3
        loc = torch.randn(b, k, p, dtype=dtype, requires_grad=True)
        lmat = _tril_param(p, k, b, 5).to(dtype).requires_grad_(True)
        df = torch.tensor(df0, dtype=dtype, requires_grad=True)
        x = torch.randn(b, p, dtype=dtype) * 5
        mix = StudentTMixture(torch.zeros(b, k, dtype=dtype), loc, scale_tril=lmat, df=df)
        design = torch.randn(b, 1, 2, p, dtype=dtype)
        loss = -(
            mix.log_prob(x).sum()
            + mix.mixture_log_prob(
                mix.component_observation_log_prob(x[:, :2], design, torch.ones(b, 2, dtype=dtype))
                + mix.component_censored_log_prob(x[:, :2], design, torch.ones(b, 2, dtype=dtype))
            ).sum()
        )
        loss.backward()
        for t in (loc, lmat, df):
            assert t.grad is not None and torch.isfinite(t.grad).all(), (df0, dtype)

    def test_infinite_df_has_no_nan_gradient(self):
        loc = torch.randn(3, 2, 2, dtype=F64, requires_grad=True)
        df = torch.tensor([math.inf, 3.0], dtype=F64, requires_grad=True)
        mix = StudentTMixture(
            torch.zeros(3, 2, dtype=F64), loc, scale_diag=torch.ones(3, 2, 2, dtype=F64), df=df
        )
        mix.log_prob(torch.randn(3, 2, dtype=F64)).sum().backward()
        assert torch.isfinite(loc.grad).all()
        assert df.grad[0] == 0 and torch.isfinite(df.grad[1])


class TestRobustness:
    def test_float32_agrees_with_float64(self):
        logits, loc, cov = _params(b=64, k=3, p=5, seed=12)
        x = torch.randn(64, 5, dtype=F64)
        design = torch.randn(64, 3, 9, 5, dtype=F64)
        noise = torch.rand(64, 9, dtype=F64) + 0.1
        z = torch.randn(64, 9, dtype=F64)
        mask = torch.rand(64, 5) > 0.3
        out = {}
        for dtype in (torch.float32, F64):
            mix = _mixture(logits.to(dtype), loc.to(dtype), cov.to(dtype), 4.0, "scale_tril")
            out[dtype] = (
                mix.log_prob(x.to(dtype)),
                mix.log_prob(x.to(dtype), mask=mask),
                mix.component_observation_log_prob(z.to(dtype), design.to(dtype), noise.to(dtype)),
                mix.marginal_quantile([0.16, 0.84]),
                mix.condition(2, z[:, 0].to(dtype), noise[:, 0].to(dtype)).loc,
            )
        for a, b in zip(out[torch.float32], out[F64], strict=True):
            assert a.dtype == torch.float32
            torch.testing.assert_close(a.double(), b, rtol=2e-5, atol=2e-5)

    @pytest.mark.parametrize("scale", [1e-4, 1.0, 1e4])
    def test_extreme_scales_are_exact_rescalings(self, scale):
        logits, loc, cov = _params(b=6, k=3, p=4, seed=13)
        x = torch.randn(6, 4, dtype=F64)
        design = torch.randn(6, 3, 7, 4, dtype=F64)
        noise = torch.rand(6, 7, dtype=F64) + 0.1
        z = torch.randn(6, 7, dtype=F64)
        mask = torch.rand(6, 4) > 0.3
        base = _mixture(logits, loc, cov, 3.0, "scale_tril")
        big = _mixture(logits, loc * scale, cov * scale**2, 3.0, "scale_tril")
        d = mask.sum(-1).double()
        torch.testing.assert_close(
            big.log_prob(x * scale), base.log_prob(x) - 4 * math.log(scale), rtol=1e-12, atol=1e-9
        )
        torch.testing.assert_close(
            big.log_prob(x * scale, mask=mask),
            base.log_prob(x, mask=mask) - d * math.log(scale),
            rtol=1e-12,
            atol=1e-9,
        )
        torch.testing.assert_close(
            big.component_observation_log_prob(z * scale, design, noise * scale**2),
            base.component_observation_log_prob(z, design, noise) - 7 * math.log(scale),
            rtol=1e-12,
            atol=1e-9,
        )
        torch.testing.assert_close(
            big.marginal_quantile([0.1, 0.5, 0.9]) / scale,
            base.marginal_quantile([0.1, 0.5, 0.9]),
            rtol=1e-10,
            atol=1e-10,
        )
        cb = big.condition(1, z[:, 0] * scale, noise[:, 0] * scale**2)
        cs = base.condition(1, z[:, 0], noise[:, 0])
        torch.testing.assert_close(cb.loc / scale, cs.loc, rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(cb.log_weights, cs.log_weights, rtol=1e-11, atol=1e-11)

    def test_all_masked_rows_contribute_zero_without_nan(self):
        b, k, p = 4, 3, 3
        loc = torch.randn(b, k, p, dtype=F64, requires_grad=True)
        lmat = _tril_param(p, k, b, 6).requires_grad_(True)
        mix = StudentTMixture(torch.randn(b, k, dtype=F64), loc, scale_tril=lmat, df=3.0)
        x = torch.full((b, p), math.nan, dtype=F64)
        mask = torch.zeros(b, p, dtype=torch.bool)
        z = torch.full((b, 5), math.nan, dtype=F64)
        valid = torch.zeros(b, 5, dtype=torch.bool)
        design = torch.randn(b, 1, 5, p, dtype=F64)
        noise = torch.full((b, 5), math.nan, dtype=F64)
        total = (
            mix.log_prob(x, mask=mask)
            + mix.mixture_log_prob(
                mix.component_observation_log_prob(z, design, noise, valid=valid)
                + mix.component_censored_log_prob(z, design, noise, valid=valid)
            )
            + mix.mixture_log_prob(
                mix.component_observation_log_prob(
                    z[:, :2], design[..., :2, :], noise[:, :2], valid=valid[:, :2]
                )
            )
        )
        assert torch.all(total == 0)
        total.sum().backward()
        assert torch.all(loc.grad == 0) and torch.all(lmat.grad == 0)

    def test_non_spd_observation_covariance_is_nan_not_error(self):
        logits, loc, cov = _params(b=2, k=1, p=2)
        mix = _mixture(logits, loc, cov, 3.0, "scale_tril")
        design = torch.zeros(2, 1, 1, 2, dtype=F64)
        out = mix.component_observation_log_prob(
            torch.zeros(2, 1, dtype=F64), design, torch.zeros(2, 1, dtype=F64)
        )
        assert torch.isnan(out).all()


@pytest.mark.cuda
def test_device_matches_cpu():
    device = os.environ.get("TORCHREGRESS_PARITY_DEVICE", "cuda")
    logits, loc, cov = _params(b=16, k=3, p=4, seed=14)
    x = torch.randn(16, 4, dtype=F64)
    design = torch.randn(16, 3, 6, 4, dtype=F64)
    noise = torch.rand(16, 6, dtype=F64) + 0.1
    z = torch.randn(16, 6, dtype=F64)

    def run(dev):
        mix = _mixture(logits.to(dev), loc.to(dev), cov.to(dev), 4.0, "scale_tril")
        return (
            mix.log_prob(x.to(dev)),
            mix.component_observation_log_prob(z.to(dev), design.to(dev), noise.to(dev)),
            mix.component_censored_log_prob(z.to(dev), design.to(dev), noise.to(dev)),
            mix.marginal_quantile([0.16, 0.5, 0.84]),
            mix.condition(0, z[:, 0].to(dev), noise[:, 0].to(dev)).scale_tril,
        )

    for a, b in zip(run("cpu"), run(device), strict=True):
        assert b.device.type == torch.device(device).type
        torch.testing.assert_close(b.cpu(), a, rtol=1e-10, atol=1e-10)
