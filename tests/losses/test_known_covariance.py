"""Tests for the known-plus-model-covariance Gaussian NLL."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import stats

from torchregress.losses import (
    GaussianNLLLoss,
    KnownCovarianceGaussianNLL,
    KnownCovarianceGaussianResult,
    LowRankGaussianLoss,
    MarginalizedMultiSourceGaussianNLLLoss,
    MultivariateGaussianLoss,
    NoisyTargetGaussianNLL,
    PackedObservations,
    get_regression_loss,
    known_covariance_gaussian_nll,
    marginalize_linear_nuisance,
)

F64 = torch.float64


def _spd(rows: int, d: int, seed: int, dtype: torch.dtype = F64) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(rows, d, d, generator=gen, dtype=dtype)
    return a @ a.mT / d + 0.05 * torch.eye(d, dtype=dtype)


def _problem(rows: int = 6, d: int = 7, k: int = 2, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    known = _spd(rows, d, seed + 1)
    residual = torch.randn(rows, d, generator=gen, dtype=F64)
    scale = 0.3 * torch.rand(rows, d, generator=gen, dtype=F64)
    factor = 0.4 * torch.randn(rows, d, k, generator=gen, dtype=F64)
    return residual, known, scale, factor


def _scipy_nll(residual, cov, mask=None) -> np.ndarray:
    out = []
    for i in range(residual.shape[0]):
        m = torch.ones(residual.shape[1], dtype=torch.bool) if mask is None else mask[i]
        if not bool(m.any()):
            out.append(0.0)
            continue
        c = cov[i][m][:, m].numpy()
        r = residual[i][m].numpy()
        out.append(-stats.multivariate_normal(np.zeros(len(r)), c).logpdf(r))
    return np.asarray(out)


def _total(known, scale, factor):
    return known + torch.diag_embed(scale.square()).to(dtype=known.dtype) + factor @ factor.mT


class TestScipyParity:
    def test_dense_known_plus_model_matches_scipy(self) -> None:
        residual, known, scale, factor = _problem()
        res = known_covariance_gaussian_nll(
            residual, known, model_scale=scale, model_factor=factor, return_whitened=True
        )
        assert isinstance(res, KnownCovarianceGaussianResult)
        cov = _total(known, scale, factor)
        np.testing.assert_allclose(res.nll.numpy(), _scipy_nll(residual, cov), atol=1e-11)
        quad = (residual.unsqueeze(-2) @ torch.linalg.solve(cov, residual.unsqueeze(-1))).flatten()
        torch.testing.assert_close(res.chi2, quad, atol=1e-11, rtol=0)
        torch.testing.assert_close(res.log_det, torch.linalg.slogdet(cov)[1], atol=1e-11, rtol=0)
        torch.testing.assert_close(res.chi2, res.whitened.square().sum(-1), atol=1e-12, rtol=0)
        assert torch.equal(res.dof, torch.full((6,), 7.0, dtype=F64))
        assert torch.equal(res.jitter, torch.zeros(6, dtype=F64))

    def test_shared_known_covariance_and_factor_broadcast(self) -> None:
        residual, known, scale, factor = _problem()
        res = known_covariance_gaussian_nll(
            residual, known[0], model_scale=scale[0], model_factor=factor[0]
        )
        cov = _total(known[0], scale[0], factor[0]).expand(6, -1, -1)
        np.testing.assert_allclose(res.nll.numpy(), _scipy_nll(residual, cov), atol=1e-11)

    def test_whitened_residual_is_the_cholesky_whitening(self) -> None:
        residual, known, scale, factor = _problem()
        res = known_covariance_gaussian_nll(
            residual, known, model_scale=scale, model_factor=factor, return_whitened=True
        )
        chol = torch.linalg.cholesky(_total(known, scale, factor))
        ref = torch.linalg.solve_triangular(chol, residual.unsqueeze(-1), upper=False).squeeze(-1)
        torch.testing.assert_close(res.whitened, ref, atol=1e-11, rtol=0)

    def test_whitened_residuals_are_standard_normal(self) -> None:
        gen = torch.Generator().manual_seed(3)
        d, n = 5, 20000
        known = _spd(1, d, 4)[0]
        chol = torch.linalg.cholesky(known)
        draws = (chol @ torch.randn(d, n, generator=gen, dtype=F64)).T
        res = known_covariance_gaussian_nll(draws, known, return_whitened=True)
        cov = torch.cov(res.whitened.T)
        torch.testing.assert_close(cov, torch.eye(d, dtype=F64), atol=0.04, rtol=0)
        assert abs(float(res.chi2.mean()) - d) < 0.1


class TestMasking:
    def test_masked_equals_principal_submatrix_marginal(self) -> None:
        residual, known, scale, factor = _problem(rows=8)
        gen = torch.Generator().manual_seed(5)
        mask = torch.rand(8, 7, generator=gen) > 0.35
        mask[0] = False
        mask[1] = True
        mask[2] = False
        mask[2, 3] = True
        res = known_covariance_gaussian_nll(
            residual, known, model_scale=scale, model_factor=factor, mask=mask, return_whitened=True
        )
        cov = _total(known, scale, factor)
        np.testing.assert_allclose(res.nll.numpy(), _scipy_nll(residual, cov, mask), atol=1e-11)
        torch.testing.assert_close(res.dof, mask.sum(-1).to(F64))
        assert res.nll[0] == 0 and res.chi2[0] == 0
        assert torch.all(res.whitened[~mask] == 0)

    def test_nan_at_masked_entries_is_ignored_with_finite_gradients(self) -> None:
        residual, known, scale, factor = _problem()
        mask = torch.ones(6, 7, dtype=torch.bool)
        mask[:, 2] = False
        mask[3, 5] = False
        r_nan = residual.clone()
        r_nan[~mask] = float("nan")
        k_nan = known.clone()
        k_nan[:, 2, :] = float("nan")
        k_nan[:, :, 2] = float("nan")
        r_nan.requires_grad_(True)
        res = known_covariance_gaussian_nll(r_nan, k_nan, model_scale=scale, mask=mask)
        ref = known_covariance_gaussian_nll(residual, known, model_scale=scale, mask=mask)
        torch.testing.assert_close(res.nll, ref.nll, atol=1e-12, rtol=0)
        res.nll.sum().backward()
        assert r_nan.grad is not None and torch.isfinite(r_nan.grad).all()
        assert torch.all(r_nan.grad[~mask] == 0)

    def test_loss_mean_drops_empty_rows(self) -> None:
        residual, known, _, _ = _problem()
        mask = torch.ones(6, 7, dtype=torch.bool)
        mask[4] = False
        loss = KnownCovarianceGaussianNLL()
        per_row = known_covariance_gaussian_nll(residual, known, mask=mask).nll
        value = loss(torch.zeros_like(residual), residual, known_covariance=known, mask=mask)
        torch.testing.assert_close(value, per_row[mask.any(-1)].mean())


class TestSpecialCases:
    def test_diagonal_case_equals_existing_diagonal_losses(self) -> None:
        gen = torch.Generator().manual_seed(7)
        y_pred = torch.randn(9, 5, generator=gen, dtype=F64)
        target = torch.randn(9, 5, generator=gen, dtype=F64)
        known_var = torch.rand(9, 5, generator=gen, dtype=F64) + 0.1
        log_s2 = torch.randn(9, 5, generator=gen, dtype=F64)
        scale = torch.exp(0.5 * log_s2)
        res = known_covariance_gaussian_nll(
            target - y_pred, known_variance=known_var, model_scale=scale
        )
        noisy = NoisyTargetGaussianNLL(reduction="none", min_variance=0.0)(
            (y_pred, log_s2), target, target_variance=known_var
        )
        torch.testing.assert_close(res.nll, noisy.sum(-1), atol=1e-12, rtol=0)
        gauss = GaussianNLLLoss(reduction="none", min_variance=1e-300, eps=0.0, log_variance=False)
        total_var = known_var + scale.square()
        torch.testing.assert_close(
            res.nll, gauss((y_pred, total_var), target).sum(-1), atol=1e-12, rtol=0
        )
        dense = known_covariance_gaussian_nll(
            target - y_pred, torch.diag_embed(known_var).to(dtype=F64), model_scale=scale
        )
        torch.testing.assert_close(res.nll, dense.nll, atol=1e-12, rtol=0)
        torch.testing.assert_close(res.chi2, dense.chi2, atol=1e-12, rtol=0)

    def test_dense_known_only_matches_multivariate_gaussian_loss(self) -> None:
        residual, known, _, _ = _problem()
        res = known_covariance_gaussian_nll(residual, known)
        ref = MultivariateGaussianLoss(jitter=0.0, eps=0.0, reduction="none")(
            torch.zeros_like(residual), residual, known
        )
        torch.testing.assert_close(res.nll, ref, atol=1e-12, rtol=0)

    @pytest.mark.parametrize("masked", [False, True])
    def test_low_rank_matches_woodbury(self, masked: bool) -> None:
        residual, _, scale, factor = _problem(rows=7, d=9, k=3)
        gen = torch.Generator().manual_seed(11)
        known_var = torch.rand(7, 9, generator=gen, dtype=F64) + 0.2
        mask = (torch.rand(7, 9, generator=gen) > 0.3) if masked else None
        res = known_covariance_gaussian_nll(
            residual, known_variance=known_var, model_scale=scale, model_factor=factor, mask=mask
        )
        # Woodbury identity with the low-rank term as a linear nuisance a ~ N(0, I).
        woodbury = marginalize_linear_nuisance(
            torch.zeros_like(residual),
            residual,
            1.0 / (known_var + scale.square()),
            factor,
            prior_precision=1.0,
            mask=mask,
        )
        torch.testing.assert_close(res.nll, woodbury.nll, atol=1e-11, rtol=0)
        if not masked:
            low_rank = LowRankGaussianLoss(min_variance=0.0, jitter=0.0, reduction="none")(
                torch.zeros_like(residual), residual, factor, known_var + scale.square()
            )
            torch.testing.assert_close(res.nll, low_rank, atol=1e-11, rtol=0)

    def test_matches_marginalized_multisource(self) -> None:
        # R = known + diag(s^2), H L = U with L = I: same covariance R + U U^T.
        residual, known, scale, factor = _problem(rows=5, d=6, k=2)
        gen = torch.Generator().manual_seed(2)
        mask = torch.rand(5, 6, generator=gen) > 0.25
        res = known_covariance_gaussian_nll(
            residual, known, model_scale=scale, model_factor=factor, mask=mask
        )
        r_meas = known + torch.diag_embed(scale.square()).to(dtype=F64)
        obs = PackedObservations(residual, torch.zeros_like(residual), factor, r_meas, mask)
        eye = torch.eye(2, dtype=F64).expand(5, 2, 2)
        ref = MarginalizedMultiSourceGaussianNLLLoss(reduction="none", method="direct").per_star(
            torch.zeros(5, 2, dtype=F64), eye, obs
        )
        torch.testing.assert_close(res.nll, ref, atol=1e-11, rtol=0)


class TestGradients:
    def test_gradcheck_dense_all_inputs(self) -> None:
        residual, known, scale, factor = _problem(rows=3, d=4, k=2)
        mask = torch.tensor(
            [[True, True, False, True], [True, True, True, True], [False, True, True, False]]
        )
        a = torch.linalg.cholesky(known).clone()

        def fn(r, a_, s, u):
            k = a_ @ a_.mT  # symmetric by construction
            return known_covariance_gaussian_nll(r, k, model_scale=s, model_factor=u, mask=mask).nll

        inputs = tuple(x.clone().requires_grad_(True) for x in (residual, a, scale + 0.1, factor))
        assert torch.autograd.gradcheck(fn, inputs)

    def test_gradgradcheck_residual_and_scale(self) -> None:
        # Newton-type optimisers (e.g. inference.maximize_score) need Hessians.
        residual, known, scale, factor = _problem(rows=2, d=3, k=1)

        def fn(r, s):
            return known_covariance_gaussian_nll(r, known, model_scale=s, model_factor=factor).nll

        inputs = (residual.clone().requires_grad_(True), (scale + 0.2).requires_grad_(True))
        assert torch.autograd.gradgradcheck(fn, inputs)

    def test_gradcheck_diagonal(self) -> None:
        gen = torch.Generator().manual_seed(1)
        r = torch.randn(4, 3, generator=gen, dtype=F64, requires_grad=True)
        v = (torch.rand(4, 3, generator=gen, dtype=F64) + 0.2).requires_grad_(True)
        s = (torch.rand(3, generator=gen, dtype=F64) + 0.1).requires_grad_(True)

        def fn(r_, v_, s_):
            return known_covariance_gaussian_nll(r_, known_variance=v_, model_scale=s_).nll

        assert torch.autograd.gradcheck(fn, (r, v, s))


class TestLearnedScatter:
    def test_fractional_terms_follow_the_formula(self) -> None:
        torch.manual_seed(0)
        loss = KnownCovarianceGaussianNLL(4, scatter="fractional", rank=2, init_fraction=0.05)
        y = torch.randn(3, 4)
        variance, factor = loss.model_covariance_terms(y)
        a = torch.exp(loss.log_fraction)
        b = torch.exp(loss.log_floor)
        torch.testing.assert_close(variance, (a * y).square() + b.square())
        torch.testing.assert_close(factor, y.unsqueeze(-1) * loss.factor_weight)
        assert {n for n, _ in loss.named_parameters()} == {
            "log_fraction",
            "log_floor",
            "factor_weight",
        }

    def test_learned_and_given_terms_add(self) -> None:
        torch.manual_seed(1)
        d = 5
        loss = KnownCovarianceGaussianNLL(d, scatter="fractional", rank=1, reduction="none")
        known = _spd(4, d, 3, dtype=torch.float32)
        y = torch.randn(4, d)
        t = torch.randn(4, d)
        s_given = torch.rand(4, d)
        u_given = torch.randn(4, d, 2) * 0.1
        got = loss(y, t, known_covariance=known, model_scale=s_given, model_factor=u_given)
        variance, factor = loss.model_covariance_terms(y)
        ref = known_covariance_gaussian_nll(
            t - y,
            known,
            model_scale=torch.sqrt(variance + s_given.square()),
            model_factor=torch.cat([factor, u_given], dim=-1),
        ).nll
        torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)

    def test_fixed_zero_floor_and_constant_factor(self) -> None:
        loss = KnownCovarianceGaussianNLL(
            3, scatter="constant", learn_floor=False, init_floor=0.0, rank=1, factor="constant"
        )
        variance, factor = loss.model_covariance_terms(torch.ones(2, 3))
        assert torch.equal(variance, torch.zeros(2, 3))
        assert factor.shape == (1, 3, 1)
        assert [n for n, _ in loss.named_parameters()] == ["factor_weight"]

    def test_maximum_likelihood_recovers_fractional_scatter(self) -> None:
        # Known per-row noise plus 8 % fractional and 0.05 additive model scatter.
        gen = torch.Generator().manual_seed(4)
        n, d = 4000, 3
        mu = 1.0 + 4.0 * torch.rand(n, d, generator=gen, dtype=F64)
        known_sd = 0.05 + 0.2 * torch.rand(n, d, generator=gen, dtype=F64)
        model_sd = torch.sqrt((0.08 * mu).square() + 0.05**2)
        y = mu + torch.randn(n, d, generator=gen, dtype=F64) * torch.sqrt(
            known_sd.square() + model_sd.square()
        )
        loss = KnownCovarianceGaussianNLL(
            d, scatter="fractional", init_fraction=0.01, init_floor=0.2
        )
        loss = loss.to(F64)
        opt = torch.optim.Adam(loss.parameters(), lr=0.05)
        for _ in range(400):
            opt.zero_grad()
            value = loss(mu, y, known_variance=known_sd.square())
            value.backward()
            opt.step()
        frac = torch.exp(loss.log_fraction).detach()
        floor = torch.exp(loss.log_floor).detach()
        assert torch.allclose(frac, torch.full((d,), 0.08, dtype=F64), atol=0.012)
        assert torch.allclose(floor, torch.full((d,), 0.05, dtype=F64), atol=0.04)

    def test_constructor_validation(self) -> None:
        with pytest.raises(ValueError, match="n_features"):
            KnownCovarianceGaussianNLL(scatter="fractional")
        with pytest.raises(ValueError, match="scatter"):
            KnownCovarianceGaussianNLL(3, scatter="bogus")
        with pytest.raises(ValueError, match="rank"):
            KnownCovarianceGaussianNLL(3, rank=-1)
        with pytest.raises(ValueError, match="init_floor"):
            KnownCovarianceGaussianNLL(3, scatter="constant", init_floor=0.0)
        loss = KnownCovarianceGaussianNLL(3, scatter="constant")
        with pytest.raises(ValueError, match="features"):
            loss.model_covariance_terms(torch.ones(2, 4))


class TestNumerics:
    def test_badly_scaled_covariance_in_float32_matches_float64(self) -> None:
        # Feature scales spanning 1e-4 .. 1e4: equilibration makes float32 accurate.
        d = 12
        corr = _spd(4, d, 9)
        sd = torch.sqrt(torch.diagonal(corr, dim1=-2, dim2=-1))
        corr = corr / sd.unsqueeze(-1) / sd.unsqueeze(-2)
        units = torch.logspace(-4, 4, d, dtype=F64)
        known = corr * units.unsqueeze(-1) * units.unsqueeze(-2)
        gen = torch.Generator().manual_seed(1)
        residual = units * torch.randn(4, d, generator=gen, dtype=F64)
        ref = known_covariance_gaussian_nll(residual, known)
        got = known_covariance_gaussian_nll(residual.float(), known.float())
        torch.testing.assert_close(got.chi2.double(), ref.chi2, rtol=1e-3, atol=1e-3)
        # The log-determinant spans ~ +-100 nats; compare the nll relative to that.
        torch.testing.assert_close(got.nll.double(), ref.nll, rtol=1e-4, atol=2e-3)
        # scipy's eigenvalue cut-off declares the unequilibrated matrix singular; compare on
        # the equilibrated problem: log N(r; 0, D C D) = log N(r / u; 0, C) - sum(log u).
        scaled = _scipy_nll(residual / units, corr) + float(torch.log(units).sum())
        np.testing.assert_allclose(ref.nll.numpy(), scaled, rtol=1e-9)

    def test_nearly_singular_covariance_float64(self) -> None:
        # Condition number ~1e12, reference in 50-digit arithmetic on the same matrix.
        mpmath = pytest.importorskip("mpmath")
        d = 6
        gen = torch.Generator().manual_seed(2)
        q, _ = torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=F64))
        scale = torch.logspace(-3, 3, d, dtype=F64)
        known = (q * torch.logspace(0, -12, d, dtype=F64)) @ q.T
        known = 0.5 * (known + known.T) * scale.unsqueeze(-1) * scale.unsqueeze(-2)
        residual = torch.randn(3, d, generator=gen, dtype=F64) * 1e-4 * scale
        res = known_covariance_gaussian_nll(residual, known)
        with mpmath.workdps(50):
            k_mp = mpmath.matrix(known.tolist())
            chol = mpmath.cholesky(k_mp)
            ref_logdet = float(2 * sum(mpmath.log(chol[i, i]) for i in range(d)))
            ref_chi2 = []
            for row in residual.tolist():
                w = mpmath.lu_solve(chol, mpmath.matrix(row))
                ref_chi2.append(float(sum(x**2 for x in w)))
        # Cholesky is backward stable on the equilibrated matrix: eigenvalue errors are
        # relative ~ d * eps * cond(R) (~1e-3 here), independent of the 1e-3..1e3 units.
        assert abs(float(res.log_det[0]) - ref_logdet) < 1e-3
        torch.testing.assert_close(res.chi2, torch.tensor(ref_chi2, dtype=F64), rtol=1e-3, atol=0)

    def test_singular_known_covariance_needs_model_term_or_jitter(self) -> None:
        d = 5
        gen = torch.Generator().manual_seed(3)
        v = torch.randn(d, 2, generator=gen, dtype=F64)
        known = (v @ v.T).expand(2, d, d)  # rank 2
        residual = torch.randn(2, d, generator=gen, dtype=F64)
        with pytest.raises(ValueError, match="positive definite"):
            known_covariance_gaussian_nll(residual, known)
        # A model floor regularises it exactly (no jitter needed).
        floor = torch.full((d,), 0.1, dtype=F64)
        res = known_covariance_gaussian_nll(residual, known, model_scale=floor)
        assert torch.equal(res.jitter, torch.zeros(2, dtype=F64))
        np.testing.assert_allclose(
            res.nll.numpy(), _scipy_nll(residual, known + torch.diag(floor.square())), atol=1e-10
        )
        # Opt-in jitter factorises it and reports the amount used.
        res = known_covariance_gaussian_nll(residual, known, max_jitter=1e-6)
        assert torch.isfinite(res.nll).all()
        assert torch.all(res.jitter > 0) and torch.all(res.jitter <= 1e-6)

    def test_jitter_is_applied_only_to_failing_rows(self) -> None:
        d = 4
        good = _spd(1, d, 1)[0]
        bad = torch.ones(d, d, dtype=F64)  # rank 1
        known = torch.stack([good, bad])
        residual = torch.ones(2, d, dtype=F64) * 0.1
        res = known_covariance_gaussian_nll(residual, known, max_jitter=1e-4)
        assert res.jitter[0] == 0 and res.jitter[1] > 0
        ref = known_covariance_gaussian_nll(residual[:1], known[:1])
        torch.testing.assert_close(res.nll[:1], ref.nll)

    def test_compute_dtype_float64_for_float32_inputs(self) -> None:
        residual, known, scale, factor = _problem()
        ref = known_covariance_gaussian_nll(residual, known, model_scale=scale, model_factor=factor)
        got = known_covariance_gaussian_nll(
            residual.float(),
            known.float(),
            model_scale=scale.float(),
            model_factor=factor.float(),
            compute_dtype=F64,
        )
        assert got.nll.dtype == torch.float32
        torch.testing.assert_close(got.nll.double(), ref.nll, atol=1e-5, rtol=1e-6)


class TestValidation:
    def test_invalid_inputs_raise(self) -> None:
        residual, known, scale, _ = _problem()
        with pytest.raises(ValueError, match="not both"):
            known_covariance_gaussian_nll(residual, known, known_variance=scale)
        with pytest.raises(ValueError, match="shape"):
            known_covariance_gaussian_nll(residual, known[:, :3, :3])
        with pytest.raises(ValueError, match=">= 0"):
            known_covariance_gaussian_nll(residual, known_variance=-scale - 0.1)
        asym = known.clone()
        asym[:, 0, 1] += 1.0
        with pytest.raises(ValueError, match="symmetric"):
            known_covariance_gaussian_nll(residual, asym)
        bad = known.clone()
        bad[0, 1, 1] = float("inf")
        with pytest.raises(ValueError, match="finite"):
            known_covariance_gaussian_nll(residual, bad)
        with pytest.raises(ValueError, match="variance must be > 0"):
            known_covariance_gaussian_nll(residual)
        with pytest.raises(TypeError, match="boolean"):
            known_covariance_gaussian_nll(residual, known, mask=torch.ones(6, 7))
        with pytest.raises(ValueError, match="jitter"):
            known_covariance_gaussian_nll(residual, known, jitter=-1.0)

    def test_negative_known_variance_on_masked_dims_is_ignored(self) -> None:
        residual, _, scale, _ = _problem()
        kv = scale.square() + 0.1
        mask = torch.ones(6, 7, dtype=torch.bool)
        mask[:, 0] = False
        kv_bad = kv.clone()
        kv_bad[:, 0] = -1.0
        a = known_covariance_gaussian_nll(residual, known_variance=kv_bad, mask=mask)
        b = known_covariance_gaussian_nll(residual, known_variance=kv, mask=mask)
        torch.testing.assert_close(a.nll, b.nll)


class TestLossClass:
    def test_reductions_and_row_weights(self) -> None:
        residual, known, scale, _ = _problem()
        y = torch.zeros_like(residual)
        per_row = KnownCovarianceGaussianNLL(reduction="none")(
            y, residual, known_covariance=known, model_scale=scale
        )
        ref = known_covariance_gaussian_nll(residual, known, model_scale=scale).nll
        torch.testing.assert_close(per_row, ref)
        total = KnownCovarianceGaussianNLL(reduction="sum")(
            y, residual, known_covariance=known, model_scale=scale
        )
        torch.testing.assert_close(total, ref.sum())
        w = torch.linspace(0.5, 2.0, 6, dtype=F64)
        mean = KnownCovarianceGaussianNLL()(
            y, residual, known_covariance=known, model_scale=scale, weights=w
        )
        torch.testing.assert_close(mean, (ref * w).sum() / w.sum())

    def test_evaluate_returns_whitened(self) -> None:
        residual, known, _, _ = _problem()
        res = KnownCovarianceGaussianNLL().evaluate(
            torch.zeros_like(residual), residual, known_covariance=known, return_whitened=True
        )
        assert res.whitened is not None and res.whitened.shape == residual.shape

    def test_registry(self) -> None:
        assert get_regression_loss("known_covariance_gaussian_nll") is KnownCovarianceGaussianNLL


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_matches_cpu() -> None:
    residual, known, scale, factor = _problem()
    mask = torch.rand(6, 7, generator=torch.Generator().manual_seed(0)) > 0.2
    ref = known_covariance_gaussian_nll(
        residual, known, model_scale=scale, model_factor=factor, mask=mask
    )
    got = known_covariance_gaussian_nll(
        residual.cuda(),
        known.cuda(),
        model_scale=scale.cuda(),
        model_factor=factor.cuda(),
        mask=mask.cuda(),
    )
    torch.testing.assert_close(got.nll.cpu(), ref.nll)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_matches_cpu_float32() -> None:
    residual, known, scale, factor = _problem()
    ref = known_covariance_gaussian_nll(residual, known, model_scale=scale, model_factor=factor)
    dev = torch.device("mps")
    got = known_covariance_gaussian_nll(
        residual.float().to(dev),
        known.float().to(dev),
        model_scale=scale.float().to(dev),
        model_factor=factor.float().to(dev),
    )
    torch.testing.assert_close(got.nll.cpu().double(), ref.nll, atol=1e-4, rtol=1e-4)


def test_constant_matches_log_two_pi() -> None:
    res = known_covariance_gaussian_nll(
        torch.zeros(1, 3, dtype=F64), known_variance=torch.ones(3, dtype=F64)
    )
    assert math.isclose(float(res.nll), 1.5 * math.log(2 * math.pi), rel_tol=1e-14)


def test_docs_example_runs() -> None:
    page = Path(__file__).resolve().parents[2] / "docs" / "losses" / "known_covariance.md"
    blocks = re.findall(r"```python\s*\n(.*?)```", page.read_text(encoding="utf-8"), re.DOTALL)
    assert blocks
    for block in blocks:
        exec(compile(block, str(page), "exec"), {})
