"""Tests for the marginalised linear-nuisance Gaussian NLL."""

from __future__ import annotations

import math

import pytest
import torch
from torch.distributions import LowRankMultivariateNormal, MultivariateNormal

from torchregress.losses import (
    LinearNuisanceMarginal,
    MarginalizedNuisanceGaussianNLLLoss,
    marginalize_linear_nuisance,
)

F64 = torch.float64


def _problem(rows: int = 4, n: int = 25, m: int = 3, seed: int = 0, dtype: torch.dtype = F64):
    gen = torch.Generator().manual_seed(seed)
    t = torch.linspace(-1.0, 1.0, n, dtype=dtype)
    basis = torch.stack([t**k for k in range(m)], dim=-1)  # polynomial baseline
    y_pred = torch.randn(rows, n, generator=gen, dtype=dtype)
    target = torch.randn(rows, n, generator=gen, dtype=dtype)
    precision = torch.rand(rows, n, generator=gen, dtype=dtype) + 0.5
    return y_pred, target, precision, basis


def _spd(m: int, seed: int = 1) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(m, m, generator=gen, dtype=F64)
    return a @ a.T + 0.5 * torch.eye(m, dtype=F64)


def _dense_nll(y_pred, target, precision, basis, prior_cov):
    cov = torch.diag_embed(1.0 / precision).to(dtype=precision.dtype) + basis @ prior_cov @ basis.mT
    zero = torch.zeros(target.shape[-1], dtype=target.dtype)
    return -MultivariateNormal(zero, covariance_matrix=cov).log_prob(target - y_pred)


class TestDenseParity:
    def test_full_prior_covariance_matches_multivariate_normal(self) -> None:
        y_pred, target, precision, basis = _problem()
        prior_cov = _spd(3)
        res = marginalize_linear_nuisance(
            y_pred, target, precision, basis, prior_covariance=prior_cov
        )
        assert isinstance(res, LinearNuisanceMarginal)
        ref = _dense_nll(y_pred, target, precision, basis, prior_cov)
        torch.testing.assert_close(res.nll, ref, atol=1e-11, rtol=0)

    def test_diagonal_prior_precision_matches_multivariate_normal(self) -> None:
        y_pred, target, precision, basis = _problem()
        p = torch.tensor([0.5, 2.0, 7.0], dtype=F64)
        res = marginalize_linear_nuisance(y_pred, target, precision, basis, prior_precision=p)
        ref = _dense_nll(y_pred, target, precision, basis, torch.diag_embed(1.0 / p).to(dtype=F64))
        torch.testing.assert_close(res.nll, ref, atol=1e-11, rtol=0)
        scalar = marginalize_linear_nuisance(y_pred, target, precision, basis, prior_precision=2.0)
        ref2 = _dense_nll(y_pred, target, precision, basis, torch.eye(3, dtype=F64) / 2.0)
        torch.testing.assert_close(scalar.nll, ref2, atol=1e-11, rtol=0)

    def test_per_row_basis_and_prior_match_low_rank_normal(self) -> None:
        y_pred, target, precision, basis = _problem(rows=3)
        gen = torch.Generator().manual_seed(4)
        basis_rows = basis + 0.1 * torch.randn(3, *basis.shape, generator=gen, dtype=F64)
        prior_cov = torch.stack([_spd(3, seed=s) for s in range(3)])
        res = marginalize_linear_nuisance(
            y_pred, target, precision, basis_rows, prior_covariance=prior_cov
        )
        factor = basis_rows @ torch.linalg.cholesky(prior_cov)
        ref = -LowRankMultivariateNormal(y_pred, factor, 1.0 / precision).log_prob(target)
        torch.testing.assert_close(res.nll, ref, atol=1e-10, rtol=0)

    def test_posterior_mean_and_covariance_match_dense_formulas(self) -> None:
        y_pred, target, precision, basis = _problem(rows=2)
        prior_cov = _spd(3)
        res = marginalize_linear_nuisance(
            y_pred, target, precision, basis, prior_covariance=prior_cov
        )
        for i in range(2):
            lam = torch.diag(precision[i]).to(dtype=F64)
            post_prec = torch.linalg.inv(prior_cov) + basis.T @ lam @ basis
            post_cov = torch.linalg.inv(post_prec)
            mean = post_cov @ basis.T @ lam @ (target[i] - y_pred[i])
            torch.testing.assert_close(res.coefficients[i], mean, atol=1e-11, rtol=0)
            torch.testing.assert_close(res.coefficient_covariance[i], post_cov, atol=1e-11, rtol=0)
            # E[Ua | r] equals the GP-style conditional mean U S U^T C^{-1} r.
            cov = torch.diag(1.0 / precision[i]).to(dtype=F64) + basis @ prior_cov @ basis.T
            cond = basis @ prior_cov @ basis.T @ torch.linalg.solve(cov, target[i] - y_pred[i])
            torch.testing.assert_close(res.nuisance[i], cond, atol=1e-11, rtol=0)
        torch.testing.assert_close(res.residual, target - y_pred - res.nuisance)

    @pytest.mark.parametrize("rows", [1, 2, 5])  # rows <= (m + 1) / 2 takes the per-row path
    def test_shared_and_expanded_basis_agree(self, rows: int) -> None:
        y_pred, target, precision, basis = _problem(rows=rows, m=4)
        shared = marginalize_linear_nuisance(y_pred, target, precision, basis, prior_precision=1.0)
        expanded = marginalize_linear_nuisance(
            y_pred, target, precision, basis.expand(rows, *basis.shape), prior_precision=1.0
        )
        for a, b in zip(shared, expanded):
            torch.testing.assert_close(a, b, atol=1e-12, rtol=0)


class TestMasking:
    @pytest.mark.parametrize("shared", [True, False])
    def test_masking_equals_dropping_samples(self, shared: bool) -> None:
        y_pred, target, precision, basis = _problem(rows=4, n=30)
        mask = torch.rand(4, 30, generator=torch.Generator().manual_seed(3)) > 0.3
        b = basis if shared else basis.expand(4, *basis.shape).clone()
        target_bad = target.clone()
        target_bad[~mask] = float("nan")
        precision_bad = precision.clone()
        precision_bad[~mask] = float("inf")
        if not shared:
            b[~mask] = float("nan")
        prior = torch.tensor([1.0, 0.5, 0.0], dtype=F64)  # flat in one direction
        res = marginalize_linear_nuisance(
            y_pred, target_bad, precision_bad, b, prior_precision=prior, mask=mask
        )
        for i in range(4):
            keep = mask[i]
            ref = marginalize_linear_nuisance(
                y_pred[i, keep],
                target[i, keep],
                precision[i, keep],
                basis[keep],
                prior_precision=prior,
            )
            torch.testing.assert_close(res.nll[i], ref.nll, atol=1e-12, rtol=0)
            torch.testing.assert_close(res.coefficients[i], ref.coefficients, atol=1e-12, rtol=0)
            torch.testing.assert_close(
                res.coefficient_covariance[i], ref.coefficient_covariance, atol=1e-12, rtol=0
            )
        if shared:
            # The fitted nuisance is evaluated at masked samples too (interpolation).
            torch.testing.assert_close(res.nuisance, res.coefficients @ basis.T)

    def test_nan_at_masked_samples_does_not_leak_into_gradients(self) -> None:
        y_pred, target, precision, basis = _problem()
        mask = torch.ones_like(target, dtype=torch.bool)
        mask[:, 3:6] = False
        target = target.clone()
        target[~mask] = float("nan")
        pred = y_pred.clone().requires_grad_(True)
        loss = MarginalizedNuisanceGaussianNLLLoss()(pred, target, precision, basis, mask=mask)
        loss.backward()
        assert torch.isfinite(loss)
        assert pred.grad is not None and torch.isfinite(pred.grad).all()
        assert (pred.grad[~mask] == 0).all()

    def test_empty_row_with_proper_prior_is_zero_and_dropped(self) -> None:
        y_pred, target, precision, basis = _problem(rows=3)
        mask = torch.ones_like(target, dtype=torch.bool)
        mask[1] = False
        res = marginalize_linear_nuisance(
            y_pred, target, precision, basis, prior_precision=2.0, mask=mask
        )
        torch.testing.assert_close(res.nll[1], torch.tensor(0.0, dtype=F64), atol=1e-14, rtol=0)
        torch.testing.assert_close(res.coefficients[1], torch.zeros(3, dtype=F64))
        torch.testing.assert_close(res.coefficient_covariance[1], torch.eye(3, dtype=F64) / 2.0)
        loss = MarginalizedNuisanceGaussianNLLLoss()(
            y_pred, target, precision, basis, prior_precision=2.0, mask=mask
        )
        torch.testing.assert_close(loss, res.nll[[0, 2]].mean())


class TestFlatPrior:
    def test_flat_limit_equals_profile_plus_log_det(self) -> None:
        y_pred, target, precision, basis = _problem(rows=3, n=40, m=4)
        flat = marginalize_linear_nuisance(y_pred, target, precision, basis)
        resid = target - y_pred
        for i in range(3):
            w = precision[i].sqrt()
            # Explicit profile: weighted least squares for the nuisance.
            coef = torch.linalg.lstsq(basis * w[:, None], resid[i] * w).solution
            torch.testing.assert_close(flat.coefficients[i], coef, atol=1e-11, rtol=0)
            chi2 = (precision[i] * (resid[i] - basis @ coef) ** 2).sum()
            profile = 0.5 * (40 * math.log(2 * math.pi) - precision[i].log().sum() + chi2)
            gram = basis.T @ (precision[i, :, None] * basis)
            torch.testing.assert_close(
                flat.nll[i], profile + 0.5 * torch.logdet(gram), atol=1e-11, rtol=0
            )
            torch.testing.assert_close(
                flat.coefficient_covariance[i], torch.linalg.inv(gram), atol=1e-11, rtol=0
            )

    def test_flat_is_the_limit_of_vanishing_prior_precision(self) -> None:
        y_pred, target, precision, basis = _problem()
        flat = marginalize_linear_nuisance(y_pred, target, precision, basis)
        errors = []
        for eps in (1e-2, 1e-4, 1e-6):
            proper = marginalize_linear_nuisance(
                y_pred, target, precision, basis, prior_precision=eps
            )
            # Restricted-likelihood convention: drop -1/2 log|P| = -1/2 m log(eps).
            errors.append((proper.nll + 0.5 * 3 * math.log(eps) - flat.nll).abs().max().item())
        assert errors[0] > errors[1] > errors[2]
        assert errors[2] < 1e-5

    def test_flat_nll_is_invariant_to_nuisance_shifts(self) -> None:
        y_pred, target, precision, basis = _problem()
        flat = marginalize_linear_nuisance(y_pred, target, precision, basis)
        a = torch.tensor([3.0, -2.0, 5.0], dtype=F64)
        shifted = marginalize_linear_nuisance(y_pred, target + basis @ a, precision, basis)
        torch.testing.assert_close(shifted.nll, flat.nll, atol=1e-10, rtol=0)
        torch.testing.assert_close(shifted.coefficients, flat.coefficients + a, atol=1e-10, rtol=0)

    def test_partially_flat_prior_matches_dense_limit(self) -> None:
        y_pred, target, precision, basis = _problem()
        mixed = torch.tensor([0.0, 3.0, 0.5], dtype=F64)
        res = marginalize_linear_nuisance(y_pred, target, precision, basis, prior_precision=mixed)
        eps = 1e-7
        cov = torch.diag_embed(1.0 / torch.tensor([eps, 3.0, 0.5], dtype=F64)).to(dtype=F64)
        ref = _dense_nll(y_pred, target, precision, basis, cov) + 0.5 * math.log(eps)
        torch.testing.assert_close(res.nll, ref, atol=1e-5, rtol=0)


class TestFailClosed:
    def test_empty_row_with_flat_prior_raises(self) -> None:
        y_pred, target, precision, basis = _problem()
        mask = torch.ones_like(target, dtype=torch.bool)
        mask[2] = False
        with pytest.raises(ValueError, match="1 row"):
            marginalize_linear_nuisance(y_pred, target, precision, basis, mask=mask)

    def test_unconstrained_flat_column_raises(self) -> None:
        y_pred, target, precision, basis = _problem(n=20)
        local = torch.zeros(20, 1, dtype=F64)
        local[:4] = 1.0  # a local term supported only on the first samples
        mask = torch.ones_like(target, dtype=torch.bool)
        mask[0, :4] = False
        with pytest.raises(ValueError, match="1 row"):
            marginalize_linear_nuisance(
                y_pred, target, precision, torch.cat([basis, local], -1), mask=mask
            )
        # The same masked row is fine once that column has a proper prior.
        prior = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=F64)
        res = marginalize_linear_nuisance(
            y_pred,
            target,
            precision,
            torch.cat([basis, local], -1),
            prior_precision=prior,
            mask=mask,
        )
        assert torch.isfinite(res.nll).all()

    def test_collinear_columns_raise(self) -> None:
        y_pred, target, precision, basis = _problem()
        with pytest.raises(ValueError, match="not numerically positive definite"):
            marginalize_linear_nuisance(
                y_pred, target, precision, torch.cat([basis, 2.0 * basis[:, :1]], -1)
            )

    @pytest.mark.parametrize(
        "kwargs,error",
        [
            ({"prior_precision": -1.0}, ValueError),
            ({"prior_covariance": -torch.eye(3, dtype=F64)}, ValueError),
            ({"prior_covariance": torch.eye(2, dtype=F64)}, ValueError),
            ({"prior_precision": 1.0, "prior_covariance": torch.eye(3, dtype=F64)}, ValueError),
            ({"mask": torch.ones(4, 25)}, TypeError),
        ],
    )
    def test_invalid_arguments(self, kwargs, error) -> None:
        y_pred, target, precision, basis = _problem()
        with pytest.raises(error):
            marginalize_linear_nuisance(y_pred, target, precision, basis, **kwargs)

    def test_shape_mismatch_raises(self) -> None:
        y_pred, target, precision, basis = _problem()
        with pytest.raises(ValueError):
            marginalize_linear_nuisance(y_pred, target, precision, basis[:-1])


class TestGradients:
    def test_gradcheck_all_inputs_covariance_prior(self) -> None:
        y_pred, target, precision, basis = _problem(rows=2, n=8, m=2)
        gen = torch.Generator().manual_seed(9)
        basis_rows = (basis + 0.1 * torch.randn(2, 8, 2, generator=gen, dtype=F64)).requires_grad_(
            True
        )
        a = torch.randn(2, 2, generator=gen, dtype=F64)
        prior_cov = (a @ a.T + torch.eye(2, dtype=F64)).requires_grad_(True)
        mask = torch.ones(2, 8, dtype=torch.bool)
        mask[1, 2] = False
        inputs = (
            y_pred.clone().requires_grad_(True),
            precision.clone().requires_grad_(True),
            basis_rows,
            prior_cov,
        )

        def fn(yp, lam, u, cov):
            sym = 0.5 * (cov + cov.T)
            res = marginalize_linear_nuisance(yp, target, lam, u, prior_covariance=sym, mask=mask)
            return res.nll, res.coefficients, res.residual

        assert torch.autograd.gradcheck(fn, inputs)

    def test_gradcheck_diagonal_and_flat_prior(self) -> None:
        y_pred, target, precision, basis = _problem(rows=2, n=10, m=3)
        # p = 0 is a boundary of the prior's domain: differentiate the flat limit
        # separately from a strictly positive diagonal prior.
        p = torch.tensor([0.3, 0.7, 2.0], dtype=F64, requires_grad=True)
        pred = y_pred.clone().requires_grad_(True)
        shared = basis.clone().requires_grad_(True)

        def fn(yp, u, prior):
            res = marginalize_linear_nuisance(yp, target, precision, u, prior_precision=prior)
            return res.nll, res.coefficient_covariance

        def flat(yp, u):
            res = marginalize_linear_nuisance(yp, target, precision, u)
            return res.nll, res.coefficients

        assert torch.autograd.gradcheck(fn, (pred, shared, p))
        assert torch.autograd.gradcheck(flat, (pred, shared))

    def test_float32_is_close_to_float64(self) -> None:
        y_pred, target, precision, basis = _problem(rows=3, n=200, m=4)
        r64 = marginalize_linear_nuisance(y_pred, target, precision, basis, prior_precision=1.0)
        r32 = marginalize_linear_nuisance(
            y_pred.float(), target.float(), precision.float(), basis.float(), prior_precision=1.0
        )
        assert r32.nll.dtype == torch.float32
        torch.testing.assert_close(r32.nll.double(), r64.nll, rtol=1e-5, atol=1e-4)


class TestLossClass:
    def test_reductions_and_row_weights(self) -> None:
        y_pred, target, precision, basis = _problem()
        nll = marginalize_linear_nuisance(y_pred, target, precision, basis, prior_precision=1.0).nll
        args = (y_pred, target, precision, basis)
        torch.testing.assert_close(
            MarginalizedNuisanceGaussianNLLLoss(reduction="none")(*args, prior_precision=1.0), nll
        )
        torch.testing.assert_close(
            MarginalizedNuisanceGaussianNLLLoss(reduction="sum")(*args, prior_precision=1.0),
            nll.sum(),
        )
        w = torch.tensor([1.0, 2.0, 0.0, 1.0], dtype=F64)
        weighted = MarginalizedNuisanceGaussianNLLLoss()(*args, prior_precision=1.0, weights=w)
        torch.testing.assert_close(weighted, (w * nll).sum() / w.sum())

    def test_fitting_through_the_loss_ignores_the_baseline(self) -> None:
        """A model trained through the flat-prior loss recovers the signal, not the baseline."""
        n = 64
        t = torch.linspace(-1.0, 1.0, n, dtype=F64)
        basis = torch.stack([torch.ones_like(t), t], dim=-1)
        signal = torch.exp(-0.5 * (t / 0.1) ** 2)
        target = (2.0 * signal + 5.0 - 3.0 * t).unsqueeze(0)  # amplitude 2 on a linear baseline
        precision = torch.full_like(target, 100.0)
        amp = torch.zeros(1, dtype=F64, requires_grad=True)
        opt = torch.optim.LBFGS([amp], max_iter=50)
        loss_fn = MarginalizedNuisanceGaussianNLLLoss()

        def closure():
            opt.zero_grad()
            loss = loss_fn(amp * signal.unsqueeze(0), target, precision, basis)
            loss.backward()
            return loss

        opt.step(closure)
        torch.testing.assert_close(amp.detach(), torch.tensor([2.0], dtype=F64), atol=1e-8, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_matches_cpu() -> None:
    y_pred, target, precision, basis = _problem()
    mask = torch.rand(target.shape, generator=torch.Generator().manual_seed(0)) > 0.2
    cpu = marginalize_linear_nuisance(
        y_pred, target, precision, basis, prior_precision=1.0, mask=mask
    )
    gpu = marginalize_linear_nuisance(
        y_pred.cuda(),
        target.cuda(),
        precision.cuda(),
        basis.cuda(),
        prior_precision=1.0,
        mask=mask.cuda(),
    )
    for a, b in zip(cpu, gpu):
        torch.testing.assert_close(b.cpu(), a)
