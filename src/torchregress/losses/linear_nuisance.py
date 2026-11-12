"""
Gaussian likelihood with linear nuisance parameters marginalised analytically.

Many sequence-regression targets carry smooth distortions that the model should
not be scored on: a continuum or baseline, a flux-calibration polynomial, a
gain and offset. Writing them as a linear nuisance ``U a`` with a Gaussian
prior ``a ~ N(0, S)`` and integrating ``a`` out gives the exact Gaussian

    target - y_pred ~ N(0, diag(1/precision) + U S U^T)

(Rybicki & Press 1992). :func:`marginalize_linear_nuisance` evaluates its
negative log-likelihood row by row with the Woodbury identity and the matrix
determinant lemma -- one ``m x m`` Cholesky per row, never an ``N x N`` matrix
-- and also returns the posterior mean of the nuisance coefficients (the fitted
baseline) and the conditional residual. :class:`MarginalizedNuisanceGaussianNLLLoss`
is the reducing loss.

The likelihood is exactly invariant to any distortion in the span of ``U`` in
the flat-prior limit, and penalises it only through the prior otherwise: it is
the likelihood-based counterpart of the local gain/offset insensitivity of
:func:`torchregress.losses.structural_similarity_1d`.

References:
    Rybicki & Press (1992): "Interpolation, realization, and reconstruction of
    noisy, irregularly sampled data." ApJ 398: 169-176.
    Patterson & Thompson (1971): "Recovery of inter-block information when
    block sizes are unequal." Biometrika 58(3): 545-554.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple, Optional

import torch
from torch import Tensor

from .base import DistributionLoss
from .loss_registry import register_regression_loss

_LOG_2PI = math.log(2.0 * math.pi)


class LinearNuisanceMarginal(NamedTuple):
    """Result of :func:`marginalize_linear_nuisance` (leading shape ``[...]``).

    Attributes:
        nll: Marginal negative log-likelihood per row, ``[...]``, constants
            included (restricted likelihood in flat-prior directions).
        coefficients: Posterior mean of the nuisance coefficients ``[..., m]``
            (the MAP nuisance; the generalised least-squares fit in the flat limit).
        coefficient_covariance: Posterior covariance of the coefficients
            ``[..., m, m]``.
        nuisance: Fitted nuisance ``U @ coefficients`` at every sample ``[..., N]``,
            masked samples included (evaluated with the basis as given).
        residual: Conditional residual ``target - y_pred - nuisance`` ``[..., N]``;
            masked entries carry whatever the inputs held there.
    """

    nll: Tensor
    coefficients: Tensor
    coefficient_covariance: Tensor
    nuisance: Tensor
    residual: Tensor


def _sym_index(m: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    """Upper-triangle indices of an ``m x m`` matrix and the full -> packed map."""
    iu = torch.triu_indices(m, m, device=device)
    packed = torch.empty((m, m), dtype=torch.long, device=device)
    pos = torch.arange(iu.shape[1], device=device)
    packed[iu[0], iu[1]] = pos
    packed[iu[1], iu[0]] = pos
    return iu[0], iu[1], packed.reshape(-1)


def marginalize_linear_nuisance(
    y_pred: Tensor,
    target: Tensor,
    precision: Tensor,
    basis: Tensor,
    *,
    prior_precision: float | Tensor | None = None,
    prior_covariance: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
) -> LinearNuisanceMarginal:
    r"""Exact Gaussian NLL with linear nuisance parameters integrated out.

    With residual :math:`r = y - \hat y`, diagonal precision
    :math:`\Lambda = \mathrm{diag}(\lambda)` over the :math:`n` unmasked samples,
    nuisance basis :math:`U \in \mathbb R^{N\times m}` and coefficient prior
    :math:`a \sim \mathcal N(0, P^{-1})`, the marginal is
    :math:`r \sim \mathcal N(0, \Lambda^{-1} + U P^{-1} U^\top)` and

    .. math::

        -\log p(r) = \tfrac12\Big[n\log 2\pi - \sum_i \log\lambda_i
        + \log\lvert A\rvert - \log\lvert P\rvert
        + (r - U\hat a)^\top\Lambda(r - U\hat a) + \hat a^\top P\hat a\Big],

    .. math::

        A = P + U^\top\Lambda U,\qquad \hat a = A^{-1}U^\top\Lambda r,
        \qquad \operatorname{Cov}(a\mid r) = A^{-1}

    (Woodbury identity and matrix determinant lemma). The quadratic form is
    evaluated as the conditional residual plus prior penalty, which equals
    :math:`r^\top\Lambda r - \hat a^\top U^\top\Lambda r` without its
    cancellation. With ``prior_covariance`` :math:`S = LL^\top` the same
    algebra runs in whitened coordinates :math:`a = Lc` (``P = I``), so ``S``
    is never inverted.

    Flat directions (``prior_precision`` entries equal to 0) are the
    improper-prior limit: their :math:`\log p_j` is dropped from
    :math:`\log\lvert P\rvert`, which makes the result the *restricted*
    likelihood (REML; Patterson & Thompson 1971). It differs from the profile
    likelihood (nuisance fixed at its least-squares value) by
    :math:`\tfrac12\log\lvert U^\top\Lambda U\rvert` in those directions, a
    term that does not depend on ``y_pred``.

    Args:
        y_pred: Predictions ``[..., N]``.
        target: Observations, broadcastable with ``y_pred``.
        precision: Known inverse variances :math:`\lambda`, broadcastable to
            ``[..., N]``; must be > 0 on unmasked samples (0 gives ``+inf``:
            use ``mask`` to drop a sample).
        basis: Nuisance basis ``[N, m]`` shared by all rows (must be finite),
            or per row ``[..., N, m]``.
        prior_precision: Diagonal prior precision of the coefficients: a
            float, ``[m]`` or ``[..., m]``, entries ``>= 0``; 0 means a flat
            prior in that direction. Default (with ``prior_covariance`` also
            ``None``): flat in every direction.
        prior_covariance: Full prior covariance ``[m, m]`` or ``[..., m, m]``,
            positive definite. Mutually exclusive with ``prior_precision``.
        mask: Optional boolean mask (``True`` = observed), broadcastable to
            ``[..., N]``. Masked samples drop out exactly (as if absent), and
            non-finite values there are ignored.

    Returns:
        :class:`LinearNuisanceMarginal` with the per-row NLL, the posterior
        mean and covariance of the coefficients, the fitted nuisance and the
        conditional residual.

    Raises:
        ValueError: On shape errors, a non-positive-definite ``prior_covariance``
            or negative ``prior_precision``, or a row whose posterior precision
            :math:`A` is not numerically positive definite: some flat direction
            is not constrained by that row's unmasked data (an empty row with a
            flat prior, a basis column that is zero on every unmasked sample,
            collinear columns), or non-finite precision/basis on unmasked
            samples. Non-finite ``y_pred``/``target`` on unmasked samples
            propagate to ``NaN`` instead.
    """
    if prior_precision is not None and prior_covariance is not None:
        raise ValueError("pass at most one of prior_precision and prior_covariance")
    if basis.dim() < 2:
        raise ValueError(f"basis must be [N, m] or [..., N, m], got shape {tuple(basis.shape)}")
    n, m = basis.shape[-2], basis.shape[-1]
    if m < 1:
        raise ValueError("basis must have at least one column")
    dtype = torch.promote_types(torch.promote_types(y_pred.dtype, target.dtype), basis.dtype)
    if not dtype.is_floating_point:
        raise TypeError("y_pred, target and basis must be floating-point tensors")
    device = y_pred.device

    obs_shape = torch.broadcast_shapes(y_pred.shape, target.shape, precision.shape)
    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError("mask must have dtype torch.bool")
        obs_shape = torch.broadcast_shapes(obs_shape, mask.shape)
    if len(obs_shape) < 1 or obs_shape[-1] != n:
        raise ValueError(
            f"observations have shape {tuple(obs_shape)}, expected [..., {n}] to match basis"
        )
    shared_basis = basis.dim() == 2
    lead = (
        obs_shape[:-1] if shared_basis else torch.broadcast_shapes(obs_shape[:-1], basis.shape[:-2])
    )
    prior_lead: tuple[int, ...] = ()
    if isinstance(prior_precision, Tensor) and prior_precision.dim() > 1:
        prior_lead = tuple(prior_precision.shape[:-1])
    if prior_covariance is not None:
        if prior_covariance.dim() < 2 or prior_covariance.shape[-2:] != (m, m):
            raise ValueError(
                f"prior_covariance must be [..., {m}, {m}], got {tuple(prior_covariance.shape)}"
            )
        prior_lead = tuple(prior_covariance.shape[:-2])
    lead = torch.broadcast_shapes(lead, prior_lead)
    full = (*lead, n)
    rows = math.prod(lead)

    y_hat = torch.broadcast_to(y_pred.to(dtype), full)
    y_obs = torch.broadcast_to(target.to(device=device, dtype=dtype), full)
    lam = torch.broadcast_to(precision.to(device=device, dtype=dtype), full).reshape(rows, n)
    r = (y_obs - y_hat).reshape(rows, n)
    valid: Optional[Tensor] = None
    if mask is not None:
        valid = torch.broadcast_to(mask.to(device), full).reshape(rows, n)
        zero = torch.zeros((), dtype=dtype, device=device)
        r = torch.where(valid, r, zero)
        lam = torch.where(valid, lam, zero)
        log_lam = torch.where(valid, torch.log(torch.where(valid, lam, torch.ones_like(lam))), zero)
        n_obs = valid.sum(dim=-1).to(dtype)
    else:
        log_lam = torch.log(lam)
        n_obs = torch.full((rows,), float(n), dtype=dtype, device=device)

    u = basis.to(device=device, dtype=dtype)
    lam_r = lam * r
    if shared_basis and not bool(torch.isfinite(u).all()):
        raise ValueError("a shared basis must be finite (pass a per-row basis to mask entries)")
    if shared_basis and 2 * rows > m + 1:
        # G = U^T diag(lam) U for every row via the packed products U_j U_k:
        # N m (m + 1) / 2 shared elements instead of a [rows, N, m] intermediate
        # (the per-row path below is used when that one is not larger).
        j, k, unpack = _sym_index(m, device)
        gram = (lam @ (u[:, j] * u[:, k]))[:, unpack].reshape(rows, m, m)
        b = lam_r @ u
        u_rows = None
    else:
        u_rows = torch.broadcast_to(u, (*lead, n, m)).reshape(rows, n, m)
        if valid is not None:
            u_rows = torch.where(
                valid.unsqueeze(-1), u_rows, torch.zeros((), dtype=dtype, device=device)
            )
        gram = (u_rows * lam.unsqueeze(-1)).mT @ u_rows
        b = (u_rows.mT @ lam_r.unsqueeze(-1)).squeeze(-1)

    # Prior in coordinates with diagonal precision p: a = T c.
    transform: Optional[Tensor] = None
    if prior_covariance is not None:
        cov = torch.broadcast_to(prior_covariance.to(device=device, dtype=dtype), (*lead, m, m))
        chol_s, info = torch.linalg.cholesky_ex(cov.reshape(rows, m, m))
        if bool((info != 0).any()):
            raise ValueError("prior_covariance must be symmetric positive definite")
        transform = chol_s
        gram = chol_s.mT @ gram @ chol_s
        b = (chol_s.mT @ b.unsqueeze(-1)).squeeze(-1)
        p = torch.ones((rows, m), dtype=dtype, device=device)
        log_det_prior = torch.zeros((rows,), dtype=dtype, device=device)
    else:
        raw = 0.0 if prior_precision is None else prior_precision
        p_t = torch.as_tensor(raw, dtype=dtype, device=device)
        if not bool(torch.all(torch.isfinite(p_t) & (p_t >= 0))):
            raise ValueError("prior_precision must be finite and >= 0")
        p = torch.broadcast_to(p_t, (*lead, m)).reshape(rows, m)
        positive = p > 0
        log_det_prior = torch.where(
            positive, torch.log(torch.where(positive, p, torch.ones_like(p))), 0.0
        ).sum(-1)

    # A = diag(p) + G, Jacobi-scaled Cholesky. A pivot^2 below m * eps of the
    # unit-diagonal matrix means a direction is fixed only to round-off.
    a_mat = gram + torch.diag_embed(p).to(device=device, dtype=dtype)
    diag = torch.diagonal(a_mat, dim1=-2, dim2=-1)
    ok_diag = torch.isfinite(diag) & (diag > 0)
    scale = torch.where(ok_diag, diag, torch.ones_like(diag)).rsqrt()
    scaled = a_mat * scale.unsqueeze(-1) * scale.unsqueeze(-2)
    chol, info = torch.linalg.cholesky_ex(scaled)
    pivots = torch.diagonal(chol, dim1=-2, dim2=-1)
    tol = m * torch.finfo(dtype).eps
    good = (
        ok_diag.all(-1)
        & (info == 0)
        & torch.isfinite(pivots).all(-1)
        & (pivots.square() > tol).all(-1)
    )
    if not bool(good.all()):
        bad = int((~good).sum())
        raise ValueError(
            f"{bad} row(s) have a posterior precision U^T diag(precision) U + prior "
            "that is not numerically positive definite: a flat-prior direction is not "
            "constrained by the unmasked data (empty row, all-zero or collinear basis "
            "columns), or the unmasked precision/basis are not finite. Use a proper "
            "prior or drop those rows/columns."
        )
    rhs = (scale * b).unsqueeze(-1)
    coef_w = scale * torch.cholesky_solve(rhs, chol).squeeze(-1)
    log_det_a = 2.0 * torch.log(pivots).sum(-1) - 2.0 * torch.log(scale).sum(-1)

    coef = coef_w if transform is None else (transform @ coef_w.unsqueeze(-1)).squeeze(-1)
    if u_rows is None:
        fit = coef @ u.mT
    else:
        fit = (u_rows @ coef.unsqueeze(-1)).squeeze(-1)
    cond = r - fit
    chi2 = (lam * cond * cond).sum(-1) + (p * coef_w * coef_w).sum(-1)
    nll = 0.5 * (n_obs * _LOG_2PI - log_lam.sum(-1) + log_det_a - log_det_prior + chi2)

    inv_scaled = torch.cholesky_inverse(chol)
    post_cov = inv_scaled * scale.unsqueeze(-1) * scale.unsqueeze(-2)
    if transform is not None:
        post_cov = transform @ post_cov @ transform.mT

    coef_out = coef.reshape(*lead, m)
    if shared_basis:
        nuisance = coef_out @ u.mT
    else:
        nuisance = (torch.broadcast_to(u, (*lead, n, m)) @ coef_out.unsqueeze(-1)).squeeze(-1)
    residual = y_obs - y_hat - nuisance
    return LinearNuisanceMarginal(
        nll=nll.reshape(lead),
        coefficients=coef_out,
        coefficient_covariance=post_cov.reshape(*lead, m, m),
        nuisance=nuisance,
        residual=residual,
    )


@register_regression_loss("marginalized_nuisance_gaussian_nll")
class MarginalizedNuisanceGaussianNLLLoss(DistributionLoss):
    r"""Gaussian NLL of ``[..., N]`` rows with linear nuisance marginalised out.

    One NLL per row, :math:`-\log\mathcal N(y - \hat y;\,0,\,\Lambda^{-1} + USU^\top)`,
    computed by :func:`marginalize_linear_nuisance` (see there for the formula,
    the flat-prior convention and the fail-closed conditions). Reduction is over
    rows: ``"mean"`` averages the per-row joint NLL (not per sample); rows with
    no unmasked sample are dropped from ``"mean"``/``"sum"``. ``weights`` are
    per-row weights broadcastable to the leading shape ``[...]``.

    Args:
        reduction: ``"mean"`` | ``"sum"`` | ``"none"``.

    Use :func:`marginalize_linear_nuisance` directly for the fitted nuisance
    (posterior mean coefficients) and conditional residuals.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        precision: Tensor,
        basis: Tensor,
        prior_precision: float | Tensor | None = None,
        prior_covariance: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Return the reduced marginal NLL.

        Args:
            y_pred: Predictions ``[..., N]``.
            target: Observations ``[..., N]``.
            precision: Known inverse variances, broadcastable to ``[..., N]``.
            basis: Nuisance basis ``[N, m]`` or ``[..., N, m]``.
            prior_precision: Diagonal coefficient prior precision (0 = flat);
                default flat in every direction.
            prior_covariance: Full coefficient prior covariance (alternative).
            mask: Optional boolean sample mask broadcastable to ``[..., N]``.
            weights: Optional per-row weights broadcastable to ``[...]``.
        """
        result = marginalize_linear_nuisance(
            y_pred,
            target,
            precision,
            basis,
            prior_precision=prior_precision,
            prior_covariance=prior_covariance,
            mask=mask,
        )
        row_mask = None
        if mask is not None:
            full = (*result.nll.shape, basis.shape[-2])
            row_mask = torch.broadcast_to(mask.to(result.nll.device), full).any(-1)
        return self._reduce(result.nll, row_mask, weights)
