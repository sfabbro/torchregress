r"""Gaussian likelihood with a per-sample known covariance plus a learned model covariance.

Measurements often come with a published, per-object noise covariance (for
example the coefficient covariance of a spectral fit), while the model that
predicts them is not perfect.  Scoring the residual with the published
covariance alone over-weights the objects and directions the model fits worst
(Gauss–Markov efficiency assumes a correct model); adding a fixed variance
floor erases the noise model wherever the published errors are smaller than
the floor.  The principled middle ground is to add a *learned* model
covariance to the known one and fit both terms by maximum likelihood:

.. math::

    r = y - \hat y \sim \mathcal N\!\left(0,\; C_{\text{known}} + C_{\text{model}}\right),
    \qquad
    C_{\text{model}} = \operatorname{diag}(s^2) + U U^\top .

:func:`known_covariance_gaussian_nll` evaluates the exact negative
log-likelihood per row,

.. math::

    -\log p(r) = \tfrac12\left[r^\top C^{-1} r + \log\lvert C\rvert + n\log 2\pi\right],
    \qquad C = C_{\text{known}} + C_{\text{model}},

over the ``n`` observed dimensions of the row, with a batched Cholesky
factorisation, and returns the per-row :math:`\chi^2 = r^\top C^{-1} r`, the
log-determinant, the number of observed dimensions and (optionally) the
whitened residual :math:`w = L^{-1} r` (:math:`C = LL^\top`), whose squared
norm is the :math:`\chi^2` used for goodness-of-fit diagnostics.
:class:`KnownCovarianceGaussianNLL` is the reducing loss and can own the learned
model covariance:

* ``scatter="fractional"``: per-feature variance :math:`(a \odot \hat y)^2 + b^2`
  with learned :math:`a, b \ge 0` (independent multiplicative and additive
  components add in variance; the expression is smooth in :math:`\hat y`, so
  second derivatives exist everywhere, unlike :math:`a\lvert\hat y\rvert + b`);
* ``scatter="constant"``: per-feature variance :math:`b^2`;
* ``rank=k``: a learned factor :math:`U = \hat y \odot V` (``factor="fractional"``,
  correlated multiplicative errors such as a calibration that scales with
  flux) or :math:`U = V` (``factor="constant"``), :math:`V \in \mathbb R^{D\times k}`.

Model terms predicted by a network are passed per row as ``model_scale`` and
``model_factor``; they add to the learned ones.

Missing dimensions are marginalised exactly: the Gaussian marginal of the
observed coordinates has the principal submatrix of :math:`C` as covariance.
Rows are padded to a common size by giving each masked coordinate unit
variance, zero covariance and zero residual, which after a permutation makes
the padded matrix block diagonal: the padded factor restricted to the observed
rows is the factor of the observed submatrix and the padding adds nothing to
the quadratic form or the log-determinant (as in
:class:`~torchregress.losses.MarginalizedMultiSourceGaussianNLLLoss`).

Numerics.  Published covariances are often badly scaled and nearly singular.
The total covariance is equilibrated first (Jacobi scaling to unit diagonal,
:math:`C = D^{1/2} R D^{1/2}`, which is optimal up to a factor :math:`n` among
diagonal scalings for the condition number; van der Sluis 1969) and :math:`R`
is factorised, so the factorisation error depends on the conditioning of the
correlation structure, not on the units.  A row whose :math:`R` is not
numerically positive definite raises, unless ``max_jitter > 0``: then the
diagonal of :math:`R` is raised (relative jitter, i.e. :math:`C + \epsilon\,
\operatorname{diag}(C)`) by powers of ten from ``jitter`` up to ``max_jitter``
for the failing rows only, and the amount used is returned per row.

References:
    van der Sluis, A. (1969). Condition numbers and equilibration of matrices.
    Numer. Math. 14, 14-23.
    Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*,
    2nd ed., SIAM, ch. 10 (Cholesky factorisation).
    Hogg, D. W. & Villar, S. (2021). Fitting very flexible models: linear
    regression with large numbers of parameters. PASP 133, 093001 (model
    scatter added to known noise).
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple, Optional

import torch
from torch import Tensor, nn

from .base import DistributionLoss
from .loss_registry import register_regression_loss

__all__ = [
    "KnownCovarianceGaussianNLL",
    "KnownCovarianceGaussianResult",
    "known_covariance_gaussian_nll",
]

_LOG_2PI = math.log(2.0 * math.pi)
_SCATTER_KINDS = ("none", "constant", "fractional")
_FACTOR_KINDS = ("constant", "fractional")


class KnownCovarianceGaussianResult(NamedTuple):
    """Per-row result of :func:`known_covariance_gaussian_nll` (rows ``[B]``).

    Attributes:
        nll: Negative log-likelihood ``[B]``, constants included; 0 for a row
            with no observed dimension.
        chi2: Quadratic form :math:`r^\\top C^{-1} r` over the observed
            dimensions ``[B]``.
        log_det: :math:`\\log\\lvert C_{oo}\\rvert` of the observed block ``[B]``.
        dof: Number of observed dimensions ``[B]`` (same dtype as ``nll``).
        whitened: Whitened residual :math:`L^{-1} r` ``[B, D]`` (Cholesky factor
            of the observed block in the original feature order, 0 at masked
            dimensions), or ``None`` unless ``return_whitened=True``.
        jitter: Relative jitter added to each row's equilibrated covariance
            ``[B]`` (0 unless a factorisation failed and ``max_jitter > 0``).
    """

    nll: Tensor
    chi2: Tensor
    log_det: Tensor
    dof: Tensor
    whitened: Optional[Tensor]
    jitter: Tensor


def _work_dtype(residual: Tensor, compute_dtype: Optional[torch.dtype]) -> torch.dtype:
    if compute_dtype is not None:
        if not compute_dtype.is_floating_point:
            raise TypeError(f"compute_dtype must be floating point, got {compute_dtype}")
        return compute_dtype
    if residual.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return residual.dtype


def _rows(name: str, value: Tensor, batch: int, tail: tuple[Optional[int], ...]) -> Tensor:
    """Validate ``[*tail]`` or ``[B or 1, *tail]`` and return it with a leading row dim.

    ``None`` in ``tail`` is a free size (e.g. the rank of a factor).
    """
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value)}")
    if value.dim() == len(tail):
        value = value.unsqueeze(0)
    ok = value.dim() == len(tail) + 1 and value.shape[0] in (1, batch)
    ok = ok and all(t is None or value.shape[i + 1] == t for i, t in enumerate(tail))
    if not ok:
        want = ", ".join("k" if t is None else str(t) for t in tail)
        raise ValueError(f"{name} has shape {tuple(value.shape)}; expected [{want}] or [B, {want}]")
    return value


def _evaluate(
    residual: Tensor,
    *,
    known_covariance: Optional[Tensor],
    known_variance: Optional[Tensor],
    model_variance: Optional[Tensor],
    model_factor: Optional[Tensor],
    mask: Optional[Tensor],
    return_whitened: bool,
    jitter: float,
    max_jitter: float,
    compute_dtype: Optional[torch.dtype],
) -> KnownCovarianceGaussianResult:
    if not isinstance(residual, Tensor) or not residual.is_floating_point():
        raise TypeError("residual must be a floating-point torch.Tensor")
    if residual.dim() != 2:
        raise ValueError(f"residual must be [B, D], got shape {tuple(residual.shape)}")
    if known_covariance is not None and known_variance is not None:
        raise ValueError("pass known_covariance or known_variance, not both")
    if not (math.isfinite(jitter) and math.isfinite(max_jitter)) or jitter < 0 or max_jitter < 0:
        raise ValueError("jitter and max_jitter must be finite and >= 0")
    batch, dims = residual.shape
    work = _work_dtype(residual, compute_dtype)
    out_dtype = residual.dtype if residual.dtype not in (torch.float16, torch.bfloat16) else work
    device = residual.device
    zero = torch.zeros((), dtype=work, device=device)

    if mask is None:
        observed = torch.ones(batch, dims, dtype=torch.bool, device=device)
    else:
        observed = torch.as_tensor(mask, device=device)
        if observed.dtype != torch.bool:
            raise TypeError("mask must be a boolean tensor (True = observed)")
        observed = torch.broadcast_to(observed, (batch, dims))
    maskf = observed.to(work)
    r = residual.to(work) if mask is None else torch.where(observed, residual.to(work), zero)

    dense = known_covariance is not None or model_factor is not None or return_whitened
    flags: list[tuple[Tensor, str]] = []

    # Diagonal contributions (known variance and model variance), zero where masked.
    diag = torch.zeros(batch, dims, dtype=work, device=device)
    if known_variance is not None:
        kv = _rows("known_variance", known_variance, batch, (dims,)).to(device=device, dtype=work)
        kv = torch.where(observed, kv, zero)
        flags.append(((kv < 0).any(), "known_variance must be >= 0 on observed dimensions"))
        flags.append((~torch.isfinite(kv).all(), "known_variance must be finite where observed"))
        diag = diag + kv
    if model_variance is not None:
        mv = model_variance.to(device=device, dtype=work)
        mv = torch.where(observed, torch.broadcast_to(mv, (batch, dims)), zero)
        flags.append((~torch.isfinite(mv).all(), "model scale must be finite where observed"))
        diag = diag + mv

    applied = torch.zeros(batch, dtype=work, device=device)
    if not dense:
        var = torch.where(observed, diag, torch.ones_like(diag))
        flags.append(((var <= 0).any(), "total variance must be > 0 on observed dimensions"))
        _raise_flags(flags)
        whitened = r * torch.rsqrt(var)
        chi2 = whitened.square().sum(-1)
        log_det = torch.log(var).sum(-1)
        w_out: Optional[Tensor] = None
    else:
        # Masked rows/columns are zeroed in every term and get unit variance below, so the
        # padded matrix is a permutation of blockdiag(C_oo, I): exact marginalisation.
        masked_any = mask is not None
        cov: Optional[Tensor] = None
        if known_covariance is not None:
            kc = _rows("known_covariance", known_covariance, batch, (dims, dims))
            # Symmetry tolerance of the input precision (a float32 matrix computed as
            # s_i rho_ij s_j can be asymmetric at its own rounding level).
            tol = math.sqrt(torch.finfo(kc.dtype).eps) if kc.is_floating_point() else 0.0
            kc = kc.to(device=device, dtype=work)
            if masked_any:
                pair = observed.unsqueeze(-1) & observed.unsqueeze(-2)
                kc = torch.where(pair, kc, zero)
            with torch.no_grad():
                flags.append(
                    (~torch.isfinite(kc).all(), "known_covariance must be finite where observed")
                )
                # Per-row relative test: max |K - K^T| <= tol * max |K|.
                asym = (kc - kc.mT).abs().amax(dim=(-2, -1)) > tol * kc.abs().amax(dim=(-2, -1))
                flags.append((asym.any(), "known_covariance must be symmetric"))
            # Cholesky reads one triangle; its backward returns a symmetric gradient.
            cov = kc
        if model_factor is not None:
            uf = _rows("model_factor", model_factor, batch, (dims, None)).to(
                device=device, dtype=work
            )
            if masked_any:
                uf = torch.where(observed.unsqueeze(-1), uf, zero)
            flags.append((~torch.isfinite(uf).all(), "model_factor must be finite where observed"))
            if cov is None:
                cov = uf @ uf.mT
            elif uf.shape[0] == cov.shape[0] or cov.shape[0] == 1:
                cov = torch.baddbmm(cov, uf, uf.mT)
            else:
                # Shared factor with a per-row known covariance: baddbmm does not
                # broadcast its two factors, so form the one outer product and add it.
                cov = cov + uf @ uf.mT
        diag_total = diag + (1.0 - maskf) if masked_any else diag
        if cov is None:
            cov = torch.diag_embed(diag_total).to(device=device, dtype=work)
        else:
            cov = torch.diagonal_scatter(
                cov.expand(batch, dims, dims),
                torch.diagonal(cov, dim1=-2, dim2=-1) + diag_total,
                0,
                -2,
                -1,
            )
        scale = torch.diagonal(cov, dim1=-2, dim2=-1)
        flags.append(
            (
                ~(torch.isfinite(scale).all() & (scale > 0).all()),
                "total covariance must have a positive finite diagonal on observed dimensions",
            )
        )
        good = torch.isfinite(scale) & (scale > 0)
        q = torch.rsqrt(torch.where(good, scale, torch.ones_like(scale)))
        # Jacobi equilibration: unit diagonal, so the factorisation error depends on
        # the correlation structure only, not on the units of the features.
        corr = cov * q.unsqueeze(-1) * q.unsqueeze(-2)
        if jitter > 0:
            corr_j = corr + torch.diag_embed(maskf * jitter).to(device=device, dtype=work)
            applied = applied + jitter * observed.any(-1).to(work)
        else:
            corr_j = corr
        factor, info = torch.linalg.cholesky_ex(corr_j)
        failed = info != 0
        flags.append((failed.any(), "__cholesky__"))
        status = torch.stack([flag for flag, _ in flags]).tolist() if flags else []
        for (_, message), bad in zip(flags[:-1], status[:-1], strict=True):
            if bad:
                raise ValueError(message)
        if status[-1]:
            if not max_jitter > jitter:
                raise ValueError(
                    "total covariance is not numerically positive definite for some rows; "
                    "pass max_jitter > 0 to regularise them"
                )
            amounts = _retry_jitter(corr, maskf, failed, jitter, max_jitter)
            applied = torch.where(failed, amounts, applied)
            corr_j = corr + torch.diag_embed(maskf * applied.unsqueeze(-1)).to(
                device=device, dtype=work
            )
            factor, info = torch.linalg.cholesky_ex(corr_j)
            if bool((info != 0).any()):
                raise ValueError("total covariance could not be factorised within max_jitter")
        whitened = torch.linalg.solve_triangular(
            factor, (q * r).unsqueeze(-1), upper=False
        ).squeeze(-1)
        chi2 = whitened.square().sum(-1)
        log_det = 2.0 * torch.log(torch.diagonal(factor, dim1=-2, dim2=-1)).sum(-1) + torch.log(
            torch.where(observed, scale, torch.ones_like(scale))
        ).sum(-1)
        w_out = whitened if return_whitened else None

    dof = maskf.sum(-1)
    nll = 0.5 * (chi2 + log_det + dof * _LOG_2PI)
    nll = torch.where(dof > 0, nll, zero)
    return KnownCovarianceGaussianResult(
        nll.to(out_dtype),
        chi2.to(out_dtype),
        log_det.to(out_dtype),
        dof.to(out_dtype),
        None if w_out is None else w_out.to(out_dtype),
        applied.detach().to(out_dtype),
    )


def _raise_flags(flags: list[tuple[Tensor, str]]) -> None:
    """Raise the first failing check; one host synchronisation for all of them."""
    if not flags:
        return
    status = torch.stack([flag for flag, _ in flags]).tolist()
    for (_, message), bad in zip(flags, status, strict=True):
        if bad:
            raise ValueError(message)


def _retry_jitter(
    corr: Tensor, maskf: Tensor, failed: Tensor, jitter: float, max_jitter: float
) -> Tensor:
    """Smallest power-of-ten relative jitter (up to ``max_jitter``) that factorises each row."""
    eps = torch.finfo(corr.dtype).eps
    amount = max(jitter, 0.0)
    amounts = torch.zeros(corr.shape[0], dtype=corr.dtype, device=corr.device)
    pending = failed.clone()
    with torch.no_grad():
        base = corr.detach()
        while bool(pending.any()):
            if amount >= max_jitter:
                raise ValueError("total covariance could not be factorised within max_jitter")
            amount = min(max_jitter, max(16.0 * eps, amount * 10.0))
            trial = base + torch.diag_embed(maskf * amount).to(device=base.device, dtype=base.dtype)
            _, info = torch.linalg.cholesky_ex(trial)
            ok = pending & (info == 0)
            amounts = torch.where(ok, torch.full_like(amounts, amount), amounts)
            pending = pending & ~ok
    return amounts


def known_covariance_gaussian_nll(
    residual: Tensor,
    known_covariance: Optional[Tensor] = None,
    *,
    known_variance: Optional[Tensor] = None,
    model_scale: Optional[Tensor] = None,
    model_factor: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
    return_whitened: bool = False,
    jitter: float = 0.0,
    max_jitter: float = 0.0,
    compute_dtype: Optional[torch.dtype] = None,
) -> KnownCovarianceGaussianResult:
    r"""Exact Gaussian NLL with known plus model covariance, per row.

    .. math::

        -\log p(r) = \tfrac12\left[r_o^\top C_{oo}^{-1} r_o + \log\lvert C_{oo}\rvert
        + n\log 2\pi\right],\qquad
        C = C_{\text{known}} + \operatorname{diag}(s^2) + UU^\top,

    where :math:`o` are the ``n`` observed dimensions of the row.

    Args:
        residual: ``[B, D]`` residuals :math:`r = y - \hat y`.
        known_covariance: Known covariance ``[D, D]`` or ``[B, D, D]``
            (``[1, D, D]`` broadcast), symmetric (to :math:`\sqrt{\epsilon}`
            relative; the symmetric part is used) and positive semidefinite on the
            observed block.  Mutually exclusive with ``known_variance``.
        known_variance: Diagonal known covariance ``[D]`` or ``[B, D]``, ``>= 0``.
        model_scale: Model standard deviation :math:`s`, ``[D]`` or ``[B, D]``;
            only :math:`s^2` enters.
        model_factor: Model covariance factor :math:`U`, ``[D, k]`` or
            ``[B, D, k]`` (``[1, D, k]`` broadcast).
        mask: Boolean ``[B, D]`` or ``[D]`` (``True`` = observed).  Masked
            dimensions are marginalised out exactly (principal submatrix); any
            value there, including ``NaN``, is ignored.  A row with nothing
            observed has ``nll = 0`` and ``dof = 0``.
        return_whitened: Also return :math:`w = L^{-1} r` ``[B, D]``.
        jitter: Relative jitter always added to the equilibrated covariance
            (default 0).
        max_jitter: Largest relative jitter tried, by powers of ten, for rows
            whose factorisation fails (default 0: such rows raise).
        compute_dtype: Dtype of the factorisation (e.g. ``torch.float64`` for
            float32 inputs; not supported on MPS).  Results are returned in the
            residual's dtype (float32 for half-precision inputs).

    Returns:
        :class:`KnownCovarianceGaussianResult` ``(nll, chi2, log_det, dof,
        whitened, jitter)``.

    Raises:
        ValueError: On shape errors, both ``known_covariance`` and
            ``known_variance``, negative or non-finite known variances, a
            non-finite or asymmetric known covariance, non-finite model terms, a
            non-positive total variance, or a row whose total covariance is not
            numerically positive definite (with ``max_jitter = 0``).  All checks
            share one host synchronisation.  Non-finite residuals on observed
            dimensions propagate to ``NaN``.

    The diagonal-only case (no dense known covariance, no factor, no whitened
    output) is evaluated elementwise without a factorisation.  For a diagonal
    known covariance plus a low-rank term in large ``D`` the same likelihood is
    available in :math:`O(Dk^2)` through
    :func:`~torchregress.losses.marginalize_linear_nuisance` with
    ``precision = 1 / (known_variance + s^2)``, ``basis = U`` and
    ``prior_precision = 1``.
    """
    with torch.autocast(device_type=residual.device.type, enabled=False):
        model_variance = None if model_scale is None else _square(model_scale, residual)
        return _evaluate(
            residual,
            known_covariance=known_covariance,
            known_variance=known_variance,
            model_variance=model_variance,
            model_factor=model_factor,
            mask=mask,
            return_whitened=return_whitened,
            jitter=float(jitter),
            max_jitter=float(max_jitter),
            compute_dtype=compute_dtype,
        )


def _square(scale: Tensor, residual: Tensor) -> Tensor:
    if not isinstance(scale, Tensor):
        raise TypeError("model_scale must be a torch.Tensor")
    batch, dims = residual.shape
    scale = _rows("model_scale", scale, batch, (dims,))
    return scale.square()


@register_regression_loss("known_covariance_gaussian_nll")
class KnownCovarianceGaussianNLL(DistributionLoss):
    r"""Gaussian NLL with a per-sample known covariance plus a learned model covariance.

    One NLL per row of :math:`r = y - \hat y \sim \mathcal N(0, C_{\text{known}} +
    C_{\text{model}})` (see :func:`known_covariance_gaussian_nll`), reduced over
    rows: ``"mean"`` averages the per-row joint NLL over rows with at least one
    observed dimension, ``weights`` are per-row ``[B]``.

    The model covariance is the sum of the terms passed to :meth:`forward`
    (``model_scale``, ``model_factor``: e.g. predicted by a network) and the
    learned terms configured here:

    ======================  ===================================================
    ``scatter``             learned per-feature variance
    ======================  ===================================================
    ``"none"`` (default)    none
    ``"constant"``          :math:`b^2`
    ``"fractional"``        :math:`(a\odot\hat y)^2 + b^2`
    ======================  ===================================================

    plus, for ``rank = k > 0``, :math:`UU^\top` with :math:`U = \hat y \odot V`
    (``factor="fractional"``) or :math:`U = V` (``factor="constant"``),
    :math:`V\in\mathbb R^{D\times k}`.  :math:`a = e^{\alpha}` and
    :math:`b = e^{\beta}` are learned on the log scale (``learn_floor=False``
    keeps :math:`b` fixed at ``init_floor``, which may be 0).  Add the loss's
    parameters to the optimiser.

    Args:
        n_features: ``D``; required when a learned term is configured.
        scatter: ``"none"``, ``"constant"`` or ``"fractional"``.
        rank: Rank ``k`` of the learned factor (0 = none).
        factor: ``"fractional"`` or ``"constant"`` learned factor.
        init_fraction: Initial :math:`a` (``scatter="fractional"``).
        init_floor: Initial :math:`b`, in target units.
        learn_floor: Learn :math:`b` (default) or keep it fixed.
        init_factor_scale: Standard deviation of the random initial :math:`V`
            entries (a zero factor is a stationary point); default
            ``init_fraction / sqrt(k)``, or ``init_floor / sqrt(k)`` for a constant
            factor with ``init_floor > 0``.
        jitter, max_jitter, compute_dtype: As in
            :func:`known_covariance_gaussian_nll`.
        reduction: ``"mean"`` | ``"sum"`` | ``"none"`` (per-row NLL ``[B]``).
    """

    def __init__(
        self,
        n_features: Optional[int] = None,
        *,
        scatter: str = "none",
        rank: int = 0,
        factor: str = "fractional",
        init_fraction: float = 1e-2,
        init_floor: float = 1e-3,
        learn_floor: bool = True,
        init_factor_scale: Optional[float] = None,
        jitter: float = 0.0,
        max_jitter: float = 0.0,
        compute_dtype: Optional[torch.dtype] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        if scatter not in _SCATTER_KINDS:
            raise ValueError(f"scatter must be one of {_SCATTER_KINDS}, got {scatter!r}")
        if factor not in _FACTOR_KINDS:
            raise ValueError(f"factor must be one of {_FACTOR_KINDS}, got {factor!r}")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise ValueError(f"rank must be a non-negative integer, got {rank!r}")
        learned = scatter != "none" or rank > 0
        if learned and (n_features is None or isinstance(n_features, bool) or int(n_features) < 1):
            raise ValueError("n_features (a positive integer) is required for learned terms")
        if jitter < 0 or max_jitter < 0 or not math.isfinite(jitter + max_jitter):
            raise ValueError("jitter and max_jitter must be finite and >= 0")
        self.n_features = None if n_features is None else int(n_features)
        self.scatter = scatter
        self.rank = rank
        self.factor = factor
        self.learn_floor = learn_floor
        self.jitter = float(jitter)
        self.max_jitter = float(max_jitter)
        self.compute_dtype = compute_dtype

        self.log_fraction: Optional[nn.Parameter] = None
        self.log_floor: Optional[nn.Parameter] = None
        self.factor_weight: Optional[nn.Parameter] = None
        floor_fixed: Optional[Tensor] = None
        if scatter != "none":
            d = int(self.n_features or 0)
            if scatter == "fractional":
                if not init_fraction > 0:
                    raise ValueError("init_fraction must be > 0 for scatter='fractional'")
                self.log_fraction = nn.Parameter(torch.full((d,), math.log(init_fraction)))
            if learn_floor:
                if not init_floor > 0:
                    raise ValueError("init_floor must be > 0 when learn_floor=True")
                self.log_floor = nn.Parameter(torch.full((d,), math.log(init_floor)))
            else:
                if init_floor < 0:
                    raise ValueError("init_floor must be >= 0")
                floor_fixed = torch.full((d,), float(init_floor))
        self.register_buffer("floor", floor_fixed)
        if rank > 0:
            d = int(self.n_features or 0)
            if init_factor_scale is None:
                base = init_floor if factor == "constant" and init_floor > 0 else init_fraction
                init_factor_scale = base / math.sqrt(rank)
            if not init_factor_scale > 0:
                raise ValueError("init_factor_scale must be > 0")
            self.factor_weight = nn.Parameter(torch.randn(d, rank) * init_factor_scale)

    def model_covariance_terms(self, y_pred: Tensor) -> tuple[Optional[Tensor], Optional[Tensor]]:
        """Learned model variance ``[B, D]`` and factor ``[B or 1, D, k]`` at ``y_pred``.

        Either entry is ``None`` when the corresponding term is not configured.
        """
        variance: Optional[Tensor] = None
        if self.scatter != "none":
            if y_pred.shape[-1] != self.n_features:
                raise ValueError(
                    f"y_pred has {y_pred.shape[-1]} features, the loss was built for "
                    f"{self.n_features}"
                )
            if self.log_floor is not None:
                floor_var = torch.exp(2.0 * self.log_floor).to(y_pred.dtype)
            else:
                floor = self.floor
                assert isinstance(floor, Tensor)
                floor_var = floor.to(device=y_pred.device, dtype=y_pred.dtype).square()
            variance = floor_var.expand_as(y_pred)
            if self.log_fraction is not None:
                frac = torch.exp(self.log_fraction).to(y_pred.dtype)
                variance = (frac * y_pred).square() + variance
        factor: Optional[Tensor] = None
        if self.factor_weight is not None:
            v = self.factor_weight.to(y_pred.dtype)
            if y_pred.shape[-1] != v.shape[0]:
                raise ValueError(
                    f"y_pred has {y_pred.shape[-1]} features, the loss was built for {v.shape[0]}"
                )
            factor = y_pred.unsqueeze(-1) * v if self.factor == "fractional" else v.unsqueeze(0)
        return variance, factor

    def evaluate(
        self,
        y_pred: Tensor,
        target: Tensor,
        known_covariance: Optional[Tensor] = None,
        known_variance: Optional[Tensor] = None,
        model_scale: Optional[Tensor] = None,
        model_factor: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        return_whitened: bool = False,
    ) -> KnownCovarianceGaussianResult:
        """Unreduced per-row result (NLL, chi2, log-det, dof, whitened, jitter)."""
        if not isinstance(y_pred, Tensor) or not isinstance(target, Tensor):
            raise TypeError("y_pred and target must be tensors")
        if y_pred.shape != target.shape or y_pred.dim() != 2:
            raise ValueError(
                f"y_pred {tuple(y_pred.shape)} and target {tuple(target.shape)} must both be [B, D]"
            )
        with torch.autocast(device_type=y_pred.device.type, enabled=False):
            learned_var, learned_factor = self.model_covariance_terms(y_pred)
            variance = learned_var
            if model_scale is not None:
                given = _square(model_scale, y_pred)
                variance = given if variance is None else variance + given
            factor = learned_factor
            if model_factor is not None:
                given_f = _rows(
                    "model_factor", model_factor, y_pred.shape[0], (y_pred.shape[1], None)
                )
                if factor is None:
                    factor = given_f
                else:
                    batch = max(factor.shape[0], given_f.shape[0])
                    factor = torch.cat(
                        [
                            factor.expand(batch, -1, -1),
                            given_f.to(factor.dtype).expand(batch, -1, -1),
                        ],
                        dim=-1,
                    )
            return _evaluate(
                target - y_pred,
                known_covariance=known_covariance,
                known_variance=known_variance,
                model_variance=variance,
                model_factor=factor,
                mask=mask,
                return_whitened=return_whitened,
                jitter=self.jitter,
                max_jitter=self.max_jitter,
                compute_dtype=self.compute_dtype,
            )

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        known_covariance: Optional[Tensor] = None,
        known_variance: Optional[Tensor] = None,
        model_scale: Optional[Tensor] = None,
        model_factor: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Reduced NLL over rows.

        Args:
            y_pred: Predictions ``[B, D]``.
            target: Observations ``[B, D]``.
            known_covariance: Known covariance ``[D, D]`` or ``[B, D, D]``.
            known_variance: Diagonal known covariance ``[D]`` or ``[B, D]``.
            model_scale: Extra model standard deviation ``[D]`` or ``[B, D]``.
            model_factor: Extra model factor ``[D, k]`` or ``[B, D, k]``.
            mask: Boolean ``[B, D]`` (``True`` = observed); masked dimensions
                are marginalised out.
            weights: Optional per-row weights ``[B]``.
        """
        result = self.evaluate(
            y_pred,
            target,
            known_covariance=known_covariance,
            known_variance=known_variance,
            model_scale=model_scale,
            model_factor=model_factor,
            mask=mask,
        )
        row_mask = None if mask is None else result.dof > 0
        return self._reduce(result.nll, row_mask, weights)
