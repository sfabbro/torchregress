r"""Batched mixtures of multivariate Student-t distributions.

:class:`StudentTMixture` is a batch of ``K``-component mixtures over a
``P``-dimensional vector ``x``,

.. math::

    p(x) = \sum_{k=1}^K w_k\, t_{\nu_k}(x \mid \mu_k, \Sigma_k),

with full (covariance or precision Cholesky) or diagonal scale matrices and
per-component degrees of freedom ``nu`` (``math.inf`` is the Gaussian, so a
Gaussian mixture is the ``df=inf`` special case of the same code). Every
operation is vectorised over the batch *and* the components: ``K`` is folded
into the batch of the linear algebra, and nothing loops over components or
rows in Python.

Operations:

* :meth:`~StudentTMixture.log_prob` / :meth:`~StudentTMixture.component_log_prob`
  of full or partially observed vectors (per-row masks; the unobserved
  coordinates are marginalised exactly);
* :meth:`~StudentTMixture.component_observation_log_prob`: noisy linear
  observations ``z = A x + b + e``, ``e ~ N(0, diag(noise_var))``, with per-row
  validity masks and per-component ``A``, ``b`` (maps linearised at each
  component's location);
* :meth:`~StudentTMixture.component_censored_log_prob`: upper/lower limits on
  1-D linear observations with the exact Student-t log-CDF;
* :meth:`~StudentTMixture.mixture_log_prob`: combine per-component terms;
* :meth:`~StudentTMixture.condition`: condition on a noisy observation of one
  coordinate (returns a new mixture);
* :meth:`~StudentTMixture.marginal_cdf`, :meth:`~StudentTMixture.marginal_quantile`,
  :meth:`~StudentTMixture.sample`, :attr:`~StudentTMixture.mean`,
  :attr:`~StudentTMixture.covariance_matrix`, :attr:`~StudentTMixture.variance`.

References:
    Kotz & Nadarajah (2004), *Multivariate t Distributions and Their
    Applications*, Cambridge University Press (marginals, conditionals).
    Ding (2016), "On the conditional distribution of the multivariate t
    distribution", The American Statistician 70(3): 293-295.
    Lange, Little & Taylor (1989), "Robust statistical modeling using the t
    distribution", JASA 84(408): 881-896.
    Bishop (1994), "Mixture Density Networks", NCRG/94/004.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Distribution, constraints

from .families import _student_t_log_cdf_vec

__all__ = ["StudentTMixture"]

_LOG_2PI = math.log(2.0 * math.pi)

#: Above ``x = nu/2`` of this size :func:`_lgamma_ratio` uses the difference of two
#: Stirling series instead of ``lgamma(x + a) - lgamma(x)`` (which cancels to an
#: absolute error of ``eps * x log x``). The series truncation error at 50 is below
#: ``1e-15`` for any ``a >= 0``.
_LGAMMA_SERIES_X = 50.0

#: Elements per chunk of the ``[rows, P, Q, K]`` (or ``[rows, P, K]``) Student-t
#: CDF evaluations of :meth:`StudentTMixture.marginal_cdf` and
#: :meth:`StudentTMixture.marginal_quantile`. Bounds the float64 temporaries of the
#: incomplete-beta continued fraction (about ten per element).
_CDF_CHUNK = 1 << 20


# ---------------------------------------------------------------------------
# scalar Student-t pieces (float64 work, tensor-valued df, df = inf exact)
# ---------------------------------------------------------------------------


def _work_dtype(device: torch.device) -> torch.dtype:
    """float64 for the O(B K) Student-t transforms (float32 on MPS, which has no f64)."""
    return torch.float32 if device.type == "mps" else torch.float64


def _lgamma_ratio(x: Tensor, a: Tensor) -> Tensor:
    r"""``log Γ(x + a) − log Γ(x) − a log x`` for finite ``x > 0``, ``a ≥ 0``.

    Tends to 0 as ``x → ∞`` (it is the ``nu``-dependent part of the Student-t
    normaliser). Below ``_LGAMMA_SERIES_X`` it is the ``lgamma`` difference; above,
    the difference of the Stirling series of both terms,

    .. math::

        (x + a - \tfrac12)\log(1 + a/x) - a + \sum_m \frac{B_{2m}}{2m(2m-1)}
        \left[(x+a)^{1-2m} - x^{1-2m}\right],

    which has no cancellation worse than ``eps * a``.
    """
    small = x < _LGAMMA_SERIES_X
    xs = torch.where(small, x, torch.full_like(x, _LGAMMA_SERIES_X))
    direct = torch.lgamma(xs + a) - torch.lgamma(xs) - a * torch.log(xs)
    xl = torch.where(small, torch.full_like(x, _LGAMMA_SERIES_X), x)
    inv = 1.0 / xl
    inva = 1.0 / (xl + a)
    inv2, inva2 = inv * inv, inva * inva
    series = (
        (xl + a - 0.5) * torch.log1p(a * inv)
        - a
        + (inva - inv) / 12.0
        - (inva * inva2 - inv * inv2) / 360.0
        + (inva * inva2 * inva2 - inv * inv2 * inv2) / 1260.0
    )
    return torch.where(small, direct, series)


def _t_log_density(quad: Tensor, half_log_det: Tensor, df: Tensor, dim: Tensor) -> Tensor:
    r"""Log density of a ``dim``-variate ``t_df`` from its Mahalanobis term.

    .. math::

        \log t_\nu = \log\Gamma(\tfrac{\nu+d}{2}) - \log\Gamma(\tfrac\nu2)
        - \tfrac d2\log(\nu\pi) - \tfrac12\log|\Sigma|
        - \tfrac{\nu+d}{2}\log\!\left(1 + \frac{q}{\nu}\right),

    evaluated in float64 (it is ``O(rows x K)``) and returned in ``quad``'s dtype;
    ``df = inf`` gives the Gaussian exactly, and the infinite branch never
    receives a NaN gradient. ``dim = 0`` gives 0 (no observation).
    """
    out_dtype = torch.promote_types(quad.dtype, half_log_det.dtype)
    work = _work_dtype(quad.device)
    q = quad.to(work)
    nu = df.to(work)
    d = dim.to(work)
    inf = torch.isinf(nu)
    nu_s = torch.where(inf, torch.ones_like(nu), nu)
    half_d = 0.5 * d
    finite = _lgamma_ratio(0.5 * nu_s, half_d) - 0.5 * (nu_s + d) * torch.log1p(q / nu_s)
    gauss = -0.5 * q
    core = torch.where(inf, gauss, finite) - half_d * _LOG_2PI
    return core.to(out_dtype) - half_log_det.to(out_dtype)


def _t_log_cdf(z: Tensor, df: Tensor) -> Tensor:
    """``log T_df(z)`` (standard Student-t log-CDF; ``df = inf`` is ``log Φ``).

    Exact incomplete-beta evaluation (:func:`torchregress.losses.families._student_t_log_cdf_vec`)
    in float64. ``z = ±inf`` gives ``0`` / ``-inf``.
    """
    work = _work_dtype(z.device)
    zw, nu = torch.broadcast_tensors(z.to(work), df.to(work))
    inf_df = torch.isinf(nu)
    pos_inf, neg_inf = zw == math.inf, zw == -math.inf
    z_safe = torch.where(pos_inf | neg_inf, torch.zeros_like(zw), zw)
    if bool(inf_df.all()):
        out = torch.special.log_ndtr(z_safe)
    else:
        nu_s = torch.where(inf_df, torch.ones_like(nu), nu)
        out = _student_t_log_cdf_vec(z_safe, nu_s)
        if bool(inf_df.any()):
            out = torch.where(inf_df, torch.special.log_ndtr(z_safe), out)
    out = torch.where(pos_inf, torch.zeros_like(out), out)
    out = torch.where(neg_inf, torch.full_like(out, -math.inf), out)
    return out


def _t_tail_bound(alpha: Tensor, df: Tensor) -> Tensor:
    r"""A ``c ≥ 0`` with ``T_df(−c) ≤ alpha`` (a guaranteed quantile bracket).

    For ``c² ≥ ν``, bounding ``(1 − t)^{−1/2}`` in the incomplete-beta integral of
    the tail gives ``T_ν(−c) ≤ G_ν c^{−ν}`` with
    ``G_ν = ν^{ν/2} / (√2 · (ν/2) B(ν/2, 1/2))``; for the Gaussian
    ``Φ(−c) ≤ e^{−c²/2}/2``.
    """
    nu = df
    inf = torch.isinf(nu)
    nu_s = torch.where(inf, torch.ones_like(nu), nu)
    half = 0.5 * nu_s
    log_beta = torch.lgamma(half) + math.lgamma(0.5) - torch.lgamma(half + 0.5)
    log_g = half * torch.log(nu_s) - 0.5 * math.log(2.0) - torch.log(half) - log_beta
    log_alpha = torch.log(alpha)
    c_t = torch.maximum(torch.sqrt(nu_s), torch.exp((log_g - log_alpha) / nu_s))
    c_g = torch.sqrt((2.0 * (-math.log(2.0) - log_alpha)).clamp_min(0.0))
    return torch.where(inf, c_g, c_t)


def _validated_df(df: float | Tensor, like: Tensor) -> Tensor:
    if isinstance(df, Tensor):
        return df.to(device=like.device, dtype=like.dtype)
    value = float(df)
    if math.isnan(value) or value <= 0.0:
        raise ValueError(f"df must be > 0 (math.inf for the Gaussian), got {df}")
    return torch.tensor(value, device=like.device, dtype=like.dtype)


def _chunk_rows(elements_per_row: int) -> int:
    return max(1, _CDF_CHUNK // max(1, elements_per_row))


# ---------------------------------------------------------------------------
# the distribution
# ---------------------------------------------------------------------------


class StudentTMixture(Distribution):
    r"""Batch of mixtures of multivariate Student-t distributions.

    .. math::

        p(x) = \sum_{k=1}^K w_k\, t_{\nu_k}(x \mid \mu_k, \Sigma_k), \qquad
        t_\nu(x \mid \mu, \Sigma) = \frac{\Gamma(\frac{\nu+P}{2})}
        {\Gamma(\frac\nu2)(\nu\pi)^{P/2}|\Sigma|^{1/2}}
        \left(1 + \frac{(x-\mu)^\top\Sigma^{-1}(x-\mu)}{\nu}\right)^{-\frac{\nu+P}{2}}.

    ``Σ_k`` is the *scale* matrix (the covariance is ``ν/(ν−2) Σ_k`` for
    ``ν > 2``). ``df = math.inf`` is the Gaussian ``N(μ_k, Σ_k)`` exactly.

    Args:
        logits: Unnormalised log mixture weights ``[..., K]`` (``log_softmax`` is
            applied; normalised log weights pass through unchanged).
        loc: Component locations ``[..., K, P]``.
        scale_tril: Lower Cholesky factors ``L_k`` of the scale matrices,
            ``Σ_k = L_k L_kᵀ``, ``[..., K, P, P]`` (positive diagonal).
        precision_tril: Lower Cholesky factors ``T_k`` of the precision,
            ``Σ_k⁻¹ = T_k T_kᵀ``, ``[..., K, P, P]`` (the parameterisation of
            :class:`~torchregress.losses.MixtureDensityLoss` ``"full"`` heads; the
            log density is linear in ``T_k`` and needs no solve).
        scale_diag: Diagonal scales ``[..., K, P]`` (``Σ_k = diag(s_k²)``); keeps
            every operation ``O(P)`` in memory per component until a full scale
            matrix is needed (masked/observation terms, conditioning).
        df: Degrees of freedom ``ν > 0``: a float or a tensor broadcastable to
            ``[..., K]`` (per row and/or per component, learnable). ``math.inf``
            entries are Gaussian components.
        validate_args: ``True`` checks finiteness, positive scale diagonals and
            ``df > 0`` (one host sync). Shapes are always checked.

    Exactly one of ``scale_tril``, ``precision_tril`` and ``scale_diag`` is given.
    ``batch_shape`` is the broadcast of the leading shapes of all parameters;
    ``event_shape`` is ``(P,)``.

    Per-component terms (``component_*`` methods, ``[..., K]``) are added before
    the mixture is formed with :meth:`mixture_log_prob`: in a mixture the
    observations of one row are independent *given the component*, so their
    log-likelihoods add inside the ``logsumexp``, not outside.

    Example:
        >>> import torch
        >>> from torchregress.losses import StudentTMixture
        >>> B, K, P = 4, 3, 2
        >>> L = torch.diag_embed(torch.rand(B, K, P) + 0.5)
        >>> mix = StudentTMixture(torch.zeros(B, K), torch.randn(B, K, P), scale_tril=L, df=5.0)
        >>> mix.log_prob(torch.randn(B, P)).shape
        torch.Size([4])
    """

    arg_constraints: dict[str, constraints.Constraint] = {}
    support = constraints.real_vector
    has_rsample = False

    def __init__(
        self,
        logits: Tensor,
        loc: Tensor,
        *,
        scale_tril: Tensor | None = None,
        precision_tril: Tensor | None = None,
        scale_diag: Tensor | None = None,
        df: float | Tensor = math.inf,
        validate_args: bool | None = None,
    ) -> None:
        given = [s is not None for s in (scale_tril, precision_tril, scale_diag)]
        if sum(given) != 1:
            raise ValueError("Give exactly one of scale_tril, precision_tril or scale_diag")
        if loc.dim() < 2:
            raise ValueError("loc must be [..., K, P]")
        if logits.dim() < 1:
            raise ValueError("logits must be [..., K]")
        n_comp, n_dim = loc.shape[-2], loc.shape[-1]
        if logits.shape[-1] != n_comp:
            raise ValueError(f"logits has {logits.shape[-1]} components, loc has {n_comp}")
        df_t = _validated_df(df, loc)
        if df_t.dim() == 0:
            df_t = df_t.reshape(1)
        shapes = [logits.shape[:-1], loc.shape[:-2], df_t.shape[:-1]]
        if df_t.shape[-1] not in (1, n_comp):
            raise ValueError(f"df must broadcast to [..., {n_comp}], got {tuple(df_t.shape)}")
        if scale_diag is not None:
            if scale_diag.shape[-2:] != (n_comp, n_dim):
                raise ValueError(f"scale_diag must be [..., {n_comp}, {n_dim}]")
            shapes.append(scale_diag.shape[:-2])
            self._form = "diag"
            factor = scale_diag
        else:
            factor = scale_tril if scale_tril is not None else precision_tril
            assert factor is not None
            if factor.shape[-3:] != (n_comp, n_dim, n_dim):
                raise ValueError(
                    f"{'scale_tril' if scale_tril is not None else 'precision_tril'} must be "
                    f"[..., {n_comp}, {n_dim}, {n_dim}]"
                )
            shapes.append(factor.shape[:-3])
            self._form = "scale_tril" if scale_tril is not None else "precision_tril"
        batch_shape = torch.broadcast_shapes(*shapes)
        self.log_weights = F.log_softmax(logits, dim=-1).expand(*batch_shape, n_comp)
        self.loc = loc.expand(*batch_shape, n_comp, n_dim)
        self.df = df_t.expand(*batch_shape, n_comp)
        tail = (n_comp, n_dim) if self._form == "diag" else (n_comp, n_dim, n_dim)
        self._factor = factor.expand(*batch_shape, *tail)
        self.n_components = n_comp
        if validate_args:
            self._check_values()
        super().__init__(batch_shape, torch.Size((n_dim,)), validate_args=False)

    def _check_values(self) -> None:
        diag = self._factor if self._form == "diag" else self._factor.diagonal(dim1=-2, dim2=-1)
        checks = torch.stack(
            [
                ~torch.isfinite(self.loc).all(),
                ~torch.isfinite(self._factor).all(),
                (diag <= 0).any(),
                torch.isnan(self.df).any() | (self.df <= 0).any(),
                torch.isnan(self.log_weights).any(),
            ]
        ).tolist()
        messages = (
            "loc must be finite",
            "scale parameters must be finite",
            "scale factors must have a positive diagonal",
            "df must be > 0",
            "logits must not be NaN",
        )
        for bad, message in zip(checks, messages, strict=True):
            if bad:
                raise ValueError(message)

    def __repr__(self) -> str:
        return (
            f"StudentTMixture(batch_shape={tuple(self.batch_shape)}, "
            f"n_components={self.n_components}, event_shape={tuple(self.event_shape)}, "
            f"form={self._form!r})"
        )

    # ------------------------------------------------------------------
    # scale-matrix representations
    # ------------------------------------------------------------------

    @property
    def scale_tril(self) -> Tensor:
        """Lower Cholesky factors ``L_k`` of the scale matrices ``[..., K, P, P]``.

        For a precision factor ``T`` this is ``Rᵀ`` from the QR decomposition of
        ``T⁻¹`` (``Σ = T⁻ᵀT⁻¹ = RᵀR``), which avoids forming ``Σ`` (no squared
        condition number).
        """
        if self._form == "scale_tril":
            return self._factor
        if self._form == "diag":
            return torch.diag_embed(self._factor)
        inv = self._precision_inverse()
        r = torch.linalg.qr(inv, mode="r")[1]
        sign = torch.where(r.diagonal(dim1=-2, dim2=-1) < 0, -1.0, 1.0).to(r.dtype)
        return (r * sign.unsqueeze(-1)).mT

    def _precision_inverse(self) -> Tensor:
        """``T⁻¹`` (lower) for the precision form; ``Σ = (T⁻¹)ᵀ T⁻¹``."""
        t = self._factor
        eye = torch.eye(t.shape[-1], dtype=t.dtype, device=t.device).expand_as(t)
        return torch.linalg.solve_triangular(t, eye, upper=False)

    @property
    def component_scale(self) -> Tensor:
        """Marginal Student-t scales ``sqrt(diag Σ_k)`` ``[..., K, P]``.

        The marginal of coordinate ``p`` in component ``k`` is
        ``t_{ν_k}(μ_kp, component_scale[..., k, p])``; its standard deviation is
        ``scale · sqrt(ν/(ν−2))`` for ``ν > 2``.
        """
        if self._form == "diag":
            return self._factor
        if self._form == "scale_tril":
            return self._factor.square().sum(-1).sqrt()
        return self._precision_inverse().square().sum(-2).sqrt()

    def _times_sqrt(self, a: Tensor) -> Tensor:
        """``A S_k`` for ``A [..., K|1, M, P]`` with ``Σ_k = S_k S_kᵀ`` → ``[..., K, M, P]``."""
        if self._form == "diag":
            return a * self._factor.unsqueeze(-2)
        if self._form == "scale_tril":
            return a @ self._factor
        # S = T^{-T}: A T^{-T} = (T^{-1} A^T)^T, one triangular solve.
        t = self._factor
        rhs = a.mT.expand(*torch.broadcast_shapes(a.mT.shape[:-2], t.shape[:-2]), *a.mT.shape[-2:])
        return torch.linalg.solve_triangular(t, rhs, upper=False).mT

    def _sqrt_matrix(self) -> Tensor:
        """A square root ``S_k`` of ``Σ_k`` (not necessarily triangular) ``[..., K, P, P]``."""
        if self._form == "diag":
            return torch.diag_embed(self._factor)
        if self._form == "scale_tril":
            return self._factor
        return self._precision_inverse().mT

    # ------------------------------------------------------------------
    # densities
    # ------------------------------------------------------------------

    def mixture_log_prob(self, component_log_prob: Tensor) -> Tensor:
        """``log Σ_k w_k exp(c_k)`` for per-component log terms ``c [..., K]`` → ``[...]``.

        Sum all per-component terms of a row (full or masked targets,
        observations, censored limits) before calling this. A row whose terms
        are all zero (nothing observed) gives exactly zero.
        """
        return torch.logsumexp(self.log_weights + component_log_prob, dim=-1)

    def log_prob(self, value: Tensor, mask: Tensor | None = None) -> Tensor:
        """Mixture log density ``[...]`` of ``value [..., P]`` (see :meth:`component_log_prob`)."""
        return self.mixture_log_prob(self.component_log_prob(value, mask))

    def component_log_prob(self, value: Tensor, mask: Tensor | None = None) -> Tensor:
        r"""Per-component log densities ``log t_{ν_k}(x | μ_k, Σ_k)`` ``[..., K]``.

        Args:
            value: ``[..., P]``, broadcastable to ``sample_shape + batch_shape + (P,)``.
            mask: Optional boolean ``[..., P]`` (``True`` = observed). The density
                is then the exact marginal of the observed coordinates, a
                ``t_ν(μ_o, Σ_oo)`` of dimension ``d = mask.sum(-1)`` (row-wise);
                unobserved entries may hold ``NaN``. Rows with nothing observed
                give 0.
        """
        if mask is not None:
            return self._masked_component_log_prob(value, mask)
        resid = value.unsqueeze(-2) - self.loc
        n_dim = self.event_shape[0]
        if self._form == "diag":
            u2 = (resid / self._factor).square().sum(-1)
            half_log_det = torch.log(self._factor).sum(-1)
        elif self._form == "scale_tril":
            lmat = self._factor
            rhs = resid.unsqueeze(-1)
            shape = torch.broadcast_shapes(lmat.shape[:-2], rhs.shape[:-2])
            u = torch.linalg.solve_triangular(
                lmat.expand(*shape, n_dim, n_dim), rhs.expand(*shape, n_dim, 1), upper=False
            )
            u2 = u.square().sum((-2, -1))
            half_log_det = torch.log(lmat.diagonal(dim1=-2, dim2=-1)).sum(-1)
        else:
            t = self._factor
            u2 = (resid.unsqueeze(-2) @ t).square().sum((-2, -1))
            half_log_det = -torch.log(t.diagonal(dim1=-2, dim2=-1)).sum(-1)
        dim = torch.full((), float(n_dim), dtype=u2.dtype, device=u2.device)
        return _t_log_density(u2, half_log_det, self.df, dim)

    def _masked_component_log_prob(self, value: Tensor, mask: Tensor) -> Tensor:
        if mask.dtype != torch.bool:
            raise TypeError("mask must have dtype torch.bool")
        mask = mask.to(value.device)
        value = torch.where(mask, value, torch.zeros_like(value))
        m = mask.unsqueeze(-2)  # [..., 1, P]
        resid = torch.where(m, value.unsqueeze(-2) - self.loc, torch.zeros((), dtype=value.dtype))
        dim = mask.sum(-1, keepdim=True).to(resid.dtype)  # [..., 1]
        if self._form == "diag":
            s = self._factor
            u2 = torch.where(m, resid / s, torch.zeros_like(resid)).square().sum(-1)
            half_log_det = torch.where(m, torch.log(s), torch.zeros_like(s)).sum(-1)
            return _t_log_density(u2, half_log_det, self.df, dim)
        # Observed marginal through a padded full-size system: unobserved
        # coordinates become independent unit variables with zero residual, which
        # add exactly nothing to the quadratic form or the log-determinant.
        mf = m.to(resid.dtype)
        rows = mf.unsqueeze(-1) * self._sqrt_matrix()
        cov = rows @ rows.mT + torch.diag_embed(1.0 - mf)
        return self._gaussian_form_terms(resid, cov, dim)

    def _gaussian_form_terms(self, resid: Tensor, cov: Tensor, dim: Tensor) -> Tensor:
        """Student-t log density from a residual and an SPD matrix (NaN if not SPD)."""
        chol, info = torch.linalg.cholesky_ex(cov)
        shape = torch.broadcast_shapes(chol.shape[:-1], resid.shape)
        u = torch.linalg.solve_triangular(
            chol.expand(*shape, shape[-1]), resid.expand(shape).unsqueeze(-1), upper=False
        )
        u2 = u.square().sum((-2, -1))
        half_log_det = torch.log(chol.diagonal(dim1=-2, dim2=-1)).sum(-1)
        out = _t_log_density(u2, half_log_det, self.df, dim)
        return torch.where(info == 0, out, torch.full_like(out, math.nan))

    def component_observation_log_prob(
        self,
        value: Tensor,
        design: Tensor,
        noise_var: Tensor,
        *,
        offset: Tensor | None = None,
        valid: Tensor | None = None,
    ) -> Tensor:
        r"""Per-component log-likelihood of noisy linear observations ``[..., K]``.

        The observation model is ``z = A_k x + b_k + e``, ``e ~ N(0, diag(n))``,
        with ``x`` drawn from component ``k``. Each component gives, in closed form,

        .. math::

            z \mid k \;\sim\; t_{\nu_k}\!\left(A_k\mu_k + b_k,\;
            A_k \Sigma_k A_k^\top + \mathrm{diag}(n)\right)

        over the valid entries of ``z`` (dimension ``d = valid.sum(-1)``).

        This is exact for ``ν = ∞``. For finite ``ν`` it treats the noise as
        sharing the Student-t mixing variable of ``x`` ("Gaussian noise inside
        the t"): the exact density of ``A x + e`` with independent Gaussian ``e``
        is not a Student-t. The approximation is exact when the noise is
        negligible against ``A Σ Aᵀ``; when the noise dominates it keeps
        Student-t tails where the exact density is closer to Gaussian.

        Args:
            value: Observations ``z`` ``[..., D]``; invalid entries may be ``NaN``.
            design: ``A`` ``[..., K, D, P]`` per component (for maps linearised
                at each component's location) or ``[..., 1, D, P]`` shared.
            noise_var: Noise variances ``[..., D]``, > 0 where valid. They are
                not floored: a zero variance on a valid entry gives ``NaN`` (fail
                closed) unless ``D <= P`` and ``A Σ Aᵀ`` is non-singular.
            offset: Optional ``b`` ``[..., K, D]`` or ``[..., 1, D]``. For a map
                ``f`` linearised at ``μ_k`` with Jacobian ``J_k``, pass
                ``design=J_k`` and ``offset=f(μ_k) − J_k μ_k``.
            valid: Optional boolean ``[..., D]`` (``True`` = use the entry). Rows
                with no valid entry give exactly 0.

        Cost: one ``D x D`` Cholesky per row and component when ``D <= P``,
        otherwise the information form with one ``P x P`` Cholesky
        (``(I + Wᵀ W)``, ``W = n^{-1/2} A S_k``), whose quadratic form is a sum
        of squares (no subtraction of large terms).
        """
        n_dim = self.event_shape[0]
        if design.dim() < 3 or design.shape[-1] != n_dim:
            raise ValueError(f"design must be [..., K or 1, D, {n_dim}]")
        if design.shape[-3] not in (1, self.n_components):
            raise ValueError(
                f"design must have a component axis of size 1 or {self.n_components} at dim -3"
            )
        n_obs = design.shape[-2]
        if value.shape[-1] != n_obs:
            raise ValueError(f"value has {value.shape[-1]} entries, design has {n_obs} rows")
        if valid is None:
            valid = torch.ones(value.shape, dtype=torch.bool, device=value.device)
        elif valid.dtype != torch.bool:
            raise TypeError("valid must have dtype torch.bool")
        valid = valid.to(value.device)
        mean = (design @ self.loc.unsqueeze(-1)).squeeze(-1)  # [..., K, D]
        if offset is not None:
            mean = mean + offset
        v = valid.unsqueeze(-2)  # [..., 1, D]
        z = torch.where(valid, value, torch.zeros_like(value)).unsqueeze(-2)
        resid = torch.where(v, z - mean, torch.zeros((), dtype=mean.dtype, device=mean.device))
        noise = noise_var.unsqueeze(-2)
        dim = valid.sum(-1, keepdim=True).to(resid.dtype)  # [..., 1]
        rows = self._times_sqrt(design)  # [..., K, D, P]
        if n_obs <= n_dim:
            vf = v.to(rows.dtype)
            rows = rows * vf.unsqueeze(-1)
            diag = torch.where(v, noise, torch.ones((), dtype=rows.dtype, device=rows.device))
            cov = rows @ rows.mT + torch.diag_embed(diag)
            return self._gaussian_form_terms(resid, cov, dim)
        # Information form: N^{-1/2} with zero precision on invalid entries.
        w = torch.where(v, torch.rsqrt(torch.where(v, noise, torch.ones_like(noise))), 0.0)
        u = resid * w
        wa = rows * w.unsqueeze(-1)
        eye = torch.eye(n_dim, dtype=wa.dtype, device=wa.device)
        cap = eye + wa.mT @ wa
        chol, info = torch.linalg.cholesky_ex(cap)
        rhs = (wa.mT @ u.unsqueeze(-1)).expand(*chol.shape[:-1], 1)
        xi = torch.cholesky_solve(rhs, chol)
        fit = (wa @ xi).squeeze(-1)
        u2 = (u - fit).square().sum(-1) + xi.square().sum((-2, -1))
        half_log_det = torch.log(chol.diagonal(dim1=-2, dim2=-1)).sum(-1) - torch.where(
            v, torch.log(w.clamp_min(torch.finfo(w.dtype).tiny)), 0.0
        ).sum(-1)
        out = _t_log_density(u2, half_log_det, self.df, dim)
        return torch.where(info == 0, out, torch.full_like(out, math.nan))

    def component_censored_log_prob(
        self,
        limit: Tensor,
        design: Tensor,
        noise_var: Tensor,
        *,
        offset: Tensor | None = None,
        upper: bool | Tensor = True,
        valid: Tensor | None = None,
    ) -> Tensor:
        r"""Per-component log-probabilities of censored linear observations ``[..., K]``.

        Each of the ``M`` terms is a 1-D observation ``y_m = a_{km}ᵀ x + b_{km} + e_m``,
        ``e_m ~ N(0, n_m)``, known only to satisfy ``y_m ≤ limit_m`` (``upper``)
        or ``y_m ≥ limit_m``. Under component ``k`` (same noise approximation as
        :meth:`component_observation_log_prob`),

        .. math::

            \log P(y_m \le \ell_m \mid k) = \log T_{\nu_k}\!\left(
            \frac{\ell_m - a_{km}^\top\mu_k - b_{km}}
            {\sqrt{a_{km}^\top\Sigma_k a_{km} + n_m}}\right),

        with the exact Student-t log-CDF (``log Φ`` for ``ν = ∞``), accurate deep
        in both tails. The ``M`` valid terms are summed (treated as independent
        given the component).

        Args:
            limit: Limits ``[..., M]``.
            design: ``[..., K, M, P]`` or ``[..., 1, M, P]``.
            noise_var: ``[..., M]``, ``≥ 0``.
            offset: Optional ``[..., K, M]`` or ``[..., 1, M]``.
            upper: ``True`` for upper limits, ``False`` for lower limits, or a
                boolean tensor ``[..., M]`` mixing both.
            valid: Optional boolean ``[..., M]``; invalid terms contribute 0.
        """
        n_dim = self.event_shape[0]
        if design.dim() < 3 or design.shape[-1] != n_dim:
            raise ValueError(f"design must be [..., K or 1, M, {n_dim}]")
        mean = (design @ self.loc.unsqueeze(-1)).squeeze(-1)  # [..., K, M]
        if offset is not None:
            mean = mean + offset
        noise = noise_var.unsqueeze(-2)
        var = self._times_sqrt(design).square().sum(-1) + noise
        lim = limit.unsqueeze(-2)
        if valid is not None:
            if valid.dtype != torch.bool:
                raise TypeError("valid must have dtype torch.bool")
            v = valid.to(limit.device).unsqueeze(-2)
            lim = torch.where(v, lim, mean.detach())
            var = torch.where(v, var, torch.ones_like(var))
        z = (lim - mean) / torch.sqrt(var)
        if isinstance(upper, Tensor):
            z = torch.where(upper.to(z.device).unsqueeze(-2), z, -z)
        elif not upper:
            z = -z
        terms = _t_log_cdf(z, self.df.unsqueeze(-1)).to(mean.dtype)
        if valid is not None:
            terms = torch.where(v, terms, torch.zeros_like(terms))
        return terms.sum(-1)

    # ------------------------------------------------------------------
    # conditioning
    # ------------------------------------------------------------------

    def condition(self, index: int, value: Tensor, noise_var: Tensor) -> StudentTMixture:
        r"""Condition on a noisy observation ``y = x_j + e``, ``e ~ N(0, n)``.

        Per component (noise inside the t, as in
        :meth:`component_observation_log_prob`), with ``s = Σ_jj + n``,
        ``c = Σ_{:,j}`` and ``q = (y − μ_j)²/s``:

        .. math::

            x \mid y, k \sim t_{\nu+1}\!\left(\mu + c\,\frac{y-\mu_j}{s},\;
            \frac{\nu + q}{\nu + 1}\left(\Sigma - \frac{c c^\top}{s}\right)\right),
            \qquad w_k' \propto w_k\, t_{\nu_k}(y \mid \mu_{kj}, \sqrt{s_k}).

        The conditional Cholesky factor is ``L G`` with ``G`` the closed-form
        Cholesky factor of ``I − v vᵀ``, ``v = L_{j,:}/√s``, built from tail sums
        ``t_i = (n + Σ_{m>i} L_{jm}²)/s`` without cancellation (so a nearly exact
        observation keeps relative accuracy). ``n = 0`` gives a degenerate
        (zero-scale) coordinate ``j``.

        Args:
            index: Observed coordinate ``j``.
            value: Observations ``[...]`` (broadcast to ``batch_shape``).
            noise_var: Noise variances ``[...]``. Rows where ``value`` or
                ``noise_var`` is not finite, or ``noise_var < 0``, are returned
                unchanged (same weights, locations, scales and ``df``).

        Returns:
            A new :class:`StudentTMixture` in ``scale_tril`` form with ``df + 1``
            on conditioned rows.
        """
        n_dim = self.event_shape[0]
        j = int(index)
        if not -n_dim <= j < n_dim:
            raise IndexError(f"index {index} out of range for {n_dim} coordinates")
        j %= n_dim
        lmat = self.scale_tril
        dtype = lmat.dtype
        value = torch.as_tensor(value, dtype=dtype, device=lmat.device).expand(self.batch_shape)
        noise_var = torch.as_tensor(noise_var, dtype=dtype, device=lmat.device).expand(
            self.batch_shape
        )
        ok = torch.isfinite(value) & torch.isfinite(noise_var) & (noise_var >= 0)
        z = torch.where(ok, value, torch.zeros_like(value)).unsqueeze(-1)  # [..., 1]
        n = torch.where(ok, noise_var, torch.ones_like(noise_var)).unsqueeze(-1)
        row = lmat[..., j, :]  # [..., K, P] = l (support 0..j)
        cov_col = (lmat @ row.unsqueeze(-1)).squeeze(-1)  # Σ[:, j]
        sq = row.square()
        # tail[i] = sum_{m > i} l_m^2 (exact reverse cumulative sum).
        tail = sq.flip(-1).cumsum(-1).flip(-1) - sq
        s = sq.sum(-1) + n  # Σ_jj + n
        t_i = (n.unsqueeze(-1) + tail) / s.unsqueeze(-1)  # [..., K, P], t_i
        t_prev = torch.cat([torch.ones_like(t_i[..., :1]), t_i[..., :-1]], dim=-1)
        vv = row / torch.sqrt(s).unsqueeze(-1)
        # G_ii = sqrt(t_i / t_{i-1}); G_ik = -v_i v_k / sqrt(t_{k-1} t_k) for i > k.
        safe_prev = torch.where(t_prev > 0, t_prev, torch.ones_like(t_prev))
        g_diag = torch.where(vv == 0, torch.ones_like(vv), torch.sqrt(t_i / safe_prev))
        denom = t_prev * t_i
        safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
        col_scale = torch.where(denom > 0, torch.rsqrt(safe_denom), torch.zeros_like(denom))
        g_low = -(vv.unsqueeze(-1) * (vv * col_scale).unsqueeze(-2)).tril(-1)
        g = g_low + torch.diag_embed(g_diag)
        resid = z - self.loc[..., j]  # [..., K]
        q = resid.square() / s
        nu = self.df
        inf = torch.isinf(nu)
        nu_s = torch.where(inf, torch.ones_like(nu), nu)
        factor = torch.where(inf, torch.ones_like(nu), (nu_s + q) / (nu_s + 1.0))
        new_l = (lmat @ g) * torch.sqrt(factor)[..., None, None]
        new_loc = self.loc + cov_col * (resid / s).unsqueeze(-1)
        one = torch.ones((), dtype=dtype, device=lmat.device)
        pred = _t_log_density(q, 0.5 * torch.log(s), nu, one)
        new_logw = torch.log_softmax(self.log_weights + pred, dim=-1)
        okk = ok.unsqueeze(-1)
        return StudentTMixture(
            torch.where(okk, new_logw, self.log_weights),
            torch.where(okk.unsqueeze(-1), new_loc, self.loc),
            scale_tril=torch.where(okk[..., None, None], new_l, lmat),
            df=torch.where(okk, nu + 1.0, nu),
        )

    # ------------------------------------------------------------------
    # marginal CDF and quantiles
    # ------------------------------------------------------------------

    def _marginal_params(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """``(log w [..., 1, K], loc [..., P, K], scale [..., P, K], df [..., 1, K])``."""
        return (
            self.log_weights.unsqueeze(-2),
            self.loc.mT,
            self.component_scale.mT,
            self.df.unsqueeze(-2),
        )

    @staticmethod
    def _log_cdf_and_log_pdf(
        x: Tensor, logw: Tensor, loc: Tensor, scale: Tensor, df: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Mixture marginal ``log F`` and ``log f`` at ``x [..., 1]``; params ``[..., K]``."""
        z = (x - loc) / scale
        zero = scale == 0
        if bool(zero.any()):  # point-mass (exactly conditioned) components
            step = torch.where(x >= loc, math.inf, -math.inf).to(z.dtype)
            z = torch.where(zero, step, z)
        log_cdf = torch.logsumexp(logw + _t_log_cdf(z, df).to(z.dtype), dim=-1)
        one = torch.ones((), dtype=z.dtype, device=z.device)
        safe = torch.where(zero, torch.ones_like(scale), scale)
        zf = torch.where(zero, torch.zeros_like(z), z)
        comp = _t_log_density(zf.square(), torch.log(safe), df, one)
        comp = torch.where(zero, torch.full_like(comp, -math.inf), comp)
        log_pdf = torch.logsumexp(logw + comp, dim=-1)
        return log_cdf, log_pdf

    def marginal_cdf(self, value: Tensor) -> Tensor:
        r"""Per-coordinate marginal CDF ``F_p(x_p) = Σ_k w_k T_{ν_k}((x_p − μ_kp)/σ_kp)``.

        Args:
            value: ``[..., P]``, broadcastable to ``sample_shape + batch_shape + (P,)``
                (e.g. a ``[G, 1, P]`` grid against a ``[B]`` batch).

        Returns:
            ``[..., P]`` probabilities, evaluated as a ``logsumexp`` of exact
            log-CDFs in float64 (no clipping) and chunked so the temporaries stay
            bounded for large grids.
        """
        logw, loc, scale, df = self._marginal_params()
        shape = torch.broadcast_shapes(value.shape, (*self.batch_shape, self.event_shape[0]))
        x = value.to(loc.dtype).expand(shape)
        n_batch = len(self.batch_shape)
        lead = len(shape) - n_batch - 1
        flat = x.reshape(-1, *shape[lead:]) if lead else x.unsqueeze(0)
        per_row = math.prod(shape[lead:]) * self.n_components
        step = _chunk_rows(per_row)
        out = [
            self._log_cdf_and_log_pdf(chunk.unsqueeze(-1), logw, loc, scale, df)[0].exp()
            for chunk in flat.split(step, dim=0)
        ]
        return torch.cat(out, dim=0).reshape(shape)

    def marginal_quantile(
        self,
        q: float | Sequence[float] | Tensor,
        *,
        tol: float = 1e-12,
        max_iter: int = 100,
    ) -> Tensor:
        r"""Per-coordinate marginal quantiles ``F_p^{-1}(q)``.

        Solved per row, coordinate and level by safeguarded Newton iteration on the
        mixture CDF ("rtsafe"): a guaranteed bracket from Student-t tail bounds of
        every component (``F(lo) ≤ q ≤ F(hi)``, valid for any ``ν > 0``), Newton
        steps with the exact mixture density, and bisection whenever a step leaves
        the bracket or does not halve the previous one, so it converges for
        multimodal mixtures and heavy tails alike. Everything is float64 and
        chunked over rows.

        Args:
            q: A level or a 1-D sequence of levels in ``(0, 1)``.
            tol: Stop when every step is below ``tol · (|x| + σ̄)``
                (``σ̄`` the weight-averaged component scale).
            max_iter: Iteration cap (bisection alone needs ~60 halvings in
                float64; Newton converges in a handful).

        Returns:
            ``batch_shape + (P,)`` for a scalar level, ``batch_shape + (P, Q)``
            for ``Q`` levels.
        """
        logw, loc, scale, df = self._marginal_params()
        work = _work_dtype(loc.device)
        levels = torch.as_tensor(q, dtype=work, device=loc.device)
        scalar = levels.dim() == 0
        levels = levels.reshape(-1)
        if not bool(((levels > 0) & (levels < 1)).all()):
            raise ValueError("quantile levels must be in (0, 1)")
        n_dim, n_comp = self.event_shape[0], self.n_components
        batch = self.batch_shape
        params = [
            p.to(work).expand(*batch, *p.shape[-2:]).reshape(-1, *p.shape[-2:])
            for p in (logw, loc, scale, df)
        ]
        n_rows = params[1].shape[0]
        step = _chunk_rows(n_dim * levels.numel() * n_comp)
        outs = []
        for start in range(0, n_rows, step):
            sl = slice(start, start + step)
            outs.append(
                StudentTMixture._solve_quantiles(levels, *(p[sl] for p in params), tol, max_iter)
            )
        out = torch.cat(outs, dim=0) if outs else params[1].new_zeros(0, n_dim, levels.numel())
        out = out.reshape(*batch, n_dim, levels.numel()).to(self.loc.dtype)
        return out.squeeze(-1) if scalar else out

    @staticmethod
    def _solve_quantiles(
        levels: Tensor,
        logw: Tensor,
        loc: Tensor,
        scale: Tensor,
        df: Tensor,
        tol: float,
        max_iter: int,
    ) -> Tensor:
        """Quantiles ``[N, P, Q]`` for ``[N, 1|P, K]`` parameters (rtsafe, active set)."""
        n_rows, n_dim, n_comp = loc.shape
        n_q = levels.numel()
        shape = (n_rows, n_dim, n_q, n_comp)
        # One element per (row, coordinate, level); parameters [E, K].
        logw_e, loc_e, scale_e, df_e = (
            p.unsqueeze(-2).expand(shape).reshape(-1, n_comp) for p in (logw, loc, scale, df)
        )
        target = levels.view(1, 1, -1).expand(shape[:-1]).reshape(-1)
        qk = target.unsqueeze(-1)
        # Bracket: every component's CDF is <= q at lo and >= q at hi.
        lo = (loc_e - _t_tail_bound(qk, df_e) * scale_e).amin(-1)
        hi = (loc_e + _t_tail_bound(1.0 - qk, df_e) * scale_e).amax(-1)
        w = logw_e.exp()
        ref = (w * scale_e).sum(-1)
        # Start: weighted average of approximate component quantiles (normal
        # quantile plus the first Cornish-Fisher term of the t; exact at nu = inf).
        zq = torch.special.ndtri(qk)
        inv_nu = torch.where(torch.isinf(df_e), torch.zeros_like(df_e), 1.0 / df_e)
        tq = zq + (zq**3 + zq) * inv_nu / 4.0
        x = torch.minimum(torch.maximum((w * (loc_e + tq * scale_e)).sum(-1), lo), hi)
        dx_old = hi - lo
        active = torch.arange(x.numel(), device=x.device)
        for _ in range(max_iter):
            if active.numel() == 0:
                break
            xa, lo_a, hi_a, q_a = x[active], lo[active], hi[active], target[active]
            log_cdf, log_pdf = StudentTMixture._log_cdf_and_log_pdf(
                xa.unsqueeze(-1), logw_e[active], loc_e[active], scale_e[active], df_e[active]
            )
            g = log_cdf.exp() - q_a
            lo_a = torch.where(g < 0, xa, lo_a)
            hi_a = torch.where(g > 0, xa, hi_a)
            step = g / log_pdf.exp()
            newton = xa - step
            # Newton inside the closed bracket and at least halving the step before
            # last; otherwise bisect (guaranteed progress on flat or multimodal CDFs).
            use_newton = (
                torch.isfinite(newton)
                & (newton >= lo_a)
                & (newton <= hi_a)
                & (step.abs() * 2.0 <= dx_old[active].abs())
            )
            x_new = torch.where(use_newton, newton, 0.5 * (lo_a + hi_a))
            x_new = torch.where(g == 0, xa, x_new)
            dx = x_new - xa
            limit = tol * (x_new.abs() + ref[active])
            done = (
                (g == 0)
                | (use_newton & (step.abs() <= limit))
                | (dx.abs() <= limit)
                | ((hi_a - lo_a) <= limit)
            )
            x[active], lo[active], hi[active], dx_old[active] = x_new, lo_a, hi_a, dx
            active = active[~done]
        return x.reshape(shape[:-1])

    # ------------------------------------------------------------------
    # moments and sampling
    # ------------------------------------------------------------------

    def _variance_factor(self) -> Tensor:
        """``ν/(ν−2)`` per component (1 for Gaussians, ``inf`` for ``1 < ν ≤ 2``, NaN below)."""
        nu = self.df
        inf = torch.isinf(nu)
        nu_s = torch.where(inf, torch.full_like(nu, 3.0), nu)
        kappa = torch.where(nu_s > 2, nu_s / (nu_s - 2.0), torch.full_like(nu_s, math.inf))
        kappa = torch.where(nu_s <= 1, torch.full_like(nu_s, math.nan), kappa)
        return torch.where(inf, torch.ones_like(nu), kappa)

    @property
    def mean(self) -> Tensor:
        """Mixture mean ``Σ_k w_k μ_k`` ``[..., P]`` (NaN where some ``ν_k ≤ 1``)."""
        m = (self.log_weights.exp().unsqueeze(-1) * self.loc).sum(-2)
        undefined = (self.df <= 1).any(-1, keepdim=True)
        return torch.where(undefined, torch.full_like(m, math.nan), m)

    @property
    def covariance_matrix(self) -> Tensor:
        r"""Mixture covariance ``Σ_k w_k (κ_k Σ_k + (μ_k − m)(μ_k − m)ᵀ)``, ``κ = ν/(ν−2)``.

        ``[..., P, P]``; ``inf`` where some ``1 < ν_k ≤ 2``, NaN where some ``ν_k ≤ 1``.
        """
        w = self.log_weights.exp()
        kappa = self._variance_factor()
        sq = self._sqrt_matrix()
        scale_mat = sq @ sq.mT
        dev = self.loc - (w.unsqueeze(-1) * self.loc).sum(-2, keepdim=True)
        between = dev.unsqueeze(-1) * dev.unsqueeze(-2)
        weighted = w[..., None, None] * (kappa[..., None, None] * scale_mat + between)
        return weighted.sum(-3)

    @property
    def variance(self) -> Tensor:
        """Marginal mixture variances ``[..., P]`` (the diagonal of :attr:`covariance_matrix`)."""
        w = self.log_weights.exp().unsqueeze(-1)
        kappa = self._variance_factor().unsqueeze(-1)
        dev = self.loc - (w * self.loc).sum(-2, keepdim=True)
        return (w * (kappa * self.component_scale.square() + dev.square())).sum(-2)

    def sample(self, sample_shape: torch.Size | Sequence[int] = torch.Size()) -> Tensor:
        """Draw ``sample_shape + batch_shape + (P,)`` samples (component index, then
        ``x = μ_k + S_k ε / sqrt(g)``, ``g ~ χ²_ν/ν``); not reparameterised."""
        shape = torch.Size(sample_shape) + self.batch_shape
        n_dim = self.event_shape[0]
        with torch.no_grad():
            idx = torch.distributions.Categorical(logits=self.log_weights).sample(
                torch.Size(sample_shape)
            )
            eps = torch.randn(*shape, n_dim, dtype=self.loc.dtype, device=self.loc.device)
            loc = _gather_component(self.loc, idx, shape, (n_dim,))
            if self._form == "diag":
                draw = _gather_component(self._factor, idx, shape, (n_dim,)) * eps
            else:
                fac = _gather_component(self._factor, idx, shape, (n_dim, n_dim))
                if self._form == "scale_tril":
                    draw = (fac @ eps.unsqueeze(-1)).squeeze(-1)
                else:  # x = mu + T^{-T} eps has covariance (T T^T)^{-1}
                    draw = torch.linalg.solve_triangular(
                        fac.mT, eps.unsqueeze(-1), upper=True
                    ).squeeze(-1)
            nu = _gather_component(self.df, idx, shape, ())
            inf = torch.isinf(nu)
            nu_s = torch.where(inf, torch.ones_like(nu), nu)
            g = torch.distributions.Chi2(nu_s).sample() / nu_s
            g = torch.where(inf, torch.ones_like(g), g)
            return loc + draw * torch.rsqrt(g).unsqueeze(-1)


def _gather_component(
    param: Tensor, idx: Tensor, shape: torch.Size, tail: tuple[int, ...]
) -> Tensor:
    """Select component ``idx [shape]`` of ``param [batch, K, *tail]`` → ``[shape, *tail]``."""
    n_comp = param.shape[len(param.shape) - len(tail) - 1]
    expanded = param.expand(*shape, n_comp, *tail)
    index = idx.reshape(*shape, 1, *([1] * len(tail))).expand(*shape, 1, *tail)
    return expanded.gather(len(shape), index).squeeze(len(shape))
