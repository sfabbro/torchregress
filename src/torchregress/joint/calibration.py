r"""Post-hoc recalibration of joint predictive distributions.

Each recalibrator is fitted on held-out pairs :math:`(x_i, y_i)` through the
predictive of those rows, ``fit(predictive_cal, y_cal)``, and then wraps any
predictive of the same head, ``transform(predictive)``, into another
:class:`~torchregress.joint.predictive.JointPredictive`:

* :class:`ScaleRecalibrator`: one scale factor per target about the predictive
  mean.  The cheapest correction and, in forecasting competitions, the one that
  most often decided the ranking (a single sigma multiplier recovered almost all
  of the Ariel 2025 winner's margin).
* :class:`LatentRecalibrator`: latent recalibration of Dheur and Ben Taieb
  (NeurIPS 2025): a radial map of the standard-normal latent space that makes
  the norm of the latent residual follow its calibration distribution.  Keeps a
  density and a latent map; corrects the norm, not the direction.
* :class:`MarginalRecalibrator`: split conformal predictive systems (conformal
  calibrators, Vovk et al.) applied per target, with the dependence (copula) of
  the base predictive kept unchanged.  The only one of the three with a
  finite-sample calibration guarantee, and that guarantee is per target and
  marginal over :math:`x`.

None of these gives calibration conditional on an individual :math:`x`; they
correct average miscalibration over the distribution of the calibration rows,
under exchangeability of calibration and test rows.

References:

1. V. Dheur and S. Ben Taieb, "Multivariate latent recalibration for conditional
   normalizing flows", NeurIPS 2025 (arXiv 2505.16636).
2. V. Vovk, J. Shen, V. Manokhin, M. Xie, "Nonparametric predictive distributions
   based on conformal prediction", Machine Learning 108 (2019).
3. V. Vovk, I. Petej, P. Toccaceli, A. Gammerman, E. Ahlberg, L. Carlsson,
   "Conformal calibrators", COPA 2020.
4. J. A. Gregory and R. Delbourgo, "Piecewise rational quadratic interpolation to
   monotonic data", IMA J. Numer. Anal. 2 (1982).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal, Optional, Union

import torch
from scipy.stats import chi2
from torch import Tensor

from ..utils.distributions import normal_cdf
from ._common import fixed_moment_seed, seeded, target_index
from .predictive import AffineJoint, GaussianJoint, JointPredictive, SampleJoint

__all__ = [
    "LatentRecalibratedJoint",
    "LatentRecalibrator",
    "MarginalRecalibratedJoint",
    "MarginalRecalibrator",
    "ScaleRecalibrator",
    "latent_calibration_error",
]

_F64 = torch.float64
_MEAN_SEED = 0  # seed of Monte Carlo predictive means in ScaleRecalibrator
_LOG_TINY = -700.0  # log of the smallest latent norm we evaluate (exp(-700) > 0 in float64)


def _check_pair(predictive: JointPredictive, y: Any) -> Tensor:
    """Targets as a tensor on the predictive's dtype and device (arrays accepted)."""
    y = predictive.as_input(y)
    if y.dim() != 2 or y.shape != (predictive.batch_size, predictive.event_dim):
        raise ValueError(
            f"y must be [n, d] = [{predictive.batch_size}, {predictive.event_dim}], "
            f"got {tuple(y.shape)}"
        )
    return y


def _check_fit_mask(y: Tensor, mask: Optional[Tensor]) -> Tensor:
    if mask is None:
        return torch.ones(y.shape, dtype=torch.bool, device=y.device)
    mask = torch.as_tensor(mask)
    if mask.dtype != torch.bool or mask.shape != y.shape:
        raise ValueError(f"mask must be boolean {tuple(y.shape)}, got {mask.dtype} {mask.shape}")
    if not bool(mask.any(dim=0).all()):
        raise ValueError("every target needs at least one observed calibration value")
    return mask.to(y.device)


# ═══════════════════════════════════════════════════════════════════════════════
# per-target scale
# ═══════════════════════════════════════════════════════════════════════════════


class ScaleRecalibrator:
    r"""One scale factor per target about the predictive mean.

    The recalibrated predictive is the law of

    .. math::
        Y' = m(x) + s \odot \big(Y - m(x)\big), \qquad Y \sim p(\cdot \mid x),

    with :math:`m(x)` the predictive mean and :math:`s \in \mathbb R^d_{>0}`
    shared by all rows.  With ``about="center"``, :math:`m(x)` is
    :meth:`~torchregress.joint.JointPredictive.center` instead: the same point
    for Gaussian and mixture predictives, the image of the latent origin for a
    flow, where the mean would cost ``n_samples`` draws per row at every
    :meth:`transform`.  Any fixed :math:`m(x)` gives a valid change of
    variables; the two choices differ by a shift of
    :math:`(1 - s)\,(m_{\text{center}} - m_{\text{mean}})`, second order when
    :math:`s` is near one.  Its density is
    :math:`\log p'(y) = \log p\big(m + (y - m) / s\big) - \sum_j \log s_j`,
    its covariance :math:`S \Sigma(x) S` with :math:`S = \mathrm{diag}(s)`, and the
    dependence structure (copula) of the base predictive is unchanged.

    Objectives:

    * ``"variance"``: :math:`s_j^2 = \frac1{n_j}\sum_i (y_{ij} - m_{ij})^2 /
      \Sigma_{i,jj}`, the mean squared standardised residual of target ``j``
      (over its observed rows), so that standardised residuals have unit
      variance on the calibration set;
    * ``"nll"``: the :math:`s` maximising the held-out log density
      :math:`\sum_i \log p'_i(y_i)` (needs ``has_density``; with a mask, needs
      ``exact_marginals``).  Fitted by L-BFGS on :math:`\log s`, started at the
      ``"variance"`` solution.

    A target whose calibration residuals are all zero (a constant target, or
    one the mean predicts exactly) keeps :math:`s_j = 1` in both objectives.

    The guarantee is only that of an M-estimator: with exchangeable calibration
    and test rows, the fitted factors converge to those that are optimal on
    average over :math:`x`; a predictive whose miscalibration varies with
    :math:`x` stays miscalibrated locally.

    Args:
        objective: ``"variance"`` or ``"nll"``.
        n_samples: Monte Carlo draws for the mean and covariance of heads that
            do not have them in closed form.
        max_iter: L-BFGS iterations for ``"nll"``.
        about: ``"mean"`` (default) or ``"center"``, the point :math:`m(x)` the
            scaling is taken about.  With ``"center"`` and ``objective="nll"``
            no draws are needed (the search starts at :math:`s = 1`).

    Attributes:
        scale_: Fitted factors ``[d]``.
    """

    def __init__(
        self,
        objective: str = "variance",
        n_samples: int = 1024,
        max_iter: int = 100,
        about: str = "mean",
    ):
        if objective not in ("variance", "nll"):
            raise ValueError(f"objective must be 'variance' or 'nll', got {objective!r}")
        if about not in ("mean", "center"):
            raise ValueError(f"about must be 'mean' or 'center', got {about!r}")
        self.objective = objective
        self.about = about
        self.n_samples = int(n_samples)
        self.max_iter = int(max_iter)
        self.scale_: Optional[Tensor] = None

    def fit(
        self, predictive: JointPredictive, y: Tensor, mask: Optional[Tensor] = None
    ) -> "ScaleRecalibrator":
        """Fit the factors on calibration rows ``y [n, d]`` (``mask``: ``True`` = observed)."""
        y = _check_pair(predictive, y)
        obs = _check_fit_mask(y, mask)
        # The variance solution (and the draws it may cost) is the answer of
        # objective="variance" and the start of "nll" about the mean; "nll" about the
        # center starts at s = 1 and needs no draws.
        moments = self.objective == "variance" or self.about == "mean"
        with torch.no_grad(), seeded(_MEAN_SEED), fixed_moment_seed(_MEAN_SEED):
            # Covariance before the location: Monte Carlo moments then draw once for both.
            cov = predictive.covariance(self.n_samples).detach() if moments else None
            mean = self._location(predictive)
            var = torch.ones_like(mean) if cov is None else cov.diagonal(dim1=-2, dim2=-1)
        y = y.to(mean)
        resid = torch.where(obs, y - mean, torch.zeros_like(y))
        obs_f = obs.to(mean.dtype)
        msr = (resid.square() / var * obs_f).sum(0) / obs_f.sum(0)
        # A target with no residual spread (constant, or predicted exactly) has no
        # scale to fit: its factor stays 1 in both objectives.
        free = torch.isfinite(msr) & (msr > 0)
        scale = torch.ones_like(msr)
        if moments:
            scale = torch.where(free, msr.sqrt(), scale)
        if self.objective == "nll" and bool(free.any()):
            scale = self._fit_nll(predictive, mean, resid, obs, scale, mask is not None, free)
        self.scale_ = scale.detach()
        return self

    def _location(self, predictive: JointPredictive) -> Tensor:
        """The point ``m(x) [n, d]`` the scaling is taken about."""
        if self.about == "center":
            return predictive.center(self.n_samples).detach()
        return predictive.mean(self.n_samples).detach()

    def _fit_nll(
        self,
        predictive: JointPredictive,
        mean: Tensor,
        resid: Tensor,
        obs: Tensor,
        start: Tensor,
        masked: bool,
        free: Tensor,
    ) -> Tensor:
        if not predictive.has_density:
            raise ValueError("objective='nll' needs a predictive with a density")
        if masked and not predictive.exact_marginals:
            raise ValueError("objective='nll' with a mask needs exact_marginals")
        log_s = start.log().clone().requires_grad_(True)
        obs_f = obs.to(mean.dtype)
        fixed = ~free

        def full(candidate: Tensor) -> Tensor:
            return torch.where(fixed, torch.zeros_like(candidate), candidate)

        optimiser = torch.optim.LBFGS(
            [log_s], max_iter=self.max_iter, line_search_fn="strong_wolfe"
        )

        def closure() -> Tensor:
            ls = full(log_s)
            u = mean + resid / ls.exp()
            lp = predictive.log_prob(u, obs if masked else None)
            loss = -(lp - (ls * obs_f).sum(-1)).mean()
            (grad,) = torch.autograd.grad(loss, [log_s])
            log_s.grad = grad
            return loss.detach()

        optimiser.step(closure)  # type: ignore[no-untyped-call]

        def objective(candidate: Tensor) -> float:
            candidate = full(candidate)
            with torch.no_grad():
                lp = predictive.log_prob(mean + resid / candidate.exp(), obs if masked else None)
                return float(-(lp - (candidate * obs_f).sum(-1)).mean())

        # Gradients are blind to densities that are piecewise constant (quantile and
        # binned marginals): the density term has zero slope almost everywhere, so
        # L-BFGS only sees the Jacobian term.  Compare candidates by value, refine the
        # best one by a derivative-free pattern search, and never return factors that
        # score worse than leaving the predictive alone.
        candidates = [log_s.detach().clone(), start.log().clone(), torch.zeros_like(start)]
        values = [objective(c) for c in candidates]
        finite = [v if math.isfinite(v) else math.inf for v in values]
        best = candidates[finite.index(min(finite))].clone()
        best_value = min(finite)
        if best.numel() <= 64:
            step = 0.2
            for _ in range(4):
                for j in range(best.numel()):
                    if bool(fixed[j]):
                        continue
                    for delta in (step, -step):
                        trial = best.clone()
                        trial[j] += delta
                        value = objective(trial)
                        if math.isfinite(value) and value < best_value - 1e-9:
                            best, best_value = trial, value
                            break
                step *= 0.5
        return full(best).exp()

    def transform(self, predictive: JointPredictive) -> AffineJoint:
        r"""Wrap ``predictive`` as the law of :math:`m + s \odot (Y - m)`.

        Returns an :class:`~torchregress.joint.predictive.AffineJoint` with
        ``loc = (1 - s) m`` and ``scale = s``: draws, density (with masks),
        latent map, marginals, mean and covariance are all consistent with the
        rescaling, and the capability flags of ``predictive`` are kept.  When
        the predictive mean is a Monte Carlo estimate it is drawn with a fixed
        seed, so the result is reproducible and the global torch RNG is left
        untouched.
        """
        if self.scale_ is None:
            raise RuntimeError("call fit first")
        if predictive.event_dim != self.scale_.shape[0]:
            raise ValueError(f"predictive has d = {predictive.event_dim}, fitted d = {len(self)}")
        # Monte Carlo means are drawn with a fixed seed (the caller's RNG state is
        # restored), so the same predictive always gets the same recalibration.
        with torch.no_grad(), seeded(_MEAN_SEED), fixed_moment_seed(_MEAN_SEED):
            mean = self._location(predictive)
        scale = self.scale_.to(mean)
        return AffineJoint(predictive, (1.0 - scale) * mean, scale)

    def __len__(self) -> int:
        return 0 if self.scale_ is None else int(self.scale_.shape[0])


# ═══════════════════════════════════════════════════════════════════════════════
# monotone rational-quadratic spline (closed-form value, inverse and derivative)
# ═══════════════════════════════════════════════════════════════════════════════


class _MonotoneSpline:
    r"""Strictly increasing :math:`C^1` interpolant through ``(xk, yk)``, linear outside.

    Monotone rational-quadratic (Gregory and Delbourgo, 1982) on each interval,
    with knot derivatives from the harmonic mean of the adjacent secants
    (Fritsch and Butland), so it is monotone for any increasing data and exact
    for data on a straight line.  Outside ``[xk_0, xk_{K-1}]`` it continues
    linearly with the end derivatives.  ``tail_span`` sets each end derivative
    to the secant over that many intervals (1: the end interval), which steadies
    the tails when the knots are noisy.  Value, inverse (a quadratic root) and
    derivative are closed form.  Float64 throughout.
    """

    def __init__(self, xk: Tensor, yk: Tensor, tail_span: int = 1) -> None:
        xk, yk = xk.to(_F64), yk.to(_F64)
        if xk.dim() != 1 or xk.shape != yk.shape or xk.shape[0] < 2:
            raise ValueError("knots must be two 1-D tensors of the same length >= 2")
        if not bool((xk.diff() > 0).all() and (yk.diff() > 0).all()):
            raise ValueError("knots must be strictly increasing in both coordinates")
        secant = yk.diff() / xk.diff()
        inner = 2.0 * secant[:-1] * secant[1:] / (secant[:-1] + secant[1:])
        self.xk, self.yk = xk, yk
        span = int(min(max(tail_span, 1), xk.shape[0] - 1))
        left = (yk[span] - yk[0]) / (xk[span] - xk[0])
        right = (yk[-1] - yk[-1 - span]) / (xk[-1] - xk[-1 - span])
        self.deriv = torch.cat([left.reshape(1), inner, right.reshape(1)])

    def to(self, device: torch.device) -> "_MonotoneSpline":
        out = _MonotoneSpline.__new__(_MonotoneSpline)
        out.xk, out.yk, out.deriv = (t.to(device) for t in (self.xk, self.yk, self.deriv))
        return out

    def _bin(self, knots: Tensor, v: Tensor) -> tuple[Tensor, ...]:
        k = knots.shape[0]
        idx = torch.searchsorted(knots, v.contiguous(), right=True).sub(1).clamp(0, k - 2)
        x0, x1 = self.xk[idx], self.xk[idx + 1]
        y0, y1 = self.yk[idx], self.yk[idx + 1]
        return x0, x1 - x0, y0, y1 - y0, self.deriv[idx], self.deriv[idx + 1]

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """``(y, log dy/dx)`` at ``x``."""
        x = x.to(_F64)
        xc = x.clamp(self.xk[0], self.xk[-1])
        x0, w, y0, h, d0, d1 = self._bin(self.xk, xc)
        s = h / w
        xi = (xc - x0) / w
        t = xi * (1.0 - xi)
        den = s + (d0 + d1 - 2.0 * s) * t
        y = y0 + h * (s * xi.square() + d0 * t) / den
        log_dy = (
            2.0 * s.log()
            + (d1 * xi.square() + 2.0 * s * t + d0 * (1.0 - xi).square()).log()
            - 2.0 * den.log()
        )
        lo, hi = x < self.xk[0], x > self.xk[-1]
        y = torch.where(lo, self.yk[0] + self.deriv[0] * (x - self.xk[0]), y)
        y = torch.where(hi, self.yk[-1] + self.deriv[-1] * (x - self.xk[-1]), y)
        log_dy = torch.where(lo, self.deriv[0].log().expand_as(log_dy), log_dy)
        log_dy = torch.where(hi, self.deriv[-1].log().expand_as(log_dy), log_dy)
        return y, log_dy

    def inverse(self, y: Tensor) -> tuple[Tensor, Tensor]:
        """``(x, log dy/dx at x)`` for ``y``."""
        y = y.to(_F64)
        yc = y.clamp(self.yk[0], self.yk[-1])
        x0, w, y0, h, d0, d1 = self._bin(self.yk, yc)
        s = h / w
        dy = yc - y0
        c_sum = d0 + d1 - 2.0 * s
        a = h * (s - d0) + dy * c_sum
        b = h * d0 - dy * c_sum
        c = -s * dy
        disc = (b.square() - 4.0 * a * c).clamp_min(0.0)
        xi = (2.0 * c / (-b - disc.sqrt())).clamp(0.0, 1.0)
        x = x0 + xi * w
        _, log_dy = self.forward(x)
        lo, hi = y < self.yk[0], y > self.yk[-1]
        x = torch.where(lo, self.xk[0] + (y - self.yk[0]) / self.deriv[0], x)
        x = torch.where(hi, self.xk[-1] + (y - self.yk[-1]) / self.deriv[-1], x)
        log_dy = torch.where(lo, self.deriv[0].log().expand_as(log_dy), log_dy)
        log_dy = torch.where(hi, self.deriv[-1].log().expand_as(log_dy), log_dy)
        return x, log_dy


# ═══════════════════════════════════════════════════════════════════════════════
# latent recalibration
# ═══════════════════════════════════════════════════════════════════════════════


def _chi_cdf(norm: Tensor, d: int) -> Tensor:
    r""":math:`F_{\chi_d}(l) = P(d/2, l^2/2)` (regularised lower gamma), in float64."""
    l64 = norm.to(_F64)
    return torch.special.gammainc(torch.full_like(l64, 0.5 * d), 0.5 * l64.square())


def latent_calibration_error(predictive: JointPredictive, y: Tensor) -> Tensor:
    r"""Latent calibration error (L-ECE) of a predictive with a latent map.

    With :math:`L_i = \lVert T(y_i; x_i)\rVert_2` the norm of the latent residual
    and :math:`U_i = F_{\chi_d}(L_i)`, a calibrated predictive has
    :math:`U \sim \mathcal U(0, 1)`.  The error is

    .. math::
        \text{L-ECE} = \int_0^1 \big|\hat F_U(a) - a\big|\, \mathrm da ,

    computed exactly for the empirical CDF :math:`\hat F_U` of the ``n`` values
    (in float64, since the :math:`\chi_d` CDF saturates in single precision).
    Its value for a calibrated predictive is of order
    :math:`\sqrt{\pi/32}\,/\sqrt n \approx 0.31/\sqrt n` (the mean absolute
    Brownian bridge), not zero.

    Args:
        predictive: Predictive with ``has_latent``.
        y: Targets ``[n, d]``.

    Returns:
        Scalar float64 tensor on the device of ``y``.
    """
    if not predictive.has_latent:
        raise ValueError(f"{type(predictive).__name__} has no latent map")
    y = _check_pair(predictive, y)
    with torch.no_grad():
        z = predictive.to_latent(y)
    u = _chi_cdf(z.to(_F64).norm(dim=-1), predictive.event_dim).sort().values
    n = u.shape[0]
    edges = torch.cat([u.new_zeros(1), u, u.new_ones(1)])
    level = torch.arange(n + 1, device=u.device, dtype=_F64) / n

    def antiderivative(a: Tensor) -> Tensor:  # d/da [0.5 (a - c)|a - c|] = |a - c|
        return 0.5 * (a - level) * (a - level).abs()

    return (antiderivative(edges[1:]) - antiderivative(edges[:-1])).sum()


class LatentRecalibratedJoint(JointPredictive):
    r"""A predictive whose latent norm has been remapped by a radial map :math:`R`.

    With :math:`T` the latent map of the base predictive (``to_latent``) and
    :math:`R(z) = r(\lVert z\rVert)\, z / \lVert z\rVert` for a strictly
    increasing :math:`r`, the recalibrated predictive is the law of
    :math:`Y' = T^{-1}(R(Z))`, :math:`Z \sim \mathcal N(0, I_d)`.  Its latent map
    is :math:`R^{-1} \circ T`, so it can be recalibrated or conformalised again.

    Density.  With :math:`z' = T(y)`, :math:`l' = \lVert z'\rVert`,
    :math:`l = r^{-1}(l')`, :math:`z = R^{-1}(z') = (l / l') z'`: the radial map has
    Jacobian determinant :math:`\lvert\det\nabla R(z)\rvert = (r(l)/l)^{d-1} r'(l)`
    (one radial direction stretched by :math:`r'(l)`, :math:`d-1` tangential ones by
    :math:`r(l)/l`), and the base density is
    :math:`f(y) = \varphi_d(z')\,\lvert\det\nabla T(y)\rvert`, so

    .. math::
        \log f'(y) = \log f(y) + \big[\log\varphi_d(z) - \log\varphi_d(z')\big]
                     - (d - 1)\log\frac{l'}{l} - \log r'(l),

    where :math:`\log\varphi_d(z) - \log\varphi_d(z') = -\tfrac12(l^2 - l'^2)`.
    :math:`r` is represented as :math:`\log r = g(\log l)` with :math:`g` a
    monotone rational-quadratic spline, so :math:`\log r'(l) = g(a) - a +
    \log g'(a)` with :math:`a = \log l`, and the last two terms combine to
    :math:`-d\,(\log l' - \log l) - \log g'(\log l)`.

    Capabilities: ``has_latent`` is true, ``has_density`` is that of the base,
    ``exact_marginals`` is false (a radial map in latent space does not act on
    target subsets; ``marginal`` integrates by sampling).
    """

    has_latent = True
    exact_marginals = False

    def __init__(self, base: JointPredictive, spline: _MonotoneSpline) -> None:
        if not base.has_latent:
            raise ValueError(f"{type(base).__name__} has no latent map")
        self.base = base
        self.spline = spline
        self.has_density = base.has_density
        probe = base.sample(1)
        self._dtype, self._device = probe.dtype, probe.device
        self._spline_dev = spline.to(probe.device)

    @property
    def batch_size(self) -> int:
        return self.base.batch_size

    @property
    def event_dim(self) -> int:
        return self.base.event_dim

    @property
    def dtype(self) -> torch.dtype:
        return self.base.dtype

    @property
    def device(self) -> torch.device:
        return self.base.device

    def _radial(self, z: Tensor, inverse: bool) -> tuple[Tensor, Tensor, Tensor]:
        """Apply ``R`` (or ``R^{-1}``); returns (image, log|z| before, log|z| after)."""
        z64 = z.to(_F64)
        log_in = z64.norm(dim=-1).log().clamp_min(_LOG_TINY)
        spline = self._spline_dev.to(z.device)
        log_out = spline.inverse(log_in)[0] if inverse else spline.forward(log_in)[0]
        out = z64 * (log_out - log_in).exp().unsqueeze(-1)
        return out.to(z.dtype), log_in, log_out

    def sample(self, n_samples: int) -> Tensor:
        n, d = self.batch_size, self.event_dim
        z = torch.randn(n, n_samples, d, device=self._device, dtype=self._dtype)
        with torch.no_grad():
            return self.from_latent(z)

    def to_latent(self, y: Tensor) -> Tensor:
        return self._radial(self.base.to_latent(self._check(y)), inverse=True)[0]

    def from_latent(self, z: Tensor) -> Tensor:
        return self.base.from_latent(self._radial(self._check(z), inverse=False)[0])

    def log_prob(self, y: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if mask is not None:
            raise NotImplementedError(
                "a latent-recalibrated predictive cannot integrate out missing targets"
            )
        if not self.has_density:
            raise NotImplementedError(f"{type(self.base).__name__} has no density")
        y = self._check(y)
        base_lp = self.base.log_prob(y)
        z_prime = self.base.to_latent(y)
        log_lp = z_prime.to(_F64).norm(dim=-1).log().clamp_min(_LOG_TINY)  # log l'
        log_l, log_g = self._spline_dev.to(y.device).inverse(log_lp)  # log l, log g'(log l)
        d = self.event_dim
        corr = -0.5 * ((2.0 * log_l).exp() - (2.0 * log_lp).exp()) - d * (log_lp - log_l) - log_g
        return base_lp + corr.to(base_lp.dtype)


class LatentRecalibrator:
    r"""Latent recalibration (Dheur and Ben Taieb, NeurIPS 2025).

    For a predictive with a latent map :math:`T`, the latent norm
    :math:`L = \lVert T(Y; X)\rVert_2` follows the :math:`\chi_d` distribution when
    the predictive is calibrated.  On calibration data, fit a smooth, strictly
    increasing estimate :math:`\hat F_L` of the CDF of :math:`L` and set

    .. math::
        r(l) = \hat F_L^{-1}\big(F_{\chi_d}(l)\big), \qquad
        R(z) = r(\lVert z\rVert)\,\frac{z}{\lVert z\rVert}.

    The recalibrated predictive (:class:`LatentRecalibratedJoint`) samples
    :math:`y = T^{-1}(R(z))`, :math:`z \sim \mathcal N(0, I)`, so its own latent
    norm :math:`\lVert T(Y')\rVert = r(\lVert Z\rVert)` has CDF :math:`\hat F_L`.

    Estimator.  :math:`r` is fitted directly in log-log space: at levels
    :math:`p_k = (k + \tfrac12)/K`, :math:`k = 0..K-1`, the knots are
    :math:`(\log F_{\chi_d}^{-1}(p_k),\ \log \hat q_L(p_k))` with :math:`\hat q_L`
    the empirical quantile function of the calibration norms; a monotone
    rational-quadratic spline interpolates them, continued linearly (a power law
    in :math:`l`) beyond the end knots.  The implied
    :math:`\hat F_L = F_{\chi_d} \circ r^{-1}` is smooth and strictly increasing,
    :math:`r`, :math:`r^{-1}` and :math:`r'` are closed form on the device, and the
    :math:`\chi_d` quantiles are computed once, in float64, at fit time (the
    :math:`\chi_d` CDF saturates in single precision).  On a calibrated predictive
    the knots lie on the identity up to quantile noise, so :math:`R` is the
    identity up to :math:`O_p(n^{-1/2})`.

    What it does not do: the map is radial, so it corrects the distribution of
    the norm (over- or under-confidence, tail weight) and not the direction or
    shape of the latent residual; and it is fitted on average over :math:`x`.
    The latent calibration error is measured by :func:`latent_calibration_error`.

    Args:
        n_knots: Number of quantile knots ``K``; default
            :math:`\mathrm{clip}(\mathrm{round}(n^{1/3}/2), 4, 20)`.  The relative
            noise of the fitted slope :math:`r'` grows like :math:`\sqrt{K/n}`, and
            it enters the density directly, so few knots are preferred; more knots
            follow non-power-law norm distributions more closely.
    """

    def __init__(self, n_knots: Optional[int] = None) -> None:
        if n_knots is not None and n_knots < 2:
            raise ValueError("n_knots must be at least 2")
        self.n_knots = n_knots
        self.spline_: Optional[_MonotoneSpline] = None

    def fit(self, predictive: JointPredictive, y: Tensor) -> "LatentRecalibrator":
        """Fit ``r`` on calibration rows ``y [n, d]``."""
        if not predictive.has_latent:
            raise ValueError(f"{type(predictive).__name__} has no latent map")
        y = _check_pair(predictive, y)
        with torch.no_grad():
            z = predictive.to_latent(y)
        log_norm = z.to(_F64).norm(dim=-1).log().clamp_min(_LOG_TINY)
        n, d = log_norm.shape[0], predictive.event_dim
        k = self.n_knots or int(min(20, max(4, round(0.5 * n ** (1.0 / 3.0)))))
        levels = (torch.arange(k, dtype=_F64) + 0.5) / k
        chi_q = torch.as_tensor(chi2.ppf(levels.numpy(), d), dtype=_F64).sqrt().log()
        emp_q = torch.quantile(log_norm, levels.to(log_norm.device)).cpu()
        # Strictly increasing values (ties only arise from repeated norms).
        step = 1e-9 * torch.arange(k, dtype=_F64)
        emp_q = torch.cummax(emp_q - step, dim=0).values + step
        self.spline_ = _MonotoneSpline(chi_q, emp_q)
        return self

    def transform(self, predictive: JointPredictive) -> LatentRecalibratedJoint:
        """Wrap ``predictive`` with the fitted radial latent map."""
        if self.spline_ is None:
            raise RuntimeError("call fit first")
        return LatentRecalibratedJoint(predictive, self.spline_)


# ═══════════════════════════════════════════════════════════════════════════════
# conformal calibrators per target
# ═══════════════════════════════════════════════════════════════════════════════


def _draw_edge(s: int) -> float:
    """Normal score of the top plotting position ``(S - 1/2) / S``."""
    return float(torch.special.ndtri(torch.tensor((s - 0.5) / s, dtype=_F64)))


def _draws_scores(sorted_draws: Tensor, spread: Tensor, y: Tensor) -> Tensor:
    r"""Normal scores of ``y [n]`` under the smoothed marginal CDF of sorted draws ``[n, S]``.

    :math:`\hat F(x_{(k)}) = (k + \tfrac12)/S`, linear between order statistics,
    and Gaussian tails with the draws' standard deviation beyond the extremes
    (continuous and strictly increasing for distinct draws).
    """
    n, s = sorted_draws.shape
    edge = _draw_edge(s)
    idx = torch.searchsorted(sorted_draws, y.unsqueeze(1), right=True).squeeze(1) - 1
    idx = idx.clamp(0, s - 2)
    x0 = sorted_draws.gather(1, idx.unsqueeze(1)).squeeze(1)
    x1 = sorted_draws.gather(1, (idx + 1).unsqueeze(1)).squeeze(1)
    frac = ((y - x0) / (x1 - x0).clamp_min(torch.finfo(_F64).tiny)).clamp(0.0, 1.0)
    inner = torch.special.ndtri((idx.to(_F64) + frac + 0.5) / s)
    low = -edge + (y - sorted_draws[:, 0]) / spread
    high = edge + (y - sorted_draws[:, -1]) / spread
    out = torch.where(y < sorted_draws[:, 0], low, inner)
    return torch.where(y > sorted_draws[:, -1], high, out)


def _draws_quantile(sorted_draws: Tensor, spread: Tensor, w: Tensor) -> Tensor:
    """Inverse of :func:`_draws_scores` at normal scores ``w [S']`` shared by rows: ``[n, S']``."""
    s = sorted_draws.shape[1]
    edge = _draw_edge(s)
    pos = (normal_cdf(w) * s - 0.5).clamp(0.0, s - 1.0)
    lo = pos.floor().long().clamp_max(s - 1)
    hi = (lo + 1).clamp_max(s - 1)
    frac = pos - lo.to(_F64)
    value = sorted_draws[:, lo] * (1 - frac) + sorted_draws[:, hi] * frac
    tail = (w + edge).clamp_max(0.0) + (w - edge).clamp_min(0.0)
    return value + spread.unsqueeze(1) * tail


class MarginalRecalibratedJoint(JointPredictive):
    r"""Gaussian predictive with conformally recalibrated marginals and its copula kept.

    With base marginal CDFs :math:`F_j(y) = \Phi((y - \mu_j)/\sigma_j)` and fitted
    calibrators :math:`G_j`, written in normal-score form
    :math:`\tilde G_j = \Phi^{-1} \circ G_j \circ \Phi` (a strictly increasing
    spline fitted by :class:`MarginalRecalibrator`), the recalibrated predictive
    is the law of

    .. math::
        Y'_j = \mu_j + \sigma_j\, \tilde G_j^{-1}\!\big((V_j - \mu_j)/\sigma_j\big),
        \qquad V \sim p(\cdot \mid x),

    a coordinate-wise increasing map, so the Gaussian copula of the base is
    unchanged and the marginal CDF of target ``j`` is :math:`G_j \circ F_j`
    (:meth:`marginal_cdf`).  With :math:`w_j = (y_j - \mu_j)/\sigma_j`, the
    pre-image is :math:`v_j = \mu_j + \sigma_j \tilde G_j(w_j)` and the density is

    .. math::
        \log p'(y) = \log p(v) + \sum_j \log \tilde G_j'(w_j)

    (summed over observed targets under a mask).  The latent map is the base
    latent map of :math:`v`.  Because the map acts per target, marginals and
    masked densities are exact: all three capability flags are kept.  Working
    with normal scores rather than PITs avoids the loss of precision of
    :math:`\Phi(w)` near one.
    """

    has_density = True
    has_latent = True
    exact_marginals = True

    def __init__(self, base: GaussianJoint, calibrators: Sequence[_MonotoneSpline]) -> None:
        if len(calibrators) != base.event_dim:
            raise ValueError("one calibrator per target is required")
        self.base = base
        self.calibrators = [c.to(base.loc.device) for c in calibrators]
        self._mu = base.loc.to(_F64)
        self._sd = base.covariance().diagonal(dim1=-2, dim2=-1).to(_F64).sqrt()

    @property
    def batch_size(self) -> int:
        return self.base.batch_size

    @property
    def event_dim(self) -> int:
        return self.base.event_dim

    @property
    def dtype(self) -> torch.dtype:
        return self.base.dtype

    @property
    def device(self) -> torch.device:
        return self.base.device

    def _loc_scale(self, like: Tensor) -> tuple[Tensor, Tensor]:
        if like.dim() == 3:
            return self._mu.unsqueeze(1), self._sd.unsqueeze(1)
        return self._mu, self._sd

    def _map(self, w: Tensor, inverse: bool) -> tuple[Tensor, Tensor]:
        """Apply ``G~_j`` (or its inverse) to normal scores ``w [..., d]``; value and log slope."""
        vals, log_slopes = [], []
        for j, spline in enumerate(self.calibrators):
            v, log_s = spline.inverse(w[..., j]) if inverse else spline.forward(w[..., j])
            vals.append(v)
            log_slopes.append(log_s)
        return torch.stack(vals, -1), torch.stack(log_slopes, -1)

    def marginal_cdf(self, y: Tensor) -> Tensor:
        r"""Recalibrated marginal CDFs :math:`G_j(F_j(y_j))`, ``[n, d]`` or ``[n, S, d]``."""
        y = self._check(y)
        mu, sd = self._loc_scale(y)
        return normal_cdf(self._map((y.to(_F64) - mu) / sd, inverse=False)[0]).to(y.dtype)

    def _to_base(self, y: Tensor) -> tuple[Tensor, Tensor]:
        mu, sd = self._loc_scale(y)
        w_tilde, log_jac = self._map((y.to(_F64) - mu) / sd, inverse=False)
        return (mu + sd * w_tilde).to(y.dtype), log_jac

    def _from_base(self, v: Tensor) -> Tensor:
        mu, sd = self._loc_scale(v)
        w, _ = self._map((v.to(_F64) - mu) / sd, inverse=True)
        return (mu + sd * w).to(v.dtype)

    def sample(self, n_samples: int) -> Tensor:
        return self._from_base(self.base.sample(n_samples))

    def log_prob(self, y: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        y = self._check(y)
        v, log_jac = self._to_base(y)
        if mask is not None:
            mask = self._check_mask(y, mask)
            m = mask if y.dim() == 2 else mask.unsqueeze(1)
            log_jac = torch.where(m, log_jac, torch.zeros_like(log_jac))
            v = torch.where(m, v, torch.zeros_like(v))
        return self.base.log_prob(v, mask) + log_jac.sum(-1).to(y.dtype)

    def to_latent(self, y: Tensor) -> Tensor:
        return self.base.to_latent(self._to_base(self._check(y))[0])

    def from_latent(self, z: Tensor) -> Tensor:
        return self._from_base(self.base.from_latent(self._check(z)))

    def marginal(self, dims: Union[int, Sequence[int]]) -> "MarginalRecalibratedJoint":
        idx = target_index(dims, self.event_dim)
        return MarginalRecalibratedJoint(
            self.base.marginal(idx), [self.calibrators[i] for i in idx]
        )


class MarginalRecalibrator:
    r"""Per-target conformal calibrators (split conformal predictive systems).

    For target ``j`` and the ``m`` calibration rows with that target observed,
    compute the probability integral transforms
    :math:`u_{ij} = F_j(y_{ij} \mid x_i)` under the predictive's marginal, sort
    them, :math:`u_{(1)} < \dots < u_{(m)}`, and let :math:`G_j` be a continuous,
    strictly increasing map of :math:`[0, 1]` onto itself through the points
    :math:`(u_{(k)}, k/(m+1))` for the knot ranks :math:`k \in \mathcal K`:
    :math:`K` evenly spaced ranks from :math:`1` to :math:`m` (by default
    :math:`K = \min(m, \mathrm{clip}(\mathrm{round}(\sqrt m / 2), 10, 100))`), or every
    rank with ``n_knots="all"``.  The recalibrated
    marginal CDF is :math:`G_j \circ F_j`; the dependence of the base predictive
    is kept by applying :math:`F_j^{-1} \circ G_j^{-1} \circ F_j` coordinate-wise
    to its draws (or, for a Gaussian, to its density by a change of variables).

    :math:`G_j` is built in normal-score form,
    :math:`\tilde G_j = \Phi^{-1} \circ G_j \circ \Phi`: a monotone
    rational-quadratic spline through :math:`(\Phi^{-1}(u_{(k)}),
    \Phi^{-1}(k/(m+1)))`, continued linearly beyond the end knots (a Gaussian-type
    tail) with slopes taken over about :math:`\sqrt{|\mathcal K|}` end intervals.
    It is the identity up to the noise of the calibration PITs when the base is
    calibrated.

    Finite-sample property.  If the calibration pairs and the test pair are
    exchangeable and the PITs are almost surely distinct (continuous marginal
    CDFs, or randomised PITs), the rank of the test PIT among the :math:`m+1` PITs
    is uniform, so the recalibrated PIT :math:`V_j = G_j(F_j(Y_j \mid X))` satisfies

    .. math::
        P\big(V_j \le k/(m+1)\big) = k/(m+1), \qquad k \in \mathcal K,

    because :math:`V_j \le k/(m+1)` exactly when the test PIT is at most
    :math:`u_{(k)}`.  Between knots both :math:`P(V_j \le a)` and :math:`a` are
    increasing, so :math:`\sup_a |P(V_j \le a) - a|` is at most the largest gap
    between consecutive knot levels in :math:`\{0\} \cup \{k/(m+1)\}_{k \in
    \mathcal K} \cup \{1\}`.  For the default (end knots at ranks 1 and
    :math:`m`, :math:`K` evenly spaced) that gap is about :math:`1/(K-1)`, e.g.
    0.048 at :math:`m = 2000` (:math:`K = 22`).  With ``n_knots="all"`` it is
    :math:`1/(m+1)`: probabilistic calibration of each target to within
    :math:`1/(m+1)`.  Either way the statement is marginal over :math:`x` and the
    calibration set, and it holds for one
    target at a time; it says nothing about calibration at an individual
    :math:`x` or about the joint (copula) calibration, which is inherited from
    the base.

    Marginal CDFs.  For a :class:`~torchregress.joint.predictive.GaussianJoint`
    they are closed form and :meth:`transform` returns a
    :class:`MarginalRecalibratedJoint` (density, latent map and exact marginals
    kept).  For any other predictive, :math:`F_j` is the smoothed marginal CDF of
    ``n_samples`` draws of the row, :math:`\hat F_j(x_{(k)}) = (k + \tfrac12)/S`
    at the order statistics, linear in between and with Gaussian tails (scaled
    by the draws' standard deviation) beyond the extremes; it is continuous and
    strictly increasing, so the finite-sample property above holds for it
    (fresh draws per row keep the rows exchangeable).  :meth:`transform` then
    returns a draws-only :class:`~torchregress.joint.predictive.SampleJoint`:
    each draw :math:`x` of target ``j`` is moved to
    :math:`\hat F_j^{-1}(G_j^{-1}(\hat F_j(x)))`, computed on the same draws.
    The map is strictly increasing, so ranks within each target, hence the
    empirical copula of the draws, are kept exactly.  Beyond the extreme draws
    the recalibrated tail is the Gaussian tail of :math:`\hat F_j` stretched by
    :math:`\tilde G_j`, an extrapolation rather than an estimate.

    Knots and density.  With every rank as a knot, :math:`G_j'` follows the
    spacings of the calibration PITs, which are exponential-like noise, so the
    recalibrated *density* is rough: about :math:`\psi(2) \approx 0.42` nats per
    target worse in log score even on a calibrated predictive.  The default
    trades the :math:`1/(m+1)` bound for a smooth density; the excess log score
    over the true density grows roughly like :math:`K/m` per target (measured on
    calibrated and mis-scaled Gaussians, :math:`d = 3`: 0.055 nats at
    :math:`m = 300`, 0.026 at :math:`m = 2000`, 0.006–0.009 at :math:`m = 8000`,
    in total over the three targets).

    Args:
        n_knots: Number of knot ranks per target; ``None`` for the automatic
            :math:`\min(m, \mathrm{clip}(\mathrm{round}(\sqrt m / 2), 10, 100))`,
            ``"all"`` for every rank (exact :math:`1/(m+1)` bound, rough density).
        n_samples: Draws per row for predictives without closed-form marginals
            (used both in ``fit`` and in ``transform``).

    Attributes:
        calibrators_: One fitted :math:`\tilde G_j` per target.
    """

    def __init__(
        self, n_knots: Union[int, None, Literal["all"]] = None, n_samples: int = 1000
    ) -> None:
        if n_samples < 2:
            raise ValueError("n_samples must be at least 2")
        if isinstance(n_knots, str):
            if n_knots != "all":
                raise ValueError(f"n_knots must be an int, None or 'all', got {n_knots!r}")
        elif n_knots is not None and n_knots < 2:
            raise ValueError("n_knots must be at least 2")
        self.n_knots = n_knots
        self.n_samples = int(n_samples)
        self.calibrators_: Optional[list[_MonotoneSpline]] = None

    def _knot_count(self, m: int) -> int:
        if self.n_knots == "all":
            return m
        if self.n_knots is None:
            return min(m, int(min(100, max(10, round(0.5 * math.sqrt(m))))))
        return min(m, int(self.n_knots))

    def scores(self, predictive: JointPredictive, y: Tensor) -> Tensor:
        r"""Normal scores :math:`\Phi^{-1}(F_j(y_j \mid x))`, ``[n, d]`` float64.

        Closed form for a Gaussian (the standardised residual, without forming
        the PIT); otherwise under the smoothed marginal CDF of ``n_samples``
        draws per row (see the class docstring).
        """
        y = _check_pair(predictive, y)
        if isinstance(predictive, GaussianJoint):
            sd = predictive.covariance().diagonal(dim1=-2, dim2=-1).to(_F64).sqrt()
            return (y.to(_F64) - predictive.loc.to(_F64)) / sd
        with torch.no_grad():
            sorted_draws = predictive.sample(self.n_samples).to(_F64).sort(dim=1).values
        spread = sorted_draws.std(dim=1).clamp_min(torch.finfo(_F64).tiny)
        y64 = y.to(sorted_draws)
        cols = [
            _draws_scores(sorted_draws[:, :, j].contiguous(), spread[:, j], y64[:, j])
            for j in range(predictive.event_dim)
        ]
        return torch.stack(cols, dim=1)

    def pit(self, predictive: JointPredictive, y: Tensor) -> Tensor:
        """Marginal PITs ``F_j(y_j | x)``, ``[n, d]`` float64 (see :meth:`scores`)."""
        return normal_cdf(self.scores(predictive, y))

    def fit(
        self, predictive: JointPredictive, y: Tensor, mask: Optional[Tensor] = None
    ) -> "MarginalRecalibrator":
        """Fit ``G_j`` on calibration rows ``y [n, d]`` (``mask``: ``True`` = observed)."""
        y = predictive.as_input(y)
        obs = _check_fit_mask(y, mask)
        w = self.scores(predictive, torch.where(obs, y, torch.zeros_like(y)))
        calibrators = []
        for j in range(predictive.event_dim):
            wj = w[obs[:, j], j].sort().values.cpu()
            m = wj.shape[0]
            if m < 2 or bool((wj.diff() <= 0).any()):
                raise ValueError(f"target {j} needs at least two distinct calibration PITs")
            ranks = torch.arange(1, m + 1)
            k = self._knot_count(m)
            if k < m:
                ranks = torch.linspace(1, m, k, dtype=_F64).round().long().unique()
            levels = torch.special.ndtri(ranks.to(_F64) / (m + 1))
            span = round(len(ranks) ** 0.5)
            calibrators.append(_MonotoneSpline(wj[ranks - 1], levels, tail_span=span))
        self.calibrators_ = calibrators
        return self

    def transform(self, predictive: JointPredictive) -> JointPredictive:
        """Recalibrated predictive (with a density for a Gaussian, draws-only otherwise)."""
        if self.calibrators_ is None:
            raise RuntimeError("call fit first")
        if predictive.event_dim != len(self.calibrators_):
            raise ValueError(
                f"predictive has d = {predictive.event_dim}, fitted d = {len(self.calibrators_)}"
            )
        if isinstance(predictive, GaussianJoint):
            return MarginalRecalibratedJoint(predictive, self.calibrators_)
        with torch.no_grad():
            draws = predictive.sample(self.n_samples)
        n, s, d = draws.shape
        sorted_draws, order = draws.to(_F64).sort(dim=1)
        spread = sorted_draws.std(dim=1).clamp_min(torch.finfo(_F64).tiny)  # [n, d]
        levels = (torch.arange(s, device=draws.device, dtype=_F64) + 0.5) / s
        out = torch.empty_like(draws)
        for j in range(d):
            spline = self.calibrators_[j].to(draws.device)
            w_new = spline.inverse(torch.special.ndtri(levels))[0]  # G~_j^{-1}
            col = sorted_draws[:, :, j].contiguous()
            value = _draws_quantile(col, spread[:, j], w_new)
            out[:, :, j] = torch.empty_like(col).scatter_(1, order[:, :, j], value).to(draws)
        return SampleJoint(out)
