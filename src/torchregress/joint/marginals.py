r"""Per-target predictive marginals: the univariate half of a Sklar joint.

A :class:`Marginals` object holds ``n`` rows times ``d`` targets of univariate
predictive distributions :math:`F_j(\cdot \mid x_i)`.  It is how a per-target
engine (a Gaussian head, a quantile regressor, a histogram head of a tabular
foundation model, a sampler) enters the joint machinery: the dependence ladder
in :mod:`torchregress.joint.dependence` works on the normal scores

.. math::
    z_j = \Phi^{-1}\big(F_j(y_j \mid x)\big),

and :class:`~torchregress.joint.dependence.SklarJoint` turns a density on the
score space back into a joint density on the targets.

Every adapter returns the CDF, its inverse and the log density with the same
layouts: ``y [n, d]`` or ``[n, S, d]`` in, the same shape out.  The CDFs are
continuous and strictly increasing and the densities strictly positive, so the
scores are finite and the change of variables is exact:

* :class:`GaussianMarginals` -- :math:`\mathcal N(\mu_{ij}, \sigma_{ij}^2)`;
* :class:`QuantileMarginals` -- a grid of predicted quantiles, piecewise-linear
  CDF between the knots and exponential tails beyond them (the TabICL
  convention);
* :class:`BinnedMarginals` -- bin edges and bin probabilities (histogram heads,
  TabPFN's bar distribution), piecewise-constant density with exponential tails;
* :class:`SampleMarginals` -- draws, read as an empirical CDF interpolated
  linearly between the order statistics, with exponential tails.

References:
    Sklar, A. (1959). Fonctions de répartition à n dimensions et leurs marges.
    Publ. Inst. Statist. Univ. Paris 8, 229--231.

    Qu, J., Holzmüller, D., Varoquaux, G., Le Morvan, M. (2025). TabICL: a
    tabular foundation model for in-context learning on large data
    (exponential tails on a quantile grid).

    Hollmann, N. et al. (2025). Accurate predictions on small data with a
    tabular foundation model. Nature 637 (TabPFN's bar distribution).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional, Union

import torch
from torch import Tensor

from ..utils.distributions import normal_cdf
from ._common import target_index

__all__ = [
    "BinnedMarginals",
    "GaussianMarginals",
    "Marginals",
    "QuantileMarginals",
    "SampleMarginals",
]

_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)
#: Smallest gap between consecutive knots, relative to the knot spread.
_TIE_GAP = 1e-6


class Marginals(ABC):
    r"""``n`` rows times ``d`` targets of continuous univariate predictives.

    Subclasses implement :meth:`cdf`, :meth:`icdf`, :meth:`log_prob` and
    :meth:`select_targets`; the normal-score maps are derived from them.
    """

    # ------------------------------------------------------------- required
    @property
    @abstractmethod
    def batch_size(self) -> int:
        """Number of rows ``n``."""

    @property
    @abstractmethod
    def event_dim(self) -> int:
        """Number of targets ``d``."""

    @abstractmethod
    def cdf(self, y: Tensor) -> Tensor:
        """:math:`F_j(y_j \\mid x_i)` for ``y [n, d]`` or ``[n, S, d]``; same shape."""

    @abstractmethod
    def icdf(self, u: Tensor) -> Tensor:
        """Inverse of :meth:`cdf`, for ``u`` in ``(0, 1)`` of shape ``[n, d]`` or ``[n, S, d]``."""

    @abstractmethod
    def log_prob(self, y: Tensor) -> Tensor:
        """Per-target log density :math:`\\log f_j(y_j \\mid x_i)`, same shape as ``y``."""

    @abstractmethod
    def select_targets(self, dims: Union[int, Sequence[int]]) -> "Marginals":
        """The marginals of the targets ``dims`` (in that order)."""

    # -------------------------------------------------------------- derived
    def normal_scores(self, y: Tensor, eps: float = 1e-6) -> Tensor:
        r"""Normal scores :math:`z = \Phi^{-1}(\mathrm{clip}(F(y), \varepsilon, 1 - \varepsilon))`.

        Clipping keeps the scores finite (:math:`|z| \le \Phi^{-1}(1 - \varepsilon)`,
        4.75 for the default); it moves the joint density only on the set
        where some :math:`F_j` is within :math:`\varepsilon` of 0 or 1.
        """
        u = self.cdf(y).clamp(eps, 1.0 - eps)
        return torch.special.ndtri(u)

    def from_normal_scores(self, z: Tensor, eps: float = 1e-6) -> Tensor:
        r"""Inverse of :meth:`normal_scores`.

        :math:`y = F^{-1}(\mathrm{clip}(\Phi(z), \varepsilon, 1 - \varepsilon))`.
        """
        u = normal_cdf(z).clamp(eps, 1.0 - eps)
        return self.icdf(u)

    def median(self) -> Tensor:
        """Per-target median ``[n, d]``."""
        dtype = self._dtype()
        device = self._device()
        half = torch.full((self.batch_size, self.event_dim), 0.5, dtype=dtype, device=device)
        return self.icdf(half)

    # -------------------------------------------------------------- helpers
    @abstractmethod
    def _dtype(self) -> torch.dtype: ...

    @abstractmethod
    def _device(self) -> torch.device: ...

    def _check(self, y: Tensor) -> Tensor:
        if y.dim() not in (2, 3) or y.shape[0] != self.batch_size or y.shape[-1] != self.event_dim:
            raise ValueError(
                f"expected [n, d] or [n, S, d] with n = {self.batch_size} and "
                f"d = {self.event_dim}, got {tuple(y.shape)}"
            )
        return y

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n={self.batch_size}, d={self.event_dim})"


# ═══════════════════════════════════════════════════════════════════════════════
# Gaussian
# ═══════════════════════════════════════════════════════════════════════════════


class GaussianMarginals(Marginals):
    r"""Independent normal marginals :math:`\mathcal N(\mu_{ij}, \sigma_{ij}^2)`.

    The normal score is affine, :math:`z = (y - \mu)/\sigma`, so
    :meth:`normal_scores` and :meth:`from_normal_scores` are computed in
    closed form without clipping: a Gaussian copula on these marginals is
    exactly a multivariate normal.

    Args:
        mean: ``[n, d]``.
        std: Positive ``[n, d]``.
    """

    def __init__(self, mean: Tensor, std: Tensor) -> None:
        if mean.dim() != 2 or std.shape != mean.shape:
            raise ValueError(
                f"mean and std must both be [n, d], got {tuple(mean.shape)} and {tuple(std.shape)}"
            )
        if not bool((std > 0).all()):
            raise ValueError("std must be positive")
        self.loc = mean
        self.scale = std.to(device=mean.device, dtype=mean.dtype)

    @property
    def batch_size(self) -> int:
        return int(self.loc.shape[0])

    @property
    def event_dim(self) -> int:
        return int(self.loc.shape[1])

    def _dtype(self) -> torch.dtype:
        return self.loc.dtype

    def _device(self) -> torch.device:
        return self.loc.device

    def _params(self, like: Tensor) -> tuple[Tensor, Tensor]:
        loc, scale = self.loc.to(like.dtype), self.scale.to(like.dtype)
        if like.dim() == 3:
            return loc.unsqueeze(1), scale.unsqueeze(1)
        return loc, scale

    def cdf(self, y: Tensor) -> Tensor:
        loc, scale = self._params(self._check(y))
        return normal_cdf((y - loc) / scale)

    def icdf(self, u: Tensor) -> Tensor:
        loc, scale = self._params(self._check(u))
        return loc + scale * torch.special.ndtri(u)

    def log_prob(self, y: Tensor) -> Tensor:
        loc, scale = self._params(self._check(y))
        r = (y - loc) / scale
        return -0.5 * r.square() - scale.log() - _HALF_LOG_2PI

    def normal_scores(self, y: Tensor, eps: float = 1e-6) -> Tensor:
        loc, scale = self._params(self._check(y))
        return (y - loc) / scale

    def from_normal_scores(self, z: Tensor, eps: float = 1e-6) -> Tensor:
        loc, scale = self._params(self._check(z))
        return loc + scale * z

    def median(self) -> Tensor:
        return self.loc

    def select_targets(self, dims: Union[int, Sequence[int]]) -> "GaussianMarginals":
        idx = target_index(dims, self.event_dim)
        return GaussianMarginals(self.loc[:, idx], self.scale[:, idx])


# ═══════════════════════════════════════════════════════════════════════════════
# piecewise-linear CDF with exponential tails
# ═══════════════════════════════════════════════════════════════════════════════


class _PiecewiseMarginals(Marginals):
    r"""CDF through knots :math:`(q_k, \tau_k)_{k=1}^K`, linear between them.

    Outside the knots the tails are exponential,

    .. math::
        F(y) = \tau_1\, e^{(y - q_1)/b_L} \quad (y < q_1), \qquad
        1 - F(y) = (1 - \tau_K)\, e^{-(y - q_K)/b_R} \quad (y > q_K),

    so the density is piecewise constant inside, exponential outside,
    strictly positive everywhere, and integrates to one.

    Internal layout: ``knots [n, d, K]`` strictly increasing, ``levels``
    broadcastable to it and strictly increasing in ``(0, 1)``, tail scales
    ``[n, d]``.
    """

    knots: Tensor
    levels: Tensor
    left_scale: Tensor
    right_scale: Tensor

    def _set_parts(
        self, knots: Tensor, levels: Tensor, left_scale: Tensor, right_scale: Tensor
    ) -> None:
        self.knots = knots
        self.levels = levels.to(device=knots.device, dtype=knots.dtype)
        self.left_scale = left_scale
        self.right_scale = right_scale

    def _with_parts(
        self, knots: Tensor, levels: Tensor, left_scale: Tensor, right_scale: Tensor
    ) -> "_PiecewiseMarginals":
        out = type(self).__new__(type(self))
        out.__dict__.update(self.__dict__)  # scalar settings such as tail_mass
        out._set_parts(knots, levels, left_scale, right_scale)
        return out

    @property
    def batch_size(self) -> int:
        return int(self.knots.shape[0])

    @property
    def event_dim(self) -> int:
        return int(self.knots.shape[1])

    def _dtype(self) -> torch.dtype:
        return self.knots.dtype

    def _device(self) -> torch.device:
        return self.knots.device

    # layout: [n, d] -> [n, d, 1], [n, S, d] -> [n, d, S]
    @staticmethod
    def _inner(y: Tensor) -> Tensor:
        return (y.unsqueeze(-1) if y.dim() == 2 else y.movedim(1, 2)).contiguous()

    @staticmethod
    def _outer(v: Tensor, like: Tensor) -> Tensor:
        return v.squeeze(-1) if like.dim() == 2 else v.movedim(2, 1)

    def _grid(self, like: Tensor) -> tuple[Tensor, Tensor]:
        q = self.knots.to(like.dtype)
        t = self.levels.to(like.dtype).expand_as(q).contiguous()
        return q, t

    def _segment(self, k: Tensor, q: Tensor, t: Tensor) -> tuple[Tensor, ...]:
        n_knots = q.shape[-1]
        lo = (k - 1).clamp(0, n_knots - 2)
        hi = lo + 1
        return q.gather(-1, lo), q.gather(-1, hi), t.gather(-1, lo), t.gather(-1, hi)

    def cdf(self, y: Tensor) -> Tensor:
        yi = self._inner(self._check(y))
        q, t = self._grid(yi)
        n_knots = q.shape[-1]
        k = torch.searchsorted(q, yi, right=True)
        q_lo, q_hi, t_lo, t_hi = self._segment(k, q, t)
        inside = t_lo + (t_hi - t_lo) * ((yi - q_lo) / (q_hi - q_lo)).clamp(0.0, 1.0)
        q1, qk = q[..., :1], q[..., -1:]
        t1, tk = t[..., :1], t[..., -1:]
        bl = self.left_scale.to(yi.dtype).unsqueeze(-1)
        br = self.right_scale.to(yi.dtype).unsqueeze(-1)
        left = t1 * torch.exp(((yi - q1) / bl).clamp(max=0.0))
        right = 1.0 - (1.0 - tk) * torch.exp(-((yi - qk) / br).clamp(min=0.0))
        out = torch.where(k == 0, left, torch.where(k >= n_knots, right, inside))
        return self._outer(out, y)

    def icdf(self, u: Tensor) -> Tensor:
        ui = self._inner(self._check(u))
        q, t = self._grid(ui)
        n_knots = q.shape[-1]
        k = torch.searchsorted(t, ui, right=True)
        q_lo, q_hi, t_lo, t_hi = self._segment(k, q, t)
        inside = q_lo + (q_hi - q_lo) * ((ui - t_lo) / (t_hi - t_lo)).clamp(0.0, 1.0)
        q1, qk = q[..., :1], q[..., -1:]
        t1, tk = t[..., :1], t[..., -1:]
        bl = self.left_scale.to(ui.dtype).unsqueeze(-1)
        br = self.right_scale.to(ui.dtype).unsqueeze(-1)
        left = q1 + bl * torch.log((ui / t1).clamp(max=1.0))
        right = qk - br * torch.log(((1.0 - ui) / (1.0 - tk)).clamp(max=1.0))
        out = torch.where(k == 0, left, torch.where(k >= n_knots, right, inside))
        return self._outer(out, u)

    def log_prob(self, y: Tensor) -> Tensor:
        yi = self._inner(self._check(y))
        q, t = self._grid(yi)
        n_knots = q.shape[-1]
        k = torch.searchsorted(q, yi, right=True)
        q_lo, q_hi, t_lo, t_hi = self._segment(k, q, t)
        inside = torch.log(t_hi - t_lo) - torch.log(q_hi - q_lo)
        q1, qk = q[..., :1], q[..., -1:]
        t1, tk = t[..., :1], t[..., -1:]
        bl = self.left_scale.to(yi.dtype).unsqueeze(-1)
        br = self.right_scale.to(yi.dtype).unsqueeze(-1)
        left = torch.log(t1 / bl) + ((yi - q1) / bl).clamp(max=0.0)
        right = torch.log((1.0 - tk) / br) - ((yi - qk) / br).clamp(min=0.0)
        out = torch.where(k == 0, left, torch.where(k >= n_knots, right, inside))
        return self._outer(out, y)

    def select_targets(self, dims: Union[int, Sequence[int]]) -> "_PiecewiseMarginals":
        idx = target_index(dims, self.event_dim)
        levels = self.levels if self.levels.shape[1] == 1 else self.levels[:, idx]
        return self._with_parts(
            self.knots[:, idx], levels, self.left_scale[:, idx], self.right_scale[:, idx]
        )


def _strictly_increasing(values: Tensor) -> Tensor:
    r"""Sort along the last axis and separate exact ties.

    With :math:`\delta` a small gap (``1e-6`` of the spread of the row, never
    below a few float ulps), returns
    :math:`q'_k = \mathrm{cummax}_k(q_k - k\delta) + k\delta`, which equals the
    sorted input wherever consecutive gaps are already at least
    :math:`\delta` and is otherwise the smallest strictly increasing
    correction.
    """
    q = torch.sort(values, dim=-1).values
    spread = q[..., -1:] - q[..., :1]
    magnitude = q.abs().amax(dim=-1, keepdim=True)
    base = torch.where(spread > 0, spread, magnitude + 1.0)
    floor = 8.0 * torch.finfo(q.dtype).eps * (magnitude + 1.0)
    delta = torch.maximum(_TIE_GAP * base, floor)
    step = torch.arange(q.shape[-1], device=q.device, dtype=q.dtype) * delta
    return torch.cummax(q - step, dim=-1).values + step


class QuantileMarginals(_PiecewiseMarginals):
    r"""Marginals from a grid of predicted quantiles (quantile regression, TabICL).

    With levels :math:`0 < \tau_1 < \dots < \tau_Q < 1` and knots
    :math:`q_1 < \dots < q_Q`, the CDF is linear between the knots and has
    exponential tails :math:`F(y) = \tau_1 e^{(y - q_1)/b_L}` below
    :math:`q_1` and :math:`1 - F(y) = (1 - \tau_Q) e^{-(y - q_Q)/b_R}` above
    :math:`q_Q`.  The tail scales are fitted by least squares (through the
    outermost knot) to the ``tail_knots`` outermost knots on each side:
    :math:`\log(\tau_k / \tau_1) \approx (q_k - q_1)/b_L`, and symmetrically on
    the right.  With ``tail_knots = 2`` the tail passes through the two
    outermost knots.

    Repairs, both documented properties of the input rather than silent
    fixes of the model:

    * **crossing quantiles** are sorted per row and target (the rearrangement
      of Chernozhukov, Fernández-Val and Galichon, 2010, which never increases
      the quantile loss);
    * **exact ties** are separated by a gap of ``1e-6`` of the knot spread so
      the CDF is strictly increasing (a tie is a point mass, which has no
      density; the repaired marginal puts that mass on a short interval).

    Args:
        levels: Probability levels ``[Q]``, strictly increasing in ``(0, 1)``, ``Q >= 2``.
        quantiles: ``[n, d, Q]``.
        tail_knots: Knots per side used to fit the tail scales (at least 2,
            at most ``Q``).  Default ``max(2, round(Q / 50))``: 20 for a
            999-level grid, as in TabICL.
    """

    def __init__(self, levels: Tensor, quantiles: Tensor, tail_knots: Optional[int] = None) -> None:
        if levels.dim() != 1 or levels.shape[0] < 2:
            raise ValueError(f"levels must be [Q] with Q >= 2, got {tuple(levels.shape)}")
        if quantiles.dim() != 3 or quantiles.shape[-1] != levels.shape[0]:
            raise ValueError(
                f"quantiles must be [n, d, Q] with Q = {levels.shape[0]}, "
                f"got {tuple(quantiles.shape)}"
            )
        levels = levels.to(device=quantiles.device, dtype=quantiles.dtype)
        if not bool((levels > 0).all() and (levels < 1).all() and (levels.diff() > 0).all()):
            raise ValueError("levels must be strictly increasing inside (0, 1)")
        n_levels = int(levels.shape[0])
        m = max(2, round(n_levels / 50)) if tail_knots is None else int(tail_knots)
        if not 2 <= m <= n_levels:
            raise ValueError(f"tail_knots must lie in [2, {n_levels}], got {m}")
        q = _strictly_increasing(quantiles)
        left, right = self._tail_scales(levels, q, m)
        self._set_parts(q, levels.reshape(1, 1, -1), left, right)

    @staticmethod
    def _tail_scales(levels: Tensor, q: Tensor, m: int) -> tuple[Tensor, Tensor]:
        # log(tau_k / tau_1) = (q_k - q_1) / b: least squares through the origin.
        c_left = torch.log(levels[1:m] / levels[0])  # [m - 1]
        a_left = q[..., 1:m] - q[..., :1]  # [n, d, m - 1]
        left = (a_left * c_left).sum(-1) / c_left.square().sum()
        tail = 1.0 - levels
        c_right = torch.log(tail[-m:-1].flip(0) / tail[-1])
        a_right = q[..., -1:] - q[..., -m:-1].flip(-1)
        right = (a_right * c_right).sum(-1) / c_right.square().sum()
        return left, right


class SampleMarginals(QuantileMarginals):
    r"""Marginals from draws ``[n, S, d]`` (ensembles, generators, forests).

    The order statistics :math:`x_{(1)} \le \dots \le x_{(S)}` of each row and
    target are read as quantiles at the plotting positions
    :math:`\tau_k = k / (S + 1)`: the CDF interpolates linearly between them
    and has exponential tails beyond :math:`x_{(1)}` and :math:`x_{(S)}`
    (fitted as in :class:`QuantileMarginals`).  Tied draws are separated as in
    :class:`QuantileMarginals`.

    Args:
        samples: ``[n, S, d]`` with ``S >= 2``.
        tail_knots: As in :class:`QuantileMarginals`.
    """

    def __init__(self, samples: Tensor, tail_knots: Optional[int] = None) -> None:
        if samples.dim() != 3 or samples.shape[1] < 2:
            raise ValueError(f"samples must be [n, S, d] with S >= 2, got {tuple(samples.shape)}")
        n_draws = int(samples.shape[1])
        levels = torch.arange(1, n_draws + 1, device=samples.device, dtype=samples.dtype) / (
            n_draws + 1
        )
        super().__init__(levels, samples.movedim(1, 2), tail_knots=tail_knots)


class BinnedMarginals(_PiecewiseMarginals):
    r"""Histogram marginals: bin edges and bin probabilities (histogram heads, TabPFN).

    Inside :math:`[e_0, e_B]` the density is piecewise constant,
    :math:`(1 - t)\, p_k / (e_k - e_{k-1})` on bin :math:`k`; outside, the
    tail mass :math:`t/2` on each side decays exponentially with the width of
    the outermost bin as scale.  Bins with zero probability would make the
    CDF flat, so the probabilities are first floored,
    :math:`p \leftarrow (1 - \lambda) p + \lambda / B` with
    :math:`\lambda` = ``min_prob`` :math:`\cdot B`.

    Args:
        edges: Strictly increasing bin edges ``[n, d, B+1]``, ``[d, B+1]`` or ``[B+1]``.
        probs: Non-negative bin probabilities ``[n, d, B]`` summing to one over ``B``.
        tail_mass: Total mass :math:`t` placed outside the edges (half on each side).
        min_prob: Probability floor per bin (as a share of a uniform bin).
    """

    def __init__(
        self, edges: Tensor, probs: Tensor, tail_mass: float = 1e-3, min_prob: float = 1e-6
    ) -> None:
        if probs.dim() != 3 or probs.shape[-1] < 1:
            raise ValueError(f"probs must be [n, d, B], got {tuple(probs.shape)}")
        n, d, n_bins = probs.shape
        edges = edges.to(device=probs.device, dtype=probs.dtype)
        if edges.shape[-1] != n_bins + 1 or edges.dim() not in (1, 2, 3):
            raise ValueError(
                f"edges must be [B+1], [d, B+1] or [n, d, B+1] with B = {n_bins}, "
                f"got {tuple(edges.shape)}"
            )
        edges = edges.expand(n, d, n_bins + 1)
        if not bool((edges.diff(dim=-1) > 0).all()):
            raise ValueError("edges must be strictly increasing")
        if bool((probs < 0).any()):
            raise ValueError("probs must be non-negative")
        if not 0.0 < tail_mass < 1.0:
            raise ValueError("tail_mass must lie in (0, 1)")
        if not 0.0 <= min_prob * n_bins < 1.0:
            raise ValueError("min_prob * B must lie in [0, 1)")
        p = probs / probs.sum(-1, keepdim=True)
        lam = min_prob * n_bins
        p = (1.0 - lam) * p + lam / n_bins
        cum = torch.cumsum(p, dim=-1)
        cum = cum / cum[..., -1:]
        zero = torch.zeros_like(cum[..., :1])
        levels = 0.5 * tail_mass + (1.0 - tail_mass) * torch.cat([zero, cum], dim=-1)
        widths = edges.diff(dim=-1)
        self.tail_mass = float(tail_mass)
        self._set_parts(edges.contiguous(), levels, widths[..., 0], widths[..., -1])
