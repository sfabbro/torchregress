r"""Maximum-likelihood estimates and likelihood-ratio intervals from score functions.

This module turns any learned or analytic *score function* into frequentist
point estimates and confidence intervals for a continuous, multivariate
parameter :math:`\theta \in \mathbb{R}^P` restricted to a box.  It is the
continuous counterpart of the candidate-bank helpers in
:mod:`torchregress.inference.ratio`: instead of scoring a fixed grid, it
maximizes the score by batched multistart Newton ascent and inverts the
likelihood-ratio test by profile root finding.  The cost scales with the number
of rows ``N`` and polynomially in ``P`` (not exponentially as a grid does).

Score convention
----------------
A score function is any callable

.. math::

    s(\theta, x) = \log p(x \mid \theta) + c(x),

``score(theta: Tensor[M, P], x: Tensor[M, ...]) -> Tensor[M]``, evaluated row by
row (row ``m`` of the output depends only on row ``m`` of ``theta`` and ``x``)
and twice differentiable in ``theta``.  The additive term :math:`c(x)` is
arbitrary and cancels in every quantity computed here.  Examples:

* a ratio classifier trained with :class:`~torchregress.losses.NeuralRatioLoss`
  (its logit is :math:`\log p(x\mid\theta) - \log p(x)`), via
  :func:`ratio_classifier_score`;
* a conditional normalizing flow, e.g. from
  :class:`~torchregress.losses.ContrastiveFlowLoss`, via :func:`flow_score`;
* an analytic log-likelihood written in torch, used as is.

Statistics
----------
With :math:`\hat\theta(x) = \arg\max_{\theta \in B} s(\theta, x)` and
:math:`s_{\max} = s(\hat\theta, x)`, the likelihood-ratio statistic of a
hypothesis :math:`\theta_0` is

.. math::

    \lambda(\theta_0; x) = s_{\max}(x) - s(\theta_0, x) \;\ge\; 0,

i.e. half the usual :math:`-2 \log \Lambda`.  The profile interval of
coordinate :math:`j` at threshold :math:`c` is

.. math::

    I_j(x) = \Big\{ t : \max_{\theta:\ \theta_j = t} s(\theta, x)
             \;\ge\; s_{\max}(x) - c \Big\}.

Under Wilks' theorem :math:`2\lambda \sim \chi^2_{\mathrm{df}}`, so
:math:`c =` :func:`wilks_threshold` ``(confidence, df=1)`` gives approximate
coverage (0.5 for 68.27 %, 1.92 for 95 %).  For exact finite-sample coverage
calibrate :math:`c` per row by Neyman construction (e.g. a critical-value
regression) and pass it as a per-row tensor.

Algorithms
----------
All optimization runs in unit-box coordinates :math:`u \in [0, 1]^P` with
:math:`\theta = \mathrm{lo} + u\,(\mathrm{hi} - \mathrm{lo})`, so tolerances and
step caps are relative to the box and parameters with very different units
(e.g. effective temperature and metallicity) need no rescaling.

* **Multistart.** ``n_candidates`` scrambled Sobol points (shared by all rows)
  are scored in chunks.  Starts are chosen topographically (Torn & Viitanen,
  1994): a candidate whose score is at least that of its ``2P`` nearest
  candidates is a discrete local maximum, and the ``n_starts`` best such
  maxima per row (topped up with the best remaining candidates, plus an
  optional ``init``) are refined together.  Ranking all candidates by score
  instead would spend every start in the basin with the broadest high region
  and miss a narrow global mode whose candidates all score below it; ``2P``
  is the number of axis neighbours of a lattice point, the smallest
  neighbourhood that surrounds a candidate on every side.
* **Refinement.** Projected Newton ascent (Bertsekas, 1982) with exact
  per-row Hessians from autograd.  Coordinates on a bound whose gradient points
  out of the box are held fixed; on the free block the Hessian's eigenvalues
  are replaced by their absolute values with a floor of
  :math:`\sqrt{\epsilon}\,\max|\lambda|` (saddle-free Newton), which makes every
  direction an ascent direction; the step is capped at one box width and an
  Armijo backtracking line search along the projection arc guarantees
  monotone ascent.  Newton is affine invariant and converges quadratically,
  which for ``P <= ~10`` outweighs the ``P`` extra backward passes per
  iteration compared to first-order methods (that need hundreds of steps and a
  learning rate in parameter units).
* **Profile intervals.** For each side of :math:`\hat\theta_j` the profile is
  scanned on ``n_scan`` equally spaced points up to the bound (the inner
  maximization over :math:`\theta_{-j}` is warm-started from the previous
  point, i.e. numerical continuation), the *outermost* scan point above the
  threshold brackets the crossing, and the crossing is refined with Illinois
  regula falsi on the signed root
  :math:`r(t) = \sqrt{2(s_{\max} - p(t))}`.  For a Gaussian likelihood
  :math:`r` is exactly linear in :math:`t`, so the first interpolation is
  exact; in general the bracket is kept and convergence is superlinear.

References
----------
.. [1] Bertsekas, D. P. (1982). Projected Newton methods for optimization
   problems with simple constraints. *SIAM J. Control Optim.*, 20(2), 221-246.
.. [2] Torn, A. & Viitanen, S. (1994). Topographical global optimization using
   pre-sampled points. *J. Global Optim.*, 5, 267-276.
.. [3] Dauphin, Y. et al. (2014). Identifying and attacking the saddle point
   problem in high-dimensional non-convex optimization. *NeurIPS*.
.. [4] Venzon, D. J. & Moolgavkar, S. H. (1988). A method for computing
   profile-likelihood-based confidence intervals. *Appl. Statist.*, 37, 87-94.
.. [5] Dowell, M. & Jarratt, P. (1971). A modified regula falsi method for
   computing the root of an equation. *BIT*, 11, 168-174.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from typing import NamedTuple

import torch
from torch import Tensor, nn

__all__ = [
    "ProfileInterval",
    "ScoreMaximum",
    "flow_score",
    "likelihood_ratio_statistic",
    "maximize_score",
    "profile_interval",
    "ratio_classifier_score",
    "wilks_threshold",
]

#: A score function ``score(theta [M, P], x [M, ...]) -> [M]``.
ScoreFn = Callable[[Tensor, Tensor], Tensor]

#: Default maximum number of ``(theta, x)`` pairs per score call.
_DEFAULT_CHUNK = 32768
#: Armijo sufficient-increase constant (Nocedal & Wright's standard value).
_ARMIJO = 1e-4
#: Backtracking halvings; 2**-40 of a box width is below float64 resolution of u.
_MAX_BACKTRACKS = 40


class ScoreMaximum(NamedTuple):
    """Result of :func:`maximize_score` (unpacks as ``theta, score, at_bound``)."""

    theta: Tensor
    """``[N, P]`` maximizer :math:`\\hat\\theta` (the MLE for a likelihood score)."""
    score: Tensor
    """``[N]`` maximum score :math:`s_{\\max} = s(\\hat\\theta, x)`."""
    at_bound: Tensor
    """``[N, P]`` bool, ``True`` where :math:`\\hat\\theta_p` lies on a box bound."""


class ProfileInterval(NamedTuple):
    """Result of :func:`profile_interval` (unpacks as ``lower, upper, ...``)."""

    lower: Tensor
    """``[N]`` lower endpoint."""
    upper: Tensor
    """``[N]`` upper endpoint."""
    lower_at_bound: Tensor
    """``[N]`` bool, ``True`` where the interval reaches the lower box bound."""
    upper_at_bound: Tensor
    """``[N]`` bool, ``True`` where the interval reaches the upper box bound."""


# ─── adapters ────────────────────────────────────────────────────────────────


def ratio_classifier_score(classifier: nn.Module) -> ScoreFn:
    r"""Score function of a neural-ratio classifier.

    The logit of a classifier trained with
    :class:`~torchregress.losses.NeuralRatioLoss` estimates
    :math:`\log p(x \mid \theta) - \log p(x)`, i.e. a score in the sense of
    this module.  The classifier is evaluated on ``concat([theta, x])`` through
    :func:`~torchregress.losses.nre.neural_ratio_logits`.

    Args:
        classifier: The classifier module (e.g. from
            :func:`~torchregress.losses.nre.build_ratio_classifier`) or the
            :class:`~torchregress.losses.NeuralRatioLoss` holding it.  Put it
            in ``eval()`` mode first: dropout or batch statistics would make the
            score random or couple rows.

    Returns:
        ``score(theta [M, P], x [M, D]) -> [M]``.
    """
    from ..losses.nre import NeuralRatioLoss, neural_ratio_logits

    module = classifier.classifier if isinstance(classifier, NeuralRatioLoss) else classifier
    if not isinstance(module, nn.Module):
        raise TypeError(f"classifier must be a torch.nn.Module, got {type(module)}")

    def score(theta: Tensor, x: Tensor) -> Tensor:
        return neural_ratio_logits(module, theta, x)

    return score


def flow_score(flow: nn.Module) -> ScoreFn:
    r"""Score function of a conditional normalizing flow :math:`p(x \mid \theta)`.

    Args:
        flow: A :class:`~torchregress.losses.ContrastiveFlowLoss` or
            :class:`~torchregress.losses.NormalizingFlowLoss` (its
            ``log_prob(theta, x)`` is used), or a conditional flow such as a
            zuko flow, called as ``flow(theta).log_prob(x)``.  The flow must be
            conditioned on :math:`\theta` (context) and model :math:`x`.

    Returns:
        ``score(theta [M, P], x [M, D]) -> [M]``, a normalized
        :math:`\log p(x \mid \theta)`.
    """
    from ..losses.nflows import NormalizingFlowLoss

    if isinstance(flow, NormalizingFlowLoss):
        loss = flow

        def loss_score(theta: Tensor, x: Tensor) -> Tensor:
            return loss.log_prob(theta, x)

        return loss_score
    if not isinstance(flow, nn.Module):
        raise TypeError(f"flow must be a torch.nn.Module, got {type(flow)}")
    module = flow

    def module_score(theta: Tensor, x: Tensor) -> Tensor:
        target = x.unsqueeze(-1) if x.dim() == 1 else x
        return module(theta).log_prob(target)

    return module_score


# ─── validation helpers ─────────────────────────────────────────────────────


def _check_x(x: Tensor) -> Tensor:
    if not isinstance(x, Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x)}")
    if not x.is_floating_point():
        raise TypeError(f"x must be a floating-point tensor, got dtype {x.dtype}")
    if x.dim() < 1:
        raise ValueError("x must have a leading row dimension [N, ...]")
    return x.detach()


def _check_bounds(bounds: Tensor | Sequence[Sequence[float]], x: Tensor) -> tuple[Tensor, Tensor]:
    box = torch.as_tensor(bounds, dtype=x.dtype, device=x.device)
    if box.dim() != 2 or box.shape[1] != 2 or box.shape[0] < 1:
        raise ValueError(f"bounds must have shape [P, 2], got {tuple(box.shape)}")
    if not bool(torch.isfinite(box).all()):
        raise ValueError("bounds must be finite")
    lo, hi = box[:, 0].contiguous(), box[:, 1].contiguous()
    if not bool((hi > lo).all()):
        raise ValueError("bounds must satisfy lower < upper for every parameter")
    return lo, hi


def _check_rows(name: str, value: Tensor, shape: tuple[int, ...], x: Tensor) -> Tensor:
    tensor = torch.as_tensor(value).detach().to(dtype=x.dtype, device=x.device)
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite")
    return tensor


def _check_int(name: str, value: int, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def _check_chunk(chunk_size: int | None) -> int:
    if chunk_size is None:
        return _DEFAULT_CHUNK
    return _check_int("chunk_size", chunk_size, 1)


def _default_tol(tol: float | None, dtype: torch.dtype) -> float:
    if tol is None:
        return math.sqrt(torch.finfo(dtype).eps)
    tol = float(tol)
    if not tol > 0 or not math.isfinite(tol):
        raise ValueError(f"tol must be a positive finite float, got {tol}")
    return tol


def _chunks(total: int, size: int) -> Iterator[slice]:
    for start in range(0, total, size):
        yield slice(start, min(start + size, total))


def _eigh(matrix: Tensor) -> tuple[Tensor, Tensor]:
    """Batched symmetric eigendecomposition (host fallback where unsupported)."""
    if matrix.device.type == "mps":
        values, vectors = torch.linalg.eigh(matrix.cpu())
        return values.to(matrix.device), vectors.to(matrix.device)
    values, vectors = torch.linalg.eigh(matrix)
    return values, vectors


def _call_score(score: ScoreFn, theta: Tensor, x: Tensor) -> Tensor:
    """Evaluate ``score`` on matched rows and validate its ``[M]`` output."""
    out = score(theta, x)
    if not isinstance(out, Tensor):
        raise TypeError(f"score must return a torch.Tensor, got {type(out)}")
    if out.dim() == 2 and out.shape[1] == 1:
        out = out.squeeze(1)
    if tuple(out.shape) != (theta.shape[0],):
        raise ValueError(
            f"score must return one value per row with shape [{theta.shape[0]}], "
            f"got {tuple(out.shape)}"
        )
    return out.to(x.dtype)


# ─── objective in unit-box coordinates ──────────────────────────────────────


class _Objective:
    """Score as a function of unit-box coordinates, optionally with one fixed column.

    ``u`` holds the free coordinates ``[b, Q]``; with ``index`` set, ``fixed``
    ``[b]`` supplies coordinate ``index`` (``Q = P - 1``).  ``rows`` ``[b]``
    selects the matching rows of ``x``.
    """

    def __init__(
        self,
        score: ScoreFn,
        x: Tensor,
        lo: Tensor,
        hi: Tensor,
        index: int | None,
        chunk_size: int,
    ) -> None:
        self.score = score
        self.x = x
        self.lo = lo
        self.hi = hi
        self.index = index
        self.chunk_size = chunk_size

    def _call(self, u: Tensor, rows: Tensor, fixed: Tensor | None) -> Tensor:
        if self.index is not None:
            assert fixed is not None
            j = self.index
            u = torch.cat([u[:, :j], fixed.unsqueeze(1), u[:, j:]], dim=1)
        return _call_score(self.score, torch.lerp(self.lo, self.hi, u), self.x[rows])

    def values(self, u: Tensor, rows: Tensor, fixed: Tensor | None) -> Tensor:
        parts = []
        with torch.no_grad():
            for sl in _chunks(u.shape[0], self.chunk_size):
                parts.append(self._call(u[sl], rows[sl], None if fixed is None else fixed[sl]))
        return torch.cat(parts) if parts else u.new_empty(0)

    def derivatives(
        self, u: Tensor, rows: Tensor, fixed: Tensor | None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Value ``[b]``, gradient ``[b, Q]`` and Hessian ``[b, Q, Q]`` in ``u``."""
        q = u.shape[1]
        values, grads, hessians = [], [], []
        for sl in _chunks(u.shape[0], self.chunk_size):
            with torch.enable_grad():
                leaf = u[sl].detach().requires_grad_(True)
                f = self._call(leaf, rows[sl], None if fixed is None else fixed[sl])
                (g,) = torch.autograd.grad(f.sum(), leaf, create_graph=True, allow_unused=True)
                if g is None:
                    g = torch.zeros_like(leaf)
                columns = []
                for i in range(q):
                    h = None
                    if g.requires_grad:
                        # Rows are independent, so d(sum_m g[m, i])/du[m] is row m's
                        # Hessian row i.
                        (h,) = torch.autograd.grad(
                            g[:, i].sum(), leaf, retain_graph=i < q - 1, allow_unused=True
                        )
                    columns.append(torch.zeros_like(leaf) if h is None else h)
                hessian = torch.stack(columns, dim=1)
            values.append(f.detach())
            grads.append(g.detach())
            hessians.append(hessian.detach())
        if not values:
            return u.new_empty(0), u.new_empty(0, q), u.new_empty(0, q, q)
        return torch.cat(values), torch.cat(grads), torch.cat(hessians)


def _newton_direction(u: Tensor, g: Tensor, hessian: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Projected saddle-free Newton direction for maximization in ``[0, 1]^Q``.

    Returns ``(direction, decrement, free_gradient)``; ``decrement`` is
    ``g_F^T |H_F|^{-1} g_F`` (twice the quadratic model's predicted increase)
    computed before the step is capped to one box width.
    """
    eps = torch.finfo(u.dtype).eps
    active = ((u <= 0) & (g < 0)) | ((u >= 1) & (g > 0))
    keep = (~active).to(u.dtype)
    free_g = g * keep
    curvature = -hessian * keep.unsqueeze(2) * keep.unsqueeze(1)
    curvature = 0.5 * (curvature + curvature.transpose(1, 2))
    eigval, eigvec = _eigh(curvature)
    magnitude = eigval.abs()
    # Floor: sqrt(eps) relative conditioning, plus an absolute term so that a zero
    # Hessian (locally linear score) gives a finite, gradient-direction step.
    floor = math.sqrt(eps) * magnitude.amax(dim=1, keepdim=True) + eps * (
        1.0 + free_g.abs().amax(dim=1, keepdim=True)
    )
    coeff = (eigvec.transpose(1, 2) @ free_g.unsqueeze(2)).squeeze(2) / magnitude.clamp_min(floor)
    direction = (eigvec @ coeff.unsqueeze(2)).squeeze(2) * keep
    decrement = (free_g * direction).sum(dim=1)
    largest = direction.abs().amax(dim=1, keepdim=True)
    direction = direction / largest.clamp_min(1.0)
    return direction, decrement, free_g


def _line_search(
    objective: _Objective,
    u: Tensor,
    rows: Tensor,
    fixed: Tensor | None,
    f0: Tensor,
    g: Tensor,
    direction: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Armijo backtracking along the projection arc; returns ``(u, f, success)``."""
    u_new, f_new = u.clone(), f0.clone()
    success = torch.zeros(u.shape[0], dtype=torch.bool, device=u.device)
    pending = torch.arange(u.shape[0], device=u.device)
    alpha = 1.0
    for _ in range(_MAX_BACKTRACKS):
        if pending.numel() == 0:
            break
        base = u[pending]
        trial = (base + alpha * direction[pending]).clamp(0.0, 1.0)
        value = objective.values(trial, rows[pending], None if fixed is None else fixed[pending])
        gain = (g[pending] * (trial - base)).sum(dim=1)
        start = f0[pending]
        moved = (trial != base).any(dim=1)
        ok = moved & (value >= start) & (value - start >= _ARMIJO * gain) & torch.isfinite(value)
        accepted = pending[ok]
        u_new[accepted] = trial[ok]
        f_new[accepted] = value[ok]
        success[accepted] = True
        pending = pending[~ok]
        alpha *= 0.5
    return u_new, f_new, success


def _ascend(
    objective: _Objective,
    u: Tensor,
    rows: Tensor,
    fixed: Tensor | None,
    *,
    steps: int,
    tol: float,
) -> tuple[Tensor, Tensor]:
    """Batched projected Newton ascent from ``u``; returns ``(u, score)``.

    Each row stops when the predicted increase falls below ``tol`` nats (plus
    the rounding level of the score), after which one last full Newton step is
    taken, or when no ascent step can be found at working precision.
    """
    if u.shape[1] == 0 or steps == 0 or u.shape[0] == 0:
        return u, objective.values(u, rows, fixed)
    eps = torch.finfo(u.dtype).eps
    u = u.clone()
    f = torch.full((u.shape[0],), float("nan"), dtype=u.dtype, device=u.device)
    done = torch.zeros(u.shape[0], dtype=torch.bool, device=u.device)
    for _ in range(steps):
        idx = (~done).nonzero().squeeze(1)
        if idx.numel() == 0:
            break
        ui, ri = u[idx], rows[idx]
        fi = None if fixed is None else fixed[idx]
        f0, g, hessian = objective.derivatives(ui, ri, fi)
        f[idx] = f0
        direction, decrement, free_g = _newton_direction(ui, g, hessian)
        level = tol + 8.0 * eps * f0.abs()
        small = 0.5 * decrement <= level
        u_out, f_out = ui.clone(), f0.clone()
        stop = small.clone()
        if bool(small.any()):
            # Converged: the quadratic model is accurate, take the full Newton step
            # unless it loses more than rounding.
            s = small.nonzero().squeeze(1)
            trial = (ui[s] + direction[s]).clamp(0.0, 1.0)
            value = objective.values(trial, ri[s], None if fi is None else fi[s])
            take = value >= f0[s] - level[s]
            u_out[s[take]] = trial[take]
            f_out[s[take]] = value[take]
        big = (~small).nonzero().squeeze(1)
        if big.numel():
            fb = None if fi is None else fi[big]
            u_try, f_try, ok = _line_search(
                objective, ui[big], ri[big], fb, f0[big], g[big], direction[big]
            )
            failed = (~ok).nonzero().squeeze(1)
            if failed.numel():
                # Rare: projection broke the Newton direction's ascent property.
                # Retry with the box-scaled projected gradient (Bertsekas, 1982).
                grad_dir = free_g[big][failed]
                grad_dir = grad_dir / grad_dir.abs().amax(dim=1, keepdim=True).clamp_min(
                    torch.finfo(u.dtype).tiny
                )
                fbf = None if fb is None else fb[failed]
                u_g, f_g, ok_g = _line_search(
                    objective,
                    ui[big][failed],
                    ri[big][failed],
                    fbf,
                    f0[big][failed],
                    g[big][failed],
                    grad_dir,
                )
                u_try[failed] = u_g
                f_try[failed] = f_g
                ok[failed] = ok_g
            u_out[big] = u_try
            f_out[big] = f_try
            stop[big] = ~ok
        u[idx] = u_out
        f[idx] = f_out
        done[idx] = stop
    return u, f


# ─── public API ─────────────────────────────────────────────────────────────


def maximize_score(
    score: ScoreFn,
    x: Tensor,
    bounds: Tensor | Sequence[Sequence[float]],
    *,
    n_candidates: int = 256,
    n_starts: int = 4,
    steps: int = 50,
    init: Tensor | None = None,
    tol: float | None = None,
    chunk_size: int | None = None,
    generator: torch.Generator | None = None,
) -> ScoreMaximum:
    r"""Batched global maximization of a score over a box (the MLE).

    Computes :math:`\hat\theta_n = \arg\max_{\theta \in B} s(\theta, x_n)` for
    every row ``n`` simultaneously: ``n_candidates`` quasi-random candidates
    are scored, the best ``n_starts`` discrete local maxima among them (plus
    ``init``) are refined by projected Newton ascent, and the best refined
    point is returned.  There is
    no Python loop over rows; the loops run over chunks and Newton iterations.

    Args:
        score: ``score(theta [M, P], x [M, ...]) -> [M]``, equal to
            :math:`\log p(x \mid \theta)` up to a function of ``x``, twice
            differentiable in ``theta`` and evaluated row by row.
        x: ``[N, ...]`` observations (floating point; defines dtype and device).
        bounds: ``[P, 2]`` lower/upper bounds of the box :math:`B`.
        n_candidates: Scrambled Sobol candidates per row (powers of two keep the
            Sobol balance).  Must be large enough that every basin that can hold
            the global maximum contains a candidate.  Defaults to ``256``.
        n_starts: Best candidates per row refined by Newton ascent.  Defaults
            to ``4``.
        steps: Maximum Newton iterations.  Defaults to ``50``.
        init: Optional ``[N, P]`` extra start per row (e.g. a posterior head's
            point estimate), clamped into the box.
        tol: Convergence tolerance on the predicted score increase, in nats.
            Defaults to :math:`\sqrt{\epsilon}` of the dtype.
        chunk_size: Maximum ``(theta, x)`` pairs per score call.  Defaults to
            ``32768``.
        generator: Random generator for the Sobol scramble; fixes the result.
            ``None`` uses the global torch RNG.

    Returns:
        :class:`ScoreMaximum` ``(theta [N, P], score [N], at_bound [N, P])``,
        detached from the autograd graph.

    Raises:
        TypeError: If ``x`` is not a floating-point tensor or ``score`` returns
            a non-tensor.
        ValueError: On invalid shapes, bounds or integer arguments.

    Notes:
        The search is global only up to the candidate resolution: a mode
        narrower than the candidate spacing (about
        ``n_candidates ** (-1 / P)`` box widths) can be missed.  Rows whose
        score is not finite at every start return the best finite start (or
        ``NaN`` score if none).
    """
    x = _check_x(x)
    lo, hi = _check_bounds(bounds, x)
    n_rows, n_params = x.shape[0], lo.numel()
    n_candidates = _check_int("n_candidates", n_candidates, 1)
    n_starts = _check_int("n_starts", n_starts, 1)
    if n_starts > n_candidates:
        raise ValueError(f"n_starts ({n_starts}) must not exceed n_candidates ({n_candidates})")
    steps = _check_int("steps", steps, 0)
    chunk = _check_chunk(chunk_size)
    tol = _default_tol(tol, x.dtype)
    dtype, device = x.dtype, x.device
    objective = _Objective(score, x, lo, hi, None, chunk)

    gen_device = generator.device if generator is not None else torch.device("cpu")
    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator, device=gen_device).item())
    sobol = torch.quasirandom.SobolEngine(n_params, scramble=True, seed=seed)
    base = sobol.draw(n_candidates, dtype=torch.float64).to(dtype=dtype, device=device)
    # Topographical start selection: k nearest candidates of each candidate.
    k = min(2 * n_params, n_candidates - 1)
    neighbours: Tensor | None = None
    if k > 0:
        dist = torch.cdist(base, base)
        dist.fill_diagonal_(math.inf)
        neighbours = dist.topk(k, dim=1, largest=False).indices  # [C, k]

    starts_u = torch.empty(n_rows, n_starts, n_params, dtype=dtype, device=device)
    block = max(1, chunk // n_candidates)
    for sl in _chunks(n_rows, block):
        rows = torch.arange(sl.start, sl.stop, device=device)
        value = objective.values(
            base.repeat(rows.numel(), 1), rows.repeat_interleave(n_candidates), None
        ).view(-1, n_candidates)
        value = torch.nan_to_num(value, nan=-math.inf)
        order = value.argsort(dim=1, descending=True, stable=True)
        if neighbours is not None:
            local_max = (value >= value[:, neighbours].amax(dim=2)) & torch.isfinite(value)
            # Stable re-sort keeps the score order within both groups.
            first = local_max.gather(1, order).to(torch.int8)
            order = order.gather(1, first.argsort(dim=1, descending=True, stable=True))
        starts_u[sl] = base[order[:, :n_starts]]

    if init is not None:
        init_t = _check_rows("init", init, (n_rows, n_params), x)
        init_u = ((init_t - lo) / (hi - lo)).clamp(0.0, 1.0)
        starts_u = torch.cat([starts_u, init_u.unsqueeze(1)], dim=1)
    n_total = starts_u.shape[1]

    rows_all = torch.arange(n_rows, device=device).repeat_interleave(n_total)
    u, f = _ascend(objective, starts_u.reshape(-1, n_params), rows_all, None, steps=steps, tol=tol)
    u, f = u.view(n_rows, n_total, n_params), f.view(n_rows, n_total)
    best = torch.nan_to_num(f, nan=-math.inf).argmax(dim=1)
    u_best = u[torch.arange(n_rows, device=device), best]
    f_best = f[torch.arange(n_rows, device=device), best]
    theta = torch.lerp(lo, hi, u_best)
    at_bound = (u_best <= 0) | (u_best >= 1)
    return ScoreMaximum(theta, f_best, at_bound)


def likelihood_ratio_statistic(
    score: ScoreFn,
    x: Tensor,
    theta0: Tensor,
    s_max: Tensor,
    *,
    clamp: bool = True,
    chunk_size: int | None = None,
) -> Tensor:
    r"""Likelihood-ratio statistic :math:`\lambda = s_{\max} - s(\theta_0, x)`.

    This is half the usual :math:`-2\log\Lambda`; under Wilks' theorem a
    hypothesis is rejected at level :math:`1-c` when
    ``lambda > wilks_threshold(c, df)``.

    Args:
        score: Score function (see :func:`maximize_score`).
        x: ``[N, ...]`` observations.
        theta0: Hypotheses, either ``[N, P]`` (one per row, result ``[N]``) or
            a bank ``[N, K, P]`` / ``[1, K, P]`` (result ``[N, K]``).  Pass a
            shared bank ``[K, P]`` as ``bank[None]``.
        s_max: ``[N]`` maximum scores from :func:`maximize_score`.
        clamp: Clamp at 0 (default).  A negative raw value means ``theta0``
            scores above ``s_max``: either ``s_max`` is not the global maximum
            or ``theta0`` lies outside the box it was maximized over.  Such a
            hypothesis is at least as likely as the reported MLE and is never
            rejected, which ``lambda = 0`` encodes; pass ``clamp=False`` to
            inspect the raw differences.
        chunk_size: Maximum pairs per score call.  Defaults to ``32768``.

    Returns:
        ``[N]`` or ``[N, K]`` statistic, detached.
    """
    x = _check_x(x)
    n_rows = x.shape[0]
    chunk = _check_chunk(chunk_size)
    s_max = _check_rows("s_max", s_max, (n_rows,), x)
    theta0 = torch.as_tensor(theta0).detach().to(dtype=x.dtype, device=x.device)
    rows = torch.arange(n_rows, device=x.device)

    def values(theta: Tensor, theta_rows: Tensor) -> Tensor:
        with torch.no_grad():
            parts = [
                _call_score(score, theta[sl], x[theta_rows[sl]])
                for sl in _chunks(theta.shape[0], chunk)
            ]
        return torch.cat(parts) if parts else theta.new_empty(0)

    if theta0.dim() == 2:
        if theta0.shape[0] != n_rows:
            raise ValueError(f"theta0 [N, P] must have N={n_rows} rows, got {theta0.shape[0]}")
        stat = s_max - values(theta0, rows)
    elif theta0.dim() == 3:
        if theta0.shape[0] not in (1, n_rows):
            raise ValueError(
                f"theta0 bank must have shape [N or 1, K, P], got {tuple(theta0.shape)}"
            )
        n_hyp = theta0.shape[1]
        flat = theta0.expand(n_rows, -1, -1).reshape(n_rows * n_hyp, -1)
        value = values(flat, rows.repeat_interleave(n_hyp))
        stat = s_max.unsqueeze(1) - value.view(n_rows, n_hyp)
    else:
        raise ValueError(f"theta0 must be [N, P] or [N, K, P], got {tuple(theta0.shape)}")
    return stat.clamp_min(0.0) if clamp else stat


def profile_interval(
    score: ScoreFn,
    x: Tensor,
    theta_hat: Tensor,
    s_max: Tensor,
    index: int,
    threshold: float | Tensor,
    bounds: Tensor | Sequence[Sequence[float]],
    *,
    n_scan: int = 8,
    steps: int = 20,
    max_iter: int = 50,
    tol: float | None = None,
    chunk_size: int | None = None,
) -> ProfileInterval:
    r"""Profile-likelihood interval for one parameter, batched over rows.

    Returns the outermost crossings of

    .. math::

        p_j(t) = \max_{\theta:\ \theta_j = t} s(\theta, x)
        \;=\; s_{\max} - c

    on each side of :math:`\hat\theta_j`, i.e. the smallest interval holding
    every point of the scanned profile above the threshold.  If the
    likelihood-ratio set is disconnected (a secondary mode above threshold) the
    interval spans the gap: this is the conservative choice that keeps the
    coverage of the set; inspect the profile itself to see the gap.

    Args:
        score: Score function (see :func:`maximize_score`).
        x: ``[N, ...]`` observations.
        theta_hat: ``[N, P]`` maximizer from :func:`maximize_score`.
        s_max: ``[N]`` maximum score from :func:`maximize_score`.
        index: Parameter :math:`j` to profile (negative indices allowed).
        threshold: :math:`c \ge 0`, scalar or ``[N]`` per row, e.g.
            ``wilks_threshold(0.95)`` (1.92) or a Neyman critical value.
        bounds: ``[P, 2]`` box (the one used for :func:`maximize_score`).
        n_scan: Profile scan points per side, equally spaced from
            :math:`\hat\theta_j` to the bound.  Disconnected regions narrower
            than the spacing can be missed.  Defaults to ``8``.
        steps: Maximum Newton iterations of each warm-started inner
            maximization over :math:`\theta_{-j}`.  Defaults to ``20``.
        max_iter: Maximum root-refinement iterations per endpoint.  Defaults
            to ``50``.
        tol: Root tolerance on the signed root
            :math:`\sqrt{2(s_{\max}-p_j)}`, i.e. roughly in standard
            deviations of :math:`\theta_j`.  Defaults to
            :math:`\sqrt{\epsilon}` of the dtype.
        chunk_size: Maximum pairs per score call.  Defaults to ``32768``.

    Returns:
        :class:`ProfileInterval` ``(lower [N], upper [N], lower_at_bound [N],
        upper_at_bound [N])``.  An endpoint whose side of the profile stays
        above the threshold up to the box edge is set to that bound and
        flagged.

    Notes:
        * For ``P = 1`` the profile is the score itself.
        * The inner maximization follows the branch of
          :math:`\arg\max_{\theta_{-j}}` continued from :math:`\hat\theta`; if a
          different branch takes over far from :math:`\hat\theta`, the profile
          is underestimated there.
        * Points whose profile exceeds ``s_max`` (``theta_hat`` was not the
          global maximum) count as inside.
    """
    x = _check_x(x)
    lo, hi = _check_bounds(bounds, x)
    n_rows, n_params = x.shape[0], lo.numel()
    if isinstance(index, bool) or not isinstance(index, int) or not -n_params <= index < n_params:
        raise ValueError(
            f"index must be an integer in [{-n_params}, {n_params - 1}], got {index!r}"
        )
    j = index % n_params
    theta_hat = _check_rows("theta_hat", theta_hat, (n_rows, n_params), x)
    s_max = _check_rows("s_max", s_max, (n_rows,), x)
    thr = torch.as_tensor(threshold).detach().to(dtype=x.dtype, device=x.device)
    if thr.dim() > 1 or (thr.dim() == 1 and thr.shape[0] != n_rows):
        raise ValueError(f"threshold must be a scalar or have shape [{n_rows}]")
    thr = thr.expand(n_rows)
    if not bool(torch.isfinite(thr).all()) or bool((thr < 0).any()):
        raise ValueError("threshold must be finite and non-negative")
    n_scan = _check_int("n_scan", n_scan, 1)
    steps = _check_int("steps", steps, 0)
    max_iter = _check_int("max_iter", max_iter, 1)
    chunk = _check_chunk(chunk_size)
    root_tol = _default_tol(tol, x.dtype)
    inner_tol = _default_tol(None, x.dtype)
    device, dtype = x.device, x.dtype
    eps = torch.finfo(dtype).eps

    objective = _Objective(score, x, lo, hi, j, chunk)
    u_hat = ((theta_hat - lo) / (hi - lo)).clamp(0.0, 1.0)
    free = [c for c in range(n_params) if c != j]
    # Both sides stacked: entries [0, N) go down, [N, 2N) go up.
    sign = torch.cat(
        [
            -torch.ones(n_rows, dtype=dtype, device=device),
            torch.ones(n_rows, dtype=dtype, device=device),
        ]
    )
    rows = torch.arange(n_rows, device=device).repeat(2)
    uj = u_hat[:, j].repeat(2)
    reach = torch.where(sign < 0, uj, 1.0 - uj)
    s_top = s_max.repeat(2)
    r_star = torch.sqrt(2.0 * thr).repeat(2)
    v_hat = u_hat[:, free].repeat(2, 1)

    def signed_root_gap(p: Tensor, sel: Tensor) -> Tensor:
        gap = torch.sqrt(2.0 * (s_top[sel] - p).clamp_min(0.0)) - r_star[sel]
        return torch.nan_to_num(gap, nan=math.inf)

    def profile(sel: Tensor, tau: Tensor, v0: Tensor) -> tuple[Tensor, Tensor]:
        fixed = (uj[sel] + sign[sel] * tau).clamp(0.0, 1.0)
        return _ascend(objective, v0, rows[sel], fixed, steps=steps, tol=inner_tol)

    every = torch.arange(2 * n_rows, device=device)
    # Scan: track the outermost inside point and the scan point right after it.
    tau_in, h_in, v_in = torch.zeros_like(reach), -r_star, v_hat.clone()
    tau_out, h_out, v_out = reach.clone(), torch.full_like(reach, math.inf), v_hat.clone()
    previous_inside = torch.ones_like(reach, dtype=torch.bool)
    v_prev = v_hat
    for k in range(1, n_scan + 1):
        tau_k = reach * (k / n_scan)
        v_k, p_k = profile(every, tau_k, v_prev)
        h_k = signed_root_gap(p_k, every)
        inside = h_k <= 0
        tau_in = torch.where(inside, tau_k, tau_in)
        h_in = torch.where(inside, h_k, h_in)
        v_in = torch.where(inside.unsqueeze(1), v_k, v_in)
        leaving = ~inside & previous_inside
        tau_out = torch.where(leaving, tau_k, tau_out)
        h_out = torch.where(leaving, h_k, h_out)
        v_out = torch.where(leaving.unsqueeze(1), v_k, v_out)
        previous_inside = inside
        v_prev = v_k
    at_bound = previous_inside
    result = torch.where(at_bound, reach, tau_in)

    # Illinois regula falsi on the signed-root gap inside [tau_in, tau_out].
    a, fa, va = tau_in, h_in, v_in
    b, fb, vb = tau_out, h_out, v_out
    last = torch.zeros_like(reach, dtype=torch.long)  # 1: replaced a, 2: replaced b
    active = ~at_bound
    xtol = 4.0 * eps
    for _ in range(max_iter):
        sel = active.nonzero().squeeze(1)
        if sel.numel() == 0:
            break
        a_s, b_s, fa_s, fb_s = a[sel], b[sel], fa[sel], fb[sel]
        t = a_s - fa_s * (b_s - a_s) / (fb_s - fa_s)
        bisect = ~torch.isfinite(t) | (t <= a_s) | (t >= b_s)
        t = torch.where(bisect, 0.5 * (a_s + b_s), t)
        near_a = (t - a_s).abs() <= (b_s - t).abs()
        v0 = torch.where(near_a.unsqueeze(1), va[sel], vb[sel])
        v_t, p_t = profile(sel, t, v0)
        h_t = signed_root_gap(p_t, sel)
        inside = h_t <= 0
        # Illinois: halve the retained endpoint's value when the same side is
        # replaced twice in a row.
        fb[sel] = torch.where(inside & (last[sel] == 1), 0.5 * fb_s, fb_s)
        fa[sel] = torch.where(~inside & (last[sel] == 2), 0.5 * fa_s, fa_s)
        a[sel] = torch.where(inside, t, a_s)
        fa[sel] = torch.where(inside, h_t, fa[sel])
        va[sel] = torch.where(inside.unsqueeze(1), v_t, va[sel])
        b[sel] = torch.where(inside, b_s, t)
        fb[sel] = torch.where(inside, fb[sel], h_t)
        vb[sel] = torch.where(inside.unsqueeze(1), vb[sel], v_t)
        last[sel] = torch.where(inside, 1, 2)
        converged = (h_t.abs() <= root_tol) | (b[sel] - a[sel] <= xtol)
        result[sel] = torch.where(converged, t, 0.5 * (a[sel] + b[sel]))
        active[sel] = ~converged

    endpoint = torch.lerp(lo[j], hi[j], (uj + sign * result).clamp(0.0, 1.0))
    lower = torch.where(at_bound[:n_rows], lo[j], endpoint[:n_rows])
    upper = torch.where(at_bound[n_rows:], hi[j], endpoint[n_rows:])
    return ProfileInterval(lower, upper, at_bound[:n_rows].clone(), at_bound[n_rows:].clone())


def wilks_threshold(confidence: float, df: int = 1) -> float:
    r"""Wilks threshold :math:`c = \tfrac12 \chi^2_{\mathrm{df}}(\text{confidence})`.

    The likelihood-ratio statistic :math:`\lambda` of this module is half the
    usual :math:`-2\log\Lambda`, so the asymptotic level-``confidence`` region
    is :math:`\{\theta_0 : \lambda(\theta_0) \le c\}`.  Use ``df=1`` for a
    profile interval of one parameter (0.5 for 68.27 %, 1.92 for 95 %) and
    ``df=P`` for a joint region of ``P`` parameters.  Computed exactly as
    :math:`c = P^{-1}(\mathrm{df}/2, \text{confidence})`, the inverse
    regularized lower incomplete gamma function (``scipy.special.gammaincinv``).

    Args:
        confidence: Confidence level in ``(0, 1)``.
        df: Degrees of freedom (positive integer).  Defaults to ``1``.

    Returns:
        The threshold as a Python float.
    """
    from scipy.special import gammaincinv

    level = float(confidence)
    if not 0.0 < level < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    df = _check_int("df", df, 1)
    return float(gammaincinv(0.5 * df, level))
