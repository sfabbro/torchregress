"""Batched normal-equation solves for many small (non-negative) least-squares problems."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

#: Default KKT tolerance of the non-negative solve. A constrained variable at its
#: bound is freed only while its multiplier ``μ_j`` on the Jacobi-scaled equations
#: exceeds ``kkt_tol · (1 + |g_s|_∞)``: a mixed absolute/relative threshold, in
#: units of σ when the weights are true precisions. Freeing a variable with
#: multiplier ``μ_j`` lowers the objective by ``μ_j² / (2 S_j) ≥ μ_j² / 2``, where
#: ``S_j ≤ 1`` is the Schur complement of its scaled diagonal on the free block,
#: so collinear columns (small ``S_j``) amplify the forgone decrease.
NNLS_KKT_TOL = 1.0e-10


@dataclass(frozen=True)
class NormalEquationsSolution:
    """Per-row solution of ``min ½ xᵀNx − gᵀx`` (optionally with ``x_i ≥ 0``).

    Attributes:
        solution: ``[..., n]`` minimiser; ``NaN`` on ``singular`` rows.
        standard_errors: ``[..., n]`` ``√((N_FF)⁻¹)_ii`` on the free set ``F``
            (the inverse of the normal matrix restricted to it; of its ridged
            scaled form on ``ridge_used`` rows), ``1/√N_ii`` for a
            variable held at its bound, ``inf`` for an empty column (zero
            diagonal: no information) and ``NaN`` on ``singular`` rows.
        free: ``[..., n]`` variables neither at a bound nor empty.
        dead: ``[..., n]`` empty columns (``N_ii ≤ 0``), pinned at 0 (``False`` on
            rows with non-finite input).
        ridge_used: ``[...]`` rows whose (free-block) factorisation failed or was
            numerically rank-deficient and took the ridge: their solution and
            standard errors are regularised.
        singular: ``[...]`` rows singular even with the ridge, or with non-finite
            input (any ``NaN``/``±inf`` in their ``N`` or ``g``); not solved, and
            ``free`` is ``False`` on them.
    """

    solution: Tensor
    standard_errors: Tensor
    free: Tensor
    dead: Tensor
    ridge_used: Tensor
    singular: Tensor


def solve_normal_equations(
    normal: Tensor,
    rhs: Tensor,
    *,
    nonnegative: Tensor | None = None,
    start: Tensor | None = None,
    ridge: float = 1.0e-8,
    kkt_tol: float = NNLS_KKT_TOL,
    pivot_tol: float | None = None,
) -> NormalEquationsSolution:
    r"""Solve many small least-squares problems given their normal equations.

    Each row solves

    .. math::

        \min_x \tfrac12 x^\mathsf{T} N x - g^\mathsf{T} x
        \quad\text{s.t.}\quad x_i \ge 0 \ (i \in \mathcal C),

    for a positive semi-definite ``N = XᵀWX (+ prior precision)`` and
    ``g = XᵀWy (+ prior term)``; ``𝒞`` is empty unless ``nonnegative`` is given.
    The system is Jacobi-scaled (unit diagonal) before factorisation, so the
    ``ridge`` is relative. It is added to the scaled diagonal (and the row
    flagged in ``ridge_used``) where the Cholesky factorisation fails or is
    numerically rank-deficient: its smallest pivot ``L_kk²`` is at most
    ``pivot_tol`` (default: ``ridge``) times its largest. On ridged rows the
    scaled condition number is then about ``n/ridge`` at most, and the solution and standard errors are
    those of ``N_s + ridge·I``: regularised, so a non-identified direction gets
    a finite (ridge-limited) error rather than ``inf``. The pivot test is cheap
    but not rank-revealing: an unridged factorisation can still be badly
    conditioned. Empty columns (``N_ii ≤ 0``) are pinned at 0.

    Without ``nonnegative`` the solve is one batched Cholesky with no host
    synchronisation. With it, a batched Lawson--Hanson active-set method runs on
    the scaled normal equations: each outer step frees the constrained variable
    with the largest scaled gradient (KKT multiplier) above
    ``kkt_tol · (1 + |g_s|_∞)``; each inner step solves the free block and, when a
    constrained variable would turn non-positive, moves to the first bound on
    the segment and pins it. ``start`` warm-starts the free set (e.g. the
    previous solution of a nearby problem). Rows terminate independently, and
    every step works on the rows still in play only: about one host
    synchronisation per active-set step, at ``O(b·n³)`` for the ``b`` rows still
    pending, plus one final batched factorisation for the standard errors.

    Args:
        normal: ``[..., n, n]`` symmetric positive semi-definite matrices.
        rhs: ``[..., n]`` right-hand sides ``g``.
        nonnegative: optional boolean ``[n]`` (or broadcastable to ``rhs``) mask of
            the variables constrained to ``x_i ≥ 0``.
        start: optional ``[..., n]`` warm start of the non-negative solve (clipped
            to the bounds; non-finite entries start at 0). Ignored without
            ``nonnegative``.
        ridge: positive relative ridge added to the scaled diagonal where the
            factorisation fails or its pivots span more than ``1/pivot_tol``.
        pivot_tol: smallest-to-largest Cholesky pivot ratio at or below which a
            successful factorisation counts as rank-deficient and takes the
            ridge. ``None`` (default) uses ``ridge``; ``0`` takes the ridge only
            where the factorisation fails.
        kkt_tol: positive KKT tolerance of the non-negative solve (threshold
            ``kkt_tol · (1 + |g_s|_∞)`` on the scaled multipliers).

    Returns:
        :class:`NormalEquationsSolution`.
    """
    _validate(normal, rhs, ridge, kkt_tol)
    if pivot_tol is None:
        pivot_tol = ridge
    elif not (math.isfinite(pivot_tol) and pivot_tol >= 0.0):
        raise ValueError("pivot_tol must be finite and >= 0, or None")
    batch_shape = rhs.shape[:-1]
    n = rhs.shape[-1]
    normal_2d = normal.reshape(-1, n, n)
    rhs_2d = rhs.reshape(-1, n)
    # Rows with non-finite input are not solved. They are swapped for a trivial
    # system first so that they cannot poison batch-wide quantities (the KKT
    # tolerance, the multipliers, the active-set loop) of the other rows.
    bad = ~torch.isfinite(normal_2d).all(-1).all(-1) | ~torch.isfinite(rhs_2d).all(-1)
    eye = torch.eye(n, dtype=rhs.dtype, device=rhs.device)
    normal_2d = torch.where(bad[:, None, None], eye, normal_2d)
    rhs_2d = torch.where(bad[:, None], torch.zeros_like(rhs_2d), rhs_2d)
    scaled, gs, scale, dead = _jacobi_scale(normal_2d, rhs_2d)
    if nonnegative is None:
        x_s, se_s, free, ridged, singular = _solve_unconstrained(scaled, gs, dead, ridge, pivot_tol)
    else:
        if nonnegative.dtype != torch.bool:
            raise TypeError("nonnegative must be a boolean mask")
        constrained = torch.broadcast_to(nonnegative.to(rhs.device), rhs.shape).reshape(-1, n)
        x0 = torch.zeros_like(rhs_2d)
        if start is not None:
            if start.shape != rhs.shape:
                raise ValueError(f"start shape {tuple(start.shape)} must be {tuple(rhs.shape)}")
            warm = start.to(rhs_2d).reshape(-1, n)
            x0 = torch.where(dead | ~torch.isfinite(warm), x0, warm * scale)
            x0 = torch.where(constrained & ~(x0 > 0.0), torch.zeros_like(x0), x0)
        x_s, se_s, free, ridged, singular = _solve_nonnegative(
            scaled, gs, dead, constrained, x0, ridge, pivot_tol, kkt_tol
        )
    solution = x_s / scale
    standard_errors = torch.where(dead, torch.full_like(se_s, math.inf), se_s / scale)
    singular = singular | bad | ~torch.isfinite(solution).all(-1)
    nan = torch.full_like(solution, math.nan)
    solution = torch.where(singular[:, None], nan, solution)
    standard_errors = torch.where(singular[:, None], nan, standard_errors)
    return NormalEquationsSolution(
        solution=solution.reshape(rhs.shape),
        standard_errors=standard_errors.reshape(rhs.shape),
        free=(free & ~singular[:, None]).reshape(rhs.shape),
        dead=dead.reshape(rhs.shape),
        ridge_used=ridged.reshape(batch_shape),
        singular=singular.reshape(batch_shape),
    )


def _validate(normal: Tensor, rhs: Tensor, ridge: float, kkt_tol: float) -> None:
    if not (normal.is_floating_point() and rhs.is_floating_point()):
        raise TypeError("normal and rhs must be floating point")
    if normal.dtype != rhs.dtype or normal.device != rhs.device:
        raise ValueError("normal and rhs must share dtype and device")
    if rhs.ndim < 1 or rhs.shape[-1] == 0:
        raise ValueError("rhs must have shape [..., n] with n >= 1")
    if normal.shape != rhs.shape + rhs.shape[-1:]:
        raise ValueError(
            f"normal shape {tuple(normal.shape)} must be rhs shape + (n,): "
            f"{tuple(rhs.shape + rhs.shape[-1:])}"
        )
    if not (math.isfinite(ridge) and ridge > 0.0):
        raise ValueError("ridge must be finite and positive")
    if not (math.isfinite(kkt_tol) and kkt_tol > 0.0):
        raise ValueError("kkt_tol must be finite and positive")


def _jacobi_scale(normal: Tensor, rhs: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Unit-diagonal scaling; empty columns decoupled with a unit diagonal."""
    diag = torch.diagonal(normal, dim1=-2, dim2=-1)
    dead = diag <= 0.0
    scale = torch.where(dead, torch.ones_like(diag), diag.sqrt())
    scaled = normal / (scale.unsqueeze(-1) * scale.unsqueeze(-2))
    pair = dead.unsqueeze(-1) | dead.unsqueeze(-2)
    scaled = torch.where(pair, torch.zeros_like(scaled), scaled) + torch.diag_embed(
        dead.to(normal.dtype)
    )
    gs = torch.where(dead, torch.zeros_like(rhs), rhs / scale)
    return scaled, gs, scale, dead


def _rank_deficient(chol: Tensor, free: Tensor | None, pivot_tol: float) -> Tensor:
    """Rows whose Cholesky pivots ``d_k = L_kk²`` span more than ``1/pivot_tol``.

    A factorisation of an (exactly) singular matrix often succeeds with a pivot
    at round-off level; its inverse is then meaningless. The pivot ratio is a
    cheap necessary test (``d_min ≥ λ_min``), not a rank-revealing one.
    """
    if pivot_tol == 0.0:
        return torch.zeros(chol.shape[:-2], dtype=torch.bool, device=chol.device)
    pivots = torch.diagonal(chol, dim1=-2, dim2=-1).square()
    if free is None:
        low, high = pivots.amin(-1), pivots.amax(-1)
    else:
        low = torch.where(free, pivots, torch.full_like(pivots, math.inf)).amin(-1)
        high = torch.where(free, pivots, torch.zeros_like(pivots)).amax(-1)
    return low <= pivot_tol * high


def _solve_unconstrained(
    scaled: Tensor, gs: Tensor, dead: Tensor, ridge: float, pivot_tol: float
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    n = scaled.shape[-1]
    eye = torch.eye(n, dtype=scaled.dtype, device=scaled.device)
    chol, info = torch.linalg.cholesky_ex(scaled)
    ridged = (info != 0) | _rank_deficient(chol, None, pivot_tol)
    # Both factorisations are computed unconditionally: no host synchronisation.
    chol_r, info_r = torch.linalg.cholesky_ex(scaled + ridge * eye)
    singular = ridged & (info_r != 0)
    chol_r = torch.where(singular[:, None, None], eye, chol_r)
    chol = torch.where(ridged[:, None, None], chol_r, chol)
    x = torch.cholesky_solve(gs.unsqueeze(-1), chol).squeeze(-1)
    inv_diag = torch.diagonal(torch.cholesky_inverse(chol), dim1=-2, dim2=-1)
    return x, inv_diag.clamp_min(0.0).sqrt(), ~dead, ridged, singular


def _free_block_cholesky(
    scaled: Tensor, free: Tensor, ridge: float, pivot_tol: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Cholesky factor of ``scaled`` on the free block (identity elsewhere)."""
    n = scaled.shape[-1]
    eye = torch.eye(n, dtype=scaled.dtype, device=scaled.device)
    pair = free.unsqueeze(-1) & free.unsqueeze(-2)
    system = torch.where(pair, scaled, torch.zeros_like(scaled)) + torch.diag_embed(
        (~free).to(scaled.dtype)
    )
    chol, info = torch.linalg.cholesky_ex(system)
    ridged = (info != 0) | _rank_deficient(chol, free, pivot_tol)
    # Both factorisations are computed unconditionally: no host synchronisation.
    chol_r, info_r = torch.linalg.cholesky_ex(
        system + ridge * torch.diag_embed(free.to(scaled.dtype))
    )
    singular = ridged & (info_r != 0)
    chol_r = torch.where(singular[:, None, None], eye, chol_r)
    chol = torch.where(ridged[:, None, None], chol_r, chol)
    return chol, ridged, singular


def _solve_nonnegative(
    scaled: Tensor,
    gs: Tensor,
    dead: Tensor,
    constrained: Tensor,
    x: Tensor,
    ridge: float,
    pivot_tol: float,
    kkt_tol: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Batched Lawson--Hanson on Jacobi-scaled normal equations (feasible ``x``).

    Every step works on the rows still in play only (``going`` for the outer
    KKT step, ``pending`` for the inner feasibility steps), gathered by index:
    each step costs ``O(b·n³)`` on those rows and one host synchronisation (the
    index of the rows that carry on), whatever the batch size.
    """
    b, n = gs.shape
    free = ~dead & (~constrained | (x > 0.0))
    tol = kkt_tol * (1.0 + gs.abs().amax(-1))
    ridged = torch.zeros(b, dtype=torch.bool, device=gs.device)
    singular = torch.zeros_like(ridged)
    going = torch.arange(b, device=gs.device)
    for _ in range(3 * n + 3):
        pending = going
        for _ in range(n + 1):
            if pending.numel() == 0:
                break
            free_p, gs_p, x_p = free[pending], gs[pending], x[pending]
            bounded = free_p & constrained[pending]
            chol, used, sing = _free_block_cholesky(scaled[pending], free_p, ridge, pivot_tol)
            ridged[pending] = ridged[pending] | used
            singular[pending] = singular[pending] | sing
            z = torch.cholesky_solve(
                torch.where(free_p, gs_p, torch.zeros_like(gs_p)).unsqueeze(-1), chol
            ).squeeze(-1)
            z = torch.where(free_p, z, torch.zeros_like(z))
            infeasible = ~sing.unsqueeze(-1) & bounded & ~(z > 0.0)
            stuck = infeasible.any(-1)
            accept = ~sing & ~stuck
            x_p = torch.where(accept.unsqueeze(-1), z, x_p)
            # Move from x (feasible) towards z up to the first bound on the way.
            ratio = torch.where(infeasible, x_p / (x_p - z), torch.full_like(x_p, math.inf))
            alpha = ratio.amin(-1, keepdim=True).clamp(0.0, 1.0)
            moved = x_p + alpha * (z - x_p)
            hit = stuck.unsqueeze(-1) & (
                (infeasible & (ratio <= alpha)) | (bounded & ~(moved > 0.0))
            )
            x_p = torch.where(stuck.unsqueeze(-1), moved, x_p)
            x[pending] = torch.where(hit, torch.zeros_like(x_p), x_p)
            free[pending] = free_p & ~hit
            pending = pending[stuck]  # the one host synchronisation of the step
        free_g, x_g = free[going], x[going]
        multiplier = gs[going] - (scaled[going] @ x_g.unsqueeze(-1)).squeeze(-1)
        candidate = (
            ~singular[going].unsqueeze(-1)
            & constrained[going]
            & ~free_g
            & ~dead[going]
            & (multiplier > tol[going].unsqueeze(-1))
        )
        best = torch.where(candidate, multiplier, torch.full_like(x_g, -math.inf)).argmax(-1)
        more = candidate.any(-1)
        pick = torch.nn.functional.one_hot(best, n).to(torch.bool) & more.unsqueeze(-1)
        free[going] = free_g | pick
        going = going[more]  # the one host synchronisation of the step
        if going.numel() == 0:
            break
    # One more factorisation of the final free sets gives the standard errors.
    chol, _, _ = _free_block_cholesky(scaled, free, ridge, pivot_tol)
    inv_diag = torch.diagonal(torch.cholesky_inverse(chol), dim1=-2, dim2=-1)
    se = torch.where(free, inv_diag.clamp_min(0.0).sqrt(), torch.ones_like(x))
    return x, se, free, ridged, singular
