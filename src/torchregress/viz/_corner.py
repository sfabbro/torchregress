"""Numerical helpers for corner plots: draws, smoothed densities, levels, labels.

Nothing here draws.  The smoothing is a binned Gaussian kernel density estimate
evaluated by FFT convolution on a fixed grid (``O(n + G^2 log G)`` for ``n``
draws on a ``G x G`` grid), so corner plots stay fast and stable from ``10^3``
to ``10^6`` draws.  Contour levels are calibrated on the draws themselves: the
level of mass ``m`` is the ``(1 - m)`` quantile of the smoothed density
evaluated at the draws, so the contour encloses a fraction ``m`` of the draws
(the sample estimate of a highest-density region of mass ``m``).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve

from torchregress.joint.predictive import JointPredictive
from torchregress.joint.regions import JointRegion, MahalanobisScore, MaxResidualScore

# Paul Tol "bright" qualitative palette (colour-blind safe, distinct in greyscale
# when paired with line styles).
TOL_BRIGHT = ("#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB")
OVERLAY_STYLES = ("-", "--", ":", "-.", (0, (5, 1, 1, 1, 1, 1)), (0, (1, 3)))
REGION_COLOR = "#1a1a1a"
REGION_STYLE = (0, (6, 2, 1.5, 2))

# Subsample size for rank correlations and robust moments.
_MOMENT_SUBSAMPLE = 20_000


# ═══════════════════════════════════════════════════════════════════════════════
# inputs
# ═══════════════════════════════════════════════════════════════════════════════


def as_numpy(x: Any) -> np.ndarray:
    """Tensor (any device or dtype) or array-like to a float64 array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().to(torch.float64).numpy()
    return np.asarray(x, dtype=np.float64)


def draws_of(obj: Any, *, row: int, n_samples: int, name: str) -> np.ndarray:
    """Draws ``[S, d]`` (float64) from an array, a tensor or a ``JointPredictive``.

    A ``JointPredictive`` is sampled for its row ``row`` only when it can select
    rows (``select``), otherwise all rows are sampled and ``row`` is kept.  A
    ``[n, S, d]`` array of draws is indexed by ``row``.
    """
    if isinstance(obj, JointPredictive):
        if not 0 <= row < obj.batch_size:
            raise ValueError(f"row must be in [0, {obj.batch_size}), got {row}")
        if n_samples < 2:
            raise ValueError("n_samples must be at least 2")
        with torch.no_grad():
            select = getattr(obj, "select", None)
            if callable(select):
                draws = select(torch.tensor([row])).sample(n_samples)[0]
            else:
                draws = obj.sample(n_samples)[row]
        out = as_numpy(draws)
    else:
        out = as_numpy(obj)
        if out.ndim == 3:
            if not 0 <= row < out.shape[0]:
                raise ValueError(f"row must be in [0, {out.shape[0]}), got {row}")
            out = out[row]
        if out.ndim == 1:
            out = out[:, None]
    if out.ndim != 2:
        raise ValueError(f"{name} must be [n_samples, d] draws, got shape {out.shape}")
    finite = np.isfinite(out).all(axis=1)
    if not finite.all():
        warnings.warn(
            f"{name}: dropped {int((~finite).sum())} draw(s) with non-finite values",
            RuntimeWarning,
            stacklevel=3,
        )
        out = out[finite]
    if out.shape[0] < 2:
        raise ValueError(f"{name} needs at least two finite draws")
    return out


def subsample(x: np.ndarray, size: int, seed: int = 0) -> np.ndarray:
    """At most ``size`` rows of ``x`` (deterministic)."""
    if x.shape[0] <= size:
        return x
    idx = np.random.default_rng(seed).choice(x.shape[0], size=size, replace=False)
    return x[np.sort(idx)]


# ═══════════════════════════════════════════════════════════════════════════════
# levels and target selection
# ═══════════════════════════════════════════════════════════════════════════════


def contour_masses(levels: Optional[Sequence[float]], convention: str) -> tuple[list[float], str]:
    """Probability masses (ascending) and the caption describing them."""
    if convention == "mass":
        lv = [0.68, 0.95] if levels is None else [float(v) for v in levels]
        if not lv or any(not 0.0 < v < 1.0 for v in lv):
            raise ValueError("with level_convention='mass', levels must lie in (0, 1)")
        masses = sorted(lv)
        pct = ", ".join(f"{100 * m:.3g}%" for m in masses)
        return masses, f"2-D contours: highest-density regions holding {pct} of the draws"
    if convention == "sigma":
        lv = [1.0, 2.0] if levels is None else [float(v) for v in levels]
        if not lv or any(v <= 0.0 for v in lv):
            raise ValueError("with level_convention='sigma', levels must be positive sigmas")
        lv = sorted(lv)
        masses = [1.0 - math.exp(-0.5 * s * s) for s in lv]
        sig = ", ".join(f"{s:g}σ" for s in lv)
        pct = ", ".join(f"{100 * m:.3g}%" for m in masses)
        return masses, (f"2-D contours: {sig} levels of a 2-D Gaussian, holding {pct} of the draws")
    raise ValueError(f"level_convention must be 'mass' or 'sigma', got {convention!r}")


def spearman_matrix(x: np.ndarray) -> np.ndarray:
    """Spearman rank correlation matrix ``[d, d]`` (constant columns give 0)."""
    sub = subsample(x, _MOMENT_SUBSAMPLE)
    ranks = sub.argsort(axis=0).argsort(axis=0).astype(np.float64)
    ranks -= ranks.mean(axis=0)
    sd = np.sqrt((ranks**2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = (ranks.T @ ranks) / np.outer(sd, sd)
    corr = np.where(np.isfinite(corr), corr, 0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def most_dependent_dims(x: np.ndarray, cap: int) -> list[int]:
    """Up to ``cap`` targets, taken from the pairs with the largest ``|Spearman rho|``."""
    d = x.shape[1]
    corr = np.abs(spearman_matrix(x))
    iu, ju = np.triu_indices(d, k=1)
    order = np.argsort(-corr[iu, ju], kind="stable")
    chosen: list[int] = []
    for k in order:
        for t in (int(iu[k]), int(ju[k])):
            if t not in chosen and len(chosen) < cap:
                chosen.append(t)
        if len(chosen) >= cap:
            break
    return sorted(chosen)


# ═══════════════════════════════════════════════════════════════════════════════
# robust moments and plot ranges
# ═══════════════════════════════════════════════════════════════════════════════


def robust_scale(v: np.ndarray) -> float:
    """``min(sd, IQR / 1.349)``, falling back to the one that is positive (0 if none)."""
    q25, q75 = np.quantile(v, [0.25, 0.75])
    iqr = (q75 - q25) / 1.349
    sd = float(np.std(v))
    positive = [s for s in (iqr, sd) if s > 0 and math.isfinite(s)]
    return float(min(positive)) if positive else 0.0


def plot_range(
    columns: Sequence[np.ndarray], tail: float = 0.002, pad: float = 0.1
) -> tuple[float, float]:
    """Axis range from the ``tail`` and ``1 - tail`` quantiles of every dataset, padded."""
    lo = min(float(np.quantile(c, tail)) for c in columns)
    hi = max(float(np.quantile(c, 1.0 - tail)) for c in columns)
    span = hi - lo
    if not span > 0:
        centre = 0.5 * (lo + hi)
        half = max(abs(centre) * 0.05, 1e-3) if centre != 0 else 0.5
        return centre - half, centre + half
    return lo - pad * span, hi + pad * span


# ═══════════════════════════════════════════════════════════════════════════════
# smoothed densities
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Density1D:
    grid: np.ndarray  # [G] bin centres
    density: np.ndarray  # [G] normalised to integrate to the fraction of draws in range
    at_draws: np.ndarray  # [S] density at each draw (0 outside the range)


@dataclass
class Density2D:
    x: np.ndarray  # [G] bin centres of the column variable
    y: np.ndarray  # [G] bin centres of the row variable
    density: np.ndarray  # [G, G], indexed [y, x]
    at_draws: np.ndarray  # [S]


def _bin_index(v: np.ndarray, lo: float, hi: float, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    pos = (v - lo) / (hi - lo) * n_bins
    inside = (pos >= 0) & (pos < n_bins)
    return np.clip(pos.astype(np.int64), 0, n_bins - 1), inside


# Fraction of the plotted span added on each side of the KDE grid, so that draws
# just outside the axes still contribute and contours are not pulled in at the
# edges (boundary bias of a truncated KDE).
_GRID_EXTENSION = 0.3


def _extend(lo: float, hi: float) -> tuple[float, float]:
    span = hi - lo
    return lo - _GRID_EXTENSION * span, hi + _GRID_EXTENSION * span


def kde_1d(
    v: np.ndarray, lo: float, hi: float, *, smooth: float = 1.0, n_grid: int = 800
) -> Density1D:
    """Binned Gaussian KDE (bandwidth ``1.3 smooth`` times Silverman's robust rule).

    The grid covers ``[lo, hi]`` extended by 30% of the span on each side.
    """
    n = v.shape[0]
    lo, hi = _extend(lo, hi)
    width = (hi - lo) / n_grid
    scale = robust_scale(v)
    bw = smooth * 1.3 * scale * n ** (-0.2) if scale > 0 else width
    sigma_bins = max(bw / width, 1.0)
    idx, inside = _bin_index(v, lo, hi, n_grid)
    counts = np.bincount(idx[inside], minlength=n_grid).astype(np.float64)
    half = int(min(math.ceil(4.0 * sigma_bins), n_grid))
    offs = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offs / sigma_bins) ** 2)
    kernel /= kernel.sum()
    dens = np.clip(fftconvolve(counts, kernel, mode="same"), 0.0, None) / (n * width)
    grid = lo + (np.arange(n_grid) + 0.5) * width
    pos = (v - lo) / width - 0.5
    at = np.interp(pos, np.arange(n_grid), dens, left=0.0, right=0.0)
    at = np.where((v >= lo) & (v <= hi), at, 0.0)
    return Density1D(grid, dens, at)


def kde_2d(
    x: np.ndarray,
    y: np.ndarray,
    xr: tuple[float, float],
    yr: tuple[float, float],
    *,
    smooth: float = 1.0,
    n_grid: int = 240,
) -> Density2D:
    r"""Binned 2-D Gaussian KDE with a full, robust bandwidth matrix.

    The bandwidth matrix is :math:`H = (1.7\,s\,n^{-1/6})^2 \Sigma` (Scott's
    rule widened by 1.7, ``s = smooth``), with :math:`\Sigma` built from robust
    scales and the Spearman correlation of the draws, so heavy tails do not
    inflate it; it is widened when needed so the kernel spans at least one grid
    cell per axis.  The grid covers the plotted ranges ``xr`` and ``yr``
    extended by 30% on each side.
    """
    n = x.shape[0]
    xr, yr = _extend(*xr), _extend(*yr)
    wx = (xr[1] - xr[0]) / n_grid
    wy = (yr[1] - yr[0]) / n_grid
    sx, sy = robust_scale(x), robust_scale(y)
    sx = sx if sx > 0 else wx
    sy = sy if sy > 0 else wy
    sub = subsample(np.stack([x, y], axis=1), _MOMENT_SUBSAMPLE)
    rho_s = float(spearman_matrix(sub)[0, 1])
    rho = float(np.clip(2.0 * math.sin(math.pi * rho_s / 6.0), -0.97, 0.97))
    factor = smooth * 1.7 * n ** (-1.0 / 6.0)
    # bandwidth in grid units, floored at one cell
    bx = max(factor * sx / wx, 1.0)
    by = max(factor * sy / wy, 1.0)
    cov = np.array([[bx * bx, rho * bx * by], [rho * bx * by, by * by]])
    inv = np.linalg.inv(cov)
    hx = int(min(math.ceil(4.0 * bx), n_grid))
    hy = int(min(math.ceil(4.0 * by), n_grid))
    ox = np.arange(-hx, hx + 1, dtype=np.float64)
    oy = np.arange(-hy, hy + 1, dtype=np.float64)
    gx, gy = np.meshgrid(ox, oy)
    quad = inv[0, 0] * gx * gx + 2.0 * inv[0, 1] * gx * gy + inv[1, 1] * gy * gy
    kernel = np.exp(-0.5 * quad)
    kernel /= kernel.sum()

    ix, in_x = _bin_index(x, xr[0], xr[1], n_grid)
    iy, in_y = _bin_index(y, yr[0], yr[1], n_grid)
    inside = in_x & in_y
    counts = np.bincount(iy[inside] * n_grid + ix[inside], minlength=n_grid * n_grid)
    counts = counts.reshape(n_grid, n_grid).astype(np.float64)
    dens = np.clip(fftconvolve(counts, kernel, mode="same"), 0.0, None) / (n * wx * wy)

    xc = xr[0] + (np.arange(n_grid) + 0.5) * wx
    yc = yr[0] + (np.arange(n_grid) + 0.5) * wy
    coords = np.stack([(y - yr[0]) / wy - 0.5, (x - xr[0]) / wx - 0.5])
    at = map_coordinates(dens, coords, order=1, mode="constant", cval=0.0)
    at = np.where(inside, at, 0.0)
    return Density2D(xc, yc, dens, at)


def mass_thresholds(at_draws: np.ndarray, masses: Sequence[float]) -> list[float]:
    """Density thresholds whose super-level sets hold the masses of the draws.

    The threshold for mass ``m`` is the ``1 - m`` quantile of the density at the
    draws, so ``mean(at_draws >= threshold) ~= m`` (exact up to ties).  ``nan``
    when the draws inside the grid do not reach that mass.
    """
    out = []
    positive = float((at_draws > 0).mean())
    for m in masses:
        if m > positive + 1e-12:
            out.append(float("nan"))
            continue
        out.append(float(np.quantile(at_draws, 1.0 - m, method="higher")))
    return out


def hpd_mask_1d(dens: Density1D, mass: float) -> np.ndarray:
    """Grid mask of the 1-D highest-density set holding ``mass`` of the draws."""
    (t,) = mass_thresholds(dens.at_draws, [mass])
    if not math.isfinite(t):
        return np.ones_like(dens.grid, dtype=bool)
    return dens.density >= t


# ═══════════════════════════════════════════════════════════════════════════════
# titles
# ═══════════════════════════════════════════════════════════════════════════════


def split_unit(name: str) -> tuple[str, str]:
    """``"mass [kg]"`` -> ``("mass", "kg")``; names without a bracketed unit unchanged."""
    s = name.rstrip()
    if s.endswith("]") and " [" in s:
        base, unit = s[:-1].rsplit(" [", 1)
        return base, unit
    return name, ""


def _decimals(err: float) -> int:
    """Decimals that keep two significant digits of ``err``."""
    return 1 - int(math.floor(math.log10(err)))


def _fmt(v: float, decimals: int) -> str:
    if decimals <= 0:
        step = 10.0 ** (-decimals)
        return f"{round(v / step) * step:.0f}"
    return f"{v:.{decimals}f}"


def format_estimate(median: float, minus: float, plus: float) -> str:
    """``median^{+plus}_{-minus}`` (mathtext) with two significant digits of the error.

    The smaller of the two errors sets the precision, as in ``getdist`` and
    ``chainconsumer``.  Very large or small values share one power of ten.
    """
    errs = [e for e in (minus, plus) if e > 0 and math.isfinite(e)]
    if not errs or not math.isfinite(median):
        return f"${median:.3g}$"
    exponent = 0
    big = max(abs(median), max(errs))
    if big >= 1e4 or big < 1e-3:
        exponent = int(math.floor(math.log10(big)))
        scale = 10.0**exponent
        median, minus, plus = median / scale, minus / scale, plus / scale
        errs = [e / scale for e in errs]
    dec = _decimals(min(errs))
    m, lo, hi = _fmt(median, dec), _fmt(minus, dec), _fmt(plus, dec)
    if m.startswith("-") and float(m) == 0.0:
        m = m[1:]
    core = f"{m}^{{+{hi}}}_{{-{lo}}}"
    if exponent:
        return rf"$({core})\times 10^{{{exponent}}}$"
    return f"${core}$"


# ═══════════════════════════════════════════════════════════════════════════════
# region projections
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RegionLevel:
    """One level of a joint region, for one row, in the original target coordinates."""

    label: str
    kind: str  # "ellipse", "box", "draws" or "unbounded"
    mean: Optional[np.ndarray] = None  # [d]
    cov: Optional[np.ndarray] = None  # [d, d] (ellipse)
    radius: float = 0.0  # ellipse radius in Mahalanobis units
    lower: Optional[np.ndarray] = None  # [d] (box)
    upper: Optional[np.ndarray] = None
    draws: Optional[np.ndarray] = None  # [S, d] accepted draws


def _row_region(region: JointRegion, row: int) -> JointRegion:
    """The region of one row (cheap sampling); the full region when rows cannot be selected."""
    pred = region.predictive
    select = getattr(pred, "select", None)
    n = pred.batch_size
    state = region._state
    if not callable(select) or any(
        not isinstance(v, torch.Tensor) or v.shape[:1] != (n,) for v in state.values()
    ):
        return region
    idx = torch.tensor([row])
    sub_state = {k: v[row : row + 1] for k, v in state.items()}
    sub = JointRegion(
        select(idx),
        region.score,
        region.alpha,
        region.threshold[row : row + 1],
        sub_state,
        region._masked,
        region._scalar,
    )
    return sub


def region_levels(region: JointRegion, *, row: int, n_draws: int) -> list[RegionLevel]:
    """Projections data of every level of ``region`` for row ``row``."""
    pred = region.predictive
    if not 0 <= row < pred.batch_size:
        raise ValueError(f"row must be in [0, {pred.batch_size}) for the region, got {row}")
    alphas = [region.alpha] if isinstance(region.alpha, (int, float)) else list(region.alpha)
    name = region.score.name
    threshold = region.threshold[row].detach().cpu().to(torch.float64)
    out: list[RegionLevel] = []
    single: Optional[JointRegion] = None
    for lvl, a in enumerate(alphas):
        label = f"{100 * (1 - float(a)):.3g}% conformal region ({name})"
        t = float(threshold[lvl])
        if math.isinf(t) and t > 0:
            out.append(RegionLevel(label, "unbounded"))
            continue
        score = region.score
        state = region._state
        if isinstance(score, MahalanobisScore) and not region._masked:
            out.append(
                RegionLevel(
                    label,
                    "ellipse",
                    mean=as_numpy(state["mean"][row]),
                    cov=as_numpy(state["cov"][row]),
                    radius=t,
                )
            )
        elif isinstance(score, MahalanobisScore):
            d = pred.event_dim
            r = float(score._radius(threshold[lvl : lvl + 1], True, d)[0])
            out.append(
                RegionLevel(
                    label,
                    "ellipse",
                    mean=as_numpy(state["mean"][row]),
                    cov=as_numpy(state["cov"][row]),
                    radius=r,
                )
            )
        elif isinstance(score, MaxResidualScore):
            mean = as_numpy(state["mean"][row])
            std = as_numpy(state["std"][row])
            out.append(RegionLevel(label, "box", lower=mean - t * std, upper=mean + t * std))
        else:
            if single is None:
                single = _row_region(region, row)
            sub_row = 0 if single is not region else row
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                acc = single.sample(n_draws, level=lvl if single.n_levels > 1 else None)
            draws = as_numpy(acc[sub_row])
            draws = draws[np.isfinite(draws).all(axis=1)]
            if draws.shape[0] < 10:
                out.append(RegionLevel(label, "unbounded"))
                continue
            out.append(RegionLevel(label + ", outline of accepted draws", "draws", draws=draws))
    return out


def ellipse_points(mean: np.ndarray, cov: np.ndarray, radius: float, n: int = 200) -> np.ndarray:
    """Boundary ``[n, 2]`` of ``{u : (u - m)^T cov^{-1} (u - m) <= radius^2}`` in 2-D.

    The projection of a ``d``-dimensional ellipsoid onto two coordinates is the
    ellipse of the corresponding 2x2 sub-covariance with the same radius.
    """
    theta = np.linspace(0.0, 2.0 * math.pi, n)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    vals, vecs = np.linalg.eigh(cov)
    root = vecs * np.sqrt(np.clip(vals, 0.0, None))
    return mean + radius * circle @ root.T


def region_interval(level: RegionLevel, j: int) -> Optional[tuple[float, float]]:
    """1-D projection ``[lo, hi]`` of a region level onto target ``j``."""
    if level.kind == "ellipse":
        assert level.mean is not None and level.cov is not None
        half = level.radius * math.sqrt(max(float(level.cov[j, j]), 0.0))
        return float(level.mean[j]) - half, float(level.mean[j]) + half
    if level.kind == "box":
        assert level.lower is not None and level.upper is not None
        return float(level.lower[j]), float(level.upper[j])
    if level.kind == "draws":
        assert level.draws is not None
        return float(level.draws[:, j].min()), float(level.draws[:, j].max())
    return None
