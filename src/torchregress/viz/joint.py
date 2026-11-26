"""One-call diagnostics figure for a joint predictive distribution.

Every number on the figure comes from :mod:`torchregress.joint.diagnostics`
(or from :class:`~torchregress.joint.JointRegion` for regions); this module only
draws.  Each panel carries a one-line reading aid.
"""

from __future__ import annotations

import math
import textwrap
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - hit only without the viz extra
    raise ImportError(
        "torchregress.viz requires matplotlib; pip install 'torchregress[viz]'"
    ) from exc

import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import kstest

from torchregress.joint._common import seeded
from torchregress.joint.diagnostics import (
    conditional_coverage_gap,
    coverage_by_mask,
    dependence_error,
    holm_adjust,
    hpd_calibration,
    prerank_histograms,
    worst_slab_coverage,
)
from torchregress.joint.predictive import JointPredictive
from torchregress.joint.regions import JointRegion
from torchregress.viz._corner import TOL_BRIGHT, as_numpy

__all__ = ["plot_joint_diagnostics"]

_MAIN = TOL_BRIGHT[0]
_ALERT = TOL_BRIGHT[1]
_GREY = "#8a8a8a"
_BAND = "#dddddd"

# One-line reading aids.
_PRERANK_AIDS = {
    "location": "Slope: mean bias. U shape: too narrow; hump: too wide.",
    "scale": "Not flat: the spread of the targets around their average is off.",
    "dependence": "High ranks: targets move together more than predicted; low: less.",
    "energy": "High ranks: observations outlying (too narrow); low: too wide.",
    "average_rank": "Slope: mean bias. U shape: too narrow; hump: too wide.",
    "band_depth": "Low ranks: observations outlying (too narrow); high: too wide.",
}
_PRERANK_TITLES = {
    "location": "Location",
    "scale": "Scale",
    "dependence": "Dependence",
    "energy": "Energy",
    "average_rank": "Average rank",
    "band_depth": "Band depth",
}


def _p(p: float) -> str:
    """``"p = 0.123"`` or ``"p < 0.001"``."""
    if not math.isfinite(p):
        return "p n/a"
    return "p < 0.001" if p < 1e-3 else f"p = {p:.3f}"


def _title(ax: Axes, title: str, aid: str, fontsize: float = 9.5) -> None:
    """Bold title stacked above a one-line italic reading aid (wrapped to the panel)."""
    fig = ax.figure
    width_in = ax.get_position().width * fig.get_figwidth()
    chars = max(24, int(width_in * 15.5))
    aid_text = ax.text(
        0.0,
        1.03,
        textwrap.fill(aid, chars),
        transform=ax.transAxes,
        fontsize=7.3,
        style="italic",
        color="#444444",
        va="bottom",
        ha="left",
    )
    ax.annotate(
        textwrap.fill(title, max(20, int(width_in * 13.5))),
        xy=(0.0, 1.0),
        xycoords=aid_text,
        xytext=(0, 2),
        textcoords="offset points",
        fontsize=fontsize,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def _unavailable(ax: Axes, title: str, why: str) -> None:
    ax.set_axis_off()
    ax.set_title(title, fontsize=9.5, fontweight="bold", loc="left")
    ax.text(
        0.5,
        0.5,
        why,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.5,
        color=_GREY,
        wrap=True,
    )


def _clean(ax: Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=7.5, direction="out", length=3)


# ═══════════════════════════════════════════════════════════════════════════════
# panels
# ═══════════════════════════════════════════════════════════════════════════════


def _panel_hpd(ax: Axes, levels: torch.Tensor, confidence: float) -> None:
    res = hpd_calibration(levels, n_bins=40, confidence=confidence)
    nom, emp = as_numpy(res.nominal), as_numpy(res.empirical)
    ax.fill_between(
        nom,
        as_numpy(res.lower),
        as_numpy(res.upper),
        color=_BAND,
        lw=0,
        label=f"{100 * confidence:.0f}% simultaneous band (DKW)",
    )
    ax.plot([0, 1], [0, 1], color=_GREY, lw=1.0, ls="--", label="Calibrated")
    ax.plot(nom, emp, color=_MAIN, lw=2.0, label="Empirical")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Nominal HPD mass", fontsize=8.5)
    ax.set_ylabel("Fraction of observations inside", fontsize=8.5)
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    verdict = "inside band" if bool(res.within_band) else "leaves band"
    _title(
        ax,
        f"Joint HPD calibration (KS {_p(float(res.p_value))}, {verdict})",
        "Below the diagonal: regions too small (over-confident); above: too large.",
    )
    _clean(ax)


def _matrix(ax: Axes, m: np.ndarray, names: Sequence[str], vlim: float, cmap: str) -> Any:
    d = m.shape[0]
    im = ax.imshow(m, cmap=cmap, vmin=-vlim, vmax=vlim, interpolation="nearest")
    ax.set_xticks(range(d))
    ax.set_yticks(range(d))
    show_names = d <= 10
    ax.set_xticklabels(names if show_names else [], fontsize=7.5, rotation=0)
    ax.set_yticklabels(names if show_names else [], fontsize=7.5)
    ax.tick_params(length=0)
    if d <= 8:
        for i in range(d):
            for j in range(d):
                v = m[i, j]
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.8 if d > 5 else 7.5,
                    color="white" if abs(v) > 0.6 * vlim else "#222222",
                )
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def _panel_dependence(
    axes: Sequence[Axes], samples: torch.Tensor, y: torch.Tensor, names: Sequence[str], fig: Figure
) -> None:
    res = dependence_error(samples, y)
    pred, emp = as_numpy(res.predicted), as_numpy(res.empirical)
    diff = emp - pred
    im_p = _matrix(axes[0], pred, names, 1.0, "RdBu_r")
    im_e = _matrix(axes[1], emp, names, 1.0, "RdBu_r")
    lim = max(float(np.abs(diff).max()), 0.05)
    im_d = _matrix(axes[2], diff, names, lim, "PuOr_r")
    for ax, im in zip(axes, (im_p, im_e, im_d)):
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=7)
    _title(axes[0], "Predicted correlation", "Average over rows of the predictive correlation.")
    _title(axes[1], "Empirical correlation", "Correlation of the standardised residuals.")
    _title(
        axes[2],
        f"Empirical − predicted (Frobenius {float(res.frobenius):.2f})",
        "Near 0: dependence right. Marginal errors also move it: read with the PITs.",
    )


def _panel_prerank(
    ax: Axes, name: str, counts: np.ndarray, expected: np.ndarray, p_holm: float
) -> None:
    rel = counts / expected
    n_bins = rel.size
    centres = (np.arange(n_bins) + 0.5) / n_bins
    # pointwise 95% binomial band around the uniform expectation
    total = counts.sum()
    prob = expected / total
    half = 1.96 * np.sqrt(total * prob * (1 - prob)) / expected
    ax.fill_between(
        np.r_[0, centres, 1],
        1 - np.r_[half[0], half, half[-1]],
        1 + np.r_[half[0], half, half[-1]],
        color=_BAND,
        lw=0,
        step="mid",
    )
    colour = _ALERT if p_holm < 0.05 else _MAIN
    ax.bar(centres, rel, width=1.0 / n_bins, color=colour, alpha=0.8, edgecolor="white", lw=0.6)
    ax.axhline(1.0, color=_GREY, lw=1.0, ls="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(2.0, float(rel.max()) * 1.1))
    ax.set_xlabel("Normalised rank", fontsize=8)
    _title(
        ax,
        f"{_PRERANK_TITLES.get(name, name)} (Holm {_p(p_holm)})",
        _PRERANK_AIDS.get(name, "Flat: calibrated for this pre-rank."),
        fontsize=9.0,
    )
    _clean(ax)


def _panel_pit(ax: Axes, pit: np.ndarray, name: str, p_holm: float, eps: float) -> None:
    grid = np.linspace(0.0, 1.0, 201)
    ecdf = np.searchsorted(np.sort(pit), grid, side="right") / pit.size
    diff = ecdf - grid
    ax.fill_between(grid, -eps, eps, color=_BAND, lw=0)
    ax.axhline(0.0, color=_GREY, lw=1.0, ls="--")
    colour = _ALERT if np.abs(diff).max() > eps else _MAIN
    ax.plot(grid, diff, color=colour, lw=1.8)
    lim = max(1.6 * eps, float(np.abs(diff).max()) * 1.15)
    ax.set_xlim(0, 1)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("PIT", fontsize=8)
    _title(
        ax,
        f"PIT of {name} (KS, Holm {_p(p_holm)})",
        "∩ above 0: predictive too high; ∪: too low; +/− S: too narrow; −/+ S: too wide.",
        fontsize=9.0,
    )
    _clean(ax)


def _panel_coverage(ax: Axes, covered: np.ndarray, nominal: Sequence[float]) -> None:
    n = covered.shape[0]
    ones = torch.ones(n, 1, dtype=torch.bool)
    cov, se = [], []
    for lvl in range(covered.shape[1]):
        res = coverage_by_mask(torch.as_tensor(covered[:, lvl]), ones)
        cov.append(float(res.coverage[0]))
        se.append(float(res.standard_error[0]))
    nom = np.asarray(nominal)
    lo = min(float(nom.min()), min(cov)) - 0.08
    ax.plot([0, 1], [0, 1], color=_GREY, ls="--", lw=1.0, label="Nominal")
    ax.errorbar(
        nom,
        cov,
        yerr=1.96 * np.asarray(se),
        fmt="o",
        color=_MAIN,
        ms=5,
        capsize=3,
        label="Empirical ± 1.96 se",
    )
    ax.set_xlim(max(lo, 0.0), 1.0)
    ax.set_ylim(max(lo, 0.0), 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("Nominal coverage 1 − α", fontsize=8.5)
    ax.set_ylabel("Empirical coverage", fontsize=8.5)
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    _title(
        ax,
        "Region coverage by level",
        "On the diagonal: the conformal guarantee holds on these rows.",
    )
    _clean(ax)


def _panel_volume(ax: Axes, log_vol: np.ndarray, nominal: Sequence[float]) -> None:
    finite = np.isfinite(log_vol)
    lo = float(log_vol[finite].min()) if finite.any() else 0.0
    hi = float(log_vol[finite].max()) if finite.any() else 1.0
    bins = np.linspace(lo, hi + 1e-9, 35)
    n_inf = int((~finite).sum())
    for lvl in range(log_vol.shape[1]):
        v = log_vol[:, lvl]
        v = v[np.isfinite(v)]
        if v.size:
            ax.hist(
                v,
                bins=bins,
                histtype="step",
                lw=1.6,
                color=TOL_BRIGHT[lvl % len(TOL_BRIGHT)],
                label=f"{100 * nominal[lvl]:.3g}% (median {np.median(v):.2f})",
            )
    ax.set_xlabel("log volume of the region", fontsize=8.5)
    ax.set_ylabel("Rows", fontsize=8.5)
    ax.legend(fontsize=7, frameon=False, loc="best")
    extra = f" ({n_inf} unbounded)" if n_inf else ""
    _title(
        ax,
        "Region size" + extra,
        "Smaller is sharper at equal coverage; a long right tail: very uncertain rows.",
    )
    _clean(ax)


def _panel_slab(
    ax: Axes, x: torch.Tensor, covered: torch.Tensor, nominal: float, seed: int
) -> None:
    res = worst_slab_coverage(x, covered, delta=0.2, seed=seed)
    labels = ["All rows\n(evaluation half)", f"Worst slab\n({100 * float(res.mass):.0f}% of rows)"]
    vals = [float(res.marginal_coverage), float(res.coverage)]
    n_eval = x.shape[0] - x.shape[0] // 2
    se_all = math.sqrt(max(vals[0] * (1 - vals[0]), 1e-12) / n_eval)
    errs = [1.96 * se_all, 1.96 * float(res.standard_error)]
    colours = [_MAIN, _ALERT if vals[1] + errs[1] < nominal else _MAIN]
    ax.bar([0, 1], vals, yerr=errs, color=colours, alpha=0.85, capsize=4, width=0.55)
    ax.axhline(nominal, color=_GREY, ls="--", lw=1.0, label=f"Nominal {100 * nominal:.3g}%")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Coverage", fontsize=8.5)
    ax.set_ylim(0.0, 1.12)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    _title(
        ax,
        f"Worst-slab coverage (gap {float(res.gap):+.3f})",
        "Slab well below nominal: coverage depends on x (slab chosen on the other half).",
    )
    _clean(ax)


def _panel_ert(ax: Axes, x: torch.Tensor, covered: torch.Tensor, alpha: float, seed: int) -> None:
    res = conditional_coverage_gap(x, covered, alpha, seed=seed)
    prob = as_numpy(res.probability)
    ax.hist(prob, bins=np.linspace(0, 1, 41), color=_MAIN, alpha=0.8, edgecolor="white", lw=0.4)
    ax.axvline(1 - alpha, color=_GREY, ls="--", lw=1.0, label=f"Nominal {100 * (1 - alpha):.3g}%")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Out-of-fold P(covered | x)", fontsize=8.5)
    ax.set_ylabel("Rows", fontsize=8.5)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    _title(
        ax,
        f"Conditional coverage: ERT {float(res.brier):.4f} ± {float(res.brier_se):.4f} "
        f"({_p(float(res.p_value))})",
        "Mass away from the nominal line, ERT > 0: coverage varies with x.",
    )
    _clean(ax)


# ═══════════════════════════════════════════════════════════════════════════════
# figure
# ═══════════════════════════════════════════════════════════════════════════════


def _alphas(region: JointRegion) -> List[float]:
    a = region.alpha
    return [float(a)] if isinstance(a, (int, float)) else [float(v) for v in a]


def plot_joint_diagnostics(
    predictive: JointPredictive,
    y: Union[torch.Tensor, np.ndarray],
    *,
    x: Optional[Union[torch.Tensor, np.ndarray]] = None,
    region: Optional[JointRegion] = None,
    n_samples: int = 256,
    preranks: Optional[Sequence[str]] = None,
    target_names: Optional[Sequence[str]] = None,
    max_targets: int = 6,
    region_level: Optional[int] = None,
    confidence: float = 0.95,
    seed: int = 0,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[Union[str, Path]] = None,
    return_figure: bool = False,
    return_axes: bool = False,
) -> Union[Figure, Tuple[Figure, Dict[str, Any]], None]:
    r"""Calibration diagnostics of a joint predictive in one figure, read top to bottom.

    Rows (every statistic from :mod:`torchregress.joint.diagnostics`):

    1. **Joint HPD calibration** (:func:`~torchregress.joint.hpd_calibration`
       of :meth:`~torchregress.joint.JointPredictive.hpd_level`) with its
       simultaneous DKW band, next to the **predicted and empirical correlation
       matrices** and their difference
       (:func:`~torchregress.joint.dependence_error`).
    2. **Pre-rank histograms** (:func:`~torchregress.joint.prerank_histograms`)
       relative to the uniform expectation, Holm-adjusted p-values in the titles.
    3. **Per-target PIT** as ECDF-minus-uniform curves with a DKW band that is
       simultaneous over the shown targets (Bonferroni); the PIT of target
       :math:`j` is the randomised rank of :math:`y_j` among the draws
       (the ``location`` pre-rank of that single target), with KS p-values
       Holm-adjusted across targets.  At most ``max_targets`` targets are shown
       (those with the smallest p-values when there are more).
    4. With ``region``: **coverage by level**
       (:func:`~torchregress.joint.coverage_by_mask` on all rows) and the
       **distribution of log-volumes** (:meth:`JointRegion.log_volume
       <torchregress.joint.JointRegion.log_volume>`); with ``x`` as well, the
       **worst-slab coverage** (:func:`~torchregress.joint.worst_slab_coverage`)
       and the **ERT conditional-coverage** estimate
       (:func:`~torchregress.joint.conditional_coverage_gap`) at level
       ``region_level``.

    Every panel has a one-line reading aid.  All checks are about calibration on
    average over rows (or over slabs of ``x``), never at an individual ``x``.

    Args:
        predictive: Predictive of ``n`` held-out rows.
        y: Observed targets ``[n, d]``.
        x: Features ``[n, p]`` for the conditional-coverage panels.
        region: A :class:`~torchregress.joint.JointRegion` for the same rows.
        n_samples: Draws per row for the pre-ranks, PITs, dependence and HPD
            levels.
        preranks: Pre-rank names (default: all applicable).
        target_names: Names of the targets.
        max_targets: Largest number of PIT panels.
        region_level: Index into the region's ``alpha`` for the conditional
            panels (default: the level with the highest coverage).
        confidence: Confidence of the simultaneous bands.
        seed: Seed of every random choice in the figure: the predictive draws,
            the HPD reference draws, the region volumes, the random tie-breaking,
            the slab split and the ERT folds.  The figure is reproducible for a
            given seed, and the caller's torch RNG state is left as it was.
        title: Figure title.
        figsize: Figure size (default from the number of rows).
        save_path: Save the figure there (200 dpi).
        return_figure: Return the figure (otherwise it is shown and closed).
        return_axes: Return ``(figure, axes)`` with ``axes`` a dict from panel
            name (``"hpd"``, ``"corr_predicted"``, ``"corr_empirical"``,
            ``"corr_difference"``, ``"prerank_<name>"``, ``"pit_<j>"``,
            ``"coverage"``, ``"volume"``, ``"slab"``, ``"ert"``) to its axes.

    Returns:
        ``(Figure, axes)`` with ``return_axes``, the ``Figure`` with
        ``return_figure``, else ``None``.
    """
    if not isinstance(predictive, JointPredictive):
        raise TypeError("predictive must be a torchregress.joint.JointPredictive")
    n, d = predictive.batch_size, predictive.event_dim
    # Every random draw of the figure (predictive samples, HPD reference draws,
    # region volumes) comes from torch RNGs seeded by ``seed``; the caller's RNG
    # state is restored afterwards.
    with seeded(seed), torch.no_grad():
        samples = predictive.sample(n_samples)
    y_t = torch.as_tensor(y).to(device=samples.device, dtype=samples.dtype)
    if y_t.shape != (n, d):
        raise ValueError(f"y must be [n, d] = [{n}, {d}], got {tuple(y_t.shape)}")
    x_t = None
    if x is not None:
        x_t = torch.as_tensor(x)
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(1)
        if x_t.shape[0] != n:
            raise ValueError(f"x must have n = {n} rows, got {tuple(x_t.shape)}")
        x_t = x_t.to(device=samples.device, dtype=torch.float64)
    if region is not None and (
        region.predictive.batch_size != n or region.predictive.event_dim != d
    ):
        raise ValueError("region must cover the same rows and targets as predictive")
    names = (
        list(target_names) if target_names is not None else [f"$y_{{{j + 1}}}$" for j in range(d)]
    )
    if len(names) != d:
        raise ValueError(f"target_names must have {d} entries")

    # ── statistics ───────────────────────────────────────────────────────────
    pr = prerank_histograms(samples, y_t, preranks=preranks, seed=seed)
    gen = torch.Generator().manual_seed(seed + 1)
    pit_cols, pit_p = [], []
    for j in range(d):
        rk = prerank_histograms(
            samples[..., j : j + 1],
            y_t[:, j : j + 1],
            preranks=["location"],
            standardize=False,
            seed=seed + 2 + j,
        ).ranks[:, 0]
        u = torch.rand(n, generator=gen, dtype=torch.float64)
        pit = (rk.cpu().to(torch.float64) + u) / (n_samples + 1)
        pit_cols.append(pit.numpy())
        pit_p.append(float(kstest(pit.numpy(), "uniform").pvalue))
    pit_holm = as_numpy(holm_adjust(torch.tensor(pit_p, dtype=torch.float64)))
    order = np.argsort(pit_holm, kind="stable")
    shown = sorted(order[:max_targets].tolist()) if d > max_targets else list(range(d))
    eps = math.sqrt(math.log(2.0 * len(shown) / (1.0 - confidence)) / (2.0 * n))

    hpd_levels = None
    if predictive.has_density:
        with seeded(seed + 1), torch.no_grad():
            hpd_levels = predictive.hpd_level(y_t, n_samples=n_samples)

    # ── layout ───────────────────────────────────────────────────────────────
    has_region = region is not None
    rows = 3 + int(has_region)
    heights = [3.6, 2.5, 2.5] + ([3.4] if has_region else [])
    width = 16.0
    if figsize is None:
        figsize = (width, sum(heights) + 0.75 * rows + (0.6 if title else 0.3))
    fig = plt.figure(figsize=figsize)
    outer = fig.add_gridspec(
        rows,
        1,
        height_ratios=heights,
        hspace=0.5,
        left=0.05,
        right=0.98,
        bottom=0.05,
        top=1 - (1.35 if title else 1.0) / figsize[1],
    )
    axes: Dict[str, Any] = {}

    top = outer[0].subgridspec(1, 4, wspace=0.42)
    axes["hpd"] = fig.add_subplot(top[0])
    if hpd_levels is not None:
        _panel_hpd(axes["hpd"], hpd_levels, confidence)
    else:
        _unavailable(
            axes["hpd"],
            "Joint HPD calibration",
            f"{type(predictive).__name__} has no density:\nHPD levels need log_prob.",
        )
    dep_axes = [fig.add_subplot(top[k]) for k in (1, 2, 3)]
    axes["corr_predicted"], axes["corr_empirical"], axes["corr_difference"] = dep_axes
    if d >= 2:
        _panel_dependence(dep_axes, samples, y_t, names, fig)
    else:
        for ax in dep_axes:
            _unavailable(ax, "Dependence", "One target: no dependence to check.")

    n_pr = len(pr.names)
    pre = outer[1].subgridspec(1, max(n_pr, 4), wspace=0.32)
    counts, expected = as_numpy(pr.counts), as_numpy(pr.expected)
    for k, name in enumerate(pr.names):
        ax = fig.add_subplot(pre[k])
        _panel_prerank(ax, name, counts[k], expected, float(pr.p_values_holm[k]))
        if k == 0:
            ax.set_ylabel("Count / uniform", fontsize=8)
        axes[f"prerank_{name}"] = ax

    pit_gs = outer[2].subgridspec(1, max(len(shown), 4), wspace=0.32)
    for k, j in enumerate(shown):
        ax = fig.add_subplot(pit_gs[k])
        _panel_pit(ax, pit_cols[j], names[j], float(pit_holm[j]), eps)
        if k == 0:
            ax.set_ylabel("ECDF − uniform", fontsize=8)
        axes[f"pit_{j}"] = ax
    if len(shown) < d:
        axes[f"pit_{shown[0]}"].annotate(
            f"Showing the {len(shown)} of {d} targets with the smallest p-values",
            xy=(0, -0.42),
            xycoords="axes fraction",
            fontsize=7.5,
            color="#444444",
        )

    if region is not None:
        alphas = _alphas(region)
        nominal = [1.0 - a for a in alphas]
        with seeded(seed + 2), torch.no_grad():
            cov_t = region.contains(y_t)
            log_vol, _ = region.log_volume()
        covered = as_numpy(cov_t).reshape(n, -1).astype(bool)
        lv_np = as_numpy(log_vol).reshape(n, -1)
        reg = outer[3].subgridspec(1, 4, wspace=0.42)
        axes["coverage"] = fig.add_subplot(reg[0])
        _panel_coverage(axes["coverage"], covered, nominal)
        axes["volume"] = fig.add_subplot(reg[1])
        _panel_volume(axes["volume"], lv_np, nominal)
        if region_level is None:
            region_level = int(np.argmin(alphas))
        if not 0 <= region_level < len(alphas):
            raise ValueError(f"region_level must be in [0, {len(alphas)})")
        if x_t is not None:
            cov_l = torch.as_tensor(covered[:, region_level]).to(x_t.device)
            axes["slab"] = fig.add_subplot(reg[2])
            _panel_slab(axes["slab"], x_t, cov_l, nominal[region_level], seed)
            axes["ert"] = fig.add_subplot(reg[3])
            _panel_ert(axes["ert"], x_t, cov_l, alphas[region_level], seed)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1 - 0.12 / figsize[1], va="top")
    fig.text(
        0.05,
        1 - (0.5 if title else 0.15) / figsize[1],
        f"{n} rows, {d} targets, {n_samples} draws per row. Read top to bottom: joint "
        "calibration and dependence, multivariate ranks, per-target PITs"
        + (", regions." if has_region else ".")
        + " Red: a departure at the 5% level.",
        fontsize=8.5,
        color="#444444",
        va="top",
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if return_axes:
        return fig, axes
    if return_figure:
        return fig
    plt.show()
    plt.close(fig)  # TR-VIZ-05: close figures we created
    return None
