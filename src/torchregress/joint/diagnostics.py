r"""Calibration diagnostics for joint predictive distributions.

Each diagnostic targets one way in which a joint predictive can be wrong and
returns tensors (or a small frozen dataclass of tensors); nothing here plots.

* :func:`prerank_histograms`: multivariate rank histograms from pre-rank
  functions (location, scale, dependence, energy, average rank, band depth),
  with a chi-square uniformity test per pre-rank and Holm adjustment across them.
* :func:`nees_consistency`: the normalised estimation error squared
  :math:`\chi^2` band of covariance heads, with per-eigen-direction variances.
* :func:`hpd_calibration`: calibration curve of highest-density levels with a
  simultaneous DKW band and a Kolmogorov--Smirnov test.
* :func:`coverage_by_mask`: region coverage per missingness pattern.
* :func:`worst_slab_coverage`: minimum coverage over slabs of feature space
  (a lower bound on how far conditional coverage can fall).
* :func:`conditional_coverage_gap`: the excess-risk-of-the-target-coverage
  (ERT) estimate of a conditional-coverage violation.
* :func:`dependence_error`: predicted against empirical correlation of
  standardised residuals.

All tests here are about calibration on average over the rows (and, for the
slab and ERT diagnostics, over groups of rows defined by the features); none
of them can certify calibration at an individual :math:`x`.

References:

1. S. Allen, J. Ziegel, D. Ginsbourger, "Assessing the calibration of
   multivariate probabilistic forecasts", QJRMS (2024; arXiv 2307.05846).
2. T. L. Thorarinsdottir, M. Scheuerer, C. Heinz, "Assessing the calibration of
   high-dimensional ensemble forecasts using rank histograms", J. Comput. Graph.
   Stat. 25 (2016).
3. Y. Bar-Shalom, X. R. Li, T. Kirubarajan, "Estimation with Applications to
   Tracking and Navigation", Wiley (2001), section 5.4 (NEES).
4. M. Cauchois, S. Gupta, J. Duchi, "Knowing what you know: valid and validated
   confidence sets in multiclass and multilabel prediction", JMLR 22 (2021).
5. V. Dheur, M. Fontana, Y. Estievenart, N. Desobry, S. Ben Taieb, "Multi-output
   conformal regression: a unified comparative study", ICML 2025 (arXiv 2501.10533).
6. Braun et al., "Excess risk of the target coverage", ICML 2026 (arXiv 2512.11779).
7. A. Dvoretzky, J. Kiefer, J. Wolfowitz (1956); P. Massart, "The tight constant
   in the DKW inequality", Ann. Probab. 18 (1990).
8. S. Holm, "A simple sequentially rejective multiple test procedure", Scand. J.
   Stat. 6 (1979).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

import torch
from scipy.stats import chi2, chisquare, kstest
from torch import Tensor

from ._common import masked_gaussian_quadratic

__all__ = [
    "CoverageByMask",
    "CoverageGap",
    "DependenceError",
    "HPDCalibration",
    "NEESConsistency",
    "PRERANKS",
    "PreRankHistograms",
    "WorstSlabCoverage",
    "conditional_coverage_gap",
    "coverage_by_mask",
    "dependence_error",
    "hpd_calibration",
    "holm_adjust",
    "nees_consistency",
    "prerank_histograms",
    "worst_slab_coverage",
]

_F64 = torch.float64
PRERANKS = ("location", "scale", "dependence", "energy", "average_rank", "band_depth")
_PAIR_BUDGET = 1 << 22  # elements per chunk of pairwise-distance tensors


def _check_samples(samples: Tensor, y: Tensor) -> None:
    if samples.dim() != 3 or y.dim() != 2 or samples.shape[0] != y.shape[0]:
        raise ValueError(
            f"samples must be [n, S, d] and y [n, d], got {tuple(samples.shape)} and "
            f"{tuple(y.shape)}"
        )
    if samples.shape[2] != y.shape[1]:
        raise ValueError("samples and y must have the same number of targets")


def holm_adjust(p_values: Tensor) -> Tensor:
    r"""Holm step-down adjusted p-values (family-wise error control, any dependence).

    With sorted p-values :math:`p_{(1)} \le \dots \le p_{(m)}`,
    :math:`\tilde p_{(i)} = \max_{k \le i} \min\{1, (m - k + 1) p_{(k)}\}`.
    """
    m = p_values.shape[0]
    order = p_values.argsort()
    factors = torch.arange(m, 0, -1, device=p_values.device, dtype=p_values.dtype)
    adjusted = torch.cummax(p_values[order] * factors, dim=0).values.clamp(max=1.0)
    out = torch.empty_like(p_values)
    out[order] = adjusted
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# pre-rank histograms
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PreRankHistograms:
    """Result of :func:`prerank_histograms`.

    Attributes:
        names: Pre-rank names, in the order of the columns below.
        ranks: ``[n, P]`` ranks of the observation in ``{0, ..., S}``.
        counts: ``[P, n_bins]`` histogram counts of the ranks.
        expected: ``[n_bins]`` expected count per bin under uniformity.
        p_values: ``[P]`` chi-square uniformity p-values.
        p_values_holm: ``[P]`` Holm-adjusted p-values across the pre-ranks.
    """

    names: tuple[str, ...]
    ranks: Tensor
    counts: Tensor
    expected: Tensor
    p_values: Tensor
    p_values_holm: Tensor


def _pairwise_mean_distance(pool: Tensor) -> Tensor:
    """Mean Euclidean distance of each pool member to the others: ``[n, m, d] -> [n, m]``."""
    n, m, _ = pool.shape
    chunk = max(1, _PAIR_BUDGET // (m * m))
    parts = [torch.cdist(block, block).sum(-1) / (m - 1) for block in pool.split(chunk)]
    return torch.cat(parts) if parts else pool.new_zeros(n, m)


def _coordinate_ranks(pool: Tensor) -> Tensor:
    """Rank (1..m) of each member's coordinate within the pool, ``[n, m, d]``."""
    return pool.argsort(dim=1).argsort(dim=1).to(pool.dtype) + 1.0


def _prerank(name: str, pool: Tensor) -> Tensor:
    d = pool.shape[-1]
    m = pool.shape[1]
    if name == "location":
        return pool.mean(-1)
    if name == "scale":
        return (pool - pool.mean(-1, keepdim=True)).square().mean(-1)
    if name == "dependence":
        if d < 2:
            raise ValueError("the dependence pre-rank needs at least two targets")
        total = pool.sum(-1)
        pair_sum = 0.5 * (total.square() - pool.square().sum(-1))
        return pair_sum / (d * (d - 1) / 2)
    if name == "energy":
        return _pairwise_mean_distance(pool)
    if name == "average_rank":
        return _coordinate_ranks(pool).mean(-1)
    if name == "band_depth":
        r = _coordinate_ranks(pool)
        return ((r - 1.0) * (m - r)).mean(-1)
    raise ValueError(f"unknown pre-rank {name!r}; choose from {PRERANKS}")


def prerank_histograms(
    samples: Tensor,
    y: Tensor,
    preranks: Optional[Sequence[str]] = None,
    standardize: bool = True,
    n_bins: int = 10,
    seed: int = 0,
) -> PreRankHistograms:
    r"""Multivariate rank histograms from pre-rank functions.

    For row ``i``, pool the observation with the ``S`` draws,
    :math:`\{x_0 = y_i, x_1, \dots, x_S\}`, optionally standardise each target by
    the pool's mean and standard deviation (a symmetric function of the pool,
    so exchangeability is preserved), apply a pre-rank
    :math:`\rho: \mathbb R^d \to \mathbb R` (which may depend on the pool
    symmetrically) and record the rank of :math:`\rho(x_0)` among
    :math:`\rho(x_1), \dots, \rho(x_S)`, ties broken uniformly at random:
    :math:`R_i \in \{0, \dots, S\}`.  If :math:`y_i` is exchangeable with the
    draws (the predictive is calibrated in distribution), :math:`R_i` is uniform
    on :math:`\{0, \dots, S\}` for every pre-rank; each pre-rank makes a
    particular departure visible.

    Pre-ranks (on standardised coordinates when ``standardize``):

    * ``location``: :math:`\bar x = \frac1d\sum_j x_j` (Allen et al. 2024); a
      biased mean gives a sloped histogram.
    * ``scale``: :math:`\frac1d\sum_j (x_j - \bar x)^2`; dispersion errors.
    * ``dependence``: the mean pairwise product
      :math:`\binom d2^{-1}\sum_{j<k} x_j x_k` over *all* pairs, a
      correlation-type statistic that does not assume ordered targets (Allen et
      al.'s lag-1 variogram does); needs ``d >= 2`` and is meaningful only with
      ``standardize``.
    * ``energy``: the mean Euclidean distance to the other pool members
      (an energy-score centrality); outlying observations rank high.
    * ``average_rank``: the mean over targets of the coordinate-wise rank in
      the pool (Thorarinsdottir et al. 2016).
    * ``band_depth``: the mean over targets of :math:`(r_j - 1)(S + 1 - r_j)`,
      the number of pool pairs whose coordinate range brackets :math:`x_j`
      (Thorarinsdottir et al. 2016); low for outlying observations.

    Uniformity is tested per pre-rank with a chi-square test on ``n_bins`` bins
    of consecutive ranks (exact expected counts when ``S + 1`` is not a
    multiple of ``n_bins``), and Holm's procedure adjusts the p-values across
    pre-ranks.  The chi-square approximation needs about five expected counts
    per bin.  The ranks are computed in ``O(n S d)`` memory except ``energy``,
    which is chunked over rows (``O(S^2)`` per row).

    Args:
        samples: Draws ``[n, S, d]``.
        y: Observations ``[n, d]``.
        preranks: Names, a subset of ``PRERANKS``.  Default: all of them,
            except ``dependence`` when ``d < 2`` (it needs two targets; naming
            it explicitly with one target is an error).
        standardize: Standardise each target by the pooled mean and standard
            deviation of the row.
        n_bins: Histogram bins (at most ``S + 1``).
        seed: Seed of the random tie-breaking.
    """
    _check_samples(samples, y)
    if preranks is None:
        names = tuple(p for p in PRERANKS if samples.shape[2] >= 2 or p != "dependence")
    else:
        names = tuple(preranks)
    if not names:
        raise ValueError("at least one pre-rank is required")
    n, s, _ = samples.shape
    pool = torch.cat([y.unsqueeze(1).to(samples), samples], dim=1).to(_F64)
    if standardize:
        sd = pool.std(dim=1, keepdim=True).clamp_min(torch.finfo(_F64).tiny)
        pool = (pool - pool.mean(dim=1, keepdim=True)) / sd
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ranks = []
    for name in names:
        rho = _prerank(name, pool)
        obs, draws = rho[:, :1], rho[:, 1:]
        below = (draws < obs).sum(1)
        ties = (draws == obs).sum(1)
        u = torch.rand(n, generator=gen, dtype=_F64).to(rho.device)
        ranks.append(below + (u * (ties + 1).to(_F64)).floor().long())
    rank = torch.stack(ranks, dim=1)  # [n, P]
    bins = min(int(n_bins), s + 1)
    bin_of_rank = torch.arange(s + 1, device=rank.device) * bins // (s + 1)
    expected = torch.bincount(bin_of_rank, minlength=bins).to(_F64) * n / (s + 1)
    counts = torch.stack(
        [torch.bincount(bin_of_rank[rank[:, p]], minlength=bins) for p in range(len(names))]
    ).to(_F64)
    p_values = torch.tensor(
        [float(chisquare(c.cpu().numpy(), expected.cpu().numpy()).pvalue) for c in counts],
        dtype=_F64,
        device=rank.device,
    )
    return PreRankHistograms(names, rank, counts, expected, p_values, holm_adjust(p_values))


# ═══════════════════════════════════════════════════════════════════════════════
# NEES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class NEESConsistency:
    """Result of :func:`nees_consistency`.

    Attributes:
        nees: ``[n]`` normalised estimation errors squared.
        dof: ``[n]`` degrees of freedom (observed targets) per row.
        statistic: Mean NEES (scalar).
        lower: Lower end of the acceptance band (scalar).
        upper: Upper end of the acceptance band (scalar).
        verdict: ``"over-confident"`` (above the band), ``"consistent"`` or
            ``"conservative"`` (below the band).
        eigen_variances: ``[d]`` mean squared whitened residual along the
            eigenvectors of each row's covariance, from the largest eigenvalue to
            the smallest (complete rows only; NaN if there are none).  Each should
            be near one; a value above one flags under-estimated variance along
            those directions.
    """

    nees: Tensor
    dof: Tensor
    statistic: Tensor
    lower: Tensor
    upper: Tensor
    verdict: str
    eigen_variances: Tensor


def nees_consistency(
    mean: Tensor,
    covariance: Tensor,
    y: Tensor,
    confidence: float = 0.95,
    mask: Optional[Tensor] = None,
) -> NEESConsistency:
    r"""Normalised estimation error squared (NEES) consistency test.

    For each row, :math:`\varepsilon_i = (y_i - m_i)^\top \Sigma_i^{-1} (y_i - m_i)`
    over the observed targets :math:`O_i`; under a correct Gaussian predictive,
    :math:`\varepsilon_i \sim \chi^2_{|O_i|}` independently, so
    :math:`N\bar\varepsilon \sim \chi^2_{D}` with :math:`D = \sum_i |O_i|` and the
    two-sided acceptance band at level :math:`1 - a` is

    .. math::
        \bar\varepsilon \in \big[\chi^2_{D}(a/2)/N,\ \chi^2_{D}(1 - a/2)/N\big].

    Above the band the covariance is too small (over-confident); below it, too
    large (conservative).  The test is exact for Gaussian predictives and an
    approximation (it checks the second moment) for others.

    Args:
        mean: ``[n, d]``.
        covariance: ``[n, d, d]`` (or ``[d, d]``).
        y: ``[n, d]``.
        confidence: :math:`1 - a`.
        mask: Boolean ``[n, d]`` (``True`` = observed).
    """
    if mean.dim() != 2 or y.shape != mean.shape:
        raise ValueError(f"mean and y must be [n, d], got {tuple(mean.shape)}, {tuple(y.shape)}")
    n, d = mean.shape
    cov = covariance.to(_F64)
    if cov.dim() == 2:
        cov = cov.expand(n, d, d)
    if cov.shape != (n, d, d):
        raise ValueError(f"covariance must be [n, d, d], got {tuple(covariance.shape)}")
    obs = torch.ones(n, d, dtype=torch.bool, device=mean.device) if mask is None else mask
    if obs.shape != (n, d) or obs.dtype != torch.bool:
        raise ValueError("mask must be boolean [n, d]")
    diff = y.to(_F64) - mean.to(_F64)
    resid = torch.where(obs, diff, torch.zeros_like(diff))
    # Exact marginalisation of the missing targets (known_covariance_gaussian_nll).
    nees, _, _ = masked_gaussian_quadratic(resid, cov, obs)
    dof = obs.sum(-1)
    total = int(dof.sum())
    a = 1.0 - confidence
    lower = chi2.ppf(a / 2, total) / n
    upper = chi2.ppf(1 - a / 2, total) / n
    stat = nees.mean()
    verdict = (
        "over-confident"
        if float(stat) > upper
        else ("conservative" if float(stat) < lower else "consistent")
    )
    complete = obs.all(-1)
    if bool(complete.any()):
        evals, evecs = torch.linalg.eigh(cov[complete])  # ascending
        proj = (evecs.mT @ resid[complete].unsqueeze(-1)).squeeze(-1)
        eig_var = (proj.square() / evals).mean(0).flip(0)
    else:
        eig_var = torch.full((d,), float("nan"), dtype=_F64, device=mean.device)
    band = torch.tensor([lower, upper], dtype=_F64, device=mean.device)
    return NEESConsistency(nees, dof, stat, band[0], band[1], verdict, eig_var)


# ═══════════════════════════════════════════════════════════════════════════════
# HPD calibration curve
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class HPDCalibration:
    """Result of :func:`hpd_calibration`.

    Attributes:
        nominal: ``[n_bins + 1]`` grid of nominal levels in ``[0, 1]``.
        empirical: ``[n_bins + 1]`` empirical CDF of the HPD levels on the grid
            (the fraction of observations inside the HPD region of that mass).
        lower: ``[n_bins + 1]`` lower edge of the simultaneous DKW band.
        upper: ``[n_bins + 1]`` upper edge.
        within_band: Whether the empirical curve stays inside the band.
        ks_statistic: Kolmogorov-Smirnov distance to the uniform distribution.
        p_value: Kolmogorov-Smirnov p-value.
    """

    nominal: Tensor
    empirical: Tensor
    lower: Tensor
    upper: Tensor
    within_band: Tensor
    ks_statistic: Tensor
    p_value: Tensor


def hpd_calibration(levels: Tensor, n_bins: int = 20, confidence: float = 0.95) -> HPDCalibration:
    r"""Calibration curve of highest-density (HPD) levels.

    For a calibrated predictive the HPD level
    :math:`h_i = P_{Y' \sim p(\cdot\mid x_i)}[p(Y') > p(y_i)]` is uniform, so the
    fraction of observations inside the HPD region of mass :math:`a`,
    :math:`\hat F(a) = \frac1n\sum_i \mathbf 1[h_i \le a]`, should equal
    :math:`a`.  Above the diagonal the regions over-cover (conservative), below
    they under-cover (over-confident).  The band is the DKW–Massart band
    :math:`a \pm \sqrt{\log(2/\alpha)/(2n)}`, which contains the whole curve with
    probability at least :math:`1-\alpha` under calibration (simultaneously over
    :math:`a`); the KS test uses the same statistic.  This checks average
    coverage over the rows only.

    Args:
        levels: HPD levels ``[n]`` in ``[0, 1]`` (``JointPredictive.hpd_level``).
        n_bins: Grid resolution of the returned curve.
        confidence: :math:`1-\alpha` of the band.
    """
    if levels.dim() != 1 or levels.numel() == 0:
        raise ValueError("levels must be a non-empty [n] tensor")
    lv = levels.to(_F64)
    n = lv.shape[0]
    grid = torch.linspace(0.0, 1.0, n_bins + 1, dtype=_F64, device=lv.device)
    emp = (lv.unsqueeze(0) <= grid.unsqueeze(1)).to(_F64).mean(1)
    eps = math.sqrt(math.log(2.0 / (1.0 - confidence)) / (2.0 * n))
    test = kstest(lv.cpu().numpy(), "uniform")
    ks = torch.tensor(float(test.statistic), dtype=_F64, device=lv.device)
    return HPDCalibration(
        nominal=grid,
        empirical=emp,
        lower=(grid - eps).clamp(0.0, 1.0),
        upper=(grid + eps).clamp(0.0, 1.0),
        within_band=ks <= eps,
        ks_statistic=ks,
        p_value=torch.tensor(float(test.pvalue), dtype=_F64, device=lv.device),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# coverage diagnostics
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class CoverageByMask:
    """Result of :func:`coverage_by_mask`.

    Attributes:
        patterns: ``[k, d]`` distinct missingness patterns (``True`` = observed).
        counts: ``[k]`` rows per pattern.
        coverage: ``[k]`` coverage per pattern.
        standard_error: ``[k]`` binomial standard error per pattern.
    """

    patterns: Tensor
    counts: Tensor
    coverage: Tensor
    standard_error: Tensor


def coverage_by_mask(covered: Tensor, mask: Tensor) -> CoverageByMask:
    r"""Coverage per missingness pattern.

    Marginal coverage over all rows can hide under-coverage for specific
    patterns of observed targets (Zaffran et al. 2023).  Returns, for each
    distinct pattern, the number of rows, the coverage and its binomial standard
    error :math:`\sqrt{\hat c(1-\hat c)/n_k}`.

    Args:
        covered: ``[n]`` boolean (or 0/1) coverage indicators.
        mask: ``[n, d]`` boolean, ``True`` = observed.
    """
    if covered.dim() != 1 or mask.dim() != 2 or mask.shape[0] != covered.shape[0]:
        raise ValueError("covered must be [n] and mask [n, d]")
    patterns, inverse, counts = torch.unique(
        mask.to(torch.bool), dim=0, return_inverse=True, return_counts=True
    )
    hits = torch.zeros(patterns.shape[0], dtype=_F64, device=covered.device)
    hits.index_add_(0, inverse, covered.to(_F64))
    cov = hits / counts.to(_F64)
    se = (cov * (1.0 - cov) / counts.to(_F64)).sqrt()
    return CoverageByMask(patterns, counts, cov, se)


@dataclass(frozen=True)
class WorstSlabCoverage:
    """Result of :func:`worst_slab_coverage`.

    Attributes:
        coverage: Coverage in the selected slab on the evaluation half.
        standard_error: Its binomial standard error.
        marginal_coverage: Coverage of the evaluation half.
        gap: ``marginal_coverage - coverage``.
        mass: Fraction of the evaluation half inside the slab.
        selection_coverage: Coverage of the slab on the selection half
            (optimistically low, since it was minimised there).
        direction: ``[p]`` unit direction (on standardised features).
        bounds: ``[2]`` slab bounds on the projected standardised features.
    """

    coverage: Tensor
    standard_error: Tensor
    marginal_coverage: Tensor
    gap: Tensor
    mass: Tensor
    selection_coverage: Tensor
    direction: Tensor
    bounds: Tensor


def worst_slab_coverage(
    x: Tensor,
    covered: Tensor,
    delta: float = 0.2,
    n_directions: int = 1000,
    seed: int = 0,
    n_grid: int = 50,
) -> WorstSlabCoverage:
    r"""Worst-slab coverage (Cauchois et al. 2021; protocol of Dheur et al. 2025).

    A slab is :math:`\{x: a \le v^\top x \le b\}` for a unit direction :math:`v`.
    On a random half of the rows (selection), over ``n_directions`` directions
    drawn uniformly on the sphere (features standardised with the selection
    half's moments) and slab ends on a grid of ``n_grid`` empirical quantiles of
    :math:`v^\top x`, find the slab of empirical mass :math:`\ge \delta` with
    the lowest coverage; report its coverage on the other half (evaluation).

    Because the slab is chosen on data independent of the evaluation half,
    the reported coverage is an unbiased estimate of the coverage in that
    slab, and a value clearly below the marginal coverage shows that coverage
    depends on :math:`x`.  A value near the marginal coverage does not prove
    conditional coverage: it only rules out large deviations on slabs.

    Args:
        x: Features ``[n, p]``.
        covered: Coverage indicators ``[n]``.
        delta: Minimum slab mass.
        n_directions: Random directions.
        seed: Seed for the split and the directions.
        n_grid: Quantile grid for the slab ends.
    """
    if x.dim() != 2 or covered.shape != (x.shape[0],):
        raise ValueError("x must be [n, p] and covered [n]")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    n, p = x.shape
    gen = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(n, generator=gen).to(x.device)
    sel, ev = perm[: n // 2], perm[n // 2 :]
    x64, cov = x.to(_F64), covered.to(_F64)
    mu, sd = x64[sel].mean(0), x64[sel].std(0).clamp_min(1e-12)
    xs = (x64 - mu) / sd
    v = torch.randn(n_directions, p, generator=gen, dtype=_F64).to(x.device)
    v = v / v.norm(dim=1, keepdim=True)
    proj = xs[sel] @ v.T  # [m, M]
    m = proj.shape[0]
    q = torch.linspace(0.0, 1.0, n_grid + 1, dtype=_F64, device=x.device)
    edges = torch.quantile(proj, q, dim=0).T  # [M, G+1]
    # counts below each edge, per direction: [M, G+1]
    order = proj.argsort(0)
    sorted_proj = torch.gather(proj, 0, order).T.contiguous()  # [M, m]
    cum_cov = torch.cat(
        [torch.zeros(n_directions, 1, dtype=_F64, device=x.device), cov[sel][order].T.cumsum(1)],
        dim=1,
    )  # [M, m + 1]
    lo_idx = torch.searchsorted(sorted_proj, edges.contiguous(), right=False)  # first >= edge
    hi_idx = torch.searchsorted(sorted_proj, edges.contiguous(), right=True)  # past <= edge
    # slab [edge_a, edge_b] contains rows lo_idx[a] .. hi_idx[b] - 1
    count = hi_idx.unsqueeze(1) - lo_idx.unsqueeze(2)  # [M, a, b]
    hits = torch.gather(cum_cov, 1, hi_idx).unsqueeze(1) - torch.gather(
        cum_cov, 1, lo_idx
    ).unsqueeze(2)
    valid = count >= math.ceil(delta * m)
    rate = torch.where(valid, hits / count.clamp_min(1), torch.full_like(hits, math.inf))
    flat = rate.reshape(n_directions, -1)
    best_per_dir, arg = flat.min(1)
    k = int(best_per_dir.argmin())
    a_i, b_i = divmod(int(arg[k]), n_grid + 1)
    lo, hi = edges[k, a_i], edges[k, b_i]
    t = xs[ev] @ v[k]
    inside = (t >= lo) & (t <= hi)
    n_in = int(inside.sum())
    c_ev = cov[ev][inside].mean() if n_in else cov.new_tensor(float("nan"))
    marg = cov[ev].mean()
    se = (c_ev * (1 - c_ev) / max(n_in, 1)).sqrt()
    return WorstSlabCoverage(
        coverage=c_ev,
        standard_error=se,
        marginal_coverage=marg,
        gap=marg - c_ev,
        mass=inside.to(_F64).mean(),
        selection_coverage=best_per_dir[k],
        direction=v[k],
        bounds=torch.stack([lo, hi]),
    )


@dataclass(frozen=True)
class CoverageGap:
    """Result of :func:`conditional_coverage_gap`.

    Attributes:
        brier: ERT estimate for the squared (Brier) loss.
        brier_se: Its standard error over rows.
        absolute: ERT estimate for the absolute loss.
        absolute_se: Its standard error.
        p_value: One-sided normal p-value of ``brier > 0``.
        probability: ``[n]`` out-of-fold predicted coverage probabilities.
    """

    brier: Tensor
    brier_se: Tensor
    absolute: Tensor
    absolute_se: Tensor
    p_value: Tensor
    probability: Tensor


def _fit_logistic(features: Tensor, target: Tensor, ridge: float) -> tuple[Tensor, Tensor]:
    w = torch.zeros(features.shape[1], dtype=_F64, device=features.device, requires_grad=True)
    p0 = target.mean().clamp(1e-3, 1 - 1e-3)
    b = (p0 / (1 - p0)).log().clone().requires_grad_(True)
    opt = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")

    def closure() -> Tensor:
        opt.zero_grad()
        logits = features @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        loss = loss + ridge * w.square().sum()
        loss.backward()  # type: ignore[no-untyped-call]
        return loss

    opt.step(closure)  # type: ignore[no-untyped-call]
    return w.detach(), b.detach()


def conditional_coverage_gap(
    x: Tensor,
    covered: Tensor,
    alpha: float,
    n_splits: int = 5,
    seed: int = 0,
    n_fourier: int = 64,
    ridge: float = 1e-3,
) -> CoverageGap:
    r"""Excess risk of the target coverage (ERT; Braun et al., ICML 2026).

    With :math:`Z_i = \mathbf 1[y_i \in C(x_i)]`, fit a classifier
    :math:`h: x \mapsto P(Z = 1 \mid x)` out of fold (``n_splits`` folds) and
    estimate

    .. math::
        \mathrm{ERT}_\ell = \frac1n\sum_i \big[\ell(1-\alpha, Z_i) - \ell(h(x_i), Z_i)\big]

    for the squared loss :math:`\ell(q, z) = (q - z)^2` and the absolute loss
    :math:`\ell(q, z) = |q - z|`.  If coverage is exactly :math:`1-\alpha` at
    every :math:`x`, the constant :math:`1-\alpha` is Bayes-optimal for the
    squared loss and no classifier beats it in expectation, so a positive value
    (beyond its standard error) certifies a conditional-coverage violation.  It
    is also positive when the *marginal* coverage differs from
    :math:`1-\alpha`.  A value near zero does not prove conditional coverage: the
    classifier may simply be too weak.  The p-value is a one-sided normal
    approximation for the squared-loss estimate.

    Classifier: ridge-penalised logistic regression, fitted with L-BFGS in
    float64, on the training fold's standardised features and ``n_fourier``
    random Fourier features :math:`\sqrt{2/D}\cos(W x + c)`,
    :math:`W \sim \mathcal N(0, I/p)`, :math:`c \sim \mathcal U(0, 2\pi)`
    (deterministic given ``seed``).

    Args:
        x: Features ``[n, p]``.
        covered: Coverage indicators ``[n]``.
        alpha: Target miscoverage.
        n_splits: Folds.
        seed: Seed of the folds and the random features.
        n_fourier: Random Fourier features (0 for a linear classifier).
        ridge: Ridge penalty on the weights.
    """
    if x.dim() != 2 or covered.shape != (x.shape[0],):
        raise ValueError("x must be [n, p] and covered [n]")
    n, p = x.shape
    if n_splits < 2 or n_splits > n:
        raise ValueError("n_splits must be in [2, n]")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(n, generator=gen).to(x.device)
    freq = (torch.randn(p, n_fourier, generator=gen, dtype=_F64) / math.sqrt(p)).to(x.device)
    phase = (torch.rand(n_fourier, generator=gen, dtype=_F64) * 2 * math.pi).to(x.device)
    x64, z = x.to(_F64), covered.to(_F64)
    prob = torch.empty(n, dtype=_F64, device=x.device)
    for fold in perm.tensor_split(n_splits):
        train = torch.ones(n, dtype=torch.bool, device=x.device)
        train[fold] = False
        mu, sd = x64[train].mean(0), x64[train].std(0).clamp_min(1e-12)

        def featurise(rows: Tensor, mu: Tensor = mu, sd: Tensor = sd) -> Tensor:
            xs = (rows - mu) / sd
            rff = math.sqrt(2.0 / max(n_fourier, 1)) * torch.cos(xs @ freq + phase)
            return torch.cat([xs, rff], dim=1)

        w, b = _fit_logistic(featurise(x64[train]), z[train], ridge)
        prob[fold] = torch.sigmoid(featurise(x64[fold]) @ w + b)
    target = 1.0 - alpha
    brier = (target - z).square() - (prob - z).square()
    absolute = (target - z).abs() - (prob - z).abs()
    b_mean, b_se = brier.mean(), brier.std() / math.sqrt(n)
    a_mean, a_se = absolute.mean(), absolute.std() / math.sqrt(n)
    zscore = float(b_mean / b_se.clamp_min(1e-300))
    p_value = 0.5 * math.erfc(zscore / math.sqrt(2.0))
    return CoverageGap(
        brier=b_mean,
        brier_se=b_se,
        absolute=a_mean,
        absolute_se=a_se,
        p_value=torch.tensor(p_value, dtype=_F64, device=x.device),
        probability=prob,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# dependence
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class DependenceError:
    """Result of :func:`dependence_error`.

    Attributes:
        predicted: ``[d, d]`` average over rows of the predictive correlation.
        empirical: ``[d, d]`` correlation of the standardised residuals.
        frobenius: Frobenius norm of their difference.
    """

    predicted: Tensor
    empirical: Tensor
    frobenius: Tensor


def dependence_error(samples: Tensor, y: Tensor) -> DependenceError:
    r"""Predicted against empirical dependence between targets.

    With per-row draw means :math:`m_i` and standard deviations :math:`s_i`,
    the standardised residuals :math:`r_{ij} = (y_{ij} - m_{ij}) / s_{ij}` of a
    calibrated predictive have, for targets :math:`j \ne k`,
    :math:`E[r_j r_k] = E_x[\rho_{jk}(x)]`, the average predictive correlation.
    The function returns the average of the per-row correlation matrices of the
    draws, the empirical correlation matrix of the standardised residuals over
    the rows, and the Frobenius norm of their difference (only off-diagonal
    entries differ).  It isolates dependence errors when the marginals are
    calibrated; a mean or scale error also moves the empirical matrix, so read
    it next to marginal diagnostics.

    Args:
        samples: Draws ``[n, S, d]``.
        y: Observations ``[n, d]``.
    """
    _check_samples(samples, y)
    draws = samples.to(_F64)
    centred = draws - draws.mean(1, keepdim=True)
    sd = centred.square().mean(1).sqrt().clamp_min(1e-300)  # [n, d]
    std_draws = centred / sd.unsqueeze(1)
    predicted = (std_draws.mT @ std_draws / draws.shape[1]).mean(0)
    resid = (y.to(_F64) - draws.mean(1)) / sd
    empirical = torch.corrcoef(resid.T)
    if empirical.dim() == 0:
        empirical = empirical.reshape(1, 1)
    return DependenceError(predicted, empirical, torch.linalg.norm(predicted - empirical))
