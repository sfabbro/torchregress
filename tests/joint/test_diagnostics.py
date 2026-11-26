"""Joint calibration diagnostics: quiet when calibrated, firing on the error they target."""

from __future__ import annotations

import math

import pytest
import torch

from torchregress.joint.diagnostics import (
    PRERANKS,
    conditional_coverage_gap,
    coverage_by_mask,
    dependence_error,
    holm_adjust,
    hpd_calibration,
    nees_consistency,
    prerank_histograms,
    worst_slab_coverage,
)
from torchregress.joint.predictive import GaussianJoint

F64 = torch.float64
D, N, S = 4, 500, 50
RHO = 0.5


def _truth_chol() -> torch.Tensor:
    corr = torch.full((D, D), RHO, dtype=F64) + (1 - RHO) * torch.eye(D, dtype=F64)
    return torch.linalg.cholesky(corr)


def _cases(seed: int = 0) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Observations from N(mu_i, R) and draws [N, S, D] of four predictives.

    calibrated: N(mu_i, R); location: mean shifted by +0.3 sd on every target;
    scale: covariance 0.8^2 R (over-confident); dependence: N(mu_i, I), right
    marginals, no correlation.
    """
    gen = torch.Generator().manual_seed(seed)
    chol = _truth_chol()
    mu = torch.randn(N, D, generator=gen, dtype=F64)
    y = mu + torch.randn(N, D, generator=gen, dtype=F64) @ chol.T
    eps = torch.randn(N, S, D, generator=gen, dtype=F64)
    m = mu.unsqueeze(1)
    return y, {
        "calibrated": m + eps @ chol.T,
        "location": m + 0.3 + eps @ chol.T,
        "scale": m + 0.8 * eps @ chol.T,
        "dependence": m + eps,
    }


class TestPreRanks:
    def test_ranks_are_in_range_and_histograms_sum(self) -> None:
        y, cases = _cases()
        res = prerank_histograms(cases["calibrated"], y)
        assert res.names == PRERANKS
        assert res.ranks.shape == (N, len(PRERANKS))
        assert int(res.ranks.min()) >= 0 and int(res.ranks.max()) <= S
        torch.testing.assert_close(res.counts.sum(1), torch.full((6,), float(N), dtype=F64))
        torch.testing.assert_close(res.expected.sum(), torch.tensor(float(N), dtype=F64))

    def test_quiet_when_calibrated(self) -> None:
        # N = 500 rows, S = 50 draws, 10 bins (50 expected counts per bin).
        y, cases = _cases()
        res = prerank_histograms(cases["calibrated"], y)
        assert bool((res.p_values_holm > 0.01).all()), res.p_values

    def test_each_prerank_fires_on_its_error(self) -> None:
        y, cases = _cases()
        p = {k: prerank_histograms(v, y).p_values for k, v in cases.items() if k != "calibrated"}
        col = {name: i for i, name in enumerate(PRERANKS)}
        # The targeted pre-rank fires (p < 1e-6) on its error ...
        assert float(p["location"][col["location"]]) < 1e-6
        assert float(p["scale"][col["scale"]]) < 1e-6
        assert float(p["dependence"][col["dependence"]]) < 1e-6
        # ... and the scale pre-rank, invariant to a common shift, stays quiet on it.
        assert float(p["location"][col["scale"]]) > 0.01
        # The targeted pre-rank is the most significant one for location and scale.
        assert int(p["location"].argmin()) in (col["location"], col["average_rank"])
        assert int(p["scale"].argmin()) == col["scale"]

    def test_dependence_prerank_needs_two_targets(self) -> None:
        with pytest.raises(ValueError, match="two targets"):
            prerank_histograms(torch.randn(5, 10, 1), torch.randn(5, 1), preranks=["dependence"])

    def test_holm_adjustment(self) -> None:
        p = torch.tensor([0.01, 0.04, 0.03, 0.5], dtype=F64)
        expected = torch.tensor([0.04, 0.09, 0.09, 0.5], dtype=F64)
        torch.testing.assert_close(holm_adjust(p), expected)


class TestDependenceError:
    def test_fires_only_on_dependence(self) -> None:
        # Off-diagonal correlation estimates at N = 500 have standard error
        # (1 - rho^2) / sqrt(N) ~ 0.034; 12 entries give a Frobenius noise ~ 0.12.
        y, cases = _cases()
        frob = {k: float(dependence_error(v, y).frobenius) for k, v in cases.items()}
        for quiet in ("calibrated", "location", "scale"):
            assert frob[quiet] < 0.3, frob
        assert frob["dependence"] > 1.0, frob
        res = dependence_error(cases["dependence"], y)
        off = ~torch.eye(D, dtype=torch.bool)
        assert float(res.predicted[off].abs().max()) < 0.1
        assert float((res.empirical[off] - RHO).abs().max()) < 0.15


class TestNEES:
    def test_band_and_verdicts(self) -> None:
        gen = torch.Generator().manual_seed(1)
        n, d = 2000, 3
        chol = _truth_chol()[:d, :d]
        cov = chol @ chol.T
        mean = torch.randn(n, d, generator=gen, dtype=F64)
        y = mean + torch.randn(n, d, generator=gen, dtype=F64) @ chol.T
        ok = nees_consistency(mean, cov.expand(n, d, d), y)
        assert ok.verdict == "consistent"
        assert float(ok.lower) < 3.0 < float(ok.upper)
        # Each whitened eigen-direction variance is 1 +- 4 standard errors (sqrt(2/n)).
        assert float((ok.eigen_variances - 1).abs().max()) < 4 * math.sqrt(2 / n)
        assert nees_consistency(mean, 0.8 * cov, y).verdict == "over-confident"
        assert nees_consistency(mean, 1.25 * cov, y).verdict == "conservative"

    def test_eigen_directions_localise_the_error(self) -> None:
        gen = torch.Generator().manual_seed(2)
        n, d = 2000, 3
        cov = torch.diag(torch.tensor([4.0, 1.0, 0.25], dtype=F64)).to(dtype=F64)
        mean = torch.zeros(n, d, dtype=F64)
        # True variance is twice the predicted one along the largest direction only.
        y = torch.randn(n, d, generator=gen, dtype=F64) * torch.tensor([8.0, 1.0, 0.25]).sqrt()
        res = nees_consistency(mean, cov, y.to(F64))
        se = math.sqrt(2 / n)
        assert abs(float(res.eigen_variances[0]) - 2.0) < 4 * 2 * se
        assert float((res.eigen_variances[1:] - 1).abs().max()) < 4 * se

    def test_mask_counts_observed_dof(self) -> None:
        gen = torch.Generator().manual_seed(3)
        n, d = 1000, 3
        mean = torch.zeros(n, d, dtype=F64)
        cov = torch.eye(d, dtype=F64)
        y = torch.randn(n, d, generator=gen, dtype=F64)
        mask = torch.rand(n, d, generator=gen) > 0.3
        y_nan = torch.where(mask, y, torch.full_like(y, float("nan")))
        res = nees_consistency(mean, cov, y_nan, mask=mask)
        torch.testing.assert_close(res.dof, mask.sum(-1))
        torch.testing.assert_close(
            res.nees, torch.where(mask, y, torch.zeros_like(y)).square().sum(-1)
        )
        assert res.verdict == "consistent"


class TestHPDCalibration:
    def test_uniform_levels_are_inside_the_band(self) -> None:
        gen = torch.Generator().manual_seed(0)
        res = hpd_calibration(torch.rand(2000, generator=gen, dtype=F64))
        assert bool(res.within_band) and float(res.p_value) > 0.01
        assert res.empirical.shape == res.nominal.shape == (21,)
        assert bool((res.lower <= res.empirical).all() and (res.empirical <= res.upper).all())

    def test_over_confident_predictive_fails(self) -> None:
        torch.manual_seed(0)
        gen = torch.Generator().manual_seed(0)
        n, d = 1000, 2
        mean = torch.randn(n, d, generator=gen, dtype=F64)
        y = mean + torch.randn(n, d, generator=gen, dtype=F64)
        eye = torch.eye(d, dtype=F64)
        good = hpd_calibration(GaussianJoint(mean, eye).hpd_level(y))
        bad = hpd_calibration(GaussianJoint(mean, 0.7 * eye).hpd_level(y))
        assert float(good.p_value) > 0.01 and bool(good.within_band)
        assert float(bad.p_value) < 1e-6 and not bool(bad.within_band)
        # Over-confident: fewer observations than nominal inside each HPD region.
        assert float((bad.empirical - bad.nominal)[1:-1].max()) < 0


class TestCoverage:
    def test_coverage_by_mask_counts_patterns(self) -> None:
        mask = torch.tensor([[1, 1], [1, 0], [1, 1], [1, 0], [1, 0]], dtype=torch.bool)
        covered = torch.tensor([1, 0, 1, 1, 1], dtype=torch.bool)
        res = coverage_by_mask(covered, mask)
        rows = {tuple(p.tolist()): i for i, p in enumerate(res.patterns)}
        full, partial = rows[(True, True)], rows[(True, False)]
        assert int(res.counts[full]) == 2 and int(res.counts[partial]) == 3
        assert float(res.coverage[full]) == 1.0
        assert abs(float(res.coverage[partial]) - 2 / 3) < 1e-12
        assert abs(float(res.standard_error[partial]) - math.sqrt(2 / 27)) < 1e-12

    @staticmethod
    def _coverage(x_dependent: bool, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        # n = 4000, p = 3.  Coverage 0.9 everywhere, or 0.98 / 0.82 by the sign of x_0.
        gen = torch.Generator().manual_seed(seed)
        n = 4000
        x = torch.randn(n, 3, generator=gen, dtype=F64)
        prob = torch.full((n,), 0.9, dtype=F64)
        if x_dependent:
            prob = torch.where(x[:, 0] > 0, 0.98, 0.82).to(F64)
        return x, torch.rand(n, generator=gen, dtype=F64) < prob

    def test_worst_slab_detects_x_dependent_coverage(self) -> None:
        x, covered = self._coverage(True)
        res = worst_slab_coverage(x, covered, delta=0.2, n_directions=200)
        # The evaluation half's slab coverage is unbiased; the gap (0.08 for the
        # x_0 < 0 half-space) is > 3 standard errors (one-sided p < 0.0014).
        assert float(res.gap) > 3 * float(res.standard_error)
        assert float(res.mass) > 0.1
        assert abs(float(res.direction.norm()) - 1) < 1e-12

    def test_worst_slab_quiet_when_coverage_is_constant(self) -> None:
        x, covered = self._coverage(False)
        res = worst_slab_coverage(x, covered, delta=0.2, n_directions=200)
        # Unbiased for the slab's coverage (0.9): within 3 standard errors of marginal.
        assert abs(float(res.gap)) < 3 * float(res.standard_error)
        assert float(res.selection_coverage) <= float(res.coverage) + 0.05

    def test_ert_detects_x_dependent_coverage(self) -> None:
        x, covered = self._coverage(True)
        res = conditional_coverage_gap(x, covered, alpha=0.1)
        assert float(res.brier) > 4 * float(res.brier_se) and float(res.p_value) < 1e-4
        assert res.probability.shape == (4000,)

    def test_ert_quiet_when_coverage_is_constant(self) -> None:
        x, covered = self._coverage(False)
        res = conditional_coverage_gap(x, covered, alpha=0.1)
        assert float(res.brier) < 2 * float(res.brier_se) and float(res.p_value) > 0.01
        # Deterministic given the seed.
        again = conditional_coverage_gap(x, covered, alpha=0.1)
        torch.testing.assert_close(again.probability, res.probability)
