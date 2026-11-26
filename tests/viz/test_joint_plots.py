"""Corner plots and the joint diagnostics panel (Agg backend, single process)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.contour import ContourSet
from matplotlib.figure import Figure

from torchregress.joint import (
    GaussianJoint,
    HPDScore,
    JointConformal,
    MahalanobisScore,
    MaxResidualScore,
    SampleJoint,
)
from torchregress.viz import plot_corner_plot, plot_joint_diagnostics
from torchregress.viz._corner import (
    format_estimate,
    kde_2d,
    mass_thresholds,
    most_dependent_dims,
)

F64 = torch.float64


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _gaussian_draws(n: int = 20_000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, 0.7, -0.2], [0.7, 1.5, 0.3], [-0.2, 0.3, 0.8]])
    return rng.multivariate_normal([0.0, 1.0, -1.0], cov, size=n)


def _legend_text(fig: Figure) -> str:
    leg = fig.legends[0]
    parts = [leg.get_title().get_text()] + [t.get_text() for t in leg.get_texts()]
    return "\n".join(parts)


def _line_contours(ax) -> list:
    return [c for c in ax.collections if isinstance(c, ContourSet) and not c.filled]


def _gaussian_region(score, alpha=0.1, n=400, d=3):
    gen = torch.Generator().manual_seed(0)
    mean = torch.randn(n, d, generator=gen, dtype=F64)
    chol = torch.linalg.cholesky(torch.eye(d, dtype=F64) + 0.5 * torch.ones(d, d, dtype=F64))
    pred = GaussianJoint(mean, chol)
    y = mean + torch.randn(n, d, generator=gen, dtype=F64) @ chol.T
    half = n // 2
    cal = GaussianJoint(mean[:half], chol)
    test = GaussianJoint(mean[half:], chol)
    region = JointConformal(score, alpha=alpha).calibrate(cal, y[:half]).region(test)
    return pred, test, region, y[half:]


# ── backward compatibility ────────────────────────────────────────────────────


class TestBackwardCompatibility:
    def test_every_legacy_argument(self, tmp_path):
        rng = np.random.default_rng(1)
        samples = rng.normal(size=(600, 3))
        members = [rng.normal(loc=m, size=(300, 3)) for m in (0.0, 0.3)]
        path = tmp_path / "corner.png"
        fig = plot_corner_plot(
            samples,
            member_samples=members,
            param_names=["a", "b", "c"],
            true_vals=np.zeros(3),
            quantiles=(0.05, 0.5, 0.95),
            show_titles=True,
            show_uncertainty_decomposition=True,
            show_scatter=True,
            title="Legacy",
            color="#1f77b4",
            truth_color="#e31a1c",
            member_colors=["#ff7f0e", "#2ca02c"],
            figsize=(7, 7),
            save_path=path,
            return_figure=True,
        )
        assert isinstance(fig, Figure)
        assert path.exists()
        assert tuple(fig.get_size_inches()) == (7.0, 7.0)
        assert "sigma" in fig.axes[0].get_title() or "σ" in fig.axes[0].get_title()

    def test_tensor_inputs_any_dtype(self):
        samples = torch.randn(300, 2, dtype=torch.bfloat16)
        fig = plot_corner_plot(
            samples, param_names=["a", "b"], true_vals=torch.tensor([0.5, -0.5]), return_figure=True
        )
        assert isinstance(fig, Figure)

    def test_default_path_shows_and_closes(self):
        before = plt.get_fignums()
        assert plot_corner_plot(np.random.default_rng(0).normal(size=(200, 2))) is None
        assert plt.get_fignums() == before

    def test_quantiles_empty_hides_titles(self):
        fig, axes = plot_corner_plot(_gaussian_draws(2000)[:, :2], quantiles=(), return_axes=True)
        assert axes[0, 0].get_title() == ""

    def test_param_names_length_checked(self):
        with pytest.raises(ValueError, match="param_names length"):
            plot_corner_plot(np.zeros((10, 2)) + np.arange(10)[:, None], param_names=["a"])


# ── shapes and paths ──────────────────────────────────────────────────────────


class TestShapes:
    @pytest.mark.parametrize("d", [1, 2, 3])
    def test_axes_shape(self, d):
        fig, axes = plot_corner_plot(_gaussian_draws(3000)[:, :d], return_axes=True)
        assert axes.shape == (d, d)
        for i in range(d):
            for j in range(i + 1, d):
                assert not axes[i, j].axison

    def test_d1_has_title_and_legend(self):
        fig, axes = plot_corner_plot(_gaussian_draws(3000)[:, :1], return_axes=True)
        assert "=" in axes[0, 0].get_title()
        assert "1-D: median and 68% central interval" in _legend_text(fig)
        assert "2-D contours" not in _legend_text(fig)

    def test_predictive_row(self):
        pred = GaussianJoint(
            torch.tensor([[0.0, 0.0], [10.0, -10.0]], dtype=F64),
            torch.eye(2, dtype=F64),
        )
        fig, axes = plot_corner_plot(pred, row=1, n_samples=4000, return_axes=True)
        lo, hi = axes[1, 0].get_xlim()
        assert lo < 10.0 < hi
        assert "4,000 draws" in _legend_text(fig)

    def test_batched_draws_row(self):
        draws = torch.randn(3, 500, 2)
        draws[2] += 50.0
        _, axes = plot_corner_plot(draws, row=2, return_axes=True)
        lo, hi = axes[1, 0].get_xlim()
        assert lo < 50.0 < hi

    def test_many_draws(self):
        draws = np.random.default_rng(3).normal(size=(1_000_000, 2))
        fig, axes = plot_corner_plot(draws, return_axes=True, show_scatter=False)
        assert len(_line_contours(axes[1, 0])) == 1

    def test_dims_selection(self):
        _, axes = plot_corner_plot(_gaussian_draws(2000), dims=[2, 0], return_axes=True)
        assert axes.shape == (2, 2)
        assert axes[1, 0].get_xlabel() == "$y_{3}$"

    def test_large_d_selects_dependent_pairs(self):
        rng = np.random.default_rng(4)
        d = 12
        cov = np.eye(d)
        cov[1, 9] = cov[9, 1] = 0.9
        cov[3, 4] = cov[4, 3] = -0.8
        draws = rng.multivariate_normal(np.zeros(d), cov, size=5000)
        assert {1, 9, 3, 4} <= set(most_dependent_dims(draws, 4))
        fig, axes = plot_corner_plot(draws, max_dims=4, return_axes=True)
        assert axes.shape == (4, 4)
        assert "Showing 4 of 12 targets" in _legend_text(fig)
        fig, axes = plot_corner_plot(draws, return_axes=True)
        assert axes.shape == (8, 8)

    def test_constant_target(self):
        draws = _gaussian_draws(2000)
        draws[:, 1] = 3.0
        fig, axes = plot_corner_plot(draws, return_axes=True)
        assert "constant" in axes[1, 1].get_title()
        lo, hi = axes[1, 1].get_xlim()
        assert lo < 3.0 < hi and hi - lo > 0

    def test_near_constant_target(self):
        draws = _gaussian_draws(2000)
        draws[:, 0] = 1.0 + 1e-9 * draws[:, 0]
        fig = plot_corner_plot(draws, return_figure=True)
        assert isinstance(fig, Figure)

    def test_heavy_tails_do_not_set_limits(self):
        rng = np.random.default_rng(5)
        draws = rng.standard_cauchy(size=(20_000, 2))
        _, axes = plot_corner_plot(draws, return_axes=True)
        lo, hi = axes[1, 0].get_xlim()
        assert hi - lo < 0.05 * (draws[:, 0].max() - draws[:, 0].min())

    def test_figsize_grows_with_panels(self):
        small = plot_corner_plot(_gaussian_draws(1000)[:, :2], return_figure=True)
        large = plot_corner_plot(_gaussian_draws(1000), return_figure=True)
        assert large.get_size_inches()[0] > small.get_size_inches()[0]


# ── contour mass ──────────────────────────────────────────────────────────────


class TestContourMass:
    @pytest.mark.parametrize(
        "convention,levels,masses",
        [
            ("mass", None, (0.68, 0.95)),
            ("mass", (0.5, 0.9), (0.5, 0.9)),
            ("sigma", None, (1 - np.exp(-0.5), 1 - np.exp(-2.0))),
        ],
    )
    def test_drawn_contours_enclose_stated_mass(self, convention, levels, masses):
        draws = _gaussian_draws(20_000)
        fig, axes = plot_corner_plot(
            draws, levels=levels, level_convention=convention, return_axes=True
        )
        pts = draws[:, [0, 1]]
        (lines,) = _line_contours(axes[1, 0])
        # contour levels are thresholds sorted ascending: outer (largest mass) first
        fractions = sorted(float(np.mean(path.contains_points(pts))) for path in lines.get_paths())
        assert fractions == pytest.approx(sorted(masses), abs=0.015)

    def test_thresholds_hold_mass_of_draws_bimodal(self):
        rng = np.random.default_rng(6)
        a = rng.normal(size=30_000)
        b = np.where(rng.random(30_000) < 0.4, -2.0, 1.5) + 0.4 * rng.normal(size=30_000)
        dens = kde_2d(a, b, (-4, 4), (-4, 4))
        for m, t in zip((0.68, 0.95), mass_thresholds(dens.at_draws, (0.68, 0.95))):
            assert float(np.mean(dens.at_draws >= t)) == pytest.approx(m, abs=1e-3)

    def test_level_labels(self):
        fig = plot_corner_plot(_gaussian_draws(2000), return_figure=True)
        assert "68%, 95% of the draws" in _legend_text(fig)
        fig = plot_corner_plot(_gaussian_draws(2000), level_convention="sigma", return_figure=True)
        text = _legend_text(fig).replace("\n", " ")
        assert "1σ, 2σ" in text and "39.3%, 86.5%" in text

    def test_bad_levels(self):
        with pytest.raises(ValueError, match="levels must lie"):
            plot_corner_plot(_gaussian_draws(500), levels=(1.5,))
        with pytest.raises(ValueError, match="level_convention"):
            plot_corner_plot(_gaussian_draws(500), level_convention="hpd")

    def test_hpd_interval_and_hist(self):
        fig = plot_corner_plot(
            _gaussian_draws(3000), interval="hpd", marginal="hist", return_figure=True
        )
        assert "68% highest-density interval" in _legend_text(fig).replace("\n", " ")


# ── regions and overlays ──────────────────────────────────────────────────────


class TestOverlays:
    def test_mahalanobis_ellipse(self):
        _, test, region, y = _gaussian_region(MahalanobisScore())
        fig, axes = plot_corner_plot(test, row=0, region=region, true_vals=y[0], return_axes=True)
        assert "90% conformal region (mahalanobis)" in _legend_text(fig)
        assert "not a density contour" in _legend_text(fig).replace("\n", " ")
        # exact ellipse: 2-D projection radius equals the threshold in Mahalanobis units
        t = float(region.threshold[0, 0])
        cov = region._state["cov"][0][:2, :2].numpy()
        mean = region._state["mean"][0][:2].numpy()
        ellipse = [ln for ln in axes[1, 0].get_lines() if len(ln.get_xdata()) == 200]
        assert len(ellipse) == 1
        pts = np.column_stack([ellipse[0].get_xdata(), ellipse[0].get_ydata()]) - mean
        quad = np.einsum("ni,ij,nj->n", pts, np.linalg.inv(cov), pts)
        assert np.sqrt(quad) == pytest.approx(t, rel=1e-6)

    def test_box_region_multi_level(self):
        _, test, region, _ = _gaussian_region(MaxResidualScore(), alpha=[0.2, 0.1])
        fig = plot_corner_plot(test, region=region, return_figure=True)
        text = _legend_text(fig)
        assert "80% conformal region (max_residual)" in text
        assert "90% conformal region (max_residual)" in text

    def test_outline_of_accepted_draws(self):
        _, test, region, _ = _gaussian_region(HPDScore(n_samples=64), n=200)
        fig, axes = plot_corner_plot(
            test, region=region, n_samples=3000, region_samples=1500, return_axes=True
        )
        assert "outline of accepted draws" in _legend_text(fig)
        assert len(_line_contours(axes[1, 0])) >= 2  # density contours and the outline

    def test_others_overlay(self):
        a = _gaussian_draws(3000)
        b = _gaussian_draws(3000, seed=2) * 2.0
        pred = SampleJoint(torch.as_tensor(b).unsqueeze(0))
        fig, axes = plot_corner_plot(
            a, label="Posterior", others={"Prior": b, "Prior (joint)": pred}, return_axes=True
        )
        text = _legend_text(fig)
        assert "Posterior (3,000 draws)" in text and "Prior (3,000 draws)" in text
        assert "Prior (joint) (" in text
        assert len(_line_contours(axes[1, 0])) == 3

    def test_others_dimension_mismatch(self):
        with pytest.raises(ValueError, match="targets"):
            plot_corner_plot(_gaussian_draws(500), others={"x": np.zeros((10, 2))})


# ── titles ────────────────────────────────────────────────────────────────────


class TestEstimateFormat:
    def test_two_significant_digits_of_smaller_error(self):
        assert format_estimate(1.23456, 0.0123, 0.0456) == "$1.235^{+0.046}_{-0.012}$"
        assert format_estimate(0.3, 1.0, 1.0) == "$0.3^{+1.0}_{-1.0}$"
        assert format_estimate(1234.0, 56.0, 78.0) == "$1234^{+78}_{-56}$"

    def test_large_values_share_exponent(self):
        assert (
            format_estimate(123456.0, 1200.0, 1500.0) == r"$(1.235^{+0.015}_{-0.012})\times 10^{5}$"
        )

    def test_units_in_title(self):
        _, axes = plot_corner_plot(
            _gaussian_draws(2000)[:, :2], param_names=["mass [kg]", "b"], return_axes=True
        )
        title = axes[0, 0].get_title()
        assert title.startswith("mass = ") and title.endswith(" kg")
        assert axes[1, 0].get_xlabel() == "mass [kg]"


# ── diagnostics panel ─────────────────────────────────────────────────────────


class TestJointDiagnostics:
    def test_full_panel(self):
        _, test, region, y = _gaussian_region(MahalanobisScore(), alpha=[0.5, 0.1])
        x = torch.randn(y.shape[0], 2, dtype=F64)
        fig, axes = plot_joint_diagnostics(
            test, y, x=x, region=region, n_samples=64, return_axes=True
        )
        expected = {
            "hpd",
            "corr_predicted",
            "corr_empirical",
            "corr_difference",
            "coverage",
            "volume",
            "slab",
            "ert",
        }
        assert expected <= set(axes)
        assert {f"prerank_{n}" for n in ("location", "dependence", "energy")} <= set(axes)
        assert {"pit_0", "pit_1", "pit_2"} <= set(axes)
        texts = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
        assert "Below the diagonal: regions too small" in texts

    def test_minimal_and_lifecycle(self):
        _, test, _, y = _gaussian_region(MahalanobisScore())
        before = plt.get_fignums()
        assert plot_joint_diagnostics(test, y, n_samples=32) is None
        assert plt.get_fignums() == before

    def test_draws_only_single_target(self):
        draws = torch.randn(80, 40, 1, dtype=F64)
        y = torch.randn(80, 1, dtype=F64)
        fig, axes = plot_joint_diagnostics(SampleJoint(draws), y, n_samples=40, return_axes=True)
        assert not axes["hpd"].axison  # no density: panel says so
        assert "prerank_dependence" not in axes

    def test_target_cap(self):
        gen = torch.Generator().manual_seed(0)
        draws = torch.randn(60, 30, 9, generator=gen, dtype=F64)
        y = torch.randn(60, 9, generator=gen, dtype=F64)
        _, axes = plot_joint_diagnostics(SampleJoint(draws), y, n_samples=30, return_axes=True)
        assert sum(k.startswith("pit_") for k in axes) == 6

    def test_shape_errors(self):
        _, test, _, y = _gaussian_region(MahalanobisScore())
        with pytest.raises(ValueError, match="y must be"):
            plot_joint_diagnostics(test, y[:, :2])
        with pytest.raises(TypeError):
            plot_joint_diagnostics(y, y)


# ── layout and reproducibility (usability pass) ───────────────────────────────


def _overlap(a, b) -> bool:
    return a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1


@pytest.mark.parametrize("k", [1, 2])
def test_corner_legend_stays_outside_the_grid_for_one_or_two_targets(k):
    draws = _gaussian_draws(2000)[:, :k]
    fig, axes = plot_corner_plot(
        draws, true_vals=draws[0], param_names=[f"t{j} [m]" for j in range(k)], return_axes=True
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_box = fig.legends[0].get_window_extent(renderer)
    for i in range(k):
        for j in range(i + 1):
            assert not _overlap(legend_box, axes[i, j].get_tightbbox(renderer)), (i, j)
    fig_box = fig.bbox
    assert legend_box.x0 >= fig_box.x0 - 1 and legend_box.y0 >= fig_box.y0 - 1


def test_joint_diagnostics_seed_makes_the_figure_reproducible():
    _, test, region, y = _gaussian_region(MahalanobisScore(), alpha=[0.5, 0.1])

    def curves(seed: int) -> list:
        fig, axes = plot_joint_diagnostics(
            test, y, region=region, n_samples=32, seed=seed, return_axes=True
        )
        out = [line.get_ydata().copy() for line in axes["hpd"].get_lines()]
        out += [p.get_height() for p in axes["prerank_location"].patches]
        plt.close(fig)
        return out

    torch.manual_seed(5)
    state = torch.get_rng_state()
    first = curves(3)
    assert torch.equal(torch.get_rng_state(), state)  # the caller's RNG is left alone
    torch.manual_seed(99)  # a different global state must not matter
    second = curves(3)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
