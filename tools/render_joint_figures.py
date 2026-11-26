"""Regenerate the joint-plot images of ``docs/methods/visualization.md``.

Deterministic (fixed seeds, CPU, float64)::

    pixi run python tools/render_joint_figures.py [--out docs/assets/images/joint]

Writes one PNG per case: corner plots of a correlated Gaussian, a banana and
bimodal predictive, a heavy-tailed predictive, six targets with a conformal
region and true values, two overlaid predictives, twelve targets with automatic
pair selection, and the one-call joint diagnostics panel.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from torchregress.joint import (  # noqa: E402
    GaussianJoint,
    JointConformal,
    MahalanobisScore,
)
from torchregress.viz import plot_corner_plot, plot_joint_diagnostics  # noqa: E402

DEFAULT_OUT = Path("docs/assets/images/joint")
_F64 = torch.float64


def _seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _save(fig: plt.Figure, out: Path, name: str) -> Path:
    path = out / name
    # Fixed metadata keeps the PNG bytes reproducible.
    fig.savefig(path, dpi=130, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return path


def correlated_gaussian(out: Path) -> Path:
    _seed(0)
    cov = torch.tensor([[1.0, 0.8, -0.3], [0.8, 1.5, -0.5], [-0.3, -0.5, 0.6]], dtype=_F64)
    pred = GaussianJoint.from_covariance(torch.tensor([[0.3, 1.2, -0.4]], dtype=_F64), cov)
    fig = plot_corner_plot(
        pred,
        n_samples=20_000,
        param_names=["$y_1$", "$y_2$ [m]", "$y_3$"],
        true_vals=torch.tensor([0.5, 1.0, -0.2]),
        title="Correlated Gaussian predictive",
        return_figure=True,
    )
    return _save(fig, out, "corner_gaussian.png")


def banana_bimodal(out: Path) -> Path:
    rng = np.random.default_rng(1)
    n = 30_000
    a = rng.normal(0.0, 1.0, n)
    b = 0.6 * a**2 + rng.normal(0.0, 0.35, n) - 0.6
    mode = rng.random(n) < 0.4
    c = np.where(mode, rng.normal(-1.5, 0.4, n), rng.normal(1.2, 0.5, n)) + 0.3 * a
    draws = np.stack([a, b, c], axis=1)
    fig = plot_corner_plot(
        draws,
        param_names=["$a$", "$b$", "$c$"],
        interval="hpd",
        title="Banana and bimodal predictive (HPD intervals)",
        return_figure=True,
    )
    return _save(fig, out, "corner_banana_bimodal.png")


def heavy_tailed(out: Path) -> Path:
    rng = np.random.default_rng(2)
    n, nu = 50_000, 2.5
    cov = np.array([[1.0, 0.6, 0.0], [0.6, 1.0, 0.4], [0.0, 0.4, 1.0]])
    z = rng.multivariate_normal(np.zeros(3), cov, size=n)
    w = rng.chisquare(nu, size=n) / nu
    draws = z / np.sqrt(w)[:, None] + np.array([10.0, -3.0, 0.0])
    fig = plot_corner_plot(
        draws,
        param_names=["flux [mJy]", "index", "offset [arcsec]"],
        level_convention="sigma",
        title="Heavy-tailed predictive (Student t, ν = 2.5; sigma levels)",
        return_figure=True,
    )
    return _save(fig, out, "corner_heavy_tailed.png")


def _gaussian_batch(n: int, d: int, seed: int) -> tuple[GaussianJoint, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=gen, dtype=_F64)
    a = torch.randn(d, d, generator=gen, dtype=_F64) / math.sqrt(d)
    cov = a @ a.T + 0.3 * torch.eye(d, dtype=_F64)
    mean = torch.stack([torch.sin(x[:, 0] * (k + 1) / d) + x[:, 1] for k in range(d)], dim=1)
    chol = torch.linalg.cholesky(cov)
    y = mean + torch.randn(n, d, generator=gen, dtype=_F64) @ chol.T
    return GaussianJoint(mean, chol), y


def region_and_truth(out: Path) -> Path:
    _seed(3)
    d = 6
    pred, y = _gaussian_batch(1500, d, seed=3)
    cal = GaussianJoint(pred.loc[:1000], pred.scale_tril[:1000])
    test = GaussianJoint(pred.loc[1000:], pred.scale_tril[1000:])
    region = JointConformal(MahalanobisScore(), alpha=0.1).calibrate(cal, y[:1000]).region(test)
    fig = plot_corner_plot(
        test,
        row=0,
        n_samples=20_000,
        region=region,
        true_vals=y[1000],
        param_names=[f"$y_{{{k + 1}}}$" for k in range(d)],
        title="Six targets: 90% conformal region (Mahalanobis) and the observed value",
        return_figure=True,
    )
    return _save(fig, out, "corner_region_truth.png")


def two_predictives(out: Path) -> Path:
    _seed(4)
    prior = GaussianJoint.from_covariance(
        torch.zeros(1, 3, dtype=_F64), torch.eye(3, dtype=_F64) * 2.0
    )
    post_cov = torch.tensor([[0.20, 0.12, 0.0], [0.12, 0.25, -0.08], [0.0, -0.08, 0.15]])
    posterior = GaussianJoint.from_covariance(
        torch.tensor([[0.8, -0.5, 1.1]], dtype=_F64), post_cov.to(_F64)
    )
    fig = plot_corner_plot(
        posterior,
        label="Posterior predictive",
        others={"Prior predictive": prior},
        n_samples=20_000,
        param_names=["$y_1$", "$y_2$", "$y_3$"],
        title="Two predictives overlaid",
        return_figure=True,
    )
    return _save(fig, out, "corner_overlay.png")


def many_targets(out: Path) -> Path:
    rng = np.random.default_rng(5)
    d, n = 12, 20_000
    cov = np.eye(d)
    for i, j, r in [(0, 3, 0.85), (2, 7, -0.75), (4, 5, 0.6), (9, 11, 0.5), (1, 10, 0.35)]:
        cov[i, j] = cov[j, i] = r
    draws = rng.multivariate_normal(np.zeros(d), cov, size=n)
    fig = plot_corner_plot(
        draws,
        param_names=[f"$y_{{{k + 1}}}$" for k in range(d)],
        max_dims=6,
        title="Twelve targets: automatic choice of the most dependent pairs",
        return_figure=True,
    )
    return _save(fig, out, "corner_many_targets.png")


def diagnostics(out: Path) -> Path:
    _seed(6)
    d = 3
    pred, y = _gaussian_batch(2000, d, seed=6)
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(2000, 2, generator=gen, dtype=_F64)
    # Heteroscedastic truth; the head is over-confident where x[:, 0] > 0.
    scale = torch.exp(0.35 * x[:, 1]).view(-1, 1, 1)
    y = pred.loc + (y - pred.loc) * scale[:, :, 0]
    shrink = torch.where(x[:, :1] > 0, 0.7, 1.0).unsqueeze(-1)
    model = GaussianJoint(pred.loc, pred.scale_tril * scale * shrink)
    cal = GaussianJoint(model.loc[:1000], model.scale_tril[:1000])
    test = GaussianJoint(model.loc[1000:], model.scale_tril[1000:])
    region = (
        JointConformal(MahalanobisScore(), alpha=[0.5, 0.2, 0.1])
        .calibrate(cal, y[:1000])
        .region(test)
    )
    fig = plot_joint_diagnostics(
        test,
        y[1000:],
        x=x[1000:],
        region=region,
        target_names=["$y_1$", "$y_2$", "$y_3$"],
        title="Joint diagnostics of an over-confident Gaussian head",
        return_figure=True,
    )
    return _save(fig, out, "joint_diagnostics.png")


CASES = {
    "gaussian": correlated_gaussian,
    "banana": banana_bimodal,
    "heavy": heavy_tailed,
    "region": region_and_truth,
    "overlay": two_predictives,
    "many": many_targets,
    "diagnostics": diagnostics,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only", nargs="*", choices=sorted(CASES), default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcdefaults()
    for name in args.only or list(CASES):
        print(CASES[name](args.out))


if __name__ == "__main__":
    main()
