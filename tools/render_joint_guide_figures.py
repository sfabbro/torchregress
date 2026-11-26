"""Regenerate the images of the joint guide, ``docs/guide/joint-regression.md``.

Deterministic (fixed seeds, CPU)::

    pixi run python tools/render_joint_guide_figures.py [--out docs/assets/images/joint/guide]
    pixi run python tools/render_joint_guide_figures.py --only example

Each function runs the guide's code and saves its figures:

* ``example`` (section 3): corner plot of one prediction (also shown in section 1),
  diagnostics of the model and of a model that ignores the dependence between targets.
* ``npe`` (section 5): learned posterior of the mixture head against the exact
  Normal-Inverse-Gamma posterior of one simulated data set.

About 30 s on a laptop CPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from torchregress.joint import (  # noqa: E402
    GaussianJoint,
    JointConformal,
    JointRegressor,
    LatentScore,
)
from torchregress.viz import plot_corner_plot, plot_joint_diagnostics  # noqa: E402

DEFAULT_OUT = Path("docs/assets/images/joint/guide")
Saver = Callable[[Any, str], None]


def _saver(out: Path) -> Saver:
    def save(fig: Any, name: str) -> None:
        path = out / name
        # Fixed metadata keeps the PNG bytes reproducible.
        fig.savefig(path, dpi=110, bbox_inches="tight", metadata={"Software": None})
        plt.close(fig)
        print(path)

    return save


# ═══════════════════════════════════════════════════════════════════════════════
# section 3: the worked example
# ═══════════════════════════════════════════════════════════════════════════════


def simulate_quickstart(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Two features, three targets; y1 and y2 share most of their noise."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 2, generator=g) * 4 - 2  # uniform on [-2, 2]^2
    noise = 0.3 + 0.2 * x[:, 0].abs()  # noisier at the edges of x1
    e = torch.randn(n, 3, generator=g)
    y = torch.empty(n, 3)
    y[:, 0] = torch.sin(x[:, 0]) + noise * e[:, 0]
    y[:, 1] = 0.5 * x[:, 1] + noise * (0.9 * e[:, 0] + 0.44 * e[:, 1])
    y[:, 2] = x[:, 0] * x[:, 1] + 0.3 * e[:, 2]
    return x, y


def example(save: Saver) -> None:
    simulate = simulate_quickstart
    torch.manual_seed(0)  # the draws below use torch's global generator

    x, y = simulate(8000, seed=0)  # training data
    x_new, y_new = simulate(2000, seed=1)  # new rows, never seen by fit

    model = JointRegressor("gaussian", device="cpu").fit(x, y)  # 1. fit
    dist = model.predict(x_new)  # 2. one joint distribution per new row
    draws = dist.sample(1000)  # 3. [2000, 1000, 3] plausible target vectors
    region = model.region(x_new, alpha=0.1)  # 4. a 90% region per row
    inside = region.contains(y_new)  # 5. [2000] booleans: is the truth inside?
    n_held = model.n_val_ + model.n_recal_ + model.n_conformal_
    print(f"rows: {model.n_train_} train, {n_held} held out")
    print(f"corr(y1, y2) in row 0: {torch.corrcoef(draws[0].T)[0, 1]:.2f}")
    print(f"joint region covers:   {inside.float().mean():.3f} of the new rows")

    fig = plot_corner_plot(
        dist,
        row=0,
        n_samples=20_000,
        true_vals=y_new[0],
        region=region,
        param_names=["$y_1$", "$y_2$", "$y_3$"],
        return_figure=True,
    )
    save(fig, "example_corner.png")

    per_target = [model.region(x_new, alpha=0.1, interest=[j]) for j in range(3)]
    each = torch.stack([r.contains(y_new[:, [j]]) for j, r in enumerate(per_target)], dim=1)
    print(f"each 90% interval covers its own target: {each.float().mean(0).numpy().round(3)}")
    print(f"all three intervals cover at once:       {each.all(1).float().mean():.3f}")
    noisy = x_new[:, 0].abs() > 1
    print(f"joint region, |x1| > 1:  {inside[noisy].float().mean():.3f}")
    print(f"joint region, |x1| <= 1: {inside[~noisy].float().mean():.3f}")

    fig = plot_joint_diagnostics(dist, y_new, x=x_new, region=region, return_figure=True)
    save(fig, "example_diagnostics.png")

    variances = dist.covariance().diagonal(dim1=-2, dim2=-1)
    independent = GaussianJoint.from_covariance(dist.mean(), torch.diag_embed(variances))
    fig = plot_joint_diagnostics(independent, y_new, return_figure=True)
    save(fig, "example_diagnostics_independent.png")

    x_cal, y_cal = simulate(2000, seed=3)
    cal = model.predict(x_cal)
    cal_independent = GaussianJoint.from_covariance(
        cal.mean(), torch.diag_embed(cal.covariance().diagonal(dim1=-2, dim2=-1))
    )
    for name, p_cal, p_new in (("full", cal, dist), ("independent", cal_independent, independent)):
        r = JointConformal(LatentScore(), alpha=0.1).calibrate(p_cal, y_cal).region(p_new)
        print(
            f"{name:<12} coverage {r.contains(y_new).float().mean():.3f}, "
            f"median log volume {r.log_volume()[0].median():.2f}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# section 5: a posterior with an exact answer
# ═══════════════════════════════════════════════════════════════════════════════

T = np.linspace(-1.0, 1.0, 10)
X_LINE = np.stack([np.ones_like(T), T], axis=1)  # [10, 2]: intercept a, slope b


def simulate_line(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(a, b, log sigma) from the prior, then 10 noisy points on the line a + b t."""
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(0.5 / rng.gamma(3.0, 1.0, size=n))
    beta = sigma[:, None] * rng.standard_normal((n, 2))
    data = beta @ X_LINE.T + sigma[:, None] * rng.standard_normal((n, 10))
    return data.astype(np.float32), np.column_stack([beta, np.log(sigma)]).astype(np.float32)


def exact_sample(data: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    """Exact posterior draws [n, n_samples, 3]."""
    rng = np.random.default_rng(seed)
    data = data.astype(np.float64)
    prec = np.eye(2) + X_LINE.T @ X_LINE
    loc = data @ X_LINE @ np.linalg.inv(prec)
    rate = 0.5 + 0.5 * ((data**2).sum(1) - np.einsum("ni,ij,nj->n", loc, prec, loc))
    sigma2 = rate[:, None] / rng.gamma(8.0, 1.0, size=(len(data), n_samples))
    z = rng.standard_normal((len(data), n_samples, 2)) @ np.linalg.cholesky(np.linalg.inv(prec)).T
    beta = loc[:, None] + np.sqrt(sigma2)[..., None] * z
    return np.concatenate([beta, 0.5 * np.log(sigma2)[..., None]], axis=-1)


def npe(save: Saver) -> None:
    data, theta = simulate_line(20_000, seed=0)
    data_obs, theta_obs = simulate_line(2_000, seed=1)
    mixture = JointRegressor("mixture", device="cpu").fit(data, theta)
    torch.manual_seed(0)
    fig = plot_corner_plot(
        mixture.predict(data_obs),
        row=17,
        n_samples=20_000,
        label="Learned (mixture head)",
        others={"Exact posterior": exact_sample(data_obs[17:18], 20_000, seed=2)[0]},
        true_vals=theta_obs[17],
        param_names=["$a$", "$b$", r"$\log\sigma$"],
        return_figure=True,
    )
    save(fig, "npe_corner_exact.png")


CASES = {"example": example, "npe": npe}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only", nargs="*", choices=sorted(CASES), default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcdefaults()
    for name in args.only or list(CASES):
        print(f"== {name}")
        CASES[name](_saver(args.out))


if __name__ == "__main__":
    main()
