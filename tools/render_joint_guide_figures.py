"""Regenerate the images of the three joint-prediction entry pages.

Deterministic (fixed seeds, CPU)::

    pixi run python tools/render_joint_guide_figures.py [--out docs/assets/images/joint/guide]
    pixi run python tools/render_joint_guide_figures.py --only quickstart

Each function below runs the worked example of one page, with the same code as
the page, and saves its figures:

* ``quickstart`` (``docs/guide/joint-quickstart.md``): corner plot of one
  prediction, diagnostics of a good model and of a model that ignores the
  dependence between targets.
* ``mcmc`` (``docs/guide/joint-for-mcmc-users.md``): learned posterior against
  the exact Normal-Inverse-Gamma posterior, and the marginal of two parameters
  with the third integrated out.
* ``sbi`` (``docs/guide/joint-for-sbi-users.md``): two-moons posterior against
  reference draws, and the diagnostics panel over fresh simulations.

The script prints the numbers quoted on the pages.  It takes about three
minutes on a laptop CPU, most of it training the flow of the ``sbi`` example.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.special import gammaln  # noqa: E402

from torchregress.inference import (  # noqa: E402
    NeymanConfidenceSet,
    binned_conditional_coverage,
)
from torchregress.joint import (  # noqa: E402
    GaussianJoint,
    JointConformal,
    JointRegressor,
    LatentRecalibrator,
    LatentScore,
    hpd_calibration,
    latent_calibration_error,
    prerank_histograms,
)
from torchregress.metrics.sbi import (  # noqa: E402
    sbc_uniformity_pvalue,
    simulation_based_calibration_ranks,
    tarp_coverage,
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
# docs/guide/joint-quickstart.md
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


def quickstart(save: Saver) -> None:
    simulate = simulate_quickstart
    torch.manual_seed(0)  # the draws below use torch's global generator

    x, y = simulate(8000, seed=0)  # training data
    x_new, y_new = simulate(2000, seed=1)  # new rows, never seen by fit

    model = JointRegressor("gaussian", device="cpu").fit(x, y)  # 1. fit
    dist = model.predict(x_new)  # 2. one joint distribution per new row
    draws = dist.sample(1000)  # 3. [2000, 1000, 3] plausible target vectors
    region = model.region(x_new, alpha=0.1)  # 4. a 90% prediction region per row
    inside = region.contains(y_new)  # 5. [2000] booleans: is the truth inside?

    print(
        f"rows: {model.n_train_} train, {model.n_val_ + model.n_recal_ + model.n_conformal_} held out"
    )
    print(f"mean of row 0:         {dist.mean()[0].numpy().round(2)}")
    print(f"corr(y1, y2) in row 0: {torch.corrcoef(draws[0].T)[0, 1]:.2f}")
    print(f"joint region covers:   {inside.float().mean():.3f} of the new rows")

    per_target = [model.region(x_new, alpha=0.1, interest=[j]) for j in range(3)]
    each = torch.stack([r.contains(y_new[:, [j]]) for j, r in enumerate(per_target)], dim=1)
    print(f"each 90% interval covers its own target: {each.float().mean(0).numpy().round(3)}")
    print(f"all three intervals cover at once:       {each.all(1).float().mean():.3f}")
    wider = [model.region(x_new, alpha=0.1 / 3, interest=[j]) for j in range(3)]
    each_w = torch.stack([r.contains(y_new[:, [j]]) for j, r in enumerate(wider)], dim=1)
    print(f"all three 96.7% intervals cover at once: {each_w.all(1).float().mean():.3f}")

    fig = plot_corner_plot(
        dist,  # the predictive of the 2000 new rows...
        row=0,  # ...of which we look at the first
        n_samples=20_000,
        true_vals=y_new[0],  # what actually happened
        region=region,  # its 90% prediction region
        param_names=["$y_1$", "$y_2$", "$y_3$"],
        return_figure=True,
    )
    save(fig, "quickstart_corner.png")

    noisy = x_new[:, 0].abs() > 1  # the noisier half of the inputs
    log_volume, _ = region.log_volume()
    print(f"coverage, all rows:  {inside.float().mean():.3f}")
    print(f"coverage, |x1| > 1:  {inside[noisy].float().mean():.3f}")
    print(f"coverage, |x1| <= 1: {inside[~noisy].float().mean():.3f}")
    print(
        f"median log volume:   {log_volume[noisy].median():.2f} (|x1| > 1), "
        f"{log_volume[~noisy].median():.2f} (|x1| <= 1)"
    )

    fig = plot_joint_diagnostics(dist, y_new, x=x_new, region=region, return_figure=True)
    save(fig, "quickstart_diagnostics.png")

    variances = dist.covariance().diagonal(dim1=-2, dim2=-1)
    independent = GaussianJoint.from_covariance(dist.mean(), torch.diag_embed(variances))
    fig = plot_joint_diagnostics(independent, y_new, return_figure=True)
    save(fig, "quickstart_diagnostics_independent.png")

    x_cal, y_cal = simulate(2000, seed=3)  # fresh rows to calibrate both regions on
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

    y_holes = y.clone()
    hide = torch.rand(8000, generator=torch.Generator().manual_seed(2)) < 0.3
    y_holes[hide, 2] = float("nan")  # y3 not measured for 30% of the training rows

    model_h = JointRegressor("gaussian", device="cpu").fit(x, y_holes)
    dist_h = model_h.predict(x_new)
    region_h = model_h.region(x_new, alpha=0.1)
    seen = torch.ones_like(y_new, dtype=torch.bool)
    seen[:1000, 2] = False  # suppose y3 is missing for the first 1000 new rows
    print(f"complete new rows covered:     {region_h.contains(y_new)[1000:].float().mean():.3f}")
    print(
        f"rows without y3 covered:       "
        f"{region_h.contains(y_new, mask=seen)[:1000].float().mean():.3f}"
    )
    print(f"log p(y1, y2) of row 0, mask:     {dist_h.log_prob(y_new, mask=seen)[0]:.4f}")
    print(
        f"log p(y1, y2) of row 0, marginal: {dist_h.marginal([0, 1]).log_prob(y_new[:, :2])[0]:.4f}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# docs/guide/joint-for-mcmc-users.md
# ═══════════════════════════════════════════════════════════════════════════════

T = np.linspace(-1.0, 1.0, 10)
X_DESIGN = np.stack([np.ones_like(T), T], axis=1)  # [10, 2]: intercept a, slope b
A0, B0 = 3.0, 0.5  # sigma^2 ~ InvGamma(3, 0.5);  (a, b) | sigma ~ N(0, sigma^2 I)


def simulate_line(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw (a, b, log sigma) from the prior, then 10 noisy points on the line a + b t."""
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(B0 / rng.gamma(A0, 1.0, size=n))
    beta = sigma[:, None] * rng.standard_normal((n, 2))
    data = beta @ X_DESIGN.T + sigma[:, None] * rng.standard_normal((n, 10))
    theta = np.column_stack([beta, np.log(sigma)])
    return data.astype(np.float32), theta.astype(np.float32)


def exact_posterior(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Normal-Inverse-Gamma posterior of each row of data [n, 10] (conjugate)."""
    data = data.astype(np.float64)
    prec = np.eye(2) + X_DESIGN.T @ X_DESIGN
    cov = np.linalg.inv(prec)
    loc = data @ X_DESIGN @ cov  # [n, 2]
    shape = A0 + X_DESIGN.shape[0] / 2
    rate = B0 + 0.5 * ((data**2).sum(1) - np.einsum("ni,ij,nj->n", loc, prec, loc))
    return loc, cov, shape, rate


def exact_log_prob(theta: np.ndarray, data: np.ndarray) -> np.ndarray:
    """log p(a, b, log sigma | data) for theta [n, 3]."""
    loc, cov, shape, rate = exact_posterior(data)
    beta, s = theta[:, :2].astype(np.float64), theta[:, 2].astype(np.float64)
    sigma2 = np.exp(2 * s)
    r = beta - loc
    quad = np.einsum("ni,ij,nj->n", r, np.linalg.inv(cov), r) / sigma2
    log_normal = -0.5 * quad - np.log(2 * np.pi * sigma2) - 0.5 * np.log(np.linalg.det(cov))
    log_inv_gamma = (
        shape * np.log(rate) - gammaln(shape) - (shape + 1) * np.log(sigma2) - rate / sigma2
    )
    return log_normal + log_inv_gamma + np.log(2.0) + 2 * s  # + log |d sigma^2 / d log sigma|


def exact_marginal_log_prob(ab: np.ndarray, data: np.ndarray) -> np.ndarray:
    """log p(a, b | data) with log sigma integrated out: a bivariate Student t."""
    loc, cov, shape, rate = exact_posterior(data)
    nu, scale = 2 * shape, (rate / shape)[:, None, None] * cov  # [n, 2, 2]
    r = ab.astype(np.float64) - loc
    quad = np.einsum("ni,nij,nj->n", r, np.linalg.inv(scale), r)
    log_det = np.log(np.linalg.det(scale))
    return (
        gammaln((nu + 2) / 2) - gammaln(nu / 2) - np.log(nu * np.pi) - 0.5 * log_det
        - (nu + 2) / 2 * np.log1p(quad / nu)
    )  # fmt: skip


def exact_sample(data: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    """Exact posterior draws [n, n_samples, 3]."""
    rng = np.random.default_rng(seed)
    loc, cov, shape, rate = exact_posterior(data)
    sigma2 = rate[:, None] / rng.gamma(shape, 1.0, size=(data.shape[0], n_samples))
    z = rng.standard_normal((data.shape[0], n_samples, 2)) @ np.linalg.cholesky(cov).T
    beta = loc[:, None] + np.sqrt(sigma2)[..., None] * z
    return np.concatenate([beta, 0.5 * np.log(sigma2)[..., None]], axis=-1)


def log_target(th: np.ndarray, y: np.ndarray) -> float:
    """Unnormalised log posterior of (a, b, log sigma): what you would give a sampler."""
    a, b, s = th
    sigma2 = math.exp(2 * s)
    log_lik = -0.5 * ((y - a - b * T) ** 2).sum() / sigma2 - len(y) * s
    log_prior = -0.5 * (a * a + b * b) / sigma2 - 2 * s  # (a, b) | sigma ~ N(0, sigma^2 I)
    log_prior += -(A0 + 1) * 2 * s - B0 / sigma2 + 2 * s  # InvGamma(sigma^2), Jacobian
    return log_lik + log_prior


def mcmc(save: Saver) -> None:
    simulate = simulate_line
    torch.manual_seed(0)

    data, theta = simulate(20_000, seed=0)  # 20,000 (data, parameters) pairs
    data_obs, theta_obs = simulate(2_000, seed=1)  # fresh simulations to check against

    gaussian = JointRegressor("gaussian", device="cpu").fit(data, theta)
    mixture = JointRegressor("mixture", device="cpu").fit(data, theta)

    exact = exact_log_prob(theta_obs, data_obs)
    for name, model in (("gaussian", gaussian), ("mixture", mixture)):
        post = model.predict(data_obs)  # 2,000 amortised posteriors in one call
        gap = exact - post.log_prob(theta_obs).double().numpy()  # log p - log q at the truth
        hpd = hpd_calibration(post.hpd_level(theta_obs))
        ranks = simulation_based_calibration_ranks(post.sample(255), theta_obs)
        sbc = sbc_uniformity_pvalue(ranks)
        print(
            f"{name:<9} gap {gap.mean():.3f} +- {gap.std() / np.sqrt(gap.size):.3f} nats, "
            f"HPD KS p = {float(hpd.p_value):.2g}, SBC min p = {sbc['min_p_value']:.2g}"
        )

    row = 17
    post = mixture.predict(data_obs)
    reference = exact_sample(data_obs[row : row + 1], 20_000, seed=2)[0]  # [20000, 3]
    for name, model in (("gaussian", gaussian), ("mixture", mixture)):
        log_q = model.predict(data_obs[row : row + 1]).log_prob(reference[None])
        log_p = exact_log_prob(reference, np.repeat(data_obs[row : row + 1], 20_000, axis=0))
        print(
            f"{name:<9} KL(exact || learned), row {row}: {(log_p - log_q[0].double().numpy()).mean():.3f} nats"
        )
    fig = plot_corner_plot(
        post,
        row=row,
        n_samples=20_000,
        label="Learned (mixture head)",
        others={
            "Exact posterior": reference,
            "Learned (Gaussian head)": gaussian.predict(data_obs),
        },
        true_vals=theta_obs[row],
        param_names=["$a$", "$b$", r"$\log\sigma$"],
        return_figure=True,
    )
    save(fig, "mcmc_corner_overlay.png")

    ab = post.marginal([0, 1])  # (a, b) with log sigma integrated out
    gap_ab = (
        exact_marginal_log_prob(theta_obs[:, :2], data_obs)
        - ab.log_prob(theta_obs[:, :2]).double().numpy()
    )
    print(
        f"marginal of (a, b): gap {gap_ab.mean():.3f} +- {gap_ab.std() / np.sqrt(gap_ab.size):.3f} nats"
    )
    region_ab = mixture.region(data_obs, alpha=0.1, interest=[0, 1])
    print(f"90% region for (a, b) covers {region_ab.contains(theta_obs[:, :2]).float().mean():.3f}")
    fig = plot_corner_plot(
        ab,
        row=row,
        n_samples=20_000,
        label="Learned marginal",
        others={"Exact marginal (Student t)": reference[:, :2]},
        region=region_ab,
        true_vals=theta_obs[row, :2],
        param_names=["$a$", "$b$"],
        return_figure=True,
    )
    save(fig, "mcmc_marginal_ab.png")

    rng = np.random.default_rng(3)
    current = theta_obs[row].astype(np.float64)
    log_cur, chain = log_target(current, data_obs[row]), []
    for step in range(60_000):  # random-walk Metropolis; keep every 25th draw after burn-in
        proposal = current + 0.15 * rng.standard_normal(3)
        log_prop = log_target(proposal, data_obs[row])
        if math.log(rng.random()) < log_prop - log_cur:
            current, log_cur = proposal, log_prop
        if step >= 10_000 and step % 25 == 0:
            chain.append(current.copy())
    draws = np.asarray(chain)  # [2000, 3] from the sampler you trust
    for name, model in (("gaussian", gaussian), ("mixture", mixture)):
        q = model.predict(data_obs[row : row + 1])  # the learned posterior of this data set
        levels = q.hpd_level(draws[None], n_samples=4096)[0]  # where each draw sits in q
        print(
            f"{name:<9} sampler draws vs learned: HPD KS p = {float(hpd_calibration(levels).p_value):.2g}"
        )
    print(f"sampler mean {draws.mean(0).round(3)}, exact mean {reference.mean(0).round(3)}")
    print(f"sampler sd   {draws.std(0).round(3)}, exact sd   {reference.std(0).round(3)}")


# ═══════════════════════════════════════════════════════════════════════════════
# docs/guide/joint-for-sbi-users.md
# ═══════════════════════════════════════════════════════════════════════════════


def simulate_two_moons(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Two moons: theta ~ U(-1, 1)^2, x in R^2 (Greenberg et al. 2019)."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-1.0, 1.0, size=(n, 2))
    a = rng.uniform(-math.pi / 2, math.pi / 2, size=n)
    r = rng.normal(0.1, 0.01, size=n)
    p = np.column_stack([r * np.cos(a) + 0.25, r * np.sin(a)])
    shift = np.column_stack([-np.abs(theta.sum(1)), theta[:, 1] - theta[:, 0]]) / math.sqrt(2)
    return (p + shift).astype(np.float32), theta.astype(np.float32)


def two_moons_reference(x_obs: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    """Exact posterior draws for one observation x_obs [2], by inverting the simulator."""
    rng = np.random.default_rng(seed)
    kept, total = [], 0
    while total < n_samples:
        a = rng.uniform(-math.pi / 2, math.pi / 2, size=n_samples)
        r = rng.normal(0.1, 0.01, size=n_samples)
        p = np.column_stack([r * np.cos(a) + 0.25, r * np.sin(a)])
        u = (x_obs - p) * math.sqrt(2)  # u = (-|theta_1 + theta_2|, theta_2 - theta_1)
        s = -u[:, 0] * rng.choice([-1.0, 1.0], size=n_samples)  # the sign is not identified
        th = np.column_stack([(s - u[:, 1]) / 2, (s + u[:, 1]) / 2])
        th = th[(u[:, 0] <= 0) & (np.abs(th) <= 1).all(1)]  # inside the prior box
        kept.append(th)
        total += len(th)
    return np.concatenate(kept)[:n_samples].astype(np.float32)


def sbi(save: Saver) -> None:
    simulate, reference_posterior = simulate_two_moons, two_moons_reference
    torch.manual_seed(0)

    x_train, theta_train = simulate(20_000, seed=0)
    npe = JointRegressor("flow", device="cpu").fit(x_train, theta_train)  # data -> parameters
    names = [r"$\theta_1$", r"$\theta_2$"]
    print(
        f"simulations: {npe.n_train_} train, {npe.n_val_} early stopping, "
        f"{npe.n_recal_} recalibration, {npe.n_conformal_} region calibration"
    )

    x_test, theta_test = simulate(2_000, seed=2)  # fresh simulations
    truth = torch.as_tensor(theta_test)
    post = npe.predict(x_test)  # 2,000 amortised posteriors
    draws = post.sample(256)  # [2000, 256, 2]: the layout SBC and TARP expect
    sbc = sbc_uniformity_pvalue(simulation_based_calibration_ranks(draws, truth))
    tarp = tarp_coverage(draws, truth, torch.rand(2_000, 2) * 2 - 1)  # references from the prior
    hpd = hpd_calibration(post.hpd_level(truth))  # expected coverage of the HPD regions
    print(
        f"SBC min p {sbc['min_p_value']:.2g} (Bonferroni {sbc['bonferroni_min_p_value']:.2g}), "
        f"TARP error {float(tarp['tarp_error']):.3f}, "
        f"HPD KS {float(hpd.ks_statistic):.3f} (p {float(hpd.p_value):.2g})"
    )
    region = npe.region(x_test, alpha=[0.32, 0.1, 0.05])
    fig = plot_joint_diagnostics(
        post, truth, x=x_test, region=region, target_names=names, return_figure=True
    )
    save(fig, "sbi_diagnostics.png")

    x_cal, theta_cal = simulate(5_000, seed=6)  # fresh simulations for the regions
    npe.calibrate(x_cal, theta_cal)  # replaces the held-out calibration split
    default = npe.region(x_test, alpha=0.1)
    latent = JointConformal(LatentScore(), alpha=0.1).calibrate(npe.predict(x_cal), theta_cal)
    for name, r in ((f"default ({default.score.name})", default), ("latent", latent.region(post))):
        print(
            f"{name:<13} 90% region: coverage {r.contains(theta_test).float().mean():.3f}, "
            f"median log area {r.log_volume()[0].median():.2f}"
        )
    x_more, theta_more = simulate(20_000, seed=5)
    covered = npe.region(x_more, alpha=[0.32, 0.1, 0.05]).contains(theta_more)
    print(f"default regions, 20,000 simulations: {covered.float().mean(0).numpy().round(3)}")

    x_obs = np.zeros((1, 2), dtype=np.float32)  # the observation x = (0, 0)
    fig = plot_corner_plot(
        npe.predict(x_obs),
        n_samples=20_000,
        label="NPE (JointRegressor, flow)",
        others={"Reference posterior": reference_posterior(x_obs[0], 20_000, seed=1)},
        region=npe.region(x_obs, alpha=0.1),
        param_names=names,
        return_figure=True,
    )
    save(fig, "sbi_corner_two_moons.png")

    x_rc, theta_rc = simulate(2_000, seed=3)
    recal = LatentRecalibrator().fit(npe.predict(x_rc), torch.as_tensor(theta_rc))
    post_rc = recal.transform(post)
    hpd_rc = hpd_calibration(post_rc.hpd_level(truth))
    print(
        f"L-ECE {latent_calibration_error(post, truth):.3f} -> {latent_calibration_error(post_rc, truth):.3f}, "
        f"HPD KS {float(hpd.ks_statistic):.3f} -> {float(hpd_rc.ks_statistic):.3f} (p {float(hpd_rc.p_value):.2g}), "
        f"log score {post.log_prob(truth).mean():.3f} -> {post_rc.log_prob(truth).mean():.3f}"
    )
    for name, p in (("before", post), ("after", post_rc)):
        pr = prerank_histograms(p.sample(256), truth)
        print(
            f"pre-rank Holm p {name:<6}",
            {k: round(float(v), 4) for k, v in zip(pr.names, pr.p_values_holm)},
        )

    grid = torch.linspace(-1.0, 1.0, 101)

    def profile_statistic(pred: Any, theta_1: torch.Tensor) -> torch.Tensor:
        """lambda(theta_1; x) = -max over theta_2 of log q(theta_1, theta_2 | x), per row."""
        n, k = len(theta_1), grid.numel()
        candidates = torch.stack([theta_1[:, None].expand(n, k), grid.expand(n, k)], dim=-1)
        return -pred.log_prob(candidates).amax(dim=1)  # log_prob of [n, k, 2] is [n, k]

    x_nc, theta_nc = simulate(10_000, seed=4)  # calibration simulations
    theta_nc = torch.as_tensor(theta_nc)
    lam_nc = profile_statistic(npe.predict(x_nc), theta_nc[:, 0])
    cs = NeymanConfidenceSet.calibrate(theta_nc, lam_nc, alpha=0.1)  # C(theta_1, theta_2)
    c_max = cs.critical_value(torch.cartesian_prod(grid, grid)).reshape(101, 101).amax(dim=1)

    def critical(theta_1: torch.Tensor) -> torch.Tensor:  # max over theta_2, on the grid
        return c_max[((theta_1 + 1) * 50).round().long().clamp(0, 100)]

    neyman = profile_statistic(post, truth[:, 0]) <= critical(truth[:, 0])
    q = post.sample(1000)[..., 0].quantile(torch.tensor([0.05, 0.95]), dim=1)
    credible = (q[0] <= truth[:, 0]) & (truth[:, 0] <= q[1])  # central 90% credible interval
    print(
        f"coverage of theta_1: Neyman {neyman.float().mean():.3f}, credible {credible.float().mean():.3f}"
    )
    for name, cov in (("Neyman", neyman), ("credible", credible)):
        by_theta_1 = binned_conditional_coverage(truth, cov, nominal=0.9, bins=4)[0]
        print(
            f"{name:<8} by theta_1 quarter: {by_theta_1.coverage.numpy().round(3)}, "
            f"flagged {by_theta_1.under_covered.tolist()}"
        )
    lam_obs = profile_statistic(npe.predict(np.repeat(x_obs, 101, axis=0)), grid)
    accepted = grid[lam_obs <= c_max]
    print(f"90% confidence set for theta_1 at x = (0, 0): {accepted.numpy().round(2)}")


CASES = {"quickstart": quickstart, "mcmc": mcmc, "sbi": sbi}


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
