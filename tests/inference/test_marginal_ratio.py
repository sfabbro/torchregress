"""Marginal neural ratio estimation (``MarginalRatioEstimator``) and ``RatioPosterior``.

Exact answers come from a linear-Gaussian simulator with nuisance parameters,

    theta ~ N(0, I_P),   x = A theta + sigma * eps,   eps ~ N(0, I_D),

with ``D = 4`` observations, two parameters of interest ``(0, 1)`` and ``P - 2``
nuisances whose loadings are scaled by ``1 / sqrt(P - 2)`` (the nuisance noise in
``x`` has the same total size for every ``P``).  The posterior is Gaussian,
``Sigma = (I + A^T A / sigma^2)^-1``, ``mu = Sigma A^T x / sigma^2``, and the
marginal of the interest pair is ``N(mu_I, Sigma_II)``, so the exact marginal log
ratio is ``log N(theta_I; mu_I, Sigma_II) - log N(theta_I; 0, I)``.

Budget of the trained estimators: 10 000 simulations, batch 256, ``K = 128``
in-batch negatives (CNRE, gamma = 1), at most 12 epochs with early stopping
(about 10 s each on a laptop CPU).  The tolerances below are stated against that
budget.
"""

from __future__ import annotations

import math
import statistics
import time
import warnings
from functools import cache

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
import torch
from matplotlib.figure import Figure
from torch import Tensor

from torchregress.inference import MarginalRatioEstimator, RatioPosterior
from torchregress.joint import DensityScore, JointConformal, as_joint, hpd_calibration
from torchregress.losses import NeuralRatioLoss, in_batch_negative_index
from torchregress.metrics import (
    sbc_uniformity_pvalue,
    simulation_based_calibration_ranks,
    tarp_coverage,
)
from torchregress.viz import plot_corner_plot

D = 4
SIGMA = 0.5
INTEREST = (0, 1)
N_TRAIN = 10_000
F64 = torch.float64


def _design(p: int) -> Tensor:
    g = torch.Generator().manual_seed(1)
    a_interest = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7, -0.5], [0.3, 0.6]])
    a_nuisance = torch.randn(D, p - 2, generator=g) / math.sqrt(p - 2)
    return torch.cat([a_interest, a_nuisance], dim=1)


def _simulate(p: int, n: int, seed: int) -> tuple[Tensor, Tensor]:
    g = torch.Generator().manual_seed(seed)
    theta = torch.randn(n, p, generator=g)
    x = theta @ _design(p).T + SIGMA * torch.randn(n, D, generator=g)
    return theta, x


def _exact_marginal(p: int, x: Tensor) -> tuple[Tensor, Tensor]:
    """Exact posterior mean ``[n, 2]`` and covariance ``[2, 2]`` of the interest pair."""
    a = _design(p).to(F64)
    cov = torch.linalg.inv(torch.eye(p, dtype=F64) + a.T @ a / SIGMA**2)
    mean = x.to(F64) @ (cov @ a.T / SIGMA**2).T
    idx = list(INTEREST)
    return mean[:, idx], cov[idx][:, idx]


def _exact_log_ratio(p: int, theta: Tensor, x: Tensor) -> Tensor:
    """Exact ``log r`` of candidates: ``theta [n, S, 2]`` and ``x [n, D]`` -> ``[n, S]``."""
    mean, cov = _exact_marginal(p, x)
    post = torch.distributions.MultivariateNormal(mean.unsqueeze(1), cov)
    th = theta.to(F64)
    return post.log_prob(th) - _prior_log_prob(th)


def _prior_log_prob(theta: Tensor) -> Tensor:
    return torch.distributions.Normal(0.0, 1.0).log_prob(theta).sum(-1)


def _prior_bank(m: int, seed: int) -> Tensor:
    return torch.randn(m, 2, generator=torch.Generator().manual_seed(seed))


@cache
def _trained(p: int) -> MarginalRatioEstimator:
    theta, x = _simulate(p, N_TRAIN, seed=0)
    torch.manual_seed(0)
    est = MarginalRatioEstimator(p, [INTEREST], observation_dim=D)
    return est.fit(theta, x, max_epochs=12, patience=4, batch_size=256, lr=2e-3, seed=0)


# ═══════════════════════════════════════════════════════════════════════════════
# in-batch negatives
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(("n", "k"), [(2, 1), (7, 3), (64, 32), (65, 64)])
def test_in_batch_negatives_never_use_the_rows_own_theta(n: int, k: int) -> None:
    g = torch.Generator().manual_seed(n + k)
    for _ in range(20):
        index = in_batch_negative_index(n, k, generator=g)
        assert index.shape == (n, k) and index.dtype == torch.long
        rows = torch.arange(n).unsqueeze(1)
        assert not bool((index == rows).any())
        # K distinct partners per row, and every row is a negative of exactly K rows.
        assert all(len(set(r.tolist())) == k for r in index)
        assert torch.equal(torch.bincount(index.flatten(), minlength=n), torch.full((n,), k))


def test_in_batch_negatives_are_random_partners() -> None:
    g = torch.Generator().manual_seed(0)
    n, k = 16, 1
    counts = torch.zeros(n, n)
    for _ in range(4000):
        index = in_batch_negative_index(n, k, generator=g)
        counts[torch.arange(n), index[:, 0]] += 1
    off_diagonal = counts[~torch.eye(n, dtype=torch.bool)]
    expected = 4000 / (n - 1)
    # Each partner j != i is equally likely: 4000 / 15 = 267 draws, sd ~ 16.
    assert float((off_diagonal - expected).abs().max()) < 6 * math.sqrt(expected)


def test_in_batch_negatives_reject_bad_sizes() -> None:
    with pytest.raises(ValueError, match="at least two rows"):
        in_batch_negative_index(1, 1)
    with pytest.raises(ValueError, match="n_negatives"):
        in_batch_negative_index(4, 4)
    with pytest.raises(ValueError, match="n_negatives"):
        in_batch_negative_index(4, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# estimator: shapes, marginals, the loss
# ═══════════════════════════════════════════════════════════════════════════════


def test_marginal_helpers() -> None:
    assert MarginalRatioEstimator.univariate([2, 0]) == [(2,), (0,)]
    assert MarginalRatioEstimator.pairs([0, 1, 3]) == [(0, 1), (0, 3), (1, 3)]
    assert MarginalRatioEstimator.jointly([0, 3]) == [(0, 3)]


def test_constructor_validation() -> None:
    with pytest.raises(ValueError, match="observation_dim"):
        MarginalRatioEstimator(3, [(0,)])
    with pytest.raises(ValueError, match="outside"):
        MarginalRatioEstimator(3, [(0, 3)], observation_dim=2)
    with pytest.raises(ValueError, match="distinct"):
        MarginalRatioEstimator(3, [(0, 0)], observation_dim=2)
    with pytest.raises(ValueError, match="distinct"):
        MarginalRatioEstimator(3, [(0,), 0], observation_dim=2)
    est = MarginalRatioEstimator(3, [(0,), (1, 2)], observation_dim=2)
    with pytest.raises(ValueError, match="name the marginal"):
        est.log_ratio(torch.zeros(4, 1), torch.zeros(4, 2))
    with pytest.raises(ValueError, match="was not trained"):
        est.log_ratio(torch.zeros(4, 2), torch.zeros(4, 2), (2, 1))


def test_shapes_with_several_marginals() -> None:
    torch.manual_seed(0)
    p, n = 5, 6
    marginals = MarginalRatioEstimator.univariate([0, 1, 2]) + MarginalRatioEstimator.pairs([0, 1])
    est = MarginalRatioEstimator(p, marginals, observation_dim=3, embedding_dim=8)
    theta, x = torch.randn(n, p), torch.randn(n, 3)
    for key in marginals:
        lr = est.log_ratio(theta[:, list(key)], x, key)
        assert lr.shape == (n,)
        cand = torch.randn(n, 7, len(key))
        scores = est.candidate_log_ratio(cand, x, key)
        assert scores.shape == (n, 7)
        torch.testing.assert_close(scores[:, 3], est.log_ratio(cand[:, 3], x, key))
        bank = torch.randn(9, len(key))
        assert est.candidate_log_ratio(bank, x, key).shape == (n, 9)
    # 1-D marginals accept a 1-D theta and an int key.
    torch.testing.assert_close(
        est.log_ratio(theta[:, 2], x, 2), est.log_ratio(theta[:, [2]], x, (2,))
    )
    with torch.no_grad():
        post = est.posterior(x, torch.randn(50, p), marginal=(0, 1), min_ess=0)
    assert post.event_dim == 2 and post.batch_size == n
    assert post.prior_samples.shape == (50, 2)
    assert post.sample(5).shape == (n, 5, 2)


def test_loss_is_the_sum_of_neural_ratio_losses_on_shared_pairs() -> None:
    torch.manual_seed(0)
    p, n = 4, 10
    marginals = [(0,), (1, 3)]
    est = MarginalRatioEstimator(p, marginals, observation_dim=2, n_negatives=3)
    theta, x = torch.randn(n, p), torch.randn(n, 2)
    total = est.loss(theta, x, generator=torch.Generator().manual_seed(5))
    index = in_batch_negative_index(n, 3, generator=torch.Generator().manual_seed(5))
    emb = est.embed(x)
    expected = torch.zeros(())
    for key in marginals:
        loss_fn = NeuralRatioLoss(est.head(key), criterion="cnre", gamma=1.0)
        th = theta[:, list(key)]
        expected = expected + loss_fn(th, emb, negative_context=th[index])
    torch.testing.assert_close(total, expected)
    # Default K is half the batch.
    est_default = MarginalRatioEstimator(p, marginals, observation_dim=2)
    assert est_default._n_negatives(256) == 128 and est_default._n_negatives(3) == 1


def test_all_marginals_train_together() -> None:
    torch.manual_seed(0)
    p = 3
    theta, x = _simulate(p, 2000, seed=11)
    x = x[:, :2]
    marginals = MarginalRatioEstimator.univariate([0, 1, 2]) + MarginalRatioEstimator.pairs([0, 1])
    est = MarginalRatioEstimator(p, marginals, observation_dim=2, embedding_dim=16)
    est.loss(theta[:64], x[:64]).backward()
    for key in marginals:
        assert all(prm.grad is not None for prm in est.head(key).parameters())
    est.zero_grad()
    est.fit(theta, x, max_epochs=6, patience=6, batch_size=128, lr=3e-3)
    result = est.training_result_
    assert result is not None and result.best < result.history[0]
    assert not est.training
    # Each head separates joint pairs from shuffled ones.
    theta_t, x_t = _simulate(p, 500, seed=12)
    x_t = x_t[:, :2]
    shuffled = theta_t[torch.randperm(500, generator=torch.Generator().manual_seed(0))]
    with torch.no_grad():
        for key in marginals:
            joint = est.log_ratio(theta_t[:, list(key)], x_t, key).mean()
            product = est.log_ratio(shuffled[:, list(key)], x_t, key).mean()
            assert float(joint) > float(product) + 0.3, key


def test_custom_embedding() -> None:
    torch.manual_seed(0)
    embedding = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(6, 5))
    est = MarginalRatioEstimator(2, [(0,)], embedding=embedding, embedding_dim=5)
    theta, x = torch.randn(8, 2), torch.randn(8, 2, 3)
    assert est.loss(theta, x).dim() == 0
    assert est.log_ratio(theta[:, :1], x).shape == (8,)
    bad = MarginalRatioEstimator(2, [(0,)], embedding=embedding, embedding_dim=4)
    with pytest.raises(ValueError, match="embedding must return"):
        bad.embed(x)


# ═══════════════════════════════════════════════════════════════════════════════
# RatioPosterior with the exact ratio: the sampler and the normalisation
# ═══════════════════════════════════════════════════════════════════════════════


def _exact_posterior(p: int, x: Tensor, bank: Tensor) -> RatioPosterior:
    return RatioPosterior(
        lambda th, obs: _exact_log_ratio(p, th, obs),
        x,
        bank,
        prior_log_prob=_prior_log_prob,
    )


def test_exact_ratio_posterior_moments_and_normalisation_within_monte_carlo_error() -> None:
    p, n, m, s = 16, 40, 20_000, 4000
    _, x = _simulate(p, n, seed=21)
    bank = _prior_bank(m, seed=22).to(F64)
    post = _exact_posterior(p, x, bank)
    mean, cov = _exact_marginal(p, x)
    sd = cov.diagonal().sqrt()
    ess = post.effective_sample_size()
    assert ess.shape == (n,) and bool((ess > 300).all())

    # Normalisation: E_p[r] = 1 exactly; relative Monte Carlo error sqrt(1/ESS - 1/M).
    z = post.normalisation()
    rel_se = (1.0 / ess - 1.0 / m).sqrt()
    assert float(((z - 1.0) / rel_se).abs().max()) < 4.5

    # Weighted moments: standard error of a self-normalised mean ~ sd / sqrt(ESS).
    z_mean = (post.mean() - mean) / (sd / ess.unsqueeze(1).sqrt())
    assert float(z_mean.abs().max()) < 4.5
    assert float(z_mean.pow(2).mean()) < 2.0

    # SIR draws: Var(mean of S draws) ~ sd^2 (1 / S + 1 / ESS).
    torch.manual_seed(0)
    draws = post.sample(s)
    assert draws.shape == (n, s, 2) and draws.dtype == F64
    se = sd * (1.0 / s + 1.0 / ess.unsqueeze(1)).sqrt()
    z_draws = (draws.mean(1) - mean) / se
    assert float(z_draws.abs().max()) < 4.5
    centred = draws - draws.mean(1, keepdim=True)
    sample_cov = centred.mT @ centred / (s - 1)
    # Relative error of a variance from ~ESS effective draws: sqrt(2 / ESS).
    var_ratio = sample_cov.diagonal(dim1=-2, dim2=-1) / cov.diagonal()
    assert float(((var_ratio - 1.0) / (2.0 / ess.unsqueeze(1)).sqrt()).abs().max()) < 4.5

    # log_prob: exact posterior density up to the Monte Carlo log Z.
    theta_eval = mean + 0.5 * sd
    exact_lp = torch.distributions.MultivariateNormal(mean, cov).log_prob(theta_eval)
    torch.testing.assert_close(post.log_prob(theta_eval), exact_lp - z.log())
    assert float((post.log_prob(theta_eval) - exact_lp).abs().max()) < 4.5 * float(rel_se.max())
    lp3 = post.log_prob(draws[:, :5])
    assert lp3.shape == (n, 5)
    torch.testing.assert_close(lp3[:, 2], post.log_prob(draws[:, 2]))


def test_ratio_posterior_low_ess_warning_and_sampler_prior() -> None:
    x = torch.zeros(3, 1)

    def peaked(theta: Tensor, obs: Tensor) -> Tensor:
        return -200.0 * theta.pow(2).sum(-1)

    post = RatioPosterior(peaked, x, lambda m: torch.randn(m, 2), n_proposals=500, min_ess=50)
    assert post.n_proposals == 500
    assert float(post.effective_sample_size().max()) < 50
    with pytest.warns(RuntimeWarning, match="effective sample size below 50"):
        post.sample(10)
    quiet = RatioPosterior(peaked, x, torch.randn(500, 2), min_ess=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        quiet.sample(10)


def test_ratio_posterior_without_prior_density() -> None:
    post = RatioPosterior(
        lambda th, obs: -th.pow(2).sum(-1), torch.zeros(2, 3), torch.randn(100), min_ess=0
    )
    assert post.event_dim == 1 and not post.has_density
    with pytest.raises(NotImplementedError, match="prior_log_prob"):
        post.log_prob(torch.zeros(2, 1))
    assert post.marginal(0).sample(4).shape == (2, 4, 1)
    with pytest.raises(ValueError, match="log_ratio returned"):
        RatioPosterior(
            lambda th, obs: th.sum(-1).sum(-1), torch.zeros(2, 3), torch.randn(10)
        ).sample(1)


# ═══════════════════════════════════════════════════════════════════════════════
# trained estimators against the exact marginal posterior, P = 4, 16, 64
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("p", [4, 16, 64])
def test_marginal_log_ratio_matches_the_exact_one(p: int) -> None:
    est = _trained(p)
    theta_t, x_t = _simulate(p, 2000, seed=5)
    shuffled = theta_t[torch.randperm(2000, generator=torch.Generator().manual_seed(3))]
    with torch.no_grad():
        joint = est.log_ratio(theta_t[:, :2], x_t).to(F64)
        product = est.log_ratio(shuffled[:, :2], x_t).to(F64)
    exact_joint = _exact_log_ratio(p, theta_t[:, :2].unsqueeze(1), x_t)[:, 0]
    exact_product = _exact_log_ratio(p, shuffled[:, :2].unsqueeze(1), x_t)[:, 0]
    both, exact_both = torch.cat([joint, product]), torch.cat([exact_joint, exact_product])
    corr = float(torch.corrcoef(torch.stack([both, exact_both]))[0, 1])
    mae_joint = float((joint - exact_joint).abs().mean())
    print(
        f"P={p}: corr={corr:.4f} MAE(joint pairs)={mae_joint:.3f} "
        f"MAE(all)={float((both - exact_both).abs().mean()):.3f} "
        f"sd(exact, joint pairs)={float(exact_joint.std()):.3f}"
    )
    # Held-out joint pairs (where the posterior mass is) and shuffled pairs (prior
    # draws, much of them in the far tail of the posterior).
    assert corr > 0.97
    assert mae_joint < 0.15


@pytest.mark.parametrize("p", [4, 16, 64])
def test_ratio_posterior_moments_and_normalisation(p: int) -> None:
    est = _trained(p)
    _, x = _simulate(p, 200, seed=6)
    bank = _prior_bank(16_384, seed=7)
    with torch.no_grad():
        post = est.posterior(x, bank, prior_log_prob=_prior_log_prob, min_ess=0)
    mean, cov = _exact_marginal(p, x)
    sd = cov.diagonal().sqrt()
    torch.manual_seed(1)
    draws = post.sample(2000).to(F64)
    centred = draws - draws.mean(1, keepdim=True)
    sample_cov = centred.mT @ centred / 1999
    mean_err = (draws.mean(1) - mean) / sd
    sd_ratio = sample_cov.diagonal(dim1=-2, dim2=-1).sqrt() / sd
    corr = sample_cov[:, 0, 1] / sample_cov.diagonal(dim1=-2, dim2=-1).sqrt().prod(-1)
    exact_corr = float(cov[0, 1] / sd.prod())
    z = post.normalisation()
    ess = post.effective_sample_size()
    print(
        f"P={p}: mean error rms={float(mean_err.pow(2).mean().sqrt()):.3f} sd "
        f"(max {float(mean_err.abs().max()):.3f}); sd ratio={sd_ratio.mean(0).tolist()}; "
        f"corr {float(corr.mean()):.3f} vs exact {exact_corr:.3f}; "
        f"Z mean={float(z.mean()):.3f} median |log Z|={float(z.log().abs().median()):.3f}; "
        f"ESS median={float(ess.median()):.0f} min={float(ess.min()):.0f}"
    )
    # Monte Carlo error alone (2000 SIR draws from a bank with ESS ~ 1000+) is
    # ~0.04 sd on the mean; the rest is the estimator error at this budget.
    assert float(mean_err.pow(2).mean().sqrt()) < 0.2
    assert bool(((sd_ratio.mean(0) - 1.0).abs() < 0.1).all())
    assert abs(float(corr.mean()) - exact_corr) < 0.1
    # Normalisation diagnostic: E_p[r] = 1 for the exact ratio.
    assert abs(float(z.mean()) - 1.0) < 0.06
    assert float(z.log().abs().median()) < 0.1


def test_head_cost_does_not_grow_with_the_number_of_nuisances() -> None:
    def head_parameters(est: MarginalRatioEstimator) -> int:
        return sum(prm.numel() for prm in est.ratio_losses.parameters())

    def step_time(est: MarginalRatioEstimator, theta: Tensor, x: Tensor) -> float:
        times = []
        for r in range(18):
            start = time.perf_counter()
            est.loss(theta, x).backward()
            est.zero_grad()
            if r >= 3:
                times.append(time.perf_counter() - start)
        return statistics.median(times)

    marginal_params, joint_params, marginal_time, joint_time = {}, {}, {}, {}
    for p in (4, 16, 64):
        torch.manual_seed(0)
        theta, x = torch.randn(256, p), torch.randn(256, D)
        marginal = MarginalRatioEstimator(p, [INTEREST], observation_dim=D)
        joint = MarginalRatioEstimator(
            p, MarginalRatioEstimator.jointly(range(p)), observation_dim=D
        )
        marginal_params[p], joint_params[p] = head_parameters(marginal), head_parameters(joint)
        marginal_time[p] = step_time(marginal, theta, x)
        joint_time[p] = step_time(joint, theta, x)
    print(f"head parameters: marginal {marginal_params}, joint {joint_params}")
    print(
        "time per step (ms): marginal "
        f"{ {p: round(1e3 * t, 1) for p, t in marginal_time.items()} }, joint "
        f"{ {p: round(1e3 * t, 1) for p, t in joint_time.items()} }"
    )
    assert marginal_params[4] == marginal_params[16] == marginal_params[64]
    # The joint head's first layer reads all P parameters: 64 weights per parameter.
    assert joint_params[64] - joint_params[4] == 64 * 60
    assert joint_params[16] > marginal_params[16]
    # Same computation for every P (median of 15 steps; 1.5 absorbs timer noise).
    assert marginal_time[64] < 1.5 * marginal_time[4]


# ═══════════════════════════════════════════════════════════════════════════════
# calibration over fresh simulations (P = 4) and the joint API
# ═══════════════════════════════════════════════════════════════════════════════


def test_sbc_and_expected_coverage_are_uniform_for_p4() -> None:
    p, n, s = 4, 500, 99
    est = _trained(p)
    theta, x = _simulate(p, n, seed=31)
    truth = theta[:, list(INTEREST)]
    with torch.no_grad():
        post = est.posterior(x, _prior_bank(16_384, 32), prior_log_prob=_prior_log_prob, min_ess=0)
    torch.manual_seed(2)
    draws = post.sample(s)
    ranks = simulation_based_calibration_ranks(draws, truth)
    sbc = sbc_uniformity_pvalue(ranks, n_bins=10, n_posterior_samples=s)
    reference = torch.randn(n, 2, generator=torch.Generator().manual_seed(33))
    tarp = tarp_coverage(draws, truth, reference)
    levels = post.hpd_level(truth, n_samples=512)
    hpd = hpd_calibration(levels)
    print(
        f"SBC n={n} S={s} 10 bins: p-values {sbc['p_values']}; TARP error "
        f"{tarp['tarp_error']:.3f}; HPD KS {float(hpd.ks_statistic):.3f} "
        f"p={float(hpd.p_value):.3f}"
    )
    # Chi-square test of uniform ranks per parameter, Bonferroni over the two.
    assert sbc["bonferroni_min_p_value"] > 0.01
    # Expected coverage of HPD regions at every level: KS test against uniform.
    assert float(hpd.p_value) > 0.01
    # TARP has no calibrated null; with n = 500 its noise floor is ~ 1.36 / sqrt(n) = 0.06.
    assert tarp["tarp_error"] < 0.1


def test_conformal_region_on_a_ratio_posterior_covers_and_plots() -> None:
    p, alpha = 4, 0.2
    est = _trained(p)
    bank = _prior_bank(8192, 41)
    theta_cal, x_cal = _simulate(p, 500, seed=42)
    theta_test, x_test = _simulate(p, 1000, seed=43)
    with torch.no_grad():
        post_cal = as_joint(est, x_cal, prior_samples=bank, prior_log_prob=_prior_log_prob)
        post_test = est.posterior(x_test, bank, prior_log_prob=_prior_log_prob)
    assert isinstance(post_cal, RatioPosterior)
    conformal = JointConformal(DensityScore(), alpha=alpha).calibrate(post_cal, theta_cal[:, :2])
    coverage = float(conformal.region(post_test).contains(theta_test[:, :2]).float().mean())
    print(f"conformal coverage {coverage:.3f} at nominal {1 - alpha}")
    # Finite-sample guarantee over the simulated population: E[coverage] in
    # [0.8, 0.8 + 1 / 501]; 1000 test rows give a binomial sd of 0.013.
    assert abs(coverage - (1 - alpha)) < 3 * math.sqrt(alpha * (1 - alpha) / 1000) + 1 / 501

    with torch.no_grad():
        small = est.posterior(x_test[:3], bank, prior_log_prob=_prior_log_prob, min_ess=0)
    fig = plot_corner_plot(
        small,
        row=0,
        region=conformal.region(small),
        n_samples=200,
        region_samples=100,
        return_figure=True,
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_as_joint_routes_the_estimator_to_its_posterior() -> None:
    torch.manual_seed(0)
    est = MarginalRatioEstimator(3, [(0,), (1, 2)], observation_dim=2)
    x, prior = torch.randn(4, 2), torch.randn(64, 3)
    with torch.no_grad():
        routed = as_joint(est, x, prior_samples=prior, marginal=(1, 2))
        direct = est.posterior(x, prior, (1, 2))
    assert isinstance(routed, RatioPosterior)
    torch.testing.assert_close(routed.bank_log_ratio, direct.bank_log_ratio)
    torch.testing.assert_close(routed.prior_samples, prior[:, [1, 2]])
    assert as_joint(routed) is routed
    with pytest.raises(ValueError, match="observations"):
        as_joint(est)
