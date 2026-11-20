# Likelihood-Ratio Inference

> ← [WALDO Wald Regions](waldo.md) | [Neural Ratio Estimation](../losses/nre.md) →

`torchregress.inference.likelihood` turns a score function, learned or analytic, into
**maximum-likelihood estimates** and **profile-likelihood confidence intervals** for a
continuous multivariate parameter $\theta \in \mathbb{R}^P$ inside a box. It is batched over
rows (no Python loop over observations) and does not depend on the estimator: a ratio classifier,
a conditional flow and a closed-form log-likelihood all go through the same code.

## Why a frequentist estimate

A posterior mean trained on a skewed population is pulled toward the bulk of that population.
For stellar parameters this means metal-poor stars come out too metal-rich. The likelihood
$p(x \mid \theta)$ does not contain the training prior, so its maximizer and the
likelihood-ratio intervals around it are free of that pull. A ratio estimator trained with
[`NeuralRatioLoss`](../losses/nre.md) gives exactly such a score. A grid scan over $\theta$
([`grid_log_posterior`](../api/inference.md#likelihood-ratio-posterior-reweighting-inferenceratio))
works for one parameter. It stops scaling at $P = 3$ to $5$ parameters and $10^5$ to $10^6$
objects. This module replaces the grid with optimization and root finding.

## When to use which

| Situation | Tool | Why |
|:----------|:-----|:----|
| One parameter, a fixed grid is affordable | [`maximum_likelihood_estimate`](../api/inference.md#likelihood-ratio-posterior-reweighting-inferenceratio) on a scan | Exact on the grid, no optimizer |
| $P \ge 2$ continuous parameters, many objects | `maximize_score` + `profile_interval` | Cost grows with $N$ and polynomially in $P$, not exponentially |
| Hypothesis test or joint region for all parameters | `likelihood_ratio_statistic` with `wilks_threshold(c, df=P)` | No per-parameter profiling needed |
| You want posterior samples under a chosen prior | NPE flow or [posterior reweighting](../api/inference.md#likelihood-ratio-posterior-reweighting-inferenceratio) | A posterior answers a different question than the MLE |
| Only a predictive mean and covariance are available | [WALDO](waldo.md) | Wald regions need no likelihood |

## Score convention

A **score function** is any callable

```text
score(theta: Tensor[M, P], x: Tensor[M, ...]) -> Tensor[M]
```

with $s(\theta, x) = \log p(x \mid \theta) + c(x)$. The term $c(x)$ is arbitrary and cancels.
The score must be evaluated row by row (no batch statistics) and be twice differentiable in
$\theta$.

| Source | Adapter |
|:-------|:--------|
| `NeuralRatioLoss` classifier (logit $= \log p(x \mid \theta) - \log p(x)$) | `ratio_classifier_score(classifier_or_loss)` |
| `ContrastiveFlowLoss` / `NormalizingFlowLoss`, or a zuko conditional flow | `flow_score(flow_or_loss)` |
| Closed-form log-likelihood in torch | use it directly |

Put learned modules in `eval()` mode first.

## Statistics

With $\hat\theta = \arg\max_{\theta \in B} s(\theta, x)$ and $s_{\max} = s(\hat\theta, x)$:

$$
\lambda(\theta_0; x) = s_{\max}(x) - s(\theta_0, x),
\qquad
I_j(x) = \Big\{ t : \max_{\theta:\,\theta_j = t} s(\theta, x) \ge s_{\max}(x) - c \Big\}.
$$

$\lambda$ is half the usual $-2 \log \Lambda$. Under Wilks' theorem, `wilks_threshold(confidence,
df)` $= \tfrac12 \chi^2_{\mathrm{df}}(\text{confidence})$ gives the asymptotic threshold: $c = 0.5$
for a 68.27 % interval and $c = 1.92$ for 95 % (`df=1` for one profiled parameter, `df=P` for a
joint region). For finite-sample coverage, estimate a Neyman critical value $c(x)$ per row (for
example by quantile regression of $\lambda(\theta_{\text{true}})$ on simulations) and pass it as a
tensor.

## Worked example

```python
import torch
from torchregress.inference import (
    likelihood_ratio_statistic,
    maximize_score,
    profile_interval,
    wilks_threshold,
)

# Linear-Gaussian simulator: x = A theta + noise, P = 3 parameters, 6 observables.
torch.manual_seed(0)
A = torch.randn(6, 3, dtype=torch.float64)
noise = 0.3
theta_true = torch.rand(1000, 3, dtype=torch.float64) * 4 - 2
x = theta_true @ A.T + noise * torch.randn(1000, 6, dtype=torch.float64)


def score(theta, x):  # log p(x | theta) up to a constant
    return -0.5 * (((x - theta @ A.T) / noise) ** 2).sum(-1)


bounds = torch.tensor([[-3.0, 3.0]] * 3, dtype=torch.float64)
theta_hat, s_max, at_bound = maximize_score(
    score, x, bounds, generator=torch.Generator().manual_seed(0)
)

c95 = wilks_threshold(0.95)  # 1.92
lower, upper, lo_flag, hi_flag = profile_interval(score, x, theta_hat, s_max, 2, c95, bounds)
coverage = ((lower <= theta_true[:, 2]) & (theta_true[:, 2] <= upper)).double().mean()

# Likelihood-ratio test of the true parameters (joint region, df = 3).
lam = likelihood_ratio_statistic(score, x, theta_true, s_max)
joint = (lam <= wilks_threshold(0.95, df=3)).double().mean()
print(f"profile coverage {coverage:.3f}, joint coverage {joint:.3f}")
```

With a trained ratio estimator, replace `score` with `ratio_classifier_score(classifier)` and
pass a posterior head's point estimate as `init=` to add one more start per row.

## Algorithms

All work happens in unit-box coordinates $u \in [0, 1]^P$, so tolerances are relative to the box
and parameters in different units (kelvin and dex) need no rescaling.

**Multistart (`maximize_score`).** `n_candidates` scrambled Sobol points are scored in chunks.
Starts are chosen *topographically* (Törn & Viitanen, 1994): a candidate that scores at least as
high as its $2P$ nearest candidates is a discrete local maximum. The best `n_starts` such maxima
per row are refined, plus an optional `init`. Ranking the raw candidates by score would put every
start in the basin with the broadest high region. A narrow global mode whose candidates all score
lower would then be lost.

**Refinement.** All starts of all rows are refined at once by projected Newton ascent (Bertsekas,
1982) with exact $P \times P$ Hessians from autograd:

- coordinates on a bound whose gradient points out of the box are held fixed;
- Hessian eigenvalues are replaced by their absolute values, floored at
  $\sqrt{\epsilon}\max|\lambda|$ (saddle-free Newton), so every step is an ascent direction;
- the step is capped at one box width and accepted by Armijo backtracking along the projection
  arc.

Newton is affine invariant and converges quadratically. For $P \lesssim 10$ this beats first-order
optimizers, which need hundreds of steps and a learning rate in parameter units. Converged rows
drop out of the batch. The maximizer lands exactly on a bound when it lies there, and `at_bound`
flags it.

**Profile intervals (`profile_interval`).** On each side of $\hat\theta_j$, the profile
$p_j(t) = \max_{\theta_{-j}} s$ is scanned on `n_scan` equally spaced points up to the bound. The
inner maximization is warm-started from the previous point (numerical continuation). The
**outermost** scan point above the threshold brackets the crossing. The crossing is then refined
by Illinois regula falsi on the signed root $r(t) = \sqrt{2(s_{\max} - p_j(t))}$. For a Gaussian
likelihood $r$ is linear in $t$, so the first interpolation is exact. In general the bracket is
kept and convergence is superlinear. If the profile stays above the threshold up to the box edge,
the endpoint is set to the bound and flagged.

!!! info "Multimodal profiles: the outermost crossing"
    When the likelihood-ratio set is disconnected (a secondary mode above the threshold),
    `profile_interval` returns the smallest interval that contains every scanned point above the
    threshold, so the interval spans the gap. This is the conservative choice: the interval keeps
    the coverage of the set. The connected component around $\hat\theta$ alone would under-cover.
    Regions narrower than the scan spacing $|{\rm bound} - \hat\theta_j| / n_{\rm scan}$ can be
    missed.

## Cost

Measured on 4 CPU threads, float32, an MLP score on `concat([theta, x])` with 256-d `x` and
hidden widths (256, 256, 128), $P = 3$, $N = 10\,000$ rows, default settings: `maximize_score`
takes about 26 s and each `profile_interval` about 11 s. Candidate scoring
($N \times$ `n_candidates` = 2.6 M forward passes, about 19 s) is most of the `maximize_score`
time. Newton refinement needs about two Hessian evaluations per start. A profile costs about
nine warm-started inner maximizations per side, most converging in one Newton step.
`tools/benchmark_likelihood_inference.py` reproduces the numbers.

## Limitations

- The search is global only up to the candidate resolution, about `n_candidates ** (-1/P)` box
  widths. Raise `n_candidates` for narrow modes.
- The inner profile maximization follows one branch by continuation. If a different branch of
  $\arg\max_{\theta_{-j}}$ takes over far from $\hat\theta$, the profile is underestimated there.
- Wilks thresholds are asymptotic. A learned score is only as good as the estimator, so check
  coverage on simulations or calibrate per-row thresholds.
- Non-smooth scores (ReLU networks) have zero Hessians almost everywhere. Newton then falls back
  to box-scaled gradient steps, which converge more slowly. Prefer smooth activations (the
  `build_ratio_classifier` default is SiLU).

## References

| # | Reference |
|:-:|:----------|
| 1 | D. P. Bertsekas. "Projected Newton methods for optimization problems with simple constraints." *SIAM J. Control Optim.* 20(2), 221–246, **1982** — projected Newton ascent with an active set. |
| 2 | A. Törn, S. Viitanen. "Topographical global optimization using pre-sampled points." *J. Global Optim.* 5, 267–276, **1994** — start selection from discrete local maxima. |
| 3 | Y. Dauphin et al. "Identifying and attacking the saddle point problem in high-dimensional non-convex optimization." *NeurIPS*, **2014** — absolute-eigenvalue (saddle-free) Newton. |
| 4 | D. J. Venzon, S. H. Moolgavkar. "A method for computing profile-likelihood-based confidence intervals." *Appl. Statist.* 37, 87–94, **1988** — profile-likelihood endpoints. |
| 5 | M. Dowell, P. Jarratt. "A modified regula falsi method for computing the root of an equation." *BIT* 11, 168–174, **1971** — the Illinois algorithm. |
| 6 | S. S. Wilks. "The large-sample distribution of the likelihood ratio for testing composite hypotheses." *Ann. Math. Statist.* 9(1), 60–62, **1938** — the $\chi^2$ threshold. |

## Next steps

- [Neural ratio estimation](../losses/nre.md): train the score
- [Inference API](../api/inference.md#maximum-likelihood-and-profile-intervals-inferencelikelihood): signatures
- [WALDO Wald regions](waldo.md): confidence sets from predictive moments instead of a likelihood
