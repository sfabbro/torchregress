# Neyman Confidence Sets (LF2I)

> ← [WALDO Wald Regions](waldo.md) | [Model-Agnostic UQ Wrappers](estimators.md) →

Confidence sets for a parameter $\theta \in \mathbb{R}^P$ with (approximately)
nominal coverage **at every $\theta$**, whatever the distribution of $\theta$ in
the training or calibration population.  This is the Neyman construction with a
*parameter-dependent* critical value learned by quantile regression, as in
likelihood-free frequentist inference (LF2I; Dalmasso et al.) and WALDO
(Masserano et al.).  It applies to any test statistic: a likelihood ratio from a
ratio estimator, the WALDO statistic, a negative log posterior, ...

!!! abstract "Why this matters"
    A posterior interval learned from a skewed population inherits that
    population as a prior: in the rare part of parameter space it is pulled
    towards the bulk and under-covers (for example, 68 % intervals that contain
    the true metallicity of only ~40 % of metal-poor stars).  A split-conformal
    threshold fixes the *average* coverage but not the coverage in that region.
    The Neyman construction targets $P(\theta \in R(X) \mid \theta) = 1-\alpha$
    for each $\theta$ separately.

---

## The construction

Let $\lambda(\theta; x)$ be a test statistic that is **large** when $\theta$ is
incompatible with the observation $x$.  The level-$(1-\alpha)$ Neyman set is

$$
R(x) = \{\theta : \lambda(\theta; x) \le C_\alpha(\theta)\},
\qquad
C_\alpha(\theta) = F^{-1}_{\lambda(\theta; X)}(1-\alpha),
\quad X \sim p(x \mid \theta),
$$

and by construction

$$
P_{X \mid \theta}\bigl(\theta \in R(X)\bigr)
= P_{X \mid \theta}\bigl(\lambda(\theta; X) \le C_\alpha(\theta)\bigr)
\ge 1 - \alpha
\quad \text{for every } \theta .
$$

$C_\alpha(\theta)$ is a conditional quantile.  Given calibration pairs
$(\theta_i, x_i)$ drawn from $p(x \mid \theta)$, set
$\lambda_i = \lambda(\theta_i; x_i)$ and estimate $C_\alpha$ by quantile
regression of $\lambda_i$ on $\theta_i$, i.e. by minimising the pinball loss

$$
\hat C_\alpha = \arg\min_{C} \sum_{i=1}^{N}
\rho_{1-\alpha}\bigl(\lambda_i - C(\theta_i)\bigr),
\qquad
\rho_\tau(u) = \max\{\tau u, (\tau - 1) u\}.
$$

The distribution of the $\theta_i$ only decides *where* $C_\alpha$ is estimated
precisely; the target $C_\alpha(\theta)$ does not depend on it.  This is what
makes the construction prior-free.  In LF2I the pairs come from a simulator;
here they are any labelled data that follow the same $p(x \mid \theta)$ as the
test data (for example stars with spectroscopic labels).

A constant critical value $C$ (the split-conformal threshold) is the special
case that only guarantees *marginal* coverage, averaged over the calibration
population of $\theta$.  When the distribution of $\lambda(\theta; X)$ changes
with $\theta$ (heteroscedastic noise, parameter boundaries, a statistic learned
on a skewed population), it over-covers some regions and under-covers others.

---

## API

| Object | Role |
|:-------|:-----|
| `CriticalValueRegressor(alpha, method="mlp" \| "knn", ...)` | Fits $\hat C_\alpha(\theta)$ for one or several levels. `fit(theta [N, P], statistic [N])`, `predict(theta [M, P]) -> [M]` or `[M, L]`. |
| `NeymanConfidenceSet(critical_value)` | Holds a fitted regressor (or any callable $\theta \mapsto C$). `contains(statistic, theta)`, `region(statistic_fn, theta_bank, x)`, `critical_value(theta)`, and `NeymanConfidenceSet.calibrate(theta, statistic, alpha, **regressor_kwargs)`. |
| `binned_conditional_coverage(theta, covered, nominal=...)` | Coverage per bin of each parameter with Wilson intervals (Bonferroni over bins). |
| `smooth_conditional_coverage(theta, covered, nominal=...)` | Logistic-regression estimate of $P(\text{covered} \mid \theta)$ with a Scheffé simultaneous band. |

All live in `torchregress.inference` (see the
[Inference API](../api/inference.md#neyman-confidence-sets-lf2i)).

### Critical-value estimators

| | `method="mlp"` (default) | `method="knn"` |
|:--|:--|:--|
| Estimate | Quantile-regression MLP (pinball loss via `torchregress.utils.quantile.multi_quantile_loss`) | Finite-sample quantile, order statistic $\lceil (k+1)(1-\alpha) \rceil$, of the $k$ nearest calibration statistics |
| Non-crossing levels | Head predicts the lowest quantile plus softplus increments | Automatic (same sorted neighbours) |
| Inputs | Standardised $\theta$; statistic log-transformed when non-negative (quantiles are equivariant under increasing maps) | Standardised $\theta$ |
| Training | Adam, held-out early stopping, reduce-on-plateau, seeded and deterministic | None |
| Default $k$ | | $\lceil N^{4/(4+P)}/2 \rceil$, at least $\lceil 10/\alpha \rceil$ |
| Strengths | Pools information across $\theta$; most accurate in the tests, especially for $P \ge 2$ | Assumption-light cross-check, instant |
| Weaknesses | Needs training; extrapolates outside the calibration support | One-sided neighbourhoods at the sparse edge of the support bias $\hat C$ towards the dense side; noisy for $P \ge 3$ |

Both estimators are tensor-native (CPU, CUDA, MPS; float32 and float64).  The
k-NN distances are computed in chunks with `torch.cdist`.

---

## Worked example: calibrated vs constant threshold by region

A skewed population (each $\theta_j \sim 3\,\mathrm{Beta}(1, 4)$, so large
$\theta$ is rare), noise that grows with $\theta$, and a statistic that ignores
it (a unit-variance Wald statistic, as an estimator trained on the bulk of the
population would produce).  The distribution of $\lambda(\theta; X)$ is
$s(\theta)^2 \chi^2_2$, so no constant threshold can be right everywhere.

```python
import math

import torch

from torchregress.inference import (
    NeymanConfidenceSet,
    binned_conditional_coverage,
    smooth_conditional_coverage,
)


def simulate(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Skewed population (rare at large theta), noise growing with theta."""
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(n, 2, generator=g, dtype=torch.float64)
    theta = 3.0 * (1.0 - (1.0 - u) ** 0.25)  # 3 * Beta(1, 4) per parameter
    scale = 0.3 + 0.5 * theta.sum(-1, keepdim=True) / math.sqrt(2)
    x = theta + scale * torch.randn(n, 2, generator=g, dtype=torch.float64)
    statistic = ((x - theta) ** 2).sum(-1)  # unit-variance Wald statistic
    return theta, statistic


alpha = 0.32
theta_cal, stat_cal = simulate(20_000, seed=0)  # labelled calibration pairs
theta_test, stat_test = simulate(10_000, seed=1)  # held-out pairs

# Constant threshold: the split-conformal (marginal) quantile.
constant = stat_test <= torch.quantile(stat_cal, 1 - alpha)
# Neyman construction: learned C(theta) (quantile MLP).
cs = NeymanConfidenceSet.calibrate(theta_cal, stat_cal, alpha=alpha)
neyman = cs.contains(stat_test, theta_test)

key = theta_test.sum(-1) / math.sqrt(2)
edges = [0.0, 0.4, 0.8, 1.2, 1.6, 4.3]
for lo, hi in zip(edges[:-1], edges[1:]):
    r = (key >= lo) & (key < hi)
    print(
        f"[{lo:.1f}, {hi:.1f})  n={int(r.sum()):5d}  constant={constant[r].double().mean():.3f}"
        f"  neyman={neyman[r].double().mean():.3f}"
    )

for name, covered in (("constant", constant), ("neyman", neyman)):
    b0, b1 = binned_conditional_coverage(theta_test, covered, nominal=1 - alpha, bins=5)
    smooth = smooth_conditional_coverage(theta_test, covered, nominal=1 - alpha)
    print(name, b0.under_covered.tolist(), float(smooth.under_covered.double().mean()))
```

Held-out coverage of the 68 % sets by region of $(\theta_1 + \theta_2)/\sqrt{2}$
(10 000 test pairs; the k-NN column is the same run with `method="knn"`):

| Region | Test pairs | Constant threshold | Neyman, MLP | Neyman, k-NN |
|:-------|-----------:|-------------------:|------------:|-------------:|
| [0.0, 0.4) | 1904 | 0.950 | 0.692 | 0.699 |
| [0.4, 0.8) | 3291 | 0.785 | 0.692 | 0.684 |
| [0.8, 1.2) | 2594 | 0.592 | 0.680 | 0.663 |
| [1.2, 1.6) | 1387 | 0.430 | 0.678 | 0.674 |
| [1.6, 4.3) (rare) | 824 | 0.289 | 0.683 | 0.658 |
| all | 10 000 | 0.676 | 0.686 | 0.678 |

Both thresholds have nominal *marginal* coverage, but the constant one covers
95 % of the common, low-noise objects and 29 % of the rare ones.  The
diagnostics flag the constant threshold's bins 2-5 along $\theta_1$ (Wilson
intervals entirely below 0.68) and 40 % of the test points under the smooth
estimate (49 % significantly *over*-covered); for the Neyman sets they flag
nothing.

!!! tip "Calibration size sets the precision, not the target"
    With 10 000 instead of 20 000 calibration pairs the exact conditional
    coverage of the same MLP in the densest corner ($\theta$ near 0, where the
    scale changes fastest) ranges from 0.64 to 0.69 over seeds.  Run the diagnostics on held-out data and
    add calibration pairs where they flag a region.

---

## Using a statistic from a model

`NeymanConfidenceSet` never looks inside the statistic.  For a candidate bank
of parameters, pass a callable `statistic_fn(theta_bank, x) -> [N, K]`, where
`theta_bank` is a shared `[K, P]` grid or a per-row `[N, K, P]` bank:

```python
import torch

from torchregress.inference import NeymanConfidenceSet

# A ratio estimator log r(x, theta) gives lambda = -log r (or -2 log LR).
def statistic_fn(theta_bank: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    bank = theta_bank if theta_bank.dim() == 3 else theta_bank.expand(x.shape[0], -1, -1)
    return -log_ratio_model(x, bank)  # [N, K]

# Calibrate on labelled pairs: the statistic at each row's own parameter.
stat_cal = statistic_fn(theta_cal.unsqueeze(1), x_cal)[:, 0]
cs = NeymanConfidenceSet.calibrate(theta_cal, stat_cal, alpha=[0.32, 0.05])

masks = cs.region(statistic_fn, theta_bank, x_test, chunk_size=1024)  # [N, K, 2] bool
```

With several levels the masks are nested (the 68 % set lies inside the 95 % set)
because the critical values never cross.  The WALDO statistic works the same
way, and a single-level `CriticalValueRegressor` can be passed directly as
`critical_value_regressor` to [`WaldoConformal`](waldo.md) and its sample-based
variants, replacing the binned one-dimensional critical-value curve with one
learned on the full parameter vector.

---

## Conditional-coverage diagnostics

LF2I's key check estimates $P(\theta \in R(X) \mid \theta)$ on held-out pairs
from the indicators $\mathbb{1}[\lambda(\theta_i; x_i) \le \hat C(\theta_i)]$:

- **Binned** (`binned_conditional_coverage`): equal-width bins along each
  parameter (so rare regions get their own bins; `binning="quantile"` for equal
  counts, or explicit edges), Wilson score intervals, Bonferroni-corrected over
  all non-empty bins by default.  `under_covered` marks bins whose upper bound
  is below nominal.
- **Smooth** (`smooth_conditional_coverage`): logistic regression on a
  polynomial basis of the standardised parameters (powers up to `degree=3`
  plus pairwise products), fitted by ridge-penalised Newton/IRLS.  The band is
  $\hat\eta \pm c\,\mathrm{se}$ on the logit with
  $c = \sqrt{\chi^2_d(\text{confidence})}$ over the $d$ basis functions
  (Scheffé), so it holds simultaneously at all evaluation points;
  `simultaneous=False` gives pointwise bands.

!!! warning "What the guarantee is, and is not"
    The coverage at $\theta$ is nominal only up to the error of
    $\hat C_\alpha(\theta)$.  That error is small where calibration pairs are
    dense and grows where they are sparse; outside the calibration support
    $\hat C$ is an extrapolation.  The statistic must also be computed the same
    way on calibration and test data, and the calibration pairs must follow the
    test-time $p(x \mid \theta)$ (selection on $x$ breaks this).  Always run the
    diagnostics.

---

## When to use which

| Situation | Use |
|:----------|:----|
| Exchangeable calibration and test sets, average coverage is enough | Constant threshold ([split conformal](conformal/index.md), `WaldoConformal` without critical-value model) |
| Coverage must hold for every parameter value, population skewed or shifted | `NeymanConfidenceSet` with the MLP critical-value regressor |
| Quick cross-check of the MLP, $P \le 2$ | `CriticalValueRegressor(method="knn")` |
| Only a predictive mean/covariance available | WALDO statistic ([WALDO](waldo.md)) + `NeymanConfidenceSet` |
| Likelihood-ratio estimator available | $\lambda = -\log \hat r(x, \theta)$ ([NRE](../losses/nre.md)) + `NeymanConfidenceSet` |

## References

| # | Reference |
|:-:|:----------|
| 1 | N. Dalmasso, L. Masserano, D. Zhao, R. Izbicki, A. B. Lee. ["Likelihood-free frequentist inference: bridging classical statistics and machine learning for reliable simulator-based inference."](https://arxiv.org/abs/2107.03920) *Electronic Journal of Statistics* (2024). |
| 2 | L. Masserano, T. Dorigo, R. Izbicki, M. Kuusela, A. B. Lee. ["Simulator-based inference with WALDO: confidence regions by leveraging prediction algorithms and posterior estimators for inverse problems."](https://arxiv.org/abs/2205.15680) AISTATS 2023. |
| 3 | J. Neyman. "Outline of a theory of statistical estimation based on the classical theory of probability." *Phil. Trans. R. Soc. A* 236 (1937) 333-380. |
| 4 | R. Koenker, G. Bassett. "Regression quantiles." *Econometrica* 46 (1978) 33-50. |
| 5 | E. B. Wilson. "Probable inference, the law of succession, and statistical inference." *JASA* 22 (1927) 209-212. |
