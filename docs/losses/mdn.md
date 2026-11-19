# Mixture Density Networks (MDN)

> ← [Evidential Regression](advanced.md) | [Normalizing Flows](nflows.md) →

Mixture Density Networks model the conditional distribution $p(y \mid x)$ as a **mixture of Gaussian (or multivariate Student-$t$) components**, capturing multimodality, heteroscedastic noise, heavy tails, and complex distributional shapes.

!!! abstract "When to use"
    When the output distribution has **multiple modes** — e.g., inverse problems with multiple solutions, financial regimes, or multi-path trajectory prediction.

!!! tip "Noisy Inputs"
    If your input features ($X$) are noisy, standard MDNs may produce biased results. Use [**InputNoiseMDNLoss**](eiv.md#multimodal-eiv-variants) to marginalize over the input noise for more robust predictions.

---

## Mathematical Background

$$\boxed{\;p(y \mid x) = \sum_{k=1}^K \pi_k(x)\;\mathcal{N}\!\bigl(y \mid \mu_k(x),\, \Sigma_k(x)\bigr)\;}$$

| Symbol | Predicted by model | Constraint |
|:------:|:------------------|:-----------|
| $\pi_k$ | Mixture weights | $\pi_k > 0$, $\sum_k \pi_k = 1$ (softmax) |
| $\mu_k$ | Component means | Unconstrained |
| $\Sigma_k$ | Component covariance | PD (softplus + min_std) |

The NLL is:

$$\mathcal{L}_{\text{MDN}} = -\log \sum_{k=1}^K \pi_k\;\mathcal{N}(y \mid \mu_k, \Sigma_k)$$

computed via log-sum-exp for numerical stability.

See the [MixtureDensityLoss API](../api/losses.md) for constructor parameters and the [`MixtureDensityLoss`](../api/losses.md) alias.

## Usage

```python
from torchregress.losses import MixtureDensityLoss

loss_fn = MixtureDensityLoss(
    n_components=3,    # K = number of Gaussian components
    n_features=2,      # d = dimensionality of targets
    covariance_type="diagonal",  # or "full"
    min_std=1e-3,
)
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `n_components` | `int` | — | Number of mixture components $K$ |
| `n_features` | `int` | — | Target dimensionality $d$ |
| `covariance_type` | `str` | `"diagonal"` | `"diagonal"` or `"full"` |
| `full_parameterization` | `str` | `"precision"` | For `"full"`: the triangular factor parameterises the precision or the covariance (see [below](#full-covariance-precision-or-covariance-factor)) |
| `min_std` | `float` | `1e-3` | Minimum std for stability |
| `df` | `float` | `math.inf` | Component degrees of freedom: `math.inf` is Gaussian, finite gives Student-$t$ components (see [below](#student-t-components)) |
| `learn_df` | `bool` | `False` | Learn one shared `df = df · exp(raw_df)` (log scale, starting exactly at the finite `df`) |

### Model Output Size

=== "Diagonal covariance"

    $$\text{output\_dim} = \underbrace{K}_{\text{weights}} + \underbrace{K \cdot d}_{\text{means}} + \underbrace{K \cdot d}_{\text{log-stds}} = K(1 + 2d)$$

    ```python
    # K=3, d=2 → output_dim = 3(1 + 4) = 15
    model = nn.Sequential(
        nn.Linear(in_dim, 128), nn.ReLU(),
        nn.Linear(128, 15),
    )
    ```

=== "Full covariance"

    $$\text{output\_dim} = K + K \cdot d + K \cdot \frac{d(d+1)}{2}$$

    The last block parameterises lower-triangular factors $T_k$ (positive diagonal). By default they are Cholesky factors of the **precision**, $\Sigma_k^{-1} = T_k T_k^\top$; with `full_parameterization="covariance"`, $\Sigma_k = T_k T_k^\top$. The layout is the same for both.

### Full covariance: precision or covariance factor

With `covariance_type="full"` the network outputs, per component, a lower-triangular $T_k$. `full_parameterization` sets what it means:

| `full_parameterization` | Meaning of $T_k$ | Component log-density |
|:------------------------|:-----------------|:----------------------|
| `"precision"` (default since 0.3.0, as in sbi's MDN) | $P_k = \Sigma_k^{-1} = T_k T_k^\top$ | $-\tfrac12 \lVert T_k^\top (y-\mu_k)\rVert^2 + \sum_i \log T_{k,ii} - \tfrac d2 \log 2\pi$ |
| `"covariance"` (pre-0.3.0 behaviour) | $\Sigma_k = T_k T_k^\top$ | $-\tfrac12 \lVert T_k^{-1} (y-\mu_k)\rVert^2 - \sum_i \log T_{k,ii} - \tfrac d2 \log 2\pi$ |

The precision form is linear in $T_k$ and needs no triangular solve, which makes it much easier to optimise when there are many targets. It is not better everywhere. Harness run #6 (`density_parity`, test NLL, mean of 3 seeds, lower is better) against run #5 (covariance form), with sbi's MDN on the same splits:

| Dataset | Targets $d$ | `"covariance"` (run #5) | `"precision"` (run #6) | sbi MDN |
|:--------|:-----------:|------------------------:|-----------------------:|--------:|
| mt_scm20d | 16 | 3.11 | **0.33** | -0.01 |
| mt_jura | 3 | 2.53 | **1.95** | 1.92 |
| energy_efficiency | 2 | **-1.32** | -0.76 | -1.33 |
| mt_enb | 2 | **-1.18** | -0.85 | -1.24 |

!!! tip "Which parameterisation to use"
    Keep the default `"precision"`: it wins clearly on many targets (scm20d: 3.11 → 0.33). For low-dimensional targets ($d \le 2$ here) the covariance form did better, so also train with `full_parameterization="covariance"` and keep whichever has the lower **validation** NLL.

!!! warning "Sampling from a precision factor"
    In precision mode `_extract_distribution_parameters` returns the precision factor $T$ ($P = T T^\top$), not a covariance Cholesky factor. A covariance square root is $T^{-\top}$: a draw is $\mu + T^{-\top}\varepsilon$, **not** $\mu + T\varepsilon$. Use `loss_fn.sample(...)` (and `predict_interval`), which handle both parameterisations, rather than sampling from the extracted factors yourself. A harness suite that used $T$ as if it were $L$ got 0.00 interval coverage.

### Student-t components

With a finite `df`, every component is a multivariate Student-$t$ with the same location and scale-matrix layout:

$$
p(y \mid x) = \sum_{k=1}^K \pi_k(x)\; t_{\nu}\!\bigl(y \mid \mu_k(x),\, \Sigma_k(x)\bigr),
$$

where $\Sigma_k$ is the scale matrix (covariance $\tfrac{\nu}{\nu-2}\Sigma_k$). Both families are evaluated by one implementation, [`StudentTMixture`](student_t_mixture.md); the Gaussian MDN is its $\nu = \infty$ case, and the $\nu$-dependent terms converge to it as $O(1/\nu)$.

```python
import torch

from torchregress.losses import MixtureDensityLoss

loss_fn = MixtureDensityLoss(n_components=3, n_features=2, covariance_type="full", df=5.0)
# or learn one shared df: MixtureDensityLoss(..., df=5.0, learn_df=True)
y_pred = torch.randn(64, loss_fn.expected_output_size)
loss = loss_fn(y_pred, torch.randn(64, 2))

mixture = loss_fn.distribution(y_pred)  # StudentTMixture
q = mixture.marginal_quantile([0.05, 0.5, 0.95])  # exact per-target quantiles [64, 2, 3]
```

!!! tip "When Student-t components help"
    Heavy-tailed or contaminated targets: a Gaussian component must widen (or a spare component must chase outliers) to cover them, while a Student-$t$ component down-weights them. With `learn_df=True`, include `loss_fn.parameters()` in the optimiser, ideally as its own group at a learning rate of about `1e-2` (`df` is learned on the log scale, so it moves by a fixed fraction per step toward heavy tails or the Gaussian limit). On held-out NLL a learned `df` was never more than 0.035 nats behind the best Gaussian mixture in the harness benchmark, while a fixed `df = 5` lost up to 0.13 nats on Gaussian-like data, so prefer `learn_df=True` to a fixed small `df`. `predict_mean_std` returns infinite standard deviations for $\nu \le 2$.

`loss_fn.distribution(y_pred)` also gives the masked (partially observed) log density, conditioning on a measured target and closed-form likelihoods of noisy linear observations; see [Student-t mixtures](student_t_mixture.md).

### Training

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    params = model(x_train)           # (batch, output_dim)
    loss = loss_fn(params, y_train)   # MDN NLL
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## Inference Methods

### `predict_mean_std`

Gaussian-approximation summary (mean and variance of the mixture):

```python
mean, std = loss_fn.predict_mean_std(model(x_test))
```

!!! warning "Multimodal caveat"
    For genuinely multimodal distributions, the mean may fall **between** modes where no data exists.  Use `predict_interval` (exact marginal quantiles) or `sample` instead.

### `predict_interval`

Central intervals from the exact quantiles of each target's marginal mixture CDF (`StudentTMixture.marginal_quantile`): deterministic, and valid for multimodal mixtures and heavy tails. (`n_samples` is accepted and ignored; it was the Monte Carlo sample size of earlier releases.)

```python
lower, upper = loss_fn.predict_interval(model(x_test), confidence=0.95)
```

### `sample`

Draw samples from the learned mixture:

```python
samples = loss_fn.sample(model(x_test), n_samples=1000)
# samples shape: [n_samples, batch, n_features]
```

---

## Practical Tips

!!! tip "Avoiding component collapse"
    Monitor component usage during training:
    ```python
    log_weights, means, scales = loss_fn._extract_distribution_parameters(params)
    active = (log_weights.exp() > 0.05).float().mean()  # should stay > 0.5
    ```

!!! tip "Initialisation"
    Initialise the final layer with small weights to prevent early commitment to one component.

!!! tip "Start simple"
    Begin with $K = 3$ diagonal components. Increase $K$ only if validation NLL improves.

---

## Limitations & Optimization Challenges

While MDNs are highly expressive, they present several training and optimization challenges:

1. **Numerical Instability & Log-Sum-Exp**:
   Computing the NLL directly via $p(y \mid x) = \sum_k \pi_k \mathcal{N}(y \mid \mu_k, \sigma_k^2)$ involves exponentials that can easily underflow or overflow. **torchregress** internally utilizes the `log-sum-exp` trick:
   $$\log p(y \mid x) = \text{log-sum-exp}\left(\log \pi_k + \log \mathcal{N}(y \mid \mu_k, \sigma_k^2)\right)$$
   Always use a non-zero `min_std` (e.g., $10^{-3}$) to prevent variance collapse ($\sigma_k^2 \rightarrow 0$), which causes division by zero.

2. **Computational Scaling**:
   The forward pass cost is $\mathcal{O}(K \cdot d)$ for diagonal covariance and $\mathcal{O}(K \cdot d^2)$ for full covariance per sample. NLL evaluation adds determinant and solve operations that scale as $\mathcal{O}(d^3)$ for full covariance. For large $K$ or high-dimensional targets, prefer diagonal covariance or consider flow-based alternatives.

3. **Mode Collapse**:
   The network may collapse into using only a single mixture component, effectively ignoring the others ($\pi_k \approx 0$ for $k > 1$). To monitor this, check active components during validation:
   $$\text{active\_fraction} = \frac{1}{B} \sum_{i=1}^B \sum_{k=1}^K \mathbf{1}_{\pi_k^{(i)} > 0.05}$$
   Mode collapse is especially common with large $K$ (> 10) or strong regularization.

4. **Initialization Sensitivity**:
   If initialized poorly, components can "claim" the same regions of target space. Initialize the final output layer weight matrices with small random values to diversify components early in training.

5. **Dtype Sensitivity**:
   The log-sum-exp trick is numerically sensitive in `float16` or `bfloat16`. Use `float32` for the NLL computation and only cast down after the loss is computed. Mixed-precision training with MDNs requires careful gradient scaling.

6. **Covariance Type Constraints**:
   With `"full"` covariance the head has $K \cdot d(d+1)/2$ factor outputs and memory grows as $\mathcal{O}(K \cdot d^2)$ per sample. The default precision parameterisation trains on 16 targets (scm20d above); for very large $d$ prefer diagonal components or a flow.

---

## Next steps

- [Student-t mixtures](student_t_mixture.md) — the distribution behind the head: masks, conditioning, linear observations, censoring, exact quantiles
- [Normalizing Flows](nflows.md) — more flexible distributions for complex targets
- [EIV + MDN](eiv.md#multimodal-eiv-variants) — marginalize over input noise with MDN heads
- [MDN Ensembles](../methods/ensemble/methods.md#mdnensemblemodel) — mixture-of-mixtures for decomposed uncertainty
- [Multimodal comparison](../examples/multimodal_method_comparison.md) — benchmark MDN vs flows vs SLS

---

## References

| # | Reference |
|:-:|:----------|
| 1 | C.M. Bishop. ["Mixture Density Networks."](https://publications.aston.ac.uk/id/eprint/373/) Neural Computing Research Group Report NCRG/94/004, **1994**. |
| 2 | C.M. Bishop. *Pattern Recognition and Machine Learning*, §5.6. Springer, **2006**. |
| 3 | K.L. Lange, R.J.A. Little and J.M.G. Taylor. ["Robust statistical modeling using the t distribution."](https://doi.org/10.1080/01621459.1989.10478852) *JASA* 84(408):881–896, **1989**. |
