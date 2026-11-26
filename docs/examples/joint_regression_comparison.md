# Joint Regression Comparison

→ Guide: [Joint multi-target regression](../guide/joint-regression.md). API: [`JointRegressor`](../api/joint.md), [`GaussianJoint`](../api/joint.md), [`energy_score`](../api/metrics.md), [`variogram_score`](../api/metrics.md).

This example fits every head of `JointRegressor` on one three-target problem whose true conditional density is known, and scores them on the same test rows next to the oracle and to "independent targets" (the Gaussian head's per-target means and variances with the correlations dropped).

Script: `examples/joint_regression_comparison.py`

## The problem

Four features, three targets. Given $\mathbf x$ the targets follow a two-component Gaussian mixture,

$$
\mathbf y \mid \mathbf x \;\sim\; \pi(\mathbf x)\,\mathcal N\big(\mathbf m(\mathbf x) + \boldsymbol\delta,\ \Sigma(\mathbf x)\big) + \big(1 - \pi(\mathbf x)\big)\,\mathcal N\big(\mathbf m(\mathbf x) - \boldsymbol\delta,\ \Sigma(\mathbf x)\big),
$$

with $\boldsymbol\delta = (0.8, -0.8, 0.4)$, $\pi(\mathbf x) = \sigma(2 x_2)$, a noise scale that grows with $|x_0|$, and a correlation between targets 0 and 1 equal to $0.8 \tanh(x_3)$. The two modes move all three targets together, so a single Gaussian cannot represent them, and the correlation changes sign with $x_3$, so independent targets cannot represent it either.

## What it compares

| Row | What it is |
|:--|:--|
| `oracle` | The true conditional density (the floor for every score) |
| `independent targets` | The Gaussian head's means and variances, correlations set to zero |
| `JointRegressor('gaussian')` | One Gaussian per row with a full covariance |
| `JointRegressor('mixture')` | Mixture density network |
| `JointRegressor('flow')` | Conditional spline flow |
| `JointRegressor('sklar')` | Per-target marginals (a quantile or a mixture network, whichever scores better on validation rows) joined by the dependence ladder |
| `JointRegressor('energy')` | Generative network trained by the energy score (no density) |
| `JointRegressor('auto')` | The default: the density heads stacked by validation log score |

## Metrics reported

| Column | Definition | Better |
|:--|:--|:-:|
| `NLL`, `NLLGap` | Mean test negative log density, and its excess over the oracle (nats) | lower |
| `Energy` | Energy score $\mathbb E\lVert \mathbf Y - \mathbf y\rVert - \tfrac12 \mathbb E\lVert \mathbf Y - \mathbf Y'\rVert$ from 200 draws | lower |
| `Variogram` | Variogram score of order 0.5 from the same draws (sensitive to dependence) | lower |
| `Cov90` | Share of test rows whose whole target vector is inside the 90% conformal region | 0.90 |
| `LogVol90` | Median log volume of that region | lower at equal coverage |
| `BoxCov90` | Share of rows with every target inside its own central 90% interval | (below 0.90 by construction) |
| `train_s`, `eval_s` | Wall-clock seconds | lower |

## Output of one run

Default configuration (4,000 training rows, 1,000 test rows, seed 0, laptop CPU with 4 threads):

```text
Method                     | NLL    | NLLGap | Energy | Variogram | Cov90  | LogVol90 | BoxCov90 | train_s | eval_s
---------------------------+--------+--------+--------+-----------+--------+----------+----------+---------+---------
oracle                     | 2.0742 | 0.0000 | 0.7454 | 0.3175    | -      | -        | 0.7140   | -       | -
independent targets        | 3.0660 | 0.9919 | 0.7846 | 0.3827    | -      | -        | 0.7610   | -       | -
JointRegressor('gaussian') | 2.6468 | 0.5726 | 0.7722 | 0.3489    | 0.9070 | 2.5530   | 0.7640   | 1.1029  | 0.1306
JointRegressor('mixture')  | 2.2375 | 0.1634 | 0.7518 | 0.3201    | 0.8970 | 2.0923   | 0.7360   | 0.8170  | 0.4885
JointRegressor('flow')     | 2.3342 | 0.2601 | 0.7623 | 0.3260    | 0.9150 | 2.4272   | 0.7300   | 2.8196  | 12.8047
JointRegressor('sklar')    | 2.4893 | 0.4152 | 0.7699 | 0.3319    | 0.8850 | 2.4013   | 0.7260   | 7.1927  | 13.3084
JointRegressor('energy')   | -      | -      | 0.7642 | 0.3328    | 0.9060 | 2.9326   | 0.6960   | 1.7958  | 0.2675
JointRegressor('auto')     | 2.2128 | 0.1386 | 0.7523 | 0.3209    | 0.8930 | 2.1056   | 0.7150   | 15.4346 | 10.3603
```

Reading it, for this problem and this seed:

- Dropping the correlations costs 0.42 nats per row against the same Gaussian head (0.99 against 0.57), and it shows in the variogram score (0.383 against 0.349) much more than in the energy score.
- The stacked default has the smallest gap to the oracle (0.14 nats). The mixture head alone is close (0.16 nats, with a region of the same size), because the truth is a mixture.
- Every conformal region is within two standard errors of 0.90 (one standard error is about 0.015 with 600 calibration rows and 1,000 test rows). Per-target 90% intervals taken together cover 70% to 76%.
- The energy head needs no density family and evaluates in a quarter of a second where the flow needs thirteen, but it has no log density and its region is the largest.
- The Sklar route picks mixture marginals and the conditional-mixture dependence rung on its validation rows and ends at 0.42 nats, between the single Gaussian (0.57) and the flow (0.26): the two modes move all three targets together, and that shared mode has to be learned by the dependence model from the normal scores.

!!! warning "One problem, one seed"
    The ranking is an illustration of what each head can and cannot represent, not a benchmark. The harness suite `joint_regression` in torchregress-harness runs the same heads on several designs and real datasets with seeds and external baselines.

## Fairness and comparability

- One seed for the data, the splits, initialisation and the evaluation draws; every head sees the same rows.
- Same `max_epochs` and `patience` for every head; estimator defaults otherwise (60% of the rows train, the rest validate, recalibrate and calibrate the region).
- `independent targets` is not trained separately: it reuses the Gaussian head, so the difference between the two rows is the correlation alone.

## When this fails

- A two-component Gaussian mixture favours the mixture head; on heavy-tailed or strongly non-Gaussian marginals the order changes.
- With 4,000 rows the flow and the Sklar dependence model are under-trained relative to what they reach on larger samples.
- Region volumes are Monte Carlo estimates.

## Run

```bash
pixi run python examples/joint_regression_comparison.py
```

## Summary artifact

```bash
pixi run python examples/joint_regression_comparison.py \
  --summary-json-path reports/example_summaries/joint_regression_comparison.json
```

## References

| # | Reference |
|:-:|:----------|
| 1 | Gneiting, T. & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *JASA* 102(477). |
| 2 | Scheuerer, M. & Hamill, T. M. (2015). Variogram-based proper scoring rules for probabilistic forecasts of multivariate quantities. *Monthly Weather Review* 143(4). |
| 3 | Shen, X. & Meinshausen, N. (2024). Engression: extrapolation through the lens of distributional regression. *JRSS-B*. |
