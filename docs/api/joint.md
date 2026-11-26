# Joint API

Reference for `torchregress.joint`: joint predictive distributions over several
targets, the estimator that fits them, the dependence ladder, conformal regions,
recalibration and calibration diagnostics. Derivations, guarantees, the choice of
head, rung and score, and a complete example are in the
[Joint Multi-Target Regression guide](../guide/joint-regression.md).

```python
import torchregress as tr

joint = tr.joint  # lazy submodule
from torchregress.joint import JointRegressor, JointConformal, LatentScore
```

**Layouts.** Every predictive holds `n` rows of distributions over `d` targets.
Draws are `[n, S, d]` (batch first, always). Densities take `[n, d]` and return
`[n]`, or take `[n, S, d]` and return `[n, S]`. Masks are boolean `[n, d]` with
`True` = observed; a masked target is integrated out (marginalised), not dropped
from a reduction. The sample-based scores of `torchregress.metrics`
(`energy_score`, `variogram_score`, `crps_from_samples`) take draws sample-first,
`[S, n, d]`: pass `samples.movedim(1, 0)`.

---

## Estimator

Module `torchregress.joint.estimator`.

| Symbol | Signature | Description |
|:-------|:----------|:------------|
| `JointRegressor` | `JointRegressor(method="auto", *, marginals=None, hidden=(128, 128), n_components=5, rank=None, max_epochs=300, patience=30, batch_size=256, lr=2e-3, weight_decay=1e-5, val_fraction=0.15, recal_fraction=0.1, conformal_fraction=0.15, recalibrate=True, score="auto", n_folds=3, device=None, seed=0)` | Arrays in, joint predictives and conformal regions out. `method` is `"auto"` (stack of heads), `"gaussian"`, `"mixture"`, `"flow"` or `"sklar"`. `score="auto"`: `LatentScore` for a Gaussian predictive, `HPDScore` for any other predictive with a density, `SampleBallScore` for draws only, masked `MahalanobisScore` for conformal rows with missing targets. |
| `MarginalEngine` | `MarginalEngine()` | Interface for per-target predictors used by the Sklar route: implement `fit(x, y, mask=None)` and `predict(x) -> Marginals`. |
| `QuantileNetworkEngine` | `QuantileNetworkEngine(n_quantiles=31, hidden=(128, 128), *, max_epochs=300, patience=30, batch_size=256, lr=2e-3, weight_decay=1e-5, val_fraction=0.15, seed=0)` | Built-in marginal engine: one network predicting a non-crossing grid of quantiles per target, trained with the pinball loss; missing targets are left out of the loss. |

`JointRegressor` methods and attributes:

| Member | Description |
|:-------|:------------|
| `fit(X, Y)` | Fit on `X [N, p]` and `Y [N, d]` (`NaN` = missing target, `±inf` is an error, at least 50 rows). Standardises, fits the heads, stacks them, fits the scale recalibration and stores the conformal split. Deterministic for a given `seed`; the caller's torch RNG state is restored. Returns `self`. |
| `predict(X)` | `JointPredictive` of the rows of `X`, in data units, recalibrated. |
| `calibrate(X, Y)` | Replace the conformal rows with `X [N, p]`, `Y [N, d]` (for example fresh simulations) that were not used for fitting; same validation as `fit`; clears the cached calibrations. Returns `self`. |
| `region(X, alpha=0.1, *, interest=None)` | `JointRegion` calibrated on the held-out conformal split. `alpha` may be a list (nested regions); `interest` restricts the region to a subset of targets, integrating the others out first. When conformal rows have missing targets, `contains(y)` is calibrated on the complete conformal rows (guarantee for complete test rows; `ValueError` if there are fewer than `1/alpha - 1`) and `contains(y, mask=m)` on the masked score of all rows (guarantee for the same missingness mechanism). |
| `members_`, `weights_` | Names of the fitted heads and their stacking weights `[K]`. |
| `validation_log_score_` | Mean validation log density per head in the units of the data (comparable with `predict(X).log_prob(Y).mean()` before recalibration; over observed targets when targets are missing). |
| `n_train_`, `n_val_`, `n_recal_`, `n_conformal_` | Rows in each split: `round(fraction * N)` (at least 8) for validation, recalibration and conformal rows, the rest (60% by default) for training; `n_conformal_` follows `calibrate`. |
| `dependence_scores_` | Validation log score of each dependence rung (Sklar route). |
| `scale_` | Per-target recalibration factors `[d]` actually applied (ones when the recalibration was skipped or rejected), or `None`. |

---

## Predictive distributions

Module `torchregress.joint.predictive`.

| Symbol | Signature | Description |
|:-------|:----------|:------------|
| `JointPredictive` | abstract | Base class. Required: `batch_size`, `event_dim`, `sample(n_samples)`. Declared by every built-in head: `dtype`, `device` (no draw needed). Optional: `log_prob(y, mask=None)`, `to_latent(y)`, `from_latent(z)`, `marginal(dims)`. Derived: `mean(n_samples=512)`, `covariance(n_samples=512)`, `hpd_level(y, n_samples=512)` (`[n, d] -> [n]` or `[n, S, d] -> [n, S]`, candidates of a row sharing one set of reference draws), `as_input(y)`. Inputs (`y`, `z`, masks) may be tensors of any floating dtype on any device or NumPy arrays: they are cast to the predictive's `dtype` and moved to its `device`, and results stay there. Flags: `has_density`, `has_latent`, `exact_marginals`. |
| `GaussianJoint` | `GaussianJoint(mean, scale_tril)`; `GaussianJoint.from_covariance(mean, covariance)`; `GaussianJoint.from_low_rank(mean, cov_factor, cov_diag)` | Multivariate normal with Cholesky factor `[n, d, d]` or shared `[d, d]`. Closed-form marginals, masked densities, latent map and `condition(dims, values)`; `select(index)` picks rows. |
| `DistributionJoint` | `DistributionJoint(distribution)` | Adapter for a `torch.distributions.Distribution` with batch `[n]` and event `[d]` (`StudentTMixture`, `MultivariateNormal`, zuko flows). Masked densities when the distribution's `log_prob` takes a `mask`; latent map for zuko flows. |
| `FlowJoint` | `FlowJoint(flow_loss, context)` | Predictive of a `NormalizingFlowLoss` head for network outputs `context [n, c]`; samples through the loss in blocks. |
| `ConditionalModelJoint` | `ConditionalModelJoint(model, context, event_dim=None)` | Predictive of any model with `log_prob(context, target) -> [n]` and `sample(context, n_samples) -> [n, S, d]`: `NormalizingFlowLoss`, `ContrastiveFlowLoss`, `ensemble.DensityEnsemble`, `models.MarginalContrastiveDensity`, `models.NormalizingFlowMap`. `[n, S, d]` densities repeat the context; no latent map, no masks, marginals by sampling. |
| `SampleJoint` | `SampleJoint(samples)` | Draws-only predictive from stored draws `[n, S0, d]`; `sample(S)` subsamples or resamples them. |
| `MaskedMarginalJoint` | `MaskedMarginalJoint(base, dims)` | Exact marginal of a head that integrates out masked targets (mixtures and stacks of them): keeps a density. Returned by `marginal(dims)` for heads with `has_density` and `exact_marginals`. |
| `ProjectedJoint` | `ProjectedJoint(base, dims)` | Draws-only marginal of another predictive: the result of `marginal(dims)` for heads that cannot integrate targets out exactly (flows). |
| `AffineJoint` | `AffineJoint(base, loc, scale)` | Per-target change of units `y = loc + scale * u`, with the Jacobian in the density. Used to map standardised heads back to data units. |
| `MixtureJoint` | `MixtureJoint(members, weights=None)` | Weighted mixture of predictives (`[K]` or per-row `[n, K]` weights); the stacked predictive of `JointRegressor`. |

---

## Marginals

Module `torchregress.joint.marginals`. Each holds `n` rows times `d` targets of
continuous univariate predictives with `cdf`, `icdf`, `log_prob`,
`normal_scores(y, eps=1e-6)`, `from_normal_scores(z, eps=1e-6)`, `median()` and
`select_targets(dims)`.

| Symbol | Signature | Description |
|:-------|:----------|:------------|
| `Marginals` | abstract | Base class; subclasses implement `cdf`, `icdf`, `log_prob` and `select_targets`. |
| `GaussianMarginals` | `GaussianMarginals(mean, std)` | Normal marginals; normal scores are the standardised residuals, without clipping. |
| `QuantileMarginals` | `QuantileMarginals(levels, quantiles, tail_knots=None)` | Quantile grid `[n, d, Q]`: linear CDF between knots, exponential tails fitted on the outer knots; crossing quantiles are sorted and ties separated. |
| `BinnedMarginals` | `BinnedMarginals(edges, probs, tail_mass=1e-3, min_prob=1e-6)` | Histogram marginals (bin edges and probabilities), piecewise-constant density with exponential tails. |
| `SampleMarginals` | `SampleMarginals(samples, tail_knots=None)` | Marginals read from draws `[n, S, d]` as quantiles at plotting positions `k / (S + 1)`. |

---

## Dependence

Module `torchregress.joint.dependence`. A `Dependence` models the density of the
normal scores; `SklarJoint` joins it to `Marginals`.

| Symbol | Signature | Description |
|:-------|:----------|:------------|
| `SklarJoint` | `SklarJoint(marginals, scores, eps=1e-6)` | Joint predictive from per-target marginals and a predictive of their normal scores; a valid density for any score density. Flags follow `scores`. |
| `Dependence` | abstract | Base of the ladder: `fit(z, x=None)` on out-of-fold training scores, `predictive(n, x=None)`, `log_prob(z, x=None)`, `marginal_drift(z, x=None, n_samples=256)`; attributes `name`, `has_density`, `conditional`, `fit_time_`. |
| `IndependenceDependence` | `IndependenceDependence()` | Rung 0: independent standard-normal scores (product of the marginals). |
| `SchaakeShuffle` | `SchaakeShuffle(n_neighbors=None)` | Rung 1: empirical copula of the training scores; with `n_neighbors=k`, the `k` nearest training rows in standardised `x`. Draws only. |
| `GaussianCopula` | `GaussianCopula(shrinkage="auto", rank=None)` | Rung 2: constant Gaussian copula with Ledoit–Wolf shrinkage (or a fixed intensity) and optional low-rank-plus-diagonal correlation; `correlation_`, `shrinkage_`. |
| `ConditionalGaussianCopula` | `ConditionalGaussianCopula(hidden=(64, 64), max_epochs=200, patience=20, lr=1e-3, batch_size=256, val_fraction=0.15, seed=0)` | Rung 3: Gaussian score density with covariance depending on `x` (modified Cholesky), trained by NLL from the independence start. |
| `FlowDependence` | `FlowDependence(n_transforms=3, hidden=(64, 64), max_epochs=300, patience=25, lr=1e-3, bound=5.0, bins=8, batch_size=256, val_fraction=0.15, seed=0)` | Rung 5: conditional (or unconditional) spline flow on the scores, initialised at the identity, trained by NLL with early stopping. |
| `TemplateJoint` | `TemplateJoint(template, n, index=None)` | Draws-only predictive resampling template rows (per-row candidate rows via `index [n, k]`); the predictive of `SchaakeShuffle`. |
| `select_dependence` | `select_dependence(candidates, z_val, x_val=None, n_se=2.0)` | Lowest rung that scores at least as well as independence and within `n_se` paired standard errors of the best on held-out scores. Returns `(chosen, {name: mean log q})`. |

---

## Conformal regions

Module `torchregress.joint.regions`.

| Symbol | Signature | Description |
|:-------|:----------|:------------|
| `JointConformal` | `JointConformal(score, alpha=0.1)` | Split conformal calibration. `calibrate(predictive, y, *, mask=None, weights=None, groups=None)` stores `thresholds_`, `group_ids_`, `n_calibration_`; `region(predictive, *, test_weights=None, groups=None)` returns a `JointRegion`. Thresholds come from `losses.conformal.finite_sample_quantile` (rank `ceil((n+1)(1-alpha))`) and its weighted counterpart, as in `MultivariateScoreConformal`. |
| `JointRegion` | built by `JointConformal.region` | Calibrated regions, one per row and level: `contains(y, mask=None)`, `log_volume(n_samples=1024)` (estimate and standard error), `sample(n_samples, *, level=None, max_rounds=50)`, `bounding_box(n_samples=2048)`; attributes `predictive`, `score`, `alpha`, `threshold [n, L]`, `mask_region` (region used by `contains` when a mask is passed, set by `JointRegressor.region`). A `NaN` score counts as `+inf`. |
| `ConformalScore` | abstract | Base score: `score(predictive, y, mask=None)`; attributes `name`, `supports_mask`. |
| `DensityScore` | `DensityScore()` | Negative log density (density level set). Needs a density; masks need `exact_marginals`. |
| `HPDScore` | `HPDScore(n_samples=256)` | Highest-density level from reference draws (adaptive level set). Needs a density. |
| `LatentScore` | `LatentScore()` | Norm of the latent residual (pre-image of a latent ball). Needs a latent map. |
| `MahalanobisScore` | `MahalanobisScore()` | Mahalanobis distance from the predictive mean and covariance (ellipsoid); with a mask, the chi-square CDF of the observed quadratic form. Closed-form volume and box. |
| `MaxResidualScore` | `MaxResidualScore()` | Largest standardised absolute residual (box with a joint guarantee). Accepts masks; closed-form volume and box. |
| `SampleBallScore` | `SampleBallScore(n_samples=64)` | Distance to the nearest predictive draw (union of balls). Draws only. |

---

## Recalibration

Module `torchregress.joint.calibration`. Each recalibrator has
`fit(predictive_cal, y_cal)` and `transform(predictive)`.

| Symbol | Signature | Description |
|:-------|:----------|:------------|
| `ScaleRecalibrator` | `ScaleRecalibrator(objective="variance", n_samples=1024, max_iter=100)` | One scale factor per target about the predictive mean (`"variance"` or held-out `"nll"`); `fit(predictive, y, mask=None)`, `scale_`; `transform` returns an `AffineJoint`. |
| `LatentRecalibrator` | `LatentRecalibrator(n_knots=None)` | Radial map of the latent space so that the latent norm follows its calibration distribution; `spline_`; `transform` returns a `LatentRecalibratedJoint`. Needs a latent map. |
| `LatentRecalibratedJoint` | built by `LatentRecalibrator.transform` | Predictive with a remapped latent norm; keeps a density and a latent map, marginals by sampling. |
| `MarginalRecalibrator` | `MarginalRecalibrator(n_knots=None, n_samples=1000)` | Per-target conformal calibrators (split conformal predictive systems), dependence kept; `fit(predictive, y, mask=None)`, `scores`, `pit`, `calibrators_`. `transform` returns a `MarginalRecalibratedJoint` for a `GaussianJoint` and a draws-only `SampleJoint` otherwise. |
| `MarginalRecalibratedJoint` | built by `MarginalRecalibrator.transform` | Gaussian predictive with recalibrated marginals and its copula kept; density, latent map, exact marginals and `marginal_cdf(y)`. |
| `latent_calibration_error` | `latent_calibration_error(predictive, y)` | Latent calibration error (L-ECE) of a predictive with a latent map: distance of the chi-CDF of the latent norms from uniform. Scalar float64. |

---

## Diagnostics

Module `torchregress.joint.diagnostics`. Each function returns tensors in a small
frozen dataclass, exported from `torchregress.joint`: `PreRankHistograms`,
`NEESConsistency`, `HPDCalibration`, `CoverageByMask`, `WorstSlabCoverage`,
`CoverageGap`, `DependenceError` (fields in their docstrings). `PRERANKS` and
`holm_adjust` stay in `torchregress.joint.diagnostics`.

| Symbol | Signature | Description |
|:-------|:----------|:------------|
| `prerank_histograms` | `prerank_histograms(samples, y, preranks=None, standardize=True, n_bins=10, seed=0)` | Multivariate rank histograms for the `location`, `scale`, `dependence`, `energy`, `average_rank` and `band_depth` pre-ranks (default: all, without `dependence` when `d = 1`), with chi-square p-values and Holm adjustment. Returns `PreRankHistograms`. |
| `nees_consistency` | `nees_consistency(mean, covariance, y, confidence=0.95, mask=None)` | Normalised estimation error squared, its chi-square acceptance band, verdict and per-eigen-direction variances. Returns `NEESConsistency`. |
| `hpd_calibration` | `hpd_calibration(levels, n_bins=20, confidence=0.95)` | Calibration curve of HPD levels with a DKW band and a KS test. Returns `HPDCalibration`. |
| `coverage_by_mask` | `coverage_by_mask(covered, mask)` | Coverage and binomial standard error per missingness pattern. Returns `CoverageByMask`. |
| `worst_slab_coverage` | `worst_slab_coverage(x, covered, delta=0.2, n_directions=1000, seed=0, n_grid=50)` | Lowest coverage over feature-space slabs, selected on one half and evaluated on the other. Returns `WorstSlabCoverage`. |
| `conditional_coverage_gap` | `conditional_coverage_gap(x, covered, alpha, n_splits=5, seed=0, n_fourier=64, ridge=1e-3)` | Excess risk of the target coverage (ERT) from an out-of-fold classifier of the coverage indicator. Returns `CoverageGap`. |
| `dependence_error` | `dependence_error(samples, y)` | Average predicted correlation against the empirical correlation of standardised residuals, and their Frobenius distance. Returns `DependenceError`. |

---

## See also

- [Joint Multi-Target Regression](../guide/joint-regression.md): concepts, guarantees and a worked example.
- Entry pages with worked examples: [New to Joint Prediction](../guide/joint-quickstart.md), [Coming from MCMC and HMC](../guide/joint-for-mcmc-users.md), [Coming from Simulation-Based Inference](../guide/joint-for-sbi-users.md).
- [Losses API](losses.md): `MultivariateGaussianLoss`, `LowRankGaussianLoss`, `MixtureDensityLoss`, `NormalizingFlowLoss`.
- [Conformal API](conformal.md): single-target and coordinate-wise conformal predictors; `MultivariateScoreConformal` and `ConformalizedDensity` are the raw-tensor counterparts of `JointConformal` with `MahalanobisScore` / `DensityScore` (see [Relation to existing tools](../guide/joint-regression.md#12-relation-to-existing-tools)).
- [Metrics API](metrics.md): `EnergyScore`, `VariogramScore` and other multivariate scores.
