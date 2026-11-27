# Joint API

Reference for `torchregress.joint`: joint predictive distributions over several
targets, the estimator that fits them, the dependence ladder, conformal regions,
recalibration and calibration diagnostics. Explanations, guarantees and a worked
example are in the [Joint Multi-Target Regression guide](../guide/joint-regression.md).

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

| Layer | Names | Section |
|:--|:--|:--|
| Shortcut | `JointRegressor`, `MarginalEngine`, `QuantileNetworkEngine`, `MixtureNetworkEngine` | [Estimator](#estimator) |
| From a head to a joint predictive | `as_joint`, `CopulaSplineJoint`, `SLSRegion` (and `inference.RatioPosterior`) | [Adapters](#adapters-for-the-librarys-heads) |
| Joint predictives | `JointPredictive`, `GaussianJoint`, `DistributionJoint`, `FlowJoint`, `ConditionalModelJoint`, `SampleJoint`, `MixtureJoint`, `AffineJoint`, `MaskedMarginalJoint`, `ProjectedJoint` | [Predictive distributions](#predictive-distributions) |
| Per-target predictions and dependence | `Marginals` and subclasses, `SklarJoint`, the `Dependence` ladder, `select_dependence` | [Marginals](#marginals), [Dependence](#dependence) |
| Regions | `JointConformal`, `JointRegion`, `CrossConformalRegion`, the scores | [Conformal regions](#conformal-regions) |
| Recalibration | `ScaleRecalibrator`, `LatentRecalibrator`, `MarginalRecalibrator` | [Recalibration](#recalibration) |
| Diagnostics | `prerank_histograms`, `hpd_calibration`, `nees_consistency`, `dependence_error`, `coverage_by_mask`, `worst_slab_coverage`, `conditional_coverage_gap` | [Diagnostics](#diagnostics) |
| Plots | `plot_corner_plot`, `plot_joint_diagnostics` (in `torchregress.viz`) | [Joint predictive plots](../methods/visualization.md#4-joint-predictive-plots-corner-plots-and-diagnostics) |

---

## Estimator

Module `torchregress.joint.estimator`.

| Symbol | Signature | Description |
|:-------|:----------|:------------|
| `JointRegressor` | `JointRegressor(method="auto", *, marginals=None, hidden=(128, 128), n_components=5, rank=None, max_epochs=300, patience=30, batch_size="auto", lr=2e-3, weight_decay=1e-5, val_fraction=0.1, recal_fraction=0.05, conformal_fraction=0.1, max_holdout=20_000, cv=None, recalibrate=True, clip_features=True, score="auto", n_folds=3, n_members=None, noise_dim=None, energy_draws=2, device=None, seed=0)` | Arrays in, joint predictives and conformal regions out. `method` is `"auto"` (stack of heads), `"gaussian"`, `"mixture"`, `"flow"`, `"sklar"` or `"energy"` (engression, Shen and Meinshausen 2024: a generator $g(\mathbf x, \boldsymbol\varepsilon)$ with Gaussian noise of dimension `noise_dim` (default $d$) concatenated to the features, trained by the energy score of `energy_draws` draws per row; draws only, so no log score, no HPD region, never in `"auto"`; missing targets enter through the energy score of the observed sub-vector). `score="auto"`: `LatentScore` for a Gaussian predictive, `HPDScore` for any other predictive with a density, `SampleBallScore` for draws only, masked `MahalanobisScore` for conformal rows with missing targets. `batch_size="auto"`: 256, doubled while an epoch would take more than 400 steps, up to 4096. `max_holdout`: cap on the rows of each held-out split, so that beyond about 100,000 rows the extra rows train the heads. The scale recalibration is taken about `center()` for heads with a density, so scoring needs no draws; for the draws-only energy head it is the variance objective about the Monte Carlo mean, kept only when it lowers the validation energy score. `n_members` is the number of independently initialised copies of each network head, averaged with equal weights (a deep ensemble): `None` means 3 for `method="auto"` and 1 for a single named head; with more than one copy a head's predictive is a mixture of its copies. `clip_features` clamps every feature to the range of the training rows before it enters the networks, so a wildly out-of-range input (a glitched sensor) is predicted like the nearest in-range input instead of by extrapolation; rows inside the range are untouched. `cv=K` ($K \ge 2$) cross-fits for small data: `K` fold models, each fitted on the other folds (inner validation split for early stopping only, one copy per head unless `n_members` is given), stacking and the scale recalibration on the out-of-fold predictive of every row, `predict` the equal-weight average of the fold models, and `region` a `CrossConformalRegion`; `folds_` holds the fold of each row, `fold_choices_` the per-fold Sklar choices. |
| `MarginalEngine` | `MarginalEngine()` | Interface for per-target predictors used by the Sklar route: implement `fit(x, y, mask=None)` and `predict(x) -> Marginals`. |
| `QuantileNetworkEngine` | `QuantileNetworkEngine(n_quantiles=31, hidden=(128, 128), *, max_epochs=300, patience=30, batch_size=256, lr=2e-3, weight_decay=1e-5, val_fraction=0.15, seed=0)` | Built-in marginal engine: one network predicting a non-crossing grid of quantiles per target, trained with the pinball loss; missing targets are left out of the loss. `predict` returns `QuantileMarginals`. |
| `MixtureNetworkEngine` | `MixtureNetworkEngine(n_components=5, hidden=(128, 128), *, min_std=1e-2, max_epochs=300, patience=30, batch_size=256, lr=2e-3, weight_decay=1e-5, val_fraction=0.15, seed=0)` | Built-in marginal engine: one network predicting a univariate Gaussian mixture per target (the output layout and parametrisation of a one-feature diagonal `MixtureDensityLoss`: `log_softmax` weights, `softplus + min_std` standard deviations), trained by the sum of the marginal negative log densities; missing targets are left out of the loss. Starts at the same unit-variance mixture for every row. `predict` returns `MixtureMarginals`. Expects standardised targets (`min_std` is in their units), as `JointRegressor` supplies. No component is wider than 1.25 times the largest deviation of its target from the training mean, nor centred further than that from it. |

`JointRegressor` methods and attributes:

| Member | Description |
|:-------|:------------|
| `fit(X, Y)` | Fit on `X [N, p]` and `Y [N, d]` (`NaN` = missing target, `±inf` is an error, at least 50 rows). Standardises, fits the heads, stacks them, fits the scale recalibration and stores the conformal split. Deterministic for a given `seed`; the caller's torch RNG state is restored. Returns `self`. |
| `predict(X)` | `JointPredictive` of the rows of `X`, in data units, recalibrated. |
| `calibrate(X, Y)` | Replace the conformal rows with `X [N, p]`, `Y [N, d]` (for example fresh simulations) that were not used for fitting; same validation as `fit`; clears the cached calibrations. With `cv`, `region` is then the split conformal region of the averaged predictive on these rows ($1 - \alpha$) instead of the cross-conformal one. Returns `self`. |
| `region(X, alpha=0.1, *, interest=None)` | `JointRegion` calibrated on the held-out conformal split. `alpha` may be a list (nested regions); `interest` restricts the region to a subset of targets, integrating the others out first. When conformal rows have missing targets, `contains(y)` is calibrated on the complete conformal rows (guarantee for complete test rows; `ValueError` if there are fewer than `1/alpha - 1`) and `contains(y, mask=m)` on the masked score of all rows (guarantee for the same missingness mechanism). With `cv` (and no `calibrate`): a `CrossConformalRegion` from the out-of-fold scores of the training rows (kept on the estimator; cached per `interest` and `score`), the score chosen per fold predictive, the same two statements for missing targets with the cross-conformal bound in place of $1 - \alpha$. |
| `members_`, `weights_` | Names of the fitted heads and their stacking weights `[K]`. |
| `validation_log_score_` | Mean validation log density per head in the units of the data (comparable with `predict(X).log_prob(Y).mean()` before recalibration; over observed targets when targets are missing). Heads with a density only. |
| `validation_energy_score_` | Mean validation energy score (lower is better) of each head without a density (`"energy"`), in the units of the data, over the observed targets of each row, from 16 fixed draws per row. |
| `n_parameters_` | Number of network parameters of the fitted model: every ensemble copy of every head, plus the networks of the marginal engine and of the chosen dependence rung on the Sklar route. |
| `n_train_`, `n_val_`, `n_recal_`, `n_conformal_` | Rows in each split: `round(fraction * N)` (at least 8, at most `max_holdout`) for validation, recalibration and conformal rows, the rest (75% with the default fractions 0.1, 0.05 and 0.1 and fewer than about 200,000 rows) for training; `n_conformal_` follows `calibrate`. With `cv`: the training and inner validation rows of one fold model (the smallest over the folds), and `n_recal_ = n_conformal_ = N`. |
| `folds_`, `fold_choices_` | With `cv`: the fold of each training row `[N]`, and the per-fold choices of the Sklar route (`{"marginal_engine": [...], "dependence": [...]}`; `marginal_engine_` and `dependence_` report the most frequent). `None` and `{}` otherwise. |
| `batch_size_` | Minibatch size used by the last `fit` (resolves `batch_size="auto"`). |
| `dependence_scores_` | Validation log score of each dependence rung (Sklar route). |
| `marginal_scores_`, `marginal_engine_` | Sklar route: mean held-out sum of marginal log densities $\sum_j \log \hat f_j(y_j \mid \mathbf x)$ of each marginal engine on the complete validation rows, in the units of the data, and the engine used. With `marginals=None` both built-in engines are fitted once on the training rows (`"quantile"`, `"mixture"`) and the larger score wins; only that engine is refitted out of fold. A user-supplied engine is used as is (its class name, one entry). Empty and `None` for the other methods. |
| `scale_` | Per-target recalibration factors `[d]` actually applied (ones when the recalibration was skipped or rejected), or `None`. |

---

## Predictive distributions

Module `torchregress.joint.predictive`.

| Symbol | Signature | Description |
|:-------|:----------|:------------|
| `JointPredictive` | abstract | Base class. Required: `batch_size`, `event_dim`, `sample(n_samples)`. Declared by every built-in head: `dtype`, `device` (no draw needed). Optional: `log_prob(y, mask=None)`, `to_latent(y)`, `from_latent(z)`, `marginal(dims)`. Derived: `mean(n_samples=512)`, `center(n_samples=512)` (a location without sampling: the image of the latent origin when there is a latent map, else the mean; wrappers combine), `covariance(n_samples=512)` (Monte Carlo moments are computed once per predictive object and sample size and reused, unless the draws carry gradients; a request for the covariance draws once for both), `hpd_level(y, n_samples=512)` (`[n, d] -> [n]` or `[n, S, d] -> [n, S]`, candidates of a row sharing one set of reference draws), `as_input(y)`. Inputs (`y`, `z`, masks) may be tensors of any floating dtype on any device or NumPy arrays: they are cast to the predictive's `dtype` and moved to its `device`, and results stay there. Flags: `has_density`, `has_latent`, `exact_marginals`. |
| `GaussianJoint` | `GaussianJoint(mean, scale_tril)`; `GaussianJoint.from_covariance(mean, covariance)`; `GaussianJoint.from_low_rank(mean, cov_factor, cov_diag)`; `GaussianJoint.from_known_covariance(y_pred, *, loss=None, known_covariance=None, known_variance=None, model_scale=None, model_factor=None)` | Multivariate normal with Cholesky factor `[n, d, d]` or shared `[d, d]`. Closed-form marginals, masked densities, latent map and `condition(dims, values)`; `select(index)` picks rows. `from_known_covariance` is the predictive of `KnownCovarianceGaussianNLL`: the total covariance (known plus given plus the loss's learned model terms), whose masked density equals the loss's NLL with the same mask. |
| `DistributionJoint` | `DistributionJoint(distribution)` | Adapter for a `torch.distributions.Distribution` with batch `[n]` and event `[d]` (`StudentTMixture`, `MultivariateNormal`, zuko flows). Masked densities when the distribution's `log_prob` takes a `mask`; latent map for zuko flows. |
| `FlowJoint` | `FlowJoint(flow_loss, context)` | Predictive of a `NormalizingFlowLoss` head for network outputs `context [n, c]`; samples through the loss in blocks. |
| `ConditionalModelJoint` | `ConditionalModelJoint(model, context, event_dim=None)` | Predictive of any model with `sample(context, n_samples) -> [n, S, d]` and, for a density, `log_prob(context, target) -> [n]`: `NormalizingFlowLoss`, `ContrastiveFlowLoss`, `ensemble.DensityEnsemble`, `models.MarginalContrastiveDensity`, `models.NormalizingFlowMap`. `[n, S, d]` densities repeat the context; no latent map, no masks, marginals by sampling. A model without `log_prob` (the energy head of `JointRegressor`, a user-trained generator) gives a draws-only predictive (`has_density` is `False`) that runs the model on every `sample`. |
| `SampleJoint` | `SampleJoint(samples)` | Draws-only predictive from stored draws `[n, S0, d]`; `sample(S)` subsamples or resamples them. |
| `MaskedMarginalJoint` | `MaskedMarginalJoint(base, dims)` | Exact marginal of a head that integrates out masked targets (mixtures and stacks of them): keeps a density. Returned by `marginal(dims)` for heads with `has_density` and `exact_marginals`. |
| `ProjectedJoint` | `ProjectedJoint(base, dims)` | Draws-only marginal of another predictive: the result of `marginal(dims)` for heads that cannot integrate targets out exactly (flows). |
| `AffineJoint` | `AffineJoint(base, loc, scale)` | Per-target change of units `y = loc + scale * u`, with the Jacobian in the density. Used to map standardised heads back to data units. |
| `MixtureJoint` | `MixtureJoint(members, weights=None)` | Weighted mixture of predictives (`[K]` or per-row `[n, K]` weights); the stacked predictive of `JointRegressor`. |

---

## Adapters for the library's heads

Module `torchregress.joint.adapters`. The table of every multi-target tool and
its one-line call is in the guide,
[I have X, call Y](../guide/joint-regression.md#2-i-have-x-call-y).

| Symbol | Signature | Description |
|:-------|:----------|:------------|
| `as_joint` | `as_joint(obj, inputs=None, /, **kwargs)` | The `JointPredictive` of a head, loss (with the network output it was trained on), ensemble or distribution, or the `SLSRegion` of a calibrated `SLSConformal`. Dispatches on the type of `obj`: Gaussian losses (`GaussianNLLLoss`, `MultivariateGaussianLoss`, `LowRankGaussianLoss` with `rank=` for a concatenated output, `KnownCovarianceGaussianNLL` with the known covariance as keywords), `MixtureDensityLoss`, `InputNoiseMDNLoss`, `EvidentialRegressionLoss`, `NormalizingFlowLoss`, `NoiseAwareDensityLoss`, `JointBinnedDensityLoss`, `MultiQuantileLoss`, a one-feature `MixtureDensityLoss` applied per target (output `[n, d, 3K]`, Gaussian components: `SklarJoint` of `MixtureMarginals`) and `Marginals` (`dependence=`, `x=`), `GaussianCopulaSplineHead`, `FlowMatchingMLP` (`n_samples=`, `steps=`), `inference.MarginalRatioEstimator` with observations `x` (`prior_samples=`, `marginal=`, `prior_log_prob=`, `n_proposals=`, `min_ess=`; returns a `RatioPosterior`), `DensityEnsemble`, heteroscedastic and MDN ensembles, `HeteroscedasticBNN`, `torch.distributions`, any model with `log_prob(context, y)` and `sample(context, n)`, and any generator with `sample(context, n)` only (a draws-only `ConditionalModelJoint`). Raises `TypeError` (naming the supported types) otherwise, and for point-prediction ensembles and SWAG weight samplers. |
| `CopulaSplineJoint` | `CopulaSplineJoint(head, context)` | Predictive of a `models.GaussianCopulaSplineHead`: density and exact masked densities from `intrinsic_log_prob`, batch-first draws, latent map $z = L^{-1}(T^{-1}(y) - \mu)$; `select(index)` picks rows. |
| `SLSRegion` | `SLSRegion(conformal, x, *, groups=None, test_weights=None)` | A `JointRegion` for a calibrated `losses.SLSConformal`: `contains` equals `SLSConformal.contains`, `threshold [n, 1]` is the frontier level $\hat r\, q_\tau(x)$, `log_volume` is exact for one frontier component and an unbiased Monte Carlo estimate for a union, `sample` draws uniformly from the region, `bounding_box` from those draws. No masks; `predictive` carries only the rows. |

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
| `MixtureMarginals` | `MixtureMarginals(weights, means, stds, *, tol=1e-12, max_iter=100)` | A univariate Gaussian mixture per row and target, parameters `[n, d, K]` (weights renormalised over `K`). Closed-form `cdf` (in float64, from $\log F$ below the median and $\log(1 - F)$ above it) and `log_prob`; `icdf` by the safeguarded Newton–bisection solver of `StudentTMixture.marginal_quantile` (at most `max_iter` iterations, steps below `tol` $\cdot(\lvert y\rvert + \bar\sigma)$), solved on $1 - F$ above the median. `normal_scores` / `from_normal_scores` use the tail probability on the side of the median, so float32 upper-tail scores keep their precision. Attributes `log_weights`, `loc`, `scale`. |
| `QuantileMarginals` | `QuantileMarginals(levels, quantiles, tail_knots=None, tail_floor=0.5)` | Quantile grid `[n, d, Q]`: linear CDF between knots, exponential tails fitted on the outer knots; crossing quantiles are sorted and ties separated. `tail_floor` floors each tail scale at that share of a Gaussian tail matched to the interquartile range, so nearly coinciding outer knots cannot give a near-zero scale and absurdly small densities just outside the grid (`0` turns it off). |
| `BinnedMarginals` | `BinnedMarginals(edges, probs, tail_mass=1e-3, min_prob=1e-6, tail_floor=0.5)` | Histogram marginals (bin edges and probabilities), piecewise-constant density with exponential tails; `tail_floor` as in `QuantileMarginals`. |
| `SampleMarginals` | `SampleMarginals(samples, tail_knots=None, tail_floor=0.5, n_knots="auto")` | Marginals read from draws `[n, S, d]`: order statistics as quantiles at plotting positions `r / (S + 1)`. By default `round(2 sqrt(S))` evenly spaced order statistics are kept as knots, because a density from single spacings is too low by 0.42 nats per target on average; `n_knots="all"` keeps every draw. `tail_floor` as in `QuantileMarginals`. |

---

## Dependence

Module `torchregress.joint.dependence`. A `Dependence` models the density of the
normal scores; `SklarJoint` joins it to `Marginals`.

| Symbol | Signature | Description |
|:-------|:----------|:------------|
| `SklarJoint` | `SklarJoint(marginals, scores, eps=1e-6)` | Joint predictive from per-target marginals and a predictive of their normal scores; a valid density for any score density. Flags follow `scores`. |
| `Dependence` | abstract | Base of the ladder: `fit(z, x=None)` on out-of-fold training scores, `predictive(n, x=None)`, `log_prob(z, x=None)`, `marginal_drift(z, x=None, n_samples=256)`; attributes `name`, `has_density`, `conditional`, `fit_time_`. |
| `IndependenceDependence` | `IndependenceDependence(affine=False)` | Rung 0: independent standard-normal scores (product of the marginals). With `affine=True`, a per-target location and scale of the scores, $\mathcal N(\mathbf m, D^2)$ (name `affine_independence`). |
| `SchaakeShuffle` | `SchaakeShuffle(n_neighbors=None)` | Rung 1: empirical copula of the training scores; with `n_neighbors=k`, the `k` nearest training rows in standardised `x`. Draws only. |
| `GaussianCopula` | `GaussianCopula(shrinkage="auto", rank=None, affine=False)` | Rung 2: constant Gaussian copula with Ledoit–Wolf shrinkage (or a fixed intensity) and optional low-rank-plus-diagonal correlation; `correlation_`, `shrinkage_`. With `affine=True`, $\mathcal N(\mathbf m, D R D)$ with a fitted per-target location and scale (name `affine_gaussian_copula`). `JointRegressor(method="sklar")` fits both affine rungs. |
| `ConditionalGaussianCopula` | `ConditionalGaussianCopula(hidden=(64, 64), max_epochs=200, patience=20, lr=1e-3, batch_size=256, val_fraction=0.15, seed=0)` | Rung 3: Gaussian score density with covariance depending on `x` (modified Cholesky), trained by NLL from the independence start. |
| `MixtureDependence` | `MixtureDependence(n_components=3, hidden=(64, 64), max_epochs=300, patience=25, lr=1e-3, batch_size=256, val_fraction=0.15, seed=0)` | Rung 4: Gaussian mixture on the scores with weights, means and covariances depending on `x` (a mixture density network; constant without `x`), trained by NLL with early stopping. For dependence with a few regimes (targets that move together between modes). Its predictive keeps exact marginals and exact densities under missing targets, which the flow rung does not (name `conditional_mixture`). |
| `FlowDependence` | `FlowDependence(n_transforms=3, hidden=(64, 64), max_epochs=300, patience=25, lr=1e-3, bound=5.0, bins=8, batch_size=256, val_fraction=0.15, seed=0)` | Rung 5: conditional (or unconditional) spline flow on the scores, initialised at the identity, trained by NLL with early stopping. |
| `TemplateJoint` | `TemplateJoint(template, n, index=None)` | Draws-only predictive resampling template rows (per-row candidate rows via `index [n, k]`); the predictive of `SchaakeShuffle`. |
| `select_dependence` | `select_dependence(candidates, z_val, x_val=None, n_se=2.0)` | Lowest rung that scores at least as well as independence and within `n_se` paired standard errors of the best on held-out scores. Returns `(chosen, {name: mean log q})`. |

---

## Conformal regions

Module `torchregress.joint.regions`.

| Symbol | Signature | Description |
|:-------|:----------|:------------|
| `JointConformal` | `JointConformal(score, alpha=0.1)` | Split conformal calibration. `calibrate(predictive, y, *, mask=None, weights=None, groups=None)` stores `thresholds_`, `group_ids_`, `n_calibration_`; `region(predictive, *, test_weights=None, groups=None)` returns a `JointRegion`. Thresholds come from `losses.conformal.finite_sample_quantile` (rank `ceil((n+1)(1-alpha))`) and its weighted counterpart, as in `MultivariateScoreConformal`. |
| `CrossConformalRegion` | `CrossConformalRegion(predictive, score, alpha, fold_predictives, fold_scores, *, masked=False)`; built by `JointRegressor.region` with `cv` | Cross-conformal regions (Vovk 2015): $R_\alpha(\mathbf x) = \{\mathbf y : (1 + \sum_i \mathbf 1[s_i \ge t_{k(i)}(\mathbf y)])/(N + 1) > \alpha\}$ from the out-of-fold scores $s_i$ of each fold and the scores $t_k(\mathbf y)$ of the candidate under each fold predictive. Coverage $\ge 1 - 2\alpha - \min\{2(1 - 1/K)/(N/K + 1), (1 - K/N)/(K + 1)\}$ for exchangeable data (Barber et al. 2021, Theorem 4 and Remark 1), not $1 - \alpha$. A `JointRegion`: `threshold` is the rank bound $\lceil (N+1)(1-\alpha) \rceil - 1$, `score` the rank (`score.bases`: the score of each fold), `log_volume` by Monte Carlo (importance sampling from `predictive`, the average of the fold predictives; a uniform box for sample-ball scores), `n_calibration_`. |
| `JointRegion` | built by `JointConformal.region` (`SLSRegion` for SLS regions) | Calibrated regions, one per row and level: `contains(y, mask=None)`, `log_volume(n_samples=1024)` (estimate and standard error), `sample(n_samples, *, level=None, max_rounds=50)`, `bounding_box(n_samples=2048)`; attributes `predictive`, `score`, `alpha`, `threshold [n, L]`, `mask_region` (region used by `contains` when a mask is passed, set by `JointRegressor.region`). A `NaN` score counts as `+inf`. |
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
| `ScaleRecalibrator` | `ScaleRecalibrator(objective="variance", n_samples=1024, max_iter=100, about="mean")` | One scale factor per target about the predictive mean, or about `center()` with `about="center"` (no draws when the predictive has a latent map) (`"variance"` or held-out `"nll"`); `fit(predictive, y, mask=None)`, `scale_`; `transform` returns an `AffineJoint`. |
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

- [Joint Multi-Target Regression](../guide/joint-regression.md): the guide (picture, worked example, mathematics, recipes).
- [Losses API](losses.md): `MultivariateGaussianLoss`, `LowRankGaussianLoss`, `MixtureDensityLoss`, `NormalizingFlowLoss`; `as_joint` turns each into a `JointPredictive`.
- [Conformal API](conformal.md): single-target and coordinate-wise conformal predictors; `MultivariateScoreConformal` and `ConformalizedDensity` are the raw-tensor counterparts of `JointConformal` with `MahalanobisScore` / `DensityScore` (see [I have X, call Y](../guide/joint-regression.md#2-i-have-x-call-y)).
- [Metrics API](metrics.md): `EnergyScore`, `VariogramScore` and other multivariate scores.
