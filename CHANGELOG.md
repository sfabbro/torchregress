# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Cross-fitting for small data: `JointRegressor(cv=K)`. `K` fold models are fitted, each on
  the other `K - 1` folds (inner validation split for early stopping only, one copy per head
  unless `n_members` is given); the stacking weights and the scale recalibration are fitted on
  the out-of-fold predictive of every row; `predict` is the equal-weight average of the fold
  models; `region` returns a `torchregress.joint.CrossConformalRegion` (cross-conformal
  prediction, Vovk 2015) built from the out-of-fold scores of every training row, with coverage
  at least `1 - 2 alpha - min{2(1 - 1/K)/(N/K + 1), (1 - K/N)/(K + 1)}` for exchangeable data
  (Barber, Candès, Ramdas, Tibshirani 2021, Theorem 4 and Remark 1), not `1 - alpha`.
  `calibrate` keeps working (split conformal on the external rows). New attributes `folds_` and
  `fold_choices_`. On 200 synthetic data sets of 120 rows (`d = 2`, `K = 5`, `alpha = 0.2`,
  Gaussian head) the cross-conformal region covered 0.807 on average with a standard deviation
  of 0.046 across data sets, against 0.846 and 0.107 for the default single split (12 conformal
  rows); the proven bound is 0.536. On two small real data sets (193 and 410 training rows,
  four seeds, `JointRegressor()` against `JointRegressor(cv=5)`) the range of realised 90%
  coverage went from 0.75-0.97 to 0.92-0.96 and from 0.87-0.96 to 0.97-0.99 (conservative),
  and the log score did not change beyond the spread across seeds. `cv=None` (the default)
  leaves every fitted number unchanged.
- `JointRegressor.n_parameters_`: number of network parameters of the fitted model (every
  ensemble copy, and the Sklar route's engine and dependence networks).
- Dependence ladder rung 4, `torchregress.joint.MixtureDependence(n_components=3, ...)`: a
  Gaussian mixture on the normal scores whose weights, means and covariances depend on `x`
  (a mixture density network; constant without `x`), for dependence with a few regimes such
  as targets that move together between modes. Its predictive keeps exact marginals and exact
  densities under missing targets, which the flow rung does not. On the ladder test with a
  hidden-sign mixture (4,000 training, 2,000 held-out rows, 3 targets) it reaches 3.434 nats
  against 3.420 for the truth and 3.479 for the flow rung, and fits four times faster;
  on Gaussian dependence it is within 0.01 nats of the well-specified rung.
  `JointRegressor(method="sklar")` fits it between the conditional Gaussian copula and the
  flow (name `conditional_mixture`); on `examples/joint_regression_comparison.py` the Sklar
  route's gap to the oracle went from 0.56 to 0.42 nats.
- Mixture marginals for the Sklar route of `torchregress.joint`: `MixtureMarginals(weights,
  means, stds)` holds a univariate Gaussian mixture per row and target (`[n, d, K]`), with a
  closed-form CDF (float64, from the log CDF below the median and the log survival above it)
  and density, quantiles by the safeguarded Newton-bisection solver of
  `StudentTMixture.marginal_quantile` (now shared), and normal scores taken on the side of the
  median where the tail probability is small (in float32 the default path moved upper-tail
  scores by up to 3e-3). `MixtureNetworkEngine(n_components=5, ...)` is a marginal engine that
  predicts one mixture per target in the parametrisation of a one-feature `MixtureDensityLoss`
  and trains it by the sum of marginal negative log densities (missing targets left out), with
  the training scaffolding of `QuantileNetworkEngine`. `as_joint` of a one-feature
  `MixtureDensityLoss` applied per target (output `[n, d, 3K]`) gives the `SklarJoint` of its
  `MixtureMarginals` (`dependence=`, `x=`). On the problem of the joint comparison example
  (2,400 training rows) the gap to the true marginal log density is +0.09 nats against +0.32
  for the quantile engine.
- `examples/joint_regression_comparison.py` (docs page "Joint Regression Comparison"): every
  `JointRegressor` head and the stacked default against the true conditional density of a
  three-target problem, next to the oracle and to independent targets: NLL gap, energy and
  variogram scores, coverage and volume of the joint 90% region, runtime, and a summary JSON.
- Energy-score (engression) head for `torchregress.joint.JointRegressor`: `method="energy"`
  fits a generator `g(x, eps)` (Gaussian noise of dimension `noise_dim`, default `d`,
  concatenated to the features; Shen and Meinshausen 2024) by the energy score of
  `energy_draws=2` draws per row with the unbiased cross term (Gneiting and Raftery 2007),
  early-stopped on the validation energy score with the same noise every epoch. Its
  predictive is draws only (no log score, no HPD region, not in `"auto"`): regions use the
  sample-ball score, recalibration the variance objective (kept only if it lowers the
  validation energy score), and `validation_energy_score_` reports the validation score
  (`validation_log_score_` has no entry for it). Missing targets enter through the energy
  score of the observed sub-vector. One forward pass per draw block: 1,000 rows x 200 draws
  take 0.09 s at `d = 16` against 15 s for the flow head (4-core laptop CPU), 0.2 s at
  `d = 64`. `ConditionalModelJoint` now also wraps a model with `sample(context, n)` and no
  `log_prob` (a draws-only predictive, `has_density` per instance), and `as_joint` accepts
  such a generator.

- Marginal neural ratio estimation: `torchregress.inference.MarginalRatioEstimator(n_params,
  marginals, observation_dim=None, *, embedding=None, ...)` learns the posterior of chosen
  subsets of the parameters directly (the marginal estimation of TMNRE, Miller et al. 2021; no
  truncation, no rounds), one `build_ratio_classifier` head per marginal on one shared embedding
  of `x`, each trained by its own `NeuralRatioLoss` (CNRE, `gamma=1` by default) with negatives
  drawn inside the batch (`K` = half the batch). Nuisance parameters are integrated out by the
  simulator, so a head's size and step time do not depend on their number (tested for 2, 14 and
  62 nuisances). `fit` uses the early-stopping trainer of `torchregress.joint`; `univariate`,
  `pairs` and `jointly` build marginal lists. `torchregress.inference.RatioPosterior(log_ratio,
  x, prior_samples, *, prior_log_prob=None, ...)` is the `JointPredictive` of
  `p(theta) r(theta, x)` for any learned log ratio: importance resampling of prior draws with
  its effective sample size and a low-ESS warning, a Monte-Carlo-normalised `log_prob`, and
  `normalisation()` (the CNRE check `E_p[r] = 1`). `MarginalRatioEstimator.posterior(...)` and
  `torchregress.joint.as_joint(estimator, x, prior_samples=..., marginal=...)` return it, so
  joint regions, diagnostics and corner plots apply. On a linear-Gaussian simulator with exact
  answers the learned marginal posteriors match the exact means to 0.1 posterior sd and pass
  SBC and HPD expected-coverage tests. New helper `torchregress.losses.in_batch_negative_index`.
- Every multi-target tool of the library reaches `torchregress.joint` in one call:
  `torchregress.joint.as_joint(obj, inputs, **kwargs)` dispatches on the head, loss (with the
  network output it was trained on), ensemble or distribution and returns its
  `JointPredictive` (Gaussian, low-rank and known-covariance Gaussian losses, MDN and
  Student-t mixtures, input-noise MDN, evidential, every `create_flow_model` flow type,
  contrastive flows, noise-aware and joint-binned densities, multi-quantile heads and
  `Marginals` with a dependence rung, the copula spline head, MCD, flow maps, flow matching,
  density / heteroscedastic / MDN ensembles, `HeteroscedasticBNN`), or the `JointRegion` of a
  calibrated `SLSConformal`; it raises a `TypeError` naming the supported types otherwise.
  New adapters: `CopulaSplineJoint` (density and exact masked densities of
  `GaussianCopulaSplineHead`, batch-first draws, latent map) and `SLSRegion` (super-level-set
  regions as a `JointRegion`: the membership of `SLSConformal`, exact volume of a
  volume-preserving frontier and Monte Carlo volume of a union frontier, uniform draws,
  bounding box, corner plots and diagnostics). `GaussianJoint.from_known_covariance` is the
  predictive of `KnownCovarianceGaussianNLL` (total covariance, learned terms included).
  `SLSConformal.contains(x, y, *, groups=None, test_weights=None)` gives the membership of the
  set `predict_interval_from_grid` bounds. The joint guide has a table of every multi-target
  tool and its one-line call; `tests/joint/test_all_heads.py` checks each row (densities equal
  the head's own, masked densities where it has them, conformal coverage on a well-specified
  case, corner plots).
- `torchregress.viz.plot_joint_diagnostics(predictive, y, *, x=None, region=None, ...)`: a
  one-call figure of joint HPD calibration, pre-rank histograms, per-target PITs with
  simultaneous bands, correlation error, and region coverage, size and conditional coverage,
  all computed by `torchregress.joint.diagnostics`, with a plain-language reading aid on every
  panel. `plot_corner_plot` now accepts a `JointPredictive` (`row=`, `n_samples=`) and draws
  2-D highest-density contours that enclose the stated fraction of the draws (fast FFT KDE,
  10^6 draws), with the convention always stated in the legend: `levels=`,
  `level_convention="mass"` (68% / 95%) or `"sigma"` (39.3% / 86.5%). New options: smoothed or
  histogram 1-D marginals with central or HPD intervals and `median^{+}_{-}` titles;
  conformal-region overlays (`region=`); colour-blind-safe comparisons (`others=`); `dims=`
  and automatic selection of the most dependent targets above `max_dims`; quantile-based
  limits for heavy tails; `return_axes=`. `tools/render_joint_figures.py` regenerates the
  documentation images.
- `torchregress.joint` (lazy submodule, 56 exports): joint predictive distributions over
  `d` targets. `JointPredictive` gives every head one interface (`sample` -> `[n, S, d]`,
  `log_prob` on `[n, d]` or `[n, S, d]`, `marginal(dims)`, latent maps, masks) with declared
  capabilities (`has_density`, `has_latent`, `exact_marginals`); adapters `GaussianJoint`,
  `DistributionJoint`, `FlowJoint`, `SampleJoint`, `ProjectedJoint`, `AffineJoint`,
  `MixtureJoint`, and `ConditionalModelJoint` for any density head with `log_prob(context, y)`
  and `sample(context, n)` (`NormalizingFlowLoss`, `DensityEnsemble`,
  `MarginalContrastiveDensity`, `NormalizingFlowMap`). `SklarJoint` joins any `Marginals` (`GaussianMarginals`,
  `QuantileMarginals`, `BinnedMarginals`, `SampleMarginals`) with a density on normal scores,
  a valid joint density for any score model; the dependence ladder `IndependenceDependence`,
  `SchaakeShuffle`, `GaussianCopula` (Ledoit-Wolf shrinkage, optional low rank),
  `ConditionalGaussianCopula` (modified Cholesky in `x`), `FlowDependence`
  (identity-initialised spline flow) is gated by `select_dependence` (lowest rung within 2
  paired standard errors of the best, never below independence). `JointConformal` /
  `JointRegion`: split conformal regions with `P(Y in R(X)) >= 1 - alpha` under
  exchangeability (marginal over `x`, not per `x`), for density, HPD, latent-ball,
  Mahalanobis (with missing targets via the chi-square CDF), max-residual (box with a joint
  guarantee) and sample-ball scores, weighted and Mondrian variants, nested levels,
  closed-form or importance-sampled log-volumes, rejection sampling and bounding boxes.
  Recalibration: `ScaleRecalibrator`, `LatentRecalibrator` (Dheur and Ben Taieb 2025),
  `MarginalRecalibrator` (per-target conformal calibrators, finite-sample per-target
  calibration). Diagnostics: `prerank_histograms` (six pre-ranks, Holm-adjusted),
  `nees_consistency`, `hpd_calibration`, `coverage_by_mask`, `worst_slab_coverage`,
  `conditional_coverage_gap` (ERT), `dependence_error`, `latent_calibration_error`, with
  their result dataclasses (`PreRankHistograms`, `NEESConsistency`, `HPDCalibration`,
  `CoverageByMask`, `WorstSlabCoverage`, `CoverageGap`, `DependenceError`) exported.
  The package reuses the library's primitives: conformal thresholds from
  `losses.conformal.finite_sample_quantile` and its weighted helper (the same thresholds as
  `MultivariateScoreConformal` for the same scores), HPD levels from
  `metrics.sample_hpd_level`, masked Gaussian densities from
  `known_covariance_gaussian_nll`, `utils.distributions.normal_cdf`,
  `utils.quantile.pinball_residual`, and one mixture-weight EM shared with
  `DensityEnsemble.fit_weights`; `inverse_softplus` moved to `utils.distributions`.
  `JointRegressor().fit(X, Y)` (`NaN` = missing target) fits Gaussian, mixture, flow and Sklar
  heads on standardised targets, stacks them by held-out log score, recalibrates per-target
  scales and conformalises on a separate split; `region(X, alpha, interest=...)` calibrates a
  region for targets of interest; with missing targets in the conformal rows,
  `region(X).contains(y)` is calibrated on the complete conformal rows (guarantee for complete
  test rows) and `contains(y, mask=m)` on the masked score (same missingness mechanism).
  `fit` leaves the caller's torch RNG state untouched. Inputs to `log_prob`, `to_latent`,
  `from_latent`, `hpd_level` (which also takes `[n, S, d]` candidates), `JointConformal.calibrate`
  and `JointRegion.contains` may be NumPy arrays or tensors of any floating dtype and device; every
  predictive declares `dtype` and `device`. `score="auto"` uses `LatentScore` only for Gaussian
  predictives and `HPDScore` for other densities. `JointRegressor.calibrate(X, Y)` replaces the
  conformal rows; `n_train_`, `n_val_`, `n_recal_`, `n_conformal_` report the split and
  `validation_log_score_` is in data units. `viz.plot_joint_diagnostics(seed=...)` is reproducible
  without touching the global RNG, and corner plots of one or two targets keep the legend outside
  the grid. Tests (289 in `tests/joint/`) check: coverage of every score
  within 4 standard errors of 0.9 (4,000 calibration and 8,000 test rows, well-specified and
  misspecified predictives) and of `JointRegressor` regions at two levels, with missing targets
  and for targets of interest; the ladder choosing the simulated rung, with held-out NLL within
  0.01 (constant copula) and 0.03 nats (x-dependent copula) of the truth and the flow rung
  0.22 nats better than the Gaussian copula on non-Gaussian dependence (`d = 3`, 4,000
  training rows); latent recalibration taking L-ECE from above 0.15 to below 0.03 on
  mis-scaled Gaussians. New guide page "Joint Multi-Target Regression" and API page "Joint";
  the multi-target guide no longer claims a joint guarantee for `MultiTargetConformal`, and its
  worked example now runs.
- `torchregress.losses.KnownCovarianceGaussianNLL` and `known_covariance_gaussian_nll`:
  Gaussian NLL of `[B, D]` residuals under a per-row **known** covariance (dense
  `[B, D, D]` / shared `[D, D]`, or diagonal `known_variance`) plus a **model**
  covariance `diag(s^2) + U U^T`, learned by the loss (`scatter="fractional"`:
  `(a * y_pred)^2 + b^2`; `scatter="constant"`: `b^2`; `rank=k` with a fractional
  `U = y_pred * V` or constant `U = V` factor; `a`, `b` on the log scale) and/or passed
  per row (`model_scale`, `model_factor`; they add). Exact batched Cholesky of the
  Jacobi-equilibrated total covariance, masked dimensions marginalised exactly
  (principal submatrix, `NaN` at masked entries ignored), fail-closed validation in
  one host sync, opt-in relative jitter for failing rows only (reported per row).
  The functional returns `KnownCovarianceGaussianResult(nll, chi2, log_det, dof,
  whitened, jitter)` for chi2 / whitened-residual calibration diagnostics;
  `KnownCovarianceGaussianNLL.evaluate` gives the same with the learned terms. New
  guide page "Known + Model Covariance" and `tools/benchmark_known_covariance.py`.
- `torchregress.inference.likelihood`: estimator-agnostic maximum-likelihood estimates and
  profile-likelihood intervals for continuous multivariate parameters in a box, from any score
  `score(theta [M, P], x [M, ...]) -> [M]` equal to `log p(x | theta)` up to a function of `x`.
  `maximize_score` (batched over rows: scrambled Sobol candidates, topographical start selection
  over the `2P` nearest candidates, projected saddle-free Newton ascent with exact autograd
  Hessians and Armijo backtracking, optional `init` start, `at_bound` flags, deterministic given a
  generator); `profile_interval` (continuation scan of the profile to each bound with warm-started
  inner Newton maximization, outermost crossing refined by Illinois regula falsi on the signed root
  `sqrt(2 (s_max - p))`, scalar or per-row thresholds, bound flags);
  `likelihood_ratio_statistic` (per-row hypotheses or banks, clamped at 0); `wilks_threshold`
  (`chi2` quantile / 2); adapters `ratio_classifier_score` (`NeuralRatioLoss` classifiers) and
  `flow_score` (`ContrastiveFlowLoss` / conditional flows). New guide page
  "Likelihood-Ratio Inference" and `tools/benchmark_likelihood_inference.py`.
- `torchregress.inference` Neyman confidence sets with parameter-dependent critical values
  (LF2I; Dalmasso et al., arXiv:2107.03920): `{theta : lambda(theta; x) <= C_alpha(theta)}`
  for any statistic oriented so that large values reject (likelihood ratio, WALDO, ...), with
  coverage targeting every `theta` independently of the calibration population.
  `CriticalValueRegressor(alpha, method="mlp" | "knn")` learns `C_alpha(theta)` from labelled
  calibration pairs `theta [N, P]`, `statistic [N]` for one or several levels: a
  quantile-regression MLP (pinball loss from `utils.quantile`, standardised inputs, head =
  lowest quantile + softplus increments so levels never cross, initialised at the marginal
  quantiles, held-out early stopping with reduce-on-plateau, seeded and deterministic; a
  non-negative statistic is fitted on the log scale, exact by quantile equivariance) or a
  chunked k-nearest-neighbour finite-sample quantile. The k-NN estimator takes
  `statistic_fn` (with `fit(..., x=...)`): the neighbours' observations are re-scored at the
  query parameter, which keeps coverage for count data, where the exact critical value is a
  saw-tooth that a smooth or pooled estimate cannot follow (Poisson with background, 20 000
  pairs: minimum coverage 0.67 / 0.89 at 68.27% / 90% against 0.55 / 0.81 for the MLP); and
  `confidence`, the binomial upper confidence bound for the quantile in place of the
  finite-sample order statistic. `NeymanConfidenceSet` gives
  `contains(statistic, theta)` and candidate-bank masks `region(statistic_fn, theta_bank, x)`;
  `binned_conditional_coverage` (Wilson intervals, Bonferroni over bins) and
  `smooth_conditional_coverage` (ridge-penalised logistic regression with a Scheffé
  simultaneous band) estimate `P(covered | theta)` on held-out data and flag the regions
  significantly below nominal. A single-level `CriticalValueRegressor` is accepted as
  `critical_value_regressor` by `WaldoConformal`, `WaldoPosteriorCalibrator` and
  `WaldoDensityConformal` (their behaviour is otherwise unchanged).
- `torchregress.losses.StudentTMixture`: batched mixture of multivariate Student-t
  distributions (a `torch.distributions.Distribution`; `df = math.inf` components are
  Gaussian), with covariance-Cholesky (`scale_tril`), precision-Cholesky (`precision_tril`)
  or diagonal (`scale_diag`) scale matrices and `df` per row and/or component (fixed or a
  tensor with gradients). The components are folded into the batch of every linear-algebra
  call (no Python loop over components or rows): `log_prob` / `component_log_prob` of full
  or partially observed targets (exact marginals, rows with nothing observed give 0);
  `component_observation_log_prob` for noisy linear observations `z = A_k x + b_k + e` with
  per-component designs and validity masks (closed form `t(A mu, A Sigma A^T + diag(n))`,
  noise treated inside the t, exact for Gaussians; `D x D` Cholesky for `D <= P`, otherwise
  an information form whose quadratic is a sum of squares); `component_censored_log_prob`
  for upper/lower limits with the exact Student-t log-CDF; `mixture_log_prob` to mix summed
  per-component terms; `condition` on a noisy coordinate (`df + 1`, `(df + q)/(df + 1)`
  scale, reweighted components; the conditional Cholesky factor is a closed-form rank-one
  downdate built from tail sums, so a nearly exact measurement keeps full relative
  accuracy, applied in O(P²) per component without forming the P x P downdate factor);
  `marginal_cdf` and `marginal_quantile` (safeguarded Newton with a guaranteed
  Student-t tail-bound bracket and an active set, float64, chunked; no probability
  clipping; levels above 1/2 are solved on the survival function, so upper-tail quantiles
  are as accurate relative to `1 - q` as lower ones are to `q`); `sample`, `mean`, `variance`, `covariance_matrix`. The `df`-dependent
  normaliser uses a Stirling-difference series above `df = 100`, so the Gaussian limit is
  approached as `O(1/df)` with no numerical floor. `tools/benchmark_student_t_mixture.py`
  times it against a per-component loop.
- `MixtureDensityLoss(df=..., learn_df=...)`: Student-t components (fixed `df`, or one
  shared `df = df_init * exp(raw_df)` learned on the log scale: `raw_df` starts at 0, so the
  initial `df` is exact in any dtype, and an optimiser step moves `df` by a fixed fraction,
  reaching the Gaussian regime geometrically instead of stalling near `df ~ 10`), and `MixtureDensityLoss.distribution(y_pred)`
  returning the head's `StudentTMixture`. The Gaussian MDN is the `df = math.inf` case of
  the same code path.
- `torchregress.losses.structural_similarity_1d` and `StructuralSimilarity1dLoss` (`1 - SSIM`):
  structural similarity along the last dimension of `[..., N]` sequences (spectra, light
  curves, profiles), batched over leading dimensions. Masks and per-sample weights (e.g.
  inverse variances) enter every local mean, variance and covariance by normalised
  convolution; a zero weight is a masked sample; windows with less than `min_coverage`
  of their kernel mass valid are excluded; masked samples may hold `NaN`. Gaussian or uniform window, `padding="valid"`
  (default), `"same"` (sequence masked beyond its ends) or `"reflect"` (reproduces
  `torchmetrics` 2-D SSIM on `[B, 1, N, 1]` with a `(k, 1)` kernel to round-off); per-row
  `data_range` (default: the target row's valid range, independent of the batch);
  `return_map`. Documented as a similarity index without a noise model, dependent on
  `data_range`, and insensitive to local gain/offset.
- `torchregress.losses.marginalize_linear_nuisance` and `MarginalizedNuisanceGaussianNLLLoss`:
  exact Gaussian NLL of `[..., N]` rows with known per-sample precision and a linear
  nuisance `U a` (continuum, baseline, calibration polynomial) integrated out (Rybicki &
  Press 1992), via Woodbury and the matrix determinant lemma: one Jacobi-scaled `m x m`
  Cholesky per row, never an `N x N` matrix. Shared `[N, m]` or per-row bases, full prior
  covariance or diagonal prior precision with flat (improper) directions giving the
  restricted likelihood, exact masking, and the posterior nuisance coefficients, their
  covariance, the fitted nuisance and the conditional residual (`LinearNuisanceMarginal`).
  Rows whose flat directions are not constrained by the unmasked data raise instead of
  being regularised. Runs in at least float32 with autocast disabled, so AMP does not
  demote the Gram products and solves. Forward + backward at N=10,000, m=50, float64, 2 threads on a shared
  host: about 1,000 rows/s with a shared basis (`tools/benchmark_linear_nuisance.py`).
- Student-t family with known per-observation precision:
  `torchregress.losses.student_t_nll(residual, precision, nu, *, mask=None, dim=-1)` is
  the exact `t_nu(0, 1/precision)` NLL with all constants, masked and summed per row
  (`nu=math.inf` is the Gaussian; masked entries may be non-finite);
  `effective_precision(precision, model, eta)` is the variance model
  `1/precision + eta^2 model^2` (a tensor `eta` is validated with one host sync per call); `student_t_efficiency(nu, noise="gaussian")` is
  `E[psi']^2/E[psi^2]` of the t location score under Gaussian noise (closed form for nu < 3,
  200-node Gauss-Hermite above; round-off accurate for every nu > 0) or t
  noise (the Fisher factor `(nu+1)/(nu+3)`). `robust_weights` and
  `iteratively_reweighted_fit` accept `weight="student_t"` (`(nu+1)/(nu+u^2)`, the IRLS/MM
  weight psi(u)/u of the exact t NLL; its weighted normal matrix estimates the Gaussian,
  not the t Fisher, information). `StudentTLoss` accepts `nu=math.inf`.
- `torchregress.algorithms.batched_iteratively_reweighted_fit(problem, initial_theta, target,
  precision, mask=None, config=None, *, initial_weights=None)`: IRLS for many independent
  rows (`[B, N]` observations with known precisions and masks; one spectrum or source per
  row) with no per-row Python loop and one host sync per controller iteration (with a damped
  solve, plus one for the decrease test and one per damping attempt; the caller's solve is
  extra). Student-t weights by default (any `robust_weights` family), variance
  `1/precision + eta^2 m^2`, exact per-row NLL plus an optional penalty, per-row
  `IRLSRowStatus` (converged / max_iterations / stalled / failed / empty, mapped onto
  `IRLSTermination` in its docstring) with early exit of finished rows. Convergence is
  certified on the full step in standard-error units (`BatchedIRLSFitConfig.step_tol`); an
  optional Levenberg-Marquardt `damped_solve` is accepted on objective decrease and never
  counts as converged. Failed and empty rows are `NaN`, not zeros. The caller owns the
  weighted solve (`BatchedIRLSProblem`, `BatchedIRLSStep`, `BatchedIRLSFitConfig`,
  `BatchedIRLSFitResult`); a solve may report `ridge_used`, accumulated per row into
  `BatchedIRLSFitResult.ridge_used`. At N=4096, n=8 (float64, CPU, 2 threads, on a shared
  4-CPU host) it fits 260-730 rows/s for B=100..10,000, 4-12x a per-row loop over
  `iteratively_reweighted_fit` in the same run (`tools/benchmark_batched_irls.py`;
  `--max-outlier-fraction` draws per-row outlier fractions so rows converge at different
  iterations and the compaction of finished rows shows in its row-solve count). The
  controller holds about three `[B, N]` working buffers of the running rows (plus two
  transients) on top of the caller's data.
- `torchregress.algorithms.solve_normal_equations(normal, rhs, *, nonnegative=None, start=None,
  ridge=1e-8, kkt_tol=1e-10, rank_tol=0.0)`: many small least-squares problems from their
  normal equations in one call. Jacobi-scaled batched Cholesky (no host sync on an
  accelerator) with a flagged ridge fallback (by default only where the factorisation fails, so identified rows are never
  biased; opt-in `rank_tol > 0` also takes it where the smallest-to-largest eigenvalue ratio
  of the scaled final system is at most `rank_tol`, testing only the final free set of the
  non-negative solve and re-solving flagged rows as the strictly convex ridged problem;
  solution and standard errors are then regularised) and standard
  errors; with a `nonnegative` mask, batched Lawson-Hanson NNLS on
  the scaled normal equations with a warm-startable free set (each active-set step works on
  the rows still pending, about one host sync per step; the standard errors reuse each
  row's last free-block factor). Empty columns are pinned at 0 with
  infinite standard error; singular or non-finite rows return `NaN` and are flagged.
- `torchregress.models.MarginalContrastiveDensity`: Marginal Contrastive Discrimination
  (Meziani, Ndiaye & Riu 2022), `p(y|x) = p(y) r(x, y)` from an unconditional flow and a
  CNRE ratio classifier. Monte-Carlo-normalized `log_prob`, importance-resampled `sample`
  and an `effective_sample_size` diagnostic; works with `model_hpd_level` and
  `DensityEnsemble`.
- `NeuralRatioLoss(criterion="cnre", gamma=...)`: bias-free contrastive neural ratio
  estimation (Miller, Weniger & Forré 2022). An extra "independent" class fixes the
  normalisation, so the logit estimates `log p(x|theta) - log p(x)` exactly at any number
  of negatives; `neural_ratio_loss` gains `gamma`.
- `torchregress.ensemble.DensityEnsemble`: equal-weight mixture of independently trained
  conditional flows (or any head with `log_prob`/`sample`), with
  `entropy_decomposition()` returning total / aleatoric / epistemic (member mutual
  information) entropies.
- `torchregress.inference.WifiRatioEnsemble` and `MixtureFractionEstimate`: wifi
  ensembles (Benevedes & Thaler, PRD 112, 056024, 2025). Frequentist standard errors on
  learned log density ratios (convex weight fit over frozen members plus sandwich
  covariance) and on mixture fractions, with the ratio uncertainty propagated by
  Gong–Samaniego.
- `DensityEnsemble.fit_weights` / `log_prob_std`: maximum-likelihood mixture weights on
  held-out data with a sandwich covariance, giving standard errors on `log p(y|x)`.
- `torchregress.metrics.sample_hpd_level`, `sample_hpd_coverage`, `model_hpd_level`:
  multivariate highest-density-region calibration from predictive draws, testing the
  joint multi-target shape that per-dimension PIT cannot.
- `create_flow_model(flow_type="tarflow")` (`torchregress.losses.tarflow.TransformerFlow`):
  conditional transformer autoregressive flow after TarFlow (Zhai et al., ICML 2025) with
  STARFlow-style deep-shallow blocks (`n_layers=(4, 1, 1)`). A causal Transformer over a
  context token and the preceding targets parameterizes rational-quadratic splines
  (default) or affine maps. On a synthetic skewed copula its KL gap is about half of NSF's
  at 8 targets and a quarter at 16; on multimodal targets and for 2-4 targets NSF/MAF
  remain better and sample much faster (see the flow docs).
- Affine rungs of the dependence ladder: `IndependenceDependence(affine=True)` and
  `GaussianCopula(affine=True)` fit a per-target location and scale of the normal scores,
  `q(z) = N(m, D R D)` (names `affine_independence`, `affine_gaussian_copula`). The
  conditional and flow rungs can move and rescale the score marginals, so with globally
  biased or mis-scaled marginals they gained log score without modelling any dependence; a
  cheap affine rung now absorbs that miscalibration and the richer rungs are kept by
  `select_dependence` only for what they add beyond it. `JointRegressor(method="sklar")` fits
  independence, affine independence, Gaussian copula, affine Gaussian copula, conditional
  Gaussian copula and flow, and passes its batch size to the conditional and flow rungs.
- `JointPredictive.center(n_samples=512)`: a location summary that avoids sampling (the image
  of the latent origin when the predictive has a latent map, otherwise the mean, closed form
  for mixtures; wrappers combine the centers of what they wrap).
  `ScaleRecalibrator(..., about="mean" | "center")` takes its per-target scaling about either.
### Changed

- `JointRegressor(method="auto")` now trains three independently initialised copies of each
  network head (Gaussian, mixture, flow) and averages them with equal weights before
  stacking across head types: a deep ensemble (`n_members=None`: 3 for `"auto"`, 1 for a
  single named head; set `n_members=1` for the old behaviour). The first full-size harness
  run had the library's own 3-flow `DensityEnsemble` ahead of the stacked default on every
  dataset. Measured on three harness designs with known densities (4,000 rows, one seed):
  test log-score gap to the oracle 0.26, 0.37, 0.41 nats with one copy and 0.21, 0.30, 0.33
  with three (flow head alone 0.38, 0.45, 0.58 against 0.25, 0.34, 0.42; mixture head 0.37,
  0.49, 0.78 against 0.31, 0.41, 0.52), for two to three times the fit time (13 to 19 s
  against 27 to 41 s on a laptop CPU).
- `JointRegressor` holds out less by default: `val_fraction=0.1`, `recal_fraction=0.05`,
  `conformal_fraction=0.1` (were 0.15, 0.10, 0.15), so 75% of the rows train the heads instead
  of 60%. Measured on three harness designs with known densities (4,000 rows, two seeds), the
  test log-score gap to the oracle fell by 0.02 to 0.12 nats for the Gaussian and mixture
  heads on every design, and region coverage was unchanged within its sampling noise. The
  first full-size harness run had shown a hand-written Gaussian MLP trained on 85% of the rows
  ahead of the library's Gaussian head (0.33 against 0.41 nats).
- `JointRegressor(method="sklar")` and the Sklar member of `method="auto"` with
  `marginals=None` now fit both built-in marginal engines (`QuantileNetworkEngine`,
  `MixtureNetworkEngine` with `n_components` components) once on the training rows and keep
  the one with the larger mean held-out sum of marginal log densities on the complete
  validation rows (`marginal_scores_`, `marginal_engine_`); only that engine is cross-fitted
  for the normal scores. Fitted numbers move: on the joint comparison example the mixture
  engine is chosen, the Sklar test NLL gap to the oracle goes from 0.569 to 0.560 nats and the
  stacked default's from 0.146 to 0.143. A user-supplied `marginals=` engine is used as before.
- The diagnostics of `torchregress.joint` (`prerank_histograms`, `nees_consistency`,
  `hpd_calibration`, `coverage_by_mask`, `worst_slab_coverage`, `conditional_coverage_gap`,
  `dependence_error`) accept NumPy arrays and lists as well as tensors, like the rest of the
  joint API.
- `JointRegressor` recalibrates its per-target scale about `center()` instead of the
  Monte Carlo mean, so `predict(X).log_prob(Y)` on a flow or stacked predictive no longer
  draws samples. Measured on one laptop CPU run (4 threads, d = 4, 2,000 test rows, flow
  head): scoring 5.9 s to 0.03 s, `region(...).contains(...)` 17.7 s to 5.9 s, fit of a
  5-epoch flow 5.7 s to 0.9 s.
- `JointRegressor(batch_size="auto", max_holdout=20_000)` (new defaults; attribute
  `batch_size_`): each held-out split is capped at `max_holdout` rows, so with a million
  rows 94% train the heads instead of 60%, and the batch size grows from 256 up to 4096 so
  that an epoch stays under about 400 steps. Behaviour is unchanged below about 100,000
  training rows.
- `plot_corner_plot` draws the 68% and 95% mass contours by default (it drew the 39.3% and
  86.5% "sigma" levels before; pass `level_convention="sigma"` for those). Default colours
  and target names changed; every existing argument keeps its meaning.

- `MixtureDensityLoss` evaluates its log density, moments and samples through
  `StudentTMixture`. The diagonal and full log densities no longer add `eps` to the
  (already `min_std`-floored) scales, so Gaussian NLLs change by `O(eps / min_std)`; the
  private `_log_prob_diagonal`, `_log_prob_full` and `_component_marginal_std` are removed.
  `NoiseAwareDensityLoss` uses its analytic diagonal-MDN shortcuts only for Gaussian
  components.
- `MixtureDensityLoss.forward(mask=...)` with a per-feature mask uses the exact marginal
  NLL of each row's observed features (rows with none are dropped) instead of dropping
  every partially labelled row; masked target entries may be NaN. Per-feature `weights`
  are averaged over each row's observed features.
- `MixtureDensityLoss.predict_interval` returns exact marginal quantiles
  (`StudentTMixture.marginal_quantile`) instead of Monte Carlo estimates from 10,000 draws;
  `n_samples` is accepted and ignored.
- `solve_normal_equations` does less work per call, with bit-identical results (6,144
  randomized cases, both dtypes, all flags): the non-negative solve's standard errors reuse
  each row's last free-block Cholesky factor instead of factorising every final free block
  again; the ridged factorisation runs only for the rows whose plain one failed (on the CPU;
  on an accelerator both are still computed, without a host sync); active-set steps on every
  row skip the gathers, and the empty-column and non-finite-row handling are skipped on the
  CPU when no row needs them. Per call, 2 threads, `n` 6-20 with 4-16 constrained: 0.55-0.70x
  the previous time at `B = 1`, 0.6-0.95x at `B = 8-512`; the unconstrained solve 0.5-0.6x.
  uspm's S1 refine (one object per call) takes 0.88x the time it took with the previous version.
  `tools/benchmark_normal_equations.py` measures it (`--uspm-reference` also times uspm's
  former `nnls_normal`); `scripts/check_uspm_irls_parity.py` now compares against uspm's
  pre-adoption implementation frozen in `scripts/uspm_b461f8f_reference.py`.
- `NormalizingFlowLoss.sample` draws large requests in blocks of at most 8192 draws
  (contexts x samples) and frees each block's inverse-transform garbage before the next.
  Results follow the same distribution; requests up to 8192 draws are unchanged
  bit for bit. 51,200 NSF draws at 16 targets went from 84 s to 33 s.

### Fixed

- `JointRegressor` clamps every feature to the range of its training rows before it enters
  the networks (`clip_features=True`, the default; `False` restores the old behaviour). On the
  river-flow set `mt_rf1` one test row has a feature 178 standard deviations out, and its log
  density was -1.4e11 under the Gaussian head, which made the mean log score meaningless for
  every head. With the clamp (seed 0, 1,000 test rows): Gaussian 11.1 nats, mixture 9.7 (was
  30.4), flow 8.9 (13.1), Sklar 7.0 (54.0), against 10.1 for the harness's 3-flow ensemble;
  regions cover 0.91 to 0.93. Rows inside the training range are untouched, and the clamp is
  applied to calibration and test rows alike.
- `GaussianJoint.from_covariance`, `from_low_rank`, `marginal`, `condition` and
  `MahalanobisScore` factorise through the jittered Cholesky of the package, so a covariance
  that is positive definite only up to float32 rounding (condition numbers of 1e7 and beyond
  on `mt_rf1`) no longer raises `linalg.cholesky: not positive-definite`. `JointRegressor()`
  and its Gaussian head failed that way in the full-size harness run.
- `SampleMarginals` no longer uses every draw as a quantile knot: a density built from single
  spacings is too low by `1 - gamma` = 0.42 nats per target on average, whatever the number
  of draws (measured -0.40 to -0.44 for 50, 200 and 1,000 Gaussian draws). The default keeps
  `round(2 sqrt(S))` evenly spaced order statistics (`n_knots="auto"`; measured -0.16, -0.08
  and -0.04); `n_knots="all"` restores the old grid, whose CDF and draws were never affected.
  Found in the harness, where the sample-marginals row sat 1.3 nats above the quantile grid it
  was drawn from on three targets.
- `QuantileMarginals`, `SampleMarginals` and `BinnedMarginals`: the exponential tail scales
  are floored at `tail_floor` (default 0.5) times the scale of a Gaussian tail matched to the
  interquartile range. Outer knots that almost coincide (noisy extreme quantiles, two close
  order statistics, a narrow outermost bin) gave a tail scale near zero, hence log densities
  of -100 and below just outside the grid. Found in the first full-size harness run, where
  the quantile-grid ladder rows had a log-score gap of 1.3 to 1.7 nats with a standard error
  of 0.38 on one design; with the floor the mean marginal log density on that design goes
  from -1.57 +/- 0.09 to -1.40 +/- 0.04 and the worst row from -129 to -16. `tail_floor=0`
  restores the old tails.
- `metrics.energy_score`: the distance to the observation is `torch.linalg.vector_norm`, whose
  gradient is zero where a draw equals the observation; the square root of a sum of squares
  gave a NaN gradient there, which a float32 generator trained by the score hits within a few
  hundred steps. Values are unchanged.
- `maximize_score` / `profile_interval`: the projected-Newton step diagonalised every active
  row-start's Hessian in one `torch.linalg.eigh` call; on CUDA the batched solver's workspace
  grows with the batch (6M row-starts asked for 253 GiB). Batches are now split into chunks of
  8192 matrices (results unchanged).
- `NormalizingFlowLoss` spline-range warning identified spline flows by searching the class
  name for "nsf", which also matched "TransformerFlow"; it now checks for zuko's NSF or a
  `tail_bound` attribute.
- `test_large_df_tends_to_gaussian` failed for about 4% of global RNG states: at
  `df = 1e12` the test's own subtraction of two log densities has an absolute error of
  `df * eps * |log p|` (about 2e-3); its tolerance now includes that bound.
- `test_recommended_tail_bound_is_data_range_with_margin_clipped` used an unseeded
  `torch.randn(500)` and failed for about 1.6% of global RNG states; the input is now seeded.
- Docs: corrected author lists for Miller, Weniger & Forré (CNRE) and Elsharkawy & Kahn;
  added a survey of contrastive/ratio flow methods to `docs/losses/flow_research.md`.

## [0.3.0] - 2026-11-04

Bug fixes, performance of existing code, docs, and harness parity on top of the
0.3.0b1 API cut. Three method families landed after the beta and ship in 0.3.0
(see *Added*).

### Added

- Multi-source regression: `MarginalizedMultiSourceGaussianNLLLoss` and
  `PackedObservations` (joint NLL over all noisy measurements of one shared
  Gaussian latent, linearized observation maps, full measurement covariance,
  masked missing dimensions), `RobustMultiSourceGaussianNLLLoss`, and
  `GaussianLatentForwardObservationNLLLoss` (quadrature through a nonlinear
  forward map). Calibration heads `CommonScaleSplineHead`,
  `RelativeSurveySplineHead`, `ConditionBiasHead` and
  `bspline_basis_with_derivative`. Docs: `docs/losses/multisource.md`.
- `torchregress.models.GaussianCopulaSplineHead`: a conditional joint spline
  flow (full-covariance Gaussian followed by per-coordinate monotone
  rational-quadratic splines) with exact marginalization of missing
  coordinates. Requires the `flows` extra.
- `PoissonGaussianLoss` (registry key `"poisson_gaussian"`): NLL of Poisson shot
  noise convolved with Gaussian read noise, `y = g K + b + eps`, with fixed or
  learnable gain, offset and read-noise variance. Finite support with
  `tail_sigma` and renormalisation, an observation-conditioned window for
  outliers, chunked evaluation and analytic gradients (ported from torchsky).
  Docs: `docs/losses/poisson_gaussian.md`.

### Changed

- Performance: `GaussianCopulaSplineHead.intrinsic_log_prob` and `observed_log_prob`
  no longer loop in Python over distinct missingness patterns. The exact observed-subvector
  Gaussian marginal is computed for the whole batch with one batched Cholesky of
  `diag(m) L Lᵀ diag(m) + diag(1 - m)`; fully observed batches use the predicted Cholesky
  factor `L` directly (no `L Lᵀ` re-factorisation). Input checks are read back in one
  device-to-host copy. Results and gradients match the per-pattern implementation to
  float64 round-off. CPU, batch 2048, forward+backward: D=12 with random missingness
  (≈1090 patterns) 1.43 → 0.07 s/step; D=6 0.09 → 0.04 s/step.

- Performance: `MarginalizedMultiSourceGaussianNLLLoss` and
  `RobustMultiSourceGaussianNLLLoss` now score all objects as one padded batch.
  They use batched Cholesky and triangular solves, so there is no per-object
  Python loop and no per-object host synchronization. Unobserved slots are
  decoupled unit-variance coordinates with zero residual and zero Jacobian, so
  they contribute nothing. The robust mixture scores each inlier/outlier
  assignment for the whole batch and takes a batched `logsumexp`. Values and
  gradients match the previous object-by-object code (float64: values within
  2e-15 relative, gradients within 1e-11 relative). On CPU with float64,
  50k objects, 12 slots and D=3, forward plus backward runs at about 1.6e5
  objects/s, up from 1.0e3 (`auto`). The robust variant runs at 1.7e4
  objects/s, up from 70. Inputs are validated once per batch. If a batch has
  several different errors, the first failing check type is reported. Checks
  are no longer reported per object. `last_diagnostics` is now a read-only
  mapping that is copied to Python lists only when it is read. Timing script:
  `python -m tools.benchmark_multisource`.

- Performance: `SkewTLoss` / `skew_t_nll` no longer synchronise with the host
  on accelerators. The Student-t CDF in the skew-t density evaluated its
  incomplete-beta continued fraction with an early-exit Lentz loop that read a
  convergence flag back to the host every term (on CUDA ~1,500 kernel launches
  and ~17 device-to-host reads per training step, and no CUDA-graph capture).
  Off the CPU the same continued fraction is now evaluated at a fixed depth
  (63 Lentz iterations; the early-exit loop never needs more than ~70 for
  dof ≤ 1e10) as a binary-tree product of its 2×2 recurrence matrices, with no
  value-dependent control flow, so training steps can be captured and
  replayed as CUDA graphs. The CPU keeps the early-exit loop (faster there,
  unchanged numerics). Values and gradients match the early-exit evaluation
  to float64 round-off (≤1e-12 relative for dof ≤ 1e3) and scipy's
  `t.logcdf` to 1e-10 relative over dof 1+1e-6 … 1e3. H100 MIG 1g.12gb, MLP +
  `SkewTLoss` Adam step, batch 4096: 42.8 ms eager → 31.4 ms eager, 2.2 ms
  as a CUDA graph (batch 1024: 34.4 → 28.8 / 1.4 ms). Timing script:
  `python tools/benchmark_skew_t.py --device cuda`.
- Renamed `PoissonGaussianMixtureLoss` to `PoissonGaussianContaminationMixtureLoss`
  (registry key `"poisson_gaussian_contamination_mixture"`) and
  `EnhancedPoissonGaussianMixtureLoss` to
  `EnhancedPoissonGaussianContaminationMixtureLoss`
  (`"enhanced_poisson_gaussian_contamination_mixture"`), with matching config
  dataclasses (`PoissonGaussianContaminationMixtureConfig`,
  `EnhancedPoissonGaussianContaminationMixtureConfig`). They compute
  `w*Pois + (1-w)*N` with a shared rate, not the detector convolution
  (`PoissonGaussianLoss`). Behaviour is unchanged. The example is now
  `examples/poisson_gaussian_contamination_mixture_demo.py`.
  `PoissonGaussianLikelihoodRatioLoss` is unchanged: it is not a mixture.
- Examples: each synthetic / real-data pair is one script with
  `--dataset {synthetic,diabetes}` (causal, censored, EIV, multimodal,
  noisy-label, OOD selective prediction, ordinal, uncertain ground truth); the
  `*_realdata_comparison.py` files are gone. Overlapping demos are one script per
  topic: `loss_comparison.py --suite`, `conformal_regression_example.py --mode`,
  `contrastive_flow_parameter_estimation.py --mode`, `count_regression_demo.py
  --family` (replaces the Poisson and Tweedie demos),
  `test_time_bayesian_linear_head_demo.py --mode`,
  `gaussian_wasserstein_bound_demo.py --mode`. Summary JSON values are unchanged.
- The `external_comparison_*` examples (MAPIE, crepes, torchcp, BoTorch,
  scikit-learn) moved to torchregress-harness suites, and
  `examples/benchmarks/self_agreement_higgs_ood.py` to torchregress-research.
- Density-conformal calibration and prediction stream over rows, so peak memory
  no longer grows with the split size (1M+ rows on CPU or GPU):
  - `ConformalizedDensity` / `ContrastiveFlowConformal` evaluate the density in
    chunks of at most `chunk_size` (new keyword-only argument, default 131072)
    `(context, y)` pairs in `calibrate`, `calibrate_log_prob` and `contains`.
    Scalar `predict_interval` evaluates `chunk_size // grid_size` contexts per
    density call instead of one Python iteration per row (about 8x faster on
    2000 rows with a small NSF on CPU). Results are identical for row-wise
    densities.
  - `WaldoDensityConformal` calls `sample_fn` on row chunks (new keyword-only
    `chunk_size`, default `2**20 // n_samples` rows) and computes moments and
    Wald scores per chunk. A split that fits in one chunk gives bit-identical
    results for a fixed seed; with several chunks a sampler that draws one
    `[S, N, D]` noise block consumes the generator in a different order, so
    thresholds differ by Monte Carlo error only (same score distribution).
  - `WaldoPosteriorCalibrator` computes sample moments and scores per row chunk
    (new keyword-only `chunk_size`), without a second `[N, S, D]` temporary.
    Results are unchanged.
  - `WaldoConformal.predict_interval` is vectorised over rows (shared grid and
    critical values, no per-row Python loop or host sync; about 100x faster on
    20000 rows) and `WaldoConformal.predict_region` processes rows in bounded
    chunks. Outputs are unchanged.
  - The grid level-set search shared by `CTI`, `SLSConformal` and
    `ConformalizedDensity` no longer synchronises with the device.
  - `tools/benchmark_conformal_streaming.py` times these paths at realistic
    sizes (`--device cuda` for GPU).
- Performance: `crps_from_binned_cdf` is computed in closed form for all rows
  and bins at once (target clipped to each bin, float64, row chunks of bounded
  memory) instead of a Python loop over rows and bins, on the device of the
  first tensor argument. Return types are unchanged (float for `mean`/`sum`,
  NumPy array for `none`). Agrees with the loop to 2e-15; 1e5 rows x 64 bins on
  CPU: 119 s -> 0.11 s (1e6 rows: 1.2 s).
- Memory: `neighbour_target_gap`, `target_ordering`, `frozen_readout_metrics`
  and `representation_geometry_report` no longer build N x N (or N_test x
  N_train) distance matrices: k-NN and pair distances are computed in row
  chunks (O(chunk * N) memory) on the input's device, and tensors with
  `requires_grad` are accepted. `target_ordering` keeps its rule (all pairs
  when N(N-1)/2 <= `max_pairs`, else `max_pairs` seeded random pairs) and gives
  bit-identical values. N = 1e5, D = 16 now runs on CPU in about 25 s; before,
  the gap alone needed an 80 GB distance matrix.
- `FTTransformer` (and `AutoregressiveTimeLayer`) build their `norm_first`
  encoders with `enable_nested_tensor=False`, which silences the PyTorch
  warning on every construction. Behaviour is unchanged.

### Fixed

- `VIRRegressor` training is stable again across seeds, platforms and torch
  versions. The `kl_weight` latent regulariser averaged
  `KL(q(z|x) || N(0, I))` over the latent dimensions instead of summing them,
  so it was `latent_dim` times weaker than documented. Without a scale anchor,
  the label-smoothing recolouring let the encoder expand the latent scale every
  epoch (latent and gradient norms grew roughly 100x). Late epochs then became
  chaotic and tail error regressed on some seeds, for example in the pip/torch
  2.15 CI job. The KL is now summed over latent dimensions and averaged over
  the batch; the default weight is unchanged.
- `create_flow_model(flow_type="maf")` and the affine-coupling types
  (`"realnvp"`/`"cinn"`/`"nice"`) no longer produce exploding samples. A far-tail
  base draw could give an out-of-range coordinate; the ReLU conditioner then
  extrapolated to a huge shift and a saturated tiny scale (zuko's affine layer
  allows `|log s| < 6.9`, a 1000x stretch per layer), and the inverse pass
  cascaded geometrically over coordinates and layers. On
  `synthetic_multitarget_big` (density_parity, seed 1, GPU) the MAF had a sound
  NLL (0.957) but finite samples up to 2.4e12 and CRPS 1.3e4; all three seeds
  showed the same tail on larger draws, and so does zuko's MAF used through sbi.
  The affine flows keep zuko's layout and state-dict keys but bound each layer's
  log-scale softly by 3 (`slope=exp(-3)`, overridable with `slope=`) and feed the
  conditioners `x / sqrt(1 + (x / B)^2)` with `B = 10` (standardized units;
  `bound=` sets `B`). Both directions use the same conditioner, so the flows
  remain exact bijections with exact `log_prob`. **Outputs change**: newly
  trained MAF/RealNVP/cINN flows differ, and existing checkpoints load unchanged
  but evaluate under the new layers (pass `slope=1e-3` and a very large `bound`
  to reproduce the old map). `bound` is now accepted for `"maf"` and the
  affine-coupling types as the conditioner clip half-width.
- Sampling a normalizing flow (`NormalizingFlowLoss.sample` and everything built on it:
  `ContrastiveFlowLoss`, `quantile`/`cdf`, WALDO sample functions) and the spline
  inverse of `GaussianCopulaSplineHead.intrinsic_log_prob` / `observed_log_prob` no
  longer strand tensors in reference cycles. torch's `Transform.inv` caches its
  inverse with a strong reference, so every zuko inverse pass left a
  `transform <-> inverse` cycle holding its spline parameters (about 11 KiB per
  6-D NSF draw) until Python's object-count-driven GC ran; on CUDA chunked
  sampling of a few hundred thousand draws ran an 11 GiB device out of memory.
  The young generation is now collected after each call; results are unchanged.
- Medians follow the textbook (NumPy/SciPy) convention, averaging the two
  middle values for even counts, instead of `torch.median`'s lower middle value:
  `neighbour_target_gap`, the `*_NMAD` entries of `frozen_readout_metrics`,
  `calculate_mad` (and so `reweight_fixed_predictions(variance_type="robust")`),
  the masked `scale="mad"` estimate of `iteratively_reweighted_fit`, the
  `VarianceTemperatureScaler` `clip` centre/MAD and variance median,
  `BalancedMSEBMC`'s robust scale and fitted median, `RepresentationShiftInflator`'s
  reference scale, the median-heuristic bandwidths of `random_fourier_features`
  and the test-time Bayesian head, and the MC-sample centre in `conformal.py`.
  Results move slightly for even-sized inputs.
  `neighbour_target_gap`'s docstring now says what it computes: the median over
  points of the mean absolute target difference to the k nearest neighbours.
- MAD-to-sigma consistency constant is `1 / Phi^-1(0.75) = 1.482602218505602`
  instead of the rounded `1.4826` everywhere (`MedianAbsoluteDeviation`,
  `median_absolute_deviation`, `NormalizedMedianAbsoluteDeviation`, representation
  `nmad` and readout NMAD, IRLS robust scales, `BalancedMSEBMC` robust scale,
  `VarianceTemperatureScaler` `clip`).
- `PoissonGaussianMixtureLoss` and `EnhancedPoissonGaussianMixtureLoss` docs
  and docstrings described the shot-plus-read-noise convolution and cited
  Seifert et al. (2023), which uses it. They compute a two-component mixture;
  the docs now say so, report its low-count bias, and point to
  `PoissonGaussianLoss`. Behaviour is unchanged.
- Examples whose real-data track never seeded the global RNG (censored, uncertain
  ground truth) now do, so their Diabetes results are reproducible.
- `WaldoConformal` (and `WaldoPosteriorCalibrator` / `WaldoDensityConformal`,
  which use it) set the Wald threshold with an interpolated `torch.quantile`
  instead of the finite-sample order statistic `ceil((n+1)(1-alpha))`, so
  regions under-covered (0.882 vs 0.90 nominal at n=50 over 2000 repeats; now
  0.9075). Global, per-bin and `WaldoCriticalValueRegressor` thresholds all use
  `finite_sample_quantile`. A bin with too few points for `alpha` has an
  infinite threshold, and the interpolated critical value next to it is `+inf`
  instead of `nan`. Found by the torchregress-harness multivariate rows.
- `probability_integral_transform(return_histogram=True)` raised on CUDA tensors
  (`torch.histogram` has no CUDA kernel); it now bins with bucketize/bincount
  under the same convention.
- `ConformalLoss` no longer swallows unknown keyword arguments (including the
  removed `debias`). Docs and tests that still advertised the no-op are gone.
- Removed the thin `PredictiveBatch.from_density` / `from_samples` constructors
  (use the dataclass fields directly).
- Removed the identity factory `create_mdn_loss` (use `MixtureDensityLoss`).
- `WaldoDensityConformal` and `WaldoPosteriorCalibrator` with
  `covariance_type="diagonal"` raised a shape error for multi-target samples
  (they passed the full sample covariance). They now use the per-target sample
  variances.
- `GaussianLatentForwardObservationNLLLoss` placed its Gauss–Hermite nodes on
  the latent prior, so it was badly wrong when the observation noise is narrow
  relative to the latent spread pushed through the map, even for a linear map
  (harness repro: -0.30, +0.47, -0.10 nats at Q=64, 128, 256; up to hundreds of
  nats on synthetic SNR 7-50 fluxes at Q=32). It now uses adaptive
  Gauss–Hermite quadrature: a fixed number of derivative-free bracketing and
  parabolic passes on `[N,Q]` node grids centre and scale the rule on the
  integrand's peak. The centre and scale are a stop-gradient change of
  variables, there are no host syncs, and it runs under `torch.inference_mode`.
  The rule is exact for linear maps at any order of 3 or more, and the
  synthetic flux grid's maximum error at Q=32 drops to 2e-7 nats. The map is
  now called several times per forward (once per pass without autograd), so it
  must be deterministic and side-effect free. Forward cost at equal order is
  about 2.5-5x on CPU, but Q=16 now beats the old Q=256. The quadrature
  constants are no longer float64 buffers: they are converted to the input
  device and dtype at use, so `.to("mps")` works. They no longer appear in
  `state_dict()`.

## [0.3.0b1] - 2026-10-26

API cut. One public name per behavior.

### Removed

- Compatibility aliases: `MDNLoss` (use `MixtureDensityLoss`), `ExpectileCrossover`, `QuantileCrossover`, `AsymmetricLeastSquaresLoss`, `CORALLoss` (use `CumulativeLinkLoss`; `CORALHead` is unchanged), `JackknifePlus` (use `CVPlus`; leave-one-out is `K = n`), `SelfAgreementTrainer` (use `TeacherStudentTrainer`), `iteratively_reweighted_least_squares` (use `reweight_fixed_predictions` or `iteratively_reweighted_fit`), `RepresentationShiftCalibrator`, `SignificantSubspaceAligner`, `prediction_interval_coverage`, `MeanIntervalWidth`.
- Torchmetrics re-exports and the calibration-metric re-exports from `torchregress.metrics`. Calibration metrics are `torchregress.calibration` only.
- The ignored `debias` argument on `CQR`, `UACQR`, and `ConformalLoss`.
- `utils.quantile.quantile_loss` is `pinball_residual`. `fetch_openml_regression_with_sklearn_fallback` is `fetch_openml_regression`.
- Registry keys `"als"` and `"coral"`.

### Added
- True noise-adaptive Balanced MSE (`BalancedMSEBMC`, `balanced_mse_bmc_loss`;
  Ren et al., CVPR 2022) with fitted label prior and target-noise scale; this is
  distinct from the historical inverse-bin-frequency `BalancedMSELoss`.
- RankSim ranking-similarity regularizer (`RankSimLoss`, `rank_sim_regularizer`;
  Gong et al., ICML 2022) for representation-level imbalanced regression, with
  a blackbox-interpolation gradient and composition with task losses.
- Variational Imbalanced Regression (`VIRLoss`, `VIRRegressor`) for long-tailed
  regression with evidential (Normal-Inverse-Gamma) uncertainty: label-similarity
  latent smoothing, smoothed label-density reweighting, evidential regularisation,
  and `predict(return_std=True)`. Docs: `docs/methods/algorithms/vir.md`.
- Corrected the historical `BalancedMSELoss` documentation: it implements inverse
  bin-frequency weighting, not Ren et al.'s noise-adaptive Balanced MSE (BMC);
  the misleading citation was removed and the distinction is documented.
- Contrastive likelihood-ratio / neural-ratio-estimation tooling (2026-10-17):
  - `losses.NeuralRatioLoss` (+ `build_ratio_classifier`, `neural_ratio_logits`,
    `neural_ratio_loss`): classifier-based neural ratio estimation with balanced binary
    (NRE-A) and multiclass (NRE-B/InfoNCE-style) criteria, paired or mismatched negatives,
    and flow-loss-style shape/ambiguity validation. The logit estimates
    `log p(x|theta) - log p(x)`, so grid-argmax is the MLE and the training prior cancels
    under a prior-sampled reference. Docs: `docs/losses/nre.md`.
  - `inference.posterior_log_weights`, `effective_sample_size`, `grid_log_posterior`,
    `maximum_likelihood_estimate`: finite-bank likelihood-ratio/posterior reweighting,
    Kish ESS diagnostics, and MLE/MAP point estimates with documented prior/proposal
    corrections. Docs: `docs/api/inference.md`, `docs/methods/inference.md`.
  - `metrics.simulation_based_calibration_ranks`, `sbc_rank_histogram`,
    `sbc_uniformity_pvalue`, `tarp_coverage`: simulation-based calibration and TARP
    posterior-coverage diagnostics. Docs: `docs/metrics/calibration.md`,
    `docs/api/metrics.md`.
  - `ContrastiveFlowLoss`: documented `temperature=1, margin=0` likelihood-ratio regime with
    a `likelihood_ratio_mode` property, chunked negative evaluation, and target-shape
    validation; the flow guide now frames it correctly as InfoNCE/multiclass NRE and
    distinguishes it from the Elsharkawy-Kahn contrastive-normalizing-flow method.
  - The contrastive-flow comparison example now also trains classifier NRE and an NPE-style
    posterior flow, and reports SBC/TARP diagnostics next to point accuracy and runtime.
- **Post-freeze method additions (user-requested audit, 2026-10-17; targeted at 0.4.0):**
  - `algorithms.ForwardInversionRegressor` (+ `build_decoder_mlp`): generative `y -> x`
    surrogate (statistical inverse regression / auto-decoder construction) trained with
    known per-observable errors and inverted per object by maximum likelihood (damped
    Gauss–Newton or L-BFGS) with Fisher covariance `(J^T Sigma_x^{-1} J)^{-1}`; removes
    input attenuation and inverse-regression prior bias. Training supports the full
    heteroscedastic Gaussian likelihood (default), a robust Student-t likelihood
    (`likelihood="student_t"`) and fixed-weight heteroskedastic least squares
    (`likelihood="wls"`). `predict(..., return_info=True)` returns per-object stationarity
    and Fisher-condition diagnostics for the local, non-convex test-time inversion, and the
    guide documents when a discriminative model is the better choice. Docs:
    `docs/methods/algorithms/forward_inversion.md`.
  - `losses.SplitMeasurementLoss` / `split_measurement_loss` and
    `losses.CorrectedScoreEIVLoss` / `corrected_score_eiv_loss`: paired-exposure
    cross-product loss (no noise model required) and the Stefanski–Nakamura corrected
    score (subtracts the measurement-error curvature instead of adding a TLS gradient
    penalty). Docs: `docs/losses/split_eiv.md`.
  - `losses.PrimeLoss` and `losses.ConR` (+ `conr_loss`, `uniform_proxy_targets`):
    representation-level imbalanced-regression losses (PRIME proxies with uniformly
    spaced targets; ConR contrastive collapse penalty). Feature Distribution Smoothing
    already existed (`losses.FeatureDistributionSmoother`) and is cross-linked, not
    duplicated. Docs: `docs/losses/prime_conr.md`.
  - `algorithms.ContinuousGroupDRO` and `algorithms.GroupDROTrainer`: exponentiated-gradient
    worst-target-regime training over continuous target bins (equalizes per-regime mean
    loss, unlike CVaR's worst individual samples). Docs: `docs/methods/algorithms/group_dro.md`.
  - `calibration.VarianceRestoringCalibrator`: marginal de-shrinkage by moment matching or
    1-D optimal transport, with `describe()` diagnostics; documented as marginal (not
    conditional) calibration. Docs: `docs/methods/calibration.md`.
  - `method_catalog` metadata, `docs/api/*`, loss-guide indexes and site navigation updated
    for all of the above.
- `MixtureDensityLoss.sample` supports `covariance_type="full"` (draws `mu + L eps` from the selected component's covariance Cholesky factor); previously only diagonal mixtures could be sampled.
- `inference.random_fourier_features` and `inference.median_heuristic_bandwidth`: random Fourier feature nuisance basis for `orthogonal_partially_linear` with a median-heuristic bandwidth (median pairwise distance of the standardised `z` on a seeded subsample), per-covariate polynomial terms (`polynomial_degree`, default 3: across five confounding designs it keeps 95% coverage within 0.90-0.95, where degree 1 drops to 0.78 on cubic confounding) and a seeded draw. `orthogonal_partially_linear(ridge="gcv" | "loo")` picks the nuisance ridge penalty per nuisance and training fold by generalized cross-validation or exact leave-one-out error (closed form from one SVD, unpenalised intercept); a float `ridge` (default `1e-6`) behaves exactly as before. Over 200 replications at n = 500 the old harness recipe (unit-bandwidth RFF, `ridge=1e-2`, no linear part) gave Coverage95 0.785 / bias +0.008 on CCDDHNR-2018 and 0.820 / +0.060 on the cubic design; the new defaults give 0.955 / +0.004 and 0.900 / +0.016, and `polynomial_degree=3` gives 0.945 / -0.002 on the cubic design (plan item B5).
- `losses.create_flow_model(bound=...)` sets the NSF spline tail bound (zuko fixes it at 5), and `losses.recommended_tail_bound(y_standardised)` derives it from the training target (data range with 25% head-room, clipped to [5, 25]). `NormalizingFlowLoss`'s out-of-range warning reports the actual bound. New "Training recipe for NLL and calibration" section in `docs/losses/nflows.md` (validation early stopping, derived bound, 16 bins): test NLL 1.272 -> 1.201 (skewed), 1.935 -> 1.852 (Student-t 3 df), 6.144 -> 5.417 (`load_diabetes`) with equal or better 90% interval scores, versus the fixed 60-epoch recipe of the harness `pzflow_parity` row (plan item B4). Library defaults (`bins=8`, bound 5) are unchanged.
- `test_time.select_high_confidence(scores=...)`: optional per-row ranking score (default: maximum probability).
- `torchregress.models`: `TabularMLP` (periodic numeric embeddings, SiLU blocks, dropout), `TabularPreprocessor` (median/IQR scaling, smooth clipping, NaN imputation with missing indicators), `fit_tabular` / `fit_tabular_ensemble` (AdamW, one-cycle schedule, early stopping with restore-best-weights, target standardisation) and `TabularFit`. Works with any loss, including `GaussianNLLLoss` (`2 * D` outputs) and `MultiQuantileLoss`. Docs: `docs/methods/tabular_mlp.md`, `docs/api/models.md`. `models` is a new lazy top-level submodule (plan item B2).
- `torchregress.estimators` (experimental, may change in 0.4): model-agnostic UQ wrappers. `ConformalRegressor(base, method="split" | "normalized" | "cqr" | "cv+" | "jackknife+", alpha, n_folds, prefit)` with `fit` / `predict` / `predict_interval(X, alpha, test_weights)` puts `SplitConformal`, `CQR` or `CVPlus` intervals on any scikit-learn or LightGBM model, a quantile pair/triple, a torch module with a training callable, or a fitted `TabularFit`; NumPy in gives NumPy out and tensors give tensors. `CalibratedRegressor(base, method="vts" | "isotonic+vts")` fits `VarianceTemperatureScaler` (and `IsotonicMeanCalibrator`) on a held-out split of any Gaussian-output base and offers `predict_dist` / `predict_interval`, optionally with conformal widths. Tests check parity with direct use of the conformal classes and with MAPIE `SplitConformalRegressor` / `ConformalizedQuantileRegressor` (prefit) to 1e-6. Docs: `docs/methods/estimators.md`, `docs/api/estimators.md`. `estimators` is a new lazy top-level submodule (plan item B1).
- `estimators.calibrated_deep_ensemble(X, y, n_members, val_fraction, loss, alpha, conformal, seed, **fit_kwargs)`: Gaussian `TabularMLP` ensemble via `fit_tabular_ensemble` with mixture moments (mean member variance plus variance of member means), a variance temperature fitted on a held-out carve and optional normalized split-conformal intervals; returns a `CalibratedRegressor` (plan item B3). Experimental, may change in 0.4.

- Examples for the 0.3.0 APIs: `tabular_mlp_tutorial.py` (`TabularMLP`, `fit_tabular`, `TabularPreprocessor` with NaNs), `calibrated_regressor_comparison.py` (`CalibratedRegressor`, `calibrated_deep_ensemble`, `ConformalRegressor`), `nsf_tail_bound_comparison.py` (`recommended_tail_bound` vs the fixed NSF bound on Student-t noise) and `orthogonal_inference_rff_demo.py` (random Fourier nuisances, median bandwidth, `ridge="gcv"`), each under 15 s on 2 CPU cores and smoke-tested.

### Changed

- Declared public (added to `__all__`; they were already importable):
  `CQR`, `CTI`, `MultivariateScoreConformal`,
  `NonExchangeableConformalRegressor` and `BaseEIVLoss` in
  `torchregress.losses`, and `SelfAgreementTrainer` (alias of
  `TeacherStudentTrainer`) in `torchregress.semi_supervised`. Listed in
  `docs/api/losses.md`, `docs/api/conformal.md` and
  `docs/api/semi_supervised.md`.
- `MixtureDensityLoss(covariance_type="full")` now reads its lower-triangular
  factor as the Cholesky factor of the **precision** (`full_parameterization="precision"`,
  new default, as in sbi's MDN): the log-density is linear in the factor, with no
  triangular solve. The covariance form (`full_parameterization="covariance"`, the
  previous behaviour) did not fit multi-target data: on the 16-target scm20d
  benchmark, train / test NLL 0.42 / 3.32 vs -3.45 / 0.31 with the precision form
  (sbi's MDN: 0.04). The output layout is unchanged. `predict_mean_std` and
  `predict_interval` now support full covariance (per-dimension marginals).
  The precision form is not better everywhere: in harness run #6
  (`density_parity`, test NLL) it improves mt_scm20d 3.11 → 0.33 and mt_jura
  2.53 → 1.95 but is worse on the 2-target energy_efficiency (-1.32 → -0.76)
  and mt_enb (-1.18 → -0.85). `docs/losses/mdn.md` now explains both
  parameterisations, recommends comparing `full_parameterization="covariance"`
  on validation NLL for low-dimensional targets, and warns that the extracted
  precision factor `T` is not a covariance square root (use `.sample()`).

- `quantiles_to_density_grid` (and `PredictiveBatch.with_density` for quantile
  batches) keeps the tail mass: `tau_0` below the first quantile and
  `1 - tau_K` above the last are spread uniformly over the grid margins
  instead of being dropped and renormalised. The grid CDF now equals `tau_k`
  at every knot; for levels 0.05/0.95 a 90% interval read off the grid covers
  0.900 of Gaussian draws instead of 0.854 (PRD-001; supersedes TR-COR-02).
- `docs/losses/sls.md`: new section measuring the width of SLS bounding-box intervals against `GaussianNLL` / `MultivariateGaussian` on the three harness multivariate designs (plan item B7). The 1.2-1.3x width gap at nominal level shrinks to 1.04-1.11x at equal joint coverage; raw `SLSLoss` thresholds cover only 0.45-0.56 of held-out targets (0.91 in-sample), so the coverage guarantee needs `SLSConformal`. No library default changed: warmup, window, flow size and boundary-sample count do not reduce width at equal coverage on all designs.
- 28 plumbing helpers are no longer re-exported from `torchregress.utils` (and
  dropped from its `__all__`); they remain importable from their submodules, for
  example `from torchregress.utils.tensor_ops import convert_to_tensor`.
  `distributions`: `normal_cdf`. `gaussian_output`: `parse_heteroscedastic_output`,
  `low_rank_output_dim`, `split_low_rank_gaussian_output`. `numpy_stats`:
  `subsample_rows`, `winsorize`. `ordinal`: `labels_to_levels`,
  `normalize_class_probs`, `class_probs_to_levels`. `pytorch_compat`: `get_device`.
  `quantile`: `quantile_loss`, `multi_quantile_loss`. `tensor_ops`: `apply_mask`,
  `convert_to_tensor`, `ensure_batch_dim`, `prepare_cross_covariance`,
  `prepare_model_input_for_gradients`, `compute_model_gradients`,
  `calculate_gaussian_nll`, `calculate_propagated_variance`. `validation`:
  `validate_reduction`, `validate_range`, `validate_quantile`, `validate_weights`,
  `validate_metric_inputs`, `validate_sample_weight`, `check_tensor`. `security`:
  `validate_url`. `parse_heteroscedastic_output`, `low_rank_output_dim` and
  `split_low_rank_gaussian_output` stay public in `torchregress.ensemble` /
  `torchregress.losses`.
- Dependency floors raised to the newest releases resolvable from conda-forge:
  `torch>=2.13`, `numpy>=2.5`, `scipy>=1.18`, `torchmetrics>=1.9`; extras
  `matplotlib>=3.11`, `zuko>=1.6`, `scikit-learn>=1.9`, `pandas>=3.0`,
  `polars>=1.43`, `pyarrow>=25.0`, `pytest>=9.1`. Pre-commit hooks: ruff
  v0.16.10, pre-commit-hooks v6.0.0. Tested on torch 2.14.1 / numpy 2.5.3 /
  scipy 1.18.1 (PyPI) and the pixi lock (conda-forge torch 2.13.0).

### Removed
- The `torchregress.metrics.calibration` shim module. Its 7 names
  (`ExpectedCalibrationError`, `MarginalCalibrationError`, `bias`,
  `calibration_metrics_report`, `calibration_score`,
  `expected_calibration_error`, `marginal_calibration_error`) stay exported
  from both `torchregress.metrics` and `torchregress.calibration`: replace
  `from torchregress.metrics.calibration import X` with
  `from torchregress.metrics import X` or `from torchregress.calibration import X`
  (the implementation lives in `torchregress.calibration.metrics`).

### Fixed
- `InverseGaussianLoss` and `GammaLoss` (and `TweedieLoss` with `p >= 2`) now
  raise `ValueError` ("... requires strictly positive targets") when any
  unmasked target is `<= 0`. Both distributions have support `y > 0` and their
  deviances are undefined at `y = 0`; the `eps` guards silently turned a zero
  target into a large finite penalty (about 5e7 per entry for the Inverse
  Gaussian, about 17 for the Gamma at `mu = 1`) that dominated training.
  Masked-out entries are not checked. Targets are not shifted or clipped.
- `LocalConformal`, `LocalConformalMAD` and `DensityConformal` built dense
  (n_test x n_cal) kernel matrices with several temporaries alive at once, so
  memory grew with the product of the two sizes: at ~2x10^4 x 1.5x10^4 rows the
  harness `conformal/zurich_delays` job was killed by a 32 GB limit. Kernel
  weights and the weighted-quantile search now run in test-row blocks
  (O(n_cal) memory per block); results are identical.
- `CVPlus` / `JackknifePlus` materialised and fully sorted two
  (n_train x n_test) candidate tensors (CV+ uses every training row); the
  order statistics now come from `torch.kthvalue` over test-row blocks, with
  identical endpoints.
- `metrics.ood` kernel density scores and `NeighborhoodCovariancePseudoLabeler`
  (which built an (n_query, n_ref, D) difference tensor) now work in query-row
  blocks; results are identical.
- `algorithms.WarmupMCTrainer` iterated a `DataLoader` over a `TensorDataset`
  (one Python `__getitem__` per row per epoch); it now shuffles and slices on
  the device, as `LatentNN` does.
- `algorithms.LatentNN` scaled quadratically with the number of rows: the per-row
  latent inputs were one dense `nn.Parameter` under `torch.optim.Adam`, so every
  mini-batch step updated all N x D values and moments (O(N^2 D / B) per epoch;
  8x the rows took 36x longer, and the harness `folktables_eiv` job with 1.2M rows
  ran for 2.5 days without finishing). Latents now use a lazy, row-sparse Adam
  (only the batch rows' values and moments move, per-row bias correction), batches
  are drawn on the device instead of through a `DataLoader`, and Monte Carlo
  `predict(sigma_x=...)` runs in row chunks. Training is linear in N; full-batch
  training is unchanged (identical to dense Adam). Mini-batch results change:
  rows outside the batch no longer drift on stale momentum.
- `fit_tabular` / `TabularFit.predict`: `GaussianNLLLoss(log_variance=False)` heads are mapped back to the original scale with a new `"variance"` layout (`var * sigma^2`) instead of being treated as a location output; `standardize_target=False` keeps the Gaussian layout, so `calibrated_deep_ensemble(..., standardize_target=False)` works (audit C2-003, C2-006).
- `ConformalRegressor`, `CalibratedRegressor` and `calibrated_deep_ensemble` raise `ValueError` for NaN or infinite targets instead of returning NaN intervals or a temperature fitted on a subset (audit C2-004).
- `TabularPreprocessor`: the smooth clip no longer overflows for huge values (mapped to 0 above 1e154) or turns infinite inputs into NaN (audit C2-002).
- `inference.median_heuristic_bandwidth` is exact for data far from the origin and rejects non-finite input; `test_time.select_high_confidence(scores=)` ranks NaN scores lowest (audit C2-001, C2-005).
- `test_time.ShiftFactoredPredictiveTransport`: the confident-row selection (`top_fraction`, default 0.5) now ranks rows on the unrenormalised in-grid peak probability (maximum bin probability times the predictive mass inside the support grid) instead of the renormalised maximum. Renormalisation inflated the peak of predictives cut by the grid edge, so on fine grids the selection favoured them and the EM prior mean overshot a label shift by up to 0.32 |shift| at 256 bins (0.14 at 64); it is now below 0.07 |shift| at 16, 64 and 256 bins for shifts of +/-0.8. Selected rows, and therefore `target_prior` and the adapted predictive, change for a given input (plan item B6).
- `test_time.ShiftFactoredPredictiveTransport` with a Gaussian predictive: the prior correction is applied to the native mean / std as a moment change (transported minus untransported binned moments, weighted by the row's in-grid predictive mass) instead of returning the moments of the transported grid. Those added the discretisation and the truncation at the grid edge whenever `transport_applied` was true, so a near-identity prior moved the predictive (up to 3.7 sigma for edge rows) and target predictives beyond the source label range collapsed onto the edge bin: on a 2.5 SD label shift the target NLL went from 0.19 (untransported) to 7-4300 depending on `n_support`, now 0.24-0.26 (0.19-0.24 without uncertainty inflation); on a 0.8 SD shift without inflation 0.029 -> 0.046 before, 0.019 now (16 bins). New metadata `prior_mass_outside_source_range` and a `UserWarning` when the raw EM prior puts more than 0.2 of its mass on the support bins outside the source label range (label-shift support violation or unreliable EM). The docs page now warns that the stabilised prior's shrink weight scales as 1/`n_support` (2% of the EM correction at the default 256 bins on a calibrated Gaussian predictive) and that uncertainty inflation widens the predictive under label shift (harness finding 8 follow-up).
- `inference.orthogonal_partially_linear`: both nuisance regressions (`E[x|z]`, `E[y|z]`) now share one cross-fitting split. Independent splits biased `theta` (-0.026, about 7.7 standard errors, over 200 replications of the DoubleML CCDDHNR-2018 design at n = 500); found by the harness `orthogonal_inference` suite against DoubleML on the same nuisance basis. Point estimates for a given `seed` change.
- **0.3.0 release audit: 61 defects fixed**, each with a regression test in
  `tests/audit/` (full list with severities: `docs/reports/audit_0.3.0.md`).
  Results that change for existing code:
  - *Conformal prediction.* Thresholds are `+inf` (infinite intervals) when
    `ceil((n+1)(1-alpha)) > n`, per Mondrian group, or when the weighted mass
    cannot reach `1-alpha`; they used to fall back to the largest score and
    under-cover at small `n`. Weighted conformal follows Tibshirani et al.
    (2019): only relative weights matter, the test point defaults to the mean
    calibration weight, and a new `test_weights=` argument gives per-point
    thresholds. CV+/Jackknife+ use the lower rank `floor(alpha (n+1))`.
    `DensityConformal` scores use the density at `y_pred` in both phases.
    Negative Mondrian group ids work; R2C, CTI, DCP and `SLSConformal` honour
    `groups=`. `SemiConformalCalibrator` computes in float64.
  - *Losses.* `EvidentialRegressionLoss` uses the NIG predictive
    (`Omega = 2 beta (1 + nu)`, Amini et al. 2020, Eq. 8). Skew-t NLL values
    change (incomplete-beta bug). Skew-normal, censored/AFT, zero-inflated
    Poisson and GEV are exact in the tails instead of saturating.
    `BetaNLLLoss` and `CVaRLoss` handle 1-D `[B]` inputs. Weighted `'mean'`
    is normalised by the sum of weights in the flow, multi-expectile and
    imbalanced losses (FocalR, BalancedMSE, BinReweightedMSE,
    DensityWeighted, LDS, PropensityWeighted), and `BaseLoss` accepts `[B, D]`
    masks with `[B]` weights. `ExpectileCrossoverLoss` raises on
    non-ascending levels. EIV losses propagate gradients through the
    Jacobian; latent-marginalisation and input-noise losses now marginalise;
    `SLSLoss.forward` no longer mutates state (new `evaluate_frontier`).
  - *Metrics.* `distribution_metrics_report["crps"]` is exact (was 3-4% low).
    Quantile PIT randomises out-of-range targets (seedable `generator=`).
    MedAE/MAD/NMAD use the true median and per-output averaging (sklearn);
    trimmed MSE matches `scipy.stats.trim_mean`. `VarioScore.higher_is_better`
    is `True`. Absolute variance floors and covariance jitter became
    `min_variance=` / relative `jitter=` arguments.
  - *Calibration.* `VarianceTemperatureScaler` has no hidden `[0.05, 20]`
    clamp (`temperature_bounds=` restores it) and returns the closed-form MLE
    by default. `IsotonicMeanCalibrator` is true isotonic regression.
  - *Inference and causal.* `ppi_quantile_ci` inverts the rectified CDF
    (Angelopoulos et al. 2023); the old estimator was inconsistent. PPI and
    DR estimators keep float64; the PPI bootstrap no longer allocates an
    `[n_boot, N]` matrix. DR `fold_bootstrap` standard errors are correct
    (were 0.56x at two folds); trimming every unit raises.
  - *Ensembles.* Predicted variances use the `GaussianNLLLoss` training range
    `[1e-6, e^30]` (was `[e^-8, e^6]`); `predict` runs in eval mode and
    restores the previous modes; `base_seed` leaves the global RNG alone.
    `bars_to_density_grid` puts no mass outside the bin edges.
  - *Semi-supervised.* The documented trust-weight options (`tau`,
    `weight_power`, `hard_weight_threshold`, `batch_relative_mode`,
    `batch_trust_top_k`) take effect; `TeacherStudentTrainer(tau=)` defaults
    to 0.2, the value the code always used. Unknown options raise.
- `test_time.ShiftFactoredPredictiveTransport.ppi_target_ci` keeps float64 inputs.
- **Audit batches 4–5** (`algorithms`, `test_time`, `utils`, `prediction`,
  `comparison`, `viz`): 37 more defects fixed (one documented, decision
  pending), tests in `tests/audit/test_audit_batch4.py`
  and `test_audit_batch5.py`.
  - *IRLS* weights are `w(r/sigma)` times the initial precision; they used to
    compound over iterations (`w^n_iter`). `variance_type="robust"` takes the
    MAD over samples (it was zero for `(N, 1)` targets). Concatenated
    `[mean, log_sigma]` outputs work with the default config.
  - *SIMEX* accepts float64, is scale-equivariant (no absolute jitter), and
    `predict` averages independent remeasurements. *Regression calibration*
    honours `posterior(sigma_u=)` and uses relative eigenvalue floors.
  - *Weighted / OT conformal and transport conformal* return infinite
    intervals when the finite-sample guarantee needs them, and the transport
    threshold is the exact order statistic (it was one too high).
  - `DomainClassifierRatioEstimator` corrects for unequal pool sizes;
    `DelayedLabelResidualAdapter` no longer deflates the first batch;
    `IVON.state_dict()` stores the step count; `HeteroscedasticLaplaceRegressor`
    accepts a plain linear head and `n_samples=1`; `BayesianLinearHead`
    honours `generator=`; transport, label-shift and pseudo-label helpers keep
    float64.
  - `comparison.compute_point_metrics` no longer broadcasts `[N, 1]` against
    `[N]` (N×N errors) and averages R² per output (scikit-learn).
  - `masked_mean`/`masked_sum` ignore NaN at masked positions; Box-Cox and
    Yeo-Johnson are accurate for small lambda; `normal_cdf` is accurate in the
    lower tail; validators reject NaN; `quantiles_to_density_grid` sorts
    crossing quantiles (its truncation to `[q_0, q_K]` is now documented);
    numeric-coded nominal OpenML columns load.
  - Viz: residual histogram and QQ plot handle `[N, 1]` predictions, the PIT
    histogram spans [0, 1], the Qini area is finite, RMSCE ignores empty bins,
    `plot_performance_comparison` knows more lower-is-better metrics, and
    plots that create their own figure close it.
- Found by the harness after the audit (tests in
  `tests/audit/test_audit_postharness.py`):
  - `SkewTLoss` / `skew_t_nll`: the gradient with respect to the skewness was
    exactly zero at `alpha = 0`, so heads initialised there never learned
    skew. The Student-t CDF term is now smooth at 0 and log-space in the
    tails; values are unchanged (match scipy).
  - `ShiftFactoredPredictiveTransport`: the source prior is the mean source
    predictive distribution on the support grid (it was an eps-clamped label
    histogram), and support-grid EM stops on a log-likelihood plateau. EM no
    longer collapses onto empty margin bins (prior TV ~1 with no shift) and
    prior transport is applied under label shift. `estimate_target_prior_em`
    gives classes with no source mass a target prior of 0; new
    `LabelShiftEMConfig.loglik_tol` (default off).
  - Docs: EM label-shift estimation needs calibrated source posteriors (BBSE
    does not).
- Metrics return float64 for float64 inputs (18 metrics used to return
  float32), and functional metric wrappers move their internal torchmetrics
  state to the input device. Calibration-error histograms no longer use
  `torch.histogram`, which has no CUDA kernel.
- `import torchregress.test_time` works without scikit-learn (only the default
  classifier of `WeightedConformalRegressionAdapter` needs it).
- Examples: every script runs against the current API; external comparisons
  use MAPIE 1.x, torchcp 1.2 and scikit-learn's `TweedieRegressor`; the
  Tweedie examples no longer apply `exp` twice.
- `CORALLoss` no longer overwrites the `CumulativeLinkLoss` docstring: it is
  now a thin subclass of `CumulativeLinkLoss` with its own docstring (same
  computation, same `"coral"` registry key).
- `torchregress.method_catalog` declares an explicit `__all__` (its 4
  dataclasses, `CapabilityValue` and 5 functions), so `import *` no longer
  re-exports `Any`, `Dict`, `dataclass` and the other typing imports.
- Docs: removed names that do not exist (`PackedEnsembleRegressor`,
  `MCDropoutModel`, `BayesianModelAveraging`, `mse`, `mean_absolute_percentage_error`,
  `conformal_loss`, `create_eiv_loss`, ...) and corrected calls the functions
  do not accept (`mask=`/`weights=` on point metrics, which take `sample_weight`;
  `AsymmetricLeastSquaresLoss(tau=)`, `create_grid_figure(n_cols=)`,
  `plot_pit_histogram(bins=)`, `prediction_interval_coverage(confidence=)`).
  `plot_target_density_error_overlap` is now shown as `(y_true, y_pred)`.
- `highest_posterior_density_coverage`: the docstring and the metrics guide now
  state that `alpha` is the nominal HPD mass, not the miscoverage (no
  behaviour change).

### Packaging and CI
- SPDX licence metadata (`setuptools>=77`), project URLs, typed classifier;
  the sdist ships the test suite and CHANGELOG.
- CI `package` job and release `install-smoke` matrix (Linux/macOS, Python
  3.12–3.14): build, check contents (`scripts/release/check_dist.py`),
  `twine check --strict`, install with core dependencies only and import every
  submodule (`scripts/release/smoke_install.py`).
- `docs.yml` publishes the documentation to GitHub Pages.
- Local CUDA testing: `cuda` marker, CPU-vs-device parity suite for 77 losses
  and 94 metrics (`tests/test_device_parity.py`), `pixi run test-cuda`
  writing `reports/cuda/<date>_<sha>.txt`.

### Tests
- `tests/metrics/test_reference_parity.py` checks all 97 `metrics` exports
  against scoringrules, properscoring, scipy, scikit-learn or closed forms
  (14 have no reference and say why); `tests/metrics/test_scoring_properties.py`
  adds Hypothesis invariants (CRPS non-negativity and equivariance,
  propriety, pinball monotonicity). `scoringrules`, `properscoring` and
  `hypothesis` join the `test` extra.

### Added
- `losses.FaithfulGaussianLoss`: `mean_weight` accepts a per-output vector (length `D`, validated finite and non-negative), and a new `mean_loss="huber"` option (with `huber_delta`) bounds the pull of outlying targets on the mean. Defaults unchanged. `mean_weight` is now a registered buffer.
- `utils.BSplineDensityBasis`: unit-integral (M-spline) B-spline basis for simplex-parameterised 1D densities — exact bin integrals, cumulative moments and `E|U - x|` via piecewise Gauss–Legendre.
- `losses.DiscreteWasserstein1Loss` / `discrete_wasserstein1`: 1-Wasserstein between mass vectors on a shared 1D grid (mass or point targets).
- `losses.RankNContrastLoss` / `rank_n_contrast_loss`: Rank-N-Contrast (Zha et al. 2023) with tie-aware rank denominators, mask/weights support.

## [0.2.0] - 2026-08-27

### Fixed
- `BetaNLLLoss` preserves `[B,D]` elementwise before `_reduce` so partial masks `[[True,False]]` no longer discard whole rows (NEW-HIGH-02)
- `PoissonLikelihoodRatioLoss` clamps `exp(y_pred.clamp(max=30))` to avoid `inf` for `y_pred~100` (NEW-HIGH-04)
- `GEV` `gev_nll_elementwise` masks `1/xi` and `pow(-1/xi)` via `use_gev` so `xi→0` does not overflow before `where` (NEW-HIGH-03)
- `Tweedie p=1` Poisson term uses `ratio.clamp(min=eps)` not `log(+eps)` bias (NEW-MED-03)
- `prediction.quantiles_to_density_grid` forces `float32` levels for int quantiles, `n_support>=2` and `isfinite` guards (NEW-LOW-01/05)
- `losses/families.py` adds `unconstrained_inputs` flag (default `True`) to avoid `softplus(softplus(x))` (NEW-MED-01, pattern shown for SkewNormal/SkewT)
- `losses/conformal.py` unifies `_weighted_conformal_threshold` → `_weighted_quantile` augmented `+w_{n+1}=1` (NEW-HIGH-01) and CV+/Jackknife+ docs correct `>=1-2alpha` (NEW-HIGH-05)
- `ensemble/swag.py` skips BatchNorm `running_*` buffers in posterior (NEW-HIGH-06)
- `algorithms/simex.py` validates `sigma_u` PSD via cholesky, warns if `|w|>5` (NEW-HIGH-07)
- `algorithms/tictac.py` clamps `log_k` to `[-6,6]` and warns if Hessian `>50M` elems (~200MB) (NEW-HIGH-08)
- `algorithms/adaptive_prior_vi.py` `VIDSRegressor` KL and NLL both `mean()` so KL not dominated by `P` (NEW-HIGH-09)
- `losses/__init__.py` adds explicit `__all__` (139 symbols) for `audit_api_coverage.py` soundness (TR-API-01 latent)

### Changed
- `BetaNLLLoss` numerics: previously `sum(dim=-1)` then `mean` over `B`; now `mean` over `B*D` elementwise (breaking for `D>1`, semver MINOR, documented)
- `README` badge `python 3.12 | 3.13 | 3.14 | 3.15` now matches `pyproject.toml:14 requires-python <3.16` (NEW-LOW-09)

### Added
- `CHANGELOG.md` (Keep a Changelog) and release-script `prepare_release.sh` now enforces bump

## [0.1.0] - 2026-08-26

- Initial PyPI release candidate (unreleased). Pre-1.0 blockers TR-COR-01…09 (all fixed in `6ce4b9e`).
- Features: 19 loss families, `PredictiveBatch`, `torchmetrics` metrics, conformal (split/CQR/CTI/weighted/Mondrian/CV+), `SemiConformalCalibrator`, ensembles (DeepEnsemble/BatchEnsemble/SWAG/Laplace/BNN/MC-Dropout), algorithms (IRLS/SIMEX/RC/LatentNN/TICTAC/VIDS), causal `dr_*`, PPI, test-time (BLR/OT/COSA/subspace/transport).

[Unreleased]: https://github.com/astroai/torchregress/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/astroai/torchregress/compare/v0.3.0b1...v0.3.0
[0.3.0b1]: https://github.com/astroai/torchregress/compare/v0.2.0...v0.3.0b1
[0.2.0]: https://github.com/astroai/torchregress/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/astroai/torchregress/releases/tag/v0.1.0
