# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
## [Unreleased]

**Algorithm freeze for 0.3.0 (from 2026-10-05):** no new methods or method changes
until 0.3.0 ships; only fixes, tests, docs and packaging. Methods to add or
remove are decided for 0.4.0.

### Changed
- Dependency floors raised to the newest releases resolvable from conda-forge:
  `torch>=2.13`, `numpy>=2.5`, `scipy>=1.18`, `torchmetrics>=1.9`; extras
  `matplotlib>=3.11`, `zuko>=1.6`, `scikit-learn>=1.9`, `pandas>=3.0`,
  `polars>=1.43`, `pyarrow>=25.0`, `pytest>=9.1`. Pre-commit hooks: ruff
  v0.16.10, pre-commit-hooks v6.0.0. Tested on torch 2.14.1 / numpy 2.5.3 /
  scipy 1.18.1 (PyPI) and the pixi lock (conda-forge torch 2.13.0).

### Fixed
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

[Unreleased]: https://github.com/astroai/torchregress/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/astroai/torchregress/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/astroai/torchregress/releases/tag/v0.1.0
