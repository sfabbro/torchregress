# 0.3.0 release audit

Audit of the highest-risk modules before the first PyPI release (release plan
Phase 2.2, batches 1–3), run 2026-10-06 at commit `5a334d3`.

**Evidence rule.** A finding counts only when a test reproduces it: every row
below had a test that failed on `5a334d3`, and that test now lives in
`tests/audit/` and passes. Tests are named `test_<ID>_<slug>`.

**Rubric** applied to every public export: maths against the cited paper,
float32/float64 numerics at extreme inputs, `gradcheck` in float64,
statistical guarantees checked by simulation (coverage inside the binomial
interval), the `reduction`/`mask`/`weights` contract of `losses/base.py`, and
dtype/device hygiene.

| Batch | Scope | Findings | Fix commit | Tests |
|:--|:--|--:|:--|:--|
| A | `losses`: Gaussian family, families, flows, quantile/expectile, robust, Poisson, censored, evidential, Wasserstein | 20 | `f6ebbbc` | `tests/audit/test_audit_losses_a.py` |
| B | `losses/conformal.py`, EIV, imbalanced, SLS, uncertain GT, `calibration/semicp.py` | 14 | `6c63316` | `tests/audit/test_audit_conformal_losses_b.py` |
| M | `metrics`, `calibration` (post-hoc) | 14 | `4f102ac` | `tests/audit/test_audit_metrics.py` |
| I | `inference`, `causal`, `ensemble`, `semi_supervised`, `prediction` | 13 | `39a8879` | `tests/audit/test_audit_inference.py` |

Severity: **High** = wrong numbers or a broken guarantee in normal use;
**Medium** = edge cases or an API-contract violation; **Low** = minor.

## Batch A: losses

| ID | Sev | Export | Finding |
|:--|:--|:--|:--|
| A-LOSS-016 | High | `EvidentialRegressionLoss` | NIG NLL used Ω = 2β instead of 2β(1+ν) (Amini et al. 2020, Eq. 8): trained the Student-t of μ, not of y, inconsistent with `predict_interval` |
| A-LOSS-003 | High | `SkewTLoss` | Incomplete-beta symmetry branch divided by the wrong argument; Student-t CDF wrong, NLL off by up to 0.7 nat |
| A-LOSS-002 | High | `BetaNLLLoss` | 1-D `[B]` inputs summed over the batch; `'mean'` was B× too large, mask and weights ignored |
| A-LOSS-012 | High | `CVaRLoss` | 1-D inputs reduced over all dims or raised |
| A-LOSS-004 | High | `SkewNormalNLLLoss` | log Φ clamped at float32 tiny: NLL saturates, zero gradient on the short-tail side |
| A-LOSS-017 | High | `GaussianWassersteinBoundLoss` | `eigh` backward gave NaN gradients for repeated eigenvalues (e.g. isotropic init) |
| A-LOSS-001 | Medium | `BaseLoss._reduce` | `[B, D]` mask with `[B]` weights raised |
| A-LOSS-019 | Medium | `BaseLoss._reduce` | `[B, D]` weights on per-sample losses were summed over D |
| A-LOSS-015 | Medium | `CensoredGaussianNLLLoss`, `AFTLoss` | censored terms saturated at 18.42 nats beyond ~5.6σ |
| A-LOSS-009 | Medium | `NormalizingFlowLoss`, `ContrastiveFlowLoss` | bypassed `_reduce`: weighted mean not normalised, masked `'none'` not zero-filled |
| A-LOSS-010 | Medium | `MultiExpectileLoss`, `ExpectileCrossoverLoss` | own mean: weights not normalised, masked samples in the denominator |
| A-LOSS-011 | Medium | `ExpectileCrossoverLoss` | silently sorted levels while columns stayed put |
| A-LOSS-007 | Medium | `BetaRegressionNLLLoss`, `SQRLoss` | `[B, 1]` targets broadcast to a `[B, B]` loss |
| A-LOSS-013 | Medium | `ZeroInflatedPoissonNLLLoss` | probability-space eps saturation; vanishing gradient on confident logits |
| A-LOSS-005 | Medium | `GEVNLLLoss` | float32 error of 0.17 nat just above the Gumbel threshold |
| A-LOSS-008 | Medium | family losses (`unconstrained_inputs=False`) | inverse softplus overflowed for scales above ~88 |
| A-LOSS-020 | Low | `FaithfulGaussianLoss` | cached mean-term flag went stale after `load_state_dict` |
| A-LOSS-006 | Low | `GEVNLLLoss` | NaN gradient from the unused Gumbel branch |
| A-LOSS-014 | Low | `NegativeBinomialNLLLoss` | θ in float32 for float64 inputs |
| A-LOSS-018 | Low | `losses.__all__` | `StudentTLoss`, `ExpectileCrossover` missing |

## Batch B: conformal prediction and losses

| ID | Sev | Export | Finding |
|:--|:--|:--|:--|
| B-CONF-002 | High | weighted conformal, `NonExchangeableConformalRegressor`, `MultivariateScoreConformal` | test-point weight fixed at 1 on the raw scale and never +inf: coverage 0.77 vs ≥ 0.9 under covariate shift (Tibshirani et al. 2019) |
| B-CONF-003 | High | `SemiConformalCalibrator` | returned the max score instead of +inf; float32 CDF picked the wrong order statistic |
| B-CONF-004 | High | CV+ / Jackknife+ | lower rank `ceil` instead of `floor(α(n+1))` (Barber et al. 2021); no ±inf endpoints |
| B-CONF-005 | High | `DensityConformal` | calibration and prediction used different densities: scores not exchangeable, coverage 0 on a bimodal target |
| B-CONF-006 | High | Mondrian conformal | negative group ids read the wrong threshold; R2C, CTI, DCP, SLSConformal used the first group's threshold for all points |
| B-LOSS-002 | High | `LatentMarginalizationLoss`, `InputNoiseAugmentationLoss` | marginalisation silently skipped (base loss returned a scalar), weights ignored |
| B-LOSS-001 | High | Functional/Structural EIV | Jacobian detached: no gradient through J Σx Jᵀ |
| B-LOSS-003 | High | `OrthogonalDistanceRegressionLoss` | absolute 1e-3 jitter (loss 0.55 vs 1.15 closed form); batch-mean unrolled steps; masked rows counted |
| B-LOSS-005 | High | `BalancedMSELoss` | an empty bin collapsed the loss by ~1e-8 |
| B-CONF-001 | Medium | `finite_sample_quantile` and users | clamped to the max score when ⌈(n+1)(1−α)⌉ > n: coverage below 1−α at small n |
| B-LOSS-004 | Medium | `SLSLoss` | `forward` mutated frontier state; the two passes used different β |
| B-LOSS-006 | Medium | FocalR, BalancedMSE, BinReweightedMSE, DensityWeighted, LDS, PropensityWeighted | weights bypassed `_reduce` |
| B-LOSS-007 | Medium | `DensityWeightedLoss` | normalised over the batch without `sample_indices`; NaN targets poisoned the loss |
| B-LOSS-008 | Low | pseudo-label losses | NaN placeholder targets gave NaN |

## Batch M: metrics and calibration

| ID | Sev | Export | Finding |
|:--|:--|:--|:--|
| M-MET-001 | High | `distribution_metrics_report` | CRPS from 7 sample quantiles, 3–4% low |
| M-MET-006 | High | `median_absolute_error` (multi-output) | median of row means instead of mean of per-output medians (sklearn) |
| M-MET-002 | Medium | quantile PIT | out-of-range targets got PIT exactly 0/1: KS and χ² invalid for calibrated forecasts |
| M-MET-003 | Medium | `energy_score`, `kernel_density_score` | float32 `cdist` matmul mode lost up to 16% |
| M-MET-004 | Medium | `crps_from_samples`, `vario_score` | NaN for a single sample |
| M-MET-005 | Medium | `VarioScore` | `higher_is_better` had the wrong orientation |
| M-MET-008 | Medium | trimmed MSE | asymmetric trimming |
| M-MET-009 | Medium | `EnsembleIntervalMetrics` | `reset()` left child metrics stale |
| M-MET-010 | Medium | `gaussian_nll`, ensemble NLL/intervals | absolute variance floors distorted small-scale targets |
| M-MET-011 | Medium | Mahalanobis, TAC | absolute covariance jitter, not scale-equivariant |
| M-CAL-001 | Medium | `VarianceTemperatureScaler` | undocumented T ∈ [0.05, 20] clamp |
| M-CAL-002 | Medium | `VarianceTemperatureScaler` | default fit stopped 5% short of the MLE |
| M-CAL-003 | Medium | `IsotonicMeanCalibrator` | not isotonic regression (ties not pooled, centroid interpolation) |
| M-MET-007 | Low | MedAE, MAD, NMAD | lower median for even n |

## Batch I: inference, causal, ensembles

| ID | Sev | Export | Finding |
|:--|:--|:--|:--|
| I-INF-001 | High | `ppi_quantile_ci` | returned Q(f_u) + median residual, not the PPI quantile: 0% coverage (Angelopoulos et al. 2023) |
| I-CAU-001 | High | `dr_ate`/`dr_cate` (`fold_bootstrap`) | bootstrapped K fold means: SE 0.56× at K = 2, coverage 0.56–0.65 |
| I-ENS-001 | High | heteroscedastic/batch/packed/SWAG ensembles | predicted log-variance clamped to [−8, 6] vs the training range [log 1e-6, 30] |
| I-INF-002 | Medium | PPI functions | float64 downcast to float32 (SE 0 at large offsets) |
| I-CAU-002 | Medium | DR estimators | float32 downcast |
| I-CAU-003 | Medium | DR estimators | silent NaN when trimming removed every unit |
| I-ENS-002 | Medium | ensemble `predict` | ran in train mode after `fit` (dropout on, BatchNorm stats overwritten) |
| I-INF-004 | Medium | PPI bootstrap | `[n_boot, N]` index matrix: 15 GB at N = 1e6 |
| I-INF-003 | Medium | semi-supervised trust weights | documented options silently ignored or raised |
| I-ENS-003 | Medium | `bars_to_density_grid` | density outside the bin edges |
| I-INF-005 | Low | `orthogonal_partially_linear` | identification check not shift-invariant |
| I-ENS-004 | Low | ensembles with `base_seed` | reseeded the global RNG |
| I-ENS-005 | Low | ensemble docs | documented removed APIs |

## Checked and found correct

Reference implementations matched on: `GaussianNLLLoss`, `GaussianCRPSLoss`,
multivariate and low-rank Gaussian, Student-t, MDN, Johnson SU, beta
regression, GEV away from ξ = 0, Barron, Tweedie, Poisson, discrete
Wasserstein, quantile/expectile losses; split conformal, CQR, UACQR, Monte
Carlo, multi-target, Mahalanobis-score, R2C and CTI coverage; `crps_gaussian`,
`crps_from_samples`, `interval_score`, pinball, KS, Gaussian Wasserstein,
ordinal and point metrics; `ppi_mean_ci`, PPI++, calibrated PPI, PPI OLS,
DML (0.951 coverage at n = 500), AIPW; ensemble variance decomposition,
`BatchEnsembleLinear`, `VariationalLinear`, `MCDropoutWrapper`.

`tests/metrics/test_reference_parity.py` now checks every metrics export
against scoringrules, properscoring, scipy, scikit-learn or a closed form,
and fails when a new export has neither a test nor a stated reason.

## Not yet audited

Batches 4–5 of the plan: `algorithms`, `test_time`, `constraints` (spot
checked), `utils`, `viz`. Known item: `test_time.transport.ppi_target_ci`
still casts its inputs to float32 before calling the PPI functions.
