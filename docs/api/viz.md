# Visualization API

Complete reference for `torchregress.viz`. For the conceptual overview of
recommended plots and decision guidance, see
[Visualization methods](../methods/visualization.md).

> **Style.** All plot functions accept a `figsize`, `title`, and `return_figure`
> flag. They also accept an optional `ax` to embed in an existing figure. Most
> plots default to a notebook-sized `(10, 6)` figure and a bold, grid-on style.
> Call `tr.viz.set_style("whitegrid", "notebook")` once at the start of a
> notebook for consistent typography.

---

## Diagnostic plots (`viz.diagnostic`)

| Symbol | Description |
|:-------|:------------|
| `plot_reliability_diagram` | `(y_pred_quantiles, y_true, …)` — Predicted-quantile vs. observed-proportion reliability diagram with MACE annotation. |
| `plot_gaussian_reliability_diagram` | `(y_pred, y_pred_std, y_true, n_levels=10, …)` — Reliability for Gaussian (mean, std) predictions across `n_levels` confidence levels. |
| `plot_pit_histogram` | `(y_pred, y_pred_std, y_true, n_bins=20, …)` — PIT histogram for Gaussian predictions with KS-uniform statistic. |
| `plot_binned_metrics` | `(y_pred, y_pred_std, y_true, n_bins=5, metric="rmse", …)` — Per-target-bin RMSE / MAE / bias / NMAD / PICP / MPIW. |
| `plot_residuals` | `(y_pred, y_true, y_pred_std=None, y_true_std=None, censoring_indicator=None, …)` — Residual vs. predicted scatter; automatically standardises when stds are provided; marks censored points. |
| `plot_prediction_intervals` | `(y_pred, y_lower, y_upper, y_true=None, x=None, sorted_by_pred=False, …)` — Band of lower/upper with observed coverage annotation. |
| `plot_qq_plot` | `(y_pred, y_true, …)` — Q-Q plot for residual normality. |
| `plot_residual_histogram` | `(y_pred, y_true, n_bins=30, show_kde=True, …)` — Residual histogram with optional KDE + Normal overlay. |
| `plot_distribution_comparison` | `(predicted_samples, y_true, n_samples_to_show=5, …)` — Per-sample KDE / histogram of predictive distribution vs. true value. |
| `plot_calibration_curve` | `(y_pred_probs, y_true, n_bins=10, add_hist=True, …)` — Probability calibration curve (also works for binned regression). |
| `plot_target_density_error_overlap` | `(y_true, y_pred, n_bins=20, …)` — Twin-axis plot of target density and local MAE — great for imbalanced regression. |
| `plot_conditional_density_slices` | `(density_fn, x_slices, y_grid, y_true_slices=None, …)` — 1D conditional density slices for multimodal / flow / MDN models. |
| `plot_uncertainty_vs_error` | `(y_pred, y_pred_std, y_true, aleatoric_var=None, epistemic_var=None, …)` — Predicted σ vs. absolute error with Spearman ρ; optional decomposition stack plot. |
| `plot_censored_survival_curves` | `(predicted_survival, time_grid, observed_times, censoring_indicators, …)` — Predicted survival mean ± std vs. empirical Kaplan-Meier. |

---

## Monitoring plots (`viz.monitoring`)

| Symbol | Description |
|:-------|:------------|
| `plot_learning_curves` | `(train_history, val_history=None, metrics_to_plot=None, smoothing=0.0, log_scale=None, scientific_notation=True, …)` — Train / val metric curves in a grid; optional exponential smoothing; optional log scale. |
| `plot_validation_metrics` | `(epochs, metrics, n_cols=3, error_bars=None, …)` — Validation metric curves with optional error bars. |
| `plot_early_stopping` | `(train_losses, val_losses, patience=10, delta=0.0, …)` — Best-epoch and stop-epoch markers with patience-window shading. |
| `plot_lr_find_results` | `(learning_rates, losses, smoothing=0.05, suggestion_method="valley", …)` — LR-finder plot with `valley` / `steepest` / `minimum` suggestion. |

---

## Results visualisation (`viz.results`)

| Symbol | Description |
|:-------|:------------|
| `plot_performance_comparison` | `(metrics, highlight_best=True, plot_type="bar", sort_by=None, …)` — Bar / radar / heatmap comparison across models and metrics, with ↑/↓ direction indicators. |
| `plot_parameter_sensitivity` | `(parameter_values, metric_values, n_cols=2, …)` — Per-parameter effect on each metric; line or bar plot. |
| `plot_feature_importance` | `(feature_names, importance_values, importance_errors=None, horizontal=True, top_n=None, …)` — Bar chart of feature importances with optional error bars. |
| `plot_model_ensemble_contributions` | `(predictions, ensemble_prediction, model_weights=None, …)` — Per-model predictions + ensemble line. |
| `plot_simex_extrapolation` | `(lambda_values, simulated_values, extrapolator, …)` — SIMEX points + extrapolation curve + corrected estimate at `λ = -1`. |
| `plot_risk_coverage_curve` | `(y_true, y_pred, rejection_scores, …)` — Selective-prediction risk-coverage curve (model / random / oracle) with AURCC. |
| `plot_causal_uplift_qini` | `(uplift_scores, treatment, y_obs, …)` — Qini curve for uplift modelling with Qini area metric. |
| `plot_corner_plot` | `(samples, *, member_samples=None, param_names=None, true_vals=None, quantiles=(0.16, 0.5, 0.84), show_titles=True, show_uncertainty_decomposition=True, show_scatter=None, title=None, color="#4477AA", truth_color="#CC3311", member_colors=None, figsize=None, save_path=None, return_figure=False, row=0, n_samples=10000, dims=None, max_dims=8, levels=None, level_convention="mass", smooth=1.0, marginal="kde", interval="central", label=None, others=None, region=None, region_samples=4000, return_axes=False)` — Corner plot of draws `[S, d]`, draws `[n, S, d]` or a `JointPredictive` (row `row`): smoothed 1-D marginals with median and shaded interval, 2-D highest-density contours enclosing the stated mass of the draws (68%/95% by default, or `level_convention="sigma"`), true values, a `JointRegion` overlay, other predictives (`others=`), automatic target selection above `max_dims`. The legend states the contour convention; it sits in the empty upper triangle for three or more targets, below the grid for two and to the right for one. `return_axes=True` returns `(fig, axes[k, k])`. |

---

## Joint diagnostics (`viz.joint`)

| Symbol | Description |
|:-------|:------------|
| `plot_joint_diagnostics` | `(predictive, y, *, x=None, region=None, n_samples=256, preranks=None, target_names=None, max_targets=6, region_level=None, confidence=0.95, seed=0, title=None, figsize=None, save_path=None, return_figure=False, return_axes=False)` — One figure, read top to bottom: joint HPD calibration with its DKW band, predicted vs empirical correlation and their difference, pre-rank histograms with Holm-adjusted p-values, per-target PIT ECDF differences with a simultaneous band, and with `region` coverage by level and log-volumes, with `x` worst-slab coverage and the ERT conditional-coverage estimate. Every statistic comes from `torchregress.joint.diagnostics`; each panel has a one-line reading aid. `seed` fixes every random draw of the figure (predictive draws, HPD reference draws, region volumes, tie-breaking, slab split, ERT folds), so the figure is reproducible, and the caller's torch RNG state is left as it was. `return_axes=True` returns `(fig, {panel name: Axes})`. See [Joint predictive plots](../methods/visualization.md#4-joint-predictive-plots-corner-plots-and-diagnostics). |

---

## Utility helpers (`viz.utils`)

| Symbol | Description |
|:-------|:------------|
| `set_style` | `(style="whitegrid", context="notebook", font_scale=1.0, rc=None)` — One-shot matplotlib / seaborn style. |
| `create_grid_figure` | `(n_plots, figsize=(12, 8), nrows=None, ncols=None, sharex=False, sharey=False)` — Create a grid of subplots with auto-layout; returns `(Figure, list[Axes])`. |
| `add_identity_line` | `(ax, color="gray", linestyle="--", alpha=0.8, label=None)` — Add a `y = x` reference line. |
| `add_zero_line` | `(ax, axis="y", color="gray", linestyle="--", alpha=0.8, label=None)` — Add a zero reference line. |
| `save_figure` | `(fig, filename, directory="./figures", formats=["png", "pdf"], dpi=300, transparent=False, bbox_inches="tight")` — Save a figure in multiple formats. |
| `add_annotations` | `(ax, annotations, loc="upper right", fontsize=10, frameon=True, title=None)` — Add a text-box annotation of metric values. |
| `create_color_palette` | `(n_colors, palette_name="tab10", as_hex=False, as_cmap=False)` — Generate a colour palette (uses seaborn if available). |
| `enable_latex_rendering` | `(enable=True)` — Toggle LaTeX rendering (returns `bool` of success). |
| `is_lower_better` | `(metric_name)` — `True` when the metric name contains a term of `LOWER_IS_BETTER_TERMS` (`error`, `loss`, `rmse`, `nll`, `crps`, `interval_score`, `nmad`, `mad`, `mpiw`, `width`, `train_s`, `eval_s`, …); used for the ↑/↓ direction and best-model highlighting in `plot_performance_comparison`. |
| `format_metric_label` | `(metric_name, use_latex=True)` — Map common metric names to LaTeX strings (`RMSE`, `MAE`, `R²`, …). |

---

## Quick example

```python
import torch
from torchregress.viz import (
    set_style, plot_residuals, plot_pit_histogram, plot_learning_curves,
    plot_performance_comparison,
)
set_style("whitegrid", "notebook")

# Diagnostics
plot_residuals(y_pred, y_true, y_pred_std=y_pred_std, return_figure=False)
plot_pit_histogram(y_pred, y_pred_std, y_true, return_figure=False)

# Training monitoring
plot_learning_curves({"loss": train_losses, "mse": train_mses},
                      val_history={"loss": val_losses, "mse": val_mses},
                      metrics_to_plot=["loss", "mse"], smoothing=0.1)

# Comparison
plot_performance_comparison(
    {"DeepEns": {"rmse": 0.41, "mae": 0.32, "r2": 0.91},
     "MDN":    {"rmse": 0.45, "mae": 0.36, "r2": 0.89}},
    plot_type="bar", highlight_best=True, return_figure=False,
)
```

## Next steps

- [Visualization methods](../methods/visualization.md) — conceptual overview, decision guidance
- [Calibration metrics](../metrics/calibration.md) — PIT, ECE counterparts for diagnostic plots
- [Point metrics](../metrics/point.md) — RMSE, MAE counterparts for residual/scatter plots
