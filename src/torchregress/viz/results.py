"""
Results visualization tools for regression models.

This module provides visualization utilities for presenting
and comparing regression model results.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - hit only without the viz extra
    raise ImportError(
        "torchregress.viz requires matplotlib; pip install 'torchregress[viz]'"
    ) from exc

import math
import textwrap
from collections.abc import Sequence

import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, MaxNLocator, ScalarFormatter
from matplotlib.transforms import blended_transform_factory

from torchregress.joint.predictive import JointPredictive
from torchregress.joint.regions import JointRegion
from torchregress.viz._corner import (
    OVERLAY_STYLES,
    REGION_COLOR,
    REGION_STYLE,
    TOL_BRIGHT,
    RegionLevel,
    as_numpy,
    contour_masses,
    draws_of,
    ellipse_points,
    format_estimate,
    hpd_mask_1d,
    kde_1d,
    kde_2d,
    mass_thresholds,
    most_dependent_dims,
    plot_range,
    region_interval,
    region_levels,
    split_unit,
    subsample,
)

# Import visualization utilities
from torchregress.viz.utils import create_color_palette, is_lower_better


def plot_performance_comparison(
    metrics: Dict[str, Dict[str, Union[float, np.ndarray]]],
    highlight_best: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Model Performance Comparison",
    metrics_to_include: Optional[List[str]] = None,
    higher_is_better: Optional[Dict[str, bool]] = None,
    sort_by: Optional[str] = None,
    color_palette: str = "tab10",
    plot_type: str = "bar",
    return_figure: bool = False,
    ax: Optional[plt.Axes] = None,
) -> Optional[Figure]:
    """
    Create a comparison plot for multiple models across different metrics.

    Args:
        metrics: Dictionary mapping model names to dictionaries of metrics
        highlight_best: Whether to highlight the best model for each metric
        figsize: Figure size (width, height) when creating a new figure
        title: Plot title
        metrics_to_include: List of metrics to include (defaults to all common metrics)
        higher_is_better: Dictionary specifying for each metric if higher values are better
        sort_by: Metric name to sort models by
        color_palette: Name of color palette to use
        plot_type: Type of plot ('bar', 'radar', or 'heatmap')
        return_figure: If True, return figure object instead of displaying
        ax: Optional matplotlib axes for plotting (only used for bar plot)

    Returns:
        If return_figure=True, returns matplotlib Figure object
    """
    # Extract common metrics across all models if not specified
    if metrics_to_include is None:
        metrics_to_include = list(
            set.intersection(*[set(model_metrics.keys()) for model_metrics in metrics.values()])
        )
        metrics_to_include = sorted(metrics_to_include)

    # Filter to only include metrics that exist for all models
    metrics_to_include = [
        m
        for m in metrics_to_include
        if all(m in model_metrics for model_metrics in metrics.values())
    ]

    if not metrics_to_include:
        raise ValueError("No common metrics found across all models")

    # Determine whether higher or lower is better for each metric
    if higher_is_better is None:
        higher_is_better = {}
        for metric in metrics_to_include:
            # TR-VIZ-01: single shared direction registry instead of per-file term lists
            higher_is_better[metric] = not is_lower_better(metric)

    # Build numpy arrays for easier manipulation
    model_names = list(metrics.keys())
    n_models = len(model_names)
    n_metrics = len(metrics_to_include)

    # Build values array (n_models, n_metrics)
    values_arr = np.zeros((n_models, n_metrics))
    for j, metric in enumerate(metrics_to_include):
        for i, model in enumerate(model_names):
            values_arr[i, j] = float(metrics[model][metric])

    # Sort by specified metric if requested
    if sort_by is not None and sort_by in metrics_to_include:
        sort_col = metrics_to_include.index(sort_by)
        ascending = not higher_is_better.get(sort_by, True)
        order = np.argsort(values_arr[:, sort_col])
        if not ascending:
            order = order[::-1]
        values_arr = values_arr[order]
        model_names = [model_names[i] for i in order]

    # Store metric names and values as list-based structures (was DataFrame)
    metric_names_list = list(metrics_to_include)

    # Determine best model for each metric if highlighting is requested
    best_values = {}
    if highlight_best:
        for j, metric in enumerate(metrics_to_include):
            if higher_is_better.get(metric, True):
                best_values[metric] = float(values_arr[:, j].max())
            else:
                best_values[metric] = float(values_arr[:, j].min())

    # Determine colors for models
    colors = cast(List[Any], create_color_palette(len(model_names), palette_name=color_palette))

    # Create plot based on specified type
    if plot_type == "bar":
        return _plot_performance_bar(
            model_names,
            metric_names_list,
            values_arr,
            best_values,
            figsize,
            title,
            higher_is_better,
            colors,
            return_figure,
            ax,
        )
    elif plot_type == "radar":
        return _plot_performance_radar(
            model_names,
            metric_names_list,
            values_arr,
            best_values,
            figsize,
            title,
            higher_is_better,
            colors,
            return_figure,
        )
    elif plot_type == "heatmap":
        return _plot_performance_heatmap(
            model_names,
            metric_names_list,
            values_arr,
            best_values,
            figsize,
            title,
            higher_is_better,
            return_figure,
        )
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")


def _add_performance_bars(
    ax: plt.Axes,
    model_names: list[str],
    metric_names: list[str],
    values_arr: np.ndarray,
    best_values: Dict[str, float],
    colors: List,
    bar_width: float,
    indices: np.ndarray,
) -> None:
    """Helper to add bars and value annotations."""
    for i, model_name in enumerate(model_names):
        values = values_arr[i]
        bars = ax.bar(
            indices + i * bar_width,
            values,
            bar_width,
            label=model_name,
            color=colors[i % len(colors)],
        )

        # Highlight best model for each metric
        if best_values:
            for j, metric in enumerate(metric_names):
                if np.isclose(values_arr[i, j], best_values[metric]):
                    bars[j].set_edgecolor("black")
                    bars[j].set_linewidth(2)

        # Add value labels on top of bars
        for j, v in enumerate(values):
            # Format value based on magnitude
            if abs(v) < 0.01 or abs(v) >= 1000:
                value_str = f"{v:.2e}"
            elif abs(v) < 0.1:
                value_str = f"{v:.3f}"
            else:
                value_str = f"{v:.2f}"

            ax.text(
                indices[j] + i * bar_width,
                v * 1.01,
                value_str,
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=45,
            )


def _format_performance_axes(
    ax: plt.Axes,
    metric_names: list[str],
    title: str,
    indices: np.ndarray,
    bar_width: float,
    n_models: int,
    higher_is_better: Dict[str, bool],
) -> None:
    """Helper to format axes and labels."""
    # Add labels, title and legend
    ax.set_xlabel("Metric", fontweight="bold")
    ax.set_ylabel("Value", fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(indices + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(metric_names, rotation=45, ha="right", fontweight="bold")

    # Skip LaTeX rendering if not available or fails
    # Use rc_context to avoid globally mutating rcParams
    try:
        import shutil

        if shutil.which("latex") is not None:
            with plt.rc_context(
                {
                    "text.usetex": True,
                    "font.family": "serif",
                    "font.serif": ["Computer Modern Roman"],
                }
            ):
                # Replace common metric names with LaTeX
                metric_labels = []
                for metric in metric_names:
                    if "rmse" in metric.lower():
                        metric_labels.append(r"$\mathrm{RMSE}$")
                    elif "mae" in metric.lower():
                        metric_labels.append(r"$\mathrm{MAE}$")
                    elif "r2" in metric.lower():
                        metric_labels.append(r"$R^2$")
                    else:
                        metric_labels.append(metric)
                ax.set_xticklabels(metric_labels, rotation=45, ha="right")
    except RuntimeError:
        pass  # Skip LaTeX rendering if it fails at render time
    # Add grid
    ax.grid(True, axis="y", alpha=0.3)

    # Add annotations for which direction is better
    for i, metric in enumerate(metric_names):
        direction = "↑" if higher_is_better.get(metric, True) else "↓"
        ax.annotate(
            direction,
            xy=(i, 0),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=12,
        )


def _plot_performance_bar(
    model_names: list[str],
    metric_names: list[str],
    values_arr: np.ndarray,
    best_values: Dict[str, float],
    figsize: Tuple[int, int],
    title: str,
    higher_is_better: Dict[str, bool],
    colors: List,
    return_figure: bool,
    ax: Optional[plt.Axes],
) -> Optional[Figure]:
    """Helper function for bar plot comparison."""
    # Create plot if no axes provided
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = cast(Figure, ax.figure)

    # Number of models and metrics
    n_models = len(model_names)
    n_metrics = len(metric_names)

    # Set width of bars
    bar_width = 0.8 / n_models

    # Set positions of bars on x-axis
    indices = np.arange(n_metrics)

    _add_performance_bars(
        ax, model_names, metric_names, values_arr, best_values, colors, bar_width, indices
    )
    _format_performance_axes(
        ax, metric_names, title, indices, bar_width, n_models, higher_is_better
    )

    fig.tight_layout()

    if return_figure:
        return fig
    elif created_fig:  # Only show if we created the figure here
        plt.show()
        plt.close(fig)  # TR-VIZ-05: close figures we created

    return None


def _plot_performance_radar(
    model_names: list[str],
    metric_names: list[str],
    values_arr: np.ndarray,
    best_values: Dict[str, float],
    figsize: Tuple[int, int],
    title: str,
    higher_is_better: Dict[str, bool],
    colors: List,
    return_figure: bool,
) -> Optional[Figure]:
    """Helper function for radar plot comparison."""
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, polar=True)

    # Number of metrics (variables)
    n_metrics = len(metric_names)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()

    # Add the last angle to close the polygon
    angles += angles[:1]

    # Normalize metrics to range [0, 1] for radar plot
    norm_values = np.zeros_like(values_arr)
    for j in range(n_metrics):
        col = values_arr[:, j]
        min_val = float(col.min())
        max_val = float(col.max())
        metric = metric_names[j]
        if higher_is_better.get(metric, True):
            norm_values[:, j] = (col - min_val) / (max_val - min_val) if max_val > min_val else 0.5
        else:
            norm_values[:, j] = (
                1.0 - (col - min_val) / (max_val - min_val) if max_val > min_val else 0.5
            )

    # Plot each model
    for i, model_name in enumerate(model_names):
        values = norm_values[i].tolist()
        values += values[:1]  # Close the polygon

        ax.plot(angles, values, color=colors[i % len(colors)], linewidth=2, label=model_name)
        ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.1)

    # Set labels and ticks
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [
            f"{metric}\n({'↑' if higher_is_better.get(metric, True) else '↓'})"
            for metric in metric_names
        ]
    )

    # Set y-ticks
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], color="grey", size=8)
    cast(Any, ax).set_rlabel_position(0)

    # Add title and legend
    plt.title(title, size=14, y=1.1)
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

    if return_figure:
        return fig
    else:
        plt.show()
        plt.close(fig)  # TR-VIZ-05: close figures we created
        return None


def _plot_performance_heatmap(
    model_names: list[str],
    metric_names: list[str],
    values_arr: np.ndarray,
    best_values: Dict[str, float],
    figsize: Tuple[int, int],
    title: str,
    higher_is_better: Dict[str, bool],
    return_figure: bool,
) -> Optional[Figure]:
    """Helper function for heatmap plot comparison."""
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Normalize metrics to range [0, 1] for heatmap
    n_metrics = len(metric_names)
    norm_values = np.zeros_like(values_arr)
    for j in range(n_metrics):
        col = values_arr[:, j]
        min_val = float(col.min())
        max_val = float(col.max())
        metric = metric_names[j]
        if higher_is_better.get(metric, True):
            norm_values[:, j] = (col - min_val) / (max_val - min_val) if max_val > min_val else 0.5
        else:
            norm_values[:, j] = (
                1.0 - (col - min_val) / (max_val - min_val) if max_val > min_val else 0.5
            )

    # Create heatmap
    im = ax.imshow(norm_values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Performance (normalized)", rotation=-90, va="bottom")

    # Set ticks and labels
    ax.set_xticks(np.arange(n_metrics))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels(
        [
            f"{metric}\n({'↑' if higher_is_better.get(metric, True) else '↓'})"
            for metric in metric_names
        ]
    )
    ax.set_yticklabels(model_names)

    # Rotate x-tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data and add annotations with actual values
    for i in range(len(model_names)):
        for j in range(n_metrics):
            metric_value = values_arr[i, j]
            is_best = best_values and np.isclose(metric_value, best_values[metric_names[j]])

            # Format text based on value type
            if isinstance(metric_value, float):
                text = f"{metric_value:.4f}"
            else:
                text = str(metric_value)

            # Add bold font for best values
            if is_best:
                text = f"$\\bf{{{text}}}$"

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="black" if norm_values[i, j] > 0.5 else "white",
            )

    # Add title
    ax.set_title(title)

    fig.tight_layout()

    if return_figure:
        return fig
    else:
        plt.show()
        plt.close(fig)  # TR-VIZ-05: close figures we created
        return None


def plot_parameter_sensitivity(
    parameter_values: Dict[str, List[Union[float, int, str]]],
    metric_values: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Parameter Sensitivity Analysis",
    n_cols: int = 2,
    higher_is_better: Optional[Dict[str, bool]] = None,
    highlight_best: bool = True,
    plot_type: str = "line",
    color_palette: str = "viridis",
    return_figure: bool = False,
) -> Optional[Figure]:
    """
    Create a parameter sensitivity plot to analyze how parameters affect model performance.

    Args:
        parameter_values: Dictionary mapping parameter names to lists of values tested
        metric_values: Dictionary mapping metric names to lists of resulting values
        figsize: Figure size (width, height)
        title: Plot title
        n_cols: Number of columns in the grid
        higher_is_better: Dictionary specifying for each metric if higher values are better
        highlight_best: Whether to highlight the best parameter value for each metric
        plot_type: Type of plot ('line' or 'bar')
        color_palette: Name of color palette to use
        return_figure: If True, return figure object instead of displaying

    Returns:
        If return_figure=True, returns matplotlib Figure object
    """
    # Determine if higher or lower is better for each metric
    if higher_is_better is None:
        higher_is_better = {}
        for metric in metric_values:
            # TR-VIZ-01: single shared direction registry instead of per-file term lists
            higher_is_better[metric] = not is_lower_better(metric)

    # Create grid of subplots
    n_metrics = len(metric_values)
    n_rows = int(np.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Handle single subplot case
    if n_metrics == 1:
        axes = np.array([axes])

    # Ensure axes is always a 1D array
    axes = axes.flatten()

    # Plot each parameter's effect on each metric
    for i, (metric, values) in enumerate(metric_values.items()):
        ax = axes[i]

        for k, (param_name, param_values) in enumerate(parameter_values.items()):
            # Skip if number of parameter values doesn't match metric values
            if len(param_values) != len(values):
                continue

            # Sort values by parameter if parameter is numeric
            if all(isinstance(p, (int, float)) for p in param_values):
                sorted_order = [int(j) for j in np.argsort(param_values)]
                sorted_params = [param_values[j] for j in sorted_order]
                sorted_values = [values[j] for j in sorted_order]
            else:
                sorted_params = param_values
                sorted_values = values

            # Determine best value
            if higher_is_better.get(metric, True):
                best_idx = np.argmax(sorted_values)
            else:
                best_idx = np.argmin(sorted_values)

            # Get colors
            if plot_type == "line":
                # For line plot, use a single color
                pass  # Use default color cycle
            else:
                # For bar plot, use a color gradient
                cmap = plt.get_cmap(color_palette)
                colors = [cmap(i / len(sorted_params)) for i in range(len(sorted_params))]

            # Create plot based on type
            if plot_type == "line":
                ax.plot(sorted_params, sorted_values, "o-", label=param_name)

                # Highlight best value
                if highlight_best:
                    ax.scatter(
                        [sorted_params[best_idx]],
                        [sorted_values[best_idx]],
                        color="red",
                        s=100,
                        zorder=10,
                        label=f"Best: {sorted_params[best_idx]}",
                    )
            else:  # bar plot
                # TR-VIZ-06..21: group bars per parameter instead of overlaying them
                x_pos = np.arange(len(sorted_params))
                n_params = max(len(parameter_values), 1)
                bar_width = 0.8 / n_params
                offset = (k - (n_params - 1) / 2.0) * bar_width
                bars = ax.bar(
                    x_pos + offset, sorted_values, width=bar_width, color=colors, label=param_name
                )
                ax.set_xticks(x_pos)
                ax.set_xticklabels([str(p) for p in sorted_params], rotation=45)

                # Highlight best value
                if highlight_best:
                    bars[best_idx].set_edgecolor("red")
                    bars[best_idx].set_linewidth(2)
                    bars[best_idx].set_hatch("//")
                if len(parameter_values) > 1:
                    ax.legend(loc="best")

        # Set title and labels
        is_higher_better = higher_is_better.get(metric, True)
        direction = "↑" if is_higher_better else "↓"
        ax.set_title(f"{metric} {direction}")

        if plot_type == "line":
            ax.set_xlabel("Parameter Value")
            ax.set_ylabel("Metric Value")
            ax.legend(loc="best")
        else:  # bar plot
            # TR-VIZ-06..21: label every parameter iterated on this subplot,
            # not just whichever happened to be last in the loop.
            ax.set_xlabel(" / ".join(parameter_values.keys()))
            ax.set_ylabel("Metric Value")

        # Add grid
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)

    # Set overall title
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)

    if return_figure:
        return fig
    else:
        plt.show()
        plt.close(fig)  # TR-VIZ-05: close figures we created
        return None


def _normalize_importance_inputs(
    importance_values: Union[np.ndarray, List[float], torch.Tensor],
    importance_errors: Optional[Union[np.ndarray, List[float], torch.Tensor]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Helper to convert importance inputs to numpy arrays."""
    if isinstance(importance_values, torch.Tensor):
        importance_values = importance_values.detach().cpu().numpy()
    importance_values_np = np.array(importance_values)

    importance_errors_np = None
    if importance_errors is not None:
        if isinstance(importance_errors, torch.Tensor):
            importance_errors = importance_errors.detach().cpu().numpy()
        importance_errors_np = np.array(importance_errors)

    return importance_values_np, importance_errors_np


def _sort_and_limit_features(
    feature_names: List[str],
    importance_values: np.ndarray,
    importance_errors: Optional[np.ndarray],
    sort_values: bool,
    horizontal: bool,
    top_n: Optional[int],
) -> Tuple[List[str], np.ndarray, Optional[np.ndarray]]:
    """Helper to sort and limit feature importance values."""
    if sort_values:
        idx = np.argsort(importance_values)
        if not horizontal:  # For vertical, we want descending order
            idx = idx[::-1]

        feature_names = [feature_names[i] for i in idx]
        importance_values = importance_values[idx]

        if importance_errors is not None:
            importance_errors = importance_errors[idx]

    if top_n is not None and top_n < len(feature_names):
        if horizontal:  # For horizontal, we want highest values at the top
            feature_names = feature_names[-top_n:]
            importance_values = importance_values[-top_n:]
            if importance_errors is not None:
                importance_errors = importance_errors[-top_n:]
        else:  # For vertical, we've already sorted in descending order
            feature_names = feature_names[:top_n]
            importance_values = importance_values[:top_n]
            if importance_errors is not None:
                importance_errors = importance_errors[:top_n]

    return feature_names, importance_values, importance_errors


def _plot_importance_bars(
    ax: plt.Axes,
    feature_names: List[str],
    importance_values: np.ndarray,
    importance_errors: Optional[np.ndarray],
    horizontal: bool,
    color: str,
    error_color: str,
    show_values: bool,
) -> None:
    """Helper to plot the bars for feature importance."""
    y_pos = np.arange(len(feature_names))

    if horizontal:
        bars = ax.barh(
            y_pos,
            importance_values,
            color=color,
            xerr=importance_errors,
            error_kw=dict(ecolor=error_color),
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.invert_yaxis()  # Highest values at the top

        if show_values:
            for bar in bars:
                width = bar.get_width()
                label_position = max(width * 1.05, 0.01)
                ax.text(
                    label_position, bar.get_y() + bar.get_height() / 2, f"{width:.3f}", va="center"
                )
    else:
        bars = ax.bar(
            y_pos,
            importance_values,
            color=color,
            yerr=importance_errors,
            error_kw=dict(ecolor=error_color),
        )
        ax.set_xticks(y_pos)
        ax.set_xticklabels(feature_names, rotation=45, ha="right")

        if show_values:
            for bar in bars:
                height = bar.get_height()
                label_position = height * 1.05
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    label_position,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                )


def plot_feature_importance(
    feature_names: List[str],
    importance_values: Union[np.ndarray, List[float], torch.Tensor],
    importance_errors: Optional[Union[np.ndarray, List[float], torch.Tensor]] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Feature Importance",
    color: str = "skyblue",
    error_color: str = "black",
    sort_values: bool = True,
    horizontal: bool = True,
    top_n: Optional[int] = None,
    return_figure: bool = False,
    show_values: bool = True,
    ax: Optional[plt.Axes] = None,
) -> Optional[Figure]:
    """
    Create a feature importance plot to visualize which features are most important.

    Args:
        feature_names: Names of features
        importance_values: Importance score for each feature
        importance_errors: Optional error/uncertainty for importance values
        figsize: Figure size (width, height) when creating a new figure
        title: Plot title
        color: Bar color
        error_color: Error bar color
        sort_values: Whether to sort features by importance
        horizontal: Whether to create a horizontal bar chart
        top_n: Optionally limit to top N features
        return_figure: If True, return figure object instead of displaying
        show_values: Whether to display importance values on bars
        ax: Optional matplotlib axes for plotting

    Returns:
        If return_figure=True, returns matplotlib Figure object
    """
    # Convert to numpy array for easier manipulation
    importance_values_np, importance_errors_np = _normalize_importance_inputs(
        importance_values, importance_errors
    )

    # Sort and limit features
    feature_names, importance_values_np, importance_errors_np = _sort_and_limit_features(
        feature_names, importance_values_np, importance_errors_np, sort_values, horizontal, top_n
    )

    # Create plot if no axes provided
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = cast(Figure, ax.figure)

    # Create bars
    _plot_importance_bars(
        ax,
        feature_names,
        importance_values_np,
        importance_errors_np,
        horizontal,
        color,
        error_color,
        show_values,
    )

    # Set labels and title
    if horizontal:
        ax.set_xlabel("Importance")
    else:
        ax.set_ylabel("Importance")

    ax.set_title(title)

    # Add grid
    if horizontal:
        ax.grid(True, axis="x", alpha=0.3)
    else:
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()

    if return_figure:
        return fig
    elif created_fig:  # Only show if we created the figure here
        plt.show()
        plt.close(fig)  # TR-VIZ-05: close figures we created

    return None


def plot_model_ensemble_contributions(
    predictions: Dict[str, np.ndarray],
    ensemble_prediction: np.ndarray,
    model_weights: Optional[Dict[str, float]] = None,
    x: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Model Ensemble Contributions",
    xlabel: str = "Sample Index",
    ylabel: str = "Prediction",
    color_palette: str = "tab10",
    return_figure: bool = False,
    ax: Optional[plt.Axes] = None,
) -> Optional[Figure]:
    """
    Visualize how different models contribute to an ensemble prediction.

    Args:
        predictions: Dictionary mapping model names to their predictions
        ensemble_prediction: The final ensemble prediction
        model_weights: Optional dictionary of weights for each model
        x: Optional x-coordinates (e.g., time points or feature values)
        figsize: Figure size (width, height) when creating a new figure
        title: Plot title
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        color_palette: Name of color palette to use
        return_figure: If True, return figure object instead of displaying
        ax: Optional matplotlib axes for plotting

    Returns:
        If return_figure=True, returns matplotlib Figure object
    """
    # Create plot if no axes provided
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = cast(Figure, ax.figure)

    # Create x values if not provided
    if x is None:
        x = np.arange(len(ensemble_prediction))

    # Get colors
    colors = cast(List[Any], create_color_palette(len(predictions), palette_name=color_palette))

    # Plot individual model predictions
    for i, (model_name, preds) in enumerate(predictions.items()):
        weight = model_weights.get(model_name, 1.0) if model_weights else 1.0
        label = f"{model_name} (w={weight:.2f})" if model_weights else model_name
        ax.plot(x, preds, color=colors[i], alpha=0.5, linestyle="--", label=label)

    # Plot ensemble prediction
    ax.plot(x, ensemble_prediction, color="black", linewidth=2, label="Ensemble")

    # Add labels, title and legend
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")

    # Add grid
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if return_figure:
        return fig
    elif created_fig:  # Only show if we created the figure here
        plt.show()
        plt.close(fig)  # TR-VIZ-05: close figures we created

    return None


def plot_simex_extrapolation(
    lambda_values: Union[torch.Tensor, np.ndarray],
    simulated_values: Union[torch.Tensor, np.ndarray],
    extrapolator: Any,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "SIMEX Extrapolation Diagnostics",
    return_figure: bool = False,
    ax: Optional[plt.Axes] = None,
) -> Optional[Figure]:
    """
    Plots SIMEX simulation points and the fitted extrapolation curve back to lambda = -1.
    Useful for diagnosing measurement error (Error-in-Variables) correction.

    Args:
        lambda_values: [M] array of added noise multipliers (typically >= 0).
        simulated_values: [M] or [M, P] array of simulated metrics/coefficient values.
        extrapolator: A callable function or fitted extrapolator model
            (e.g. poly1d) that takes lambda and returns prediction.
        figsize: Figure size (width, height)
        title: Plot title
        return_figure: If True, return figure object
        ax: Optional matplotlib axes
    """
    from torchregress.metrics.utils import convert_to_tensor

    lambda_values = convert_to_tensor(lambda_values).detach().cpu().numpy().flatten()
    simulated_values = convert_to_tensor(simulated_values).detach().cpu().numpy()

    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = cast(Figure, ax.figure)

    # Plot simulation points
    if len(simulated_values.shape) > 1 and simulated_values.shape[1] > 1:
        for j in range(simulated_values.shape[1]):
            ax.scatter(
                lambda_values, simulated_values[:, j], alpha=0.8, marker="o", label=f"Param {j}"
            )
    else:
        ax.scatter(
            lambda_values,
            simulated_values.flatten(),
            color="navy",
            s=50,
            zorder=5,
            label="Simulated Metrics",
        )

    # Extrapolation curve
    lambdas_fine = np.linspace(-1.0, max(lambda_values) * 1.1, 200)

    try:
        if hasattr(extrapolator, "predict"):
            pred_fine = extrapolator.predict(lambdas_fine.reshape(-1, 1)).flatten()
            pred_minus_1 = extrapolator.predict(np.array([[-1.0]])).item()
        elif callable(extrapolator):
            pred_fine = extrapolator(lambdas_fine).flatten()
            pred_minus_1 = extrapolator(-1.0)
        else:
            pred_fine = np.polyval(extrapolator, lambdas_fine).flatten()
            pred_minus_1 = np.polyval(extrapolator, -1.0)

        ax.plot(
            lambdas_fine,
            pred_fine,
            color="crimson",
            linestyle="-",
            linewidth=2.5,
            label="Fitted Extrapolation",
        )
        ax.scatter(
            [-1.0],
            [pred_minus_1],
            color="forestgreen",
            marker="*",
            s=250,
            zorder=6,
            label=f"Corrected Estimate: {pred_minus_1:.4f}",
        )
    except Exception as e:
        print(f"Warning: Could not plot extrapolation curve: {e}")

    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5, label="Naive / Observed Noise")
    ax.axvline(x=-1, color="forestgreen", linestyle="--", alpha=0.5, label="Theoretical Zero Noise")

    ax.set_xlabel("Noise Multiplier (λ)", fontweight="bold")
    ax.set_ylabel("Parameter Estimate / Metric Value", fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    if return_figure:
        return fig
    elif created_fig:
        fig.tight_layout()
        plt.show()
        plt.close(fig)  # TR-VIZ-05: close figures we created
    return None


def plot_risk_coverage_curve(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    rejection_scores: Union[torch.Tensor, np.ndarray],
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Risk-Coverage Selective Prediction Curve",
    return_figure: bool = False,
    ax: Optional[plt.Axes] = None,
) -> Optional[Figure]:
    """
    Plots validation risk (MSE/MAE) vs. coverage (percentage of remaining dataset kept).
    Allows evaluating out-of-distribution scores or uncertainty estimators for selective prediction.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        rejection_scores: Rejection score for each sample (higher score = reject first).
                          Typically predicted uncertainty std, typicality, or entropy score.
        figsize: Figure size (width, height)
        title: Plot title
        return_figure: If True, return figure
        ax: Optional axes
    """
    from torchregress.metrics.utils import convert_to_tensor

    y_true = convert_to_tensor(y_true).detach().cpu().numpy().flatten()
    y_pred = convert_to_tensor(y_pred).detach().cpu().numpy().flatten()
    rejection_scores = convert_to_tensor(rejection_scores).detach().cpu().numpy().flatten()

    errors = (y_true - y_pred) ** 2

    # Sort samples by rejection scores ascending
    sort_idx = np.argsort(rejection_scores)
    sorted_errors = errors[sort_idx]

    n_samples = len(y_true)
    coverages = np.linspace(1 / n_samples, 1.0, n_samples)

    # Cumulative mean risk for model-based rejection
    model_risk = np.cumsum(sorted_errors) / np.arange(1, n_samples + 1)

    # Random rejection (constant expected risk)
    random_risk = np.full(n_samples, np.mean(errors))

    # Oracle rejection (sorting by actual error)
    oracle_idx = np.argsort(errors)
    oracle_sorted_errors = errors[oracle_idx]
    oracle_risk = np.cumsum(oracle_sorted_errors) / np.arange(1, n_samples + 1)

    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = cast(Figure, ax.figure)

    ax.plot(coverages, model_risk, color="navy", linewidth=2.5, label="Model Rejection Policy")
    ax.plot(
        coverages,
        random_risk,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label="Random Rejection",
    )
    ax.plot(
        coverages,
        oracle_risk,
        color="forestgreen",
        linestyle=":",
        linewidth=2.0,
        label="Oracle (Ideal)",
    )

    aurcc_model = np.mean(model_risk)
    aurcc_random = np.mean(random_risk)
    aurcc_oracle = np.mean(oracle_risk)

    # Re-import add_annotations here to ensure it's available in this scope
    from torchregress.viz.utils import add_annotations

    annotations = {
        "AURCC Model": aurcc_model,
        "AURCC Random": aurcc_random,
        "AURCC Oracle": aurcc_oracle,
    }

    add_annotations(ax, annotations, loc="upper right")

    ax.set_xlabel("Coverage (Fraction of data kept)", fontweight="bold")
    ax.set_ylabel("Remaining Risk (MSE)", fontweight="bold")
    ax.set_xlim(0, 1.02)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    if return_figure:
        return fig
    elif created_fig:
        fig.tight_layout()
        plt.show()
        plt.close(fig)  # TR-VIZ-05: close figures we created
    return None


def plot_causal_uplift_qini(
    uplift_scores: Union[torch.Tensor, np.ndarray],
    treatment: Union[torch.Tensor, np.ndarray],
    y_obs: Union[torch.Tensor, np.ndarray],
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Causal Uplift Qini Curve",
    return_figure: bool = False,
    ax: Optional[plt.Axes] = None,
) -> Optional[Figure]:
    """
    Plots the Qini curve showing cumulative uplift per treated fraction.
    Useful for evaluating doubly robust causal uplift models.

    Args:
        uplift_scores: Predicted individual treatment effect (ITE) or uplift score.
                       Higher score means the treatment is predicted to have larger positive effect.
        treatment: Binary treatment indicator (1 = treated, 0 = control)
        y_obs: Observed outcome
        figsize: Figure size (width, height)
        title: Plot title
        return_figure: If True, return figure
        ax: Optional axes
    """
    from torchregress.metrics.utils import convert_to_tensor

    uplift_scores = convert_to_tensor(uplift_scores).detach().cpu().numpy().flatten()
    treatment = convert_to_tensor(treatment).detach().cpu().numpy().flatten().astype(int)
    y_obs = convert_to_tensor(y_obs).detach().cpu().numpy().flatten()

    sort_idx = np.argsort(uplift_scores)[::-1]
    treatment_sorted = treatment[sort_idx]
    y_sorted = y_obs[sort_idx]

    n_samples = len(uplift_scores)
    qini_x = np.arange(1, n_samples + 1) / n_samples

    cum_treated = np.cumsum(treatment_sorted)
    cum_control = np.arange(1, n_samples + 1) - cum_treated

    cum_y_treated = np.cumsum(y_sorted * treatment_sorted)
    cum_y_control = np.cumsum(y_sorted * (1 - treatment_sorted))

    with np.errstate(divide="ignore", invalid="ignore"):
        qini_y = cum_y_treated - cum_y_control * (
            cum_treated / np.where(cum_control == 0, 1.0, cum_control)
        )
    # TR-VIZ-06..21: prefixes that contain only one arm have no defined uplift —
    # mask them as NaN so the curve splits instead of fabricating values.
    no_arm = (cum_treated == 0) | (cum_control == 0)
    qini_y = np.where(no_arm, np.nan, qini_y)

    qini_x = np.insert(qini_x, 0, 0.0)
    qini_y = np.insert(qini_y, 0, 0.0)

    random_y = qini_x * qini_y[-1]
    finite = np.isfinite(qini_y)

    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = cast(Figure, ax.figure)

    ax.plot(qini_x, qini_y, color="navy", linewidth=2.5, label="Model Policy")
    ax.plot(qini_x, random_y, color="gray", linestyle="--", linewidth=1.5, label="Random Policy")

    # Integrate over the finite part of the curve only: single-arm prefixes are NaN
    # (see above) and would otherwise make the area NaN for every input.
    # Use np.trapezoid (NumPy 2.0+) or fallback to np.trapz / scipy.integrate.trapezoid
    area_x = qini_x[finite]
    area_y = (qini_y - random_y)[finite]
    if area_x.size < 2:
        qini_area = float("nan")
    elif hasattr(np, "trapezoid"):
        qini_area = np.trapezoid(area_y, area_x)
    else:
        try:
            qini_area = np.trapz(area_y, area_x)
        except AttributeError:
            from scipy.integrate import trapezoid

            qini_area = trapezoid(area_y, area_x)

    # Re-import add_annotations here to ensure it's available in this scope
    from torchregress.viz.utils import add_annotations

    annotations = {
        "Qini Area Metric": qini_area,
    }
    add_annotations(ax, annotations, loc="lower right")

    ax.set_xlabel("Fraction of Population Treated (ranked by uplift)", fontweight="bold")
    ax.set_ylabel("Cumulative Uplift (Incremental Outcome)", fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    if return_figure:
        return fig
    elif created_fig:
        fig.tight_layout()
        plt.show()
        plt.close(fig)  # TR-VIZ-05: close figures we created
    return None


def _mass_contour_levels(
    density: np.ndarray, masses: Tuple[float, ...] = (0.393, 0.865)
) -> List[float]:
    """Density contour levels enclosing the given probability masses (TR-VIZ-06..21).

    Sorts densities descending and picks the density value where the normalized
    cumulative mass first reaches each target. The defaults correspond to the
    1- and 2-sigma credible regions of a 2-D Gaussian:
    ``1 - exp(-0.5) = 0.393`` and ``1 - exp(-2) = 0.865``.
    """
    flat = np.sort(np.asarray(density, dtype=float).ravel())[::-1]
    cum = np.cumsum(flat)
    total = cum[-1]
    if not np.isfinite(total) or total <= 0:
        return []
    cum /= total

    levels: List[float] = []
    for m in masses:
        idx = int(np.searchsorted(cum, m))
        levels.append(float(flat[min(idx, len(flat) - 1)]))
    return sorted(levels)


def _mix_white(color: str, frac: float) -> Tuple[float, float, float]:
    """``color`` blended with white (``frac`` = 1 gives white)."""
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * frac, g + (1 - g) * frac, b + (1 - b) * frac)


def _shades(color: str, n: int) -> List[Tuple[float, float, float]]:
    """``n`` sequential fills of ``color``, outermost (lightest) first."""
    if n == 1:
        return [_mix_white(color, 0.55)]
    return [_mix_white(color, 0.8 - 0.55 * k / (n - 1)) for k in range(n)]


def _inner_ticks(lo: float, hi: float, n: int = 4, margin: float = 0.1) -> np.ndarray:
    """Round ticks inside ``[lo, hi]``, away from both edges (neighbours never collide)."""
    ticks = MaxNLocator(n).tick_values(lo, hi)
    span = hi - lo
    keep = (ticks >= lo + margin * span) & (ticks <= hi - margin * span)
    return ticks[keep] if keep.sum() >= 2 else ticks[(ticks >= lo) & (ticks <= hi)]


def _style_ticks(ax: plt.Axes, fontsize: float, x: bool, y: bool) -> None:
    ax.tick_params(direction="in", labelsize=fontsize, length=3, top=False, right=False)
    if x:
        ax.xaxis.set_major_locator(FixedLocator(_inner_ticks(*ax.get_xlim())))
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    if y:
        ax.yaxis.set_major_locator(FixedLocator(_inner_ticks(*ax.get_ylim())))
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))


_TRUTH_EFFECTS = [pe.Stroke(linewidth=2.6, foreground="white"), pe.Normal()]


def plot_corner_plot(
    samples: Any,
    *,
    member_samples: Optional[List[Any]] = None,
    param_names: Optional[List[str]] = None,
    true_vals: Optional[Union[np.ndarray, torch.Tensor]] = None,
    quantiles: Tuple[float, ...] = (0.16, 0.50, 0.84),
    show_titles: bool = True,
    show_uncertainty_decomposition: bool = True,
    show_scatter: Optional[bool] = None,
    title: Optional[str] = None,
    color: str = "#4477AA",
    truth_color: str = "#CC3311",
    member_colors: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[Union[str, Path]] = None,
    return_figure: bool = False,
    row: int = 0,
    n_samples: int = 10_000,
    dims: Optional[Sequence[int]] = None,
    max_dims: int = 8,
    levels: Optional[Sequence[float]] = None,
    level_convention: str = "mass",
    smooth: float = 1.0,
    marginal: str = "kde",
    interval: str = "central",
    label: Optional[str] = None,
    others: Optional[Dict[str, Any]] = None,
    region: Optional[JointRegion] = None,
    region_samples: int = 4000,
    return_axes: bool = False,
) -> Union[Figure, Tuple[Figure, np.ndarray], None]:
    r"""Corner plot of a joint predictive: 1-D marginals and 2-D highest-density contours.

    The 2-D contours are highest-density regions of a smoothed density of the
    draws (binned Gaussian KDE by FFT, robust bandwidth, fast up to ``10^6``
    draws).  Each contour of mass :math:`m` is the level
    :math:`t_m` with :math:`\frac1S\sum_s \mathbf 1[\hat p(y_s) \ge t_m] = m`,
    so it encloses a fraction :math:`m` of the draws.  The figure always
    states which levels are drawn: ``level_convention="mass"`` (default
    68% and 95% of the mass) or ``"sigma"`` (the :math:`k\sigma` levels of a
    2-D Gaussian, :math:`1 - e^{-k^2/2}`: 39.3% and 86.5%, the default of the
    ``corner`` package).

    Args:
        samples: Draws ``[n_samples, d]`` (array or tensor on any device and
            dtype), draws ``[n, S, d]`` (row ``row`` is plotted), or a
            :class:`~torchregress.joint.JointPredictive` (row ``row`` is
            sampled ``n_samples`` times).
        member_samples: Draws of ensemble members (same forms); adds dashed
            member marginals and contours and, with
            ``show_uncertainty_decomposition``, aleatoric/epistemic titles.
        param_names: Target names; a trailing ``" [unit]"`` is kept on the axis
            label and moved after the value in the title.
        true_vals: True or observed values ``[d]`` (lines and a marker).
        quantiles: Quantiles of the 1-D summary: the smallest and largest give
            the shaded central interval, the middle one the median line and
            title (``()`` hides both).
        show_titles: Title each 1-D panel ``name = median^{+hi}_{-lo}`` with two
            significant digits of the smaller error.
        show_uncertainty_decomposition: With members, title with total,
            aleatoric and epistemic standard deviations instead.
        show_scatter: Draw the draws outside the outermost contour (default:
            on without members).
        title: Figure title.
        color: Colour of ``samples`` (sequential fills are derived from it).
        truth_color: Colour of the true values.
        member_colors: Colours of the members.
        figsize: Figure size; by default it grows with the number of panels.
        save_path: Save the figure there (300 dpi).
        return_figure: Return the figure (otherwise it is shown and closed).
        row: Row of a predictive or ``[n, S, d]`` draws to plot.
        n_samples: Draws per predictive.
        dims: Targets to show (indices); default all, or for ``d > max_dims``
            the targets of the most dependent pairs by absolute Spearman
            correlation (the figure says so).
        max_dims: Largest number of targets shown without ``dims``.
        levels: Contour levels: masses in ``(0, 1)`` or numbers of sigmas,
            depending on ``level_convention``.
        level_convention: ``"mass"`` or ``"sigma"``.
        smooth: Multiplier of the KDE bandwidth (Scott's rule for 2-D panels,
            Silverman's for 1-D).
        marginal: ``"kde"`` (smooth curve) or ``"hist"`` (step histogram).
        interval: Shaded 1-D interval, ``"central"`` (from ``quantiles``) or
            ``"hpd"`` (highest density, same mass).
        label: Legend label of ``samples``.
        others: ``{label: draws or predictive}`` overlaid in a colour-blind-safe
            palette with distinct line styles (prior vs posterior, two models,
            before vs after recalibration).
        region: A :class:`~torchregress.joint.JointRegion` whose row ``row`` is
            overlaid: exact ellipse or box projections for the Mahalanobis and
            max-residual scores, otherwise the outline of accepted draws.
        region_samples: Accepted draws for a region without a closed form.
        return_axes: Return ``(figure, axes)`` with ``axes`` the ``[k, k]``
            array of panels (``k`` targets shown; upper triangle hidden).

    Returns:
        ``(Figure, axes)`` with ``return_axes``, the ``Figure`` with
        ``return_figure``, else ``None``.
    """
    masses, contour_caption = contour_masses(levels, level_convention)
    if marginal not in ("kde", "hist"):
        raise ValueError(f"marginal must be 'kde' or 'hist', got {marginal!r}")
    if interval not in ("central", "hpd"):
        raise ValueError(f"interval must be 'central' or 'hpd', got {interval!r}")
    if max_dims < 1:
        raise ValueError("max_dims must be at least 1")

    primary = draws_of(samples, row=row, n_samples=n_samples, name="samples")
    d_all = primary.shape[1]
    others_np: List[Tuple[str, np.ndarray]] = []
    for lab, obj in (others or {}).items():
        arr = draws_of(obj, row=row, n_samples=n_samples, name=f"others[{lab!r}]")
        if arr.shape[1] != d_all:
            raise ValueError(f"others[{lab!r}] has {arr.shape[1]} targets, expected {d_all}")
        others_np.append((str(lab), arr))
    members_np: List[np.ndarray] = []
    for m_idx, ms in enumerate(member_samples or []):
        arr = draws_of(ms, row=row, n_samples=n_samples, name=f"member_samples[{m_idx}]")
        if arr.shape[1] != d_all:
            raise ValueError(
                f"member_samples[{m_idx}] has {arr.shape[1]} targets, expected {d_all}"
            )
        members_np.append(arr)

    if param_names is None:
        names_all = [f"$y_{{{i + 1}}}$" for i in range(d_all)]
    elif len(param_names) != d_all:
        msg = f"param_names length ({len(param_names)}) must match n_params ({d_all})"
        raise ValueError(msg)
    else:
        names_all = list(param_names)
    truth_all = None if true_vals is None else as_numpy(true_vals).reshape(-1)

    region_lv: List[RegionLevel] = []
    if region is not None:
        if region.predictive.event_dim != d_all:
            raise ValueError(
                f"region has {region.predictive.event_dim} targets, the draws have {d_all}"
            )
        region_lv = region_levels(region, row=row, n_draws=region_samples)

    note = None
    if dims is not None:
        sel = [int(t) for t in dims]
        if not sel or len(set(sel)) != len(sel) or any(not 0 <= t < d_all for t in sel):
            raise ValueError(f"dims must be distinct indices in [0, {d_all}), got {list(dims)}")
    elif d_all > max_dims:
        sel = most_dependent_dims(primary, max_dims)
        note = (
            f"Showing {len(sel)} of {d_all} targets: those in the most dependent "
            "pairs (|Spearman ρ|). Pass dims= to choose."
        )
    else:
        sel = list(range(d_all))
    k = len(sel)

    # ── data per shown target ────────────────────────────────────────────────
    base_label = label or ("Predictive" if isinstance(samples, JointPredictive) else "Draws")
    palette = [c for c in TOL_BRIGHT if c.lower() != color.lower()]
    sets: List[Tuple[str, np.ndarray, str, Any]] = [
        (f"{base_label} ({primary.shape[0]:,} draws)", primary[:, sel], color, "-")
    ]
    for o_idx, (lab, arr) in enumerate(others_np):
        style = OVERLAY_STYLES[(o_idx + 1) % len(OVERLAY_STYLES)]
        lab_n = f"{lab} ({arr.shape[0]:,} draws)"
        sets.append((lab_n, arr[:, sel], palette[o_idx % len(palette)], style))
    members_sel = [m[:, sel] for m in members_np]
    names = [names_all[t] for t in sel]
    truth = None
    if truth_all is not None:
        truth = np.array([truth_all[t] if t < truth_all.size else np.nan for t in sel])

    ranges: List[Tuple[float, float]] = []
    for c in range(k):
        lo, hi = plot_range([s[1][:, c] for s in sets])
        span = hi - lo
        extras: List[float] = []
        if truth is not None and np.isfinite(truth[c]):
            extras.append(float(truth[c]))
        for lv in region_lv:
            iv = region_interval(lv, sel[c])
            if iv is not None:
                extras.extend(v for v in iv if math.isfinite(v))
        for v in extras:
            if lo - 2 * span <= v <= hi + 2 * span:
                lo, hi = min(lo, v - 0.05 * span), max(hi, v + 0.05 * span)
        ranges.append((lo, hi))
    constant = [
        all(float(np.ptp(s[1][:, c])) <= 1e-12 * max(1.0, abs(float(s[1][0, c]))) for s in sets)
        for c in range(k)
    ]

    if show_scatter is None:
        show_scatter = not members_np and not others_np

    alea_std = epi_std = tot_std = None
    if len(members_sel) > 1:
        member_means = np.array([np.mean(ms, axis=0) for ms in members_sel])
        member_vars = np.array([np.var(ms, axis=0) for ms in members_sel])
        alea_var = np.mean(member_vars, axis=0)
        epi_var = np.var(member_means, axis=0)
        alea_std, epi_std, tot_std = (
            np.sqrt(alea_var),
            np.sqrt(epi_var),
            np.sqrt(alea_var + epi_var),
        )
    if member_colors is None and members_sel:
        member_colors = [
            palette[(m + len(others_np)) % len(palette)] for m in range(len(members_sel))
        ]

    # ── figure geometry ──────────────────────────────────────────────────────
    panel = 2.1 if k <= 3 else 1.8 if k <= 5 else 1.5 if k <= 7 else 1.3
    left, bottom, right = 0.85, 0.8, 0.2
    top = 0.45 + (0.35 if title else 0.0)
    # The legend and caption go in the empty upper triangle for k >= 3.  For k = 2
    # they would cover the top-left panel, so they go in a band below the grid; for
    # k = 1, in a column to the right.
    legend_w = 2.6 if k == 1 else 0.0
    legend_h = 0.0
    if k == 2:
        n_handles = len(sets) + len(members_sel) + int(truth is not None) + len(region_lv)
        n_caption = 2 + int(bool(region_lv)) + (2 if note else 0)
        legend_h = 0.35 + 0.2 * math.ceil(n_handles / 2) + 0.17 * n_caption
    if figsize is None:
        figsize = (left + right + legend_w + panel * k, bottom + legend_h + top + panel * k)
    fig_w, fig_h = float(figsize[0]), float(figsize[1])
    fig, axes = plt.subplots(k, k, figsize=(fig_w, fig_h), squeeze=False)
    fig.subplots_adjust(
        left=min(left / fig_w, 0.3),
        right=max(1 - (right + legend_w) / fig_w, 0.5),
        bottom=min((bottom + legend_h) / fig_h, 0.45),
        top=max(1 - top / fig_h, 0.6),
        wspace=0.07,
        hspace=0.07,
    )
    label_fs = 10.0 if k <= 4 else 9.0
    tick_fs = 7.5 if k <= 5 else 6.5
    title_fs = 9.0 if k <= 5 else 8.0

    q_sorted = sorted(quantiles) if quantiles else []
    interval_mass = (q_sorted[-1] - q_sorted[0]) if len(q_sorted) >= 2 else 0.0
    median_q = 0.5 if 0.5 in q_sorted or not q_sorted else q_sorted[len(q_sorted) // 2]

    # ── 1-D marginals ────────────────────────────────────────────────────────
    for c in range(k):
        ax = axes[c, c]
        lo, hi = ranges[c]
        peak_all = 0.0
        for s_idx, (lab, arr, col, style) in reversed(list(enumerate(sets))):
            v = arr[:, c]
            is_primary = s_idx == 0
            if constant[c]:
                ax.axvline(float(v[0]), color=col, linestyle=style, linewidth=1.4)
                peak_all = max(peak_all, 1.0)
                continue
            if marginal == "kde":
                dens = kde_1d(v, lo, hi, smooth=smooth)
                yv = dens.density / max(float(dens.density.max()), 1e-300)
                grid = dens.grid
                ax.plot(grid, yv, color=col, linestyle=style, linewidth=1.5 if is_primary else 1.3)
            else:
                n_bins = int(np.clip(np.sqrt(v.size) / 1.5, 20, 60))
                counts, edges = np.histogram(v, bins=n_bins, range=(lo, hi))
                yv = counts / max(float(counts.max()), 1.0)
                grid = 0.5 * (edges[:-1] + edges[1:])
                ax.stairs(yv, edges, color=col, linestyle=style, linewidth=1.3)
                dens = kde_1d(v, lo, hi, smooth=smooth) if interval == "hpd" else None
            peak_all = max(peak_all, 1.0)
            if not is_primary or not q_sorted:
                continue
            if interval_mass > 0:
                if interval == "central":
                    qlo, qhi = np.quantile(v, [q_sorted[0], q_sorted[-1]])
                    mask = (grid >= qlo) & (grid <= qhi)
                else:
                    assert dens is not None
                    hmask = hpd_mask_1d(dens, interval_mass)
                    mask = np.interp(grid, dens.grid, hmask.astype(float)) > 0.5
                ax.fill_between(
                    grid,
                    0,
                    yv,
                    where=mask,
                    color=_mix_white(col, 0.45),
                    linewidth=0,
                    step="mid" if marginal == "hist" else None,
                    zorder=0,
                )
            med = float(np.quantile(v, median_q))
            ax.plot([med, med], [0, float(np.interp(med, grid, yv))], color=col, linewidth=1.2)

        for m_idx, ms in enumerate(members_sel):
            if constant[c]:
                continue
            dens_m = kde_1d(ms[:, c], lo, hi, smooth=smooth)
            ref = kde_1d(sets[0][1][:, c], lo, hi, smooth=smooth).density.max()
            ym = dens_m.density / max(float(ref), 1e-300)
            peak_all = max(peak_all, float(ym.max()))
            assert member_colors is not None
            ax.plot(dens_m.grid, ym, color=member_colors[m_idx], linewidth=1.0, linestyle="--")

        if truth is not None and np.isfinite(truth[c]):
            ax.axvline(truth[c], color=truth_color, linewidth=1.3, path_effects=_TRUTH_EFFECTS)

        trans = blended_transform_factory(ax.transData, ax.transAxes)
        for l_idx, lv in enumerate(region_lv):
            iv = region_interval(lv, sel[c])
            if iv is None:
                continue
            yb = 0.035 + 0.045 * l_idx
            ax.plot(
                list(iv),
                [yb, yb],
                transform=trans,
                color=REGION_COLOR,
                linestyle=_region_style(l_idx),
                linewidth=1.3,
                marker="|",
                markersize=6,
                clip_on=False,
            )

        ax.set_xlim(lo, hi)
        ax.set_ylim(0, 1.12 * peak_all)
        ax.set_yticks([])
        for side in ("left", "right", "top"):
            ax.spines[side].set_visible(False)
        base_name, unit = split_unit(names[c])
        if show_titles and len(q_sorted) >= 2 and not constant[c]:
            if alea_std is not None and show_uncertainty_decomposition:
                assert epi_std is not None and tot_std is not None
                p_title = (
                    rf"$\sigma_{{tot}}$={tot_std[c]:.2g}"
                    + "\n"
                    + rf"($\sigma_{{al}}$={alea_std[c]:.2g}, "
                    + rf"$\sigma_{{ep}}$={epi_std[c]:.2g})"
                )
            else:
                v0 = sets[0][1][:, c]
                ql, qm, qh = np.quantile(v0, [q_sorted[0], median_q, q_sorted[-1]])
                p_title = f"{base_name} = {format_estimate(qm, qm - ql, qh - qm)}"
                if unit:
                    p_title += f" {unit}"
            ax.set_title(p_title, fontsize=title_fs, pad=4)
        elif show_titles and constant[c]:
            ax.set_title(f"{base_name} = {sets[0][1][0, c]:.4g} (constant)", fontsize=title_fs)

    # ── 2-D panels ───────────────────────────────────────────────────────────
    for i in range(k):
        for j in range(k):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                continue
            xr, yr = ranges[j], ranges[i]
            if constant[i] or constant[j]:
                for _lab, arr, col, style in sets:
                    pts = subsample(arr[:, [j, i]], 2000)
                    ax.scatter(
                        pts[:, 0], pts[:, 1], s=2, color=col, alpha=0.4, lw=0, rasterized=True
                    )
            else:
                for s_idx, (_lab, arr, col, style) in reversed(list(enumerate(sets))):
                    _draw_pair(
                        ax,
                        arr[:, j],
                        arr[:, i],
                        xr,
                        yr,
                        masses,
                        col,
                        style,
                        primary=s_idx == 0,
                        scatter=bool(show_scatter) and s_idx == 0,
                        smooth=smooth,
                        linewidth=1.2 if s_idx == 0 else 1.5,
                    )
                for m_idx, ms in enumerate(members_sel):
                    assert member_colors is not None
                    _draw_pair(
                        ax,
                        ms[:, j],
                        ms[:, i],
                        xr,
                        yr,
                        masses[:1],
                        member_colors[m_idx],
                        "--",
                        primary=False,
                        scatter=False,
                        smooth=smooth,
                        fill=False,
                        linewidth=0.9,
                    )
            for l_idx, lv in enumerate(region_lv):
                _draw_region_2d(ax, lv, sel[j], sel[i], xr, yr, _region_style(l_idx))
            if truth is not None:
                tx, ty = truth[j], truth[i]
                if np.isfinite(tx):
                    ax.axvline(tx, color=truth_color, linewidth=1.1, path_effects=_TRUTH_EFFECTS)
                if np.isfinite(ty):
                    ax.axhline(ty, color=truth_color, linewidth=1.1, path_effects=_TRUTH_EFFECTS)
                if np.isfinite(tx) and np.isfinite(ty):
                    ax.plot(
                        tx,
                        ty,
                        marker="s",
                        markersize=4.5,
                        color=truth_color,
                        markeredgecolor="white",
                        markeredgewidth=0.8,
                        zorder=6,
                    )
            ax.set_xlim(*xr)
            ax.set_ylim(*yr)

    # ── ticks and labels ─────────────────────────────────────────────────────
    for i in range(k):
        for j in range(i + 1):
            ax = axes[i, j]
            _style_ticks(ax, tick_fs, x=True, y=i != j)
            if i == k - 1:
                ax.set_xlabel(names[j], fontsize=label_fs)
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(names[i], fontsize=label_fs)
            elif i != j:
                ax.set_yticklabels([])
    fig.align_labels(axs=[axes[k - 1, j] for j in range(k)])
    fig.align_labels(axs=[axes[i, 0] for i in range(k)])

    # ── legend and convention caption ────────────────────────────────────────
    handles: List[Any] = []
    for s_idx, (lab, _arr, col, style) in enumerate(sets):
        if s_idx == 0:
            handles.append(
                Patch(facecolor=_mix_white(col, 0.45), edgecolor=col, linewidth=1.2, label=lab)
            )
        else:
            handles.append(Line2D([], [], color=col, linestyle=style, linewidth=1.5, label=lab))
    for m_idx in range(len(members_sel)):
        assert member_colors is not None
        handles.append(
            Line2D([], [], color=member_colors[m_idx], linestyle="--", label=f"Member {m_idx + 1}")
        )
    if truth is not None:
        handles.append(
            Line2D(
                [],
                [],
                color=truth_color,
                marker="s",
                markersize=4.5,
                linewidth=1.2,
                markeredgecolor="white",
                label="True / observed value",
            )
        )
    for l_idx, lv in enumerate(region_lv):
        suffix = " (unbounded, not drawn)" if lv.kind == "unbounded" else ""
        handles.append(
            Line2D(
                [],
                [],
                color=REGION_COLOR,
                linestyle=_region_style(l_idx),
                linewidth=1.4,
                label=lv.label + suffix,
            )
        )
    caption_lines = []
    if k >= 2:
        caption_lines.append(contour_caption + " (smoothed density)")
    if q_sorted and interval_mass > 0:
        kind = "central" if interval == "central" else "highest-density"
        caption_lines.append(f"1-D: median and {100 * interval_mass:.3g}% {kind} interval")
    if region_lv:
        caption_lines.append(
            "Dash-dot: conformal region (calibrated coverage), not a density contour"
        )
    if note:
        caption_lines.append(note)
    wrap = 46 if k >= 3 else 64 if k == 2 else 40
    caption = "\n".join(textwrap.fill(line, wrap) for line in caption_lines)
    legend_fs = 8.0 if k >= 3 else 7.5
    ncol = 1
    if k == 1:
        pos = axes[0, 0].get_position()
        anchor: Tuple[float, float] = (pos.x1 + 0.15 / fig_w, pos.y1)
        loc = "upper left"
    elif k == 2:
        pos = axes[1, 0].get_position()
        anchor = (pos.x0, legend_h / fig_h)
        loc = "upper left"
        ncol = 2
    else:
        pos = axes[0, k - 1].get_position()
        anchor = (pos.x1, pos.y1)
        loc = "upper right"
    leg = fig.legend(
        handles=handles,
        loc=loc,
        bbox_to_anchor=anchor,
        ncol=ncol,
        fontsize=legend_fs,
        frameon=False,
        title=caption or None,
        title_fontsize=legend_fs,
        alignment="left",
        handlelength=2.6,
        borderaxespad=0.0,
    )
    leg.get_title().set_color("#333333")

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1 - 0.15 / fig_h, va="top")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if return_axes:
        return fig, axes
    if return_figure:
        return fig
    plt.show()
    plt.close(fig)  # TR-VIZ-05: close figures we created
    return None


_REGION_STYLES: Tuple[Any, ...] = (REGION_STYLE, (0, (2, 1.5)), (0, (8, 3)))


def _region_style(level: int) -> Any:
    return _REGION_STYLES[level % len(_REGION_STYLES)]


def _draw_pair(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    xr: Tuple[float, float],
    yr: Tuple[float, float],
    masses: Sequence[float],
    color: str,
    style: Any,
    *,
    primary: bool,
    scatter: bool,
    smooth: float,
    fill: bool = True,
    linewidth: float = 1.2,
) -> None:
    """Highest-density contours (filled for the primary set) of one pair of targets."""
    dens = kde_2d(x, y, xr, yr, smooth=smooth)
    thresholds = mass_thresholds(dens.at_draws, masses)  # descending with mass
    finite = [(m, t) for m, t in zip(masses, thresholds) if math.isfinite(t) and t > 0]
    if not finite:
        return
    ts = sorted({t for _, t in finite})
    zmax = float(dens.density.max())
    if fill and primary:
        bounds = ts + [max(zmax, ts[-1]) * 1.0001 + 1e-300]
        ax.contourf(
            dens.x, dens.y, dens.density, levels=bounds, colors=_shades(color, len(ts)), zorder=1
        )
    elif fill:
        ax.contourf(
            dens.x,
            dens.y,
            dens.density,
            levels=[ts[0], max(zmax, ts[0]) * 1.0001],
            colors=[color],
            alpha=0.14,
            zorder=1,
        )
    ax.contour(
        dens.x,
        dens.y,
        dens.density,
        levels=ts,
        colors=[color],
        linewidths=linewidth,
        linestyles=[style],
        zorder=3,
    )
    if scatter:
        outside = dens.at_draws < ts[0]
        pts = subsample(np.stack([x[outside], y[outside]], axis=1), 2500, seed=1)
        if pts.size:
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=1.6,
                color=color,
                alpha=0.45,
                lw=0,
                rasterized=True,
                zorder=0,
            )


def _draw_region_2d(
    ax: plt.Axes,
    lv: RegionLevel,
    tx: int,
    ty: int,
    xr: Tuple[float, float],
    yr: Tuple[float, float],
    style: Any,
) -> None:
    """2-D projection of one region level onto targets ``(tx, ty)``."""
    kw: Dict[str, Any] = dict(color=REGION_COLOR, linestyle=style, linewidth=1.4, zorder=5)
    if lv.kind == "ellipse":
        assert lv.mean is not None and lv.cov is not None
        idx = [tx, ty]
        pts = ellipse_points(lv.mean[idx], lv.cov[np.ix_(idx, idx)], lv.radius)
        ax.plot(pts[:, 0], pts[:, 1], **kw)
    elif lv.kind == "box":
        assert lv.lower is not None and lv.upper is not None
        x0, x1 = lv.lower[tx], lv.upper[tx]
        y0, y1 = lv.lower[ty], lv.upper[ty]
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], **kw)
    elif lv.kind == "draws":
        assert lv.draws is not None
        px, py = lv.draws[:, tx], lv.draws[:, ty]
        dens = kde_2d(px, py, xr, yr, smooth=0.7)
        (t,) = mass_thresholds(dens.at_draws, [0.99])
        if math.isfinite(t) and t > 0:
            ax.contour(
                dens.x,
                dens.y,
                dens.density,
                levels=[t],
                colors=[REGION_COLOR],
                linewidths=1.4,
                linestyles=[style],
                zorder=5,
            )
