"""
Joint multi-target regression: which head, and what does dependence buy?

Fits ``torchregress.joint.JointRegressor`` with each of its heads on one synthetic
problem whose true conditional density is known, and scores them on the same
test rows next to two references:

- the oracle (the true conditional density), and
- "independent targets": the Gaussian head's own per-target means and variances
  with the correlations dropped, which is what reporting per-target predictions
  amounts to.

The data: three targets and four features.  Given ``x`` the targets follow a
two-component Gaussian mixture (a bimodal region that moves the three targets
together) with a noise scale that grows with ``|x0|`` and a correlation between
targets 0 and 1 that changes sign with ``x3``.  A single Gaussian cannot represent
the two modes; independent targets cannot represent the correlation.

Reported per row: test negative log density (``NLL``, and its gap to the oracle),
energy and variogram scores from 200 draws, coverage and median log volume of the
90% conformal region for the whole vector, the coverage of per-target 90% intervals
taken jointly (they are not a joint statement), and wall-clock time.

The goal is decision support on one problem, not a benchmark.
"""

import argparse
import math
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal

from torchregress.comparison import (
    print_comparison_summary,
    print_fairness_notes,
    set_comparison_seed,
    timed_call,
    write_comparison_summary_json,
)
from torchregress.joint import GaussianJoint, JointPredictive, JointRegressor
from torchregress.metrics import energy_score, variogram_score

EXAMPLE_PATH = "examples/joint_regression_comparison.py"
N_TARGETS = 3
N_FEATURES = 4
METHODS = ("gaussian", "mixture", "flow", "sklar", "energy", "auto")
METRIC_ORDER = [
    "NLL",
    "NLLGap",
    "Energy",
    "Variogram",
    "Cov90",
    "LogVol90",
    "BoxCov90",
    "train_s",
    "eval_s",
]


@dataclass
class JointComparisonConfig:
    """Sizes and budget; every head gets the same ``max_epochs`` and ``patience``."""

    n_train: int = 4000
    n_test: int = 1000
    max_epochs: int = 200
    patience: int = 20
    eval_samples: int = 200
    alpha: float = 0.1
    seed: int = 0
    methods: tuple[str, ...] = METHODS


def true_conditional(x: torch.Tensor) -> MixtureSameFamily:
    """The data-generating conditional density of ``y`` given ``x [n, 4]``."""
    n = x.shape[0]
    centre = torch.stack([torch.sin(x[:, 0]), 0.5 * x[:, 1] * x[:, 0], 0.5 * x[:, 2]], dim=1)
    shift = torch.tensor([0.8, -0.8, 0.4], dtype=x.dtype)
    means = torch.stack([centre + shift, centre - shift], dim=1)  # [n, 2, 3]
    scale = torch.tensor([0.3, 0.4, 0.3], dtype=x.dtype) * (1.0 + 0.5 * x[:, :1].abs())
    rho = 0.8 * torch.tanh(x[:, 3])
    corr = torch.eye(N_TARGETS).to(device=x.device, dtype=x.dtype).repeat(n, 1, 1)
    corr[:, 0, 1] = corr[:, 1, 0] = rho
    corr[:, 1, 2] = corr[:, 2, 1] = 0.3
    cov = corr * scale.unsqueeze(-1) * scale.unsqueeze(-2)
    weight = torch.sigmoid(2.0 * x[:, 2])
    mixing = Categorical(probs=torch.stack([weight, 1.0 - weight], dim=1))
    components = MultivariateNormal(means, covariance_matrix=cov.unsqueeze(1).expand(n, 2, 3, 3))
    return MixtureSameFamily(mixing, components)


def make_data(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``n`` rows of features ``[n, 4]`` and targets ``[n, 3]``."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, N_FEATURES, generator=gen)
    with torch.random.fork_rng():
        torch.manual_seed(seed + 1)
        y = true_conditional(x).sample()
    return x, y


def independent_targets(predictive: JointPredictive) -> GaussianJoint:
    """The same means and per-target variances with every correlation set to zero."""
    variances = predictive.covariance().diagonal(dim1=-2, dim2=-1)
    return GaussianJoint.from_covariance(
        predictive.mean(),
        torch.diag_embed(variances).to(device=variances.device, dtype=variances.dtype),
    )


def box_coverage(draws: torch.Tensor, y: torch.Tensor, alpha: float) -> float:
    """Share of rows with every target inside its own central ``1 - alpha`` interval."""
    lo = torch.quantile(draws, alpha / 2, dim=1)
    hi = torch.quantile(draws, 1 - alpha / 2, dim=1)
    return float(((y >= lo) & (y <= hi)).all(dim=1).float().mean())


def score_predictive(
    predictive: JointPredictive, y: torch.Tensor, oracle_nll: float, cfg: JointComparisonConfig
) -> dict[str, Any]:
    """Proper scores of a joint predictive on test targets ``y [n, 3]``."""
    with torch.random.fork_rng():
        torch.manual_seed(cfg.seed)
        draws = predictive.sample(cfg.eval_samples).cpu()
    row: dict[str, Any] = {
        "Energy": float(energy_score(draws.movedim(1, 0), y)),
        "Variogram": float(variogram_score(draws.movedim(1, 0), y)),
        "BoxCov90": box_coverage(draws, y, cfg.alpha),
    }
    if predictive.has_density:
        nll = float(-predictive.log_prob(y).mean())
        row["NLL"] = nll
        row["NLLGap"] = nll - oracle_nll
    return row


def run_method(
    method: str,
    data: dict[str, torch.Tensor],
    oracle_nll: float,
    cfg: JointComparisonConfig,
) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    """Fit one head; return its row and, for the Gaussian head, the independent row."""
    model = JointRegressor(
        method,  # type: ignore[arg-type]
        max_epochs=cfg.max_epochs,
        patience=cfg.patience,
        seed=cfg.seed,
        device="cpu",
    )
    _, train_s = timed_call(model.fit, data["x_train"], data["y_train"])

    def evaluate() -> tuple[dict[str, Any], JointPredictive]:
        predictive = model.predict(data["x_test"])
        scores = score_predictive(predictive, data["y_test"], oracle_nll, cfg)
        # Sampling scores and Monte Carlo volumes draw random numbers: seed them so the
        # row does not depend on what was fitted before it.
        with torch.random.fork_rng():
            torch.manual_seed(cfg.seed)
            region = model.region(data["x_test"], alpha=cfg.alpha)
            scores["Cov90"] = float(region.contains(data["y_test"]).float().mean())
            log_volume, _ = region.log_volume()
        finite = log_volume[torch.isfinite(log_volume)]
        if finite.numel() > 0:
            scores["LogVol90"] = float(finite.median())
        return scores, predictive

    (scores, predictive), eval_s = timed_call(evaluate)
    notes = f"members {model.members_}"
    if model.dependence_ is not None:
        notes += f", dependence rung {model.dependence_}"
    row = {
        "Method": f"JointRegressor({method!r})",
        **scores,
        "train_s": train_s,
        "eval_s": eval_s,
        "Notes": notes,
    }
    independent_row = None
    if method == "gaussian":
        independent = independent_targets(predictive)
        independent_row = {
            "Method": "independent targets",
            **score_predictive(independent, data["y_test"], oracle_nll, cfg),
            "Notes": "Gaussian head's means and variances, correlations dropped; no region",
        }
    return row, independent_row


def main(
    cfg: Optional[JointComparisonConfig] = None, summary_json_path: Optional[str] = None
) -> list[dict[str, Any]]:
    cfg = cfg or JointComparisonConfig()
    set_comparison_seed(cfg.seed)
    x_train, y_train = make_data(cfg.n_train, cfg.seed)
    x_test, y_test = make_data(cfg.n_test, cfg.seed + 1000)
    data = {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}

    truth = true_conditional(x_test)
    oracle_nll = float(-truth.log_prob(y_test).mean())
    with torch.random.fork_rng():
        torch.manual_seed(cfg.seed)
        oracle_draws = truth.sample((cfg.eval_samples,))  # [S, n, d]
    rows: list[dict[str, Any]] = [
        {
            "Method": "oracle",
            "NLL": oracle_nll,
            "NLLGap": 0.0,
            "Energy": float(energy_score(oracle_draws, y_test)),
            "Variogram": float(variogram_score(oracle_draws, y_test)),
            "BoxCov90": box_coverage(oracle_draws.movedim(0, 1), y_test, cfg.alpha),
            "Notes": "the true conditional density",
        }
    ]
    for method in cfg.methods:
        row, independent_row = run_method(method, data, oracle_nll, cfg)
        if independent_row is not None:
            rows.append(independent_row)
        rows.append(row)

    print_fairness_notes(
        title="Joint regression heads",
        seed_policy=f"one seed ({cfg.seed}) for data, splits, initialisation and evaluation draws",
        train_budget=(
            f"n_train={cfg.n_train}, at most {cfg.max_epochs} epochs, patience {cfg.patience}, "
            "estimator defaults otherwise (60% of the rows train, the rest validate, "
            "recalibrate and calibrate the region)"
        ),
        metric_policy=(
            f"{cfg.n_test} test rows; NLL from the density, energy and variogram scores from "
            f"{cfg.eval_samples} draws, region coverage at {1 - cfg.alpha:.0%}"
        ),
    )
    print_comparison_summary("Joint regression: heads and references", rows, METRIC_ORDER)
    se = math.sqrt(cfg.alpha * (1 - cfg.alpha) / cfg.n_test)
    notes = [
        "One synthetic problem and one seed: read the ranking as an illustration.",
        f"Cov90 is the conformal region's coverage of the whole vector; one standard error is "
        f"about {se:.3f} on {cfg.n_test} test rows.",
        "BoxCov90 takes each target's own 90% interval and asks for all three at once: it is "
        "below 0.90 by construction, which is why per-target intervals are not a joint region.",
        "The energy head has no density: no NLL, and its region is a ball around its draws.",
        "LogVol90 is the median log volume of the region; smaller is better at equal coverage.",
    ]
    print("\nReading notes:")
    for note in notes:
        print(f"- {note}")
    if summary_json_path is not None:
        out = write_comparison_summary_json(
            summary_json_path,
            example=EXAMPLE_PATH,
            task="joint_multi_target_regression",
            config=cfg,
            rows=rows,
            notes=notes,
        )
        print(f"\nWrote summary JSON: {out}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare the heads of JointRegressor")
    parser.add_argument("--summary-json-path", type=str, default=None)
    parser.add_argument("--n-train", type=int, default=JointComparisonConfig.n_train)
    args = parser.parse_args()
    main(JointComparisonConfig(n_train=args.n_train), summary_json_path=args.summary_json_path)
