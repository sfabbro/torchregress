r"""`JointRegressor`: arrays in, calibrated joint predictive distributions out.

.. code-block:: python

    model = JointRegressor().fit(X, Y)          # Y [n, d]; NaN marks a missing target
    dist = model.predict(X_new)                 # JointPredictive, one distribution per row
    dist.sample(256); dist.log_prob(Y_new); dist.marginal([0, 2])
    region = model.region(X_new, alpha=0.1)     # conformal region, finite-sample coverage
    region.contains(Y_new); region.log_volume()

What ``fit`` does, in order:

1. **Standardise** features and targets (statistics of the training rows).  Heads are
   fitted on standardised targets and their predictives are mapped back with
   :class:`~torchregress.joint.AffineJoint`, so raw units never meet a bounded spline.
2. **Fit the heads** on the training split with early stopping on a validation split.
3. **Stack** the heads' densities with weights that maximise the validation log score
   (a convex problem; Yao et al. 2018), giving a mixture predictive.
4. **Recalibrate** on a held-out split: one scale factor per target, chosen by held-out
   log score.
5. **Conformalise** on another held-out split: :meth:`region` returns a set with
   :math:`P(Y \in R(X)) \ge 1 - \alpha` under exchangeability, whatever the model.

Steps 4 and 5 use different rows: a recalibrator fitted on the conformal rows would break
the exchangeability the coverage statement rests on.
"""

from __future__ import annotations

import copy
import math
import warnings
from collections.abc import Sequence
from typing import Any, Literal, Optional, Union

import torch
from torch import Tensor, nn

from ..ensemble.density import _em_mixture_weights
from ..losses.mdn import MixtureDensityLoss
from ..losses.nflows import NormalizingFlowLoss, create_flow_model, recommended_tail_bound
from ..utils.distributions import inverse_softplus
from ..utils.quantile import pinball_residual
from ._common import mlp, safe_cholesky, seeded
from ._common import train as train_module
from .calibration import ScaleRecalibrator
from .dependence import (
    ConditionalGaussianCopula,
    Dependence,
    FlowDependence,
    GaussianCopula,
    IndependenceDependence,
    SklarJoint,
    select_dependence,
)
from .marginals import Marginals, QuantileMarginals
from .predictive import (
    AffineJoint,
    DistributionJoint,
    FlowJoint,
    GaussianJoint,
    JointPredictive,
    MixtureJoint,
)
from .regions import (
    ConformalScore,
    HPDScore,
    JointConformal,
    JointRegion,
    LatentScore,
    MahalanobisScore,
    SampleBallScore,
)

__all__ = ["JointRegressor", "MarginalEngine", "QuantileNetworkEngine"]

Method = Literal["auto", "gaussian", "mixture", "flow", "sklar"]
ArrayLike = Any  # torch.Tensor or anything torch.as_tensor accepts (numpy arrays)


# ═══════════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════════


class _Head(nn.Module):
    """A network that turns standardised features into a joint predictive."""

    supports_missing = False

    def predictive(self, x: Tensor) -> JointPredictive:
        raise NotImplementedError

    def loss(self, x: Tensor, y: Tensor, mask: Optional[Tensor]) -> Tensor:
        """Mean negative log density of the observed targets."""
        raise NotImplementedError


class _GaussianHead(_Head):
    r"""Mean and Cholesky factor of :math:`\mathcal N(\mu(x), L(x)L(x)^\top)`.

    The output layer starts at zero, so training starts from the unit-variance
    independent Gaussian of the standardised targets.  With ``rank`` set the
    covariance is :math:`VV^\top + \mathrm{diag}(D)` with ``rank`` columns; the
    rows of the output layer that produce :math:`V` start small but not at zero
    (the gradient of :math:`VV^\top` is linear in :math:`V`, so a zero factor
    would never move), and the factorisation adds diagonal jitter to rows that
    are not numerically positive definite.
    """

    supports_missing = True

    def __init__(self, in_dim: int, d: int, hidden: Sequence[int], rank: Optional[int]) -> None:
        super().__init__()
        self.d = d
        self.rank = rank
        n_cov = d * (d + 1) // 2 if rank is None else d + d * rank
        self.net = mlp(in_dim, hidden, d + n_cov)
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        if rank is not None:
            # Small random factor: V V^T has a zero gradient at V = 0.
            with torch.no_grad():
                width = last.weight.shape[1]
                last.weight[2 * d :].normal_(0.0, 1e-2 / math.sqrt(width))
                last.bias[2 * d :].normal_(0.0, 1e-2)
        self._diag_shift = float(inverse_softplus(torch.tensor(1.0, dtype=torch.float64)))
        rows, cols = torch.tril_indices(d, d, offset=-1)
        self.register_buffer("_rows", rows, persistent=False)
        self.register_buffer("_cols", cols, persistent=False)

    def _moments(self, x: Tensor) -> tuple[Tensor, Tensor]:
        out = self.net(x)
        d = self.d
        mean = out[:, :d]
        diag = nn.functional.softplus(out[:, d : 2 * d] + self._diag_shift) + 1e-4
        if self.rank is None:
            tril = torch.diag_embed(diag).to(device=out.device, dtype=out.dtype)
            rows, cols = self._rows, self._cols
            assert isinstance(rows, Tensor) and isinstance(cols, Tensor)
            tril = tril.index_put(
                (torch.arange(out.shape[0], device=out.device).unsqueeze(1), rows, cols),
                out[:, 2 * d :],
            )
            return mean, tril
        factor = out[:, 2 * d :].reshape(-1, d, self.rank)
        cov = factor @ factor.mT + torch.diag_embed(diag.square()).to(
            device=out.device, dtype=out.dtype
        )
        return mean, safe_cholesky(cov)

    def predictive(self, x: Tensor) -> GaussianJoint:
        return GaussianJoint(*self._moments(x))

    def loss(self, x: Tensor, y: Tensor, mask: Optional[Tensor]) -> Tensor:
        return -self.predictive(x).log_prob(y, mask).mean()


class _MixtureHead(_Head):
    """Mixture density network (Gaussian components; exact missing-target densities)."""

    supports_missing = True

    def __init__(self, in_dim: int, d: int, hidden: Sequence[int], n_components: int) -> None:
        super().__init__()
        covariance = "full" if d <= 8 else "diagonal"
        self.mdn = MixtureDensityLoss(
            n_components=n_components, n_features=d, covariance_type=covariance
        )
        self.net = mlp(in_dim, hidden, self.mdn.expected_output_size)

    def predictive(self, x: Tensor) -> DistributionJoint:
        return DistributionJoint(self.mdn.distribution(self.net(x)))

    def loss(self, x: Tensor, y: Tensor, mask: Optional[Tensor]) -> Tensor:
        return -self.predictive(x).log_prob(y, mask).mean()


class _FlowHead(_Head):
    """Conditional neural spline flow on the standardised targets."""

    def __init__(
        self, in_dim: int, d: int, hidden: Sequence[int], bound: float, n_transforms: int
    ) -> None:
        super().__init__()
        context = int(hidden[-1]) if hidden else in_dim
        self.net = mlp(in_dim, hidden[:-1], context) if hidden else nn.Sequential(nn.Identity())
        flow = create_flow_model(
            n_features=d,
            context_dim=context,
            flow_type="nsf",
            n_transforms=n_transforms,
            hidden_features=[64, 64],
            bound=bound,
        )
        self.flow_loss = NormalizingFlowLoss(flow=flow)

    def predictive(self, x: Tensor) -> FlowJoint:
        return FlowJoint(self.flow_loss, self.net(x))

    def loss(self, x: Tensor, y: Tensor, mask: Optional[Tensor]) -> Tensor:
        return -self.flow_loss.log_prob(self.net(x), y).mean()


def _stacking_weights(log_density: Tensor, n_iter: int = 500) -> Tensor:
    r"""Weights on the simplex maximising :math:`\sum_i \log \sum_k w_k p_k(y_i \mid x_i)`.

    ``log_density`` is ``[N, K]``; EM (shared with
    :meth:`~torchregress.ensemble.DensityEnsemble.fit_weights`).
    """
    return _em_mixture_weights(log_density.T, max_iter=n_iter, weight_tol=1e-9, floor=1e-12)


# ═══════════════════════════════════════════════════════════════════════════════
# marginal engines (the first stage of the Sklar route)
# ═══════════════════════════════════════════════════════════════════════════════


class MarginalEngine:
    """Anything that predicts per-target distributions.

    Wrap a tabular foundation model, boosted quantiles or any other per-target
    predictor by implementing two methods; :class:`JointRegressor` supplies the
    dependence between targets.
    """

    def fit(self, x: Tensor, y: Tensor, mask: Optional[Tensor] = None) -> "MarginalEngine":
        """Fit on features ``[N, p]`` and targets ``[N, d]`` (``mask``: observed entries)."""
        raise NotImplementedError

    def predict(self, x: Tensor) -> Marginals:
        """Per-target predictive distributions for the rows of ``x``."""
        raise NotImplementedError


class QuantileNetworkEngine(MarginalEngine):
    """Built-in marginal engine: one network predicting a grid of quantiles per target.

    Trained with the pinball loss on ``n_quantiles`` equally spaced levels; missing
    targets are left out of the loss.

    Args:
        n_quantiles: Size of the quantile grid (levels ``k / (K + 1)``).
        hidden: Hidden-layer widths.
        max_epochs, patience, batch_size, lr, weight_decay: Training budget.
        val_fraction: Rows held out for early stopping.
        seed: Seed of the initialisation, the split and the shuffling.
    """

    def __init__(
        self,
        n_quantiles: int = 31,
        hidden: Sequence[int] = (128, 128),
        *,
        max_epochs: int = 300,
        patience: int = 30,
        batch_size: int = 256,
        lr: float = 2e-3,
        weight_decay: float = 1e-5,
        val_fraction: float = 0.15,
        seed: int = 0,
    ) -> None:
        if n_quantiles < 3:
            raise ValueError("n_quantiles must be at least 3")
        self.n_quantiles = int(n_quantiles)
        self.hidden = tuple(int(h) for h in hidden)
        self.max_epochs, self.patience, self.batch_size = max_epochs, patience, batch_size
        self.lr, self.weight_decay, self.val_fraction, self.seed = (
            lr,
            weight_decay,
            val_fraction,
            seed,
        )
        self._net: Optional[nn.Sequential] = None
        self._d = 0

    @property
    def levels(self) -> Tensor:
        k = self.n_quantiles
        return torch.arange(1, k + 1, dtype=torch.float32) / (k + 1)

    def _quantiles(self, x: Tensor) -> Tensor:
        assert self._net is not None
        out = self._net(x).reshape(x.shape[0], self._d, self.n_quantiles)
        # First level plus non-negative increments: quantiles cannot cross.
        first = out[..., :1]
        steps = nn.functional.softplus(out[..., 1:])
        return torch.cat([first, first + torch.cumsum(steps, dim=-1)], dim=-1)

    def fit(self, x: Tensor, y: Tensor, mask: Optional[Tensor] = None) -> "QuantileNetworkEngine":
        """Fit on ``x [N, p]`` and ``y [N, d]``; the caller's torch RNG state is left as it was."""
        with seeded(self.seed):
            return self._fit(x, y, mask)

    def _fit(self, x: Tensor, y: Tensor, mask: Optional[Tensor]) -> "QuantileNetworkEngine":
        gen = torch.Generator().manual_seed(self.seed)
        n, self._d = y.shape
        torch.manual_seed(self.seed)
        self._net = mlp(x.shape[1], self.hidden, self._d * self.n_quantiles).to(x.device, x.dtype)
        # Start at the standard normal quantiles for every row (targets are standardised
        # by the caller): zero output weights, biases set to the grid and its increments.
        last = self._net[-1]
        assert isinstance(last, nn.Linear)
        grid = torch.special.ndtri(self.levels.double())
        steps = (grid[1:] - grid[:-1]).clamp_min(1e-4)
        bias = torch.cat([grid[:1], inverse_softplus(steps)])
        with torch.no_grad():
            last.weight.zero_()
            last.bias.copy_(bias.repeat(self._d).to(last.bias))
        observed = torch.ones_like(y, dtype=torch.bool) if mask is None else mask
        target = torch.where(observed, y, torch.zeros_like(y))
        levels = self.levels.to(x.device, x.dtype)
        perm = torch.randperm(n, generator=gen)
        n_val = max(1, int(round(self.val_fraction * n)))
        val, train = perm[:n_val].to(x.device), perm[n_val:].to(x.device)

        def pinball(rows: Tensor) -> Tensor:
            loss = pinball_residual(
                self._quantiles(x[rows]), target[rows].unsqueeze(-1), levels
            ).mean(-1)
            keep = observed[rows].to(loss.dtype)
            return (loss * keep).sum() / keep.sum().clamp_min(1.0)

        train_module(
            self._net,
            lambda idx: pinball(train[idx.to(train.device)]),
            int(train.numel()),
            lambda: pinball(val),
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            generator=gen,
            optimizer="adamw",
            clip_norm=10.0,
            skip_nonfinite=True,
            name="the quantile engine",
        )
        return self

    def predict(self, x: Tensor) -> QuantileMarginals:
        if self._net is None:
            raise RuntimeError("Call fit() first.")
        with torch.no_grad():
            return QuantileMarginals(self.levels.to(x.device, x.dtype), self._quantiles(x))


# ═══════════════════════════════════════════════════════════════════════════════
# the estimator
# ═══════════════════════════════════════════════════════════════════════════════


class JointRegressor:
    r"""Joint predictive distributions :math:`p(y \mid x)` for :math:`y \in \mathbb R^d`.

    Args:
        method: ``"auto"`` (default) fits several heads and stacks their densities by
            validation log score.  A single head can be requested instead:
            ``"gaussian"`` (full or low-rank covariance, the reference model),
            ``"mixture"`` (mixture density network), ``"flow"`` (conditional spline
            flow) or ``"sklar"`` (per-target marginals joined by the lowest rung of the
            dependence ladder that validation supports: independence, constant Gaussian
            copula, covariate-dependent Gaussian copula, conditional flow).
        marginals: Marginal engine for the Sklar route (a :class:`MarginalEngine`);
            default :class:`QuantileNetworkEngine`.  Engines are refitted out of fold
            to produce the normal scores the dependence model is trained on.
        hidden: Hidden-layer widths of the heads.
        n_components: Mixture components.
        rank: Rank of the Gaussian head's low-rank covariance; ``None`` uses a full
            Cholesky factor up to 32 targets and rank 8 beyond.
        max_epochs, patience, batch_size, lr, weight_decay: Training budget per head.
        val_fraction, recal_fraction, conformal_fraction: Held-out shares for early
            stopping and stacking, for recalibration, and for conformal calibration.
        recalibrate: Fit the per-target scale recalibration.
        score: Conformity score of :meth:`region`; ``"auto"`` picks the latent norm
            (:class:`~torchregress.joint.LatentScore`) for a Gaussian predictive (whose
            latent ball is its highest-density region), the HPD level
            (:class:`~torchregress.joint.HPDScore`) for every other predictive with a
            density, a sample-ball score for draws-only predictives, and the masked
            Mahalanobis score for conformal rows with missing targets.  See
            :meth:`region` for the trade-off.
        n_folds: Folds for the out-of-fold normal scores of the Sklar route.
        device: Training device (default: CUDA when available, else CPU).
        seed: Seed for splits, initialisation and shuffling.

    Missing targets (``NaN`` in ``Y``) are integrated out exactly by the Gaussian and
    mixture heads.  The flow head needs complete rows, so ``"auto"`` leaves it out when
    targets are missing.

    **How** ``fit`` **splits the rows.**  A random permutation (from ``seed``) gives
    ``round(val_fraction * N)`` validation rows (early stopping and stacking),
    ``round(recal_fraction * N)`` recalibration rows and
    ``round(conformal_fraction * N)`` conformal rows (each at least 8); the rest
    (60% with the defaults 15%, 10%, 15%) trains the heads.  The counts are
    ``n_val_``, ``n_recal_``, ``n_conformal_`` and ``n_train_``.  :meth:`calibrate`
    replaces the conformal rows with data you supply.

    Attributes:
        members_: Names of the fitted heads.
        weights_: Stacking weights (``[K]``).
        validation_log_score_: Mean validation log density per head, in the units of
            the data (comparable with ``predict(X).log_prob(Y).mean()``, before
            recalibration); over the observed targets when targets are missing.
        n_train_, n_val_, n_recal_, n_conformal_: Rows in each split
            (``n_conformal_`` follows :meth:`calibrate`).
        dependence_scores_: Validation log score of each dependence rung (Sklar route).
        dependence_: Name of the rung the validation gate chose (Sklar route).
        scale_: Per-target recalibration factors (``None`` when not fitted).
    """

    def __init__(
        self,
        method: Method = "auto",
        *,
        marginals: Optional[MarginalEngine] = None,
        hidden: Sequence[int] = (128, 128),
        n_components: int = 5,
        rank: Optional[int] = None,
        max_epochs: int = 300,
        patience: int = 30,
        batch_size: int = 256,
        lr: float = 2e-3,
        weight_decay: float = 1e-5,
        val_fraction: float = 0.15,
        recal_fraction: float = 0.1,
        conformal_fraction: float = 0.15,
        recalibrate: bool = True,
        score: Union[str, ConformalScore] = "auto",
        n_folds: int = 3,
        device: Optional[Union[str, torch.device]] = None,
        seed: int = 0,
    ) -> None:
        if method not in ("auto", "gaussian", "mixture", "flow", "sklar"):
            raise ValueError(f"unknown method {method!r}")
        fractions = (val_fraction, recal_fraction, conformal_fraction)
        if any(f <= 0 for f in fractions) or sum(fractions) >= 0.9:
            raise ValueError("held-out fractions must be positive and sum to less than 0.9")
        self.method: Method = method
        self.marginals = marginals
        self.hidden = tuple(int(h) for h in hidden)
        self.n_components = int(n_components)
        self.rank = rank
        self.max_epochs, self.patience, self.batch_size = (
            int(max_epochs),
            int(patience),
            int(batch_size),
        )
        self.lr, self.weight_decay = float(lr), float(weight_decay)
        self.val_fraction, self.recal_fraction = float(val_fraction), float(recal_fraction)
        self.conformal_fraction = float(conformal_fraction)
        self.recalibrate = bool(recalibrate)
        self.score = score
        if int(n_folds) < 2:
            raise ValueError(f"n_folds must be at least 2, got {n_folds}")
        self.n_folds = int(n_folds)
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.seed = int(seed)

        self.members_: list[str] = []
        self.weights_: Optional[Tensor] = None
        self.validation_log_score_: dict[str, float] = {}
        self.dependence_scores_: dict[str, float] = {}
        self.dependence_: Optional[str] = None
        self.scale_: Optional[Tensor] = None
        self.n_features_in_: Optional[int] = None
        self.n_train_: Optional[int] = None
        self.n_val_: Optional[int] = None
        self.n_recal_: Optional[int] = None
        self.n_conformal_: Optional[int] = None
        self.n_targets_: Optional[int] = None
        self._heads: dict[str, _Head] = {}
        self._engine: Optional[MarginalEngine] = None
        self._dependence: Optional[Dependence] = None
        self._recalibrator: Optional[ScaleRecalibrator] = None
        self._x_stats: Optional[tuple[Tensor, Tensor]] = None
        self._y_stats: Optional[tuple[Tensor, Tensor]] = None
        self._conformal_rows: Optional[tuple[Tensor, Tensor, Tensor]] = None
        self._conformal_cache: dict[Any, tuple[JointConformal, Optional[JointConformal]]] = {}

    # ----------------------------------------------------------------- fit
    def fit(self, X: ArrayLike, Y: ArrayLike) -> "JointRegressor":
        """Fit on features ``X [N, p]`` and targets ``Y [N, d]`` (``NaN`` = missing).

        Deterministic for a given ``seed``; the caller's torch RNG state is left as
        it was.  ``±inf`` in ``Y`` is an error (only ``NaN`` marks a missing target).
        """
        with seeded(self.seed):
            return self._fit(X, Y)

    def _fit(self, X: ArrayLike, Y: ArrayLike) -> "JointRegressor":
        x = torch.as_tensor(X, dtype=torch.float32)
        y = torch.as_tensor(Y, dtype=torch.float32)
        if x.dim() != 2 or y.dim() != 2 or x.shape[0] != y.shape[0]:
            raise ValueError("X must be [N, p] and Y [N, d] with the same number of rows")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("X must be finite")
        n, d = y.shape
        if bool(torch.isinf(y).any()):
            raise ValueError("Y contains +-inf; mark missing targets with NaN")
        observed = torch.isfinite(y)
        if not bool(observed.any(dim=0).all()):
            raise ValueError("every target needs at least one observed value")
        if n < 50:
            raise ValueError("at least 50 rows are needed to hold out calibration splits")
        self.n_features_in_, self.n_targets_ = int(x.shape[1]), int(d)
        has_missing = not bool(observed.all())

        gen = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(n, generator=gen)
        n_val = max(8, int(round(self.val_fraction * n)))
        n_rec = max(8, int(round(self.recal_fraction * n)))
        n_conf = max(8, int(round(self.conformal_fraction * n)))
        val, rec = perm[:n_val], perm[n_val : n_val + n_rec]
        conf, train = perm[n_val + n_rec : n_val + n_rec + n_conf], perm[n_val + n_rec + n_conf :]

        x_mean, x_std = x[train].mean(0), x[train].std(0).clamp_min(1e-6)
        filled = torch.where(observed, y, torch.zeros_like(y))
        count = observed[train].sum(0).clamp_min(1)
        y_mean = (filled[train] * observed[train]).sum(0) / count
        y_var = ((filled[train] - y_mean).square() * observed[train]).sum(0) / count
        y_std = y_var.sqrt().clamp_min(1e-6)
        self._x_stats, self._y_stats = (x_mean, x_std), (y_mean, y_std)
        log_y_std = y_std.log().to(self.device)
        self.n_train_, self.n_val_ = int(train.numel()), int(val.numel())
        self.n_recal_, self.n_conformal_ = int(rec.numel()), int(conf.numel())

        dev = self.device
        xs = ((x - x_mean) / x_std).to(dev)
        ys = torch.where(observed, (y - y_mean) / y_std, torch.zeros_like(y)).to(dev)
        mask = observed.to(dev)
        mask_arg: Optional[Tensor] = mask if has_missing else None
        train_d, val_d = train.to(dev), val.to(dev)

        names = self._member_names(d, has_missing)
        log_density: list[Tensor] = []
        self._heads, self.validation_log_score_ = {}, {}
        for offset, name in enumerate(names):
            torch.manual_seed(self.seed + offset)
            if name == "sklar":
                self._fit_sklar(xs, ys, mask, train_d, val_d, gen)
            else:
                head = self._build_head(name, xs.shape[1], d, ys[train_d], mask[train_d]).to(dev)
                rows_tr = train_d if head.supports_missing else train_d[mask[train_d].all(1)]
                rows_va = val_d if head.supports_missing else val_d[mask[val_d].all(1)]
                m_arg = mask_arg if head.supports_missing else None

                def batch_loss(idx: Tensor, head: _Head = head, rows: Tensor = rows_tr) -> Tensor:
                    r = rows[idx.to(rows.device)]
                    return head.loss(xs[r], ys[r], None if m_arg is None else m_arg[r])

                def validation(head: _Head = head, rows: Tensor = rows_va) -> Tensor:
                    return head.loss(xs[rows], ys[rows], None if m_arg is None else m_arg[rows])

                train_module(
                    head,
                    batch_loss,
                    int(rows_tr.numel()),
                    validation,
                    max_epochs=self.max_epochs,
                    patience=self.patience,
                    batch_size=self.batch_size,
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                    generator=gen,
                    optimizer="adamw",
                    clip_norm=10.0,
                    skip_nonfinite=True,
                    name=f"the {name} head",
                )
                for param in head.parameters():
                    param.requires_grad_(False)
                self._heads[name] = head
            masked = has_missing and name != "sklar" and self._heads[name].supports_missing
            rows_lp = val_d if (masked or not has_missing) else val_d[mask[val_d].all(1)]
            with torch.no_grad():
                lp = self._member_predictive(name, xs[rows_lp]).log_prob(
                    ys[rows_lp], mask[rows_lp] if masked else None
                )
            log_density.append(lp)
            # Report in data units (the Jacobian of the standardisation of the
            # observed targets), comparable with predict(X).log_prob(Y).  Stacking
            # uses lp: the shift is the same for every head, so the weights agree.
            obs_rows = (mask[rows_lp] if masked else torch.ones_like(mask[rows_lp])).to(lp.dtype)
            jac = (obs_rows * log_y_std.to(obs_rows)).sum(-1)
            self.validation_log_score_[name] = float((lp - jac).mean())
        self.members_ = names
        self.weights_ = _stack(torch.stack(log_density, dim=1).double()).to(torch.float32).cpu()

        self._recalibrator, self.scale_ = None, None
        complete_rec = rec[observed[rec].all(1)]
        if self.recalibrate and complete_rec.numel() >= 8:
            base = self._base_predictive(x[complete_rec])
            objective = "nll" if base.has_density else "variance"
            recalibrator = ScaleRecalibrator(objective=objective, n_samples=256)
            recalibrator.fit(base, y[complete_rec])
            fitted = recalibrator.scale_
            assert fitted is not None
            self.scale_ = torch.ones_like(fitted.detach().cpu())
            # A factor within 2% of one changes nothing that matters; skipping it saves
            # the predictive-mean evaluation on every later predict.  scale_ reports
            # the factors actually applied (ones when the recalibration is skipped).
            if float(fitted.log().abs().max()) > 0.02:
                keep = True
                complete_val = val[observed[val].all(1)]
                if base.has_density and complete_val.numel() >= 8:
                    # Accept only if it helps on rows the recalibrator has not seen.
                    held = self._base_predictive(x[complete_val])
                    y_held = y[complete_val].to(self.device)
                    with torch.no_grad():
                        before = held.log_prob(y_held).mean()
                        after = recalibrator.transform(held).log_prob(y_held).mean()
                    keep = bool(after > before)
                if keep:
                    self._recalibrator = recalibrator
                    self.scale_ = fitted.detach().cpu()
        self._conformal_rows = (x[conf], filled[conf], observed[conf])
        self._conformal_cache = {}
        return self

    def calibrate(self, X: ArrayLike, Y: ArrayLike) -> "JointRegressor":
        """Replace the conformal rows with calibration data ``X [N, p]``, ``Y [N, d]``.

        For example fresh simulations.  These rows must **not** have been used to
        fit the model (training, validation or recalibration): the coverage
        guarantee of :meth:`region` rests on their exchangeability with test
        rows under the fitted model.  ``NaN`` marks a missing target (``±inf`` is
        an error).  Clears the cached calibrations; returns ``self``.
        """
        if self.n_features_in_ is None or self.n_targets_ is None:
            raise RuntimeError("Call fit() first.")
        x = torch.as_tensor(X, dtype=torch.float32).cpu()
        y = torch.as_tensor(Y, dtype=torch.float32).cpu()
        if x.dim() != 2 or y.dim() != 2 or x.shape[0] != y.shape[0]:
            raise ValueError("X must be [N, p] and Y [N, d] with the same number of rows")
        if x.shape[1] != self.n_features_in_ or y.shape[1] != self.n_targets_:
            raise ValueError(
                f"X must have {self.n_features_in_} columns and Y {self.n_targets_}, "
                f"got {x.shape[1]} and {y.shape[1]}"
            )
        if not bool(torch.isfinite(x).all()):
            raise ValueError("X must be finite")
        if bool(torch.isinf(y).any()):
            raise ValueError("Y contains +-inf; mark missing targets with NaN")
        if x.shape[0] < 1:
            raise ValueError("at least one calibration row is needed")
        observed = torch.isfinite(y)
        filled = torch.where(observed, y, torch.zeros_like(y))
        self._conformal_rows = (x, filled, observed)
        self._conformal_cache = {}
        self.n_conformal_ = int(x.shape[0])
        return self

    def _member_names(self, d: int, has_missing: bool) -> list[str]:
        if self.method != "auto":
            if self.method == "flow" and has_missing:
                warnings.warn(
                    "the flow head needs complete rows: rows with missing targets are dropped",
                    UserWarning,
                    stacklevel=3,
                )
            return [self.method]
        names = ["gaussian", "mixture"]
        if not has_missing:
            names.append("sklar")
            if d <= 16:
                names.append("flow")
        return names

    def _build_head(self, name: str, in_dim: int, d: int, y: Tensor, mask: Tensor) -> _Head:
        if name == "gaussian":
            rank = self.rank if self.rank is not None else (None if d <= 32 else 8)
            return _GaussianHead(in_dim, d, self.hidden, rank)
        if name == "mixture":
            return _MixtureHead(in_dim, d, self.hidden, self.n_components)
        bound = recommended_tail_bound(y[mask.all(1)].detach().cpu())
        return _FlowHead(in_dim, d, self.hidden, bound, n_transforms=3)

    def _fit_sklar(
        self, xs: Tensor, ys: Tensor, mask: Tensor, train: Tensor, val: Tensor, gen: torch.Generator
    ) -> None:
        """Marginal engine, out-of-fold normal scores, and the dependence ladder."""
        template = (
            self.marginals
            if self.marginals is not None
            else QuantileNetworkEngine(
                hidden=self.hidden,
                max_epochs=self.max_epochs,
                patience=self.patience,
                batch_size=self.batch_size,
                lr=self.lr,
                weight_decay=self.weight_decay,
                seed=self.seed,
            )
        )
        complete = train[mask[train].all(1)]
        if self.n_folds > int(complete.numel()):
            raise ValueError(
                f"n_folds = {self.n_folds} exceeds the {int(complete.numel())} complete "
                "training rows of the Sklar route"
            )
        folds = torch.randperm(int(complete.numel()), generator=gen).to(complete.device)
        z_parts, row_parts = [], []
        for fold in range(self.n_folds):
            held = complete[folds[fold :: self.n_folds]]
            keep = torch.ones(complete.numel(), dtype=torch.bool, device=complete.device)
            keep[folds[fold :: self.n_folds]] = False
            rest = complete[keep]
            engine = copy.deepcopy(template).fit(xs[rest], ys[rest], mask[rest])
            z_parts.append(engine.predict(xs[held]).normal_scores(ys[held]))
            row_parts.append(held)
        z_train, rows = torch.cat(z_parts), torch.cat(row_parts)
        self._engine = copy.deepcopy(template).fit(xs[train], ys[train], mask[train])
        val_complete = val[mask[val].all(1)]
        z_val = self._engine.predict(xs[val_complete]).normal_scores(ys[val_complete])
        candidates: list[Dependence] = [IndependenceDependence()]
        if ys.shape[1] > 1:  # a copula on one target is the independence copula
            candidates += [
                GaussianCopula(),
                ConditionalGaussianCopula(seed=self.seed),
                FlowDependence(seed=self.seed),
            ]
        for candidate in candidates:
            candidate.fit(z_train, xs[rows])
        self._dependence, table = select_dependence(candidates, z_val, xs[val_complete])
        self.dependence_scores_ = {k: float(v) for k, v in table.items()}
        self.dependence_ = self._dependence.name

    # ------------------------------------------------------------- predict
    def _member_predictive(self, name: str, xs: Tensor) -> JointPredictive:
        """Predictive of one head in standardised units."""
        if name == "sklar":
            assert self._engine is not None and self._dependence is not None
            scores = self._dependence.predictive(xs.shape[0], xs)
            return SklarJoint(self._engine.predict(xs), scores)
        return self._heads[name].predictive(xs)

    def _base_predictive(self, x: Tensor) -> JointPredictive:
        """Stacked predictive in data units, before recalibration."""
        if self._x_stats is None or self._y_stats is None or self.weights_ is None:
            raise RuntimeError("Call fit() first.")
        xs = ((x - self._x_stats[0]) / self._x_stats[1]).to(self.device)
        with torch.no_grad():
            members = [self._member_predictive(name, xs) for name in self.members_]
        inner: JointPredictive
        if len(members) == 1:
            inner = members[0]
        else:
            keep = self.weights_ > 1e-4
            if not bool(keep.any()):
                keep = self.weights_ == self.weights_.max()
            kept = [m for m, k in zip(members, keep.tolist()) if k]
            weights = self.weights_[keep] / self.weights_[keep].sum()
            inner = kept[0] if len(kept) == 1 else MixtureJoint(kept, weights.to(self.device))
        loc, scale = self._y_stats
        if isinstance(inner, GaussianJoint):
            # Stay Gaussian in data units: closed-form marginals, conditionals and CDFs.
            return inner.affine(loc.to(self.device), scale.to(self.device))
        return AffineJoint(inner, loc.to(self.device), scale.to(self.device))

    def predict(self, X: ArrayLike) -> JointPredictive:
        """Joint predictive distributions for the rows of ``X`` (data units, recalibrated)."""
        x = torch.as_tensor(X, dtype=torch.float32)
        if x.dim() != 2 or x.shape[1] != self.n_features_in_:
            raise ValueError(f"X must be [n, {self.n_features_in_}]")
        base = self._base_predictive(x)
        return base if self._recalibrator is None else self._recalibrator.transform(base)

    # -------------------------------------------------------------- region
    def _score_for(self, predictive: JointPredictive, has_missing: bool) -> ConformalScore:
        if isinstance(self.score, ConformalScore):
            return self.score
        if self.score != "auto":
            raise ValueError("score must be 'auto' or a ConformalScore")
        if has_missing:
            return MahalanobisScore()
        if _is_gaussian(predictive):
            # The latent ball of a Gaussian is its highest-density region: exact and fast.
            return LatentScore()
        if predictive.has_density:
            return HPDScore()
        return SampleBallScore()

    def region(
        self,
        X: ArrayLike,
        alpha: Union[float, Sequence[float]] = 0.1,
        *,
        interest: Optional[Sequence[int]] = None,
    ) -> JointRegion:
        r"""Conformal prediction region for the rows of ``X``.

        Without missing targets in the conformal rows, the region has
        :math:`P(Y \in R(X)) \ge 1 - \alpha` for a test pair exchangeable with
        the conformal pairs.

        **Score.**  With ``score="auto"`` a Gaussian predictive (including its
        affine and recalibrated-scale forms) gets
        :class:`~torchregress.joint.LatentScore`: its latent ball is the
        highest-density region, computed exactly.  Every other predictive with a
        density (mixture, flow, Sklar, stack) gets
        :class:`~torchregress.joint.HPDScore`: the highest-density set, the
        smallest region at a given coverage when the model is right, which follows
        multimodal predictives (a latent ball pulled back through a flow can be
        several times larger).  HPD costs ``n_samples`` draws and densities per
        row; pass ``score=LatentScore()`` to the estimator for the fast choice on
        flows.  Draws-only predictives get
        :class:`~torchregress.joint.SampleBallScore`.  The score used is
        ``region.score`` (``region.score.name``).

        With missing targets in the conformal rows, two calibrations are kept:

        * ``region.contains(y)`` (no mask, complete test rows) is calibrated on
          the complete conformal rows only, so
          :math:`P(Y \in R(X) \mid \text{row complete}) \ge 1 - \alpha` when the
          complete conformal rows and the test row are exchangeable (for
          example when targets are missing completely at random).  Volumes,
          samples and boxes refer to this region.  A ``ValueError`` is raised
          when there are too few complete conformal rows for the requested
          ``alpha`` (fewer than :math:`1/\alpha - 1`).
        * ``region.contains(y, mask=m)`` uses the masked calibration on every
          conformal row (the :math:`\chi^2`-scaled
          :class:`~torchregress.joint.MahalanobisScore` of the observed
          sub-vector, Braun et al. 2025), whose guarantee holds for test rows
          whose missingness is generated by the same mechanism as in the
          conformal rows.  It is *not* a guarantee for complete rows passed
          with an all-true mask.

        Args:
            X: Features ``[n, p]``.
            alpha: Miscoverage level(s).
            interest: Targets of interest.  The region is then built for that
                sub-vector, with the other (nuisance) targets integrated out of the
                predictive before calibration.
        """
        if self._conformal_rows is None:
            raise RuntimeError("Call fit() first.")
        dims = None if interest is None else tuple(int(i) for i in interest)
        levels = (
            (float(alpha),) if isinstance(alpha, (int, float)) else tuple(float(a) for a in alpha)
        )
        key = (levels, dims, isinstance(alpha, (int, float)), self.score)
        x_cal, y_cal, m_cal = self._conformal_rows

        def restrict(pred: JointPredictive) -> JointPredictive:
            return pred if dims is None else pred.marginal(list(dims))

        if key not in self._conformal_cache:
            y_c = (y_cal if dims is None else y_cal[:, list(dims)]).to(self.device)
            m_c = (m_cal if dims is None else m_cal[:, list(dims)]).to(self.device)
            complete = m_c.all(1)
            masked_conformal: Optional[JointConformal] = None
            if bool(complete.all()):
                cal = restrict(self.predict(x_cal))
                rows_x, rows_y = x_cal, y_c
            else:
                n_complete = int(complete.sum())
                need = math.ceil(1.0 / min(levels) - 1.0 - 1e-9)
                if n_complete < max(need, 1):
                    raise ValueError(
                        f"only {n_complete} conformal rows have every target observed; a "
                        f"complete-row region at alpha = {min(levels)} needs at least "
                        f"{max(need, 1)} (raise conformal_fraction or alpha)"
                    )
                full = restrict(self.predict(x_cal))
                masked_score = (
                    self.score
                    if isinstance(self.score, ConformalScore) and self.score.supports_mask
                    else MahalanobisScore()
                )
                if not isinstance(self.score, ConformalScore) or self.score.supports_mask:
                    masked_conformal = JointConformal(masked_score, alpha)
                    masked_conformal.calibrate(full, y_c, mask=m_c)
                keep = complete.cpu()
                rows_x, rows_y = x_cal[keep], y_c[complete]
                cal = restrict(self.predict(rows_x))
            conformal = JointConformal(self._score_for(cal, False), alpha)
            conformal.calibrate(cal, rows_y)
            self._conformal_cache[key] = (conformal, masked_conformal)
        conformal, masked_conformal = self._conformal_cache[key]
        test = restrict(self.predict(X))
        region = conformal.region(test)
        if masked_conformal is not None:
            region.mask_region = masked_conformal.region(test)
        return region


def _is_gaussian(predictive: JointPredictive) -> bool:
    """A Gaussian predictive, possibly through per-target affine maps (units, scale)."""
    while isinstance(predictive, AffineJoint):
        predictive = predictive.base
    return isinstance(predictive, GaussianJoint)


def _stack(log_density: Tensor) -> Tensor:
    """Stacking weights ``[K]`` from validation log densities ``[N, K]``.

    Rows with a ``NaN`` are left out, and members that are ``NaN`` on every row
    get weight zero.  ``-inf`` (a member with zero density on a row) is kept:
    EM gives that member no responsibility there.  Rows where every remaining
    member is ``-inf`` carry no information and are left out.  Uniform weights
    (over the usable members) when nothing usable remains.
    """
    k = log_density.shape[1]
    usable = ~torch.isnan(log_density).all(0)
    if not bool(usable.any()):
        return torch.full((k,), 1.0 / k, dtype=log_density.dtype)
    sub = log_density[:, usable]
    rows = ~torch.isnan(sub).any(1) & (sub > -math.inf).any(1)
    weights = torch.zeros(k, dtype=log_density.dtype, device=log_density.device)
    if not bool(rows.any()):
        weights[usable] = 1.0 / int(usable.sum())
    else:
        weights[usable] = _stacking_weights(sub[rows])
    return weights
