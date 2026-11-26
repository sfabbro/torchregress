r"""Neyman confidence sets with parameter-dependent critical values (LF2I).

For a parameter :math:`\theta \in \mathbb R^P`, an observation :math:`x` and any
test statistic :math:`\lambda(\theta; x)` that is *large* when :math:`\theta` is
incompatible with :math:`x` (minus a log-likelihood ratio, the WALDO statistic,
a negative log posterior, ...), the Neyman construction

.. math::

    R(x) = \{\theta : \lambda(\theta; x) \le C_\alpha(\theta)\},
    \qquad
    C_\alpha(\theta) = F^{-1}_{\lambda(\theta; X)}(1 - \alpha),
    \quad X \sim p(x \mid \theta),

has coverage :math:`P_{X\mid\theta}(\theta \in R(X)) \ge 1 - \alpha` at *every*
:math:`\theta`, whatever the distribution of :math:`\theta` in the population.
Likelihood-free frequentist inference (LF2I; Dalmasso et al. 2024) estimates the
critical-value function :math:`C_\alpha` by quantile regression of
:math:`\lambda_i = \lambda(\theta_i; x_i)` on :math:`\theta_i` over a calibration
set of pairs :math:`(\theta_i, x_i)` drawn from :math:`p(x \mid \theta)`.  The
calibration distribution of :math:`\theta` only affects how precisely
:math:`C_\alpha` is estimated (where it has support), not the target: the method
is prior-free.  Here no simulator is needed; the calibration pairs are labelled
data that follow the same :math:`p(x \mid \theta)` as the test data.

This module provides

* :class:`CriticalValueRegressor` -- the critical-value regression, with a
  non-crossing multi-level quantile MLP (pinball loss) and a k-nearest-neighbour
  conditional finite-sample quantile;
* :class:`NeymanConfidenceSet` -- membership tests and candidate-bank regions
  for any statistic;
* :func:`binned_conditional_coverage` and :func:`smooth_conditional_coverage`
  -- LF2I's diagnostic: estimate :math:`P(\theta \in R(X) \mid \theta)` on a
  held-out set and report where it is significantly below nominal.

A constant critical value (the split-conformal threshold) only guarantees
*marginal* coverage, averaged over the calibration population of
:math:`\theta`; when the distribution of :math:`\lambda(\theta; X)` depends on
:math:`\theta` it over-covers some regions and under-covers others, typically
the rare ones.

References
----------
.. [1] Dalmasso, N., Masserano, L., Zhao, D., Izbicki, R., & Lee, A. B. (2024).
   Likelihood-free frequentist inference: bridging classical statistics and
   machine learning for reliable simulator-based inference. *Electronic
   Journal of Statistics*. https://arxiv.org/abs/2107.03920
.. [2] Masserano, L., Dorigo, T., Izbicki, R., Kuusela, M., & Lee, A. B. (2023).
   Simulator-based inference with WALDO: confidence regions by leveraging
   prediction algorithms and posterior estimators for inverse problems.
   *AISTATS*. https://arxiv.org/abs/2205.15680
.. [3] Koenker, R., & Bassett, G. (1978). Regression quantiles. *Econometrica*,
   46(1), 33-50.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..utils.distributions import inverse_softplus
from ..utils.quantile import multi_quantile_loss

__all__ = [
    "BinnedCoverage",
    "CriticalValueRegressor",
    "NeymanConfidenceSet",
    "SmoothCoverage",
    "binned_conditional_coverage",
    "smooth_conditional_coverage",
]

# Elements per vectorised chunk in the k-NN distance computation.
_ELEMENT_BUDGET = 1 << 22

# Statistic evaluations per chunk when the k-NN estimator re-scores neighbours.
_RESCORE_BUDGET = 1 << 18

AlphaLike = Union[float, Sequence[float]]
CriticalValueFn = Callable[[Tensor], Tensor]
StatisticFn = Callable[[Tensor, Tensor], Tensor]


# ═══════════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _as_2d(theta: Tensor, name: str = "theta") -> Tensor:
    if not isinstance(theta, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if theta.dim() == 1:
        theta = theta.unsqueeze(-1)
    if theta.dim() != 2:
        raise ValueError(f"{name} must be [N] or [N, P], got shape {tuple(theta.shape)}")
    if not theta.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    return theta


def _parse_alpha(alpha: AlphaLike) -> tuple[list[float], bool]:
    """Miscoverage levels as a list, and whether a scalar was given."""
    scalar = isinstance(alpha, (int, float))
    values = [float(alpha)] if scalar else [float(a) for a in alpha]  # type: ignore[union-attr]
    if not values:
        raise ValueError("alpha must contain at least one level")
    for a in values:
        if not 0.0 < a < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {a}")
    if len(set(values)) != len(values):
        raise ValueError(f"alpha levels must be distinct, got {values}")
    return values, scalar


def _standardizer(theta: Tensor) -> tuple[Tensor, Tensor]:
    """Column mean and standard deviation (constant columns get scale 1)."""
    mean = theta.mean(dim=0)
    std = theta.std(dim=0, unbiased=False) if theta.shape[0] > 1 else torch.ones_like(mean)
    std = torch.where(std > 0, std, torch.ones_like(std))
    return mean, std


def _finite_sample_index(n: int, alphas: Tensor) -> Tensor:
    """0-based order-statistic index ``ceil((n+1)(1-alpha)) - 1`` per level."""
    return torch.ceil((n + 1) * (1.0 - alphas)).long() - 1


def _confidence_index(n: int, alphas: Tensor, confidence: float) -> Tensor:
    """0-based order-statistic index that upper-bounds each quantile with confidence.

    For i.i.d. draws, ``#{X_i <= q} ~ Binomial(n, 1 - alpha)`` where ``q`` is the
    ``1 - alpha`` quantile, so the smallest ``m`` with ``P(Bin <= m) >= confidence``
    gives ``P(X_(m+1) >= q) >= confidence`` (a distribution-free upper confidence
    bound for a quantile; ``m = n`` means that no order statistic certifies it).
    """
    j = torch.arange(n + 1, dtype=torch.float64)
    out = []
    for alpha in alphas.tolist():
        level = 1.0 - alpha
        log_pmf = (
            torch.lgamma(torch.tensor(n + 1.0, dtype=torch.float64))
            - torch.lgamma(j + 1.0)
            - torch.lgamma(n - j + 1.0)
            + j * math.log(level)
            + (n - j) * math.log1p(-level)
        )
        cdf = torch.cumsum(torch.exp(log_pmf), dim=0)
        # 1e-12 absorbs the rounding of the cumulative sum at the last term.
        hit = torch.nonzero(cdf >= confidence - 1e-12)
        out.append(int(hit[0]) if hit.numel() else n)
    return torch.tensor(out, dtype=torch.long)


def _build_mlp(in_dim: int, hidden: Sequence[int], n_levels: int) -> nn.Sequential:
    """SiLU MLP with ``n_levels`` raw outputs; the last layer is the head."""
    layers: list[nn.Module] = []
    width = in_dim
    for h in hidden:
        layers += [nn.Linear(width, int(h)), nn.SiLU()]
        width = int(h)
    layers.append(nn.Linear(width, n_levels))
    return nn.Sequential(*layers)


def _non_crossing(raw: Tensor) -> Tensor:
    """Lowest quantile plus cumulative softplus increments: non-decreasing levels."""
    if raw.shape[-1] == 1:
        return raw
    steps = F.softplus(raw[:, 1:])
    return torch.cat([raw[:, :1], raw[:, :1] + torch.cumsum(steps, dim=-1)], dim=-1)


# ═══════════════════════════════════════════════════════════════════════════════
# critical-value regression
# ═══════════════════════════════════════════════════════════════════════════════


class CriticalValueRegressor:
    r"""Estimate :math:`C_\alpha(\theta)`, the :math:`1-\alpha` quantile of a statistic.

    Fit on calibration pairs ``theta [N, P]`` and ``statistic [N]`` (the
    statistic evaluated at each row's own parameter), then predict the
    critical value at new parameters.  Several confidence levels can be fitted
    jointly; their predictions never cross.

    Args:
        alpha: Miscoverage level(s) in ``(0, 1)``; a scalar gives ``[M]``
            predictions, a sequence gives ``[M, L]`` with columns in the order
            given (the confidence levels are ``1 - alpha``).
        method: ``"mlp"`` (quantile-regression network, the default) or
            ``"knn"`` (k-nearest-neighbour finite-sample quantile: no training,
            assumption-light, a cross-check; less accurate than the MLP in the
            tests, markedly so for ``P >= 2``).
        hidden: MLP hidden-layer widths.
        max_epochs: Maximum MLP training epochs.
        batch_size: MLP mini-batch size.
        lr: Adam learning rate.
        weight_decay: Adam weight decay.
        val_fraction: Fraction of the calibration rows held out for early
            stopping on the validation pinball loss (``0`` disables early
            stopping and trains for ``max_epochs``).
        patience: Epochs without validation improvement before stopping; the
            learning rate is halved after every ``patience // 3`` of them
            (reduce-on-plateau), and the best validation checkpoint is kept.
        n_neighbors: k for the k-NN estimator.  ``None`` uses
            ``ceil(N ** (4 / (4 + P)) / 2)``, raised to ``ceil(10 / min(alpha))``
            so that about ten neighbours lie above the highest quantile, and
            capped at ``N``.  The exponent balances the ``O(1/k)`` variance
            against the ``O((k/N)^(2/P))`` squared bias of a twice-differentiable
            quantile function; the factor 1/2 was the best of {1/4, 1/2, 1} on
            the heteroscedastic benchmark of the test-suite.
        transform: Monotone reparametrisation of the statistic for the MLP
            fit.  ``"log"`` fits the quantiles of ``log(lambda + c)`` with
            ``c = 0.01 * mean(lambda)`` (requires ``lambda >= 0``); ``"none"``
            fits ``lambda`` itself; ``"auto"`` (default) uses ``"log"`` when
            every calibration statistic is non-negative.  Quantiles are
            equivariant under increasing maps, so this changes how easily the
            network represents ``C(theta)`` (a statistic whose scale varies by
            orders of magnitude across ``theta`` becomes additive), not what
            is estimated.  The k-NN estimator is equivariant and ignores it.
        seed: Seed for the MLP initialisation, validation split and shuffling.
        statistic_fn: k-NN only.  ``statistic_fn(theta, x) -> [n, K]`` in the
            convention of :meth:`NeymanConfidenceSet.region` (``theta`` is
            ``[n, K, P]``, one candidate row per observation row here, ``K = 1``).
            With it (and ``x`` passed to :meth:`fit`), the critical value at a
            query :math:`\theta` is the order statistic of
            :math:`\lambda(\theta; x_i)` over the k neighbours: their
            *observations* are re-scored at the query parameter instead of
            reusing :math:`\lambda(\theta_i; x_i)`.  The reference distribution
            then differs from the exact one only through
            :math:`p(x \mid \theta_i)` vs :math:`p(x \mid \theta)`, which is
            what makes it valid for **discrete** statistics (counts): their
            exact critical value is a saw-tooth in :math:`\theta` that no
            pooled or smooth estimate can follow, and missing one tooth drops
            a whole atom of probability from the set.
        confidence: k-NN only.  ``None`` (default) uses the finite-sample
            order statistic ``ceil((k+1)(1-alpha))``, whose coverage is at
            least ``1 - alpha`` *on average over calibration sets* and so
            fluctuates around nominal by about ``sqrt(alpha(1-alpha)/k)`` from
            one :math:`\theta` to the next.  A value in ``(0, 1)`` uses the
            binomial upper confidence bound for the quantile instead: at each
            query, with probability at least ``confidence`` over the
            calibration draw, the critical value is no smaller than the
            ``1 - alpha`` quantile of the neighbours' reference distribution
            (``+inf`` when k is too small to certify it).  The sets are then
            conservative by about ``z * sqrt(alpha(1-alpha)/k)`` in coverage.

    The inputs are standardised (column mean and standard deviation of the
    calibration ``theta``; the statistic too for the MLP).  The MLP head
    predicts the lowest quantile plus non-negative (softplus) increments, so
    levels cannot cross; it is initialised at the constant (marginal)
    quantiles, so training starts from the split-conformal threshold and only
    learns the :math:`\theta` dependence.  The k-NN estimator returns, for each
    query, the order statistic ``ceil((k+1)(1-alpha))`` of its k neighbours'
    statistics (``+inf`` when ``alpha < 1/(k+1)``), which is monotone in the
    level by construction.

    Fitted state lives on the device and dtype of the calibration tensors;
    :meth:`predict` returns values on the device and dtype of the query.

    Note:
        The construction's guarantee is only as good as the estimate of
        :math:`C_\alpha`: validate with :func:`binned_conditional_coverage` /
        :func:`smooth_conditional_coverage` on held-out data.  Predictions far
        outside the calibration support are extrapolations.
    """

    def __init__(
        self,
        alpha: AlphaLike = 0.1,
        *,
        method: Literal["mlp", "knn"] = "mlp",
        hidden: Sequence[int] = (64, 64),
        max_epochs: int = 300,
        batch_size: int = 512,
        lr: float = 3e-3,
        weight_decay: float = 0.0,
        val_fraction: float = 0.2,
        patience: int = 25,
        n_neighbors: Optional[int] = None,
        transform: Literal["auto", "none", "log"] = "auto",
        seed: int = 0,
        statistic_fn: Optional[StatisticFn] = None,
        confidence: Optional[float] = None,
    ) -> None:
        self._alphas, self._scalar = _parse_alpha(alpha)
        if method not in ("mlp", "knn"):
            raise ValueError(f"method must be 'mlp' or 'knn', got {method!r}")
        if any(int(h) < 1 for h in hidden):
            raise ValueError("hidden widths must be positive")
        if int(max_epochs) < 1 or int(batch_size) < 1 or int(patience) < 1:
            raise ValueError("max_epochs, batch_size and patience must be positive")
        if not 0.0 <= float(val_fraction) < 1.0:
            raise ValueError("val_fraction must be in [0, 1)")
        if not float(lr) > 0.0 or float(weight_decay) < 0.0:
            raise ValueError("lr must be positive and weight_decay non-negative")
        if transform not in ("auto", "none", "log"):
            raise ValueError(f"transform must be 'auto', 'none' or 'log', got {transform!r}")
        if n_neighbors is not None and int(n_neighbors) < 1:
            raise ValueError("n_neighbors must be None or a positive integer")
        if statistic_fn is not None and not callable(statistic_fn):
            raise TypeError("statistic_fn must be None or a callable")
        if confidence is not None and not 0.0 < float(confidence) < 1.0:
            raise ValueError(f"confidence must be None or in (0, 1), got {confidence}")
        if method != "knn" and (statistic_fn is not None or confidence is not None):
            raise ValueError("statistic_fn and confidence apply to method='knn' only")
        self.method = method
        self.hidden = tuple(int(h) for h in hidden)
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.val_fraction = float(val_fraction)
        self.patience = int(patience)
        self.n_neighbors = None if n_neighbors is None else int(n_neighbors)
        self.transform = transform
        self.seed = int(seed)
        self.statistic_fn = statistic_fn
        self.confidence = None if confidence is None else float(confidence)

        self.method_: Optional[str] = None
        self.n_features_: Optional[int] = None
        self.n_neighbors_: Optional[int] = None
        self.best_epoch_: Optional[int] = None
        self.history_: dict[str, list[float]] = {"train": [], "val": []}
        self._theta_mean: Optional[Tensor] = None
        self._theta_std: Optional[Tensor] = None
        self._stat_mean: Optional[Tensor] = None
        self._stat_std: Optional[Tensor] = None
        self.transform_: Optional[str] = None
        self._offset: float = 0.0
        self._model: Optional[nn.Sequential] = None
        self._z_cal: Optional[Tensor] = None
        self._stat_cal: Optional[Tensor] = None
        self._x_cal: Optional[Tensor] = None

    # ------------------------------------------------------------------ props
    @property
    def alpha(self) -> Union[float, list[float]]:
        """Miscoverage level(s) as given."""
        return self._alphas[0] if self._scalar else list(self._alphas)

    @property
    def levels(self) -> list[float]:
        """Confidence levels ``1 - alpha`` (prediction-column order)."""
        return [1.0 - a for a in self._alphas]

    @property
    def is_fitted(self) -> bool:
        return self.method_ is not None

    # -------------------------------------------------------------------- fit
    def fit(
        self, theta: Tensor, statistic: Tensor, x: Optional[Tensor] = None
    ) -> "CriticalValueRegressor":
        """Fit the critical-value function.

        Args:
            theta: Calibration parameters ``[N, P]`` (or ``[N]`` for ``P = 1``).
            statistic: ``lambda(theta_i; x_i)`` per row, ``[N]``; finite.
            x: Calibration observations with ``N`` rows; required (and only
                used) with ``statistic_fn``.

        Returns:
            ``self``.
        """
        if (x is None) != (self.statistic_fn is None):
            raise ValueError("x is required with statistic_fn, and only used with it")
        if x is not None and x.shape[0] != _as_2d(theta).shape[0]:
            raise ValueError("x and theta must agree on the number of rows")
        theta = _as_2d(theta).detach()
        statistic = statistic.detach().reshape(-1).to(theta.device, theta.dtype)
        n, p = theta.shape
        if statistic.numel() != n:
            raise ValueError("theta and statistic must agree on the number of rows")
        if n < 2:
            raise ValueError("at least two calibration rows are required")
        if not bool(torch.isfinite(theta).all()) or not bool(torch.isfinite(statistic).all()):
            raise ValueError("theta and statistic must be finite")
        method = self.method
        self._theta_mean, self._theta_std = _standardizer(theta)
        z = (theta - self._theta_mean) / self._theta_std
        self.n_features_ = p
        if method == "knn":
            k = self.n_neighbors
            if k is None:
                k = max(
                    math.ceil(n ** (4.0 / (4.0 + p)) / 2.0), math.ceil(10.0 / min(self._alphas))
                )
            self.n_neighbors_ = min(int(k), n)
            self._z_cal, self._stat_cal = z, statistic
            self._x_cal = None if x is None else x.detach()
            self._model = None
        else:
            self._fit_mlp(z, statistic)
            self._z_cal, self._stat_cal, self._x_cal = None, None, None
        self.method_ = method
        return self

    def _sorted_alphas(self) -> tuple[list[float], Tensor]:
        """Alphas in ascending quantile-level order and the inverse permutation."""
        order = sorted(range(len(self._alphas)), key=lambda i: -self._alphas[i])
        inverse = torch.empty(len(order), dtype=torch.long)
        inverse[torch.tensor(order)] = torch.arange(len(order))
        return [self._alphas[i] for i in order], inverse

    def _fit_mlp(self, z: Tensor, statistic: Tensor) -> None:
        device, dtype = z.device, z.dtype
        n, p = z.shape
        alphas, _ = self._sorted_alphas()
        levels = torch.tensor([1.0 - a for a in alphas], device=device, dtype=dtype)

        nonneg = bool((statistic >= 0).all())
        mean_stat = float(statistic.mean())
        transform = self.transform
        if transform == "auto":
            transform = "log" if nonneg and mean_stat > 0 else "none"
        if transform == "log":
            if not nonneg or not mean_stat > 0:
                raise ValueError("transform='log' needs non-negative statistics with mean > 0")
            self._offset = 0.01 * mean_stat
            target = torch.log(statistic + self._offset)
        else:
            self._offset = 0.0
            target = statistic
        self.transform_ = transform
        self._stat_mean, self._stat_std = _standardizer(target.unsqueeze(-1))
        y = (target - self._stat_mean[0]) / self._stat_std[0]

        gen = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(n, generator=gen).to(device)
        n_val = int(round(self.val_fraction * n))
        if n_val < 1 or n - n_val < 2:
            n_val = 0
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        model = _build_mlp(p, self.hidden, len(alphas))
        head = model[-1]
        assert isinstance(head, nn.Linear)
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    bound = 1.0 / math.sqrt(module.in_features)
                    module.weight.copy_(
                        torch.rand(module.weight.shape, generator=gen) * 2 * bound - bound
                    )
                    module.bias.copy_(
                        torch.rand(module.bias.shape, generator=gen) * 2 * bound - bound
                    )
            # Start at the constant (marginal) quantiles: the network learns the
            # theta dependence on top of the split-conformal threshold.
            q0 = torch.quantile(y[train_idx].cpu().double(), levels.cpu().double())
            bias = q0.clone()
            if q0.numel() > 1:
                gaps = (q0[1:] - q0[:-1]).clamp_min(1e-3)
                bias[1:] = inverse_softplus(gaps)
            head.weight.zero_()
            head.bias.copy_(bias.to(head.bias.dtype))
        model = model.to(device=device, dtype=dtype)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_val = math.inf
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = 0
        wait = 0
        history: dict[str, list[float]] = {"train": [], "val": []}
        n_train = train_idx.numel()
        for epoch in range(self.max_epochs):
            model.train()
            order = train_idx[torch.randperm(n_train, generator=gen).to(device)]
            total = torch.zeros((), device=device, dtype=dtype)
            for start in range(0, n_train, self.batch_size):
                idx = order[start : start + self.batch_size]
                loss = multi_quantile_loss(_non_crossing(model(z[idx])), y[idx], levels).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total = total + loss.detach() * idx.numel()
            history["train"].append(float(total) / n_train)
            if n_val == 0:
                best_epoch = epoch
                continue
            model.eval()
            with torch.no_grad():
                val = float(
                    multi_quantile_loss(_non_crossing(model(z[val_idx])), y[val_idx], levels).mean()
                )
            history["val"].append(val)
            if val < best_val:
                best_val, best_epoch, wait = val, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                wait += 1
                if wait >= self.patience:
                    break
                if wait % max(1, self.patience // 3) == 0:
                    for group in optimizer.param_groups:
                        group["lr"] *= 0.5
        if n_val > 0:
            model.load_state_dict(best_state)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self._model = model
        self.best_epoch_ = best_epoch
        self.history_ = history

    # ---------------------------------------------------------------- predict
    def predict(self, theta: Tensor) -> Tensor:
        """Critical values at ``theta [M, P]``: ``[M]`` (scalar alpha) or ``[M, L]``."""
        if not self.is_fitted or self._theta_mean is None or self._theta_std is None:
            raise RuntimeError("Call fit() first.")
        query = _as_2d(theta)
        if query.shape[1] != self.n_features_:
            raise ValueError(f"theta must have {self.n_features_} columns, got {query.shape[1]}")
        mean, std = self._theta_mean, self._theta_std
        z = (query.detach().to(mean.device, mean.dtype) - mean) / std
        if self.method_ == "knn":
            out = self._predict_knn(z, query.detach())
        else:
            out = self._predict_mlp(z)
        out = out.to(query.device, query.dtype)
        return out[:, 0] if self._scalar else out

    def predict_critical(self, features: Tensor) -> Tensor:
        """Single-level critical values ``[M]``; the WALDO regressor interface.

        Lets the regressor be passed as ``critical_value_regressor`` to
        :class:`~torchregress.losses.WaldoConformal` and its sample-based
        variants, which call ``fit(target, scores)`` and
        ``predict_critical(target)``.
        """
        if len(self._alphas) != 1:
            raise ValueError("predict_critical needs a single alpha level; use predict()")
        return self.predict(features).reshape(-1)

    def _predict_mlp(self, z: Tensor) -> Tensor:
        if self._model is None or self._stat_mean is None or self._stat_std is None:
            raise RuntimeError("Call fit() first.")
        _, inverse = self._sorted_alphas()
        with torch.no_grad():
            pred = _non_crossing(self._model(z)) * self._stat_std[0] + self._stat_mean[0]
            if self.transform_ == "log":
                # exp is increasing and the clamp is monotone: levels stay ordered.
                pred = (torch.exp(pred) - self._offset).clamp_min(0.0)
        return pred[:, inverse.to(pred.device)]

    def _predict_knn(self, z: Tensor, query: Tensor) -> Tensor:
        if self._z_cal is None or self._stat_cal is None or self.n_neighbors_ is None:
            raise RuntimeError("Call fit() first.")
        k = self.n_neighbors_
        alphas = torch.tensor(self._alphas, dtype=torch.float64)
        if self.confidence is None:
            index = _finite_sample_index(k, alphas)
        else:
            index = _confidence_index(k, alphas, self.confidence)
        infinite = (index >= k).to(z.device)
        safe = index.clamp_max(k - 1).to(z.device)
        rows = max(1, _ELEMENT_BUDGET // max(self._z_cal.shape[0], 1))
        if self.statistic_fn is not None:
            # Each row re-scores k observations: bound the statistic batch too.
            rows = max(1, min(rows, _RESCORE_BUDGET // k))
        parts = []
        for start in range(0, z.shape[0], rows):
            dist = torch.cdist(z[start : start + rows], self._z_cal)
            nearest = torch.topk(dist, k, dim=1, largest=False).indices
            if self.statistic_fn is None or self._x_cal is None:
                stats = self._stat_cal[nearest]
            else:
                stats = self._rescore(query[start : start + rows], nearest)
            values = torch.sort(stats, dim=1).values
            crit = values[:, safe]
            parts.append(torch.where(infinite, torch.full_like(crit, math.inf), crit))
        if not parts:
            return torch.empty(0, len(self._alphas), device=z.device, dtype=z.dtype)
        return torch.cat(parts)

    def _rescore(self, query: Tensor, nearest: Tensor) -> Tensor:
        """``lambda(theta_m; x_i)`` for the neighbours ``i`` of each query ``m``: ``[M, k]``."""
        assert self.statistic_fn is not None and self._x_cal is not None
        assert self._stat_cal is not None
        m, k = nearest.shape
        x = self._x_cal[nearest.reshape(-1).to(self._x_cal.device)]
        bank = query.to(x.device).unsqueeze(1).expand(m, k, query.shape[1]).reshape(m * k, 1, -1)
        with torch.no_grad():
            stat = self.statistic_fn(bank, x)
        if stat.shape != (m * k, 1):
            raise ValueError(
                f"statistic_fn must return [n, K] = [{m * k}, 1], got {tuple(stat.shape)}"
            )
        if not bool(torch.isfinite(stat).all()):
            raise ValueError("statistic_fn returned non-finite values at a query parameter")
        return stat.reshape(m, k).to(self._stat_cal.device, self._stat_cal.dtype)


# ═══════════════════════════════════════════════════════════════════════════════
# confidence sets
# ═══════════════════════════════════════════════════════════════════════════════


class NeymanConfidenceSet:
    r"""Neyman confidence set :math:`\{\theta : \lambda(\theta; x) \le C(\theta)\}`.

    Agnostic to where the statistic comes from: pass statistic values already
    computed at candidate parameters (:meth:`contains`), or a callable that
    evaluates the statistic on a candidate bank (:meth:`region`).  The
    statistic must be oriented so that *large* values reject (e.g. minus a
    log-likelihood ratio, or the WALDO statistic).

    Args:
        critical_value: A fitted :class:`CriticalValueRegressor`, or any
            callable ``theta [M, P] -> [M]`` (or ``[M, L]``) giving critical
            values.

    Example:
        >>> cs = NeymanConfidenceSet.calibrate(theta_cal, stat_cal, alpha=0.32)
        >>> covered = cs.contains(stat_test, theta_test)            # [N] bool
        >>> masks = cs.region(statistic_fn, theta_bank, x_test)     # [N, K] bool
    """

    def __init__(self, critical_value: Union[CriticalValueRegressor, CriticalValueFn]) -> None:
        if isinstance(critical_value, CriticalValueRegressor):
            if not critical_value.is_fitted:
                raise ValueError("the CriticalValueRegressor must be fitted")
        elif not callable(critical_value):
            raise TypeError("critical_value must be a CriticalValueRegressor or a callable")
        self.critical_value_model = critical_value

    @classmethod
    def calibrate(
        cls,
        theta: Tensor,
        statistic: Tensor,
        alpha: AlphaLike = 0.1,
        *,
        x: Optional[Tensor] = None,
        **regressor_kwargs: Any,
    ) -> "NeymanConfidenceSet":
        """Fit a :class:`CriticalValueRegressor` on calibration pairs and wrap it.

        Args:
            theta: Calibration parameters ``[N, P]``.
            statistic: ``lambda(theta_i; x_i)`` per row, ``[N]``.
            alpha: Miscoverage level(s).
            x: Calibration observations ``[N, ...]``, with
                ``method="knn", statistic_fn=...`` (re-scored neighbours; use
                this for count data and other discrete statistics).
            **regressor_kwargs: Forwarded to :class:`CriticalValueRegressor`.
        """
        regressor = CriticalValueRegressor(alpha, **regressor_kwargs)
        return cls(regressor.fit(theta, statistic, x))

    def critical_value(self, theta: Tensor) -> Tensor:
        """Critical values at ``theta [..., P]``: shape ``[...]`` or ``[..., L]``."""
        if theta.dim() < 1:
            raise ValueError("theta must have at least one dimension")
        lead = theta.shape[:-1]
        flat = theta.reshape(-1, theta.shape[-1])
        model = self.critical_value_model
        crit = model.predict(flat) if isinstance(model, CriticalValueRegressor) else model(flat)
        if crit.dim() == 1:
            return crit.reshape(lead)
        return crit.reshape(*lead, crit.shape[-1])

    def contains(self, statistic: Tensor, theta: Tensor) -> Tensor:
        """Membership ``statistic <= C(theta)``.

        Args:
            statistic: Statistic values ``[...]`` at the candidate parameters.
            theta: Candidate parameters ``[..., P]``; for ``P = 1`` a tensor of
                the same shape as ``statistic`` is also accepted.

        Returns:
            Boolean ``[...]``, or ``[..., L]`` for a multi-level regressor.
        """
        if theta.shape == statistic.shape:
            theta = theta.unsqueeze(-1)
        if theta.shape[:-1] != statistic.shape:
            raise ValueError(
                "theta must be [..., P] with leading shape equal to statistic's shape, got "
                f"{tuple(theta.shape)} and {tuple(statistic.shape)}"
            )
        crit = self.critical_value(theta).to(statistic.device, statistic.dtype)
        if crit.shape == statistic.shape:
            return statistic <= crit
        return statistic.unsqueeze(-1) <= crit

    def region(
        self,
        statistic_fn: Callable[[Tensor, Tensor], Tensor],
        theta_bank: Tensor,
        x: Tensor,
        *,
        chunk_size: Optional[int] = None,
    ) -> Tensor:
        """Confidence-set membership masks over a candidate bank.

        Args:
            statistic_fn: ``statistic_fn(theta_bank, x) -> [n, K]`` returning
                ``lambda(theta_k; x_i)``; it receives the shared bank ``[K, P]``
                or the matching rows ``[n, K, P]`` of a per-row bank, and a
                row slice of ``x``.
            theta_bank: Candidate parameters, shared ``[K, P]`` or per row
                ``[N, K, P]``.
            x: Observations with ``N`` rows (first dimension).
            chunk_size: Rows of ``x`` per ``statistic_fn`` call (default: all).

        Returns:
            Boolean ``[N, K]`` (or ``[N, K, L]`` for a multi-level regressor).
        """
        if theta_bank.dim() not in (2, 3):
            raise ValueError("theta_bank must be [K, P] or [N, K, P]")
        n = x.shape[0]
        if theta_bank.dim() == 3 and theta_bank.shape[0] != n:
            raise ValueError("a per-row theta_bank must have x.shape[0] rows")
        if chunk_size is not None and int(chunk_size) < 1:
            raise ValueError("chunk_size must be None or a positive integer")
        rows = n if chunk_size is None else int(chunk_size)
        shared = theta_bank.dim() == 2
        crit_shared = self.critical_value(theta_bank) if shared else None
        parts = []
        for start in range(0, n, max(rows, 1)):
            bank = theta_bank if shared else theta_bank[start : start + rows]
            stat = statistic_fn(bank, x[start : start + rows])
            k = theta_bank.shape[-2]
            if stat.shape != (min(rows, n - start), k):
                raise ValueError(
                    f"statistic_fn must return [n, K] = [{min(rows, n - start)}, {k}], "
                    f"got {tuple(stat.shape)}"
                )
            crit = crit_shared if crit_shared is not None else self.critical_value(bank)
            crit = crit.to(stat.device, stat.dtype)
            if shared:
                crit = crit.unsqueeze(0)
            parts.append(stat <= crit if crit.dim() == 2 else stat.unsqueeze(-1) <= crit)
        if not parts:
            return torch.zeros(0, theta_bank.shape[-2], dtype=torch.bool, device=theta_bank.device)
        return torch.cat(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# conditional-coverage diagnostics
# ═══════════════════════════════════════════════════════════════════════════════


def _work_dtype(device: torch.device) -> torch.dtype:
    return torch.float32 if device.type == "mps" else torch.float64


def _coverage_inputs(theta: Tensor, covered: Tensor, nominal: float) -> tuple[Tensor, Tensor]:
    theta = _as_2d(theta)
    if not 0.0 < float(nominal) < 1.0:
        raise ValueError(f"nominal must be in (0, 1), got {nominal}")
    covered = covered.reshape(-1)
    if covered.numel() != theta.shape[0]:
        raise ValueError("theta and covered must agree on the number of rows")
    if theta.shape[0] < 2:
        raise ValueError("at least two rows are required")
    if not bool(torch.isfinite(theta).all()):
        raise ValueError("theta must be finite")
    dtype = _work_dtype(theta.device)
    return theta.to(dtype), covered.to(theta.device, dtype)


@dataclass(frozen=True)
class BinnedCoverage:
    """Empirical coverage in bins of one parameter, with Wilson intervals.

    Attributes:
        parameter: Column index of ``theta``.
        edges: Bin edges ``[B + 1]``.
        count: Rows per bin ``[B]``.
        coverage: Fraction covered per bin (``nan`` for empty bins).
        lower, upper: Wilson confidence interval per bin (``nan`` if empty).
        under_covered: Bins whose upper bound is below nominal.
        over_covered: Bins whose lower bound is above nominal.
        nominal: Target coverage.
    """

    parameter: int
    edges: Tensor
    count: Tensor
    coverage: Tensor
    lower: Tensor
    upper: Tensor
    under_covered: Tensor
    over_covered: Tensor
    nominal: float


def _wilson(k: Tensor, n: Tensor, z: float) -> tuple[Tensor, Tensor]:
    """Wilson score interval for ``k`` successes in ``n`` trials (``nan`` if ``n = 0``)."""
    nn_ = n.clamp_min(1)
    p = k / nn_
    z2 = z * z
    denom = 1.0 + z2 / nn_
    centre = (p + z2 / (2 * nn_)) / denom
    half = z * torch.sqrt(p * (1 - p) / nn_ + z2 / (4 * nn_ * nn_)) / denom
    nan = torch.full_like(p, math.nan)
    empty = n == 0
    return (
        torch.where(empty, nan, (centre - half).clamp(0.0, 1.0)),
        torch.where(empty, nan, (centre + half).clamp(0.0, 1.0)),
    )


def binned_conditional_coverage(
    theta: Tensor,
    covered: Tensor,
    *,
    nominal: float,
    bins: Union[int, Sequence[Tensor]] = 10,
    binning: Literal["uniform", "quantile"] = "uniform",
    confidence: float = 0.95,
    bonferroni: bool = True,
) -> list[BinnedCoverage]:
    r"""Coverage :math:`P(\theta \in R(X) \mid \theta)` binned along each parameter.

    Args:
        theta: Held-out true parameters ``[N, P]``.
        covered: Indicator ``[N]`` that each row's set contains its parameter.
        nominal: Target coverage ``1 - alpha``.
        bins: Number of bins per parameter, or explicit edges per parameter.
        binning: ``"uniform"`` (equal width over the observed range; the
            default, so that rare parameter regions get their own bins) or
            ``"quantile"`` (equal counts).
        confidence: Confidence of the two-sided Wilson intervals.
        bonferroni: Split ``1 - confidence`` over all non-empty bins of all
            parameters, so a flag means a family-wise significant departure.

    Returns:
        One :class:`BinnedCoverage` per parameter.
    """
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if binning not in ("uniform", "quantile"):
        raise ValueError("binning must be 'uniform' or 'quantile'")
    theta_w, cov_w = _coverage_inputs(theta, covered, nominal)
    p = theta_w.shape[1]
    if isinstance(bins, int):
        if bins < 1:
            raise ValueError("bins must be a positive integer")
        edge_list = []
        for j in range(p):
            col = theta_w[:, j]
            if binning == "uniform":
                edge_list.append(
                    torch.linspace(
                        float(col.min()),
                        float(col.max()),
                        bins + 1,
                        device=col.device,
                        dtype=col.dtype,
                    )
                )
            else:
                q = torch.linspace(0.0, 1.0, bins + 1, device=col.device, dtype=col.dtype)
                edge_list.append(torch.quantile(col, q).unique())
    else:
        if len(bins) != p:
            raise ValueError("explicit bins must give one edge tensor per parameter")
        edge_list = [e.to(theta_w.device, theta_w.dtype).reshape(-1) for e in bins]
        for e in edge_list:
            if e.numel() < 2 or not bool((e[1:] > e[:-1]).all()):
                raise ValueError("bin edges must be strictly increasing with >= 2 entries")

    counts, hits = [], []
    for j, edges in enumerate(edge_list):
        col = theta_w[:, j].contiguous()
        idx = torch.bucketize(col, edges[1:-1].contiguous(), right=True)
        inside = (col >= edges[0]) & (col <= edges[-1])
        nb = edges.numel() - 1
        counts.append(
            torch.zeros(nb, device=col.device, dtype=col.dtype).index_add_(
                0, idx[inside], torch.ones_like(col[inside])
            )
        )
        hits.append(
            torch.zeros(nb, device=col.device, dtype=col.dtype).index_add_(
                0, idx[inside], cov_w[inside]
            )
        )
    n_tests = sum(int((c > 0).sum()) for c in counts) if bonferroni else 1
    tail = (1.0 - float(confidence)) / 2.0 / max(n_tests, 1)
    z = float(torch.special.ndtri(torch.tensor(1.0 - tail, dtype=torch.float64)))

    out_dtype = theta.dtype
    results = []
    for j, (edges, n_j, k_j) in enumerate(zip(edge_list, counts, hits)):
        lower, upper = _wilson(k_j, n_j, z)
        coverage = torch.where(n_j > 0, k_j / n_j.clamp_min(1), torch.full_like(n_j, math.nan))
        results.append(
            BinnedCoverage(
                parameter=j,
                edges=edges.to(out_dtype),
                count=n_j.long(),
                coverage=coverage.to(out_dtype),
                lower=lower.to(out_dtype),
                upper=upper.to(out_dtype),
                under_covered=(n_j > 0) & (upper < nominal),
                over_covered=(n_j > 0) & (lower > nominal),
                nominal=float(nominal),
            )
        )
    return results


@dataclass(frozen=True)
class SmoothCoverage:
    r"""Smooth estimate of :math:`P(\theta \in R(X) \mid \theta)`.

    Attributes:
        probability: Estimated coverage at the evaluation points ``[M]``.
        lower, upper: Confidence band at the evaluation points.
        under_covered: Points whose upper bound is below nominal.
        over_covered: Points whose lower bound is above nominal.
        nominal: Target coverage.
        coefficients: Logistic-regression coefficients on the basis.
        covariance: Their (inverse penalised Fisher information) covariance.
    """

    probability: Tensor
    lower: Tensor
    upper: Tensor
    under_covered: Tensor
    over_covered: Tensor
    nominal: float
    coefficients: Tensor
    covariance: Tensor


def _poly_basis(z: Tensor, degree: int, interactions: bool) -> Tensor:
    cols = [torch.ones_like(z[:, 0])]
    for d in range(1, degree + 1):
        cols.extend((z**d).unbind(dim=1))
    if interactions:
        p = z.shape[1]
        for i in range(p):
            for j in range(i + 1, p):
                cols.append(z[:, i] * z[:, j])
    return torch.stack(cols, dim=1)


def smooth_conditional_coverage(
    theta: Tensor,
    covered: Tensor,
    *,
    nominal: float,
    evaluate_at: Optional[Tensor] = None,
    degree: int = 3,
    interactions: bool = True,
    ridge: float = 1e-3,
    confidence: float = 0.95,
    simultaneous: bool = True,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> SmoothCoverage:
    r"""Logistic-regression estimate of :math:`P(\theta \in R(X) \mid \theta)`.

    LF2I's coverage diagnostic: regress the coverage indicators on the
    parameters with a logistic model on a polynomial basis of the standardised
    parameters (per-parameter powers up to ``degree`` plus, optionally,
    pairwise products), fitted by penalised Newton/IRLS (ridge penalty on all
    coefficients, which keeps the fit finite even when every row is covered).

    The band on the logit is :math:`\hat\eta \pm c\,\mathrm{se}`, with ``se``
    from the inverse penalised Fisher information.  With ``simultaneous=True``
    (default) :math:`c = \sqrt{\chi^2_{d}(\text{confidence})}` over the ``d``
    basis functions (Scheffé), so the band holds jointly over all evaluation
    points and a flag is not an artefact of testing many points; otherwise
    :math:`c` is the pointwise normal quantile.

    Args:
        theta: Held-out true parameters ``[N, P]``.
        covered: Coverage indicators ``[N]``.
        nominal: Target coverage ``1 - alpha``.
        evaluate_at: Parameters ``[M, P]`` at which to report the estimate
            (default: ``theta``).
        degree: Polynomial degree per parameter.
        interactions: Include pairwise products of the standardised parameters.
        ridge: Ridge penalty (in units of the log-likelihood).
        confidence: Band confidence level.
        simultaneous: Scheffé simultaneous band (default) or pointwise.
        max_iter, tol: Newton iteration controls.

    Returns:
        :class:`SmoothCoverage`.
    """
    if int(degree) < 0:
        raise ValueError("degree must be non-negative")
    if not float(ridge) > 0.0:
        raise ValueError("ridge must be positive")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    theta_w, y = _coverage_inputs(theta, covered, nominal)
    mean, std = _standardizer(theta_w)
    basis = _poly_basis((theta_w - mean) / std, int(degree), interactions)
    d = basis.shape[1]
    eye = torch.eye(d, device=basis.device, dtype=basis.dtype)
    beta = torch.zeros(d, device=basis.device, dtype=basis.dtype)
    hessian = eye * float(ridge)
    for _ in range(int(max_iter)):
        prob = torch.sigmoid(basis @ beta)
        grad = basis.T @ (prob - y) + float(ridge) * beta
        weight = prob * (1 - prob)
        hessian = (basis * weight.unsqueeze(-1)).T @ basis + float(ridge) * eye
        step = torch.linalg.solve(hessian, grad)
        beta = beta - step
        if float(step.abs().max()) < tol:
            break
    prob = torch.sigmoid(basis @ beta)
    weight = prob * (1 - prob)
    hessian = (basis * weight.unsqueeze(-1)).T @ basis + float(ridge) * eye
    covariance = torch.linalg.inv(hessian)

    if evaluate_at is None:
        eval_basis = basis
        out_device, out_dtype = theta.device, theta.dtype
    else:
        ev = _as_2d(evaluate_at, "evaluate_at")
        if ev.shape[1] != theta_w.shape[1]:
            raise ValueError("evaluate_at must have the same number of columns as theta")
        out_device, out_dtype = ev.device, ev.dtype
        ev = ev.to(theta_w.device, theta_w.dtype)
        eval_basis = _poly_basis((ev - mean) / std, int(degree), interactions)
    eta = eval_basis @ beta
    se = torch.sqrt(((eval_basis @ covariance) * eval_basis).sum(-1).clamp_min(0.0))
    if simultaneous:
        from scipy.stats import chi2

        c = math.sqrt(float(chi2.ppf(float(confidence), d)))
    else:
        tail = (1.0 - float(confidence)) / 2.0
        c = float(torch.special.ndtri(torch.tensor(1.0 - tail, dtype=torch.float64)))
    lower = torch.sigmoid(eta - c * se)
    upper = torch.sigmoid(eta + c * se)

    def _out(t: Tensor) -> Tensor:
        return t.to(out_device, out_dtype)

    return SmoothCoverage(
        probability=_out(torch.sigmoid(eta)),
        lower=_out(lower),
        upper=_out(upper),
        under_covered=(upper < nominal).to(out_device),
        over_covered=(lower > nominal).to(out_device),
        nominal=float(nominal),
        coefficients=_out(beta),
        covariance=_out(covariance),
    )
