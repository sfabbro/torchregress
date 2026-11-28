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
   log score (by matching the variance of standardised residuals for a draws-only
   predictive, kept only if it lowers the validation energy score).
5. **Conformalise** on another held-out split: :meth:`region` returns a set with
   :math:`P(Y \in R(X)) \ge 1 - \alpha` under exchangeability, whatever the model.

Steps 4 and 5 use different rows: a recalibrator fitted on the conformal rows would break
the exchangeability the coverage statement rests on.

With ``cv=K`` (cross-fitting) the rows are not split once: ``K`` fold models each train
on all rows but one fold, stacking and recalibration use the out-of-fold densities of
every row, ``predict`` averages the fold models and ``region`` is cross-conformal (a
weaker guarantee, see :class:`~torchregress.joint.CrossConformalRegion`).
"""

from __future__ import annotations

import copy
import inspect
import math
import warnings
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any, Literal, Optional, Union, cast

import torch
from torch import Tensor, nn

from ..ensemble.density import _em_mixture_weights
from ..losses.mdn import MixtureDensityLoss
from ..losses.nflows import NormalizingFlowLoss, create_flow_model, recommended_tail_bound
from ..metrics.distribution import energy_score
from ..utils.distributions import inverse_softplus
from ..utils.quantile import pinball_residual
from ._common import mlp, safe_cholesky, seeded, target_index, to_tensor
from ._common import train as train_module
from .calibration import ScaleRecalibrator
from .dependence import (
    ConditionalGaussianCopula,
    Dependence,
    FlowDependence,
    GaussianCopula,
    IndependenceDependence,
    MixtureDependence,
    SklarJoint,
    select_dependence,
)
from .marginals import Marginals, MixtureMarginals, QuantileMarginals
from .predictive import (
    AffineJoint,
    ConditionalModelJoint,
    DistributionJoint,
    FlowJoint,
    GaussianJoint,
    JointPredictive,
    MixtureJoint,
)
from .regions import (
    ConformalScore,
    CrossConformalRegion,
    HPDScore,
    JointConformal,
    JointRegion,
    LatentScore,
    MahalanobisScore,
    SampleBallScore,
)

__all__ = ["JointRegressor", "MarginalEngine", "MixtureNetworkEngine", "QuantileNetworkEngine"]

_ENERGY_VAL_DRAWS = 16  # fixed draws per validation row for the energy head
#: ``method="auto"`` trains the flow head up to this many targets.  A draw from an
#: autoregressive flow costs one step per target and layer.  At 32 targets (harness design
#: ``joint_nuisance_d32``, 4,000 rows, seed 0) the default with the flow head is 3.3 nats
#: from the oracle against 10.6 without it; it fits in 155 s instead of 51 s.  Beyond 32
#: targets it has not been measured.
_AUTO_FLOW_MAX_TARGETS = 32
#: Folds of the cross-fitted point model of ``location="mse"`` (fewer on tiny data).
_LOCATION_FOLDS = 5
_LOCATION_CHUNK = 65_536  # rows per forward pass of the point networks

Method = Literal["auto", "gaussian", "mixture", "flow", "sklar", "energy"]


def _location_mean(nets: Iterable[nn.Module], xs: Tensor) -> Tensor:
    """Average of the point networks of ``location="mse"`` at ``xs``, in chunks of rows."""
    members = list(nets)
    with torch.no_grad():
        parts = [
            torch.stack([net(part) for net in members]).mean(0)
            for part in xs.split(_LOCATION_CHUNK)
        ]
    return torch.cat(parts)


def _auto_batch_size(n_train: int) -> int:
    """Minibatch size of ``batch_size="auto"`` for ``n_train`` training rows.

    256, halved while an epoch would have fewer than 8 steps (down to 16) and doubled
    while it would have more than 400 (up to 4096).  Early stopping counts epochs: with
    300 rows and batches of 256 an epoch is two steps and a patience of 30 epochs is
    60 steps, too few to fit a head.  On two real sets with 145 and 308 training rows
    (three seeds each) batches of 16 and 32 improved the stacked log score by 0.26 and
    0.68 nats over batches of 256.
    """
    batch = 256
    while batch > 16 and n_train < 8 * batch:
        batch //= 2
    while batch < 4096 and n_train > 400 * batch:
        batch *= 2
    return batch


ArrayLike = Any  # a tensor, a NumPy array, nested lists, or a pandas or polars frame or series


def _float_tensor(data: ArrayLike, name: str) -> Tensor:
    """``data`` as a float32 tensor (a tensor stays on its device)."""
    try:
        return to_tensor(data).to(torch.float32)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError(
            f"{name} must hold numbers (a tensor, a NumPy array, nested lists, or a pandas "
            f"or polars frame with numeric columns), got {type(data).__name__}: {exc}"
        ) from exc


def _targets(data: ArrayLike) -> Tensor:
    """Targets as a float32 ``[N, d]`` tensor; a one-dimensional ``Y`` is one target."""
    y = _float_tensor(data, "Y")
    return y.unsqueeze(1) if y.dim() == 1 else y


def _rows_error(x: Tensor, y: Tensor) -> ValueError:
    return ValueError(
        "X must be [N, p] and Y [N, d] (or [N] for one target) with the same number of "
        f"rows, got X {tuple(x.shape)} and Y {tuple(y.shape)}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════════


class _Head(nn.Module):
    """A network that turns standardised features into a joint predictive."""

    supports_missing = False
    #: The predictive has a density (``False``: draws only, no log score).
    has_density = True

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

    def __init__(
        self, in_dim: int, d: int, hidden: Sequence[int], n_components: int, max_std: float
    ) -> None:
        super().__init__()
        covariance = "full" if d <= 8 else "diagonal"
        # No component centred outside the range of the (standardised) training targets,
        # and none with an average variance beyond that range squared: ``max_std`` bounds
        # both.  A component of small weight with a huge spread or a far-away mean costs
        # the likelihood nothing and ruins the draws.  Components within the bounds are
        # exactly what they were.
        self.bound = float(max_std)
        self.mdn = MixtureDensityLoss(
            n_components=n_components, n_features=d, covariance_type=covariance, max_std=max_std
        )
        self.net = mlp(in_dim, hidden, self.mdn.expected_output_size)

    def predictive(self, x: Tensor) -> DistributionJoint:
        log_weights, means, factor = self.mdn._extract_distribution_parameters(self.net(x))
        means = means.clamp(-self.bound, self.bound)
        return DistributionJoint(self.mdn._mixture((log_weights, means, factor)))

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


def _masked_energy_score(draws: Tensor, y: Tensor, mask: Optional[Tensor]) -> Tensor:
    r"""Energy score of each row, ``draws [n, m, d]``, ``y [n, d]`` -> ``[n]``.

    .. math::
        \mathrm{ES}_i = \frac1m \sum_j \lVert y_i - g_{ij} \rVert
        - \frac{1}{2m(m-1)} \sum_{j \ne k} \lVert g_{ij} - g_{ik} \rVert,

    unbiased for :math:`\mathbb E\lVert Y' - y\rVert - \tfrac12 \mathbb E\lVert Y' - Y''\rVert`.
    It is :func:`~torchregress.metrics.energy_score` (differentiable, so the
    training loss and the metric are one expression).  With ``mask`` (``True`` =
    observed) the unobserved coordinates are set to zero in the draws and in
    ``y``, so the norms run over the observed coordinates of the row only: the
    energy score of the predictive marginal of the observed sub-vector, a proper
    score for that marginal.  A row with nothing observed scores zero.
    """
    if mask is not None:
        m = mask.to(draws.dtype)
        draws, y = draws * m.unsqueeze(1), y * m
    out = energy_score(draws.movedim(1, 0), y, reduction="none")
    assert isinstance(out, Tensor)
    return out


class _EnergyHead(_Head):
    r"""Engression generator :math:`\hat y = g(x, \varepsilon)`, trained by the energy score.

    Gaussian noise :math:`\varepsilon \sim \mathcal N(0, I_k)`, ``k = noise_dim``,
    is concatenated to the standardised features at the input layer (Shen and
    Meinshausen 2024).  The loss is :func:`_masked_energy_score` of ``n_draws``
    draws per row, averaged over the rows with at least one observed target.
    There is no density: the predictive is a draws-only
    :class:`~torchregress.joint.ConditionalModelJoint` that runs the network on
    every ``sample`` call.
    """

    supports_missing = True
    has_density = False
    _CHUNK_ELEMS = 1 << 24  # elements of one forward block in ``sample``

    def __init__(
        self, in_dim: int, d: int, hidden: Sequence[int], noise_dim: int, n_draws: int
    ) -> None:
        super().__init__()
        self.d, self.noise_dim, self.n_draws = d, int(noise_dim), int(n_draws)
        self.net = mlp(in_dim + self.noise_dim, hidden, d)
        # Widest activation of one draw (input, hidden layers, output): sets the chunk.
        self._width = in_dim + self.noise_dim + max([d, *(int(h) for h in hidden)])

    def noise(
        self, n: int, s: int, like: Tensor, generator: Optional[torch.Generator] = None
    ) -> Tensor:
        """Standard normal noise ``[n, s, noise_dim]`` (``generator`` draws on the CPU)."""
        if generator is None:
            return torch.randn(n, s, self.noise_dim, device=like.device, dtype=like.dtype)
        eps = torch.randn(n, s, self.noise_dim, generator=generator, dtype=like.dtype)
        return eps.to(like.device)

    def generate(self, x: Tensor, eps: Tensor) -> Tensor:
        """Draws ``[n, s, d]`` for features ``x [n, p]`` and noise ``eps [n, s, k]``."""
        n, s, _ = eps.shape
        inputs = torch.cat([x.unsqueeze(1).expand(n, s, x.shape[-1]), eps.to(x)], dim=-1)
        return self.net(inputs)  # type: ignore[no-any-return]

    def sample(self, context: Tensor, n_samples: int = 1) -> Tensor:
        """Draws ``[n, S, d]`` (the flow-head protocol of ``ConditionalModelJoint``)."""
        n = int(context.shape[0])
        chunk = max(1, self._CHUNK_ELEMS // max(1, n * self._width))
        with torch.no_grad():
            parts = [
                self.generate(context, self.noise(n, min(chunk, n_samples - start), context))
                for start in range(0, n_samples, chunk)
            ]
        return torch.cat(parts, dim=1)

    def predictive(self, x: Tensor) -> ConditionalModelJoint:
        return ConditionalModelJoint(self, x, event_dim=self.d)

    def score(self, x: Tensor, y: Tensor, mask: Optional[Tensor], eps: Tensor) -> Tensor:
        """Mean energy score over rows with an observed target, for given noise."""
        es = _masked_energy_score(self.generate(x, eps), y, mask)
        if mask is None:
            return es.mean()
        keep = mask.any(-1)
        return es.sum() / keep.sum().clamp_min(1)

    def loss(self, x: Tensor, y: Tensor, mask: Optional[Tensor]) -> Tensor:
        """Mean energy score of the observed targets, ``n_draws`` fresh draws per row."""
        return self.score(x, y, mask, self.noise(x.shape[0], self.n_draws, x))


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


class _NetworkEngine(MarginalEngine):
    """One network from the features to per-target distribution parameters.

    Shared scaffolding of the built-in engines: the training budget, the seed, the
    early-stopping split and the masked loss.  A subclass gives the output width per
    target, the initial output-layer bias (the output weights start at zero, so
    every row starts at the same distribution), the per-entry loss ``[b, d]`` of the
    parameters ``[b, d, m]`` and the :class:`Marginals` they describe.  Missing
    targets are left out of the loss: it is the mean over the observed entries.
    """

    _name = "network engine"

    def __init__(
        self,
        hidden: Sequence[int],
        max_epochs: int,
        patience: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        val_fraction: float,
        seed: int,
    ) -> None:
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

    # ---------------------------------------------------------- subclass API
    @property
    def _width(self) -> int:
        """Network outputs per target."""
        raise NotImplementedError

    def _initial_bias(self) -> Tensor:
        """Output-layer bias of one target ``[_width]`` (float64)."""
        raise NotImplementedError

    def _entry_loss(self, params: Tensor, y: Tensor) -> Tensor:
        """Loss of each entry ``[b, d]`` from parameters ``[b, d, _width]``."""
        raise NotImplementedError

    def _marginals(self, params: Tensor) -> Marginals:
        raise NotImplementedError

    def _transform(self, out: Tensor) -> Tensor:
        """Network output ``[b, d * _width]`` to the parameters ``[b, d, _width]``."""
        return out.reshape(out.shape[0], self._d, self._width)

    # ------------------------------------------------------------- training
    def _params(self, x: Tensor) -> Tensor:
        assert self._net is not None
        return self._transform(self._net(x))

    def fit(self, x: Tensor, y: Tensor, mask: Optional[Tensor] = None) -> "_NetworkEngine":
        """Fit on ``x [N, p]`` and ``y [N, d]``; the caller's torch RNG state is left as it was."""
        with seeded(self.seed):
            return self._fit(x, y, mask)

    def _fit(self, x: Tensor, y: Tensor, mask: Optional[Tensor]) -> "_NetworkEngine":
        gen = torch.Generator().manual_seed(self.seed)
        n, self._d = y.shape
        torch.manual_seed(self.seed)
        self._net = mlp(x.shape[1], self.hidden, self._d * self._width).to(x.device, x.dtype)
        last = self._net[-1]
        assert isinstance(last, nn.Linear)
        with torch.no_grad():
            last.weight.zero_()
            last.bias.copy_(self._initial_bias().repeat(self._d).to(last.bias))
        observed = torch.ones_like(y, dtype=torch.bool) if mask is None else mask
        target = torch.where(observed, y, torch.zeros_like(y))
        perm = torch.randperm(n, generator=gen)
        n_val = max(1, int(round(self.val_fraction * n)))
        val, train = perm[:n_val].to(x.device), perm[n_val:].to(x.device)

        def loss(rows: Tensor) -> Tensor:
            entry = self._entry_loss(self._params(x[rows]), target[rows])
            keep = observed[rows].to(entry.dtype)
            return (entry * keep).sum() / keep.sum().clamp_min(1.0)

        train_module(
            self._net,
            lambda idx: loss(train[idx.to(train.device)]),
            int(train.numel()),
            lambda: loss(val),
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            generator=gen,
            optimizer="adamw",
            clip_norm=10.0,
            skip_nonfinite=True,
            name=f"the {self._name}",
        )
        return self

    def predict(self, x: Tensor) -> Marginals:
        if self._net is None:
            raise RuntimeError("Call fit() first.")
        with torch.no_grad():
            return self._marginals(self._params(x))


class QuantileNetworkEngine(_NetworkEngine):
    """Built-in marginal engine: one network predicting a grid of quantiles per target.

    Trained with the pinball loss on ``n_quantiles`` equally spaced levels; missing
    targets are left out of the loss.  Its marginals are :class:`QuantileMarginals`
    (piecewise-linear CDF between the predicted quantiles, exponential tails).

    Args:
        n_quantiles: Size of the quantile grid (levels ``k / (K + 1)``).
        hidden: Hidden-layer widths.
        max_epochs, patience, batch_size, lr, weight_decay: Training budget.
        val_fraction: Rows held out for early stopping.
        seed: Seed of the initialisation, the split and the shuffling.
    """

    _name = "quantile engine"

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
        super().__init__(
            hidden, max_epochs, patience, batch_size, lr, weight_decay, val_fraction, seed
        )

    @property
    def levels(self) -> Tensor:
        k = self.n_quantiles
        return torch.arange(1, k + 1, dtype=torch.float32) / (k + 1)

    @property
    def _width(self) -> int:
        return self.n_quantiles

    def _initial_bias(self) -> Tensor:
        # The standard normal quantiles for every row (targets are standardised by
        # the caller): the first level and the softplus pre-images of the increments.
        grid = torch.special.ndtri(self.levels.double())
        steps = (grid[1:] - grid[:-1]).clamp_min(1e-4)
        return torch.cat([grid[:1], inverse_softplus(steps)])

    def _transform(self, out: Tensor) -> Tensor:
        out = out.reshape(out.shape[0], self._d, self.n_quantiles)
        # First level plus non-negative increments: quantiles cannot cross.
        first = out[..., :1]
        steps = nn.functional.softplus(out[..., 1:])
        return torch.cat([first, first + torch.cumsum(steps, dim=-1)], dim=-1)

    def _entry_loss(self, params: Tensor, y: Tensor) -> Tensor:
        levels = self.levels.to(params.device, params.dtype)
        return pinball_residual(params, y.unsqueeze(-1), levels).mean(-1)

    def _marginals(self, params: Tensor) -> QuantileMarginals:
        return QuantileMarginals(self.levels.to(params.device, params.dtype), params)

    def fit(self, x: Tensor, y: Tensor, mask: Optional[Tensor] = None) -> "QuantileNetworkEngine":
        """Fit on ``x [N, p]`` and ``y [N, d]``; the caller's torch RNG state is left as it was."""
        super().fit(x, y, mask)
        return self

    def predict(self, x: Tensor) -> QuantileMarginals:
        out = super().predict(x)
        assert isinstance(out, QuantileMarginals)
        return out


class MixtureNetworkEngine(_NetworkEngine):
    r"""Built-in marginal engine: one network predicting a Gaussian mixture per target.

    For each target :math:`j` the network outputs ``n_components`` logits, means and
    scale pre-activations, read exactly as a diagonal
    :class:`~torchregress.losses.MixtureDensityLoss` with one feature reads its
    output (``log_softmax`` weights, ``softplus(.) + min_std`` standard deviations),
    and is trained by the sum over targets of the marginal negative log densities
    :math:`-\sum_j \log f_j(y_j \mid x)` (the mean over the observed entries:
    missing targets are left out of the loss).  Its marginals are
    :class:`MixtureMarginals`.

    The output weights start at zero, so every row starts at the same mixture: equal
    weights, means at the normal quantiles :math:`\Phi^{-1}((k - 1/2)/K)` shrunk so
    that the mixture has unit variance, and standard deviations of 0.5 (targets are
    standardised by the caller).  Distinct means break the symmetry between
    components, which identical components would keep for ever.

    Defaults.  ``n_components = 5``: on the problem of
    ``examples/joint_regression_comparison.py`` 3, 5 and 8 components were within
    0.05 nats of each other in held-out marginal log density, and five follow skewed
    or heavy-tailed marginals better than three.  ``min_std = 1e-2`` (in units of the
    standardised target): a component narrower than one hundredth of the target's
    spread is a spike on a few training points rather than a feature of the
    conditional distribution, and the floor keeps the log density bounded when values
    repeat.
    No component is wider than 1.25 times the largest deviation of its target from
    the training mean, and none is centred further than that from the mean: beyond
    the range of the data a component of small weight costs the likelihood nothing and
    gives absurd draws.

    Args:
        n_components: Mixture components per target, ``K``.
        hidden: Hidden-layer widths.
        min_std: Floor on the component standard deviations.
        max_epochs, patience, batch_size, lr, weight_decay: Training budget.
        val_fraction: Rows held out for early stopping.
        seed: Seed of the initialisation, the split and the shuffling.
    """

    _name = "mixture engine"
    #: Default for engines saved before the mean range existed (their instances lack it).
    _mean_range: Optional[tuple[Tensor, Tensor]] = None

    def __init__(
        self,
        n_components: int = 5,
        hidden: Sequence[int] = (128, 128),
        *,
        min_std: float = 1e-2,
        max_epochs: int = 300,
        patience: int = 30,
        batch_size: int = 256,
        lr: float = 2e-3,
        weight_decay: float = 1e-5,
        val_fraction: float = 0.15,
        seed: int = 0,
    ) -> None:
        if n_components < 1:
            raise ValueError("n_components must be at least 1")
        if not min_std > 0:
            raise ValueError("min_std must be positive")
        self.n_components = int(n_components)
        self.min_std = float(min_std)
        self._mean_range: Optional[tuple[Tensor, Tensor]] = None
        self._mdn = MixtureDensityLoss(
            n_components=self.n_components,
            n_features=1,
            covariance_type="diagonal",
            min_std=self.min_std,
        )
        super().__init__(
            hidden, max_epochs, patience, batch_size, lr, weight_decay, val_fraction, seed
        )

    @property
    def _width(self) -> int:
        return int(self._mdn.expected_output_size)

    def _initial_bias(self) -> Tensor:
        k = self.n_components
        std = 0.5
        centres = torch.special.ndtri((torch.arange(k, dtype=torch.float64) + 0.5) / k)
        spread = centres.square().mean().sqrt()
        means = centres * (math.sqrt(1.0 - std**2) / spread) if k > 1 else centres
        raw_std = inverse_softplus(torch.full((k,), std - self.min_std, dtype=torch.float64))
        return torch.cat([torch.zeros(k, dtype=torch.float64), means, raw_std])

    def _bounded(self, params: Tensor) -> Any:
        """The mixtures of ``params`` (batch ``[b, d]``, one feature) with means kept in range."""
        log_weights, means, stds = self._mdn._extract_distribution_parameters(params)
        if self._mean_range is not None:
            lo, hi = self._mean_range  # [d, 1, 1] each
            means = torch.maximum(torch.minimum(means, hi.to(means)), lo.to(means))
        return self._mdn._mixture((log_weights, means, stds))

    def _mixture(self, params: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Log weights, means and standard deviations ``[b, d, K]``."""
        mix = self._bounded(params)
        return mix.log_weights, mix.loc[..., 0], mix.component_scale[..., 0]

    def _entry_loss(self, params: Tensor, y: Tensor) -> Tensor:
        return -self._bounded(params).log_prob(y.unsqueeze(-1))

    def _marginals(self, params: Tensor) -> MixtureMarginals:
        log_w, loc, scale = self._mixture(params)
        return MixtureMarginals(log_w.exp(), loc, scale)

    def _fit(self, x: Tensor, y: Tensor, mask: Optional[Tensor]) -> "_NetworkEngine":
        # No component wider than the range of its own target, and none centred outside
        # it: the range is 1.25 times the largest deviation from the target's mean among
        # the observed training values.  A component of small weight beyond it costs the
        # likelihood nothing and ruins the draws.
        observed = torch.ones_like(y, dtype=torch.bool) if mask is None else mask
        count = observed.sum(0).clamp_min(1)
        centre = torch.where(observed, y, torch.zeros_like(y)).sum(0) / count
        spread = torch.where(observed, (y - centre).abs(), torch.zeros_like(y)).amax(0)
        bound = (1.25 * spread).clamp_min(2.0 * self.min_std)
        self._mdn.max_std = bound.reshape(-1, 1, 1)  # stds are [b, d, K, 1]
        self._mean_range = (
            (centre - bound).reshape(-1, 1, 1),  # means are [b, d, K, 1]
            (centre + bound).reshape(-1, 1, 1),
        )
        return super()._fit(x, y, mask)

    def fit(self, x: Tensor, y: Tensor, mask: Optional[Tensor] = None) -> "MixtureNetworkEngine":
        """Fit on ``x [N, p]`` and ``y [N, d]``; the caller's torch RNG state is left as it was."""
        super().fit(x, y, mask)
        return self

    def predict(self, x: Tensor) -> MixtureMarginals:
        out = super().predict(x)
        assert isinstance(out, MixtureMarginals)
        return out


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
            flow), ``"sklar"`` (per-target marginals joined by the lowest rung of the
            dependence ladder that validation supports: independence, constant Gaussian
            copula, covariate-dependent Gaussian copula, conditional Gaussian mixture,
            conditional flow) or
            ``"energy"`` (engression: a generator :math:`g(x, \varepsilon)` trained by
            the energy score; draws only, see below).  ``"auto"`` never includes
            ``"energy"`` (its stacking is by log score).
        marginals: Marginal engine for the Sklar route (a :class:`MarginalEngine`),
            used as is.  ``None`` (default) fits both built-in engines,
            :class:`QuantileNetworkEngine` and :class:`MixtureNetworkEngine` (with
            ``n_components`` components), once each on the training rows and keeps
            the one with the larger mean held-out sum of marginal log densities
            :math:`\sum_j \log f_j(y_j \mid x)` on the complete validation rows (the
            rows the dependence ladder is selected on); the scores are
            ``marginal_scores_`` and the choice ``marginal_engine_``.  The engines
            are not nested, so there is no simpler one to prefer: the larger score
            wins.  Only the chosen engine is refitted out of fold to produce the
            normal scores the dependence model is trained on.
        hidden: Hidden-layer widths of the heads.
        n_components: Mixture components of the mixture head and, per target, of the
            built-in :class:`MixtureNetworkEngine` of the Sklar route.
        rank: Rank of the Gaussian head's low-rank covariance; ``None`` uses a full
            Cholesky factor up to 32 targets and rank 8 beyond.
        max_epochs, patience, lr, weight_decay: Training budget per head.
        batch_size: Minibatch size, or ``"auto"`` (default): 256, halved while an
            epoch would have fewer than 8 steps, down to 16 (below about 2,000 training
            rows), and doubled while it would take more than 400 steps, up to 4096
            (256 up to about 100,000 training rows, 4096 from about 800,000).  The
            size used is ``batch_size_``.
        val_fraction, recal_fraction, conformal_fraction: Held-out shares for early
            stopping and stacking, for recalibration, and for conformal calibration.
        max_holdout: Cap on the rows of each held-out split.  Beyond it more
            held-out rows add little (a conformal quantile from 20,000 rows is
            within :math:`5 \times 10^{-5}` of its level) and the rows go to
            training instead.
        cv: ``None`` (default): one split, as described below.  An integer
            :math:`K \ge 2`: cross-fitting for small data sets, see
            **Cross-fitting** below.  ``K`` fits instead of one.
        location: ``None`` (default): the heads model the targets directly.  ``"mse"``:
            a point model is fitted first by squared error and the heads model its
            residuals, see **Location model** below.  Five more network fits of the
            same ``hidden`` size (per fold model with ``cv``).
        recalibrate: Fit the per-target scale recalibration.
        clip_features: Clamp every feature to the range seen in the training rows
            before it enters the networks (default).  A network has no information
            beyond that range and extrapolates arbitrarily: one test row with a
            feature 178 standard deviations out (a sensor glitch in a river-flow
            data set) had a log density of :math:`-10^{11}` without it.  Rows inside
            the range are untouched; the clamp is applied to calibration and test
            rows alike, so the coverage statement of :meth:`region` is unchanged.
        score: Conformity score of :meth:`region`; ``"auto"`` picks the latent norm
            (:class:`~torchregress.joint.LatentScore`) for a Gaussian predictive (whose
            latent ball is its highest-density region), the HPD level
            (:class:`~torchregress.joint.HPDScore`) for every other predictive with a
            density, a sample-ball score for draws-only predictives, and the masked
            Mahalanobis score for conformal rows with missing targets.  See
            :meth:`region` for the trade-off.
        n_folds: Folds for the out-of-fold normal scores of the Sklar route.
        n_members: Independently initialised copies of each network head (Gaussian,
            mixture, flow), averaged with equal weights: a deep ensemble
            (Lakshminarayanan et al. 2017).  ``None`` (default) means 3 for
            ``method="auto"`` and 1 for a single named head.  With more than one
            copy the head's predictive is a mixture of its copies (a Gaussian head
            is then no longer a single Gaussian); stacking is across head types.
            The Sklar and energy heads always have one copy.
        noise_dim: Dimension :math:`k` of the Gaussian noise of the energy head,
            concatenated to the features at the input layer.  ``None`` (default) uses
            :math:`k = d`, the least noise for a predictive that is not confined to a
            lower-dimensional set (:math:`k < d` confines it to a :math:`k`-dimensional
            manifold; :math:`k > d` is allowed).  The engression Python package
            defaults to 100; on the tests of this head (``d = 3``, two features) 100
            gave a worse held-out energy score and marginal calibration than
            :math:`k = d` at the same budget.
        energy_draws: Draws :math:`m \ge 2` per row in the energy-score loss (2,
            the choice of Shen and Meinshausen 2024; the estimate is unbiased for
            every :math:`m`, larger :math:`m` only lowers its variance).
        device: Training device (default: CUDA when available, else CPU).
        seed: Seed for splits, initialisation and shuffling.

    Missing targets (``NaN`` in ``Y``) are integrated out exactly by the Gaussian and
    mixture heads.  The flow head needs complete rows, so ``"auto"`` leaves it out when
    targets are missing; it also leaves it out beyond 32 targets, where its cost has not
    been measured (``method="flow"`` has no such limit).

    **Location model** (``location="mse"``).  A head trained by likelihood fits scale
    and shape together with the centre, and with early stopping its centre can end
    up less accurate than a squared-error network's (on a 16-target real set the
    Gaussian head's predictive mean had a root mean square error of 0.63 standard
    deviations against 0.37).  With ``location="mse"`` the predictive is

    .. math:: Y = m(x) + s \odot R, \qquad R \mid x \sim \hat p_R(\cdot \mid x),

    where :math:`m` is the average of five networks trained by squared error on
    the training rows, each without one fifth of them (a ``val_fraction`` of the
    rest stops its training early; fewer networks below 40 training rows, with the
    same ``batch_size_`` as the heads), :math:`s` the per-target spread
    of their out-of-fold residuals, and :math:`\hat p_R` whatever ``method`` fits,
    to the residuals.  A training row's residual uses the network that did not see
    it (a density fitted to in-sample residuals is overconfident on new rows); the
    validation, recalibration and conformal rows and every row at ``predict`` use
    the average of the five, none of which saw them, so the stacking weights, the
    recalibration and the coverage statement of :meth:`region` hold as without it.
    It is not always a gain: it helped where the heads' centres were poor (that
    16-target set: energy score 413 to 241 for the Gaussian head) and hurt on a set
    with 308 training rows, where networks trained on four fifths of the rows are
    worse than the heads' own centres (default log score 1.73 to 2.57).  Compare
    ``validation_log_score_`` with and without it.

    **The energy head** (``method="energy"``, engression; Shen and Meinshausen 2024)
    is a network :math:`g(x, \varepsilon)`, :math:`\varepsilon \sim \mathcal N(0, I_k)`,
    trained to minimise the energy score (Gneiting and Raftery 2007) of
    ``energy_draws`` draws per row,

    .. math::
        \frac1m \sum_j \lVert y - g_j \rVert
        - \frac{1}{2m(m-1)} \sum_{j \ne k} \lVert g_j - g_k \rVert .

    It assumes no density family and draws in one forward pass, but has no density:
    its predictive is draws only (``has_density`` is ``False``), so there is no log
    score, no HPD or latent region (``score="auto"`` gives the sample-ball score) and
    recalibration uses the variance objective.  With missing targets the norms run
    over the observed coordinates of each row: that is the energy score of the
    predictive marginal of the observed sub-vector, a proper score for that marginal
    (not for the joint), so the head is fitted on every row in that sense.  Early
    stopping uses the validation energy score with the same noise every epoch,
    reported in ``validation_energy_score_``.

    **How** ``fit`` **splits the rows.**  A random permutation (from ``seed``) gives
    ``round(val_fraction * N)`` validation rows (early stopping and stacking),
    ``round(recal_fraction * N)`` recalibration rows and
    ``round(conformal_fraction * N)`` conformal rows (each at least 8 and at most
    ``max_holdout``); the rest (75% with the defaults 10%, 5%, 10%, more once the
    cap binds: 94% at a million rows) trains the heads.  The counts are
    ``n_val_``, ``n_recal_``, ``n_conformal_`` and ``n_train_``.  :meth:`calibrate`
    replaces the conformal rows with data you supply.

    **Cross-fitting** (``cv=K``).  One seeded permutation cuts the ``N`` rows into
    ``K`` folds (sizes differ by at most one; ``folds_`` holds the fold of each
    row).  Fold model :math:`k` is a ``JointRegressor`` with the same settings
    fitted on the other ``K - 1`` folds, with its own standardisation, feature
    clamp and an inner validation split (``val_fraction`` of its rows, at most
    ``max_holdout``) for early stopping only: it never sees fold :math:`k`.  No
    recalibration or conformal rows are held out.  ``n_members=None`` means one
    copy of each head per fold model (the ``K`` fold models are the ensemble).
    Then:

    * **Stacking** on the out-of-fold log density of every row (each row scored by
      the fold model that did not see it; ``validation_log_score_`` is the
      out-of-fold mean per head).  The energy head gets an out-of-fold energy
      score instead and is not stacked.
    * **Prediction**: :meth:`predict` is the equal-weight average of the ``K``
      stacked fold predictives, one flat :class:`~torchregress.joint.MixtureJoint`
      over (fold, head) components with weights ``w_head / K``, each component
      carrying the recalibration of its fold.
    * **Recalibration**: one :class:`~torchregress.joint.ScaleRecalibrator` fitted
      on the out-of-fold predictive of every complete row and applied to every
      fold model, each scaled about its own center.  No rows are left to accept or
      reject the factors on: factors within 2% of one are skipped, and with a
      density the ``"nll"`` objective never returns factors that score worse
      than leaving the predictive alone on those rows (an in-sample statement).
    * **Regions**: cross-conformal (:meth:`region`).  The fold models used for its
      scores are stacked with their own inner-validation weights and not
      recalibrated, so that fold model :math:`k` depends on no row of fold
      :math:`k`, as the guarantee requires.  The training rows are kept on the
      estimator to compute the out-of-fold scores.

    The cost is ``K`` fits and ``K`` times the scoring work at :meth:`region`.

    Attributes:
        members_: Names of the fitted heads.
        weights_: Stacking weights (``[K]``).
        validation_log_score_: Mean validation log density per head, in the units of
            the data (comparable with ``predict(X).log_prob(Y).mean()``, before
            recalibration); over the observed targets when targets are missing.
            Heads with a density only: the energy head has no log score.
        validation_energy_score_: Mean validation energy score (lower is better) of
            each head without a density (``"energy"``), in the units of the data and
            over the observed targets of each row, from fixed draws (16 per row).
        n_train_, n_val_, n_recal_, n_conformal_: Rows in each split
            (``n_conformal_`` follows :meth:`calibrate`).  With ``cv``: the
            training and inner validation rows of one fold model (the smallest
            over the folds), and ``N`` recalibration and conformal rows (every row,
            out of fold).
        folds_: Fold of each training row ``[N]`` (``cv`` only, else ``None``).
        fold_choices_: Per-fold choices of the Sklar route with ``cv``:
            ``{"marginal_engine": [...], "dependence": [...]}`` (``marginal_engine_``
            and ``dependence_`` are then the most frequent choice, and
            ``marginal_scores_`` and ``dependence_scores_`` the means over the
            folds of each fold model's inner validation scores); empty otherwise.
        batch_size_: Minibatch size used by the last :meth:`fit`.
        n_parameters_: Number of network parameters of the fitted model (all copies,
            all fold models).
        dependence_scores_: Validation log score of each dependence rung (Sklar route).
        dependence_: Name of the rung the validation gate chose (Sklar route).
        marginal_scores_: Mean held-out sum of marginal log densities of each marginal
            engine fitted for the Sklar route (``"quantile"`` and ``"mixture"``, or the
            class name of a user-supplied engine), in the units of the data, on the
            complete validation rows.
        marginal_engine_: Name of the marginal engine the Sklar route uses.
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
        batch_size: Union[int, str] = "auto",
        lr: float = 2e-3,
        weight_decay: float = 1e-5,
        val_fraction: float = 0.1,
        recal_fraction: float = 0.05,
        conformal_fraction: float = 0.1,
        max_holdout: int = 20_000,
        cv: Optional[int] = None,
        location: Optional[str] = None,
        recalibrate: bool = True,
        clip_features: bool = True,
        score: Union[str, ConformalScore] = "auto",
        n_folds: int = 3,
        n_members: Optional[int] = None,
        noise_dim: Optional[int] = None,
        energy_draws: int = 2,
        device: Optional[Union[str, torch.device]] = None,
        seed: int = 0,
    ) -> None:
        if method not in ("auto", "gaussian", "mixture", "flow", "sklar", "energy"):
            raise ValueError(f"unknown method {method!r}")
        fractions = (val_fraction, recal_fraction, conformal_fraction)
        if any(f <= 0 for f in fractions) or sum(fractions) >= 0.9:
            raise ValueError("held-out fractions must be positive and sum to less than 0.9")
        self.method: Method = method
        self.marginals = marginals
        self.hidden = tuple(int(h) for h in hidden)
        self.n_components = int(n_components)
        self.rank = rank
        if isinstance(batch_size, str):
            if batch_size != "auto":
                raise ValueError(f"batch_size must be an integer or 'auto', got {batch_size!r}")
        elif int(batch_size) < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if int(max_holdout) < 8:
            raise ValueError(f"max_holdout must be at least 8, got {max_holdout}")
        self.max_epochs, self.patience = int(max_epochs), int(patience)
        self.batch_size: Union[int, str] = (
            batch_size if isinstance(batch_size, str) else int(batch_size)
        )
        self.max_holdout = int(max_holdout)
        if cv is not None and (isinstance(cv, bool) or int(cv) != cv or int(cv) < 2):
            raise ValueError(f"cv must be None or an integer of at least 2, got {cv!r}")
        self.cv: Optional[int] = None if cv is None else int(cv)
        self.batch_size_: Optional[int] = None
        self.lr, self.weight_decay = float(lr), float(weight_decay)
        self.val_fraction, self.recal_fraction = float(val_fraction), float(recal_fraction)
        self.conformal_fraction = float(conformal_fraction)
        self.recalibrate = bool(recalibrate)
        self.clip_features = bool(clip_features)
        self.score = score
        if int(n_folds) < 2:
            raise ValueError(f"n_folds must be at least 2, got {n_folds}")
        self.n_folds = int(n_folds)
        if noise_dim is not None and int(noise_dim) < 1:
            raise ValueError(f"noise_dim must be positive, got {noise_dim}")
        if int(energy_draws) < 2:
            raise ValueError(f"energy_draws must be at least 2, got {energy_draws}")
        self.noise_dim = None if noise_dim is None else int(noise_dim)
        self.energy_draws = int(energy_draws)
        if location not in (None, "mse"):
            raise ValueError(f"location must be None or 'mse', got {location!r}")
        self.location = location
        if n_members is not None and int(n_members) < 1:
            raise ValueError(f"n_members must be at least 1, got {n_members}")
        self.n_members = None if n_members is None else int(n_members)
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.seed = int(seed)

        self.members_: list[str] = []
        self.weights_: Optional[Tensor] = None
        self.validation_log_score_: dict[str, float] = {}
        self.validation_energy_score_: dict[str, float] = {}
        self.dependence_scores_: dict[str, float] = {}
        self.dependence_: Optional[str] = None
        self.marginal_scores_: dict[str, float] = {}
        self.marginal_engine_: Optional[str] = None
        self.scale_: Optional[Tensor] = None
        self.n_features_in_: Optional[int] = None
        self.n_train_: Optional[int] = None
        self.n_val_: Optional[int] = None
        self.n_recal_: Optional[int] = None
        self.n_conformal_: Optional[int] = None
        self.n_targets_: Optional[int] = None
        self._heads: dict[str, list[_Head]] = {}
        self._engine: Optional[MarginalEngine] = None
        self._marginal_candidates: dict[str, MarginalEngine] = {}
        self._dependence: Optional[Dependence] = None
        self._recalibrator: Optional[ScaleRecalibrator] = None
        self._x_stats: Optional[tuple[Tensor, Tensor]] = None
        self._x_range: Optional[tuple[Tensor, Tensor]] = None
        self._y_stats: Optional[tuple[Tensor, Tensor]] = None
        # Location model (location="mse"): the fold networks and the per-target scale of
        # their out-of-fold residuals, in units of the standardised targets.
        self._location: Optional[nn.ModuleList] = None
        self._location_div: Optional[Tensor] = None
        self._conformal_rows: Optional[tuple[Tensor, Tensor, Tensor]] = None
        self._conformal_cache: dict[Any, tuple[JointConformal, Optional[JointConformal]]] = {}
        # Cross-fitting (cv): fold of each row, the fold models, the training rows
        # (x, filled y, observed) kept for out-of-fold region scores, their cache.
        self.folds_: Optional[Tensor] = None
        self.fold_choices_: dict[str, list[Optional[str]]] = {}
        self._fold_models: list[JointRegressor] = []
        self._cross_rows: Optional[tuple[Tensor, Tensor, Tensor]] = None
        self._cross_cache: dict[Any, Any] = {}
        # A fold model: no recalibration or conformal rows, the head names of the
        # outer model, its own stacking weights kept for the cross-conformal scores.
        self._holdout = True
        self._forced_names: Optional[list[str]] = None
        self._own_weights: Optional[Tensor] = None

    # ----------------------------------------------------------------- fit
    def fit(self, X: ArrayLike, Y: ArrayLike) -> "JointRegressor":
        """Fit on features ``X [N, p]`` and targets ``Y [N, d]`` (``NaN`` = missing).

        ``X`` and ``Y`` are tensors, NumPy arrays, nested lists, or pandas or polars
        frames with numeric columns; a one-dimensional ``Y`` is a single target.
        Deterministic for a given ``seed``; the caller's torch RNG state is left as
        it was.  ``±inf`` in ``Y`` is an error (only ``NaN`` marks a missing target).
        """
        with seeded(self.seed):
            return self._fit(X, Y)

    def _fit(self, X: ArrayLike, Y: ArrayLike) -> "JointRegressor":
        # The split, the statistics and the kept rows live on the CPU, whatever device the
        # inputs came on; the networks see them on ``self.device``.
        x = _float_tensor(X, "X").cpu()
        y = _targets(Y).cpu()
        if x.dim() != 2 or y.dim() != 2 or x.shape[0] != y.shape[0]:
            raise _rows_error(x, y)
        if not bool(torch.isfinite(x).all()):
            raise ValueError("X must be finite")
        n, d = y.shape
        if bool(torch.isinf(y).any()):
            raise ValueError("Y contains +-inf; mark missing targets with NaN")
        observed = torch.isfinite(y)
        if not bool(observed.any(dim=0).all()):
            raise ValueError("every target needs at least one observed value")
        if n < 50 and self._holdout:
            raise ValueError("at least 50 rows are needed to hold out calibration splits")
        if n < 16:
            raise ValueError("a fold model needs at least 16 rows")
        self.n_features_in_, self.n_targets_ = int(x.shape[1]), int(d)
        has_missing = not bool(observed.all())
        self._fold_models, self.folds_, self.fold_choices_ = [], None, {}
        self._cross_rows, self._cross_cache = None, {}
        if self.cv is not None:
            return self._fit_crossfit(x, y, observed, has_missing)

        gen = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(n, generator=gen)
        cap = self.max_holdout
        n_val = min(cap, max(8, int(round(self.val_fraction * n))))
        n_rec = min(cap, max(8, int(round(self.recal_fraction * n)))) if self._holdout else 0
        n_conf = min(cap, max(8, int(round(self.conformal_fraction * n)))) if self._holdout else 0
        val, rec = perm[:n_val], perm[n_val : n_val + n_rec]
        conf, train = perm[n_val + n_rec : n_val + n_rec + n_conf], perm[n_val + n_rec + n_conf :]

        seen = observed[train].sum(0)
        if bool((seen < 2).any()):
            # One value has no spread and none has no location: the standardisation would
            # fall back to mean 0 and spread 1e-6, a point mass wherever the data are.
            short = {int(j): int(seen[j]) for j in (seen < 2).nonzero().flatten()}
            raise ValueError(
                f"each target needs at least 2 observed values among the {train.numel()} "
                f"training rows; observed counts by target index: {short}"
                + (" (in one fold model of cv)" if self._holdout is False else "")
            )
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
        if isinstance(self.batch_size, str):
            self.batch_size_ = _auto_batch_size(self.n_train_)
        else:
            self.batch_size_ = self.batch_size

        dev = self.device
        xs = (x - x_mean) / x_std
        # Standardised range of the training rows; held-out rows are clamped to it here
        # exactly as new rows are at predict time.
        self._x_range = (xs[train].amin(0), xs[train].amax(0))
        xs = self._standardised(x).to(dev)
        ys = torch.where(observed, (y - y_mean) / y_std, torch.zeros_like(y)).to(dev)
        mask = observed.to(dev)
        mask_arg: Optional[Tensor] = mask if has_missing else None
        train_d, val_d = train.to(dev), val.to(dev)

        self._location, self._location_div = None, None
        if self.location is not None:
            # The heads model what a cross-fitted point model leaves: from here on ``ys``
            # are its residuals in units of their own spread, and ``y_std`` that spread
            # in data units.
            ys, residual_scale = self._fit_location(xs, ys, mask, train_d)
            y_std = y_std * residual_scale.cpu()
            self._y_stats = (y_mean, y_std)
            log_y_std = y_std.log().to(dev)

        names = self._member_names(d, has_missing)
        log_density: list[Tensor] = []
        self._heads, self.validation_log_score_, self.validation_energy_score_ = {}, {}, {}
        # Nothing of a previous fit survives (a refit may not fit the Sklar head at all).
        self.marginal_scores_, self.marginal_engine_ = {}, None
        self.dependence_scores_, self.dependence_ = {}, None
        self._engine, self._dependence, self._marginal_candidates = None, None, {}
        for offset, name in enumerate(names):
            torch.manual_seed(self.seed + offset)
            if name == "sklar":
                self._fit_sklar(xs, ys, mask, train_d, val_d, gen)
            else:
                copies = 1 if name == "energy" else self._n_copies()
                self._heads[name] = []
                for copy_id in range(1, copies):
                    # Extra copies of a deep ensemble: same rows, another initialisation.
                    torch.manual_seed(self.seed + offset + 7919 * copy_id)
                    extra = self._build_head(name, xs.shape[1], d, ys[train_d], mask[train_d]).to(
                        dev
                    )
                    self._train_head(
                        extra, name, xs, ys, mask_arg, train_d, val_d, mask, gen, offset
                    )
                    self._heads[name].append(extra)
                torch.manual_seed(self.seed + offset)
                head = self._build_head(name, xs.shape[1], d, ys[train_d], mask[train_d]).to(dev)
                rows_tr = train_d if head.supports_missing else train_d[mask[train_d].all(1)]
                rows_va = val_d if head.supports_missing else val_d[mask[val_d].all(1)]
                m_arg = mask_arg if head.supports_missing else None

                def batch_loss(idx: Tensor, head: _Head = head, rows: Tensor = rows_tr) -> Tensor:
                    r = rows[idx.to(rows.device)]
                    return head.loss(xs[r], ys[r], None if m_arg is None else m_arg[r])

                # The energy head is validated on the same noise every epoch, so that
                # epochs are compared on the same draws.
                eps_va: Optional[Tensor] = None
                if isinstance(head, _EnergyHead):
                    noise_gen = torch.Generator().manual_seed(self.seed + offset)
                    eps_va = head.noise(int(rows_va.numel()), _ENERGY_VAL_DRAWS, xs, noise_gen)

                def validation(
                    head: _Head = head, rows: Tensor = rows_va, eps: Optional[Tensor] = eps_va
                ) -> Tensor:
                    m_rows = None if m_arg is None else m_arg[rows]
                    if isinstance(head, _EnergyHead) and eps is not None:
                        return head.score(xs[rows], ys[rows], m_rows, eps)
                    return head.loss(xs[rows], ys[rows], m_rows)

                train_module(
                    head,
                    batch_loss,
                    int(rows_tr.numel()),
                    validation,
                    max_epochs=self.max_epochs,
                    patience=self.patience,
                    batch_size=self.batch_size_,
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
                self._heads[name].insert(0, head)
                if isinstance(head, _EnergyHead) and eps_va is not None:
                    # Data units: the standardisation shift cancels in the score, the
                    # scale does not.  No log score: this head is never stacked.
                    with torch.no_grad():
                        y_std_d = y_std.to(dev)
                        draws = head.generate(xs[rows_va], eps_va) * y_std_d
                        es = _masked_energy_score(
                            draws,
                            ys[rows_va] * y_std_d,
                            None if m_arg is None else m_arg[rows_va],
                        )
                    n_obs = es.numel() if m_arg is None else int(m_arg[rows_va].any(-1).sum())
                    self.validation_energy_score_[name] = float(es.sum()) / max(n_obs, 1)
                    continue
            masked = has_missing and name != "sklar" and self._heads[name][0].supports_missing
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
        if len(names) == 1:
            self.weights_ = torch.ones(1)
        else:
            stacked = torch.stack(log_density, dim=1).cpu().double()
            self.weights_ = _stack(stacked).to(torch.float32).cpu()
        self._own_weights = self.weights_

        self._recalibrator, self.scale_ = None, None
        complete_rec = rec[observed[rec].all(1)]
        if self.recalibrate and complete_rec.numel() >= 8:
            base = self._base_predictive(x[complete_rec])
            objective = "nll" if base.has_density else "variance"
            # About the center: no draws from flow members at fit or at any later predict.
            recalibrator = ScaleRecalibrator(
                objective=objective,
                n_samples=256,
                about="center" if base.has_density else "mean",
            )
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
                elif complete_val.numel() >= 8:
                    # Draws only: the same test by energy score, on common draws (the
                    # recalibrated draws are the base draws rescaled about the mean).
                    held = self._base_predictive(x[complete_val])
                    y_held = y[complete_val].to(self.device)
                    scores = []
                    for pred in (held, recalibrator.transform(held)):
                        with torch.no_grad(), seeded(self.seed):
                            draws = pred.sample(_ENERGY_VAL_DRAWS)
                        scores.append(float(_masked_energy_score(draws, y_held, None).mean()))
                    keep = scores[1] < scores[0]
                if keep:
                    self._recalibrator = recalibrator
                    self.scale_ = fitted.detach().cpu()
        self._conformal_rows = (x[conf], filled[conf], observed[conf])
        self._conformal_cache = {}
        return self

    # ----------------------------------------------------------- cross-fitting
    def _fold_model(self, k: int, names: list[str]) -> "JointRegressor":
        """Unfitted model of fold ``k``: same budget, no recalibration or conformal rows."""
        model = JointRegressor(
            self.method,
            marginals=self.marginals,
            hidden=self.hidden,
            n_components=self.n_components,
            rank=self.rank,
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            val_fraction=self.val_fraction,
            recal_fraction=self.recal_fraction,
            conformal_fraction=self.conformal_fraction,
            max_holdout=self.max_holdout,
            location=self.location,
            recalibrate=False,
            clip_features=self.clip_features,
            score=self.score,
            n_folds=self.n_folds,
            n_members=1 if self.n_members is None else self.n_members,
            noise_dim=self.noise_dim,
            energy_draws=self.energy_draws,
            device=self.device,
            seed=self.seed + 1009 * (k + 1),
        )
        model._holdout = False
        model._forced_names = list(names)
        return model

    def _fit_crossfit(
        self, x: Tensor, y: Tensor, observed: Tensor, has_missing: bool
    ) -> "JointRegressor":
        """Fold models, out-of-fold stacking and recalibration (``cv = K``)."""
        assert self.cv is not None
        n, d = y.shape
        k_folds = self.cv
        if k_folds > n:
            raise ValueError(f"cv = {k_folds} exceeds the {n} rows: every fold needs a row")
        gen = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(n, generator=gen)
        folds = torch.empty(n, dtype=torch.long)
        for k in range(k_folds):
            folds[perm[k::k_folds]] = k
        names = self._member_names(d, has_missing)
        models: list[JointRegressor] = []
        for k in range(k_folds):
            rows = (folds != k).nonzero().squeeze(1)
            models.append(self._fold_model(k, names).fit(x[rows], y[rows]))
        self._fold_models, self.folds_, self.members_ = models, folds, names
        dev = self.device
        filled = torch.where(observed, y, torch.zeros_like(y))

        # Out-of-fold log densities (data units) of each head, NaN where not scored.
        log_density = torch.full((n, len(names)), math.nan, dtype=torch.float64)
        energy_sum: dict[str, float] = {}
        energy_rows: dict[str, int] = {}
        for k, model in enumerate(models):
            rows = (folds == k).nonzero().squeeze(1)
            assert model._y_stats is not None
            mu, sd = model._y_stats
            xs = model._standardised(x[rows]).to(dev)
            obs = observed[rows].to(dev)
            ys = model._head_targets(xs, filled[rows].to(dev), obs)
            log_sd = sd.log().to(dev)
            for h, name in enumerate(names):
                if name == "energy":
                    head = model._heads[name][0]
                    assert isinstance(head, _EnergyHead)
                    noise_gen = torch.Generator().manual_seed(self.seed + h)
                    eps = head.noise(int(rows.numel()), _ENERGY_VAL_DRAWS, xs, noise_gen)
                    with torch.no_grad():
                        es = _masked_energy_score(
                            head.generate(xs, eps) * sd.to(dev),
                            ys * sd.to(dev),
                            obs if has_missing else None,
                        )
                    energy_sum[name] = energy_sum.get(name, 0.0) + float(es.sum())
                    energy_rows[name] = energy_rows.get(name, 0) + int(obs.any(-1).sum())
                    continue
                masked = has_missing and name != "sklar" and model._heads[name][0].supports_missing
                sel = (
                    torch.arange(rows.numel(), device=dev)
                    if masked or not has_missing
                    else obs.all(1).nonzero().squeeze(1)
                )
                if sel.numel() == 0:
                    continue
                with torch.no_grad():
                    lp = model._member_predictive(name, xs[sel]).log_prob(
                        ys[sel], obs[sel] if masked else None
                    )
                seen = obs[sel] if masked else torch.ones_like(obs[sel])
                jac = (seen.to(lp.dtype) * log_sd.to(lp.dtype)).sum(-1)
                log_density[rows[sel.cpu()], h] = (lp - jac).cpu().double()
        self.validation_log_score_ = {
            name: float(log_density[:, h][~torch.isnan(log_density[:, h])].mean())
            for h, name in enumerate(names)
            if name != "energy"
        }
        self.validation_energy_score_ = {
            name: energy_sum[name] / max(energy_rows[name], 1) for name in energy_sum
        }
        if len(names) == 1:
            self.weights_ = torch.ones(1)
        else:
            self.weights_ = _stack(log_density).to(torch.float32).cpu()

        # Sizes, choices of the Sklar route (per fold, the most frequent reported).
        self.n_train_ = min(int(m.n_train_ or 0) for m in models)
        self.n_val_ = min(int(m.n_val_ or 0) for m in models)
        self.n_recal_ = self.n_conformal_ = n
        self.batch_size_ = max(int(m.batch_size_ or 0) for m in models)
        self.fold_choices_ = {}
        if "sklar" in names:
            self.fold_choices_ = {
                "marginal_engine": [m.marginal_engine_ for m in models],
                "dependence": [m.dependence_ for m in models],
            }
        self.marginal_engine_ = _most_frequent([m.marginal_engine_ for m in models])
        self.dependence_ = _most_frequent([m.dependence_ for m in models])
        self.marginal_scores_ = _mean_tables([m.marginal_scores_ for m in models])
        self.dependence_scores_ = _mean_tables([m.dependence_scores_ for m in models])

        # One recalibration from the out-of-fold predictive of every complete row.
        self._recalibrator, self.scale_ = None, None
        blocks: list[JointPredictive] = []
        targets: list[Tensor] = []
        for k, model in enumerate(models):
            rows = ((folds == k) & observed.all(1)).nonzero().squeeze(1)
            if rows.numel() > 0:
                blocks.append(self._fold_base(k, x[rows]))
                targets.append(y[rows])
        n_complete = sum(int(t.shape[0]) for t in targets)
        if self.recalibrate and n_complete >= 8:
            base = _RowBlockJoint(blocks)
            recalibrator = ScaleRecalibrator(
                objective="nll" if base.has_density else "variance",
                n_samples=256,
                about="center" if base.has_density else "mean",
            )
            recalibrator.fit(base, torch.cat(targets))
            fitted = recalibrator.scale_
            assert fitted is not None
            self.scale_ = torch.ones_like(fitted.detach().cpu())
            # No rows are left to accept or reject the factors on: only the 2% rule
            # (and, for densities, the recalibrator never scoring worse than s = 1).
            if float(fitted.log().abs().max()) > 0.02:
                self._recalibrator = recalibrator
                self.scale_ = fitted.detach().cpu()
        self._conformal_rows = None
        # A copy: the scores are computed at the first region() call, from these rows.
        self._cross_rows = (x.clone(), filled, observed)
        self._cross_cache = {}
        return self

    def _fold_base(self, k: int, x: Tensor, *, own: bool = False) -> JointPredictive:
        """Stacked predictive of fold model ``k`` in data units, before recalibration.

        ``own=True`` uses the fold model's own stacking weights (fitted on its inner
        validation rows) instead of the out-of-fold weights: the cross-conformal
        scores need a fold model that depends on no row of its fold.
        """
        model = self._fold_models[k]
        weights = model._own_weights if own else self.weights_
        assert weights is not None
        return model._base_predictive(x, weights=weights)

    def _fold_predictives(self, x: Tensor) -> list[JointPredictive]:
        """Recalibrated stacked predictive of every fold model for the rows of ``x``."""
        out = []
        for k in range(len(self._fold_models)):
            base = self._fold_base(k, x)
            out.append(base if self._recalibrator is None else self._recalibrator.transform(base))
        return out

    def calibrate(self, X: ArrayLike, Y: ArrayLike) -> "JointRegressor":
        """Replace the conformal rows with calibration data ``X [N, p]``, ``Y [N, d]``.

        For example fresh simulations.  These rows must **not** have been used to
        fit the model (training, validation or recalibration): the coverage
        guarantee of :meth:`region` rests on their exchangeability with test
        rows under the fitted model.  ``NaN`` marks a missing target (``±inf`` is
        an error).  Clears the cached calibrations; returns ``self``.

        With ``cv`` the external rows replace the cross-conformal calibration:
        :meth:`region` is then the split conformal region of the averaged
        predictive of :meth:`predict` on these rows, with the
        :math:`1 - \alpha` guarantee (refit to go back to cross-conformal).
        """
        if self.n_features_in_ is None or self.n_targets_ is None:
            raise RuntimeError("Call fit() first.")
        x = _float_tensor(X, "X").cpu()
        y = _targets(Y).cpu()
        if x.dim() != 2 or y.dim() != 2 or x.shape[0] != y.shape[0]:
            raise _rows_error(x, y)
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
        # A copy of X: the scores are computed at the first region() call, and the
        # caller's array may have changed by then.
        self._conformal_rows = (x.clone(), filled, observed)
        self._conformal_cache = {}
        # External rows replace the cross-conformal calibration (cv): split conformal.
        self._cross_rows, self._cross_cache = None, {}
        self.n_conformal_ = int(x.shape[0])
        return self

    def _member_names(self, d: int, has_missing: bool) -> list[str]:
        if self._forced_names is not None:
            return list(self._forced_names)
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
            if d <= _AUTO_FLOW_MAX_TARGETS:
                names.append("flow")
        return names

    def _build_head(self, name: str, in_dim: int, d: int, y: Tensor, mask: Tensor) -> _Head:
        if name == "energy":
            noise_dim = d if self.noise_dim is None else self.noise_dim
            return _EnergyHead(in_dim, d, self.hidden, noise_dim, self.energy_draws)
        if name == "gaussian":
            rank = self.rank if self.rank is not None else (None if d <= 32 else 8)
            return _GaussianHead(in_dim, d, self.hidden, rank)
        if name == "mixture":
            spread = recommended_tail_bound(y[mask].detach().cpu())
            return _MixtureHead(in_dim, d, self.hidden, self.n_components, max_std=spread)
        bound = recommended_tail_bound(y[mask.all(1)].detach().cpu())
        return _FlowHead(in_dim, d, self.hidden, bound, n_transforms=3)

    def _marginal_templates(self) -> dict[str, MarginalEngine]:
        """Unfitted marginal engines to choose from: the user's, or both built-in ones."""
        if self.marginals is not None:
            return {type(self.marginals).__name__: self.marginals}
        assert self.batch_size_ is not None
        budget: dict[str, Any] = {
            "hidden": self.hidden,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "batch_size": self.batch_size_,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
        }
        return {
            "quantile": QuantileNetworkEngine(**budget),
            "mixture": MixtureNetworkEngine(self.n_components, **budget),
        }

    def _fit_sklar(
        self, xs: Tensor, ys: Tensor, mask: Tensor, train: Tensor, val: Tensor, gen: torch.Generator
    ) -> None:
        """Marginal engine, out-of-fold normal scores, and the dependence ladder."""
        assert self.batch_size_ is not None and self._y_stats is not None
        batch = self.batch_size_
        complete = train[mask[train].all(1)]
        if self.n_folds > int(complete.numel()):
            raise ValueError(
                f"n_folds = {self.n_folds} exceeds the {int(complete.numel())} complete "
                "training rows of the Sklar route"
            )
        folds = torch.randperm(int(complete.numel()), generator=gen).to(complete.device)
        val_complete = val[mask[val].all(1)]
        # Choose the engine from one fit of each on the training rows, by the held-out
        # sum of marginal log densities (data units); only the chosen one is refitted
        # out of fold.
        log_jac = float(self._y_stats[1].log().sum())
        fitted: dict[str, MarginalEngine] = {}
        self.marginal_scores_ = {}
        templates = self._marginal_templates()
        for name, template in templates.items():
            engine = copy.deepcopy(template).fit(xs[train], ys[train], mask[train])
            with torch.no_grad():
                lp = engine.predict(xs[val_complete]).log_prob(ys[val_complete]).sum(-1)
            fitted[name] = engine
            self.marginal_scores_[name] = float(lp.mean()) - log_jac
        finite = {k: v for k, v in self.marginal_scores_.items() if math.isfinite(v)}
        chosen = max(finite, key=finite.__getitem__) if finite else next(iter(templates))
        self.marginal_engine_ = chosen
        self._marginal_candidates = fitted
        template, self._engine = templates[chosen], fitted[chosen]
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
        z_val = self._engine.predict(xs[val_complete]).normal_scores(ys[val_complete])
        # Simplest first.  The affine rungs absorb a global bias or mis-scaling of the
        # marginals, so the conditional and flow rungs are kept for dependence only.
        candidates: list[Dependence] = [
            IndependenceDependence(),
            IndependenceDependence(affine=True),
        ]
        if ys.shape[1] > 1:  # a copula on one target is the independence copula
            candidates += [
                GaussianCopula(),
                GaussianCopula(affine=True),
                ConditionalGaussianCopula(seed=self.seed, batch_size=batch),
                MixtureDependence(seed=self.seed, batch_size=batch),
                FlowDependence(seed=self.seed, batch_size=batch),
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
        copies = [head.predictive(xs) for head in self._heads[name]]
        return copies[0] if len(copies) == 1 else MixtureJoint(copies)

    @property
    def n_parameters_(self) -> int:
        """Number of network parameters of the fitted model.

        Every ensemble copy of every head is counted; for the Sklar route, the
        networks of the marginal engine and of the chosen dependence rung
        (closed-form rungs have none).
        """
        if self.weights_ is None:
            raise RuntimeError("Call fit() first.")
        if self._fold_models:
            return sum(model.n_parameters_ for model in self._fold_models)

        def count(holder: object) -> int:
            modules = [holder] if isinstance(holder, nn.Module) else list(vars(holder).values())
            return sum(
                int(param.numel())
                for module in modules
                if isinstance(module, nn.Module)
                for param in module.parameters()
            )

        total = sum(count(head) for copies in self._heads.values() for head in copies)
        for part in (self._engine, self._dependence, self._location):
            if part is not None:
                total += count(part)
        return total

    # ------------------------------------------------------------ location
    def _fit_location(
        self, xs: Tensor, ys: Tensor, mask: Tensor, train_d: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Fit the point model of ``location="mse"``; return the heads' targets and their scale.

        ``K`` networks are trained by squared error on the training rows, each on the
        other ``K - 1`` folds (with an inner validation split for early stopping).  A
        training row is predicted by the network that did not see it; every other row
        (validation, recalibration, conformal, and any row at ``predict``) by the
        average of the ``K`` networks, none of which saw it.  In-sample residuals would
        be too small: a density fitted to them is overconfident on new rows.

        Returns the residuals of the standardised targets ``ys`` divided by the
        per-target spread of the out-of-fold residuals (zero where a target is
        missing), and that spread ``[d]``.
        """
        dev, d = xs.device, ys.shape[1]
        n_train = int(train_d.numel())
        k_folds = max(2, min(_LOCATION_FOLDS, n_train // 8))
        gen = torch.Generator().manual_seed(self.seed + 7411)
        order = train_d[torch.randperm(n_train, generator=gen).to(dev)]
        folds = [order[k::k_folds] for k in range(k_folds)]
        assert self.batch_size_ is not None
        nets: list[nn.Module] = []
        out_of_fold = torch.zeros_like(ys)
        for k, held in enumerate(folds):
            rows = torch.cat([f for j, f in enumerate(folds) if j != k])
            n_val = max(1, int(round(self.val_fraction * rows.numel())))
            rows_va, rows_tr = rows[:n_val], rows[n_val:]
            torch.manual_seed(self.seed + 7411 + k)
            net = mlp(xs.shape[1], self.hidden, d).to(dev)

            def loss(r: Tensor, net: nn.Module = net) -> Tensor:
                weight = mask[r].to(ys.dtype)
                return ((net(xs[r]) - ys[r]).square() * weight).sum() / weight.sum().clamp_min(1)

            def batch_loss(idx: Tensor, rows: Tensor = rows_tr, loss: Any = loss) -> Tensor:
                return cast(Tensor, loss(rows[idx.to(rows.device)]))

            def validation(rows: Tensor = rows_va, loss: Any = loss) -> Tensor:
                return cast(Tensor, loss(rows))

            train_module(
                net,
                batch_loss,
                int(rows_tr.numel()),
                validation,
                max_epochs=self.max_epochs,
                patience=self.patience,
                batch_size=self.batch_size_,
                lr=self.lr,
                weight_decay=self.weight_decay,
                generator=gen,
                optimizer="adamw",
                clip_norm=10.0,
                skip_nonfinite=True,
                name=f"location network {k}",
            )
            for param in net.parameters():
                param.requires_grad_(False)
            with torch.no_grad():
                out_of_fold[held] = net(xs[held])
            nets.append(net)
        self._location = nn.ModuleList(nets)
        # Rows outside the training split: the average of the networks.  The training
        # rows keep their out-of-fold prediction and are not evaluated again.
        other = torch.ones(xs.shape[0], dtype=torch.bool, device=dev)
        other[train_d] = False
        location = out_of_fold
        location[other] = _location_mean(nets, xs[other])
        residual = torch.where(mask, ys - location, torch.zeros_like(ys))
        observed = mask[train_d].to(ys.dtype)
        spread = (
            ((residual[train_d].square() * observed).sum(0) / observed.sum(0).clamp_min(1))
            .sqrt()
            .clamp_min(1e-6)
        )
        self._location_div = spread
        return residual / spread, spread

    def _location_at(self, xs: Tensor) -> Tensor:
        """The point model at standardised features ``xs``, in units of the residual spread."""
        assert self._location is not None and self._location_div is not None
        mean = _location_mean(self._location, xs)
        return mean / self._location_div.to(mean)

    def _head_targets(self, xs: Tensor, y: Tensor, observed: Tensor) -> Tensor:
        """Targets in the space the heads model (zero where missing), from data-unit ``y``."""
        assert self._y_stats is not None
        mu, sd = self._y_stats
        target = (y - mu.to(y)) / sd.to(y)
        if self._location is not None:
            target = target - self._location_at(xs)
        return torch.where(observed, target, torch.zeros_like(target))

    def _standardised(self, x: Tensor) -> Tensor:
        """Standardised features, clamped to the training range when ``clip_features``."""
        assert self._x_stats is not None
        xs = (x - self._x_stats[0]) / self._x_stats[1]
        if self.clip_features and self._x_range is not None:
            xs = torch.maximum(torch.minimum(xs, self._x_range[1]), self._x_range[0])
        return xs

    def _n_copies(self) -> int:
        """Ensemble copies of each network head (see ``n_members``)."""
        if self.n_members is not None:
            return self.n_members
        return 3 if self.method == "auto" else 1

    def _train_head(
        self,
        head: _Head,
        name: str,
        xs: Tensor,
        ys: Tensor,
        mask_arg: Optional[Tensor],
        train_d: Tensor,
        val_d: Tensor,
        mask: Tensor,
        gen: torch.Generator,
        offset: int,
    ) -> None:
        """Train one density head with early stopping on the validation log score."""
        rows_tr = train_d if head.supports_missing else train_d[mask[train_d].all(1)]
        rows_va = val_d if head.supports_missing else val_d[mask[val_d].all(1)]
        m_arg = mask_arg if head.supports_missing else None

        def batch_loss(idx: Tensor) -> Tensor:
            r = rows_tr[idx.to(rows_tr.device)]
            return head.loss(xs[r], ys[r], None if m_arg is None else m_arg[r])

        def validation() -> Tensor:
            return head.loss(xs[rows_va], ys[rows_va], None if m_arg is None else m_arg[rows_va])

        train_module(
            head,
            batch_loss,
            int(rows_tr.numel()),
            validation,
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=self.batch_size_,
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

    def _base_predictive(self, x: Tensor, weights: Optional[Tensor] = None) -> JointPredictive:
        """Stacked predictive in data units, before recalibration.

        ``weights`` replaces the stacking weights ``weights_`` (fold models of
        ``cv``, whose weights come from the out-of-fold densities).
        """
        if self._x_stats is None or self._y_stats is None or self.weights_ is None:
            raise RuntimeError("Call fit() first.")
        stack_w = self.weights_ if weights is None else weights
        xs = self._standardised(x).to(self.device)
        with torch.no_grad():
            members = [self._member_predictive(name, xs) for name in self.members_]
        inner: JointPredictive
        if len(members) == 1:
            inner = members[0]
        else:
            keep = stack_w > 1e-4
            if not bool(keep.any()):
                keep = stack_w == stack_w.max()
            kept = [m for m, k in zip(members, keep.tolist()) if k]
            weights = stack_w[keep] / stack_w[keep].sum()
            # One flat mixture: the copies of an ensembled head share its stacking weight.
            flat: list[JointPredictive] = []
            flat_w: list[float] = []
            for member, weight in zip(kept, weights.tolist()):
                parts = member.members if isinstance(member, MixtureJoint) else [member]
                flat += parts
                flat_w += [weight / len(parts)] * len(parts)
            flat_weights = torch.tensor(flat_w, dtype=torch.float32, device=self.device)
            inner = flat[0] if len(flat) == 1 else MixtureJoint(flat, flat_weights)
        loc, scale = self._y_stats
        loc, scale = loc.to(self.device), scale.to(self.device)
        if self._location is not None:
            loc = loc + scale * self._location_at(xs)  # [n, d]: the point model's prediction
        if isinstance(inner, GaussianJoint):
            # Stay Gaussian in data units: closed-form marginals, conditionals and CDFs.
            return inner.affine(loc, scale)
        return AffineJoint(inner, loc, scale)

    def predict(self, X: ArrayLike) -> JointPredictive:
        """Joint predictive distributions for the rows of ``X`` (data units, recalibrated).

        ``X [n, p]`` must be finite, as at :meth:`fit`: a row with a ``NaN`` feature
        has no prediction, and an error says so rather than a ``NaN`` density later.
        """
        x = self._features(X)
        if self._fold_models:
            return _average(self._fold_predictives(x))
        base = self._base_predictive(x)
        return base if self._recalibrator is None else self._recalibrator.transform(base)

    def _features(self, X: ArrayLike) -> Tensor:
        """Features of new rows as a finite float32 ``[n, p]`` tensor."""
        if self.n_features_in_ is None:
            raise RuntimeError("Call fit() first.")
        x = _float_tensor(X, "X")
        if x.dim() != 2 or x.shape[1] != self.n_features_in_:
            one_row = x.dim() == 1 and x.numel() == self.n_features_in_
            hint = "; for a single row pass X[None]" if one_row else ""
            raise ValueError(f"X must be [n, {self.n_features_in_}], got {tuple(x.shape)}{hint}")
        finite = torch.isfinite(x).all(dim=1)
        if not bool(finite.all()):
            raise ValueError(
                f"X must be finite: {int((~finite).sum())} of {x.shape[0]} rows hold NaN or inf"
            )
        return x.cpu()  # the feature statistics are on the CPU; networks move it

    def __repr__(self) -> str:
        # The arguments that differ from their defaults, in the order of the signature.
        shown = []
        for name, param in inspect.signature(type(self).__init__).parameters.items():
            if name == "self" or not hasattr(self, name):
                continue
            value = getattr(self, name)
            default = param.default
            if isinstance(default, (list, tuple)):
                default = tuple(default)
            if value is default or (type(value) is type(default) and value == default):
                continue
            if isinstance(value, torch.device):
                value = str(value)  # the device in use, also when it was left to the default
            shown.append(f"{name}={value!r}")
        return f"{type(self).__name__}({', '.join(shown)})"

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

        **Cross-conformal** (``cv=K``, no :meth:`calibrate`).  Every training row
        :math:`i` is scored by the fold model that did not see it,
        :math:`s_i = s(\hat p_{k(i)}(\cdot \mid x_i), y_i)`; a candidate
        :math:`y` at :math:`x` gets the ``K`` scores
        :math:`t_k(y) = s(\hat p_k(\cdot \mid x), y)` and

        .. math::
            R_\alpha(x) = \Big\{ y : \frac{1 + \sum_{i=1}^N
            \mathbf 1[s_i \ge t_{k(i)}(y)]}{N + 1} > \alpha \Big\}

        (Vovk 2015), a :class:`~torchregress.joint.CrossConformalRegion`.  For
        exchangeable rows Barber et al. (2021, Theorem 4 and Remark 1) prove
        :math:`P(Y \in R_\alpha(X)) \ge 1 - 2\alpha - \min\{2(1 - 1/K)/(N/K + 1),
        (1 - K/N)/(K + 1)\}`, not :math:`1 - \alpha`; realised coverage is
        typically close to :math:`1 - \alpha`.  The score is chosen per fold
        predictive as above (``region.score.bases``, one per fold) with the same number of draws
        for training and test rows.  With missing targets the two calibrations
        above (complete rows; ``contains(y, mask=)``) are kept, from out-of-fold
        scores and with this weaker bound in place of :math:`1 - \alpha`.  The
        out-of-fold scores are cached per (``interest``, ``score``).

        Args:
            X: Features ``[n, p]``.
            alpha: Miscoverage level(s).
            interest: Targets of interest.  The region is then built for that
                sub-vector, with the other (nuisance) targets integrated out of the
                predictive before calibration.
        """
        rows = self._cross_rows if self._cross_rows is not None else self._conformal_rows
        if rows is None:
            raise RuntimeError("Call fit() first.")
        dims = None if interest is None else tuple(target_index(interest, rows[1].shape[1]))
        levels = (
            (float(alpha),) if isinstance(alpha, (int, float)) else tuple(float(a) for a in alpha)
        )
        if self._cross_rows is not None:
            return self._cross_region(X, alpha, levels, dims)
        if self._conformal_rows is None:
            raise RuntimeError("Call fit() first.")
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

    def _cross_region(
        self,
        X: ArrayLike,
        alpha: Union[float, Sequence[float]],
        levels: tuple[float, ...],
        dims: Optional[tuple[int, ...]],
    ) -> JointRegion:
        """Cross-conformal region from the out-of-fold scores of the training rows."""
        assert self._cross_rows is not None and self.folds_ is not None
        x_cal, y_cal, m_cal = self._cross_rows
        folds = self.folds_
        n_folds = len(self._fold_models)

        def restrict(pred: JointPredictive) -> JointPredictive:
            return pred if dims is None else pred.marginal(list(dims))

        def fold_scores(
            scores_k: Sequence[ConformalScore], rows: Tensor, masked: bool
        ) -> list[Tensor]:
            out = []
            for k, score in enumerate(scores_k):
                idx = (rows & (folds == k)).nonzero().squeeze(1)
                if idx.numel() == 0:
                    out.append(torch.empty(0))
                    continue
                pred = restrict(self._fold_base(k, x_cal[idx], own=True))
                score.check(pred)
                if masked:
                    score.check_mask(pred)
                y_k = pred.as_input(y_c[idx])
                m_k = m_c[idx].to(y_k.device) if masked else None
                with torch.no_grad():
                    s = score._evaluate(pred, y_k, m_k, score._prepare(pred))
                if m_k is not None:
                    s = s[m_k.any(-1)]  # rows with nothing observed carry no score
                out.append(s.detach().cpu())
            return out

        y_c = y_cal if dims is None else y_cal[:, list(dims)]
        m_c = m_cal if dims is None else m_cal[:, list(dims)]
        complete = m_c.all(1)
        n_complete = int(complete.sum())
        need = math.ceil(1.0 / min(levels) - 1.0 - 1e-9)
        if n_complete < max(need, 1):
            raise ValueError(
                f"only {n_complete} training rows have every target observed; a "
                f"complete-row region at alpha = {min(levels)} needs at least {max(need, 1)}"
            )
        key = (dims, self.score)
        if key not in self._cross_cache:
            # Chosen per fold predictive (the fold models' own stacks may differ).
            score = [
                self._score_for(restrict(self._fold_base(k, x_cal[:1], own=True)), False)
                for k in range(n_folds)
            ]
            scores = fold_scores(score, complete, False)
            masked_score: Optional[ConformalScore] = None
            masked_scores: list[Tensor] = []
            if not bool(complete.all()) and (
                not isinstance(self.score, ConformalScore) or self.score.supports_mask
            ):
                masked_score = (
                    self.score
                    if isinstance(self.score, ConformalScore) and self.score.supports_mask
                    else MahalanobisScore()
                )
                masked_scores = fold_scores([masked_score] * n_folds, m_c.any(1), True)
            self._cross_cache[key] = (score, scores, masked_score, masked_scores)
        score, scores, masked_score, masked_scores = self._cross_cache[key]
        x = self._features(X)
        tests = [restrict(self._fold_base(k, x, own=True)) for k in range(n_folds)]
        proposal = _average(tests)
        region = CrossConformalRegion(proposal, score, alpha, tests, scores)
        if masked_score is not None:
            region.mask_region = CrossConformalRegion(
                proposal, masked_score, alpha, tests, masked_scores, masked=True
            )
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


def _most_frequent(values: Sequence[Optional[str]]) -> Optional[str]:
    """The most frequent non-``None`` value (the first one seen among ties), or ``None``."""
    present = [v for v in values if v is not None]
    if not present:
        return None
    counts = Counter(present)
    top = max(counts.values())
    return next(v for v in present if counts[v] == top)


def _mean_tables(tables: Sequence[dict[str, float]]) -> dict[str, float]:
    """Key-wise mean of score tables (keys of the first table present in every one)."""
    if not tables or not tables[0]:
        return {}
    keys = [k for k in tables[0] if all(k in t for t in tables)]
    return {k: sum(t[k] for t in tables) / len(tables) for k in keys}


def _compose(part: JointPredictive, loc: Tensor, scale: Tensor) -> JointPredictive:
    """``loc + scale * Y`` for ``Y ~ part``, one wrapper deep (Gaussians stay Gaussian)."""
    if isinstance(part, GaussianJoint):
        return part.affine(loc, scale)
    if isinstance(part, AffineJoint):
        inner_loc, inner_scale = part.loc.to(loc), part.scale.to(scale)
        return AffineJoint(part.base, loc + scale * inner_loc, scale * inner_scale)
    return AffineJoint(part, loc, scale)


def _flatten(pred: JointPredictive) -> list[tuple[JointPredictive, float]]:
    """Components and weights of nested mixtures and affine maps, as one flat mixture."""
    if isinstance(pred, MixtureJoint) and pred.weights.dim() == 1:
        out: list[tuple[JointPredictive, float]] = []
        for member, weight in zip(pred.members, pred.weights.tolist()):
            out += [(part, weight * w) for part, w in _flatten(member)]
        return out
    if isinstance(pred, AffineJoint):
        return [(_compose(part, pred.loc, pred.scale), w) for part, w in _flatten(pred.base)]
    return [(pred, 1.0)]


def _average(preds: Sequence[JointPredictive]) -> JointPredictive:
    """Equal-weight average of predictives as one flat :class:`MixtureJoint`."""
    parts: list[JointPredictive] = []
    weights: list[float] = []
    for pred in preds:
        for part, w in _flatten(pred):
            if w > 0.0:
                parts.append(part)
                weights.append(w / len(preds))
    if len(parts) == 1:
        return parts[0]
    w64 = torch.tensor(weights, dtype=torch.float64)
    w32 = (w64 / w64.sum()).to(torch.float32)
    return MixtureJoint(parts, w32.to(parts[0].device))


class _RowBlockJoint(JointPredictive):
    """Rows ``0..a`` from predictive A, ``a..b`` from B, ...: blocks stacked along the rows.

    The out-of-fold predictive of cross-fitting (each row predicted by the fold
    model that did not see it).  Capabilities are those every block has.
    """

    def __init__(self, blocks: Sequence[JointPredictive]) -> None:
        if not blocks:
            raise ValueError("at least one block is required")
        d = blocks[0].event_dim
        if any(b.event_dim != d for b in blocks):
            raise ValueError("blocks must agree on d")
        self.blocks = list(blocks)
        self._sizes = [b.batch_size for b in blocks]
        self.has_density = all(b.has_density for b in blocks)
        self.has_latent = all(b.has_latent for b in blocks)
        self.exact_marginals = all(b.exact_marginals for b in blocks)

    @property
    def batch_size(self) -> int:
        return sum(self._sizes)

    @property
    def event_dim(self) -> int:
        return self.blocks[0].event_dim

    @property
    def dtype(self) -> torch.dtype:
        return self.blocks[0].dtype

    @property
    def device(self) -> torch.device:
        return self.blocks[0].device

    def _split(self, t: Tensor) -> tuple[Tensor, ...]:
        return torch.split(t, self._sizes, dim=0)

    def sample(self, n_samples: int) -> Tensor:
        return torch.cat([b.sample(n_samples) for b in self.blocks], dim=0)

    def log_prob(self, y: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        y = self._check(y)
        masks: Sequence[Optional[Tensor]] = (
            [None] * len(self.blocks) if mask is None else self._split(self._check_mask(y, mask))
        )
        return torch.cat(
            [b.log_prob(part, m) for b, part, m in zip(self.blocks, self._split(y), masks)], dim=0
        )

    def to_latent(self, y: Tensor) -> Tensor:
        y = self._check(y)
        return torch.cat([b.to_latent(p) for b, p in zip(self.blocks, self._split(y))], dim=0)

    def from_latent(self, z: Tensor) -> Tensor:
        z = self._check(z)
        return torch.cat([b.from_latent(p) for b, p in zip(self.blocks, self._split(z))], dim=0)

    def marginal(self, dims: Union[int, Sequence[int]]) -> "_RowBlockJoint":
        return _RowBlockJoint([b.marginal(dims) for b in self.blocks])

    def mean(self, n_samples: int = 512) -> Tensor:
        return torch.cat([b.mean(n_samples) for b in self.blocks], dim=0)

    def center(self, n_samples: int = 512) -> Tensor:
        return torch.cat([b.center(n_samples) for b in self.blocks], dim=0)

    def covariance(self, n_samples: int = 512) -> Tensor:
        return torch.cat([b.covariance(n_samples) for b in self.blocks], dim=0)
