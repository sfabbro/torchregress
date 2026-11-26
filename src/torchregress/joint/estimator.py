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
from ..metrics.distribution import energy_score
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
    HPDScore,
    JointConformal,
    JointRegion,
    LatentScore,
    MahalanobisScore,
    SampleBallScore,
)

__all__ = ["JointRegressor", "MarginalEngine", "MixtureNetworkEngine", "QuantileNetworkEngine"]

_ENERGY_VAL_DRAWS = 16  # fixed draws per validation row for the energy head

Method = Literal["auto", "gaussian", "mixture", "flow", "sklar", "energy"]
ArrayLike = Any  # torch.Tensor or anything torch.as_tensor accepts (numpy arrays)


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

    Args:
        n_components: Mixture components per target, ``K``.
        hidden: Hidden-layer widths.
        min_std: Floor on the component standard deviations.
        max_epochs, patience, batch_size, lr, weight_decay: Training budget.
        val_fraction: Rows held out for early stopping.
        seed: Seed of the initialisation, the split and the shuffling.
    """

    _name = "mixture engine"

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

    def _mixture(self, params: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Log weights, means and standard deviations ``[b, d, K]``."""
        mix = self._mdn.distribution(params)  # batch [b, d], one feature
        return mix.log_weights, mix.loc[..., 0], mix.component_scale[..., 0]

    def _entry_loss(self, params: Tensor, y: Tensor) -> Tensor:
        return -self._mdn.distribution(params).log_prob(y.unsqueeze(-1))

    def _marginals(self, params: Tensor) -> MixtureMarginals:
        log_w, loc, scale = self._mixture(params)
        return MixtureMarginals(log_w.exp(), loc, scale)

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
        batch_size: Minibatch size, or ``"auto"`` (default): 256, doubled while an
            epoch would take more than 400 steps, up to 4096 (256 up to about
            100,000 training rows, 4096 from about 800,000).  The size used is
            ``batch_size_``.
        val_fraction, recal_fraction, conformal_fraction: Held-out shares for early
            stopping and stacking, for recalibration, and for conformal calibration.
        max_holdout: Cap on the rows of each held-out split.  Beyond it more
            held-out rows add little (a conformal quantile from 20,000 rows is
            within :math:`5 \times 10^{-5}` of its level) and the rows go to
            training instead.
        recalibrate: Fit the per-target scale recalibration.
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
    targets are missing.

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
            (``n_conformal_`` follows :meth:`calibrate`).
        batch_size_: Minibatch size used by the last :meth:`fit`.
        n_parameters_: Number of network parameters of the fitted model (all copies).
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
        recalibrate: bool = True,
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
        self.batch_size_: Optional[int] = None
        self.lr, self.weight_decay = float(lr), float(weight_decay)
        self.val_fraction, self.recal_fraction = float(val_fraction), float(recal_fraction)
        self.conformal_fraction = float(conformal_fraction)
        self.recalibrate = bool(recalibrate)
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
        cap = self.max_holdout
        n_val = min(cap, max(8, int(round(self.val_fraction * n))))
        n_rec = min(cap, max(8, int(round(self.recal_fraction * n))))
        n_conf = min(cap, max(8, int(round(self.conformal_fraction * n))))
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
        if isinstance(self.batch_size, str):
            batch = 256
            while batch < 4096 and self.n_train_ > 400 * batch:
                batch *= 2
            self.batch_size_ = batch
        else:
            self.batch_size_ = self.batch_size

        dev = self.device
        xs = ((x - x_mean) / x_std).to(dev)
        ys = torch.where(observed, (y - y_mean) / y_std, torch.zeros_like(y)).to(dev)
        mask = observed.to(dev)
        mask_arg: Optional[Tensor] = mask if has_missing else None
        train_d, val_d = train.to(dev), val.to(dev)

        names = self._member_names(d, has_missing)
        log_density: list[Tensor] = []
        self._heads, self.validation_log_score_, self.validation_energy_score_ = {}, {}, {}
        self.marginal_scores_, self.marginal_engine_ = {}, None
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
            stacked = torch.stack(log_density, dim=1).double()
            self.weights_ = _stack(stacked).to(torch.float32).cpu()

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
        if name == "energy":
            noise_dim = d if self.noise_dim is None else self.noise_dim
            return _EnergyHead(in_dim, d, self.hidden, noise_dim, self.energy_draws)
        if name == "gaussian":
            rank = self.rank if self.rank is not None else (None if d <= 32 else 8)
            return _GaussianHead(in_dim, d, self.hidden, rank)
        if name == "mixture":
            return _MixtureHead(in_dim, d, self.hidden, self.n_components)
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

        def count(holder: object) -> int:
            modules = [holder] if isinstance(holder, nn.Module) else list(vars(holder).values())
            return sum(
                int(param.numel())
                for module in modules
                if isinstance(module, nn.Module)
                for param in module.parameters()
            )

        total = sum(count(head) for copies in self._heads.values() for head in copies)
        for part in (self._engine, self._dependence):
            if part is not None:
                total += count(part)
        return total

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
