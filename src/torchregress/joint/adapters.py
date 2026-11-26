r"""Routes from the library's multi-target heads, losses and regions into :mod:`torchregress.joint`.

:func:`as_joint` dispatches on the type of a head, loss, ensemble or
distribution and returns the matching :class:`~torchregress.joint.JointPredictive`
(or, for :class:`~torchregress.losses.SLSConformal`, a
:class:`~torchregress.joint.JointRegion`).  Two heads need their own adapter:

* :class:`CopulaSplineJoint` wraps
  :class:`~torchregress.models.GaussianCopulaSplineHead`, whose draws are
  sample-first and whose density takes a mask, with its exact masked densities
  and its latent map.  A dedicated adapter rather than a generalised
  :class:`~torchregress.joint.ConditionalModelJoint`: the head has a latent map
  and exact marginals that the flow-head protocol does not describe, so
  generalising would add head-specific branches to a protocol class.
* :class:`SLSRegion` wraps a calibrated
  :class:`~torchregress.losses.SLSConformal` (a region method, not a density)
  as a :class:`~torchregress.joint.JointRegion`, with exact volumes and uniform
  draws from the geometry of the frontier.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, Optional, Union, cast

import torch
from torch import Tensor
from torch.distributions import Distribution, Independent, StudentT

from ..ensemble._variance import member_variance_from_logvar
from ..ensemble.base import BaseEnsembleModel, eval_mode
from ..ensemble.bnn import HeteroscedasticBNN
from ..ensemble.density import DensityEnsemble
from ..ensemble.models import (
    HeteroscedasticBatchEnsembleModel,
    HeteroscedasticEnsembleModel,
    MDNEnsembleModel,
)
from ..ensemble.swag import SWAG, MultiSWAG
from ..losses.conformal import SLSConformal
from ..losses.eiv import InputNoiseMDNLoss
from ..losses.evidential import EvidentialRegressionLoss
from ..losses.gaussian import GaussianNLLLoss, LowRankGaussianLoss, MultivariateGaussianLoss
from ..losses.known_covariance import KnownCovarianceGaussianNLL
from ..losses.mdn import MixtureDensityLoss
from ..losses.nflows import NormalizingFlowLoss
from ..losses.noise_aware_density import JointBinnedDensityLoss, NoiseAwareDensityLoss
from ..losses.quantile import MultiQuantileLoss
from ..losses.sls import MahalanobisFrontier, UnionFrontier
from ..models.copula_spline import GaussianCopulaSplineHead
from ..models.flow_matching import FlowMatchingMLP, sample_cfm
from ..utils.gaussian_output import parse_heteroscedastic_output, split_low_rank_gaussian_output
from ..utils.pytorch_compat import _collect_transform_cycles
from .dependence import Dependence, SklarJoint
from .marginals import Marginals, MixtureMarginals, QuantileMarginals
from .predictive import (
    ConditionalModelJoint,
    DistributionJoint,
    FlowJoint,
    GaussianJoint,
    JointPredictive,
    MixtureJoint,
    SampleJoint,
)
from .regions import ConformalScore, JointRegion, _State

__all__ = ["CopulaSplineJoint", "SLSRegion", "as_joint"]


# ═══════════════════════════════════════════════════════════════════════════════
# Gaussian copula spline head
# ═══════════════════════════════════════════════════════════════════════════════


class CopulaSplineJoint(JointPredictive):
    r"""Predictive of a :class:`~torchregress.models.GaussianCopulaSplineHead`.

    The head is :math:`y = T(u)`, :math:`u \sim \mathcal N(\mu(x), L(x) L(x)^\top)`
    with coordinatewise monotone splines :math:`T`.  Densities (and masked
    densities, which integrate missing targets out exactly) are the head's
    ``intrinsic_log_prob``; draws are the head's, batch first; the latent map
    is :math:`z = L^{-1}(T^{-1}(y) - \mu)`, standard normal under the head.

    Args:
        head: The trained head.
        context: Head input for the rows, ``[n, context_dim]``.
    """

    has_density = True
    has_latent = True
    exact_marginals = True

    def __init__(self, head: GaussianCopulaSplineHead, context: Tensor) -> None:
        if not isinstance(head, GaussianCopulaSplineHead):
            raise TypeError(f"head must be a GaussianCopulaSplineHead, got {type(head).__name__}")
        if context.dim() != 2 or context.shape[0] < 1:
            raise ValueError(f"context must be [n, context_dim], got {tuple(context.shape)}")
        self.head = head
        self.context = context

    @property
    def batch_size(self) -> int:
        return int(self.context.shape[0])

    @property
    def event_dim(self) -> int:
        return int(self.head.target_dim)

    @property
    def dtype(self) -> torch.dtype:
        return self.head._out.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.context.device

    def sample(self, n_samples: int) -> Tensor:
        with torch.no_grad():
            return self.head.sample(self.context, int(n_samples)).movedim(0, 1)

    def log_prob(self, y: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        y = self._check(y)
        full = (
            torch.ones(self.batch_size, self.event_dim, dtype=torch.bool, device=y.device)
            if mask is None
            else self._check_mask(y, mask)
        )
        if y.dim() == 2:
            return self.head.intrinsic_log_prob(self.context, y, full)
        return self.head.intrinsic_log_prob(self.context, y.movedim(1, 0), full).movedim(0, 1)

    def to_latent(self, y: Tensor) -> Tensor:
        y = self._check(y)
        with _collect_transform_cycles(), torch.no_grad():
            mean, lower, transform = self.head._distribution_parameters(self.context)
            u = transform.inv(y if y.dim() == 2 else y.movedim(1, 0))  # [n, d] or [S, n, d]
            resid = u - mean
            rhs = resid.unsqueeze(-1) if resid.dim() == 2 else resid.permute(1, 2, 0)
            z = torch.linalg.solve_triangular(lower, rhs, upper=False)  # [n, d, 1 or S]
        return z.squeeze(-1) if y.dim() == 2 else z.mT

    def from_latent(self, z: Tensor) -> Tensor:
        z = self._check(z)
        with torch.no_grad():
            mean, lower, transform = self.head._distribution_parameters(self.context)
            z3 = z if z.dim() == 3 else z.unsqueeze(1)  # [n, S, d]
            u = mean.unsqueeze(1) + z3 @ lower.mT
            y = transform(u.movedim(1, 0)).movedim(0, 1)
        return y if z.dim() == 3 else y[:, 0]

    def select(self, index: Tensor) -> "CopulaSplineJoint":
        """Rows ``index`` (a long or boolean index)."""
        return CopulaSplineJoint(self.head, self.context[index])


# ═══════════════════════════════════════════════════════════════════════════════
# super-level-set regions
# ═══════════════════════════════════════════════════════════════════════════════


class _RegionRows(JointPredictive):
    """Rows, targets, dtype and device of a region without a predictive distribution."""

    def __init__(self, context: Tensor, event_dim: int) -> None:
        self.context = context
        self._d = int(event_dim)

    @property
    def batch_size(self) -> int:
        return int(self.context.shape[0])

    @property
    def event_dim(self) -> int:
        return self._d

    @property
    def dtype(self) -> torch.dtype:
        return self.context.dtype

    @property
    def device(self) -> torch.device:
        return self.context.device

    def sample(self, n_samples: int) -> Tensor:
        raise NotImplementedError(
            "an SLS region has no predictive distribution; SLSRegion.sample draws "
            "uniformly from the region"
        )


class _FrontierScore(ConformalScore):
    """The SLS frontier :math:`G(x, y)` of a calibrated :class:`SLSConformal`."""

    name = "sls"

    def __init__(self, conformal: SLSConformal) -> None:
        self.conformal = conformal

    def _prepare(self, predictive: JointPredictive) -> _State:
        return {"context": getattr(predictive, "context")}

    def _evaluate(
        self, predictive: JointPredictive, y: Tensor, mask: Optional[Tensor], state: _State
    ) -> Tensor:
        return self.conformal._frontier_values(state["context"], y)

    def __repr__(self) -> str:
        return "SLSFrontier()"


class SLSRegion(JointRegion):
    r"""Calibrated super-level-set regions of an :class:`~torchregress.losses.SLSConformal`.

    The region of row :math:`x` is the set :class:`SLSConformal
    <torchregress.losses.SLSConformal>` calibrates,
    :math:`R(x) = \{y : G(x, y) \le t(x)\}` with :math:`t(x) = \hat r\, q_\tau(x)`
    (``threshold``), so :meth:`contains` reproduces ``SLSConformal.contains``
    exactly and carries its marginal guarantee
    :math:`P(Y \in R(X)) \ge 1 - \alpha`.

    Geometry.  A frontier component is
    :math:`G(x, y) = (F_x(y) - \mu)^\top P (F_x(y) - \mu)` with a
    volume-preserving flow :math:`F_x`, so its level set is the image of an
    ellipsoid of volume :math:`V_d\, t^{d/2} \det P^{-1/2}`, exactly, and
    pushing uniform points of the ellipsoid through :math:`F_x^{-1}` gives
    uniform points of the region.  For a union frontier (``K > 1``) the soft
    minimum lies above the smallest component, so the region lies inside the
    union of the component sets; :meth:`sample` draws uniformly from that union
    (component chosen by volume, accepted with probability one over the number
    of components that contain the point, kept when inside the region), and
    :meth:`log_volume` is the matching unbiased Monte Carlo estimate.  The
    region has no predictive distribution: ``predictive`` only carries the rows
    (``sample`` raises), and masks are not supported.

    Args:
        conformal: A calibrated :class:`~torchregress.losses.SLSConformal`.
        x: Test features ``[n, context_dim]``.
        groups: Mondrian group ids (required when calibrated with groups).
        test_weights: Optional ``[n]`` test-point weights (weighted calibration).
    """

    def __init__(
        self,
        conformal: SLSConformal,
        x: Tensor,
        *,
        groups: Optional[Tensor] = None,
        test_weights: Optional[Tensor] = None,
    ) -> None:
        if not isinstance(conformal, SLSConformal):
            raise TypeError(f"conformal must be an SLSConformal, got {type(conformal).__name__}")
        if x.dim() != 2:
            raise ValueError(f"x must be [n, context_dim], got {tuple(x.shape)}")
        level = conformal._frontier_level(x, groups=groups, test_weights=test_weights)
        rows = _RegionRows(x.detach(), int(conformal.sls_loss.d))
        super().__init__(
            rows,
            _FrontierScore(conformal),
            conformal.alpha,
            level.detach().unsqueeze(-1),
            {"context": rows.context},
            False,
            True,
        )
        self.conformal = conformal
        self._context = rows.context

    # ------------------------------------------------------------- geometry
    def _components(self) -> list[tuple[MahalanobisFrontier, Tensor]]:
        """Frontier components with their per-row level ``[n]`` (``G'_k <= t_k``)."""
        frontier = self.conformal.sls_loss.frontier
        x = self._context
        t = self.threshold[:, 0].to(x)
        if isinstance(frontier, MahalanobisFrontier):
            return [(frontier, t)]
        if not isinstance(frontier, UnionFrontier):
            raise TypeError(f"unsupported SLS frontier {type(frontier).__name__}")
        ctx = x if frontier.context_dim > 0 else None
        with torch.no_grad():
            p = frontier._get_mixture_weights(ctx, freeze=False).to(x)
        p = p.expand(x.shape[0], frontier.K).clamp(min=1e-8)
        d = frontier.d
        comps = []
        for k, comp in enumerate(frontier.components):
            assert isinstance(comp, MahalanobisFrontier)
            comps.append((comp, t * p[:, k] ** (2.0 / d)))
        return comps

    def _ellipsoid(self, comp: MahalanobisFrontier) -> tuple[Tensor, Tensor, Tensor]:
        """Centre ``[n, d]``, ``A [n, d, d]`` with ``A A^T = P^{-1}``, and ``log|P| / 2 [n]``."""
        x = self._context
        n, d = x.shape[0], comp.d
        ctx = x if comp.context_dim > 0 else None
        with torch.no_grad():
            mu, params = comp._get_params(ctx)
            l_or_dv, half_log_det = comp._get_L_matrix_and_logdet(params)
            if isinstance(l_or_dv, Tensor):  # full: P = L^T L, A = L^{-1}
                lower = l_or_dv.expand(n, d, d)
                eye = torch.eye(d, dtype=lower.dtype, device=lower.device).expand(n, d, d)
                factor = torch.linalg.solve_triangular(lower, eye, upper=False)
            else:  # low rank: P = diag(D) + V V^T
                diag, v = l_or_dv
                precision = torch.diag_embed(diag).to(device=v.device, dtype=v.dtype) + v @ v.mT
                chol = torch.linalg.cholesky(precision.expand(n, d, d))
                eye = torch.eye(d, dtype=chol.dtype, device=chol.device).expand(n, d, d)
                factor = torch.linalg.solve_triangular(chol, eye, upper=False).mT
        return mu.expand(n, d).to(x), factor.to(x), half_log_det.expand(n).to(x)

    def _push(self, comp: MahalanobisFrontier, latent: Tensor) -> Tensor:
        """Map latent points ``[n, B, d]`` of a component back to targets."""
        x = self._context
        n, b, _ = latent.shape
        ctx = None
        if comp.context_dim > 0:
            ctx = x.unsqueeze(1).expand(n, b, x.shape[-1])
        with torch.no_grad():
            inverse = cast(Callable[[Tensor, Optional[Tensor]], Tensor], comp.flow.inverse)
            return inverse(latent, ctx)

    def _component_log_volumes(self, comps: Sequence[tuple[MahalanobisFrontier, Tensor]]) -> Tensor:
        d = self.predictive.event_dim
        log_ball = 0.5 * d * math.log(math.pi) - math.lgamma(0.5 * d + 1.0)
        out = []
        for comp, t in comps:
            _, _, half_log_det = self._ellipsoid(comp)
            out.append(log_ball + 0.5 * d * t.log() - half_log_det)
        return torch.stack(out, dim=-1)  # [n, K]

    def _union_proposals(
        self, comps: Sequence[tuple[MahalanobisFrontier, Tensor]], n_draws: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Uniform draws from the union of the component sets.

        Returns draws ``[n, B, d]``, the number of components containing each draw
        ``[n, B]`` and the component log-volumes ``[n, K]``.
        """
        x = self._context
        n, d = x.shape[0], self.predictive.event_dim
        log_vol = self._component_log_volumes(comps)
        finite = torch.isfinite(log_vol).all(-1, keepdim=True)
        probs = torch.softmax(torch.where(finite, log_vol, torch.zeros_like(log_vol)), dim=-1)
        pick = torch.multinomial(probs.double(), n_draws, replacement=True)  # [n, B]
        direction = torch.randn(n, n_draws, d, dtype=x.dtype, device=x.device)
        direction = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
        radius = torch.rand(n, n_draws, 1, dtype=x.dtype, device=x.device) ** (1.0 / d)
        ball = direction * radius
        draws = torch.zeros(n, n_draws, d, dtype=x.dtype, device=x.device)
        for k, (comp, t) in enumerate(comps):
            mu, factor, _ = self._ellipsoid(comp)
            scale = t.clamp(min=0.0).sqrt().view(n, 1, 1)
            latent = mu.unsqueeze(1) + scale * (ball @ factor.mT)
            y_k = self._push(comp, latent)
            draws = torch.where((pick == k).unsqueeze(-1), y_k, draws)
        count = torch.zeros(n, n_draws, dtype=x.dtype, device=x.device)
        for comp, t in comps:
            ctx = None
            if comp.context_dim > 0:
                ctx = x.unsqueeze(1).expand(n, n_draws, x.shape[-1])
            with torch.no_grad():
                g_k, _ = comp(draws, ctx)
            count = count + (g_k <= t.unsqueeze(1)).to(count)
        return draws, count.clamp(min=1.0), log_vol

    def _propose(self, n_draws: int, level: int) -> tuple[Tensor, Tensor]:
        comps = self._components()
        draws, count, _ = self._union_proposals(comps, n_draws)
        accept = torch.rand_like(count) < 1.0 / count
        inside = self._inside(draws)[..., level]
        finite = torch.isfinite(self.threshold[:, level]).to(inside.device).unsqueeze(1)
        return draws, inside & accept & finite

    # ------------------------------------------------------------ interface
    def sample(
        self, n_samples: int, *, level: Optional[int] = None, max_rounds: int = 50
    ) -> Tensor:
        """Uniform draws ``[n, S, d]`` from each region (there is no predictive to restrict).

        Exact uniform draws for ``K = 1``; for a union frontier, rejection from
        the union of the component sets (see the class docstring).  Rows with an
        infinite threshold (an unbounded region) are filled with ``nan`` and
        emit a ``RuntimeWarning``, as do rows short of ``S`` accepted draws
        after ``max_rounds`` rounds (see :meth:`JointRegion.sample
        <torchregress.joint.JointRegion.sample>`).
        """
        return super().sample(n_samples, level=level, max_rounds=max_rounds)

    def log_volume(self, n_samples: int = 1024) -> tuple[Tensor, Tensor]:
        r"""Log Lebesgue volume of each region and its standard error, ``[n]``.

        Exact for one frontier component (``K = 1``):
        :math:`\log V_d + \tfrac d2 \log t - \tfrac12 \log\det P` (standard
        error zero).  For a union frontier, the unbiased estimate
        :math:`\hat V = \sum_k V_k \cdot \frac1B \sum_b \mathbf 1[y_b \in R] / m(y_b)`
        from ``n_samples`` uniform draws of the union, with the delta-method
        standard error of its logarithm.  ``+inf`` for an infinite threshold.
        """
        comps = self._components()
        t = self.threshold[:, 0]
        if len(comps) == 1:
            est = self._component_log_volumes(comps)[:, 0]
            se = torch.zeros_like(est)
        else:
            draws, count, log_vol = self._union_proposals(comps, int(n_samples))
            inside = self._inside(draws)[..., 0]
            w = inside.to(count) / count  # [n, B]
            mean = w.mean(dim=1)
            est = torch.logsumexp(log_vol, dim=-1) + mean.log()
            se = w.std(dim=1) / (math.sqrt(n_samples) * mean)
        infinite = torch.isposinf(t).to(est.device)
        est = torch.where(infinite, torch.full_like(est, math.inf), est)
        se = torch.where(infinite, torch.zeros_like(se), se)
        return est, se

    def __repr__(self) -> str:
        return (
            f"SLSRegion(alpha={self.alpha}, K={self.conformal.sls_loss.K}, "
            f"n={self.predictive.batch_size}, d={self.predictive.event_dim})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# dispatch
# ═══════════════════════════════════════════════════════════════════════════════


def _diagonal_gaussian(mean: Tensor, variance: Tensor) -> GaussianJoint:
    return GaussianJoint(
        mean, torch.diag_embed(variance.sqrt()).to(device=mean.device, dtype=mean.dtype)
    )


def _gaussian_mixture(means: Tensor, variances: Tensor) -> MixtureJoint:
    """Equal-weight mixture of diagonal Gaussians, members ``[K, n, d]``."""
    return MixtureJoint([_diagonal_gaussian(m, v) for m, v in zip(means, variances, strict=True)])


def _need(inputs: Any, what: str) -> Any:
    if inputs is None:
        raise ValueError(f"as_joint needs {what} as its second argument")
    return inputs


def _sklar(marginals: Marginals, dependence: Any, x: Optional[Tensor]) -> SklarJoint:
    n, d = marginals.batch_size, marginals.event_dim
    if dependence is None:
        ref = torch.zeros((), dtype=torch.get_default_dtype())
        for t in vars(marginals).values():
            if isinstance(t, Tensor) and t.is_floating_point():
                ref = t
                break
        eye = torch.eye(d, dtype=ref.dtype, device=ref.device)
        scores: JointPredictive = GaussianJoint(ref.new_zeros(n, d), eye)
    elif isinstance(dependence, Dependence):
        scores = dependence.predictive(n, x)
    elif isinstance(dependence, JointPredictive):
        scores = dependence
    else:
        raise TypeError(
            "dependence must be a fitted torchregress.joint.Dependence or a JointPredictive "
            f"of the normal scores, got {type(dependence).__name__}"
        )
    return SklarJoint(marginals, scores)


def _per_target_mixture(loss: MixtureDensityLoss, out: Tensor) -> MixtureMarginals:
    """:class:`MixtureMarginals` of a one-feature MDN applied per target, output ``[n, d, m]``."""
    if not loss.gaussian_components:
        raise ValueError(
            "per-target mixture heads need Gaussian components (df = inf): MixtureMarginals "
            "has no Student-t components"
        )
    with torch.no_grad():
        mix = loss.distribution(out.detach())
    return MixtureMarginals(mix.log_weights.exp(), mix.loc[..., 0], mix.component_scale[..., 0])


_SUPPORTED = (
    "JointPredictive (returned unchanged), torch.distributions.Distribution with batch [n] "
    "and event [d] (StudentTMixture, MultivariateNormal, zuko flows), GaussianNLLLoss, "
    "MultivariateGaussianLoss, LowRankGaussianLoss, KnownCovarianceGaussianNLL, "
    "MixtureDensityLoss (also one feature per target, output [n, d, m]), InputNoiseMDNLoss, "
    "EvidentialRegressionLoss, NormalizingFlowLoss "
    "(and ContrastiveFlowLoss), NoiseAwareDensityLoss, JointBinnedDensityLoss, "
    "MultiQuantileLoss, Marginals, GaussianCopulaSplineHead, FlowMatchingMLP, DensityEnsemble, "
    "HeteroscedasticEnsembleModel, HeteroscedasticBatchEnsembleModel, MDNEnsembleModel, "
    "HeteroscedasticBNN, SLSConformal, MarginalRatioEstimator, any model with "
    "log_prob(context, y) and "
    "sample(context, n) (MarginalContrastiveDensity, NormalizingFlowMap), and any "
    "generator with sample(context, n) only (draws-only predictive)"
)


def as_joint(obj: Any, inputs: Any = None, /, **kwargs: Any) -> Union[JointPredictive, JointRegion]:
    r"""The joint predictive (or joint region) of a multi-target head, loss or ensemble.

    One call from what a model was trained with to :mod:`torchregress.joint`
    (``obj``, then ``inputs`` and keywords, then the result):

    * ``JointPredictive``: returned unchanged (``RatioPosterior`` included).
    * ``MarginalRatioEstimator``, observations ``x`` (``prior_samples=``,
      ``marginal=``, ``prior_log_prob=``, ``n_proposals=``, ``min_ess=``):
      ``obj.posterior(x, ...)``, a ``RatioPosterior`` of one marginal.
    * ``torch.distributions.Distribution`` (batch ``[n]``, event ``[d]``):
      ``DistributionJoint(obj)``.
    * ``GaussianNLLLoss``, ``(mean, log_var)`` or ``[n, 2d]``: diagonal
      ``GaussianJoint`` with the loss's variance floor and ``eps``.
    * ``MultivariateGaussianLoss``, ``(mean, covariance)``: ``GaussianJoint``
      of the loss's covariance (jitter and learned adjustment included).
    * ``LowRankGaussianLoss``, ``(mean, cov_factor, cov_diag)`` or
      ``[n, d (r + 2)]`` with ``rank=r``: ``GaussianJoint.from_low_rank``.
    * ``KnownCovarianceGaussianNLL``, ``y_pred`` with ``known_covariance=``,
      ``known_variance=``, ``model_scale=``, ``model_factor=``:
      ``GaussianJoint.from_known_covariance``.
    * ``MixtureDensityLoss``, network output: ``DistributionJoint`` of
      ``loss.distribution(output)``.  A one-feature loss (``n_features=1``,
      Gaussian components) applied per target, output ``[n, d, m]``
      (``dependence=``, ``x=``): ``SklarJoint`` of ``MixtureMarginals``.
    * ``InputNoiseMDNLoss``, ``x_obs`` (``sigma_x=``, ``n_samples=``):
      ``MixtureJoint`` over the input-noise draws.
    * ``EvidentialRegressionLoss``, network output: independent Student-t
      targets (``DistributionJoint``).
    * ``NormalizingFlowLoss`` (and ``ContrastiveFlowLoss``), context: ``FlowJoint``.
    * ``NoiseAwareDensityLoss``: ``as_joint(loss.base_loss, inputs)`` (the
      clean predictive).
    * ``JointBinnedDensityLoss``, joint logits: ``ConditionalModelJoint``.
    * ``MultiQuantileLoss``, ``[n, Q, d]`` (``dependence=``, ``x=``,
      ``tail_knots=``): ``SklarJoint`` of ``QuantileMarginals``.
    * ``Marginals`` (``dependence=``, ``x=``): ``SklarJoint``.
    * ``GaussianCopulaSplineHead``, context: :class:`CopulaSplineJoint`.
    * ``FlowMatchingMLP``, context (``n_samples=256``, ``steps=32``):
      ``SampleJoint`` of ``sample_cfm`` draws.
    * ``DensityEnsemble``, ``x``: ``MixtureJoint`` of the members with the
      ensemble's weights.
    * ``HeteroscedasticEnsembleModel``, ``HeteroscedasticBatchEnsembleModel``,
      ``HeteroscedasticBNN`` (``n_samples=``), ``x``: ``MixtureJoint`` of
      diagonal Gaussians; ``MDNEnsembleModel``, ``x``: ``MixtureJoint`` of the
      member mixtures.
    * ``SLSConformal`` (calibrated), ``x`` (``groups=``, ``test_weights=``):
      :class:`SLSRegion`, a ``JointRegion``.
    * Any model with ``log_prob(context, y)`` and ``sample(context, n)``
      (``MarginalContrastiveDensity``, ``NormalizingFlowMap``), context:
      ``ConditionalModelJoint`` (``event_dim=``).
    * Any generator with ``sample(context, n)`` and no ``log_prob`` (a
      user-trained engression network, for example), context: draws-only
      ``ConditionalModelJoint`` (``event_dim=``); draws are taken on demand.

    The ``dependence`` of a quantile head is a fitted
    :class:`~torchregress.joint.Dependence` (fit it on out-of-fold normal
    scores, see the joint guide) or a ``JointPredictive`` of the scores;
    independence when omitted.  Point-prediction ensembles and posterior
    samplers (``BaseEnsembleModel``, SWAG, Laplace, MC dropout,
    ``BayesianNeuralNetwork``) describe the spread of the network output, not
    a predictive of ``y``, and raise.

    Raises:
        TypeError: For an unsupported object (the message names the supported types).
    """
    # inference.marginal_ratio builds on joint.predictive, so it is imported here
    # rather than at module level (a module-level import would be circular).
    from ..inference.marginal_ratio import MarginalRatioEstimator

    if isinstance(obj, JointPredictive):
        return obj
    if isinstance(obj, MarginalRatioEstimator):
        return obj.posterior(_need(inputs, "the observations x"), **kwargs)
    if isinstance(obj, SLSConformal):
        return SLSRegion(obj, _need(inputs, "the test features x"), **kwargs)
    if isinstance(obj, Distribution):
        return DistributionJoint(obj)
    if isinstance(obj, Marginals):
        return _sklar(obj, kwargs.get("dependence"), kwargs.get("x"))

    # ---------------------------------------------------------- Gaussian losses
    if isinstance(obj, KnownCovarianceGaussianNLL):
        return GaussianJoint.from_known_covariance(_need(inputs, "y_pred"), loss=obj, **kwargs)
    if isinstance(obj, MultivariateGaussianLoss):
        mean, cov = _need(inputs, "(mean, covariance)")
        with torch.no_grad():
            cov = obj._prepare_covariance(cov, mean.shape[0], mean.shape[-1])
        return GaussianJoint.from_covariance(mean.detach(), cov)
    if isinstance(obj, LowRankGaussianLoss):
        out = _need(inputs, "(mean, cov_factor, cov_diag) or the concatenated output")
        if isinstance(out, Tensor):
            rank = int(kwargs["rank"])
            out = split_low_rank_gaussian_output(out, out.shape[-1] // (rank + 2), rank)
        mean, factor, diag = out
        with torch.no_grad():
            factor, diag = obj._prepare_low_rank(factor, diag, mean.shape[0], mean.shape[-1])
        return GaussianJoint.from_low_rank(mean.detach(), factor, diag)
    if isinstance(obj, GaussianNLLLoss):
        with torch.no_grad():
            mean, var = obj._extract_distribution_parameters(_need(inputs, "the head output"))
        return _diagonal_gaussian(mean.detach(), var + obj.eps)

    # ------------------------------------------------------- other density heads
    if isinstance(obj, MixtureDensityLoss):
        out = _need(inputs, "the network output")
        if obj.n_features == 1 and out.dim() == 3:
            return _sklar(_per_target_mixture(obj, out), kwargs.get("dependence"), kwargs.get("x"))
        return DistributionJoint(obj.distribution(out))
    if isinstance(obj, InputNoiseMDNLoss):
        draws = obj.sample_predictions(_need(inputs, "x_obs"), **kwargs)
        return MixtureJoint([DistributionJoint(obj.base_loss.distribution(p)) for p in draws])
    if isinstance(obj, EvidentialRegressionLoss):
        with torch.no_grad():
            gamma, nu, alpha, beta = obj._extract_nig_parameters(_need(inputs, "the head output"))
        scale = (beta * (1.0 + nu) / (nu * alpha)).sqrt()
        return DistributionJoint(Independent(StudentT(2.0 * alpha, gamma, scale), 1))
    if isinstance(obj, NormalizingFlowLoss):
        return FlowJoint(obj, _need(inputs, "the flow context"))
    if isinstance(obj, NoiseAwareDensityLoss):
        return as_joint(obj.base_loss, inputs, **kwargs)
    if isinstance(obj, JointBinnedDensityLoss):
        return ConditionalModelJoint(obj, _need(inputs, "the joint logits"), obj.n_targets)
    if isinstance(obj, MultiQuantileLoss):
        q = _need(inputs, "the quantile predictions [n, Q, d]")
        if isinstance(q, (list, tuple)):
            q = torch.stack(list(q), dim=1)
        levels = obj.quantiles.to(q)  # type: ignore[union-attr]
        marginals = QuantileMarginals(levels, q.detach().transpose(1, 2), kwargs.get("tail_knots"))
        return _sklar(marginals, kwargs.get("dependence"), kwargs.get("x"))
    if isinstance(obj, GaussianCopulaSplineHead):
        return CopulaSplineJoint(obj, _need(inputs, "the head context"))
    if isinstance(obj, FlowMatchingMLP):
        with torch.no_grad():
            draws = sample_cfm(
                obj,
                _need(inputs, "the context"),
                n_samples=int(kwargs.get("n_samples", 256)),
                steps=int(kwargs.get("steps", 32)),
            )
        return SampleJoint(draws)

    # --------------------------------------------------------------- ensembles
    if isinstance(obj, DensityEnsemble):
        x = _need(inputs, "x")
        members = [
            cast(JointPredictive, as_joint(obj.heads[m], c))
            for m, c in enumerate(obj.member_contexts(x))
        ]
        weights = cast(Tensor, obj.mixture_weights).detach().clone()
        return MixtureJoint(members, weights / weights.sum())
    if isinstance(obj, HeteroscedasticEnsembleModel):
        with eval_mode(obj), torch.no_grad():
            outputs = obj(_need(inputs, "x"))
        parts = [parse_heteroscedastic_output(p) for p in outputs]
        means = torch.stack([m for m, _ in parts])
        variances = torch.stack([member_variance_from_logvar(v) for _, v in parts])
        return _gaussian_mixture(means, variances)
    if isinstance(obj, HeteroscedasticBatchEnsembleModel):
        with eval_mode(obj), torch.no_grad():
            out = obj(_need(inputs, "x"))
        means = out["means"].movedim(1, 0)
        return _gaussian_mixture(means, member_variance_from_logvar(out["log_vars"].movedim(1, 0)))
    if isinstance(obj, MDNEnsembleModel):
        with eval_mode(obj), torch.no_grad():
            outputs = obj(_need(inputs, "x"))
        return MixtureJoint([DistributionJoint(obj.loss.distribution(p)) for p in outputs])
    if isinstance(obj, HeteroscedasticBNN):
        with eval_mode(obj):
            means, log_vars = obj.mc_forward(_need(inputs, "x"), kwargs.get("n_samples"))
        return _gaussian_mixture(means, log_vars.exp())
    if isinstance(obj, BaseEnsembleModel):
        raise TypeError(
            f"{type(obj).__name__} predicts points: the spread of its members is the "
            "uncertainty of the mean, not a predictive of y. Use members with a Gaussian "
            "or mixture output (HeteroscedasticEnsembleModel, MDNEnsembleModel)."
        )

    if isinstance(obj, (SWAG, MultiSWAG)):
        raise TypeError(
            f"{type(obj).__name__} samples network weights: the spread of its output is the "
            "uncertainty of the mean, not a predictive of y"
        )

    # ------------------------------------------------------ flow-head protocol
    # sample(context, n), with log_prob(context, y) for a density; draws only otherwise.
    if callable(getattr(obj, "sample", None)):
        if inputs is None:
            raise ValueError("as_joint needs the model context as its second argument")
        return ConditionalModelJoint(obj, inputs, kwargs.get("event_dim"))
    raise TypeError(f"as_joint does not support {type(obj).__name__}; supported: {_SUPPORTED}")
