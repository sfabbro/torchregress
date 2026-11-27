r"""Joint predictive distributions: one interface for every multi-target head.

A :class:`JointPredictive` is a batch of ``n`` predictive distributions
:math:`p(y \mid x_i)` over targets :math:`y \in \mathbb R^d`.  Whatever produced
it (a Gaussian head, a mixture, a conditional flow, a set of draws, marginals
joined by a dependence model), it exposes the same operations with the same
layouts, so regions, recalibration, scores and plots are written once:

* ``sample(S)`` returns ``[n, S, d]`` (batch first, always);
* ``log_prob(y)`` takes ``[n, d]`` and returns ``[n]``, or takes ``[n, S, d]``
  and returns ``[n, S]`` (the densities of ``S`` candidate points per row);
* ``marginal(dims)`` returns the joint predictive of a subset of targets, the
  operation that removes *nuisance* targets;
* ``to_latent(y)`` maps targets to a space where the predictive is standard
  normal, and ``from_latent(z)`` maps back, when the head has such a map.

Capabilities differ between heads and are declared, not guessed:
``has_density``, ``has_latent`` and ``exact_marginals`` (whether ``marginal`` and
``log_prob(mask=...)`` are exact or fall back to Monte Carlo).
"""

from __future__ import annotations

import inspect
import math
import weakref
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Optional, Union

import torch
from torch import Tensor

from ..losses.known_covariance import KnownCovarianceGaussianNLL, _total_covariance
from ..metrics.distribution import sample_hpd_level
from ._common import MOMENT_SEED, masked_gaussian_quadratic, safe_cholesky, seeded, target_index


def _same_token(kept: Optional[tuple[Any, ...]], token: tuple[Any, ...]) -> bool:
    """Whether two ``JointPredictive._moment_token`` values describe the same state."""
    if kept is None or len(kept) != len(token):
        return False
    for a, b in zip(kept, token, strict=True):
        if isinstance(a, weakref.ref) or isinstance(b, weakref.ref):
            if not (isinstance(a, weakref.ref) and isinstance(b, weakref.ref)):
                return False
            if a() is None or a() is not b():
                return False
        elif a != b:
            return False
    return True


__all__ = [
    "AffineJoint",
    "ConditionalModelJoint",
    "DistributionJoint",
    "FlowJoint",
    "GaussianJoint",
    "JointPredictive",
    "MaskedMarginalJoint",
    "MixtureJoint",
    "ProjectedJoint",
    "SampleJoint",
]

_LOG_2PI = math.log(2.0 * math.pi)
_HPD_CHUNK_ELEMS = 1 << 22  # elements of the [n, chunk, S] reference block in hpd_level


def _reference_tensor(obj: Any, depth: int = 0) -> Optional[Tensor]:
    """A floating tensor held by a distribution (for its dtype and device), if any."""
    if depth > 4 or obj is None:
        return None
    for name in ("loc", "logits", "probs", "scale_tril", "scale", "rate", "concentration"):
        try:
            value = getattr(obj, name, None)
        except Exception:  # lazy properties may fail on some distributions
            value = None
        if isinstance(value, Tensor) and value.is_floating_point():
            return value
    for value in vars(obj).values():
        if isinstance(value, Tensor) and value.is_floating_point():
            return value
    for value in vars(obj).values():
        if isinstance(value, torch.nn.Module):
            param = _first_floating_parameter(value)
            if param is not None:
                return param
        elif hasattr(value, "batch_shape") or hasattr(value, "transforms"):
            found = _reference_tensor(value, depth + 1)
            if found is not None:
                return found
        elif isinstance(value, (list, tuple)):
            for item in value:
                if hasattr(item, "__dict__"):
                    found = _reference_tensor(item, depth + 1)
                    if found is not None:
                        return found
    return None


def _first_floating_parameter(obj: Any) -> Optional[Tensor]:
    if isinstance(obj, torch.nn.Module):
        for t in list(obj.parameters()) + list(obj.buffers()):
            if t.is_floating_point():
                return t
    return None


class JointPredictive(ABC):
    """A batch of ``n`` joint predictive distributions over ``d`` targets."""

    #: ``log_prob`` is available.
    has_density: bool = False
    #: ``to_latent`` is available (a map to a standard-normal space).
    has_latent: bool = False
    #: ``marginal`` keeps a density and ``log_prob`` accepts a missing-target mask.
    exact_marginals: bool = False

    # ------------------------------------------------------------- required
    @property
    @abstractmethod
    def batch_size(self) -> int:
        """Number of rows ``n``."""

    @property
    @abstractmethod
    def event_dim(self) -> int:
        """Number of targets ``d``."""

    @property
    def dtype(self) -> torch.dtype:
        """Floating dtype of the predictive: inputs are cast to it, outputs come in it."""
        raise NotImplementedError(f"{type(self).__name__} does not declare its dtype")

    @property
    def device(self) -> torch.device:
        """Device of the predictive: inputs are moved to it, outputs stay on it."""
        raise NotImplementedError(f"{type(self).__name__} does not declare its device")

    @abstractmethod
    def sample(self, n_samples: int) -> Tensor:
        """Draws ``[n, S, d]`` (batch first).

        :func:`~torchregress.metrics.energy_score`,
        :func:`~torchregress.metrics.variogram_score` and
        :func:`~torchregress.metrics.crps_from_samples` take draws sample-first,
        ``[S, n, d]``: pass ``samples.movedim(1, 0)``.  With ``S == n`` a wrong
        layout is not detected by their shape checks.
        """

    # ------------------------------------------------------------- optional
    def log_prob(self, y: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Log density at ``y [n, d] -> [n]`` or ``y [n, S, d] -> [n, S]``.

        Args:
            y: Targets.
            mask: Boolean ``[n, d]`` (``True`` = observed).  Masked targets are
                integrated out, so the result is the log density of the
                observed sub-vector (zero for a row with nothing observed).
                Only heads with ``exact_marginals``.
        """
        raise NotImplementedError(f"{type(self).__name__} has no density")

    def to_latent(self, y: Tensor) -> Tensor:
        """Map ``y`` (``[n, d]`` or ``[n, S, d]``) to the standard-normal latent space."""
        raise NotImplementedError(f"{type(self).__name__} has no latent map")

    def from_latent(self, z: Tensor) -> Tensor:
        """Inverse of :meth:`to_latent`: latent points (``[n, d]`` or ``[n, S, d]``) to targets."""
        raise NotImplementedError(f"{type(self).__name__} has no latent map")

    def marginal(self, dims: Union[int, Sequence[int]]) -> "JointPredictive":
        """Joint predictive of the targets ``dims`` (the others are integrated out).

        Heads with ``exact_marginals`` return a predictive with a density (the
        other targets are integrated out through ``log_prob(mask=...)``); the
        rest integrate by sampling and return a draws-only predictive.
        """
        idx = target_index(dims, self.event_dim)
        if self.has_density and self.exact_marginals:
            return MaskedMarginalJoint(self, idx)
        return ProjectedJoint(self, idx)

    # -------------------------------------------------------------- derived
    def __getstate__(self) -> dict[str, Any]:
        # Kept moments are not part of the predictive: a copy or a pickle computes its own.
        state = self.__dict__.copy()
        state.pop("_moment_cache", None)
        return state

    def _moment_token(self) -> tuple[Any, ...]:
        """State the kept moments depend on; they are dropped when it changes.

        Items are compared with ``==``, except weak references, which match when they
        point to the same live object (an ``id`` would be reused once it is freed).
        """
        return ()

    def _monte_carlo_moments(
        self, n_samples: int, covariance: bool
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Mean ``[n, d]`` and, if asked, covariance ``[n, d, d]`` from ``n_samples`` draws.

        Kept on the predictive per ``n_samples`` and reused: a predictive is a fixed
        batch of distributions, so the same object returns the same moments, and
        scores that need them (Mahalanobis, residual and recalibration steps) do not
        draw again.  A request for the covariance draws once for both moments; a
        mean asked for first is kept as it is.  The draws come from the caller's
        generators, as any other draw (inside
        :func:`~torchregress.joint._common.fixed_moment_seed` from a fixed seed,
        kept separately).  Draws that carry gradients are never kept: their moments
        stay differentiable and are redrawn at every call; moments first computed
        under ``torch.no_grad()`` are checked once more when gradients are enabled.
        """
        seed = MOMENT_SEED.get()
        token = self._moment_token()
        cache: dict[Any, Any] = self.__dict__.setdefault("_moment_cache", {})
        if not _same_token(cache.get("token"), token):
            cache.clear()  # every kept entry belongs to the previous state
            cache["token"] = token
        key = (n_samples, seed)
        kept = cache.get(key)
        grad = torch.is_grad_enabled()
        if kept is not None and (kept["grad_checked"] or not grad):
            if kept["cov"] is not None or not covariance:
                return kept["mean"], kept["cov"]
        if seed is None:
            draws = self.sample(n_samples)
        else:
            with seeded(seed):
                draws = self.sample(n_samples)
        mean = draws.mean(dim=1)
        cov: Optional[Tensor] = None
        if covariance:
            centred = draws - mean.unsqueeze(1)
            cov = centred.mT @ centred / max(n_samples - 1, 1)
        if draws.requires_grad:
            return mean, cov
        if kept is not None:  # what was returned before stays what is returned
            mean = kept["mean"]
            cov = kept["cov"] if kept["cov"] is not None else cov
        checked = grad or (kept is not None and kept["grad_checked"])
        cache[key] = {"mean": mean, "cov": cov, "grad_checked": checked}
        return mean, cov

    def mean(self, n_samples: int = 512) -> Tensor:
        """Predictive mean ``[n, d]``.

        Closed form where the head has one; otherwise a Monte Carlo estimate from
        ``n_samples`` draws, computed once per predictive and ``n_samples`` and
        reused by later calls (the same object returns the same value).
        """
        return self._monte_carlo_moments(n_samples, covariance=False)[0].clone()

    def center(self, n_samples: int = 512) -> Tensor:
        r"""A location summary ``[n, d]`` that avoids sampling where it can.

        The image of the latent origin, :math:`T(0; x)`, for a predictive with a
        latent map (the mean of a Gaussian; for a flow, the point the mode of the
        base distribution is sent to); otherwise the predictive mean, which is
        closed form for mixtures and Monte Carlo (``n_samples`` draws) for the
        rest.  Wrappers combine the centers of what they wrap.  Use it where a
        deterministic reference point per row is needed and the mean would cost
        ``n_samples`` draws per row, as in
        :class:`~torchregress.joint.ScaleRecalibrator` with ``about="center"``.
        """
        if self.has_latent:
            origin = torch.zeros(
                self.batch_size, self.event_dim, dtype=self.dtype, device=self.device
            )
            return self.from_latent(origin)
        return self.mean(n_samples)

    def covariance(self, n_samples: int = 512) -> Tensor:
        """Predictive covariance ``[n, d, d]``.

        Closed form where the head has one; otherwise a Monte Carlo estimate from
        ``n_samples`` draws, computed once per predictive and ``n_samples``; the
        mean is taken from the same draws unless it was computed before.
        """
        cov = self._monte_carlo_moments(n_samples, covariance=True)[1]
        assert cov is not None
        return cov.clone()

    def hpd_level(self, y: Any, n_samples: int = 512) -> Tensor:
        r"""Highest-density level of ``y``: ``[n, d] -> [n]``, ``[n, S, d] -> [n, S]``.

        The predictive mass of the smallest highest-density region that
        contains ``y``,

        .. math::
            h(x, y) = P_{Y' \sim p(\cdot \mid x)}\big[p(Y' \mid x) > p(y \mid x)\big],

        estimated from ``n_samples`` draws (ties count one half) by
        :func:`~torchregress.metrics.sample_hpd_level`, which requires finite
        log-densities.  With ``[n, S, d]`` candidates, the ``S`` points of a row
        share one set of reference draws.  Uniform on ``[0, 1]`` for a
        calibrated predictive.
        """
        y = self._check(y)
        with torch.no_grad():
            target = self.log_prob(y)
            draws = self.log_prob(self.sample(n_samples))
        if y.dim() == 2:
            return sample_hpd_level(target, draws)
        n, s = target.shape
        m = draws.shape[1]
        chunk = max(1, _HPD_CHUNK_ELEMS // max(1, n * m))
        parts = []
        for start in range(0, s, chunk):
            block = target[:, start : start + chunk]  # [n, c]
            c = block.shape[1]
            ref = draws.unsqueeze(1).expand(n, c, m).reshape(n * c, m)
            parts.append(sample_hpd_level(block.reshape(n * c), ref).reshape(n, c))
        return torch.cat(parts, dim=1)

    # -------------------------------------------------------------- helpers
    def as_input(self, y: Any) -> Tensor:
        """``y`` (tensor or array, any floating dtype and device) as a tensor in this
        predictive's dtype and on its device."""
        t = torch.as_tensor(y)
        try:
            dtype, device = self.dtype, self.device
        except NotImplementedError:
            return t if t.is_floating_point() else t.to(torch.get_default_dtype())
        return t.to(device=device, dtype=dtype)

    def _check(self, y: Any) -> Tensor:
        y = self.as_input(y)
        if y.dim() not in (2, 3) or y.shape[0] != self.batch_size or y.shape[-1] != self.event_dim:
            raise ValueError(
                f"y must be [n, d] or [n, S, d] with n = {self.batch_size} and "
                f"d = {self.event_dim}, got {tuple(y.shape)}"
            )
        return y

    def _check_mask(self, y: Tensor, mask: Any) -> Tensor:
        mask = torch.as_tensor(mask)
        if mask.dtype != torch.bool or mask.shape != (self.batch_size, self.event_dim):
            raise ValueError(
                f"mask must be boolean [n, d] = [{self.batch_size}, {self.event_dim}], "
                f"got {mask.dtype} {tuple(mask.shape)}"
            )
        return mask.to(y.device)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n={self.batch_size}, d={self.event_dim})"


# ═══════════════════════════════════════════════════════════════════════════════
# Gaussian
# ═══════════════════════════════════════════════════════════════════════════════


class GaussianJoint(JointPredictive):
    r"""Multivariate normal predictive :math:`\mathcal N(\mu_i, L_i L_i^\top)`.

    The reference head: marginals, conditionals, missing targets and the latent
    map :math:`z = L^{-1}(y - \mu)` are all closed-form.

    Args:
        mean: ``[n, d]``.
        scale_tril: Lower-triangular Cholesky factor ``[n, d, d]`` (or ``[d, d]``,
            shared by all rows) with a positive diagonal.
    """

    has_density = True
    has_latent = True
    exact_marginals = True

    def __init__(self, mean: Tensor, scale_tril: Tensor) -> None:
        if mean.dim() != 2:
            raise ValueError(f"mean must be [n, d], got {tuple(mean.shape)}")
        n, d = mean.shape
        if scale_tril.dim() == 2:
            scale_tril = scale_tril.unsqueeze(0).expand(n, d, d)
        if scale_tril.shape != (n, d, d):
            raise ValueError(
                f"scale_tril must be [n, d, d] or [d, d], got {tuple(scale_tril.shape)}"
            )
        self.loc = mean
        self.scale_tril = scale_tril.to(mean.device, mean.dtype)

    @classmethod
    def from_covariance(cls, mean: Tensor, covariance: Tensor) -> "GaussianJoint":
        """Build from a covariance ``[n, d, d]`` (or ``[d, d]``)."""
        return cls(mean, safe_cholesky(covariance))

    @classmethod
    def from_low_rank(cls, mean: Tensor, cov_factor: Tensor, cov_diag: Tensor) -> "GaussianJoint":
        r"""Build from :math:`\Sigma = V V^\top + \mathrm{diag}(D)` (``V [n, d, r]``, ``D [n, d]``).

        The dense factor is formed once (``O(d^3)``); use it for ``d`` up to a
        few hundred.
        """
        cov = cov_factor @ cov_factor.mT + torch.diag_embed(cov_diag).to(
            device=cov_factor.device, dtype=cov_factor.dtype
        )
        return cls(mean, safe_cholesky(cov))

    @classmethod
    def from_known_covariance(
        cls,
        y_pred: Tensor,
        *,
        loss: Any = None,
        known_covariance: Optional[Tensor] = None,
        known_variance: Optional[Tensor] = None,
        model_scale: Optional[Tensor] = None,
        model_factor: Optional[Tensor] = None,
    ) -> "GaussianJoint":
        r"""Gaussian predictive of a :class:`~torchregress.losses.KnownCovarianceGaussianNLL`.

        :math:`\mathcal N(\hat y, C)`, where
        :math:`C = C_{\text{known}} + \mathrm{diag}(s^2) + UU^\top` is the total
        covariance whose likelihood the loss evaluates: the known measurement
        covariance (or variance) of each row, the given model terms, and the
        learned terms of ``loss`` (none when ``loss`` is omitted).  The masked
        density equals the loss's NLL with the same mask.

        Args:
            y_pred: Predictions ``[n, d]``.
            loss: The trained loss whose learned covariance terms are included.
            known_covariance, known_variance, model_scale, model_factor: As in
                :meth:`KnownCovarianceGaussianNLL.forward
                <torchregress.losses.KnownCovarianceGaussianNLL.forward>`.
        """
        head = KnownCovarianceGaussianNLL() if loss is None else loss
        if not isinstance(head, KnownCovarianceGaussianNLL):
            raise TypeError(f"loss must be a KnownCovarianceGaussianNLL, got {type(loss).__name__}")
        if y_pred.dim() != 2:
            raise ValueError(f"y_pred must be [n, d], got {tuple(y_pred.shape)}")
        with torch.no_grad():
            variance, factor = head._model_terms(y_pred, model_scale, model_factor)
            cov = _total_covariance(y_pred, known_covariance, known_variance, variance, factor)
        return cls.from_covariance(y_pred.detach(), cov)

    @property
    def batch_size(self) -> int:
        return int(self.loc.shape[0])

    @property
    def event_dim(self) -> int:
        return int(self.loc.shape[1])

    @property
    def dtype(self) -> torch.dtype:
        return self.loc.dtype

    @property
    def device(self) -> torch.device:
        return self.loc.device

    def sample(self, n_samples: int) -> Tensor:
        n, d = self.loc.shape
        eps = torch.randn(n, n_samples, d, device=self.loc.device, dtype=self.loc.dtype)
        return self.loc.unsqueeze(1) + eps @ self.scale_tril.mT

    def to_latent(self, y: Tensor) -> Tensor:
        y = self._check(y)
        resid = (y if y.dim() == 3 else y.unsqueeze(1)) - self.loc.unsqueeze(1)
        z = torch.linalg.solve_triangular(self.scale_tril, resid.mT, upper=False).mT
        return z if y.dim() == 3 else z[:, 0]

    def from_latent(self, z: Tensor) -> Tensor:
        z = self._check(z)
        z3 = z if z.dim() == 3 else z.unsqueeze(1)
        y = self.loc.unsqueeze(1) + z3 @ self.scale_tril.mT
        return y if z.dim() == 3 else y[:, 0]

    def log_prob(self, y: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        y = self._check(y)
        if mask is not None:
            return self._masked_log_prob(y, self._check_mask(y, mask))
        z = self.to_latent(y)
        half_log_det = self.scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        if y.dim() == 3:
            half_log_det = half_log_det.unsqueeze(1)
        return -0.5 * z.square().sum(-1) - half_log_det - 0.5 * self.event_dim * _LOG_2PI

    def _masked_log_prob(self, y: Tensor, mask: Tensor) -> Tensor:
        # Exact marginalisation of the masked targets (principal submatrix of the
        # covariance), shared with known_covariance_gaussian_nll.
        resid = y - (self.loc if y.dim() == 2 else self.loc.unsqueeze(1))
        quad, log_det, dof = masked_gaussian_quadratic(resid, self.covariance(), mask)
        return -0.5 * (quad + log_det + dof * _LOG_2PI)

    def marginal(self, dims: Union[int, Sequence[int]]) -> "GaussianJoint":
        idx = torch.as_tensor(target_index(dims, self.event_dim), device=self.loc.device)
        cov = self.covariance()[:, idx][:, :, idx]
        return GaussianJoint(self.loc[:, idx], safe_cholesky(cov))

    def affine(self, loc: Tensor, scale: Tensor) -> "GaussianJoint":
        r"""The Gaussian of :math:`\mathrm{loc} + \mathrm{scale} \odot Y`, per-target ``[d]``."""
        loc, scale = loc.to(self.loc), scale.to(self.loc)
        return GaussianJoint(loc + scale * self.loc, scale.unsqueeze(-1) * self.scale_tril)

    def condition(self, dims: Union[int, Sequence[int]], values: Tensor) -> "GaussianJoint":
        r"""Predictive of the remaining targets given ``y[dims] = values [n, k]``.

        :math:`\mu_{h \mid r} = \mu_h + \Sigma_{hr}\Sigma_{rr}^{-1}(y_r - \mu_r)`,
        :math:`\Sigma_{h \mid r} = \Sigma_{hh} - \Sigma_{hr}\Sigma_{rr}^{-1}\Sigma_{rh}`.
        """
        given = target_index(dims, self.event_dim)
        rest = [i for i in range(self.event_dim) if i not in given]
        if not rest:
            raise ValueError("conditioning on every target leaves nothing to predict")
        if values.shape != (self.batch_size, len(given)):
            raise ValueError(f"values must be [n, {len(given)}], got {tuple(values.shape)}")
        r = torch.as_tensor(given, device=self.loc.device)
        h = torch.as_tensor(rest, device=self.loc.device)
        cov = self.covariance()
        chol_rr = safe_cholesky(cov[:, r][:, :, r])
        cov_hr = cov[:, h][:, :, r]
        gain = torch.cholesky_solve(cov_hr.mT, chol_rr).mT  # Sigma_hr Sigma_rr^{-1}
        resid = (values.to(self.loc) - self.loc[:, r]).unsqueeze(-1)
        mean = self.loc[:, h] + (gain @ resid).squeeze(-1)
        cov_cond = cov[:, h][:, :, h] - gain @ cov_hr.mT
        return GaussianJoint(mean, safe_cholesky(cov_cond))

    def mean(self, n_samples: int = 512) -> Tensor:
        return self.loc

    def covariance(self, n_samples: int = 512) -> Tensor:
        return self.scale_tril @ self.scale_tril.mT

    def select(self, index: Tensor) -> "GaussianJoint":
        """Rows ``index`` (a long or boolean index)."""
        return GaussianJoint(self.loc[index], self.scale_tril[index])


# ═══════════════════════════════════════════════════════════════════════════════
# torch.distributions adapter (mixtures, flows, anything with batch [n], event [d])
# ═══════════════════════════════════════════════════════════════════════════════


class DistributionJoint(JointPredictive):
    """Adapter for a ``torch.distributions.Distribution`` with batch ``[n]`` and event ``[d]``.

    Covers :class:`~torchregress.losses.StudentTMixture` (the MDN predictive,
    with exact missing-target densities), ``MultivariateNormal`` and
    ``LowRankMultivariateNormal``, and zuko flow distributions (which also give
    the latent map).

    Args:
        distribution: The predictive distribution.
    """

    def __init__(self, distribution: Any) -> None:
        batch, event = tuple(distribution.batch_shape), tuple(distribution.event_shape)
        if len(batch) != 1 or len(event) != 1:
            raise ValueError(
                "the distribution must have batch shape [n] and event shape [d], "
                f"got {batch} and {event}"
            )
        self.distribution = distribution
        self._n, self._d = int(batch[0]), int(event[0])
        self.has_density = True
        # zuko's NormalizingFlow exposes the forward map and its base.
        self.has_latent = hasattr(distribution, "transform") and hasattr(distribution, "base")
        try:
            accepts_mask = "mask" in inspect.signature(distribution.log_prob).parameters
        except (TypeError, ValueError):
            accepts_mask = False
        self.exact_marginals = accepts_mask
        self._ref = _reference_tensor(distribution)

    @property
    def batch_size(self) -> int:
        return self._n

    @property
    def event_dim(self) -> int:
        return self._d

    @property
    def dtype(self) -> torch.dtype:
        if self._ref is None:
            raise NotImplementedError("cannot infer the dtype of the distribution")
        return self._ref.dtype

    @property
    def device(self) -> torch.device:
        if self._ref is None:
            raise NotImplementedError("cannot infer the device of the distribution")
        return self._ref.device

    def sample(self, n_samples: int) -> Tensor:
        with torch.no_grad():
            return self.distribution.sample((n_samples,)).movedim(0, 1)  # type: ignore[no-any-return]

    def log_prob(self, y: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        y = self._check(y)
        if mask is not None:
            if not self.exact_marginals:
                raise NotImplementedError(
                    f"{type(self.distribution).__name__} cannot integrate out missing targets"
                )
            mask = self._check_mask(y, mask)
            if y.dim() == 2:
                return self.distribution.log_prob(y, mask=mask)  # type: ignore[no-any-return]
            out = self.distribution.log_prob(y.movedim(1, 0), mask=mask.unsqueeze(0))
            return out.movedim(0, 1)  # type: ignore[no-any-return]
        if y.dim() == 2:
            return self.distribution.log_prob(y)  # type: ignore[no-any-return]
        return self.distribution.log_prob(y.movedim(1, 0)).movedim(0, 1)  # type: ignore[no-any-return]

    def to_latent(self, y: Tensor) -> Tensor:
        if not self.has_latent:
            raise NotImplementedError(f"{type(self.distribution).__name__} has no latent map")
        y = self._check(y)
        if y.dim() == 2:
            return self.distribution.transform(y)  # type: ignore[no-any-return]
        return self.distribution.transform(y.movedim(1, 0)).movedim(0, 1)  # type: ignore[no-any-return]

    def from_latent(self, z: Tensor) -> Tensor:
        if not self.has_latent:
            raise NotImplementedError(f"{type(self.distribution).__name__} has no latent map")
        z = self._check(z)
        inverse = self.distribution.transform.inv
        if z.dim() == 2:
            return inverse(z)  # type: ignore[no-any-return]
        return inverse(z.movedim(1, 0)).movedim(0, 1)  # type: ignore[no-any-return]

    def mean(self, n_samples: int = 512) -> Tensor:
        try:
            value = self.distribution.mean
        except (NotImplementedError, AttributeError):
            return super().mean(n_samples)
        return value if isinstance(value, Tensor) else super().mean(n_samples)

    def covariance(self, n_samples: int = 512) -> Tensor:
        try:
            value = self.distribution.covariance_matrix
        except (NotImplementedError, AttributeError):
            return super().covariance(n_samples)
        return value if isinstance(value, Tensor) else super().covariance(n_samples)


class FlowJoint(DistributionJoint):
    """Predictive of a conditional flow head (:class:`~torchregress.losses.NormalizingFlowLoss`).

    Sampling goes through the loss, which draws in blocks and releases the
    transform graph, instead of through the raw flow distribution.

    Args:
        flow_loss: The flow head.
        context: Network output for the rows, ``[n, context_dim]``.
    """

    def __init__(self, flow_loss: Any, context: Tensor) -> None:
        distribution = flow_loss.distribution(context)
        if len(tuple(distribution.batch_shape)) == 0:
            # An unconditional flow: the same distribution for every row.
            distribution = distribution.expand((context.shape[0],))
        super().__init__(distribution)
        self.flow_loss = flow_loss
        self.context = context

    @property
    def dtype(self) -> torch.dtype:
        param = _first_floating_parameter(self.flow_loss)
        return param.dtype if param is not None else self.context.dtype

    @property
    def device(self) -> torch.device:
        return self.context.device

    def sample(self, n_samples: int) -> Tensor:
        with torch.no_grad():
            draws = self.flow_loss.sample(self.context, n_samples=n_samples)
        return draws.unsqueeze(1) if draws.dim() == 2 else draws  # type: ignore[no-any-return]


class ConditionalModelJoint(JointPredictive):
    """Predictive of a conditional model with the flow-head API (density or draws only).

    Wraps any object with ``sample(context, n_samples) -> [n, S, d]``
    (``[n, d]`` for one draw) and, for a density, ``log_prob(context, target)
    -> [n]``: the convention of :class:`~torchregress.losses.NormalizingFlowLoss`
    (and :class:`~torchregress.losses.ContrastiveFlowLoss`),
    :class:`~torchregress.ensemble.DensityEnsemble`,
    :class:`~torchregress.models.MarginalContrastiveDensity` and
    :class:`~torchregress.models.NormalizingFlowMap`, the same one
    :func:`~torchregress.metrics.model_hpd_level` uses.  Densities of
    ``[n, S, d]`` candidates are computed by repeating each context ``S``
    times.  There is no latent map and no missing-target mask; marginals are
    by sampling.  For a :class:`~torchregress.losses.NormalizingFlowLoss`,
    :class:`FlowJoint` also gives the latent map and exact masks when the flow
    supports them.  ``MarginalContrastiveDensity.log_prob`` normalises by Monte
    Carlo, so its densities (and density-based regions) carry that error.

    A model without ``log_prob`` (an implicit generator, such as the energy-score
    head of :class:`~torchregress.joint.JointRegressor` or a user-trained
    generative network) gives a draws-only predictive: ``has_density`` is then
    ``False`` for this instance, and ``sample`` runs the model on every call
    (draws are not stored), so mean, covariance and marginals are Monte Carlo.

    Args:
        model: The conditional model.
        context: Model input for the rows, ``[n, ...]``.
        event_dim: Number of targets ``d``; inferred from one draw when omitted.
    """

    def __init__(self, model: Any, context: Tensor, event_dim: Optional[int] = None) -> None:
        if not callable(getattr(model, "sample", None)):
            raise TypeError(
                "model must define sample(context, n) (and log_prob(context, target) for a density)"
            )
        self.has_density = callable(getattr(model, "log_prob", None))
        if context.dim() < 1 or context.shape[0] < 1:
            raise ValueError(f"context must be [n, ...] with n >= 1, got {tuple(context.shape)}")
        self.model = model
        self.context = context
        if event_dim is None:
            with torch.no_grad():
                probe = model.sample(context[:1], n_samples=1)
            event_dim = int(probe.shape[-1])
        self._d = int(event_dim)

    @property
    def batch_size(self) -> int:
        return int(self.context.shape[0])

    @property
    def event_dim(self) -> int:
        return self._d

    @property
    def dtype(self) -> torch.dtype:
        param = _first_floating_parameter(self.model)
        if param is not None:
            return param.dtype
        ctx = self.context
        return ctx.dtype if ctx.is_floating_point() else torch.get_default_dtype()

    @property
    def device(self) -> torch.device:
        return self.context.device

    def sample(self, n_samples: int) -> Tensor:
        with torch.no_grad():
            draws = self.model.sample(self.context, n_samples=n_samples)
        if draws.dim() == 2:
            draws = draws.unsqueeze(1)
        expected = (self.batch_size, n_samples, self.event_dim)
        if tuple(draws.shape) != expected:
            raise ValueError(f"model.sample returned {tuple(draws.shape)}, expected {expected}")
        return draws  # type: ignore[no-any-return]

    def log_prob(self, y: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if not self.has_density:
            raise NotImplementedError(f"{type(self.model).__name__} has no density (draws only)")
        if mask is not None:
            raise NotImplementedError(
                f"{type(self.model).__name__} cannot integrate out missing targets"
            )
        y = self._check(y)
        n = self.batch_size
        if y.dim() == 2:
            return self.model.log_prob(self.context, y).reshape(n)  # type: ignore[no-any-return]
        s = int(y.shape[1])
        context = self.context.repeat_interleave(s, dim=0)
        out = self.model.log_prob(context, y.reshape(n * s, self.event_dim))
        return out.reshape(n, s)  # type: ignore[no-any-return]


# ═══════════════════════════════════════════════════════════════════════════════
# draws only
# ═══════════════════════════════════════════════════════════════════════════════


class SampleJoint(JointPredictive):
    """A predictive known only through draws ``[n, S0, d]`` (implicit generators, forests).

    ``sample(S)`` returns all stored draws when ``S == S0``, a random subset
    without replacement (independently per row) when ``S < S0``, and resamples
    with replacement when ``S > S0``.
    """

    def __init__(self, samples: Tensor) -> None:
        if samples.dim() != 3 or samples.shape[1] < 1:
            raise ValueError(f"samples must be [n, S, d] with S >= 1, got {tuple(samples.shape)}")
        self.samples = samples

    def _moment_token(self) -> tuple[Any, ...]:
        # Replacing the stored draws, or editing them in place, makes kept moments stale
        # (an edit through ``samples.data`` is not tracked by torch and is not seen).
        version = None if torch.is_inference(self.samples) else self.samples._version
        return (weakref.ref(self.samples), version)

    @property
    def batch_size(self) -> int:
        return int(self.samples.shape[0])

    @property
    def event_dim(self) -> int:
        return int(self.samples.shape[2])

    @property
    def dtype(self) -> torch.dtype:
        return self.samples.dtype

    @property
    def device(self) -> torch.device:
        return self.samples.device

    def sample(self, n_samples: int) -> Tensor:
        n, stored, _ = self.samples.shape
        if n_samples == stored:
            return self.samples
        if n_samples < stored:
            keys = torch.rand(n, stored, device=self.samples.device)
            index = keys.argsort(dim=1)[:, :n_samples]
        else:
            index = torch.randint(stored, (n, n_samples), device=self.samples.device)
        return torch.gather(
            self.samples, 1, index.unsqueeze(-1).expand(n, n_samples, self.event_dim)
        )

    def marginal(self, dims: Union[int, Sequence[int]]) -> "SampleJoint":
        return SampleJoint(self.samples[:, :, target_index(dims, self.event_dim)])

    def select(self, index: Tensor) -> "SampleJoint":
        """Rows ``index`` (a long or boolean index)."""
        return SampleJoint(self.samples[index])


class ProjectedJoint(JointPredictive):
    """Draws-only marginal of another predictive (its targets ``dims``)."""

    def __init__(self, base: JointPredictive, dims: Sequence[int]) -> None:
        self.base = base
        self.dims = target_index(dims, base.event_dim)

    @property
    def batch_size(self) -> int:
        return self.base.batch_size

    @property
    def event_dim(self) -> int:
        return len(self.dims)

    @property
    def dtype(self) -> torch.dtype:
        return self.base.dtype

    @property
    def device(self) -> torch.device:
        return self.base.device

    def sample(self, n_samples: int) -> Tensor:
        return self.base.sample(n_samples)[:, :, self.dims]

    def mean(self, n_samples: int = 512) -> Tensor:
        return self.base.mean(n_samples)[:, self.dims]

    def center(self, n_samples: int = 512) -> Tensor:
        return self.base.center(n_samples)[:, self.dims]

    def covariance(self, n_samples: int = 512) -> Tensor:
        # The moments of a projection are the projected moments of the base, which the
        # base computes once (in closed form where it can).
        return self.base.covariance(n_samples)[:, self.dims][:, :, self.dims]


class MaskedMarginalJoint(JointPredictive):
    """Exact marginal of a predictive that integrates out masked targets.

    The density of the kept targets is the base density with every other target
    masked; draws are the base draws restricted to the kept targets.
    """

    has_density = True
    exact_marginals = True

    def __init__(self, base: JointPredictive, dims: Sequence[int]) -> None:
        if not (base.has_density and base.exact_marginals):
            raise ValueError("the base predictive must integrate out masked targets exactly")
        self.base = base
        self.dims = target_index(dims, base.event_dim)

    @property
    def batch_size(self) -> int:
        return self.base.batch_size

    @property
    def event_dim(self) -> int:
        return len(self.dims)

    @property
    def dtype(self) -> torch.dtype:
        return self.base.dtype

    @property
    def device(self) -> torch.device:
        return self.base.device

    def sample(self, n_samples: int) -> Tensor:
        return self.base.sample(n_samples)[:, :, self.dims]

    def _embed(self, y: Tensor, mask: Optional[Tensor]) -> tuple[Tensor, Tensor]:
        """Place ``y`` in the base's target space and build the matching mask."""
        full = y.new_zeros(*y.shape[:-1], self.base.event_dim)
        full[..., self.dims] = y
        base_mask = torch.zeros(
            self.batch_size, self.base.event_dim, dtype=torch.bool, device=y.device
        )
        base_mask[:, self.dims] = True if mask is None else mask.to(y.device)
        return full, base_mask

    def log_prob(self, y: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        y = self._check(y)
        if mask is not None:
            mask = self._check_mask(y, mask)
            y = torch.where(mask if y.dim() == 2 else mask.unsqueeze(1), y, torch.zeros_like(y))
        return self.base.log_prob(*self._embed(y, mask))

    def marginal(self, dims: Union[int, Sequence[int]]) -> "MaskedMarginalJoint":
        idx = target_index(dims, self.event_dim)
        return MaskedMarginalJoint(self.base, [self.dims[i] for i in idx])

    def mean(self, n_samples: int = 512) -> Tensor:
        return self.base.mean(n_samples)[:, self.dims]

    def center(self, n_samples: int = 512) -> Tensor:
        return self.base.center(n_samples)[:, self.dims]

    def covariance(self, n_samples: int = 512) -> Tensor:
        return self.base.covariance(n_samples)[:, self.dims][:, :, self.dims]


# ═══════════════════════════════════════════════════════════════════════════════
# wrappers
# ═══════════════════════════════════════════════════════════════════════════════


class AffineJoint(JointPredictive):
    r"""Per-target affine change of units, :math:`y = \mathrm{loc} + \mathrm{scale} \odot u`.

    Estimators fit heads on standardised targets and wrap the head's predictive
    in this class, so that densities, draws, marginals and latent maps come back
    in the units of the data.  The density picks up the Jacobian
    :math:`-\sum_j \log \mathrm{scale}_j` (over observed targets under a mask).

    Args:
        base: Predictive of the standardised targets ``u``.
        loc: ``[d]`` (or ``[n, d]``).
        scale: Positive ``[d]`` (or ``[n, d]``).
    """

    def __init__(self, base: JointPredictive, loc: Tensor, scale: Tensor) -> None:
        d = base.event_dim
        if loc.shape[-1] != d or scale.shape[-1] != d:
            raise ValueError(f"loc and scale must end in d = {d}")
        if not bool((scale > 0).all()):
            raise ValueError("scale must be positive")
        self.base = base
        self.loc = loc.reshape(-1, d)
        self.scale = scale.reshape(-1, d)
        self.has_density = base.has_density
        self.has_latent = base.has_latent
        self.exact_marginals = base.exact_marginals

    @property
    def batch_size(self) -> int:
        return self.base.batch_size

    @property
    def event_dim(self) -> int:
        return self.base.event_dim

    @property
    def dtype(self) -> torch.dtype:
        return self.base.dtype

    @property
    def device(self) -> torch.device:
        return self.base.device

    def _standardise(self, y: Tensor) -> Tensor:
        loc, scale = self.loc.to(y), self.scale.to(y)
        if y.dim() == 3:
            loc, scale = loc.unsqueeze(1), scale.unsqueeze(1)
        return (y - loc) / scale

    def sample(self, n_samples: int) -> Tensor:
        u = self.base.sample(n_samples)
        return self.loc.to(u).unsqueeze(1) + self.scale.to(u).unsqueeze(1) * u

    def log_prob(self, y: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        y = self._check(y)
        if mask is not None:
            mask = self._check_mask(y, mask)
        inner = self.base.log_prob(self._standardise(y), mask)
        log_scale = self.scale.to(inner).log().expand(self.batch_size, self.event_dim)
        if mask is not None:
            log_scale = torch.where(
                mask.to(log_scale.device), log_scale, torch.zeros_like(log_scale)
            )
        jac = log_scale.sum(-1)
        return inner - (jac.unsqueeze(1) if y.dim() == 3 else jac)

    def to_latent(self, y: Tensor) -> Tensor:
        return self.base.to_latent(self._standardise(self._check(y)))

    def from_latent(self, z: Tensor) -> Tensor:
        u = self.base.from_latent(self._check(z))
        loc, scale = self.loc.to(u), self.scale.to(u)
        if u.dim() == 3:
            loc, scale = loc.unsqueeze(1), scale.unsqueeze(1)
        return loc + scale * u

    def marginal(self, dims: Union[int, Sequence[int]]) -> "AffineJoint":
        idx = target_index(dims, self.event_dim)
        return AffineJoint(self.base.marginal(idx), self.loc[:, idx], self.scale[:, idx])

    def mean(self, n_samples: int = 512) -> Tensor:
        inner = self.base.mean(n_samples)
        return self.loc.to(inner) + self.scale.to(inner) * inner

    def center(self, n_samples: int = 512) -> Tensor:
        inner = self.base.center(n_samples)
        return self.loc.to(inner) + self.scale.to(inner) * inner

    def covariance(self, n_samples: int = 512) -> Tensor:
        inner = self.base.covariance(n_samples)
        scale = self.scale.to(inner).expand(self.batch_size, self.event_dim)
        return inner * scale.unsqueeze(-1) * scale.unsqueeze(-2)


class MixtureJoint(JointPredictive):
    r"""Weighted mixture of predictives, :math:`p = \sum_k w_k\, p_k` (stacked ensembles).

    Args:
        members: Predictives with the same ``n`` and ``d``.
        weights: Non-negative ``[K]`` (or per-row ``[n, K]``) summing to one;
            uniform when omitted.
    """

    def __init__(
        self, members: Sequence[JointPredictive], weights: Optional[Tensor] = None
    ) -> None:
        if not members:
            raise ValueError("at least one member is required")
        n, d = members[0].batch_size, members[0].event_dim
        if any(m.batch_size != n or m.event_dim != d for m in members):
            raise ValueError("members must agree on n and d")
        k = len(members)
        if weights is None:
            weights = torch.full((k,), 1.0 / k)
        if weights.shape[-1] != k or weights.dim() not in (1, 2):
            raise ValueError(f"weights must be [K] or [n, K] with K = {k}")
        if bool((weights < 0).any()) or not torch.allclose(
            weights.sum(-1), torch.ones((), dtype=weights.dtype, device=weights.device), atol=1e-6
        ):
            raise ValueError("weights must be non-negative and sum to one")
        self.members = list(members)
        self.weights = weights
        self.has_density = all(m.has_density for m in members)
        self.exact_marginals = all(m.exact_marginals for m in members)

    @property
    def batch_size(self) -> int:
        return self.members[0].batch_size

    @property
    def event_dim(self) -> int:
        return self.members[0].event_dim

    @property
    def dtype(self) -> torch.dtype:
        return self.members[0].dtype

    @property
    def device(self) -> torch.device:
        return self.members[0].device

    def _row_weights(self, like: Tensor) -> Tensor:
        w = self.weights.to(like.device, like.dtype)
        return w.expand(self.batch_size, len(self.members))

    def sample(self, n_samples: int) -> Tensor:
        # Each row picks its own member for each draw (so rows stay independent, which
        # Monte Carlo conformity scores rely on).  Member k is then asked only for as
        # many draws as the row that needs it most, not for S draws.
        n, d = self.batch_size, self.event_dim
        pick = torch.multinomial(
            self.weights.double().expand(n, len(self.members)), n_samples, replacement=True
        )  # [n, S]
        onehot = torch.nn.functional.one_hot(pick, len(self.members))  # [n, S, K]
        counts = onehot.sum(1)  # [n, K]
        # Position of each draw among the draws of the same row and member.
        occurrence = (onehot.cumsum(1) - 1).gather(2, pick.unsqueeze(-1)).squeeze(-1)  # [n, S]
        out: Optional[Tensor] = None
        for k, member in enumerate(self.members):
            need = int(counts[:, k].max())
            if need == 0:
                continue
            draws = member.sample(need)  # [n, need, d]
            index = occurrence.clamp_max(need - 1).to(draws.device)
            chosen = draws.gather(1, index.unsqueeze(-1).expand(n, n_samples, d))
            if out is None:
                out = torch.zeros_like(chosen)
            out = torch.where((pick == k).to(draws.device).unsqueeze(-1), chosen, out)
        assert out is not None
        return out

    def log_prob(self, y: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if not self.has_density:
            raise NotImplementedError("a member has no density")
        y = self._check(y)
        if mask is not None:
            mask = self._check_mask(y, mask)
        parts = torch.stack([m.log_prob(y, mask) for m in self.members], dim=-1)
        log_w = self._row_weights(parts).log()
        if y.dim() == 3:
            log_w = log_w.unsqueeze(1)
        return torch.logsumexp(parts + log_w, dim=-1)

    def mean(self, n_samples: int = 512) -> Tensor:
        means = torch.stack([m.mean(n_samples) for m in self.members], dim=1)  # [n, K, d]
        return (self._row_weights(means).unsqueeze(-1) * means).sum(1)

    def center(self, n_samples: int = 512) -> Tensor:
        centers = torch.stack([m.center(n_samples) for m in self.members], dim=1)  # [n, K, d]
        return (self._row_weights(centers).unsqueeze(-1) * centers).sum(1)

    def covariance(self, n_samples: int = 512) -> Tensor:
        # Law of total covariance: mean of the members' covariances plus the
        # covariance of their means.
        # Covariances first: a member without closed forms then draws once for both.
        covs = torch.stack([m.covariance(n_samples) for m in self.members], dim=1)
        means = torch.stack([m.mean(n_samples) for m in self.members], dim=1)  # [n, K, d]
        w = self._row_weights(means)
        centre = (w.unsqueeze(-1) * means).sum(1, keepdim=True)
        spread = (means - centre).unsqueeze(-1) * (means - centre).unsqueeze(-2)
        return (w[..., None, None] * (covs + spread)).sum(1)

    def marginal(self, dims: Union[int, Sequence[int]]) -> "MixtureJoint":
        idx = target_index(dims, self.event_dim)
        return MixtureJoint([m.marginal(idx) for m in self.members], self.weights)
