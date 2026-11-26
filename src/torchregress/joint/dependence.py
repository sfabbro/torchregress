r"""Dependence on normal scores: marginals plus a copula-like model make a joint.

By Sklar's theorem, any joint distribution with continuous marginal CDFs
:math:`F_j(\cdot \mid x)` factors into its marginals and a copula.  With normal
scores :math:`z_j = \Phi^{-1}(F_j(y_j \mid x))` and any density
:math:`q(z \mid x)` on the score space, the change of variables
:math:`y \mapsto z` gives the joint density

.. math::
    \log p(y \mid x) = \sum_{j=1}^d \log f_j(y_j \mid x) + \log q(z \mid x)
        - \sum_{j=1}^d \log \varphi(z_j),

which is a valid density for every :math:`q` (:class:`SklarJoint`).  The
marginals come from any per-target engine (:mod:`torchregress.joint.marginals`);
the dependence comes from the lowest rung of a ladder that held-out scores
support (:func:`select_dependence`):

=====  ===============================  ===============================  =======
Rung   Model of :math:`q(z \mid x)`     Fit                              Density
=====  ===============================  ===============================  =======
0      :class:`IndependenceDependence`  none                             yes
1      :class:`SchaakeShuffle`          template of training scores      no
2      :class:`GaussianCopula`          shrunk correlation, closed form  yes
3      :class:`ConditionalGaussianCopula` NLL, modified Cholesky in x    yes
5      :class:`FlowDependence`          NLL, conditional spline flow     yes
=====  ===============================  ===============================  =======

(Rung 4 of the design, vine copulas, is not implemented.)

:math:`q` is a *copula* density only if its marginals are standard normal.  A
fitted :math:`q` whose marginals drift also reshapes the per-target
predictives; that acts as a recalibration of the marginals and is allowed, but
then the joint is a marginally preconditioned model rather than "marginals plus
copula".  :meth:`Dependence.marginal_drift` reports it.

References:
    Sklar, A. (1959). Fonctions de répartition à n dimensions et leurs marges.

    Clark, M. et al. (2004). The Schaake shuffle. J. Hydrometeorology 5.

    Pinson, P. et al. (2009). From probabilistic forecasts to statistical
    scenarios of short-term wind power production. Wind Energy 12.

    Ledoit, O., Wolf, M. (2004). A well-conditioned estimator for
    large-dimensional covariance matrices. J. Multivariate Anal. 88.

    Klein, N., Hothorn, T., Barbanti, L., Kneib, T. (2022). Multivariate
    conditional transformation models. Scand. J. Statist. 49.

    Wen, R., Torkkola, K. (2019). Deep generative quantile-copula models for
    probabilistic forecasting. ICML time-series workshop.

    Ashok, A. et al. (2024). TACTiS-2: better, faster, simpler attentional
    copulas for multivariate time series. ICLR.

    Arpogaus, M., Kneib, T., Nagler, T., Rügamer, D. (2025). Hybrid Bernstein
    normalizing flows for flexible multivariate density regression.
    arXiv:2505.14164.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Optional, Union

import torch
from torch import Tensor, nn

from ..losses.nflows import NormalizingFlowLoss, create_flow_model
from ._common import mlp, safe_cholesky, target_index, train
from .marginals import Marginals
from .predictive import DistributionJoint, FlowJoint, GaussianJoint, JointPredictive

__all__ = [
    "ConditionalGaussianCopula",
    "Dependence",
    "FlowDependence",
    "GaussianCopula",
    "IndependenceDependence",
    "SchaakeShuffle",
    "SklarJoint",
    "TemplateJoint",
    "select_dependence",
]

_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


# ═══════════════════════════════════════════════════════════════════════════════
# the joint
# ═══════════════════════════════════════════════════════════════════════════════


class SklarJoint(JointPredictive):
    r"""Joint predictive from per-target marginals and a predictive of their normal scores.

    With :math:`z_j = \Phi^{-1}(\mathrm{clip}(F_j(y_j \mid x), \varepsilon, 1 - \varepsilon))`,

    .. math::
        \log p(y \mid x) = \sum_j \log f_j(y_j \mid x) + \log q(z \mid x)
            - \sum_j \log \varphi(z_j).

    * ``sample``: :math:`z \sim q`, :math:`u = \Phi(z)`, :math:`y = F^{-1}(u)`.
    * ``log_prob(y, mask)``: when ``scores.exact_marginals``, missing targets
      are integrated out exactly (the score marginal of the observed
      coordinates replaces :math:`q`, and only observed targets contribute
      :math:`\log f_j - \log\varphi`).
    * ``to_latent`` / ``from_latent``: through the scores' latent map.
    * ``marginal(dims)``: the selected marginals with ``scores.marginal(dims)``.
    * ``has_density``, ``has_latent``, ``exact_marginals``: those of ``scores``.

    Exactness: the density integrates to one when the clipping is inactive
    (``eps`` only alters it where some :math:`F_j` lies within ``eps`` of 0
    or 1).  When :math:`q` has standard-normal marginals the per-target
    marginals of the joint are exactly :math:`F_j`.  When it does not (a
    fitted :math:`q` can drift, see :meth:`Dependence.marginal_drift`), the
    joint is still a valid density but its marginals are :math:`F_j`
    recalibrated by :math:`q`'s marginals, not :math:`F_j`.

    Args:
        marginals: ``n`` rows times ``d`` targets of univariate predictives.
        scores: Joint predictive of the normal scores (``n`` rows, ``d`` dims).
        eps: Clipping of the PIT values before :math:`\Phi^{-1}`.
    """

    def __init__(self, marginals: Marginals, scores: JointPredictive, eps: float = 1e-6) -> None:
        if marginals.batch_size != scores.batch_size or marginals.event_dim != scores.event_dim:
            raise ValueError(
                f"marginals are n={marginals.batch_size}, d={marginals.event_dim} but scores are "
                f"n={scores.batch_size}, d={scores.event_dim}"
            )
        if not 0.0 < eps < 0.5:
            raise ValueError("eps must lie in (0, 0.5)")
        self.marginals = marginals
        self.scores = scores
        self.eps = float(eps)
        self.has_density = scores.has_density
        self.has_latent = scores.has_latent
        self.exact_marginals = scores.exact_marginals

    @property
    def batch_size(self) -> int:
        return self.scores.batch_size

    @property
    def event_dim(self) -> int:
        return self.scores.event_dim

    @property
    def dtype(self) -> torch.dtype:
        return self.marginals._dtype()

    @property
    def device(self) -> torch.device:
        return self.marginals._device()

    def sample(self, n_samples: int) -> Tensor:
        return self.marginals.from_normal_scores(self.scores.sample(n_samples), self.eps)

    def normal_scores(self, y: Tensor) -> Tensor:
        """Normal scores of ``y`` under the marginals (same shape)."""
        return self.marginals.normal_scores(self._check(y), self.eps)

    def log_prob(self, y: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if not self.has_density:
            raise NotImplementedError(f"{type(self.scores).__name__} has no density")
        y = self._check(y)
        observed: Optional[Tensor] = None
        if mask is not None:
            if not self.exact_marginals:
                raise NotImplementedError(
                    f"{type(self.scores).__name__} cannot integrate out missing targets"
                )
            mask = self._check_mask(y, mask)
            observed = mask if y.dim() == 2 else mask.unsqueeze(1)
            fill = self.marginals.median().to(y)
            fill = fill if y.dim() == 2 else fill.unsqueeze(1)
            y = torch.where(observed, y, fill)
        z = self.marginals.normal_scores(y, self.eps)
        log_phi = -0.5 * z.square() - _HALF_LOG_2PI
        jac = self.marginals.log_prob(y) - log_phi
        if observed is not None:
            jac = torch.where(observed, jac, torch.zeros_like(jac))
        return jac.sum(-1) + self.scores.log_prob(z, mask)

    def to_latent(self, y: Tensor) -> Tensor:
        return self.scores.to_latent(self.normal_scores(y))

    def from_latent(self, z: Tensor) -> Tensor:
        return self.marginals.from_normal_scores(self.scores.from_latent(self._check(z)), self.eps)

    def marginal(self, dims: Union[int, Sequence[int]]) -> "SklarJoint":
        idx = target_index(dims, self.event_dim)
        return SklarJoint(self.marginals.select_targets(idx), self.scores.marginal(idx), self.eps)


# ═══════════════════════════════════════════════════════════════════════════════
# the ladder
# ═══════════════════════════════════════════════════════════════════════════════


class Dependence(ABC):
    r"""A model of the score density :math:`q(z \mid x)`, fitted on training normal scores.

    ``fit(z, x)`` learns from scores ``z [N, d]`` (and optionally covariates
    ``x [N, p]``); ``predictive(n, x)`` returns a :class:`JointPredictive`
    over the scores of ``n`` new rows, to be passed to :class:`SklarJoint`.
    """

    #: Short name used in the selection table.
    name: str = "dependence"
    #: ``predictive(...).log_prob`` is available.
    has_density: bool = True

    def __init__(self) -> None:
        self._event_dim: Optional[int] = None
        self._dtype: torch.dtype = torch.get_default_dtype()
        self._device: torch.device = torch.device("cpu")
        #: Whether the model was fitted with covariates (and needs them to predict).
        self.conditional: bool = False
        #: Wall-clock seconds of the last :meth:`fit`.
        self.fit_time_: float = 0.0

    @abstractmethod
    def fit(self, z: Tensor, x: Optional[Tensor] = None) -> "Dependence":
        """Fit on training normal scores ``z [N, d]`` (and covariates ``x [N, p]``).

        The scores must be **out of fold**: computed from marginals that were
        not fitted on these rows (cross-fitting, or a held-out split).
        In-sample PITs are too uniform and too central, so the dependence
        stage would learn the marginal model's overfit rather than the
        dependence of the targets.  This is the caller's responsibility.
        """

    @abstractmethod
    def predictive(self, n: int, x: Optional[Tensor] = None) -> JointPredictive:
        """Predictive over the scores of ``n`` rows (covariates ``x [n, p]`` when conditional)."""

    # -------------------------------------------------------------- derived
    @property
    def event_dim(self) -> int:
        """Number of targets ``d`` seen by :meth:`fit`."""
        if self._event_dim is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted")
        return self._event_dim

    def log_prob(self, z: Tensor, x: Optional[Tensor] = None) -> Tensor:
        """Log score density ``[N]`` of ``z [N, d]`` (no gradient)."""
        if not self.has_density:
            raise NotImplementedError(f"{type(self).__name__} has no density")
        with torch.no_grad():
            return self.predictive(int(z.shape[0]), x).log_prob(z)

    def marginal_drift(
        self, z: Tensor, x: Optional[Tensor] = None, n_samples: int = 256
    ) -> dict[str, Tensor]:
        r"""How far the fitted :math:`q` is from having standard-normal marginals.

        For the rows of ``z`` (and ``x``), the implied marginal of each score
        :math:`z_j` under :math:`q(\cdot \mid x_i)` has mean :math:`m_{ij}`
        and variance :math:`s^2_{ij}`; returned per target ``[d]``:

        * ``"mean"``: :math:`\frac1n \sum_i m_{ij}` (0 for a copula),
        * ``"std"``: :math:`(\frac1n \sum_i s^2_{ij})^{1/2}` (1 for a copula),
        * ``"score_mean"``, ``"score_std"``: the same moments of the held-out
          scores ``z`` themselves (0 and 1 when the marginals are calibrated).

        Exact for Gaussian rungs, Monte Carlo with ``n_samples`` draws
        otherwise.  Large drift means :math:`q` is recalibrating the
        marginals, see :class:`SklarJoint`.
        """
        with torch.no_grad():
            pred = self.predictive(int(z.shape[0]), x)
            if isinstance(pred, (GaussianJoint, TemplateJoint)):
                mean = pred.mean()
                var = pred.covariance().diagonal(dim1=-2, dim2=-1)
            else:
                draws = pred.sample(n_samples)
                mean, var = draws.mean(1), draws.var(1)
        return {
            "mean": mean.mean(0),
            "std": var.mean(0).sqrt(),
            "score_mean": z.mean(0),
            "score_std": z.std(0),
        }

    # -------------------------------------------------------------- helpers
    def _remember(self, z: Tensor, x: Optional[Tensor]) -> None:
        if z.dim() != 2 or z.shape[0] < 2:
            raise ValueError(f"z must be [N, d] with N >= 2, got {tuple(z.shape)}")
        if x is not None and (x.dim() != 2 or x.shape[0] != z.shape[0]):
            raise ValueError(f"x must be [N, p] with N = {z.shape[0]}, got {tuple(x.shape)}")
        self._event_dim = int(z.shape[1])
        self._dtype = z.dtype
        self._device = z.device

    def _zeros(self, n: int) -> Tensor:
        return torch.zeros(n, self.event_dim, dtype=self._dtype, device=self._device)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class IndependenceDependence(Dependence):
    r"""Rung 0: :math:`q(z \mid x) = \mathcal N(0, I)`.

    The joint is the product of the marginals,
    :math:`p(y \mid x) = \prod_j f_j(y_j \mid x)`.
    """

    name = "independence"

    def fit(self, z: Tensor, x: Optional[Tensor] = None) -> "IndependenceDependence":
        start = time.perf_counter()
        self._remember(z, None)
        self.fit_time_ = time.perf_counter() - start
        return self

    def predictive(self, n: int, x: Optional[Tensor] = None) -> GaussianJoint:
        d = self.event_dim
        eye = torch.eye(d).to(device=self._device, dtype=self._dtype)
        return GaussianJoint(self._zeros(n), eye)


def _low_rank_correlation(corr: Tensor, rank: int, n_iter: int = 50) -> Tensor:
    r"""Principal-factor fit :math:`R \approx V V^\top + \mathrm{diag}(\psi)` with unit diagonal."""
    d = corr.shape[0]
    psi = torch.zeros(d, dtype=corr.dtype, device=corr.device)
    factor = corr.new_zeros(d, rank)
    for _ in range(n_iter):
        reduced = corr - torch.diag(psi).to(device=corr.device, dtype=corr.dtype)
        vals, vecs = torch.linalg.eigh(reduced)
        top_vals = vals[-rank:].clamp(min=0.0)
        factor = vecs[:, -rank:] * top_vals.sqrt()
        psi = (1.0 - factor.square().sum(-1)).clamp(min=1e-3)
    approx = factor @ factor.mT + torch.diag(psi).to(device=corr.device, dtype=corr.dtype)
    inv_sd = approx.diagonal().rsqrt()
    return approx * inv_sd.unsqueeze(0) * inv_sd.unsqueeze(1)


class GaussianCopula(Dependence):
    r"""Rung 2: constant Gaussian copula, :math:`q(z \mid x) = \mathcal N(0, R)`.

    :math:`R` is the sample correlation :math:`S` of the training scores,
    shrunk towards the identity, :math:`R = (1 - \lambda) S + \lambda I`.
    With ``shrinkage="auto"`` the intensity is the Ledoit--Wolf estimate
    restricted to the off-diagonal entries (the diagonal of a correlation
    matrix is not estimated): with standardised rows :math:`x_i`,

    .. math::
        \lambda = \min\Big(1, \frac{\sum_i \sum_{j \ne k}(x_{ij}x_{ik} - S_{jk})^2 / N^2}
            {\sum_{j \ne k} S_{jk}^2}\Big).

    With ``rank = r`` the shrunk correlation is replaced by a
    low-rank-plus-diagonal fit :math:`V V^\top + \mathrm{diag}(\psi)`
    (principal factors, rescaled to unit diagonal), ``r (d + 1)``
    parameters instead of ``d (d - 1) / 2``.  Covariates are ignored.

    Attributes:
        correlation_: The fitted :math:`R` ``[d, d]``.
        shrinkage_: The intensity :math:`\lambda` used.
    """

    name = "gaussian_copula"

    def __init__(self, shrinkage: Union[str, float] = "auto", rank: Optional[int] = None) -> None:
        super().__init__()
        if isinstance(shrinkage, str):
            if shrinkage != "auto":
                raise ValueError(f"shrinkage must be 'auto' or a float, got {shrinkage!r}")
        elif not 0.0 <= float(shrinkage) <= 1.0:
            raise ValueError("shrinkage must lie in [0, 1]")
        if rank is not None and rank < 1:
            raise ValueError("rank must be at least 1")
        self.shrinkage = shrinkage
        self.rank = rank
        self.correlation_: Optional[Tensor] = None
        self.shrinkage_: float = 0.0
        self._chol: Optional[Tensor] = None

    def fit(self, z: Tensor, x: Optional[Tensor] = None) -> "GaussianCopula":
        start = time.perf_counter()
        self._remember(z, None)
        n_rows, d = z.shape
        centred = z - z.mean(0)
        std = centred.square().mean(0).sqrt().clamp(min=torch.finfo(z.dtype).tiny)
        xs = centred / std
        sample = xs.mT @ xs / n_rows
        eye = torch.eye(d).to(device=z.device, dtype=z.dtype)
        if isinstance(self.shrinkage, str):
            off = sample - eye
            delta = off.square().sum()
            row_sq = xs.square().sum(-1)
            beta = (
                row_sq.square().sum()
                - n_rows * sample.square().sum()
                - (xs.square() - 1.0).square().sum()
            ) / n_rows**2
            lam = 1.0 if float(delta) <= 0.0 else float((beta / delta).clamp(0.0, 1.0))
        else:
            lam = float(self.shrinkage)
        corr = (1.0 - lam) * sample + lam * eye
        # A constant score column has zero sample variance; a correlation matrix
        # has a unit diagonal whatever the data.
        corr = corr - torch.diag_embed(corr.diagonal()).to(device=z.device, dtype=z.dtype) + eye
        if self.rank is not None and self.rank < d:
            corr = _low_rank_correlation(corr, self.rank)
        self.shrinkage_ = lam
        self.correlation_ = corr
        self._chol = safe_cholesky(corr)
        self.fit_time_ = time.perf_counter() - start
        return self

    def predictive(self, n: int, x: Optional[Tensor] = None) -> GaussianJoint:
        if self._chol is None:
            raise RuntimeError("GaussianCopula is not fitted")
        return GaussianJoint(self._zeros(n), self._chol)


# ─────────────────────────────────────────────────────────────── training loop


def _standardiser(x: Tensor) -> tuple[Tensor, Tensor]:
    mean = x.mean(0)
    std = x.std(0) if x.shape[0] > 1 else torch.ones_like(mean)
    std = torch.where(std > 0, std, torch.ones_like(std))
    return mean, std


def _fit_score_model(
    module: nn.Module,
    nll: Callable[[Tensor, Optional[Tensor]], Tensor],
    z: Tensor,
    x: Optional[Tensor],
    *,
    max_epochs: int,
    patience: int,
    lr: float,
    batch_size: int,
    val_fraction: float,
    seed: int,
) -> dict[str, Any]:
    """Adam on mini-batches with held-out early stopping; the best weights are restored.

    The weights before the first step are a candidate, so the result is never
    worse than the initial model on the internal validation rows.
    """
    n_rows = int(z.shape[0])
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_rows, generator=gen).to(z.device)
    n_val = min(max(1, round(val_fraction * n_rows)), n_rows - 1)
    val, rows = perm[:n_val], perm[n_val:]

    def batch(idx: Tensor) -> Tensor:
        b = rows[idx.to(z.device)]
        return nll(z[b], None if x is None else x[b]).mean()

    def validation() -> Tensor:
        return nll(z[val], None if x is None else x[val]).mean()

    result = train(
        module,
        batch,
        int(rows.numel()),
        validation,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        lr=lr,
        generator=gen,
        optimizer="adam",
        include_initial=True,
        name=type(module).__name__,
    )
    module.requires_grad_(False)
    return {
        "initial_val_nll": result.initial,
        "best_val_nll": result.best,
        "epochs": len(result.history),
        "history": result.history,
    }


class _CholeskyNet(nn.Module):
    """``x -> (L(x), log s(x))``: unit lower-triangular ``L`` and per-target log-scales."""

    def __init__(self, n_in: int, d: int, hidden: Sequence[int]) -> None:
        super().__init__()
        self.d = d
        self.n_off = d * (d - 1) // 2
        self.net = mlp(n_in, hidden, self.n_off + d, zero_last=True)

    def factors(self, x: Tensor) -> tuple[Tensor, Tensor]:
        out = self.net(x)
        d = self.d
        rows, cols = torch.tril_indices(d, d, offset=-1, device=x.device)
        eye = torch.eye(d).to(device=x.device, dtype=x.dtype)
        tril = eye.expand(x.shape[0], d, d).clone()
        tril[:, rows, cols] = out[:, : self.n_off]
        return tril, out[:, self.n_off :]


class ConditionalGaussianCopula(Dependence):
    r"""Rung 3: Gaussian score density whose covariance depends on ``x``.

    .. math::
        q(z \mid x) = \mathcal N\big(0,\ L(x)\, D(x)^2 L(x)^\top\big),

    with :math:`L(x)` unit-diagonal lower triangular (the modified Cholesky
    parametrisation of Pourahmadi, 1999) and :math:`D(x) = \mathrm{diag}(s(x))`
    positive per-target scales, both output by an MLP of the standardised
    covariates and trained by the Gaussian negative log-likelihood of the
    scores.  The output layer is zero-initialised, so before training
    :math:`L = I`, :math:`s = 1` and :math:`q(z \mid x) = \mathcal N(0, I)` for
    every ``x``: training starts at the independence rung.  Early stopping
    on a held-out ``val_fraction`` of the rows restores the best weights
    (the initial weights included).

    The diagonal of :math:`L D^2 L^\top` is free, so the fitted :math:`q` can
    rescale the score marginals; see :meth:`Dependence.marginal_drift`.

    Args:
        hidden: MLP hidden widths.
        max_epochs, patience, lr, batch_size, val_fraction: Adam training and
            early stopping (``max_epochs=0`` returns the initial model).
        seed: Initialisation and shuffling seed.
    """

    name = "conditional_gaussian_copula"

    def __init__(
        self,
        hidden: Sequence[int] = (64, 64),
        max_epochs: int = 200,
        patience: int = 20,
        lr: float = 1e-3,
        batch_size: int = 256,
        val_fraction: float = 0.15,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.hidden = tuple(int(h) for h in hidden)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.val_fraction = float(val_fraction)
        self.seed = int(seed)
        self.net_: Optional[_CholeskyNet] = None
        self.training_: dict[str, Any] = {}
        self._x_loc: Optional[Tensor] = None
        self._x_scale: Optional[Tensor] = None

    def _nll(self, z: Tensor, xs: Tensor) -> Tensor:
        assert self.net_ is not None
        tril, log_s = self.net_.factors(xs)
        w = torch.linalg.solve_triangular(
            tril, z.unsqueeze(-1), upper=False, unitriangular=True
        ).squeeze(-1) * torch.exp(-log_s)
        return 0.5 * w.square().sum(-1) + log_s.sum(-1) + z.shape[-1] * _HALF_LOG_2PI

    def fit(self, z: Tensor, x: Optional[Tensor] = None) -> "ConditionalGaussianCopula":
        if x is None:
            raise ValueError("ConditionalGaussianCopula needs covariates x")
        start = time.perf_counter()
        self._remember(z, x)
        self.conditional = True
        x = x.to(device=z.device, dtype=z.dtype)
        self._x_loc, self._x_scale = _standardiser(x)
        xs = (x - self._x_loc) / self._x_scale
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            net = _CholeskyNet(int(x.shape[1]), int(z.shape[1]), self.hidden)
        self.net_ = net.to(device=z.device, dtype=z.dtype)

        def nll(zb: Tensor, xb: Optional[Tensor]) -> Tensor:
            assert xb is not None
            return self._nll(zb, xb)

        self.training_ = _fit_score_model(
            self.net_,
            nll,
            z,
            xs,
            max_epochs=self.max_epochs,
            patience=self.patience,
            lr=self.lr,
            batch_size=self.batch_size,
            val_fraction=self.val_fraction,
            seed=self.seed,
        )
        self.fit_time_ = time.perf_counter() - start
        return self

    def predictive(self, n: int, x: Optional[Tensor] = None) -> GaussianJoint:
        if self.net_ is None or self._x_loc is None or self._x_scale is None:
            raise RuntimeError("ConditionalGaussianCopula is not fitted")
        if x is None or x.shape[0] != n:
            raise ValueError(f"x must be [n, p] with n = {n}")
        xs = (x.to(device=self._device, dtype=self._dtype) - self._x_loc) / self._x_scale
        with torch.no_grad():
            tril, log_s = self.net_.factors(xs)
        return GaussianJoint(self._zeros(n), tril * torch.exp(log_s).unsqueeze(-2))


class _UnconditionalFlowJoint(FlowJoint):
    """:class:`FlowJoint` of an unconditional flow, repeated over ``n`` rows."""

    def __init__(self, flow_loss: NormalizingFlowLoss, n: int, like: Tensor) -> None:
        dist = flow_loss.flow()
        expanded = type(dist)(dist.transform, dist.base.expand((n,)))
        DistributionJoint.__init__(self, expanded)
        self.flow_loss = flow_loss
        self.context = like.new_zeros(n, 0)


def _zero_output_layer(layer: nn.Module) -> None:
    """Zero the parameters that produce a flow layer's spline parameters (identity map).

    Conditional and autoregressive layers have a hyper-network ``hyper``: its last
    module with a weight is zeroed.  Unconditional element-wise layers hold the
    spline parameters directly (``phi``): they are zeroed.
    """
    hyper = getattr(layer, "hyper", None)
    if isinstance(hyper, nn.Module):
        outputs = [
            m for m in hyper.modules() if isinstance(getattr(m, "weight", None), torch.Tensor)
        ]
        if not outputs:
            raise RuntimeError(f"no output layer found in {type(layer).__name__}.hyper")
        nn.init.zeros_(outputs[-1].weight)
        bias = getattr(outputs[-1], "bias", None)
        if isinstance(bias, torch.Tensor):
            nn.init.zeros_(bias)
        return
    phi = getattr(layer, "phi", None)
    params = list(phi.parameters()) if isinstance(phi, nn.Module) else []
    if not params:
        raise RuntimeError(f"cannot locate the spline parameters of {type(layer).__name__}")
    with torch.no_grad():
        for p in params:
            p.zero_()


class FlowDependence(Dependence):
    r"""Rung 5: conditional neural spline flow on the normal scores.

    :math:`q(z \mid x)` is a masked autoregressive rational-quadratic spline
    flow (:func:`~torchregress.losses.nflows.create_flow_model` with
    ``flow_type="nsf"``) conditioned on the standardised covariates, or
    unconditional when ``fit`` gets no ``x``; trained by negative
    log-likelihood with held-out early stopping (best weights restored, the
    initial ones included); the predictive is a
    :class:`~torchregress.joint.FlowJoint`.

    **Identity initialisation.**  The last layer of every autoregressive
    conditioner is set to zero.  A zero output means equal bin widths and
    heights and unit knot derivatives (zuko's parametrisation,
    :math:`\exp(0) = 1`), which is exactly the identity spline on
    :math:`[-B, B]`; outside :math:`[-B, B]` the spline is the identity by
    construction.  Every layer is therefore the identity map for every
    ``x``, and the initial flow is exactly :math:`q(z \mid x) = \mathcal N(0, I)`
    (to floating-point rounding): training starts at the independence rung.
    The earlier layers keep their random initialisation, so the gradient
    with respect to the zeroed layer is not zero.

    Scores are clipped to :math:`|z| \le \Phi^{-1}(1 - \varepsilon)` (4.75 for
    the default ``eps``), inside the default ``bound = 5``.

    Args:
        n_transforms: Autoregressive layers.
        hidden: Conditioner hidden widths.
        max_epochs, patience, lr, batch_size, val_fraction: Training and early
            stopping (``max_epochs=0`` returns the identity flow).
        bound: Spline half-width :math:`B`.
        bins: Spline bins.
        seed: Initialisation and shuffling seed.
    """

    name = "flow"

    def __init__(
        self,
        n_transforms: int = 3,
        hidden: Sequence[int] = (64, 64),
        max_epochs: int = 300,
        patience: int = 25,
        lr: float = 1e-3,
        bound: float = 5.0,
        bins: int = 8,
        batch_size: int = 256,
        val_fraction: float = 0.15,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.n_transforms = int(n_transforms)
        self.hidden = tuple(int(h) for h in hidden)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.lr = float(lr)
        self.bound = float(bound)
        self.bins = int(bins)
        self.batch_size = int(batch_size)
        self.val_fraction = float(val_fraction)
        self.seed = int(seed)
        self.flow_loss_: Optional[NormalizingFlowLoss] = None
        self.training_: dict[str, Any] = {}
        self._x_loc: Optional[Tensor] = None
        self._x_scale: Optional[Tensor] = None

    def _context(self, x: Optional[Tensor], n: int) -> Tensor:
        if self._x_loc is None or self._x_scale is None:
            return torch.zeros(n, 0, dtype=self._dtype, device=self._device)
        if x is None or x.shape[0] != n:
            raise ValueError(f"x must be [n, p] with n = {n}")
        return (x.to(device=self._device, dtype=self._dtype) - self._x_loc) / self._x_scale

    def fit(self, z: Tensor, x: Optional[Tensor] = None) -> "FlowDependence":
        start = time.perf_counter()
        self._remember(z, x)
        d = int(z.shape[1])
        if x is not None:
            x = x.to(device=z.device, dtype=z.dtype)
            self._x_loc, self._x_scale = _standardiser(x)
            self.conditional = True
            context_dim = int(x.shape[1])
        else:
            self._x_loc = self._x_scale = None
            self.conditional = False
            context_dim = 0
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            flow = create_flow_model(
                n_features=d,
                context_dim=context_dim,
                flow_type="nsf",
                n_transforms=self.n_transforms,
                hidden_features=list(self.hidden),
                bound=self.bound,
                bins=self.bins,
            )
        transforms: Any = getattr(flow, "transform")
        for layer in transforms.transforms:
            _zero_output_layer(layer)
        flow = flow.to(device=z.device, dtype=z.dtype)
        loss = NormalizingFlowLoss(flow)
        self.flow_loss_ = loss
        ctx = self._context(x, int(z.shape[0]))

        def nll(zb: Tensor, cb: Optional[Tensor]) -> Tensor:
            c = cb if cb is not None else zb.new_zeros(zb.shape[0], 0)
            return -loss.log_prob(c, zb)

        self.training_ = _fit_score_model(
            flow,
            nll,
            z,
            ctx,
            max_epochs=self.max_epochs,
            patience=self.patience,
            lr=self.lr,
            batch_size=self.batch_size,
            val_fraction=self.val_fraction,
            seed=self.seed,
        )
        self.fit_time_ = time.perf_counter() - start
        return self

    def predictive(self, n: int, x: Optional[Tensor] = None) -> FlowJoint:
        if self.flow_loss_ is None:
            raise RuntimeError("FlowDependence is not fitted")
        ctx = self._context(x, n)
        if self.conditional:
            return FlowJoint(self.flow_loss_, ctx)
        return _UnconditionalFlowJoint(self.flow_loss_, n, ctx)


# ─────────────────────────────────────────────────────────────── Schaake shuffle


class TemplateJoint(JointPredictive):
    r"""Draws-only predictive that resamples rows of a template ``[N, d]``.

    Row :math:`i` draws uniformly from the template rows ``index[i]``
    (``[n, k]``), or from all ``N`` rows when ``index`` is ``None``; the
    predictive is the empirical distribution of those rows, so :meth:`mean`
    and :meth:`covariance` are exact.

    Args:
        template: ``[N, d]``.
        n: Number of rows.
        index: Optional long ``[n, k]`` of template rows per predictive row.
    """

    def __init__(self, template: Tensor, n: int, index: Optional[Tensor] = None) -> None:
        if template.dim() != 2:
            raise ValueError(f"template must be [N, d], got {tuple(template.shape)}")
        if index is not None and (index.dim() != 2 or index.shape[0] != n):
            raise ValueError(f"index must be [n, k] with n = {n}")
        self.template = template
        self.index = index
        self._n = int(n)

    @property
    def batch_size(self) -> int:
        return self._n

    @property
    def event_dim(self) -> int:
        return int(self.template.shape[1])

    @property
    def dtype(self) -> torch.dtype:
        return self.template.dtype

    @property
    def device(self) -> torch.device:
        return self.template.device

    def _rows(self) -> Tensor:
        # [n, k, d] view of the template rows available to each predictive row
        if self.index is None:
            return self.template.unsqueeze(0).expand(self._n, -1, -1)
        return self.template[self.index]

    def sample(self, n_samples: int) -> Tensor:
        device = self.template.device
        if self.index is None:
            rows = torch.randint(int(self.template.shape[0]), (self._n, n_samples), device=device)
        else:
            pos = torch.randint(int(self.index.shape[1]), (self._n, n_samples), device=device)
            rows = self.index.gather(1, pos)
        return self.template[rows]

    def marginal(self, dims: Union[int, Sequence[int]]) -> "TemplateJoint":
        return TemplateJoint(
            self.template[:, target_index(dims, self.event_dim)], self._n, self.index
        )

    def mean(self, n_samples: int = 512) -> Tensor:
        if self.index is None:
            return self.template.mean(0).expand(self._n, -1)
        return self._rows().mean(1)

    def covariance(self, n_samples: int = 512) -> Tensor:
        if self.index is None:
            centred = self.template - self.template.mean(0)
            cov = centred.mT @ centred / self.template.shape[0]
            return cov.expand(self._n, -1, -1)
        rows = self._rows()
        centred = rows - rows.mean(1, keepdim=True)
        return centred.mT @ centred / rows.shape[1]


def _rank_normal_scores(z: Tensor) -> Tensor:
    """Per-column normal scores of the ranks, :math:`\\Phi^{-1}(r / (N + 1))` (ties by order)."""
    n_rows = z.shape[0]
    ranks = torch.argsort(torch.argsort(z, dim=0, stable=True), dim=0, stable=True)
    return torch.special.ndtri((ranks + 1).to(z.dtype) / (n_rows + 1))


class SchaakeShuffle(Dependence):
    r"""Rung 1: the empirical copula of the training scores (Schaake shuffle).

    The template is the set of training score vectors, each coordinate
    replaced by the normal score of its rank,
    :math:`t_{ij} = \Phi^{-1}(r_{ij} / (N + 1))`, so every template
    coordinate is the same standard-normal grid and the template carries
    only the rank dependence of the training scores (their empirical
    copula).  A draw for a new row is a template row drawn uniformly; pushed
    through :class:`SklarJoint`, this is the Schaake shuffle: marginal
    draws reordered to the ranks of a historical template (Clark et al.,
    2004).

    With ``n_neighbors = k`` the template of a new row is restricted to the
    ``k`` training rows nearest in standardised ``x`` (the "SimSchaake"
    variant of Scheuerer et al., 2017), so the dependence can vary with
    ``x``.  There is no density.

    Args:
        n_neighbors: Template size per row (``None``: all training rows).
    """

    name = "schaake"
    has_density = False

    def __init__(self, n_neighbors: Optional[int] = None) -> None:
        super().__init__()
        if n_neighbors is not None and n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1")
        self.n_neighbors = n_neighbors
        self.template_: Optional[Tensor] = None
        self._x_train: Optional[Tensor] = None
        self._x_loc: Optional[Tensor] = None
        self._x_scale: Optional[Tensor] = None

    def fit(self, z: Tensor, x: Optional[Tensor] = None) -> "SchaakeShuffle":
        start = time.perf_counter()
        self._remember(z, x)
        self.template_ = _rank_normal_scores(z)
        if self.n_neighbors is not None:
            if x is None:
                raise ValueError("n_neighbors needs covariates x")
            x = x.to(device=z.device, dtype=z.dtype)
            self._x_loc, self._x_scale = _standardiser(x)
            self._x_train = (x - self._x_loc) / self._x_scale
            self.conditional = True
        self.fit_time_ = time.perf_counter() - start
        return self

    def _neighbours(self, x: Tensor, chunk: int = 1024) -> Tensor:
        assert self._x_train is not None and self._x_loc is not None and self._x_scale is not None
        assert self.n_neighbors is not None
        k = min(self.n_neighbors, int(self._x_train.shape[0]))
        xs = (x.to(device=self._device, dtype=self._dtype) - self._x_loc) / self._x_scale
        parts = [
            torch.cdist(block, self._x_train).topk(k, dim=1, largest=False).indices
            for block in xs.split(chunk)
        ]
        return torch.cat(parts, dim=0)

    def predictive(self, n: int, x: Optional[Tensor] = None) -> TemplateJoint:
        if self.template_ is None:
            raise RuntimeError("SchaakeShuffle is not fitted")
        if self.n_neighbors is None:
            return TemplateJoint(self.template_, n)
        if x is None or x.shape[0] != n:
            raise ValueError(f"x must be [n, p] with n = {n}")
        return TemplateJoint(self.template_, n, self._neighbours(x))


# ─────────────────────────────────────────────────────────────── selection


def select_dependence(
    candidates: Sequence[Dependence],
    z_val: Tensor,
    x_val: Optional[Tensor] = None,
    n_se: float = 2.0,
) -> tuple[Dependence, dict[str, float]]:
    r"""Choose the lowest rung that held-out scores support.

    Every fitted density candidate is scored by its held-out mean log score
    density :math:`\bar\ell_c = \frac1N \sum_i \log q_c(z_i \mid x_i)`.  An
    :class:`IndependenceDependence` is added in front when the list has
    none, and draws-only candidates (:class:`SchaakeShuffle`) are skipped.
    Reading the list as a ladder from simplest to most complex, the choice is
    the first candidate that

    * scores at least as well as independence, :math:`\bar\ell_c \ge \bar\ell_0`, and
    * is within ``n_se`` standard errors of the best,
      :math:`\bar\ell_{\max} - \bar\ell_c \le n_{se}\,\mathrm{SE}`, with SE the
      standard error of the paired per-row differences.

    So the choice is never worse than independence on the validation data,
    and a more complex rung is kept only when its gain is resolved.
    ``n_se = 0`` gives the plain arg-max.

    Args:
        candidates: Fitted dependence models, simplest first.
        z_val: Held-out normal scores ``[N, d]`` (out of fold, as for ``fit``).
        x_val: Held-out covariates ``[N, p]`` (needed by conditional candidates).
        n_se: Tolerance in standard errors.

    Returns:
        The chosen model and the table ``{name: mean held-out log q}``
        (duplicate names get a ``#k`` suffix).
    """
    if n_se < 0:
        raise ValueError("n_se must be non-negative")
    ladder = [c for c in candidates if c.has_density]
    if not any(isinstance(c, IndependenceDependence) for c in ladder):
        ladder.insert(0, IndependenceDependence().fit(z_val))
    else:
        ladder.sort(key=lambda c: not isinstance(c, IndependenceDependence))
    names: list[str] = []
    scores: list[Tensor] = []
    for c in ladder:
        name, k = c.name, 2
        while name in names:
            name, k = f"{c.name}#{k}", k + 1
        names.append(name)
        scores.append(c.log_prob(z_val, x_val if c.conditional else None))
    raw_means = torch.stack([s.mean() for s in scores])
    # A candidate whose held-out score is not finite (NaN from a failed evaluation)
    # can never be chosen.
    means = torch.where(torch.isfinite(raw_means), raw_means, torch.full_like(raw_means, -math.inf))
    best = int(means.argmax())
    n_rows = int(z_val.shape[0])
    chosen = best
    for i in range(len(ladder)):
        if bool(means[i] < means[0]):
            continue
        diff = scores[best] - scores[i]
        se = float(diff.std() / math.sqrt(n_rows)) if n_rows > 1 else 0.0
        if float(means[best] - means[i]) <= n_se * se:
            chosen = i
            break
    return ladder[chosen], {name: float(m) for name, m in zip(names, raw_means)}
