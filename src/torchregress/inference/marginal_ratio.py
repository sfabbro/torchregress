r"""Marginal neural ratio estimation and posteriors from learned ratios.

:class:`MarginalRatioEstimator` learns the posterior of chosen subsets
("marginals") of the parameters directly, one ratio per marginal, without
learning the joint (Miller, Cole, Forré, Louppe & Weniger 2021, the
marginal-estimation half of TMNRE).  For a marginal :math:`\vartheta =
\theta_M` of the parameters :math:`\theta \in \mathbb R^P`, the
likelihood-to-evidence ratio

.. math::

    r_M(\vartheta, x) = \frac{p(\vartheta \mid x)}{p(\vartheta)}
    = \frac{p(\vartheta, x)}{p(\vartheta)\, p(x)}

is what a classifier separating joint pairs :math:`(\vartheta_i, x_i)` from
shuffled pairs :math:`(\vartheta_j, x_i)` estimates.  The other parameters
never enter the classifier: they are integrated out by the simulator, because
:math:`(\vartheta_i, x_i)` is a draw of :math:`p(\vartheta, x) = \int
p(\theta, x)\, d\theta_{-M}`.  The cost of a ratio head therefore depends on
the size of its marginal and of the embedding of :math:`x`, not on the number
of nuisance parameters.

:class:`RatioPosterior` turns a learned ratio and draws of the prior into a
:class:`~torchregress.joint.JointPredictive` over the parameters of a marginal,
:math:`p(\vartheta \mid x) = p(\vartheta)\, r(\vartheta, x)`: draws by
sampling-importance-resampling of the prior draws, a density normalised by the
Monte Carlo estimate of :math:`E_{p(\vartheta)}[r(\vartheta, x)]`, and that
estimate itself as a diagnostic (1 for a correctly normalised ratio).  Regions,
diagnostics and corner plots of :mod:`torchregress.joint` and
:mod:`torchregress.viz` then work on it unchanged.

Not here: truncation of the prior around one observation and sequential rounds
(the "T" of TMNRE).  The estimator is amortised over the training prior.

References
----------
.. [1] Miller, B. K., Cole, A., Forré, P., Louppe, G., & Weniger, C. (2021).
   Truncated Marginal Neural Ratio Estimation. *NeurIPS*.
   https://arxiv.org/abs/2107.01214
.. [2] Miller, B. K., Weniger, C., & Forré, P. (2022). Contrastive Neural Ratio
   Estimation. *NeurIPS*. https://arxiv.org/abs/2210.06170
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from itertools import combinations
from typing import Optional, Union, cast

import torch
from torch import Tensor, nn

from ..joint._common import TrainResult, mlp, seeded, train
from ..joint.predictive import JointPredictive
from ..losses.nre import (
    NeuralRatioLoss,
    build_ratio_classifier,
    in_batch_negative_index,
    neural_ratio_logits,
)
from .ratio import effective_sample_size, posterior_log_weights

__all__ = ["MarginalRatioEstimator", "RatioPosterior"]

#: Largest number of input elements (pairs x head input width) scored in one chunk.
_PAIR_ELEMENTS = 1 << 23

#: ``log_ratio(theta [n, S, d], x [n, ...]) -> [n, S]``.
LogRatioFn = Callable[[Tensor, Tensor], Tensor]
#: Prior draws ``[M, d]`` or a sampler ``n -> [n, d]``.
PriorSource = Union[Tensor, Callable[[int], Tensor]]


def _marginal_key(marginal: Union[int, Sequence[int]], n_params: int) -> tuple[int, ...]:
    key = (int(marginal),) if isinstance(marginal, int) else tuple(int(i) for i in marginal)
    if not key:
        raise ValueError("a marginal must name at least one parameter")
    if len(set(key)) != len(key):
        raise ValueError(f"the parameters of a marginal must be distinct, got {key}")
    for i in key:
        if not 0 <= i < n_params:
            raise ValueError(f"parameter index {i} is outside [0, {n_params})")
    return key


class MarginalRatioEstimator(nn.Module):
    r"""One ratio head per requested marginal of the parameters, on a shared embedding of ``x``.

    For every marginal :math:`M` (a tuple of parameter indices) a head
    :math:`s_M(\theta_M, e(x))` estimates :math:`\log r_M = \log p(\theta_M \mid
    x) - \log p(\theta_M)`, where :math:`e` is a single embedding network shared
    by all heads.  The heads are :func:`~torchregress.losses.build_ratio_classifier`
    MLPs on ``concat([theta_M, e(x)])`` and each is trained with its own
    :class:`~torchregress.losses.NeuralRatioLoss` (default
    ``criterion="cnre"``, ``gamma=1``): the loss of a batch is the sum over
    marginals, all marginals scored on the same pairs.

    Negatives come from the batch itself
    (:func:`~torchregress.losses.in_batch_negative_index`): for row ``i`` the
    parameters of ``K`` other rows, which are prior draws independent of
    ``x_i``.  ``K`` defaults to half the batch, the regime the CNRE paper
    recommends (large ``K``; the optimum is the exact ratio at every ``K``).

    Training rows ``(theta, x)`` must be independent draws of the prior and the
    simulator: the learned ratio is relative to that prior, and
    :meth:`posterior` must be given draws of the same prior.

    Cost.  A head's input width is ``len(M) + embedding_dim``; neither depends
    on ``n_params``.  The nuisance parameters only cost the ``[N, P]`` tensor
    of parameters, which the heads never read beyond their own columns.

    Args:
        n_params: Number of parameters ``P`` in the training ``theta``.
        marginals: The marginals to learn, tuples of parameter indices (an int
            is a 1-D marginal).  :meth:`univariate`, :meth:`pairs` and
            :meth:`jointly` build the usual lists, e.g.
            ``univariate([0, 1]) + pairs([0, 1])``.
        observation_dim: Width of ``x`` (``[N, D]``) for the default embedding;
            ignored when ``embedding`` is given.
        embedding: Network mapping ``x [N, ...]`` to ``[N, embedding_dim]``
            (for structured observations: a CNN, a set encoder).  Default: an
            MLP ``observation_dim -> embedding_hidden -> embedding_dim``.
        embedding_dim: Output width of the embedding.
        embedding_hidden: Hidden widths of the default embedding.
        head_hidden: Hidden widths of every ratio head.
        criterion: ``"cnre"`` (default), ``"bce"`` or ``"multiclass"``, as in
            :class:`~torchregress.losses.NeuralRatioLoss`.  ``"multiclass"``
            (NRE-B) leaves the normalisation of the ratio free, so
            :meth:`RatioPosterior.log_prob` and the normalisation diagnostic
            are then not meaningful.
        gamma: CNRE prior odds (``criterion="cnre"``).
        n_negatives: Negatives per row ``K``; ``None`` is half the batch.

    Example:
        >>> est = MarginalRatioEstimator(
        ...     n_params=10, marginals=MarginalRatioEstimator.pairs([0, 1]), observation_dim=5
        ... )
        >>> est.fit(theta, x, max_epochs=50)  # doctest: +SKIP
        >>> post = est.posterior(x_obs, prior_theta, marginal=(0, 1))  # doctest: +SKIP
        >>> post.sample(1000).shape  # [n_obs, 1000, 2]  # doctest: +SKIP
    """

    theta_loc: Tensor
    theta_scale: Tensor
    x_loc: Optional[Tensor]
    x_scale: Optional[Tensor]

    def __init__(
        self,
        n_params: int,
        marginals: Sequence[Union[int, Sequence[int]]],
        observation_dim: Optional[int] = None,
        *,
        embedding: Optional[nn.Module] = None,
        embedding_dim: int = 32,
        embedding_hidden: Sequence[int] = (64, 64),
        head_hidden: Sequence[int] = (64, 64),
        criterion: str = "cnre",
        gamma: float = 1.0,
        n_negatives: Optional[int] = None,
    ) -> None:
        super().__init__()
        n_params = int(n_params)
        if n_params < 1:
            raise ValueError(f"n_params must be positive, got {n_params}")
        if isinstance(marginals, (int, str)) or not marginals:
            raise ValueError("marginals must be a non-empty list of index tuples")
        keys = [_marginal_key(m, n_params) for m in marginals]
        if len(set(keys)) != len(keys):
            raise ValueError(f"marginals must be distinct, got {keys}")
        if n_negatives is not None and int(n_negatives) < 1:
            raise ValueError(f"n_negatives must be at least 1, got {n_negatives}")
        embedding_dim = int(embedding_dim)
        if embedding is None:
            if observation_dim is None:
                raise ValueError("pass observation_dim for the default embedding, or an embedding")
            observation_dim = int(observation_dim)
            embedding = mlp(observation_dim, embedding_hidden, embedding_dim)
            self.register_buffer("x_loc", torch.zeros(observation_dim))
            self.register_buffer("x_scale", torch.ones(observation_dim))
        else:
            if not isinstance(embedding, nn.Module):
                raise TypeError(f"embedding must be a torch.nn.Module, got {type(embedding)}")
            self.x_loc = None
            self.x_scale = None
        self.n_params = n_params
        self.marginals: list[tuple[int, ...]] = keys
        self.embedding = embedding
        self.embedding_dim = embedding_dim
        self.n_negatives = None if n_negatives is None else int(n_negatives)
        self.ratio_losses = nn.ModuleList(
            NeuralRatioLoss(
                build_ratio_classifier(len(key), embedding_dim, hidden_dims=head_hidden),
                criterion=criterion,
                gamma=gamma,
            )
            for key in keys
        )
        self.register_buffer("theta_loc", torch.zeros(n_params))
        self.register_buffer("theta_scale", torch.ones(n_params))
        self.training_result_: Optional[TrainResult] = None

    # ------------------------------------------------------------ marginals
    @staticmethod
    def univariate(dims: Sequence[int]) -> list[tuple[int, ...]]:
        """Every 1-D marginal of ``dims``: ``[(i,) for i in dims]``."""
        return [(int(i),) for i in dims]

    @staticmethod
    def pairs(dims: Sequence[int]) -> list[tuple[int, ...]]:
        """Every 2-D marginal of ``dims`` (in the order of ``dims``)."""
        return [(int(i), int(j)) for i, j in combinations(dims, 2)]

    @staticmethod
    def jointly(dims: Sequence[int]) -> list[tuple[int, ...]]:
        """The single marginal of ``dims`` taken jointly: ``[tuple(dims)]``."""
        return [tuple(int(i) for i in dims)]

    def _index(self, marginal: Union[int, Sequence[int], None]) -> int:
        if marginal is None:
            if len(self.marginals) == 1:
                return 0
            raise ValueError(f"name the marginal: this estimator has {self.marginals}")
        key = _marginal_key(marginal, self.n_params)
        try:
            return self.marginals.index(key)
        except ValueError:
            raise ValueError(
                f"marginal {key} was not trained; trained marginals: {self.marginals}"
            ) from None

    def head(self, marginal: Union[int, Sequence[int]]) -> nn.Module:
        """The ratio head of a marginal (a classifier on ``concat([theta_M, e(x)])``)."""
        return self._head(self._index(marginal))

    def _head(self, i: int) -> nn.Module:
        return cast(NeuralRatioLoss, self.ratio_losses[i]).classifier

    # ------------------------------------------------------------- features
    def embed(self, x: Tensor) -> Tensor:
        """Embedding ``e(x)`` of observations ``[N, ...]``, ``[N, embedding_dim]``."""
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if self.x_loc is not None and self.x_scale is not None:
            x = (x - self.x_loc) / self.x_scale
        out = self.embedding(x)
        if not isinstance(out, Tensor) or out.dim() != 2 or out.shape[1] != self.embedding_dim:
            shape = tuple(out.shape) if isinstance(out, Tensor) else type(out)
            raise ValueError(f"the embedding must return [N, {self.embedding_dim}], got {shape}")
        return out

    def _standard_theta(self, theta: Tensor, key: Sequence[int]) -> Tensor:
        dims = list(key)
        return (theta - self.theta_loc[dims]) / self.theta_scale[dims]

    def log_ratio(
        self, theta: Tensor, x: Tensor, marginal: Union[int, Sequence[int], None] = None
    ) -> Tensor:
        r"""Estimated :math:`\log r_M(\theta_M, x)` on matched rows, ``[N]``.

        Args:
            theta: Parameters of the marginal ``[N, len(M)]`` (1-D for a 1-D
                marginal), in the order of the marginal's indices.
            x: Observations ``[N, ...]``.
            marginal: A trained marginal (optional when there is only one).
        """
        i = self._index(marginal)
        key = self.marginals[i]
        if theta.dim() == 1:
            theta = theta.unsqueeze(-1)
        if theta.dim() != 2 or theta.shape[1] != len(key):
            raise ValueError(f"theta must be [N, {len(key)}] for marginal {key}")
        head = self._head(i)
        return neural_ratio_logits(head, self._standard_theta(theta, key), self.embed(x))

    def candidate_log_ratio(
        self, theta: Tensor, x: Tensor, marginal: Union[int, Sequence[int], None] = None
    ) -> Tensor:
        r"""Log ratios of ``S`` candidates per observation, ``[n, S]``.

        The embedding of each row of ``x`` is computed once and paired with its
        candidates, in chunks.  This is the ``log_ratio`` function of the
        :class:`RatioPosterior` that :meth:`posterior` returns.

        Args:
            theta: Candidates ``[n, S, len(M)]``, or a bank ``[S, len(M)]``
                shared by every row.
            x: Observations ``[n, ...]``.
            marginal: A trained marginal (optional when there is only one).
        """
        i = self._index(marginal)
        key = self.marginals[i]
        d = len(key)
        emb = self.embed(x)
        n = emb.shape[0]
        if theta.dim() == 2:
            theta = theta.unsqueeze(0).expand(n, -1, -1)
        if theta.dim() != 3 or theta.shape[0] != n or theta.shape[2] != d:
            raise ValueError(f"theta must be [n, S, {d}] or [S, {d}] with n = {n}")
        s = theta.shape[1]
        theta = self._standard_theta(theta.to(emb), key)
        head = self._head(i)
        rows = max(1, _PAIR_ELEMENTS // max(1, s * (d + self.embedding_dim)))
        blocks = []
        for start in range(0, n, rows):
            th = theta[start : start + rows]
            b = th.shape[0]
            e = emb[start : start + rows].unsqueeze(1).expand(b, s, -1)
            logits = neural_ratio_logits(head, th.reshape(b * s, d), e.reshape(b * s, -1))
            blocks.append(logits.view(b, s))
        return torch.cat(blocks, dim=0)

    # -------------------------------------------------------------- training
    def _n_negatives(self, n: int) -> int:
        k = max(1, n // 2) if self.n_negatives is None else self.n_negatives
        return min(k, n - 1)

    def loss(
        self, theta: Tensor, x: Tensor, *, generator: Optional[torch.Generator] = None
    ) -> Tensor:
        """Sum over marginals of the ratio losses on a batch (scalar).

        Every marginal is scored on the same pairs: the batch rows are the
        positives, and the negatives of row ``i`` are the parameters of ``K``
        other rows (:func:`~torchregress.losses.in_batch_negative_index`).

        Args:
            theta: Parameters ``[N, P]`` (N >= 2).
            x: Observations ``[N, ...]``.
            generator: Optional CPU generator for the choice of negatives.
        """
        if theta.dim() != 2 or theta.shape[1] != self.n_params:
            raise ValueError(f"theta must be [N, {self.n_params}], got {tuple(theta.shape)}")
        n = int(theta.shape[0])
        if x.shape[0] != n:
            raise ValueError(f"theta and x must have the same rows, got {n} and {x.shape[0]}")
        index = in_batch_negative_index(
            n, self._n_negatives(n), generator=generator, device=theta.device
        )
        emb = self.embed(x)
        theta = theta.to(emb)
        total = emb.new_zeros(())
        for key, loss_fn in zip(self.marginals, self.ratio_losses):
            th = self._standard_theta(theta[:, list(key)], key)
            total = total + loss_fn(th, emb, negative_context=th[index])
        return total

    def fit(
        self,
        theta: Tensor,
        x: Tensor,
        *,
        max_epochs: int = 200,
        patience: int = 20,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        val_fraction: float = 0.1,
        standardize: bool = True,
        seed: int = 0,
    ) -> "MarginalRatioEstimator":
        """Train every head and the embedding on simulations ``(theta, x)``.

        Mini-batch AdamW with early stopping on the loss of held-out rows (the
        trainer of :mod:`torchregress.joint`): the best validation checkpoint
        is restored and the module is left in eval mode.  The validation loss
        uses a fixed choice of negatives, so epochs are compared on the same
        pairs.  The caller's torch RNG state is left as it was.

        Args:
            theta: Parameters ``[N, P]``, independent prior draws.
            x: Simulated observations ``[N, ...]``, one per row of ``theta``.
            max_epochs, patience, batch_size, lr, weight_decay: Training budget.
            val_fraction: Rows held out for early stopping.
            standardize: Standardise ``theta`` (and ``x`` for the default
                embedding) with the training mean and standard deviation.
                The ratio is invariant to this reparametrisation.
            seed: Seed of the split, the shuffling and the negatives.

        Returns:
            ``self``; the training record is in ``training_result_``.
        """
        if theta.dim() != 2 or theta.shape[1] != self.n_params:
            raise ValueError(f"theta must be [N, {self.n_params}], got {tuple(theta.shape)}")
        n = int(theta.shape[0])
        if x.shape[0] != n:
            raise ValueError(f"theta and x must have the same rows, got {n} and {x.shape[0]}")
        if not 0.0 < val_fraction < 1.0:
            raise ValueError(f"val_fraction must lie in (0, 1), got {val_fraction}")
        n_val = max(2, int(round(val_fraction * n)))
        if n - n_val < 2:
            raise ValueError(f"need at least {n_val + 2} rows, got {n}")
        with seeded(seed):
            gen = torch.Generator().manual_seed(seed)
            perm = torch.randperm(n, generator=gen).to(theta.device)
            val, tr = perm[:n_val], perm[n_val:]
            if standardize:
                with torch.no_grad():
                    self.theta_loc.copy_(theta[tr].mean(0))
                    self.theta_scale.copy_(theta[tr].std(0).clamp_min(1e-12))
                    if self.x_loc is not None and self.x_scale is not None:
                        xt = x[tr].reshape(len(tr), -1)
                        self.x_loc.copy_(xt.mean(0))
                        self.x_scale.copy_(xt.std(0).clamp_min(1e-12))
            n_train = int(tr.numel())

            def batch_loss(rows: Tensor) -> Tensor:
                rows = tr[rows.to(tr.device)]
                if rows.numel() < 2:  # a last batch of one row: add its neighbour
                    pos = int((tr == rows[0]).nonzero()[0, 0])
                    rows = torch.cat([rows, tr[(pos + 1) % n_train].reshape(1)])
                return self.loss(theta[rows], x[rows], generator=gen)

            def val_loss() -> Tensor:
                val_gen = torch.Generator().manual_seed(seed + 1)
                total, count = 0.0, 0
                for start in range(0, n_val, batch_size):
                    rows = val[start : start + batch_size]
                    if rows.numel() < 2:
                        rows = val[-2:]
                    total += float(self.loss(theta[rows], x[rows], generator=val_gen)) * len(rows)
                    count += len(rows)
                return torch.tensor(total / count)

            self.training_result_ = train(
                self,
                batch_loss,
                n_train,
                val_loss,
                max_epochs=max_epochs,
                patience=patience,
                batch_size=batch_size,
                lr=lr,
                generator=gen,
                optimizer="adamw",
                weight_decay=weight_decay,
                name="the marginal ratio estimator",
            )
        return self

    # ------------------------------------------------------------- inference
    def posterior(
        self,
        x: Tensor,
        prior_samples: PriorSource,
        marginal: Union[int, Sequence[int], None] = None,
        prior_log_prob: Optional[Callable[[Tensor], Tensor]] = None,
        *,
        n_proposals: int = 4096,
        min_ess: float = 100.0,
    ) -> "RatioPosterior":
        r"""Posterior of one marginal for observations ``x``, a :class:`RatioPosterior`.

        Args:
            x: Observations ``[n, ...]``.
            prior_samples: Draws of the training prior: ``[M, len(M)]`` for the
                marginal, ``[M, P]`` full parameter draws (the marginal's
                columns are taken), or a sampler ``m -> [m, len(M)]`` called
                once with ``n_proposals``.
            marginal: A trained marginal (optional when there is only one).
            prior_log_prob: Log density of the prior of the marginal,
                ``[..., len(M)] -> [...]``; gives the posterior a density.
            n_proposals: Draws taken from a sampler.
            min_ess: Effective sample size below which :meth:`RatioPosterior.sample`
                warns.
        """
        i = self._index(marginal)
        key = self.marginals[i]
        if isinstance(prior_samples, Tensor):
            bank = prior_samples
            if bank.dim() == 1:
                bank = bank.unsqueeze(-1)
            if bank.dim() == 2 and bank.shape[1] == self.n_params and len(key) != self.n_params:
                bank = bank[:, list(key)]
            prior: PriorSource = bank
        else:
            prior = prior_samples

        def log_ratio(theta: Tensor, obs: Tensor) -> Tensor:
            return self.candidate_log_ratio(theta, obs, key)

        return RatioPosterior(
            log_ratio,
            x,
            prior,
            prior_log_prob=prior_log_prob,
            n_proposals=n_proposals,
            min_ess=min_ess,
        )


class RatioPosterior(JointPredictive):
    r"""Posterior :math:`p(\vartheta \mid x) = p(\vartheta)\, r(\vartheta, x)` from a learned ratio.

    A :class:`~torchregress.joint.JointPredictive` over the ``d`` parameters of
    one marginal for ``n`` observations, built from a log-ratio function and a
    bank of ``M`` draws :math:`\vartheta_k \sim p(\vartheta)` of the prior the
    ratio was trained against.  With :math:`r_{ik} = r(\vartheta_k, x_i)`:

    * ``sample(S)``: sampling-importance-resampling, ``S`` bank draws per row
      with probabilities :math:`r_{ik} / \sum_j r_{ij}`
      (:func:`~torchregress.inference.posterior_log_weights`; no proposal
      correction, the bank is the prior).  The draws repeat bank points; the
      Kish :meth:`effective_sample_size` says how many distinct draws they are
      worth, and ``sample`` warns when a row falls below ``min_ess``.
    * ``normalisation()``: :math:`\hat Z_i = \frac1M \sum_k r_{ik}`, the Monte
      Carlo estimate of :math:`E_{p(\vartheta)}[r(\vartheta, x_i)]`, which is
      1 for a correctly normalised ratio (the diagnostic of the CNRE paper).
      Its relative standard error is about
      :math:`\sqrt{1/\mathrm{ESS}_i - 1/M}`.
    * ``log_prob(theta)`` (when ``prior_log_prob`` is given):
      :math:`\log r(\vartheta, x_i) + \log p(\vartheta) - \log \hat Z_i`, the
      density whose SIR draws ``sample`` returns.
    * ``mean()`` and ``covariance()``: the importance-weighted moments of the
      bank (no resampling noise).

    The log ratio and the weights of the bank are computed once (without
    gradients) and cached.  ``marginal(dims)`` is draws-only.

    Args:
        log_ratio: ``log_ratio(theta [n, S, d], x [n, ...]) -> [n, S]``, the
            log ratio of ``S`` candidates per observation; for a classifier
            ``clf`` on ``concat([theta, x])`` trained with
            :class:`~torchregress.losses.NeuralRatioLoss`, wrap
            :func:`~torchregress.losses.neural_ratio_logits`.
            :meth:`MarginalRatioEstimator.posterior` passes
            :meth:`MarginalRatioEstimator.candidate_log_ratio`.
        x: Observations ``[n, ...]``.
        prior_samples: Prior draws ``[M, d]`` (``[M]`` for ``d = 1``), or a
            sampler ``m -> [m, d]`` called once with ``n_proposals``.
        prior_log_prob: Prior log density ``[..., d] -> [...]``.  Without it the
            posterior has no density (``has_density`` is ``False``).
        n_proposals: Draws taken from a sampler.
        min_ess: Effective sample size below which ``sample`` warns.
    """

    has_latent = False
    exact_marginals = False

    def __init__(
        self,
        log_ratio: LogRatioFn,
        x: Tensor,
        prior_samples: PriorSource,
        *,
        prior_log_prob: Optional[Callable[[Tensor], Tensor]] = None,
        n_proposals: int = 4096,
        min_ess: float = 100.0,
    ) -> None:
        if not callable(log_ratio):
            raise TypeError("log_ratio must be callable: (theta [n, S, d], x [n, ...]) -> [n, S]")
        if not isinstance(x, Tensor) or x.dim() < 1 or x.shape[0] < 1:
            raise ValueError("x must be a tensor [n, ...] with n >= 1")
        if isinstance(prior_samples, Tensor):
            bank = prior_samples
        elif callable(prior_samples):
            with torch.no_grad():
                bank = prior_samples(int(n_proposals))
            if not isinstance(bank, Tensor):
                raise TypeError("the prior sampler must return a tensor [m, d]")
        else:
            raise TypeError("prior_samples must be a tensor [M, d] or a sampler m -> [m, d]")
        bank = bank.detach()
        if bank.dim() == 1:
            bank = bank.unsqueeze(-1)
        if bank.dim() != 2 or bank.shape[0] < 2:
            raise ValueError(f"prior draws must be [M, d] with M >= 2, got {tuple(bank.shape)}")
        if not bank.is_floating_point():
            bank = bank.to(torch.get_default_dtype())
        if not bool(torch.isfinite(bank).all()):
            raise ValueError("prior draws must be finite")
        if not math.isfinite(float(min_ess)) or float(min_ess) < 0:
            raise ValueError(f"min_ess must be a finite non-negative number, got {min_ess}")
        self.log_ratio_fn = log_ratio
        self.x = x
        self.prior_samples = bank
        self.prior_log_prob = prior_log_prob
        self.min_ess = float(min_ess)
        self.has_density = prior_log_prob is not None
        self._bank_log_ratio: Optional[Tensor] = None

    # ------------------------------------------------------------- layout
    @property
    def batch_size(self) -> int:
        return int(self.x.shape[0])

    @property
    def event_dim(self) -> int:
        return int(self.prior_samples.shape[1])

    @property
    def dtype(self) -> torch.dtype:
        return self.prior_samples.dtype

    @property
    def device(self) -> torch.device:
        return self.prior_samples.device

    @property
    def n_proposals(self) -> int:
        """Number of prior draws ``M`` in the bank."""
        return int(self.prior_samples.shape[0])

    # ------------------------------------------------------------- weights
    def _scores(self, theta: Tensor) -> Tensor:
        """``log r`` of candidates ``[n, S, d]``, ``[n, S]``, without gradients."""
        with torch.no_grad():
            out = self.log_ratio_fn(theta, self.x)
        if not isinstance(out, Tensor) or tuple(out.shape) != tuple(theta.shape[:2]):
            shape = tuple(out.shape) if isinstance(out, Tensor) else type(out)
            raise ValueError(f"log_ratio returned {shape}, expected {tuple(theta.shape[:2])}")
        return out.to(device=self.device, dtype=self.dtype)

    @property
    def bank_log_ratio(self) -> Tensor:
        r"""``[n, M]`` log ratios :math:`\log r(\vartheta_k, x_i)` of the prior bank (cached)."""
        if self._bank_log_ratio is None:
            bank = self.prior_samples.unsqueeze(0).expand(self.batch_size, -1, -1)
            scores = self._scores(bank)
            if not bool(torch.isfinite(scores).all()):
                raise ValueError("the log ratio is not finite on some prior draws")
            self._bank_log_ratio = scores
        return self._bank_log_ratio

    def log_weights(self) -> Tensor:
        """``[n, M]`` normalised log resampling weights of the prior bank."""
        return posterior_log_weights(self.bank_log_ratio, dim=-1)

    def effective_sample_size(self) -> Tensor:
        """Kish effective sample size of the resampling weights per row, ``[n]`` (1 to ``M``)."""
        return effective_sample_size(self.bank_log_ratio, dim=-1)

    def log_normalisation(self) -> Tensor:
        r"""``[n]`` :math:`\log \hat Z_i = \log \frac1M \sum_k r(\vartheta_k, x_i)`."""
        return torch.logsumexp(self.bank_log_ratio, dim=-1) - math.log(self.n_proposals)

    def normalisation(self) -> Tensor:
        r"""``[n]`` Monte Carlo :math:`\hat Z_i \approx E_{p(\vartheta)}[r(\vartheta, x_i)]`.

        1 for a correctly normalised ratio.

        The relative Monte Carlo standard error is about
        :math:`\sqrt{1/\mathrm{ESS}_i - 1/M}`
        (:meth:`effective_sample_size`), so a departure from 1 larger than a few
        times that is an error of the ratio, not of the bank.
        """
        return self.log_normalisation().exp()

    # ------------------------------------------------------------- predictive
    def sample(self, n_samples: int) -> Tensor:
        """``[n, S, d]`` draws by sampling-importance-resampling of the prior bank."""
        n_samples = int(n_samples)
        if n_samples < 1:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        ess = self.effective_sample_size()
        low = ess < self.min_ess
        if bool(low.any()):
            warnings.warn(
                f"RatioPosterior: {int(low.sum())} of {self.batch_size} rows have an effective "
                f"sample size below {self.min_ess:g} (smallest {float(ess.min()):.1f} of "
                f"{self.n_proposals} prior draws); the draws repeat a few bank points. Use more "
                "prior draws, or a prior closer to the posterior with the ratio corrected.",
                RuntimeWarning,
                stacklevel=2,
            )
        probs = self.log_weights().double().exp()
        index = torch.multinomial(probs, n_samples, replacement=True)
        return self.prior_samples[index]

    def log_prob(self, y: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if self.prior_log_prob is None:
            raise NotImplementedError(
                "RatioPosterior has a density only when prior_log_prob is given"
            )
        if mask is not None:
            raise NotImplementedError("RatioPosterior cannot integrate out masked parameters")
        y = self._check(y)
        theta = y.unsqueeze(1) if y.dim() == 2 else y
        with torch.no_grad():
            prior = self.prior_log_prob(theta)
        if not isinstance(prior, Tensor) or tuple(prior.shape) != tuple(theta.shape[:2]):
            raise ValueError("prior_log_prob must map [..., d] to [...]")
        out = self._scores(theta) + prior.to(self.dtype) - self.log_normalisation().unsqueeze(1)
        return out[:, 0] if y.dim() == 2 else out

    def mean(self, n_samples: int = 512) -> Tensor:
        """Importance-weighted mean of the bank, ``[n, d]`` (``n_samples`` is ignored)."""
        return self.log_weights().exp() @ self.prior_samples

    def covariance(self, n_samples: int = 512) -> Tensor:
        """Importance-weighted covariance of the bank, ``[n, d, d]`` (``n_samples`` is ignored)."""
        w = self.log_weights().exp()  # [n, M]
        centred = self.prior_samples.unsqueeze(0) - self.mean().unsqueeze(1)  # [n, M, d]
        return (centred * w.unsqueeze(-1)).mT @ centred
