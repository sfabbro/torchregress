"""
Mixture Density Network (MDN) loss functions.

This module provides loss functions for Mixture Density Networks,
which model outputs as mixtures of Gaussian or multivariate Student-t
distributions (both evaluated by :class:`~torchregress.losses.StudentTMixture`;
the Gaussian is its ``df = inf`` case).

Reference: Bishop, C. M. (1994). Mixture Density Networks. Neural Computing
Research Group Report NCRG/94/004, Aston University.
"""

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .base import DistributionLoss
from .loss_registry import register_regression_loss
from .student_t_mixture import StudentTMixture

TensorTriplet = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@register_regression_loss("mdn")
class MixtureDensityLoss(DistributionLoss):
    """
    Negative Log-Likelihood loss for Mixture Density Networks.

    Models the output as a mixture of Gaussian (default) or multivariate
    Student-t distributions with either diagonal or full scale matrices.

    Args:
        n_components (int): Number of mixture components
        n_features (int): Number of output features
        covariance_type (str): Type of covariance matrix ('diagonal' or 'full')
        full_parameterization (str): For ``covariance_type='full'``, what the lower
            triangular factor ``T`` built from the network output parameterises:
            ``'precision'`` (default; precision ``P = T T^T``, as in sbi's MDN) or
            ``'covariance'`` (``Sigma = T T^T``, the pre-0.3 behaviour).  The
            precision form makes the log-density linear in ``T`` (no triangular
            solve), which trains far better in several dimensions: on the
            16-target scm20d benchmark the covariance form did not even fit the
            training data (train NLL 0.42 vs -3.45, test 3.32 vs 0.31).  The
            output layout is the same for both.
        min_std (float): Floor added to every diagonal scale (softplus + min_std)
        eps (float): Floor on the mixture variance in ``predict_mean_std``
        df (float): Degrees of freedom of the components. ``math.inf`` (default) is
            the Gaussian MDN; a finite ``df > 0`` gives multivariate Student-t
            components ``t_df(mu_k, Sigma_k)`` (``Sigma_k`` the scale matrix built
            from the same output layout), which tolerate outliers and heavy tails.
        learn_df (bool): Learn one shared ``df = df_init * exp(raw_df)`` (``raw_df``
            a parameter of this module, starting at 0, so the initial ``df`` is exact
            in any dtype). On the log scale an Adam step changes ``df`` by a fixed
            fraction, so it moves as readily toward heavy tails as toward the
            Gaussian limit (reached geometrically, not linearly). Give ``raw_df``
            its own parameter group at about ``1e-2``: on a linear-head toy it
            reached ``df ≈ 3.4`` (truth 3) in 1000 steps and ``df ≈ 60`` on
            Gaussian data, where a linear parametrisation stalled near 10.
        reduction (str): Specifies the reduction to apply: 'none' | 'mean' | 'sum'
            Default: 'mean'

    Mathematical Formulation:
        For a mixture density network with K components, the negative log
        likelihood is given by:

        NLL = -log(∑(w_k * p_k(y|μ_k, Σ_k)))

        where w_k are mixture weights, and p_k is the density of the k-th
        component: N(μ_k, Σ_k) for ``df = inf``, otherwise t_df(μ_k, Σ_k).

    Notes:
        - The model output is expected to contain all distribution parameters
          concatenated in a single tensor.
        - For 'diagonal' covariance: output size = n_components + 2*n_components*n_features
        - For 'full' covariance: output size = n_components + n_components*n_features +
          n_components*n_features*(n_features+1)/2

    Examples:
        >>> import torch
        >>> loss_fn = MixtureDensityLoss(n_components=2, n_features=3, covariance_type='diagonal')
        >>> # Create model predictions with mixture weights, means, and stds
        >>> y_pred = torch.randn(10, 2 + 2*2*3)  # Batch of 10, 2 components, 3 features
        >>> target = torch.randn(10, 3)  # Ground truth values
        >>> loss = loss_fn(y_pred, target)
    """

    def __init__(
        self,
        n_components: int,
        n_features: int,
        covariance_type: str = "diagonal",
        min_std: float = 1e-3,
        eps: float = 1e-8,
        reduction: str = "mean",
        full_parameterization: str = "precision",
        df: float = math.inf,
        learn_df: bool = False,
    ):
        super().__init__(reduction=reduction)
        df = float(df)
        if math.isnan(df) or df <= 0:
            raise ValueError(f"df must be > 0 (math.inf for Gaussian components), got {df}")
        if learn_df and math.isinf(df):
            raise ValueError("learn_df=True needs a finite initial df")
        self.learn_df = bool(learn_df)
        self._df = df
        if self.learn_df:
            self.raw_df = nn.Parameter(torch.zeros(()))
        self.n_components = n_components
        self.n_features = n_features
        self.covariance_type = covariance_type.lower()
        self.full_parameterization = full_parameterization.lower()
        if self.full_parameterization not in ("precision", "covariance"):
            raise ValueError(
                "full_parameterization must be 'precision' or 'covariance', got "
                f"{full_parameterization!r}"
            )
        self.min_std = min_std
        self.eps = eps
        self.log_2pi = math.log(2 * math.pi)

        if self.covariance_type not in ["diagonal", "full"]:
            raise ValueError(
                f"Unsupported covariance_type: {covariance_type}. Expected 'diagonal' or 'full'"
            )

        # Calculate expected output size for validation
        if self.covariance_type == "diagonal":
            # n_components (mixture weights) + n_components * n_features (means) +
            # n_components * n_features (stds)
            self.expected_output_size = n_components + 2 * n_components * n_features
        else:  # 'full'
            # n_components + n_components * n_features (means) +
            # n_components * n_features * (n_features + 1) / 2 (covs)
            # We use triangular parameterization for covariance matrices
            self.expected_output_size = (
                n_components
                + n_components * n_features
                + n_components * n_features * (n_features + 1) // 2
            )

    def _extract_distribution_parameters(self, y_pred: torch.Tensor) -> TensorTriplet:
        """
        Extract mixture parameters from model predictions.

        Args:
            y_pred (torch.Tensor): Model predictions. For diagonal covariance, shape should be
                   [..., n_components + n_components*n_features + n_components*n_features]
                   For full covariance, shape depends on the parameterization.

        Returns:
            tuple: (log_mixture_weights, means, stds or cov_factors)
        """
        # Validate output size
        if y_pred.shape[-1] != self.expected_output_size:
            raise ValueError(
                f"Model output size {y_pred.shape[-1]} doesn't match expected size "
                f"{self.expected_output_size}"
            )

        batch_shape = y_pred.shape[:-1]

        # Extract mixture weight logits - always the first n_components elements
        logits = y_pred[..., : self.n_components]
        # Use log-softmax so extreme logits retain nonzero gradients
        log_weights = F.log_softmax(logits, dim=-1)

        # Extract means - always after the mixture weights
        means_start = self.n_components
        means_end = means_start + self.n_components * self.n_features
        means = y_pred[..., means_start:means_end].reshape(
            *batch_shape, self.n_components, self.n_features
        )

        # Extract covariance parameters - depends on covariance_type
        if self.covariance_type == "diagonal":
            # For diagonal, we extract log_stds and convert to stds
            log_stds_start = means_end
            log_stds = y_pred[..., log_stds_start:].reshape(
                *batch_shape, self.n_components, self.n_features
            )

            # Apply softplus with shift for better numerical stability
            # This ensures stds are always positive and not too close to zero
            stds = F.softplus(log_stds) + self.min_std

            return log_weights, means, stds
        else:  # 'full'
            # For full covariance, we extract parameters for lower triangular matrices
            # using Cholesky decomposition parameterization
            tril_indices = torch.tril_indices(self.n_features, self.n_features)
            n_tril_elements = len(tril_indices[0])

            tril_start = means_end
            tril_values = y_pred[..., tril_start:].reshape(
                *batch_shape, self.n_components, n_tril_elements
            )

            # Build the Cholesky factor tensor without performing a second
            # read-then-in-place-write on the same tensor that already carries
            # gradient.  The previous implementation did
            # ``L[..., tril] = tril_values`` and then
            # ``L[..., diag, diag] = softplus(L[..., diag, diag]) + min_std``,
            # which mutates ``L`` in-place after it has been registered into
            # the computation graph by the first scatter; this raises
            # ``RuntimeError: a leaf Variable that requires grad is being used
            # in an in-place operation`` during backward for full-covariance
            # Mixture Density Networks.  The replacement performs the scatter
            # exactly once on a fresh non-graded leaf (which is safe) and
            # then constructs the diagonal transformation purely functionally,
            # combining via ``torch.where`` so no buffer is mutated after the
            # scatter.  Using ``__setitem__`` here is intentional: the
            # advanced-indexing semantics for ``L[..., i, j] = tril_values``
            # differ from ``L.index_put((i, j), tril_values)`` (the latter
            # interprets ``i`` and ``j`` as two separate dim replacements,
            # which breaks the broadcast).  ``L_offdiag`` is the only tensor
            # we ever write to in-place; the diagonal and final ``L_matrices``
            # are produced functionally.
            L_offdiag = torch.zeros(
                *batch_shape,
                self.n_components,
                self.n_features,
                self.n_features,
                device=y_pred.device,
                dtype=tril_values.dtype,
            )
            i_idx = tril_indices[0].to(y_pred.device)
            j_idx = tril_indices[1].to(y_pred.device)
            L_offdiag[..., i_idx, j_idx] = tril_values  # registers grad via tril_values

            # Compute the diagonal transform purely functionally from the raw
            # diagonal entries of ``tril_values``.  ``diag_positions[k]`` is
            # the location within the tril-flat array that maps to ``(k, k)``.
            diag_in_tril = (i_idx == j_idx).nonzero(as_tuple=True)[0]
            raw_diag = tril_values.index_select(-1, diag_in_tril)  # (*batch, C, F)
            softplus_diag = F.softplus(raw_diag) + self.min_std  # (*batch, C, F)
            # Coverage invariants (TOR003): chain .to() on torch.diag_embed
            # because torch.diag_embed does not accept device=/dtype= kwargs natively.
            L_diag = torch.diag_embed(softplus_diag, dim1=-2, dim2=-1).to(
                device=softplus_diag.device, dtype=softplus_diag.dtype
            )

            diag_mask = torch.eye(
                self.n_features,
                dtype=torch.bool,
                device=y_pred.device,
            ).expand_as(L_offdiag)
            L_matrices = torch.where(diag_mask, L_diag, L_offdiag)

            return log_weights, means, L_matrices

    @property
    def df(self) -> float | torch.Tensor:
        """Component degrees of freedom (``math.inf``: Gaussian; a tensor when learned)."""
        if self.learn_df:
            return self._df * torch.exp(self.raw_df)
        return self._df

    @property
    def gaussian_components(self) -> bool:
        """Whether the components are Gaussian (``df = inf``, not learned)."""
        return not self.learn_df and math.isinf(self._df)

    def _mixture(self, params: TensorTriplet) -> StudentTMixture:
        log_weights, means, factor = params
        df = self.df
        if isinstance(df, torch.Tensor):
            df = df.to(device=means.device, dtype=means.dtype)
        if self.covariance_type == "diagonal":
            return StudentTMixture(log_weights, means, scale_diag=factor, df=df)
        if self.full_parameterization == "precision":
            return StudentTMixture(log_weights, means, precision_tril=factor, df=df)
        return StudentTMixture(log_weights, means, scale_tril=factor, df=df)

    def distribution(self, y_pred: torch.Tensor) -> StudentTMixture:
        """The predicted mixture as a :class:`~torchregress.losses.StudentTMixture`.

        Gives exact log densities of partially observed targets, marginal CDFs and
        quantiles, conditioning and linear-observation likelihoods for the head.
        """
        return self._mixture(self._extract_distribution_parameters(y_pred))

    def _calculate_nll(
        self,
        target: torch.Tensor,
        params: TensorTriplet,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Calculate negative log likelihood for mixture model.

        Args:
            target (torch.Tensor): Target values [..., n_features]
            params (tuple): Tuple of (log_weights, means, stds_or_L)
            mask (torch.Tensor, optional): Boolean feature mask ``[..., n_features]``
                (``True`` = observed); the exact marginal density of the observed
                features is used.

        Returns:
            torch.Tensor: Negative log likelihood values
        """
        return -self._mixture(params).log_prob(target, mask=mask)

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Calculate mixture density negative log-likelihood loss.

        Args:
            y_pred (torch.Tensor): Model predictions (format depends on covariance_type)
            target (torch.Tensor): Ground truth values [..., n_features]
            mask (torch.Tensor, optional): Boolean validity mask, per feature
                ``[..., n_features]`` or per sample ``[...]``. With a feature mask a
                partially labelled row contributes the exact marginal NLL of its
                observed features (the mixture's marginal over the observed
                coordinates, no imputation); rows with no observed feature are
                dropped. Masked target entries may hold any value (e.g. NaN).
            weights (torch.Tensor, optional): Optional sample weights ``[...]``, or
                per-feature weights ``[..., n_features]`` (averaged over each row's
                observed features).

        Returns:
            torch.Tensor: Negative log-likelihood loss

        Raises:
            ValueError: If target shape doesn't match expected features
        """
        # Basic input validation
        if target.shape[-1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features in target, got {target.shape[-1]}"
            )

        # Extract distribution parameters
        params = self._extract_distribution_parameters(y_pred)

        feature_mask = sample_mask = None
        if mask is not None:
            mask = mask.to(device=target.device, dtype=torch.bool)
            if mask.shape == target.shape:
                feature_mask = mask
                sample_mask = mask.any(dim=-1)
                if weights is not None and weights.dim() == target.dim():
                    w = torch.where(mask, weights, torch.zeros_like(weights))
                    weights = w.sum(-1) / mask.sum(-1).clamp_min(1)
            else:
                sample_mask = mask
        if feature_mask is not None and bool(feature_mask.all()):
            feature_mask = None  # fully observed: the cheaper unmasked density
        if feature_mask is None and sample_mask is not None:
            # Rows excluded by a sample mask may hold non-finite targets.
            target = torch.where(sample_mask.unsqueeze(-1), target, torch.zeros_like(target))

        # Per-sample NLL (exact marginal over the observed features)
        nll = self._calculate_nll(target, params, feature_mask)

        # A9: single mask/weight/reduction path shared with every other loss
        return self._reduce(nll, sample_mask, weights)

    def predict_mean_std(self, y_pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the mean and standard deviation of the mixture distribution.

        Note: This uses the Gaussian approximation which may not be accurate
        for multimodal mixtures. For proper prediction intervals, use
        `predict_interval()` instead.

        Args:
            y_pred: Model predictions [batch, output_size]

        Returns:
            tuple: (mean, std) each of shape [batch, n_features]
        """
        mixture = self.distribution(y_pred)
        # Law of total variance; Student-t components contribute df/(df-2) Sigma_kk
        # (infinite for df <= 2).
        mixture_std = torch.sqrt(mixture.variance.clamp(min=self.eps))
        return mixture.mean, mixture_std

    def predict_interval(
        self,
        y_pred: torch.Tensor,
        confidence: float = 0.95,
        n_samples: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Central prediction interval of each target from the exact mixture marginal.

        The bounds are the ``(1 - confidence)/2`` and ``(1 + confidence)/2``
        quantiles of each feature's marginal mixture CDF
        (:meth:`StudentTMixture.marginal_quantile`: exact, deterministic, valid for
        multimodal mixtures and heavy tails), not Monte Carlo estimates.

        Args:
            y_pred: Model predictions [batch, output_size]
            confidence: Confidence level in (0, 1) (default 0.95 for 95% CI)
            n_samples: Ignored (the quantiles are exact); kept for compatibility
                with the earlier Monte Carlo implementation.

        Returns:
            tuple: (lower, upper) each of shape [batch, n_features]

        Example:
            >>> loss_fn = MixtureDensityLoss(n_components=3, n_features=1)
            >>> y_pred = model(x_test)
            >>> lower, upper = loss_fn.predict_interval(y_pred, confidence=0.95)
            >>> in_interval = (y_test >= lower) & (y_test <= upper)
            >>> coverage = in_interval.float().mean()  # Should be ~0.95
        """
        if not 0.0 < confidence < 1.0:
            raise ValueError(f"confidence must be in (0, 1), got {confidence}")
        alpha = 1.0 - confidence
        bounds = self.distribution(y_pred).marginal_quantile([alpha / 2, 1.0 - alpha / 2])
        return bounds[..., 0], bounds[..., 1]

    def log_prob(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-sample log mixture density ``[B]``."""
        params = self._extract_distribution_parameters(y_pred)
        nll = self._calculate_nll(target, params)
        return -nll.reshape(-1)

    def sample(
        self,
        y_pred: torch.Tensor,
        n_samples: int = 100,
    ) -> torch.Tensor:
        """
        Sample from the mixture distribution.

        Args:
            y_pred: Model predictions [batch, output_size]
            n_samples: Number of samples to generate

        Returns:
            samples: [n_samples, batch, n_features]
        """
        return self.distribution(y_pred).sample((n_samples,))
