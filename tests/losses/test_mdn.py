import math

import pytest
import torch
import torch.nn.functional as F

from torchregress.losses.mdn import MixtureDensityLoss


class TestMixtureDensityLoss:
    """Tests for MixtureDensityLoss implementation."""

    def test_initialization(self):
        """Test initialization with different parameters."""
        # Default initialization
        loss_fn = MixtureDensityLoss(n_components=2, n_features=3)
        assert loss_fn.n_components == 2
        assert loss_fn.n_features == 3
        assert loss_fn.covariance_type == "diagonal"
        assert loss_fn.reduction == "mean"
        assert loss_fn.expected_output_size == 14  # 2 + 2*2*3 = 14

        # Full covariance initialization
        loss_fn = MixtureDensityLoss(n_components=3, n_features=2, covariance_type="full")
        assert loss_fn.covariance_type == "full"
        assert loss_fn.expected_output_size == 18  # 3 + 3*2 + 3*(2*3/2) = 3 + 6 + 9 = 18

        # Custom parameters
        loss_fn = MixtureDensityLoss(
            n_components=2,
            n_features=3,
            covariance_type="diagonal",
            min_std=0.05,
            eps=1e-10,
            reduction="sum",
        )
        assert loss_fn.min_std == 0.05
        assert loss_fn.eps == 1e-10
        assert loss_fn.reduction == "sum"

        # Invalid covariance type
        with pytest.raises(ValueError, match="Unsupported covariance_type"):
            MixtureDensityLoss(n_components=2, n_features=3, covariance_type="invalid")

    def test_extract_distribution_parameters_diagonal(self):
        """Test parameter extraction for diagonal covariance."""
        n_components, n_features = 2, 3
        loss_fn = MixtureDensityLoss(n_components, n_features)

        # Create dummy prediction tensor
        batch_size = 5
        y_pred = torch.randn(batch_size, loss_fn.expected_output_size)

        log_weights, means, stds = loss_fn._extract_distribution_parameters(y_pred)

        # Check shapes
        assert log_weights.shape == (batch_size, n_components)
        assert means.shape == (batch_size, n_components, n_features)
        assert stds.shape == (batch_size, n_components, n_features)

        # Exponentiated log-weights should sum to 1
        assert torch.allclose(log_weights.exp().sum(dim=-1), torch.ones(batch_size))

        # Standard deviations should be positive and at least min_std
        assert (stds >= loss_fn.min_std).all()

        # Test with incorrect output size
        with pytest.raises(ValueError, match="Model output size .* doesn't match expected size"):
            loss_fn._extract_distribution_parameters(torch.randn(batch_size, 10))

    def test_extract_distribution_parameters_full(self):
        """Test parameter extraction for full covariance."""
        n_components, n_features = 2, 3
        loss_fn = MixtureDensityLoss(n_components, n_features, covariance_type="full")

        # Create dummy prediction tensor
        batch_size = 5
        y_pred = torch.randn(batch_size, loss_fn.expected_output_size)

        log_weights, means, L_matrices = loss_fn._extract_distribution_parameters(y_pred)

        # Check shapes
        assert log_weights.shape == (batch_size, n_components)
        assert means.shape == (batch_size, n_components, n_features)
        assert L_matrices.shape == (batch_size, n_components, n_features, n_features)

        # Exponentiated log-weights should sum to 1
        assert torch.allclose(log_weights.exp().sum(dim=-1), torch.ones(batch_size))

        # Check that matrices are lower triangular with positive diagonal
        for b in range(batch_size):
            for c in range(n_components):
                L = L_matrices[b, c]
                # Upper triangular part (excluding diagonal) should be zero
                assert torch.allclose(
                    torch.triu(L, diagonal=1), torch.zeros_like(torch.triu(L, diagonal=1))
                )
                # Diagonal elements should be positive
                assert (torch.diagonal(L) > 0).all()

    def test_log_prob_diagonal(self):
        """Component log densities for diagonal covariance (via the shared mixture)."""
        n_components, n_features = 2, 3
        loss_fn = MixtureDensityLoss(n_components, n_features)

        batch_size = 5
        y_pred = torch.randn(batch_size, loss_fn.expected_output_size)
        target = torch.randn(batch_size, n_features)
        log_weights, means, stds = loss_fn._extract_distribution_parameters(y_pred)

        log_probs = loss_fn.distribution(y_pred).component_log_prob(target)
        assert log_probs.shape == (batch_size, n_components)

        b, c = 0, 0
        norm_dist = (target[b] - means[b, c]) / stds[b, c]
        expected_log_prob = (
            -0.5 * torch.sum(norm_dist**2)
            - torch.sum(torch.log(stds[b, c]))
            - 0.5 * n_features * loss_fn.log_2pi
        )
        assert torch.isclose(log_probs[b, c], expected_log_prob)
        expected_nll = -torch.logsumexp(
            log_weights
            + torch.distributions.Normal(means, stds).log_prob(target.unsqueeze(-2)).sum(-1),
            dim=-1,
        )
        torch.testing.assert_close(
            loss_fn._calculate_nll(target, (log_weights, means, stds)), expected_nll
        )

    def test_log_prob_full(self):
        """Component log densities for a covariance-Cholesky full head."""
        n_components, n_features = 2, 2
        # Covariance-Cholesky formula (precision form: tests/audit/test_audit_c2.py).
        loss_fn = MixtureDensityLoss(
            n_components, n_features, covariance_type="full", full_parameterization="covariance"
        )

        batch_size = 3
        y_pred = torch.randn(batch_size, loss_fn.expected_output_size)
        target = torch.randn(batch_size, n_features)
        _, means, L_matrices = loss_fn._extract_distribution_parameters(y_pred)

        log_probs = loss_fn.distribution(y_pred).component_log_prob(target)
        assert log_probs.shape == (batch_size, n_components)

        expected = torch.distributions.MultivariateNormal(means, scale_tril=L_matrices).log_prob(
            target.unsqueeze(-2)
        )
        torch.testing.assert_close(log_probs, expected, rtol=1e-5, atol=1e-5)

    def test_forward_diagonal(self):
        """Test forward pass with diagonal covariance."""
        n_components, n_features = 2, 3
        batch_size = 5

        for reduction in ["none", "mean", "sum"]:
            loss_fn = MixtureDensityLoss(n_components, n_features, reduction=reduction)
            y_pred = torch.randn(batch_size, loss_fn.expected_output_size)
            target = torch.randn(batch_size, n_features)

            loss = loss_fn(y_pred, target)

            if reduction == "none":
                assert loss.shape == (batch_size,)
            else:
                assert loss.ndim == 0  # Scalar

    def test_forward_full(self):
        """Test forward pass with full covariance."""
        n_components, n_features = 2, 2  # Using n_features=2 for simplicity
        batch_size = 5

        for reduction in ["none", "mean", "sum"]:
            loss_fn = MixtureDensityLoss(
                n_components, n_features, covariance_type="full", reduction=reduction
            )
            y_pred = torch.randn(batch_size, loss_fn.expected_output_size)
            target = torch.randn(batch_size, n_features)

            loss = loss_fn(y_pred, target)

            if reduction == "none":
                assert loss.shape == (batch_size,)
            else:
                assert loss.ndim == 0  # Scalar

    def test_masked_loss(self):
        """Test loss calculation with masks."""
        n_components, n_features = 2, 3
        loss_fn = MixtureDensityLoss(n_components, n_features)

        batch_size = 6
        y_pred = torch.randn(batch_size, loss_fn.expected_output_size)
        target = torch.randn(batch_size, n_features)

        # Create a mask where half the samples are masked
        mask = torch.ones(batch_size, n_features, dtype=torch.bool)
        mask[: batch_size // 2] = False

        # Test with different reductions
        for reduction in ["none", "mean", "sum"]:
            loss_fn.reduction = reduction
            masked_loss = loss_fn(y_pred, target, mask=mask)

            # Calculate reference - only on unmasked samples
            reference_loss = loss_fn(y_pred[batch_size // 2 :], target[batch_size // 2 :])

            if reduction == "none":
                # First half should be zero, second half should match reference
                assert torch.allclose(masked_loss[: batch_size // 2], torch.zeros(batch_size // 2))
                assert torch.allclose(masked_loss[batch_size // 2 :], reference_loss)
            else:
                assert torch.isclose(masked_loss, reference_loss)

        # Test with sample-level mask
        sample_mask = torch.ones(batch_size, dtype=torch.bool)
        sample_mask[: batch_size // 2] = False

        loss_fn.reduction = "mean"
        sample_masked_loss = loss_fn(y_pred, target, mask=sample_mask)
        feature_masked_loss = loss_fn(y_pred, target, mask=mask)

        # Both should give the same result since MDN converts feature masks to sample masks
        assert torch.isclose(sample_masked_loss, feature_masked_loss)

    def test_weighted_loss(self):
        """Test loss calculation with weights."""
        n_components, n_features = 2, 3
        loss_fn = MixtureDensityLoss(n_components, n_features, reduction="sum")

        batch_size = 5
        y_pred = torch.randn(batch_size, loss_fn.expected_output_size)
        target = torch.randn(batch_size, n_features)

        # Create sample weights
        weights = torch.rand(batch_size)

        # Calculate losses
        loss_fn.reduction = "none"
        unweighted_losses = loss_fn(y_pred, target)

        loss_fn.reduction = "sum"
        weighted_loss = loss_fn(y_pred, target, weights=weights)

        # Manual weighting
        expected_weighted_loss = (unweighted_losses * weights).sum()
        assert torch.isclose(weighted_loss, expected_weighted_loss)

        # Test with feature-level weights
        feature_weights = torch.rand(batch_size, n_features)
        feature_weighted_loss = loss_fn(y_pred, target, weights=feature_weights)

        # Should use average of feature weights as sample weights
        expected_feature_weighted_loss = (unweighted_losses * feature_weights.mean(dim=-1)).sum()
        assert torch.isclose(feature_weighted_loss, expected_feature_weighted_loss)

    def test_input_validation(self):
        """Test input validation."""
        n_components, n_features = 2, 3
        loss_fn = MixtureDensityLoss(n_components, n_features)

        # Valid inputs
        y_pred = torch.randn(5, loss_fn.expected_output_size)
        target = torch.randn(5, n_features)
        loss_fn(y_pred, target)  # Should not raise

        # Invalid target shape
        invalid_target = torch.randn(5, n_features + 1)
        with pytest.raises(ValueError, match=f"Expected {n_features} features"):
            loss_fn(y_pred, invalid_target)

        # Invalid prediction shape
        invalid_pred = torch.randn(5, loss_fn.expected_output_size + 1)
        with pytest.raises(ValueError, match="Model output size .* doesn't match expected size"):
            loss_fn(invalid_pred, target)

    def test_mdn_edge_cases(self):
        """Test MixtureDensityLoss with edge cases: zeros, empty tensors, extreme values, NaN/Inf."""
        n_components = 3
        n_features = 1
        loss_fn = MixtureDensityLoss(n_components=n_components, n_features=n_features)

        # Test with zeros
        y_pred_zeros = torch.zeros(10, loss_fn.expected_output_size)
        y_true_zeros = torch.zeros(10, n_features)
        assert torch.isfinite(loss_fn(y_pred_zeros, y_true_zeros))

        # Test with empty tensors
        y_pred_empty = torch.tensor([]).reshape(0, loss_fn.expected_output_size)
        y_true_empty = torch.tensor([]).reshape(0, n_features)
        result = loss_fn(y_pred_empty, y_true_empty)
        assert result.numel() == 0 or result == 0 or torch.isnan(result)

        # Test with extreme values
        y_pred_large = torch.randn(1, loss_fn.expected_output_size)
        y_true_small = torch.zeros(1, n_features)
        assert torch.isfinite(loss_fn(y_pred_large, y_true_small))

        # Test with masks
        y_pred_masked = torch.randn(3, loss_fn.expected_output_size)
        y_true_masked = torch.tensor([[1.5], [2.5], [3.5]])
        mask = torch.tensor([[True], [True], [False]])
        assert torch.isfinite(loss_fn(y_pred_masked, y_true_masked, mask))


class TestMDNLossNumericalStability:
    def test_mdn_gradient_flow(self):
        """Test that gradients flow through MixtureDensityLoss properly."""

        # Create simple model and test gradient flow
        n_samples = 5
        n_components = 2
        n_features = 2

        loss_fn = MixtureDensityLoss(
            n_components=n_components, n_features=n_features, reduction="mean"
        )
        y_pred = torch.randn(n_samples, loss_fn.expected_output_size, requires_grad=True)
        target = torch.randn(n_samples, n_features)

        loss = loss_fn(y_pred, target)
        loss.backward()

        assert y_pred.grad is not None
        assert torch.all(torch.isfinite(y_pred.grad))

    def test_extreme_values(self):
        """Test stability with extreme values."""

        n_samples = 3
        n_components = 2
        n_features = 1

        mdn_loss = MixtureDensityLoss(
            n_components=n_components, n_features=n_features, reduction="mean"
        )

        # Test with normal values
        y_pred = torch.randn(n_samples, mdn_loss.expected_output_size, requires_grad=True)
        target = torch.randn(n_samples, n_features)

        loss = mdn_loss(y_pred, target)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.all(torch.isfinite(y_pred.grad))

    def test_nan_inf_handling(self):
        """Test how MDN loss handles NaN and Inf values with masks."""

        n_samples = 3
        n_components = 2
        n_features = 1

        mdn_loss = MixtureDensityLoss(
            n_components=n_components, n_features=n_features, reduction="mean"
        )

        # Create data
        y_pred = torch.randn(n_samples, mdn_loss.expected_output_size, requires_grad=True)
        target = torch.randn(n_samples, n_features)

        # Create mask to ignore second sample
        mask = torch.tensor([[True], [False], [True]])

        # This should only use the valid elements
        loss = mdn_loss(y_pred, target, mask)
        assert torch.isfinite(loss)
        loss.backward()

        # All gradients should be finite
        assert torch.all(torch.isfinite(y_pred.grad))


class TestMDNMixtureAggregationNumerics:
    """TR-COR-04: mixture aggregation must use log_softmax + logsumexp so
    extreme logits keep finite losses and nonzero gradients for every
    component, and the loss must equal a direct numerical mixture NLL."""

    def test_extreme_logits_finite_loss_and_full_gradient(self):
        """Extreme weight logits [-100, 0, 100] must yield a finite loss and
        nonzero gradient on ALL components' logits.

        d(loss)/d(logit_k) = w_k - r_k (weight minus posterior
        responsibility).  The pre-fix ``log(softmax + eps)`` truncated tail
        log-weights at the constant ``log(eps)``, killing their gradient.
        The geometry below keeps every responsibility r_k above 1e-12 so
        each logit receives a healthy gradient under the fixed code."""
        n_samples, n_components, n_features = 2, 3, 1
        loss_fn = MixtureDensityLoss(n_components=n_components, n_features=n_features)

        y_pred = torch.zeros(n_samples, loss_fn.expected_output_size)
        # Extreme weight logits: the -100/-100 tails carry ~0 probability mass.
        y_pred[:, :n_components] = torch.tensor([-100.0, 0.0, 100.0])
        # All sigmas at min_std (log_std -> -inf under softplus + min_std).
        y_pred[:, -n_components:] = -20.0
        target = torch.zeros(n_samples, n_features)
        # Means place every component's weighted log-density m_k = logw_k +
        # logp_k inside one ~27-nat window, so each responsibility
        # r_k = softmax(m)_k stays above 1e-12 and grad = w_k - r_k is
        # bounded away from zero for all three components.
        y_pred[:, n_components : 2 * n_components] = torch.tensor([0.0, 0.01389, 0.020041])
        y_pred.requires_grad_(True)

        loss = loss_fn(y_pred, target)
        assert torch.isfinite(loss)
        loss.backward()

        logits_grad = y_pred.grad[:, :n_components]
        assert torch.all(torch.isfinite(logits_grad))
        assert (logits_grad.abs() > 1e-12).all(), (
            f"gradient must reach ALL mixture components' logits, got {logits_grad}"
        )

    def test_matches_direct_numerical_mixture_nll(self):
        torch.manual_seed(7)
        n_components, n_features = 3, 2
        loss_fn = MixtureDensityLoss(n_components=n_components, n_features=n_features)

        y_pred = torch.randn(4, loss_fn.expected_output_size)
        target = torch.randn(4, n_features)

        loss = loss_fn(y_pred, target)

        # Direct numerical reference: build the same Gaussian components and
        # evaluate log(sum_k w_k * N(y | mu_k, sigma_k)) in probability space.
        weights = F.softmax(y_pred[:, :n_components], dim=-1)
        means = y_pred[:, n_components : n_components * 3].reshape(4, n_components, n_features)
        stds = (
            F.softplus(y_pred[:, n_components * 3 :].reshape(4, n_components, n_features))
            + loss_fn.min_std
        )

        diff = (target.unsqueeze(1) - means) / stds  # [4, K, D]
        comp = torch.exp(-0.5 * (diff**2).sum(-1)) / (
            (2 * math.pi) ** (n_features / 2) * stds.prod(-1)
        )  # [4, K]
        expected = -torch.log((weights * comp).sum(-1)).mean()

        assert torch.allclose(loss, expected, rtol=1e-5, atol=1e-6)


class TestStudentTComponents:
    """``df`` option: Student-t components through ``StudentTMixture`` (df=inf is Gaussian)."""

    @pytest.mark.parametrize(
        "cov_type,param",
        [("diagonal", "precision"), ("full", "precision"), ("full", "covariance")],
    )
    @pytest.mark.parametrize("df", [1.5, 6.0])
    def test_matches_scipy_multivariate_t(self, cov_type, param, df):
        from scipy import stats

        torch.manual_seed(3)
        loss_fn = MixtureDensityLoss(
            3, 2, covariance_type=cov_type, full_parameterization=param, df=df, reduction="none"
        )
        y_pred = torch.randn(5, loss_fn.expected_output_size, dtype=torch.float64)
        target = torch.randn(5, 2, dtype=torch.float64)
        logw, means, factor = loss_fn._extract_distribution_parameters(y_pred)
        if cov_type == "diagonal":
            cov = torch.diag_embed(factor.square()).to(device=factor.device, dtype=factor.dtype)
        elif param == "precision":
            cov = torch.linalg.inv(factor @ factor.mT)
        else:
            cov = factor @ factor.mT
        expected = []
        for b in range(5):
            dens = [
                stats.multivariate_t(means[b, k].numpy(), cov[b, k].numpy(), df=df).logpdf(
                    target[b].numpy()
                )
                for k in range(3)
            ]
            expected.append(-torch.logsumexp(logw[b] + torch.tensor(dens), dim=0))
        torch.testing.assert_close(
            loss_fn(y_pred, target), torch.stack(expected), rtol=1e-10, atol=1e-10
        )

    def test_large_df_approaches_gaussian_head(self):
        torch.manual_seed(4)
        gauss = MixtureDensityLoss(2, 3, covariance_type="full")
        y_pred = torch.randn(8, gauss.expected_output_size, dtype=torch.float64)
        target = torch.randn(8, 3, dtype=torch.float64)
        t_loss = MixtureDensityLoss(2, 3, covariance_type="full", df=1e9)
        torch.testing.assert_close(
            t_loss(y_pred, target), gauss(y_pred, target), rtol=1e-7, atol=1e-7
        )
        assert gauss.gaussian_components and not t_loss.gaussian_components

    def test_learned_df_receives_gradient(self):
        torch.manual_seed(5)
        loss_fn = MixtureDensityLoss(2, 2, df=4.0, learn_df=True)
        assert float(loss_fn.df) == pytest.approx(4.0)
        y_pred = torch.randn(16, loss_fn.expected_output_size, requires_grad=True)
        target = torch.randn(16, 2) * 3
        loss_fn(y_pred, target).backward()
        assert loss_fn.raw_df.grad is not None and torch.isfinite(loss_fn.raw_df.grad)
        assert torch.isfinite(y_pred.grad).all()
        assert any(p is loss_fn.raw_df for p in loss_fn.parameters())

    def test_learned_df_is_exact_and_log_scale(self):
        loss_fn = MixtureDensityLoss(2, 2, df=5.0, learn_df=True).double()
        assert float(loss_fn.df) == 5.0  # exact in float64 (raw_df starts at 0)
        assert loss_fn.df.dtype == torch.float64
        # One Adam step moves df by the same fraction at any df (log scale).
        for start in (3.0, 30.0):
            loss_fn = MixtureDensityLoss(1, 1, df=start, learn_df=True)
            opt = torch.optim.Adam(loss_fn.parameters(), lr=1e-2)
            y_pred = torch.zeros(64, loss_fn.expected_output_size)
            loss_fn(y_pred, torch.randn(64, 1) * 5).backward()  # heavy tails: df down
            opt.step()
            assert float(loss_fn.df) == pytest.approx(start * math.exp(-1e-2), rel=1e-5)

    def test_partially_observed_rows_use_the_exact_marginal(self):
        torch.manual_seed(8)
        for cov in ("diagonal", "full"):
            loss_fn = MixtureDensityLoss(3, 3, covariance_type=cov, df=4.0, reduction="none")
            y_pred = torch.randn(6, loss_fn.expected_output_size, dtype=torch.float64)
            target = torch.randn(6, 3, dtype=torch.float64)
            mask = torch.tensor(
                [[1, 1, 1], [1, 0, 1], [0, 1, 0], [0, 0, 0], [1, 1, 0], [0, 0, 1]],
                dtype=torch.bool,
            )
            target_nan = torch.where(mask, target, torch.full_like(target, math.nan))
            got = loss_fn(y_pred, target_nan, mask=mask)
            exact = -loss_fn.distribution(y_pred).log_prob(target, mask=mask)
            expected = torch.where(mask.any(-1), exact, torch.zeros_like(exact))
            torch.testing.assert_close(got, expected)
            assert torch.isfinite(got).all()
            # Rows with no observed feature are excluded from the mean.
            loss_fn.reduction = "mean"
            mean = loss_fn(y_pred, target_nan, mask=mask)
            torch.testing.assert_close(mean, exact[mask.any(-1)].mean())

    def test_invalid_df(self):
        with pytest.raises(ValueError, match="df"):
            MixtureDensityLoss(2, 2, df=0.0)
        with pytest.raises(ValueError, match="finite"):
            MixtureDensityLoss(2, 2, learn_df=True)

    def test_sampling_moments_and_distribution(self):
        torch.manual_seed(6)
        loss_fn = MixtureDensityLoss(2, 2, covariance_type="full", df=5.0)
        y_pred = torch.randn(3, loss_fn.expected_output_size, dtype=torch.float64)
        samples = loss_fn.sample(y_pred, n_samples=200_000)
        assert samples.shape == (200_000, 3, 2)
        mean, std = loss_fn.predict_mean_std(y_pred)
        torch.testing.assert_close(samples.mean(0), mean, rtol=0, atol=0.03)
        torch.testing.assert_close(samples.std(0), std, rtol=0.05, atol=0.0)
        mixture = loss_fn.distribution(y_pred)
        lower, upper = loss_fn.predict_interval(y_pred, confidence=0.9)
        exact = mixture.marginal_quantile([0.05, 0.95])
        torch.testing.assert_close(lower, exact[..., 0], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(upper, exact[..., 1], rtol=1e-12, atol=1e-12)
        # The exact bounds agree with the Monte Carlo quantiles of the samples.
        mc = torch.quantile(samples, torch.tensor([0.05, 0.95], dtype=samples.dtype), dim=0)
        torch.testing.assert_close(lower, mc[0], rtol=0.05, atol=0.02)
        torch.testing.assert_close(upper, mc[1], rtol=0.05, atol=0.02)


class TestMaxStd:
    """Upper bound on the spread of every component (``max_std``)."""

    FORMS = [("diagonal", "precision"), ("full", "precision"), ("full", "covariance")]

    @staticmethod
    def _loss(covariance: str, form: str, max_std, d: int = 3, k: int = 4) -> MixtureDensityLoss:
        return MixtureDensityLoss(
            n_components=k,
            n_features=d,
            covariance_type=covariance,
            full_parameterization=form,
            max_std=max_std,
        )

    @staticmethod
    def _largest(loss: MixtureDensityLoss, out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Largest standard deviation of any component, and largest root-mean variance."""
        _, _, factor = loss._extract_distribution_parameters(out)
        if loss.covariance_type == "diagonal":
            return factor.amax(), factor.square().mean(-1).sqrt().amax()
        if loss.full_parameterization == "precision":
            eye = torch.eye(factor.shape[-1], dtype=factor.dtype)
            factor = torch.linalg.solve_triangular(factor, eye.expand_as(factor), upper=False).mT
        cov = factor @ factor.mT
        top = torch.linalg.eigvalsh(cov)[..., -1].amax().sqrt()
        mean = (cov.diagonal(dim1=-2, dim2=-1).mean(-1)).amax().sqrt()
        return top, mean

    @pytest.mark.parametrize(("covariance", "form"), FORMS)
    def test_no_component_is_wider_than_the_bound(self, covariance: str, form: str) -> None:
        gen = torch.Generator().manual_seed(0)
        bounded = self._loss(covariance, form, 4.0)
        free = self._loss(covariance, form, None)
        # outputs far from zero: near-singular precisions, huge and tiny scales
        out = 6.0 * torch.randn(
            64, bounded.expected_output_size, generator=gen, dtype=torch.float64
        )
        _, free_mean = self._largest(free, out)
        top, mean = self._largest(bounded, out)
        assert float(free_mean) > 2 * 4.0  # the unbounded head does go beyond the bound
        assert float(mean) <= 4.0 * (1 + 1e-9)  # average variance over the features
        assert float(top) <= 4.0 * math.sqrt(3) * (1 + 1e-9)  # hence every direction
        if covariance == "diagonal":
            assert float(top) <= 4.0

    @pytest.mark.parametrize(("covariance", "form"), FORMS)
    def test_components_within_the_bound_are_untouched(self, covariance: str, form: str) -> None:
        gen = torch.Generator().manual_seed(1)
        bounded = self._loss(covariance, form, 50.0)
        free = self._loss(covariance, form, None)
        out = 0.3 * torch.randn(32, bounded.expected_output_size, generator=gen)
        y = torch.randn(32, 3, generator=gen)
        for a, b in zip(
            bounded._extract_distribution_parameters(out),
            free._extract_distribution_parameters(out),
            strict=True,
        ):
            assert torch.equal(a, b)
        assert torch.equal(bounded(out, y), free(out, y))

    @pytest.mark.parametrize("form", ["precision", "covariance"])
    def test_a_component_beyond_the_bound_keeps_its_shape(self, form: str) -> None:
        gen = torch.Generator().manual_seed(4)
        bounded = self._loss("full", form, 1.0, d=3, k=2)
        free = self._loss("full", form, None, d=3, k=2)
        out = torch.randn(6, bounded.expected_output_size, generator=gen, dtype=torch.float64)
        out[:, 2 + 2 * 3 :] *= 0.2 if form == "precision" else 5.0  # wide components
        _, _, f_b = bounded._extract_distribution_parameters(out)
        _, _, f_f = free._extract_distribution_parameters(out)
        ratio = f_b / f_f
        lower = torch.tril(torch.ones(3, 3, dtype=torch.bool))
        # one factor per component: the same matrix up to a scalar
        per_component = ratio[..., lower].reshape(6, 2, -1)
        spread = per_component.amax(-1) - per_component.amin(-1)
        assert float(spread.abs().amax()) < 1e-9
        assert bool((per_component[..., 0] != 1).any())  # and the bound was active

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("form", ["precision", "covariance"])
    def test_draws_stay_in_range_for_nearly_singular_factors(self, form: str, dtype) -> None:
        torch.manual_seed(2)
        bounded = self._loss("full", form, 3.0, d=2, k=3)
        free = self._loss("full", form, None, d=2, k=3)
        out = torch.zeros(4, bounded.expected_output_size, dtype=dtype)
        # one component with its precision diagonal at the min_std floor (1e-3), or with
        # a scale of softplus(40) = 40, and a huge off-diagonal entry
        tril_start = 3 + 3 * 2
        out[:, tril_start] = -40.0 if form == "precision" else 40.0
        out[:, tril_start + 1] = 1e4
        wild = free.distribution(out).sample((4000,))
        tame = bounded.distribution(out).sample((4000,))
        assert float(wild.abs().amax()) > (1e3 if form == "precision" else 100.0)
        assert float(tame.abs().amax()) < 3.0 * math.sqrt(2) * 8  # eight bounded deviations
        mix = bounded.distribution(out)
        draws = mix.sample((20000,))
        assert bool(torch.isfinite(mix.log_prob(draws)).all())
        cov = torch.cov(draws[:, 0].T.double())
        assert float(cov.diagonal().mean()) <= 3.0**2 * 1.1

    def test_gradients_are_finite_and_reach_a_bounded_component(self) -> None:
        gen = torch.Generator().manual_seed(3)
        for covariance, form in self.FORMS:
            loss = self._loss(covariance, form, 2.0)
            out = (5.0 * torch.randn(16, loss.expected_output_size, generator=gen)).requires_grad_()
            y = torch.randn(16, 3, generator=gen)
            loss(out, y).backward()
            assert bool(torch.isfinite(out.grad).all())
            if covariance == "full":
                assert float(out.grad[:, 4 + 4 * 3 :].abs().sum()) > 0

    def test_one_bound_per_feature_for_diagonal_components(self) -> None:
        loss = self._loss("diagonal", "precision", torch.tensor([1.0, 2.0, 30.0]))
        out = torch.full((5, loss.expected_output_size), 20.0)
        _, _, stds = loss._extract_distribution_parameters(out)
        torch.testing.assert_close(
            stds.amax(dim=(0, 1)), torch.tensor([1.0, 2.0, 20.0 + 1e-3]), rtol=1e-4, atol=1e-4
        )

    def test_rejects_bad_bounds(self) -> None:
        with pytest.raises(ValueError, match="max_std"):
            self._loss("diagonal", "precision", 0.0)
        with pytest.raises(ValueError, match="max_std"):
            self._loss("diagonal", "precision", float("inf"))
        with pytest.raises(ValueError, match="one max_std"):
            self._loss("full", "precision", torch.tensor([1.0, 2.0, 3.0]))

    @pytest.mark.parametrize("offdiag", [10.0, 100.0, 1e4])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_a_precision_factor_whose_inverse_overflows_is_still_bounded(
        self, dtype, offdiag: float
    ) -> None:
        # Eight features: with the diagonal at the floor and large off-diagonals the
        # inverse of the precision factor grows like (offdiag / 1e-3)^7 and overflows
        # float32, where an "is the norm finite" test would let the component through.
        torch.manual_seed(5)
        d, k = 8, 2
        bounded = self._loss("full", "precision", 3.0, d=d, k=k)
        out = torch.zeros(3, bounded.expected_output_size, dtype=dtype)
        i, j = torch.tril_indices(d, d)
        start = k + k * d
        out[:, start : start + len(i)] = torch.where(i == j, -50.0, offdiag).to(dtype)
        top, mean = self._largest(bounded, out.double())
        assert float(mean) <= 3.0 * (1 + 1e-6)
        draws = bounded.distribution(out).sample((2000,))
        assert bool(torch.isfinite(draws).all())
        assert float(draws.abs().amax()) < 3.0 * math.sqrt(d) * 8
        mix = bounded.distribution(out)
        assert bool(torch.isfinite(mix.log_prob(draws)).all())

    def test_objects_saved_before_the_bound_existed_still_work(self) -> None:
        loss = self._loss("full", "precision", None)
        del loss.__dict__["max_std"]  # what an older pickle restores
        out = torch.zeros(2, loss.expected_output_size)
        assert bool(torch.isfinite(loss(out, torch.zeros(2, 3))))
