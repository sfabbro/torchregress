"""Synthetic checks for a correlated spline flow with sparse targets."""

import math

import pytest
import torch

pytest.importorskip("zuko")

from torchregress.models.copula_spline import GaussianCopulaSplineHead


def test_identity_flow_exact_marginals_for_arbitrary_masks():
    torch.manual_seed(2)
    head = GaussianCopulaSplineHead(4, 6, width=8).double()
    x = torch.randn(5, 4, dtype=torch.double)
    y = torch.randn(5, 6, dtype=torch.double)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    y[~mask] = torch.nan
    got = head.intrinsic_log_prob(x, y, mask)
    safe = y.nan_to_num()
    _, lower, _ = head._distribution_parameters(x)
    sd = lower.diagonal(dim1=-2, dim2=-1)
    torch.testing.assert_close(sd, torch.ones_like(sd), atol=1e-7, rtol=0)
    expected = ((-0.5 * ((safe / sd).square() + math.log(2 * math.pi)) - sd.log()) * mask).sum(-1)
    torch.testing.assert_close(got, expected, atol=1e-12, rtol=1e-12)


def test_correlated_complete_and_partial_densities_match_gaussian_reference():
    head = GaussianCopulaSplineHead(2, 3, width=4).double()
    # Packed lower triangular order: (0,0),(1,0),(1,1),(2,0),(2,1),(2,2).
    with torch.no_grad():
        head.net[-1].bias[3 + 1] = 0.6
        head.net[-1].bias[3 + 3] = -0.25
    x = torch.zeros(3, 2, dtype=torch.double)
    y = torch.tensor([[0.4, -0.8, 0.2]] * 3, dtype=torch.double)
    mask = torch.tensor([[1, 1, 1], [1, 0, 1], [0, 1, 0]], dtype=torch.bool)
    got = head.intrinsic_log_prob(x, y, mask)
    mean, lower, _ = head._distribution_parameters(x)
    covariance = lower @ lower.transpose(-1, -2)
    ref = []
    for i in range(3):
        cols = mask[i].nonzero().flatten()
        law = torch.distributions.MultivariateNormal(
            mean[i, cols], covariance_matrix=covariance[i][cols][:, cols]
        )
        ref.append(law.log_prob(y[i, cols]))
    torch.testing.assert_close(got, torch.stack(ref), atol=1e-12, rtol=1e-12)


def test_quoted_noise_matches_independent_gaussian_convolution():
    head = GaussianCopulaSplineHead(2, 3, width=4).double()
    x = torch.zeros(1, 2, dtype=torch.double)
    y = torch.tensor([[0.0, 0.3, 0.0]], dtype=torch.double)
    mask = torch.tensor([[False, True, False]])
    sigma = torch.tensor([[float("nan"), 0.25, float("nan")]], dtype=torch.double)
    nodes = torch.quasirandom.SobolEngine(3, scramble=True, seed=9).draw(4096, dtype=torch.double)
    nodes = torch.special.ndtri(nodes)
    got, ess = head.observed_log_prob(x, y, sigma, mask, nodes)
    expected = torch.distributions.Normal(0.0, math.sqrt(1 + 0.25**2)).log_prob(y[0, 1])
    torch.testing.assert_close(got[0], expected, atol=2e-4, rtol=0)
    assert 0 < ess.item() <= 1
    zero, zero_ess = head.observed_log_prob(x, y, torch.zeros_like(sigma), mask, nodes[:16])
    torch.testing.assert_close(zero, head.intrinsic_log_prob(x, y, mask), atol=1e-12, rtol=0)
    torch.testing.assert_close(zero_ess, torch.ones_like(zero_ess), atol=1e-12, rtol=0)


def test_native_samples_are_coupled_and_gradients_reach_flow_parameters():
    head = GaussianCopulaSplineHead(2, 3, width=8).double()
    with torch.no_grad():
        head.net[-1].bias[3 + 1] = 0.7
        head.net[-1].bias[3 + 6 + 0] = 0.3  # first spline width logit
    x = torch.zeros(1, 2, dtype=torch.double)
    draws = head.sample(x, 12000, generator=torch.Generator().manual_seed(4))[:, 0]
    assert torch.corrcoef(draws.T)[0, 1] > 0.4
    y = torch.tensor([[0.3, -0.5, 0.8]], dtype=torch.double)
    loss = -head.intrinsic_log_prob(x, y, torch.ones_like(y, dtype=torch.bool)).mean()
    loss.backward()
    assert torch.isfinite(head.net[-1].bias.grad).all()
    assert head.net[-1].bias.grad.abs().sum() > 0


def test_nonlinear_partial_density_matches_numerical_missing_integral():
    head = GaussianCopulaSplineHead(2, 2, width=4, bins=4).double()
    with torch.no_grad():
        head.net[-1].bias[2 + 1] = 0.6  # nonzero Gaussian base correlation
        head.net[-1].bias[2 + 3 + 0] = 0.8  # nonlinear first-coordinate spline
        head.net[-1].bias[2 + 3 + 4] = -0.5
    observed = torch.tensor([[0.4, 0.0]], dtype=torch.double)
    partial = head.intrinsic_log_prob(
        torch.zeros(1, 2, dtype=torch.double), observed, torch.tensor([[True, False]])
    ).exp()[0]
    grid = torch.linspace(-12, 12, 4001, dtype=torch.double)
    values = torch.stack((torch.full_like(grid, 0.4), grid), dim=-1)
    joint = head.intrinsic_log_prob(
        torch.zeros(len(grid), 2, dtype=torch.double),
        values,
        torch.ones_like(values, dtype=torch.bool),
    ).exp()
    numerical = torch.trapezoid(joint, grid)
    torch.testing.assert_close(partial, numerical, atol=1e-6, rtol=1e-6)


def test_output_only_first_step_then_context_gradient():
    torch.manual_seed(19)
    head = GaussianCopulaSplineHead(3, 2, width=8).double()
    context = torch.randn(5, 3, dtype=torch.double, requires_grad=True)
    values = torch.randn(5, 2, dtype=torch.double)
    mask = torch.tensor([[1, 0], [0, 1], [1, 1], [1, 0], [0, 1]], dtype=torch.bool)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.01)
    loss = -head.intrinsic_log_prob(context, values, mask).mean()
    loss.backward()
    assert torch.count_nonzero(context.grad) == 0
    assert head.net[-1].weight.grad.abs().sum() > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    context.grad = None
    (-head.intrinsic_log_prob(context, values, mask).mean()).backward()
    assert torch.isfinite(context.grad).all() and context.grad.abs().sum() > 0


def test_partial_nonlinear_density_gradcheck_and_empty_rows():
    torch.manual_seed(37)
    head = GaussianCopulaSplineHead(2, 3, width=4, bins=4).double()
    with torch.no_grad():
        head.net[-1].weight.normal_(0, 0.04)
        head.net[-1].bias[3 + 1] = 0.4
    context = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
    values = torch.tensor([[0.2, -0.3, 0.7], [float("nan")] * 3], dtype=torch.double)
    mask = torch.tensor([[True, False, True], [False, False, False]])
    assert torch.autograd.gradcheck(
        lambda c: head.intrinsic_log_prob(c, values, mask), (context,), atol=2e-5, rtol=1e-4
    )
    assert head.intrinsic_log_prob(context, values, mask)[1] == 0
    noisy, ess = head.observed_log_prob(
        context,
        values,
        torch.full_like(values, float("nan")).masked_fill(mask, 0.0),
        mask,
        torch.zeros(8, 3, dtype=torch.double),
    )
    assert noisy[1] == 0 and ess[1] == 1


def test_invalid_masks_and_nonfinite_observations_fail_closed():
    head = GaussianCopulaSplineHead(2, 3, width=4).double()
    x = torch.zeros(1, 2, dtype=torch.double)
    y = torch.tensor([[float("nan"), 0.0, 0.0]], dtype=torch.double)
    with pytest.raises(ValueError, match="Observed values"):
        head.intrinsic_log_prob(x, y, torch.tensor([[True, False, False]]))
    with pytest.raises(ValueError, match="boolean"):
        head.intrinsic_log_prob(x, y.nan_to_num(), torch.ones_like(y))


def test_autocast_uses_at_least_float32_for_covariance_and_splines():
    head = GaussianCopulaSplineHead(2, 3, width=4)
    x = torch.zeros(2, 2, dtype=torch.bfloat16, requires_grad=True)
    y = torch.zeros(2, 3, dtype=torch.bfloat16)
    sigma = torch.full_like(y, 0.2)
    mask = torch.tensor([[True, False, True], [False, True, False]])
    nodes = torch.zeros(8, 3, dtype=torch.bfloat16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        score, ess = head.observed_log_prob(x, y, sigma, mask, nodes)
        draws = head.sample(x, 3)
    assert score.dtype == ess.dtype == draws.dtype == torch.float32
    assert torch.isfinite(score).all() and torch.isfinite(draws).all()
    (-score.mean()).backward()
    assert torch.isfinite(head.net[-1].weight.grad).all()
