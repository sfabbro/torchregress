"""Analytic and independent numerical checks for scalar forward likelihoods."""

import math

import numpy as np
import pytest
import torch
from scipy.integrate import quad

from torchregress.losses.forward_quadrature import GaussianLatentForwardObservationNLLLoss


def test_linear_map_matches_analytic_gaussian_and_gradients():
    mean = torch.tensor([-0.3, 0.5, 1.0], dtype=torch.double, requires_grad=True)
    std = torch.tensor([0.2, 0.4, 0.1], dtype=torch.double, requires_grad=True)
    y = torch.tensor([-1.0, 0.2, 1.8], dtype=torch.double)
    error = torch.tensor([0.7, 0.8, 0.5], dtype=torch.double, requires_grad=True)
    slope = torch.tensor(1.3, dtype=torch.double, requires_grad=True)
    bias = torch.tensor(-0.2, dtype=torch.double, requires_grad=True)
    actual = GaussianLatentForwardObservationNLLLoss(64, "none")(
        mean, std, y, error, lambda z: slope * z + bias
    )
    expected = -torch.distributions.Normal(
        slope * mean + bias, torch.sqrt(error.square() + (slope * std).square())
    ).log_prob(y)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    params = (mean, std, error, slope, bias)
    g_actual = torch.autograd.grad(actual.sum(), params, retain_graph=True)
    g_expected = torch.autograd.grad(expected.sum(), params)
    for a, e in zip(g_actual, g_expected, strict=True):
        torch.testing.assert_close(a, e, atol=2e-11, rtol=2e-11)


def test_nonlinear_map_against_independent_adaptive_integration():
    mean, std, y, error = 0.2, 0.35, -0.15, 0.7
    value, _ = quad(
        lambda t: (
            np.exp(-0.5 * t * t)
            / np.sqrt(2 * np.pi)
            * np.exp(-0.5 * ((y - np.exp(0.4 * (mean + std * t))) / error) ** 2)
            / (error * np.sqrt(2 * np.pi))
        ),
        -12,
        12,
        epsabs=1e-13,
        epsrel=1e-13,
    )
    args = [torch.tensor([v], dtype=torch.double) for v in (mean, std, y, error)]
    actual = GaussianLatentForwardObservationNLLLoss(64)(*args, lambda z: torch.exp(0.4 * z))
    assert actual.item() == pytest.approx(-math.log(value), abs=1e-12)


def test_zero_latent_scatter_is_normalized_observation_likelihood():
    args = [torch.tensor([v], dtype=torch.double) for v in (0.7, 0, -0.2, 0.4)]
    actual = GaussianLatentForwardObservationNLLLoss(32)(*args, lambda z: z.square())
    expected = -torch.distributions.Normal(0.49, 0.4).log_prob(
        torch.tensor(-0.2, dtype=torch.double)
    )
    torch.testing.assert_close(actual, expected)


def test_mask_and_weights_select_before_map_and_ignore_missing_nans():
    nan = float("nan")
    mean = torch.tensor([0.2, nan, nan, 0.5], dtype=torch.double, requires_grad=True)
    std = torch.tensor([0.1, nan, nan, 0.2], dtype=torch.double)
    y = torch.tensor([-0.1, nan, nan, 0.6], dtype=torch.double)
    error = torch.tensor([0.8, nan, nan, 0.9], dtype=torch.double)
    mask = torch.tensor([True, False, True, True])
    weights = torch.tensor([1.0, 5.0, 0.0, 3.0], dtype=torch.double)

    def mapping(z):
        assert z.shape == (2, 32) and torch.isfinite(z).all()
        return z

    per = GaussianLatentForwardObservationNLLLoss(32, "none")(
        mean, std, y, error, mapping, mask, weights
    )
    assert per[1] == per[2] == 0
    actual = GaussianLatentForwardObservationNLLLoss()(mean, std, y, error, mapping, mask, weights)
    torch.testing.assert_close(actual, per.sum() / 4)
    actual.backward()
    assert torch.isfinite(mean.grad).all() and mean.grad[1] == mean.grad[2] == 0


@pytest.mark.parametrize("size", [0, 3])
def test_all_empty_has_differentiable_zero_and_never_calls_map(size):
    mean = torch.full((size,), float("nan"), requires_grad=True)
    std = torch.full((size,), float("nan"), requires_grad=True)

    def forbidden(_):
        raise AssertionError("Map must not run for empty observations")

    loss = GaussianLatentForwardObservationNLLLoss()(
        mean, std, mean, std, forbidden, mask=torch.zeros(size, dtype=torch.bool)
    )
    assert loss == 0
    loss.backward()
    assert torch.isfinite(mean.grad).all() and torch.isfinite(std.grad).all()


def test_autocast_and_half_residual_promote_before_arithmetic():
    args = [torch.tensor([v], dtype=torch.float16) for v in (0, 0, 60000, 1)]
    fn = GaussianLatentForwardObservationNLLLoss()

    def mapping(z):
        return z * 0 - 60000

    expected = fn(*args, mapping)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = fn(*args, mapping)
    assert actual.dtype == torch.float32 and torch.isfinite(actual)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.item() == pytest.approx(0.5 * 120000**2, rel=1e-6)


def test_nonlinear_gradcheck():
    args = [
        torch.tensor([v], dtype=torch.double, requires_grad=True) for v in (0.2, 0.3, -0.1, 0.8)
    ]
    fn = GaussianLatentForwardObservationNLLLoss(32)
    assert torch.autograd.gradcheck(lambda *a: fn(*a, lambda z: torch.exp(0.2 * z)), tuple(args))


@pytest.mark.parametrize("index,value", [(1, -1), (3, 0), (3, -1), (2, float("inf"))])
def test_invalid_active_values_rejected(index, value):
    args = [torch.tensor([v]) for v in (0.0, 0.2, -0.4, 0.5)]
    args[index] = torch.tensor([float(value)])
    with pytest.raises(ValueError):
        GaussianLatentForwardObservationNLLLoss()(*args, lambda z: z)


def test_invalid_map_and_constructor_rejected():
    args = [torch.tensor([v]) for v in (0.0, 0.2, -0.4, 0.5)]
    for mapping in (lambda z: z[:, 0], lambda z: z * float("nan")):
        with pytest.raises(ValueError, match="observation_map"):
            GaussianLatentForwardObservationNLLLoss()(*args, mapping)
    for points in (0, 257, True, 2.5):
        with pytest.raises(ValueError):
            GaussianLatentForwardObservationNLLLoss(points)
