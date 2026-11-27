"""The grouped inverse of autoregressive flows: the same map as zuko's sweeps, less work."""

from __future__ import annotations

import copy

import pytest
import torch
from zuko.flows import NSF
from zuko.transforms import AutoregressiveTransform

from torchregress.losses.nflows import (
    NormalizingFlowLoss,
    _conditioner_plan,
    _GroupedAutoregressiveTransform,
    create_flow_model,
)

CASES = [
    ("nsf", {}),
    ("nsf", {"bound": 9.0}),
    ("nsf", {"passes": 2}),
    ("nsf", {"randperm": True}),
    ("nsf", {"activation": torch.nn.ELU}),
    ("maf", {}),
    ("bpf", {"degree": 6}),
]


def _flow(flow_type: str, kwargs: dict, features: int, context: int, seed: int = 0):
    torch.manual_seed(seed)
    flow = create_flow_model(
        features, context, flow_type, n_transforms=3, hidden_features=[24, 24], **kwargs
    )
    with torch.no_grad():  # away from the near-identity initialisation
        for param in flow.parameters():
            param.add_(0.05 * torch.randn_like(param))
    return flow


def _inputs(features: int, context: int, n: int = 64, seed: int = 1):
    gen = torch.Generator().manual_seed(seed)
    c = torch.randn(n, context, generator=gen) if context else None
    return c, 1.5 * torch.randn(n, features, generator=gen)


def _layers(flow, c):
    transform = flow.transform(c) if c is not None else flow.transform()
    return transform, list(transform.transforms)


@pytest.mark.parametrize(("flow_type", "kwargs"), CASES)
@pytest.mark.parametrize(("features", "context"), [(2, 3), (5, 4), (12, 6), (3, 0)])
def test_grouped_inverse_is_the_sweep_inverse(flow_type, kwargs, features, context) -> None:
    flow = _flow(flow_type, kwargs, features, context)
    c, y = _inputs(features, context)
    with torch.no_grad():
        transform, layers = _layers(flow, c)
        assert len(layers) == 3
        for layer in layers:
            assert type(layer) is _GroupedAutoregressiveTransform
            assert _conditioner_plan(layer._layer, y.device) is not None  # the fast path runs
            grouped = layer._inverse(y)
            sweeps = AutoregressiveTransform._inverse(layer, y)  # zuko's reference
            torch.testing.assert_close(grouped, sweeps, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(layer(grouped), y, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(transform(transform.inv(y)), y, rtol=1e-3, atol=1e-3)


def test_samples_and_densities_equal_a_plain_zuko_flow_with_the_same_weights() -> None:
    features, context = 6, 5
    ours = _flow("nsf", {}, features, context)
    torch.manual_seed(0)
    plain = NSF(features, context, transforms=3, hidden_features=[24, 24])
    assert list(plain.state_dict()) == list(ours.state_dict())  # same keys, same layout
    plain.load_state_dict(ours.state_dict())
    c, y = _inputs(features, context)
    with torch.no_grad():
        torch.testing.assert_close(ours(c).log_prob(y), plain(c).log_prob(y))
        torch.manual_seed(7)
        fast = ours(c).sample((16,))
        torch.manual_seed(7)
        reference = plain(c).sample((16,))
    torch.testing.assert_close(fast, reference, rtol=1e-5, atol=1e-5)


def test_loss_sampling_matches_the_plain_flow_and_has_the_right_shape() -> None:
    features, context = 4, 3
    ours = _flow("nsf", {"bound": 7.0}, features, context)
    loss = NormalizingFlowLoss(flow=ours)
    c, _ = _inputs(features, context, n=10)
    torch.manual_seed(3)
    draws = loss.sample(c, n_samples=50)
    assert draws.shape == (10, 50, features) and bool(torch.isfinite(draws).all())
    torch.manual_seed(3)
    reference = ours(c).sample((50,)).transpose(0, 1)
    torch.testing.assert_close(draws, reference)


def test_pathwise_gradients_equal_those_of_the_sweep_inverse() -> None:
    flow = _flow("nsf", {}, 5, 4)
    c, y = _inputs(5, 4, n=16)
    layer = flow.transform(c).transforms[1]
    params = [p for p in flow.transform.transforms[1].parameters()]
    grouped = torch.autograd.grad(layer._inverse(y).square().sum(), params)
    sweeps = torch.autograd.grad(AutoregressiveTransform._inverse(layer, y).square().sum(), params)
    for fast, reference in zip(grouped, sweeps, strict=True):
        torch.testing.assert_close(fast, reference, rtol=1e-4, atol=1e-5)


def test_hidden_units_of_rank_k_ignore_coordinates_not_known_at_sweep_k() -> None:
    # The plan evaluates a unit once, in the sweep given by its rank: it must then be a
    # function of the context and of the coordinates of order below that rank only.
    features, context = 6, 3
    flow = _flow("nsf", {"randperm": True}, features, context)
    layer = flow.transform.transforms[0]
    plan = _conditioner_plan(layer, torch.device("cpu"))
    assert plan is not None
    linears, activations = plan["linears"], plan["activations"]
    x = torch.randn(features + context)

    def hidden(inputs: torch.Tensor, depth: int) -> torch.Tensor:
        out = inputs
        for linear, activation in zip(linears[: depth + 1], activations, strict=False):
            out = activation(linear(out))
        return out

    for depth, by_rank in enumerate(plan["hidden_units"]):
        jac = torch.autograd.functional.jacobian(lambda v, d=depth: hidden(v, d), x)
        seen = torch.zeros(jac.shape[0], dtype=torch.bool)
        for rank, units in enumerate(by_rank):
            unknown = (layer.order >= rank).nonzero().squeeze(-1)  # not solved before sweep
            assert bool((jac[units][:, unknown] == 0).all())
            seen[units] = True
        # units of a rank beyond the last sweep feed no output and are never evaluated
        late = (~seen).nonzero().squeeze(-1)
        last = linears[-1].mask[:, late]
        assert not bool(last.any()) or late.numel() == 0
    for rank, (group, rows) in enumerate(zip(plan["groups"], plan["rows"], strict=True)):
        assert bool((layer.order[group] == rank).all())
        assert rows.numel() == group.numel() * layer.total


def test_a_residual_conditioner_falls_back_to_the_sweep_inverse() -> None:
    flow = _flow("nsf", {"residual": True}, 5, 4)
    c, y = _inputs(5, 4)
    with torch.no_grad():
        layer = flow.transform(c).transforms[0]
        assert _conditioner_plan(layer._layer, y.device) is None
        torch.testing.assert_close(layer._inverse(y), AutoregressiveTransform._inverse(layer, y))
        torch.testing.assert_close(layer(layer._inverse(y)), y, rtol=1e-4, atol=1e-4)


def test_one_target_flows_are_left_to_zuko() -> None:
    flow = _flow("nsf", {}, 1, 3)
    c, y = _inputs(1, 3)
    with torch.no_grad():
        transform = flow.transform(c)
        assert not any(type(t) is _GroupedAutoregressiveTransform for t in transform.transforms)
        torch.testing.assert_close(transform(transform.inv(y)), y, rtol=1e-4, atol=1e-4)


def test_copies_and_dtype_changes_keep_the_same_map() -> None:
    flow = _flow("nsf", {}, 5, 4)
    c, y = _inputs(5, 4)
    with torch.no_grad():
        reference = flow.transform(c).inv(y)  # builds the cached plan
        clone = copy.deepcopy(flow)
        torch.testing.assert_close(clone.transform(c).inv(y), reference)
        double = copy.deepcopy(flow).double()
        out = double.transform(c.double()).inv(y.double())
        assert out.dtype == torch.float64
        torch.testing.assert_close(out.float(), reference, rtol=1e-4, atol=1e-4)
