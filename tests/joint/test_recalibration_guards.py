"""Scale recalibration on densities that gradients cannot see through."""

from __future__ import annotations

import torch

from torchregress.joint import (
    GaussianJoint,
    JointRegressor,
    QuantileMarginals,
    ScaleRecalibrator,
    SklarJoint,
)

F64 = torch.float64


def _quantile_predictive(n: int, spread: float) -> SklarJoint:
    """Independent targets whose marginals are 31-knot quantile grids of N(0, spread^2)."""
    levels = torch.arange(1, 32, dtype=F64) / 32
    grid = spread * torch.special.ndtri(levels)
    marginals = QuantileMarginals(levels, grid.expand(n, 3, 31).clone())
    scores = GaussianJoint(torch.zeros(n, 3, dtype=F64), torch.eye(3, dtype=F64).to(dtype=F64))
    return SklarJoint(marginals, scores)


def test_nll_objective_works_on_piecewise_constant_densities() -> None:
    """The density has zero slope inside each quantile bin, so a gradient method stalls."""
    torch.manual_seed(0)
    y_cal, y_test = torch.randn(1500, 3, dtype=F64), torch.randn(4000, 3, dtype=F64)
    narrow = _quantile_predictive(1500, spread=0.6)  # over-confident by 1 / 0.6
    recal = ScaleRecalibrator(objective="nll").fit(narrow, y_cal)
    assert recal.scale_ is not None
    # 4 pattern-search rounds end with a step of 0.025 in log scale; the quantile grid
    # itself is a coarse description of the Gaussian, hence the 10% tolerance.
    torch.testing.assert_close(
        recal.scale_, torch.full((3,), 1 / 0.6, dtype=F64), rtol=0.10, atol=0
    )
    base_test = _quantile_predictive(4000, spread=0.6)
    before = base_test.log_prob(y_test).mean()
    after = recal.transform(base_test).log_prob(y_test).mean()
    assert float(after) > float(before) + 0.3


def test_nll_objective_never_scores_worse_than_identity_in_sample() -> None:
    torch.manual_seed(1)
    y_cal = torch.randn(800, 3, dtype=F64)
    for spread in (0.7, 1.0, 1.5):
        predictive = _quantile_predictive(800, spread)
        recal = ScaleRecalibrator(objective="nll").fit(predictive, y_cal)
        before = predictive.log_prob(y_cal).mean()
        after = recal.transform(predictive).log_prob(y_cal).mean()
        assert float(after) >= float(before) - 1e-9, spread
    calibrated = ScaleRecalibrator(objective="nll").fit(_quantile_predictive(800, 1.0), y_cal)
    assert calibrated.scale_ is not None
    assert float(calibrated.scale_.log().abs().max()) < 0.1


def test_estimator_rejects_a_recalibration_that_does_not_help_held_out() -> None:
    gen = torch.Generator().manual_seed(2)
    x = torch.randn(1200, 3, generator=gen)
    y = torch.stack([x[:, 0], x[:, 1] * x[:, 2]], 1) + 0.3 * torch.randn(1200, 2, generator=gen)
    model = JointRegressor("gaussian", seed=0, max_epochs=40, patience=8, hidden=(32, 32)).fit(x, y)
    assert model.scale_ is not None
    x_te = torch.randn(1500, 3, generator=gen)
    y_te = torch.stack([x_te[:, 0], x_te[:, 1] * x_te[:, 2]], 1) + 0.3 * torch.randn(
        1500, 2, generator=gen
    )
    with_recal = float(model.predict(x_te).log_prob(y_te).mean())
    without = float(model._base_predictive(x_te).log_prob(y_te.to(model.device)).mean())
    # Either the recalibrator was accepted because it helped on held-out rows, or it was
    # dropped and the factors reset; in both cases the estimator is not made worse by more
    # than held-out noise (0.3 / sqrt(1500) is about 0.008 nats per unit of spread).
    if bool((model.scale_ == 1).all()):
        assert with_recal == without
    else:
        assert with_recal > without - 0.03
