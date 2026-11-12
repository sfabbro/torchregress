from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy import optimize

from torchregress.algorithms import solve_normal_equations


def _least_squares_batch(
    batch: int, rows: int, n: int, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    design = torch.randn(batch, rows, n, generator=generator, dtype=torch.float64)
    design = design * torch.logspace(-2, 2, n, dtype=torch.float64)  # badly scaled columns
    target = torch.randn(batch, rows, generator=generator, dtype=torch.float64)
    weights = 0.5 + torch.rand(batch, rows, generator=generator, dtype=torch.float64)
    xw = design * weights.unsqueeze(-1)
    normal = design.transpose(1, 2) @ xw
    rhs = (xw * target.unsqueeze(-1)).sum(1)
    return design, target, weights, (normal, rhs)  # type: ignore[return-value]


def test_unconstrained_matches_direct_solve_and_inverse() -> None:
    _, _, _, (normal, rhs) = _least_squares_batch(16, 40, 6)
    out = solve_normal_equations(normal, rhs)
    expected = torch.linalg.solve(normal, rhs.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(out.solution, expected, rtol=1e-10, atol=1e-12)
    se = torch.diagonal(torch.linalg.inv(normal), dim1=-2, dim2=-1).sqrt()
    torch.testing.assert_close(out.standard_errors, se, rtol=1e-10, atol=1e-14)
    assert out.free.all() and not out.dead.any()
    assert not out.ridge_used.any() and not out.singular.any()


def test_leading_batch_dimensions_are_preserved() -> None:
    _, _, _, (normal, rhs) = _least_squares_batch(12, 30, 4)
    flat = solve_normal_equations(normal, rhs)
    shaped = solve_normal_equations(normal.reshape(3, 4, 4, 4), rhs.reshape(3, 4, 4))
    torch.testing.assert_close(shaped.solution.reshape(12, 4), flat.solution)
    assert shaped.singular.shape == (3, 4)


def test_empty_columns_are_pinned_with_infinite_standard_error() -> None:
    _, _, _, (normal, rhs) = _least_squares_batch(3, 30, 5)
    normal = normal.clone()
    rhs = rhs.clone()
    normal[:, 2, :] = 0.0
    normal[:, :, 2] = 0.0
    rhs[:, 2] = 0.0
    out = solve_normal_equations(normal, rhs)
    keep = [0, 1, 3, 4]
    reduced = normal[:, keep][:, :, keep]
    expected = torch.linalg.solve(reduced, rhs[:, keep].unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(out.solution[:, keep], expected, rtol=1e-10, atol=1e-12)
    assert torch.equal(out.solution[:, 2], torch.zeros(3, dtype=torch.float64))
    assert torch.isinf(out.standard_errors[:, 2]).all()
    assert out.dead[:, 2].all() and not out.free[:, 2].any()


def test_ridge_fallback_and_singular_rows_fail_closed() -> None:
    # Row 0 is exactly rank deficient (duplicate column), row 1 is regular, row 2 is NaN.
    normal = torch.tensor(
        [
            [[4.0, 2.0, 4.0], [2.0, 3.0, 2.0], [4.0, 2.0, 4.0]],
            [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]],
            [[math.nan] * 3] * 3,
        ],
        dtype=torch.float64,
    )
    rhs = torch.tensor([[2.0, 1.0, 2.0], [1.0, -1.0, 0.5], [1.0, 1.0, 1.0]], dtype=torch.float64)
    for nonnegative in (None, torch.tensor([False, True, False])):
        out = solve_normal_equations(normal, rhs, nonnegative=nonnegative)
        assert out.ridge_used.tolist()[:2] == [True, False]
        assert torch.isfinite(out.solution[:2]).all()
        assert out.singular.tolist() == [False, False, True]
        assert torch.isnan(out.solution[2]).all() and torch.isnan(out.standard_errors[2]).all()
    # The ridge solution splits the duplicated direction evenly.
    out = solve_normal_equations(normal[:1], rhs[:1])
    torch.testing.assert_close(out.solution[0, 0], out.solution[0, 2])


@pytest.mark.parametrize("seed", range(4))
def test_nonnegative_matches_scipy_nnls(seed: int) -> None:
    design, target, weights, (normal, rhs) = _least_squares_batch(10, 25, 7, seed=seed)
    out = solve_normal_equations(normal, rhs, nonnegative=torch.ones(7, dtype=torch.bool))
    root = weights.sqrt()
    for row in range(design.shape[0]):
        a = (design[row] * root[row, :, None]).numpy()
        b = (target[row] * root[row]).numpy()
        expected, _ = optimize.nnls(a, b)
        np.testing.assert_allclose(out.solution[row].numpy(), expected, rtol=1e-8, atol=1e-10)
    assert (out.solution >= 0).all()
    assert torch.equal(out.free, out.solution > 0)


def test_partially_constrained_matches_bounded_lstsq_and_kkt() -> None:
    design, target, weights, (normal, rhs) = _least_squares_batch(8, 30, 5, seed=9)
    nonnegative = torch.tensor([True, True, True, False, False])
    out = solve_normal_equations(normal, rhs, nonnegative=nonnegative)
    root = weights.sqrt()
    lower = np.where(nonnegative.numpy(), 0.0, -np.inf)
    for row in range(design.shape[0]):
        a = (design[row] * root[row, :, None]).numpy()
        b = (target[row] * root[row]).numpy()
        ref = optimize.lsq_linear(a, b, bounds=(lower, np.inf), tol=1e-14, method="bvls")
        np.testing.assert_allclose(out.solution[row].numpy(), ref.x, rtol=1e-7, atol=1e-9)
    gradient = rhs - (normal @ out.solution.unsqueeze(-1)).squeeze(-1)
    scale = torch.diagonal(normal, dim1=-2, dim2=-1).sqrt()
    scaled_gradient = gradient / scale
    # KKT: zero gradient on free variables, non-positive at the bound.
    assert scaled_gradient[out.free].abs().max() < 1e-8
    assert (scaled_gradient[~out.free] < 1e-8).all()


def test_nonnegative_warm_start_reaches_same_solution() -> None:
    _, _, _, (normal, rhs) = _least_squares_batch(12, 25, 6, seed=2)
    mask = torch.ones(6, dtype=torch.bool)
    cold = solve_normal_equations(normal, rhs, nonnegative=mask)
    noisy = cold.solution + 0.3 * torch.randn(cold.solution.shape, dtype=torch.float64)
    noisy[0] = math.nan
    warm = solve_normal_equations(normal, rhs, nonnegative=mask, start=noisy)
    torch.testing.assert_close(warm.solution, cold.solution, rtol=1e-9, atol=1e-11)
    torch.testing.assert_close(warm.standard_errors, cold.standard_errors, rtol=1e-9, atol=1e-11)


def test_batched_rows_equal_single_rows() -> None:
    _, _, _, (normal, rhs) = _least_squares_batch(9, 25, 5, seed=5)
    mask = torch.tensor([True, True, False, True, False])
    batched = solve_normal_equations(normal, rhs, nonnegative=mask)
    for row in range(9):
        single = solve_normal_equations(normal[row : row + 1], rhs[row : row + 1], nonnegative=mask)
        torch.testing.assert_close(single.solution[0], batched.solution[row], rtol=1e-12, atol=0)
        assert torch.equal(single.free[0], batched.free[row])


def test_standard_errors_on_free_block_and_at_bound() -> None:
    _, _, _, (normal, rhs) = _least_squares_batch(6, 25, 5, seed=1)
    out = solve_normal_equations(normal, rhs, nonnegative=torch.ones(5, dtype=torch.bool))
    for row in range(6):
        free = out.free[row]
        block = normal[row][free][:, free]
        se = torch.diagonal(torch.linalg.inv(block)).sqrt()
        torch.testing.assert_close(out.standard_errors[row][free], se, rtol=1e-10, atol=0)
        bound = normal[row].diagonal()[~free].rsqrt()
        torch.testing.assert_close(out.standard_errors[row][~free], bound, rtol=1e-12, atol=0)


def test_validation_errors() -> None:
    normal = torch.eye(3, dtype=torch.float64).expand(2, 3, 3)
    rhs = torch.ones(2, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match="normal shape"):
        solve_normal_equations(normal[:, :2], rhs)
    with pytest.raises(ValueError, match="dtype"):
        solve_normal_equations(normal.float(), rhs)
    with pytest.raises(ValueError, match="ridge"):
        solve_normal_equations(normal, rhs, ridge=0.0)
    with pytest.raises(TypeError, match="boolean"):
        solve_normal_equations(normal, rhs, nonnegative=torch.ones(3))
    with pytest.raises(ValueError, match="start shape"):
        solve_normal_equations(
            normal, rhs, nonnegative=torch.ones(3, dtype=torch.bool), start=torch.ones(3)
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_matches_cpu() -> None:
    _, _, _, (normal, rhs) = _least_squares_batch(10, 25, 6, seed=3)
    mask = torch.ones(6, dtype=torch.bool)
    cpu = solve_normal_equations(normal, rhs, nonnegative=mask)
    gpu = solve_normal_equations(normal.cuda(), rhs.cuda(), nonnegative=mask.cuda())
    torch.testing.assert_close(gpu.solution.cpu(), cpu.solution, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize(
    "where, value",
    [
        ("rhs", math.nan),
        ("rhs", math.inf),
        ("rhs", -math.inf),
        ("offdiag", math.nan),
        ("diag", math.nan),
        ("diag", math.inf),
    ],
)
@pytest.mark.parametrize("bounded", [True, False])
def test_non_finite_row_fails_closed_without_touching_good_rows(
    where: str, value: float, bounded: bool
) -> None:
    _, _, _, (normal, rhs) = _least_squares_batch(5, 25, 4, seed=7)
    nonnegative = torch.ones(4, dtype=torch.bool) if bounded else None
    clean = solve_normal_equations(normal, rhs, nonnegative=nonnegative)
    normal, rhs = normal.clone(), rhs.clone()
    bad = 2
    if where == "rhs":
        rhs[bad, 1] = value
    elif where == "offdiag":
        normal[bad, 0, 3] = value
        normal[bad, 3, 0] = value
    else:
        normal[bad, 2, 2] = value
    out = solve_normal_equations(normal, rhs, nonnegative=nonnegative)
    assert out.singular.tolist() == [row == bad for row in range(5)]
    assert torch.isnan(out.solution[bad]).all() and torch.isnan(out.standard_errors[bad]).all()
    assert not out.free[bad].any() and not out.dead[bad].any()
    good = [row for row in range(5) if row != bad]
    for name in ("solution", "standard_errors", "free", "dead"):
        assert torch.equal(getattr(out, name)[good], getattr(clean, name)[good]), name


@pytest.mark.parametrize("bounded", [False, True])
def test_numerically_singular_factorisation_takes_the_ridge(bounded: bool) -> None:
    # An exactly collinear column: Cholesky often "succeeds" with a round-off
    # pivot, and the inverse of that factor is meaningless (errors ~ 1e7).
    generator = torch.Generator().manual_seed(0)
    design = torch.randn(64, 30, 3, generator=generator, dtype=torch.float64)
    design[..., 2] = design[..., 0] + design[..., 1]
    normal = design.transpose(1, 2) @ design
    rhs = (normal @ torch.ones(64, 3, 1, dtype=torch.float64)).squeeze(-1)  # x = 1 fits
    assert (torch.linalg.cholesky_ex(normal)[1] == 0).any()  # the trap is present
    nonnegative = torch.tensor([False, False, True]) if bounded else None
    # The warm start frees the collinear column (from 0 it would never be freed:
    # its multiplier vanishes once the other two fit exactly).
    start = torch.ones_like(rhs)
    out = solve_normal_equations(normal, rhs, nonnegative=nonnegative, start=start)
    rows = out.free.all(-1)  # the collinear block is in the factorisation
    assert rows.any()
    assert out.ridge_used[rows].all() and not out.singular.any()
    # Ridge-limited errors: about 1/sqrt(ridge) = 1e4 in scaled units, not 1e7+.
    scaled_se = out.standard_errors * torch.diagonal(normal, dim1=-2, dim2=-1).sqrt()
    assert scaled_se[rows].max() < 1e5


def test_well_conditioned_rows_do_not_take_the_ridge() -> None:
    _, _, _, (normal, rhs) = _least_squares_batch(32, 40, 8, seed=11)
    for nonnegative in (None, torch.ones(8, dtype=torch.bool)):
        assert not solve_normal_equations(normal, rhs, nonnegative=nonnegative).ridge_used.any()


def test_pivot_tol_zero_takes_the_ridge_only_on_failure() -> None:
    generator = torch.Generator().manual_seed(0)
    design = torch.randn(64, 30, 3, generator=generator, dtype=torch.float64)
    design[..., 2] = design[..., 0] + design[..., 1]
    normal = design.transpose(1, 2) @ design
    rhs = (normal @ torch.ones(64, 3, 1, dtype=torch.float64)).squeeze(-1)
    root = torch.diagonal(normal, dim1=-2, dim2=-1).sqrt()
    failed = torch.linalg.cholesky_ex(normal / (root[:, :, None] * root[:, None, :]))[1] != 0
    assert failed.any() and not failed.all()
    out = solve_normal_equations(normal, rhs, pivot_tol=0.0)
    assert torch.equal(out.ridge_used, failed)
    for bad in (-1.0, math.nan):
        with pytest.raises(ValueError, match="pivot_tol"):
            solve_normal_equations(normal, rhs, pivot_tol=bad)
