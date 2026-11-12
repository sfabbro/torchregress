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
    out = solve_normal_equations(normal, rhs, nonnegative=nonnegative, start=start, rank_tol=1e-8)
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


def test_default_takes_the_ridge_only_on_failure() -> None:
    generator = torch.Generator().manual_seed(0)
    design = torch.randn(64, 30, 3, generator=generator, dtype=torch.float64)
    design[..., 2] = design[..., 0] + design[..., 1]
    normal = design.transpose(1, 2) @ design
    rhs = (normal @ torch.ones(64, 3, 1, dtype=torch.float64)).squeeze(-1)
    root = torch.diagonal(normal, dim1=-2, dim2=-1).sqrt()
    failed = torch.linalg.cholesky_ex(normal / (root[:, :, None] * root[:, None, :]))[1] != 0
    assert failed.any() and not failed.all()
    assert torch.equal(solve_normal_equations(normal, rhs).ridge_used, failed)
    assert torch.equal(solve_normal_equations(normal, rhs, rank_tol=0.0).ridge_used, failed)
    for bad in (-1.0, math.nan):
        with pytest.raises(ValueError, match="rank_tol"):
            solve_normal_equations(normal, rhs, rank_tol=bad)


def _near_collinear_pair(eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Columns ``a``, ``a(1 + eps t)`` and a constant: ill-conditioned, identified."""
    t = torch.linspace(-1.0, 1.0, 200, dtype=torch.float64)
    a = torch.exp(-0.5 * (t / 0.3) ** 2)
    return torch.stack([a, a * (1.0 + eps * t), torch.ones_like(t)], -1), t


def test_default_does_not_bias_an_identified_ill_conditioned_row() -> None:
    # Scaled smallest eigenvalue ~2e-10: badly conditioned but identified in
    # float64 (least squares agrees to ~1e-7 standard errors). A pivot-ratio
    # rank test at 1e-8 would ridge it, shift the solution by ~0.1 standard
    # errors and shrink the errors from ~6.5e3 to ~1e3.
    design, t = _near_collinear_pair(1e-4)
    generator = torch.Generator().manual_seed(0)
    y = design.sum(-1) + 0.01 * torch.randn(t.shape, generator=generator, dtype=torch.float64)
    normal, rhs = (design.T @ design)[None], (design.T @ y)[None]
    out = solve_normal_equations(normal, rhs)
    assert not out.ridge_used.any()
    reference = torch.linalg.lstsq(design, y[:, None], driver="gelsd").solution.squeeze(-1)
    assert ((out.solution[0] - reference).abs() / out.standard_errors[0]).max() < 1e-5
    se = torch.diagonal(torch.linalg.inv(normal[0])).sqrt()
    torch.testing.assert_close(out.standard_errors[0], se, rtol=1e-4, atol=0.0)


def test_default_nonnegative_solution_does_not_depend_on_the_warm_start() -> None:
    # The data push the second of a near-collinear pair negative, so it sits
    # at its bound. A rank test on the intermediate active-set free sets would
    # ridge {a, b, c}; its near-null direction then looks feasible and a warm
    # start from (1, 1, 1) would stop at a ridged, non-minimising point.
    design, t = _near_collinear_pair(1e-6)
    generator = torch.Generator().manual_seed(0)
    noise = 0.01 * torch.randn(t.shape, generator=generator, dtype=torch.float64)
    y = 2.0 * design[:, 0] + 1.0 + noise - 5e-6 * t * design[:, 0]
    normal, rhs = (design.T @ design)[None], (design.T @ y)[None]
    mask = torch.tensor([True, True, False])
    cold = solve_normal_equations(normal, rhs, nonnegative=mask)
    warm = solve_normal_equations(normal, rhs, nonnegative=mask, start=torch.ones_like(rhs))
    for out in (cold, warm):
        assert out.free[0].tolist() == [True, False, True]
        assert not out.ridge_used.any()
        assert out.standard_errors[0].max() < 1.0
    torch.testing.assert_close(warm.solution, cold.solution, rtol=1e-10, atol=1e-12)

    def objective(x: torch.Tensor) -> float:
        return float(0.5 * x @ normal[0] @ x - rhs[0] @ x)

    assert objective(warm.solution[0]) <= objective(cold.solution[0]) + 1e-12


def _product_design(batch: int) -> torch.Tensor:
    """uspm-style ``basis × Legendre`` design: 15 columns, conditioning spread over decades."""
    generator = torch.Generator().manual_seed(2)
    grid = torch.linspace(-1.0, 1.0, 400, dtype=torch.float64)
    centres = torch.rand(batch, 1, 5, generator=generator, dtype=torch.float64) * 2.0 - 1.0
    basis = 1.0 + torch.exp(-0.5 * ((grid[None, :, None] - centres) / 0.15) ** 2)
    legendre = torch.stack([torch.special.legendre_polynomial_p(grid, m) for m in range(3)], -1)
    return (basis.unsqueeze(-1) * legendre[None, :, None, :]).reshape(batch, 400, 15)


@pytest.mark.parametrize("rank_tol", [1e-12, 1e-10])
def test_rank_tol_ridges_exactly_the_rows_below_the_eigenvalue_ratio(rank_tol: float) -> None:
    # The Cholesky pivot ratio misclassifies 10-27 of these rows (it can exceed
    # lambda_min by 8 decades); the eigenvalue test must be exact and monotone.
    design = _product_design(100)
    normal = design.transpose(1, 2) @ design
    rhs = normal.sum(-1)
    unit = design / design.norm(dim=1, keepdim=True)  # Jacobi scaling of the design
    singular_values = torch.linalg.svdvals(unit)
    ratio = (singular_values[:, -1] / singular_values[:, 0]).square()
    clear = (ratio.log10() - math.log10(rank_tol)).abs() > 0.5
    expected = ratio <= rank_tol
    assert expected[clear].any() and (~expected[clear]).any()
    out = solve_normal_equations(normal, rhs, rank_tol=rank_tol)
    assert torch.equal(out.ridge_used[clear], expected[clear])
    looser = solve_normal_equations(normal, rhs, rank_tol=100.0 * rank_tol)
    assert not (out.ridge_used & ~looser.ridge_used).any()


def test_rank_tol_tests_only_the_final_free_set() -> None:
    # The final free set {a, c} is well conditioned: an opt-in rank test must
    # leave the row unridged from any start (cf. the default-rule test above).
    design, t = _near_collinear_pair(1e-6)
    generator = torch.Generator().manual_seed(0)
    noise = 0.01 * torch.randn(t.shape, generator=generator, dtype=torch.float64)
    y = 2.0 * design[:, 0] + 1.0 + noise - 5e-6 * t * design[:, 0]
    normal, rhs = (design.T @ design)[None], (design.T @ y)[None]
    mask = torch.tensor([True, True, False])
    cold = solve_normal_equations(normal, rhs, nonnegative=mask)
    for start in (None, torch.ones_like(rhs)):
        out = solve_normal_equations(normal, rhs, nonnegative=mask, start=start, rank_tol=1e-8)
        assert out.free[0].tolist() == [True, False, True] and not out.ridge_used.any()
        torch.testing.assert_close(out.solution, cold.solution, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(out.standard_errors, cold.standard_errors, rtol=1e-10, atol=0)


def test_rank_tol_nonnegative_ridged_solution_does_not_depend_on_the_start() -> None:
    # An exactly duplicated column: the minimisers form a segment. Starts with
    # both copies positive reach the rank-deficient free set {a, b, c}, which
    # the rank test flags; the ridged problem has a unique minimiser, reached
    # from every such start. (Starts with one copy at 0 stop at a vertex of
    # the segment, a full-rank free set: also a minimiser.)
    design, t = _near_collinear_pair(0.0)
    generator = torch.Generator().manual_seed(1)
    y = (
        1.5 * design[:, 0]
        + 1.0
        + 0.01 * torch.randn(t.shape, generator=generator, dtype=torch.float64)
    )
    normal, rhs = (design.T @ design)[None], (design.T @ y)[None]
    mask = torch.tensor([True, True, False])
    starts = torch.tensor([[1.0, 1.0, 1.0], [1.4, 0.1, 1.0], [0.05, 3.0, 0.0]], dtype=torch.float64)
    outs = [
        solve_normal_equations(normal, rhs, nonnegative=mask, start=start[None], rank_tol=1e-8)
        for start in starts
    ]
    cold = solve_normal_equations(normal, rhs, nonnegative=mask)

    def objective(x: torch.Tensor) -> float:
        return float(0.5 * x @ normal[0] @ x - rhs[0] @ x)

    for out in outs:
        assert abs(objective(out.solution[0]) - objective(cold.solution[0])) < 1e-8
        assert out.ridge_used.all() and not out.singular.any()
        assert out.free[0].tolist() == [True, True, True]
        # Symmetric pair: the ridged minimiser splits the coefficient equally.
        torch.testing.assert_close(out.solution[0, 0], out.solution[0, 1], rtol=1e-6, atol=0)
        torch.testing.assert_close(out.solution, outs[0].solution, rtol=1e-6, atol=1e-9)
        torch.testing.assert_close(out.standard_errors, outs[0].standard_errors, rtol=1e-6, atol=0)


def test_nonnegative_steps_work_on_the_rows_still_in_play(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torchregress.algorithms.normal_equations as module

    _, _, _, (normal, rhs) = _least_squares_batch(16, 30, 6, seed=4)
    mask = torch.ones(6, dtype=torch.bool)
    reference = solve_normal_equations(normal, rhs, nonnegative=mask)
    sizes: list[int] = []
    factor = module._free_block_cholesky

    def recording(scaled, free, ridge):  # type: ignore[no-untyped-def]
        sizes.append(scaled.shape[0])
        return factor(scaled, free, ridge)

    monkeypatch.setattr(module, "_free_block_cholesky", recording)
    out = solve_normal_equations(normal, rhs, nonnegative=mask)
    for name in ("solution", "standard_errors", "free", "ridge_used", "singular"):
        assert torch.equal(getattr(out, name), getattr(reference, name)), name
    # Rows finish at different steps: later steps factorise fewer rows, and the
    # standard errors reuse each row's last factor (no final factorisation of
    # every row).
    assert sizes[0] == 16 and min(sizes) < 16
    assert sizes.count(16) == 1


def _ridge_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Rows of which some fail the plain factorisation (an exactly collinear column)."""
    generator = torch.Generator().manual_seed(0)
    design = torch.randn(64, 30, 4, generator=generator, dtype=torch.float64)
    design[..., 2] = design[..., 0] + design[..., 1]
    design[..., 3] = design[..., 3].abs()
    normal = design.transpose(1, 2) @ design
    rhs = (design * torch.randn(64, 30, 1, generator=generator, dtype=torch.float64)).sum(1)
    return normal, rhs


_FIELDS = ("solution", "standard_errors", "free", "dead", "ridge_used", "singular")


def _assert_identical(a, b) -> None:  # type: ignore[no-untyped-def]
    """Field-by-field bit equality, NaN in the same places."""
    for name in _FIELDS:
        x, y = getattr(a, name), getattr(b, name)
        if x.is_floating_point():
            assert torch.equal(x.isnan(), y.isnan()), name
            x, y = x.nan_to_num(), y.nan_to_num()
        assert torch.equal(x, y), name


@pytest.mark.parametrize("nonnegative", [None, torch.tensor([True, True, True, False])])
@pytest.mark.parametrize("rank_tol", [0.0, 1e-8])
def test_ridging_only_the_failed_rows_matches_ridging_every_row(
    monkeypatch: pytest.MonkeyPatch, nonnegative: torch.Tensor | None, rank_tol: float
) -> None:
    import torchregress.algorithms.normal_equations as module

    normal, rhs = _ridge_batch()
    normal[5, 0, 0] = math.nan  # one non-finite row: the swap path runs too
    start = torch.ones_like(rhs)
    kwargs = {"nonnegative": nonnegative, "start": start, "rank_tol": rank_tol}
    out = solve_normal_equations(normal, rhs, **kwargs)
    assert out.ridge_used.any() and not out.ridge_used.all() and out.singular[5]
    ridge_failed_rows = module._ridge_failed_rows

    def every_row(system, chol, failed, ridge, ridged_variables, gather):  # type: ignore[no-untyped-def]
        return ridge_failed_rows(system, chol, failed, ridge, ridged_variables, False)

    # The accelerator policy: every ridged factorisation and guarded step runs
    # unconditionally, with no host read of the flags.
    monkeypatch.setattr(module, "_ridge_failed_rows", every_row)
    monkeypatch.setattr(module, "_host_any", lambda mask: True)
    _assert_identical(solve_normal_equations(normal, rhs, **kwargs), out)


@pytest.mark.parametrize("rank_tol", [0.0, 1e-8])
def test_standard_errors_reuse_equals_a_fresh_final_factorisation(
    monkeypatch: pytest.MonkeyPatch, rank_tol: float
) -> None:
    import torchregress.algorithms.normal_equations as module

    normal, rhs = _ridge_batch()
    _, _, _, (normal2, rhs2) = _least_squares_batch(64, 30, 4, seed=5)
    normal, rhs = torch.cat([normal, normal2]), torch.cat([rhs, rhs2])
    mask = torch.tensor([True, True, True, False])
    # From a positive start the collinear block is freed (and needs the ridge).
    kwargs = {"nonnegative": mask, "start": torch.ones_like(rhs), "rank_tol": rank_tol}
    out = solve_normal_equations(normal, rhs, **kwargs)
    assert out.ridge_used.any() and (out.free != mask).any()

    # Forcing every row's factor stale refactorises all final free blocks.
    active_set = module._active_set

    def stale(*args):  # type: ignore[no-untyped-def]
        *head, fresh = active_set(*args)
        return (*head, torch.zeros_like(fresh))

    monkeypatch.setattr(module, "_active_set", stale)
    _assert_identical(solve_normal_equations(normal, rhs, **kwargs), out)
    monkeypatch.undo()

    # And the standard errors are those of a fresh factorisation of the final
    # free block (of the ridged system on ridged rows).
    if rank_tol == 0.0:
        scaled, _, scale, _ = module._jacobi_scale(normal, rhs)
        chol, _, _ = module._free_block_cholesky(scaled, out.free, 1e-8)
        inv_diag = torch.diagonal(torch.cholesky_inverse(chol), dim1=-2, dim2=-1)
        se = torch.where(out.free, inv_diag.clamp_min(0.0).sqrt(), torch.ones_like(rhs))
        assert torch.equal(out.standard_errors, se / scale)
