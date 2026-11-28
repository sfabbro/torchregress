"""Private helpers shared by the modules of :mod:`torchregress.joint`."""

from __future__ import annotations

import copy
import math
import warnings
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

import numpy as np
import torch
from torch import Tensor, nn

from ..losses.known_covariance import known_covariance_gaussian_nll


def target_index(dims: Union[int, Sequence[int]], event_dim: int) -> list[int]:
    """Validate a target index (an int or distinct ints in ``[0, event_dim)``)."""
    out = [int(dims)] if isinstance(dims, int) else [int(i) for i in dims]
    if not out:
        raise ValueError("dims must name at least one target")
    if len(set(out)) != len(out):
        raise ValueError(f"dims must be distinct, got {out}")
    for i in out:
        if not 0 <= i < event_dim:
            raise ValueError(f"target index {i} is outside [0, {event_dim})")
    return out


def mlp(
    in_dim: int, hidden: Sequence[int], out_dim: int, *, zero_last: bool = False
) -> nn.Sequential:
    """SiLU multilayer perceptron; ``zero_last`` zeroes the output layer's weight and bias."""
    layers: list[nn.Module] = []
    width = in_dim
    for h in hidden:
        layers += [nn.Linear(width, int(h)), nn.SiLU()]
        width = int(h)
    last = nn.Linear(width, out_dim)
    if zero_last:
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


@dataclass
class TrainResult:
    """Outcome of :func:`train`."""

    best_epoch: int
    best: float
    initial: float
    history: list[float] = field(default_factory=list)
    n_steps: int = 0
    n_skipped: int = 0


def train(
    module: nn.Module,
    batch_loss: Callable[[Tensor], Tensor],
    n_train: int,
    val_loss: Callable[[], Tensor],
    *,
    max_epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    generator: torch.Generator,
    optimizer: Literal["adam", "adamw"] = "adamw",
    weight_decay: float = 0.0,
    clip_norm: Optional[float] = None,
    skip_nonfinite: bool = False,
    include_initial: bool = False,
    name: str = "model",
) -> TrainResult:
    """Mini-batch training with early stopping on a validation loss.

    ``batch_loss`` takes positions ``[0, n_train)`` of a batch (a random
    permutation drawn from ``generator`` each epoch) and returns a scalar loss.
    The best validation checkpoint is restored and the module is left in eval
    mode.  ``include_initial`` makes the weights before the first step a
    candidate (so the result is never worse than the initial model on the
    validation rows).  ``skip_nonfinite`` skips batches whose loss is not
    finite; a warning reports skipped batches, and an error is raised when
    every batch was skipped.
    """
    module.eval()
    best_epoch, best, initial = -1, math.inf, math.nan
    if include_initial:
        with torch.no_grad():
            best = initial = float(val_loss())
    best_state = copy.deepcopy(module.state_dict())
    result = TrainResult(best_epoch=best_epoch, best=best, initial=initial)
    if max_epochs <= 0:
        module.load_state_dict(best_state)
        module.eval()
        return result
    params = list(module.parameters())
    opt: torch.optim.Optimizer
    if optimizer == "adamw":
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    wait = 0
    for epoch in range(max_epochs):
        module.train()
        order = torch.randperm(n_train, generator=generator)
        for start in range(0, n_train, batch_size):
            loss = batch_loss(order[start : start + batch_size])
            if skip_nonfinite and not bool(torch.isfinite(loss)):
                result.n_skipped += 1
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if clip_norm is not None:
                nn.utils.clip_grad_norm_(params, clip_norm)
            opt.step()
            result.n_steps += 1
        module.eval()
        with torch.no_grad():
            current = float(val_loss())
        result.history.append(current)
        if current < best - 1e-6:
            best, best_epoch, wait = current, epoch, 0
            best_state = copy.deepcopy(module.state_dict())
        else:
            wait += 1
            if wait >= patience:
                break
    if result.n_skipped:
        if result.n_steps == 0:
            raise RuntimeError(
                f"training {name}: the loss was not finite on any of the "
                f"{result.n_skipped} batches, so no step was taken"
            )
        warnings.warn(
            f"training {name}: skipped {result.n_skipped} of "
            f"{result.n_skipped + result.n_steps} batches with a non-finite loss",
            RuntimeWarning,
            stacklevel=3,
        )
    module.load_state_dict(best_state)
    module.eval()
    result.best_epoch, result.best = best_epoch, best
    return result


def masked_gaussian_quadratic(
    resid: Tensor, covariance: Tensor, mask: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Quadratic form, log-determinant and dof of the observed sub-vector.

    For each row, with observed set :math:`O`,
    :math:`r_O^\top \Sigma_{OO}^{-1} r_O`, :math:`\log\lvert\Sigma_{OO}\rvert`
    and :math:`|O|`, from
    :func:`~torchregress.losses.known_covariance_gaussian_nll` (exact
    marginalisation of the masked targets).  ``resid`` is ``[n, d]`` or
    ``[n, S, d]`` (the ``S`` candidates of a row share its covariance and mask);
    ``covariance`` is ``[n, d, d]``; ``mask`` ``[n, d]`` (``True`` = observed).
    Outputs have the shape of ``resid`` without its last axis.

    The covariance is symmetrised first: a matrix that is symmetric only up to the
    rounding of its own dtype (``float32`` products, a covariance rescaled per target)
    would otherwise fail the symmetry check once it is upcast.
    """
    covariance = 0.5 * (covariance + covariance.mT)
    if resid.dim() == 2:
        out = known_covariance_gaussian_nll(resid, covariance, mask=mask)
        return out.chi2, out.log_det, out.dof
    n, s, d = resid.shape
    cov = covariance.unsqueeze(1).expand(n, s, d, d).reshape(n * s, d, d)
    rows = mask.unsqueeze(1).expand(n, s, d).reshape(n * s, d)
    out = known_covariance_gaussian_nll(resid.reshape(n * s, d), cov, mask=rows)
    return out.chi2.reshape(n, s), out.log_det.reshape(n, s), out.dof.reshape(n, s)


def safe_cholesky(cov: Tensor, max_rel_jitter: float = 1e-2) -> Tensor:
    """Cholesky factor of ``cov [..., d, d]``, adding diagonal jitter to rows that fail.

    Rows that factorise are untouched.  For the others, a jitter of
    ``16 eps``, then ten times more at each retry (relative to the row's mean
    diagonal), is added until they factorise; past ``max_rel_jitter`` a
    ``ValueError`` is raised.  Differentiable (the jitter is a constant).
    """
    chol, info = torch.linalg.cholesky_ex(cov)
    failed = info != 0
    if not bool(failed.any()):
        return chol
    d = cov.shape[-1]
    eye = torch.eye(d, dtype=cov.dtype, device=cov.device)
    scale = cov.detach().diagonal(dim1=-2, dim2=-1).abs().mean(-1)
    scale = scale.clamp_min(torch.finfo(cov.dtype).tiny)
    rel = 16.0 * torch.finfo(cov.dtype).eps
    while bool(failed.any()):
        if rel > max_rel_jitter:
            raise ValueError("covariance is not positive definite even with diagonal jitter")
        jitter = (rel * scale * failed.to(cov.dtype))[..., None, None] * eye
        chol, info = torch.linalg.cholesky_ex(cov + jitter)
        failed = info != 0
        rel *= 10.0
    return chol


@contextmanager
def seeded(seed: int) -> Generator[None]:
    """Run a block with the torch RNGs seeded by ``seed``, restoring the caller's state after.

    ``torch.manual_seed`` seeds the CPU, CUDA and MPS generators; ``fork_rng`` saves and
    restores the first two, the MPS state is saved and restored here.
    """
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    mps_state = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            yield
    finally:
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)


#: Seed of Monte Carlo predictive moments inside :func:`fixed_moment_seed`; ``None`` (the
#: default) draws them from the caller's generators.
MOMENT_SEED: ContextVar[Optional[int]] = ContextVar("torchregress_joint_moment_seed", default=None)


@contextmanager
def fixed_moment_seed(seed: int) -> Generator[None]:
    """Within the block, Monte Carlo ``mean`` / ``covariance`` of predictives use ``seed``.

    They are then a function of the predictive alone (kept apart from the moments drawn
    from the caller's generators), which is what a fitted transform needs to give the
    same output for the same predictive.
    """
    token = MOMENT_SEED.set(seed)
    try:
        yield
    finally:
        MOMENT_SEED.reset(token)


def to_tensor(data: Any) -> Tensor:
    """``data`` as a tensor: a tensor (returned as it is) or anything NumPy reads.

    NumPy arrays, nested lists and numbers go through :func:`torch.as_tensor`.  Any
    other object that converts to an array (a pandas or polars frame or series, an
    xarray array) is converted by NumPy first; torch alone rejects a frame and reads
    a series element by element.  A read-only array is copied, so the tensor never
    aliases memory torch may not write to.
    """
    if isinstance(data, Tensor):
        return data
    if not isinstance(data, (np.ndarray, list, tuple, int, float, bool)) and hasattr(
        data, "__array__"
    ):
        data = np.asarray(data)
    if isinstance(data, np.ndarray) and not data.flags.writeable:
        data = data.copy()
    return torch.as_tensor(data)
