"""Shared distribution primitives for losses, metrics, and calibration."""

from __future__ import annotations

import math

import torch
from torch import Tensor

_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def normal_cdf(z: Tensor) -> Tensor:
    """Standard normal CDF ``Phi(z) = 0.5 * erfc(-z / sqrt(2))``.

    The ``erfc`` form keeps full relative precision in the lower tail (``0.5 * (1 +
    erf(z / sqrt(2)))`` cancels to 0 for ``z < -8`` in float64).
    """
    return 0.5 * torch.erfc(-z * _INV_SQRT2)


def normal_icdf(u: Tensor) -> Tensor:
    """Standard normal quantile ``Phi^{-1}(u)`` (:func:`torch.special.ndtri`).

    MPS does not implement the operator; a tensor on that device is evaluated on the CPU
    and moved back (gradients pass through the moves).
    """
    if u.device.type == "mps":
        return torch.special.ndtri(u.cpu()).to(u.device)
    return torch.special.ndtri(u)


def normal_log_cdf(z: Tensor) -> Tensor:
    """``log Phi(z)`` (:func:`torch.special.log_ndtr`), on the CPU for MPS tensors.

    MPS does not implement the operator; gradients pass through the moves.
    """
    if z.device.type == "mps":
        return torch.special.log_ndtr(z.cpu()).to(z.device)
    return torch.special.log_ndtr(z)


def inverse_softplus(x: Tensor) -> Tensor:
    """Inverse of ``softplus`` for ``x > 0``: ``log(expm1(x)) = x + log(-expm1(-x))``.

    The second form never overflows (``expm1(x)`` is ``inf`` for ``x > ~88`` in
    float32).
    """
    return x + torch.log(-torch.expm1(-x))
