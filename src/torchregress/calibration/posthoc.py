"""Post-hoc calibration transforms for regression outputs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from torchregress.utils.distributions import normal_cdf


@dataclass
class VarianceTemperatureScaler:
    """Scalar variance-temperature calibration for Gaussian predictive variance.

    Fits ``T`` (and optionally an additive variance floor ``f``) by Gaussian
    negative log-likelihood on a held-out calibration split, so that the
    calibrated predictive variance is ``T * pred_var + f``.

    Two optional extensions of the scalar temperature, both off by default:

    * ``target_var``: per-sample variance of a *noisy* target (labels with their
      own uncertainty). The likelihood then uses ``T * pred_var + f + target_var``
      (as in :func:`torchregress.metrics.uncertain.noisy_target_gaussian_nll`),
      so label noise is not attributed to the model. :meth:`transform` returns
      the model's variance only.
    * ``fit_floor``: also fit ``f >= 0``. A temperature alone cannot represent an
      error budget with a floor (e.g. template or systematics error added to
      a photon-noise error), which shows up as under-coverage for the
      best-measured samples and over-coverage for the worst.

    ``clip`` drops calibration samples whose residual lies more than ``clip``
    robust standard deviations (1.4826 MAD) from the median before fitting, so
    a few catastrophic failures do not set the scale.

    References
    ----------
    .. [1] Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration
       of Modern Neural Networks. In *ICML 2017*. https://arxiv.org/abs/1706.04599
    .. [2] Kuleshov, V., Fenner, N., & Ermon, S. (2018). Accurate Uncertainties for
       Deep Learning Using Calibrated Regression. In *ICML 2018*.
       https://arxiv.org/abs/1807.00263
    """

    temperature: float = 1.0
    eps: float = 1e-8
    variance_floor: float = 0.0

    def fit(
        self,
        pred_mean: Tensor,
        pred_var: Tensor,
        target: Tensor,
        *,
        max_iter: int = 200,
        lr: float = 0.05,
        target_var: Tensor | None = None,
        fit_floor: bool = False,
        clip: float | None = None,
    ) -> "VarianceTemperatureScaler":
        if pred_mean.shape != pred_var.shape or pred_mean.shape != target.shape:
            raise ValueError("pred_mean, pred_var, and target must share shape")
        if target_var is not None and target_var.shape != pred_var.shape:
            raise ValueError("target_var must share shape with pred_var")
        if clip is not None and not clip > 0:
            raise ValueError("clip must be positive")

        mean = pred_mean.detach().double()
        # ``eps`` is applied after rescaling to the typical variance below, so a
        # target measured in tiny units (variances ~1e-10) is not clamped.
        var = pred_var.detach().double().clamp_min(torch.finfo(torch.float64).tiny)
        y = target.detach().double()
        noise = (
            torch.zeros_like(var)
            if target_var is None
            else target_var.detach().double().clamp_min(0.0)
        )
        keep = torch.isfinite(mean) & torch.isfinite(var) & torch.isfinite(y)
        keep &= torch.isfinite(noise)
        if clip is not None:
            residual = (y - mean)[keep]
            centre = residual.median()
            robust = 1.4826 * (residual - centre).abs().median()
            if float(robust) > 0:
                keep &= (y - mean - centre).abs() <= clip * robust
        if int(keep.sum()) < 2:
            raise ValueError("need at least two finite calibration samples")
        mean, var, y, noise = mean[keep], var[keep], y[keep], noise[keep]

        # Work in units of the typical predicted variance, so the optimiser sees
        # O(1) numbers whatever the physical scale of the target.
        unit = var.median().clamp_min(torch.finfo(torch.float64).tiny)
        var_u, noise_u = (var / unit).clamp_min(self.eps), noise / unit
        residual_sq = (y - mean) ** 2 / unit

        log_t = torch.nn.Parameter(torch.tensor(math.log(self.temperature), dtype=torch.float64))
        parameters = [log_t]
        log_f = None
        if fit_floor:
            start = max(self.variance_floor / float(unit), 1e-2)
            log_f = torch.nn.Parameter(torch.tensor(math.log(start), dtype=torch.float64))
            parameters.append(log_f)
        optimizer = torch.optim.Adam(parameters, lr=lr)

        def total_variance() -> Tensor:
            t = torch.exp(log_t).clamp(min=0.05, max=20.0)
            floor = (
                torch.exp(log_f)
                if log_f is not None
                else torch.tensor(self.variance_floor / float(unit), dtype=torch.float64)
            )
            return (var_u * t + floor + noise_u).clamp_min(self.eps)

        for _ in range(max_iter):
            optimizer.zero_grad(set_to_none=True)
            total = total_variance()
            nll = 0.5 * (torch.log(total) + residual_sq / total + math.log(2.0 * math.pi))
            loss = nll.mean()
            loss.backward()
            optimizer.step()

        self.temperature = float(torch.exp(log_t).clamp(min=0.05, max=20.0).item())
        if log_f is not None:
            self.variance_floor = float(torch.exp(log_f).item() * float(unit))
        return self

    def transform(self, pred_var: Tensor) -> Tensor:
        """Calibrated variance ``T * pred_var + f``, clamped below at ``eps``.

        ``eps`` is in the caller's variance units: set it below the smallest
        meaningful variance when targets live on tiny scales.
        """
        return (pred_var * self.temperature + self.variance_floor).clamp_min(self.eps)


@dataclass
class IsotonicMeanCalibrator:
    """Isotonic regression calibrator for point predictions.

    Implements the Pool Adjacent Violators Algorithm (PAVA) directly on
    PyTorch tensors so no scikit-learn dependency is required.

    References
    ----------
    .. [1] Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into accurate
       multiclass probability estimates. In *KDD 2002*. https://doi.org/10.1145/775047.775151
    """

    out_of_bounds: str = "clip"

    def __post_init__(self) -> None:
        self._x: Tensor | None = None
        self._y: Tensor | None = None

    @staticmethod
    def _pava(x: Tensor | np.ndarray, y: Tensor | np.ndarray) -> tuple[Any, Any]:
        is_numpy = isinstance(x, np.ndarray)
        x_t = torch.as_tensor(x)
        y_t = torch.as_tensor(y)

        if x_t.numel() == 0:
            if is_numpy:
                return np.array([], dtype=float), np.array([], dtype=float)
            return x_t.clone(), y_t.clone()

        order = x_t.argsort()
        x_s = x_t[order].double()
        y_s = y_t[order].double()

        blocks_x: list[Tensor] = []
        blocks_y: list[Tensor] = []

        for i in range(len(x_s)):
            blocks_x.append(x_s[i : i + 1])
            blocks_y.append(y_s[i : i + 1])

            while len(blocks_y) >= 2 and blocks_y[-2].mean() > blocks_y[-1].mean():
                bx = torch.cat([blocks_x.pop(), blocks_x.pop()])
                by = torch.cat([blocks_y.pop(), blocks_y.pop()])
                blocks_x.append(bx)
                blocks_y.append(by)

        result_x = torch.tensor([b.mean() for b in blocks_x], dtype=x_t.dtype)
        result_y = torch.tensor([b.mean() for b in blocks_y], dtype=y_t.dtype)

        if is_numpy:
            return result_x.numpy(), result_y.numpy()
        return result_x, result_y

    def _interpolate(self, x_query: Tensor) -> Tensor:
        xq = torch.as_tensor(x_query)
        if self._x is None or self._y is None:
            raise ValueError("IsotonicMeanCalibrator must be fitted before transform")

        if self._x.numel() == 0:
            return torch.zeros_like(xq)

        if self._x.numel() == 1:
            val = float(self._y[0].item())
            return torch.full_like(xq, val)

        idx = torch.searchsorted(self._x, xq).clamp(1, len(self._x) - 1)
        x_left = self._x[idx - 1]
        x_right = self._x[idx]
        y_left = self._y[idx - 1]
        y_right = self._y[idx]

        denom = x_right - x_left
        t = torch.where(denom.abs() < 1e-12, 0.0, (xq - x_left) / denom)

        if self.out_of_bounds == "clip":
            t = t.clamp(0.0, 1.0)
            y_out = y_left + t * (y_right - y_left)
            y_out = torch.where(xq <= self._x[0], self._y[0], y_out)
            y_out = torch.where(xq >= self._x[-1], self._y[-1], y_out)
            return y_out

        return y_left + t * (y_right - y_left)

    def fit(self, pred_mean: Tensor, target: Tensor) -> "IsotonicMeanCalibrator":
        x = pred_mean.detach().reshape(-1).double()
        y = target.detach().reshape(-1).double()
        if x.shape[0] != y.shape[0]:
            raise ValueError("pred_mean and target must share sample dimension")

        self._x, self._y = self._pava(x, y)
        return self

    def transform(self, pred_mean: Tensor) -> Tensor:
        x = pred_mean.detach().reshape(-1).double()
        y_hat = self._interpolate(x)
        return y_hat.reshape(pred_mean.shape).to(pred_mean.dtype)


@dataclass
class PITCalibrator:
    """Monotonic PIT-value calibrator using empirical CDF mapping."""

    eps: float = 1e-6

    def __post_init__(self) -> None:
        self._x: Tensor | None = None
        self._y: Tensor | None = None

    @staticmethod
    def pit_from_gaussian(pred_mean: Tensor, pred_std: Tensor, target: Tensor) -> Tensor:
        std = pred_std.clamp_min(1e-8)
        z = (target - pred_mean) / std
        return normal_cdf(z).clamp(min=1e-6, max=1.0 - 1e-6)

    def fit(self, pit_values: Tensor) -> "PITCalibrator":
        pit = pit_values.detach().reshape(-1).clamp(self.eps, 1.0 - self.eps)
        pit_sorted = pit.sort().values
        n = pit_sorted.shape[0]
        targets = (torch.arange(n, dtype=torch.float64) + 0.5) / n
        self._x = pit_sorted
        self._y = targets
        return self

    def transform(self, pit_values: Tensor) -> Tensor:
        if self._x is None or self._y is None:
            raise ValueError("PITCalibrator must be fitted before transform")
        pit = pit_values.detach().reshape(-1)
        idx = torch.searchsorted(self._x, pit).clamp(1, len(self._x) - 1)
        x_left = self._x[idx - 1]
        x_right = self._x[idx]
        y_left = self._y[idx - 1]
        y_right = self._y[idx]
        denom = x_right - x_left
        t = torch.where(denom.abs() < 1e-12, 0.0, (pit - x_left) / denom)
        mapped = y_left + t.clamp(0.0, 1.0) * (y_right - y_left)
        mapped = torch.where(pit <= self._x[0], self._y[0], mapped)
        mapped = torch.where(pit >= self._x[-1], self._y[-1], mapped)
        return mapped.clamp(self.eps, 1.0 - self.eps).reshape(pit_values.shape).to(pit_values.dtype)


__all__ = [
    "VarianceTemperatureScaler",
    "IsotonicMeanCalibrator",
    "PITCalibrator",
]
