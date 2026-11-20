"""Throughput of ``known_covariance_gaussian_nll`` (known plus model covariance).

Forward + backward (w.r.t. ``y_pred`` and the learned scatter) of the Gaussian NLL
for ``B`` rows of ``D`` features with a per-row dense known covariance, learned
fractional scatter and a rank-``k`` fractional factor (``KnownCovarianceGaussianNLL``),
10% masked features. The comparator is ``torch.distributions.MultivariateNormal``
on the same total covariance (no masks, covariance assembled by the caller).

    pixi run python tools/benchmark_known_covariance.py --rows 1024 8192 --dims 55 --rank 0 2
"""

from __future__ import annotations

import argparse
import itertools
import json
import time

import torch
from torch.distributions import MultivariateNormal

from torchregress.losses import KnownCovarianceGaussianNLL


def _data(rows: int, dims: int, dtype: torch.dtype, seed: int = 0):  # type: ignore[no-untyped-def]
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(rows, dims, dims, generator=g, dtype=dtype) / dims**0.5
    known = a @ a.mT + 1e-3 * torch.eye(dims, dtype=dtype)
    target = torch.randn(rows, dims, generator=g, dtype=dtype)
    mask = torch.rand(rows, dims, generator=g) > 0.1
    return known, target, mask


def _time(fn, repeats: int) -> float:  # type: ignore[no-untyped-def]
    fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1024])
    parser.add_argument("--dims", type=int, nargs="+", default=[55])
    parser.add_argument("--rank", type=int, nargs="+", default=[0, 2])
    parser.add_argument("--dtype", choices=["float32", "float64"], nargs="+", default=["float64"])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    rows_out = []
    for rows, dims, rank, dname in itertools.product(args.rows, args.dims, args.rank, args.dtype):
        dtype = getattr(torch, dname)
        known, target, mask = _data(rows, dims, dtype)
        loss = KnownCovarianceGaussianNLL(dims, scatter="fractional", rank=rank).to(dtype)
        pred = torch.zeros_like(target, requires_grad=True)

        def ours() -> None:
            pred.grad = None
            loss.zero_grad()
            loss(pred, target, known_covariance=known, mask=mask).backward()

        def native() -> None:
            pred.grad = None
            loss.zero_grad()
            variance, factor = loss.model_covariance_terms(pred)
            cov = known + torch.diag_embed(variance).to(dtype=dtype)
            if factor is not None:
                cov = cov + factor @ factor.mT
            (-MultivariateNormal(pred, covariance_matrix=cov).log_prob(target).mean()).backward()

        t_ours = _time(ours, args.repeats)
        t_native = _time(native, args.repeats)
        row = {
            "rows": rows,
            "dims": dims,
            "rank": rank,
            "dtype": dname,
            "threads": args.threads,
            "rows_per_s": round(rows / t_ours),
            "mvn_rows_per_s": round(rows / t_native),
        }
        rows_out.append(row)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
