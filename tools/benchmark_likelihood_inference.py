"""Timings of ``maximize_score`` and ``profile_interval`` on an MLP score.

The score is a ratio-classifier-shaped MLP on ``concat([theta, x])`` (SiLU,
hidden widths 256-256-128, 256-d ``x``) plus a Gaussian term centred on a
linear function of ``x`` so that the maximum is interior, the setting of a
stellar-parameter ratio estimator with ``P = 3``.  Reported per phase:

* ``candidates``: Sobol candidate scoring only (``maximize_score(steps=0)``);
* ``maximize``: the full ``maximize_score`` (candidates + Newton refinement);
* ``profile j``: one ``profile_interval`` call at the 68 % Wilks threshold;

together with the number of ``(theta, x)`` pairs evaluated without gradients
(``fwd``) and with gradients + Hessians (``hess``).

    python tools/benchmark_likelihood_inference.py --rows 10000 --threads 4
"""

from __future__ import annotations

import argparse
import time

import torch
from torch import Tensor, nn

from torchregress.inference import maximize_score, profile_interval, wilks_threshold


class _Counted:
    def __init__(self, fn):  # type: ignore[no-untyped-def]
        self.fn = fn
        self.fwd = 0
        self.hess = 0

    def __call__(self, theta: Tensor, x: Tensor) -> Tensor:
        if torch.is_grad_enabled():
            self.hess += theta.shape[0]
        else:
            self.fwd += theta.shape[0]
        return self.fn(theta, x)

    def reset(self) -> None:
        self.fwd = 0
        self.hess = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--params", type=int, default=3)
    parser.add_argument("--features", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--n-candidates", type=int, default=256)
    parser.add_argument("--n-starts", type=int, default=4)
    parser.add_argument("--n-scan", type=int, default=8)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    dtype = getattr(torch, args.dtype)
    p, d = args.params, args.features

    torch.manual_seed(0)
    mlp = (
        nn.Sequential(
            nn.Linear(p + d, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
        .to(dtype)
        .eval()
    )
    for param in mlp.parameters():
        param.requires_grad_(False)
    proj = torch.randn(d, p, dtype=dtype) / d**0.5

    def raw_score(theta: Tensor, x: Tensor) -> Tensor:
        centre = x @ proj
        quad = -0.5 * (((theta - centre) / 0.3) ** 2).sum(-1)
        return quad + mlp(torch.cat([theta, x], dim=-1)).squeeze(-1)

    score = _Counted(raw_score)
    x = torch.randn(args.rows, d, dtype=dtype)
    bounds = torch.tensor([[-3.0, 3.0]] * p, dtype=dtype)
    gen = torch.Generator().manual_seed(0)
    common = dict(n_candidates=args.n_candidates, n_starts=args.n_starts, generator=gen)

    def report(name: str, seconds: float) -> None:
        print(
            f"{name:<12} {seconds:8.2f} s   fwd pairs {score.fwd / 1e6:8.2f} M"
            f"   hess pairs {score.hess / 1e6:8.2f} M"
        )
        score.reset()

    print(f"N={args.rows} P={p} D={d} {args.dtype} threads={args.threads}")
    start = time.perf_counter()
    maximize_score(score, x, bounds, steps=0, **common)  # type: ignore[arg-type]
    report("candidates", time.perf_counter() - start)

    gen.manual_seed(0)
    start = time.perf_counter()
    theta_hat, s_max, at_bound = maximize_score(score, x, bounds, **common)  # type: ignore[arg-type]
    report("maximize", time.perf_counter() - start)
    print(f"             rows with a coordinate on a bound: {int(at_bound.any(1).sum())}")

    threshold = wilks_threshold(0.6827)
    for j in range(p):
        start = time.perf_counter()
        interval = profile_interval(
            score, x, theta_hat, s_max, j, threshold, bounds, n_scan=args.n_scan
        )
        report(f"profile {j}", time.perf_counter() - start)
        width = (interval.upper - interval.lower).median().item()
        print(f"             median width {width:.3f}")


if __name__ == "__main__":
    main()
