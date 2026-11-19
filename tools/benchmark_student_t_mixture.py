"""Timings of ``StudentTMixture`` over rows ``B``, components ``K``, labels ``P`` and observations ``D``.

Per configuration (float32, df = 5, full scale-Cholesky components, 20% of the
observations invalid):

* ``obs``: forward + backward of the mixture NLL of ``D`` noisy linear observations
  with per-component designs (``component_observation_log_prob`` +
  ``mixture_log_prob``), the training path of a head whose survey maps are
  linearised per component;
* ``loop``: the same NLL with a Python loop over components and a dense ``D x D``
  Cholesky per component (the reference implementation this replaces);
* ``logp``: forward + backward of ``log_prob`` of full ``P``-vectors;
* ``cond``: ``condition`` on one noisy coordinate (no grad);
* ``quant``: ``marginal_quantile`` at 16/50/84 % for every coordinate (no grad;
  independent of ``D``, run once per ``B, K, P``).

    pixi run python tools/benchmark_student_t_mixture.py --rows 1000 10000 100000
"""

from __future__ import annotations

import argparse
import gc
import itertools
import math
import time
from collections.abc import Callable

import torch

from torchregress.losses import StudentTMixture

DF = 5.0


def _best(fn: Callable[[], object], repeats: int) -> float:
    fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def _data(b: int, k: int, p: int, d: int, device: str, seed: int = 0):  # type: ignore[no-untyped-def]
    g = torch.Generator().manual_seed(seed)
    loc = torch.randn(b, k, p, generator=g)
    raw = torch.randn(b, k, p, p, generator=g) * 0.2
    tril = raw.tril(-1) + torch.diag_embed(torch.rand(b, k, p, generator=g) + 0.5)
    logits = torch.randn(b, k, generator=g)
    design = torch.randn(b, k, d, p, generator=g) / math.sqrt(p)
    noise = torch.rand(b, d, generator=g) * 0.1 + 0.01
    z = torch.randn(b, d, generator=g)
    valid = torch.rand(b, d, generator=g) > 0.2
    out = (logits, loc, tril, design, noise, z, valid)
    return tuple(t.to(device) for t in out)


def _loop_nll(logits, loc, tril, design, noise, z, valid):  # type: ignore[no-untyped-def]
    """Reference: per-component dense Student-t likelihood, Python loop over K."""
    logw = torch.log_softmax(logits, -1)
    vf = valid.to(z.dtype)
    comps = []
    for k in range(loc.shape[1]):
        a = design[:, k] * vf[:, :, None]
        al = a @ tril[:, k]
        cov = al @ al.mT + torch.diag_embed(torch.where(valid, noise, torch.ones_like(noise)))
        chol = torch.linalg.cholesky(cov)
        r = torch.where(valid, z - (design[:, k] @ loc[:, k, :, None]).squeeze(-1), 0.0)
        sol = torch.linalg.solve_triangular(chol, r.unsqueeze(-1), upper=False).squeeze(-1)
        q = sol.square().sum(-1)
        logdet = 2 * torch.log(torch.diagonal(chol, dim1=1, dim2=2)).sum(-1)
        dim = vf.sum(-1)
        logp = (
            torch.lgamma((DF + dim) / 2)
            - math.lgamma(DF / 2)
            - dim / 2 * math.log(DF * math.pi)
            - logdet / 2
            - (DF + dim) / 2 * torch.log1p(q / DF)
        )
        comps.append(logp)
    return -torch.logsumexp(logw + torch.stack(comps, -1), -1)


def _run(b: int, k: int, p: int, d: int, device: str, repeats: int, quantiles: bool) -> dict:
    logits, loc, tril, design, noise, z, valid = _data(b, k, p, d, device)
    loc.requires_grad_(True)
    tril.requires_grad_(True)
    out: dict[str, float] = {}

    def sync() -> None:
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    def obs() -> None:
        loc.grad = tril.grad = None
        mix = StudentTMixture(logits, loc, scale_tril=tril, df=DF)
        comp = mix.component_observation_log_prob(z, design, noise, valid=valid)
        (-mix.mixture_log_prob(comp)).sum().backward()
        sync()

    def loop() -> None:
        loc.grad = tril.grad = None
        _loop_nll(logits, loc, tril, design, noise, z, valid).sum().backward()
        sync()

    x = torch.randn(b, p, device=device)

    def logp() -> None:
        loc.grad = tril.grad = None
        (-StudentTMixture(logits, loc, scale_tril=tril, df=DF).log_prob(x)).sum().backward()
        sync()

    out["obs"] = _best(obs, repeats)
    out["loop"] = _best(loop, repeats)
    out["logp"] = _best(logp, repeats)
    if quantiles:
        with torch.no_grad():
            mix = StudentTMixture(logits, loc.detach(), scale_tril=tril.detach(), df=DF)
            out["cond"] = _best(lambda: (mix.condition(0, z[:, 0], noise[:, 0]), sync()), repeats)
            out["quant"] = _best(lambda: (mix.marginal_quantile([0.16, 0.5, 0.84]), sync()), 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, nargs="+", default=[1000, 10000, 100000])
    parser.add_argument("--components", type=int, nargs="+", default=[1, 3, 8])
    parser.add_argument("--labels", type=int, nargs="+", default=[5, 16])
    parser.add_argument("--observations", type=int, nargs="+", default=[5, 20])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=0, help="torch threads (0: default)")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    print(
        f"# device={args.device} threads={torch.get_num_threads()} torch={torch.__version__}\n"
        f"{'B':>7} {'K':>2} {'P':>3} {'D':>3} {'obs s':>8} {'loop s':>8} {'speedup':>7} "
        f"{'logp s':>8} {'cond s':>8} {'quant s':>8}"
    )
    for b, k, p, d in itertools.product(args.rows, args.components, args.labels, args.observations):
        quantiles = d == args.observations[0]
        s = _run(b, k, p, d, args.device, args.repeats, quantiles)
        cond = f"{s['cond']:>8.4f}" if quantiles else f"{'':>8}"
        quant = f"{s['quant']:>8.3f}" if quantiles else f"{'':>8}"
        print(
            f"{b:>7} {k:>2} {p:>3} {d:>3} {s['obs']:>8.4f} {s['loop']:>8.4f} "
            f"{s['loop'] / s['obs']:>7.2f} {s['logp']:>8.4f} {cond} {quant}",
            flush=True,
        )
        gc.collect()


if __name__ == "__main__":
    main()
