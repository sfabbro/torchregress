"""Every multi-target head, loss, ensemble and region of the library through ``torchregress.joint``.

One row per inventory item of the joint guide's table "Every multi-target tool":
``as_joint`` and the explicit adapter give the same predictive, its layouts are
those of ``JointPredictive``, its density is the head's own density (masked
too, where the head has masked densities), a conformal region calibrated on
draws of a well-specified case covers fresh draws at the nominal level, and the
corner plot accepts it.  ``SLSRegion`` reproduces ``SLSConformal`` exactly.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
import torch
from matplotlib.figure import Figure
from torch import Tensor, nn

from torchregress.ensemble import (
    BaseEnsembleModel,
    DensityEnsemble,
    HeteroscedasticBatchEnsembleModel,
    HeteroscedasticBNN,
    HeteroscedasticEnsembleModel,
    MDNEnsembleModel,
)
from torchregress.joint import (
    ConditionalModelJoint,
    CopulaSplineJoint,
    DensityScore,
    DistributionJoint,
    FlowJoint,
    GaussianCopula,
    GaussianJoint,
    GaussianMarginals,
    JointConformal,
    JointPredictive,
    JointRegion,
    MixtureJoint,
    QuantileMarginals,
    SampleBallScore,
    SampleJoint,
    SklarJoint,
    SLSRegion,
    as_joint,
    coverage_by_mask,
)
from torchregress.losses import (
    EvidentialRegressionLoss,
    GaussianNLLLoss,
    InputNoiseMDNLoss,
    JointBinnedDensityLoss,
    KnownCovarianceGaussianNLL,
    LowRankGaussianLoss,
    MixtureDensityLoss,
    MultiQuantileLoss,
    MultivariateGaussianLoss,
    NoiseAwareDensityLoss,
    SLSConformal,
    SLSLoss,
    build_ratio_classifier,
    create_contrastive_flow_loss,
    create_flow_loss,
    create_flow_model,
    low_rank_output_dim,
    split_low_rank_gaussian_output,
)
from torchregress.models import (
    FlowMatchingMLP,
    GaussianCopulaSplineHead,
    MarginalContrastiveDensity,
    NormalizingFlowMap,
    sample_cfm,
)
from torchregress.viz import plot_corner_plot, plot_joint_diagnostics

D = 2  # targets
C = 3  # context features
ALPHA = 0.2
N_CAL, N_TEST = 500, 800
# Coverage of a split conformal region: Beta(k, n + 1 - k) over calibration sets,
# then binomial over the test rows; sd about 0.023 here, so 0.1 is beyond 4 sd.
COVERAGE_TOL = 0.1


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _perturb(module: nn.Module, seed: int, scale: float = 0.1) -> nn.Module:
    """Move zero-initialised heads away from the identity so tests are not trivial."""
    gen = _gen(seed)
    with torch.no_grad():
        for p in module.parameters():
            if p.is_floating_point():
                p.add_(scale * torch.randn(p.shape, generator=gen, dtype=p.dtype))
    return module


@dataclass
class Case:
    """One inventory row.

    ``inputs(n, seed)`` builds the head inputs of ``n`` rows; ``explicit`` is the
    documented explicit adapter; ``own`` / ``own_masked`` are the head's own
    (masked) log densities, when it has them.
    """

    obj: Any
    inputs: Callable[[int, int], Any]
    explicit: Callable[[Any], JointPredictive]
    own: Optional[Callable[[Any, Tensor], Tensor]] = None
    own_masked: Optional[Callable[[Any, Tensor, Tensor], Tensor]] = None
    kwargs: dict = field(default_factory=dict)
    # as_joint draws random numbers (Monte Carlo heads): reseed before each route.
    stochastic: bool = False
    # Draws-only predictive whose stored draws must not be scored against themselves.
    fresh_targets: bool = False
    flags: tuple[bool, bool, bool] = (True, False, False)  # density, latent, exact marginals


def _ctx(n: int, seed: int, dim: int = C) -> Tensor:
    return torch.randn(n, dim, generator=_gen(seed))


# ---------------------------------------------------------------- Gaussian losses
def _gaussian_diag() -> Case:
    loss = GaussianNLLLoss(reduction="none")

    def inputs(n: int, seed: int) -> Tensor:
        g = _gen(seed)
        return torch.cat([torch.randn(n, D, generator=g), 0.3 * torch.randn(n, D, generator=g)], 1)

    def explicit(out: Tensor) -> JointPredictive:
        mean, log_var = out.chunk(2, dim=-1)
        scale = (log_var.exp() + loss.eps).sqrt()
        return GaussianJoint(mean, torch.diag_embed(scale).to(device=mean.device, dtype=mean.dtype))

    return Case(
        loss,
        inputs,
        explicit,
        own=lambda out, y: -loss(out, y).sum(-1),
        flags=(True, True, True),
    )


def _cov_inputs(n: int, seed: int) -> tuple[Tensor, Tensor]:
    g = _gen(seed)
    a = 0.5 * torch.randn(n, D, D, generator=g)
    eye = torch.eye(D, dtype=torch.float32)
    return torch.randn(n, D, generator=g), a @ a.mT + 0.3 * eye


def _gaussian_full() -> Case:
    loss = MultivariateGaussianLoss(reduction="none")
    eye = torch.eye(D, dtype=torch.float32)
    return Case(
        loss,
        _cov_inputs,
        lambda mc: GaussianJoint.from_covariance(mc[0], mc[1] + loss.jitter * eye),
        own=lambda mc, y: -loss(mc[0], y, mc[1]),
        flags=(True, True, True),
    )


def _gaussian_low_rank() -> Case:
    loss = LowRankGaussianLoss(reduction="none")
    rank = 1

    def inputs(n: int, seed: int) -> Tensor:
        out = torch.randn(n, low_rank_output_dim(D, rank), generator=_gen(seed))
        out[:, -D:] = out[:, -D:].abs() + 0.2  # positive diagonal
        return out

    def explicit(out: Tensor) -> JointPredictive:
        mean, factor, diag = split_low_rank_gaussian_output(out, D, rank)
        return GaussianJoint.from_low_rank(mean, factor, diag + loss.jitter)

    def own(out: Tensor, y: Tensor) -> Tensor:
        mean, factor, diag = split_low_rank_gaussian_output(out, D, rank)
        return -loss(mean, y, factor, diag)

    return Case(loss, inputs, explicit, own=own, kwargs={"rank": rank}, flags=(True, True, True))


def _known_covariance() -> Case:
    torch.manual_seed(0)
    loss = KnownCovarianceGaussianNLL(n_features=D, scatter="constant", rank=1, reduction="none")
    known = _cov_inputs(1, 99)[1][0]  # a shared known covariance [D, D]

    def evaluate(y_pred: Tensor, y: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        with torch.no_grad():
            return -loss.evaluate(y_pred, y, known_covariance=known, mask=mask).nll

    return Case(
        loss,
        lambda n, seed: torch.randn(n, D, generator=_gen(seed)),
        lambda y_pred: GaussianJoint.from_known_covariance(
            y_pred, loss=loss, known_covariance=known
        ),
        own=evaluate,
        own_masked=evaluate,
        kwargs={"known_covariance": known},
        flags=(True, True, True),
    )


# -------------------------------------------------------- mixtures and densities
def _mdn(df: float = math.inf) -> Case:
    torch.manual_seed(1)
    loss = MixtureDensityLoss(n_components=2, n_features=D, covariance_type="full", df=df)
    return Case(
        loss,
        lambda n, seed: torch.randn(n, loss.expected_output_size, generator=_gen(seed)),
        lambda out: DistributionJoint(loss.distribution(out)),
        own=lambda out, y: loss.log_prob(out, y),
        own_masked=lambda out, y, m: loss.distribution(out).log_prob(y, mask=m),
        flags=(True, False, True),
    )


def _input_noise_mdn() -> Case:
    torch.manual_seed(2)
    model = nn.Linear(C, 2 + 2 * 2 * D)
    loss = InputNoiseMDNLoss(model, n_components=2, n_features=D, sigma_x=0.2, n_samples=4)

    def explicit(x: Tensor) -> JointPredictive:
        preds = loss.sample_predictions(x)
        return MixtureJoint([DistributionJoint(loss.base_loss.distribution(p)) for p in preds])

    def own(x: Tensor, y: Tensor) -> Tensor:
        # The likelihood-mode marginal of the loss: log mean_s p(y | model(x + e_s)).
        torch.manual_seed(1234)
        preds = loss.sample_predictions(x)
        lps = torch.stack([loss.base_loss.log_prob(p, y) for p in preds])
        return torch.logsumexp(lps, 0) - math.log(len(preds))

    return Case(loss, _ctx, explicit, own=own, stochastic=True, flags=(True, False, True))


def _evidential() -> Case:
    loss = EvidentialRegressionLoss()

    def explicit(out: Tensor) -> JointPredictive:
        gamma, nu, alpha, beta = loss._extract_nig_parameters(out)
        scale = (beta * (1 + nu) / (nu * alpha)).sqrt()
        dist = torch.distributions.StudentT(2 * alpha, gamma, scale)
        return DistributionJoint(torch.distributions.Independent(dist, 1))

    def own(out: Tensor, y: Tensor) -> Tensor:
        return -loss._nig_nll(y, *loss._extract_nig_parameters(out)).sum(-1)

    return Case(
        loss,
        lambda n, seed: torch.randn(n, 4 * D, generator=_gen(seed)),
        explicit,
        own=own,
    )


def _flow(flow_type: str) -> Case:
    torch.manual_seed(3)
    extra = {"hidden_features": (16,)} if flow_type == "tarflow" else {}
    loss = create_flow_loss(
        n_features=D, context_dim=C, flow_type=flow_type, n_transforms=2, **extra
    )
    _perturb(loss, 30)
    return Case(
        loss,
        _ctx,
        lambda c: FlowJoint(loss, c),
        own=lambda c, y: loss.log_prob(c, y),
        flags=(True, True, False),
    )


def _contrastive_flow() -> Case:
    torch.manual_seed(4)
    loss = create_contrastive_flow_loss(
        n_features=D, context_dim=C, flow_type="maf", n_transforms=2
    )
    return Case(
        loss,
        _ctx,
        lambda c: FlowJoint(loss, c),
        own=lambda c, y: loss.log_prob(c, y),
        flags=(True, True, False),
    )


def _noise_aware() -> Case:
    torch.manual_seed(5)
    base = MixtureDensityLoss(n_components=2, n_features=D)
    loss = NoiseAwareDensityLoss(base, sigma=0.1)
    return Case(
        loss,
        lambda n, seed: torch.randn(n, base.expected_output_size, generator=_gen(seed)),
        lambda out: DistributionJoint(base.distribution(out)),
        own=lambda out, y: base.log_prob(out, y),
        own_masked=lambda out, y, m: base.distribution(out).log_prob(y, mask=m),
        flags=(True, False, True),
    )


def _joint_binned() -> Case:
    edges = [torch.linspace(-4.0, 4.0, 9), torch.linspace(-4.0, 4.0, 7)]
    loss = JointBinnedDensityLoss(edges)
    return Case(
        loss,
        lambda n, seed: torch.randn(n, loss.joint_bins, generator=_gen(seed)),
        lambda logits: ConditionalModelJoint(loss, logits),
        own=lambda logits, y: loss.log_prob(logits, y),
    )


_LEVELS = torch.linspace(0.05, 0.95, 19)
_NORMAL_Q = torch.distributions.Normal(0.0, 1.0).icdf(_LEVELS)
# Fitted on normal scores of correlated pairs (stands in for out-of-fold scores).
_COPULA = GaussianCopula().fit(
    torch.randn(500, D, generator=_gen(6)) @ torch.tensor([[1.0, 0.6], [0.0, 0.8]])
)


def _quantile_inputs(n: int, seed: int) -> Tensor:
    g = _gen(seed)
    loc, scale = torch.randn(n, 1, D, generator=g), 0.5 + torch.rand(n, 1, D, generator=g)
    return loc + scale * _NORMAL_Q.view(1, -1, 1)  # [n, Q, d]


def _multi_quantile() -> Case:
    loss = MultiQuantileLoss(quantiles=_LEVELS.tolist())
    return Case(
        loss,
        _quantile_inputs,
        lambda q: SklarJoint(
            QuantileMarginals(_LEVELS, q.transpose(1, 2)), _COPULA.predictive(q.shape[0])
        ),
        kwargs={"dependence": _COPULA},
        flags=(True, True, True),
    )


def _marginals() -> Case:
    def inputs(n: int, seed: int) -> GaussianMarginals:
        g = _gen(seed)
        return GaussianMarginals(
            torch.randn(n, D, generator=g), 0.5 + torch.rand(n, D, generator=g)
        )

    case = Case(
        None,
        inputs,
        lambda m: SklarJoint(m, _COPULA.predictive(m.batch_size)),
        kwargs={"dependence": _COPULA},
        flags=(True, True, True),
    )
    return case


def _copula_spline() -> Case:
    torch.manual_seed(7)
    head = _perturb(GaussianCopulaSplineHead(C, D, width=16, bins=4), 70)
    return Case(
        head,
        _ctx,
        lambda c: CopulaSplineJoint(head, c),
        own=lambda c, y: head.intrinsic_log_prob(c, y, torch.ones_like(y, dtype=torch.bool)),
        own_masked=lambda c, y, m: head.intrinsic_log_prob(c, y, m),
        flags=(True, True, True),
    )


def _mcd() -> Case:
    torch.manual_seed(8)
    marginal = create_flow_model(n_features=D, context_dim=0, flow_type="maf", n_transforms=2)
    classifier = build_ratio_classifier(context_dim=C, observable_dim=D, hidden_dims=(16,))
    model = MarginalContrastiveDensity(marginal, classifier).eval()
    return Case(
        model,
        _ctx,
        lambda c: ConditionalModelJoint(model, c),
        own=lambda c, y: model.log_prob(c, y),
    )


def _flow_map() -> Case:
    torch.manual_seed(9)
    model = _perturb(NormalizingFlowMap(n_features=D, context_dim=C, hidden=(16,), n_layers=2), 90)
    return Case(
        model,
        _ctx,
        lambda c: ConditionalModelJoint(model, c),
        own=lambda c, y: model.log_prob(c, y),
    )


def _flow_matching() -> Case:
    torch.manual_seed(10)
    model = FlowMatchingMLP(n_features=D, context_dim=C, hidden=(16,))
    return Case(
        model,
        _ctx,
        lambda c: SampleJoint(sample_cfm(model, c, n_samples=64, steps=4)),
        kwargs={"n_samples": 64, "steps": 4},
        stochastic=True,
        fresh_targets=True,
        flags=(False, False, False),
    )


# --------------------------------------------------------------------- ensembles
def _density_ensemble() -> Case:
    torch.manual_seed(11)
    heads = [
        _perturb(create_flow_loss(n_features=D, context_dim=C, flow_type="maf", n_transforms=2), s)
        for s in (110, 111)
    ]
    ensemble = DensityEnsemble(heads)
    ensemble.mixture_weights.copy_(torch.tensor([0.3, 0.7]))  # type: ignore[operator]
    return Case(
        ensemble,
        _ctx,
        lambda x: MixtureJoint([FlowJoint(h, x) for h in heads], torch.tensor([0.3, 0.7])),
        own=lambda x, y: ensemble.log_prob(x, y),
    )


class _GaussianNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(C, 2 * D)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _hetero_ensemble() -> Case:
    torch.manual_seed(12)
    ens = HeteroscedasticEnsembleModel(_GaussianNet, ensemble_size=3, base_seed=0)

    def explicit(x: Tensor) -> JointPredictive:
        with torch.no_grad():
            outs = ens(x)
        members = []
        for out in outs:
            mean, log_var = out.chunk(2, dim=-1)
            var = log_var.clamp(-8.0, 6.0).exp().clamp_min(1e-8)
            members.append(
                GaussianJoint(
                    mean, torch.diag_embed(var.sqrt()).to(device=mean.device, dtype=mean.dtype)
                )
            )
        return MixtureJoint(members)

    return Case(ens, _ctx, explicit, flags=(True, False, True))


class _Backbone(nn.Module):
    feature_dim = 8

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(C, 8)

    def forward(self, x: Tensor) -> Tensor:
        return torch.tanh(self.net(x))


def _batch_ensemble() -> Case:
    torch.manual_seed(13)
    ens = HeteroscedasticBatchEnsembleModel(_Backbone(), output_size=D, ensemble_size=3)

    def explicit(x: Tensor) -> JointPredictive:
        with torch.no_grad():
            out = ens(x)
        var = out["log_vars"].clamp(-8.0, 6.0).exp().clamp_min(1e-8)
        members = [
            GaussianJoint(
                out["means"][:, k],
                torch.diag_embed(var[:, k].sqrt()).to(device=x.device, dtype=x.dtype),
            )
            for k in range(3)
        ]
        return MixtureJoint(members)

    return Case(ens, _ctx, explicit, flags=(True, False, True))


class _MDNNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(C, 2 + 2 * 2 * D)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _mdn_ensemble() -> Case:
    torch.manual_seed(14)
    ens = MDNEnsembleModel(_MDNNet, ensemble_size=2, n_components=2, n_features=D, base_seed=0)

    def explicit(x: Tensor) -> JointPredictive:
        with torch.no_grad():
            outs = ens(x)
        return MixtureJoint([DistributionJoint(ens.loss.distribution(o)) for o in outs])

    return Case(ens, _ctx, explicit, flags=(True, False, True))


def _hetero_bnn() -> Case:
    torch.manual_seed(15)
    bnn = HeteroscedasticBNN(input_dim=C, hidden_dims=[8], output_dim=D, n_samples=6)

    def explicit(x: Tensor) -> JointPredictive:
        means, log_vars = bnn.mc_forward(x)
        members = [
            GaussianJoint(m, torch.diag_embed(v.exp().sqrt()).to(device=m.device, dtype=m.dtype))
            for m, v in zip(means, log_vars)
        ]
        return MixtureJoint(members)

    return Case(bnn, _ctx, explicit, stochastic=True, flags=(True, False, True))


def _distribution() -> Case:
    def inputs(n: int, seed: int) -> torch.distributions.Distribution:
        mean, cov = _cov_inputs(n, seed)
        return torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)

    return Case(
        None,
        inputs,
        DistributionJoint,
        own=lambda dist, y: dist.log_prob(y),
    )


CASES: dict[str, Callable[[], Case]] = {
    "gaussian_diagonal": _gaussian_diag,
    "gaussian_full": _gaussian_full,
    "gaussian_low_rank": _gaussian_low_rank,
    "known_covariance": _known_covariance,
    "mixture_density": _mdn,
    "student_t_mixture": lambda: _mdn(df=5.0),
    "input_noise_mdn": _input_noise_mdn,
    "evidential": _evidential,
    **{f"flow_{t}": (lambda t=t: _flow(t)) for t in ("nsf", "maf", "realnvp", "bpf", "tarflow")},
    "contrastive_flow": _contrastive_flow,
    "noise_aware_density": _noise_aware,
    "joint_binned_density": _joint_binned,
    "multi_quantile": _multi_quantile,
    "marginals": _marginals,
    "copula_spline": _copula_spline,
    "marginal_contrastive_density": _mcd,
    "normalizing_flow_map": _flow_map,
    "flow_matching": _flow_matching,
    "density_ensemble": _density_ensemble,
    "heteroscedastic_ensemble": _hetero_ensemble,
    "heteroscedastic_batch_ensemble": _batch_ensemble,
    "mdn_ensemble": _mdn_ensemble,
    "heteroscedastic_bnn": _hetero_bnn,
    "torch_distribution": _distribution,
}

# The keyword-free rows where the object is the input itself (a distribution, marginals).
_OBJECT_IS_INPUT = {"marginals", "torch_distribution"}


def _route(case: Case, name: str, inputs: Any) -> Any:
    if name in _OBJECT_IS_INPUT:
        return as_joint(inputs, **case.kwargs)
    return as_joint(case.obj, inputs, **case.kwargs)


def _both(case: Case, name: str, inputs: Any) -> tuple[Any, JointPredictive]:
    """The predictive through ``as_joint`` and through the explicit adapter (same RNG)."""
    torch.manual_seed(1234)
    via = _route(case, name, inputs)
    torch.manual_seed(1234)
    ref = case.explicit(inputs)
    return via, ref


@pytest.mark.parametrize("name", list(CASES))
def test_layouts_and_densities(name: str) -> None:
    case = CASES[name]()
    n = 7
    inputs = case.inputs(n, 0)
    pred, ref = _both(case, name, inputs)
    assert isinstance(pred, JointPredictive) and type(pred) is type(ref)
    assert (pred.batch_size, pred.event_dim) == (n, D)
    assert (pred.has_density, pred.has_latent, pred.exact_marginals) == case.flags
    draws = pred.sample(5)
    assert draws.shape == (n, 5, D)
    assert pred.marginal([1]).sample(3).shape == (n, 3, 1)
    if not pred.has_density:
        with pytest.raises(NotImplementedError):
            pred.log_prob(draws[:, 0])
        if case.stochastic:  # same seed: the very draws of the head
            torch.manual_seed(1234)
            again = _route(case, name, inputs)
            torch.testing.assert_close(again.sample(64), ref.sample(64))
        return
    y = draws[:, 0].detach()
    with torch.no_grad():
        lp = pred.log_prob(y)
        assert lp.shape == (n,)
        torch.testing.assert_close(lp, ref.log_prob(y))
        if case.own is not None:
            torch.testing.assert_close(lp, case.own(inputs, y), rtol=1e-5, atol=1e-5)
        lp3 = pred.log_prob(draws)
        assert lp3.shape == (n, 5)
        torch.testing.assert_close(lp3[:, 2], pred.log_prob(draws[:, 2]), rtol=1e-5, atol=1e-5)
        if pred.exact_marginals:
            mask = torch.rand(n, D, generator=_gen(1)) < 0.6
            mask[0] = False  # a row with nothing observed has log density zero
            mask[1] = True
            masked = pred.log_prob(y, mask)
            assert masked[0] == 0
            torch.testing.assert_close(masked[1], lp[1])
            if case.own_masked is not None:
                torch.testing.assert_close(
                    masked, case.own_masked(inputs, y, mask), rtol=1e-5, atol=1e-5
                )
            torch.testing.assert_close(
                pred.marginal([0]).log_prob(y[:, :1]),
                pred.log_prob(y, torch.tensor([[True, False]] * n)),
            )
    if pred.has_latent:
        z = pred.to_latent(draws)
        assert z.shape == (n, 5, D)
        torch.testing.assert_close(pred.from_latent(z), draws, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("name", list(CASES))
def test_conformal_region_covers_and_plots(name: str) -> None:
    case = CASES[name]()
    torch.manual_seed(42)
    pred_cal = _route(case, name, case.inputs(N_CAL, 1))
    pred_test = _route(case, name, case.inputs(N_TEST, 2))

    def targets(pred: JointPredictive, n: int, seed: int) -> Tensor:
        if case.fresh_targets:  # a new draw, not one of the stored draws
            return _route(case, name, case.inputs(n, seed)).sample(1)[:, 0]
        return pred.sample(1)[:, 0]

    with torch.no_grad():
        y_cal = targets(pred_cal, N_CAL, 1).detach()
        y_test = targets(pred_test, N_TEST, 2).detach()
    score = DensityScore() if pred_cal.has_density else SampleBallScore(n_samples=32)
    conformal = JointConformal(score, alpha=ALPHA).calibrate(pred_cal, y_cal)
    region = conformal.region(pred_test)
    coverage = float(region.contains(y_test).float().mean())
    assert abs(coverage - (1 - ALPHA)) < COVERAGE_TOL, coverage

    small = _route(case, name, case.inputs(3, 3))
    fig = plot_corner_plot(
        small,
        row=0,
        region=conformal.region(small),
        n_samples=200,
        region_samples=100,
        return_figure=True,
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_as_joint_rejects_unsupported_objects() -> None:
    with pytest.raises(TypeError, match="supported: JointPredictive"):
        as_joint(nn.Linear(2, 2), torch.zeros(1, 2))
    point = BaseEnsembleModel(nn.Linear(C, D), ensemble_size=2, base_seed=0)
    with pytest.raises(TypeError, match="predicts points"):
        as_joint(point, torch.zeros(1, C))
    with pytest.raises(ValueError, match="second argument"):
        as_joint(MixtureDensityLoss(n_components=2, n_features=D))


def test_known_covariance_route_without_learned_terms() -> None:
    mean, cov = _cov_inputs(5, 3)
    known = cov[0]
    model_scale = torch.full((D,), 0.3)
    pred = GaussianJoint.from_known_covariance(
        mean, known_covariance=known, model_scale=model_scale
    )
    eye = torch.eye(D, dtype=torch.float32)
    torch.testing.assert_close(pred.covariance(), (known + 0.09 * eye).expand(5, D, D))


def test_ensemble_mixtures_match_the_ensembles_own_moments() -> None:
    x = _ctx(6, 5)
    for case in (_hetero_ensemble(), _batch_ensemble()):
        pred = as_joint(case.obj, x)
        moments = case.obj.predict(x)
        torch.testing.assert_close(pred.mean(), moments["mean"])
        torch.testing.assert_close(
            pred.covariance().diagonal(dim1=-2, dim2=-1), moments["variance"], rtol=1e-5, atol=1e-6
        )
    ens = _mdn_ensemble().obj
    pred = as_joint(ens, x)
    moments = ens.predict(x)
    torch.testing.assert_close(pred.mean(), moments["mean"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        pred.covariance().diagonal(dim1=-2, dim2=-1), moments["variance"], rtol=1e-4, atol=1e-6
    )


def test_copula_spline_latent_is_standard_normal() -> None:
    case = _copula_spline()
    pred = as_joint(case.obj, _ctx(4, 6))
    torch.manual_seed(0)
    z = pred.to_latent(pred.sample(4000))
    torch.testing.assert_close(z.mean(1), torch.zeros(4, D), atol=0.08, rtol=0)
    cov = (z.mT @ z) / z.shape[1]
    eye = torch.eye(D, dtype=torch.float32).expand(4, D, D)
    torch.testing.assert_close(cov, eye, atol=0.1, rtol=0)


# ═══════════════════════════════════════════════════════════════════════════════
# super-level-set regions
# ═══════════════════════════════════════════════════════════════════════════════


def _xy(n: int, seed: int) -> tuple[Tensor, Tensor]:
    g = _gen(seed)
    x = torch.randn(n, C, generator=g)
    y = 0.5 * x[:, :D] + torch.randn(n, D, generator=g) @ torch.tensor([[0.6, 0.2], [0.0, 0.4]])
    return x, y


def _sls(k: int, mode: str = "full", n_cal: int = 400, seed: int = 0) -> SLSConformal:
    torch.manual_seed(seed)
    loss = _perturb(
        SLSLoss(d=D, context_dim=C, K=k, mode=mode, hidden_dim=16, n_transforms=2), seed + 50
    )
    conformal = SLSConformal(loss, alpha=ALPHA)
    conformal.calibrate(*_xy(n_cal, seed + 1))
    return conformal


@pytest.mark.parametrize(("k", "mode"), [(1, "full"), (1, "low_rank"), (2, "full")])
def test_sls_region_reproduces_conformal_membership(k: int, mode: str) -> None:
    conformal = _sls(k, mode)
    x, _ = _xy(20, 7)
    region = as_joint(conformal, x)
    assert isinstance(region, SLSRegion) and isinstance(region, JointRegion)
    y = 1.5 * torch.randn(20, 30, D, generator=_gen(8))
    inside = region.contains(y)
    assert inside.shape == (20, 30) and 0 < float(inside.float().mean()) < 1
    assert torch.equal(inside, conformal.contains(x, y))
    assert torch.equal(region.contains(y[:, 0]), conformal.contains(x, y[:, 0]))
    assert region.contains(y[:, 0]).shape == (20,)
    # the explicit adapter is the same region
    assert torch.equal(SLSRegion(conformal, x).contains(y), inside)


def test_sls_region_follows_mondrian_groups() -> None:
    torch.manual_seed(0)
    loss = SLSLoss(d=D, context_dim=C, hidden_dim=16, n_transforms=2)
    conformal = SLSConformal(loss, alpha=ALPHA)
    x_cal, y_cal = _xy(300, 1)
    conformal.calibrate(x_cal, y_cal, groups=(x_cal[:, 0] > 0).long())
    x, _ = _xy(15, 2)
    groups = (x[:, 0] > 0).long()
    y = torch.randn(15, 10, D, generator=_gen(3))
    region = as_joint(conformal, x, groups=groups)
    assert torch.equal(region.contains(y), conformal.contains(x, y, groups=groups))


@pytest.mark.parametrize("k", [1, 2])
def test_sls_region_has_nominal_coverage(k: int) -> None:
    conformal = _sls(k, n_cal=500)
    x, y = _xy(2000, 11)
    coverage = float(as_joint(conformal, x).contains(y).float().mean())
    # Beta(k, n + 1 - k) calibration spread (sd 0.018) plus binomial test noise (0.009).
    assert abs(coverage - (1 - ALPHA)) < 0.07, coverage


@pytest.mark.parametrize("k", [1, 2])
def test_sls_region_volume_draws_and_box(k: int) -> None:
    conformal = _sls(k)
    x, _ = _xy(4, 12)
    region = SLSRegion(conformal, x)
    torch.manual_seed(13)
    log_vol, se = region.log_volume(n_samples=4000)
    draws = region.sample(400)
    assert draws.shape == (4, 400, D) and bool(region.contains(draws).all())
    lower, upper = region.bounding_box(n_samples=400)
    assert lower.shape == (4, D) and bool((lower <= upper).all())
    # independent estimate: uniform points in a box around the region
    lo = draws.amin(1) - 0.3 * (draws.amax(1) - draws.amin(1))
    hi = draws.amax(1) + 0.3 * (draws.amax(1) - draws.amin(1))
    u = lo.unsqueeze(1) + (hi - lo).unsqueeze(1) * torch.rand(4, 40000, D, generator=_gen(14))
    frac = region.contains(u).float().mean(1)
    box = (hi - lo).log().sum(-1) + frac.log()
    box_se = ((1 - frac) / (40000 * frac)).sqrt()
    if k == 1:
        assert bool((se == 0).all())
    tol = 4 * (se.square() + box_se.square()).sqrt() + 1e-3
    assert bool(((log_vol - box).abs() <= tol).all()), (log_vol, box, tol)
    # uniform draws: the fraction in the lower half-space of the box agrees with the box points
    mid = 0.5 * (lower + upper)
    in_region = region.contains(u)
    left_u = ((u[..., 0] < mid[:, :1]) & in_region).float().sum(1) / in_region.float().sum(1)
    left_d = (draws[..., 0] < mid[:, :1]).float().mean(1)
    torch.testing.assert_close(left_d, left_u, atol=0.1, rtol=0)


def test_sls_region_in_plots_and_diagnostics() -> None:
    conformal = _sls(1)
    x, y = _xy(30, 15)
    region = as_joint(conformal, x)
    assert isinstance(region, SLSRegion)
    fig = plot_corner_plot(
        torch.randn(500, D), row=0, region=region, region_samples=200, return_figure=True
    )
    assert isinstance(fig, Figure)
    plt.close(fig)
    eye = torch.eye(D, dtype=torch.float32)
    predictive = GaussianJoint(0.5 * x[:, :D], 0.7 * eye)
    out = plot_joint_diagnostics(predictive, y, x=x, region=region, n_samples=32)
    plt.close("all")
    del out
    covered = region.contains(y)
    report = coverage_by_mask(covered, torch.ones(30, D, dtype=torch.bool))
    assert report.coverage.shape[0] == 1


def test_sls_region_without_masks_and_unbounded() -> None:
    conformal = _sls(1, n_cal=3)  # too few rows for alpha: the region is the whole space
    x, y = _xy(5, 16)
    region = SLSRegion(conformal, x)
    with pytest.raises(ValueError, match="without a mask"):
        region.contains(y, torch.ones(5, D, dtype=torch.bool))
    assert bool(region.contains(y).all())
    log_vol, _ = region.log_volume()
    assert bool(torch.isposinf(log_vol).all())
    with pytest.warns(RuntimeWarning):
        lower, upper = region.bounding_box(n_samples=10)
    assert bool(torch.isneginf(lower).all() and torch.isposinf(upper).all())
    with pytest.raises(NotImplementedError, match="no predictive distribution"):
        region.predictive.sample(2)
