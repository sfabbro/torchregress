"""The JointPredictive interface: layouts, densities, marginals, wrappers."""

from __future__ import annotations

import math

import pytest
import torch
from scipy.stats import kstest
from torch.distributions import MultivariateNormal

from torchregress.joint import (
    AffineJoint,
    DistributionJoint,
    FlowJoint,
    GaussianJoint,
    JointPredictive,
    MixtureJoint,
    ProjectedJoint,
    SampleJoint,
)
from torchregress.losses import StudentTMixture
from torchregress.losses.nflows import NormalizingFlowLoss, create_flow_model

F64 = torch.float64


def _gaussian(n: int = 6, d: int = 4, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    mean = torch.randn(n, d, generator=gen, dtype=F64)
    raw = torch.randn(n, d, d, generator=gen, dtype=F64)
    tril = raw.tril(-1) + torch.diag_embed(torch.rand(n, d, generator=gen, dtype=F64) + 0.5).to(
        dtype=F64
    )
    return mean, tril


class TestGaussianJoint:
    def test_log_prob_matches_torch_in_both_layouts(self) -> None:
        mean, tril = _gaussian()
        joint = GaussianJoint(mean, tril)
        ref = MultivariateNormal(mean, scale_tril=tril)
        y = joint.sample(5)
        assert y.shape == (6, 5, 4)
        torch.testing.assert_close(joint.log_prob(y[:, 0]), ref.log_prob(y[:, 0]))
        torch.testing.assert_close(joint.log_prob(y), ref.log_prob(y.movedim(1, 0)).movedim(0, 1))
        assert joint.has_density and joint.has_latent and joint.exact_marginals

    def test_latent_is_the_whitened_residual(self) -> None:
        mean, tril = _gaussian()
        joint = GaussianJoint(mean, tril)
        y = joint.sample(3)
        z = joint.to_latent(y)
        torch.testing.assert_close(mean.unsqueeze(1) + z @ tril.mT, y)
        torch.testing.assert_close(joint.to_latent(y[:, 1]), z[:, 1])
        torch.testing.assert_close(joint.from_latent(z), y)
        torch.testing.assert_close(joint.from_latent(z[:, 1]), y[:, 1])
        logp = -0.5 * z.square().sum(-1) - tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)[:, None]
        torch.testing.assert_close(joint.log_prob(y), logp - 2 * math.log(2 * math.pi))

    def test_sample_moments(self) -> None:
        torch.manual_seed(0)
        mean, tril = _gaussian(n=2, d=3)
        joint = GaussianJoint(mean, tril)
        draws = joint.sample(200_000)
        torch.testing.assert_close(draws.mean(1), mean, atol=0.02, rtol=0)
        emp = torch.stack([torch.cov(draws[i].T) for i in range(2)])
        torch.testing.assert_close(emp, joint.covariance(), atol=0.05, rtol=0)
        torch.testing.assert_close(joint.mean(), mean)

    def test_marginal_is_the_sub_gaussian(self) -> None:
        mean, tril = _gaussian()
        joint = GaussianJoint(mean, tril)
        sub = joint.marginal([2, 0])
        cov = (tril @ tril.mT)[:, [2, 0]][:, :, [2, 0]]
        ref = MultivariateNormal(mean[:, [2, 0]], covariance_matrix=cov)
        y = sub.sample(4)
        torch.testing.assert_close(sub.log_prob(y[:, 0]), ref.log_prob(y[:, 0]))
        assert isinstance(joint.marginal(1), GaussianJoint) and joint.marginal(1).event_dim == 1

    def test_mask_integrates_out_missing_targets(self) -> None:
        mean, tril = _gaussian()
        joint = GaussianJoint(mean, tril)
        y = joint.sample(3)
        mask = torch.tensor([[1, 0, 1, 1]] * 6, dtype=torch.bool)
        mask[0] = False  # nothing observed
        mask[1] = True  # everything observed
        got = joint.log_prob(torch.where(mask[:, None], y, torch.nan), mask)
        assert got.shape == (6, 3)
        assert bool((got[0] == 0).all())
        torch.testing.assert_close(got[1], joint.log_prob(y)[1])
        torch.testing.assert_close(
            got[2:], joint.marginal([0, 2, 3]).log_prob(y[..., [0, 2, 3]])[2:]
        )
        torch.testing.assert_close(joint.log_prob(y[:, 0], mask), got[:, 0])

    def test_condition_factorises_the_joint(self) -> None:
        mean, tril = _gaussian()
        joint = GaussianJoint(mean, tril)
        y = joint.sample(1)[:, 0]
        cond = joint.condition([1, 3], y[:, [1, 3]])
        assert cond.event_dim == 2
        total = joint.marginal([1, 3]).log_prob(y[:, [1, 3]]) + cond.log_prob(y[:, [0, 2]])
        torch.testing.assert_close(total, joint.log_prob(y))
        with pytest.raises(ValueError, match="nothing to predict"):
            joint.condition([0, 1, 2, 3], y)

    def test_constructors_and_validation(self) -> None:
        mean, tril = _gaussian()
        cov = tril @ tril.mT
        torch.testing.assert_close(GaussianJoint.from_covariance(mean, cov).scale_tril, tril)
        factor, diag = torch.randn(6, 4, 2, dtype=F64), torch.rand(6, 4, dtype=F64) + 0.1
        low = GaussianJoint.from_low_rank(mean, factor, diag)
        want = factor @ factor.mT + torch.diag_embed(diag).to(dtype=F64)
        torch.testing.assert_close(low.covariance(), want)
        shared = GaussianJoint(mean, tril[0])
        assert shared.scale_tril.shape == (6, 4, 4)
        assert GaussianJoint(mean, tril).select(torch.tensor([0, 2])).batch_size == 2
        with pytest.raises(ValueError, match=r"\[n, d\]"):
            GaussianJoint(mean[0], tril)
        with pytest.raises(ValueError, match="y must be"):
            GaussianJoint(mean, tril).log_prob(torch.zeros(6, 3, dtype=F64))
        with pytest.raises(ValueError, match="mask must be boolean"):
            GaussianJoint(mean, tril).log_prob(mean, torch.ones(6, 4))
        with pytest.raises(ValueError, match="distinct"):
            GaussianJoint(mean, tril).marginal([0, 0])


def test_hpd_levels_are_uniform_for_a_calibrated_predictive() -> None:
    torch.manual_seed(1)
    mean, tril = _gaussian(n=600, d=3, seed=4)
    joint = GaussianJoint(mean, tril)
    y = joint.sample(1)[:, 0]
    levels = joint.hpd_level(y, n_samples=400)
    assert levels.shape == (600,) and float(levels.min()) >= 0 and float(levels.max()) <= 1
    assert kstest(levels.numpy(), "uniform").pvalue > 1e-3
    # An over-confident predictive piles the levels up near one.
    narrow = GaussianJoint(mean, 0.5 * tril)
    assert float(narrow.hpd_level(y, n_samples=400).mean()) > 0.65


class TestDistributionJoint:
    def test_multivariate_normal(self) -> None:
        mean, tril = _gaussian()
        joint = DistributionJoint(MultivariateNormal(mean, scale_tril=tril))
        ref = GaussianJoint(mean, tril)
        y = joint.sample(7)
        assert y.shape == (6, 7, 4) and joint.has_density and not joint.has_latent
        torch.testing.assert_close(joint.log_prob(y), ref.log_prob(y))
        torch.testing.assert_close(joint.log_prob(y[:, 0]), ref.log_prob(y[:, 0]))
        torch.testing.assert_close(joint.mean(), mean)
        torch.testing.assert_close(joint.covariance(), ref.covariance())
        with pytest.raises(NotImplementedError, match="missing targets"):
            joint.log_prob(y[:, 0], torch.ones(6, 4, dtype=torch.bool))
        with pytest.raises(NotImplementedError, match="latent"):
            joint.to_latent(y)
        with pytest.raises(ValueError, match="batch shape"):
            DistributionJoint(MultivariateNormal(mean[0], scale_tril=tril[0]))

    def test_mixture_supports_masks(self) -> None:
        gen = torch.Generator().manual_seed(2)
        logits = torch.randn(5, 3, generator=gen, dtype=F64)
        loc = torch.randn(5, 3, 4, generator=gen, dtype=F64)
        diag = torch.rand(5, 3, 4, generator=gen, dtype=F64) + 0.3
        mix = StudentTMixture(logits, loc, scale_diag=diag, df=4.0)
        joint = DistributionJoint(mix)
        assert joint.exact_marginals
        y = joint.sample(6)
        assert y.shape == (5, 6, 4)
        mask = torch.tensor([[1, 1, 0, 1]] * 5, dtype=torch.bool)
        got = joint.log_prob(y, mask)
        assert got.shape == (5, 6)
        for s in range(6):
            torch.testing.assert_close(got[:, s], mix.log_prob(y[:, s], mask=mask))
        torch.testing.assert_close(joint.log_prob(y[:, 0], mask), got[:, 0])

    def test_flow_gives_a_latent_map(self) -> None:
        torch.manual_seed(0)
        flow = create_flow_model(n_features=3, context_dim=5, flow_type="maf", n_transforms=2)
        loss = NormalizingFlowLoss(flow=flow)
        context = torch.randn(4, 5)
        joint = FlowJoint(loss, context)
        assert joint.has_latent and joint.has_density
        y = joint.sample(8)
        assert y.shape == (4, 8, 3) and joint.sample(1).shape == (4, 1, 3)
        torch.testing.assert_close(joint.log_prob(y[:, 0]), loss.log_prob(context, y[:, 0]))
        z = joint.to_latent(y)
        assert z.shape == (4, 8, 3)
        torch.testing.assert_close(joint.to_latent(y[:, 2]), z[:, 2], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(joint.from_latent(z), y, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(joint.from_latent(z[:, 2]), y[:, 2], atol=1e-4, rtol=1e-4)
        dist = loss.distribution(context)
        _, ladj = dist.transform.call_and_ladj(y[:, 0])
        base = -0.5 * z[:, 0].square().sum(-1) - 1.5 * math.log(2 * math.pi)
        torch.testing.assert_close(joint.log_prob(y[:, 0]), base + ladj, atol=1e-4, rtol=1e-4)


class TestWrappers:
    def test_sample_and_projected_joints(self) -> None:
        draws = torch.randn(3, 10, 4)
        joint = SampleJoint(draws)
        assert not joint.has_density
        torch.testing.assert_close(joint.sample(10), draws)
        subset = joint.sample(4)
        assert subset.shape == (3, 4, 4)
        hits = (subset.unsqueeze(2) == draws.unsqueeze(1)).all(-1)  # [3, 4, 10]
        assert bool(hits.any(-1).all()) and bool((hits.sum(1) <= 1).all())  # no repeats
        more = joint.sample(25)
        assert more.shape == (3, 25, 4)
        assert bool((more.unsqueeze(2) == draws.unsqueeze(1)).all(-1).any(-1).all())
        assert joint.marginal([3, 1]).samples.shape == (3, 10, 2)
        assert joint.select(torch.tensor([True, False, True])).batch_size == 2
        with pytest.raises(NotImplementedError, match="no density"):
            joint.log_prob(draws[:, 0])

        mean, tril = _gaussian()
        proj = DistributionJoint(MultivariateNormal(mean, scale_tril=tril)).marginal([0, 3])
        assert isinstance(proj, ProjectedJoint) and proj.event_dim == 2
        assert proj.sample(9).shape == (6, 9, 2) and not proj.has_density

    def test_affine_changes_units_exactly(self) -> None:
        mean, tril = _gaussian()
        loc = torch.tensor([10.0, -3.0, 0.5, 100.0], dtype=F64)
        scale = torch.tensor([2.0, 0.1, 5.0, 30.0], dtype=F64)
        joint = AffineJoint(GaussianJoint(mean, tril), loc, scale)
        ref = GaussianJoint(loc + scale * mean, scale[:, None] * tril)
        y = ref.sample(5)
        torch.testing.assert_close(joint.log_prob(y), ref.log_prob(y))
        torch.testing.assert_close(joint.log_prob(y[:, 0]), ref.log_prob(y[:, 0]))
        torch.testing.assert_close(joint.to_latent(y), ref.to_latent(y))
        torch.testing.assert_close(joint.from_latent(ref.to_latent(y)), y)
        torch.testing.assert_close(joint.mean(), ref.mean())
        torch.testing.assert_close(joint.covariance(), ref.covariance())
        mask = torch.tensor([[0, 1, 1, 0]] * 6, dtype=torch.bool)
        torch.testing.assert_close(joint.log_prob(y, mask), ref.log_prob(y, mask))
        sub = joint.marginal([3, 1])
        torch.testing.assert_close(
            sub.log_prob(y[..., [3, 1]]), ref.marginal([3, 1]).log_prob(y[..., [3, 1]])
        )
        torch.manual_seed(0)
        draws = joint.sample(100_000)
        torch.testing.assert_close(draws.mean(1), ref.mean(), atol=0.5, rtol=0.02)
        assert joint.has_density and joint.has_latent and joint.exact_marginals
        with pytest.raises(ValueError, match="positive"):
            AffineJoint(GaussianJoint(mean, tril), loc, -scale)

    def test_mixture_of_joints(self) -> None:
        mean, tril = _gaussian()
        a, b = GaussianJoint(mean, tril), GaussianJoint(mean + 3.0, 0.5 * tril)
        weights = torch.tensor([0.25, 0.75], dtype=F64)
        mix = MixtureJoint([a, b], weights)
        y = a.sample(4)
        want = torch.logsumexp(
            torch.stack([a.log_prob(y) + math.log(0.25), b.log_prob(y) + math.log(0.75)], -1), -1
        )
        torch.testing.assert_close(mix.log_prob(y), want)
        torch.testing.assert_close(mix.mean(), 0.25 * a.mean() + 0.75 * b.mean())
        torch.manual_seed(0)
        draws = mix.sample(40_000)
        assert draws.shape == (6, 40_000, 4)
        torch.testing.assert_close(draws.mean(1), mix.mean(), atol=0.06, rtol=0)
        emp = torch.stack([torch.cov(draws[i].T) for i in range(6)])
        torch.testing.assert_close(emp, mix.covariance(), atol=0.25, rtol=0.05)
        assert mix.exact_marginals
        sub = mix.marginal([2, 0])
        want_sub = torch.logsumexp(
            torch.stack(
                [
                    a.marginal([2, 0]).log_prob(y[..., [2, 0]]) + math.log(0.25),
                    b.marginal([2, 0]).log_prob(y[..., [2, 0]]) + math.log(0.75),
                ],
                -1,
            ),
            -1,
        )
        torch.testing.assert_close(sub.log_prob(y[..., [2, 0]]), want_sub)
        uniform = MixtureJoint([a, b])
        torch.testing.assert_close(uniform.weights, torch.tensor([0.5, 0.5]))
        assert not MixtureJoint([a, SampleJoint(y)]).has_density
        with pytest.raises(ValueError, match="sum to one"):
            MixtureJoint([a, b], torch.tensor([0.5, 0.2]))
        with pytest.raises(ValueError, match="agree"):
            MixtureJoint([a, a.select(torch.tensor([0, 1]))])


def test_base_class_defaults() -> None:
    class Draws(JointPredictive):
        @property
        def batch_size(self) -> int:
            return 2

        @property
        def event_dim(self) -> int:
            return 3

        def sample(self, n_samples: int) -> torch.Tensor:
            return torch.ones(2, n_samples, 3) * torch.arange(3.0)

    joint = Draws()
    torch.testing.assert_close(joint.mean(8), torch.arange(3.0).expand(2, 3))
    assert joint.covariance(8).shape == (2, 3, 3)
    assert isinstance(joint.marginal(2), ProjectedJoint)
    assert repr(joint) == "Draws(n=2, d=3)"
    with pytest.raises(NotImplementedError):
        joint.to_latent(torch.zeros(2, 3))


def test_mixture_marginals_keep_a_density() -> None:
    gen = torch.Generator().manual_seed(3)
    logits = torch.randn(5, 3, generator=gen, dtype=F64)
    loc = torch.randn(5, 3, 4, generator=gen, dtype=F64)
    diag = torch.rand(5, 3, 4, generator=gen, dtype=F64) + 0.3
    joint = DistributionJoint(StudentTMixture(logits, loc, scale_diag=diag))
    sub = joint.marginal([3, 1])
    assert sub.has_density and sub.exact_marginals and sub.event_dim == 2
    y = joint.sample(6)
    # A diagonal Gaussian mixture has the mixture of the sub-Gaussians as marginal.
    comp = torch.distributions.Normal(loc[:, None, :, [3, 1]], diag[:, None, :, [3, 1]])
    log_w = torch.log_softmax(logits, -1)[:, None]
    want = torch.logsumexp(log_w + comp.log_prob(y[:, :, None, [3, 1]]).sum(-1), -1)
    torch.testing.assert_close(sub.log_prob(y[..., [3, 1]]), want)
    torch.testing.assert_close(sub.log_prob(y[:, 0][:, [3, 1]]), want[:, 0])
    assert sub.sample(9).shape == (5, 9, 2)
    again = sub.marginal(1)
    assert again.dims == [1] and again.event_dim == 1
    mask = torch.tensor([[True, False]] * 5)
    only_three = torch.tensor([[False, False, False, True]] * 5)
    torch.testing.assert_close(sub.log_prob(y[..., [3, 1]], mask), joint.log_prob(y, only_three))
    torch.testing.assert_close(sub.mean(), joint.mean()[:, [3, 1]])


def test_gaussian_affine_and_shared_weight_mixture_sampling() -> None:
    mean, tril = _gaussian()
    loc = torch.tensor([10.0, -3.0, 0.5, 100.0], dtype=F64)
    scale = torch.tensor([2.0, 0.1, 5.0, 30.0], dtype=F64)
    moved = GaussianJoint(mean, tril).affine(loc, scale)
    ref = AffineJoint(GaussianJoint(mean, tril), loc, scale)
    y = moved.sample(4)
    torch.testing.assert_close(moved.log_prob(y), ref.log_prob(y))
    # Per-row weights take the gather path; both paths give mixture draws.
    a, b = GaussianJoint(mean, tril), GaussianJoint(mean + 3.0, 0.5 * tril)
    per_row = MixtureJoint([a, b], torch.tensor([[0.25, 0.75]] * 6, dtype=F64))
    torch.manual_seed(0)
    draws = per_row.sample(40_000)
    torch.testing.assert_close(draws.mean(1), per_row.mean(), atol=0.06, rtol=0)


def test_mixture_draws_pick_members_independently_per_row() -> None:
    """Rows must not share one random member count: Monte Carlo scores rely on it."""
    torch.manual_seed(0)
    mean = torch.zeros(400, 2, dtype=F64)
    tril = torch.eye(2, dtype=F64).to(dtype=F64)
    near, far = GaussianJoint(mean, tril), GaussianJoint(mean + 100.0, tril)
    mix = MixtureJoint([near, far], torch.tensor([0.7, 0.3], dtype=F64))
    draws = mix.sample(64)
    from_far = (draws[..., 0] > 50).sum(1).double()  # per-row count of the far member
    # Binomial(64, 0.3): mean 19.2, standard deviation 3.67 across rows.
    assert abs(float(from_far.mean()) - 19.2) < 4 * 3.67 / 20
    assert 2.9 < float(from_far.std()) < 4.5
    # Draws of the two members are interleaved at random within a row.
    first_half = (draws[:, :32, 0] > 50).sum(1).double().mean()
    assert abs(float(first_half) - 9.6) < 4 * 2.6 / 20


class _CountingJoint(JointPredictive):
    """Draws-only predictive that counts its ``sample`` calls."""

    def __init__(self, base: GaussianJoint) -> None:
        self.base = base
        self.calls = 0

    @property
    def batch_size(self) -> int:
        return self.base.batch_size

    @property
    def event_dim(self) -> int:
        return self.base.event_dim

    def sample(self, n_samples: int) -> torch.Tensor:
        self.calls += 1
        return self.base.sample(n_samples)

    def log_prob(self, y: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.base.log_prob(y, mask)


class TestMonteCarloMoments:
    def test_mean_and_covariance_share_one_set_of_draws(self) -> None:
        torch.manual_seed(0)
        mean, tril = _gaussian(n=3, d=4)
        joint = _CountingJoint(GaussianJoint(mean, tril))
        cov = joint.covariance(4000)
        first = joint.mean(4000)
        assert joint.calls == 1  # the covariance's draws also gave the mean
        torch.testing.assert_close(joint.mean(4000), first, rtol=0, atol=0)
        torch.testing.assert_close(joint.covariance(4000), cov, rtol=0, atol=0)
        assert joint.calls == 1  # reused, not redrawn
        # the estimates are those of the draws: right to Monte Carlo error
        truth = tril @ tril.mT
        torch.testing.assert_close(first, mean, rtol=0, atol=0.25)
        torch.testing.assert_close(cov, truth, rtol=0, atol=0.25 * float(truth.abs().max()))
        assert float(torch.linalg.eigvalsh(cov).min()) > 0

    def test_the_covariance_is_only_formed_when_asked_for(self) -> None:
        joint = _CountingJoint(GaussianJoint(*_gaussian(n=3, d=4)))
        mean = joint.mean(256)
        kept = joint.__dict__["_moment_cache"][(256, None)]
        assert kept["cov"] is None and joint.calls == 1
        cov = joint.covariance(256)
        assert cov.shape == (3, 4, 4) and joint.calls == 2
        # the mean returned before stays the mean, whatever the later draws were
        torch.testing.assert_close(joint.mean(256), mean, rtol=0, atol=0)
        assert joint.calls == 2

    def test_moments_are_drawn_from_the_callers_generator_once_per_object(self) -> None:
        mean, tril = _gaussian(n=3, d=4)
        first, second = (_CountingJoint(GaussianJoint(mean, tril)) for _ in range(2))
        torch.manual_seed(11)
        a = first.mean(512)
        torch.manual_seed(11)
        b = second.mean(512)
        torch.testing.assert_close(a, b, rtol=0, atol=0)  # reproducible through the seed
        torch.manual_seed(12)
        third = _CountingJoint(GaussianJoint(mean, tril))
        assert not torch.equal(third.mean(512), a)  # and random otherwise: no shared noise
        torch.manual_seed(13)
        torch.testing.assert_close(first.mean(512), a, rtol=0, atol=0)  # kept on the object

    def test_scale_recalibration_does_not_depend_on_earlier_moment_calls(self) -> None:
        from torchregress.joint import ScaleRecalibrator

        gen = torch.Generator().manual_seed(4)
        store = torch.randn(40, 300, 3, generator=gen, dtype=F64)
        y = 1.3 * torch.randn(40, 3, generator=gen, dtype=F64)
        first = SampleJoint(store)
        recal = ScaleRecalibrator(objective="variance", n_samples=128).fit(first, y)
        second = SampleJoint(store)
        torch.manual_seed(123)
        second.mean(128)  # an earlier, differently seeded caller
        again = ScaleRecalibrator(objective="variance", n_samples=128).fit(second, y)
        torch.testing.assert_close(again.scale_, recal.scale_, rtol=0, atol=0)
        # the transform itself (location and scale of the affine map), not a Monte Carlo
        # summary of its output
        moved, reference = recal.transform(second), recal.transform(first)
        torch.testing.assert_close(moved.loc, reference.loc, rtol=0, atol=0)
        torch.testing.assert_close(moved.scale, reference.scale, rtol=0, atol=0)

    def test_another_sample_size_draws_again(self) -> None:
        joint = _CountingJoint(GaussianJoint(*_gaussian(n=2, d=3)))
        small, large = joint.mean(64), joint.mean(256)
        assert joint.calls == 2 and not torch.equal(small, large)

    def test_returned_moments_are_copies(self) -> None:
        joint = _CountingJoint(GaussianJoint(*_gaussian(n=2, d=3)))
        cov = joint.covariance(128)
        reference = cov.clone()
        cov.diagonal(dim1=-2, dim2=-1).add_(1.0)
        joint.mean(128).zero_()
        torch.testing.assert_close(joint.covariance(128), reference, rtol=0, atol=0)
        assert float(joint.mean(128).abs().sum()) > 0

    def test_differentiable_draws_keep_their_gradient_and_are_not_kept(self) -> None:
        store = torch.randn(4, 50, 3, dtype=F64, requires_grad=True)
        joint = SampleJoint(store)
        mean = joint.mean(50)
        assert mean.requires_grad
        # nothing kept (only the state token): no graph is retained
        assert set(joint.__dict__["_moment_cache"]) == {"token"}
        (grad,) = torch.autograd.grad(mean.sum(), store)
        torch.testing.assert_close(grad, torch.full_like(store, 1 / 50))
        assert joint.covariance(50).requires_grad

    def test_replacing_or_editing_the_stored_draws_forgets_their_moments(self) -> None:
        joint = SampleJoint(torch.randn(4, 50, 3, dtype=F64))
        before = joint.mean(50)
        joint.samples = joint.samples + 10.0
        torch.testing.assert_close(joint.mean(50), before + 10.0)
        joint.samples.add_(1.0)
        torch.testing.assert_close(joint.mean(50), before + 11.0)

    def test_moments_checked_under_no_grad_are_checked_again_with_gradients(self) -> None:
        store = torch.randn(4, 50, 3, dtype=F64, requires_grad=True)
        joint = SampleJoint(store)
        with torch.no_grad():
            detached = joint.mean(50)
        assert not detached.requires_grad
        assert joint.mean(50).requires_grad  # not the value kept under no_grad
        fixed = _CountingJoint(GaussianJoint(*_gaussian(n=2, d=3)))
        with torch.no_grad():
            value = fixed.mean(64)
        again = fixed.mean(64)  # draws once more to see that there is no gradient to keep
        assert fixed.calls == 2 and torch.equal(again, value)
        fixed.mean(64)
        assert fixed.calls == 2

    def test_copies_of_a_predictive_with_kept_moments_compute_their_own(self) -> None:
        import copy
        import pickle

        torch.manual_seed(0)
        joint = SampleJoint(torch.randn(4, 50, 3, dtype=F64))
        joint.mean(20)  # a random subset of the stored draws, kept on the original
        for clone in (pickle.loads(pickle.dumps(joint)), copy.deepcopy(joint)):
            torch.testing.assert_close(clone.samples, joint.samples)
            torch.testing.assert_close(clone.mean(50), joint.samples.mean(1))
            assert torch.isfinite(clone.mean(20)).all() and clone.covariance(20).shape == (4, 3, 3)
            torch.testing.assert_close(clone.mean(20), clone.mean(20), rtol=0, atol=0)
        legacy = SampleJoint.__new__(SampleJoint)  # an object pickled before the cache existed
        legacy.__dict__["samples"] = joint.samples.clone()
        torch.testing.assert_close(legacy.mean(50), joint.samples.mean(1))

    def test_a_reused_tensor_id_does_not_bring_old_moments_back(self) -> None:
        # Replace the draws twice without a moment call in between: the third tensor
        # usually gets the first one's freed id, and both have version 0.
        torch.manual_seed(0)
        stale = 0
        for _ in range(50):
            joint = SampleJoint(torch.randn(4, 20, 2))
            old = joint.mean(20)
            joint.samples = torch.randn(4, 20, 2)
            joint.samples = torch.randn(4, 20, 2)
            stale += int(torch.equal(joint.mean(20), old))
            torch.testing.assert_close(joint.mean(20), joint.samples.mean(1))
        assert stale == 0

    def test_replacing_the_draws_drops_every_kept_entry(self) -> None:
        joint = SampleJoint(torch.randn(4, 30, 2))
        joint.mean(8), joint.covariance(16), joint.mean(30)
        assert len(joint.__dict__["_moment_cache"]) == 4  # the token and three entries
        joint.samples = torch.randn(4, 30, 2)
        joint.mean(30)
        assert len(joint.__dict__["_moment_cache"]) == 2

    def test_a_no_grad_call_does_not_undo_the_gradient_check(self) -> None:
        joint = _CountingJoint(GaussianJoint(*_gaussian(n=2, d=3)))
        with torch.no_grad():
            joint.mean(64)
        joint.mean(64)  # checked once with gradients enabled
        with torch.no_grad():
            joint.covariance(64)  # adds the covariance
        calls = joint.calls
        joint.mean(64), joint.covariance(64)
        assert joint.calls == calls


class _SizedJoint(_CountingJoint):
    """Records how many draws each ``sample`` call asked for."""

    def __init__(self, base: GaussianJoint) -> None:
        super().__init__(base)
        self.sizes: list[int] = []

    def sample(self, n_samples: int) -> torch.Tensor:
        self.sizes.append(n_samples)
        return super().sample(n_samples)


def _three_gaussians() -> list[GaussianJoint]:
    mean, tril = _gaussian(n=5, d=3)
    return [
        GaussianJoint(mean, tril),
        GaussianJoint(mean + 2.0, 0.5 * tril),
        GaussianJoint(mean - 3.0, 1.5 * tril),
    ]


class TestMixtureMoments:
    """A mixture asks a sampled member for its share of the draws, not for all of them."""

    def test_sampled_members_draw_in_proportion_to_their_weight(self) -> None:
        a, b, c = _three_gaussians()
        first, last = _SizedJoint(a), _SizedJoint(c)
        mix = MixtureJoint([first, b, last], torch.tensor([0.6, 0.3, 0.1], dtype=F64))
        torch.manual_seed(0)
        mix.covariance(1000)
        mix.mean(1000)
        assert first.sizes == [600] and last.sizes == [100]  # once each, for both moments
        # a floor for members with a small weight, and never more than was asked for
        small = _SizedJoint(c)
        MixtureJoint([a, small], torch.tensor([0.99, 0.01], dtype=F64)).mean(1000)
        assert small.sizes == [32]
        small = _SizedJoint(c)
        MixtureJoint([a, small], torch.tensor([0.99, 0.01], dtype=F64)).mean(16)
        assert small.sizes == [16]
        # per-row weights: the largest weight of the member over the rows
        rows = torch.tensor([[0.9, 0.1]] * 4 + [[0.5, 0.5]], dtype=F64)
        small = _SizedJoint(c)
        MixtureJoint([a, small], rows).mean(1000)
        assert small.sizes == [500]

    def test_closed_form_members_are_exact_whatever_the_sample_size(self) -> None:
        a, b, c = _three_gaussians()
        mix = MixtureJoint([a, b, c], torch.tensor([0.6, 0.3, 0.1], dtype=F64))
        torch.testing.assert_close(mix.mean(8), mix.mean(4096), rtol=0, atol=0)
        torch.testing.assert_close(mix.covariance(8), mix.covariance(4096), rtol=0, atol=0)

    def test_the_estimate_beats_the_same_number_of_draws_from_the_mixture(self) -> None:
        # Truth from the closed forms; the sampled mixture hides them behind draws-only
        # members.  Root mean square errors over 40 repetitions of S = 256.  The mean of
        # 256 draws of the mixture also carries the spread of the member means.
        a, b, c = _three_gaussians()
        weights = torch.tensor([0.6, 0.3, 0.1], dtype=F64)
        exact = MixtureJoint([a, b, c], weights)
        true_mean, true_cov = exact.mean(), exact.covariance()
        sd = true_cov.diagonal(dim1=-2, dim2=-1).sqrt()
        err_strat, err_plain, cov_sum = [], [], torch.zeros_like(true_cov)
        reps = 40
        for rep in range(reps):
            torch.manual_seed(rep)
            mix = MixtureJoint([_CountingJoint(a), _CountingJoint(b), _CountingJoint(c)], weights)
            cov_sum += mix.covariance(256)
            err_strat.append(((mix.mean(256) - true_mean) / sd).square().mean())
            plain = exact.sample(256).mean(1)
            err_plain.append(((plain - true_mean) / sd).square().mean())
        rmse_strat = float(torch.stack(err_strat).mean().sqrt())
        rmse_plain = float(torch.stack(err_plain).mean().sqrt())
        assert rmse_strat < 0.8 * rmse_plain, (rmse_strat, rmse_plain)
        assert rmse_plain < 1.3 / 16  # 1 / sqrt(256) in standard deviations: the yardstick
        # The covariance is right on average.  Variances: mean relative error within 3%
        # (measured +0.0% to +0.9% over three sets of seeds; the noisy member means
        # inflate it by about 0.7% here).  Every entry: within 10% of the row's largest.
        average = cov_sum / reps
        variances = average.diagonal(dim1=-2, dim2=-1) / true_cov.diagonal(dim1=-2, dim2=-1)
        assert abs(float(variances.mean()) - 1.0) < 0.03, float(variances.mean())
        scale = true_cov.abs().amax((-2, -1), keepdim=True)
        assert float(((average - true_cov) / scale).abs().max()) < 0.1


class TestSeededBlock:
    def test_restores_the_cpu_generator(self) -> None:
        from torchregress.joint._common import seeded

        torch.manual_seed(5)
        state = torch.random.get_rng_state()
        with seeded(0):
            inside = torch.randn(3)
        assert torch.equal(torch.random.get_rng_state(), state)
        with seeded(0):
            torch.testing.assert_close(torch.randn(3), inside, rtol=0, atol=0)

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
    def test_restores_the_mps_generator(self) -> None:
        from torchregress.joint._common import seeded

        torch.manual_seed(5)
        expected = torch.randn(4, device="mps").cpu()
        torch.manual_seed(5)
        with seeded(0):
            torch.randn(4, device="mps")
        torch.testing.assert_close(torch.randn(4, device="mps").cpu(), expected, rtol=0, atol=0)
        # and through the recalibrator, the one place that fixes the seed of moments
        from torchregress.joint import ScaleRecalibrator

        store = torch.randn(20, 100, 2)
        recal = ScaleRecalibrator(objective="variance", n_samples=64)
        recal.fit(SampleJoint(store), torch.randn(20, 2))
        torch.manual_seed(5)
        recal.transform(SampleJoint(store))
        torch.testing.assert_close(torch.randn(4, device="mps").cpu(), expected, rtol=0, atol=0)
