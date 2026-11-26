"""SklarJoint and the dependence ladder on synthetic scores with known dependence."""

from __future__ import annotations

import math

import pytest
import torch
from scipy.stats import kstest, spearmanr
from torch.distributions import MultivariateNormal, Normal, StudentT

from torchregress.joint.dependence import (
    ConditionalGaussianCopula,
    Dependence,
    FlowDependence,
    GaussianCopula,
    IndependenceDependence,
    MixtureDependence,
    SchaakeShuffle,
    SklarJoint,
    TemplateJoint,
    select_dependence,
)
from torchregress.joint.marginals import GaussianMarginals, QuantileMarginals
from torchregress.joint.predictive import FlowJoint, GaussianJoint, MixtureJoint

F64 = torch.float64
D = 3
N_TRAIN, N_VAL = 4000, 2000


def _random_corr(n: int, d: int, gen: torch.Generator) -> torch.Tensor:
    raw = torch.randn(n, d, d + 2, generator=gen, dtype=F64)
    cov = raw @ raw.mT / (d + 2)
    sd = cov.diagonal(dim1=-2, dim2=-1).sqrt()
    return cov / sd.unsqueeze(-1) / sd.unsqueeze(-2)


# ─────────────────────────────────────────────────────────────── exact Gaussian case


class TestSklarJointGaussian:
    """Gaussian marginals + Gaussian scores = a multivariate normal, exactly."""

    def _build(self) -> tuple[SklarJoint, GaussianJoint]:
        gen = torch.Generator().manual_seed(0)
        n, d = 5, 4
        mean = torch.randn(n, d, generator=gen, dtype=F64)
        std = torch.rand(n, d, generator=gen, dtype=F64) + 0.3
        chol = torch.linalg.cholesky(_random_corr(n, d, gen))
        joint = SklarJoint(
            GaussianMarginals(mean, std), GaussianJoint(torch.zeros(n, d, dtype=F64), chol)
        )
        return joint, GaussianJoint(mean, std.unsqueeze(-1) * chol)

    def test_log_prob_both_layouts(self) -> None:
        joint, ref = self._build()
        assert joint.has_density and joint.has_latent and joint.exact_marginals
        mvn = MultivariateNormal(ref.loc, scale_tril=ref.scale_tril)
        torch.manual_seed(1)
        y = ref.sample(7) * 1.5  # include points well into the tails
        torch.testing.assert_close(joint.log_prob(y[:, 0]), mvn.log_prob(y[:, 0]))
        torch.testing.assert_close(joint.log_prob(y), mvn.log_prob(y.movedim(1, 0)).movedim(0, 1))

    def test_masked_log_prob(self) -> None:
        joint, ref = self._build()
        torch.manual_seed(2)
        y = ref.sample(6)
        mask = torch.rand(5, 4) > 0.4
        mask[0] = False  # nothing observed
        y_nan = torch.where(mask.unsqueeze(1), y, torch.full_like(y, float("nan")))
        torch.testing.assert_close(joint.log_prob(y_nan, mask), ref.log_prob(y, mask))
        torch.testing.assert_close(joint.log_prob(y_nan[:, 2], mask), ref.log_prob(y[:, 2], mask))
        torch.testing.assert_close(
            joint.log_prob(y[:, 2], mask)[0].abs(), torch.zeros((), dtype=F64)
        )

    def test_marginal(self) -> None:
        joint, ref = self._build()
        sub, sub_ref = joint.marginal([3, 1]), ref.marginal([3, 1])
        assert isinstance(sub, SklarJoint) and sub.event_dim == 2
        torch.manual_seed(3)
        y = sub_ref.sample(4)
        torch.testing.assert_close(sub.log_prob(y), sub_ref.log_prob(y))

    def test_latent_round_trip_and_sampling(self) -> None:
        joint, ref = self._build()
        torch.manual_seed(4)
        y = ref.sample(5)
        torch.testing.assert_close(joint.to_latent(y), ref.to_latent(y))
        torch.testing.assert_close(joint.from_latent(joint.to_latent(y)), y)
        torch.testing.assert_close(joint.from_latent(joint.to_latent(y[:, 1])), y[:, 1])
        # same random stream: the Sklar draw is the MVN draw
        torch.manual_seed(5)
        a = joint.sample(9)
        torch.manual_seed(5)
        torch.testing.assert_close(a, ref.sample(9))

    def test_rejects_mismatched_shapes(self) -> None:
        with pytest.raises(ValueError):
            SklarJoint(
                GaussianMarginals(torch.zeros(2, 3), torch.ones(2, 3)),
                GaussianJoint(torch.zeros(2, 2), torch.eye(2).to(device="cpu", dtype=F64)),
            )


# ─────────────────────────────────────────────────────────────── change of variables


def _x_mixture_scores(n: int, d: int, rho: float) -> MixtureJoint:
    """Equal mixture of N(0, R(+rho)) and N(0, R(-rho)) on (z1, z2): standard-normal marginals."""
    members = []
    for sign in (1.0, -1.0):
        corr = torch.eye(d, dtype=F64)
        corr[0, 1] = corr[1, 0] = sign * rho
        members.append(GaussianJoint(torch.zeros(n, d, dtype=F64), torch.linalg.cholesky(corr)))
    return MixtureJoint(members)


def _skewed_quantile_marginals(n: int, d: int) -> QuantileMarginals:
    gen = torch.Generator().manual_seed(11)
    levels = torch.linspace(0.02, 0.98, 25, dtype=F64)
    base = Normal(torch.zeros((), dtype=F64), torch.ones((), dtype=F64)).icdf(levels)
    loc = torch.randn(n, d, 1, generator=gen, dtype=F64)
    scale = torch.rand(n, d, 1, generator=gen, dtype=F64) + 0.5
    return QuantileMarginals(levels, loc + scale * (base + 0.3 * base.square()))


class TestChangeOfVariables:
    def test_total_mass_is_one_by_importance_sampling(self) -> None:
        n, d, m = 3, 2, 200_000
        marg = _skewed_quantile_marginals(n, d)
        joint = SklarJoint(marg, _x_mixture_scores(n, d, rho=0.6))
        centre = marg.median()
        width = marg.icdf(torch.full((n, d), 0.75, dtype=F64)) - marg.icdf(
            torch.full((n, d), 0.25, dtype=F64)
        )
        proposal = StudentT(3.0, centre, width)  # heavier tails than the exponential ones
        torch.manual_seed(6)
        y = proposal.sample((m,)).movedim(0, 1)  # [n, m, d]
        log_g = proposal.log_prob(y.movedim(1, 0)).sum(-1).movedim(0, 1)
        log_w = joint.log_prob(y) - log_g
        w = log_w.exp()
        est = w.mean(1)
        se = w.std(1) / math.sqrt(m)
        assert ((est - 1).abs() < 4 * se).all(), (est, se)
        assert (se < 0.02).all()  # the check has power

    def test_draws_have_the_marginals(self) -> None:
        n, d, s = 3, 2, 4000
        marg = _skewed_quantile_marginals(n, d)
        joint = SklarJoint(marg, _x_mixture_scores(n, d, rho=0.8))
        torch.manual_seed(7)
        pit = marg.cdf(joint.sample(s))
        # 6 KS tests of S = 4000 PITs against U(0, 1); Bonferroni level 1e-3 overall
        p_values = [
            kstest(pit[i, :, j].numpy(), "uniform").pvalue for i in range(n) for j in range(d)
        ]
        assert min(p_values) > 1e-3 / len(p_values), p_values

    def test_masked_target_integrates_out(self) -> None:
        # The X-mixture has standard-normal score marginals, so with only target 0
        # observed the joint density is the marginal density f_0(y_0) exactly.
        n, d = 3, 2
        marg = _skewed_quantile_marginals(n, d)
        joint = SklarJoint(marg, _x_mixture_scores(n, d, rho=0.6))
        assert joint.exact_marginals
        torch.manual_seed(8)
        y = joint.sample(5)
        mask = torch.tensor([[True, False]] * n)
        torch.testing.assert_close(joint.log_prob(y, mask), marg.log_prob(y)[..., 0])

    def test_mask_needs_exact_marginals(self) -> None:
        z = torch.randn(200, 2, generator=torch.Generator().manual_seed(2), dtype=F64)
        scores = FlowDependence(max_epochs=0).fit(z).predictive(2)
        joint = SklarJoint(_skewed_quantile_marginals(2, 2), scores)
        assert joint.has_density and not joint.exact_marginals
        with pytest.raises(NotImplementedError):
            joint.log_prob(torch.zeros(2, 2, dtype=F64), torch.ones(2, 2, dtype=torch.bool))
        assert not joint.marginal([1]).has_density  # flows marginalise by sampling


# ─────────────────────────────────────────────────────────────── the ladder


def _loadings(case: str, x: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    """One-factor loadings a(x): z = a f + sqrt(1 - a^2) e has N(0, 1) marginals."""
    t, one = x[:, 0], torch.ones_like(x[:, 0])
    if case == "a":  # independent
        return torch.zeros(len(t), D, dtype=F64)
    if case == "b":  # constant Gaussian copula
        return torch.stack([0.8 * one, 0.7 * one, -0.6 * one], 1)
    if case == "c":  # correlation that depends on x
        return torch.stack([0.9 * t, 0.9 * one, -0.9 * t], 1)
    # case "d": mixture over a hidden sign (non-Gaussian, marginals still N(0, 1))
    return torch.stack([0.9 * one, 0.9 * sign, -0.9 * sign], 1)


def _make_case(case: str, n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 2, generator=gen, dtype=F64) * 2 - 1  # x[:, 1] is irrelevant
    sign = (torch.rand(n, generator=gen, dtype=F64) < 0.5).to(F64) * 2 - 1
    a = _loadings(case, x, sign)
    f = torch.randn(n, 1, generator=gen, dtype=F64)
    z = a * f + (1 - a.square()).sqrt() * torch.randn(n, D, generator=gen, dtype=F64)
    return z, x


def _true_log_q(case: str, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    def one(a: torch.Tensor) -> torch.Tensor:
        cov = a.unsqueeze(-1) * a.unsqueeze(-2) + torch.diag_embed(1 - a.square()).to(dtype=F64)
        return MultivariateNormal(torch.zeros(D, dtype=F64), cov).log_prob(z)

    ones = torch.ones(len(z), dtype=F64)
    if case == "d":
        both = torch.stack([one(_loadings(case, x, ones)), one(_loadings(case, x, -ones))])
        return torch.logsumexp(both, 0) - math.log(2.0)
    return one(_loadings(case, x, ones))


CASES = ("a", "b", "c", "d")
EXPECTED = {
    "a": ("independence",),
    "b": ("gaussian_copula", "conditional_gaussian_copula"),
    "c": ("conditional_gaussian_copula",),
    "d": ("conditional_mixture",),  # two Gaussian regimes: the mixture rung is well specified
}


@pytest.fixture(scope="module")
def ladder() -> dict[str, dict[str, object]]:
    """Fit every rung on every case once (training N = 4000, validation N = 2000)."""
    out: dict[str, dict[str, object]] = {}
    for k, case in enumerate(CASES):
        z, x = _make_case(case, N_TRAIN, seed=10 + k)
        z_val, x_val = _make_case(case, N_VAL, seed=20 + k)
        rungs: list[Dependence] = [
            IndependenceDependence().fit(z),
            GaussianCopula().fit(z),
            ConditionalGaussianCopula().fit(z, x),
            MixtureDependence().fit(z, x),
            FlowDependence().fit(z, x),
        ]
        chosen, table = select_dependence(rungs, z_val, x_val)
        table["truth"] = float(_true_log_q(case, z_val, x_val).mean())
        out[case] = {
            "z": z, "x": x, "z_val": z_val, "x_val": x_val,
            "rungs": rungs, "chosen": chosen, "table": table,
        }  # fmt: skip
    # Printed with -s: the held-out NLL table (rungs x cases) and fit times.
    print("\nheld-out NLL (nats per row), N_train = 4000, N_val = 2000, d = 3")
    for case in CASES:
        table = out[case]["table"]
        assert isinstance(table, dict)
        rungs = out[case]["rungs"]
        assert isinstance(rungs, list)
        cells = "  ".join(f"{name}={-v:.4f}" for name, v in table.items())
        times = "  ".join(f"{r.name}={r.fit_time_:.2f}s" for r in rungs)
        print(f"case {case}: {cells}  -> {out[case]['chosen'].name}  | fit: {times}")  # type: ignore[union-attr]
    return out


class TestLadder:
    @pytest.mark.parametrize("case", CASES)
    def test_selection_picks_the_right_rung(
        self, ladder: dict[str, dict[str, object]], case: str
    ) -> None:
        chosen = ladder[case]["chosen"]
        assert isinstance(chosen, Dependence)
        assert chosen.name in EXPECTED[case], ladder[case]["table"]
        table = ladder[case]["table"]
        assert isinstance(table, dict)
        assert table[chosen.name] >= table["independence"]

    @pytest.mark.parametrize(
        ("case", "rung", "gap"),
        [
            ("a", "independence", 0.0),
            ("b", "gaussian_copula", 0.01),
            ("c", "conditional_gaussian_copula", 0.03),
        ],
    )
    def test_well_specified_rung_reaches_the_true_nll(
        self, ladder: dict[str, dict[str, object]], case: str, rung: str, gap: float
    ) -> None:
        # Held-out mean log q minus the true mean log q on the same N = 2000 rows:
        # the estimation error of the paired difference is far below these gaps
        # for a consistent fit (it is O(d^2 / N) for the Gaussian rungs).
        table = ladder[case]["table"]
        assert isinstance(table, dict)
        assert table["truth"] - table[rung] <= gap + 1e-12, table

    def test_flow_captures_non_gaussian_dependence(
        self, ladder: dict[str, dict[str, object]]
    ) -> None:
        table = ladder["d"]["table"]
        assert isinstance(table, dict)
        # better than the best Gaussian rung by a wide margin, within 0.1 nat of the truth
        assert table["flow"] - table["gaussian_copula"] > 0.15
        assert table["truth"] - table["flow"] < 0.1

    def test_mixture_rung_is_well_specified_for_two_regimes(
        self, ladder: dict[str, dict[str, object]]
    ) -> None:
        table = ladder["d"]["table"]
        assert isinstance(table, dict)
        # the truth is a two-component Gaussian mixture: within 0.03 nat of it on the
        # 2,000 held-out rows, and ahead of the flow, which needs more rows
        assert table["truth"] - table["conditional_mixture"] < 0.03, table
        assert table["conditional_mixture"] > table["flow"], table
        # on Gaussian dependence it costs little against the well-specified rungs
        for case, rung in (("b", "gaussian_copula"), ("c", "conditional_gaussian_copula")):
            other = ladder[case]["table"]
            assert isinstance(other, dict)
            assert other[rung] - other["conditional_mixture"] < 0.02, other

    def test_mixture_rung_keeps_exact_marginals_and_masks(self) -> None:
        z, x = _make_case("d", 600, seed=30)
        rung = MixtureDependence(max_epochs=30).fit(z, x)
        scores = rung.predictive(50, x[:50])
        assert scores.has_density and scores.exact_marginals
        marginals = GaussianMarginals(torch.zeros(50, D, dtype=F64), torch.ones(50, D, dtype=F64))
        joint = SklarJoint(marginals, scores)
        assert joint.has_density and joint.exact_marginals
        mask = torch.ones(50, D, dtype=torch.bool)
        mask[:, 1] = False
        sub = joint.marginal([0, 2])
        assert sub.has_density
        torch.testing.assert_close(joint.log_prob(z[:50], mask), sub.log_prob(z[:50][:, [0, 2]]))
        # unconditional fit: one mixture for every row
        const = MixtureDependence(max_epochs=30).fit(z)
        assert not const.conditional
        lp = const.log_prob(z[:20])
        assert lp.shape == (20,) and bool(torch.isfinite(lp).all())
        with pytest.raises(ValueError, match="n_components"):
            MixtureDependence(n_components=0)

    def test_gaussian_copula_recovers_the_correlation(
        self, ladder: dict[str, dict[str, object]]
    ) -> None:
        fitted = ladder["b"]["rungs"][1]  # type: ignore[index]
        assert isinstance(fitted, GaussianCopula) and fitted.correlation_ is not None
        a = _loadings("b", torch.zeros(1, 2, dtype=F64), torch.ones(1, dtype=F64))[0]
        truth = torch.outer(a, a).fill_diagonal_(1.0)
        # SE of a sample correlation ~ (1 - rho^2) / sqrt(N); 4 SE plus the shrinkage bias
        tol = 4 * (1 - truth.square()) / math.sqrt(N_TRAIN) + fitted.shrinkage_ * truth.abs()
        assert ((fitted.correlation_ - truth).abs() <= tol + 1e-12).all()

    def test_conditional_copula_recovers_the_correlation_in_x(
        self, ladder: dict[str, dict[str, object]]
    ) -> None:
        fitted = ladder["c"]["rungs"][2]  # type: ignore[index]
        assert isinstance(fitted, ConditionalGaussianCopula)
        x0 = torch.tensor([[-0.8, 0.0], [-0.3, 0.0], [0.3, 0.0], [0.8, 0.0]], dtype=F64)
        a = _loadings("c", x0, torch.ones(4, dtype=F64))
        truth = a.unsqueeze(-1) * a.unsqueeze(-2)
        cov = fitted.predictive(4, x0).covariance()
        sd = cov.diagonal(dim1=-2, dim2=-1).sqrt()
        corr = cov / sd.unsqueeze(-1) / sd.unsqueeze(-2)
        off = ~torch.eye(D, dtype=torch.bool)
        # neural estimate, no closed-form SE: absolute tolerance 0.1 on |rho| up to 0.81
        assert ((corr - truth)[:, off].abs() < 0.1).all(), (corr, truth)

    @pytest.mark.parametrize("case", CASES)
    def test_marginal_drift_is_small(self, ladder: dict[str, dict[str, object]], case: str) -> None:
        z_val, x_val = ladder[case]["z_val"], ladder[case]["x_val"]
        assert isinstance(z_val, torch.Tensor) and isinstance(x_val, torch.Tensor)
        z_val, x_val = z_val[:500], x_val[:500]
        for rung in ladder[case]["rungs"]:  # type: ignore[attr-defined]
            torch.manual_seed(8)
            drift = rung.marginal_drift(z_val, x_val if rung.conditional else None)
            assert set(drift) == {"mean", "std", "score_mean", "score_std"}
            # scores are N(0, 1) by construction; a fitted q should not move them much
            assert drift["mean"].abs().max() < 0.1, (rung.name, drift)
            assert (drift["std"] - 1).abs().max() < 0.1, (rung.name, drift)


class TestIdentityInitialisation:
    @pytest.mark.parametrize("d", [2, 32])
    @pytest.mark.parametrize("cls", [ConditionalGaussianCopula, FlowDependence])
    def test_untrained_rung_is_independence(self, cls: type, d: int) -> None:
        gen = torch.Generator().manual_seed(9)
        z = torch.randn(300, d, generator=gen, dtype=F64)
        x = torch.randn(300, 4, generator=gen, dtype=F64)
        ref = IndependenceDependence().fit(z).log_prob(z)
        rung = cls(max_epochs=0).fit(z, x)
        torch.testing.assert_close(rung.log_prob(z, x), ref, rtol=0, atol=1e-10)
        assert rung.training_["initial_val_nll"] == rung.training_["best_val_nll"]

    def test_unconditional_flow_starts_at_identity(self) -> None:
        z = torch.randn(200, 3, generator=torch.Generator().manual_seed(1), dtype=F64)
        rung = FlowDependence(max_epochs=0).fit(z)
        assert not rung.conditional
        pred = rung.predictive(200)
        assert isinstance(pred, FlowJoint)
        torch.testing.assert_close(pred.to_latent(z), z, rtol=0, atol=1e-10)
        ref = IndependenceDependence().fit(z).log_prob(z)
        torch.testing.assert_close(rung.log_prob(z), ref, rtol=0, atol=1e-10)


class TestLargeD:
    @pytest.mark.parametrize("d", [2, 32])
    def test_every_rung_runs(self, d: int) -> None:
        gen = torch.Generator().manual_seed(12)
        n = 800
        x = torch.randn(n, 3, generator=gen, dtype=F64)
        corr = _random_corr(1, d, gen)[0]
        z = torch.randn(n, d, generator=gen, dtype=F64) @ torch.linalg.cholesky(corr).mT
        rungs: list[Dependence] = [
            IndependenceDependence().fit(z),
            GaussianCopula().fit(z),
            GaussianCopula(rank=2).fit(z),
            ConditionalGaussianCopula(max_epochs=5).fit(z, x),
            FlowDependence(max_epochs=2, hidden=(32, 32)).fit(z, x),
            SchaakeShuffle(n_neighbors=50).fit(z, x),
        ]
        torch.manual_seed(13)
        for rung in rungs:
            pred = rung.predictive(4, x[:4])
            assert pred.sample(6).shape == (4, 6, d)
            if rung.has_density:
                lp = pred.log_prob(z[:4])
                assert lp.shape == (4,) and torch.isfinite(lp).all()
        chosen, table = select_dependence(rungs, z, x)
        assert "schaake" not in table and chosen.has_density
        assert table["gaussian_copula"] > table["independence"]  # the correlation is real

    def test_low_rank_copula_has_unit_diagonal(self) -> None:
        gen = torch.Generator().manual_seed(14)
        factor = torch.randn(20, 2, generator=gen, dtype=F64) * 0.6
        z = torch.randn(3000, 2, generator=gen, dtype=F64) @ factor.mT + torch.randn(
            3000, 20, generator=gen, dtype=F64
        )
        rung = GaussianCopula(rank=2).fit(z)
        full = GaussianCopula().fit(z)
        assert rung.correlation_ is not None and full.correlation_ is not None
        torch.testing.assert_close(rung.correlation_.diagonal(), torch.ones(20, dtype=F64))
        # a two-factor truth: the rank-2 fit is close to the full shrunk correlation
        assert (rung.correlation_ - full.correlation_).abs().max() < 0.1


class TestSchaakeShuffle:
    def test_template_has_standard_normal_marginals(self) -> None:
        z, x = _make_case("b", N_TRAIN, seed=30)
        rung = SchaakeShuffle().fit(z * 3.0 + 1.0)  # only ranks matter
        template = rung.template_
        assert template is not None
        grid = torch.special.ndtri(torch.arange(1, N_TRAIN + 1, dtype=F64) / (N_TRAIN + 1))
        torch.testing.assert_close(
            torch.sort(template, dim=0).values, grid.unsqueeze(1).expand(-1, D)
        )
        for j in range(D):
            assert kstest(template[:, j].numpy(), "norm").pvalue > 0.5  # a deterministic grid

    def test_draws_reproduce_training_rank_correlation(self) -> None:
        z, _ = _make_case("b", N_TRAIN, seed=31)
        rung = SchaakeShuffle().fit(z)
        pred = rung.predictive(1)
        assert isinstance(pred, TemplateJoint) and not rung.has_density
        torch.manual_seed(15)
        draws = pred.sample(20_000)[0]
        rho_train = torch.as_tensor(spearmanr(z.numpy()).statistic)
        rho_draws = torch.as_tensor(spearmanr(draws.numpy()).statistic)
        # resampling S = 20000 rows: SE of a rank correlation <= 1 / sqrt(S) = 0.007; 4 SE
        assert (rho_draws - rho_train).abs().max() < 4 / math.sqrt(20_000)

    def test_neighbours_follow_x(self) -> None:
        z, x = _make_case("c", N_TRAIN, seed=32)
        rung = SchaakeShuffle(n_neighbors=300).fit(z, x)
        x0 = torch.tensor([[-0.9, 0.0], [0.9, 0.0]], dtype=F64)
        pred = rung.predictive(2, x0)
        torch.manual_seed(16)
        draws = pred.sample(5000)
        rho = [float(spearmanr(draws[i, :, :2].numpy()).statistic) for i in range(2)]
        # true local Pearson correlation of (z1, z2) is 0.81 x: about -0.7 and +0.7 here
        assert rho[0] < -0.5 and rho[1] > 0.5, rho
        # exact moments of the template rows
        rows = rung.template_[pred.index[0]]  # type: ignore[index]
        torch.testing.assert_close(pred.mean()[0], rows.mean(0))

    def test_sklar_joint_with_schaake_is_draws_only(self) -> None:
        z, _ = _make_case("b", 500, seed=33)
        marg = _skewed_quantile_marginals(2, D)
        joint = SklarJoint(marg, SchaakeShuffle().fit(z).predictive(2))
        assert not joint.has_density
        torch.manual_seed(17)
        assert joint.sample(10).shape == (2, 10, D)
        with pytest.raises(NotImplementedError):
            joint.log_prob(torch.zeros(2, D, dtype=F64))


class TestSelection:
    def test_adds_independence_and_skips_draws_only(self) -> None:
        z, x = _make_case("a", 600, seed=40)
        cands: list[Dependence] = [GaussianCopula(shrinkage=0.0).fit(z), SchaakeShuffle().fit(z)]
        chosen, table = select_dependence(cands, z, x)
        assert list(table) == ["independence", "gaussian_copula"]
        # in-sample, the unshrunk copula always scores at least as well; within 2 SE it
        # is not resolved, so the lowest rung is kept
        assert isinstance(chosen, IndependenceDependence)

    def test_argmax_with_zero_tolerance(self) -> None:
        z, x = _make_case("b", 600, seed=41)
        cands: list[Dependence] = [GaussianCopula().fit(z), GaussianCopula(shrinkage=0.5).fit(z)]
        chosen, table = select_dependence(cands, z, x, n_se=0.0)
        assert list(table) == ["independence", "gaussian_copula", "gaussian_copula#2"]
        assert chosen is cands[0]


class TestAffineRungs:
    """Marginal miscalibration must not be read as dependence by the gate."""

    @staticmethod
    def _scores(n: int, rho: float, loc: float, scale: float, seed: int) -> torch.Tensor:
        gen = torch.Generator().manual_seed(seed)
        corr = torch.full((D, D), rho, dtype=F64)
        corr.fill_diagonal_(1.0)
        z = torch.randn(n, D, generator=gen, dtype=F64) @ torch.linalg.cholesky(corr).mT
        return loc + scale * z

    @staticmethod
    def _ladder(z: torch.Tensor) -> list[Dependence]:
        return [
            IndependenceDependence().fit(z),
            IndependenceDependence(affine=True).fit(z),
            GaussianCopula().fit(z),
            GaussianCopula(affine=True).fit(z),
            FlowDependence(max_epochs=60, seed=0).fit(z),
        ]

    def test_density_is_the_affine_gaussian(self) -> None:
        z = self._scores(3000, 0.6, 0.3, 1.5, seed=50)
        rung = GaussianCopula(shrinkage=0.0, affine=True).fit(z)
        assert rung.loc_ is not None and rung.scale_ is not None and rung.correlation_ is not None
        cov = rung.scale_.unsqueeze(-1) * rung.correlation_ * rung.scale_
        ref = MultivariateNormal(rung.loc_, covariance_matrix=cov).log_prob(z)
        torch.testing.assert_close(rung.log_prob(z), ref, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(rung.loc_, z.mean(0))
        torch.testing.assert_close(rung.scale_, z.std(0, unbiased=False))
        ind = IndependenceDependence(affine=True).fit(z)
        ref_ind = Normal(z.mean(0), z.std(0, unbiased=False)).log_prob(z).sum(-1)
        torch.testing.assert_close(ind.log_prob(z), ref_ind, rtol=1e-9, atol=1e-9)

    def test_plain_rungs_are_unchanged(self) -> None:
        z = self._scores(500, 0.5, 0.2, 1.3, seed=51)
        ind = IndependenceDependence().fit(z)
        torch.testing.assert_close(ind.log_prob(z), Normal(0.0, 1.0).log_prob(z).sum(-1))
        cop = GaussianCopula().fit(z)
        assert cop.name == "gaussian_copula" and ind.name == "independence"
        drift = cop.marginal_drift(z)
        torch.testing.assert_close(drift["mean"], torch.zeros(D, dtype=F64))
        torch.testing.assert_close(drift["std"], torch.ones(D, dtype=F64))

    def test_overdispersed_independent_scores_select_affine_independence(self) -> None:
        # over-dispersed, biased marginals and no dependence: the cheapest rung that
        # corrects the marginals is chosen, not the flow
        z_fit = self._scores(3000, 0.0, 0.25, 1.4, seed=52)
        z_val = self._scores(1500, 0.0, 0.25, 1.4, seed=53)
        chosen, table = select_dependence(self._ladder(z_fit), z_val)
        assert chosen.name == "affine_independence", table
        assert table["affine_independence"] > table["independence"] + 0.1
        # without the affine rungs the flow is chosen for its marginal correction alone
        plain = [c for c in self._ladder(z_fit) if not getattr(c, "affine", False)]
        chosen_plain, table_plain = select_dependence(plain, z_val)
        assert chosen_plain.name == "flow", table_plain

    def test_calibrated_correlated_scores_select_the_plain_copula(self) -> None:
        z_fit = self._scores(3000, 0.6, 0.0, 1.0, seed=54)
        z_val = self._scores(1500, 0.6, 0.0, 1.0, seed=55)
        chosen, table = select_dependence(self._ladder(z_fit), z_val)
        assert chosen.name == "gaussian_copula", table

    def test_miscalibrated_correlated_scores_select_the_affine_copula(self) -> None:
        z_fit = self._scores(3000, 0.6, 0.25, 1.4, seed=56)
        z_val = self._scores(1500, 0.6, 0.25, 1.4, seed=57)
        chosen, table = select_dependence(self._ladder(z_fit), z_val)
        assert chosen.name == "affine_gaussian_copula", table
        assert list(table) == [
            "independence",
            "affine_independence",
            "gaussian_copula",
            "affine_gaussian_copula",
            "flow",
        ]
