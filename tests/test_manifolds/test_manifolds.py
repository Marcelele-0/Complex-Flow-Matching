"""Tests for the manifold seam.

Two kinds of assertion live here. The parametrized ones pin the *contract* every
geometry must satisfy, so a third manifold cannot be added half-wired. The
comparison ones pin the properties the cylindrical-vs-Euclidean experiment
depends on: that the two carry the same complex signal, start from the same
noise, and integrate on the same schedule.
"""

import math

import pytest
import torch
from omegaconf import OmegaConf

from cyfm.config.adapters import manifold_from_config
from cyfm.core.manifold import BaseManifold
from cyfm.data.transforms import slice_transform, window_transforms
from cyfm.manifolds import (
    CylindricalManifold,
    EuclideanManifold,
)
from cyfm.manifolds.cylindrical import sample_cylindrical_noise
from cyfm.manifolds.euclidean import sample_matched_noise

CPU = torch.device("cpu")
monkey: dict = {}
MANIFOLDS = [CylindricalManifold, EuclideanManifold]


def _complex_slice(h: int, w: int, seed: int = 0) -> torch.Tensor:
    """A synthetic complex MRI-like slice, [1, H, W], with a bright region and air."""
    torch.manual_seed(seed)
    amp = torch.rand(1, h, w) * 0.2
    amp[:, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] += 3.0  # "tissue"
    phi = (torch.rand(1, h, w) * 2 - 1) * math.pi
    return torch.polar(amp, phi)


class TestRegistry:
    def test_defaults_to_cylindrical(self) -> None:
        """A config predating the manifold group must behave exactly as it did."""
        manifold = manifold_from_config(OmegaConf.create({}))
        assert isinstance(manifold, CylindricalManifold)

    @pytest.mark.parametrize(
        ("name", "expected"),
        [("cylindrical", CylindricalManifold), ("euclidean", EuclideanManifold)],
    )
    def test_builds_the_named_geometry(self, name: str, expected: type) -> None:
        manifold = manifold_from_config(OmegaConf.create({"manifold": {"name": name}}))
        assert isinstance(manifold, expected)

    def test_unknown_name_raises(self) -> None:
        cfg = OmegaConf.create({"manifold": {"name": "klein_bottle"}})
        with pytest.raises(ValueError, match="Unknown manifold name"):
            manifold_from_config(cfg)

    def test_loss_config_reaches_both_geometries_from_training_loss(self) -> None:
        """`training.loss.*` overrides must keep working across the manifold switch."""
        loss_cfg = {"training": {"loss": {"phase_loss_type": "l2", "vel_loss_type": "l2"}}}

        cyl = manifold_from_config(
            OmegaConf.create({"manifold": {"name": "cylindrical"}} | loss_cfg)
        )
        euc = manifold_from_config(OmegaConf.create({"manifold": {"name": "euclidean"}} | loss_cfg))

        # build_manifold is typed to the ABC, which owns no loss object. Narrowing
        # to the concrete geometry makes the reads below checkable, and pins that
        # the registry returned the class this test thinks it did.
        assert isinstance(cyl, CylindricalManifold)
        assert isinstance(euc, EuclideanManifold)

        # Each geometry reads the key that applies to it and ignores the other's.
        assert cyl._loss.phase_loss_type == "l2"
        assert euc._loss.loss_type == "l2"

    def test_euclidean_reads_its_own_loss_and_prior_keys(self) -> None:
        cfg = OmegaConf.create(
            {
                "manifold": {"name": "euclidean", "noise_prior": "gaussian"},
                "training": {"loss": {"vel_loss_type": "mse"}},
            }
        )
        manifold = manifold_from_config(cfg)

        assert isinstance(manifold, EuclideanManifold)
        assert manifold.noise_prior == "gaussian"
        assert manifold._loss.loss_type == "mse"

    def test_cylindrical_reads_phase_amplitude_weighting(self) -> None:
        default = manifold_from_config(OmegaConf.create({"manifold": {"name": "cylindrical"}}))
        off = manifold_from_config(
            OmegaConf.create(
                {
                    "manifold": {"name": "cylindrical"},
                    "training": {"loss": {"phase_amplitude_weighting": False}},
                }
            )
        )

        assert isinstance(default, CylindricalManifold)
        assert isinstance(off, CylindricalManifold)
        assert default._loss.phase_amplitude_weighting is True
        assert off._loss.phase_amplitude_weighting is False

    def test_unknown_noise_prior_raises(self) -> None:
        with pytest.raises(ValueError, match="noise_prior must be one of"):
            EuclideanManifold(noise_prior="cauchy")


@pytest.mark.parametrize("manifold_cls", MANIFOLDS)
class TestManifoldContract:
    """Every geometry must satisfy these, or an entry point will break on it."""

    def test_is_a_manifold_with_a_declared_width(self, manifold_cls: type) -> None:
        manifold = manifold_cls()
        assert isinstance(manifold, BaseManifold)
        assert manifold.state_channels in (2, 3)
        assert manifold.velocity_channels == 2

    def test_transform_produces_state_channels(self, manifold_cls: type) -> None:
        manifold = manifold_cls()
        state = slice_transform(manifold, crop_base=16)(_complex_slice(40, 40))
        # 40 -> 32 under the modulo-16 crop.
        assert state.shape == (manifold.state_channels, 32, 32)

    def test_transform_normalises_the_modulus_into_the_unit_interval(
        self, manifold_cls: type
    ) -> None:
        """PSNR's data_range=1.0 is only meaningful if this holds for every geometry."""
        manifold = manifold_cls()
        state = slice_transform(manifold, crop_base=16)(_complex_slice(32, 32, seed=1))
        modulus = torch.abs(manifold.to_complex(state.unsqueeze(0)))

        assert torch.all(modulus >= 0.0)
        assert torch.all(modulus <= 1.0 + 1e-6)
        assert modulus.max() > 0.9, "peak should reach the top of the range"

    def test_window_transforms_split_representation_from_normalisation(
        self, manifold_cls: type
    ) -> None:
        """The 2.5D pair must normalise the whole stack by ONE peak.

        Normalising per slice instead would divide each slice by its own maximum
        and destroy the relative brightness between neighbours - which is the
        signal cross-slice attention exists to read.
        """
        manifold = manifold_cls()
        slice_tf, window_tf = window_transforms(manifold, crop_base=16)

        # Deliberately unequal slice brightness, brightest last.
        stack = torch.stack([slice_tf(_complex_slice(40, 40, seed=i) * (i + 1)) for i in range(3)])
        out = window_tf(stack)

        assert out.shape == (3, manifold.state_channels, 32, 32)

        modulus = manifold.to_complex(out).abs()
        assert modulus.max().item() == pytest.approx(1.0, abs=1e-5)

        # Exactly one slice reaches the peak; the dimmer ones stay strictly below.
        per_slice = [modulus[i].max().item() for i in range(3)]
        assert sum(p > 1.0 - 1e-5 for p in per_slice) == 1
        assert per_slice[2] > per_slice[1] > per_slice[0]

    def test_noise_has_the_state_shape(self, manifold_cls: type) -> None:
        manifold = manifold_cls()
        noise = manifold.sample_noise(4, 8, 8, CPU)
        assert noise.shape == (4, manifold.state_channels, 8, 8)

    def test_bridge_endpoints_and_velocity_width(self, manifold_cls: type) -> None:
        manifold = manifold_cls()
        x_0 = manifold.sample_noise(2, 8, 8, CPU)
        x_1 = manifold.sample_noise(2, 8, 8, CPU)

        x_t, u = manifold.bridge(x_0, x_1, torch.rand(2, 1, 1, 1))
        assert x_t.shape == x_0.shape
        assert u.shape == (2, manifold.velocity_channels, 8, 8)

        # At t=0 the complex signal must be the noise; at t=1, the data. Compared
        # in the complex domain so the assertion means the same on both geometries.
        at_0, _ = manifold.bridge(x_0, x_1, torch.zeros(2, 1, 1, 1))
        at_1, _ = manifold.bridge(x_0, x_1, torch.ones(2, 1, 1, 1))
        assert torch.allclose(manifold.to_complex(at_0), manifold.to_complex(x_0), atol=1e-5)
        assert torch.allclose(manifold.to_complex(at_1), manifold.to_complex(x_1), atol=1e-5)

    def test_loss_returns_a_named_breakdown_summing_sensibly(self, manifold_cls: type) -> None:
        manifold = manifold_cls()
        x_1 = manifold.sample_noise(2, 8, 8, CPU)
        pred = torch.randn(2, manifold.velocity_channels, 8, 8)
        target = torch.randn(2, manifold.velocity_channels, 8, 8)

        total, components = manifold.loss(pred, target, target_x1=x_1)

        assert total.dim() == 0
        assert components, "every geometry must expose at least one named component"
        assert all(v.dim() == 0 for v in components.values())
        # train.py logs these as step_loss_{name}; a name with a space or slash
        # would produce a malformed W&B series.
        assert all(name.isidentifier() for name in components)

    def test_solver_step_keeps_the_state_width(self, manifold_cls: type) -> None:
        manifold = manifold_cls()
        solver = manifold.make_solver(num_steps=4)
        x_t = manifold.sample_noise(1, 8, 8, CPU)
        v_t = torch.randn(1, manifold.velocity_channels, 8, 8) * 0.1

        assert solver.step(x_t, v_t, dt=0.25).shape == x_t.shape

    def test_to_complex_shape_and_dtype(self, manifold_cls: type) -> None:
        manifold = manifold_cls()
        z = manifold.to_complex(manifold.sample_noise(3, 8, 8, CPU))
        assert z.shape == (3, 1, 8, 8)
        assert torch.is_complex(z)

    def test_to_is_chainable(self, manifold_cls: type) -> None:
        manifold = manifold_cls()
        assert manifold.to(CPU) is manifold


class TestSideBySideFairness:
    """The properties that make the two arms of the comparison comparable."""

    def test_both_pipelines_carry_the_identical_complex_signal(self) -> None:
        """The two transforms must differ in coordinates only, never in content.

        If this drifts, the two models are no longer being trained on the same
        images and no reported gap means anything.
        """
        slice_ = _complex_slice(48, 48, seed=2)

        cyl = CylindricalManifold()
        euc = EuclideanManifold()

        cyl_state = slice_transform(cyl, crop_base=16)(slice_.clone())
        euc_state = slice_transform(euc, crop_base=16)(slice_.clone())

        cyl_z = cyl.to_complex(cyl_state.unsqueeze(0))
        euc_z = euc.to_complex(euc_state.unsqueeze(0))

        assert torch.allclose(cyl_z.abs(), euc_z.abs(), atol=1e-5)
        assert torch.allclose(cyl_z.real, euc_z.real, atol=1e-5)
        assert torch.allclose(cyl_z.imag, euc_z.imag, atol=1e-5)

    def test_both_window_pipelines_carry_the_identical_complex_signal(self) -> None:
        """The 2D parity above must survive the move to slice windows.

        A 2.5D comparison is only a comparison of geometries if the two arms hand
        the cross-slice model the same complex signal, scaled by the same window
        peak. The gap is float32 round-off in the cylindrical atan2 round trip.
        """
        slices = [_complex_slice(40, 40, seed=i) * (i + 1) for i in range(3)]

        def run(manifold: BaseManifold) -> torch.Tensor:
            slice_tf, window_tf = window_transforms(manifold, crop_base=16)
            return window_tf(torch.stack([slice_tf(z.clone()) for z in slices]))

        cyl, euc = CylindricalManifold(), EuclideanManifold()
        cyl_z = cyl.to_complex(run(cyl))
        euc_z = euc.to_complex(run(euc))

        assert cyl_z.shape == euc_z.shape
        assert torch.allclose(cyl_z.real, euc_z.real, atol=1e-5)
        assert torch.allclose(cyl_z.imag, euc_z.imag, atol=1e-5)

    def test_default_euclidean_prior_is_free_of_trigonometry(self) -> None:
        """The default prior must contain no trigonometry.

        The specification forbids trigonometric embedding outright, so `matched`
        - which converts a polar draw with cos/sin - cannot be the default. This
        test exists so that decision cannot drift back silently.
        """
        assert EuclideanManifold().noise_prior == "uniform"
        built = manifold_from_config(OmegaConf.create({"manifold": {"name": "euclidean"}}))
        assert isinstance(built, EuclideanManifold)
        assert built.noise_prior == "uniform"

        # torch.randn only: sampling the default prior must call no trig op.
        called: list[str] = []
        for op in ("cos", "sin", "atan2", "angle", "polar"):
            original = getattr(torch, op)

            def spy(*args, _op=op, _orig=original, **kwargs):
                called.append(_op)
                return _orig(*args, **kwargs)

            monkey[op] = (original, spy)
            setattr(torch, op, spy)
        try:
            EuclideanManifold().sample_noise(2, 8, 8, CPU)
        finally:
            for op, (original, _) in monkey.items():
                setattr(torch, op, original)
        assert called == [], f"default prior used trigonometry: {sorted(set(called))}"

    def test_default_prior_matches_the_cylindrical_distribution(self) -> None:
        """Both arms must transport from the same law, or Table 1 varies two things.

        The paper claims the *geometry* is what helps. If the Euclidean arm also
        started from a different prior, a gap could not be attributed to geometry
        alone. `uniform` draws modulus ~U[0,1] with a uniform argument - the
        cylindrical prior's distribution - using only rand/randn, so the two are
        distributionally identical while the Euclidean path stays trig-free.
        """
        gen_a = torch.Generator(device="cpu").manual_seed(11)
        gen_b = torch.Generator(device="cpu").manual_seed(22)

        euc = EuclideanManifold().sample_noise(64, 32, 32, CPU, gen_a)
        cyl = sample_cylindrical_noise(64, 32, 32, CPU, gen_b)

        euc_mod = torch.sqrt(euc[:, 0] ** 2 + euc[:, 1] ** 2).flatten()
        cyl_mod = cyl[:, 0].flatten()

        # Modulus is U[0, 1] on both sides: mean 1/2, std 1/sqrt(12), inside the disc.
        assert torch.all(euc_mod <= 1.0 + 1e-6)
        assert abs(float(euc_mod.mean()) - float(cyl_mod.mean())) < 0.01
        assert abs(float(euc_mod.std()) - float(cyl_mod.std())) < 0.01
        assert abs(float(euc_mod.mean()) - 0.5) < 0.01
        assert abs(float(euc_mod.std()) - (1 / 12) ** 0.5) < 0.01

        # Argument is uniform on both sides. atan2 appears here, in the test only -
        # measuring an angle is not the same as embedding one in the model.
        bins = 12
        euc_hist = torch.histc(torch.atan2(euc[:, 1], euc[:, 0]).flatten(), bins, -math.pi, math.pi)
        cyl_hist = torch.histc(torch.atan2(cyl[:, 2], cyl[:, 1]).flatten(), bins, -math.pi, math.pi)
        euc_hist = euc_hist / euc_hist.sum()
        cyl_hist = cyl_hist / cyl_hist.sum()
        assert torch.allclose(euc_hist, cyl_hist, atol=0.01)
        assert torch.allclose(euc_hist, torch.full((bins,), 1.0 / bins), atol=0.01)

    def test_matched_prior_reproduces_the_cylindrical_one(self) -> None:
        """One seed must give both geometries the same complex noise field.

        The matched prior repeats the cylindrical prior's two torch.rand calls in
        the same order, so nothing in a side-by-side result can be attributed to
        the choice of prior.

        The two agree to float32 round-off rather than exactly, hence the 1e-6
        tolerance: both geometries draw the same amplitude and phase, but the
        cylindrical state gets back to a complex number through atan2 and a second
        cos/sin, while the matched prior never leaves (Re, Im). Measured gap is
        ~1e-7.
        """
        gen_a = torch.Generator(device="cpu").manual_seed(1234)
        gen_b = torch.Generator(device="cpu").manual_seed(1234)

        cyl_noise = sample_cylindrical_noise(3, 8, 8, CPU, gen_a)
        euc_noise = sample_matched_noise(3, 8, 8, CPU, gen_b)

        cyl_z = CylindricalManifold().to_complex(cyl_noise)
        euc_z = EuclideanManifold(noise_prior="matched").to_complex(euc_noise)

        assert torch.allclose(cyl_z.abs(), euc_z.abs(), atol=1e-6)
        assert torch.allclose(cyl_z.real, euc_z.real, atol=1e-6)
        assert torch.allclose(cyl_z.imag, euc_z.imag, atol=1e-6)

    def test_matched_prior_lands_in_the_unit_disc(self) -> None:
        noise = sample_matched_noise(8, 16, 16, CPU)
        modulus = torch.sqrt(noise[:, 0:1] ** 2 + noise[:, 1:2] ** 2)

        assert torch.all(modulus <= 1.0 + 1e-6)
        assert modulus.max() > 0.9

    def test_gaussian_prior_is_not_confined_to_the_disc(self) -> None:
        """The textbook prior transports from a genuinely different support.

        That is the honest difference between the three options: `matched` pairs
        the two arms on one noise field, `uniform` (the default) matches the
        cylindrical prior's law without trigonometry, and `gaussian` matches
        neither - it is kept as the "what the literature does" ablation.
        """
        manifold = EuclideanManifold(noise_prior="gaussian")
        gen = torch.Generator(device="cpu").manual_seed(0)
        noise = manifold.sample_noise(4, 32, 32, CPU, gen)
        modulus = torch.sqrt(noise[:, 0:1] ** 2 + noise[:, 1:2] ** 2)

        assert torch.any(modulus > 1.0)

    @pytest.mark.parametrize("model_name", ["c_unet"])
    def test_only_the_input_width_differs_between_the_two_architectures(
        self, model_name: str
    ) -> None:
        """Same trunk, same parameter count apart from one convolution."""
        from cyfm.utils.inference import build_model

        cfg = OmegaConf.create(
            {
                "model": {
                    "name": model_name,
                    "base_channels": 8,
                }
            }
        )
        cyl_model = build_model(cfg, CPU, in_channels=3, out_channels=2)
        euc_model = build_model(cfg, CPU, in_channels=2, out_channels=2)

        cyl_params = {n: tuple(p.shape) for n, p in cyl_model.named_parameters()}
        euc_params = {n: tuple(p.shape) for n, p in euc_model.named_parameters()}

        assert cyl_params.keys() == euc_params.keys()
        differing = {n for n in cyl_params if cyl_params[n] != euc_params[n]}
        assert differing == {"init_conv.weight"}

    @pytest.mark.parametrize("model_name", ["c_unet"])
    def test_one_seed_gives_both_arms_identical_weights_outside_init_conv(
        self, model_name: str
    ) -> None:
        """The two arms must start training from the same weights, not merely the
        same architecture.

        ``init_conv`` is the one module whose shape depends on the manifold, and
        it is constructed LAST in both models precisely so that every other module
        has already drawn from the RNG at the same stream position. Move it back
        up and this test fails: the differing element count shifts every
        subsequent draw, so the two arms would diverge in every layer and a
        one-seed comparison would be measuring initialisation as much as geometry.

        The two tensors that do differ both belong to init_conv, and the bias is
        included for a non-obvious reason: ``kaiming_uniform_`` on the weight
        consumes a different number of uniforms per manifold, which shifts the
        stream before the bias is drawn *inside the same module*. That divergence
        is therefore contained to this one convolution, which is the point.
        """
        from cyfm.utils.inference import build_model

        cfg = OmegaConf.create(
            {
                "model": {
                    "name": model_name,
                    "base_channels": 8,
                }
            }
        )
        torch.manual_seed(1234)
        cyl_model = build_model(cfg, CPU, in_channels=3, out_channels=2)
        torch.manual_seed(1234)
        euc_model = build_model(cfg, CPU, in_channels=2, out_channels=2)

        cyl_params = dict(cyl_model.named_parameters())
        euc_params = dict(euc_model.named_parameters())

        not_identical = {
            name
            for name, tensor in cyl_params.items()
            if tensor.shape != euc_params[name].shape or not torch.equal(tensor, euc_params[name])
        }
        assert not_identical == {"init_conv.weight", "init_conv.bias"}, (
            "every parameter outside init_conv must be element-wise identical; "
            f"these also differ: {sorted(not_identical - {'init_conv.weight', 'init_conv.bias'})}"
        )

    def test_both_solvers_take_the_same_number_of_model_evaluations(self) -> None:
        calls = {"cylindrical": 0, "euclidean": 0}

        for manifold in (CylindricalManifold(), EuclideanManifold()):

            def counting_model(x: torch.Tensor, t: torch.Tensor, key=manifold.name):
                calls[key] += 1
                return torch.zeros(x.shape[0], 2, x.shape[2], x.shape[3])

            manifold.make_solver(num_steps=6).sample(
                counting_model, manifold.sample_noise(1, 8, 8, CPU)
            )

        assert calls["cylindrical"] == calls["euclidean"] == 2 * 6 - 1


def test_a_bounded_head_cannot_leave_the_range_its_geometry_allows() -> None:
    """The bound is the point of the head, so it is asserted rather than assumed.

    The failure it exists to prevent is silent: an unbounded head predicts angular
    velocities above pi, which no target ever asks for, and the error only shows up
    several solver steps later.
    """
    import math

    from cyfm.models.unet import CylindricalUNet

    model = CylindricalUNet(
        base_channels=8, in_channels=3, out_channels=2, velocity_bound=(1.0, math.pi)
    )
    # Drive the final convolution hard enough that an unbounded head would run away.
    with torch.no_grad():
        model.final_conv.weight.mul_(500.0)
        model.final_conv.bias.fill_(500.0)
    velocity = model(torch.randn(4, 3, 16, 16), torch.rand(4))
    # Saturation lands on the float32 representation of the bound, and float32's pi
    # rounds above the true value, so the comparison is made in that precision.
    limits = torch.tensor([1.0, math.pi], dtype=torch.float32)
    assert float(velocity[:, 0].abs().max()) <= float(limits[0])
    assert float(velocity[:, 1].abs().max()) <= float(limits[1])
    # And it really is saturating, not merely small.
    assert float(velocity[:, 1].abs().max()) > 3.0


def test_an_unbounded_head_is_left_exactly_as_it_was() -> None:
    """Omitting the bound must not perturb the existing architecture."""
    from cyfm.models.unet import CylindricalUNet

    torch.manual_seed(0)
    plain = CylindricalUNet(base_channels=8, in_channels=3, out_channels=2)
    torch.manual_seed(0)
    same = CylindricalUNet(base_channels=8, in_channels=3, out_channels=2, velocity_bound=None)
    state = torch.randn(2, 3, 16, 16)
    time = torch.rand(2)
    assert torch.equal(plain(state, time), same(state, time))


def test_each_geometry_states_its_own_bound() -> None:
    """The cylinder's binds and the plane's is slack; that asymmetry is the argument."""
    import math

    from cyfm.manifolds.cylindrical import CylindricalManifold
    from cyfm.manifolds.euclidean import EuclideanManifold

    assert CylindricalManifold().velocity_bound == (1.0, math.pi)
    assert EuclideanManifold().velocity_bound == (2.0, 2.0)
