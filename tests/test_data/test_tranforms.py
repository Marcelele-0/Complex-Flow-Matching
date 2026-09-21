"""Transforms, and the single place each geometry's representation pipeline is composed."""

import pytest
import torch

from cyfm.core.manifold import BaseManifold
from cyfm.data.transforms import (
    AmplitudeNormalize,
    CenterCropModulo,
    CenterCropOrPad,
    ComplexToCylinderTransform,
    ComplexToEuclideanTransform,
    Compose,
    EuclideanNormalize,
    KSpaceCenterCrop,
    WindowAmplitudeNormalize,
    WindowEuclideanNormalize,
)
from cyfm.manifolds.complex_diffusion import ComplexDiffusionManifold
from cyfm.manifolds.cylindrical import CylindricalManifold
from cyfm.manifolds.euclidean import EuclideanManifold
from cyfm.utils.fft import fft2c


def test_center_crop_or_pad_crops_to_fixed_shape() -> None:
    """Larger inputs are centre-cropped to exactly the requested shape."""
    transform = CenterCropOrPad((16, 16))
    assert transform(torch.randn(1, 40, 32)).shape == (1, 16, 16)
    assert transform(torch.randn(4, 3, 17, 33)).shape == (4, 3, 16, 16)


def test_center_crop_or_pad_pads_smaller_inputs() -> None:
    """Smaller inputs are zero-padded rather than aborting the epoch."""
    transform = CenterCropOrPad(8)
    out = transform(torch.ones(1, 4, 4))

    assert out.shape == (1, 8, 8)
    assert out[0, 2:6, 2:6].eq(1.0).all()
    assert out[0, 0, 0].item() == 0.0


def test_center_crop_or_pad_preserves_complex_dtype() -> None:
    """Complex tensors survive both the crop and the pad path."""
    transform = CenterCropOrPad((8, 8))

    cropped = transform(torch.randn(1, 16, 16, dtype=torch.complex64))
    assert cropped.shape == (1, 8, 8)
    assert cropped.dtype == torch.complex64

    padded = transform(torch.randn(1, 4, 4, dtype=torch.complex64))
    assert padded.shape == (1, 8, 8)
    assert padded.dtype == torch.complex64


def test_center_crop_or_pad_is_centered() -> None:
    """The retained window is the middle of the input, not a corner."""
    x = torch.arange(36, dtype=torch.float32).reshape(1, 6, 6)
    out = CenterCropOrPad((2, 2))(x)

    torch.testing.assert_close(out, torch.tensor([[[14.0, 15.0], [20.0, 21.0]]]))


def test_center_crop_or_pad_rejects_nonpositive_size() -> None:
    """A zero or negative target is a config error."""
    with pytest.raises(ValueError, match="crop size must be positive"):
        CenterCropOrPad((0, 16))


def test_complex_to_cylinder_transform() -> None:
    """Tests if ComplexToCylinderTransform properly normalizes dimensions and mapping."""
    transform = ComplexToCylinderTransform()
    # Mocking complex data: [1, H, W]
    x = torch.randn(1, 16, 16, dtype=torch.complex64)
    out = transform(x)

    assert out.shape == (3, 16, 16)
    assert out.dtype == torch.float32


def test_amplitude_normalize() -> None:
    """Tests if only amplitude is normalized and phase vectors are left untouched."""
    transform = AmplitudeNormalize()
    # Create valid cylindrical tensor [3, H, W]
    x = torch.zeros(3, 4, 4)
    x[0, :, :] = 100.0  # Max amplitude is 100
    x[1, :, :] = 1.0  # px
    x[2, :, :] = 0.0  # py

    out = transform(x)

    # Amplitude should be scaled to exactly 1.0
    assert torch.allclose(out[0, :, :], torch.tensor(1.0))
    # Phase should remain completely untouched
    assert torch.allclose(out[1, :, :], torch.tensor(1.0))
    assert torch.allclose(out[2, :, :], torch.tensor(0.0))


def test_center_crop_modulo() -> None:
    """Tests if CenterCropModulo properly crops to given nearest base multipliers."""
    transform = CenterCropModulo(base=16)
    # Tensor with bad dimensions (e.g. 35x50)
    x = torch.randn(3, 35, 50)
    out = transform(x)

    # Nearest multiples of 16 are 32 and 48
    assert out.shape == (3, 32, 48)


def test_window_amplitude_normalize() -> None:
    """One shared scale keeps relative brightness between slices.

    Exactly one slice hits 1.0, and the ratio of any two slices' amplitudes is
    what it was before the division.
    """
    transform = WindowAmplitudeNormalize()
    x = torch.zeros(3, 3, 4, 4)  # [S, C, H, W]
    x[0, 0, :, :] = 50.0
    x[1, 0, :, :] = 100.0  # window max
    x[2, 0, :, :] = 25.0

    out = transform(x)

    assert torch.allclose(out[1, 0], torch.tensor(1.0))
    assert torch.allclose(out[0, 0], torch.tensor(0.5))
    assert torch.allclose(out[2, 0], torch.tensor(0.25))
    assert bool((out[:, 0] <= 1.0).all())
    assert (out[0, 0, 0, 0] / out[2, 0, 0, 0]).item() == pytest.approx(50.0 / 25.0)


def test_window_amplitude_normalize_rejects_non_window_input() -> None:
    """A single unstacked slice must raise rather than index the wrong axis.

    ``x[:, 0:1]`` on a ``[C, H, W]`` tensor slices rows, not the amplitude
    channel, and would normalise by something meaningless without complaining.
    """
    transform = WindowAmplitudeNormalize()
    x = torch.randn(3, 4, 4)

    with pytest.raises(ValueError):
        transform(x)


def test_window_euclidean_normalize() -> None:
    """The whole window is divided by one scalar.

    That scalar is the largest modulus anywhere in it, so exactly one pixel
    reaches 1.0 and inter-slice brightness survives.
    """
    transform = WindowEuclideanNormalize()
    x = torch.zeros(3, 2, 4, 4)  # [S, C, H, W] = (Re, Im)
    x[0, 0, :, :] = 30.0
    x[0, 1, :, :] = 40.0  # modulus 50
    x[1, 0, :, :] = 60.0
    x[1, 1, :, :] = 80.0  # modulus 100 - window max
    x[2, 0, :, :] = 15.0
    x[2, 1, :, :] = 20.0  # modulus 25

    out = transform(x)
    modulus = torch.sqrt(out[:, 0:1] ** 2 + out[:, 1:2] ** 2)

    assert modulus[1].max().item() == pytest.approx(1.0)
    assert modulus[0].max().item() == pytest.approx(0.5)
    assert modulus[2].max().item() == pytest.approx(0.25)
    assert bool((modulus <= 1.0 + 1e-6).all())


def test_window_euclidean_normalize_preserves_the_argument() -> None:
    """Scaling both channels by one factor moves the modulus and nothing else."""
    transform = WindowEuclideanNormalize()
    x = torch.randn(3, 2, 8, 8) * 7.0
    before = torch.atan2(x[:, 1:2], x[:, 0:1])

    out = transform(x.clone())
    after = torch.atan2(out[:, 1:2], out[:, 0:1])

    assert torch.allclose(before, after, atol=1e-6)


def test_window_euclidean_normalize_rejects_non_window_input() -> None:
    """A single [C, H, W] slice must raise, not silently index the wrong axis."""
    with pytest.raises(ValueError):
        WindowEuclideanNormalize()(torch.randn(2, 4, 4))


def test_center_crop_modulo_matches_per_slice_on_stacked_window() -> None:
    """Cropping a window equals cropping its slices one at a time.

    The crop only ever touches the trailing axes, so stacking cannot change it.
    """
    transform = CenterCropModulo(base=16)
    stack = torch.randn(3, 3, 35, 50)

    out_stack = transform(stack)
    assert out_stack.shape == (3, 3, 32, 48)

    for s in range(3):
        assert torch.equal(out_stack[s], transform(stack[s]))


def test_compose_pipeline() -> None:
    """Tests a full data transformation pipeline sequence."""
    pipeline = Compose(
        [ComplexToCylinderTransform(), AmplitudeNormalize(), CenterCropModulo(base=16)]
    )

    x = torch.randn(1, 35, 35, dtype=torch.complex64)
    # Give it one huge value to test normalization
    x[0, 0, 0] = torch.tensor(500.0 + 0j, dtype=torch.complex64)

    out = pipeline(x)

    assert out.shape == (3, 32, 32)
    assert out[0].max() <= 1.0


def test_complex_to_euclidean_transform() -> None:
    """Tests if ComplexToEuclideanTransform properly normalizes dimensions and mapping."""
    transform = ComplexToEuclideanTransform()
    # Mocking complex data: [1, H, W]
    x = torch.randn(1, 16, 16, dtype=torch.complex64)
    out = transform(x)

    assert out.shape == (2, 16, 16)
    assert out.dtype == torch.float32
    assert torch.allclose(out[0], x[0].real)
    assert torch.allclose(out[1], x[0].imag)


def test_euclidean_normalize_scales_by_the_peak_modulus() -> None:
    """Tests that the modulus is scaled to 1 and the argument is left untouched."""
    transform = EuclideanNormalize()
    # A point at 3+4i (modulus 5) and one at 0.3+0.4i (modulus 0.5).
    x = torch.zeros(2, 2, 1)
    x[0, 0, 0], x[1, 0, 0] = 3.0, 4.0
    x[0, 1, 0], x[1, 1, 0] = 0.3, 0.4

    out = transform(x)

    modulus = torch.sqrt(out[0] ** 2 + out[1] ** 2)
    assert torch.allclose(modulus[0, 0], torch.tensor(1.0))
    assert torch.allclose(modulus[1, 0], torch.tensor(0.1))

    # Argument preserved: the ratio Im/Re is unchanged by a common scale factor.
    assert torch.allclose(out[1] / out[0], x[1] / x[0], atol=1e-6)


def test_euclidean_normalize_leaves_an_all_zero_slice_alone() -> None:
    """A zero peak must not divide by zero, matching AmplitudeNormalize's guard."""
    out = EuclideanNormalize()(torch.zeros(2, 4, 4))
    assert torch.all(out == 0.0)


def test_euclidean_compose_pipeline() -> None:
    """Tests the full Euclidean transformation pipeline sequence."""
    pipeline = Compose(
        [ComplexToEuclideanTransform(), EuclideanNormalize(), CenterCropModulo(base=16)]
    )

    x = torch.randn(1, 35, 35, dtype=torch.complex64)
    # Give it one huge value to test normalization
    x[0, 0, 0] = torch.tensor(500.0 + 0j, dtype=torch.complex64)

    out = pipeline(x)

    assert out.shape == (2, 32, 32)
    modulus = torch.sqrt(out[0] ** 2 + out[1] ** 2)
    assert modulus.max() <= 1.0


def test_both_pipelines_normalize_by_the_same_factor() -> None:
    """The two geometries must scale the slice identically, or scoring diverges.

    AmplitudeNormalize divides |z| by its max; EuclideanNormalize divides Re and
    Im by that same max. Both are applied before the crop, so the peak is taken
    over the same uncropped slice in both cases.
    """
    x = torch.randn(1, 35, 35, dtype=torch.complex64)
    x[0, 0, 0] = torch.tensor(500.0 + 0j, dtype=torch.complex64)

    cyl = Compose([ComplexToCylinderTransform(), AmplitudeNormalize(), CenterCropModulo(base=16)])(
        x.clone()
    )
    euc = Compose([ComplexToEuclideanTransform(), EuclideanNormalize(), CenterCropModulo(base=16)])(
        x.clone()
    )

    euc_modulus = torch.sqrt(euc[0] ** 2 + euc[1] ** 2)
    assert torch.allclose(cyl[0], euc_modulus, atol=1e-5)


def test_kspace_center_crop_reduces_the_matrix() -> None:
    """The output carries the requested matrix and stays complex."""
    out = KSpaceCenterCrop(64)(torch.randn(1, 320, 320, dtype=torch.complex64))

    assert out.shape == (1, 64, 64)
    assert out.dtype == torch.complex64


def test_kspace_center_crop_at_full_size_is_the_identity() -> None:
    """Asking for the matrix the field already has skips the round trip entirely."""
    x = torch.randn(2, 1, 32, 32, dtype=torch.complex64)

    assert torch.equal(KSpaceCenterCrop((32, 32))(x), x)


def test_kspace_center_crop_keeps_dc_centered() -> None:
    """DC stays at [h // 2, w // 2], the convention cyfm.utils.fft asserts."""
    out = KSpaceCenterCrop(8)(torch.ones(1, 32, 32, dtype=torch.complex64))
    spectrum = fft2c(out)
    peak = int(spectrum.abs().argmax())

    assert divmod(peak, out.shape[-1]) == (out.shape[-2] // 2, out.shape[-1] // 2)


def test_kspace_center_crop_is_a_low_pass_not_a_roll() -> None:
    """A constant field survives as a constant field, with no half-FOV shift."""
    out = KSpaceCenterCrop(8)(torch.ones(1, 32, 32, dtype=torch.complex64))

    torch.testing.assert_close(out.abs().std(), torch.tensor(0.0), atol=1e-5, rtol=0.0)
    assert out.abs().mean().item() > 0.0


def test_kspace_center_crop_preserves_the_kept_energy() -> None:
    """Parseval, restricted to the block that was kept: the crop loses nothing else."""
    x = torch.randn(1, 32, 32, dtype=torch.complex64)
    kept = fft2c(x)[..., 12:20, 12:20]
    out = KSpaceCenterCrop(8)(x)

    torch.testing.assert_close(
        out.abs().square().sum(), kept.abs().square().sum(), atol=1e-4, rtol=1e-4
    )


def test_kspace_center_crop_refuses_to_enlarge() -> None:
    """Zero-filling k-space would interpolate, which is the thing this is not."""
    with pytest.raises(ValueError, match="cannot enlarge"):
        KSpaceCenterCrop(64)(torch.randn(1, 32, 32, dtype=torch.complex64))


def test_kspace_center_crop_rejects_a_real_field() -> None:
    """A 2-channel real state would transform into something meaningless."""
    with pytest.raises(ValueError, match="needs a complex field"):
        KSpaceCenterCrop(8)(torch.randn(2, 32, 32))


def test_kspace_center_crop_rejects_nonpositive_size() -> None:
    """A zero or negative target is a config error."""
    with pytest.raises(ValueError, match="crop size must be positive"):
        KSpaceCenterCrop((0, 64))


class TestRepresentationPipelines:
    """Every geometry's declaration must resolve, and resolve to one chain.

    ``slice_transform`` replaced a ``build_transform`` method on each manifold.
    The gain is that the composition exists once; the new failure mode it
    introduces is a geometry declaring a representation nothing maps, which would
    be a ``KeyError`` at the first batch rather than at import. These close that.
    """

    def test_every_representation_has_a_pipeline(self) -> None:
        from cyfm.core.manifold import Representation
        from cyfm.data.transforms import _PIPELINES

        assert set(Representation) == set(_PIPELINES)

    @pytest.mark.parametrize(
        "manifold_cls", [CylindricalManifold, EuclideanManifold, ComplexDiffusionManifold]
    )
    def test_every_manifold_declares_a_mapped_representation(
        self, manifold_cls: type[BaseManifold]
    ) -> None:
        from cyfm.data.transforms import _PIPELINES

        assert manifold_cls.representation in _PIPELINES

    def test_the_two_flat_arms_share_one_representation(self) -> None:
        """The Cartesian flow arm and the diffusion baseline hold pixels alike.

        This is why the declaration is a representation rather than the
        manifold's ``name``: two different geometries, one channel layout.
        """
        assert EuclideanManifold.representation is ComplexDiffusionManifold.representation

    @pytest.mark.parametrize("manifold_cls", [CylindricalManifold, EuclideanManifold])
    def test_slice_pipeline_normalises_before_cropping(
        self, manifold_cls: type[BaseManifold]
    ) -> None:
        """Order is the whole point of composing in one place.

        Cropping first would take the peak modulus over a different field of view
        for each geometry, which moves every absolute number in the tables.
        """
        from cyfm.data.transforms import CenterCropModulo, slice_transform

        pipeline = slice_transform(manifold_cls(), crop_base=16)
        stages = pipeline.transforms  # type: ignore[attr-defined]
        assert isinstance(stages[-1], CenterCropModulo)
        assert len(stages) == 3
