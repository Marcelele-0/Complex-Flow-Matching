import pytest
import torch

from cfm.data.transforms import (
    AmplitudeNormalize,
    CenterCropModulo,
    ComplexToCylinderTransform,
    ComplexToEuclideanTransform,
    Compose,
    EuclideanNormalize,
    WindowAmplitudeNormalize,
    WindowEuclideanNormalize,
)


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
    """Exactly one slice hits 1.0, and relative brightness between slices
    (the ratio of their amplitudes) is preserved by the single shared scale."""
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
    """A single [C, H, W] slice must raise, not silently index the wrong axis
    (x[:, 0:1] on [C, H, W] would slice rows, not the amplitude channel)."""
    transform = WindowAmplitudeNormalize()
    x = torch.randn(3, 4, 4)

    with pytest.raises(ValueError):
        transform(x)


def test_window_euclidean_normalize() -> None:
    """One scalar - the largest modulus anywhere in the window - scales every
    slice, so exactly one pixel hits 1.0 and inter-slice brightness survives."""
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
    """Cropping a whole [S, C, H, W] window at once must equal cropping each
    slice independently, since the crop only ever touches trailing axes."""
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
