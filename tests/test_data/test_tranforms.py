import pytest
import torch

from cfm.data.transforms import (
    AmplitudeNormalize,
    CenterCropModulo,
    ComplexToCylinderTransform,
    Compose,
    WindowAmplitudeNormalize,
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
    x[0, 0, 0] = 500.0 + 0j

    out = pipeline(x)

    assert out.shape == (3, 32, 32)
    assert out[0].max() <= 1.0
