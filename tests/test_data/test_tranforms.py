import math
import torch

from cfm.data.transforms import AmplitudeNormalize, CenterCropModulo, ComplexToCylinderTransform, Compose

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
    x[1, :, :] = 1.0    # px
    x[2, :, :] = 0.0    # py
    
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

def test_compose_pipeline() -> None:
    """Tests a full data transformation pipeline sequence."""
    pipeline = Compose([
        ComplexToCylinderTransform(),
        AmplitudeNormalize(),
        CenterCropModulo(base=16)
    ])
    
    x = torch.randn(1, 35, 35, dtype=torch.complex64)
    # Give it one huge value to test normalization
    x[0, 0, 0] = 500.0 + 0j 
    
    out = pipeline(x)
    
    assert out.shape == (3, 32, 32)
    assert out[0].max() <= 1.0