import torch

from cfm.utils.complex_ops import (
    complex_to_cylinder,
    complex_to_euclidean,
    cylinder_to_complex,
    euclidean_to_complex,
)


def test_complex_to_cylinder_shapes() -> None:
    """Checks that projection returns the correct BCHW shape."""
    batch, channels, h, w = 4, 1, 32, 32
    z = torch.randn(batch, channels, h, w, dtype=torch.complex64)

    cylinder = complex_to_cylinder(z)

    assert cylinder.shape == (batch, 3, h, w), "Wrong output shape after cylinder projection"
    assert cylinder.dtype == torch.float32, "Network expects float32, not complex"


def test_topology_constraint() -> None:
    """Checks that the phase vector always lies on the unit circle (R=1)."""
    z = torch.randn(2, 1, 16, 16, dtype=torch.complex64)
    cylinder = complex_to_cylinder(z)

    p_x = cylinder[:, 1:2, :, :]
    p_y = cylinder[:, 2:3, :, :]

    # R^2 = p_x^2 + p_y^2 should be exactly 1.0
    radius_squared = p_x**2 + p_y**2
    assert torch.allclose(
        radius_squared,
        torch.ones_like(radius_squared),
        atol=1e-6,
    ), "Phase vector left the cylinder boundary!"


def test_cycle_consistency() -> None:
    """Checks forward/backward consistency of the complex-cylindrical conversion."""
    z_original = torch.randn(8, 1, 64, 64, dtype=torch.complex64)

    cylinder = complex_to_cylinder(z_original)
    z_reconstructed = cylinder_to_complex(cylinder)

    # atol=1e-5 allows for float32 rounding differences
    assert torch.allclose(
        z_original,
        z_reconstructed,
        atol=1e-5,
    ), "Reconstruction error is too large after round-trip conversion"


def test_zero_magnitude_edge_case() -> None:
    """CRITICAL: Verifies phase behavior for zero-magnitude background pixels."""
    z_zeros = torch.zeros(1, 1, 8, 8, dtype=torch.complex64)

    cylinder = complex_to_cylinder(z_zeros)

    m = cylinder[:, 0:1, :, :]
    p_x = cylinder[:, 1:2, :, :]
    p_y = cylinder[:, 2:3, :, :]

    assert torch.all(m == 0.0), "Zero signal magnitude must stay 0"

    # Even for zero signal, phase should stay on the unit circle (avoid vanishing gradients)
    radius_squared = p_x**2 + p_y**2
    assert torch.allclose(
        radius_squared,
        torch.ones_like(radius_squared),
        atol=1e-6,
    ), "Phase collapsed in zero-magnitude pixels!"


def test_complex_to_euclidean_shapes() -> None:
    """Checks that the flat projection returns the correct BCHW shape."""
    batch, channels, h, w = 4, 1, 32, 32
    z = torch.randn(batch, channels, h, w, dtype=torch.complex64)

    euclidean = complex_to_euclidean(z)

    assert euclidean.shape == (batch, 2, h, w), "Wrong output shape after flat projection"
    assert euclidean.dtype == torch.float32, "Network expects float32, not complex"


def test_euclidean_cycle_consistency() -> None:
    """Checks forward/backward consistency of the complex-Euclidean conversion.

    This round trip is exact, not merely close: unlike the cylindrical one it
    involves no atan2 and no cos/sin, so nothing is lost.
    """
    z_original = torch.randn(8, 1, 64, 64, dtype=torch.complex64)

    z_reconstructed = euclidean_to_complex(complex_to_euclidean(z_original))

    assert torch.equal(z_original, z_reconstructed), "Flat round trip must be bit-exact"


def test_euclidean_zero_magnitude_edge_case() -> None:
    """Zero pixels need no epsilon guard here: there is no angle to compute."""
    z_zeros = torch.zeros(1, 1, 8, 8, dtype=torch.complex64)

    euclidean = complex_to_euclidean(z_zeros)

    assert torch.all(euclidean == 0.0)
    assert not torch.any(torch.isnan(euclidean))


def test_both_representations_describe_the_same_complex_number() -> None:
    """CRITICAL: the two geometries must be coordinate changes of one signal.

    If this ever fails, the cylindrical and Euclidean models are being trained on
    different data and no comparison between them means anything.
    """
    z = torch.randn(4, 1, 16, 16, dtype=torch.complex64)

    via_cylinder = cylinder_to_complex(complex_to_cylinder(z))
    via_euclidean = euclidean_to_complex(complex_to_euclidean(z))

    assert torch.allclose(via_cylinder, via_euclidean, atol=1e-5)
