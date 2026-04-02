import math

import torch

from cfm.flow.torus_math import DecoupledCylindricalL1Loss


def test_loss_shapes_and_types() -> None:
    """Checks that loss returns three scalar float32 tensors."""
    criterion = DecoupledCylindricalL1Loss()
    pred = torch.randn(4, 2, 16, 16)
    target = torch.randn(4, 2, 16, 16)

    total, l_amp, l_phase = criterion(pred, target)

    assert total.dim() == 0
    assert l_amp.dim() == 0
    assert l_phase.dim() == 0
    assert total.dtype == torch.float32


def test_amplitude_loss_logic() -> None:
    """Checks that amplitude loss is standard L1 (MAE)."""
    criterion = DecoupledCylindricalL1Loss(lambda_phase=0.0)

    pred = torch.zeros(1, 2, 1, 1)
    target = torch.zeros(1, 2, 1, 1)

    pred[0, 0, 0, 0] = 5.0
    target[0, 0, 0, 0] = 2.0

    total, l_amp, _ = criterion(pred, target)

    assert torch.isclose(l_amp, torch.tensor(3.0)), "Amplitude L1 loss was computed incorrectly"
    assert torch.isclose(total, torch.tensor(3.0)), "Total loss did not ignore phase as expected"


def test_phase_loss_topology_wrap() -> None:
    """CRITICAL: Checks shortest circular phase distance behavior."""
    criterion = DecoupledCylindricalL1Loss(lambda_phase=1.0)

    pred = torch.zeros(1, 2, 1, 1)
    target = torch.zeros(1, 2, 1, 1)

    # Case 1: Full turn should be zero error.
    pred[0, 1, 0, 0] = 2 * math.pi
    target[0, 1, 0, 0] = 0.0
    _, _, l_phase1 = criterion(pred, target)
    assert torch.isclose(
        l_phase1,
        torch.tensor(0.0),
        atol=1e-6,
    ), "Loss did not ignore a full 2*pi rotation"

    # Case 2: 350 deg vs 10 deg should give shortest path: 20 deg.
    pred[0, 1, 0, 0] = 350 * (math.pi / 180)
    target[0, 1, 0, 0] = 10 * (math.pi / 180)
    _, _, l_phase2 = criterion(pred, target)
    expected_rad = 20 * (math.pi / 180)
    assert torch.isclose(
        l_phase2,
        torch.tensor(expected_rad),
        atol=1e-6,
    ), "Loss did not take the shortest circular path"


def test_lambda_weighting() -> None:
    """Checks whether lambda scales phase error in total loss."""
    lambda_val = 0.5
    criterion = DecoupledCylindricalL1Loss(lambda_phase=lambda_val)

    pred = torch.zeros(1, 2, 1, 1)
    target = torch.zeros(1, 2, 1, 1)

    pred[0, 0, 0, 0] = 1.0  # amplitude error = 1.0
    pred[0, 1, 0, 0] = 1.0  # phase error = 1.0 rad

    total, _, _ = criterion(pred, target)

    expected_total = 1.0 + (lambda_val * 1.0)
    assert torch.isclose(
        total,
        torch.tensor(expected_total),
    ), "Lambda weight was not applied correctly to total loss"
