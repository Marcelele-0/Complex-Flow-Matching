"""Tests for VarNetReconstructor wrapping fastmri.models.VarNet."""

import hydra
import pytest
import torch
from omegaconf import OmegaConf

from cfm.core.reconstructor import BaseReconstructor
from cfm.core.registry import MODELS, RECONSTRUCTORS
from cfm.models.varnet import VarNetReconstructor


def test_varnet_registered() -> None:
    """Checks that VarNetReconstructor is properly registered in RECONSTRUCTORS and MODELS."""
    assert "varnet" in RECONSTRUCTORS
    assert issubclass(RECONSTRUCTORS.get("varnet"), BaseReconstructor)
    assert "varnet" in MODELS


def test_varnet_multicoil_reconstruct() -> None:
    """Verifies VarNetReconstructor processes multi-coil k-space and returns complex tensor."""
    b, coils, h, w = 2, 4, 32, 32
    recon = VarNetReconstructor(
        num_cascades=2,
        sens_chans=4,
        sens_pools=2,
        chans=8,
        pools=2,
    )

    kspace = torch.randn(b, coils, h, w, dtype=torch.complex64)
    mask = torch.zeros(b, 1, h, w)
    mask[:, :, :, 12:20] = 1.0

    out = recon.reconstruct(masked_kspace=kspace, mask=mask)

    assert out.shape == (b, 1, h, w)
    assert out.is_complex()
    assert out.dtype == torch.complex64
    assert torch.isfinite(out).all()


def test_varnet_single_coil_reconstruct() -> None:
    """Verifies VarNetReconstructor supports single-coil k-space input."""
    b, h, w = 2, 32, 32
    recon = VarNetReconstructor(
        num_cascades=1,
        sens_chans=4,
        sens_pools=2,
        chans=8,
        pools=2,
    )

    kspace = torch.randn(b, 1, h, w, dtype=torch.complex64)
    mask = torch.zeros(b, 1, h, w)
    mask[:, :, :, 12:20] = 1.0

    out = recon.reconstruct(masked_kspace=kspace, mask=mask)

    assert out.shape == (b, 1, h, w)
    assert out.is_complex()
    assert torch.isfinite(out).all()


def test_varnet_forward_and_api_compatibility() -> None:
    """Verifies forward() call and acceptance of optional sensitivity_maps/num_steps."""
    b, coils, h, w = 1, 2, 32, 32
    recon = VarNetReconstructor(
        num_cascades=1,
        sens_chans=4,
        sens_pools=2,
        chans=8,
        pools=2,
    )

    kspace = torch.randn(b, coils, h, w, dtype=torch.complex64)
    # 1-batch mask broadcasted across batch
    mask = torch.zeros(1, 1, h, w)
    mask[:, :, :, 12:20] = 1.0
    sens_maps = torch.randn(b, coils, h, w, dtype=torch.complex64)

    # Calling via __call__ (forward)
    out = recon(
        masked_kspace=kspace,
        mask=mask,
        sensitivity_maps=sens_maps,
        num_steps=5,
        num_low_frequencies=8,
    )

    assert out.shape == (b, 1, h, w)
    assert out.is_complex()
    assert torch.isfinite(out).all()


def test_varnet_hydra_instantiation() -> None:
    """Verifies that VarNetReconstructor can be instantiated via Hydra configuration."""
    cfg = OmegaConf.create(
        {
            "_target_": "cfm.models.varnet.VarNetReconstructor",
            "name": "varnet",
            "num_cascades": 2,
            "sens_chans": 4,
            "sens_pools": 2,
            "chans": 8,
            "pools": 2,
            "mask_center": True,
        }
    )

    model = hydra.utils.instantiate(cfg)
    assert isinstance(model, VarNetReconstructor)
    assert isinstance(model, BaseReconstructor)


def test_varnet_invalid_kspace_shape() -> None:
    """Verifies ValueError is raised for invalid k-space shape/dtype."""
    recon = VarNetReconstructor(num_cascades=1, sens_pools=2, pools=2)
    # Real tensor with wrong dim
    invalid_kspace = torch.randn(2, 32, 32)
    mask = torch.ones(2, 1, 32, 32)

    with pytest.raises(ValueError, match="Expected complex tensor"):
        recon.reconstruct(masked_kspace=invalid_kspace, mask=mask)
