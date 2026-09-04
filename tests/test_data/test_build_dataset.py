"""Tests for registry-backed dataset construction shared by the entry points."""

from __future__ import annotations

import inspect
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from cfm.core.registry import DATASETS
from cfm.data import _ENTRY_POINT_KEYS, DEFAULT_DATASET, build_dataset, build_geometry_transform
from cfm.data.fastmri import FastMRIDataset
from cfm.data.masks import PoissonDiscMaskGenerator
from cfm.utils.fft import fft2c


@pytest.fixture
def fastmri_dir(tmp_path) -> str:
    """One small multi-coil volume with sensitivity maps."""
    data_dir = tmp_path / "fastmri"
    data_dir.mkdir()
    rng = np.random.default_rng(0)

    sens = rng.standard_normal((2, 4, 32, 32)) + 1j * rng.standard_normal((2, 4, 32, 32))
    sens = (sens / np.sqrt((np.abs(sens) ** 2).sum(1, keepdims=True) + 1e-8)).astype(np.complex64)
    x_gt = (rng.standard_normal((2, 1, 32, 32)) + 1j * rng.standard_normal((2, 1, 32, 32))).astype(
        np.complex64
    )
    kspace = fft2c(torch.from_numpy((sens * x_gt).astype(np.complex64))).numpy()

    with h5py.File(data_dir / "vol.h5", "w") as hf:
        hf.create_dataset("kspace", data=kspace)
        hf.create_dataset("sensitivity_maps", data=sens)

    return str(data_dir)


def test_build_dataset_selects_by_registry_name(fastmri_dir) -> None:
    """dataset.name resolves through DATASETS rather than a hardcoded branch."""
    cfg = OmegaConf.create({"name": "fastmri", "data_dir": fastmri_dir, "use_cache": False})
    dataset = build_dataset(cfg)
    assert isinstance(dataset, FastMRIDataset)


def test_build_dataset_honours_aliases(fastmri_dir) -> None:
    """Registry aliases work without the entry points enumerating them."""
    cfg = OmegaConf.create({"name": "fast_mri", "data_dir": fastmri_dir, "use_cache": False})
    assert isinstance(build_dataset(cfg), FastMRIDataset)


def test_build_dataset_resolves_nested_config_nodes(fastmri_dir) -> None:
    """DictConfig sub-nodes reach the dataset as plain containers."""
    cfg = OmegaConf.create(
        {
            "name": "fastmri",
            "data_dir": fastmri_dir,
            "use_cache": False,
            "mask": {"type": "poisson_disc", "acceleration": 8},
        }
    )
    dataset = build_dataset(cfg)
    assert isinstance(dataset.mask_generator, PoissonDiscMaskGenerator)
    assert dataset.acceleration == 8


def test_build_dataset_overrides_win(fastmri_dir) -> None:
    """Caller-supplied arguments take precedence over the config."""
    cfg = OmegaConf.create(
        {"name": "fastmri", "data_dir": "wrong/path", "use_cache": True, "num_slices": 1}
    )
    dataset = build_dataset(cfg, data_dir=fastmri_dir, use_cache=False, mode="reconstruction")
    assert dataset.data_dir == fastmri_dir
    assert dataset.mode == "reconstruction"


def test_build_dataset_ignores_entry_point_keys(fastmri_dir) -> None:
    """split and crop_size are consumed by the caller, not the dataset constructor."""
    cfg = OmegaConf.create(
        {
            "name": "fastmri",
            "data_dir": fastmri_dir,
            "use_cache": False,
            "split": None,
            "crop_size": [16, 16],
        }
    )
    assert isinstance(build_dataset(cfg), FastMRIDataset)


def test_build_dataset_rejects_unknown_keys(fastmri_dir) -> None:
    """A key the selected dataset cannot take is a config error, not silently dropped."""
    cfg = OmegaConf.create(
        {"name": "fastmri", "data_dir": fastmri_dir, "use_cache": False, "echo_idx": 0}
    )
    with pytest.raises(TypeError, match="does not accept echo_idx"):
        build_dataset(cfg)


def test_build_dataset_unknown_name_lists_options() -> None:
    """An unregistered name reports what is available."""
    with pytest.raises(KeyError, match="Available"):
        build_dataset(OmegaConf.create({"name": "not_a_dataset"}))


@pytest.mark.parametrize(
    "config_path",
    sorted((Path(__file__).resolve().parents[2] / "conf" / "dataset").glob("*.yaml")),
    ids=lambda p: p.stem,
)
def test_shipped_dataset_configs_match_their_constructors(config_path) -> None:
    """Every key in a shipped dataset group is one its dataset actually accepts.

    build_dataset raises on unknown keys, so a config that drifts from its
    constructor would otherwise only fail at the start of a real run.
    """
    cfg = OmegaConf.load(config_path)
    name = cfg.get("name", DEFAULT_DATASET)
    dataset_cls = DATASETS.get(name)

    accepted = set(inspect.signature(dataset_cls.__init__).parameters) - {"self"}
    unknown = sorted(set(cfg.keys()) - accepted - _ENTRY_POINT_KEYS)

    assert not unknown, (
        f"{config_path.name} sets keys {dataset_cls.__name__} cannot take: {unknown}"
    )


def test_build_geometry_transform_enforces_a_fixed_shape() -> None:
    """crop_size produces one shape for every input; without it, only divisibility."""
    fixed = build_geometry_transform([16, 16], crop_base=16)
    assert fixed(torch.zeros(1, 40, 32)).shape == (1, 16, 16)
    assert fixed(torch.zeros(1, 32, 48)).shape == (1, 16, 16)

    modulo_only = build_geometry_transform(None, crop_base=16)
    assert modulo_only(torch.zeros(1, 40, 32)).shape == (1, 32, 32)
    assert modulo_only(torch.zeros(1, 32, 48)).shape == (1, 32, 48)
