"""The knee store dataset: volume-level splitting, and failing loudly when absent."""

from __future__ import annotations

import json
import pathlib

import h5py
import numpy as np
import pytest
import torch

from cyfm.data.knee_store import KneeStoreDataset, _volume_role


def write_store(directory: pathlib.Path, volumes: int = 6, slices: int = 3) -> pathlib.Path:
    """Write a tiny store with a manifest shaped like the builder's output."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "val.h5"
    total = volumes * slices
    images = np.arange(total * 4 * 4, dtype=np.float32).reshape(total, 4, 4)
    with h5py.File(path, "w") as store:
        store.create_dataset("images", data=(images + 1j * images).astype(np.complex64))
    records = [
        {"row": volume * slices + index, "volume": f"file{volume:03d}", "slice": index}
        for volume in range(volumes)
        for index in range(slices)
    ]
    path.with_suffix(".json").write_text(
        json.dumps({"slices": records, "volumes": [f"file{v:03d}" for v in range(volumes)]})
    )
    return path


def test_roles_partition_by_volume(tmp_path: pathlib.Path) -> None:
    write_store(tmp_path)
    fit = KneeStoreDataset(tmp_path, role="fit", holdout_fraction=0.34)
    holdout = KneeStoreDataset(tmp_path, role="holdout", holdout_fraction=0.34)
    every = KneeStoreDataset(tmp_path, role="all")

    assert len(fit) + len(holdout) == len(every) == 18
    assert set(fit.volumes).isdisjoint(holdout.volumes)
    # A volume never contributes to both roles, which is the point of splitting by knee.
    assert set(fit.volumes) | set(holdout.volumes) == set(every.volumes)


def test_split_is_independent_of_order_and_cohort_size(tmp_path: pathlib.Path) -> None:
    write_store(tmp_path)
    small = KneeStoreDataset(tmp_path, role="holdout", holdout_fraction=0.34).volumes
    write_store(tmp_path, volumes=12)
    large = KneeStoreDataset(tmp_path, role="holdout", holdout_fraction=0.34)
    # Adding volumes must not move an existing one across the split.
    assert set(small).issubset(large.volumes)


def test_slice_map_names_the_source_volume(tmp_path: pathlib.Path) -> None:
    write_store(tmp_path)
    dataset = KneeStoreDataset(tmp_path, role="all")
    names = {name for name, _ in dataset.slice_map}
    assert names == {f"file{v:03d}.h5" for v in range(6)}
    assert len(dataset.slice_map) == len(dataset)


def test_items_are_complex_and_transformed(tmp_path: pathlib.Path) -> None:
    write_store(tmp_path)
    plain = KneeStoreDataset(tmp_path, role="all")
    item = plain[0]
    assert item.shape == (1, 4, 4)
    assert item.is_complex()

    transformed = KneeStoreDataset(tmp_path, role="all", transform=lambda x: torch.abs(x))
    assert not transformed[0].is_complex()


def test_missing_store_says_how_to_build_it(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError, match="build_knee_pd_store"):
        KneeStoreDataset(tmp_path, role="all")


def test_rejects_impossible_configuration(tmp_path: pathlib.Path) -> None:
    write_store(tmp_path)
    with pytest.raises(ValueError, match="role must be one of"):
        KneeStoreDataset(tmp_path, role="test")
    with pytest.raises(ValueError, match="holdout_fraction"):
        KneeStoreDataset(tmp_path, role="fit", holdout_fraction=1.5)
    with pytest.raises(ValueError, match="selected no slices"):
        KneeStoreDataset(tmp_path, role="holdout", holdout_fraction=0.0)


def test_volume_role_is_stable() -> None:
    assert _volume_role("file000", 0.5, 0) == _volume_role("file000", 0.5, 0)
    roles = {_volume_role(f"file{i:03d}", 0.5, 0) for i in range(20)}
    assert roles == {"fit", "holdout"}


def test_kspace_crop_runs_before_the_transform(tmp_path: pathlib.Path) -> None:
    """The crop reduces the matrix, and the transform sees the reduced field."""
    write_store(tmp_path)

    assert KneeStoreDataset(tmp_path, role="all")[0].shape == (1, 4, 4)

    cropped = KneeStoreDataset(tmp_path, role="all", kspace_crop=2)
    item = cropped[0]
    assert item.shape == (1, 2, 2)
    assert item.is_complex()

    seen: list[tuple[int, ...]] = []

    def record(x: torch.Tensor) -> torch.Tensor:
        seen.append(tuple(x.shape))
        return x

    KneeStoreDataset(tmp_path, role="all", kspace_crop=2, transform=record)[0]
    assert seen == [(1, 2, 2)]


def test_kspace_crop_defaults_to_no_crop(tmp_path: pathlib.Path) -> None:
    """Omitting the key leaves every existing arm byte-identical."""
    write_store(tmp_path)

    assert KneeStoreDataset(tmp_path, role="all").kspace_crop is None
    torch.testing.assert_close(
        KneeStoreDataset(tmp_path, role="all")[0],
        KneeStoreDataset(tmp_path, role="all", kspace_crop=None)[0],
    )
