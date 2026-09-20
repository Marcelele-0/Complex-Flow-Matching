"""The knee store builder: selection, streaming writes, and merging per-archive parts.

The cluster processes the train split one archive per Slurm array task, so the builder
has to write without holding the cohort in memory and the parts have to join into one
store afterwards. These tests pin the three selection rules, that streaming produces the
same rows in the same order as combining by hand, and that a merge refuses the two ways
it could silently corrupt the target distribution: parts built under different
protocols, and a volume processed twice.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
from types import ModuleType

import h5py
import numpy as np
import pytest
import torch

from cyfm.data.fastmri import FastMRIDataset, _center_crop_to, _to_complex_tensor
from cyfm.data.knee_store import KneeStoreDataset

ROOT = pathlib.Path(__file__).resolve().parents[2]
BUILDER = ROOT / "scripts" / "data" / "build_knee_pd_store.py"

COILS = 2
SIDE = 16
CROP = 8
DEPTH = 4
SLICES = 2
MATRIX = f"{SIDE}x{SIDE}"


def load_builder() -> ModuleType:
    """Import scripts/data/build_knee_pd_store.py, which is not part of a package."""
    spec = importlib.util.spec_from_file_location("build_knee_pd_store", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_knee_pd_store"] = module
    spec.loader.exec_module(module)
    return module


builder = load_builder()


def write_volume(
    raw_dir: pathlib.Path,
    sens_dir: pathlib.Path,
    name: str,
    acquisition: str = "CORPD_FBK",
    coils: int = COILS,
    side: int = SIDE,
    seed: int = 0,
) -> pathlib.Path:
    """A raw volume and its ESPIRiT sidecar, small enough to combine in a test.

    The sidecar is written so the builder never calls calibration: ESPIRiT is the slow
    part of the pipeline and none of these tests are about it.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    sens_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    shape = (DEPTH, coils, side, side)
    kspace = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
    maps = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)

    path = raw_dir / f"{name}.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("kspace", data=kspace)
        handle.attrs["acquisition"] = acquisition
    with h5py.File(sens_dir / f"{name}.h5", "w") as handle:
        handle.create_dataset("sensitivity_maps", data=maps)
    return path


def build_args(raw_dir: pathlib.Path, sens_dir: pathlib.Path, out: pathlib.Path, **over: object):
    """The namespace ``build`` expects, with the test cohort's shape."""
    defaults = {
        "merge": None,
        "raw_dir": str(raw_dir),
        "sens_dir": str(sens_dir),
        "out": str(out),
        "split": "train",
        "acquisition": "CORPD_FBK",
        "coils": COILS,
        "matrices": MATRIX,
        "slices": SLICES,
        "crop": CROP,
        "limit": 0,
        "device": "cpu",
        "delete_rejected": False,
        "delete_consumed": False,
        "delete_consumed_maps": False,
    }
    return argparse.Namespace(**{**defaults, **over})


def read_manifest(store: pathlib.Path) -> dict:
    return json.loads(store.with_suffix(".json").read_text())


# --- selection -------------------------------------------------------------------


def test_accepts_only_the_protocol() -> None:
    header = {"acquisition": "CORPD_FBK", "coils": 15, "matrix": "640x368", "depth": 30}
    assert builder.rejection(header, "CORPD_FBK", 15, ("640x368", "640x372")) is None


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("acquisition", "CORPDFS_FBK", "acquisition CORPDFS_FBK"),
        ("coils", 8, "8 coils"),
        ("matrix", "640x320", "matrix 640x320"),
    ],
)
def test_each_rule_rejects_and_says_why(field: str, value: object, expected: str) -> None:
    header = {"acquisition": "CORPD_FBK", "coils": 15, "matrix": "640x368", "depth": 30}
    header[field] = value
    reason = builder.rejection(header, "CORPD_FBK", 15, ("640x368", "640x372"))
    # The reason is counted in the manifest, so the paper's volume count has an audit
    # trail rather than an unexplained shortfall.
    assert reason == expected


def test_off_protocol_volumes_never_reach_the_store(tmp_path: pathlib.Path) -> None:
    raw, sens = tmp_path / "raw", tmp_path / "sens"
    write_volume(raw, sens, "keep_a", seed=1)
    write_volume(raw, sens, "drop_fatsat", acquisition="CORPDFS_FBK", seed=2)
    write_volume(raw, sens, "drop_coils", coils=3, seed=3)
    write_volume(raw, sens, "keep_b", seed=4)

    out = tmp_path / "store" / "train.h5"
    builder.build(build_args(raw, sens, out))

    manifest = read_manifest(out)
    assert manifest["volumes"] == ["keep_a", "keep_b"]
    assert manifest["rejected"] == {"acquisition CORPDFS_FBK": 1, "3 coils": 1}
    assert all(record["acquisition"] == "CORPD_FBK" for record in manifest["slices"])
    assert all(record["matrix"] == MATRIX for record in manifest["slices"])
    assert all(record["coils"] == COILS for record in manifest["slices"])


def test_nothing_selected_writes_no_store(tmp_path: pathlib.Path) -> None:
    raw, sens = tmp_path / "raw", tmp_path / "sens"
    write_volume(raw, sens, "fatsat", acquisition="CORPDFS_FBK")
    out = tmp_path / "store" / "train.h5"

    with pytest.raises(SystemExit):
        builder.build(build_args(raw, sens, out))

    assert not out.exists()
    # The temp file must go too, or a retry inherits a half-written store.
    assert list(out.parent.glob("*.h5")) == []


# --- surviving a failure partway through an archive --------------------------------


def write_broken_sidecar(raw_dir: pathlib.Path, sens_dir: pathlib.Path, name: str) -> None:
    """A volume whose sidecar has the wrong coil count, which fails during combination."""
    write_volume(raw_dir, sens_dir, name, seed=500)
    rng = np.random.default_rng(501)
    shape = (DEPTH, COILS + 1, SIDE, SIDE)
    maps = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
    with h5py.File(sens_dir / f"{name}.h5", "w") as handle:
        handle.create_dataset("sensitivity_maps", data=maps)


def test_failure_partway_keeps_what_converted(tmp_path: pathlib.Path) -> None:
    """A volume that fails must not cost the ones already consumed before it.

    On the cluster the raw volumes are deleted as they are consumed and replacing them
    means re-downloading a ~90 GB archive, so a store that only appears at the very end
    would throw away hours of ESPIRiT on the first bad sidecar.
    """
    raw, sens = tmp_path / "raw", tmp_path / "sens"
    write_volume(raw, sens, "vol_a", seed=1)
    write_volume(raw, sens, "vol_b", seed=2)
    write_broken_sidecar(raw, sens, "vol_c")
    write_volume(raw, sens, "vol_d", seed=4)
    out = tmp_path / "store" / "part_a.h5"

    with pytest.raises(ValueError, match="must come from this k-space"):
        builder.build(build_args(raw, sens, out, delete_consumed=True))

    # The two volumes that did convert are a valid part-store, not an orphaned temp.
    assert out.exists()
    assert read_manifest(out)["volumes"] == ["vol_a", "vol_b"]
    assert list(out.parent.glob("*.tmp*.h5")) == []
    salvaged = KneeStoreDataset(out.parent, "part_a.h5", role="all")
    assert len(salvaged) == 2 * SLICES

    # Only the unprocessed volumes remain on disk, so a rerun redoes exactly those.
    assert sorted(path.name for path in raw.glob("*.h5")) == ["vol_c.h5", "vol_d.h5"]


def test_consumed_volumes_are_durable_before_deletion(tmp_path: pathlib.Path) -> None:
    raw, sens = tmp_path / "raw", tmp_path / "sens"
    write_volume(raw, sens, "vol_a", seed=7)
    write_broken_sidecar(raw, sens, "vol_b")
    out = tmp_path / "store" / "part_a.h5"

    with pytest.raises(ValueError):
        builder.build(build_args(raw, sens, out, delete_consumed=True))

    # vol_a's source is gone, so its slices had to survive the crash to exist at all.
    assert not (raw / "vol_a.h5").exists()
    with h5py.File(out, "r") as store:
        assert store["images"].shape == (SLICES, CROP, CROP)


def test_unreadable_sidecar_triggers_recalibration(tmp_path: pathlib.Path) -> None:
    """A sidecar left half-written by a killed job must not abort the whole archive."""
    sens = tmp_path / "sens"
    sens.mkdir(parents=True)
    truncated = sens / "vol_a.h5"
    truncated.write_bytes(b"\x89HDF\r\n\x1a\n truncated")

    assert builder.has_maps(truncated) is False
    assert builder.has_maps(sens / "absent.h5") is False


# --- streaming write -------------------------------------------------------------


def test_rows_are_contiguous_and_match_hand_combination(tmp_path: pathlib.Path) -> None:
    raw, sens = tmp_path / "raw", tmp_path / "sens"
    write_volume(raw, sens, "vol_a", seed=11)
    write_volume(raw, sens, "vol_b", seed=12)

    out = tmp_path / "store" / "train.h5"
    builder.build(build_args(raw, sens, out))
    manifest = read_manifest(out)

    assert [record["row"] for record in manifest["slices"]] == list(range(2 * SLICES))

    # Combine one slice independently of the builder's loop; streaming must not have
    # reordered or corrupted what a single-shot build would have produced.
    expected = []
    for name in ("vol_a", "vol_b"):
        with (
            h5py.File(raw / f"{name}.h5", "r") as raw_handle,
            h5py.File(sens / f"{name}.h5", "r") as sens_handle,
        ):
            for index in builder.central_indices(DEPTH, SLICES):
                kspace = _to_complex_tensor(raw_handle["kspace"][index])
                maps = _to_complex_tensor(sens_handle["sensitivity_maps"][index])
                image = FastMRIDataset.sense_combine(kspace, maps)
                expected.append(_center_crop_to(image, CROP, CROP)[0])
    stacked = torch.stack(expected).to(torch.complex64).numpy()

    with h5py.File(out, "r") as store:
        assert store["images"].shape == (2 * SLICES, CROP, CROP)
        np.testing.assert_allclose(store["images"][:], stacked, rtol=1e-6, atol=1e-6)


def test_store_is_chunked_one_slice_at_a_time(tmp_path: pathlib.Path) -> None:
    raw, sens = tmp_path / "raw", tmp_path / "sens"
    write_volume(raw, sens, "vol_a")
    out = tmp_path / "store" / "train.h5"
    builder.build(build_args(raw, sens, out))

    with h5py.File(out, "r") as store:
        # Training reads one row per __getitem__; a wider chunk would pull neighbours.
        assert store["images"].chunks == (1, CROP, CROP)
        assert store["images"].maxshape == (None, CROP, CROP)


# --- merge -----------------------------------------------------------------------


def make_part(tmp_path: pathlib.Path, part: str, names: list[str], seed: int) -> pathlib.Path:
    raw, sens = tmp_path / f"raw_{part}", tmp_path / f"sens_{part}"
    for offset, name in enumerate(names):
        write_volume(raw, sens, name, seed=seed + offset)
    out = tmp_path / "parts" / f"{part}.h5"
    builder.build(build_args(raw, sens, out))
    return out


def test_merge_renumbers_rows_and_unions_volumes(tmp_path: pathlib.Path) -> None:
    first = make_part(tmp_path, "a", ["vol_a", "vol_b"], seed=21)
    second = make_part(tmp_path, "b", ["vol_c"], seed=31)

    merged = tmp_path / "store" / "train.h5"
    builder.merge(
        argparse.Namespace(merge=[str(first), str(second)], out=str(merged), split="train")
    )

    manifest = read_manifest(merged)
    assert manifest["volumes"] == ["vol_a", "vol_b", "vol_c"]
    assert [record["row"] for record in manifest["slices"]] == list(range(3 * SLICES))
    assert manifest["split"] == "train"

    with h5py.File(merged, "r") as store, h5py.File(first, "r") as part_a:
        assert store["images"].shape == (3 * SLICES, CROP, CROP)
        # The first part's rows must survive the copy byte for byte and stay in front.
        np.testing.assert_array_equal(store["images"][: 2 * SLICES], part_a["images"][:])


def test_merge_sums_rejection_counts(tmp_path: pathlib.Path) -> None:
    raw_a, sens_a = tmp_path / "raw_a", tmp_path / "sens_a"
    write_volume(raw_a, sens_a, "vol_a", seed=41)
    write_volume(raw_a, sens_a, "fatsat_a", acquisition="CORPDFS_FBK", seed=42)
    first = tmp_path / "parts" / "a.h5"
    builder.build(build_args(raw_a, sens_a, first))

    raw_b, sens_b = tmp_path / "raw_b", tmp_path / "sens_b"
    write_volume(raw_b, sens_b, "vol_b", seed=43)
    write_volume(raw_b, sens_b, "fatsat_b", acquisition="CORPDFS_FBK", seed=44)
    second = tmp_path / "parts" / "b.h5"
    builder.build(build_args(raw_b, sens_b, second))

    merged = tmp_path / "store" / "train.h5"
    builder.merge(
        argparse.Namespace(merge=[str(first), str(second)], out=str(merged), split="train")
    )

    assert read_manifest(merged)["rejected"] == {"acquisition CORPDFS_FBK": 2}


def test_merge_refuses_a_different_protocol(tmp_path: pathlib.Path) -> None:
    first = make_part(tmp_path, "a", ["vol_a"], seed=51)

    raw, sens = tmp_path / "raw_b", tmp_path / "sens_b"
    write_volume(raw, sens, "vol_b", seed=52)
    second = tmp_path / "parts" / "b.h5"
    # Same data, fewer central slices: a different protocol, so a different distribution.
    builder.build(build_args(raw, sens, second, slices=1))

    merged = tmp_path / "store" / "train.h5"
    with pytest.raises(SystemExit, match="different protocol"):
        builder.merge(
            argparse.Namespace(merge=[str(first), str(second)], out=str(merged), split="train")
        )
    assert not merged.exists()


def test_merge_refuses_a_volume_processed_twice(tmp_path: pathlib.Path) -> None:
    first = make_part(tmp_path, "a", ["vol_a", "vol_b"], seed=61)
    second = make_part(tmp_path, "b", ["vol_b"], seed=71)

    merged = tmp_path / "store" / "train.h5"
    with pytest.raises(SystemExit, match="processed twice"):
        builder.merge(
            argparse.Namespace(merge=[str(first), str(second)], out=str(merged), split="train")
        )


def test_merge_refuses_a_part_from_another_split(tmp_path: pathlib.Path) -> None:
    train_part = make_part(tmp_path, "a", ["vol_a"], seed=121)

    raw, sens = tmp_path / "raw_val", tmp_path / "sens_val"
    write_volume(raw, sens, "vol_val", seed=122)
    val_part = tmp_path / "parts" / "val.h5"
    builder.build(build_args(raw, sens, val_part, split="val"))

    merged = tmp_path / "store" / "train.h5"
    # The val archive sits in FASTMRI_FULL_URLS beside the train batches; if its part
    # reached the training store, held-out knees would be trained on.
    with pytest.raises(SystemExit, match="built as split 'val'"):
        builder.merge(
            argparse.Namespace(
                merge=[str(train_part), str(val_part)], out=str(merged), split="train"
            )
        )
    assert not merged.exists()


def test_merge_needs_every_manifest(tmp_path: pathlib.Path) -> None:
    first = make_part(tmp_path, "a", ["vol_a"], seed=81)
    second = make_part(tmp_path, "b", ["vol_b"], seed=91)
    second.with_suffix(".json").unlink()

    with pytest.raises(SystemExit, match="no manifest"):
        builder.merge(
            argparse.Namespace(
                merge=[str(first), str(second)], out=str(tmp_path / "train.h5"), split="train"
            )
        )


# --- the merged store is a dataset -----------------------------------------------


def test_merged_store_splits_by_volume(tmp_path: pathlib.Path) -> None:
    parts = [
        make_part(tmp_path, "a", ["vol_a", "vol_b", "vol_c"], seed=101),
        make_part(tmp_path, "b", ["vol_d", "vol_e", "vol_f"], seed=111),
    ]
    merged = tmp_path / "store" / "train.h5"
    builder.merge(
        argparse.Namespace(merge=[str(part) for part in parts], out=str(merged), split="train")
    )

    fit = KneeStoreDataset(merged.parent, "train.h5", role="fit", holdout_fraction=0.34)
    holdout = KneeStoreDataset(merged.parent, "train.h5", role="holdout", holdout_fraction=0.34)
    every = KneeStoreDataset(merged.parent, "train.h5", role="all")

    assert len(every) == 6 * SLICES
    assert len(fit) + len(holdout) == len(every)
    # Splitting a merged store must still be patient-disjoint: the merge renumbered
    # rows, and a volume that straddled the split would leak between arms.
    assert set(fit.volumes).isdisjoint(holdout.volumes)
    assert every[0].shape == (1, CROP, CROP)
    assert every[0].dtype == torch.complex64
