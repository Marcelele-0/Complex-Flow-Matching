"""Turn raw fastMRI knee volumes into the compact CORPD store the paper trains on.

The raw cohort is a mixture of protocols, matrices and coil arrays, and unconditional
generation needs one coherent target distribution. This script keeps exactly the
volumes that belong to it, converts them to coil-combined complex images, and writes
one small HDF5 plus a manifest. Nothing downstream reads raw k-space again.

Selection (each rule checked in code, never inferred from a filename, because knee
filenames carry no contrast):

* ``acquisition`` attribute equals ``CORPD_FBK`` -- coronal proton-density weighting
  without fat suppression, the split the score-based-prior literature trains on, and
  the one whose fat-water chemical shift leaves real structure in the phase;
* a single coil count (15 across the whole local cohort), so the combined image is
  not a mixture of hardware;
* one of the accepted matrices, which drops the handful of odd acquisitions.

Per accepted volume: ESPIRiT maps (reused from a sidecar when present, computed on
the GPU otherwise) -> adjoint SENSE combination -> centre crop to 320x320, which
removes the 2x readout oversampling and unifies 640x368 with 640x372 -> the central
slices only, because the outer ones are nearly noise and would otherwise dominate a
generative target -> complex64.

Amplitude normalisation is deliberately *not* applied here: the training and
evaluation pipelines normalise each field by its own peak modulus, and doing it twice
would hide the scale the store was written at.

The store is written atomically, and with ``--delete-consumed`` the raw volume is
removed as soon as its slices are in the store, so a cluster run can stream one
archive at a time without ever holding the whole cohort.

Usage::

    uv run python scripts/data/build_knee_pd_store.py \
        --raw-dir data/fastmri_local/multicoil_val \
        --sens-dir data/fastmri_local_sens \
        --out data/knee_pd/val.h5 --split val
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import time
from typing import Any

import h5py
import numpy as np
import torch

from cfm.data.fastmri import FastMRIDataset, _center_crop_to, _to_complex_tensor
from cfm.data.torch_espirit import calibrate_fastmri_file_torch

DEFAULT_MATRICES = ("640x368", "640x372")
SENS_KEY = "sensitivity_maps"


def git_revision() -> str:
    """The commit the store was built from, or ``"unknown"`` outside a checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def inspect(path: pathlib.Path) -> dict[str, Any]:
    """Read the selection-relevant header of one raw volume.

    Args:
        path: Raw fastMRI multi-coil HDF5 file.

    Returns:
        ``acquisition``, ``coils``, ``matrix`` and ``depth`` of the volume.
    """
    with h5py.File(path, "r") as handle:
        shape = handle["kspace"].shape
        return {
            "acquisition": str(handle.attrs.get("acquisition", "?")),
            "coils": int(shape[1]),
            "matrix": f"{shape[2]}x{shape[3]}",
            "depth": int(shape[0]),
        }


def rejection(
    header: dict[str, Any], acquisition: str, coils: int, matrices: tuple[str, ...]
) -> str | None:
    """Why a volume is not part of the target distribution, or ``None`` if it is."""
    if header["acquisition"] != acquisition:
        return f"acquisition {header['acquisition']}"
    if header["coils"] != coils:
        return f"{header['coils']} coils"
    if header["matrix"] not in matrices:
        return f"matrix {header['matrix']}"
    return None


def central_indices(depth: int, count: int) -> list[int]:
    """The ``count`` slice indices centred in a volume of ``depth`` slices.

    Args:
        depth: Number of slices in the volume.
        count: How many to keep; a volume shorter than this contributes all of them.

    Returns:
        Ascending slice indices.
    """
    if depth <= count:
        return list(range(depth))
    start = (depth - count) // 2
    return list(range(start, start + count))


def combined_slices(
    raw_path: pathlib.Path,
    sens_path: pathlib.Path,
    indices: list[int],
    crop: int,
) -> torch.Tensor:
    """Coil-combine and crop the chosen slices of one volume.

    Args:
        raw_path: Raw multi-coil volume.
        sens_path: Sidecar holding its ESPIRiT maps.
        indices: Slice indices to read.
        crop: Output side length, applied as a centre crop in image space.

    Returns:
        Complex tensor ``[len(indices), crop, crop]``.
    """
    out = []
    with h5py.File(raw_path, "r") as raw, h5py.File(sens_path, "r") as sens:
        for index in indices:
            kspace = _to_complex_tensor(raw["kspace"][index])
            maps = _to_complex_tensor(sens[SENS_KEY][index])
            if kspace.shape != maps.shape:
                raise ValueError(
                    f"{raw_path.name} slice {index}: kspace {tuple(kspace.shape)} against "
                    f"maps {tuple(maps.shape)}; the maps must come from this k-space."
                )
            image = FastMRIDataset.sense_combine(kspace, maps)
            out.append(_center_crop_to(image, crop, crop)[0])
    return torch.stack(out).to(torch.complex64)


def build(args: argparse.Namespace) -> None:
    """Select, convert and store; write the manifest beside the HDF5."""
    raw_dir = pathlib.Path(args.raw_dir)
    sens_dir = pathlib.Path(args.sens_dir)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    matrices = tuple(args.matrices.split(","))

    volumes = sorted(raw_dir.glob("*.h5"))
    if args.limit:
        volumes = volumes[: args.limit]

    records: list[dict[str, Any]] = []
    rejected: dict[str, int] = {}
    images: list[torch.Tensor] = []
    started = time.time()

    for position, raw_path in enumerate(volumes, start=1):
        header = inspect(raw_path)
        reason = rejection(header, args.acquisition, args.coils, matrices)
        if reason is not None:
            rejected[reason] = rejected.get(reason, 0) + 1
            if args.delete_rejected:
                raw_path.unlink()
            continue

        sens_path = sens_dir / raw_path.name
        if not sens_path.is_file() or SENS_KEY not in h5py.File(sens_path, "r"):
            print(f"[{position}/{len(volumes)}] calibrating {raw_path.name}", flush=True)
            calibrate_fastmri_file_torch(raw_path, sens_path, device=args.device)

        indices = central_indices(header["depth"], args.slices)
        block = combined_slices(raw_path, sens_path, indices, args.crop)
        images.append(block)
        for index in indices:
            records.append(
                {
                    "row": -1,  # assigned once every block is stacked
                    "volume": raw_path.stem,
                    "slice": index,
                    "acquisition": header["acquisition"],
                    "coils": header["coils"],
                    "matrix": header["matrix"],
                    "depth": header["depth"],
                }
            )
        print(
            f"[{position}/{len(volumes)}] {raw_path.stem}: {len(indices)} slices "
            f"({header['matrix']}, {header['coils']} coils)",
            flush=True,
        )
        if args.delete_consumed:
            raw_path.unlink()
            if args.delete_consumed_maps:
                sens_path.unlink(missing_ok=True)

    if not images:
        raise SystemExit("no volume passed the selection; nothing to write")

    stacked = torch.cat(images).numpy()
    for row, record in enumerate(records):
        record["row"] = row

    tmp = out.with_suffix(f".tmp{os.getpid()}.h5")
    with h5py.File(tmp, "w") as store:
        store.create_dataset("images", data=stacked, dtype=np.complex64, compression=None)
        store.attrs["acquisition"] = args.acquisition
        store.attrs["coils"] = args.coils
        store.attrs["matrices"] = ",".join(matrices)
        store.attrs["crop"] = args.crop
        store.attrs["central_slices"] = args.slices
        store.attrs["split"] = args.split
        store.attrs["normalised"] = "no; the training transform normalises per field"
        store.attrs["git_revision"] = git_revision()
    os.replace(tmp, out)

    manifest = {
        "store": out.name,
        "split": args.split,
        "protocol": {
            "acquisition": args.acquisition,
            "coils": args.coils,
            "matrices": list(matrices),
            "crop": args.crop,
            "central_slices": args.slices,
            "coil_combination": "adjoint SENSE with ESPIRiT maps",
            "normalisation": "none in the store; per-field peak modulus at training time",
        },
        "git_revision": git_revision(),
        "volumes": sorted({record["volume"] for record in records}),
        "slices": records,
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "rejected": rejected,
        "seconds": round(time.time() - started, 1),
    }
    manifest_path = out.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=1))

    size = out.stat().st_size / 1e6
    print(
        f"\nwrote {out} : {len(records)} slices from {len(manifest['volumes'])} volumes, "
        f"{size:.0f} MB\nmanifest {manifest_path}\nrejected {rejected}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--raw-dir", required=True, help="Directory of raw multi-coil HDF5 volumes."
    )
    parser.add_argument("--sens-dir", required=True, help="Directory for ESPIRiT sidecars.")
    parser.add_argument("--out", required=True, help="Store to write, e.g. data/knee_pd/val.h5.")
    parser.add_argument("--split", default="val", help="Split name recorded in the manifest.")
    parser.add_argument("--acquisition", default="CORPD_FBK")
    parser.add_argument("--coils", type=int, default=15)
    parser.add_argument("--matrices", default=",".join(DEFAULT_MATRICES))
    parser.add_argument("--slices", type=int, default=11, help="Central slices per volume.")
    parser.add_argument("--crop", type=int, default=320)
    parser.add_argument("--limit", type=int, default=0, help="Stop after this many raw files.")
    parser.add_argument("--device", default="cuda", help="Device for ESPIRiT calibration.")
    parser.add_argument(
        "--delete-rejected", action="store_true", help="Delete volumes that fail selection."
    )
    parser.add_argument(
        "--delete-consumed", action="store_true", help="Delete a volume once its slices are stored."
    )
    parser.add_argument(
        "--delete-consumed-maps", action="store_true", help="Also delete its ESPIRiT sidecar."
    )
    build(parser.parse_args())


if __name__ == "__main__":
    main()
