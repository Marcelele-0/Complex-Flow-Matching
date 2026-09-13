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
archive at a time without ever holding the whole cohort. Slices go straight into a
resizable HDF5 dataset as each volume is combined, so peak memory is one volume rather
than the whole split.

Usage::

    uv run python scripts/data/build_knee_pd_store.py \
        --raw-dir data/fastmri_local/multicoil_val \
        --sens-dir data/fastmri_local_sens \
        --out data/knee_pd/val.h5 --split val

On the cluster the train split is processed one archive per Slurm task, which cannot
share an HDF5 file: parallel writers corrupt it. Each task writes its own store and a
single ``--merge`` afterwards joins them into the one the paper reads::

    uv run python scripts/data/build_knee_pd_store.py \
        --merge data/knee_pd/parts/*.h5 --out data/knee_pd/train.h5 --split train
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
from collections.abc import Sequence
from typing import Any

import h5py
import numpy as np
import torch

from cfm.data.fastmri import FastMRIDataset, _center_crop_to, _to_complex_tensor
from cfm.data.torch_espirit import calibrate_fastmri_file_torch

DEFAULT_MATRICES = ("640x368", "640x372")
SENS_KEY = "sensitivity_maps"

# Rows copied per read when merging: 64 slices at 320x320 complex64 is ~52 MB, which
# keeps a merge of any size within a CPU node's memory.
COPY_ROWS = 64


def git_revision() -> str:
    """The commit the store was built from.

    Falls back to a ``GIT_REVISION`` file at the project root before giving up. The
    cluster checkout is an rsync copy rather than a git repository, so ``git rev-parse``
    fails there and the store is stamped ``unknown``, which is what happened to the
    first knee train store: the only link between a binary artefact outside git and the
    code that produced it, lost at write time. ``build_stft_store.py`` already had this
    fallback, which is why the audio store carries a real revision and the knee store
    did not.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        stamp = pathlib.Path(__file__).resolve().parents[2] / "GIT_REVISION"
        if stamp.is_file():
            return stamp.read_text().strip()
        return "unknown"


def resolve_device(spec: str) -> str:
    """The device ESPIRiT should calibrate on.

    ``auto`` is the default because the cluster runs this on CPU partitions to keep the
    GPU grant for training, and a hardcoded ``cuda`` raises there.
    """
    if spec != "auto":
        return spec
    return "cuda" if torch.cuda.is_available() else "cpu"


def sha256_of(path: pathlib.Path, block: int = 1 << 20) -> str:
    """Digest a file without reading it into memory all at once."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def create_images(store: h5py.File, crop: int) -> h5py.Dataset:
    """The store's growable image dataset.

    One chunk is one slice, which is how :class:`~cfm.data.knee_store.KneeStoreDataset`
    reads it, so a training read never has to touch a neighbouring slice.
    """
    return store.create_dataset(
        "images",
        shape=(0, crop, crop),
        maxshape=(None, crop, crop),
        chunks=(1, crop, crop),
        dtype=np.complex64,
        compression=None,
    )


def append_block(images: h5py.Dataset, block: np.ndarray) -> int:
    """Append slices to the store and return the row index the block starts at."""
    start = images.shape[0]
    images.resize(start + block.shape[0], axis=0)
    images[start:] = block
    return start


def has_maps(sens_path: pathlib.Path, indices: Sequence[int] | None = None) -> bool:
    """Whether a usable ESPIRiT sidecar already exists for the slices we need.

    A sidecar left half-written by a killed job is unreadable rather than absent, and
    on a run that spans hundreds of volumes that has to mean "calibrate it again", not
    "abort the archive". Opened through a context manager because the caller runs this
    once per volume and HDF5 holds the file open until the handle is released.

    Since calibration may now cover only the central slices, presence of the dataset is
    no longer enough: a sidecar built for one slice selection would otherwise be reused
    for a wider one and hand back zeroed maps, which read as a black image rather than
    as an error. The ``calibrated_slices`` attribute is checked against what is asked
    for; a sidecar written before that attribute existed is treated as complete.
    """
    if not sens_path.is_file():
        return False
    try:
        with h5py.File(sens_path, "r") as handle:
            if SENS_KEY not in handle:
                return False
            if indices is None:
                return True
            covered = handle[SENS_KEY].attrs.get("calibrated_slices", "all")
            if covered == "all":
                return True
            have = {int(part) for part in str(covered).split(",") if part != ""}
            return set(indices) <= have
    except OSError:
        return False


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


def protocol_of(args: argparse.Namespace, matrices: tuple[str, ...]) -> dict[str, Any]:
    """The selection and conversion this store was built under.

    Recorded in full because it is what makes the store one distribution; ``--merge``
    refuses to join parts that disagree on it.
    """
    return {
        "acquisition": args.acquisition,
        "coils": args.coils,
        "matrices": list(matrices),
        "crop": args.crop,
        "central_slices": args.slices,
        "coil_combination": "adjoint SENSE with ESPIRiT maps",
        "normalisation": "none in the store; per-field peak modulus at training time",
    }


def write_attrs(
    store: h5py.File, protocol: dict[str, Any], split: str, revisions: Sequence[str]
) -> None:
    """Stamp the protocol onto the HDF5 itself, so a stray store is still readable."""
    store.attrs["acquisition"] = protocol["acquisition"]
    store.attrs["coils"] = protocol["coils"]
    store.attrs["matrices"] = ",".join(protocol["matrices"])
    store.attrs["crop"] = protocol["crop"]
    store.attrs["central_slices"] = protocol["central_slices"]
    store.attrs["split"] = split
    store.attrs["normalised"] = "no; the training transform normalises per field"
    store.attrs["git_revision"] = ",".join(revisions)


def write_manifest(
    out: pathlib.Path,
    split: str,
    protocol: dict[str, Any],
    revisions: Sequence[str],
    records: list[dict[str, Any]],
    rejected: dict[str, int],
    seconds: float,
) -> pathlib.Path:
    """Write the sidecar manifest and return its path."""
    manifest = {
        "store": out.name,
        "split": split,
        "protocol": protocol,
        "git_revision": ",".join(revisions),
        "volumes": sorted({record["volume"] for record in records}),
        "slices": records,
        "sha256": sha256_of(out),
        "rejected": rejected,
        "seconds": round(seconds, 1),
    }
    manifest_path = out.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=1))
    return manifest_path


def build(args: argparse.Namespace) -> None:
    """Select, convert and store; write the manifest beside the HDF5."""
    raw_dir = pathlib.Path(args.raw_dir)
    sens_dir = pathlib.Path(args.sens_dir)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    matrices = tuple(args.matrices.split(","))
    device = resolve_device(args.device)

    volumes = sorted(raw_dir.glob("*.h5"))
    if args.limit:
        volumes = volumes[: args.limit]

    records: list[dict[str, Any]] = []
    rejected: dict[str, int] = {}
    started = time.time()
    espirit_seconds = 0.0
    calibrated = 0

    tmp = out.with_suffix(f".tmp{os.getpid()}.h5")
    failure: Exception | None = None
    with h5py.File(tmp, "w") as store:
        images = create_images(store, args.crop)

        try:
            for position, raw_path in enumerate(volumes, start=1):
                header = inspect(raw_path)
                reason = rejection(header, args.acquisition, args.coils, matrices)
                if reason is not None:
                    rejected[reason] = rejected.get(reason, 0) + 1
                    if args.delete_rejected:
                        raw_path.unlink()
                    continue

                # The slices are chosen before calibration, not after: ESPIRiT runs
                # per slice from that slice's own k-space, so calibrating the whole
                # volume and then keeping the central few produces bit-identical maps
                # while spending roughly three times the work. Measured at ~6 min per
                # volume on CPU, that waste dominated the intake.
                indices = central_indices(header["depth"], args.slices)

                sens_path = sens_dir / raw_path.name
                if not has_maps(sens_path, indices):
                    print(
                        f"[{position}/{len(volumes)}] calibrating {raw_path.name} on {device} "
                        f"({len(indices)} of {header['depth']} slices)",
                        flush=True,
                    )
                    mark = time.time()
                    calibrate_fastmri_file_torch(
                        raw_path, sens_path, device=device, slices=indices
                    )
                    elapsed = time.time() - mark
                    espirit_seconds += elapsed
                    calibrated += 1
                    print(f"    espirit {elapsed:.1f}s", flush=True)

                block = combined_slices(raw_path, sens_path, indices, args.crop)
                start = append_block(images, block.numpy())
                for offset, index in enumerate(indices):
                    records.append(
                        {
                            "row": start + offset,
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
                    # Never drop a source before its slices are durable: the raw archive
                    # costs a ~90 GB download to replace and the maps cost an hour of CPU.
                    store.flush()
                    raw_path.unlink()
                    if args.delete_consumed_maps:
                        sens_path.unlink(missing_ok=True)
        except Exception as error:  # noqa: BLE001 - finalised below, then re-raised
            failure = error

        if not records:
            tmp.unlink(missing_ok=True)
            if failure is not None:
                raise failure
            raise SystemExit("no volume passed the selection; nothing to write")

        write_attrs(store, protocol_of(args, matrices), args.split, [git_revision()])
    os.replace(tmp, out)

    manifest_path = write_manifest(
        out,
        args.split,
        protocol_of(args, matrices),
        [git_revision()],
        records,
        rejected,
        time.time() - started,
    )

    size = out.stat().st_size / 1e6
    volumes_kept = len({record["volume"] for record in records})
    print(
        f"\nwrote {out} : {len(records)} slices from {volumes_kept} volumes, "
        f"{size:.0f} MB\nmanifest {manifest_path}\nrejected {rejected}"
    )
    if calibrated:
        print(
            f"espirit_device={device} espirit_seconds_total={espirit_seconds:.1f} "
            f"espirit_seconds_per_volume={espirit_seconds / calibrated:.1f} "
            f"volumes_calibrated={calibrated}"
        )

    if failure is not None:
        # The store above is a valid part covering everything that did convert, which
        # merge can consume like any other. Say so, name what is left, and still exit
        # non-zero so a wrapper script does not read this as a clean archive.
        remaining = sorted(path.name for path in raw_dir.glob("*.h5"))
        print(
            f"\nFAILED partway: {type(failure).__name__}: {failure}\n"
            f"The {len(records)} slices already converted are in {out} and are usable.\n"
            f"{len(remaining)} raw volume(s) still unprocessed: {', '.join(remaining) or 'none'}\n"
            "Rerun on the same --raw-dir with a DIFFERENT --out, then merge the parts.",
            file=sys.stderr,
        )
        raise failure


def merge(args: argparse.Namespace) -> None:
    """Join per-archive stores into one, renumbering rows as they are copied.

    The cluster builds one store per archive because Slurm array tasks run in parallel
    and cannot share an HDF5 file. This is the step that makes them the single store the
    paper reads.
    """
    parts = [pathlib.Path(part) for part in args.merge]
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()

    manifests: list[dict[str, Any]] = []
    for part in parts:
        manifest_path = part.with_suffix(".json")
        if not manifest_path.is_file():
            raise SystemExit(f"{part} has no manifest at {manifest_path}; cannot merge it")
        manifests.append(json.loads(manifest_path.read_text()))

    # Parts carry the split they were built as. FASTMRI_FULL_URLS lists the val archive
    # next to the train batches, and a val part merged into the training store would be
    # trained on and then scored as held out.
    for part, manifest in zip(parts, manifests, strict=True):
        if manifest.get("split") != args.split:
            raise SystemExit(
                f"{part.name} was built as split {manifest.get('split')!r}, but this merge "
                f"writes split {args.split!r}; merge each split's parts separately."
            )

    # A store that mixes protocols is the mixture distribution the whole selection
    # exists to avoid, so this is fatal rather than a warning.
    protocol = manifests[0]["protocol"]
    for part, manifest in zip(parts[1:], manifests[1:], strict=True):
        if manifest["protocol"] != protocol:
            raise SystemExit(
                f"{part.name} was built under a different protocol than {parts[0].name}:\n"
                f"  {parts[0].name}: {protocol}\n  {part.name}: {manifest['protocol']}\n"
                "Merging them would make the store a mixture of acquisitions."
            )

    origin: dict[str, pathlib.Path] = {}
    for part, manifest in zip(parts, manifests, strict=True):
        for volume in manifest["volumes"]:
            if volume in origin:
                raise SystemExit(
                    f"volume {volume} is in both {origin[volume].name} and {part.name}; "
                    "an archive was processed twice"
                )
            origin[volume] = part

    crop = int(protocol["crop"])
    records: list[dict[str, Any]] = []
    rejected: dict[str, int] = {}
    revisions: list[str] = []
    seconds = 0.0

    tmp = out.with_suffix(f".tmp{os.getpid()}.h5")
    with h5py.File(tmp, "w") as store:
        images = create_images(store, crop)
        for part, manifest in zip(parts, manifests, strict=True):
            with h5py.File(part, "r") as source:
                source_images = source["images"]
                if source_images.shape[1:] != (crop, crop):
                    raise SystemExit(
                        f"{part.name} holds {source_images.shape[1:]} slices but the "
                        f"protocol says {(crop, crop)}"
                    )
                offset = images.shape[0]
                total = int(source_images.shape[0])
                images.resize(offset + total, axis=0)
                for begin in range(0, total, COPY_ROWS):
                    end = min(begin + COPY_ROWS, total)
                    images[offset + begin : offset + end] = source_images[begin:end]

            for record in manifest["slices"]:
                moved = dict(record)
                moved["row"] = offset + int(record["row"])
                records.append(moved)
            for reason, count in manifest.get("rejected", {}).items():
                rejected[reason] = rejected.get(reason, 0) + int(count)
            revision = str(manifest.get("git_revision", "unknown"))
            if revision not in revisions:
                revisions.append(revision)
            seconds += float(manifest.get("seconds", 0.0))
            print(
                f"{part.name}: {total} slices from {len(manifest['volumes'])} volumes",
                flush=True,
            )

        write_attrs(store, protocol, args.split, revisions)
    os.replace(tmp, out)

    manifest_path = write_manifest(
        out, args.split, protocol, revisions, records, rejected, seconds + time.time() - started
    )
    size = out.stat().st_size / 1e6
    print(
        f"\nmerged {len(parts)} stores into {out} : {len(records)} slices from "
        f"{len(origin)} volumes, {size:.0f} MB\nmanifest {manifest_path}\nrejected {rejected}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--merge",
        nargs="+",
        metavar="STORE",
        help="Join these per-archive stores into --out instead of building from raw.",
    )
    parser.add_argument("--raw-dir", help="Directory of raw multi-coil HDF5 volumes.")
    parser.add_argument("--sens-dir", help="Directory for ESPIRiT sidecars.")
    parser.add_argument("--out", required=True, help="Store to write, e.g. data/knee_pd/val.h5.")
    parser.add_argument("--split", default="val", help="Split name recorded in the manifest.")
    parser.add_argument("--acquisition", default="CORPD_FBK")
    parser.add_argument("--coils", type=int, default=15)
    parser.add_argument("--matrices", default=",".join(DEFAULT_MATRICES))
    parser.add_argument("--slices", type=int, default=11, help="Central slices per volume.")
    parser.add_argument("--crop", type=int, default=320)
    parser.add_argument("--limit", type=int, default=0, help="Stop after this many raw files.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for ESPIRiT calibration: auto (cuda when present, else cpu), cuda, or cpu.",
    )
    parser.add_argument(
        "--delete-rejected", action="store_true", help="Delete volumes that fail selection."
    )
    parser.add_argument(
        "--delete-consumed", action="store_true", help="Delete a volume once its slices are stored."
    )
    parser.add_argument(
        "--delete-consumed-maps", action="store_true", help="Also delete its ESPIRiT sidecar."
    )
    args = parser.parse_args()

    if args.merge:
        merge(args)
        return
    if args.raw_dir is None or args.sens_dir is None:
        parser.error("--raw-dir and --sens-dir are required unless --merge is given")
    build(args)


if __name__ == "__main__":
    main()
