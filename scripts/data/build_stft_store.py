"""Turn a clean-speech corpus into the compact complex-STFT store the paper trains on.

The audio counterpart of ``build_knee_pd_store.py``, with the same intake discipline:
decode, transform, keep only what training reads, and delete the raw audio as it is
consumed, so a cluster node never has to hold the corpus and the converted store at
once.

LibriSpeech ``dev-clean`` and ``test-clean`` are the default cohort. They are CC BY 4.0,
already sampled at 16 kHz so nothing is resampled, and laid out as
``<speaker>/<chapter>/<utterance>.flac``, which gives the speaker id needed for the
split without a metadata file.

Only about an hour of speech is needed to fill a training set of this size, so the
constraint is the number of *speakers*, not the number of hours: the split is by
speaker, and a cohort with few speakers cannot be split without leaking a voice across
it.

Usage::

    uv run python scripts/data/build_stft_store.py \
        --raw-dir data/raw/LibriSpeech/dev-clean --out data/librispeech_stft/dev-clean.h5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time
from typing import Any

import h5py
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from cyfm.data.stores.stft import StftProtocol, forward_stft, segment_frames  # noqa: E402

CHUNK_BYTES = 1 << 20


def git_revision() -> str:
    """The commit the store was built from.

    Falls back to a ``GIT_REVISION`` file at the project root before giving up. The
    cluster checkout is an rsync copy rather than a git repository, so ``git rev-parse``
    fails there and every store built on it would otherwise be stamped ``unknown`` and
    be unattributable to a version of this code.
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


def checksum(path: pathlib.Path) -> str:
    """SHA-256 of a file, read in chunks so a large store never lands in memory."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            hasher.update(block)
    return hasher.hexdigest()


def create_frames(store: h5py.File, protocol: StftProtocol) -> h5py.Dataset:
    """The store's growable segment dataset.

    One chunk is one segment, which is how :class:`~cyfm.data.stores.audio.StftStoreDataset`
    reads it, so a training read never has to touch a neighbouring segment.
    """
    shape = (protocol.bins, protocol.frames)
    return store.create_dataset(
        "frames",
        shape=(0, *shape),
        maxshape=(None, *shape),
        chunks=(1, *shape),
        dtype=np.complex64,
        compression=None,
    )


def append_block(frames: h5py.Dataset, block: np.ndarray) -> int:
    """Append segments to the store and return the row index the block starts at."""
    start = frames.shape[0]
    frames.resize(start + block.shape[0], axis=0)
    frames[start:] = block
    return start


def utterances(raw_dir: pathlib.Path) -> list[pathlib.Path]:
    """Every FLAC under ``raw_dir``, in a deterministic order."""
    return sorted(raw_dir.rglob("*.flac"))


def speaker_of(path: pathlib.Path, raw_dir: pathlib.Path) -> str:
    """The speaker id of one utterance.

    LibriSpeech nests as ``<speaker>/<chapter>/<utterance>.flac``, so the speaker is the
    first path component below the cohort root. A flat directory falls back to the
    leading field of the file name, which is the same id in that layout.
    """
    relative = path.relative_to(raw_dir)
    if len(relative.parts) >= 2:
        return relative.parts[0]
    return path.stem.split("-")[0]


def read_waveform(path: pathlib.Path, sample_rate: int) -> torch.Tensor:
    """Decode one utterance to mono float32 at the expected rate.

    Args:
        path: Audio file.
        sample_rate: Rate the protocol expects.

    Returns:
        Real tensor ``[T]``.

    Raises:
        RuntimeError: If ``soundfile`` is not installed.
        ValueError: If the file's rate differs from the protocol's. Resampling is
            deliberately not done here: it would be a second, silent protocol, and the
            corpora this targets are already at 16 kHz.
    """
    try:
        import soundfile
    except ImportError as error:  # pragma: no cover - dependency is declared
        raise RuntimeError(
            "soundfile is required to decode audio; install the project's data extra"
        ) from error
    data, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    if rate != sample_rate:
        raise ValueError(
            f"{path.name} is {rate} Hz, protocol expects {sample_rate} Hz; "
            "this builder does not resample"
        )
    return torch.from_numpy(data.mean(axis=1))


def build(args: argparse.Namespace) -> None:
    """Convert a cohort into one store plus its manifest."""
    protocol = StftProtocol(
        sample_rate=args.sample_rate, n_fft=args.n_fft, hop=args.hop, frames=args.frames
    )
    raw_dir = pathlib.Path(args.raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"No corpus at {raw_dir}")
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    paths = utterances(raw_dir)
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise ValueError(f"No .flac files under {raw_dir}")

    started = time.time()
    records: list[dict[str, Any]] = []
    rejected: dict[str, int] = {"too_short": 0}
    revision = git_revision()

    with h5py.File(out, "w") as store:
        frames = create_frames(store, protocol)
        for index, path in enumerate(paths, start=1):
            waveform = read_waveform(path, protocol.sample_rate)
            segments = segment_frames(forward_stft(waveform, protocol), protocol)
            if segments.shape[0] == 0:
                rejected["too_short"] += 1
            else:
                block = segments.numpy().astype(np.complex64)
                start = append_block(frames, block)
                speaker = speaker_of(path, raw_dir)
                for offset in range(block.shape[0]):
                    records.append(
                        {
                            "row": start + offset,
                            "speaker": speaker,
                            "utterance": path.stem,
                            "segment": offset,
                        }
                    )
            # Flushing before the delete is what makes a killed job leave a valid
            # store rather than a store missing what it has already erased.
            store.flush()
            if args.delete_consumed:
                path.unlink()
            if index % 100 == 0 or index == len(paths):
                print(
                    f"{index}/{len(paths)} utterances, {frames.shape[0]} segments",
                    flush=True,
                )

        write_attrs(store, protocol, args.split, revision)

    if not records:
        raise ValueError(
            f"No utterance under {raw_dir} was long enough for one "
            f"{protocol.seconds:.3f} s segment"
        )

    manifest = write_manifest(
        out, args.split, protocol, revision, records, rejected, time.time() - started
    )
    speakers = sorted({record["speaker"] for record in records})
    print(
        f"wrote {out} rows={len(records)} speakers={len(speakers)} "
        f"seconds={time.time() - started:.1f} manifest={manifest}",
        flush=True,
    )


def write_attrs(
    store: h5py.File, protocol: StftProtocol, split: str, revision: str
) -> None:
    """Stamp the protocol onto the HDF5 itself, so a stray store is still readable."""
    for key, value in protocol.as_dict().items():
        store.attrs[key] = value
    store.attrs["split"] = split
    store.attrs["dropped_bin"] = "nyquist; negligible energy in speech, see cyfm.data.stores.stft"
    store.attrs["normalised"] = "no; the training transform normalises per field"
    store.attrs["git_revision"] = revision


def write_manifest(
    out: pathlib.Path,
    split: str,
    protocol: StftProtocol,
    revision: str,
    records: list[dict[str, Any]],
    rejected: dict[str, int],
    seconds: float,
) -> pathlib.Path:
    """Write the sidecar manifest and return its path."""
    manifest = {
        "store": out.name,
        "split": split,
        "protocol": protocol.as_dict(),
        "git_revision": revision,
        "speakers": sorted({record["speaker"] for record in records}),
        "segments": records,
        "sha256": checksum(out),
        "rejected": rejected,
        "seconds": round(seconds, 1),
    }
    path = out.with_suffix(".json")
    path.write_text(json.dumps(manifest, indent=2))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, help="Corpus root to walk for FLAC files.")
    parser.add_argument("--out", required=True, help="Store to write.")
    parser.add_argument("--split", default="dev-clean", help="Split name recorded in the manifest.")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--n-fft", type=int, default=128)
    parser.add_argument("--hop", type=int, default=64)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="Stop after this many utterances.")
    parser.add_argument(
        "--delete-consumed",
        action="store_true",
        help="Delete each utterance once its segments are in the store.",
    )
    build(parser.parse_args())


if __name__ == "__main__":
    main()
