"""The compact LibriSpeech STFT store, as a dataset.

The audio counterpart of :mod:`cyfm.data.stores.knee`, and deliberately the same shape:
``scripts/data/build_stft_store.py`` decodes a clean-speech corpus, takes the short-time
Fourier transform of each utterance and keeps fixed-size complex segments; this class
reads what it wrote and nothing else. Training therefore never decodes audio and does
not need the raw corpus to still exist, which is what lets the intake delete each
archive as it is consumed.

The store is complex64 and unnormalised on purpose: the manifold's transform divides
each field by its own peak modulus, exactly as it does for the knee store and the
synthetic cohorts, so normalisation lives in one place for every dataset.

The split is by **speaker**. Segments of one utterance share a voice, a room and a
microphone, so splitting them individually would leak; so would splitting by utterance,
since one speaker reads many. Hashing the speaker id makes the assignment independent
of file order and of how much of the corpus has been built so far.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from collections.abc import Callable

import h5py
import numpy as np
import torch

from cyfm.core.dataset import BaseComplexDataset
from cyfm.core.registry import DATASETS

__all__ = ["StftStoreDataset"]

ROLES = ("all", "fit", "holdout")


def _speaker_role(speaker: str, fraction: float, seed: int) -> str:
    """Assign one speaker to ``fit`` or ``holdout``, deterministically.

    The assignment hashes the speaker id, so it depends on neither the file order nor
    the store's contents: adding speakers never moves an existing one across the split,
    and the same speaker lands in the same role on every machine.

    Args:
        speaker: Speaker identifier as recorded in the manifest.
        fraction: Share of speakers to place in ``holdout``.
        seed: Salt, so a second split of the same cohort is possible.

    Returns:
        ``"holdout"`` or ``"fit"``.
    """
    digest = hashlib.sha256(f"{seed}:{speaker}".encode()).digest()
    draw = int.from_bytes(digest[:8], "big") / 2**64
    return "holdout" if draw < fraction else "fit"


@DATASETS.register("librispeech_stft")
@DATASETS.register("stft_store")
class StftStoreDataset(BaseComplexDataset):
    """Complex STFT segments read from a prebuilt store.

    Args:
        data_dir: Directory holding the store and its manifest.
        store: File name of the store inside ``data_dir``.
        role: ``"fit"`` for training speakers, ``"holdout"`` for the held-out ones,
            ``"all"`` for every segment. The split is by speaker, never by segment.
        holdout_fraction: Share of speakers held out when ``role`` is not ``"all"``.
        split_seed: Salt for the speaker-level assignment.
        transform: Applied to each complex field before it is returned; the entry
            points pass the manifold's representation pipeline here.

    Raises:
        FileNotFoundError: If the store or its manifest is absent. Training
            deliberately does not build or download it: that is a CPU-side job, not
            something to run while holding a GPU.
        ValueError: If ``role`` or ``holdout_fraction`` is out of range, or if the
            selection is empty.
    """

    def __init__(
        self,
        data_dir: str | pathlib.Path = "data/librispeech_stft",
        store: str = "librispeech.h5",
        role: str = "all",
        holdout_fraction: float = 0.2,
        split_seed: int = 0,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {role!r}")
        if not 0.0 <= holdout_fraction < 1.0:
            raise ValueError(f"holdout_fraction must be in [0, 1), got {holdout_fraction}")

        self.path = pathlib.Path(data_dir) / store
        if not self.path.is_file():
            raise FileNotFoundError(
                f"No STFT store at {self.path}. Build it from a speech corpus with:\n"
                "  uv run python scripts/data/build_stft_store.py "
                f"--raw-dir <raw> --out {self.path}\n"
                "Training does not download or decode audio: that is a CPU job."
            )

        manifest_path = self.path.with_suffix(".json")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Store {self.path} has no manifest at {manifest_path}.")
        self.manifest = json.loads(manifest_path.read_text())

        self.role = role
        self.transform = transform
        self._handle: h5py.File | None = None

        rows = self.manifest["segments"]
        keep = [
            record
            for record in rows
            if role == "all"
            or _speaker_role(str(record["speaker"]), holdout_fraction, split_seed) == role
        ]
        if not keep:
            raise ValueError(
                f"role={role!r} selected no segments out of {len(rows)}; "
                f"holdout_fraction={holdout_fraction} is likely too small for "
                f"{len(self.manifest['speakers'])} speakers."
            )
        self.rows = [int(record["row"]) for record in keep]
        # The split machinery in the entry points keys on file basenames, so the map
        # names the utterance each segment came from rather than the store it lives in.
        self.slice_map: list[tuple[str, int]] = [
            (f"{record['utterance']}.flac", int(record["segment"])) for record in keep
        ]
        self.speakers = sorted({str(record["speaker"]) for record in keep})

    def _frames(self) -> h5py.Dataset:
        """The store's segment dataset, opened once per worker process."""
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle["frames"]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """One complex STFT segment.

        Args:
            idx: Index into this role's segments.

        Returns:
            Complex tensor ``[1, bins, frames]``, or whatever ``transform`` turns it
            into.

        Raises:
            IndexError: If ``idx`` is out of range.
        """
        if not 0 <= idx < len(self.rows):
            raise IndexError(f"index {idx} out of range for {len(self.rows)} segments")
        raw = np.asarray(self._frames()[self.rows[idx]], dtype=np.complex64)
        field = torch.from_numpy(raw).unsqueeze(0)
        return field if self.transform is None else self.transform(field)
