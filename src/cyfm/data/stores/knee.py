"""The compact fastMRI knee CORPD store, as a dataset.

Raw fastMRI is a mixture of protocols, matrices and coil arrays, and a generative
target has to be one distribution. ``scripts/data/build_knee_pd_store.py`` selects the
coronal proton-density volumes without fat suppression, combines their coils with
ESPIRiT maps, crops to 320x320 and keeps the central slices; this class reads what it
wrote and nothing else. Training therefore never touches raw k-space, never calls
ESPIRiT, and does not need the raw cohort to still exist -- which is what lets the
cluster pipeline delete each archive as soon as it has been consumed.

The store is complex64 and unnormalised on purpose: the manifold's transform divides
each field by its own peak modulus, exactly as it does for the synthetic cohorts, so
the normalisation lives in one place for every dataset.
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
from cyfm.core.registry import DATASETS, register_dataset
from cyfm.data.transforms import KSpaceCenterCrop

__all__ = ["KneeStoreDataset"]

ROLES = ("all", "fit", "holdout")


def _volume_role(volume: str, fraction: float, seed: int) -> str:
    """Assign one volume to ``fit`` or ``holdout``, deterministically.

    The assignment hashes the volume id, so it depends on neither the file order nor
    the store's contents: adding volumes never moves an existing one across the split,
    and the same volume lands in the same role on every machine.

    Args:
        volume: Volume identifier as recorded in the manifest.
        fraction: Share of volumes to place in ``holdout``.
        seed: Salt, so a second split of the same cohort is possible.

    Returns:
        ``"holdout"`` or ``"fit"``.
    """
    digest = hashlib.sha256(f"{seed}:{volume}".encode()).digest()
    draw = int.from_bytes(digest[:8], "big") / 2**64
    return "holdout" if draw < fraction else "fit"


@register_dataset("fastmri_knee_pd")
@register_dataset("knee_store")
class KneeStoreDataset(BaseComplexDataset):
    """Complex knee slices read from a prebuilt store.

    Args:
        data_dir: Directory holding the store and its manifest.
        store: File name of the store inside ``data_dir``.
        role: ``"fit"`` for training volumes, ``"holdout"`` for the held-out ones,
            ``"all"`` for every slice. The split is by volume, never by slice: slices
            of one knee are highly correlated, so splitting them would leak.
        holdout_fraction: Share of volumes held out when ``role`` is not ``"all"``.
        split_seed: Salt for the volume-level assignment.
        transform: Applied to each complex field before it is returned; the entry
            points pass the manifold's representation pipeline here.
        kspace_crop: Target ``(H, W)``, or a single int, for a k-space centre crop applied
            *before* ``transform``. ``None`` keeps the store's own matrix. This is a
            resolution reduction rather than a field-of-view change, so it is not the same
            knob as the entry point's ``crop_size``; see :class:`~cyfm.data.transforms.KSpaceCenterCrop`.

    Raises:
        FileNotFoundError: If the store is absent. Training deliberately does not
            build or download it: that is a CPU-side job, not something to run while
            holding a GPU.
        ValueError: If ``role`` or ``holdout_fraction`` is out of range.
    """

    def __init__(
        self,
        data_dir: str | pathlib.Path = "data/knee_pd",
        store: str = "val.h5",
        role: str = "all",
        holdout_fraction: float = 0.2,
        split_seed: int = 0,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        kspace_crop: int | tuple[int, int] | list[int] | None = None,
    ) -> None:
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {role!r}")
        if not 0.0 <= holdout_fraction < 1.0:
            raise ValueError(f"holdout_fraction must be in [0, 1), got {holdout_fraction}")

        self.kspace_crop = None if kspace_crop is None else KSpaceCenterCrop(kspace_crop)

        self.path = pathlib.Path(data_dir) / store
        if not self.path.is_file():
            raise FileNotFoundError(
                f"No knee store at {self.path}. Build it from raw fastMRI with:\n"
                "  uv run python scripts/data/build_knee_pd_store.py "
                f"--raw-dir <raw> --sens-dir <sens> --out {self.path}\n"
                "Training does not download or preprocess: that is a CPU job."
            )

        manifest_path = self.path.with_suffix(".json")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Store {self.path} has no manifest at {manifest_path}.")
        self.manifest = json.loads(manifest_path.read_text())

        self.role = role
        self.transform = transform
        self._handle: h5py.File | None = None

        rows = self.manifest["slices"]
        keep = [
            record
            for record in rows
            if role == "all" or _volume_role(record["volume"], holdout_fraction, split_seed) == role
        ]
        if not keep:
            raise ValueError(
                f"role={role!r} selected no slices out of {len(rows)}; "
                f"holdout_fraction={holdout_fraction} is likely too small for "
                f"{len(self.manifest['volumes'])} volumes."
            )
        self.rows = [int(record["row"]) for record in keep]
        # The split machinery in the entry points keys on file basenames, so the map
        # names the volume each slice came from rather than the store it now lives in.
        self.slice_map: list[tuple[str, int]] = [
            (f"{record['volume']}.h5", int(record["slice"])) for record in keep
        ]
        self.volumes = sorted({record["volume"] for record in keep})

    def _images(self) -> h5py.Dataset:
        """The store's image dataset, opened once per worker process."""
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle["images"]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """One complex slice.

        Args:
            idx: Index into this role's slices.

        Returns:
            Complex tensor ``[1, H, W]``, or whatever ``transform`` turns it into.

        Raises:
            IndexError: If ``idx`` is out of range.
        """
        if not 0 <= idx < len(self.rows):
            raise IndexError(f"index {idx} out of range for {len(self.rows)} slices")
        raw = np.asarray(self._images()[self.rows[idx]], dtype=np.complex64)
        field = torch.from_numpy(raw).unsqueeze(0)
        if self.kspace_crop is not None:
            field = self.kspace_crop(field)
        return field if self.transform is None else self.transform(field)
