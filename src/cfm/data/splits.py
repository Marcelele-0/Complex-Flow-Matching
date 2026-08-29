"""Manifest-based split selection, shared by ``train.py`` and ``evaluate.py``.

Both entry points resolve a split name through these two functions, so one
setting gates training and scoring the same way. Keeping them in one module is
what makes a held-out claim checkable: if training and evaluation read the same
manifests through the same code, a volume cannot be in both by accident.
"""

from __future__ import annotations

import glob
import json
import os
from collections.abc import Sequence


def load_split_file_names(
    data_dir: str,
    split: str | None,
    annotations_subdir: str = "annotations/v1.0.0",
    config_key: str = "evaluate.split",
) -> set[str] | None:
    """Read the COCO-style manifest for ``split`` and return its ``.h5`` basenames.

    Args:
        data_dir: Dataset root, the same one handed to :class:`SKMTEADataset`.
        split: Manifest name (``"train"``, ``"val"``, ``"test"``), or ``None`` for
            every file in ``data_dir``, the escape hatch for a directory with no
            ``annotations/``.
        annotations_subdir: Where the manifests live under ``data_dir``.
        config_key: Config key named in the error message, so the hint points at
            the setting the caller actually reads.

    Returns:
        The set of ``.h5`` basenames in the split, or ``None`` when ``split`` is
        ``None``.

    Raises:
        FileNotFoundError: If the manifest for ``split`` does not exist.
        ValueError: If the manifest lists no images.
    """
    if split is None:
        return None

    manifest = os.path.join(data_dir, annotations_subdir, f"{split}.json")
    if not os.path.isfile(manifest):
        available = sorted(glob.glob(os.path.join(data_dir, annotations_subdir, "*.json")))
        names = [os.path.basename(p) for p in available] or "none"
        raise FileNotFoundError(
            f"No manifest for split={split!r} at {manifest}. Available: {names}. "
            f"Use {config_key}=null to evaluate every file in data_dir."
        )

    with open(manifest, encoding="utf-8") as fh:
        payload = json.load(fh)

    file_names = {os.path.basename(img["file_name"]) for img in payload.get("images", [])}
    if not file_names:
        raise ValueError(f"Manifest {manifest} lists no images under 'images'.")
    return file_names


def select_indices(
    slice_map: Sequence[tuple[str, int]],
    file_names: set[str] | None,
    max_samples: int | None = None,
    config_key: str = "evaluate.split",
) -> list[int]:
    """Pick the dataset indices belonging to a split.

    Args:
        slice_map: :attr:`SKMTEADataset.slice_map`, one ``(file_path, slice_idx)``
            per slice in dataset order.
        file_names: ``.h5`` basenames to keep, or ``None`` to keep everything.
        max_samples: Optional cap, applied by striding rather than truncation.
            Slices from one volume are highly correlated, so the first N all come
            from one end of one knee and misrepresent the split.
        config_key: Config key named in the error message.

    Returns:
        Ascending dataset indices.

    Raises:
        ValueError: If no slice matches the requested split.
    """
    if file_names is None:
        indices = list(range(len(slice_map)))
    else:
        indices = [
            i for i, (path, _) in enumerate(slice_map) if os.path.basename(path) in file_names
        ]

    if not indices:
        present = sorted({os.path.basename(p) for p, _ in slice_map})
        raise ValueError(
            f"No slices matched the split. Manifest lists {sorted(file_names or [])}, "
            f"but the files present on disk are {present}. Download the missing volumes, "
            f"pick another split, or set {config_key}=null to evaluate whatever is there."
        )

    if max_samples is not None and 0 < max_samples < len(indices):
        stride = len(indices) // max_samples
        indices = indices[::stride][:max_samples]

    return indices
