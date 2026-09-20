"""The property the paper's numbers depend on: training and scoring see different volumes."""

from __future__ import annotations

import json
import os

import pytest

from cyfm.data.splits import load_split_file_names, select_indices

SLICE_MAP = [
    ("/d/files_recon_calib-24/MTR_001.h5", 0),
    ("/d/files_recon_calib-24/MTR_001.h5", 1),
    ("/d/files_recon_calib-24/MTR_002.h5", 0),
    ("/d/files_recon_calib-24/MTR_002.h5", 1),
    ("/d/files_recon_calib-24/MTR_003.h5", 0),
]


def _write_manifest(root: str, split: str, names: list[str]) -> None:
    d = os.path.join(root, "annotations/v1.0.0")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{split}.json"), "w", encoding="utf-8") as fh:
        json.dump({"images": [{"file_name": n} for n in names]}, fh)


class TestSplitsAreHeldOut:
    def test_train_and_test_selections_are_disjoint(self, tmp_path) -> None:
        """train.py and evaluate.py read the same manifests, so a volume cannot be in both.

        This is what makes an absolute number reportable. Before both entry points
        gated on the manifest, `train.py` globbed every .h5 and the two selections
        were identical no matter what `evaluate.split` said.
        """
        root = str(tmp_path)
        _write_manifest(root, "train", ["MTR_001.h5", "MTR_002.h5"])
        _write_manifest(root, "test", ["MTR_003.h5"])

        train_idx = select_indices(SLICE_MAP, load_split_file_names(root, "train"))
        test_idx = select_indices(SLICE_MAP, load_split_file_names(root, "test"))

        assert train_idx == [0, 1, 2, 3]
        assert test_idx == [4]
        assert not set(train_idx) & set(test_idx)

    def test_null_split_keeps_everything(self, tmp_path) -> None:
        assert load_split_file_names(str(tmp_path), None) is None
        assert select_indices(SLICE_MAP, None) == list(range(len(SLICE_MAP)))

    def test_error_names_the_caller_s_config_key(self, tmp_path) -> None:
        """train.py and evaluate.py read different keys; the hint must match the caller."""
        with pytest.raises(FileNotFoundError, match="dataset.split=null"):
            load_split_file_names(str(tmp_path), "train", config_key="dataset.split")

        with pytest.raises(FileNotFoundError, match="evaluate.split=null"):
            load_split_file_names(str(tmp_path), "test")

        _write_manifest(str(tmp_path), "train", ["absent.h5"])
        with pytest.raises(ValueError, match="dataset.split=null"):
            select_indices(
                SLICE_MAP,
                load_split_file_names(str(tmp_path), "train"),
                config_key="dataset.split",
            )
