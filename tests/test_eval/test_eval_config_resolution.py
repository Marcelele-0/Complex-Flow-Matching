"""Tests for how evaluate.py resolves its mask and split from the config groups."""

from __future__ import annotations

from omegaconf import OmegaConf

from cfm.evaluate import resolve_dc_mask, resolve_split


def test_mask_inherits_trajectory_from_dataset() -> None:
    """An empty evaluate.mask scores under the trajectory the model trained on."""
    dataset = OmegaConf.create(
        {"acceleration": 4, "mask": {"type": "poisson_disc", "acceleration": 4}}
    )
    merged = resolve_dc_mask(dataset, OmegaConf.create({"mask": {}}))

    assert merged["type"] == "poisson_disc"
    assert merged["acceleration"] == 4


def test_mask_override_is_per_key_not_wholesale() -> None:
    """Overriding the acceleration must not discard the dataset's trajectory family.

    Regression test: the previous `evaluate.mask or dataset.mask` fallback meant any
    non-empty evaluate.mask replaced the whole block, so a Poisson-disc model was
    scored under the Cartesian default with no warning.
    """
    dataset = OmegaConf.create(
        {"acceleration": 4, "mask": {"type": "poisson_disc", "acceleration": 4}}
    )
    merged = resolve_dc_mask(dataset, OmegaConf.create({"mask": {"acceleration": 8}}))

    assert merged["type"] == "poisson_disc"
    assert merged["acceleration"] == 8


def test_mask_evaluate_may_change_trajectory_explicitly() -> None:
    """An explicit type in evaluate.mask still wins."""
    dataset = OmegaConf.create({"acceleration": 4, "mask": {"type": "poisson_disc"}})
    merged = resolve_dc_mask(dataset, OmegaConf.create({"mask": {"type": "cartesian"}}))

    assert merged["type"] == "cartesian"


def test_mask_acceleration_falls_back_to_dataset() -> None:
    """With no mask blocks at all, dataset.acceleration sets the operating point."""
    merged = resolve_dc_mask(OmegaConf.create({"acceleration": 12}), OmegaConf.create({}))
    assert merged["acceleration"] == 12


def test_split_uses_evaluate_key_by_default() -> None:
    """A cohort with manifests keeps the held-out gate under evaluate.split."""
    dataset = OmegaConf.create({"split": "train"})
    assert resolve_split(dataset, OmegaConf.create({"split": "test"})) == "test"


def test_split_is_none_when_dataset_declares_no_manifests() -> None:
    """dataset.split=null means the cohort ships no annotations/, so evaluate.split is moot.

    fastMRI has no manifests, and without this evaluation would abort looking for
    annotations/v1.0.0/test.json.
    """
    dataset = OmegaConf.create({"split": None})
    assert resolve_split(dataset, OmegaConf.create({"split": "test"})) is None


def test_split_explicit_null_in_evaluate() -> None:
    """evaluate.split=null still scores every file in data_dir."""
    dataset = OmegaConf.create({"split": "train"})
    assert resolve_split(dataset, OmegaConf.create({"split": None})) is None
