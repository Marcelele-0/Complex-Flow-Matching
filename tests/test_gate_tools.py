"""The fastMRI gate's tools: the batch-size search, the phase check, and the probe's wiring.

Issue #76 fixes the batch size at 320x320 and checks the store's phase before anything
is trained on it. The search's answer is written into the protocol, so an off-by-one
there changes every arm of Table 5; the phase statistic decides whether the coil
combination is trusted at all.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys
from types import ModuleType

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(name: str, relative: str) -> ModuleType:
    """Import a script, which is not part of a package."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


probe = load("probe_batch_size", "scripts/paper/probe_batch_size.py")
panels = load("knee_store_panels", "scripts/data/knee_store_panels.py")


class Card:
    """A card that holds batches up to ``limit``, recording what it was asked."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.asked: list[int] = []

    def __call__(self, batch: int) -> bool:
        self.asked.append(batch)
        return batch <= self.limit


# --- batch-size search -----------------------------------------------------------


@pytest.mark.parametrize("limit", [1, 3, 64, 100, 127])
def test_search_finds_the_exact_boundary(limit: int) -> None:
    card = Card(limit)
    assert probe.largest_fitting(card, max_batch=256) == limit
    # Every trial is real GPU steps, so the search has to stay logarithmic.
    assert len(card.asked) <= 2 * math.ceil(math.log2(256)) + 1


def test_search_stops_at_the_ceiling_when_the_card_is_not_the_limit() -> None:
    assert probe.largest_fitting(Card(10_000), max_batch=48) == 48


def test_search_reports_zero_when_nothing_fits() -> None:
    assert probe.largest_fitting(Card(0), max_batch=256) == 0


@pytest.mark.parametrize(("value", "expected"), [(0, 0), (1, 1), (100, 64), (128, 128)])
def test_largest_power_of_two(value: int, expected: int) -> None:
    assert probe.largest_power_of_two(value) == expected


@pytest.mark.parametrize("manifold", ["cylindrical", "euclidean"])
def test_probe_runs_the_real_training_step(manifold: str) -> None:
    trial = probe.build_trial(
        ["+experiment=table5_fastmri", f"manifold={manifold}", "training.coupling=ot"],
        side=32,
        device=torch.device("cpu"),
        steps=1,
    )
    # Wiring, not memory: this fails when the probe drifts from the API train.py uses,
    # which would make every number it reports describe some other training step.
    record = trial(2)
    assert record.fits
    assert record.batch == 2
    assert math.isfinite(record.seconds_per_step)


# --- phase check -----------------------------------------------------------------


def smooth_field(side: int = 96) -> np.ndarray:
    y, x = np.mgrid[0:side, 0:side]
    return (1.0 + 0.2 * np.cos(x / 9.0)) * np.exp(1j * (x + 0.5 * y) / 30.0)


def test_smooth_phase_is_far_below_noise() -> None:
    assert panels.adjacent_phase_step(smooth_field(), threshold=0.1) < 0.1


def test_random_phase_sits_at_pi_over_two() -> None:
    rng = np.random.default_rng(0)
    field = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(256, 256)))
    assert panels.adjacent_phase_step(field, threshold=0.1) == pytest.approx(np.pi / 2, abs=0.02)


def test_background_noise_does_not_count() -> None:
    rng = np.random.default_rng(1)
    field = 0.01 * np.exp(1j * rng.uniform(-np.pi, np.pi, size=(96, 96)))
    field[24:72, 24:72] = smooth_field()[24:72, 24:72]
    # The background of a correct MRI slice is noise too; only the anatomy decides.
    assert panels.adjacent_phase_step(field, threshold=0.1) < 0.1


def test_takes_the_central_stored_slice_of_each_volume() -> None:
    records = [{"volume": "b", "slice": s, "row": 10 + s} for s in (4, 5, 6)]
    records += [{"volume": "a", "slice": s, "row": s} for s in (0, 1)]
    chosen = panels.central_slices(records)
    assert [(record["volume"], record["slice"]) for record in chosen] == [("a", 1), ("b", 5)]
