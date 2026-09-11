"""Integrity of the archived evaluations behind Tables 2-3 and the appendix table.

Every entry of the three v2 archives in ``docs/reproduce/paper_results/`` carries the Hydra
overrides its training run was launched with. These tests check that each archive
holds the full grid with the expected metadata, and -- the reproducibility claim --
that composing the recorded overrides gives exactly the configuration of the
documented experiment (``conf/experiment/*.yaml``) plus the per-run arguments.
"""

from __future__ import annotations

import json
import pathlib
import re
from functools import cache
from typing import Any

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "reproduce" / "paper_results"
CONF = ROOT / "conf"

SIDES = (16, 32, 64)
COUPLINGS = ("independent", "ot")
SEEDS = (0, 1, 2, 3, 4)
STEPS = [1, 2, 4, 8, 100]

# archive file -> (experiment config, geometries it holds, loss flags every run must carry)
ARCHIVES: dict[str, tuple[str, tuple[str, ...], dict[str, str]]] = {
    "unet_eval_metrics_l2u.json": (
        "paper_unet",
        ("cylindrical", "euclidean"),
        {"amp_loss_type": "l2", "phase_amplitude_weighting": "false", "vel_loss_type": "l2"},
    ),
    "unet_eval_metrics_l1u_cylindrical.json": (
        "paper_unet_l1",
        ("cylindrical",),
        {"amp_loss_type": "l1", "phase_amplitude_weighting": "false", "vel_loss_type": "l1"},
    ),
    "unet_eval_metrics_l1w_cartesian.json": (
        "paper_unet_v1loss",
        ("euclidean",),
        {"amp_loss_type": "l1", "phase_amplitude_weighting": "true", "vel_loss_type": "l1"},
    ),
}

NAME = re.compile(
    r"^(?:un_(?P<g64>\w+?)_(?P<c64>independent|ot)_scnull_s(?P<s64>\d+)"
    r"|unsz_(?P<g>\w+?)_(?P<c>independent|ot)_(?P<side>\d+)_s(?P<s>\d+))_eval$"
)


def parse(name: str) -> tuple[int, str, str, int]:
    """Split an unprefixed evaluation name into (side, geometry, coupling, seed).

    Args:
        name: E.g. ``un_cylindrical_ot_scnull_s0_eval`` or
            ``unsz_euclidean_independent_32_s4_eval``.

    Returns:
        The field side, geometry, coupling and seed the name encodes.
    """
    match = NAME.match(name)
    assert match is not None, f"unexpected evaluation name {name!r}"
    if match["g64"] is not None:
        return 64, match["g64"], match["c64"], int(match["s64"])
    return int(match["side"]), match["g"], match["c"], int(match["s"])


def expected_names(geometries: tuple[str, ...]) -> set[str]:
    """Every evaluation name of the full grid for the given geometries."""
    names = set()
    for side in SIDES:
        for geometry in geometries:
            for coupling in COUPLINGS:
                for seed in SEEDS:
                    if side == 64:
                        names.add(f"un_{geometry}_{coupling}_scnull_s{seed}_eval")
                    else:
                        names.add(f"unsz_{geometry}_{coupling}_{side}_s{seed}_eval")
    return names


@cache
def load(filename: str) -> dict[str, Any]:
    """One archive, parsed once per test session."""
    archive: dict[str, Any] = json.loads((RESULTS / filename).read_text())
    return archive


def overrides_to_dict(overrides: list[str]) -> dict[str, str]:
    """``key=value`` override strings as a mapping (last one wins)."""
    return dict(item.split("=", 1) for item in overrides)


@pytest.mark.parametrize("filename", sorted(ARCHIVES))
def test_archive_holds_the_full_grid(filename: str) -> None:
    _, geometries, _ = ARCHIVES[filename]
    assert set(load(filename)) == expected_names(geometries)


@pytest.mark.parametrize("filename", sorted(ARCHIVES))
def test_entries_match_their_names(filename: str) -> None:
    for name, run in load(filename).items():
        side, geometry, _, seed = parse(name)
        assert run["field_shape"] == [side, side], name
        assert run["manifold"] == geometry, name
        assert run["seed"] == seed, name
        assert run["num_fields"] == 64, name
        assert run["reference_domain"] == "training transform", name
        assert [row["num_steps"] for row in run["sweep"]] == STEPS, name


@pytest.mark.parametrize("filename", sorted(ARCHIVES))
def test_recorded_overrides_carry_the_protocol(filename: str) -> None:
    _, _, loss = ARCHIVES[filename]
    for name, run in load(filename).items():
        side, geometry, coupling, seed = parse(name)
        record = run["provenance"]
        for key in ("train_overrides", "evaluate_overrides"):
            given = overrides_to_dict(record[key])
            assert given["manifold"] == geometry, (name, key)
            assert given["training.coupling"] == coupling, (name, key)
            assert given["dataset.crop_size"] == f"[{side},{side}]", (name, key)
            for field, value in loss.items():
                assert given[f"training.loss.{field}"] == value, (name, key, field)
        train = overrides_to_dict(record["train_overrides"])
        assert train["training.seed"] == str(seed), name
        assert train["training.epochs"] == "40", name
        evaluate = overrides_to_dict(record["evaluate_overrides"])
        assert evaluate["evaluate.seed"] == str(seed), name
        assert evaluate["evaluate.run_name"] == train["logging.experiment_name"], name


@pytest.mark.parametrize("filename", sorted(ARCHIVES))
def test_training_config_equals_the_documented_experiment(filename: str) -> None:
    """The recorded launch and ``+experiment=<config>`` compose to the same config."""
    experiment, _, _ = ARCHIVES[filename]
    with initialize_config_dir(config_dir=str(CONF), version_base="1.3"):
        for name, run in load(filename).items():
            side, geometry, coupling, seed = parse(name)
            recorded = run["provenance"]["train_overrides"]
            documented = [
                f"+experiment={experiment}",
                f"manifold={geometry}",
                f"training.coupling={coupling}",
                f"dataset.crop_size=[{side},{side}]",
                f"training.seed={seed}",
                f"logging.experiment_name={overrides_to_dict(recorded)['logging.experiment_name']}",
            ]
            launched = compose(config_name="config", overrides=recorded)
            described = compose(config_name="config", overrides=documented)
            assert OmegaConf.to_container(launched, resolve=False) == OmegaConf.to_container(
                described, resolve=False
            ), name
