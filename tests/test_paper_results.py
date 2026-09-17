"""Integrity of the archived evaluations behind Tables 2-3, 5 and the appendix table.

Every entry of the archives in ``docs/reproduce/paper_results/`` carries the Hydra
overrides its training run was launched with. These tests check that each archive
holds the full grid with the expected metadata, and -- the reproducibility claim --
that composing the recorded overrides gives exactly the configuration of the
documented experiment (``conf/experiment/*.yaml``) plus the per-run arguments.

Table 5 gets a parallel set of four tests rather than a fourth ``ARCHIVES`` entry.
Its grid is not the ``(side, geometry, coupling, seed)`` shape ``parse`` assumes:
there is no crop, the fields are the acquisition's own 320x320, and two of its arms
share one manifold under different sampling regimes. Forcing it into the
parametrised tests above would mean loosening the assertions that hold the three
synthetic archives -- the paper's reproducibility claim -- to their protocol.
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


# --- Table 5: fastMRI knee CORPD at 320x320 ---------------------------------
#
# The grid runs on the cluster (scripts/wcss/table5_fastmri.sbatch) and the archive
# is written by scripts/paper/export_table5.py, so these tests skip until it is
# present rather than failing a checkout that has never seen the cluster. They are
# what turns "the array came back" into "the array came back whole and on protocol".

TABLE5_ARCHIVE = "table5_fastmri_metrics.json"
TABLE5_SIDE = 320
TABLE5_NATIVE_STEPS = [1000]

# arm -> (manifold, coupling, overrides that arm adds, its swept step counts,
#         the arm whose checkpoint it scored).
# The two diffusion arms are one manifold under two sampling regimes: they share a
# training run, so the many-step arm names the matched arm as what it scored.
TABLE5_ARMS: dict[str, tuple[str, str, dict[str, str], list[int], str]] = {
    "cylindrical_ot": ("cylindrical", "ot", {}, STEPS, "cylindrical_ot"),
    "cylindrical_independent": ("cylindrical", "independent", {}, STEPS, "cylindrical_independent"),
    "euclidean_ot": ("euclidean", "ot", {}, STEPS, "euclidean_ot"),
    "euclidean_independent": ("euclidean", "independent", {}, STEPS, "euclidean_independent"),
    "complex_diffusion_heun_independent": (
        "complex_diffusion",
        "independent",
        {"manifold.nfe_mode": "heun"},
        STEPS,
        "complex_diffusion_heun_independent",
    ),
    "complex_diffusion_native_independent": (
        "complex_diffusion",
        "independent",
        {"manifold.nfe_mode": "native"},
        TABLE5_NATIVE_STEPS,
        "complex_diffusion_heun_independent",
    ),
}

table5_needs_archive = pytest.mark.skipif(
    not (RESULTS / TABLE5_ARCHIVE).exists(),
    reason=f"{TABLE5_ARCHIVE} not present; run the cluster grid and export_table5.py",
)


def table5_expected() -> dict[str, tuple[str, int]]:
    """Every Table 5 evaluation name, mapped to its arm key and seed."""
    return {
        f"t5_{arm}_s{seed}_eval": (arm, seed) for arm in TABLE5_ARMS for seed in SEEDS
    }


@table5_needs_archive
def test_table5_archive_holds_the_full_grid() -> None:
    """Thirty evaluations: six rows of five seeds, from twenty-five training runs."""
    assert set(load(TABLE5_ARCHIVE)) == set(table5_expected())


@table5_needs_archive
def test_table5_entries_match_their_names() -> None:
    """Each entry is the arm its name claims, at the acquisition's own resolution."""
    expected = table5_expected()
    for name, run in load(TABLE5_ARCHIVE).items():
        arm, seed = expected[name]
        manifold, _, _, steps, _ = TABLE5_ARMS[arm]
        assert run["field_shape"] == [TABLE5_SIDE, TABLE5_SIDE], name
        assert run["manifold"] == manifold, name
        assert run["seed"] == seed, name
        assert run["num_fields"] == 64, name
        assert run["reference_domain"] == "training transform", name
        assert [row["num_steps"] for row in run["sweep"]] == steps, name
        # The guard's evidence, not just its verdict: peak normalisation is what
        # makes these absolute numbers comparable with the synthetic tables.
        assert run["reference_peak_modulus"] <= 1.0 + 1e-4, name


@table5_needs_archive
def test_table5_recorded_overrides_carry_the_protocol() -> None:
    """Every row was launched with the same protocol, and scored its own checkpoint."""
    expected = table5_expected()
    for name, run in load(TABLE5_ARCHIVE).items():
        arm, seed = expected[name]
        manifold, coupling, extra, _, scored = TABLE5_ARMS[arm]
        record = run["provenance"]
        for key in ("train_overrides", "evaluate_overrides"):
            given = overrides_to_dict(record[key])
            # Presence first: a command that dropped a protocol key should say which
            # row and which key, not raise a bare KeyError from thirty entries deep.
            for field in ("+experiment", "manifold", "training.coupling"):
                assert field in given, (name, key, field)
            assert given["+experiment"] == "table5_fastmri", (name, key)
            assert given["manifold"] == manifold, (name, key)
            assert given["training.coupling"] == coupling, (name, key)
            # 320x320 is the store's own resolution; a crop would be a different
            # protocol from the one the #76 gate measured the batch size at.
            assert "dataset.crop_size" not in given, (name, key)
        train = overrides_to_dict(record["train_overrides"])
        assert train["training.seed"] == str(seed), name
        assert train["training.epochs"] == "40", name
        evaluate = overrides_to_dict(record["evaluate_overrides"])
        assert evaluate["evaluate.seed"] == str(seed), name
        assert evaluate["dataset.role"] == "all", name
        assert evaluate["evaluate.num_fields"] == "64", name
        for field, value in extra.items():
            assert evaluate[field] == value, (name, field)
        # The many-step row scored the matched row's checkpoint rather than one of
        # its own, so its evaluation points at that training run's name.
        assert evaluate["evaluate.run_name"] == train["logging.experiment_name"], name
        assert train["logging.experiment_name"] == f"t5_{scored}_s{seed}", name


@table5_needs_archive
def test_table5_training_config_equals_the_documented_experiment() -> None:
    """The recorded launch and ``+experiment=table5_fastmri`` compose to one config."""
    expected = table5_expected()
    with initialize_config_dir(config_dir=str(CONF), version_base="1.3"):
        for name, run in load(TABLE5_ARCHIVE).items():
            arm, seed = expected[name]
            manifold, coupling, _, _, _ = TABLE5_ARMS[arm]
            recorded = run["provenance"]["train_overrides"]
            given = overrides_to_dict(recorded)
            # The store path and the run name are properties of the machine the grid
            # ran on, not of the protocol, so they are carried over rather than
            # asserted -- exactly as the synthetic archives carry experiment_name.
            documented = [
                "+experiment=table5_fastmri",
                f"manifold={manifold}",
                f"training.coupling={coupling}",
                *(
                    f"{key}={given[key]}"
                    for key in ("dataset.data_dir", "dataset.store", "manifold.sigma_max")
                    if key in given
                ),
                *(
                    [f"manifold.nfe_mode={given['manifold.nfe_mode']}"]
                    if "manifold.nfe_mode" in given
                    else []
                ),
                f"training.epochs={given['training.epochs']}",
                f"training.seed={seed}",
                f"logging.experiment_name={given['logging.experiment_name']}",
            ]
            launched = compose(config_name="config", overrides=recorded)
            described = compose(config_name="config", overrides=documented)
            assert OmegaConf.to_container(launched, resolve=False) == OmegaConf.to_container(
                described, resolve=False
            ), name
