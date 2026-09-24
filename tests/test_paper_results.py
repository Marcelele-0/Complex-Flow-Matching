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
# The grid runs on the cluster (scripts/cluster/table5_fastmri.sbatch) and the archive
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


# The score-based baseline was trained and scored at 64x64 only, and Appendix D prints
# no diffusion row, so these two arms have no evaluations at 320 and the archive is
# legitimately partial. Named rather than silently tolerated: "twenty of thirty" has to
# mean this and nothing else, or a genuinely lost row hides inside the shortfall.
TABLE5_ABSENT_AT_320 = frozenset(
    {"complex_diffusion_heun_independent", "complex_diffusion_native_independent"}
)

# Appendix D is printed from the `p5_` re-scoring pass. The unprefixed directories are
# superseded: they predate the phase-coherence metric and carry no `phase_lag1_gap` at
# all, and their sliced W2 differs in the third decimal.
TABLE5_PREFIX = "p5_"

# That pass scored every arm with ``evaluate.seed=0``: the five seeds of a 320 row differ
# in TRAINING only and share one prior draw, exactly as at 64x64. It makes the
# arm-against-arm comparisons paired, and it makes the five-seed spread a measure of
# training variance alone -- which is what every asterisk in Appendix D rests on.
TABLE5_EVALUATE_SEED = 0


def table5_expected() -> dict[str, tuple[str, int]]:
    """Every Table 5 evaluation name, mapped to its arm key and seed."""
    return {
        f"{TABLE5_PREFIX}t5_{arm}_s{seed}_eval": (arm, seed)
        for arm in TABLE5_ARMS
        for seed in SEEDS
    }


@table5_needs_archive
def test_table5_archive_holds_the_full_grid() -> None:
    """Every flow arm at 320, five seeds; the diffusion rows exist only at 64x64."""
    archive = set(load(TABLE5_ARCHIVE))
    expected = table5_expected()
    required = {name for name, (arm, _) in expected.items() if arm not in TABLE5_ABSENT_AT_320}
    assert required <= archive, sorted(required - archive)
    assert archive <= set(expected), sorted(archive - set(expected))


@table5_needs_archive
def test_table5_entries_match_their_names() -> None:
    """Each entry is the arm its name claims, at the acquisition's own resolution."""
    expected = table5_expected()
    for name, run in load(TABLE5_ARCHIVE).items():
        arm, seed = expected[name]
        manifold, _, _, steps, _ = TABLE5_ARMS[arm]
        assert run["field_shape"] == [TABLE5_SIDE, TABLE5_SIDE], name
        assert run["manifold"] == manifold, name
        # `seed` in a metrics.json is the EVALUATION seed, fixed at 0 across this pass;
        # the training seed the name encodes lives in the provenance record.
        assert run["seed"] == TABLE5_EVALUATE_SEED, name
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
        assert evaluate["evaluate.seed"] == str(TABLE5_EVALUATE_SEED), name
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


# --- Table 5 (main text): fastMRI knee CORPD at 64x64 -----------------------
#
# The block above is Appendix D, at the acquisition's own 320x320. This one is the
# table in the main text, run on a reduced acquisition matrix (dataset.kspace_crop=64,
# a k-space truncation rather than a crop of the field of view).
#
# It is drawn from TWO sweeps of the same twenty-five checkpoints and the archive keeps
# both, because which number came from which is not recoverable from the printed table.
# `cyfm.evaluate` consumes one seeded generator down its NFE list, so a step count draws
# a different prior depending on its position: the two sweeps agree byte-for-byte on
# their shared prefix 1, 2, 4, 8 and differ at k=100. The printed columns are the
# five-point sweep; the eleven-point one is what the caption's k* is read off, and k* is
# 1, 1, 8, 8 on both. These tests hold that split in place -- a later re-export that
# quietly took k=100 from the dense sweep would move four numbers in the paper.

TABLE5C64_ARCHIVE = "table5_fastmri64_metrics.json"
TABLE5C64_SIDE = 64
TABLE5C64_DENSE_STEPS = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 100]

# name prefix -> (manifold, coupling, overrides that arm adds, swept steps, scored arm).
# The key is the evaluation name without its seed, so a prefix is part of the identity:
# `p5_` and `dense_` are the same checkpoints under two NFE lists.
TABLE5C64_ARMS: dict[str, tuple[str, str, dict[str, str], list[int], str]] = {
    "p5_t5c64_cylindrical_ot": ("cylindrical", "ot", {}, STEPS, "cylindrical_ot"),
    "p5_t5c64_cylindrical_independent": (
        "cylindrical",
        "independent",
        {},
        STEPS,
        "cylindrical_independent",
    ),
    "p5_t5c64_euclidean_ot": ("euclidean", "ot", {}, STEPS, "euclidean_ot"),
    "p5_t5c64_euclidean_independent": (
        "euclidean",
        "independent",
        {},
        STEPS,
        "euclidean_independent",
    ),
    "t5c64_complex_diffusion_heun_independent": (
        "complex_diffusion",
        "independent",
        {"manifold.nfe_mode": "heun"},
        STEPS,
        "complex_diffusion_heun_independent",
    ),
    "t5c64_complex_diffusion_native_independent": (
        "complex_diffusion",
        "independent",
        {"manifold.nfe_mode": "native"},
        TABLE5_NATIVE_STEPS,
        "complex_diffusion_heun_independent",
    ),
    "dense_t5c64_cylindrical_ot": (
        "cylindrical",
        "ot",
        {},
        TABLE5C64_DENSE_STEPS,
        "cylindrical_ot",
    ),
    "dense_t5c64_cylindrical_independent": (
        "cylindrical",
        "independent",
        {},
        TABLE5C64_DENSE_STEPS,
        "cylindrical_independent",
    ),
    "dense_t5c64_euclidean_ot": (
        "euclidean",
        "ot",
        {},
        TABLE5C64_DENSE_STEPS,
        "euclidean_ot",
    ),
    "dense_t5c64_euclidean_independent": (
        "euclidean",
        "independent",
        {},
        TABLE5C64_DENSE_STEPS,
        "euclidean_independent",
    ),
}

# Which arms moved the evaluation seed with the training seed. The flow arms did not:
# every one of their five runs was scored with ``evaluate.seed=0``, so their seeds differ
# in TRAINING only and all five sample from the same prior draw. The diffusion rows did.
#
# Both are defensible and they are not the same measurement. A flow-versus-flow
# comparison is paired on the prior and so lower-variance -- which is what the separation
# criterion in the table relies on, and every asterisk there is flow against flow -- while
# the five-seed spread of a flow arm measures training variance alone. Asserted rather
# than described because the printed table shows a spread without saying which it is.
TABLE5C64_EVAL_SEED_FOLLOWS_TRAINING = frozenset(
    {
        "t5c64_complex_diffusion_heun_independent",
        "t5c64_complex_diffusion_native_independent",
    }
)

table5c64_needs_archive = pytest.mark.skipif(
    not (RESULTS / TABLE5C64_ARCHIVE).exists(),
    reason=f"{TABLE5C64_ARCHIVE} not present; run the cluster grid and export_table5c64.py",
)


def table5c64_eval_seed(arm: str, seed: int) -> int:
    """The prior seed one arm's evaluation was launched with."""
    return seed if arm in TABLE5C64_EVAL_SEED_FOLLOWS_TRAINING else 0


def table5c64_expected() -> dict[str, tuple[str, int]]:
    """Every 64x64 evaluation name, mapped to its arm key and seed."""
    return {f"{arm}_s{seed}_eval": (arm, seed) for arm in TABLE5C64_ARMS for seed in SEEDS}


@table5c64_needs_archive
def test_table5c64_archive_holds_both_sweeps() -> None:
    """Fifty evaluations: the six printed rows plus the four flow arms swept densely."""
    assert set(load(TABLE5C64_ARCHIVE)) == set(table5c64_expected())


@table5c64_needs_archive
def test_table5c64_entries_match_their_names() -> None:
    """Each entry is the arm and the sweep its name claims, on the reduced matrix."""
    expected = table5c64_expected()
    for name, run in load(TABLE5C64_ARCHIVE).items():
        arm, seed = expected[name]
        manifold, _, _, steps, _ = TABLE5C64_ARMS[arm]
        assert run["field_shape"] == [TABLE5C64_SIDE, TABLE5C64_SIDE], name
        assert run["manifold"] == manifold, name
        # `seed` in a metrics.json is the EVALUATION seed, which for the flow arms is 0
        # on all five runs; the training seed lives in the provenance record.
        assert run["seed"] == table5c64_eval_seed(arm, seed), name
        assert run["num_fields"] == 64, name
        assert run["reference_domain"] == "training transform", name
        # The sweep's identity: this is what separates the two grids, and taking a
        # k=100 from the wrong one is a silent change to the printed table.
        assert [row["num_steps"] for row in run["sweep"]] == steps, name
        assert run["reference_peak_modulus"] <= 1.0 + 1e-4, name


@table5c64_needs_archive
def test_table5c64_shared_prefix_is_identical_across_the_two_sweeps() -> None:
    """The grids agree where the generator has drawn the same priors.

    1, 2, 4, 8 open both NFE lists, so they consume the same draws; k=100 sits fifth in
    one list and eleventh in the other and must not be asserted equal. This is the
    evidence that the two sweeps are one protocol rather than two, which is what lets
    the caption read k* off the grid the columns do not come from.

    The tolerance is float noise, not slack: two runs of the same draw agree to about
    4e-9 here, while the two grids differ at k=100 by 0.009 -- six orders of magnitude
    apart, so no real difference in the draw can hide under it.
    """
    runs = load(TABLE5C64_ARCHIVE)
    for arm, coupling in (
        ("cylindrical", "ot"),
        ("cylindrical", "independent"),
        ("euclidean", "ot"),
        ("euclidean", "independent"),
    ):
        for seed in SEEDS:
            short = runs[f"p5_t5c64_{arm}_{coupling}_s{seed}_eval"]["sweep"]
            dense = runs[f"dense_t5c64_{arm}_{coupling}_s{seed}_eval"]["sweep"]
            for steps in (1, 2, 4, 8):
                a = next(row for row in short if row["num_steps"] == steps)
                b = next(row for row in dense if row["num_steps"] == steps)
                assert a["sliced_w2_complex"] == pytest.approx(b["sliced_w2_complex"], abs=1e-6), (
                    arm,
                    coupling,
                    seed,
                    steps,
                )


@table5c64_needs_archive
def test_table5c64_recorded_overrides_carry_the_protocol() -> None:
    """Every row was launched on the reduced matrix, and scored its own checkpoint."""
    expected = table5c64_expected()
    for name, run in load(TABLE5C64_ARCHIVE).items():
        arm, seed = expected[name]
        manifold, coupling, extra, _, scored = TABLE5C64_ARMS[arm]
        record = run["provenance"]
        for key in ("train_overrides", "evaluate_overrides"):
            given = overrides_to_dict(record[key])
            for field in ("+experiment", "manifold", "training.coupling", "dataset.kspace_crop"):
                assert field in given, (name, key, field)
            assert given["+experiment"] == "table5_fastmri", (name, key)
            assert given["manifold"] == manifold, (name, key)
            assert given["training.coupling"] == coupling, (name, key)
            # The reduction is in k-space, and it is the whole point of this block:
            # an image-domain crop would cut the field of view instead of lowering
            # the acquisition matrix, and would not be the experiment the table names.
            assert given["dataset.kspace_crop"] == "64", (name, key)
            assert "dataset.crop_size" not in given, (name, key)
        train = overrides_to_dict(record["train_overrides"])
        assert train["training.seed"] == str(seed), name
        assert train["training.epochs"] == "40", name
        evaluate = overrides_to_dict(record["evaluate_overrides"])
        assert evaluate["evaluate.seed"] == str(table5c64_eval_seed(arm, seed)), name
        assert evaluate["dataset.role"] == "all", name
        assert evaluate["dataset.store"] == "val.h5", name
        assert evaluate["evaluate.num_fields"] == "64", name
        for field, value in extra.items():
            assert evaluate[field] == value, (name, field)
        assert evaluate["evaluate.run_name"] == train["logging.experiment_name"], name
        assert train["logging.experiment_name"] == f"t5c64_{scored}_s{seed}", name


@table5c64_needs_archive
def test_table5c64_training_config_equals_the_documented_experiment() -> None:
    """The recorded launch and ``+experiment=table5_fastmri`` compose to one config."""
    expected = table5c64_expected()
    with initialize_config_dir(config_dir=str(CONF), version_base="1.3"):
        for name, run in load(TABLE5C64_ARCHIVE).items():
            arm, seed = expected[name]
            manifold, coupling, _, _, _ = TABLE5C64_ARMS[arm]
            recorded = run["provenance"]["train_overrides"]
            given = overrides_to_dict(recorded)
            documented = [
                "+experiment=table5_fastmri",
                f"manifold={manifold}",
                f"training.coupling={coupling}",
                *(
                    f"{key}={given[key]}"
                    for key in (
                        "dataset.data_dir",
                        "dataset.store",
                        "dataset.kspace_crop",
                        "manifold.sigma_max",
                    )
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


# --- Table 2, speech block: LibriSpeech STFT at 64x64 -----------------------
#
# The last main-text block to get an archive. Three prefixes exist on the cluster and
# they are different experiments rather than retries: `t6_` and `t6e10_` are shorter
# budgets whose numbers differ in the second decimal, and only `t6e40_` reproduces the
# printed table. The launcher defaults to two epochs, which is how the short cohorts
# came to exist, so the published one carries its epoch count in its name and these
# tests pin it.
#
# Unlike the knee blocks, this cohort moves the EVALUATION seed with the training seed,
# so its five seeds vary both the weights and the prior draw. That makes its spread the
# wider quantity of the two and its arm-to-arm comparisons unpaired -- worth knowing
# before the two blocks' error bars are read as if they meant the same thing.

TABLE6_ARCHIVE = "table6_audio_metrics.json"
TABLE6_SIDE = 64
TABLE6_PREFIX = "t6e40_"
TABLE6_ARMS: dict[str, tuple[str, str]] = {
    "cylindrical_ot": ("cylindrical", "ot"),
    "cylindrical_independent": ("cylindrical", "independent"),
    "euclidean_ot": ("euclidean", "ot"),
    "euclidean_independent": ("euclidean", "independent"),
}

table6_needs_archive = pytest.mark.skipif(
    not (RESULTS / TABLE6_ARCHIVE).exists(),
    reason=f"{TABLE6_ARCHIVE} not present; run the cluster grid and export_table6.py",
)


def table6_expected() -> dict[str, tuple[str, int]]:
    """Every speech evaluation name, mapped to its arm key and seed."""
    return {
        f"{TABLE6_PREFIX}{arm}_s{seed}_eval": (arm, seed) for arm in TABLE6_ARMS for seed in SEEDS
    }


@table6_needs_archive
def test_table6_archive_holds_the_full_grid() -> None:
    """Twenty evaluations: four flow arms of five seeds. No diffusion row here."""
    assert set(load(TABLE6_ARCHIVE)) == set(table6_expected())


@table6_needs_archive
def test_table6_entries_match_their_names() -> None:
    """Each entry is the arm its name claims, on the speech cohort."""
    expected = table6_expected()
    for name, run in load(TABLE6_ARCHIVE).items():
        arm, seed = expected[name]
        manifold, _ = TABLE6_ARMS[arm]
        assert run["field_shape"] == [TABLE6_SIDE, TABLE6_SIDE], name
        assert run["manifold"] == manifold, name
        assert run["dataset"] == "librispeech_stft", name
        assert run["seed"] == seed, name
        assert run["num_fields"] == 64, name
        assert run["reference_domain"] == "training transform", name
        assert [row["num_steps"] for row in run["sweep"]] == STEPS, name


@table6_needs_archive
def test_table6_recorded_overrides_carry_the_protocol() -> None:
    """Forty epochs, the held-out speakers, and an evaluation seed that tracks training."""
    expected = table6_expected()
    for name, run in load(TABLE6_ARCHIVE).items():
        arm, seed = expected[name]
        manifold, coupling = TABLE6_ARMS[arm]
        record = run["provenance"]
        for key in ("train_overrides", "evaluate_overrides"):
            given = overrides_to_dict(record[key])
            for field in ("+experiment", "manifold", "training.coupling"):
                assert field in given, (name, key, field)
            assert given["+experiment"] == "table6_audio", (name, key)
            assert given["manifold"] == manifold, (name, key)
            assert given["training.coupling"] == coupling, (name, key)
        train = overrides_to_dict(record["train_overrides"])
        assert train["training.seed"] == str(seed), name
        # The launcher defaults to 2. A cohort run without EPOCHS=40 produces numbers
        # that look like these and are not, so the epoch count is asserted, not assumed.
        assert train["training.epochs"] == "40", name
        evaluate = overrides_to_dict(record["evaluate_overrides"])
        assert evaluate["evaluate.seed"] == str(seed), name
        # Speech hashes one store by speaker, so the held-out role is what keeps the
        # reference segments from speakers the arm trained on.
        assert evaluate["dataset.role"] == "holdout", name
        assert evaluate["evaluate.num_fields"] == "64", name
        assert evaluate["evaluate.run_name"] == train["logging.experiment_name"], name
        assert train["logging.experiment_name"] == f"{TABLE6_PREFIX}{arm}_s{seed}", name


@table6_needs_archive
def test_table6_training_config_equals_the_documented_experiment() -> None:
    """The recorded launch and ``+experiment=table6_audio`` compose to one config."""
    expected = table6_expected()
    with initialize_config_dir(config_dir=str(CONF), version_base="1.3"):
        for name, run in load(TABLE6_ARCHIVE).items():
            arm, seed = expected[name]
            manifold, coupling = TABLE6_ARMS[arm]
            recorded = run["provenance"]["train_overrides"]
            given = overrides_to_dict(recorded)
            documented = [
                "+experiment=table6_audio",
                f"manifold={manifold}",
                f"training.coupling={coupling}",
                *(f"{key}={given[key]}" for key in ("dataset.data_dir",) if key in given),
                f"training.epochs={given['training.epochs']}",
                f"training.seed={seed}",
                f"logging.experiment_name={given['logging.experiment_name']}",
            ]
            launched = compose(config_name="config", overrides=recorded)
            described = compose(config_name="config", overrides=documented)
            assert OmegaConf.to_container(launched, resolve=False) == OmegaConf.to_container(
                described, resolve=False
            ), name


# --- The amplitude-weighted knee arms ---------------------------------------
#
# One sentence in Section 5.2 compares the unweighted product metric against an
# amplitude-weighted phase term, and until this archive existed the weighted half of
# that comparison was in no file here: the claim was the only one in the paper without
# a source. The runs are the `t5wc64_` cohort, published from the same `p5_` re-scoring
# pass as every other block, for the same reason -- the unprefixed pass predates the
# phase-coherence metric and reproduces neither printed value.

TABLE5W_ARCHIVE = "table5_fastmri64_weighted_metrics.json"
TABLE5W_PREFIX = "p5_t5wc64_"
TABLE5W_ARMS = (
    "cylindrical_ot",
    "cylindrical_independent",
    "euclidean_ot",
    "euclidean_independent",
)

table5w_needs_archive = pytest.mark.skipif(
    not (RESULTS / TABLE5W_ARCHIVE).exists(),
    reason=f"{TABLE5W_ARCHIVE} not present; run the cluster grid and export_table5w.py",
)


@table5w_needs_archive
def test_table5w_archive_holds_the_full_grid() -> None:
    """Twenty evaluations: four arms of five seeds."""
    expected = {f"{TABLE5W_PREFIX}{arm}_s{seed}_eval" for arm in TABLE5W_ARMS for seed in SEEDS}
    assert set(load(TABLE5W_ARCHIVE)) == expected


@table5w_needs_archive
def test_table5w_runs_carry_the_weighting_and_the_crop() -> None:
    """The two overrides that make this cohort what it is, asserted rather than assumed."""
    for name, run in load(TABLE5W_ARCHIVE).items():
        given = overrides_to_dict(run["provenance"]["train_overrides"])
        assert given["training.loss.phase_amplitude_weighting"] == "true", name
        assert given["dataset.kspace_crop"] == "64", name
        assert run["field_shape"] == [64, 64], name


@table5w_needs_archive
def test_table5w_reproduces_the_comparison_the_text_makes() -> None:
    """Section 5.2's sentence, checked against both archives at once.

    The unweighted values come from the main 64x64 archive and the weighted ones from
    this archive, which is the point: the claim spans two cohorts and neither file alone
    can support it.
    """
    unweighted = load(TABLE5C64_ARCHIVE)
    weighted = load(TABLE5W_ARCHIVE)

    def converged(archive: dict[str, Any], prefix: str, metric: str) -> float:
        values = [
            next(
                row[metric]
                for row in archive[f"{prefix}cylindrical_ot_s{seed}_eval"]["sweep"]
                if row["num_steps"] == 100
            )
            for seed in SEEDS
        ]
        return sum(values) / len(values)

    assert converged(unweighted, "p5_t5c64_", "spatial_lag1_gap") == pytest.approx(0.0153, abs=5e-5)
    assert converged(weighted, TABLE5W_PREFIX, "spatial_lag1_gap") == pytest.approx(
        0.0219, abs=5e-5
    )
    assert converged(unweighted, "p5_t5c64_", "phase_lag1_gap") == pytest.approx(0.1057, abs=5e-5)
    assert converged(weighted, TABLE5W_PREFIX, "phase_lag1_gap") == pytest.approx(0.1656, abs=5e-5)
