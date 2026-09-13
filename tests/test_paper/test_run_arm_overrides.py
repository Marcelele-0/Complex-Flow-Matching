"""`run_arm.sh`'s environment overrides, checked without training anything.

The runner is what the WCSS array calls for every cell of every table, so an override
that silently fails to reach Hydra costs a whole grid rather than one run.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

RUNNER = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "paper" / "run_arm.sh"
pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _capture(tmp_path: pathlib.Path, **env: str) -> tuple[str, str]:
    """Run the runner with `uv` stubbed out, and return the train and evaluate commands."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.txt"
    (bin_dir / "uv").write_text(f'#!/usr/bin/env bash\necho "$@" >> "{log}"\n')
    (bin_dir / "uv").chmod(0o755)

    subprocess.run(
        ["bash", str(RUNNER), "arm", "native", "cylindrical", "ot", "0", "1"],
        capture_output=True,
        text=True,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path), **env},
        check=False,
    )
    lines = log.read_text().splitlines() if log.exists() else []
    train = next((line for line in lines if "cfm.train" in line), "")
    evaluate = next((line for line in lines if "cfm.evaluate" in line), "")
    return train, evaluate


def test_eval_store_reaches_evaluation_and_not_training(tmp_path: pathlib.Path) -> None:
    """Table 5's protocol: train on one store, score against another.

    fastMRI's train and val cohorts are patient-disjoint already, so the evaluation
    store *is* the split. Sending it to training as well would train on the very
    fields the run is about to be scored against.
    """
    train, evaluate = _capture(
        tmp_path, EXPERIMENT="table5_fastmri", EVAL_STORE="val.h5", HOLDOUT_ROLE="all"
    )
    assert "dataset.store=val.h5" in evaluate
    assert "dataset.role=all" in evaluate
    assert "dataset.store=" not in train
    assert "dataset.role=" not in train


def test_without_eval_store_nothing_is_overridden(tmp_path: pathlib.Path) -> None:
    """The synthetic cohorts hash one store into two roles and must keep working."""
    train, evaluate = _capture(tmp_path, EXPERIMENT="table5_fastmri", HOLDOUT_ROLE="holdout")
    assert "dataset.store=" not in evaluate
    assert "dataset.role=holdout" in evaluate
    assert "dataset.role=" not in train


def test_the_experiment_reaches_both_commands(tmp_path: pathlib.Path) -> None:
    train, evaluate = _capture(tmp_path, EXPERIMENT="table5_fastmri")
    assert "+experiment=table5_fastmri" in train
    assert "+experiment=table5_fastmri" in evaluate


def test_train_and_eval_stores_go_to_their_own_commands(tmp_path: pathlib.Path) -> None:
    """Hydra refuses the same override twice, so neither store may live in COMMON."""
    train, evaluate = _capture(
        tmp_path,
        EXPERIMENT="table5_fastmri",
        TRAIN_STORE="train.h5",
        EVAL_STORE="val.h5",
        HOLDOUT_ROLE="all",
    )
    assert "dataset.store=train.h5" in train
    assert "dataset.store=val.h5" not in train
    assert "dataset.store=val.h5" in evaluate
    assert "dataset.store=train.h5" not in evaluate
    assert evaluate.count("dataset.store=") == 1


def test_data_dir_reaches_both_so_a_job_can_stage_to_node_local(
    tmp_path: pathlib.Path,
) -> None:
    train, evaluate = _capture(
        tmp_path, EXPERIMENT="table5_fastmri", DATA_DIR="/scratch/stage", EVAL_STORE="val.h5"
    )
    assert "dataset.data_dir=/scratch/stage" in train
    assert "dataset.data_dir=/scratch/stage" in evaluate
