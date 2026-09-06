"""Unit tests for cfmri-suite CLI and Slurm launcher integration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cfm.suite import (
    DEFAULT_LAUNCHER_SCRIPT,
    build_command,
    build_hydra_args,
    build_launcher_command,
    build_parser,
    main,
    run_evaluation,
    slurm_jobs,
)


def test_parser_defaults() -> None:
    """Verify default parser options when no arguments are passed."""
    parser = build_parser()
    args = parser.parse_args([])
    assert args.matrix is False
    assert args.seeds is False
    assert args.smoke is False
    assert args.eval is False
    assert args.stats is False
    assert args.slurm is False
    assert args.submit is False
    assert args.gpus == 4
    assert args.nodes == 1
    assert args.experiment == "suite-matrix"
    assert args.extra == []


def test_parser_slurm_flags() -> None:
    """Verify parsing of Slurm-specific flags."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "--slurm",
            "--submit",
            "--gpus",
            "8",
            "--nodes",
            "2",
            "--experiment",
            "athena-ddp-matrix",
        ]
    )
    assert args.slurm is True
    assert args.submit is True
    assert args.gpus == 8
    assert args.nodes == 2
    assert args.experiment == "athena-ddp-matrix"

    # Verify -e short flag
    args_short = parser.parse_args(["-e", "short-exp"])
    assert args_short.experiment == "short-exp"


def test_parser_training_flags_and_extra() -> None:
    """Verify parsing of matrix, seeds, smoke, eval, and extra Hydra flags."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "--matrix",
            "--seeds",
            "--smoke",
            "--eval",
            "--extra",
            "training.lr=1e-4",
            "dataset.batch_size=8",
        ]
    )
    assert args.matrix is True
    assert args.seeds is True
    assert args.smoke is True
    assert args.eval is True
    assert args.extra == ["training.lr=1e-4", "dataset.batch_size=8"]


def test_build_hydra_args_combinations() -> None:
    """Test Hydra argument generation across matrix, seeds, and smoke combinations."""
    # 1. Baseline
    args = argparse.Namespace(matrix=False, seeds=False, smoke=False, extra=[])
    hydra_args, is_multirun = build_hydra_args(args)
    assert hydra_args == []
    assert is_multirun is False

    # 2. Smoke only (single run, no -m)
    args = argparse.Namespace(matrix=False, seeds=False, smoke=True, extra=[])
    hydra_args, is_multirun = build_hydra_args(args)
    assert is_multirun is False
    assert "-m" not in hydra_args
    assert "training.epochs=3" in hydra_args

    # 3. Matrix only (multirun, -m at start)
    args = argparse.Namespace(matrix=True, seeds=False, smoke=False, extra=[])
    hydra_args, is_multirun = build_hydra_args(args)
    assert is_multirun is True
    assert hydra_args[0] == "-m"
    assert "dataset=skm_tea,fastmri_local" in hydra_args
    assert "manifold=cylindrical,euclidean,complex_diffusion" in hydra_args
    assert "model=c_unet" in hydra_args

    # 4. Seeds only (multirun, -m at start)
    args = argparse.Namespace(matrix=False, seeds=True, smoke=False, extra=[])
    hydra_args, is_multirun = build_hydra_args(args)
    assert is_multirun is True
    assert hydra_args[0] == "-m"
    assert "training.seed=42,123,999" in hydra_args

    # 5. Full combo: matrix + seeds + smoke + extra
    args = argparse.Namespace(
        matrix=True,
        seeds=True,
        smoke=True,
        extra=["optimizer.lr=2e-4"],
    )
    hydra_args, is_multirun = build_hydra_args(args)
    assert is_multirun is True
    assert hydra_args[0] == "-m"
    assert "dataset=skm_tea,fastmri_local" in hydra_args
    assert "training.seed=42,123,999" in hydra_args
    assert "training.epochs=3" in hydra_args
    assert hydra_args[-1] == "optimizer.lr=2e-4"


def test_build_command_local_mode() -> None:
    """Verify local execution command generation when --slurm is not active."""
    args = argparse.Namespace(
        slurm=False,
        matrix=True,
        seeds=False,
        smoke=True,
        extra=["logging.wandb=false"],
    )
    cmd = build_command(args)
    assert cmd[0] == sys.executable
    assert cmd[1] == "src/cfm/train.py"
    assert cmd[2] == "-m"
    assert "dataset=skm_tea,fastmri_local" in cmd
    assert "training.epochs=3" in cmd
    assert cmd[-1] == "logging.wandb=false"


def test_build_launcher_command_dry_run() -> None:
    """Verify launcher command building for safe dry-run (default without --submit)."""
    args = build_parser().parse_args(["--slurm", "--smoke"])
    (experiment, overrides) = slurm_jobs(args)[0]
    cmd = build_launcher_command(args, experiment, overrides)

    assert cmd[0] == str(DEFAULT_LAUNCHER_SCRIPT)
    assert "--submit" not in cmd
    assert cmd[cmd.index("-g") + 1] == "4"
    assert cmd[cmd.index("-N") + 1] == "1"
    assert cmd[cmd.index("-e") + 1] == "suite-matrix"
    assert cmd[cmd.index("--") + 1 :] == overrides
    assert "training.epochs=3" in overrides


def test_build_launcher_command_submit() -> None:
    """Verify launcher command building when --submit is specified."""
    args = build_parser().parse_args(
        ["--slurm", "--submit", "--gpus", "8", "--nodes", "2", "-e", "a100-full-matrix"]
    )
    custom_script = Path("/custom/path/launch_slurm.sh")
    cmd = build_launcher_command(args, "a100-full-matrix", ["manifold=cylindrical"], custom_script)

    assert cmd[0] == str(custom_script)
    assert cmd[1] == "--submit"
    assert cmd[cmd.index("-g") + 1] == "8"
    assert cmd[cmd.index("-N") + 1] == "2"
    assert cmd[cmd.index("-e") + 1] == "a100-full-matrix"
    assert cmd[cmd.index("--") + 1 :] == ["manifold=cylindrical"]


def test_slurm_jobs_never_forward_multirun_flag() -> None:
    """Regression: -m must never reach launch_slurm.sh.

    train_ddp.sbatch prepends its own overrides before the caller's, and Hydra's
    parser rejects -m once any override precedes it, so a forwarded -m would make
    every submitted job die with `unrecognized arguments` after the queue wait.
    """
    args = build_parser().parse_args(["--slurm", "--matrix", "--seeds", "--smoke"])
    for experiment, overrides in slurm_jobs(args):
        cmd = build_launcher_command(args, experiment, overrides)
        assert "-m" not in overrides
        assert "-m" not in cmd
        assert not any("," in override for override in overrides)


def test_slurm_jobs_expand_grid_with_unique_experiment_names() -> None:
    """Each grid point becomes its own job under its own experiment name.

    experiment_name keys outputs/state/<name>/last.pt and train_ddp.sbatch forces
    auto_resume=true, so points sharing a name would resume from each other.
    """
    args = build_parser().parse_args(["--slurm", "--matrix", "--seeds", "-e", "run"])
    jobs = slurm_jobs(args)

    # 2 datasets x 3 manifolds x 1 model x 3 seeds
    assert len(jobs) == 18
    names = [experiment for experiment, _ in jobs]
    assert len(set(names)) == len(names)
    assert all(name.startswith("run-") for name in names)

    assert (
        "run-skm_tea-cylindrical-c_unet-42",
        [
            "dataset=skm_tea",
            "manifold=cylindrical",
            "model=c_unet",
            "training.seed=42",
        ],
    ) in jobs


def test_slurm_jobs_single_point_without_sweep() -> None:
    """Without --matrix/--seeds a single job runs under the bare experiment name."""
    args = build_parser().parse_args(["--slurm", "-e", "solo", "--extra", "manifold=euclidean"])
    assert slurm_jobs(args) == [("solo", ["manifold=euclidean"])]


def test_build_command_all_combinations() -> None:
    """Test matrix, smoke, seeds flag permutations for local execution."""
    # Permutation A: local, no matrix, no seeds, no smoke
    p1 = build_parser().parse_args([])
    cmd1 = build_command(p1)
    assert cmd1 == [sys.executable, "src/cfm/train.py"]

    # Permutation B: local, matrix only -- -m must lead, Hydra rejects it elsewhere
    p2 = build_parser().parse_args(["--matrix"])
    cmd2 = build_command(p2)
    assert cmd2[0] == sys.executable
    assert cmd2[2] == "-m"

    # Permutation C: local, matrix + smoke + seeds keeps a single leading -m
    p3 = build_parser().parse_args(["--matrix", "--smoke", "--seeds"])
    cmd3 = build_command(p3)
    assert cmd3[2] == "-m"
    assert cmd3.count("-m") == 1


def test_main_local_execution_mocked() -> None:
    """Test main() in local execution mode invoking subprocess.run."""
    with patch("cfm.suite.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        ret = main(["--smoke"])
        assert ret == 0
        mock_run.assert_called_once()
        called_cmd = mock_run.call_args[0][0]
        assert called_cmd[0] == sys.executable
        assert called_cmd[1] == "src/cfm/train.py"
        assert "training.epochs=3" in called_cmd


def test_main_slurm_execution_mocked() -> None:
    """Test main() with --slurm invoking the Slurm launcher script."""
    with patch("cfm.suite.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        ret = main(["--slurm", "--submit", "--gpus", "4", "--experiment", "test-run"])
        assert ret == 0
        mock_run.assert_called_once()
        called_cmd = mock_run.call_args[0][0]
        assert called_cmd[0] == str(DEFAULT_LAUNCHER_SCRIPT)
        assert "--submit" in called_cmd
        assert called_cmd[called_cmd.index("-g") + 1] == "4"
        assert called_cmd[called_cmd.index("-e") + 1] == "test-run"


def test_main_slurm_matrix_submits_one_job_per_grid_point() -> None:
    """--matrix dispatches six independent submissions, not one multirun job."""
    with patch("cfm.suite.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        ret = main(["--slurm", "--submit", "--matrix"])
        assert ret == 0
        assert mock_run.call_count == 6
        names = [call[0][0][call[0][0].index("-e") + 1] for call in mock_run.call_args_list]
        assert len(set(names)) == 6


def test_main_slurm_stops_at_first_launcher_failure() -> None:
    """A rejected submission stops the sweep instead of spamming the queue."""
    with patch("cfm.suite.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=2)
        ret = main(["--slurm", "--submit", "--matrix"])
        assert ret == 2
        mock_run.assert_called_once()


def test_main_slurm_eval_guidance(capsys) -> None:
    """--eval with --slurm points at evaluate.py, not at the multirun-only --eval."""
    with patch("cfm.suite.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        ret = main(["--slurm", "--eval", "--matrix"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "[INFO] --eval does not apply to --slurm" in captured.out
        assert "evaluate.run_name=<experiment>" in captured.out


def test_main_rejects_flags_after_extra() -> None:
    """--extra is REMAINDER, so a trailing suite flag must fail loudly, not silently."""
    with pytest.raises(SystemExit):
        main(["--extra", "optimizer.lr=1e-4", "--slurm", "--submit"])


def test_main_rejects_submit_without_slurm() -> None:
    """--submit without --slurm must not silently start a local run."""
    with pytest.raises(SystemExit):
        main(["--submit", "--matrix"])


def test_main_standalone_eval_no_runs(capsys) -> None:
    """Test standalone --eval when no multirun directories exist."""
    with patch("cfm.suite.glob.glob", return_value=[]):
        ret = main(["--eval"])
        assert ret == 1
        captured = capsys.readouterr()
        assert "No multirun outputs found" in captured.out


def test_main_standalone_eval_found_run(tmp_path: Path) -> None:
    """Test standalone --eval when multirun directory exists triggers run_evaluation."""
    multirun_dir = tmp_path / "outputs" / "multirun" / "2026-09-05"
    multirun_dir.mkdir(parents=True)
    with (
        patch("cfm.suite.glob.glob", return_value=[str(multirun_dir)]),
        patch("cfm.suite.run_evaluation") as mock_eval,
    ):
        ret = main(["--eval"])
        assert ret == 0
        mock_eval.assert_called_once_with(str(multirun_dir))


def test_main_local_with_eval_triggers_run_evaluation(tmp_path: Path) -> None:
    """Test local training with --eval triggers run_evaluation upon zero returncode."""
    multirun_dir = tmp_path / "outputs" / "multirun" / "2026-09-05"
    multirun_dir.mkdir(parents=True)
    with (
        patch("cfm.suite.subprocess.run", return_value=MagicMock(returncode=0)),
        patch("cfm.suite.glob.glob", return_value=[str(multirun_dir)]),
        patch("cfm.suite.run_evaluation") as mock_eval,
    ):
        ret = main(["--matrix", "--eval"])
        assert ret == 0
        mock_eval.assert_called_once_with(str(multirun_dir))


def test_run_evaluation_mocked(tmp_path: Path) -> None:
    """Verify run_evaluation inspects job dirs and writes eval_summary.json."""
    multirun_dir = tmp_path / "multirun"
    job_dir = multirun_dir / "0"
    ckpt_dir = job_dir / "checkpoints"
    hydra_dir = job_dir / ".hydra"
    ckpt_dir.mkdir(parents=True)
    hydra_dir.mkdir(parents=True)

    ckpt_file = ckpt_dir / "model_epoch_3.pt"
    ckpt_file.write_text("dummy")

    overrides_file = hydra_dir / "overrides.yaml"
    overrides_file.write_text(
        "dataset=skm_tea\nmanifold=cylindrical\nmodel=c_unet\ntraining.epochs=3\ntraining.seed=42\n"
    )

    with patch("cfm.suite.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        run_evaluation(multirun_dir)
        # Should evaluate cylindrical checkpoint and varnet baseline
        assert mock_run.call_count >= 2

    summary_file = multirun_dir / "eval_summary.json"
    assert summary_file.exists()


def test_main_stats_no_extra(capsys) -> None:
    """Verify main(['--stats']) prints usage instructions and help."""
    ret = main(["--stats"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Usage: cfmri-suite --stats --extra" in captured.out
    assert "--target" in captured.out


def test_main_stats_with_extra_demo(capsys) -> None:
    """Verify main(['--stats', '--extra', '--demo']) executes stats demo."""
    ret = main(["--stats", "--extra", "--demo"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Running in demo mode with synthetic data..." in captured.out
