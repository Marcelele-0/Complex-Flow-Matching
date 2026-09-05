"""Mission control execution suite for training and evaluation matrix runs."""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_LAUNCHER_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "launch_slurm.sh"


def run_evaluation(multirun_dir: str | Path) -> None:
    """Automatically run evaluation for all checkpoints and VarNet in a multirun dir.

    Args:
        multirun_dir: Path to the multirun output directory containing job subdirectories.
    """
    multirun_path = Path(multirun_dir)
    print(f"\nRunning automatic evaluation in {multirun_path}...")
    summary: dict[str, dict] = {}
    datasets_found: set[str] = set()

    job_dirs = sorted(glob.glob(str(multirun_path / "[0-9]*")))
    for job_dir in job_dirs:
        checkpoints = glob.glob(os.path.join(job_dir, "checkpoints", "*.pt"))
        if not checkpoints:
            continue

        latest_ckpt = max(checkpoints, key=os.path.getmtime)
        overrides_path = os.path.join(job_dir, ".hydra", "overrides.yaml")
        if not os.path.exists(overrides_path):
            continue

        with open(overrides_path, encoding="utf-8") as f:
            overrides = f.read().splitlines()

        params: dict[str, str] = {}
        for override in overrides:
            if "=" in override:
                k, v = override.lstrip("- ").split("=", 1)
                params[k] = v

        dataset = params.get("dataset", "unknown")
        manifold = params.get("manifold", "unknown")
        model = params.get("model", "unknown")
        epochs = params.get("training.epochs", "unknown")
        seed = params.get("training.seed", "0")
        datasets_found.add(dataset)

        if dataset not in summary:
            summary[dataset] = {"epochs_trained": epochs, "metrics": {}}

        key = f"{manifold}_s{seed}" if "training.seed" in params else manifold
        print(f"  [EVAL] Evaluating {key} on dataset {dataset}...")

        eval_cmd = [
            sys.executable,
            "src/cfm/evaluate.py",
            f"dataset={dataset}",
            f"manifold={manifold}",
            f"model={model}",
            "evaluate.max_samples=2",
            "evaluate.t_start=0.5",
            f"evaluate.run_name={os.path.abspath(job_dir)}",
            f"evaluate.checkpoint_path={os.path.abspath(latest_ckpt)}",
        ]

        subprocess.run(eval_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        eval_dirs = glob.glob("outputs/evaluate/*/*/")
        if eval_dirs:
            latest_eval = max(eval_dirs, key=os.path.getmtime)
            metrics_file = os.path.join(latest_eval, "metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, encoding="utf-8") as f:
                    metrics = json.load(f)
                summary[dataset]["metrics"][key] = metrics

    for dataset in sorted(datasets_found):
        print(f"  [EVAL] Evaluating varnet baseline on dataset {dataset}...")
        eval_cmd = [
            sys.executable,
            "src/cfm/evaluate.py",
            f"dataset={dataset}",
            "model=varnet",
            "evaluate.max_samples=2",
            "evaluate.run_name=SKIP",
        ]
        subprocess.run(eval_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        eval_dirs = glob.glob("outputs/evaluate/*/*/")
        if eval_dirs:
            latest_eval = max(eval_dirs, key=os.path.getmtime)
            metrics_file = os.path.join(latest_eval, "metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, encoding="utf-8") as f:
                    metrics = json.load(f)
                summary[dataset]["metrics"]["varnet"] = metrics

    summary_path = multirun_path / "eval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nEvaluation complete. Results saved to: {summary_path}")
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser for CFM mission control suite.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(description="torch-cfmri Mission Control Suite")
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Run full matrix across all models, manifolds, and datasets",
    )
    parser.add_argument(
        "--seeds",
        action="store_true",
        help="Add multi-seed variance sweep (seeds: 42, 123, 999)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run in smoke test mode (3 epochs, batch size 2)",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Automatically evaluate checkpoints after training completion",
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="Route execution through scripts/launch_slurm.sh",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually submit job to Slurm queue (default without --submit is safe dry-run)",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=4,
        help="GPUs per node forwarded to launch_slurm.sh -g (default: 4)",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Nodes count forwarded to launch_slurm.sh -N (default: 1)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="suite-matrix",
        help="Experiment name forwarded to launch_slurm.sh -e (default: 'suite-matrix')",
    )
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        help="Additional flags forwarded directly to Hydra",
        default=[],
    )
    return parser


def build_hydra_args(args: argparse.Namespace) -> tuple[list[str], bool]:
    """Construct Hydra arguments and multirun flag from parsed CLI options.

    Args:
        args: Parsed command-line arguments.

    Returns:
        tuple containing:
            - list of Hydra argument strings.
            - boolean indicating if multirun (-m) is enabled.
    """
    hydra_args: list[str] = []
    is_multirun = False

    if getattr(args, "matrix", False):
        hydra_args.extend(
            [
                "dataset=skm_tea,fastmri_local",
                "manifold=cylindrical,euclidean,complex_diffusion",
                "model=c_unet",
            ]
        )
        is_multirun = True

    if getattr(args, "seeds", False):
        hydra_args.append("training.seed=42,123,999")
        is_multirun = True

    if getattr(args, "smoke", False):
        hydra_args.extend(
            [
                "training.epochs=3",
                "training.batch_size=2",
                "evaluate.max_samples=2",
                "dataset.use_cache=false",
            ]
        )

    if is_multirun:
        hydra_args.insert(0, "-m")

    hydra_args.extend(getattr(args, "extra", []))
    return hydra_args, is_multirun


def build_command(
    args: argparse.Namespace,
    launcher_script: Path | str | None = None,
) -> list[str]:
    """Construct execution command for either local execution or Slurm launcher.

    Args:
        args: Parsed command-line arguments.
        launcher_script: Optional override for Slurm launcher script path.

    Returns:
        List of command tokens ready for subprocess.run.
    """
    hydra_args, _ = build_hydra_args(args)

    if getattr(args, "slurm", False):
        script_path = launcher_script if launcher_script is not None else DEFAULT_LAUNCHER_SCRIPT
        launcher_cmd = [str(script_path)]
        if getattr(args, "submit", False):
            launcher_cmd.append("--submit")

        gpus = getattr(args, "gpus", 4)
        nodes = getattr(args, "nodes", 1)
        experiment = getattr(args, "experiment", "suite-matrix")

        launcher_cmd.extend(["-g", str(gpus), "-N", str(nodes), "-e", str(experiment)])
        launcher_cmd.append("--")
        launcher_cmd.extend(hydra_args)
        return launcher_cmd

    return [sys.executable, "src/cfm/train.py", *hydra_args]


def main(argv: list[str] | None = None) -> int:
    """Entry point for CFM Suite CLI.

    Args:
        argv: Optional list of command-line argument strings. If None, sys.argv[1:] is used.

    Returns:
        Integer exit code (0 for success, non-zero for failure).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Standalone evaluation of existing multirun checkpoints without launching new training
    if args.eval and not (args.matrix or args.seeds or args.smoke or args.extra or args.slurm):
        multirun_dirs = glob.glob("outputs/multirun/*/")
        if multirun_dirs:
            latest_multirun = max(multirun_dirs, key=os.path.getmtime)
            run_evaluation(latest_multirun)
            return 0
        print("No multirun outputs found under outputs/multirun/ to evaluate.")
        return 1

    cmd = build_command(args)
    _, is_multirun = build_hydra_args(args)

    if args.slurm:
        if args.eval:
            print(
                "\n[INFO] --eval was requested with --slurm. Since Slurm jobs run asynchronously "
                "in the cluster queue, automatic inline evaluation cannot run immediately. "
                "Once your Slurm job completes, you can run evaluation via: cfmri-suite --eval\n"
            )
        print(f"Launching Slurm CFM Suite: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        return result.returncode

    print(f"Launching CFM Suite: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode == 0 and args.eval and is_multirun:
        multirun_dirs = glob.glob("outputs/multirun/*/")
        if multirun_dirs:
            latest_multirun = max(multirun_dirs, key=os.path.getmtime)
            run_evaluation(latest_multirun)

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
