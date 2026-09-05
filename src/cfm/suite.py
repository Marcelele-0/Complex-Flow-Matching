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


def main() -> None:
    """Entry point for CFM Suite CLI."""
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
        help="Submit jobs to SLURM cluster via submitit launcher",
    )
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        help="Additional flags forwarded directly to Hydra",
        default=[],
    )
    args = parser.parse_args()

    cmd = [sys.executable, "src/cfm/train.py"]
    is_multirun = False

    if args.matrix:
        cmd.extend(
            [
                "dataset=skm_tea,fastmri_local",
                "manifold=cylindrical,euclidean,complex_diffusion",
                "model=c_unet",
            ]
        )
        is_multirun = True

    if args.seeds:
        cmd.append("training.seed=42,123,999")
        is_multirun = True

    if args.smoke:
        cmd.extend(
            [
                "training.epochs=3",
                "training.batch_size=2",
                "evaluate.max_samples=2",
                "dataset.use_cache=false",
            ]
        )

    if args.slurm:
        cmd.append("+hydra/launcher=submitit_slurm")
        is_multirun = True

    if is_multirun:
        cmd.insert(2, "-m")

    cmd.extend(args.extra)

    print(f"Launching CFM Suite: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode == 0 and args.eval and is_multirun:
        multirun_dirs = glob.glob("outputs/multirun/*/")
        if multirun_dirs:
            latest_multirun = max(multirun_dirs, key=os.path.getmtime)
            run_evaluation(latest_multirun)

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
