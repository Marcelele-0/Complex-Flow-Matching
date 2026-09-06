"""Mission control execution suite for training and evaluation matrix runs."""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_LAUNCHER_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "launch_slurm.sh"

# One source of truth for the sweep: joined with commas it is a Hydra multirun
# override, expanded with itertools.product it is one Slurm job per grid point.
MATRIX_AXES: dict[str, list[str]] = {
    "dataset": ["skm_tea", "fastmri_local"],
    "manifold": ["cylindrical", "euclidean", "complex_diffusion"],
    "model": ["c_unet"],
}
SEED_AXIS: tuple[str, list[str]] = ("training.seed", ["42", "123", "999"])
SMOKE_OVERRIDES: list[str] = [
    "training.epochs=3",
    "training.batch_size=2",
    "evaluate.max_samples=2",
    "dataset.use_cache=false",
]


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
        "-e",
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

    if args.matrix:
        hydra_args.extend(f"{key}={','.join(values)}" for key, values in MATRIX_AXES.items())
        is_multirun = True

    if args.seeds:
        key, values = SEED_AXIS
        hydra_args.append(f"{key}={','.join(values)}")
        is_multirun = True

    if args.smoke:
        hydra_args.extend(SMOKE_OVERRIDES)

    if is_multirun:
        hydra_args.insert(0, "-m")

    hydra_args.extend(args.extra)
    return hydra_args, is_multirun


def slurm_jobs(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    """Expand the sweep into one independent Slurm job per grid point.

    Hydra multirun cannot be forwarded into a DDP job. ``train_ddp.sbatch`` prepends
    its own overrides before the caller's, and Hydra's parser rejects ``-m`` once any
    override precedes it. A single grid point per job is also the only safe layout:
    ``logging.experiment_name`` keys ``outputs/state/<name>/last.pt`` and the batch
    script forces ``auto_resume=true``, so points sharing a name would resume from
    each other's checkpoint.

    Args:
        args: Parsed command-line arguments.

    Returns:
        List of (experiment_name, hydra_overrides) pairs, one per job to submit.
    """
    axes: list[tuple[str, list[str]]] = []
    if args.matrix:
        axes.extend(MATRIX_AXES.items())
    if args.seeds:
        axes.append(SEED_AXIS)

    shared = [*(SMOKE_OVERRIDES if args.smoke else []), *args.extra]
    if not axes:
        return [(args.experiment, shared)]

    jobs: list[tuple[str, list[str]]] = []
    for point in itertools.product(*(values for _, values in axes)):
        overrides = [f"{key}={value}" for (key, _), value in zip(axes, point, strict=True)]
        jobs.append((f"{args.experiment}-{'-'.join(point)}", [*overrides, *shared]))
    return jobs


def build_command(args: argparse.Namespace) -> list[str]:
    """Construct the local ``train.py`` invocation.

    Args:
        args: Parsed command-line arguments.

    Returns:
        List of command tokens ready for subprocess.run.
    """
    hydra_args, _ = build_hydra_args(args)
    return [sys.executable, "src/cfm/train.py", *hydra_args]


def build_launcher_command(
    args: argparse.Namespace,
    experiment: str,
    overrides: list[str],
    launcher_script: Path | str | None = None,
) -> list[str]:
    """Construct the ``launch_slurm.sh`` invocation for one grid point.

    Args:
        args: Parsed command-line arguments.
        experiment: ``logging.experiment_name`` for this job, from :func:`slurm_jobs`.
        overrides: Hydra overrides for this job, forwarded after ``--``.
        launcher_script: Optional override for the launcher script path.

    Returns:
        List of command tokens ready for subprocess.run.
    """
    script_path = launcher_script if launcher_script is not None else DEFAULT_LAUNCHER_SCRIPT
    cmd = [str(script_path)]
    if args.submit:
        cmd.append("--submit")
    cmd.extend(["-g", str(args.gpus), "-N", str(args.nodes), "-e", experiment, "--", *overrides])
    return cmd


def dispatch_slurm(args: argparse.Namespace) -> int:
    """Submit one Slurm job per grid point, stopping at the first launcher failure.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Integer exit code (0 for success, the failing launcher's code otherwise).
    """
    if not DEFAULT_LAUNCHER_SCRIPT.exists():
        print(f"Slurm launcher not found at {DEFAULT_LAUNCHER_SCRIPT}.")
        print("--slurm needs a source checkout of the repository.")
        return 1

    if args.eval:
        print(
            "\n[INFO] --eval does not apply to --slurm. Jobs run asynchronously in the "
            "queue, and each writes to outputs/train/<experiment>/ rather than "
            "outputs/multirun/, which is the only tree `cfmri-suite --eval` reads. "
            "Score a finished job with:\n"
            "  uv run src/cfm/evaluate.py evaluate.run_name=<experiment>\n"
        )

    jobs = slurm_jobs(args)
    mode = "submitting" if args.submit else "dry-running (add --submit to consume allocation)"
    print(f"CFM Suite: {mode} {len(jobs)} Slurm job(s) via {DEFAULT_LAUNCHER_SCRIPT}")

    for experiment, overrides in jobs:
        cmd = build_launcher_command(args, experiment, overrides)
        print(f"\n--- {experiment} ---\n{' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\nLauncher failed for '{experiment}' (exit {result.returncode}). Stopping.")
            return result.returncode

    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for CFM Suite CLI.

    Args:
        argv: Optional list of command-line argument strings. If None, sys.argv[1:] is used.

    Returns:
        Integer exit code (0 for success, non-zero for failure).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # --extra swallows everything that follows it, so a suite flag written after it
    # would silently become a Hydra override and change which mode runs.
    misplaced = [token for token in args.extra if token.startswith("-")]
    if misplaced:
        parser.error(
            f"--extra takes Hydra overrides, not flags: {' '.join(misplaced)}. "
            "Write suite flags before --extra."
        )
    if args.submit and not args.slurm:
        parser.error("--submit only applies to --slurm dispatch.")

    # Standalone evaluation of existing multirun checkpoints without launching new training
    if args.eval and not (args.matrix or args.seeds or args.smoke or args.extra or args.slurm):
        multirun_dirs = glob.glob("outputs/multirun/*/")
        if multirun_dirs:
            latest_multirun = max(multirun_dirs, key=os.path.getmtime)
            run_evaluation(latest_multirun)
            return 0
        print("No multirun outputs found under outputs/multirun/ to evaluate.")
        return 1

    if args.slurm:
        return dispatch_slurm(args)

    cmd = build_command(args)
    _, is_multirun = build_hydra_args(args)

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
