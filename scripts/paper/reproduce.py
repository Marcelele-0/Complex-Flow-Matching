"""Run the paper's experiments from their configs in ``conf/experiment/``.

Each result of the paper is a Hydra experiment config whose ``paper:`` block says
how to reproduce it (the U-Net tables also include their training protocol, so
``+experiment=table2_unet64`` trains one cell by hand). Two kinds of ``paper:`` block:

* ``unet_grid`` -- trains and evaluates every run of a grid
  (field sides x arms x geometries x couplings x seeds). Each arm names a Hydra
  experiment config in ``conf/experiment/``; a run is one ``cfm.train`` and one
  ``cfm.evaluate`` call with that config plus geometry, coupling, field size and
  seed. A run whose evaluation already exists is skipped, so an interrupted sweep
  resumes, and runs shared between specs (Table 3 contains Table 2) are made once.
* ``commands`` -- network-free scripts, run as given.

After the runs, the spec's ``report`` commands print the result.

Usage::

    # everything, in paper order
    uv run python scripts/paper/reproduce.py --all

    # one result
    uv run python scripts/paper/reproduce.py table2_unet64

    # a quick check: one seed, the smallest field, a separate run-name tag
    uv run python scripts/paper/reproduce.py table3_unet_scaling --seeds 0 --sides 16 --tag quick_

    # print what would run
    uv run python scripts/paper/reproduce.py --all --dry-run
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

from omegaconf import OmegaConf

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPECS = ROOT / "conf" / "experiment"
EVALUATIONS = ROOT / "outputs" / "evaluate"

# Every config with a ``paper:`` block, in the order the paper presents them.
PAPER_ORDER = (
    "table1_bridges",
    "table2_unet64",
    "table3_unet_scaling",
    "tableA_loss_protocols",
    "table4_factorized",
    "sec53_ot_cost",
    "ablation_loss32",
)

# The run-name convention of the archives in docs/reproduce/paper_results/.
NAME_64 = "{prefix}un_{geometry}_{coupling}_scnull_s{seed}"
NAME = "{prefix}unsz_{geometry}_{coupling}_{side}_s{seed}"
NFE = "[1,2,4,8,100]"
NUM_FIELDS = 64
KINDS = ("unet_grid", "commands")


@dataclass(frozen=True)
class Run:
    """One training run and its evaluation."""

    name: str
    experiment: str
    side: int
    geometry: str
    coupling: str
    seed: int


def spec_path(name: str) -> pathlib.Path:
    """The config file of one paper experiment.

    Args:
        name: A config name such as ``table2_unet64``, or a path to a YAML file.

    Returns:
        The path of the config.
    """
    if name.endswith(".yaml"):
        return pathlib.Path(name)
    return SPECS / f"{name}.yaml"


def load_spec(name: str) -> dict[str, Any]:
    """Read the ``paper:`` block of one experiment config.

    Args:
        name: A config name in ``conf/experiment/`` or a path to one.

    Returns:
        The block as a plain dictionary.

    Raises:
        ValueError: If the config has no ``paper:`` mapping or names an unknown kind.
    """
    path = spec_path(name)
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    spec = config.get("paper") if isinstance(config, dict) else None
    if not isinstance(spec, dict):
        raise ValueError(f"{path}: no paper: block, so it is not a paper experiment")
    if spec.get("kind") not in KINDS:
        raise ValueError(f"{path}: kind must be one of {KINDS}, got {spec.get('kind')!r}")
    return {str(key): value for key, value in spec.items()}


def grid(
    spec: dict[str, Any],
    tag: str = "",
    seeds: list[int] | None = None,
    sides: list[int] | None = None,
) -> list[Run]:
    """Every run of a ``unet_grid`` spec, in execution order.

    Args:
        spec: A spec of kind ``unet_grid``.
        tag: Prepended to every arm's run-name prefix, to keep a quick check apart
            from the archived runs.
        seeds: Replaces the spec's seeds when given.
        sides: Replaces the spec's field sides when given.

    Returns:
        The runs, field side outermost and seed innermost.
    """
    runs = []
    for side in sides if sides is not None else spec["sides"]:
        for arm in spec["arms"]:
            template = arm.get("name", NAME_64 if side == 64 else NAME)
            for geometry in arm["geometries"]:
                for coupling in arm["couplings"]:
                    for seed in seeds if seeds is not None else spec["seeds"]:
                        name = template.format(
                            prefix=tag + arm["prefix"],
                            geometry=geometry,
                            coupling=coupling,
                            side=side,
                            seed=seed,
                        )
                        runs.append(Run(name, arm["experiment"], side, geometry, coupling, seed))
    return runs


def run_arguments(run: Run) -> list[str]:
    """Hydra overrides shared by the training and the evaluation of one run."""
    return [
        f"+experiment={run.experiment}",
        f"manifold={run.geometry}",
        f"training.coupling={run.coupling}",
        f"dataset.crop_size=[{run.side},{run.side}]",
    ]


def train_command(run: Run, epochs: int | None) -> list[str]:
    """The ``cfm.train`` call of one run; ``epochs`` overrides the config's 40."""
    command = [sys.executable, "-m", "cfm.train", *run_arguments(run)]
    command += [f"training.seed={run.seed}", f"logging.experiment_name={run.name}"]
    if epochs is not None:
        command.append(f"training.epochs={epochs}")
    return command


def evaluate_command(run: Run) -> list[str]:
    """The ``cfm.evaluate`` call of one run, at the paper's step counts."""
    return [
        sys.executable,
        "-m",
        "cfm.evaluate",
        *run_arguments(run),
        f"evaluate.run_name={run.name}",
        f"evaluate.num_fields={NUM_FIELDS}",
        f"evaluate.seed={run.seed}",
        f"evaluate.nfe={NFE}",
        f"logging.experiment_name={run.name}_eval",
    ]


def evaluated(run: Run) -> bool:
    """True when an evaluation of this run is already on disk."""
    return any(EVALUATIONS.glob(f"{run.name}_eval/*/metrics.json"))


def script_command(entry: list[Any], tag: str, seeds: list[int], sides: list[int]) -> list[str]:
    """Expand one ``commands`` / ``report`` entry into an executable command.

    Args:
        entry: ``[script, arg, ...]`` with the script relative to the repository.
            ``{tag}`` inside an argument becomes the run-name tag; an argument equal
            to ``{seeds}`` or ``{sides}`` becomes one argument per seed or side.
        tag: The run-name tag of this invocation.
        seeds: The seeds this invocation ran.
        sides: The field sides this invocation ran.

    Returns:
        The command, run with the current Python interpreter.
    """
    script, *arguments = (str(item) for item in entry)
    expanded: list[str] = []
    for argument in arguments:
        if argument == "{seeds}":
            expanded += [str(seed) for seed in seeds]
        elif argument == "{sides}":
            expanded += [str(side) for side in sides]
        else:
            expanded.append(argument.replace("{tag}", tag))
    return [sys.executable, str(ROOT / script), *expanded]


def execute(command: list[str], dry_run: bool) -> None:
    """Print a command and, unless ``dry_run``, run it from the repository root.

    Raises:
        SystemExit: If the command fails; the sweep stops at the first failure.
    """
    shown = " ".join(command[1:]) if command[0] == sys.executable else " ".join(command)
    print(f"  $ python {shown}" if command[0] == sys.executable else f"  $ {shown}", flush=True)
    if dry_run:
        return
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(f"failed with exit code {result.returncode}: {shown}")


def run_spec(name: str, args: argparse.Namespace) -> None:
    """Run one paper experiment: its grid or commands, then its report.

    Args:
        name: The config name in ``conf/experiment/``, or a path to it.
        args: The parsed command line (seeds, sides, epochs, tag, dry_run, report).
    """
    spec = load_spec(name)
    print(f"\n=== {spec['name']}  [{spec['section']}]  ({spec.get('runtime', '?')})", flush=True)
    seeds = args.seeds if args.seeds is not None else spec.get("seeds", [])
    sides = args.sides if args.sides is not None else spec.get("sides", [])
    if spec["kind"] == "unet_grid":
        runs = grid(spec, args.tag, args.seeds, args.sides)
        for index, run in enumerate(runs, start=1):
            status = "exists, skipped" if evaluated(run) else "run"
            print(f"[{index}/{len(runs)}] {run.name} ({run.experiment}): {status}", flush=True)
            if status == "run":
                execute(train_command(run, args.epochs), args.dry_run)
                execute(evaluate_command(run), args.dry_run)
    else:
        for directory in spec.get("make_dirs", []):
            if not args.dry_run:
                (ROOT / directory).mkdir(parents=True, exist_ok=True)
        for entry in spec["commands"]:
            execute(script_command(entry, args.tag, seeds, sides), args.dry_run)
        if "expected" in spec:
            print("expected:\n" + spec["expected"].rstrip(), flush=True)
    if args.report:
        for entry in spec.get("report", []):
            execute(script_command(entry, args.tag, seeds, sides), args.dry_run)


def main() -> None:
    """Parse the command line and run the selected specs in order."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "specs", nargs="*", help="Experiment config names, e.g. table2_unet64, or paths."
    )
    parser.add_argument("--all", action="store_true", help="Run every paper experiment in order.")
    parser.add_argument("--seeds", type=int, nargs="+", help="Replace the specs' seeds.")
    parser.add_argument("--sides", type=int, nargs="+", help="Replace the specs' field sides.")
    parser.add_argument("--epochs", type=int, help="Override the 40 training epochs.")
    parser.add_argument("--tag", default="", help="Prefix for run names, e.g. quick_.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    parser.add_argument(
        "--no-report", dest="report", action="store_false", help="Skip the report step."
    )
    args = parser.parse_args()

    names = list(PAPER_ORDER) if args.all else list(args.specs)
    if not names:
        parser.error(f"give experiment names ({', '.join(PAPER_ORDER)}) or --all")
    for name in names:
        run_spec(name, args)


if __name__ == "__main__":
    main()
