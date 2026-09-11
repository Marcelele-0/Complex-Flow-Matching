"""Print Tables 2 and 3 of the paper, and the per-seed checks quoted in its text.

Table 2 is the 64x64 comparison (``scripts/paper/table2_unet64.sh``); Table 3
adds 16x16 and 32x32 (``scripts/paper/table3_unet_sizes.sh``) and reuses the
64x64 runs. Every cell is the mean over seeds of the sliced W2 between generated
and reference complex values after k solver steps (Heun, NFE = 2k - 1).

Besides the tables it prints the statements in Section 5 that rest on per-seed
evidence rather than on means:

* cylinder + joint OT against the best Cartesian arm, at every size and k;
* joint OT against independent pairing on the cylinder at k <= 4 (the 11-44%
  reduction, seeds separated in 7 of 9 cells);
* OT on the Cartesian arm at 16x16 (0.421 -> 0.185 at k = 1);
* straightness of every 64x64 arm ("both geometries fit their targets equally
  straight").

Two sources:

* ``outputs/evaluate`` (default): the newest ``metrics.json`` of each run, i.e.
  whatever you trained yourself;
* ``--archive``: ``docs/reproduce/paper_results/unet_eval_metrics.json``, the evaluations
  the paper was written from, shipped so the tables can be checked without
  retraining. ``--export`` rewrites it from ``outputs/evaluate``.

Usage::

    uv run python scripts/paper/paper_tables.py                  # your runs
    uv run python scripts/paper/paper_tables.py --archive        # the paper's runs
    uv run python scripts/paper/paper_tables.py --prefix smoke_  # a short check run
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
from typing import Any

from omegaconf import OmegaConf

ROOT = pathlib.Path(__file__).resolve().parents[2]
EVALUATIONS = ROOT / "outputs" / "evaluate"
TRAININGS = ROOT / "outputs" / "train"
ARCHIVE = ROOT / "docs" / "reproduce" / "paper_results" / "unet_eval_metrics.json"

SIZES = (16, 32, 64)
STEPS = (1, 2, 4, 8, 100)
FEW_STEPS = (1, 2, 4)
GEOMETRIES = ("euclidean", "cylindrical")
COUPLINGS = ("independent", "ot")
SEEDS = (0, 1)
METRIC = "sliced_w2_complex"
LABELS = {"euclidean": "Cartesian", "cylindrical": "Cylindrical"}

Runs = dict[str, dict[str, Any]]


def evaluation_name(side: int, geometry: str, coupling: str, seed: int, prefix: str = "") -> str:
    """Directory name of one evaluation, exactly as the table scripts write it."""
    if side == 64:
        return f"{prefix}un_{geometry}_{coupling}_scnull_s{seed}_eval"
    return f"{prefix}unsz_{geometry}_{coupling}_{side}_s{seed}_eval"


def all_names(prefix: str = "") -> list[str]:
    """Every evaluation the two tables need."""
    return [
        evaluation_name(side, geometry, coupling, seed, prefix)
        for side in SIZES
        for geometry in GEOMETRIES
        for coupling in COUPLINGS
        for seed in SEEDS
    ]


def load_outputs(prefix: str) -> Runs:
    """Newest ``metrics.json`` per evaluation, keyed by the unprefixed name."""
    runs: Runs = {}
    for name in all_names(prefix):
        found = sorted(EVALUATIONS.glob(f"{name}/*/metrics.json"))
        if found:
            runs[name.removeprefix(prefix)] = json.loads(found[-1].read_text())
    return runs


def display(path: pathlib.Path) -> str:
    """``path`` relative to the repository when it lies inside it, else absolute."""
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def load_archive(path: pathlib.Path = ARCHIVE) -> Runs:
    """The evaluations the paper's tables were written from."""
    runs: Runs = json.loads(path.read_text())
    return runs


def per_seed(runs: Runs, side: int, geometry: str, coupling: str, step: int) -> list[float]:
    """The metric of one arm at one step count, one value per available seed."""
    values = []
    for seed in SEEDS:
        run = runs.get(evaluation_name(side, geometry, coupling, seed))
        if run is None:
            continue
        for entry in run["sweep"]:
            if int(entry["num_steps"]) == step:
                values.append(float(entry[METRIC]))
    return values


def cell(values: list[float]) -> str:
    """Mean over seeds, or a dash when the arm has not been run."""
    return f"{statistics.fmean(values):7.3f}" if values else "      -"


def separated_below(low: list[float], high: list[float]) -> bool:
    """True when every seed of ``low`` is below every seed of ``high``."""
    return bool(low) and bool(high) and max(low) < min(high)


def print_rows(runs: Runs, side: int, arms: list[tuple[str, str]]) -> None:
    """One table block: a row per (geometry, coupling) arm."""
    header = "".join(f"{f'k={step}':>8}" for step in STEPS)
    print(f"  {'geometry':<12} {'coupling':<12} {'n':>2}{header}")
    for geometry, coupling in arms:
        seeds = len(per_seed(runs, side, geometry, coupling, STEPS[0]))
        cells = "".join(
            f" {cell(per_seed(runs, side, geometry, coupling, step))}" for step in STEPS
        )
        print(f"  {LABELS[geometry]:<12} {coupling:<12} {seeds:>2}{cells}")


def print_table2(runs: Runs) -> None:
    """Table 2: 64x64, all four arms."""
    print("\nTABLE 2  Spatial field synthesis, 64x64, sliced W2 (mean over seeds)")
    arms = [(geometry, coupling) for geometry in GEOMETRIES for coupling in COUPLINGS]
    print_rows(runs, 64, arms)


def print_table3(runs: Runs) -> None:
    """Table 3: every size, Cartesian OT against the cylinder with and without OT."""
    print("\nTABLE 3  Dimensionality scaling, sliced W2 (mean over seeds)")
    arms = [("euclidean", "ot"), ("cylindrical", "independent"), ("cylindrical", "ot")]
    for side in SIZES:
        print(f" {side}x{side}")
        print_rows(runs, side, arms)


def best_cartesian(runs: Runs, side: int, step: int) -> list[float]:
    """Per-seed values of whichever Cartesian arm has the lower mean at this cell."""
    candidates = [per_seed(runs, side, "euclidean", c, step) for c in COUPLINGS]
    candidates = [values for values in candidates if values]
    return min(candidates, key=statistics.fmean) if candidates else []


def print_checks(runs: Runs) -> None:
    """The per-seed statements of Section 5."""
    print("\nCHECK 1  cylinder + joint OT against the best Cartesian arm, per seed")
    for side in SIZES:
        for step in STEPS:
            cylinder = per_seed(runs, side, "cylindrical", "ot", step)
            cartesian = best_cartesian(runs, side, step)
            if not cylinder or not cartesian:
                continue
            if separated_below(cylinder, cartesian):
                verdict = "cylinder"
            elif separated_below(cartesian, cylinder):
                verdict = "Cartesian"
            else:
                verdict = "overlap"
            print(
                f"  {side}x{side} k={step:<4} cylinder OT {[round(v, 3) for v in cylinder]}"
                f"  best Cartesian {[round(v, 3) for v in cartesian]}  -> {verdict}"
            )

    print("\nCHECK 2  joint OT against independent pairing on the cylinder, k <= 4")
    reductions: list[float] = []
    separated = 0
    compared = 0
    for side in SIZES:
        for step in FEW_STEPS:
            independent = per_seed(runs, side, "cylindrical", "independent", step)
            coupled = per_seed(runs, side, "cylindrical", "ot", step)
            if not independent or not coupled:
                continue
            compared += 1
            reduction = 1.0 - statistics.fmean(coupled) / statistics.fmean(independent)
            reductions.append(reduction)
            apart = separated_below(coupled, independent)
            separated += apart
            print(
                f"  {side}x{side} k={step}  independent {[round(v, 3) for v in independent]}"
                f"  OT {[round(v, 3) for v in coupled]}  reduction {reduction:6.1%}"
                f"  {'separated' if apart else 'overlap'}"
            )
    if reductions:
        print(
            f"  -> reduction {min(reductions):.0%} to {max(reductions):.0%},"
            f" seeds separated in {separated} of {compared} cells"
        )

    print("\nCHECK 3  OT on the Cartesian arm at 16x16, k = 1")
    independent = per_seed(runs, 16, "euclidean", "independent", 1)
    coupled = per_seed(runs, 16, "euclidean", "ot", 1)
    if independent and coupled:
        print(f"  {statistics.fmean(independent):.3f} -> {statistics.fmean(coupled):.3f}")

    print("\nCHECK 4  straightness at 64x64, under each checkpoint's own coupling")
    for geometry in GEOMETRIES:
        for coupling in COUPLINGS:
            values = [
                float(run["straightness"])
                for seed in SEEDS
                if (run := runs.get(evaluation_name(64, geometry, coupling, seed))) is not None
            ]
            if values:
                print(f"  {LABELS[geometry]:<12} {coupling:<12} {statistics.fmean(values):.3f}")


def provenance(name: str, prefix: str) -> dict[str, Any]:
    """Hydra overrides of one evaluation and of the training run it evaluated.

    Args:
        name: Unprefixed evaluation name, e.g. ``un_cylindrical_ot_scnull_s0_eval``.
        prefix: Run-name prefix the evaluation and its training run were written with.

    Returns:
        ``evaluate_run`` / ``evaluate_overrides`` from the newest evaluation directory
        (the one :func:`load_outputs` reads the metrics from) and ``train_run`` /
        ``train_overrides`` from the newest training directory of the same run; a key
        is absent when its directory is not present locally.
    """
    record: dict[str, Any] = {}
    evaluations = sorted(EVALUATIONS.glob(f"{prefix}{name}/*/metrics.json"))
    if evaluations:
        run_dir = evaluations[-1].parent
        record["evaluate_run"] = run_dir.name
        record["evaluate_overrides"] = read_overrides(run_dir / ".hydra" / "overrides.yaml")
    trainings = sorted(TRAININGS.glob(f"{prefix}{name.removesuffix('_eval')}/*/.hydra"))
    if trainings:
        record["train_run"] = trainings[-1].parent.name
        record["train_overrides"] = read_overrides(trainings[-1] / "overrides.yaml")
    return record


def read_overrides(path: pathlib.Path) -> list[str]:
    """The ``key=value`` override strings Hydra saved for one run.

    Args:
        path: A run's ``.hydra/overrides.yaml``.

    Returns:
        The overrides in the order they were given on the command line.
    """
    overrides = OmegaConf.to_container(OmegaConf.load(path))
    assert isinstance(overrides, list), f"{path} is not a list of overrides"
    return [str(item) for item in overrides]


def export(prefix: str, path: pathlib.Path = ARCHIVE, partial: bool = False) -> None:
    """Freeze the current evaluations as the archive the paper cites.

    Each entry is the evaluation's ``metrics.json`` plus a ``provenance`` record
    (see :func:`provenance`), so an archive states how every number was produced.

    Args:
        prefix: Run-name prefix of the evaluations to export (stripped in the archive).
        path: Archive file to write.
        partial: Allow arms of the grid to be missing, for a supplementary archive that
            holds one geometry only (e.g. the Cartesian arm under a second loss).

    Raises:
        SystemExit: If arms are missing and ``partial`` is false, or nothing matched.
    """
    runs = load_outputs(prefix)
    missing = [name for name in all_names() if name not in runs]
    if missing and not partial:
        raise SystemExit(f"Refusing to export an incomplete archive; missing: {missing}")
    if not runs:
        raise SystemExit(f"No evaluations found for prefix {prefix!r}")
    for name, run in runs.items():
        run["provenance"] = provenance(name, prefix)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(runs, indent=1, sort_keys=True) + "\n")
    print(f"Wrote {len(runs)} evaluations to {display(path)}")


def main() -> None:
    """Print both tables and the per-seed checks from one source."""
    global SEEDS
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--archive", action="store_true", help="Read the shipped archive.")
    source.add_argument("--export", action="store_true", help="Write the archive and exit.")
    parser.add_argument(
        "--prefix", default="", help="Run-name prefix used by the table scripts (PREFIX)."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(SEEDS),
        help="Seeds to read (the v1 archive has 0 1, the v2 archives 0 1 2 3 4).",
    )
    parser.add_argument(
        "--archive-file",
        type=pathlib.Path,
        default=ARCHIVE,
        help="Archive to read (--archive) or write (--export); default: the v1 archive.",
    )
    parser.add_argument(
        "--partial",
        action="store_true",
        help="With --export, allow arms to be missing (one-geometry archives).",
    )
    args = parser.parse_args()
    SEEDS = tuple(args.seeds)
    archive_file = args.archive_file.resolve()

    if args.export:
        export(args.prefix, archive_file, args.partial)
        return

    runs = load_archive(archive_file) if args.archive else load_outputs(args.prefix)
    origin = display(archive_file) if args.archive else display(EVALUATIONS)
    print(f"Source: {origin}  ({len(runs)} of {len(all_names())} evaluations present)")
    print_table2(runs)
    print_table3(runs)
    print_checks(runs)


if __name__ == "__main__":
    main()
