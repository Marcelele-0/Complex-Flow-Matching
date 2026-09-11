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
* ``--archive``: ``docs/paper_results/unet_eval_metrics.json``, the evaluations
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

ROOT = pathlib.Path(__file__).resolve().parents[2]
EVALUATIONS = ROOT / "outputs" / "evaluate"
ARCHIVE = ROOT / "docs" / "paper_results" / "unet_eval_metrics.json"

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


def load_archive() -> Runs:
    """The evaluations the paper's tables were written from."""
    runs: Runs = json.loads(ARCHIVE.read_text())
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


def export(prefix: str) -> None:
    """Freeze the current evaluations as the archive the paper cites."""
    runs = load_outputs(prefix)
    missing = [name for name in all_names() if name not in runs]
    if missing:
        raise SystemExit(f"Refusing to export an incomplete archive; missing: {missing}")
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVE.write_text(json.dumps(runs, indent=1, sort_keys=True) + "\n")
    print(f"Wrote {len(runs)} evaluations to {ARCHIVE.relative_to(ROOT)}")


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
        help="Seeds to read (the archive has 0 1; the WCSS l2u_ runs have 0 1 2).",
    )
    args = parser.parse_args()
    SEEDS = tuple(args.seeds)

    if args.export:
        export(args.prefix)
        return

    runs = load_archive() if args.archive else load_outputs(args.prefix)
    origin = ARCHIVE.relative_to(ROOT) if args.archive else EVALUATIONS.relative_to(ROOT)
    print(f"Source: {origin}  ({len(runs)} of {len(all_names())} evaluations present)")
    print_table2(runs)
    print_table3(runs)
    print_checks(runs)


if __name__ == "__main__":
    main()
