"""Rebuild the paper's result tables from whatever is already in ``outputs/``.

Every number in Tables 2, 3 and 7 comes from an evaluation that has already been
run and written to ``outputs/evaluate/<run>/<stamp>/metrics.json``. This script
reads those files and prints the tables, so a table can be regenerated at any
time -- after a crash, on a different machine, or when a caption needs checking --
without holding a GPU or retraining anything.

Run naming is the index, and it is also what keeps incomparable models apart: the
``t6`` prefix is a two-epoch smoke probe whose learning rate had already annealed
to its floor, and ``t6e40`` the forty-epoch run that supersedes it. They are
different models of the same data, so they are never merged.

``l2u_un_<manifold>_<coupling>_scnull_s<seed>`` is the
native-resolution grid and ``l2u_unsz_<manifold>_<coupling>_<side>_s<seed>`` the
scaled one; ``_dense`` marks a re-evaluation on the dense step grid. Nothing else
records which arm a checkpoint belongs to, so a rename silently orphans a run.

Usage::

    uv run python scripts/paper/collect_tables.py --outputs outputs
    uv run python scripts/paper/collect_tables.py --outputs outputs --json tables.json
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import statistics
from typing import Any

ARM_ORDER = ("cylindrical_ot", "cylindrical_independent", "euclidean_ot", "euclidean_independent")
NAME = re.compile(
    r"^(?P<family>l2u|l1u|l1w|t6e40|t6e10|t6|t5)_(?:un(?:sz)?_)?"
    r"(?P<manifold>cylindrical|euclidean)_(?P<coupling>ot|independent)"
    r"(?:_(?P<side>scnull|\d+))?_s(?P<seed>\d+)(?P<dense>_dense)?$"
)


def parse_run(name: str) -> dict[str, str] | None:
    """Split a run directory name into the arm it belongs to, or ``None``."""
    match = NAME.match(name)
    if match is None:
        return None
    parts = match.groupdict()
    parts["side"] = "native" if parts["side"] in (None, "scnull") else parts["side"]
    parts["arm"] = f"{parts['manifold']}_{parts['coupling']}"
    parts["dense"] = bool(parts["dense"])
    return parts


def load(outputs: pathlib.Path) -> list[dict[str, Any]]:
    """Every evaluation under ``outputs/evaluate`` that names a recognisable arm.

    Where a run has been evaluated more than once, the newest directory wins, so
    a re-run supersedes rather than duplicates.
    """
    records: dict[tuple[str, ...], dict[str, Any]] = {}
    for metrics in sorted((outputs / "evaluate").glob("*/*/metrics.json")):
        run = metrics.parts[-3].removesuffix("_eval")
        parsed = parse_run(run)
        if parsed is None:
            continue
        payload = json.loads(metrics.read_text())
        key = (
            parsed["family"],
            parsed["arm"],
            parsed["side"],
            parsed["seed"],
            str(parsed["dense"]),
        )
        stamp = metrics.parts[-2]
        if key in records and records[key]["stamp"] > stamp:
            continue
        records[key] = {
            **parsed,
            "stamp": stamp,
            "sweep": payload.get("sweep", []),
            "dataset": payload.get("dataset", "?"),
            "path": str(metrics),
        }
    return list(records.values())


def curves(
    records: list[dict[str, Any]], family: str, side: str
) -> dict[str, dict[int, dict[int, float]]]:
    """One error curve per arm and seed, merging every evaluation of that checkpoint.

    A run is usually evaluated twice: once on the sparse step grid and once on the dense
    one. Both read the same weights, so the curves are two samplings of one function and
    their union is the right object. Keying them separately instead would let the sparse
    curve stand in for the dense one and report the first step count reaching a target as
    100 when the dense grid shows 16 -- which is exactly the error this merge prevents.
    """
    merged: dict[str, dict[int, dict[int, float]]] = collections.defaultdict(
        lambda: collections.defaultdict(dict)
    )
    for record in records:
        if record["family"] != family or record["side"] != side:
            continue
        for row in record["sweep"]:
            merged[record["arm"]][int(record["seed"])][int(row["num_steps"])] = float(
                row["sliced_w2_complex"]
            )
    return merged


def error_table(records: list[dict[str, Any]], family: str, side: str) -> None:
    """Mean sliced W2 per arm and step count, with the seed count in brackets."""
    per_seed = curves(records, family, side)
    cells: dict[str, dict[int, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for arm, seeds in per_seed.items():
        for curve in seeds.values():
            for k, value in curve.items():
                cells[arm][k].append(value)
    if not cells:
        print(f"  (nic dla family={family} side={side})")
        return
    steps = sorted({k for arm in cells.values() for k in arm})
    shown = [k for k in steps if k in (1, 2, 4, 8, 16, 32, 64, 100)] or steps
    print(f"{'arm':<26}" + "".join(f"{'k=' + str(k):>12}" for k in shown))
    print("-" * (26 + 12 * len(shown)))
    for arm in [a for a in ARM_ORDER if a in cells] + [a for a in cells if a not in ARM_ORDER]:
        row = []
        for k in shown:
            values = cells[arm].get(k, [])
            row.append(f"{statistics.mean(values):.4f}" if values else "-")
        count = len(per_seed[arm])
        print(f"{arm:<26}" + "".join(f"{c:>12}" for c in row) + f"   n={count}")


def steps_to_target(records: list[dict[str, Any]], family: str, side: str) -> None:
    """Steps each arm needs to reach the worse arm's own k=100 error.

    Framing the target against the *worse* ceiling is what stops the comparison
    being an artefact of the two geometries converging to different asymptotes.
    """
    per_arm = curves(records, family, side)
    if not per_arm:
        return
    ceilings = {
        arm: statistics.mean([c[100] for c in seeds.values() if 100 in c])
        for arm, seeds in per_arm.items()
        if any(100 in c for c in seeds.values())
    }
    if not ceilings:
        print("  (brak k=100, nie ma jak wyznaczyc E*)")
        return
    target = max(ceilings.values())
    worst = max(ceilings.items(), key=lambda item: item[1])[0]
    print(f"\n  E* = {target:.4f}  (sufit gorszego ramienia: {worst})")
    for arm in [a for a in ARM_ORDER if a in per_arm]:
        reached = []
        for curve in per_arm[arm].values():
            hit = [k for k in sorted(curve) if curve[k] <= target]
            reached.append(hit[0] if hit else None)
        shown = ", ".join("-" if r is None else str(r) for r in reached)
        got = [r for r in reached if r is not None]
        median = f"{statistics.median(got):.0f}" if got else "-"
        print(f"  {arm:<26} k po seedach: [{shown}]   mediana {median}")


def main() -> None:
    """Collect evaluation runs into the archived JSON the tables read."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=pathlib.Path, default=pathlib.Path("outputs"))
    parser.add_argument("--json", type=pathlib.Path, default=None, help="Also dump the records.")
    args = parser.parse_args()

    records = load(args.outputs)
    print(f"ewaluacji rozpoznanych: {len(records)}")
    families = collections.Counter((r["family"], r["side"], r["dense"]) for r in records)
    for key in sorted(families):
        family, side, dense = key
        print(f"  {family:<5} side={side:<7} dense={str(dense):<5} -> {families[key]}")

    for family, side, title in (
        ("l2u", "native", "TABELA 2 -- syntetyk, rozdzielczosc natywna"),
        ("l2u", "16", "TABELA 3 -- syntetyk 16x16"),
        ("l2u", "32", "TABELA 3 -- syntetyk 32x32"),
        ("t5", "native", "BLOK MRI -- fastMRI knee CORPD 320x320"),
        ("t6e40", "native", "BLOK MOWY -- LibriSpeech STFT, 40 epok"),
        ("t6e10", "native", "BLOK MOWY -- LibriSpeech STFT, 10 epok"),
        ("t6", "native", "BLOK MOWY -- LibriSpeech STFT, sonda 2-epokowa"),
    ):
        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
        error_table(records, family, side)
        steps_to_target(records, family, side)

    if args.json is not None:
        args.json.write_text(json.dumps(records, indent=2))
        print(f"\nzapisano {args.json}")


if __name__ == "__main__":
    main()
