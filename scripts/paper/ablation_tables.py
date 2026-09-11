"""Summarise the loss ablation of ``conf/experiment/ablation_loss32.yaml``.

For every arm and solver step count it prints the mean over seeds of four
metrics -- sliced W2 of the complex values, W2 of the amplitude, circular W2 of
the phase, and the amplitude-phase dependence gap -- followed by the per-seed
values, so a reader can see spread instead of trusting a mean over three seeds.

It then prints the comparisons the ablation exists for, each with a per-seed
separation verdict (every seed of one arm below every seed of the other):

* unweighted against amplitude-weighted phase loss, per loss type;
* L2 against L1, per geometry;
* each cylindrical arm against the Cartesian arm with the same loss type.

Usage::

    uv run python scripts/paper/ablation_tables.py
    uv run python scripts/paper/ablation_tables.py --prefix smoke_ --side 16
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
EVALUATIONS = ROOT / "outputs" / "evaluate"

STEPS = (1, 2, 4, 8, 100)
METRICS = (
    ("sliced_w2_complex", "sliced W2"),
    ("w2_amplitude", "W2 amp"),
    ("w2_phase_circular", "W2 phase"),
    ("dependence_gap", "dep gap"),
)
ARMS = (
    "cylindrical_l1_weighted",
    "cylindrical_l1_unweighted",
    "cylindrical_l2_weighted",
    "cylindrical_l2_unweighted",
    "euclidean_l1",
    "euclidean_l2",
)
COMPARISONS = (
    ("cylindrical_l1_unweighted", "cylindrical_l1_weighted"),
    ("cylindrical_l2_unweighted", "cylindrical_l2_weighted"),
    ("cylindrical_l2_weighted", "cylindrical_l1_weighted"),
    ("cylindrical_l2_unweighted", "cylindrical_l1_unweighted"),
    ("euclidean_l2", "euclidean_l1"),
    ("cylindrical_l1_weighted", "euclidean_l1"),
    ("cylindrical_l1_unweighted", "euclidean_l1"),
    ("cylindrical_l2_weighted", "euclidean_l2"),
    ("cylindrical_l2_unweighted", "euclidean_l2"),
)

Row = dict[str, float]
Arms = dict[str, dict[int, dict[int, Row]]]


def load(prefix: str, side: int) -> tuple[Arms, dict[str, list[float]]]:
    """Newest ``metrics.json`` per evaluation: arm -> seed -> steps -> metrics."""
    arms: Arms = {}
    straightness: dict[str, list[float]] = {}
    for arm in ARMS:
        for path in sorted(EVALUATIONS.glob(f"{prefix}abl_{arm}_{side}_s*_eval")):
            seed = int(path.name.removesuffix("_eval").rsplit("_s", 1)[1])
            found = sorted(path.glob("*/metrics.json"))
            if not found:
                continue
            data: dict[str, Any] = json.loads(found[-1].read_text())
            arms.setdefault(arm, {})[seed] = {
                int(row["num_steps"]): {key: float(row[key]) for key, _ in METRICS}
                for row in data["sweep"]
            }
            straightness.setdefault(arm, []).append(float(data["straightness"]))
    return arms, straightness


def values(arms: Arms, arm: str, step: int, metric: str) -> list[float]:
    """Per-seed values of one metric, in seed order."""
    return [arms[arm][seed][step][metric] for seed in sorted(arms.get(arm, {}))]


def print_table(arms: Arms, straightness: dict[str, list[float]]) -> None:
    for step in STEPS:
        print(f"\n## k = {step}  (NFE = {2 * step - 1})")
        header = f"{'arm':<28}{'n':>3}" + "".join(f"{label:>11}" for _, label in METRICS)
        print(header)
        print("-" * len(header))
        for arm in ARMS:
            if arm not in arms:
                continue
            cells = [values(arms, arm, step, key) for key, _ in METRICS]
            means = "".join(f"{statistics.fmean(v):>11.4f}" for v in cells)
            print(f"{arm:<28}{len(cells[0]):>3}{means}")
            seeds = "".join(f"{'/'.join(f'{x:.3f}' for x in v):>11}" for v in cells)
            print(f"{'  per seed':<31}{seeds}")
    print("\n## straightness (mean over seeds)")
    for arm in ARMS:
        if arm in straightness:
            print(f"{arm:<28}{statistics.fmean(straightness[arm]):.4f}")


def verdict(left: list[float], right: list[float]) -> str:
    """Whether every seed of one arm lies strictly below every seed of the other.

    One seed per arm always "separates", so it is reported as untested instead.
    """
    if min(len(left), len(right)) < 2:
        return "n<2, untested"
    if max(left) < min(right):
        return "left better, separated"
    if max(right) < min(left):
        return "right better, separated"
    return "overlap"


def print_comparisons(arms: Arms) -> None:
    print("\n## comparisons (lower is better; change = left / right - 1)")
    for left, right in COMPARISONS:
        if left not in arms or right not in arms:
            continue
        print(f"\n{left}  vs  {right}")
        for step in STEPS:
            parts = []
            for key, label in METRICS[:3]:
                a = values(arms, left, step, key)
                b = values(arms, right, step, key)
                change = statistics.fmean(a) / statistics.fmean(b) - 1.0
                parts.append(f"{label} {change:+.0%} ({verdict(a, b)})")
            print(f"  k={step:<4}" + " | ".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--prefix", default="", help="Run-name prefix, e.g. smoke_.")
    parser.add_argument("--side", type=int, default=32, help="Field side length.")
    args = parser.parse_args()

    arms, straightness = load(args.prefix, args.side)
    if not arms:
        raise SystemExit(f"no evaluations found under {EVALUATIONS} for side {args.side}")
    print_table(arms, straightness)
    print_comparisons(arms)


if __name__ == "__main__":
    main()
