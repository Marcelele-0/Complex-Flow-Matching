"""Tables for the U-Net smooth-prior sweep (``scripts/sweeps/unet_prior.sh``).

Reads whatever evaluations exist, so it can be run while the sweep is still going.
Two things matter beyond the generation metrics. First, whether the white-prior
arms reproduce the data's spatial correlation at all: a pointwise model could
not (generated lag-1 0.000 against 0.956), so a U-Net doing so is the evidence
that it learns structure rather than inheriting it from a smooth prior. Second,
straightness under each checkpoint's own coupling, which says whether spatial
context finally makes the cylinder's angular target learnable.

Usage::

    uv run python scripts/sweeps/analyse_unet.py
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

EVALUATIONS = pathlib.Path(__file__).resolve().parents[2] / "outputs" / "evaluate"
STEPS = (1, 2, 4, 8, 100)
PRIORS = ("4", "null")
GEOMETRIES = ("euclidean", "cylindrical")
COUPLINGS = ("independent", "ot")
SEEDS = (0, 1)


def latest(name: str) -> dict[str, Any] | None:
    """The most recent metrics.json for one evaluation, if it exists."""
    found = sorted(EVALUATIONS.glob(f"{name}/*/metrics.json"))
    return json.loads(found[-1].read_text()) if found else None


def by_step(run: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Index a run's sweep by number of solver steps."""
    return {int(entry["num_steps"]): entry for entry in run["sweep"]}


def collect() -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """All completed runs, keyed by (prior, geometry, coupling)."""
    runs: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for prior in PRIORS:
        for geometry in GEOMETRIES:
            for coupling in COUPLINGS:
                for seed in SEEDS:
                    run = latest(f"un_{geometry}_{coupling}_sc{prior}_s{seed}_eval")
                    if run is not None:
                        runs.setdefault((prior, geometry, coupling), []).append(run)
    return runs


def mean(values: list[float]) -> float:
    """Arithmetic mean of a non-empty list."""
    return sum(values) / len(values)


def metric_table(runs: dict[tuple[str, str, str], list[dict[str, Any]]], key: str) -> None:
    """Mean over seeds of one metric at every step count."""
    print(f"\n{key}  (mean over available seeds; k = solver steps, NFE = 2k - 1)")
    header = f"{'prior':<7}{'geometry':<13}{'coupling':<13}{'n':>3}"
    print(header + "".join(f"{f'k={k}':>9}" for k in STEPS))
    for (prior, geometry, coupling), group in runs.items():
        cells = "".join(f"{mean([by_step(r)[k][key] for r in group]):>9.3f}" for k in STEPS)
        label = "white" if prior == "null" else prior
        print(f"{label:<7}{geometry:<13}{coupling:<13}{len(group):>3}{cells}")


def main() -> None:
    """Print every table the sweep is meant to produce."""
    runs = collect()
    total = sum(len(group) for group in runs.values())
    print(f"U-Net sweep: {total} of 16 evaluations available")
    if not runs:
        return

    for key in ("sliced_w2_complex", "w2_phase_circular", "w2_amplitude", "spatial_lag1_gap"):
        metric_table(runs, key)

    print("\nPER SEED, sliced W2 and phase W2 at k = 1, 2, 4")
    for (prior, geometry, coupling), group in runs.items():
        label = "white" if prior == "null" else prior
        cells = "  ".join(
            f"k{k}: sw {[round(by_step(r)[k]['sliced_w2_complex'], 3) for r in group]}"
            f" ph {[round(by_step(r)[k]['w2_phase_circular'], 3) for r in group]}"
            for k in (1, 2, 4)
        )
        print(f"  {label:<6}{geometry:<12}{coupling:<12}{cells}")

    print("\nSTRAIGHTNESS under each checkpoint's own coupling, and spatial structure at k = 100")
    print(
        f"{'prior':<7}{'geometry':<13}{'coupling':<13}{'straight':>10}"
        f"{'per seed':>20}{'lag1 generated':>16}{'lag1 data':>11}"
    )
    for (prior, geometry, coupling), group in runs.items():
        label = "white" if prior == "null" else prior
        straight = [float(r["straightness"]) for r in group]
        final = [by_step(r)[100] for r in group]
        print(
            f"{label:<7}{geometry:<13}{coupling:<13}{mean(straight):>10.3f}"
            f"{str([f'{x:.3f}' for x in straight]):>20}"
            f"{mean([f['spatial_lag1_generated'] for f in final]):>16.3f}"
            f"{mean([f['spatial_lag1_reference'] for f in final]):>11.3f}"
        )


if __name__ == "__main__":
    main()
