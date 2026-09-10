"""How the U-Net comparison moves with field size: 16x16, 32x32 and 64x64.

White prior throughout. The 64x64 runs come from ``unet_prior.sh`` (the
``sc=null`` arm), the smaller ones from ``unet_size.sh``. Reads whatever
evaluations exist, so it can run while the sweep is still going.

Usage::

    uv run python scripts/sweeps/analyse_unet_size.py
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

EVALUATIONS = pathlib.Path(__file__).resolve().parents[2] / "outputs" / "evaluate"
STEPS = (1, 2, 4, 8, 100)
SIZES = (16, 32, 64)
GEOMETRIES = ("euclidean", "cylindrical")
COUPLINGS = ("independent", "ot")
SEEDS = (0, 1)


def evaluation_name(side: int, geometry: str, coupling: str, seed: int) -> str:
    """Directory name of one evaluation."""
    if side == 64:
        return f"un_{geometry}_{coupling}_scnull_s{seed}_eval"
    return f"unsz_{geometry}_{coupling}_{side}_s{seed}_eval"


def latest(name: str) -> dict[str, Any] | None:
    """The most recent metrics.json for one evaluation, if it exists."""
    found = sorted(EVALUATIONS.glob(f"{name}/*/metrics.json"))
    return json.loads(found[-1].read_text()) if found else None


def by_step(run: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Index a run's sweep by number of solver steps."""
    return {int(entry["num_steps"]): entry for entry in run["sweep"]}


def main() -> None:
    """Print the size comparison."""
    runs: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for side in SIZES:
        for geometry in GEOMETRIES:
            for coupling in COUPLINGS:
                for seed in SEEDS:
                    run = latest(evaluation_name(side, geometry, coupling, seed))
                    if run is not None:
                        runs.setdefault((side, geometry, coupling), []).append(run)
    total = sum(len(group) for group in runs.values())
    print(f"U-Net size sweep: {total} of 24 evaluations available (white prior)")
    if not runs:
        return

    for key in ("sliced_w2_complex", "w2_phase_circular", "w2_amplitude"):
        print(f"\n{key}  (mean over available seeds; k = solver steps, NFE = 2k - 1)")
        print(
            f"{'field':<7}{'geometry':<13}{'coupling':<13}{'n':>3}"
            + "".join(f"{f'k={k}':>9}" for k in STEPS)
        )
        for (side, geometry, coupling), group in sorted(runs.items()):
            cells = "".join(
                f"{sum(by_step(r)[k][key] for r in group) / len(group):>9.3f}" for k in STEPS
            )
            print(f"{f'{side}x{side}':<7}{geometry:<13}{coupling:<13}{len(group):>3}{cells}")

    print("\nCYLINDER + OT against the best Cartesian arm, sliced W2 per seed")
    for side in SIZES:
        cylinder = runs.get((side, "cylindrical", "ot"), [])
        cartesian = [runs.get((side, "euclidean", c), []) for c in COUPLINGS]
        if not cylinder or not all(cartesian):
            continue
        for k in STEPS:
            cyl = [round(by_step(r)[k]["sliced_w2_complex"], 3) for r in cylinder]
            means = [sum(by_step(r)[k]["sliced_w2_complex"] for r in g) / len(g) for g in cartesian]
            best = cartesian[means.index(min(means))]
            euc = [round(by_step(r)[k]["sliced_w2_complex"], 3) for r in best]
            separated = max(cyl) < min(euc) or min(cyl) > max(euc)
            verdict = (
                ("cylinder" if max(cyl) < min(euc) else "Cartesian") if separated else "overlap"
            )
            print(f"  {side}x{side} k={k:<4} cylinder OT {cyl}  best Cartesian {euc}  -> {verdict}")

    print("\nSTRAIGHTNESS under each checkpoint's own coupling, spatial lag-1 at k = 100")
    for (side, geometry, coupling), group in sorted(runs.items()):
        straight = sum(float(r["straightness"]) for r in group) / len(group)
        generated = sum(by_step(r)[100]["spatial_lag1_generated"] for r in group) / len(group)
        reference = sum(by_step(r)[100]["spatial_lag1_reference"] for r in group) / len(group)
        print(
            f"  {f'{side}x{side}':<7}{geometry:<13}{coupling:<13}"
            f"straight {straight:.3f}  lag1 {generated:.3f} (data {reference:.3f})"
        )


if __name__ == "__main__":
    main()
