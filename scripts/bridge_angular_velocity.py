"""Angular velocity of the two bridges, measured on the bridges and not on a network.

Theorem 3 is a statement about paths: the Cartesian interpolant's argument can
rotate arbitrarily fast, the cylindrical one cannot. Measuring that through a
trained velocity field conflates the geometry with whatever the network learned
-- and on this project's synthetic target it did exactly that. A cylindrical
model that had not learned its phase velocity predicted almost none, so its
peak angular velocity looked comfortably below pi; the same probe on a model
that *had* learned it read 3.93, above pi. The bound belongs to the bridge's
target, never to a network's output. So this script measures the bridges alone,
in the same register as ``scripts/chord_vs_arc.py``: no network, no loss, no
training, nothing that can be attributed to a confound.

The closed form
---------------
Along the straight Cartesian path ``x_t = (1 - t) z_0 + t z_1`` with velocity
``u = z_1 - z_0``, the rate at which the argument turns is

    theta_dot(t) = Im(conj(x_t) u) / |x_t|^2  =  L / |x_t|^2,
    L = Im(conj(z_0) z_1),

because ``Im(conj(u) u) = 0`` makes the numerator constant along the path. The
peak over ``t in [0, 1]`` is therefore ``|L| / d_min^2`` with ``d_min`` the
closest approach to the origin. When that approach lies inside the segment,
``d_min = |L| / |u|`` and the peak is ``|u|^2 / |L|``: it diverges exactly when
the chord passes through the origin, which happens with positive probability
density. Its tail is then a power law of index one -- ``P(peak > s) ~ c / s`` --
so the *expected* peak angular velocity is infinite, and any sample maximum is a
statement about sample size, not about the geometry.

On the cylinder the argument turns at ``u_phi = wrap(phi_1 - phi_0)``, constant
along the path and bounded by ``pi`` from the range of the wrap. The two numbers
are the same physical quantity -- ``d arg(x_t) / dt`` of the interpolant -- which
is what makes the comparison meaningful.

Usage::

    uv run python scripts/bridge_angular_velocity.py
    uv run python scripts/bridge_angular_velocity.py --pairs 2000000 --coupling 0.5
"""

from __future__ import annotations

import argparse
import math

import torch

from cfm.data.synthetic import CylinderToy, cylinder_prior
from cfm.flow.optimal_transport import euclidean_transport_permutation, shortest_angular_diff

# Guards a division by the squared closest approach; a chord through the origin
# to float precision is the divergence itself, not a numerical accident.
_DISTANCE_FLOOR = 1e-30


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=1_000_000, help="Endpoint pairs to draw.")
    parser.add_argument(
        "--coupling", type=float, default=0.5, help="Amplitude-phase dependence of the target."
    )
    parser.add_argument(
        "--ot-batch", type=int, default=256, help="Batch size for the minibatch OT arm."
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def cartesian_peak(start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
    """Exact peak of the induced angular velocity along each straight chord.

    Args:
        start: Complex endpoints at ``t = 0``, shape ``[n]``.
        end: Complex endpoints at ``t = 1``, shape ``[n]``.

    Returns:
        ``max_t |theta_dot(t)|`` for each pair, shape ``[n]``.
    """
    velocity = end - start
    moment = (start.conj() * end).imag.abs()
    squared_speed = velocity.abs().square().clamp_min(_DISTANCE_FLOOR)
    # Closest approach of the infinite line, clipped to the segment.
    t_star = (-(start.conj() * velocity).real / squared_speed).clamp(0.0, 1.0)
    closest = (start + t_star * velocity).abs().square().clamp_min(_DISTANCE_FLOOR)
    return moment / closest


def cartesian_at(start: torch.Tensor, end: torch.Tensor, t: float) -> torch.Tensor:
    """Induced angular velocity magnitude at a single time.

    Args:
        start: Complex endpoints at ``t = 0``.
        end: Complex endpoints at ``t = 1``.
        t: Time in ``[0, 1]``.

    Returns:
        ``|theta_dot(t)|`` for each pair.
    """
    position = (1.0 - t) * start + t * end
    moment = (start.conj() * end).imag.abs()
    return moment / position.abs().square().clamp_min(_DISTANCE_FLOOR)


def verify_closed_form(start: torch.Tensor, end: torch.Tensor, grid: int = 20001) -> float:
    """Compare the closed-form peak against a dense time grid.

    The grid can only under-estimate the true supremum, so the closed form must
    be at least as large everywhere and close wherever the peak is not razor
    sharp. Returns the median relative gap, which should be tiny.

    Args:
        start: Complex endpoints at ``t = 0``.
        end: Complex endpoints at ``t = 1``.
        grid: Number of time points.

    Returns:
        Median of ``(closed - grid) / closed`` over pairs.

    Raises:
        AssertionError: If the grid ever exceeds the closed form, which would
            mean the formula is wrong.
    """
    times = torch.linspace(0.0, 1.0, grid, dtype=torch.float64)
    start64, end64 = start.to(torch.complex128), end.to(torch.complex128)
    positions = (1.0 - times[:, None]) * start64[None] + times[:, None] * end64[None]
    velocity = end64 - start64
    rates = (positions.conj() * velocity[None]).imag.abs() / positions.abs().square()
    on_grid = rates.max(dim=0).values
    closed = cartesian_peak(start64, end64)
    assert bool((on_grid <= closed * (1 + 1e-9) + 1e-12).all()), "grid exceeded the closed form"
    return float(((closed - on_grid) / closed).median())


def hill_tail_index(values: torch.Tensor, fraction: float = 0.01) -> float:
    """Hill estimator of the power-law tail index from the top order statistics.

    An index of one means ``P(X > s) ~ c / s``, for which the mean is infinite.

    Args:
        values: Positive samples.
        fraction: Share of the largest samples used.

    Returns:
        The estimated tail index.
    """
    ordered = values.sort(descending=True).values
    k = max(10, int(fraction * ordered.numel()))
    top = ordered[: k + 1].double()
    return float(1.0 / (torch.log(top[:k]) - torch.log(top[k])).mean())


def quantile_row(values: torch.Tensor) -> str:
    """Median, tail quantiles and maximum, formatted."""
    probabilities = torch.tensor([0.5, 0.9, 0.99, 0.999], dtype=values.dtype)
    q = torch.quantile(values[: min(values.numel(), 16_000_000)], probabilities)
    cells = [f"{float(x):>12.3f}" for x in q] + [f"{float(values.max()):>14.1f}"]
    return "".join(cells)


def main() -> None:
    """Run the measurement."""
    args = parse_args()
    generator = torch.Generator().manual_seed(args.seed)
    n = args.pairs

    prior = cylinder_prior(n, generator=generator).to(torch.complex128)
    toy = CylinderToy(coupling=args.coupling, dtype=torch.float64)
    target = toy.sample(n, generator=generator)

    # Minibatch OT in the plane, the pairing a Cartesian OT flow trains on.
    batch = args.ot_batch
    usable = (n // batch) * batch
    paired = target[:usable].clone()
    for offset in range(0, usable, batch):
        window = slice(offset, offset + batch)
        permutation = euclidean_transport_permutation(prior[window], target[window])
        paired[window] = target[window][permutation]

    print("=" * 100)
    print("ANGULAR VELOCITY OF THE BRIDGES   no network, exact closed form for the Cartesian chord")
    print("=" * 100)
    gap = verify_closed_form(prior[:2000], target[:2000])
    print(
        f"closed form vs 20001-point grid, median relative gap: {gap:.2e}  (grid never exceeds it)"
    )
    print(f"pairs: {n:,} | target coupling rho = {args.coupling} | OT batch {batch}")

    arms = {
        "Cartesian / independent": cartesian_peak(prior, target),
        "Cartesian / minibatch OT": cartesian_peak(prior[:usable], paired),
        "Cylindrical / independent": shortest_angular_diff(prior.angle(), target.angle()).abs(),
    }

    print("\n1. PEAK |d arg(x_t)/dt| ALONG THE PATH")
    print(f"{'arm':<28}{'median':>12}{'q90':>12}{'q99':>12}{'q99.9':>12}{'max':>14}{'> pi':>9}")
    print("-" * 100)
    for name, values in arms.items():
        exceed = float((values > math.pi).double().mean())
        print(f"{name:<28}{quantile_row(values)}{exceed:>8.1%}")

    print("\n2. AT t = 0.5, WHERE A CHORD PASSES CLOSEST TO THE ORIGIN ON AVERAGE")
    print(f"{'arm':<28}{'median':>12}{'q90':>12}{'q99':>12}{'q99.9':>12}{'max':>14}")
    print("-" * 100)
    print(f"{'Cartesian / independent':<28}{quantile_row(cartesian_at(prior, target, 0.5))}")
    print(
        f"{'Cartesian / minibatch OT':<28}{quantile_row(cartesian_at(prior[:usable], paired, 0.5))}"
    )
    print(f"{'Cylindrical / independent':<28}{quantile_row(arms['Cylindrical / independent'])}")

    print(
        "\n3. THE TAIL   Hill index on the top 1%; 1 means P(peak > s) ~ c/s and an infinite mean"
    )
    for name in ("Cartesian / independent", "Cartesian / minibatch OT"):
        print(f"  {name:<28} tail index = {hill_tail_index(arms[name]):.3f}")
    print("  Cylindrical                  bounded by pi: no tail at all")

    print("\n4. THE SAMPLE MAXIMUM IS A STATEMENT ABOUT SAMPLE SIZE, NOT GEOMETRY")
    values = arms["Cartesian / independent"]
    for size in (1_000, 10_000, 100_000, 1_000_000):
        if size <= values.numel():
            print(f"  first {size:>9,} pairs: max = {float(values[:size].max()):>14.1f}")
    print("  With a tail index near one the maximum grows roughly linearly in the sample count,")
    print("  which is why two seeds of the trained probe read 23,870 and 62,605.")


if __name__ == "__main__":
    main()
