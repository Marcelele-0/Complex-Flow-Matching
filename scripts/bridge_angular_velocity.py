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
import pathlib

import h5py
import numpy as np
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
    parser.add_argument(
        "--target-store",
        type=pathlib.Path,
        default=None,
        help="Draw the target from this HDF5 store instead of the synthetic toy.",
    )
    parser.add_argument(
        "--store-fields",
        type=int,
        default=4096,
        help="Fields to read from the store; enough to sample from, cheap to load.",
    )
    parser.add_argument(
        "--min-field-peak",
        type=float,
        default=0.0,
        help="Drop whole fields whose absolute peak is below this fraction of the loudest.",
    )
    parser.add_argument(
        "--min-amplitude",
        type=float,
        default=0.0,
        help="Drop target coefficients below this fraction of their field's peak.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def store_target(
    path: pathlib.Path,
    pairs: int,
    fields: int,
    generator: torch.Generator,
    min_amplitude: float = 0.0,
    min_field_peak: float = 0.0,
) -> tuple[torch.Tensor, str]:
    """Draw complex coefficients from a prebuilt store, normalised as training sees them.

    The manifold transform divides each field by its own peak modulus, so the amplitudes
    a bridge actually meets are relative ones. Drawing raw coefficients would compare a
    prior supported on the unit disc against a target on an arbitrary physical scale,
    and the resulting geometry would be an artefact of that mismatch rather than of the
    data. Normalising per field, exactly as the training transform does, is what makes
    the measurement comparable to the synthetic row above it.

    Args:
        path: Store written by one of the ``build_*_store`` scripts.
        pairs: Coefficients to draw.
        fields: Upper bound on fields read; a few thousand already hold far more
            coefficients than are sampled.
        generator: Seeded RNG, so a rerun prints the same numbers.
        min_amplitude: Keep only coefficients at or above this fraction of their field's
            peak. The peak angular velocity of a chord grows as the target amplitude
            falls, and a real spectrum spends most of its bins near its own noise floor,
            where the phase carries nothing. Reporting the measurement with and without
            this filter separates a geometric fact about the parametrisation from an
            artefact of counting bins nobody hears.

    Returns:
        The drawn coefficients as ``complex128``, and a label naming the store.

    Raises:
        FileNotFoundError: If the store is absent.
        KeyError: If it holds neither of the two dataset names this project writes.
    """
    if not path.is_file():
        raise FileNotFoundError(f"No store at {path}")
    with h5py.File(path, "r") as handle:
        for key in ("frames", "images"):
            if key in handle:
                break
        else:
            raise KeyError(f"{path} holds none of ('frames', 'images'); has {list(handle)}")
        available = handle[key].shape[0]
        take = min(fields, available)
        if take < available:
            chosen = torch.randperm(available, generator=generator)[:take].sort().values
            block = handle[key][chosen.numpy()]
        else:
            block = handle[key][:]
        data = torch.from_numpy(np.asarray(block, dtype=np.complex64))

    spatial = tuple(range(1, data.ndim))
    peak = data.abs().amax(dim=spatial, keepdim=True).clamp_min(1e-12)
    dropped_fields = 0
    if min_field_peak > 0.0:
        # Applied before the per-field normalisation, and that order is the point: a field
        # that holds only silence still has its own maximum divided out, so afterwards its
        # noise floor looks as loud as a vowel. Judging a field by its absolute peak is the
        # only way to tell the two apart.
        keep = peak.reshape(-1) >= min_field_peak * peak.max()
        dropped_fields = int((~keep).sum())
        if not bool(keep.any()):
            raise ValueError(f"no field of {path.name} reaches {min_field_peak} of the loudest")
        data, peak = data[keep], peak[keep]
    flat = (data / peak).reshape(-1).to(torch.complex128)
    total = flat.numel()
    label = f"{path.name} ({take - dropped_fields:,} fields"
    if dropped_fields:
        label += f" after dropping {dropped_fields / take:.1%} below {min_field_peak:g} peak"
    label += f", {total:,} coefficients"
    if min_amplitude > 0.0:
        flat = flat[flat.abs() >= min_amplitude]
        if flat.numel() == 0:
            raise ValueError(f"no coefficient of {path.name} reaches {min_amplitude}")
        label += f", {flat.numel() / total:.1%} kept above {min_amplitude:g} of peak"
    index = torch.randint(flat.numel(), (pairs,), generator=generator)
    return flat[index], label + ")"


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
    raw = -(start.conj() * velocity).real / squared_speed

    # Where the foot of the perpendicular falls inside the segment, the closest approach
    # is |moment| / |velocity| and the peak reduces to |velocity|^2 / |moment|. Forming it
    # that way rather than as moment / |start + t velocity|^2 is not a simplification but
    # a numerical necessity: a chord that passes close to the origin makes that sum a
    # difference of two nearly equal complex numbers, and on real normalised data -- where
    # most coefficients sit far below their field's peak -- the cancellation leaves the
    # result dominated by rounding. The dense-grid check in verify_closed_form fails on
    # both real stores with the cancelling form and passes with this one.
    interior = squared_speed / moment.clamp_min(_DISTANCE_FLOOR)

    # At a clamped foot the closest approach is an endpoint modulus, computed directly and
    # so free of that cancellation.
    endpoint = torch.where(raw <= 0.0, start.abs().square(), end.abs().square())
    at_endpoint = moment / endpoint.clamp_min(_DISTANCE_FLOOR)

    return torch.where((raw > 0.0) & (raw < 1.0), interior, at_endpoint)


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

    The comparison is only made where the grid itself is trustworthy. Evaluating
    ``(1 - t) z_0 + t z_1`` near the origin subtracts two nearly equal complex numbers, so
    when the closest approach falls far below the endpoint moduli the grid's position is
    rounding noise and can come out *smaller* than the true minimum -- making its rate
    exceed the closed form and, before this gate existed, aborting the whole measurement
    on real data. The closed form has no such cancellation, so those pairs are excluded
    from the check rather than from the measurement.

    Args:
        start: Complex endpoints at ``t = 0``.
        end: Complex endpoints at ``t = 1``.
        grid: Number of time points.

    Returns:
        Median of ``(closed - grid) / closed`` over the pairs the grid can resolve.

    Raises:
        AssertionError: If the grid exceeds the closed form on a pair it can resolve,
            which would mean the formula is wrong.
    """
    times = torch.linspace(0.0, 1.0, grid, dtype=torch.float64)
    start64, end64 = start.to(torch.complex128), end.to(torch.complex128)
    positions = (1.0 - times[:, None]) * start64[None] + times[:, None] * end64[None]
    velocity = end64 - start64
    rates = (positions.conj() * velocity[None]).imag.abs() / positions.abs().square()
    on_grid = rates.max(dim=0).values
    closed = cartesian_peak(start64, end64)

    scale = torch.maximum(start64.abs(), end64.abs()).clamp_min(_DISTANCE_FLOOR)
    closest = (closed / (start64.conj() * end64).imag.abs().clamp_min(_DISTANCE_FLOOR)).rsqrt()
    # A target coefficient that is exactly zero -- background outside the imaged field, and
    # the knee store holds many -- makes the chord radial: its argument never turns, so the
    # closed form correctly reports zero, while the grid evaluates 0/0 at t = 1 and returns
    # NaN. Excluding those from the comparison keeps a degenerate case from being read as a
    # failure of the formula.
    resolved = (closest / scale > 1e-6) & torch.isfinite(on_grid)
    checked = closed[resolved]
    assert bool(
        (on_grid[resolved] <= checked * (1 + 1e-9) + 1e-12).all()
    ), "grid exceeded the closed form"
    if not bool(resolved.any()):
        return float("nan")
    print(
        f"grid check: {float(resolved.double().mean()):.1%} of pairs resolvable in float64"
        f" (the rest are radial or pass within rounding of the origin)"
    )
    return float(((checked - on_grid[resolved]) / checked).median())


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


def _cell(value: float, width: int) -> str:
    """One number, switched to exponent notation before it can run into its neighbour.

    On a real store the upper quantiles of the Cartesian peak reach 1e30 and beyond,
    because the statistic has a Pareto tail of index one and the coefficients near a
    field's noise floor drive it without limit. Fixed-point formatting silently ran
    those columns together into an unreadable digit string, which looked like a
    formatting nuisance but is really the same fact: an unweighted quantile of this
    quantity is a statement about float64 rather than about the data, which is why the
    energy-weighted share exists and is the number to quote.
    """
    return f"{value:>{width}.3g}" if abs(value) >= 1e6 else f"{value:>{width}.3f}"


def quantile_row(values: torch.Tensor) -> str:
    """Median, tail quantiles and maximum, formatted."""
    probabilities = torch.tensor([0.5, 0.9, 0.99, 0.999], dtype=values.dtype)
    q = torch.quantile(values[: min(values.numel(), 16_000_000)], probabilities)
    cells = [_cell(float(x), 13) for x in q] + [_cell(float(values.max()), 15)]
    return "".join(cells)


def main() -> None:
    """Run the measurement."""
    args = parse_args()
    generator = torch.Generator().manual_seed(args.seed)
    n = args.pairs

    prior = cylinder_prior(n, generator=generator).to(torch.complex128)
    if args.target_store is None:
        toy = CylinderToy(coupling=args.coupling, dtype=torch.float64)
        target = toy.sample(n, generator=generator)
        source = f"synthetic toy, coupling rho = {args.coupling}"
    else:
        target, source = store_target(
            args.target_store,
            n,
            args.store_fields,
            generator,
            args.min_amplitude,
            args.min_field_peak,
        )

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
    print(f"pairs: {n:,} | target: {source} | OT batch {batch}")

    arms = {
        "Cartesian / independent": cartesian_peak(prior, target),
        "Cartesian / minibatch OT": cartesian_peak(prior[:usable], paired),
        "Cylindrical / independent": shortest_angular_diff(prior.angle(), target.angle()).abs(),
    }

    print("\n1. PEAK |d arg(x_t)/dt| ALONG THE PATH")
    print(f"{'arm':<28}{'median':>13}{'q90':>13}{'q99':>13}{'q99.9':>13}{'max':>15}{'> pi':>9}")
    print("-" * 104)
    for name, values in arms.items():
        exceed = float((values > math.pi).double().mean())
        print(f"{name:<28}{quantile_row(values)}{exceed:>8.1%}")

    # Weighted by the target's energy and truncated nowhere. The peak angular velocity
    # has a Pareto tail of index one, so its mean is infinite and any quantile of it
    # moves with wherever the sample is cut -- on real data most coefficients sit near
    # their field's noise floor, and including or excluding them changes an unweighted
    # quantile several-fold while changing the signal by a fraction of a percent. This
    # statistic has no such freedom: it is a bounded average, it needs no threshold,
    # and it answers what a coarse integrator actually has to cope with -- the share of
    # the signal's energy that lies on paths turning faster than pi.
    print("\n1b. SHARE OF TARGET ENERGY ON PATHS WITH PEAK > pi   no threshold, no truncation")
    # The OT arm is measured against the permuted targets it is actually paired with,
    # not against the original order; weighting one by the other silently mismatches
    # every path with a stranger's energy.
    weights = {
        "Cartesian / independent": target.abs().square(),
        "Cartesian / minibatch OT": paired.abs().square(),
        "Cylindrical / independent": target.abs().square(),
    }
    for name, values in arms.items():
        w = weights[name]
        share = float((w * (values > math.pi)).sum() / w.sum())
        print(f"  {name:<28} {share:>7.1%}")

    print("\n2. AT t = 0.5, WHERE A CHORD PASSES CLOSEST TO THE ORIGIN ON AVERAGE")
    print(f"{'arm':<28}{'median':>13}{'q90':>13}{'q99':>13}{'q99.9':>13}{'max':>15}")
    print("-" * 104)
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
