"""Table 1: how fast each bridge's argument turns, measured on the bridges.

Theorem 3 is a statement about paths: the Cartesian interpolant's argument can
rotate arbitrarily fast, the cylindrical one cannot. Measuring that through a
trained velocity field conflates the geometry with whatever the network learned
-- and on this project's synthetic target it did exactly that. A cylindrical
model that had not learned its phase velocity predicted almost none, so its peak
angular velocity looked comfortably below pi; the same probe on a model that
*had* learned it read 3.93, above pi. The bound belongs to the bridge's target,
never to a network's output. So this measures the bridges alone: no network, no
loss, no training, nothing that can be attributed to a confound.

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

What the paper prints from this is the ``> pi`` column and its energy-weighted
counterpart; the quantiles below it are diagnostic. That distinction matters:
the peak has a Pareto tail of index one, so an unweighted quantile of it moves
with wherever the sample is cut, while the energy share is a bounded average and
needs no threshold.
"""

from __future__ import annotations

import math
import pathlib
from typing import Any, ClassVar

import h5py
import numpy as np
import torch

from cyfm.core.experiment import BaseExperiment, ExperimentResult, register_experiment
from cyfm.data.synthetic import CylinderToy, cylinder_prior
from cyfm.flow.transport import euclidean_transport_permutation, shortest_angular_diff

__all__ = [
    "BridgeGeometryExperiment",
    "cartesian_at",
    "cartesian_peak",
    "closed_form_gap",
    "hill_tail_index",
    "render",
    "store_target",
]

# Guards a division by the squared closest approach; a chord through the origin
# to float precision is the divergence itself, not a numerical accident.
_DISTANCE_FLOOR = 1e-30

# Quantiles reported per arm, in column order.
QUANTILES = (0.5, 0.9, 0.99, 0.999)

ARMS = (
    "Cartesian / independent",
    "Cartesian / minibatch OT",
    "Cylindrical / independent",
)


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


def closed_form_gap(
    start: torch.Tensor, end: torch.Tensor, grid: int = 20001
) -> tuple[float, float]:
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
        ``(resolvable_share, median_relative_gap)``. The share used to be printed
        from inside this function, which is the pattern that made every probe in
        ``scripts/`` unrenderable: a measurement that reports itself cannot be
        turned into a table by anything else.

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
    assert bool((on_grid[resolved] <= checked * (1 + 1e-9) + 1e-12).all()), (
        "grid exceeded the closed form"
    )
    share = float(resolved.double().mean())
    if not bool(resolved.any()):
        return share, float("nan")
    return share, float(((checked - on_grid[resolved]) / checked).median())


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


@register_experiment("bridge_geometry")
class BridgeGeometryExperiment(BaseExperiment):
    """Peak angular velocity of the analytic bridges, on one target.

    Args:
        pairs: Endpoint pairs drawn.
        coupling: Amplitude-phase correlation of the synthetic target, ignored
            when ``target_store`` is given.
        seed: Seeds the prior, the target and the minibatch order.
        ot_batch: Batch the minibatch-OT pairing is solved over.
        target_store: A prebuilt store to draw the target from instead of the
            synthetic copula, which is how the knee and speech rows of Table 1
            are produced.
        store_fields: Fields read from that store.
        min_amplitude: Coefficient amplitude floor for a store target.
        min_field_peak: Field peak floor for a store target.
    """

    name: ClassVar[str] = "bridge_geometry"
    paper_reference: ClassVar[str] = "Table 1 (Section 5.1)"
    deterministic: ClassVar[bool] = True
    requires_data: ClassVar[bool] = False

    def __init__(
        self,
        pairs: int = 1_000_000,
        coupling: float = 0.5,
        seed: int = 0,
        ot_batch: int = 256,
        target_store: pathlib.Path | None = None,
        store_fields: int = 64,
        min_amplitude: float = 0.0,
        min_field_peak: float = 0.0,
    ) -> None:
        self.pairs = pairs
        self.coupling = coupling
        self.seed = seed
        self.ot_batch = ot_batch
        self.target_store = target_store
        self.store_fields = store_fields
        self.min_amplitude = min_amplitude
        self.min_field_peak = min_field_peak

    @property
    def requires_store(self) -> bool:
        """Whether this instance reads a store rather than generating its target."""
        return self.target_store is not None

    def execute(self) -> ExperimentResult:
        """Draw the endpoints, pair them both ways, and measure every arm.

        Returns:
            The result; see :func:`render` for what each key means.
        """
        generator = torch.Generator().manual_seed(self.seed)
        n = self.pairs

        prior = cylinder_prior(n, generator=generator).to(torch.complex128)
        if self.target_store is None:
            toy = CylinderToy(coupling=self.coupling, dtype=torch.float64)
            target = toy.sample(n, generator=generator)
            source = f"synthetic toy, coupling rho = {self.coupling}"
        else:
            target, source = store_target(
                self.target_store,
                n,
                self.store_fields,
                generator,
                self.min_amplitude,
                self.min_field_peak,
            )

        # Minibatch OT in the plane, the pairing a Cartesian OT flow trains on.
        usable = (n // self.ot_batch) * self.ot_batch
        paired = target[:usable].clone()
        for offset in range(0, usable, self.ot_batch):
            window = slice(offset, offset + self.ot_batch)
            permutation = euclidean_transport_permutation(prior[window], target[window])
            paired[window] = target[window][permutation]

        resolvable, gap = closed_form_gap(prior[:2000], target[:2000])

        peaks = {
            "Cartesian / independent": cartesian_peak(prior, target),
            "Cartesian / minibatch OT": cartesian_peak(prior[:usable], paired),
            "Cylindrical / independent": shortest_angular_diff(prior.angle(), target.angle()).abs(),
        }
        # The OT arm is measured against the permuted targets it is actually
        # paired with, not against the original order; weighting one by the other
        # silently mismatches every path with a stranger's energy.
        weights = {
            "Cartesian / independent": target.abs().square(),
            "Cartesian / minibatch OT": paired.abs().square(),
            "Cylindrical / independent": target.abs().square(),
        }
        at_half = {
            "Cartesian / independent": cartesian_at(prior, target, 0.5),
            "Cartesian / minibatch OT": cartesian_at(prior[:usable], paired, 0.5),
            "Cylindrical / independent": peaks["Cylindrical / independent"],
        }

        values: dict[str, Any] = {
            "pairs": n,
            "target": source,
            "ot_batch": self.ot_batch,
            "resolvable_share": resolvable,
            "closed_form_gap": gap,
            "arms": {
                name: {
                    **_summary(peaks[name]),
                    "exceeds_pi": float((peaks[name] > math.pi).double().mean()),
                    "energy_share": float(
                        (weights[name] * (peaks[name] > math.pi)).sum() / weights[name].sum()
                    ),
                }
                for name in ARMS
            },
            "at_half": {name: _summary(at_half[name]) for name in ARMS},
            "tail_index": {
                name: hill_tail_index(peaks[name])
                for name in ("Cartesian / independent", "Cartesian / minibatch OT")
            },
            "sample_max_growth": [
                [size, float(peaks["Cartesian / independent"][:size].max())]
                for size in (1_000, 10_000, 100_000, 1_000_000)
                if size <= n
            ],
        }
        return self.result(values, seed=self.seed)


def _summary(values: torch.Tensor) -> dict[str, Any]:
    """Quantiles and maximum of one arm.

    Args:
        values: The per-pair statistic.

    Returns:
        ``{"quantiles": [...], "max": ...}``, as plain floats.
    """
    probabilities = torch.tensor(QUANTILES, dtype=values.dtype)
    q = torch.quantile(values[: min(values.numel(), 16_000_000)], probabilities)
    return {"quantiles": [float(x) for x in q], "max": float(values.max())}


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


def _row(summary: dict[str, Any]) -> str:
    """One arm's quantile row, formatted."""
    cells = [_cell(v, 13) for v in summary["quantiles"]]
    return "".join(cells) + _cell(summary["max"], 15)


def render(result: ExperimentResult) -> str:
    """Render the measurement exactly as the script has always printed it.

    Args:
        result: What :meth:`BridgeGeometryExperiment.execute` returned.

    Returns:
        The report, ready to print.
    """
    v = result.values
    lines = [
        "=" * 100,
        "ANGULAR VELOCITY OF THE BRIDGES   no network, exact closed form for the Cartesian chord",
        "=" * 100,
        f"grid check: {v['resolvable_share']:.1%} of pairs resolvable in float64"
        " (the rest are radial or pass within rounding of the origin)",
        f"closed form vs 20001-point grid, median relative gap: {v['closed_form_gap']:.2e}"
        "  (grid never exceeds it)",
        f"pairs: {v['pairs']:,} | target: {v['target']} | OT batch {v['ot_batch']}",
        "",
        "1. PEAK |d arg(x_t)/dt| ALONG THE PATH",
        f"{'arm':<28}{'median':>13}{'q90':>13}{'q99':>13}{'q99.9':>13}{'max':>15}{'> pi':>9}",
        "-" * 104,
    ]
    for name in ARMS:
        arm = v["arms"][name]
        lines.append(f"{name:<28}{_row(arm)}{arm['exceeds_pi']:>8.1%}")

    lines += [
        "",
        "1b. SHARE OF TARGET ENERGY ON PATHS WITH PEAK > pi   no threshold, no truncation",
    ]
    for name in ARMS:
        lines.append(f"  {name:<28} {v['arms'][name]['energy_share']:>7.1%}")

    lines += [
        "",
        "2. AT t = 0.5, WHERE A CHORD PASSES CLOSEST TO THE ORIGIN ON AVERAGE",
        f"{'arm':<28}{'median':>13}{'q90':>13}{'q99':>13}{'q99.9':>13}{'max':>15}",
        "-" * 104,
    ]
    for name in ARMS:
        lines.append(f"{name:<28}{_row(v['at_half'][name])}")

    lines += [
        "",
        "3. THE TAIL   Hill index on the top 1%; 1 means P(peak > s) ~ c/s and an infinite mean",
    ]
    for name in ("Cartesian / independent", "Cartesian / minibatch OT"):
        lines.append(f"  {name:<28} tail index = {v['tail_index'][name]:.3f}")
    lines.append("  Cylindrical                  bounded by pi: no tail at all")

    lines += [
        "",
        "4. THE SAMPLE MAXIMUM IS A STATEMENT ABOUT SAMPLE SIZE, NOT GEOMETRY",
    ]
    for size, peak in v["sample_max_growth"]:
        lines.append(f"  first {size:>9,} pairs: max = {peak:>14.1f}")
    lines += [
        "  With a tail index near one the maximum grows roughly linearly in the sample count,",
        "  which is why two seeds of the trained probe read 23,870 and 62,605.",
    ]
    return "\n".join(lines)
