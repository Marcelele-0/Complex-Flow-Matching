"""Table 4 -- the Factorised Coupling Trap.

The cylindrical metric is a sum over amplitude and phase, and over pixels, so it
is tempting to solve the transport problem separately per coordinate. The
factorised minimum is always a lower bound on the joint one, and it is attained
exactly when some joint plan is simultaneously optimal in every coordinate --
which holds when both distributions are product measures, and degenerately when
source and target coincide, but not in general.

What this measures is the consequence. Coupling amplitude and phase
independently leaves both marginals bit-identical to the target, so every
marginal diagnostic passes; the circular-linear correlation between them
collapses to about 0.04 at every rho, and the reported transport cost stays flat
while the true optimum rises. The factorised plan reports a cost below the
optimum because it is not transporting the target: it reassembles endpoints from
different samples, and the law a flow-matching objective would converge to is
that reassembled law.

Two of the three sections of ``scripts/coupling_gate.py`` are here, the two Table
4 prints. The third -- whether the damage shrinks with batch size -- is a
diagnostic the paper does not publish and stays in the script.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from cyfm.core.experiment import BaseExperiment, ExperimentResult, register_experiment
from cyfm.data.synthetic import COUPLING_PRESETS, CylinderToy, Structure, cylinder_prior
from cyfm.flow.transport import (
    circular_transport_permutation,
    cylinder_transport_permutation,
    sorted_transport_permutation,
    transport_cost,
    wasserstein2_cylinder,
)
from cyfm.metrics import circular_linear_correlation

__all__ = ["Clouds", "FactorisedTrapExperiment", "draw", "marginal_gap", "render"]

# Typed as the Literal the toy accepts, not as bare str: iterating an untyped
# tuple hands CylinderToy a widened `str` and loses the only check that the
# two names here still match the ones it implements.
STRUCTURES: tuple[Structure, ...] = ("spiral", "cardioid")


@dataclass(frozen=True)
class Clouds:
    """One draw of the prior, the target, and both coupled endpoint clouds.

    Attributes:
        target_amplitude: Target amplitudes, shape ``[n]``.
        target_phase: Target phases, shape ``[n]``.
        chimera_amplitude: Endpoint amplitudes under the factorised coupling.
        chimera_phase: Endpoint phases under the factorised coupling.
        factorised_cost: Mean transport cost the factorised solution reports.
        joint_cost: Mean transport cost of the exact joint plan.
    """

    target_amplitude: torch.Tensor
    target_phase: torch.Tensor
    chimera_amplitude: torch.Tensor
    chimera_phase: torch.Tensor
    factorised_cost: float
    joint_cost: float


def draw(toy: CylinderToy, n: int, seed: int, phase_weight: float) -> Clouds:
    """Draw one prior/target pair and couple it both ways.

    Args:
        toy: The target sampler.
        n: Batch size.
        seed: Seed for this repeat.
        phase_weight: Weight on the angular term of the product metric.

    Returns:
        The clouds and the two transport costs.
    """
    prior = cylinder_prior(n, generator=torch.Generator().manual_seed(seed))
    amplitude, phase = toy.sample_polar(n, generator=torch.Generator().manual_seed(seed + 7919))
    prior_amplitude, prior_phase = prior.abs(), prior.angle()

    amplitude_perm = sorted_transport_permutation(prior_amplitude, amplitude)
    phase_perm = circular_transport_permutation(prior_phase, phase)
    chimera_amplitude, chimera_phase = amplitude[amplitude_perm], phase[phase_perm]

    joint_perm = cylinder_transport_permutation(
        prior_amplitude, prior_phase, amplitude, phase, phase_weight
    )

    return Clouds(
        target_amplitude=amplitude,
        target_phase=phase,
        chimera_amplitude=chimera_amplitude,
        chimera_phase=chimera_phase,
        factorised_cost=float(
            transport_cost(
                prior_amplitude, prior_phase, chimera_amplitude, chimera_phase, phase_weight
            )
        ),
        joint_cost=float(
            transport_cost(
                prior_amplitude,
                prior_phase,
                amplitude[joint_perm],
                phase[joint_perm],
                phase_weight,
            )
        ),
    )


def marginal_gap(first: torch.Tensor, second: torch.Tensor) -> float:
    """Two-sample Kolmogorov-Smirnov statistic between 1D clouds."""
    first_sorted, second_sorted = first.sort().values, second.sort().values
    grid = torch.cat([first_sorted, second_sorted]).sort().values
    cdf_first = torch.searchsorted(first_sorted, grid, right=True).float() / first.numel()
    cdf_second = torch.searchsorted(second_sorted, grid, right=True).float() / second.numel()
    return float((cdf_first - cdf_second).abs().max())


@register_experiment("factorised_trap")
class FactorisedTrapExperiment(BaseExperiment):
    """What coupling amplitude and phase independently costs.

    Args:
        samples: Endpoint pairs per draw. The paper uses 1024.
        seeds: Repeats averaged over. The paper uses 8, which its caption states.
        seed: Base seed; repeat ``r`` uses ``seed + 1000 * r``.
        phase_weight: Angular weight of the product metric the cost is measured in.
        structures: Target shapes swept. The paper prints the spiral rows.
    """

    name: ClassVar[str] = "factorised_trap"
    paper_reference: ClassVar[str] = "Table 4 (Section 5.4)"
    deterministic: ClassVar[bool] = True
    requires_data: ClassVar[bool] = False

    def __init__(
        self,
        samples: int = 1024,
        seeds: int = 8,
        seed: int = 0,
        phase_weight: float = 1.0,
        structures: tuple[Structure, ...] = STRUCTURES,
    ) -> None:
        self.samples = samples
        self.seeds = seeds
        self.seed = seed
        self.phase_weight = phase_weight
        self.structures = structures

    def execute(self) -> ExperimentResult:
        """Measure the joint damage and the cost gap, averaged over seeds.

        Returns:
            ``damage`` and ``cost`` rows, one per ``(structure, rho)``.
        """
        damage: list[dict[str, Any]] = []
        cost: list[dict[str, Any]] = []

        for structure in self.structures:
            for rho in COUPLING_PRESETS.values():
                toy = CylinderToy(coupling=rho, structure=structure)
                totals = dict.fromkeys(("rt", "rc", "ka", "kp", "w2", "floor"), 0.0)
                factorised = joint = 0.0

                for repeat in range(self.seeds):
                    repeat_seed = self.seed + 1000 * repeat
                    clouds = draw(toy, self.samples, repeat_seed, self.phase_weight)
                    # An independent draw of the same target, so the reported W2
                    # has a floor to be read against: two samples of one law are
                    # not at distance zero.
                    other_amplitude, other_phase = toy.sample_polar(
                        self.samples,
                        generator=torch.Generator().manual_seed(repeat_seed + 104729),
                    )
                    totals["rt"] += float(
                        circular_linear_correlation(clouds.target_amplitude, clouds.target_phase)
                    )
                    totals["rc"] += float(
                        circular_linear_correlation(clouds.chimera_amplitude, clouds.chimera_phase)
                    )
                    totals["ka"] += marginal_gap(clouds.chimera_amplitude, clouds.target_amplitude)
                    totals["kp"] += marginal_gap(clouds.chimera_phase, clouds.target_phase)
                    totals["w2"] += float(
                        wasserstein2_cylinder(
                            clouds.chimera_amplitude,
                            clouds.chimera_phase,
                            clouds.target_amplitude,
                            clouds.target_phase,
                            self.phase_weight,
                        )
                    )
                    totals["floor"] += float(
                        wasserstein2_cylinder(
                            other_amplitude,
                            other_phase,
                            clouds.target_amplitude,
                            clouds.target_phase,
                            self.phase_weight,
                        )
                    )
                    factorised += clouds.factorised_cost
                    joint += clouds.joint_cost

                mean = {key: value / self.seeds for key, value in totals.items()}
                damage.append({"structure": structure, "rho": rho, **mean})
                cost.append(
                    {
                        "structure": structure,
                        "rho": rho,
                        "factorised": factorised / self.seeds,
                        "joint": joint / self.seeds,
                        "deficit": (joint - factorised) / self.seeds,
                    }
                )

        return self.result(
            {
                "samples": self.samples,
                "seeds": self.seeds,
                "phase_weight": self.phase_weight,
                "damage": damage,
                "cost": cost,
            },
            seed=self.seed,
        )


def render(result: ExperimentResult) -> tuple[str, str]:
    """Render the two sections exactly as the script has always printed them.

    Args:
        result: What :meth:`FactorisedTrapExperiment.execute` returned.

    Returns:
        ``(joint_damage_section, cost_gap_section)``, so the script can keep
        printing its third section between them.
    """
    values = result.values
    samples, seeds = values["samples"], values["seeds"]

    damage = [
        "\n" + "=" * 96,
        f"1. MARGINALS SURVIVE, JOINT DOES NOT   (n = {samples}, {seeds} seeds)",
        "=" * 96,
        f"{'structure':<10} {'rho':>5} {'r_cl target':>12} {'r_cl chimera':>13} "
        f"{'KS(A)':>8} {'KS(theta)':>10} {'W2 chimera':>12} {'W2 floor':>10}",
        "-" * 96,
    ]
    previous = None
    for row in values["damage"]:
        if previous is not None and row["structure"] != previous:
            damage.append("-" * 96)
        previous = row["structure"]
        damage.append(
            f"{row['structure']:<10} {row['rho']:>5.2f} {row['rt']:>12.4f} {row['rc']:>13.4f} "
            f"{row['ka']:>8.5f} {row['kp']:>10.5f} {row['w2']:>12.4f} {row['floor']:>10.4f}"
        )
    damage += [
        "-" * 96,
        f"{'':<10} {'':>5}  r_cl = circular-linear correlation; KS compares the chimera",
        f"{'':<10} {'':>5}  cloud's marginals with the target's; W2 floor is an independent draw.",
    ]

    gap = [
        "\n" + "=" * 96,
        f"3. THE FACTORISED COST IS NOT ATTAINABLE   (n = {samples}, {seeds} seeds)",
        "=" * 96,
        f"{'structure':<10} {'rho':>5} {'factorised':>12} {'joint (exact)':>14} {'deficit':>10}",
        "-" * 96,
    ]
    previous = None
    for row in values["cost"]:
        if previous is not None and row["structure"] != previous:
            gap.append("-" * 96)
        previous = row["structure"]
        gap.append(
            f"{row['structure']:<10} {row['rho']:>5.2f} {row['factorised']:>12.4f} "
            f"{row['joint']:>14.4f} {row['joint'] - row['factorised']:>10.4f}"
        )
    gap += [
        "-" * 96,
        "  The joint plan is exact, so its cost is the true optimum over feasible plans.",
        "  Any deficit is the factorised relaxation scoring a plan that does not exist.",
    ]
    return "\n".join(damage), "\n".join(gap)
