"""Section 5.3: how much minibatch optimal transport buys as fields grow.

OT's advantage is a statement about *pairs*, and pairs get less distinguishable
as dimension rises. At one coefficient per field a batch of 64 has plenty of room
to be re-matched; at 320x320 the squared geodesic displacement is a sum over
102,400 coordinates, concentration of measure flattens the cost matrix, and the
best assignment is barely better than the one the dataloader happened to draw.

The paper quotes two points of this curve -- 2.8% at 64x64 against 0.5% at
320x320 -- to explain why the knee results, which are computed on real data at
the larger size, gain less from the coupling than the synthetic tables do. The
assignment still reorders almost the whole batch at every size, which is the
point: it is not that OT stops finding a permutation, it is that every
permutation costs nearly the same.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch

from cyfm.core.experiment import BaseExperiment, ExperimentResult, register_experiment
from cyfm.core.manifold import BaseManifold
from cyfm.data.toy import CylinderToyIIDDataset
from cyfm.flow.couplings import OptimalTransportCoupling
from cyfm.manifolds import build_manifold

__all__ = ["CouplingScalingExperiment", "pairing_cost", "render"]

#: Field sides swept. 320 is the knee acquisition's own side, and the only size
#: here where a paper table is computed on real data rather than synthetic fields.
SIDES = (1, 2, 4, 8, 16, 32, 64, 128, 320)


def pairing_cost(manifold: BaseManifold, prior: torch.Tensor, data: torch.Tensor) -> float:
    """Mean weighted squared geodesic displacement of an aligned pairing.

    Args:
        manifold: The geometry whose metric the displacement is measured in. Its
            ``tangent_weights`` matter: on the cylinder the angular coordinate
            spans ``[-pi, pi]`` while the amplitude is ``O(1)``, so an unweighted
            sum would declare them commensurate.
        prior: Prior states ``[B, state_channels, H, W]``.
        data: Data states, already in the order they are paired in.

    Returns:
        The mean cost per coefficient.
    """
    weights = manifold.tangent_weights.to(prior.device).reshape(1, -1, 1, 1)
    displacement = manifold.log_map(prior, data)
    return float((displacement.square() * weights).flatten(1).mean(1).mean())


@register_experiment("coupling_scaling")
class CouplingScalingExperiment(BaseExperiment):
    """Transport-cost saving of minibatch OT against field size.

    Args:
        sides: Field sides to sweep.
        batch: Batch the assignment is solved over.
        seed: Seeds the prior and the data.
        device: Where to compute; CPU if unset.
    """

    name: ClassVar[str] = "coupling_scaling"
    paper_reference: ClassVar[str] = "Section 5.3"
    deterministic: ClassVar[bool] = True
    requires_data: ClassVar[bool] = False

    def __init__(
        self,
        sides: tuple[int, ...] = SIDES,
        batch: int = 64,
        seed: int = 0,
        device: torch.device | None = None,
    ) -> None:
        self.sides = sides
        self.batch = batch
        self.seed = seed
        self.device = device or torch.device("cpu")

    def execute(self) -> ExperimentResult:
        """Solve the assignment at every size and record what it saved.

        Returns:
            One row per side: the cost drop, the share of the batch the
            assignment moved, and the cost matrix's relative spread -- which is
            the quantity that collapses, and the reason the drop does.
        """
        manifold = build_manifold({"name": "cylindrical", "spatial_correlation": None})
        coupling = OptimalTransportCoupling()
        rows: list[dict[str, Any]] = []

        for side in self.sides:
            dataset = CylinderToyIIDDataset(
                coupling=0.5, size=self.batch, crop_size=(side, side), seed=self.seed
            )
            raw = torch.stack([dataset[i] for i in range(self.batch)])
            data = manifold.from_complex(raw).to(self.device)
            prior = manifold.sample_noise(
                self.batch,
                side,
                side,
                self.device,
                generator=torch.Generator(device=self.device).manual_seed(self.seed),
            )
            cost = coupling.cost_matrix(prior, data, manifold)
            paired = coupling(prior, data, manifold)
            rows.append(
                {
                    "side": side,
                    "dim": side * side,
                    "cost_drop": 1.0
                    - pairing_cost(manifold, prior, paired) / pairing_cost(manifold, prior, data),
                    "reordered": float(
                        (~torch.isclose(paired, data).flatten(1).all(1)).double().mean()
                    ),
                    "cost_spread": float(cost.std() / cost.mean()),
                }
            )

        return self.result({"batch": self.batch, "rows": rows}, seed=self.seed)


def render(result: ExperimentResult) -> str:
    """Render the sweep as the table the probe prints.

    Args:
        result: What :meth:`CouplingScalingExperiment.execute` returned.

    Returns:
        The report, ready to print.
    """
    values = result.values
    lines = [
        "=" * 96,
        "1. OT COST REDUCTION vs FIELD SIZE   cylinder_toy_iid rho 0.5, "
        f"batch {values['batch']}, white prior",
        "=" * 96,
        f"{'field':>8}{'dim':>7}{'cost drop':>12}{'reordered':>12}{'cost spread':>14}",
    ]
    for row in values["rows"]:
        side = row["side"]
        lines.append(
            f"{f'{side}x{side}':>8}{row['dim']:>7}{row['cost_drop']:>11.1%}"
            f"{row['reordered']:>11.1%}{row['cost_spread']:>14.4f}"
        )
    return "\n".join(lines)
