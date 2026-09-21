"""Network-free measurements: what one is, and what it returns.

Five probes in ``scripts/`` produce numbers the paper prints -- Table 1's path
geometry, Table 4's factorised coupling trap, Section 5.3's transport-cost curve.
In every one of them the measurement and the report are the same function.
``report()`` prints and returns at once, or prints and returns ``None``; none has
a function that hands back the numbers.

That has a cost beyond tidiness. None of the five can be unit-tested, none can
emit JSON, and none can be rendered by ``scripts/paper/latex_tables.py`` the way
the trained arms are -- which is why four of the paper's eight tables are
hand-written LaTeX. Separating the measurement from the printing is the condition
for closing that chain, not a cosmetic preference.

An experiment therefore returns an :class:`ExperimentResult` and prints nothing.
Rendering it as text, JSON or LaTeX is somebody else's job, and can be done again
later from an archive without recomputing anything.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from cyfm.core.pipeline import BasePipeline
from cyfm.core.registry import Registry

__all__ = ["EXPERIMENTS", "BaseExperiment", "ExperimentResult", "register_experiment"]


@dataclass(frozen=True)
class ExperimentResult:
    """The numbers one measurement produced, and enough context to trust them.

    Frozen: a result is a record of what happened, and something that rendered it
    must not be able to change it before something else does.

    Attributes:
        name: The experiment's registry key.
        paper_reference: Where the numbers appear, e.g. ``"Table 1 (Section 5.1)"``.
            Carried with the result so a renderer does not have to be told.
        values: The measurement, nested as the renderer wants it. Plain Python
            containers only, so a result serialises to JSON without a custom
            encoder.
        seed: The seed every draw came from, or ``None`` for a measurement with
            no randomness in it.
        deterministic: Whether re-running with the same seed gives the same
            digits. False for a measurement whose value moves with the seed by
            more than its last printed digit -- which is a property worth
            stating, because the paper prints one of those as if it were fixed.
    """

    name: str
    paper_reference: str
    values: Mapping[str, Any] = field(default_factory=dict)
    seed: int | None = None
    deterministic: bool = True


class BaseExperiment(BasePipeline[ExperimentResult]):
    """A measurement that needs no trained network.

    Subclasses declare what they are and implement
    :meth:`~cyfm.core.pipeline.BasePipeline.execute`, which returns an
    :class:`ExperimentResult` and prints nothing.

    Attributes:
        name: Registry key.
        paper_reference: Where the numbers appear in the paper.
        deterministic: Whether the digits are fixed given the seed.
        requires_data: Whether a store has to exist on disk. False for every
            probe that generates its own target, which is what makes those
            reproducible from a bare checkout.
    """

    name: ClassVar[str]
    paper_reference: ClassVar[str]
    deterministic: ClassVar[bool] = True
    requires_data: ClassVar[bool] = False

    def result(self, values: Mapping[str, Any], seed: int | None = None) -> ExperimentResult:
        """Wrap a measurement in this experiment's declared context.

        Args:
            values: The numbers.
            seed: The seed they came from.

        Returns:
            The result.
        """
        return ExperimentResult(
            name=self.name,
            paper_reference=self.paper_reference,
            values=values,
            seed=seed,
            deterministic=self.deterministic,
        )


EXPERIMENTS: Registry[BaseExperiment] = Registry("experiments")

#: Decorator spelling, matching ``register_dataset`` and the rest.
register_experiment = EXPERIMENTS.register
