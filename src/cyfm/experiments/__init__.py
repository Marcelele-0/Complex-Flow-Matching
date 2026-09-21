"""Network-free measurements, as classes that return their numbers.

Each one produces figures the paper prints and needs no trained network, so each
is reproducible from a bare checkout in seconds. They live here rather than in
``scripts/`` because a script cannot be imported, tested or rendered; the scripts
remain as thin command-line front ends, and must, since
``conf/experiment/*.yaml`` names them by path.
"""

from __future__ import annotations

from cyfm.core.experiment import EXPERIMENTS, BaseExperiment, ExperimentResult
from cyfm.experiments.bridge_geometry import BridgeGeometryExperiment
from cyfm.experiments.coupling_scaling import CouplingScalingExperiment
from cyfm.experiments.factorised_trap import FactorisedTrapExperiment

__all__ = [
    "EXPERIMENTS",
    "BaseExperiment",
    "BridgeGeometryExperiment",
    "CouplingScalingExperiment",
    "FactorisedTrapExperiment",
    "ExperimentResult",
]
