"""The Hydra boundary.

Everything in this package knows about ``DictConfig``; nothing outside it should.
The library proper -- geometries, bridges, couplings, solvers, metrics -- takes
plain Python arguments, so it can be imported and used without Hydra in the
picture, while the reproduction layer keeps composing the configs the archives
record.
"""

from __future__ import annotations

from cyfm.config.adapters import manifold_from_config
from cyfm.config.resolve import as_plain_dict
from cyfm.config.schema import (
    EvaluateConfig,
    LoggingConfig,
    PathsConfig,
    RunConfig,
    SchedulerConfig,
    TrainingConfig,
)

__all__ = [
    "EvaluateConfig",
    "LoggingConfig",
    "PathsConfig",
    "RunConfig",
    "SchedulerConfig",
    "TrainingConfig",
    "as_plain_dict",
    "manifold_from_config",
]
