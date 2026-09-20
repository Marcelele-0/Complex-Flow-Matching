"""Entry-point pipelines: the work ``train.py`` and ``evaluate.py`` used to hold.

Both were a single Hydra ``main`` -- 535 lines and 226 lines -- with no seam
between deciding what to run and running it. Neither had a unit test, and neither
could: reaching any of that logic meant standing up Hydra and executing the whole
thing. The entry points keep their module paths, which
``tests/test_experiments.py`` pins, and delegate here.
"""

from __future__ import annotations

from cyfm.pipelines.evaluation import EvaluationPipeline, render_report
from cyfm.pipelines.reporting import NullLogger, RunLogger, WandbLogger, build_logger
from cyfm.pipelines.training import TrainingPipeline

__all__ = [
    "EvaluationPipeline",
    "NullLogger",
    "RunLogger",
    "TrainingPipeline",
    "WandbLogger",
    "build_logger",
    "render_report",
]
