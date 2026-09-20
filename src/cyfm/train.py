"""Training entry point.

The module path is pinned: ``tests/test_experiments.py`` asserts that every
archived run's command line begins ``python -m cyfm.train``, so this module must
keep both its name and its Hydra ``main``.

Everything it used to contain is in :class:`~cyfm.pipelines.training.TrainingPipeline`.
That is not a tidiness move. This was one ``main`` of 535 lines with no seam
between deciding what to run and running it, which meant none of it had a unit
test and none of it could have one: reaching any of that logic required standing
up Hydra and executing the whole thing.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from cyfm.pipelines.training import TrainingPipeline
from cyfm.utils.distributed import print_main


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Compose the config, then hand it to the pipeline.

    Args:
        cfg: The composed Hydra config.
    """
    print_main(OmegaConf.to_yaml(cfg))
    TrainingPipeline.from_config(cfg).run()


if __name__ == "__main__":
    main()
