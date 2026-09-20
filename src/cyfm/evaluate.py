"""Generative evaluation: does the model reproduce the data distribution?

This entry point replaces a paired one. The reconstruction path scored a
prediction against the specific slice it was derived from -- PSNR, SSIM, a phase
error and a data-consistency term, each meaningful only because the target was
known. Pure synthesis has no such target: a sample drawn from the prior has no
reason to match any particular slice, and every paired metric is therefore poor
*by construction* at ``t = 0``, which is precisely the setting the project now
cares about. Scoring synthesis needs distributional metrics, and those are what
this module computes.

What is measured, and why each one is here
------------------------------------------
**Sliced 2-Wasserstein on the complex plane.** The headline number. Coefficients
from generated and reference fields are pooled and compared as points in ``R^2``.
Sliced rather than exact because the exact assignment's finite-sample floor is
large enough to swallow the differences worth seeing: measured on this project's
synthetic target, the exact estimator's floor at 2048 samples is 0.078 while the
sliced estimator reaches 0.004 at 32768, against a separation of 0.188 between
genuinely different distributions.

**Exact transport on each marginal.** Amplitude on the line and phase on the
circle, both solved exactly. A model can match a pooled two-dimensional cloud
while getting one marginal wrong in a way slicing averages away, and the phase
marginal is the one this project makes claims about.

**The dependence gap.** The circular-linear correlation of the generated
coefficients against the reference's. Amplitude-phase dependence is the axis the
synthetic datasets sweep and the thing a factorised coupling destroys, so a model
that reproduces both marginals and none of the dependence has to be visible as a
number rather than as an argument.

**The spatial gap.** Lag-one autocorrelation of the amplitude field, generated
against reference. Every metric above pools coefficients and is therefore blind
to spatial structure entirely; without this one a model could match the pointwise
law perfectly and produce noise where the data has fields.

**Straightness.** The regression residual of the conditional velocity,
normalised by the displacement it had to explain. Both bridges carry a velocity
constant along the path, so this is the rectified-flow straightness statistic
directly rather than the training loss under another name. Computed under the
coupling the checkpoint was trained with, read from ``training.coupling``.

All of the above are swept over solver step counts, because the number of
function evaluations a geometry needs is a claim this project makes and a table
it has to be able to produce. Every metric lives in :mod:`cyfm.metrics` -- the
path probes included, which used to stay here on the grounds that they need the
model and the manifold. They do, and taking both as arguments costs nothing;
what it buys is that they can be tested without standing up a Hydra ``main``.

The module path is pinned: ``tests/test_experiments.py`` asserts that every
archived run's command line begins ``python -m cyfm.evaluate``. The measurement
itself is :class:`~cyfm.pipelines.evaluation.EvaluationPipeline`, which returns
the payload rather than printing it, so the same numbers can be re-rendered from
an archive without regenerating a sample.
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from cyfm.pipelines.evaluation import EvaluationPipeline, render_report


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Score a checkpoint's samples against the data distribution.

    Args:
        cfg: The composed Hydra config.
    """
    print(OmegaConf.to_yaml(cfg))

    payload = EvaluationPipeline.from_config(cfg, hydra.utils.get_original_cwd()).run()
    print(render_report(payload))

    output_dir = Path(str((cfg.get("paths") or {}).get("output_dir", ".")))
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "metrics.json"
    destination.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {destination}")


if __name__ == "__main__":
    main()
