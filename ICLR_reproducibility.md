# Reproducibility

This is the code behind the submission. It is written for a reviewer who wants to check
whether the numbers in the paper came from these experiments, and who may not want to
take our word for any of it.

There are three levels of checking here, and they cost very different amounts. The first
needs a laptop and twenty seconds. The second needs one GPU and an afternoon, and
downloads nothing. The third needs the public datasets and about thirty GPU-hours. You
can stop at whichever level convinces you.

---

## Level 1 — check the provenance. No GPU, no data, ~20 seconds

```bash
uv sync
uv run pytest tests/test_paper_results.py -q
```

This is the strongest cheap check we can offer, and it is worth saying exactly what it
proves and what it does not.

Every number printed in the paper is read from one of seven JSON archives in
`docs/reproduce/paper_results/`. An archive is keyed by evaluation run, and each entry
holds that run's measured metrics **together with the Hydra overrides the run was
launched with** — both for the evaluation and for the training run it scored.

These tests recompose those recorded overrides against the experiment configs shipped in
`conf/experiment/`, and assert that the two produce byte-identical configurations. So
they establish that **the configs in this repository are the ones that produced the
archived numbers** — not a cleaned-up description written afterwards. They also check
each grid is complete, that every entry is the arm its name claims, and that protocol
fields (epochs, seeds, coupling, resolution, evaluation role) are what the paper says.

They do **not** prove the archives were not fabricated. Nothing short of a trusted
execution log could, and we are not claiming otherwise. What they rule out is the far
more common failure: a paper whose tables and whose released configs have quietly drifted
apart.

The test file imports no torch and touches no dataset.

## Level 2 — rerun the synthetic experiments. One GPU, no downloads

The synthetic cohort is generated on the fly from a Gaussian copula, so there is nothing
to obtain:

```bash
uv run python scripts/paper/reproduce.py --all --dry-run   # see the commands first
uv run python scripts/paper/reproduce.py --all
```

That retrains and rescores the grid behind Table 2's synthetic block, Table 3 and the
appendix loss table: three field sizes, two geometries, two couplings, five seeds. On one
H100 a 64x64 arm is about five minutes for forty epochs plus evaluation, and the smaller
sizes are dominated by process start-up; the whole sweep is a few GPU-hours. A single
cell is cheaper still:

```bash
LOSS_MODE=l2u bash scripts/paper/run_arm.sh check 64 cylindrical ot 0
```

There is also a network-free control that needs no training at all and takes seconds — it
measures the distance from the raw prior to the data, which is the denominator every
few-step claim is implicitly measured against:

```bash
uv run python scripts/prior_control.py --side 16 --num-fields 16
```

## Level 3 — rerun everything, including the real data

`scripts/wcss/` holds the Slurm scripts the published cohorts were launched with, kept
verbatim apart from the anonymisation noted below. They are for one specific cluster and
will need editing for another, but they state the grids exactly:
`paper_tables_l2u.sbatch` (synthetic), `table5_fastmri.sbatch` (knee MRI, with
`KSPACE_CROP=64` for the main-text block), `table6_audio.sbatch` (speech, which needs
`EPOCHS=40`), and `recompute_cohorts.sh`, which launches all four with the settings that
are easy to get wrong.

The two real cohorts need their data:

* **fastMRI knee**, coronal proton-density without fat suppression. Access requires
  registration with the fastMRI project; we cannot redistribute it. `scripts/data/` builds
  the compact store this code reads, including the ESPIRiT coil combination.
* **LibriSpeech**, which is public. `scripts/data/build_stft_store.py` builds the STFT
  store.

Budget roughly thirty GPU-hours for all four cohorts on H100-class hardware, dominated by
the 320x320 knee grid.

### What a rerun will and will not match

Evaluation of a fixed checkpoint is exactly reproducible: the prior draw and the metric
projections are seeded, and rescoring a checkpoint returns the same numbers.

Training is seeded but **not** bitwise deterministic. We seed torch globally and seed the
dataloader, but we do not enable deterministic kernels, so convolution backward uses
non-deterministic atomics and cuDNN may select different algorithms between runs. A
retrained model therefore lands within the seed spread rather than on the printed digits.

This matters for how to read a rerun. Treat a reproduction as successful if the paper's
**claims** survive — the orderings, which differences have all five seeds separated, and
the step counts at which each arm reaches the block's asymptote — rather than requiring
the third decimal to agree. Where the paper reports a difference as separated, that is the
claim to test.

---

## What the archives contain

| archive | what it backs |
| --- | --- |
| `unet_eval_metrics_l2u.json` | Table 2, synthetic block; Table 3 |
| `unet_eval_metrics_l1u_cylindrical.json`, `..._l1w_cartesian.json` | appendix loss-protocol table |
| `table6_audio_metrics.json` | Table 2, speech block |
| `table5_fastmri64_metrics.json` | main-text knee MRI table, 64x64 |
| `table5_fastmri_metrics.json` | appendix knee MRI table, 320x320 |
| `unet_eval_metrics.json` | superseded: the arXiv v1 protocol, kept for the record and read by no current table |

234 entries in total. Each holds the full metric sweep for its run — pooled sliced $W_2$,
amplitude and circular-phase $W_2$, the two spatial autocorrelation gaps, the radial
spectrum gap, the amplitude-phase dependence, the angular-velocity probes, and the
function-evaluation count beside the requested step count.

`docs/reproduce/paper_results/README.md` documents each file's protocol, seeds and
hardware. Two properties are worth knowing before comparing error bars between blocks:

* The knee cohorts scored every arm at `evaluate.seed=0`, so their five seeds differ in
  **training only** and share one prior draw. Arm-to-arm comparisons there are paired, and
  the spread measures training variance alone.
* The speech cohort moved the evaluation seed with the training seed, so its seeds vary
  both and its comparisons are unpaired. Its spread is the wider quantity.

The knee 64x64 archive holds two sweeps of the same checkpoints under different step
grids. They agree to float noise on the step counts they share and differ at $k=100$,
because the sampler consumes one seeded generator down its step list and a step count
therefore draws a different prior depending on its position. The printed columns come
from the five-point sweep; the eleven-point sweep is what the reported step-to-asymptote
values are read off. A test asserts the shared prefix agrees, so this cannot drift
unnoticed.

## Anonymisation

For double-blind review, this tree has author names, affiliation and repository links
withheld, and cluster paths replaced with placeholders — the allocation identifier and
job numbers would otherwise name the group. `/path/to/project` and
`/path/to/node-scratch/JOBID/` are the placeholders; they appear in the Slurm scripts and
in the `dataset.data_dir` entries the archives record. Nothing else was altered, and the
provenance tests above pass against the substituted values, which is how we know the
substitution was consistent.

Internal working notes, agent configuration and the manuscript source are not included in
this bundle. The code, configs, archives, tests and cluster scripts are complete.
