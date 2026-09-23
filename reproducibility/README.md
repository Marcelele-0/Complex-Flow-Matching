# Reproducing the paper's numbers

One script per published result. Each recomputes what the paper prints and
compares it against the paper, not against a stored output — `expected.py`
quotes every value from the manuscript, naming the file and the sentence it
came from. The LaTeX itself is not in this repository; the quotes are the record.

```bash
uv run python -m reproducibility.run_all --network-free   # minutes, no data needed
uv run python -m reproducibility.run_all                  # everything available here
uv run python -m reproducibility.table1_bridge_geometry   # one result
```

Run from a checkout. There is deliberately no installed console script: this
folder is part of the repository, not of the `cyfm` wheel.

## Three outcomes, and why they are not two

| | meaning | exit |
| --- | --- | --- |
| `PASS` | the number matches the paper | 0 |
| `FAIL` | the number was computed and **disagrees** | ≠ 0 |
| `MISSING INPUT` | there was **nothing to compute it from** | ≠ 0 |

A reviewer has to be able to tell the last two apart. `FAIL` is a result that
does not hold; `MISSING INPUT` is a result this repository cannot show you. Both
are non-zero, and a script that cannot find its input prints what to run to
produce it rather than a quiet "OK".

## What runs today

| script | result | needs | runtime |
| --- | --- | --- | --- |
| `table1_bridge_geometry.py` | Table 1, analytical path geometry | nothing for the synthetic block | ~20 s |
| `table2_field_synthesis.py` | Table 2, 64x64 synthetic | the archived evaluations | ~1 s |
| `table3_unet_scaling.py` | Table 3, 16x16 and 32x32 | the archived evaluations | ~1 s |
| `section53_ot_cost.py` | Section 5.3, OT cost saving vs field size | nothing | ~10 s |
| `table4_spiral.py` | Table 4, amplitude/phase factorisation | nothing | ~4 min |
| `table4_patch_seams.py` | Table 4 / Section 5.4, patch seams | nothing | ~40 s |
| `table5_knee_mri.py` | Table 5, knee MRI 64x64 | the archived evaluations | ~1 s |

All five run on a CPU from a bare checkout. Two generate their own targets; three
re-derive their numbers from `docs/reproduce/paper_results/`, which ships with
the repository. Averaging over seeds is done here rather than read from a stored
aggregate, so what is compared is the per-run evaluations themselves.

This asks a different question from `tests/test_paper_results.py`. That test
checks every archived run's Hydra overrides against the experiment config it
claims to come from — that the runs are the ones the paper says they are. These
check that the numbers inside those runs are the numbers the paper prints.

## Known discrepancies

These are findings, not bugs, and they are recorded here rather than smoothed
over.

**Section 5.4's seam correlation does not reproduce as a digit.** The paper says
the lag-1 correlation across patch seams "falls from 0.979 to 0.044". At the seed
the shipped command uses it is **−0.008**. Neither number is wrong: across eight
seeds the statistic ranges over `[−0.008, +0.055]` with mean +0.017 and standard
deviation 0.023, so 0.044 is an ordinary draw and so is −0.008. The claim — that
seam correlation collapses from 0.979 to approximately zero — holds under every
seed, and `table4_patch_seams.py` checks that separately from the digit. The
issue is presentational: the sentence's other two numbers (0.000 and 0.979) are
stable to three decimals, which invites a reader to treat all three alike.
Run `--seeds 8` to see the spread.

**Table 5's numbers depend on which other step counts were swept.** The knee
archive carries every arm twice, as `dense_t5c64_*` and `p5_t5c64_*`. They score
the same checkpoint on the same cohort with the same seed, and differ only in
`evaluate.nfe` — eleven step counts against five. Their `k=1..8` columns agree to
float noise; their `k=100` columns do not, 0.0631 against 0.0722 for
cylindrical+OT. The evaluation sweep consumes one seeded generator across step
counts, so a row depends on which counts preceded it: `k=100` is fifth in one
list and eleventh in the other, and draws a different prior batch. The paper
prints the five-count evaluation, which matches its five columns, and that is
what `table5_knee_mri.py` checks. Nothing here is wrong, but a row is not
reproducible from its own `k` alone — the whole `nfe` list is part of its
provenance, and it is recorded in the archive.

**Two `expected:` blocks in `conf/experiment/` are stale.** They were written on
2026-09-11 and `13a1124` (09-17) changed what the probes print without updating
them. `sec53_ot_cost.yaml` lists `64x64 3.1%` where the code and *the paper* both
say 2.8%; `table4_factorized.yaml` lists the seam number discussed above. The
paper is correct in both cases; the config comments are not.

## Not reproducible from this repository

Documented, not fixed — closing either needs cluster time, not a refactor.

**The speech results have no archive.** `conf/experiment/table6_audio.yaml`
carries no `paper:` block, is absent from `reproduce.py`'s `PAPER_ORDER`, and has
no JSON under `docs/reproduce/paper_results/`. The speech block of Table 2 in the
main text therefore cannot be re-derived from a checkout. This is the largest
reproducibility hole in the project.

**The knee archives carry cluster paths.** `table5_fastmri64_metrics.json` records
provenance including `/lustre/pd03/...` and `/mnt/lscratch/slurm/...`. Those
travel with any public artefact. The archives are frozen input to
`tests/test_paper_results.py`, so rewriting them is an editorial decision that
requires re-deriving the provenance test.

## How the network-free measurements are reached

Each lives in `cyfm/experiments/` as a `BaseExperiment` subclass that returns an
`ExperimentResult` and prints nothing; a `render()` beside it produces the text
its script has always printed. The scripts in `scripts/` keep their paths, which
`conf/experiment/*.yaml` names and `tests/test_experiments.py` asserts, and their
output is unchanged byte for byte.

That split is what lets a number be checked here without re-printing a report,
and it is the condition for rendering these tables the way the trained arms are
rendered. One section of `scripts/coupling_gate.py` is deliberately left behind:
its batch-size diagnostic is not printed by the paper.
