# Reproducing the paper's numbers

One script per published result. Each recomputes what the paper prints and
compares it against the paper, not against a stored output — `expected.py`
quotes every value from `paper/ICLR Main/`, naming the file and the sentence.

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
| `table4_patch_seams.py` | Table 4 / Section 5.4, patch seams | nothing | ~40 s |

Both generate their own targets, so they reproduce from a bare checkout on a CPU.

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

## Still to convert

`scripts/coupling_gate.py` and `scripts/coupling_dimension.py` still fuse
measurement with printing, so Table 4's spiral rows and Section 5.3's cost curve
have no reproduction script yet. `scripts/bridge_angular_velocity.py` shows the
shape the others need: the measurement moved to
`cyfm.experiments.BridgeGeometryExperiment`, which returns its numbers, and the
script kept its path and its exact output.
