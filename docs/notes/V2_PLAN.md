# v2 plan

Written 2026-09-12, after the issue tracker and the project board were cleared.
Every item is an issue to file. Priorities follow one rule, so that P0 keeps meaning
something:

- **P0** — the submission cannot exist without it, or it blocks other tasks.
- **P1** — the venue version is materially weaker without it.
- **P2** — strengthens the paper, first to be cut when the schedule slips.
- **P3** — after the submission.

## State

arXiv v1 is content-complete (Overleaf `ICLR Main`, `b8a9cae`; tables generated from
the 5-seed archives; main text ends on page 9). Established: cylinder + joint OT beats
the best Cartesian arm at every k <= 8 at 16x16, 32x32 and 64x64, all 5 seeds separated
(exact two-sided Mann-Whitney p = 0.008); no significant difference at k = 100; joint OT
helps at every step count; the Factorized Coupling Trap. Stated weaknesses: synthetic
data only; the Cartesian arm has the lower circular phase W2 at k = 8 on 32x32 and
64x64; the convergence tie is a non-significant difference at n = 5, not an equivalence.

Assets that shorten the path: 120 trained checkpoints (epoch 40) on WCSS, so any
re-evaluation is minutes; 267 AXT2 brain volumes locally (4242 slices, 123 GB);
experiment configs with a `paper:` block, `reproduce.py`, archives with per-run Hydra
provenance and tests that recompose every recorded run.

## Deadline and the decision point

ICLR 2027: abstract 2026-09-18, paper 2026-09-25. ICML 2027 (~late January) is the
fallback. **P0-1 decides which**, and nothing else should be scheduled against a date
until it has run. Registering the ICLR abstract is cheap and commits us to nothing.

## Critical path (P0)

### P0-1. fastMRI gate: loader, epoch time, and how 64x64 is reached
Validate the fastMRI path on this branch with a short train/evaluate pass on the local
AXT2 subset; record dataset-construction time (ESPIRiT runs eagerly), epoch time, VRAM.
Decide and document how a 640x320 acquisition becomes 64x64: **cropping the centre of
k-space** (the whole anatomy at low resolution, the physically natural choice, and the
one that keeps the phase structure) against cropping the image (a patch of the middle).
**Done:** a fastMRI metrics.json exists, the timings are in this file, the crop rule is
implemented and documented, and a go/no-go for ICLR is written down.
**Cost:** half a day, local. **Blocks:** P0-2, P0-3, P1-5.

### P0-2. The data pipeline: raw fastMRI -> a compact 64x64 AXT2 dataset
Brain AXT2 only: one contrast, one orientation, 267 subjects locally and ~2300 in the
full cohort, which is the largest single-contrast group and the one the complex-MRI
generation literature uses. Mixing knee with brain, or contrasts with each other, makes
the target distribution a mixture, which is defensible in a reconstruction paper and not
in a generative one.
Steps: ESPIRiT sensitivity maps -> coil combination -> centre-of-k-space crop to 64x64 ->
one HDF5 per split plus a manifest (subject, volume, slice index, acquisition, checksum)
-> patient-level split recorded in the manifest, not computed at load time.
Output is ~32 KB per slice: the whole AXT2 cohort is **under 1 GB**, so training never
touches lustre and the set fits in RAM. Raw archives can be deleted after each batch.
On WCSS this runs on **CPU partitions** (`bem2-cpu-normal`, 21-day limit): downloading
and ESPIRiT are not work for an H100, and the GPU grant (~7.5k h left) is the scarce
resource, while CPU has ~45k h.
**Done:** the manifest and the HDF5 exist for the local 267 volumes, a test asserts the
split is patient-disjoint, and the pipeline reruns from raw with one command.
**Cost:** a day of work; download time depends on P0-2a. **Blocks:** P0-3, P1-5.

#### P0-2a. Fresh fastMRI links (user action)
The links in `.env` (2 mini, 19 full archives) return 403; NYU signs them for about two
weeks. Needed only to go beyond the local 267 volumes. Each brain train batch is ~180 GB
raw, of which AXT2 is ~120 GB (~260 volumes, ~4k slices), and leaves ~130 MB after
preprocessing, so batches are downloaded one at a time until P1-5 says to stop.

### P0-3. Table 5: fastMRI AXT2 at 64x64
Six rows, the same protocol as Table 2, 5 seeds: CyFM with joint OT and with independent
pairing; Cartesian with both couplings; the complex diffusion baseline (P1-3); the
phase-free floor (P2-1). Archive with provenance, LaTeX generated, not typed.
**Done:** the archive passes the same integrity tests as the synthetic ones and the table
is generated. **Cost:** ~30 runs, ~3 h on WCSS. **Depends on:** P0-1, P0-2.

### P0-4. Anonymised submission build
Drop `\iclrfinalcopy` and `\lhead{Preprint}`, anonymise the repository link, add the
required AI-use statement and a reproducibility statement, re-check the 9-page limit
after the new tables. **Cost:** an afternoon. **Depends on:** everything that adds a
table or a figure. Last item before submitting.

## P1 — a materially stronger paper

1. **Table 6: audio STFT.** Nothing audio exists in the repo. A dataset of complex STFT
   frames (64 bins x 64 frames), the same six rows as Table 5, plus two domain metrics
   (spectral distance and a phase-coherence measure). Without a second domain the paper
   reads as medical imaging at the wrong venue. **Cost:** 1-2 days plus ~30 runs.
   Open decision: which corpus (a clean-speech subset with a permissive licence).
2. **Table 7: steps to a common quality target.** For each cell, E* = the mean asymptote
   of the *worse* of the two arms, so the arm with the better ceiling is not rewarded for
   it; report the per-seed k at which each geometry first reaches E*, and the ratio.
   Needs the 120 stored checkpoints re-evaluated on a denser grid
   (k = 1, 2, 4, 8, 16, 24, 32, 48, 64, 100): today nothing is measured between 8 and 100,
   which is exactly where the curves cross. Evaluation only. **Cost:** ~15 min on WCSS.
   Also supplies the curves for Figure 3.
3. **Restore the diffusion baseline.** `complex_diffusion.py` and its predictor-corrector
   solver were dropped in `57a1cb5`; recover from `9027560`, re-test, and run it in both
   domains. Report it at matched NFE *and* in its own many-step regime, and say plainly
   that it is not a few-step method -- otherwise the comparison is a strawman.
   **Cost:** half a day plus runs.
4. **Variance decomposition, before any new table.** Re-evaluate a few checkpoints with
   five different `evaluate.seed` values at 64 and at 512 fields, to separate estimator
   noise from training-seed variance (the Cartesian asymptote at 32x32 spans 0.021-0.045
   across seeds and we do not know which source that is). Fixes the evaluation size for
   every table that follows. **Cost:** minutes.
5. **Learning curve against dataset size** (25 / 50 / 100% of the AXT2 slices), reporting
   both the held-out metric and whether the few-step advantage is stable. This answers
   "your subset is too small" with data, and tells us when to stop downloading batches.
   **Cost:** ~15 runs. **Depends on:** P0-2.
6. **Figures.** (1) chord against arc, in TikZ; (2) the angular-velocity tail, log-log,
   from the same 10^6 network-free bridges as Table 1; (3) error against steps at 64x64
   with per-seed bands, from P1-2's dense grid; (4) qualitative samples at k = 1, 2, 4,
   100 for both geometries with a cyclic colormap for phase; (5) optional: the
   transport-cost saving against dimension beside the trained benefit. Generated by a
   script from the archives, vector output. **Cost:** a day.

## P2 — strengthens, cut first

1. **Phase-free floor baseline:** amplitude-only generation with zero phase (MRI) and
   Griffin-Lim (audio). Cheap, and it is exactly the practice the paper argues against.
2. **x1-parametrisation and a pi*tanh bound.** The trained model leaves the target's
   range (|v_theta| = 5.44 at k = 2 against a target bounded by pi), and k = 2 is the
   cylinder's worst cell. Both geometries get the same parametrisation or it is unfair.
3. **Phase error stratified by amplitude:** explains the one metric we lose. Evaluation
   only.
4. **Spatial metrics** beside the pooled sliced W2, which is blind to arrangement by
   construction: lag-1 correlation and a radial power-spectrum distance.
5. **Is the data invariant to a global phase rotation?** An hour of measurement that
   decides whether U(1)-equivariance is worth any work.
6. **A distilled Cartesian baseline** (reflow or consistency), because the paper claims
   few-step generation "without distillation".
7. **One resolution above 64** (128x128), against "this is a toy size".

## P3 — after the submission

PyPI release of `torch-cfmri` and a Zenodo DOI for the archives (the data itself cannot
be redistributed; the pipeline can). Forward-backward cylindrical SDE, pixel-wise
uncertainty maps, DSB / IPF.

## Order of execution

1. **Today, no training, no new data:** P1-4 (variance), P1-2 (dense grid and Table 7),
   P2-3, P2-5. All run on the 120 stored checkpoints, minutes each.
2. **In parallel:** P0-1 on the local AXT2 subset; ask NYU for links (P0-2a).
3. **Then:** P0-2 pipeline -> P1-5 learning curve -> P0-3 Table 5.
4. **Then:** P1-1 audio with P1-3 baseline; P1-6 figures whenever there is a gap.
5. **Last:** P0-4.

## Risks

- **Links do not arrive in time.** Mitigation: the local 267 volumes carry P0-1, P0-2 and
  a first Table 5; the learning curve says whether more data changes the conclusion.
- **ESPIRiT is slower than remembered** (~15 min per 150 volumes when measured at 320x320).
  Mitigation: it is a CPU array job, one task per volume, and it runs once.
- **Audio is the largest unknown** (new dataset, new metrics, no code). It is P1, not P0,
  precisely so that it cannot sink the submission.
- **The diffusion baseline is stale code.** Mitigation: it is restored behind its own
  tests before it is quoted in a table.
