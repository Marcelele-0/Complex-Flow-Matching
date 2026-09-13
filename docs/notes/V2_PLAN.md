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

### P0-1. fastMRI gate: batch size, epoch time, and a go/no-go for ICLR
Issue #76. Every paper result is synthetic, and until one fastMRI epoch is timed nothing
on fastMRI can be scheduled and the venue cannot be chosen (ICLR 2026-09-25 against
ICML). It runs on the validation store alone (`data/knee_pd/val.h5`, P0-2): no raw data,
no NYU links, no ESPIRiT.

1. The largest batch at 320x320 that fits both geometries, on an H100 and on a 16 GB
   card. Every arm of the table shares one batch, so the hungrier geometry binds.
2. One epoch of `+experiment=table5_fastmri`: seconds per epoch, seconds to the first
   batch, peak VRAM.
3. Evaluation from `dataset.role=holdout` writes a metrics.json in the training
   transform's domain, with the raw cohort absent.
4. Amplitude and phase panels, and the mean phase step between neighbouring pixels inside
   the anatomy against the pi/2 that scrambled phase gives. A ratio near 1 means the coil
   combination is wrong and everything downstream is meaningless.

One GPU job does all four and ends by printing the lines for this section:

    sbatch scripts/wcss/gate_fastmri.sbatch

**Go/no-go rule.** Table 5 needs 20 flow runs (two geometries x two couplings x 5 seeds)
of 40 epochs. One run costs about `epoch_seconds x 40 / 3600` GPU-hours on the store it
trains on. `table5_fastmri` trains on the `fit` role of val.h5 today, which is what the
gate times; moving Table 5 to the train store multiplies that by ~6.5 (5324 slices
against ~820). Go for ICLR if one run fits well inside `lem-gpu-short`'s 3-day wall and
the 20 runs plus their evaluations fit the remaining GPU budget (~7.5k h) before the
deadline.

**Results** (job 5883147, 2026-09-13, one H100):

    max_batch=84  max_power_of_two=64  capacity_gib=93.1   both geometries alike
    epoch_seconds=8.3  first_batch_seconds=1.6  peak_vram_gib=70.77  batches=13  batch_size=64
    phase_step_median=0.045  noise_reference=1.571  ratio=0.03  volumes=93

The batch is **64**, not the provisional 32, and both geometries probe identically, so
neither binds. The phase check passes with room to spare: a ratio of 0.03 against the
1.571 that scrambled phase gives means the coil combination is right and the phase
carries structure, which is the premise of the whole paper.

**GO for ICLR.** One run on the `fit` role of val.h5 is 8.3 x 40 / 3600 = 0.09 GPU-h.
The train store is 5324 slices against 814, so x6.5: ~54 s/epoch, 36 minutes and
0.60 GPU-h per run, far inside `lem-gpu-short`'s 3-day wall. The 20 flow runs cost
~12 GPU-h out of ~7340 remaining, i.e. 0.16% of the budget. Compute is not the
constraint on this table; the missing arms are (P1-3 diffusion baseline, P2-2
phase-free floor).

Not measured, and deliberately not chased: the batch for a 16 GB card. The probe's
memory cap raises `ValueError: Expected a torch.device with a specified index` because
`set_per_process_memory_fraction` was passed the string `cuda`. The one-line fix is in
this commit, but the probe was not re-run: every arm of Table 5 runs on WCSS, so the
local number would not inform anything.

**Done:** metrics.json at `outputs/evaluate/gate_cylindrical_ot_b64_j5883147_eval/`,
panels at `outputs/gate/5883147/knee_panels.png`.
**Cost:** one GPU job, 2 minutes of an H100. **Blocks:** P0-3, P1-5 -- now unblocked.

### P0-2. The data pipeline: raw fastMRI -> a compact CORPD store
**The protocol changed from brain AXT2 to knee CORPD_FBK**; the sections above and below
have not been rewritten for it. Knee coronal proton-density without fat suppression is
one acquisition on one hardware setup -- every such volume is 15-coil in one of two
nearly identical matrices -- where brain AXT2 mixes eight coil counts, and without fat
suppression the fat-water chemical shift leaves the phase structure this paper is about.
It is also the split the score-based-prior literature trains on. The protocol of record
is `conf/dataset/fastmri_knee_pd.yaml`.

Per volume: ESPIRiT maps -> adjoint SENSE coil combination -> centre crop to 320x320
(removes the 2x readout oversampling, unifies 640x368 with 640x372) -> the central 11
slices -> complex64, at 0.82 MB per slice. Selection is checked in code from the HDF5
`acquisition` attribute, never from a filename. Normalisation is not baked in: the
manifold transform normalises each field by its own peak, as it does for the synthetic
cohorts.
On WCSS this runs on **CPU partitions** (`bem2-cpu-normal`, 21-day limit): downloading
and ESPIRiT are not work for an H100, and the GPU grant (~7.5k h left) is the scarce
resource, while CPU has ~45k h.

**Done:** `scripts/data/build_knee_pd_store.py` (selection, ESPIRiT, combination, crop,
atomic write, manifest with rejection counts) and `KneeStoreDataset`, which reads the
store and nothing else. Slices stream into a resizable HDF5 as each volume is combined,
so memory is one volume rather than the split. `--merge` joins the per-archive stores the
cluster produces, refusing parts built under different protocols or a volume processed
twice. `scripts/wcss/build_knee_store.sbatch` runs one archive per array task on
`bem2-cpu-normal`. Validation store built: 1023 slices, 93 volumes, 838 MB, seven volumes
rejected on the matrix rule.

**Remaining:** the ESPIRiT cost per volume on a CPU node is still unmeasured, which is
what decides whether 484 volumes is an hour or a day -- the builder now prints
`espirit_seconds_per_volume`, so one archive answers it. Node-local scratch on
`bem2-cpu-normal` is unverified (the 7 TB figure is documented for GPU nodes). Then the
train store itself, which needs P0-2a. Copy the existing `val.h5` to
`$PDDIR/CyFM/data/knee_pd/` rather than rebuilding it there.
**Blocks:** P0-3, P1-5.

#### P0-2a. Fresh fastMRI links (user action)
The links in `.env` (2 mini, 19 full archives) return 403; NYU signs them for about two
weeks and ours are from 2026-09-05. Needed for the train split: knee multicoil train is
5 batches and ~917 GB, of which CORPD_FBK is about half. NYU does no server-side
filtering, so the whole tar comes down and the acquisition attribute filters it as files
land.

Refreshed URLs go back into `FASTMRI_FULL_URLS` in `.env`, which stays the one place the
team maintains them; `scripts/wcss/build_knee_store.sbatch` reads that variable and
indexes it, one archive per array task. Verify before queueing anything:

    uv run python scripts/data/fastmri_urls.py --check

**Done:** every link answers 200 to a HEAD request.

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
