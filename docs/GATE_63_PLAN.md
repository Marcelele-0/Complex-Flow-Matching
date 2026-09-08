# Gate #63 — Conditional Bridge Go/No-Go

Run plan for the conditional-bridge experiment. Everything in the backlog past this
point is gated on the outcome, so the goal is a decision, not a good number.

## Why this exists

The model is trained as an unconditional prior and used at inference as a conditional
reconstructor. Measured on the held-out volume (MTR_030, 512 slices, 4x Cartesian,
24 ACS, identical metric code and masks):

| Method | PSNR (dB) | SSIM | CPE (rad) |
| --- | --- | --- | --- |
| Zero-filled (untrained adjoint) | **26.06 ± 1.14** | **0.676 ± 0.04** | **0.362 ± 0.05** |
| Cylindrical flow | 24.47 ± 1.16 | 0.588 ± 0.04 | 0.405 ± 0.05 |
| Euclidean flow | 24.13 ± 1.18 | 0.530 ± 0.06 | 0.513 ± 0.07 |

Both arms lose to a plain IFFT on every metric, including the phase metric the
geometry is designed to improve.

Zero-filled floors at the other operating points, same volume and masks:

| R | ZF PSNR (dB) | ZF SSIM | ZF CPE (rad) |
| --- | --- | --- | --- |
| 4 | 25.96 | 0.672 | 0.364 |
| 8 | 22.73 | 0.512 | 0.514 |
| 12 | 21.33 | 0.449 | 0.610 |

These floors come from the exploratory sweep. The R = 8 row was re-measured on
2026-09-07 through the same path the models are scored by and reads 22.83 / 0.516 /
0.512; the results section below uses that one, since only a floor measured by the
identical code is a fair subtrahend.

## Gate criterion

Cylindrical clears zero-filled on **PSNR and CPE** at **R >= 8**.

Losing at R = 4 is acceptable and should be reported: the floor there is high and the
phase gap between aliased and clean is small (median |dtheta| = 0.20 rad, implying
4.3% Euclidean midpoint attenuation rather than the 36.33% of Proposition 1).

- **Pass** — proceed to #67 (full cohort, patient-level statistics), then #65 / #66
  (VarNet and Score-MRI trained), then decide venue.
- **Fail** — work the fallbacks below before spending cluster time. If none of them
  moves the result, the problem is not the training regime, and the project pivots to
  the unconditional synthesis track (#47) or to complex counterfactuals.

## Results — 2026-09-07, R = 8

**Verdict: the gate as stated does not pass.** Cylindrical clears zero-filled on PSNR
and SSIM but not on CPE, which is the metric the geometry exists to improve. The
fallbacks below say why, and the answer changes what to do next rather than ending the
project.

All numbers are paired per slice against a zero-filled run measured through the
identical code path, masks and seed. Held-out volume is MTR_030 (512 slices); the
model saw MTR_201 + MTR_184.

### Held-out, the comparison that counts

| Method | PSNR (dB) | SSIM | CPE (rad) |
| --- | --- | --- | --- |
| Zero-filled (untrained adjoint) | 22.83 | 0.516 | **0.512** |
| Euclidean, conditional bridge | 23.03 | 0.546 | 0.536 |
| Cylindrical, conditional bridge | **23.38** | **0.551** | 0.527 |

Against the floor, cylindrical wins PSNR on 92.6% of slices (+0.549 dB) and SSIM on
97.7% (+0.035), and loses CPE on 80.1% (+0.015 rad). The conditional bridge fixed the
direction the unconditional one had wrong — the same arm previously lost all three by
1.6 dB — but it did not fix phase.

### The ablation: cylindrical vs Euclidean, one variable

Same network, same bridge, same solver, same masks, same seed. Only the
parameterisation differs.

| Metric | Delta | Cylindrical wins on |
| --- | --- | --- |
| PSNR | **+0.342 dB** | 87.3% of slices |
| SSIM | +0.005 | 68.0% |
| CPE | **-0.009 rad** | 64.8% |

Cylindrical leads on all three. This is the paper's central claim and it came out on
the right side, though the SSIM margin is close enough to a tie to be worth stating as
one.

**Attribution.** Euclidean buys +0.208 dB over the floor, cylindrical +0.549 dB. So
roughly 62% of the amplitude gain comes from the polar parameterisation rather than
from the conditional bridge: modelling phase on its own manifold improves *magnitude*
fidelity. That is the least obvious result here and the one worth building on.

### Fallback A — the same weights on slices the model was trained on

256 strided training slices, with the zero-filled floor re-measured on those same
slices (a different volume has a different floor, so only the deltas compare).

| Metric | ZF | Cylindrical | Delta | Wins on |
| --- | --- | --- | --- | --- |
| PSNR (dB) | 21.72 | 24.14 | **+2.42** | 100.0% |
| SSIM | 0.481 | 0.605 | **+0.124** | 100.0% |
| CPE (rad) | 0.649 | 0.519 | **-0.130** | 99.6% |

On data it has seen the model beats the floor on **all three**, phase included, by
0.13 rad on 99.6% of slices. The phase capability exists and does not transfer to a
new patient.

Read against the decision rule above, this is the overfitting branch: the deficit is
not structural, so `pi * tanh` bounding, mask conditioning and network capacity are
not the binding constraint. Read *on its own* it also looks like an argument for #67 —
but the Euclidean arm's fallback A, below, withdraws that argument.

One caution against reading it as a forecast: 73.8M parameters over 1024 near-duplicate
slices can memorise almost anything, noise included. Fallback A rules out
*incapacity*; it is close to uninformative about whether the phase residual carries
learnable structure. A one-patient-vs-two learning curve settles that for ~1.5 h of
GPU, and should run before any cluster time is committed.

### The sign flips between seen and held-out — read this before trusting fallback A

Running fallback A for the Euclidean arm as well turns the earlier reading around.

| Metric | ZF | Euclidean | Cylindrical | cyl - euc | cyl wins on |
| --- | --- | --- | --- | --- | --- |
| PSNR (dB) | 21.72 | **24.93** | 24.14 | -0.789 | 0.0% |
| SSIM | 0.481 | **0.637** | 0.605 | -0.032 | 0.4% |
| CPE (rad) | 0.649 | **0.478** | 0.519 | +0.041 | 2.0% |

On slices it has seen, **Euclidean beats cylindrical on all three, on essentially every
slice**. On held-out slices the ordering reverses, also on all three. The gain each
arm keeps when it moves to a new patient:

| | seen | held-out | retained |
| --- | --- | --- | --- |
| Euclidean - ZF | +3.210 dB | +0.208 dB | 6% |
| Cylindrical - ZF | +2.421 dB | +0.549 dB | 23% |

This is an inductive-bias result, not a capacity result. The cylindrical manifold fits
the training set *worse* and transfers *better*: it is acting as a regulariser. That
is a coherent and defensible story, and arguably a more interesting one than a raw win
— but it costs two claims made earlier in this document.

**What it invalidates.** Fallback A on the cylindrical arm alone was read as "the phase
capability exists and needs more data". Euclidean shows the same phase capability on
seen slices, and more of it (0.478 vs 0.519 against a 0.649 floor). Memorising phase is
therefore not evidence for the geometry; both parameterisations do it, and the
cylindrical one does it less well.

**What it puts at risk.** A regularisation advantage is by construction largest in the
low-data regime. Scaling to 106 patients could narrow it, or reverse it, precisely
because the constraint that helps at two patients costs capacity at a hundred. The
optimistic extrapolation from fallback A does not survive this table.

**What it makes mandatory.** The one-patient-vs-two learning curve has to run for
**both** geometries, and the quantity to read is not either arm's held-out score but
the *gap between them* as a function of data. Widening gap: the geometry is a genuine
prior and #67 is worth the cluster time. Narrowing gap: cylindrical is a small-data
regulariser, the headline claim will not survive the full cohort, and the paper needs a
different framing before anything else is spent on it.

### Where the phase deficit sits

Per-slice CPE deficit against the floor, by difficulty quartile (Q1 hardest):

| Quartile | ZF PSNR | dPSNR | ZF CPE | dCPE |
| --- | --- | --- | --- | --- |
| Q1 | 21.50 | +0.376 | 0.592 | +0.0164 |
| Q2 | 22.55 | +0.480 | 0.520 | +0.0148 |
| Q3 | 23.20 | +0.668 | 0.484 | +0.0132 |
| Q4 | 24.05 | +0.673 | 0.453 | +0.0163 |

The deficit is flat (Spearman rho = -0.126) while the amplitude gain tracks slice
easiness (rho = +0.337). Regression to the conditional mean would concentrate the
phase damage where amplitude is low and SNR poor; it does not. A near-constant offset
is more consistent with a systematic phase bias, which puts fallback D back in play
despite fallback A pointing at data scale. Slices 128-255 are the one band that *wins*
on phase (-0.010), and the volume edges lose most (+0.031), which is anatomy
dependence and fits the same story.

Settling bias against blur needs a per-pixel, amplitude-stratified breakdown rather
than per-slice aggregates.

### Statistical caveat

All p-values here are Wilcoxon over 512 slices of **one** patient. Slices within a
volume are near-duplicates, so these establish that a shift is systematic within this
volume and nothing more. No patient-level claim is available from this run; that is
what #67 is for. Seen-slice numbers are a diagnostic and must never appear in a paper.

## Prerequisite — DONE

`evaluate.py` used to build the initial state as `bridge(sample_noise(...), x_alias, t)`
unconditionally. A conditionally trained model expects the state at `t = 0` to be
**exactly** `x_alias`, with no noise mixed in; scoring the new checkpoints through the
old path would have reintroduced the very mismatch this experiment removes.

Fixed by an `evaluate.bridge` key mirroring `training.bridge`:

| File | Change |
| --- | --- |
| `src/cfm/flow/__init__.py` | `BRIDGE_ENDPOINTS`, now shared by both entry points instead of living in `train.py` |
| `src/cfm/evaluate.py` | `reconstruct_batch(..., bridge_endpoint=...)`; under `aliased` the start state is `x_alias` verbatim, no noise is drawn, and `x_1` is used only for its shape. `bridge` recorded in `metrics.json` and W&B |
| `conf/evaluate/default.yaml` | `bridge: ${training.bridge}`, so one `training.bridge=aliased` override moves training and scoring together |
| `tests/test_eval/test_evaluate.py` | `TestConditionalBridge`: start state is the alias verbatim, the target cannot reach it, the path is deterministic, and the three wrong combinations raise |

Two mismatches now raise instead of reporting a number: an `aliased` checkpoint scored
at `t_start != 0.0` (the ground truth would enter the initial condition and inflate
every metric), and `aliased` without an `x_alias` to start from.

`ruff`, `mypy` clean; 496/496 tests pass. The `noise` path is asserted bit-for-bit
unchanged, so the synthesis track (#29) is unaffected.

## What is ready

| File | Change |
| --- | --- |
| `conf/training/default.yaml` | new key `bridge: "noise" \| "aliased"`, default `noise` |
| `src/cfm/train.py` | validation (hard error when `num_slices > 1`), dataset built in `reconstruction` mode, `x_0 = manifold.from_complex(batch["input"])`, log line naming the active bridge and R |
| `src/cfm/core/reconstructor.py` | `ZeroFilledReconstructor` registered in `MODELS` |
| `conf/model/zero_filled.yaml` | config for the floor baseline |

Together with the prerequisite above, this is the whole change: one key, read
identically by both entry points.

## Steps

### 1. Train both geometries, one model per operating point

A conditional model maps *this* alias distribution to clean, so R has to match
between training and scoring: an R = 4 model fed an R = 8 alias is out of
distribution and the number would say nothing about the geometry. That is the one
place this plan departs from the unconditional runs, where a single checkpoint could
be scored at any R.

Hyperparameters must match `cylindrical_mini_bench` / `euclidean_mini_bench` exactly —
25 epochs, `base_channels=96`, batch 4, lr 2e-4, seed 0 — so that **the bridge is the
only variable** and the results are directly comparable to 24.47 / 24.13.

Train on `split: train,val` (MTR_201 + MTR_184, 1024 slices). Test volume MTR_030 is
never seen.

```
uv run python -m cfm.train manifold=cylindrical training.bridge=aliased \
  dataset.split="'train,val'" dataset.acceleration=8 dataset.mask.acceleration=8 \
  training.epochs=25 training.batch_size=4 \
  logging.experiment_name=g63_cylindrical_R8
```

Repeat for `manifold=euclidean` and for R = 4 and 12. R = 8 runs first: it is the
operating point the gate is stated at, so a run cut short still answers the question.

### 2. Evaluate each checkpoint at its own R

```
uv run python -m cfm.evaluate manifold=cylindrical training.bridge=aliased \
  evaluate.t_start=0.0 evaluate.mask.acceleration=8 evaluate.num_steps=100 \
  evaluate.run_name=g63_cylindrical_R8 \
  logging.experiment_name=g63_cylindrical_R8_test
```

`evaluate.bridge` follows `training.bridge`, and `evaluate.t_start=0.0` is required
rather than merely advisable — any other value raises.

### 3. Record

Write the resulting table into #63 with the zero-filled row alongside. Whatever it
says. Done for R = 8 above.

## Fallbacks, in order of cost

### A. Evaluate on training slices — the discriminator, ~5 minutes

Score the trained model on slices it was **trained** on (`evaluate.split=train`).

- Beats ZF on training slices, loses on the held-out one → overfitting. Data scale is
  the problem, and #67 (full cohort) is the right fix.
- **Does not beat ZF even on training slices** → not a data-scale problem. Something is
  structurally wrong, and more data will not help. Stop and diagnose before spending
  cluster time.

Run this first on any failure. It is the cheapest question that changes what you do next.

### B. Per-slice distribution — ~10 minutes

Histogram the per-slice difference (model − ZF) across all 512 test slices using
`eval_records.csv`, which both arms already emit.

A bimodal distribution, or a subgroup where the model wins, is a different result from
a uniform deficit and would point at a specific failure mode (slab edges, low-signal
slices, particular anatomy). Note that the mid-volume slice already rendered was
*above* average for both methods, so the middle of the volume is not where the model
recovers.

### C. Null-space parameterisation — ~half a day

Constrain the model to write only in the null space of the sampling operator
(range-null decomposition, DDNM-style). It cannot then touch the measured lines at all.

Caveat: hard DC projection already keeps the measured lines exact, and the arms still
lose. Zero-filling writes **zeros** into the unmeasured lines; the model writes wrong
non-zero values there, which is worse. Null-space parameterisation alone does not fix
that — it improves conditioning by removing the need to reproduce what is already
known. A guarantee of "never worse than ZF" needs a residual formulation whose output
can learn to be zero.

Worth doing if the gate passes with a thin margin, or as a diagnostic if it fails.

### D. Bound the velocity and tell the model what was measured — ~a day

Two architectural gaps, cheap enough to try together:

1. `cylindrical_unet_attention.py:186` ends in a bare `Conv2d`, so the predicted
   angular velocity is unbounded while Lemma 1 bounds only the *target* to
   `|u_theta| <= pi`. The network can emit a velocity no geodesic on `S^1` ever
   requires, and the Heun step will happily integrate it. A `pi * tanh` on the phase
   channel makes the architecture honour the lemma the paper proves, which is also a
   cleaner story than a numerical coincidence.
2. The model is never told which k-space lines were measured. It has to infer the
   sampling pattern from the aliasing, which the DC projection then silently corrects
   for. Feeding the mask (and the alias itself) as extra input channels turns a blind
   inpainting problem into a conditioned one.

These are the two suspects that survive if fallback A says the deficit is *not* a
data-scale problem. Do them in that order: the bound is a three-line change, the
conditioning changes `manifold.state_channels` and therefore every checkpoint.

### E. Full cohort — #67, cluster time

Only after A has ruled out a structural problem, and after D if it has not. This is where #20 (streaming HDF5,
106 patients) and #22 (SLURM DDP on A100) finally get used, and the only route to a
patient-level statistic. Do not spend this until A and B have been read.

## Standing caveat

The test set is **one patient**. This gate answers "did the bridge fix the direction",
not "what is the result". No patient-level statistic is possible here, so report no
significance tests from this run. That is what #67 is for.
