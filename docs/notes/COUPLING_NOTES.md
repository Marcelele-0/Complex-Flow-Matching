# The coupling is not the sort: two gates on synthetic data

Companion to `GEOMETRY_NOTES.md`. Everything here is measured on an analytic
target with no MRI cohort involved, which is the point: on images the two
geometries differ in loss, in input representation and in prior, and those three
differences invalidate any comparison between them. On single complex numbers
all three can be removed by construction, so what is left is the geometry, the
coupling and the integrator.

Recorded 2026-09-09/10. Raw console output in `docs/reproduce/gate_artefacts/`.

---

## How to reproduce

```bash
# Gate A -- is the factorised coupling a transport plan at all?
uv run python scripts/coupling_gate.py --samples 1024 --seeds 8 --plot gate_a.png

# Gate B -- what the geometry and the coupling are worth
uv run python scripts/coupling_gate_b.py --train-steps 8000 --eval-samples 2048 --seed 0
uv run python scripts/coupling_gate_b.py --train-steps 8000 --eval-samples 2048 --seed 1
uv run python scripts/coupling_gate_b.py --train-steps 8000 --eval-samples 2048 --seed 2

# the rho sweep with the Theorem 3 mechanism probes
uv run python scripts/coupling_gate_b.py --train-steps 8000 --eval-samples 2048 \
    --rho 0.0 0.2 0.4 0.6 0.8 1.0 --seeds 3 --seed 0
```

Every draw is seeded through an explicit `torch.Generator`, so a rerun at the
same `--seed` reproduces the tables exactly. No new dependencies: `scipy` and
`matplotlib` were already declared in `pyproject.toml`.

**Integration status.** The two gate scripts drive the analytic sampler
(`src/cfm/data/synthetic.py`) directly, and Gate B trains its own small MLP
(`ToyVelocityField`) with the production bridges and ODE solvers at `H = W = 1`.
Everything from section 3 on goes through the production pipeline instead: the
toy is registered as the datasets `cylinder_toy_iid` and `cylinder_toy_field`
(`conf/dataset/`), trained with `cfm.train` and scored with `cfm.evaluate`.

---

## 1. Gate A: the factorised coupling is not a coupling

### Statement

Section 5.1 of the mathematics note factorises the transport cost across
amplitude and phase and solves the two one-dimensional problems separately. That
is a plan between the joint distributions only when both are product measures:

```
min_pi Int (c_A + c_theta) dpi  =  min_{pi_A} Int c_A dpi_A + min_{pi_theta} Int c_theta dpi_theta
```

holds iff `p(A, theta) = p(A) p(theta)` on both sides. Otherwise the left side is
strictly larger and the factorised value is a lower bound no feasible plan
attains.

### Measured

`scripts/coupling_gate.py`, n = 1024, 8 seeds, no network anywhere.

| structure | rho | r_cl target | r_cl coupled | KS(A) | KS(theta) | W2 to target | W2 floor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| spiral | 0.00 | 0.0242 | 0.0417 | 0.00000 | 0.00000 | 0.0727 | 0.1060 |
| spiral | 0.50 | 0.3104 | 0.0415 | 0.00000 | 0.00000 | 0.1670 | 0.1059 |
| spiral | 1.00 | 0.8670 | 0.0396 | 0.00000 | 0.00000 | 0.3660 | 0.1051 |
| cardioid | 0.00 | 0.0242 | 0.0417 | 0.00000 | 0.00000 | 0.0727 | 0.1060 |
| cardioid | 0.50 | 0.4401 | 0.0386 | 0.00000 | 0.00000 | 0.1803 | 0.1025 |
| cardioid | 1.00 | 0.9255 | 0.0394 | 0.00000 | 0.00000 | 0.3672 | 0.0859 |

Both marginals of the coupled cloud are **bit-identical** to the target's -- the
same multiset, reordered -- so every marginal diagnostic passes while the joint
has been replaced. The circular-linear correlation of the coupled cloud is ~0.04
at every rho, against a target running to 0.87 and 0.93.

Damage against batch size (spiral, 8 seeds):

| rho | n = 64 | n = 256 | n = 1024 | n = 4096 |
| --- | --- | --- | --- | --- |
| 0.00 | 0.2066 | 0.1322 | 0.0727 | 0.0437 |
| 0.50 | 0.2426 | 0.1818 | 0.1670 | 0.1557 |
| 1.00 | 0.3842 | 0.3599 | 0.3660 | 0.3644 |

Falling with `n` is sampling noise; a plateau is a bias. At `rho > 0` it
plateaus: the coupling converges, and it converges to the product of the
marginals.

Transport cost, n = 1024:

| structure | rho | factorised | joint (exact) | deficit |
| --- | --- | --- | --- | --- |
| spiral | 0.00 | 0.1996 | 0.2055 | 0.0059 |
| spiral | 0.50 | 0.1999 | 0.2248 | 0.0249 |
| spiral | 1.00 | 0.2001 | 0.3845 | 0.1845 |
| cardioid | 0.50 | 0.2001 | 0.2323 | 0.0322 |
| cardioid | 1.00 | 0.1997 | 0.3912 | 0.1915 |

The factorised cost is **invariant in rho** while the true optimum rises. A
number that beats the optimum is not a better solution; it is the signature of an
infeasible one.

`docs/reproduce/gate_artefacts/gate_a_factorised_coupling.png` shows it directly: the three
panels of the coupled cloud are indistinguishable across rho.

### Consequence

`src/cfm/flow/coupling.py` must be a **batch-level joint assignment** on the
cylinder cost, not a factorised sort. Measured: 0.06-0.08 ms at batch 16-64,
25 ms at n = 1024. The `O(n log n)` headline buys nothing, because Delon's
algorithm is a 1D circular problem that does not arise at batch level (the cost
between two images is a sum over `H x W` coefficients) and per-coefficient is
exactly the variant that breaks.

There is therefore no "coupling gap" to defend with a Schrodinger-bridge
paragraph. The construction was never a coupling of the joint measures.

---

## 2. Gate B: the coupling and the geometry are separate effects

### Setup

`scripts/coupling_gate_b.py`. Identical across arms by construction: prior
(one circularly symmetric complex Gaussian), input encoding
`(re, im, m, cos phi, sin phi, t)`, width, depth, parameter count (34 178) and
loss (unweighted regression of the bridge's target velocity). Only the bridge,
the coupling and the integrator vary. The bridges and solvers are the production
classes at `H = W = 1`.

### Measured -- straightness, 3 seeds

Normalised regression residual; because both bridges carry a conditional velocity
constant along the path, this *is* the rectified-flow straightness statistic.

| rho | euc / indep | euc / OT | cyl / indep | cyl / OT |
| --- | --- | --- | --- | --- |
| 0.00 | 0.6969 ±0.0194 | 0.0473 ±0.0107 | 0.9001 ±0.0010 | 0.0420 ±0.0071 |
| 0.50 | 0.6905 ±0.0259 | 0.0413 ±0.0058 | 0.8846 ±0.0106 | 0.0570 ±0.0381 |
| 1.00 | 0.5841 ±0.0252 | 0.0229 ±0.0130 | 0.7913 ±0.0113 | 0.0395 ±0.0139 |

The OT coupling buys 15-26x in both geometries, with spreads an order of
magnitude below the effect.

### Measured -- one solver step, naive coupling, 3 seeds

Values far above the estimator floor, so the comparison is real.

| rho | euclidean | cylindrical | ratio |
| --- | --- | --- | --- |
| 0.0 | 1.1402 | 0.3198 | 3.57x |
| 0.5 | 1.0818 | 0.3714 | 2.91x |
| 1.0 | 1.4779 | 0.3739 | 3.95x |

Per seed, with no overlap: euclidean `[1.303, 1.025, 1.093]` against cylindrical
`[0.318, 0.314, 0.327]` at rho = 0, and `[1.448, 1.422, 1.564]` against
`[0.361, 0.387, 0.373]` at rho = 1.

### Measured -- price of one step under OT, paired within each model

`W2(n=1) - W2(n=100)` for one trained model, same reference draw and same eval
prior, so the estimator's finite-sample bias is common to both terms and cancels.
This is the comparison that works; the absolute head-to-head does not.

| rho | euclidean / OT | cylindrical / OT |
| --- | --- | --- |
| 0.0 | `[+.0128, +.0074, +.0133]` mean **+0.0112** | `[+.0074, +.0100, +.0094]` mean **+0.0089** |
| 0.5 | `[+.0234, +.0208, +.0158]` mean **+0.0200** | `[+.0085, +.0094, +.0168]` mean **+0.0116** |
| 1.0 | `[+.0403, +.0398, +.0746]` mean **+0.0516** | `[-.0032, +.0299, -.0227]` mean **+0.0013** |

Tied at rho = 0; near-separated at rho = 0.5 (one pair overlaps); **completely
separated at rho = 1**, where the smallest euclidean penalty (0.0398) exceeds the
largest cylindrical one (0.0299).

### The caveat that blocks publication of the absolute numbers

The exact-assignment W2 estimator at n = 2048 has a **floor of 0.103-0.108**
(measured as W2 between two independent target draws; 0.063-0.066 at n = 4096).
Every OT arm scores 0.085-0.112, at or below that floor. Absolute comparisons
between OT arms are therefore measuring the estimator, not the models, in both
directions -- an earlier reading of these tables that had euclidean winning at
rho >= 0.5 did not survive three seeds and should not be repeated. Only the
paired within-model difference above is readable.

### Open

- A denser rho grid with the Theorem 3 mechanism probes (smallest amplitude
  visited, peak induced angular velocity `|theta_dot| = |(x v_y - y v_x)| / A^2`
  against the cylinder's `|u_phi| <= pi` bound) is running; monotonicity in rho
  is currently a three-point observation, not a measurement.
- Sliced or entropic Wasserstein would lower the floor enough to compare arms
  absolutely and would tighten rho = 0.5.
- The unweighted loss is not scale-matched across geometries: on the cylinder
  `u_phi` spans `[-pi, pi]` while `u_m` is `O(1)`, so the phase term carries
  roughly ten times the scale and dominates the gradient. That is intrinsic to
  `ds^2 = dA^2 + dtheta^2` rather than a bug, and it is the same "why this
  metric" question a reviewer will ask. A loss-side phase weight is untested.
- Making the toy a registered dataset would let the shipped models and the
  Hydra entry points run on data with a known rho. It needs one decision: what
  shape a "toy image" is, since `BaseComplexDataset` and the U-Nets expect
  `[B, C, H, W]` with `H, W` a multiple of 16, and the current sampler emits
  scalars.

---

## 3. A spatially correlated prior

### Why

At field scale a white prior drowns minibatch OT in concentration of measure: on
64x64 fields the coupling reduces transport cost by 2.9%, and trained models are
indistinguishable with and without it. Every coefficient of a white prior is an
independent draw, so two samples differ in as many directions as there are
pixels. Smoothing the prior should lower its effective dimension.

### Construction

`manifold.spatial_correlation` (pixels), available on both geometries. The
latents are smoothed and renormalised before the modulus (`Phi(z_a)`) and phase
(`2 pi Phi(z_p)`) are built from them, so the pointwise law is the white prior's
and only the spatial dependence is new: measured modulus standard deviation
0.288-0.294 against 0.289 for uniform. The Euclidean arm replays the cylindrical
correlated RNG stream, so one seed gives both geometries the same smooth field
and the prior is not a confound between them.

### A defect found and fixed on the way

The first smoother convolved in space with circular padding. When the kernel is
wider than the field it wraps onto itself, one input pixel enters through several
taps, and normalising by the kernel's L2 norm no longer gives unit variance.
Measured on 64x64 fields: variance 1.057 at length 16 and 2.628 at length 32, the
latter turning the uniform modulus into one with standard deviation 0.361. It now
filters in the frequency domain and normalises by the root mean square of the
transfer function, which is exact at any length. Earlier artefacts are
unaffected: `cylinder_toy_field` was only ever used at lengths 4 and 6 on 64x64,
where the kernel (25 and 37 taps) is narrower than the field.

### Measured, network-free

`cylinder_toy_field`, correlation length 4, 64x64, rho = 0.5. Prior drawn by the
manifold itself.

| batch | prior correlation | OT cost reduction | mean abs u_phi, independent | mean abs u_phi, OT |
| --- | --- | --- | --- | --- |
| 64 | white | 3.0% | 1.572 | 1.541 |
| 64 | 4 | 17.4% | 1.589 | 1.402 |
| 64 | 16 | 23.2% | 1.571 | 1.318 |
| 256 | white | 3.4% | 1.571 | 1.535 |
| 256 | 4 | 19.8% | 1.572 | 1.363 |
| 256 | 16 | 24.4% | 1.563 | 1.300 |

For independent uniform phases the mean absolute angular displacement is
`pi / 2 = 1.571`.

### Reading

The coupling partly escapes concentration: from 3% to 23-24%, the level a
64-dimensional problem reaches. The angular target does not follow. Under OT its
mean magnitude falls only from 1.57 to 1.30, about 83% of its uninformative value,
so the wrap still dominates. Whether that is enough to make phase learnable is a
question for training, not for this table:

```bash
bash scripts/sweeps/smooth_prior.sh /tmp/logs_sp
```

### Trained, 2x2 x prior

Same data, `model=mlp_gate`, 40 epochs, batch 64, `grad_clip=null`, two seeds,
prior sample-matched across geometries. Sliced W2, mean of two seeds; `k` is
solver steps and the Heun solver makes NFE = `2k - 1`.

| prior | geometry | coupling | k=1 | k=2 | k=4 | k=8 | k=100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| white | Cartesian | independent | 0.365 | 0.147 | 0.073 | 0.056 | 0.051 |
| white | Cartesian | OT | 0.377 | 0.144 | 0.072 | 0.056 | 0.050 |
| white | cylindrical | independent | 0.160 | 0.152 | 0.119 | 0.104 | 0.103 |
| white | cylindrical | OT | 0.157 | 0.146 | 0.115 | 0.100 | 0.100 |
| 4 | Cartesian | independent | 0.372 | 0.148 | 0.075 | 0.057 | 0.061 |
| 4 | Cartesian | OT | 0.386 | 0.131 | 0.071 | 0.056 | 0.064 |
| 4 | cylindrical | independent | 0.161 | 0.136 | 0.100 | 0.085 | 0.085 |
| 4 | cylindrical | OT | 0.160 | **0.106** | 0.080 | 0.074 | 0.082 |
| 16 | Cartesian | independent | 0.413 | 0.147 | 0.087 | 0.073 | 0.094 |
| 16 | Cartesian | OT | 0.399 | 0.122 | 0.081 | 0.069 | 0.099 |
| 16 | cylindrical | independent | 0.229 | 0.145 | 0.099 | 0.088 | 0.096 |
| 16 | cylindrical | OT | 0.186 | 0.113 | 0.073 | 0.065 | 0.100 |

At prior correlation 4, k=2, per seed: cylindrical OT `[0.106, 0.106]` against the
best Cartesian arm `[0.130, 0.132]`, and cylindrical independent `[0.137, 0.135]`.
At k=4 the arms overlap. Raw per-seed values in
`docs/reproduce/gate_artefacts/smooth_prior_sweep.txt`.

### A measurement defect, found and fixed

`evaluate.py` scored straightness under independent pairing for every arm,
including those trained on optimal-transport pairs, which measures a different
regression than the one the model was fitted to. It now scores under
`training.coupling`. All 32 checkpoints were re-evaluated; generation metrics
reproduced exactly (largest difference 0.0), and straightness barely moved:
cylindrical OT reads 0.974 on Table 1 (was 0.972) and 0.947 at prior correlation 4
(was 0.952). The conclusion drawn from the defective number survives the correct
one. `docs/reproduce/gate_artefacts/straightness_reeval.txt` (a one-off check, kept as a historical record).

### Reading

- **A white prior makes both geometries generate spatially white fields.**
  Generated lag-1 amplitude correlation is 0.000 against 0.956 in the data, and
  every pooled metric is blind to it. A smooth prior restores it (gap 0.02 at
  correlation 4), but by pass-through rather than learning: a pointwise model maps
  each coefficient independently and so inherits the prior's spatial correlation.
  Correlation 16 overshoots (0.998). A claim about learned spatial structure needs
  a model that mixes space.
- **The crossover moves.** With a smooth prior the cylinder with OT wins at k=1
  (0.160 vs 0.372) and k=2 (0.106 vs 0.131), ties at k=4, and loses from k=8
  (0.082 vs 0.061 at k=100). Under a white prior on `cylinder_toy_iid` it lost from
  k=2.
- **The smooth prior switches the coupling on, for the cylinder, at few steps.**
  OT improves the cylinder by 22% at k=2 under correlation 4, and by 4% under a
  white prior.
- **The win is amplitude, not phase.** The cylinder is better on amplitude at every
  k (0.090 vs 0.100 at k=100) and worse on phase from k=2 (0.178 vs 0.093 at
  k=100). The phase gap narrows from about 3.7x to 1.9x, partly because the
  Cartesian arm's phase gets worse under a smooth prior (0.064 -> 0.093).
- **The phase target is still unlearnable.** Straightness of the cylinder with OT
  under its own coupling is 0.947 at correlation 4, in line with the angular
  displacement falling only from 1.57 to 1.30.
- **Over-smoothing costs everyone.** Correlation 16 is worse than 4 at convergence
  in both geometries.

---

## 4. Is "the more correlated, the better for the cylinder" established? No.

The claim came from three values of rho at dimension 1 under minibatch OT, where
the cylinder's one-step penalty went from a tie at rho = 0 to full separation at
rho = 1. A six-point sweep on the same setup did not reproduce it: under OT the
seed spreads overlapped at rho = 0, 0.2, 0.4 and 1.0 and the ratio moved without a
trend (1.4, 2.4, 2.0, 5.6, 4.1, 1.7). Under independent coupling the cylinder's
advantage was stable at about 5x and flat in rho (5.1, 5.5, 4.5, 5.1, 5.4, 5.2).
At field scale rho was never swept. The only monotone quantity was the trained
network's induced angular velocity, a probe already shown to depend on what the
network learned.

Measured on the bridges instead, no network, 200 000 pairs per value
(`scripts/bridge_angular_velocity.py --coupling RHO`):

| bridge | rho = 0 | 0.25 | 0.5 | 0.75 | 1 |
| --- | --- | --- | --- | --- | --- |
| Cartesian, independent: share of chords above pi | 46.1% | 46.1% | 46.1% | 46.1% | 46.1% |
| Cartesian, minibatch OT: share above pi | 1.1% | 1.0% | 1.0% | 1.2% | 2.2% |
| Cartesian, minibatch OT: q99.9 | 28 | 23 | 27 | 33 | 76 |
| Cylindrical: maximum | 3.14 | 3.14 | 3.14 | 3.14 | 3.14 |

Under independent coupling rho *cannot* matter, and the flat row is the proof by
measurement of an argument that holds by construction: the prior's phase is
uniform and independent of the target, so the relative angle of every chord is
uniform whatever the target's phase is, and the chord statistics depend on the
target only through its amplitude marginal -- which the copula holds fixed across
rho. Under OT there is a real effect, but only in the tail and only at the extreme:
flat to rho = 0.75, then the share of chords above pi doubles and q99.9 triples at
rho = 1. The cylinder is bounded at every rho.

So what survives is narrow: at the level of the bridges, strong amplitude-phase
dependence makes minibatch-OT Cartesian chords turn faster in their tail. Whether
that reaches trained models at field scale is unmeasured.

---

## 5. Two corrections, and a model that sees space

### The reference was in the wrong domain

Training divides every field by its own peak modulus before the model sees it,
so a model that learns the training distribution perfectly generates normalised
fields. `evaluate.py` compared those against *raw* fields: amplitude mean 0.671
raw against 0.618 normalised. The bias is the same for both geometries, but it
entered every absolute number above. It now builds the reference through the
training pipeline (`training_pipeline`), `metrics.json` records
`reference_domain: training transform`, and every pipeline checkpoint was
re-evaluated. **Absolute W2 values in sections 2 and 3 are superseded by the
figures below.**

One conclusion flipped. The MLP smooth-prior claim that the cylinder with OT wins
at k=2 does not survive: per seed `[0.096, 0.095]` against the Cartesian OT arm's
`[0.090, 0.092]` -- the plane is marginally ahead. Table 1 on the corrected
reference (sliced W2, k = 1 / 2 / 100): Cartesian independent 0.310 / 0.099 /
0.042, cylindrical OT 0.161 / 0.160 / 0.130. The cylinder wins only at one step,
and the plane is about three times better at convergence.

### The Cartesian U-Net anomaly is not the L1 loss

Both geometries train with an L1 loss by default, and L1 regresses the
conditional *median* where flow matching's guarantee needs the conditional mean.
That was the leading explanation for a Cartesian U-Net that fits its field well
(straightness 0.065) yet generates shrunk amplitudes under a smooth prior (mean
0.455 at k=100 against 0.618). Retraining it with MSE gives 0.423: no better. The
hypothesis is falsified and the cause is unknown. The prior-4 U-Net comparison
should not be used until it is explained. The Cartesian MLP under the same prior
(0.642) and the Cartesian U-Net under a white prior are both fine.

### U-Net, white prior

`scripts/sweeps/unet_prior.sh`, `c_unet` (3.6M parameters, 576 more on the
cylinder), `cylinder_toy_field`, 64x64, rho = 0.5, two seeds. Sliced W2 on the
training-domain reference:

| geometry | coupling | k=1 | k=2 | k=4 | k=8 | k=100 |
| --- | --- | --- | --- | --- | --- | --- |
| Cartesian | independent | 0.414 | 0.368 | 0.287 | 0.141 | 0.048 |
| Cartesian | OT | 0.368 | 0.327 | 0.257 | 0.121 | **0.038** |
| cylindrical | independent | 0.135 | 0.237 | 0.204 | 0.138 | 0.088 |
| cylindrical | OT | **0.120** | **0.172** | **0.163** | 0.113 | 0.069 |

Per seed, cylindrical OT against Cartesian OT: k=2 `[0.175, 0.168]` vs
`[0.319, 0.336]`, k=4 `[0.163, 0.164]` vs `[0.255, 0.259]`, k=8 `[0.094, 0.133]`
vs `[0.134, 0.109]` (overlapping). Raw tables in
`docs/reproduce/gate_artefacts/unet_prior_sweep.txt`.

- **The U-Net learns spatial structure from white noise.** Generated lag-1
  amplitude correlation 0.964-0.967 against 0.956 in the data, in both
  geometries; the pointwise MLP produced 0.000 from the same prior.
- **Spatial context makes the angular target learnable.** Straightness is about
  0.13 for both geometries, against 0.95 for the pointwise cylindrical model.
- **The cylinder wins the few-step regime cleanly** -- k=1, 2 and 4 with separated
  seeds -- ties at k=8, and loses at convergence (0.069 vs 0.038).
- **The asymptotic gap is phase, as before.** Circular W2 on phase at k=100: 0.087
  Cartesian OT against 0.173 cylindrical OT. Amplitude goes the other way: 0.022
  cylindrical against 0.062 Cartesian.
- **OT helps both geometries here**, at every step count, unlike the MLP at the
  same field size.

This is the best-supported few-step result so far: a spatial model, the standard
white prior, both geometries on the same architecture, and seeds that do not
overlap. It is still a synthetic field, and the asymptote still favours the plane.

---

## 6. Field-scale measurements recorded nowhere else

All on the training-domain reference where a trained model is involved.

### Minibatch OT against dimension, and patch-level coupling

`scripts/coupling_dimension.py`, network-free. `docs/reproduce/gate_artefacts/coupling_dimension.txt`.

| field | dimension | OT cost reduction | share of batch reordered | spread of pairwise costs |
| --- | --- | --- | --- | --- |
| 1x1 | 1 | 85.9% | 96.9% | 0.8193 |
| 4x4 | 16 | 43.7% | 100.0% | 0.2144 |
| 8x8 | 64 | 21.3% | 98.4% | 0.1069 |
| 16x16 | 256 | 11.5% | 95.3% | 0.0528 |
| 32x32 | 1024 | 5.1% | 98.4% | 0.0265 |
| 64x64 | 4096 | 3.1% | 100.0% | 0.0135 |

A larger batch buys little back (16x16: 4.9% at batch 8, 14.0% at 256), and
smoother *data* under a white prior does not help at all (64x64: 3.1% white,
1.7% at correlation length 63) -- the displacement inherits the prior's
whiteness. The coupling keeps acting on nearly the whole batch at every size;
what collapses is the room it acts in, a sixty-fold fall in the spread of
pairwise costs.

Patch-level coupling -- an independent assignment per 16x16 patch position of a
64x64 field -- leaves lag-1 correlation inside patches at 0.979 and drops it
across seams to 0.044, while sliced W2 to the data reads 0.0000 for the untouched
data, image-level OT and patch-level OT alike. The damage is in the coupled
*endpoints*, before any network: each position has its own permutation, so every
endpoint is assembled from different samples. Flow matching reproduces the
coupled endpoint distribution exactly, so no architecture trained on those
endpoints removes the seams; a better model reproduces them more faithfully.
Modelling patches as samples, with OT at patch dimension, is a different and
untested proposal whose seams appear at assembly instead.

### Dimension sweep, MLP, independent coupling

`scripts/sweeps/dim_sweep.sh`, `docs/reproduce/gate_artefacts/dim_sweep.txt`. Price of one step,
`W2(k=1) - W2(k=100)` sliced, per seed:

| field | cylindrical | Cartesian | ratio |
| --- | --- | --- | --- |
| 16x16 | `[0.073, 0.059]` | `[0.273, 0.270]` | 4.1x |
| 32x32 | `[0.044, 0.043]` | `[0.269, 0.262]` | 6.1x |
| 64x64 | `[0.037, 0.026]` | `[0.274, 0.262]` | 8.6x |

Not a claim to publish: the pointwise cylindrical model barely learns its angular
target (straightness about 0.97), so a small penalty partly reflects a model that
is equally poor at one step and at a hundred.

### Phase prior, cylindrical arm only, MLP, independent coupling

`scripts/sweeps/phase_prior.sh`, `docs/reproduce/gate_artefacts/phase_prior_sweep.txt`.
`cylinder_toy_iid`, 64x64, rho = 0.5, two seeds.

| phase spread | straightness | phase W2, k=1 | phase W2, k=100 | amplitude W2, k=100 | sliced W2, k=1 | sliced W2, k=100 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.25 | 0.222 | 0.544 | 0.362 | 0.076 | 0.200 | 0.122 |
| 0.5 | 0.404 | 0.499 | 0.400 | 0.082 | 0.191 | 0.137 |
| 1.0 | 0.703 | 0.494 | 0.403 | 0.099 | 0.190 | 0.136 |
| 2.0 | 0.948 | 0.387 | 0.367 | 0.110 | 0.152 | 0.141 |
| uniform | 0.970 | 0.425 | 0.345 | 0.110 | 0.166 | 0.135 |

A concentrated phase prior makes the angular target learnable, monotonically in
the spread, without improving phase generation -- which is best under the uniform
prior -- and it worsens the one-step regime. The bottleneck it exposed was
transport on the circle rather than regression. Within-arm only: the prior is not
shared with a Cartesian arm.

### Angular velocity of the bridges

`scripts/bridge_angular_velocity.py`, network-free,
`docs/reproduce/gate_artefacts/bridge_angular_velocity.txt`. At rho = 0.5 over a million pairs:
the Cartesian chord under independent pairing has median peak angular velocity
2.81, q99.9 1566, 46.0% of chords above pi, and a power-law tail of index 1.006,
which means an infinite expected peak; under minibatch OT the median is 0.31 and
1.0% exceed pi. The cylinder is bounded by pi by construction. The sample maximum
grows with the number of samples (885 at a thousand pairs, 1.19 million at a
million), so no maximum belongs in a paper. The closed form `|L| / d_min^2` was
checked against a 20 001-point time grid: median relative gap 4.8e-10, and the
grid never exceeds it. Section 4 has the rho sweep.

---

## 7. Where each number comes from

Evaluating a fixed checkpoint is exactly reproducible: re-evaluating 32 of them
reproduced every generation metric to 0.0. Retraining is seeded but GPU kernels
are not bitwise deterministic, so a retrained model matches to within the seed
spread quoted beside each result, not to the digit. The gate scripts and the
network-free scripts do not go through `cfm.evaluate` and were never affected by
its two defects.

**The arXiv v1 paper** has one wrapper per table in `scripts/paper/` (index in the
README, "Reproducing the paper"), and the evaluations its Tables 2 and 3 were
written from are archived in `docs/reproduce/paper_results/unet_eval_metrics.json`, so
`uv run python scripts/paper/paper_tables.py --archive` prints them without
retraining. The table below is the full research log; its "Table 1" is the MLP 2x2
of an earlier plan, not the paper's Table 1 (which is the bridge table).

| result | command | artefact | section |
| --- | --- | --- | --- |
| Gate A: the factorised coupling | `uv run python scripts/coupling_gate.py --samples 1024 --seeds 8 --plot gate_a.png` | `gate_a_factorised_coupling.png` | 1 |
| Gate B, three seeds | `uv run python scripts/coupling_gate_b.py --train-steps 8000 --eval-samples 2048 --rho 0.0 0.5 1.0 --seeds 1 --seed S` for S = 0, 1, 2 | `gate_b_seed{0,1,2}.txt` | 2 |
| Gate B, rho sweep with probes | `uv run python scripts/coupling_gate_b.py --train-steps 8000 --eval-samples 2048` | `gate_b_rho_sweep.txt` | 2, 4 |
| Bridge angular velocity | `uv run python scripts/bridge_angular_velocity.py [--coupling RHO]` | `bridge_angular_velocity.txt` | 4, 6 |
| OT against dimension, patches | `uv run python scripts/coupling_dimension.py` | `coupling_dimension.txt` | 6 |
| Table 1 | `bash scripts/sweeps/table1.sh LOGDIR` | `table1_2x2_64x64.txt` | 5 |
| Dimension sweep, MLP | `bash scripts/sweeps/dim_sweep.sh LOGDIR` | `dim_sweep.txt` | 6 |
| Phase prior | `bash scripts/sweeps/phase_prior.sh LOGDIR` | `phase_prior_sweep.txt` | 6 |
| Smooth prior, MLP | `bash scripts/sweeps/smooth_prior.sh LOGDIR` | `smooth_prior_sweep.txt` | 3, 5 |
| U-Net, 64x64 | `bash scripts/sweeps/unet_prior.sh LOGDIR` then `uv run python scripts/sweeps/analyse_unet.py` | `unet_prior_sweep.txt` | 5 |
| U-Net, 32x32 and 16x16 | `bash scripts/sweeps/unet_size.sh LOGDIR` then `uv run python scripts/sweeps/analyse_unet_size.py` | `unet_size_sweep.txt` | 8 |
| Loss ablation, 32x32 | `bash scripts/paper/ablation_loss32.sh LOGDIR` then `uv run python scripts/paper/ablation_tables.py` | console | 9 |
| v2 grid, both losses, 5 seeds | `sbatch scripts/wcss/paper_tables_l2u.sbatch` (variants in its header) | `docs/reproduce/paper_results/unet_eval_metrics_{l2u,l1u_cylindrical,l1w_cartesian}.json` | 9 |
| Loss protocols, statistics | `uv run python scripts/paper/loss_protocols.py --archive` | console | 9 |

**The v2 paper** reads its U-Net tables from the three archives above; their
protocol, Slurm jobs and per-run provenance are in `docs/reproduce/paper_results/README.md`.

---

## 8. U-Net against field size

White prior, `c_unet`, `cylinder_toy_field` with correlation length 4 pixels at
every size, rho = 0.5, two seeds. 16x16 and 32x32 from
`scripts/sweeps/unet_size.sh`, 64x64 from the white-prior arm of
`unet_prior.sh`; tables from `scripts/sweeps/analyse_unet_size.py`,
`docs/reproduce/gate_artefacts/unet_size_sweep.txt`. Sliced W2, cylinder with OT against the
Cartesian arm with OT (the better Cartesian arm at every entry that decides):

| field | arm | k=1 | k=2 | k=4 | k=8 | k=100 |
| --- | --- | --- | --- | --- | --- | --- |
| 16x16 | cylindrical OT | **0.102** | **0.100** | **0.104** | 0.106 | 0.074 |
| 16x16 | Cartesian OT | 0.185 | 0.170 | 0.118 | 0.070 | 0.061 |
| 32x32 | cylindrical OT | **0.120** | **0.114** | **0.131** | 0.100 | 0.124 |
| 32x32 | Cartesian OT | 0.320 | 0.258 | 0.192 | 0.073 | **0.051** |
| 64x64 | cylindrical OT | **0.120** | **0.172** | **0.163** | 0.113 | 0.069 |
| 64x64 | Cartesian OT | 0.368 | 0.327 | 0.257 | 0.121 | **0.038** |

Bold marks a win with separated seeds. At k=8 the seeds overlap at 16x16 and
64x64 and the plane wins at 32x32; at k=100 they overlap at 16x16 and the plane
wins at 32x32 and 64x64.

- **The few-step win holds at every size.** Cylinder with OT beats the best
  Cartesian arm at k = 1, 2 and 4 at 16x16, 32x32 and 64x64, seeds separated.
- **The asymptote closes only at the smallest field.** A tie at 16x16 (0.074 vs
  0.061, overlapping seeds); the plane is ahead at 32x32 and 64x64.
- **Not a monotone trend.** The cylinder's asymptote at 32x32 (0.124) is worse
  than at both 16x16 (0.074) and 64x64 (0.069). One favourable size is not a
  trend in size.
- **OT buys most on small fields.** Cartesian at k=1: 0.421 -> 0.185 with OT at
  16x16, 0.414 -> 0.368 at 64x64. Cylinder at k=100: 0.150 -> 0.074 at 16x16.
  Consistent with section 6: a smaller sample leaves minibatch OT more room.
- **The phase gap narrows on small fields.** Circular W2 on phase at k=100,
  cylinder OT against Cartesian OT: 0.129 vs 0.113 at 16x16, 0.173 vs 0.087 at
  64x64.

What it says about modelling patches as samples: at 16x16 the cylinder with OT
wins every step count up to four and ties the asymptote, which is the case for
training on 16x16 patches with OT at patch dimension. Two things stand between
that and a claim. Generating a whole image then requires the patches to be
assembled without seams, which is untested; and the 32x32 point does not follow
the 16x16 one, so a patch size cannot yet be chosen from a trend.

---

## 9. The loss, not the geometry, set the asymptote

Measured 2026-09-11. Until then every cylindrical U-Net was trained with L1 and a
phase term weighted by the clean amplitude `A_1 / mean(A_1)`, and every Cartesian
one with plain L1. Two properties of that objective break the flow-matching fixed
point: L1 regresses the conditional median of the target rather than its mean, and
a weight that depends on `x_1` makes the phase channel regress a reweighted
statistic that the amplitude channel does not. The flag
`training.loss.phase_amplitude_weighting` (default `true`, the old behaviour) turns
the weight off.

### Ablation, 32x32, 3 seeds, joint OT

`scripts/paper/ablation_loss32.sh`, local RTX 4070 Ti SUPER. Sliced W2, mean over
seeds; the weighted L1 rows reproduce the v1 archive (0.121 against 0.114 / 0.134).

| arm | k=1 | k=4 | k=100 | phase W2, k=100 |
| --- | --- | --- | --- | --- |
| cylinder, L1, weighted (v1) | 0.123 | 0.140 | 0.121 | 0.307 |
| cylinder, L1, unweighted | 0.093 | 0.083 | 0.074 | 0.168 |
| cylinder, L2, weighted | 0.111 | 0.089 | 0.063 | 0.205 |
| cylinder, L2, unweighted | 0.114 | 0.064 | 0.038 | 0.098 |
| Cartesian, L1 | 0.319 | 0.196 | 0.051 | 0.138 |
| Cartesian, L2 | 0.341 | 0.206 | 0.033 | 0.095 |

Removing the weight helps at every k (at k=100, -39% sliced and -45% phase,
seeds separated). With L2 and no weight the cylinder ties the Cartesian arm at
k=100 and its error becomes monotone in k; the v1 statements "the plane is ahead
at convergence" and "the cylindrical error is not monotone in k" were properties
of the loss.

### The v2 grid, 5 seeds, both losses

`docs/reproduce/paper_results/`, printed by `scripts/paper/loss_protocols.py --archive`.
Every geometry x coupling x {L1, L2} x {16, 32, 64} x 5 seeds, on WCSS. Two
comparisons, with exact two-sided Mann-Whitney p (the minimum with 5 vs 5 is
0.008, reached exactly when the seeds separate):

- **Matched loss** (L2 in both, the paper's Tables 2 and 3): cylinder + OT beats
  the better Cartesian coupling at every k <= 8 at all three sizes with separated
  seeds (12 of 12 cells); at k=100 no difference is significant (64x64: 0.040 vs
  0.046, p = 0.15).
- **Best loss per geometry** (each picks its lowest mean from {L1, L2} x
  {independent, OT}): the same, except 16x16 at k=8 (p = 0.15); at k=100, 64x64,
  0.035 (cylinder L1, OT) against 0.032 (Cartesian L1, OT), p = 0.69.
- **OT against independent pairing on the cylinder** (matched loss, k <= 4):
  3-60% lower error, seeds separated in 6 of 9 cells.

### What it does not show

- The components split. At 64x64 the cylinder is ahead on amplitude W2 at every
  k, but the plane is ahead on circular phase W2 at k=8 (32x32 and 64x64, seeds
  separated) and at k=100 against Cartesian L1 (0.063 vs 0.114, p = 0.008).
- A tie at k=100 is a non-significant difference with 5 seeds, not an
  equivalence; the 95% interval at 64x64 still allows the cylinder to be up to
  about 0.017 worse.
- The best-loss protocol selects on the evaluation itself; it flatters both
  geometries equally and is a robustness check, not the headline.
