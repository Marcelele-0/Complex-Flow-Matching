# The geometry of complex-valued probability paths

Working notes for the paper: what is proved, what is measured, what is asserted,
and how each number was produced. Every figure quoted here is reproducible from
the commands given; nothing is carried over from memory.

Written 2026-09-08. Supersedes the framing of the former `docs/GATE_63_PLAN.md`,
which measured reconstruction against a zero-filled floor; that plan was removed
from the tree on 2026-09-11 and remains in the git history (`git log --
docs/GATE_63_PLAN.md`) as the record of how the current position was reached.

---

## 1. One identity

Take two unit phasors separated by an angle `Δθ`. Two notions of distance are
available, and they are not the same thing:

```
chordal   ‖e^{iθ₁} − e^{iθ₂}‖  =  2 |sin(Δθ/2)|
geodesic  d_{S¹}(θ₁, θ₂)       =  |Δθ|
```

The chord cuts through the interior of the circle; the arc stays on it. Every
result below is a consequence of that one difference, which is why the two
literatures that suffer from it — generative modelling and reconstruction
regularisation — suffer from it in different ways without noticing they share a
cause.

---

## 2. Consequence on paths: magnitude attenuation

### Statement

Linear interpolation between two complex numbers whose phases differ passes
inside the circle, so the modulus is attenuated at intermediate times. At the
midpoint of two unit phasors:

```
|½(e^{iθ₁} + e^{iθ₂})|  =  |cos(Δθ/2)|
```

Under an i.i.d. uniform phase prior `Δθ ~ U(−π, π)`:

```
E |cos(Δθ/2)|  =  (1/2π) ∫_{−π}^{π} cos(Δθ/2) dΔθ  =  2/π  ≈  0.6366
```

so the expected attenuation is `1 − 2/π ≈ 36.34%`. The integrand is
non-negative on the domain because `Δθ/2 ∈ (−π/2, π/2)`, which is why the
absolute value drops out.

A geodesic on `ℝ⁺ × S¹` interpolates amplitude and angle separately, so its
modulus at any `t` is the interpolated amplitude and the attenuation is zero by
construction.

### Measured

```
uv run python scripts/chord_vs_arc.py
```

| `Δθ` | Euclidean midpoint modulus | geodesic | attenuation |
| --- | --- | --- | --- |
| 0.000 | 1.0000 | 1.0000 | 0.00% |
| 0.524 | 0.9659 | 1.0000 | 3.41% |
| 1.047 | 0.8660 | 1.0000 | 13.40% |
| 1.571 | 0.7071 | 1.0000 | 29.29% |
| 2.094 | 0.5000 | 1.0000 | 50.00% |
| 2.618 | 0.2588 | 1.0000 | 74.12% |
| 3.142 | 0.0000 | 1.0000 | 100.00% |

Monte Carlo under the uniform prior, 200 000 draws:

```
measured    36.3294%
predicted   36.3380%   (= 1 − 2/π)
```

No network, no loss, no learned representation, so nothing here is attributable
to the confounds listed in §6.

### Scope, stated plainly

The uniform-phase assumption holds **by construction** on an unconditional path,
where the prior is `U([0,1]) ⊗ U(S¹)`. It does **not** hold on the conditional
`aliased → clean` bridge: measured median `|Δθ|` on tissue there is 0.20 rad,
implying about 4.3% attenuation (2.0% amplitude-weighted), rising to roughly
10.5% at R = 24 and saturating, because the ACS band anchors low-frequency
phase.

This is why the generative track, not the reconstruction track, is where the
proposition describes reality. Any paper claiming otherwise invites the
objection immediately.

---

## 3. Consequence on penalties: vanishing gradient

### Statement

A squared chordal penalty and its derivative:

```
‖e^{iθ̂} − e^{iθ}‖²  =  2(1 − cos Δθ)
d/dΔθ                =  2 sin Δθ        →  0   as Δθ → π
```

A squared geodesic penalty and its derivative:

```
Δθ²
d/dΔθ  =  2 Δθ                          →  2π  as Δθ → π
```

The chordal penalty is **stationary at the antipode**: the chord there is a
diameter, the longest it can be, so moving either endpoint along the circle
changes its length only to second order. The optimiser receives no first-order
signal exactly where the phase error is largest.

### Measured

| `Δθ` | chordal `\|∇\|` | geodesic `\|∇\|` | ratio |
| --- | --- | --- | --- |
| 0.524 | 1.000 | 1.047 | 0.955 |
| 1.047 | 1.732 | 2.094 | 0.827 |
| 1.571 | 2.000 | 3.142 | 0.637 |
| 2.094 | 1.732 | 4.189 | 0.413 |
| 2.618 | 1.000 | 5.236 | 0.191 |
| 3.142 | 0.000 | 6.283 | 0.000 |

Turning the gradient statement into an observable — 200 steps of gradient
descent at lr 0.05 from a controlled initial error, residual `|Δθ|` reported:

| start | chordal residual | geodesic residual |
| --- | --- | --- |
| 0.524 | 0.0000 | 0.0000 |
| 1.047 | 0.0000 | 0.0000 |
| 1.571 | 0.0000 | 0.0000 |
| 2.094 | 0.0000 | 0.0000 |
| 2.618 | 0.0000 | 0.0000 |
| **3.142** | **3.1416** | **0.0000** |

The chordal objective leaves an error of `π` completely uncorrected. The
geodesic one removes it.

### What this does and does not license

It licenses the statement that a chordal penalty under-corrects large phase
errors, in isolation. It does **not** license attributing a defect to any
published method: the chordal terms in the reconstruction literature sit inside
ADMM couplings whose proximal operators are learned and could compensate, and
`Δθ = π` is an unstable equilibrium rather than a trap — it means slow
correction in a neighbourhood, not impossibility. Claims about other people's
methods need their own measurements.

---

## 4. Consequence on integration: instability of the Cartesian field

### Statement

Under a Cartesian flow the induced angular velocity is

```
θ̇  =  (x u_y − y u_x) / A²  =  O(A⁻¹)
```

which diverges as `A → 0` with unbounded variance. The cylindrical target
velocity is globally bounded by `|u_θ| ≤ π`, from the range of `atan2`.

Prediction: integrating a learned Cartesian field **more accurately** should
amplify that divergence, while a coarse solver steps over it. The damage should
appear predominantly in phase.

### Measured

Quality as a function of solver steps, unconditional checkpoints from
2026-08-29, held-out volume, `t_start = 0.5`, R = 8, 128 strided slices:

```
scripts/nfe_sweep_unconditional.sh
```

| steps | CYL PSNR | SSIM | CPE | EUC PSNR | SSIM | CPE | Δ PSNR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 22.703 | 0.5071 | 0.5280 | 22.741 | 0.5085 | 0.5265 | −0.038 |
| 2 | 22.510 | 0.4885 | 0.5195 | 22.417 | 0.4913 | 0.5692 | +0.092 |
| 4 | 22.276 | 0.4742 | 0.5272 | 22.019 | 0.4509 | 0.6225 | +0.257 |
| 8 | 22.042 | 0.4582 | 0.5390 | 21.704 | 0.4102 | 0.6560 | +0.338 |
| 16 | 21.888 | 0.4437 | 0.5475 | 21.545 | 0.3863 | 0.6694 | +0.343 |
| 32 | 21.822 | 0.4381 | 0.5508 | 21.501 | 0.3843 | 0.6725 | +0.320 |
| 64 | 21.785 | 0.4343 | 0.5524 | 21.471 | 0.3811 | 0.6747 | +0.314 |
| 100 | 21.773 | 0.4331 | 0.5531 | 21.461 | 0.3803 | 0.6754 | +0.312 |

Degradation from 1 step to 100:

| | PSNR | SSIM | CPE |
| --- | --- | --- | --- |
| cylindrical | −0.93 dB | −0.074 | +0.025 rad |
| euclidean | **−1.28 dB** | **−0.128** | **+0.149 rad** |

The two arms are **tied at one step** and diverge only once integration begins.
Euclidean phase error grows six times faster. The signature matches the
prediction: the damage is largest in phase, monotone across eight points, and
present only on the Cartesian side.

### The caveat that blocks publication of this table

Both models degrade with more compute, which is a symptom of underfitting — two
patients, twenty-five epochs — not a property anyone should want. A reviewer
will say "fix the model first", correctly. At one step both arms also sit below
the zero-filled floor (22.83 dB at R = 8 on this volume).

The finding is therefore a **signal about the geometry difference**, not a
result. Confirming it needs a model that does not degrade with steps, i.e.
training on a cohort with real subject diversity.

### A hypothesis that did not survive

The sweep was designed to test "a geodesic path is already straight, so it needs
fewer steps". That is **false** as measured: neither arm needs more than one
step, and both get worse with more. The surviving claim is about stability under
integration, not about cost.

An earlier sweep on the *conditional* bridge showed the same one-step behaviour
for a different and uninteresting reason: starting from a zero-filled image that
already scores 22.83 dB, the model applies a small correction, and a small
correction is a one-step problem. That setting cannot answer questions about
integration at all.

---

## 5. Not yet done: optimal coupling

### Statement, to be proved

For a separable cost and product measures, the product of the marginal optimal
plans is optimal for the joint problem. On `ℝ⁺ × S¹` with

```
c((A,θ), (A′,θ′))  =  |A − A′|²  +  d_{S¹}(θ, θ′)²
```

this factorises into one-dimensional OT on amplitude — monotone rearrangement,
i.e. sorting — and OT on the circle, which has a closed form via circular CDF
shifts (Delon, Salomon, Sobolevski 2010). Both are `O(n log n)` and exact.

The prior `U([0,1]) ⊗ U(S¹)` is a product measure and is i.i.d. across pixels,
so its pixels may be permuted freely without changing its law. That is what
makes an exact coupling available **at pixel resolution** rather than between
images.

In the ambient plane `ℂ ≅ ℝ²` no analogue exists: two-dimensional OT has no
closed form, so minibatch OT in flow matching is approximate, entropy-regularised
and bounded by batch size.

### Why it matters

With an exact optimal coupling and geodesic conditional paths, the conditional
paths do not cross, the marginal field coincides with the conditional one, and
the marginal probability path is a displacement interpolation. With independent
coupling the marginal field is an average over crossing paths and is genuinely
curved. This is the mechanism by which coupling, not geometry alone, controls
integration cost.

**Positioning, to be written carefully:** the straightness theorem is not ours —
it is why OT-CFM straightens paths in the Euclidean setting. Ours is the
observation that this manifold admits the exact coupling the theorem requires,
which is the condition for applying it at all.

### Honest caveats to state before a reviewer does

1. The factorisation is exact for product measures. The prior is one; **the data
   is not** — amplitude and phase are correlated in real MRI. The approximation
   error is exactly that dependence.
2. "The cost separates because you chose a separable metric" is a fair
   objection. The answer is that the decoupled metric is a modelling decision
   justified by physics — a phase error should cost the same at any amplitude —
   and exact cheap transport is its *payoff*, not its premise.
3. Minibatch OT is not population OT and carries a known batch-size bias.

---

## 6. Confounds in the current experimental design

The cylindrical and Euclidean arms as configured differ in **three** ways, not
one, so no comparison between them isolates geometry:

1. **Loss.** Cylindrical is `|v_A − u_A| + λ(A₁/Ā₁)|v_θ − u_θ|`, amplitude
   weighted. Euclidean is `|v_x − u_x| + |v_y − u_y|`, unweighted. Different
   objectives, not one objective in two coordinate systems.
2. **Representation.** Cylindrical receives three input channels
   `(A, cos θ, sin θ)`; Euclidean receives two `(Re, Im)`. The cylindrical model
   is handed the amplitude explicitly.
3. **Prior.** The Euclidean config sets `noise_prior: uniform`, so it is not
   standard Gaussian-prior flow matching either.

Any table in the paper must vary one of these at a time. This is a cheap fix in
the loss configuration and it is not optional: without it the empirical section
falls to a single sentence of review.

---

## 7. Reproduction

```bash
# §2 and §3, no GPU
uv run python scripts/chord_vs_arc.py

# §4, needs the 2026-08-29 unconditional checkpoints
bash scripts/nfe_sweep_unconditional.sh /tmp/logs

# the conditional-bridge sweep that answered the wrong question, kept for the record
bash scripts/nfe_sweep_conditional.sh /tmp/logs
```

Checkpoints referenced:

| run | bridge | trained | note |
| --- | --- | --- | --- |
| `cylindrical_mini_bench`, `euclidean_mini_bench` | noise | 2026-08-29 | predate the bridge key |
| `g63_cylindrical_R8`, `g63_euclidean_R8` | aliased | 2026-09-07 | R = 8, SKM-TEA, 2 patients |

All SKM-TEA numbers come from a single held-out patient (MTR_030). Slices within
one volume are near-duplicates, so per-slice p-values establish that a shift is
systematic within that volume and nothing about patients.
