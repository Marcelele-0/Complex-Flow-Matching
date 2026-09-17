# Where the paper stands, 2026-09-17

Written at the end of a long session so the next one starts from evidence rather than
from recollection. Everything numbered here was measured; guesses are labelled.

## Constraints that shape every decision below

ICLR 2027 closes 2026-09-25. The main text currently runs to 11 pages against a 9-page
limit, and today's additions lengthened it. **Compute is not the bottleneck; writing
is.** A full retrain of every table is ~100 cluster jobs and under a day of GPU, but the
prose around changed numbers is days.

## Table layout, settled

Four tables in the main text. Splitting per domain was considered and rejected: more
tables cost more pages, and we are already over.

| | contents | state |
| --- | --- | --- |
| 1 | analytical bridges, three sources, unweighted and energy-weighted columns | done |
| 2 | few-step, blocked by domain, with `k*` | synthetic and speech done, **MRI block missing** |
| 3 | resolution scaling, synthetic only | done |
| 4 | the factorised coupling trap, synthetic only | done |

Appendix A (losses) and B (audio budget) are written. **Appendix C (bounded head) is
not.** Tables 3 and 4 stay synthetic on principle, not for cost: 3 needs a controllable
family of resolutions and real MRI has one, 4 needs amplitude-phase dependence as a
knob, which only the copula gives.

## Two sentences the text must get right

**Do not claim a phase win at k = 1 on speech or MRI.** Measured against the raw prior's
phase: speech 0.0335 against 0.0352 -- indistinguishable; knee 0.2521 against 0.2773 --
9%; synthetic 0.2071 against 0.3238 -- 36%. The metric pools pixels into one cloud, so a
model that carries a uniform prior phase to a uniform output phase matches that marginal
without having learned anything, and the median |v_theta| at k = 1 on speech is 0.025 --
it barely moves the phase. **The k = 1 advantage is real and it is an amplitude
advantage.**

**Appendix C must say the dynamics change drastically and the error does not.** The
pi*tanh head takes peak |v_theta| from 9.886 to 3.142 while the k = 2 rise is unchanged
(0.0561 -> 0.1225 unbounded, 0.0588 -> 0.1261 bounded, seeds overlapping). Writing that
both are indistinguishable throws the finding away: the point is precisely that one
changes enormously and the other not at all.

## The strongest unwritten result

Sliced W2 of the raw prior against the data, no model, no integration:

```
              prior    cylinder k=1   Cartesian k=1
synthetic 64  0.1502      0.1174          0.3763
speech        0.3781      0.0454          0.0780
knee MRI      0.2239      0.0561          0.1195
```

**One Euler step in Cartesian coordinates is worse than not running the model at all**
on synthetic fields, and worse than the prior on the phase component in all three
domains. That is the sharpest statement of the few-step problem we have, and it also
pre-empts the reviewer question "what does the noise alone give".

## Next steps, in order

1. **Crop k-space to 64x64.** Code, not a flag: the store holds images after ESPIRiT, so
   it is FFT, centre crop, IFFT at load time. Physically a lower-resolution acquisition
   rather than interpolation, which is what makes it defensible.
2. **Table 5 at 64x64**, five seeds, protocol unchanged (`bounded_velocity: false`).
   About 25x cheaper per run than 320x320.
3. **Check whether joint OT removes the k = 2 rise there.** This is the one item with
   real risk and it is a hypothesis, not a formality: knee amplitudes are bimodal with
   17.6% exact zeros, unlike the synthetic toy, and that could act independently of
   dimension. If the rise survives at 64x64, the dimensionality story collapses.
4. Write in what is already measured: two rows extending section 5.3 to 128 and 320, the
   prior control row, appendix C.
5. Cut 11 pages to 9, last, because what survives depends on what step 3 returns.

## Keep 320x320 whatever happens at 64

Moving the headline to 64x64 reverses a deliberate decision -- issue #84 was demoted to
P2 with the reason "the headline fastMRI table is at the acquisition's own resolution",
and 320x320 is what the score-based-prior literature reports. New evidence justifies
revisiting that, but only if both sizes appear: 64x64 where the method works, 320x320
with the cost-drop curve explaining why it does not. Reporting only 64 invites "why does
everyone else use 320", and the honest answer would be that ours stops working there.

## Not on the table

**Latent space kills the argument.** Table 1 is about the data's amplitude distribution
near zero; in a latent space that distribution is whatever the encoder produced, the
prior is Gaussian rather than cylindrical, and the reviewer's first question is why not
plain Euclidean flow matching in R^2d.

**Schrodinger bridges are a separate paper.** And the premise that they solve this is
shaky: minibatch SB carries the same concentration-of-distances bias that kills minibatch
OT in high dimension.

**Patch-level coupling is measured and broken** -- it drives lag-one correlation across
seams to -0.008 against the data's 0.979. Patch-level *data* is a different and untested
thing, with no seams because nothing is stitched, but a 64x64 tile of a knee is a texture
crop rather than an image, which the k-space route avoids.
