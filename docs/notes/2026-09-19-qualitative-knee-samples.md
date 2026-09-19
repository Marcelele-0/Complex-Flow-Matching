# What the knee samples actually look like

Written 2026-09-19. Figure: `docs/notes/figures/knee_samples_64.png`.
Generator: `scripts/paper/knee_samples_panel.py` (no cluster, ~2 min on the laptop GPU).

```bash
uv run python scripts/paper/knee_samples_panel.py
```

Checkpoints: `t5c64_cylindrical_ot_s0` (stamp `2026-09-18_17:10`, epoch 40) and
`t5c64_euclidean_independent_s0` (stamp `2026-09-18_17:15`, epoch 40), both the 64x64
k-space-crop arms of Table 5. Reference: `data/knee_pd/val.h5`, `role=all`, slices 0 / 341
/ 682 of 1023, through `cfm.evaluate.training_pipeline` so the fields sit in the domain
training used. One prior seed shared across all four generated rows.

## The finding, stated plainly

**Neither arm generates recognisable knee anatomy.** This is the first time any sample from
these runs has been looked at, and it is not what the pooled metrics suggest.

| row | what it looks like |
| --- | --- |
| real | knee joints; phase smooth inside tissue, a sinc ripple in the air |
| CyFM + OT, $k=1$ | fine-grained noise with a faint low-frequency envelope |
| CyFM + OT, $k=100$ | large amplitude blobs, phase piecewise smooth over them |
| Cartesian, $k=1$ | near-uniform dark field; amplitude mean $0.096$ against the reference's $0.215$ |
| Cartesian, $k=100$ | blobs with harder edges and higher contrast than CyFM's |

Amplitude means over the three shown fields: reference $0.2151$, CyFM $k{=}1$ $0.2377$,
CyFM $k{=}100$ $0.1972$, Cartesian $k{=}1$ $0.0964$, Cartesian $k{=}100$ $0.2129$.

## The decision: it does not go in the paper

Taken 2026-09-19, and the figure was moved out of `paper/ICLR Main/figures/` for that
reason, so nothing can quietly reference it later.

**What this panel measures is the backbone, not the geometry.** A 64-channel U-Net trained
for 40 epochs is not an unconditional image generator for a real anatomical distribution,
and no choice of coordinates fixes that. Both arms fail the same way, which is exactly what
a capacity-and-budget limit looks like and not what a geometry difference looks like. The
paper's claim is about the transport --- few-step behaviour of two parameterisations at one
fixed budget --- and knee MRI is an instance of a complex-valued field, not the subject. This
is not a reconstruction paper and there is no diagnostic endpoint anywhere in it.

Shipping the panel would invite a question the paper does not answer and does not need to
answer, and a reader would read "the method does not work on real data" off a picture that
actually shows "this backbone at this budget does not synthesise knees".

## What it is still good for

1. **Nothing in the pipeline could show this before.** `cfm.evaluate` keeps metrics and
   discards samples, and every knee number is a distance between distributions, which a blob
   field scores respectably on. The script now exists, so any future arm can be looked at
   before its numbers are believed.
2. **One more independent argument against the over-smoothing claim.** At $k=100$ the
   Cartesian blobs have *harder* edges than the cylindrical ones. The claim was already
   flagged as needing to be resolution-dependent or dropped (Appendix D has the Cartesian arm
   closer on all three structural metrics at $320\times320$); this points the same way.
3. **The $k=1$ gap is visible.** The Cartesian single step gives an almost empty field where
   the cylindrical one has roughly the reference's amplitude level. Not used as evidence ---
   the measured $1.8\times$ already is --- but it is consistent.

If a reviewer asks for samples, this is the honest answer to hand them, together with the
budget: 40 epochs, 5324 slices, one 64-channel U-Net.

## If a reviewer asks for *better* samples --- decline, and say why

Marcel's position, recorded so the rebuttal does not improvise it. The request is refused,
not accommodated, on two independent grounds:

1. **It asks for a different paper's experiment.** Getting knee anatomy out of an
   unconditional generator needs a backbone sized for that task and a training budget to
   match --- a bigger network, more epochs, probably an architecture chosen for images rather
   than the one deliberately held fixed across every arm of every table here. That is a
   capacity result, and it would say nothing about the cylinder. Offer it as the obvious
   extension, not as a missing control.
2. **Sample quality is not the claim.** What stands or falls in this paper is a *controlled
   comparison* of two coordinate systems: one backbone, one schedule, one budget, one
   coupling protocol, five seeds, and the only difference between the arms is whether the
   bridge runs in $(\mathrm{Re}, \mathrm{Im})$ or on $[0,\infty) \times S^1$. Both arms are
   equally limited by the backbone, which is what makes the difference between them
   attributable. This is not a MICCAI submission; there is no diagnostic endpoint, no
   reconstruction task and no clinical claim anywhere in it.

**The one condition that keeps this defence honest:** it holds only while the text never
claims sample quality. Relative, measured statements are fine --- "reproduces the reference's
amplitude texture twice as closely" is a number from a table. An unmeasured qualitative
assertion is not, and the over-smoothing claim is exactly that. Leave it in and a reviewer
can answer "you made a claim about how the pictures look; show the pictures", and then the
refusal above no longer works. Dropping it is what buys the right to decline.
