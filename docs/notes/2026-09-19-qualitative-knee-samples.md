# What the knee samples actually look like

Written 2026-09-19. Figure: `paper/ICLR Main/figures/knee_samples_64.png`.
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

## Two consequences for the manuscript

1. **This cannot be a "look at our samples" figure.** Shown in the main text as evidence of
   synthesis quality it argues against us. What it does support is the $k=1$ claim: the
   Cartesian single step produces an almost empty field while the cylindrical one produces
   a field with roughly the right amplitude level, which is the $1.8\times$ pooled gap made
   visible. If it ships, it ships as a candid appendix panel making exactly that point, and
   the caption says outright that 40 epochs does not reach anatomy.
2. **It is one more argument against the over-smoothing claim.** At $k=100$ the Cartesian
   blobs have *harder* edges than the cylindrical ones. The claim was already flagged as
   needing to be resolution-dependent or dropped (Appendix D has the Cartesian arm closer
   on all three structural metrics at $320\times320$); this is an independent look at the
   same thing and it points the same way.

## The budget question a reviewer would now ask

40 epochs over 5324 slices is short for unconditional generation of a real image
distribution, and until now nothing in the pipeline could have revealed that: `cfm.evaluate`
keeps metrics and discards samples, and every knee number in the paper is a distance between
distributions, which a blob field can score respectably on. The honest reading is that the
knee block measures *few-step behaviour of two parameterisations at a fixed small budget*,
not sample quality --- which is what the text should say if the figure appears.

Not proposing a longer run here: it would cost a fresh cohort on WCSS and the deadline is
2026-09-25. Recording it so the decision is made deliberately rather than by omission.
