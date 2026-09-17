# Handoff

Written 2026-09-17 for whoever picks this up next. Read this, then
`2026-09-17-state-and-next-steps.md` for the plan and
`2026-09-17-k2-rise-investigation.md` for the open question. Everything below was
measured in this repository; where something is a guess it says so.

## The one-paragraph version

CyFM models complex-valued fields on the product manifold `[0, inf) x S^1` with a
decoupled metric, so the flow-matching bridge's angular target is bounded by pi instead
of diverging near the origin as the flat `(Re, Im)` treatment does. The contribution is
**geometry in the transport** -- the bridge, the regression target and the coupling --
not in the architecture and not in the loss; the network is a plain real U-Net. Target
venue ICLR 2027, closing 2026-09-25, and the main text is at 11 pages against a 9-page
limit.

## Where things are

| what | where |
| --- | --- |
| manuscript | `paper/ICLR Main/` in this repo, byte-identical to Overleaf |
| Overleaf working clone | `overleaf_cache/<id>/` -- gitignored, the MCP server's own checkout |
| cluster tree | `$PDDIR/CyFM` on `ui.wcss.pl`, an rsync copy, **not a git repo** |
| results | `outputs/evaluate/<run>/<stamp>/metrics.json` on the cluster |
| table rebuilder | `scripts/paper/collect_tables.py --outputs outputs` |

The paper exists in two copies and they drift. `paper/README.md` says which is newer
when they disagree. Do not edit the manuscript in `overleaf_cache/` expecting it to
reach the repo copy.

## Cluster habits that cost time to learn

**`sbatch` survives the session; `srun` does not.** An `srun` launched inside an SSH
command dies when that connection drops, and the connection drops often.

**SSH to WCSS is unreliable in bursts.** Do not set up polling loops -- several of them
at once looks like abuse and gets you nothing. Submit with `sbatch`, come back later.

**Launch training through `scripts/paper/run_arm.sh`, never a hand-written sbatch.** The
one time this was ignored, `TRAIN_STORE` was omitted, so the arm trained on `val.h5` and
was scored against `val.h5` with `role=all` -- 80% of the evaluation volumes had been
trained on, and the numbers came out 20% better than the clean protocol's. `run_arm.sh`
takes `EXTRA` for one-off Hydra overrides and `NFE` for the step grid, precisely so an
ablation does not need a second launcher.

**A run's arm is encoded only in its directory name.** `collect_tables.py` parses it;
renaming a run orphans it silently.

## What this session established

**The closed-form tail holds on real data.** Hill index 1.018 on knee tissue against the
appendix's exact 1 and the synthetic 1.006. The earlier anomaly of 2.389 was the
exactly-zero background: a chord to a zero target is radial, its argument never turns,
and including those is mixing a degenerate population into a distribution of angular
velocities.

**The flat parametrisation's cost, on three sources.** 43-49% of signal energy sits on
paths turning faster than pi, and it survives being scored in the metric the paper argues
against. Weighting an angular quantity by `|z_1|^2` reinstates exactly the `A^2` the
cylinder drops, so the choice of weighting *is* the choice of geometry; both columns are
reported and the caption says why.

**Minibatch OT has a measured ceiling.** Cost drop 12.1% / 6.1% / 2.8% / 1.4% / 0.5% at
16 / 32 / 64 / 128 / 320. At 320x320 the assignment still reorders 97% of the batch and
buys half a percent. **This splits the two contributions**: the cylinder's bound is per
pixel and scales; joint OT matches whole fields and dies with dimension.

**Three explanations tested and rejected** for the k=1 to k=2 rise: undertraining, the
background, and the trained field leaving its own pi bound. The bounded head binds hard
(9.886 to 3.142) and changes the rise by nothing.

## Two sentences that are easy to get backwards

**The k = 1 advantage is amplitude, not phase.** Against the raw prior's phase: speech
0.0335 against 0.0352 -- indistinguishable; knee 9% better; synthetic 36% better. The
metric pools pixels, so carrying a uniform prior phase to a uniform output phase matches
that marginal without learning anything, and the median `|v_theta|` at k = 1 on speech is
0.025. Do not claim a phase win at k = 1 on speech or MRI.

**The bounded-head appendix must say the dynamics change drastically and the error does
not.** Writing that both are indistinguishable discards the finding.

## Next, in order

1. Crop k-space to 64x64 -- FFT, centre crop, IFFT at load time, because the store holds
   images after ESPIRiT. Physically a lower-resolution acquisition, not interpolation.
2. Table 5 at 64x64, five seeds, `bounded_velocity: false`.
3. **Check whether joint OT removes the k = 2 rise there. This is a hypothesis, not a
   formality** -- knee amplitudes are bimodal with 17.6% exact zeros, unlike the toy, and
   that could act independently of dimension. If the rise survives, the dimensionality
   story collapses and the next session starts over on that question.
4. Write in what is measured already: section 5.3 extended to 128 and 320, the prior
   control row, the bounded-head appendix.
5. Cut 11 pages to 9, last, because what survives depends on step 3.

Keep 320x320 in the paper whatever 64x64 returns. Moving the headline down reverses a
deliberate decision (#84 was demoted to P2 because "the headline fastMRI table is at the
acquisition's own resolution") and 320 is what the score-based-prior literature reports.
Reporting only 64 invites the question whose honest answer is that ours stops working
above it.

## Not on the table, with reasons

Latent space -- Table 1 is about the data's amplitude distribution near zero, which in a
latent space is whatever the encoder made it. Schrodinger bridges -- a separate paper,
and minibatch SB carries the same concentration bias. Patch-level coupling -- measured
and broken, it drives cross-seam lag-one correlation to -0.008 against the data's 0.979.

## Outstanding, not ours

The diffusion baseline (#83) is Damian's and lands as a fifth row in the MRI block. It
blocks nothing.
