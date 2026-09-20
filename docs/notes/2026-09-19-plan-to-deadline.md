# Six days to the paper deadline

Written 2026-09-19. Full paper due 2026-09-25. The abstract is registered and stays editable,
so the repo copy of `00_abstract.tex` is the live source.

Ordered by leverage per hour, not by day, because the compute items can run while the writing
items are being done. Every number below is measured and its source is named.

## The honest claim, to be stated identically in three places

The text currently says three different things about knee MRI. One sentence, copied verbatim
into the introduction, section 5.2 and the conclusion, ends that:

> On synthetic fields and on speech, CyFM with the joint coupling has the lowest error at
> every $k \le 8$ with all five seeds separated. On knee MRI the pooled advantage is confined
> to a single step ($1.8\times$, separated), the Cartesian arm is lower from $k=2$ on that
> metric, and the ordering reverses on spatial structure from $k=4$.

A narrow true claim scores above a broad one a reviewer disproves from our own table.

## P0 --- text consistency. Half a day, no compute. Do this first.

1. `01_introduction.tex`, contributions bullet 2, still reads "Across synthetic fields, real
   speech STFT, **and raw complex fastMRI knee wavefields**, CyFM ... at every $k \le 8$ ...
   with all five seeds separated". **This is false and Table 5 disproves it on the same page:**
   `Cart+OT 0.1120*` against our `0.1200` at $k=2$, separated. Replace with the sentence above.
2. `05_experiments.tex`, the knee paragraph, never mentions $k=2$. Every favourable result gets
   a sentence and the one adverse separated result does not. Add it plainly: at $k=2$ the
   Cartesian arm is lower with separated seeds, before the ordering reverses on structure.
3. Delete "To prevent cherry-picking and provide full transparency" --- announcing it reads as
   an admission --- and "medical imaging demands sharp anatomical edges", which borrows clinical
   authority for a task that is unconditional generation with no diagnostic endpoint.
4. The over-smoothing claim must be stated as resolution-dependent or dropped. As written it
   contradicts Appendix D, where the Cartesian arm is *closer* on all three structural metrics
   at $320\times320$. If it over-smooths everywhere, it cannot be closer there.

## P1 --- the five excluded hypotheses, in the main text. Half a day, no compute.

The $k=1 \to k=2$ rise is currently unexplained and uncommented, which reads as a numerical
fault. Named, with what has been ruled out, it reads as rigour. **Five, not four:**

| ruled out | by what measurement |
| --- | --- |
| undertraining | both arms plateau by epoch 40 with LR at its floor |
| the zero background, as a scoring effect | masking it out of the metric moves $W_2$ by $0.005$ |
| the field leaving its own $\pi$ bound | a $\pi\tanh$ head binds $9.886 \to 3.142$ and the rise is unchanged ($0.0561 \to 0.1225$ against $0.0588 \to 0.1261$) |
| field dimension | the rise survives at $64\times64$, where OT's cost reduction is $2.8\%$ rather than $0.5\%$ |
| the exact-zero spike | it survives at $64\times64$, where **no** coefficient is exactly zero |

The last two are the strongest because they cost a dedicated cohort, and they are the two
missing from the current draft. Do not offer a sixth as the explanation: the amplitude-first
account is a hypothesis, and "high marginal curvature" contradicts our own
`rem:both_flat`, which proves $R \equiv 0$ for both metrics.

## P2 --- one qualitative figure. Needs a small script, no cluster.

The manuscript is about synthesising complex medical fields and contains **no generated
sample**. The only figure is the teaser schematic. This is also the only way the
over-smoothing claim becomes something a reader can check rather than take on trust.

Layout: two rows, amplitude and phase; columns ground truth, prior, Cartesian $k=1$, Cartesian
$k=100$, CyFM $k=1$, CyFM $k=100$. Knee $64\times64$.

`cyfm.evaluate` keeps metrics and discards samples, so this needs a short script that loads a
checkpoint by run name, generates, and writes PNGs. Same pattern as
`scripts/prior_control.py`: mirror the evaluation pipeline rather than reimplement it.

## P3 --- the diffusion baseline. Blocked, cause unknown, needs a fallback.

The largest hole a reviewer will name: the paper compares two parameterisations of flow
matching against each other and no external paradigm.

**Status: the arm diverges and the cause is not established.** Measured on knee $64\times64$
with `SIGMA_MAX=53.44`: matched-Heun $53.18 / 53.40 / 957 / 405 / 778$ at $k = 1,2,4,8,100$,
and $37698$ at the native $1000$ steps, against a data scale of $0.06$. A time-convention bug
in `diffusion_solver.py` was proposed and then **withdrawn**: the state at solver step $t$
carries noise scale $\sigma(t)$, because the bridge uses $\sigma(1-\tau)$ internally while the
solver passes $\tau = 1-t$, so `diffusion(t)` does match the state. There is no known one-line
fix.

Also note PR #101 verified the plumbing thoroughly --- NFE accounting, transform byte-identity,
unconditionality --- but never printed a single $W_2$, so output quality was never checked.

Next diagnostic, cheap and local: train the diffusion arm a few epochs on synthetic $16\times16$,
where a good result is known. Diverges there too, and it is the sampler or the loss; converges,
and the knee problem is the training budget, since 40 epochs is short for a score model.

**Fallback, and decide it now rather than at midnight on the 24th:** if it is not working by
the 22nd, no diffusion row ships. Write one honest sentence that no score-based baseline is
included and why. A diverged row is worse than an absent one, and $37698$ in a table is
unrecoverable.

## P4 --- close the $320\times320$ mechanism. Speculative payoff, ~15 GPU-h.

This would turn Appendix D from a limitation into a contribution. The candidate already
predicts the sign of an effect at both matrices: amplitude weighting improves the cylindrical
phase gap at $320$ ($0.119 \to 0.068$) and worsens it at $64$ ($0.106 \to 0.166$), where there
is no exactly-zero background to suppress.

The test is a **mask, not a threshold**: exclude the phase term on coefficients that are
exactly zero, which is a non-measurement rather than a small measurement. Do not chase an
amplitude threshold --- reference coherence climbs $0.913 / 0.954 / 0.982 / 0.998$ with the cut
and never plateaus, so whoever picks the threshold picks the answer.

If the structural gap at $320$ closes, the appendix gains a mechanism and a fix. If it does
not, Appendix D stands as written, which is already honest.

## P5 --- archive the numbers. Two hours, no compute.

The knee results exist only as `metrics.json` under `outputs/evaluate/p5_*` on the cluster. The
paper's own convention is an archived JSON under `docs/reproduce/paper_results/` plus a table
rebuilder, and `scripts/paper/export_table5.py` exists for exactly this. Until that is run, a
number in Table 5 cannot be regenerated from a checkout --- which is the prior-control problem
again, and that one cost an evening.

## Traps. Things measured to be false that keep reappearing in drafts.

- **The $320$ failure is not OT's dimensional collapse.** A cylindrical arm with *no coupling*
  reaches $0.3926$ on the radial spectrum gap against $0.3882$ with OT, and a Cartesian arm
  *with* minibatch OT at the same $102{,}400$ dimensions reaches $0.0756$. Offering the
  documented collapse as the explanation is falsified by the adjacent column of our own table.
- **The $64\times64$ crop is in k-space, not in the image.** It lowers the acquisition matrix;
  it does not cut a window out of tissue. The field still covers the whole FOV, and the air is
  no longer exactly zero only because truncating the spectrum smears it.
- **Air phase at $320$ is a constant, not thermal noise.** $16.7\%$ of coefficients are exactly
  zero and `atan2(0, 0)` returns $0.0$ for every one of them.
- **`phase_lag_one` is not Moran's I.** It is the mean resultant of the phase increment,
  $\mathbb{E}[\cos(\theta_{i+1} - \theta_i)]$. Do not borrow the name of a statistic a reviewer
  can check.
- **$k^\star$ is not reportable for the knee block.** $E^\star$ is the worst arm's asymptote and
  there that arm is CyFM itself, so the statistic would compare every arm against our own
  $k=100$ error. It is dashed in Table 2 for that reason.

## Where the paper actually stands

Realistic: weak accept as the centre of the distribution, with a tail into borderline reject if
the headline stays inconsistent. P0 and P1 alone move it, because they remove a reason to
reject. P2 and P3 are what add a reason to accept.

The strongest assets are Table 1 and the factorised coupling trap --- neither depends on the
cylinder winning, and nobody else has measured either. Lead with them. Led with "we win", every
row where we do not becomes ammunition.
