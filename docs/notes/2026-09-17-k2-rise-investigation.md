# The k = 1 to k = 2 rise, and what has been ruled out

Working notes, 2026-09-17. **Not paper material yet.** Everything here is measured;
the explanations are labelled where they are still guesses.

## The phenomenon

Sliced W2 rises when the solver is given a second step, instead of falling.

```
                          k=1      k=2
synthetic 16  independent 0.1160 -> 0.1549   rises
synthetic 16  joint OT    0.0980 -> 0.0858   falls
synthetic 32  independent 0.1233 -> 0.1452   rises
synthetic 32  joint OT    0.1134 -> 0.0940   falls
synthetic 64  independent 0.1214 -> 0.1478   rises
synthetic 64  joint OT    0.1174 -> 0.1141   falls
speech        independent 0.0458 -> 0.0409   falls
speech        joint OT    0.0454 -> 0.0403   falls
knee MRI      independent 0.0585 -> 0.1303   rises
knee MRI      joint OT    0.0561 -> 0.1225   RISES ANYWAY
```

On synthetic fields the joint OT coupling removes it at every resolution, which the
paper already reports. **Knee MRI is the only case where OT does not**, and speech is
the only case where it never appears. So the open question is not "why does the
cylinder dip on MRI" but **"why does joint OT work on synthetic fields and on speech
and not on knee MRI"**.

## Ruled out, with the measurement that ruled it out

**Undertraining.** Both arms plateau and tick upward by epoch 40 with the learning rate
at its floor: euclidean 0.01696 -> 0.01742, cylindrical 0.53660 -> 0.55289 between
epochs 30 and 40. Converged, not starved.

**The background.** 17.59% of knee coefficients are exactly zero, with phase a constant
0.0 from atan2(0, 0) -- a placeholder, not a measurement, carrying 0.000000% of the
energy. Masking it out of the metric changes almost nothing (0.0598 -> 0.0547) and is
worse for some arms at k = 100 (0.0649 -> 0.0720).

**The field leaving its own bound.** The trained |v_theta| crosses pi at exactly k = 2
in all three domains and grows to ~3 pi by k = 100. A pi*tanh head stops it dead (3.142
against 9.886) and the rise is unchanged: 0.0561 -> 0.1225 unbounded against
0.0588 -> 0.1261 bounded, seeds overlapping. Job 5904413, patient-disjoint split, 5
seeds. **The bound is a symptom, not the cause**, and the head does not earn a rerun.
The euclidean bound (2, 2) is slack as predicted -- 0.1194 against 0.1195 -- which is
what makes giving each geometry its own bound a fair move rather than a gift.

## The prior control, which belongs in the paper eventually

Sliced W2 of the raw prior against the data, no model, no integration:

```
              prior    cylinder k=1   Cartesian k=1
synthetic 64  0.1502      0.1174          0.3763
speech        0.3781      0.0454          0.0780
knee MRI      0.2239      0.0561          0.1195
```

Two things fall out.

**One Euler step in Cartesian coordinates is worse than not running the model at all**
on synthetic fields (0.3763 against 0.1502), and worse than the prior on the phase
component in all three domains. That is the sharpest statement of the few-step problem
available to us.

**The cylinder's k = 1 advantage is real but it is an amplitude advantage.** Phase at
k = 1 against the prior's phase: speech 0.0335 against 0.0352 -- indistinguishable;
knee 0.2521 against 0.2773 -- 9%; synthetic 0.2071 against 0.3238 -- 36%. Do not claim
a phase win at k = 1 on speech or MRI. The metric pools pixels into one cloud, so a
model that carries a uniform prior phase to a uniform output phase matches that
marginal without having learned anything, and the measured median |v_theta| at k = 1 on
speech is 0.025 -- it barely moves the phase at all.

## Where to look next

The question is about the coupling, not the head or the geometry. The transport-cost
reduction minibatch OT achieves is known to shrink with field dimension, and knee MRI at
320x320 is twenty-five times the pixel count of the largest synthetic field where OT
still works. If the pairing is effectively random at that size, the OT arm should behave
like the independent one -- which is what the table shows. Measuring the cost reduction
on the knee store against the synthetic one needs no training.
