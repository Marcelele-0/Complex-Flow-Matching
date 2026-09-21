# CyFM

Cylindrical flow matching for complex-valued fields.

Complex signals are usually generated as two real channels `(Re z, Im z)`.
Straight Cartesian paths then pass near the origin, where the induced angular
velocity is unbounded. CyFM runs flow matching on the cylinder `R+ x S^1`
instead, where the phase target is bounded by `pi`, and couples noise and data by
exact minibatch optimal transport computed jointly over whole fields in the
cylindrical metric.

This reference is generated from the docstrings. They are written to carry the
*argument* for a choice rather than a restatement of the signature, so the source
is shown beside each one.

## Where to start

- **[Contracts](contracts.md)** — what a geometry, a sampler, a coupling and a
  pipeline must do. Everything else is written against these.
- **[Geometry](geometry.md)** — the cylinder, the plane, and the score-based
  baseline that shares the plane's representation.
- **[Flow](flow.md)** — bridges, couplings, solvers and the transport primitives.
- **[Metrics](metrics.md)** — grouped by what each metric can see.
- **[Pipelines](pipelines.md)** — the training run and the evaluation sweep.
- **[Experiments](experiments.md)** — the network-free measurements behind the
  paper's analytical tables.

## Reproducing the paper

The repository carries one script per published result under `reproducibility/`,
each comparing a recomputed number against the paper and reporting `PASS`, `FAIL`
or `MISSING INPUT`. See that directory's README.
