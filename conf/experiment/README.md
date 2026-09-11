# Experiment configs

Two kinds of Hydra experiment config live here.

**Training protocols.** `+experiment=<name>` sets everything about a U-Net run except
geometry, coupling, field size and seed:

| config | objective |
| --- | --- |
| `paper_unet` | squared velocity error; on the cylinder, an unweighted phase term (the paper's) |
| `paper_unet_l1` | absolute velocity error, unweighted phase term |
| `paper_unet_v1loss` | absolute error, phase weighted by the clean amplitude (the arXiv v1 objective) |
| `paper_unet_l2w` | squared error, amplitude-weighted phase (only in the loss ablation) |

**Paper experiments.** One config per result of the paper, with a `paper:` block that
says what to run, which section it backs, its runtime and how its result is printed.
The U-Net tables also include `paper_unet`, so a single cell can be trained by hand
with `+experiment=table2_unet64`.

| config | paper result | kind | runtime |
| --- | --- | --- | --- |
| `table1_bridges` | Table 1, analytical path geometry | network-free | ~15 s |
| `table2_unet64` | Table 2, 64x64 | U-Net grid, 20 runs | ~7 h |
| `table3_unet_scaling` | Table 3, 16x16 to 64x64 (contains Table 2's runs) | U-Net grid, 60 runs | ~9 h |
| `tableA_loss_protocols` | appendix table, both losses (contains Table 3's runs) | U-Net grid, 120 runs | ~18 h |
| `table4_factorized` | Table 4 and the patch seams, Section 5.4 | network-free | ~4 min |
| `sec53_ot_cost` | OT cost saving against field size, Section 5.3 | network-free | ~10 s |
| `ablation_loss32` | the loss ablation, Section 4 | U-Net grid, 18 runs | ~1 h |

Runtimes are for one RTX 4070 Ti SUPER, one run at a time.
`scripts/paper/reproduce.py` runs them:

```bash
uv run python scripts/paper/reproduce.py --all                  # everything, in paper order
uv run python scripts/paper/reproduce.py table2_unet64          # one result
uv run python scripts/paper/reproduce.py tableA_loss_protocols \
    --seeds 0 --sides 16 --tag quick_                           # a quick check
uv run python scripts/paper/reproduce.py --all --dry-run        # what would run
```

`--seeds` and `--sides` replace an experiment's seeds and field sides, `--epochs`
overrides the 40 training epochs (smoke tests only), `--tag` prefixes the run names,
`--no-report` skips printing. A run whose evaluation already exists is skipped, so an
interrupted sweep resumes and runs shared between tables are made once. Each U-Net
run is

```bash
uv run python -m cfm.train +experiment=<protocol> manifold=<geometry> \
    training.coupling=<coupling> dataset.crop_size=[<side>,<side>] \
    training.seed=<seed> logging.experiment_name=<name>
uv run python -m cfm.evaluate +experiment=<protocol> manifold=<geometry> \
    training.coupling=<coupling> dataset.crop_size=[<side>,<side>] \
    evaluate.run_name=<name> evaluate.num_fields=64 evaluate.seed=<seed> \
    evaluate.nfe=[1,2,4,8,100] logging.experiment_name=<name>_eval
```

The archived evaluations, and how to check them without retraining, are in
`docs/reproduce/paper_results/README.md`. `tests/test_experiments.py` checks that the
U-Net experiments enumerate exactly the archived runs and that each table config
composes to its protocol.
