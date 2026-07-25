# Team Task Board

Tracked engineering / research work for Complex Flow Matching (CFM).

Repo: https://github.com/Marcelele-0/Complex-Flow-Matching

| ID | Task | Priority | Status | Owner |
|----|------|----------|--------|-------|
| [T01](01-unify-velocity-channels-and-fix-tests.md) | Unify velocity dims (3→2) and fix unit tests | P0 | Todo | — |
| [T02](02-finish-hf-experiments-and-wandb.md) | Finish HF/FFT experiments + W&B logging | P1 | Todo | — |
| [T03](03-image-evaluation-metrics.md) | Implement PSNR / SSIM / Circular Phase Error | P1 | Todo | — |
| [T04](04-dataloader-2-5d.md) | Rebuild dataloader for 2.5D (z−1, z, z+1) | P2 | Todo | — |
| [T05](05-cross-slice-attention.md) | Cross-slice attention in Cylindrical U-Net bottleneck | P2 | Todo | — |

## Suggested order

1. **T01** first — loss/tests are out of sync with the `[B, 2]` velocity contract; do not trust new experiment results until this is green.
2. **T02** + **T03** next — finish the HF sweep and get quantitative evaluation.
3. **T04** then **T05** — 2.5D data must land before cross-slice attention can train.

## How to claim a task

1. Assign yourself in the table (or open a GitHub issue / PR linked to the task file).
2. Work on a feature branch named after the task, e.g. `feat/t01-unify-velocity-channels`.
3. Mark **Status** → `In progress` → `Done` when merged.
