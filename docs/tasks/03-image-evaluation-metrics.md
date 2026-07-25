# T03 — Image evaluation metrics (PSNR, SSIM, Circular Phase Error)

**Priority:** P1
**Status:** Todo
**Depends on:** T01 (stable tensors); useful after T02 checkpoints exist
**Blocks:** quantitative paper claims / run comparison

## Goal

Fill the empty `src/cfm/utils/metrics.py` with metrics so we can **numerically** score generated knee MRI (magnitude + phase), not only eyeball PNGs.

## Context

- `metrics.py` is currently empty; README mentions validation metrics that do not exist.
- Generated outputs are complex images recovered via `cylinder_to_complex`.
- Amplitude lives on \(\mathbb{R}^+\); phase on \(S^1\) — phase error must be **circular**, not plain L1/L2 on angle.

## What to implement

### Core functions (suggested API)

```python
def peak_signal_noise_ratio(pred_amp, target_amp, data_range: float | None = None) -> float | Tensor
def structural_similarity(pred_amp, target_amp, data_range: float | None = None) -> float | Tensor
def circular_phase_error(pred_phase, target_phase, mask: Tensor | None = None) -> float | Tensor
```

Requirements:

- **PSNR / SSIM** on **amplitude** (magnitude) maps.
- **Circular Phase Error**: mean absolute shortest angular distance on \([0, 2\pi)\) (or \([-\pi, \pi)\)); support an optional **amplitude mask** so background does not dominate.
- Accept batched tensors `[B, H, W]` or `[B, 1, H, W]`; document expected ranges (normalized amp vs raw).
- Prefer pure PyTorch; if using `torchmetrics` / `skimage`, add the dependency explicitly in `pyproject.toml`.

### Wiring (minimal)

- Unit tests in `tests/test_utils/test_metrics.py` (known analytic cases: identical images → PSNR→∞ or capped, SSIM→1, phase wrap 0 vs 2π → 0 error).
- Small eval helper or Hydra entry (can live in `generate.py` or a new `src/cfm/evaluate.py`) that:
  - loads a checkpoint,
  - generates N samples **or** compares reconstruction on a hold-out batch if available,
  - prints / logs PSNR, SSIM, circular phase error.

Full train/val split is nice-to-have here but not required for v1 (can evaluate generated vs random GT batch as a sanity metric; prefer paired eval once a val split exists).

## Acceptance criteria

- [ ] `metrics.py` implements PSNR, SSIM, Circular Phase Error with clear docstrings.
- [ ] Unit tests cover shape, identity, and phase wrap cases.
- [ ] At least one scripted path reports the three metrics for a checkpoint (CLI or notebook cell is fine).

## Key files

- `src/cfm/utils/metrics.py` (empty — fill this)
- `src/cfm/utils/complex_ops.py`
- `src/cfm/generate.py`
- `tests/test_utils/` (add `test_metrics.py`)
