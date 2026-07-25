# T04 — Rebuild dataloader for 2.5D (adjacent slices)

**Priority:** P2
**Status:** Todo
**Depends on:** —
**Blocks:** T05 (cross-slice attention needs 3-slice packs)

## Goal

Change `SKMTEADataset` so each sample is a **pack of three neighboring slices** `(z−1, z, z+1)` instead of a single slice. This is the data side of a 2.5D / volumetric-consistency setup.

## Context

Current `__getitem__` (`src/cfm/data/dataset.py`):

- Indexes `(file_path, slice_idx)` for every slice.
- Loads `target[slice_idx, :, :, 0, 0]` (first echo, first coil) → one complex `[1, H, W]` (then transforms → cylindrical `[3, H, W]`).

For 2.5D we need three consecutive slices from the **same volume file**, with edge handling.

## What to change

### Dataset

- Build `slice_map` only for **center** indices that have valid neighbors, **or** include edges with replication / reflection of the missing neighbor (document the choice).
- `__getitem__` loads `z-1`, `z`, `z+1` (same file, first echo/coil).
- Return shape decision (pick one and document):
  - **A (preferred for T05):** stack as `[3, 3, H, W]` after cylindrical transform = `(slice, channel, H, W)`, **or**
  - **B:** `[9, H, W]` channels stacked, **or**
  - **C:** list/tuple of three `[3, H, W]` tensors.

Recommendation: keep per-slice transform (`ComplexToCylinderTransform`, normalize, crop) applied **independently** to each slice, then stack. Amplitude normalize should be consistent (same stats across the pack — decide: per-slice vs pack-level).

### Downstream impact (coordinate with T05)

- `train.py` / bridge / loss currently assume a single `[B, 3, H, W]` sample.
- Until T05 lands, either:
  - train only on the **center** slice (load 3 for future use but pass center to the model), or
  - land T04 + T05 on the same branch.

Do **not** silently change batch semantics without updating the model.

### Tests / config

- Update `tests/test_data/test_dataset.py` for pack shapes and edge policy.
- Optional Hydra flag, e.g. `dataset.num_slices: 1 | 3`, so 2D baseline still works.

## Acceptance criteria

- [ ] Dataset can yield packs of three adjacent slices with a documented edge policy.
- [ ] Transforms produce a documented tensor layout.
- [ ] Unit tests cover interior slices and edges.
- [ ] Config switch (or clear default) so existing 2D training is not broken without intent.

## Key files

- `src/cfm/data/dataset.py`
- `src/cfm/data/transforms.py`
- `conf/dataset/skm_tea.yaml`
- `tests/test_data/test_dataset.py`
- `tests/test_data/test_tranforms.py` (typo filename — rename only if you touch it)
