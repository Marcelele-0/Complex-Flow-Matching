# T05 — Cross-slice attention in Cylindrical U-Net

**Priority:** P2
**Status:** Todo
**Depends on:** T04 (2.5D packs `(z−1, z, z+1)`)
**Blocks:** 3D-consistent generation experiments

## Goal

Rebuild the Cylindrical U-Net so that:

1. **Encoders** still process each slice with **2D** convolutions (shared weights across the three slices).
2. At the **bottleneck**, a **cross-slice attention** mechanism exchanges information between `(z−1, z, z+1)`.
3. **Decoders** produce outputs per slice (primary target: center slice velocity, or all three — decide and document).

This preserves volumetric consistency without a full 3D U-Net.

## Context

- Current models: `CylindricalUNet` / `CylindricalUNetAttention` in `src/cfm/models/`.
- Input today: `[B, 3, H, W]` cylindrical state → output velocity `[B, 2, H, W]`.
- After T04, input becomes a 3-slice pack (e.g. `[B, 3, 3, H, W]` or equivalent).

## Proposed architecture

```
for each slice in {z-1, z, z+1}:
    h_s = Shared2DEncoder(x_s, t)     # independent 2D paths, shared weights

H = stack(h_{z-1}, h_z, h_{z+1})      # [B, S=3, C, h', w'] or tokens

H' = CrossSliceAttention(H)           # attend over slice axis (and optionally spatial tokens)

for each slice (or center only):
    v_s = Shared2DDecoder(H'_s, skips_s, t)
```

Design choices to lock in the PR description:

| Choice | Options | Suggestion |
|--------|---------|------------|
| Attention scope | slice-only vs slice×spatial tokens | Start with slice attention on pooled/spatial tokens at bottleneck only |
| Output | center slice only vs all 3 | Center-only loss first (cheaper, matches “predict middle given context”) |
| Which U-Net | replace attention variant vs new `c_unet_cross_slice` config | New Hydra model config; keep old 2D models working |
| Skips | per-slice skip connections | Yes — do not mix skips across slices until bottleneck |

## What to change

- New module or extension under `src/cfm/models/` (e.g. `cylindrical_unet_cross_slice.py`).
- Hydra config under `conf/model/`.
- Wire model selection in `train.py` / `generate.py`.
- Adapt bridge/loss call so the **supervised** target matches the chosen output (typically center-slice `x_1` / `target_v`).
- Unit test: forward shape check for pack input → `[B, 2, H, W]` (center) or `[B, 3, 2, H, W]`.

## Acceptance criteria

- [ ] Shared 2D encoder/decoder paths; cross-slice attention only at bottleneck (as designed).
- [ ] Trainable end-to-end on 2.5D packs from T04 (at least a short smoke run).
- [ ] Old single-slice models still selectable via Hydra.
- [ ] Shape unit test + brief note on memory cost vs baseline attention U-Net.

## Key files

- `src/cfm/models/cylindrical_unet_attention.py` (reference)
- `src/cfm/models/cylindrical_unet.py` (reference)
- New: `src/cfm/models/cylindrical_unet_cross_slice.py` (or similar)
- `conf/model/`
- `src/cfm/train.py`, `src/cfm/generate.py`
- `src/cfm/data/dataset.py` (T04 contract)
