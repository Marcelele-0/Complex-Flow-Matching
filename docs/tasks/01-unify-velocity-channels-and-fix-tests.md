# T01 — Unify velocity channels and fix unit tests

**Priority:** P0
**Status:** Todo
**Depends on:** —
**Blocks:** T02, T03 (reliable training/eval)

## Goal

Make `DecoupledCylindricalLoss` explicitly treat predicted/target velocity as **2 channels** `(v_amp, v_phase)` and bring unit tests back to green.

## Context

- Bridge, U-Nets, and ODE solver already use velocity shape **`[B, 2, H, W]`**.
- State / data tensors stay **`[B, 3, H, W]`** = `[amp, cos φ, sin φ]`.
- `DecoupledCylindricalLoss` in `src/cfm/flow/torus_math.py` still documents and slices as if velocity were 3-channel (`pred_v[:, 1:3]` for “cos/sin”). On `[B, 2]` this happens to use channel 1 as phase, but the API/docs/tests are inconsistent.
- Current suite: **5 failed / 18 passed** (`test_torus_math.py` ×4, `test_solver.py::test_solver_sample_loop` ×1).

## What to change

### 1. Loss (`src/cfm/flow/torus_math.py`)

- Update docstrings/types: `pred_v` / `target_v` are `[B, 2, H, W]`.
- Amplitude: channel `0` only (`[:, 0:1]`).
- Phase: channel `1` only (`[:, 1:2]`) — **angle velocity**, not cos/sin pair.
- Decide and document phase distance:
  - Prefer **circular / wrapped** distance for an angle channel if that matches the bridge target (tests already encode wrap-around expectations), **or**
  - Keep plain L1/MSE on angle and **rewrite** the topology/cosine tests to match reality.
- Keep optional HF k-space boost (`lambda_hf`, `hf_boost_factor`) working on the 2-channel error tensor.
- Optional but useful for T02: also **return `loss_hf`** (today it is computed but discarded; only `(total, amp, phi)` is returned).

### 2. Tests (`tests/test_flow/test_torus_math.py`)

- Align every test with the chosen 2-channel API.
- Fix broken assumptions:
  - `amp_loss_type="mse"` / `"huber"` if those modes are not implemented — either implement or drop from tests.
  - Phase wrap / cosine tests must match the implemented phase loss.
  - Invalid-config test must match real validation (or add validation).
- Add at least one test that HF boost changes the total loss when `lambda_hf > 0`.

### 3. Solver test (`tests/test_flow/test_solver.py`)

- Fix `test_solver_sample_loop`: zero-velocity + `randn` noise + amplitude `clamp(min=0)` currently fails the “unchanged under zero velocity” assumption.
- Use valid cylindrical noise (non-negative amp) or assert only the properties that the solver actually guarantees.

## Acceptance criteria

- [ ] Loss code and docs assume `[B, 2]` velocity only.
- [ ] `uv run pytest tests/test_flow/test_torus_math.py tests/test_flow/test_solver.py` passes.
- [ ] Full `uv run pytest tests/` is green (0 failures).

## Key files

- `src/cfm/flow/torus_math.py`
- `src/cfm/flow/bridge.py` (reference for what `target_v` actually is)
- `tests/test_flow/test_torus_math.py`
- `tests/test_flow/test_solver.py`
