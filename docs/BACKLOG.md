# Project Roadmap & Deadline Backlog

> **Primary Milestone:** NeurIPS 2026 Workshop on Geometric Distributional Deep Learning (GDDL)  
> **Submission Deadline:** September 2, 2026 (AoE)  
> **Format:** Short Paper (2–4 pages strict limit inclusive of references)

---

## 🎯 Pre-Deadline Strategy (Risk Management)

### 🛡️ Plan A (Target / Gold Standard)
- **Objective:** Full-scale generalization benchmark on held-out test split.
- **Dataset:** Full SKM-TEA (~100 train, ~30 test volumes) on PLGrid / e-Science (NVIDIA A100).
- **Steps:**
  1. Clean training from scratch with uniform time sampling (`t ~ U[0, 1]`) and `lambda_hf=0.0`.
  2. Euclidean baseline model training (Issue #9).
  3. 4x Cartesian k-space undersampling reconstruction (Issue #10).
  4. Run `src/cfm/evaluate.py split=test` to fill Table 1.
- **Trigger:** Use when cluster training finishes before the deadline.

### 🛟 Plan B (Safe Backup / Proof-of-Concept)
- **Objective:** Topological representation capacity & inductive bias benchmark.
- **Dataset:** `skm-tea-mini` (identical volume for both models).
- **Steps:**
  1. Compare Cylindrical Flow vs Euclidean Flow under identical capacity constraints.
  2. Prove that the Euclidean formulation fails on phase boundaries due to destructive interference, while Cylindrical Flow avoids it.
  3. Frame in paper as *In-Sample Topological Expressiveness / Representation Capacity*.
- **Trigger:** Fallback if cluster scheduling or data scaling is delayed.

---

## 📋 Task Backlog & Assignments

| Task | Priority | Assignee | Status | Target |
|:---|:---:|:---:|:---:|:---|
| **#7: Write THE PAPER** | P0 | Marcel / Jarvis | 🟢 4-page draft complete (3x Accept) | Overleaf synced |
| **#9: Euclidean Flow Baseline** | P0 | Damian Ryczko | 🔵 In progress | Needed for Table 1 Row 2 |
| **#10: 4x k-space Undersampling** | P1 | Open (Ready to pick up) | ⚪ Open | Reconstruction task |
| **#11: Data Consistency (DC) Metric**| P1 | Iga Wolanin | 🔵 In progress | Added to `metrics.py` |
| **#12: Multi-slice 2.5D dataset** | P2 | Anna Grelewska | 🟡 In review (PR #12) | Extension / Future work |
| **#5: Cross-slice attention** | P2 | Iga Wolanin | ⚪ Blocked on PR #12 | Extension / Future work |
