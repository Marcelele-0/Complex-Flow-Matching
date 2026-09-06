# 🚀 Complex Flow Matching: Team & Agent Handoff Guide

> **Target Venue:** ICLR 2027 Main Conference (Submission: Sep 25, 2026)
> **Repository:** `Marcelele-0/Complex-Flow-Matching`
> **Project Board:** [GitHub Project #5 — Complex FM](https://github.com/users/Marcelele-0/projects/5)

This document is the **single source of truth** for both **Human Engineers** (Marcel, Damian, Iga, Anna) and **Autonomous AI Agents** (Gemini Pro, Claude, Copilot) on how to pick tasks, write code, submit Pull Requests, and leverage the automated Self-Driving Kanban system.

---

## 🧭 1. How the Self-Driving Kanban Board Works

Our project uses an automated GitHub Project Board (`Complex FM`) with native blocker tracking:

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│     BACKLOG     │ ───►  │      READY      │ ───►  │   IN PROGRESS   │ ───►  │      DONE       │
│ (Blocked Tasks) │       │ (Pickable Tasks)│       │(Active Branch)  │       │(Merged & Closed)│
└─────────────────┘       └─────────────────┘       └─────────────────┘       └─────────────────┘
         ▲                         │
         └─────────────────────────┘
         (Auto-Unblocked upon PR Merge)
```

1. **`Ready` Column:** Only contains tasks whose dependencies are **100% resolved**.
2. **`Backlog` Column:** Contains all future tasks with active `Blocked by #ID` relationships.
3. **Automated Trigger (`unblock-auto-ready.yml`):** When a Pull Request containing `Closes #ID` is merged into `main` (or active base), GitHub Actions automatically scans the Backlog and moves all now-unblocked tasks to **`Ready`**!

---

## 🌿 2. Branching & Issue Picking Workflow

### Step 1: Pick an Issue from `Ready`
1. Go to [Project Board #5](https://github.com/users/Marcelele-0/projects/5).
2. Assign yourself to the issue (or notify the team in chat).
3. Drag the issue from `Ready` to `In progress`.

### Step 2: Create a Feature Branch
Use the standardized branch naming convention:
```bash
git checkout main
git pull origin main
git checkout -b feat/issue-<issue_number>-<short-kebab-slug>

# Example for Issue #19:
git checkout -b feat/issue-19-modular-registry-engine
```

---

## 🛠️ 3. Code & Architecture Conventions (Registry Pattern)

All core modules are organized using the **Pluggable Registry Design Pattern**:

### Manifolds (`src/cfm/manifolds/`)
```python
from cfm.manifolds.registry import register_manifold, BaseManifold

@register_manifold("cylindrical")
class CylindricalManifold(BaseManifold):
    """Product manifold S^1 x R^+ with metric g = dA^2 + dtheta^2."""
    def metric(self, x): ...
    def geodesic_path(self, x0, x1, t): ...
    def target_velocity(self, x0, x1, t): ...
```

### Models (`src/cfm/models/`)
```python
from cfm.models.registry import register_model, BaseModel

@register_model("cylindrical_unet")
class CylindricalUNet(BaseModel): ...
```

### Solvers & Evaluators (`src/cfm/solvers/`, `src/cfm/eval/`)
- All models must implement the standardized continuous flow signature: `v_t = model(t, x_t)`.
- Reconstructors must inherit from `BaseReconstructor` and enforce strict Data Consistency: $\text{DC Error} < 10^{-6}$.
- **Statistical Significance Testing:** Granular per-slice metric records are emitted to `eval_records.csv` by `evaluate.py`. Automated Wilcoxon signed-rank tests with Holm-Bonferroni correction can be executed via `cfmri-stats` CLI, `cfmri-suite --stats --extra ...`, or `scripts/run_stats.py` (including `--demo` mode).

---

## 📝 4. Creating Pull Requests (PR Protocol)

> ⚠️ **CRITICAL RULE:** Every PR **MUST** include the exact closing keyword in its description (e.g. `Closes #19` or `Resolves #20`). This triggers the automated unblocking engine!

### PR Title Convention:
* `feat(arch): implement modular registry pattern for manifolds & models (#19)`
* `feat(data): add streaming HDF5 loader for 106 SKM-TEA patients (#20)`
* `fix(metrics): fix circular phase error radian calculation (#24)`
* `test(models): add unit tests for complex diffusion baseline (#25)`

### Standard PR Description Template:
```markdown
### 🎯 Purpose & Scope
Brief 2-3 sentence summary of the feature, model, or fix introduced in this PR.

### 🔗 Issue Linkage (MANDATORY)
Closes #<issue_number>

### 🧪 Verification & Test Results
- [x] Ran unit tests: `pytest tests/` (100% passing)
- [x] Code formatting & linting: `ruff check .` && `ruff format .`
- [x] Type checking: `mypy src/`

### 📊 Benchmark / Output (if applicable)
- PSNR / SSIM / CPE metrics or sample outputs generated.
```

### Fast PR Creation via GitHub CLI (`gh`):
```bash
gh pr create \
  --title "feat(arch): implement modular registry pattern (#19)" \
  --body "### Purpose
Implements modular registry pattern for plug-and-play manifolds, models, and solvers.

### Issue Linkage
Closes #19

### Verification
- [x] pytest tests/ passed (85/85 green)
- [x] ruff check . clean" \
  --base main
```

---

## 🤖 5. Agent-Specific Instructions (For Autonomous LLMs)

When executing tasks via Gemini/Claude/Subagents:
1. **Never edit unrelated files:** Keep diffs atomic to the assigned issue.
2. **Always run unit tests before proposing PR:**
   ```bash
   pytest tests/
   ```
3. **Ensure Clean Pre-Commit:**
   ```bash
   pre-commit run --all-files
   ```
4. **Draft the PR with the exact `Closes #<ID>` keyword** so the downstream pipeline advances smoothly.
