# 🔬 Complex Flow Matching (CFM) — Project Context & Deep Memory

> **Primary Operator:** Marcel Musiałek (Marcelele-0)
> **Assistant:** Hydro (Lead Research Scientist & AI Architect)
> **Target Venue:** ICLR 2027 Main Conference (Submission: Sep 25, 2026)
> **Overleaf Project:** `6a651b7f5ac9720fb2a5d26b`
> **GitHub Project Board:** [Project #5 — Complex FM](https://github.com/users/Marcelele-0/projects/5)

---

## 🏛️ 1. Theoretical & Mathematical Foundation
- **Signal Domain:** Complex MRI signals $z = A e^{i\theta} \in \mathbb{C} \setminus \{0\}$.
- **Manifold Formulation:** Cylindrical product manifold $M = S^1 \times \mathbb{R}^+$ equipped with decoupled product metric $g = dA^2 + d\theta^2$.
- **Geodesic Trajectories:**
  $$A_t = (1-t)A_0 + t A_1, \quad \theta_t = \exp_{\theta_0}^{S^1}(t u_\theta)$$
- **Target Velocities:**
  $$u_A = A_1 - A_0, \quad u_\theta = \operatorname{atan2}(\sin(\theta_1 - \theta_0), \cos(\theta_1 - \theta_0))$$
- **Bounded-Velocity Advantage:** $|u_\theta| \le \pi$ globally bounded, whereas Euclidean velocity $\dot{\theta} = \mathcal{O}(A^{-1})$ diverges near the origin $A \to 0$.
- **Analytic Midpoint Attenuation:** Expected magnitude drop at $t=0.5$ in flat models is $1 - \frac{2}{\pi} \approx 36.33\%$.
- **Loss Function:**
  $$\mathcal{L}(\phi) = \mathbb{E}_{t, x_0, x_1} \left[ |v_A - u_A| + \lambda \frac{A_1}{\bar{A}_1} |v_\theta - u_\theta| \right], \quad \lambda = 1.0$$
  *(Linear amplitude weighting acts as SNR-aware heuristic derived from phase Fisher information).*
- **Data Consistency (DC):** Exact linear projection in measured Cartesian k-space ($DC < 10^{-6}$) applied at inference steps.

---

## 📊 2. Current Empirical Baseline (SKM-TEA-mini Held-Out Split)
*4x Cartesian Undersampling with exact Data Consistency across 512 test slices:*
- **Euclidean Baseline ($\mathbb{R}^2$):** PSNR: $24.13 \pm 1.18\text{ dB}$ | SSIM: $0.530 \pm 0.06$ | NMSE: $0.028 \pm 0.009$ | CPE: $0.513 \pm 0.07\text{ rad}$
- **Cylindrical Flow (Ours):** **PSNR: $24.47 \pm 1.16\text{ dB}$** | **SSIM: $0.588 \pm 0.04$** (+11%) | **NMSE: $0.021 \pm 0.006$** (-25%) | **CPE: $0.405 \pm 0.05\text{ rad}$** (-21%) | $p < 0.0001$ (Wilcoxon paired).

---

## 🗺️ 3. ICLR 2027 Masterplan & Active Tasks (Project Board #5)

### 🟢 READY (Active Sprint Focus):
- **#19:** `[arch] Modular Registry Pattern & Core Engine (torchcfm-complex)` $\to$ **IMMEDIATE START**
- **#20:** `[data] Streaming HDF5 loader for full SKM-TEA cohort (106 patients)`

### 📦 BACKLOG (Blocked / Future Phases):
- **Epic 1 (Data Scaling):** #21 (fastMRI Multicoil ESPIRiT), #22 (SLURM DDP on A100), #23 (Multi-acceleration masks).
- **Epic 2 (Baselines & Stats):** #24 (BaseReconstructor API), #25 (Complex Diffusion), #26 (VarNet ceiling), #27 (Multi-seed 42/123/999), #28 (Wilcoxon statistical testing).
- **Epic 3 (Synthetic Data & Downstream):** #29 (Unconditional prior), #30 (500 HDF5 volume exporter), #31 (nnU-Net cartilage segmentation), #32 (T2/T2* relaxation error).
- **Epic 4 (Radiologist Trial):** #33 (Likert rubric protocol), #34 (Streamlit A/B platform), #35 (5 Radiologists study), #36 (Fleiss' Kappa agreement).
- **Epic 5 (Manifold SDE & Uncertainty):** #37 (Cylindrical SDE solver), #38 (Pixel-wise uncertainty heatmaps), #39 (DSB IPF fitting), #40 (Circular OT coupling).
- **Epic 6 (Full Paper & Release):** #41 (9-page ICLR manuscript in `NIPS Main/`), #42 (600 DPI vector figures & maps), #43 (`pip install torchcfm-complex`).

---

## 🛠️ 4. Code & Workflow Conventions
- **Feature Branches:** `feat/issue-<num>-<slug>`
- **PR Requirement:** Include `Closes #<num>` in PR description to trigger automatic unblocking!
- **Test Suite:** `pytest tests/` (85/85 green).
