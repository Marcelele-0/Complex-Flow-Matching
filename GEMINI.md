# Project Context: Complex Flow Matching (CFM)
Target Venue: ICLR 2027 Main Conference (Submission: Sep 25, 2026)
Overleaf Project ID: `6a651b7f5ac9720fb2a5d26b`
GitHub Project: [Project #5 — Complex FM](https://github.com/users/Marcelele-0/projects/5)

## 1. Core Mathematical Framework
- **Product Manifold:** $M = S^1 \times \mathbb{R}^+$ equipped with metric $g = dA^2 + d\theta^2$.
- **Geodesic Interpolation:** $A_t = (1-t)A_0 + t A_1$, $\theta_t = \exp_{\theta_0}^{S^1}(t u_\theta)$.
- **Target Velocities:** $u_A = A_1 - A_0$, $u_\theta = \operatorname{atan2}(\sin(\theta_1 - \theta_0), \cos(\theta_1 - \theta_0))$.
- **Loss:** $L(\phi) = \mathbb{E}[ |v_A - u_A| + \lambda (A_1 / \bar{A}_1) |v_\theta - u_\theta| ]$ (linear SNR heuristic).
- **Data Consistency (DC):** Exact linear projection in measured k-space ($DC < 10^{-6}$).

## 2. Active Roadmap (25 Atomic Tasks):
- Epic 0 (#19): Modular Registry Pattern & Core Engine (torchcfm-complex).
- Epic 1 (#20-#23): SKM-TEA 106 patients HDF5 streaming, fastMRI multicoil ESPIRiT, SLURM DDP on A100.
- Epic 2 (#24-#28): BaseReconstructor, Complex Diffusion, VarNet, multi-seed runs, Wilcoxon tests.
- Epic 3 (#29-#32): 500-volume synthetic cohort, nnU-Net tissue segmentation, T2* relaxation error.
- Epic 4 (#33-#36): 5-Radiologist blinded trial, Streamlit A/B app, Fleiss' Kappa agreement.
- Epic 5 (#37-#40): Cylindrical SDE solver, pixel-wise uncertainty heatmaps, DSB IPF, circular OT.
- Epic 6 (#41-#43): 9-page ICLR manuscript, 600 DPI vector figures, `pip install torchcfm-complex`.

## 3. Team & Repository Rules:
- Primary Operator: Marcel Musiałek (Marcelele-0)
- Team: Damian Ryczko, Iga Wolanin, Anna Grelewska
- Always branch per task: `feat/issue-<num>-<slug>`
- Always include `Closes #<num>` in PR to trigger auto-unblocking!
