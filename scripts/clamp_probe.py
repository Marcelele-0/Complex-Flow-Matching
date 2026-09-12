"""How often does CylindricalODESolver's clamp at A>=0 actually fire?

Counts, per sampling run:
  - clamp_rate: fraction of (voxel, solver-step) updates whose RAW amplitude
    m_t + v_m*dt was negative, i.e. the clamp changed the value
  - zero_rate:  fraction of FINAL voxels sitting at exactly A = 0
  - min_raw:    most negative raw amplitude seen before clamping
"""

from __future__ import annotations

import sys

import torch
from hydra import compose, initialize_config_dir

from cfm.flow.solver import CylindricalODESolver
from cfm.manifolds import build_manifold
from cfm.utils.inference import build_model

CONF = "/home/marcel/Programming/papers/cfm/conf"


class CountingSolver(CylindricalODESolver):
    def __init__(self, num_steps: int) -> None:
        super().__init__(num_steps=num_steps)
        self.neg = 0
        self.total = 0
        self.min_raw = 0.0

    def step(self, x_t, v_t, dt):
        raw = x_t[:, 0:1] + v_t[:, 0:1] * dt
        self.neg += int((raw < 0).sum())
        self.total += int(raw.numel())
        self.min_raw = min(self.min_raw, float(raw.min()))
        return super().step(x_t, v_t, dt)


def main(run: str, side: int, ks: list[int], n_fields: int = 64) -> None:
    with initialize_config_dir(config_dir=CONF, version_base=None):
        cfg = compose(
            config_name="config", overrides=["+experiment=paper_unet", "manifold=cylindrical"]
        )
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifold = build_manifold(cfg).to(dev)
    model = build_model(cfg, dev, in_channels=3, out_channels=2)
    ck = torch.load(f"outputs/state/{run}/last.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    print(f"\n=== {run}  {side}x{side}  ({n_fields} fields, device={dev.type}) ===")
    print(f"{'k':>5} {'NFE':>5} {'clamp fired':>13} {'final A==0':>12} {'min raw A':>12}")
    g = torch.Generator(device=dev).manual_seed(0)
    for k in ks:
        s = CountingSolver(k)
        zeros = tot = 0
        with torch.no_grad():
            prior = manifold.sample_noise(n_fields, side, side, dev, generator=g)
            out = s.sample(model, prior)
        a = out[:, 0]
        zeros += int((a == 0).sum())
        tot += int(a.numel())
        print(
            f"{k:>5} {2*k-1:>5} {100*s.neg/max(s.total,1):>12.4f}% "
            f"{100*zeros/max(tot,1):>11.4f}% {s.min_raw:>12.5f}"
        )


if __name__ == "__main__":
    run, side = sys.argv[1], int(sys.argv[2])
    main(run, side, [1, 2, 4, 8, 100])
