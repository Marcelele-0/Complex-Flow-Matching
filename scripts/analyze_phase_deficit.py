"""Is the phase deficit a constant bias, or is it blur? (issue #73)

Both geometries lose circular phase error to zero-filling on held-out data while
winning on magnitude. Two mechanisms predict that pattern and imply different
fixes, and per-slice aggregates cannot tell them apart:

* **Regression to the conditional mean (blur).** Phase damage should concentrate
  where the target amplitude is low and there is little signal to anchor it, and
  the reconstruction should carry systematically less high-frequency energy than
  the ground truth.
* **A systematic phase bias.** The deficit should be roughly flat across
  amplitude, and high-frequency energy need not be depressed at all.

This script measures both signatures on checkpoints that already exist. It
reuses ``evaluate.py``'s own dataset, manifold, solver and reconstruction path,
so the numbers refer to the same reconstructions that produced the scored
metrics rather than to a re-implementation of them.

Usage::

    uv run python scripts/analyze_phase_deficit.py \\
        --run-name g63_cylindrical_R8 --manifold cylindrical --acceleration 8

Reported per amplitude decile *within tissue* (target amplitude above
``--mask-threshold`` of the per-slice maximum), because air carries no phase to
predict and would otherwise dominate every bin.
"""

from __future__ import annotations

import argparse
import os

import torch
from hydra import compose, initialize_config_dir
from torch.utils.data import DataLoader, Subset

from cfm.data import build_dataset, build_geometry_transform
from cfm.data.splits import load_split_file_names, select_indices
from cfm.evaluate import reconstruct_batch
from cfm.manifolds import build_manifold
from cfm.utils.fft import fft2c
from cfm.utils.inference import build_model, load_weights, resolve_checkpoint
from cfm.utils.metrics import shortest_angular_difference


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True, help="Training run holding the checkpoint.")
    p.add_argument(
        "--manifold", default="cylindrical", help="Geometry the checkpoint was trained under."
    )
    p.add_argument(
        "--acceleration", type=int, default=8, help="R for both the alias and the DC mask."
    )
    p.add_argument("--max-samples", type=int, default=128, help="Held-out slices, strided.")
    p.add_argument("--num-steps", type=int, default=100, help="Heun steps, as in evaluate.py.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--mask-threshold", type=float, default=0.05, help="Tissue cut, fraction of peak."
    )
    p.add_argument("--bins", type=int, default=10, help="Amplitude quantile bins within tissue.")
    p.add_argument(
        "--hf-radius", type=float, default=0.25, help="HF band starts at this fraction of Nyquist."
    )
    return p.parse_args()


def high_frequency_energy(image: torch.Tensor, radius: float) -> torch.Tensor:
    """Fraction of spectral energy outside a centred disc of the given radius.

    A model that regresses to the conditional mean is smoother than the truth and
    therefore carries less of its energy at high spatial frequency. This is the
    direct test of that, rather than an inference from a metric.

    Args:
        image: Complex image ``[B, 1, H, W]``.
        radius: Inner radius of the high-frequency band, as a fraction of Nyquist.

    Returns:
        Per-sample fraction in ``[0, 1]``, shape ``[B]``.
    """
    spectrum = fft2c(image).abs() ** 2
    _, _, h, w = spectrum.shape
    fy = torch.fft.fftshift(torch.fft.fftfreq(h, device=image.device)) * 2.0
    fx = torch.fft.fftshift(torch.fft.fftfreq(w, device=image.device)) * 2.0
    rr = (fy[:, None] ** 2 + fx[None, :] ** 2).sqrt()
    band = (rr > radius).to(spectrum.dtype)
    total = spectrum.sum(dim=(-2, -1, -3))
    return (spectrum * band).sum(dim=(-2, -1, -3)) / total.clamp_min(1e-12)


def main() -> None:
    """Accumulate per-pixel phase error by amplitude bin, and HF energy, then report."""
    args = parse_args()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with initialize_config_dir(config_dir=os.path.join(repo_root, "conf"), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"manifold={args.manifold}",
                "training.bridge=aliased",
                f"evaluate.run_name={args.run_name}",
                f"+evaluate.mask.acceleration={args.acceleration}",
            ],
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifold = build_manifold(cfg).to(device)
    model = build_model(
        cfg, device, in_channels=manifold.state_channels, out_channels=manifold.velocity_channels
    )
    load_weights(model, resolve_checkpoint(cfg, "evaluate", repo_root), device)

    data_dir = os.path.join(repo_root, cfg.dataset.data_dir)
    mask_spec = {"type": cfg.dataset.mask.type, "acceleration": args.acceleration}
    dataset = build_dataset(
        cfg.dataset,
        data_dir=data_dir,
        mode="reconstruction",
        acceleration=args.acceleration,
        mask=mask_spec,
        pre_transform=build_geometry_transform(cfg.dataset.get("crop_size"), crop_base=16),
    )
    indices = select_indices(
        dataset.slice_map, load_split_file_names(data_dir, "test"), args.max_samples
    )
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False)
    print(f"Scoring {len(indices)} held-out slices at R={args.acceleration} ({args.manifold}).")

    solver = manifold.make_solver(args.num_steps)
    edges = torch.linspace(0, 1, args.bins + 1)[1:-1]  # interior quantile cuts

    err_model = torch.zeros(args.bins, dtype=torch.float64)
    err_zf = torch.zeros(args.bins, dtype=torch.float64)
    counts = torch.zeros(args.bins, dtype=torch.float64)
    hf: dict[str, list[torch.Tensor]] = {"target": [], "model": [], "zero_filled": []}

    with torch.no_grad():
        for batch in loader:
            target = batch["target"].to(device)
            alias = batch["input"].to(device)
            mask = batch["mask"].to(device)

            pred_state = reconstruct_batch(
                model,
                manifold,
                solver,
                manifold.from_complex(target),
                0.0,
                x_alias=manifold.from_complex(alias),
                sampling_mask=mask,
                y_measured_kspace=fft2c(target) * mask,
                use_dc_projection=True,
                bridge_endpoint="aliased",
            )
            pred = manifold.to_complex(pred_state)

            amp = target.abs()
            peak = amp.amax(dim=(-2, -1), keepdim=True)
            tissue = amp > args.mask_threshold * peak

            d_model = shortest_angular_difference(pred.angle(), target.angle()).abs()
            d_zf = shortest_angular_difference(alias.angle(), target.angle()).abs()

            # Bin by amplitude relative to the per-slice peak, over tissue only.
            rel = (amp / peak.clamp_min(1e-12))[tissue]
            cuts = torch.quantile(rel.float(), edges.to(rel.device))
            idx = torch.bucketize(rel, cuts)

            err_model += (
                torch.zeros(args.bins)
                .to(rel.device)
                .scatter_add_(0, idx, d_model[tissue].float())
                .double()
                .cpu()
            )
            err_zf += (
                torch.zeros(args.bins)
                .to(rel.device)
                .scatter_add_(0, idx, d_zf[tissue].float())
                .double()
                .cpu()
            )
            counts += (
                torch.zeros(args.bins)
                .to(rel.device)
                .scatter_add_(0, idx, torch.ones_like(rel, dtype=torch.float32))
                .double()
                .cpu()
            )

            for name, img in (("target", target), ("model", pred), ("zero_filled", alias)):
                hf[name].append(high_frequency_energy(img, args.hf_radius).cpu())

    m = (err_model / counts.clamp_min(1)).numpy()
    z = (err_zf / counts.clamp_min(1)).numpy()

    per_bin = int(counts.mean())
    print(f"\nPer-pixel |dtheta| by target-amplitude decile, tissue only (n per bin ~ {per_bin})")
    print(f"{'decile':<10}{'ZF':>10}{'model':>10}{'deficit':>12}")
    for i in range(args.bins):
        print(f"{'D' + str(i + 1):<10}{z[i]:>10.4f}{m[i]:>10.4f}{m[i] - z[i]:>+12.4f}")
    print(
        f"{'ALL':<10}{(err_zf.sum() / counts.sum()):>10.4f}"
        f"{(err_model.sum() / counts.sum()):>10.4f}"
        f"{((err_model.sum() - err_zf.sum()) / counts.sum()):>+12.4f}"
    )

    print(f"\nHigh-frequency energy fraction (outside {args.hf_radius:.2f} of Nyquist)")
    for name in ("target", "model", "zero_filled"):
        v = torch.cat(hf[name])
        print(
            f"  {name:<12}{v.mean():.5f}  (ratio to target: "
            f"{(v.mean() / torch.cat(hf['target']).mean()):.3f})"
        )

    print(
        "\nReading: a deficit that rises steeply toward low deciles, together with a "
        "model HF ratio well below 1, is blur. A flat deficit with HF near 1 is bias."
    )


if __name__ == "__main__":
    main()
