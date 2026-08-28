import math
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from cfm.data.dataset import SKMTEADataset
from cfm.data.transforms import (
    AmplitudeNormalize,
    CenterCropModulo,
    ComplexToCylinderTransform,
    Compose,
)
from cfm.flow.bridge import GeodesicFlowBridge
from cfm.flow.torus_math import DecoupledCylindricalLoss
from cfm.models.cylindrical_unet import CylindricalUNet

# Optional Weights & Biases logging
try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # --- Performance Optimization ---
    precision = cfg.get("training", {}).get("matmul_precision", "highest")
    torch.set_float32_matmul_precision(precision)
    print(f"Set matmul precision to: {precision}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on: {device}")

    # --- W&B Configuration ---
    use_wandb = cfg.get("logging", {}).get("use_wandb", False)
    if use_wandb and HAS_WANDB:
        print("Weights & Biases logging enabled.")
        output_dir = cfg.get("paths", {}).get("output_dir", ".")
        os.makedirs(output_dir, exist_ok=True)
        wandb.init(
            project=cfg.get("logging", {}).get("project_name", "Cylindrical-Flow-Matching"),
            name=cfg.get("logging", {}).get("experiment_name", "uniform_noise_run"),
            dir=output_dir,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    else:
        use_wandb = False
        print("Local logging only.")

    # --- Data Pipeline ---
    # Ensure dataset loads target[slice, :, :, 0, 0] as a complex tensor!
    pipeline = Compose(
        [ComplexToCylinderTransform(), AmplitudeNormalize(), CenterCropModulo(base=16)]
    )

    data_dir = cfg.get("dataset", {}).get("data_dir", "../data/skm-tea-mini/v1-release")
    num_slices = cfg.get("dataset", {}).get("num_slices", 1)

    model_name = cfg.get("model", {}).get("name", "c_unet")
    base_channels = cfg.get("model", {}).get("base_channels", 64)

    # Model and dataloader must agree on the slice layout. Checked before the
    # dataset is opened so a config mistake fails immediately, rather than after
    # every .h5 in data_dir has been scanned; the mismatch would otherwise surface
    # as an opaque shape error deep inside a convolution.
    needs_slice_window = model_name == "c_unet_cross_slice"
    if needs_slice_window and num_slices == 1:
        raise ValueError(
            f"model={model_name} consumes slice windows but dataset.num_slices=1. "
            "Set dataset.num_slices to an odd value > 1, e.g. "
            f"'uv run src/cfm/train.py model={model_name} dataset.num_slices=3'."
        )
    if not needs_slice_window and num_slices > 1:
        raise ValueError(
            f"dataset.num_slices={num_slices} produces 5D slice windows, but "
            f"model={model_name} is a 2D model expecting [B, 3, H, W]. "
            "Use model=c_unet_cross_slice, or set dataset.num_slices=1."
        )

    dataset = SKMTEADataset(data_dir=data_dir, transform=pipeline, num_slices=num_slices)

    batch_size = cfg.get("training", {}).get("batch_size", 4)
    num_workers = cfg.get("training", {}).get("num_workers", 4)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )

    # --- Model Initialization ---
    match model_name:
        case "c_unet_cross_slice":
            from cfm.models.cylindrical_unet_cross_slice import CylindricalUNetCrossSlice

            channel_mults = cfg.get("model", {}).get("channel_mults", [1, 2, 4, 8, 8])
            attn_heads = cfg.get("model", {}).get("attn_heads", 4)

            model = CylindricalUNetCrossSlice(
                base_channels=base_channels,
                channel_mults=list(channel_mults),
                attn_heads=attn_heads,
            ).to(device)
            print(f"Instantiated CylindricalUNetCrossSlice with base_channels={base_channels}")

        case "c_unet_attention":
            from cfm.models.cylindrical_unet_attention import CylindricalUNetAttention

            channel_mults = cfg.get("model", {}).get("channel_mults", [1, 2, 4, 8, 8])
            use_attention = cfg.get("model", {}).get("use_attention", True)
            attn_heads = cfg.get("model", {}).get("attn_heads", 4)

            model = CylindricalUNetAttention(
                base_channels=base_channels,
                channel_mults=list(channel_mults),
                use_attention=use_attention,
                attn_heads=attn_heads,
            ).to(device)
            print(f"Instantiated CylindricalUNetAttention with base_channels={base_channels}")

        case "c_unet":
            model = CylindricalUNet(base_channels=base_channels).to(device)
            print(f"Instantiated standard CylindricalUNet with base_channels={base_channels}")

        case _:
            raise ValueError(
                f"Unknown model name specified in config: {model_name}. "
                "Please check your yaml configuration."
            )

    print("Compiling model via Triton (this may take a minute during the first epoch)...")
    model = torch.compile(model)

    # --- Loss & Bridge ---
    loss_cfg = cfg.get("training", {}).get("loss", {})
    criterion = DecoupledCylindricalLoss(
        amp_loss_type=loss_cfg.get("amp_loss_type", "l1"),
        phase_loss_type=loss_cfg.get("phase_loss_type", "l1"),
        lambda_phase=loss_cfg.get("lambda_phase", 1.0),
        lambda_hf=loss_cfg.get("lambda_hf", 0.0),
        hf_boost_factor=loss_cfg.get("hf_boost_factor", 4.0),
    ).to(device)

    bridge = GeodesicFlowBridge()

    # --- Optimizer & Scheduler ---
    lr = cfg.get("training", {}).get("learning_rate", 2e-4)
    weight_decay = cfg.get("training", {}).get("weight_decay", 1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler_cfg = cfg.get("training", {}).get("scheduler", {})
    epochs = cfg.get("training", {}).get("epochs", 100)
    eta_min = scheduler_cfg.get("eta_min", 1e-6)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)

    # Retrieves checkpoint directory from yaml config or falls back to current directory
    checkpoint_dir = cfg.get("paths", {}).get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Main Training Loop ---
    for epoch in range(epochs):
        model.train()
        epoch_loss_total = 0.0
        epoch_loss_amp = 0.0
        epoch_loss_phi = 0.0
        epoch_loss_hf = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for _batch_idx, batch in enumerate(pbar):
            x_1 = batch.to(device)

            # 2.5D batches are [B, S, 3, H, W]; plain 2D batches are [B, 3, H, W].
            # S is folded into the batch for the bridge, which slices channels as
            # [:, 0:1] and would otherwise index the slice axis instead.
            is_window = x_1.dim() == 5
            if is_window:
                b, s, c, h, w = x_1.shape
                center = s // 2
                flat = b * s
            else:
                b, c, h, w = x_1.shape
                s, center, flat = 1, 0, b

            noise_amp = torch.rand(flat, 1, h, w, device=device)  # [0, 1]
            noise_phi = torch.rand(flat, 1, h, w, device=device) * 2 * math.pi  # [0, 2pi]

            x_0 = torch.cat([noise_amp, torch.cos(noise_phi), torch.sin(noise_phi)], dim=1)

            # --- Time Sampling ---
            # One time per sample: every slice of a window is the same example, so
            # they share t. repeat_interleave keeps a sample's slices adjacent,
            # matching how the flattened tensors are unfolded again below.
            t_model = torch.rand(b, device=device)
            t_flat = t_model.repeat_interleave(s) if is_window else t_model
            t_bridge = t_flat.view(flat, 1, 1, 1)

            # --- Bridge: Interpolation and target velocity ---
            x_1_flat = x_1.reshape(flat, c, h, w) if is_window else x_1
            x_t, target_v = bridge.forward(cyl_noise=x_0, cyl_data=x_1_flat, t=t_bridge)

            if is_window:
                # Model takes the whole window but supervises the center slice only.
                x_t = x_t.view(b, s, c, h, w)
                target_v = target_v.view(b, s, 2, h, w)[:, center]
                x_1_sup = x_1[:, center]
            else:
                x_1_sup = x_1

            # --- Forward Pass ---
            optimizer.zero_grad()
            pred_v = model(x_t, t_model)

            # --- Loss with background masking ---
            loss, loss_amp, loss_phi, loss_hf = criterion(pred_v, target_v, target_x1=x_1_sup)

            # --- Backprop ---
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # --- Update metrics ---
            epoch_loss_total += loss.item()
            epoch_loss_amp += loss_amp.item()
            epoch_loss_phi += loss_phi.item()
            epoch_loss_hf += loss_hf.item()

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "amp": f"{loss_amp.item():.4f}",
                    "phi": f"{loss_phi.item():.4f}",
                    "hf": f"{loss_hf.item():.4f}",
                }
            )

            # Step-level logging
            if use_wandb:
                wandb.log(
                    {
                        "step_loss": loss.item(),
                        "step_loss_amp": loss_amp.item(),
                        "step_loss_phi": loss_phi.item(),
                        "step_loss_hf": loss_hf.item(),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                )

        # --- Epoch Summary ---
        avg_loss = epoch_loss_total / len(dataloader)
        avg_loss_amp = epoch_loss_amp / len(dataloader)
        avg_loss_phi = epoch_loss_phi / len(dataloader)
        avg_loss_hf = epoch_loss_hf / len(dataloader)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1} | Avg Loss: {avg_loss:.5f} (Amp: {avg_loss_amp:.5f}, "
            f"Phi: {avg_loss_phi:.5f}, HF: {avg_loss_hf:.5f}) | LR: {current_lr:.6f}"
        )

        scheduler.step()

        # Epoch-level logging and Sanity Check of the image
        if use_wandb:
            log_dict = {
                "epoch": epoch + 1,
                "epoch_avg_loss": avg_loss,
                "epoch_avg_loss_amp": avg_loss_amp,
                "epoch_avg_loss_phi": avg_loss_phi,
                "epoch_avg_loss_hf": avg_loss_hf,
                "learning_rate_epoch": current_lr,
            }
            # Once every 10 epochs, log the realistic amplitude target from the dataset
            if (epoch + 1) % 10 == 0:
                # Amplitude channel of the first image in the batch. x_1_sup is the
                # center slice for 2.5D windows, so this stays a 2D image either way.
                gt_amp_img = x_1_sup[0, 0].detach().cpu().numpy()
                log_dict["ground_truth_sample"] = wandb.Image(
                    gt_amp_img, caption=f"Epoch {epoch+1} Target Amp"
                )

            wandb.log(log_dict)

        # ---------------------------------------------------------
        # FIX: Cleanly save model without torch.compile artifacts
        # ---------------------------------------------------------
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")

            # Extract basic weights bypassing _orig_mod wrapper
            state_dict = (
                model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
            )

            torch.save(state_dict, checkpoint_path)
            print(f"Model saved cleanly to: {checkpoint_path}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
