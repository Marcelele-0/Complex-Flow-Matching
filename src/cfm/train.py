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
    dataset = SKMTEADataset(data_dir=data_dir, transform=pipeline)

    batch_size = cfg.get("training", {}).get("batch_size", 4)
    num_workers = cfg.get("training", {}).get("num_workers", 4)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )

    # --- Model Initialization ---
    model_name = cfg.get("model", {}).get("name", "c_unet")
    base_channels = cfg.get("model", {}).get("base_channels", 64)

    match model_name:
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

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for _batch_idx, batch in enumerate(pbar):
            x_1 = batch.to(device)
            b, _, h, w = x_1.shape

            # ---------------------------------------------------------
            # FIX: Uniform Cylindrical Noise (Distribution matching)
            # ---------------------------------------------------------
            noise_amp = torch.rand(b, 1, h, w, device=device)  # [0, 1]
            noise_phi = torch.rand(b, 1, h, w, device=device) * 2 * math.pi  # [0, 2pi]

            x_0 = torch.cat([noise_amp, torch.cos(noise_phi), torch.sin(noise_phi)], dim=1)

            # --- Time Sampling ---
            t_model = torch.rand(b, device=device)
            t_bridge = t_model.view(b, 1, 1, 1)

            # --- Bridge: Interpolation and target velocity ---
            x_t, target_v = bridge.forward(cyl_noise=x_0, cyl_data=x_1, t=t_bridge)

            # --- Forward Pass ---
            optimizer.zero_grad()
            pred_v = model(x_t, t_model)

            # --- Loss with background masking ---
            loss, loss_amp, loss_phi = criterion(pred_v, target_v, target_x1=x_1)

            # --- Backprop ---
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # --- Update metrics ---
            epoch_loss_total += loss.item()
            epoch_loss_amp += loss_amp.item()
            epoch_loss_phi += loss_phi.item()

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "amp": f"{loss_amp.item():.4f}",
                    "phi": f"{loss_phi.item():.4f}",
                }
            )

            # Step-level logging
            if use_wandb:
                wandb.log(
                    {
                        "step_loss": loss.item(),
                        "step_loss_amp": loss_amp.item(),
                        "step_loss_phi": loss_phi.item(),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                )

        # --- Epoch Summary ---
        avg_loss = epoch_loss_total / len(dataloader)
        avg_loss_amp = epoch_loss_amp / len(dataloader)
        avg_loss_phi = epoch_loss_phi / len(dataloader)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1} | Avg Loss: {avg_loss:.5f} (Amp: {avg_loss_amp:.5f}, "
            f"Phi: {avg_loss_phi:.5f}) | LR: {current_lr:.6f}"
        )

        scheduler.step()

        # Epoch-level logging and Sanity Check of the image
        if use_wandb:
            log_dict = {
                "epoch": epoch + 1,
                "epoch_avg_loss": avg_loss,
                "epoch_avg_loss_amp": avg_loss_amp,
                "epoch_avg_loss_phi": avg_loss_phi,
                "learning_rate_epoch": current_lr,
            }
            # Once every 10 epochs, log the realistic amplitude target from the dataset
            if (epoch + 1) % 10 == 0:
                gt_amp_img = (
                    x_1[0, 0].detach().cpu().numpy()
                )  # Extract the first channel (Amplitude) of the first batch image
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
