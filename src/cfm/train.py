import math
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
from cfm.utils.complex_ops import complex_to_cylinder

from cfm.models.cylindrical_unet import CylindricalUNet
from cfm.flow.torus_math import DecoupledCylindricalLoss

# Global flag for optional Weights & Biases dependency
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    
    # --- Setting TF32 precision ---
    precision = cfg.get("training", {}).get("matmul_precision", "highest")
    torch.set_float32_matmul_precision(precision)
    print(f"Set matmul precision to: {precision}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on: {device}")

    # --- Logging Configuration ---
    use_wandb = cfg.get("logging", {}).get("use_wandb", False)
    
    if use_wandb and HAS_WANDB:
        print("Weights & Biases logging enabled.")
        wandb.init(
            project=cfg.get("logging", {}).get("project_name", "Decoupled-Cylindrical-Flow"),
            name=cfg.get("logging", {}).get("experiment_name", "baseline_run"),
            config=OmegaConf.to_container(cfg, resolve=True)
        )
    elif use_wandb and not HAS_WANDB:
        print("Warning: W&B logging requested, but 'wandb' module is not installed. Proceeding with local logging.")
        use_wandb = False
    else:
        print("Local logging only. Weights & Biases is disabled.")

    # --- Data Pipeline ---
    pipeline = Compose([
        ComplexToCylinderTransform(),
        AmplitudeNormalize(),
        CenterCropModulo(base=16)
    ])
    
    data_dir = cfg.get("dataset", {}).get("data_dir", "../data/skm-tea-mini/v1-release")
    dataset = SKMTEADataset(data_dir=data_dir, transform=pipeline)
    
    batch_size = cfg.get("training", {}).get("batch_size", 4)
    num_workers = cfg.get("training", {}).get("num_workers", 4)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )

    # --- Model Definition ---
    base_channels = cfg.get("model", {}).get("base_channels", 64)
    model = CylindricalUNet(base_channels=base_channels).to(device)
    
    print("Compiling model via Triton (this may take a minute during the first epoch)...")
    model = torch.compile(model)

    # --- Environment & Teacher (Loss & Bridge) ---
    loss_cfg = cfg.get("training", {}).get("loss", {})
    criterion = DecoupledCylindricalLoss(
        amp_loss_type=loss_cfg.get("amp_loss_type", "l1"),
        phase_loss_type=loss_cfg.get("phase_loss_type", "l1"),
        lambda_phase=loss_cfg.get("lambda_phase", 1.0)
    ).to(device)
    
    bridge = GeodesicFlowBridge()
    
    # --- Optimizer & Scheduler ---
    lr = cfg.get("training", {}).get("learning_rate", 2e-4)
    weight_decay = cfg.get("training", {}).get("weight_decay", 1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Add learning rate scheduler from config
    scheduler_cfg = cfg.get("training", {}).get("scheduler", {})
    scheduler_type = scheduler_cfg.get("type", "CosineAnnealingLR")
    epochs = cfg.get("training", {}).get("epochs", 100)

    if scheduler_type == "CosineAnnealingLR":
        eta_min = scheduler_cfg.get("eta_min", 0.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=epochs, 
            eta_min=eta_min
        )
    else:
        # Default fallback
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # --- Main Training Loop ---
    
    for epoch in range(epochs):
        model.train()
        epoch_loss_total = 0.0
        epoch_loss_amp = 0.0
        epoch_loss_phi = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            x_1 = batch.to(device)
            b, _, h, w = x_1.shape
            
            # --- Generate Noise x_0 ---
            noise_real = torch.randn(b, 1, h, w, device=device)
            noise_imag = torch.randn(b, 1, h, w, device=device)
            complex_noise = torch.complex(noise_real, noise_imag)
            x_0 = complex_to_cylinder(complex_noise)
            
            # --- Sample Time t ---
            t_model = torch.rand(b, device=device) 
            t_bridge = t_model.view(b, 1, 1, 1)

            # --- Bridge: Get interpolated state and target velocity ---
            x_t, target_v = bridge.forward(cyl_noise=x_0, cyl_data=x_1, t=t_bridge)

            # --- Model: Predict velocity ---
            optimizer.zero_grad()
            pred_v = model(x_t, t_model)

            # --- Loss: Enforce topology with background masking ---
            # Pass x_1 (original cylindrical image) as the base for the mask
            loss, loss_amp, loss_phi = criterion(pred_v, target_v, target_x1=x_1)
            
            # --- Backpropagation ---
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # --- Update local metrics ---
            epoch_loss_total += loss.item()
            epoch_loss_amp += loss_amp.item()
            epoch_loss_phi += loss_phi.item()
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}", 
                "amp": f"{loss_amp.item():.4f}", 
                "phi": f"{loss_phi.item():.4f}"
            })

            # --- W&B Logging (Step level) ---
            if use_wandb:
                wandb.log({
                    "step_loss": loss.item(),
                    "step_loss_amp": loss_amp.item(),
                    "step_loss_phi": loss_phi.item(),
                    "learning_rate": optimizer.param_groups[0]['lr']
                })

        # --- Epoch Summary ---
        avg_loss = epoch_loss_total / len(dataloader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.5f} | LR: {current_lr:.6f}")
        
        # Step the learning rate scheduler
        scheduler.step()
        
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1, 
                "epoch_avg_loss": avg_loss,
                "learning_rate_epoch": current_lr
            })

        # Checkpointing
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            checkpoint_path = f"checkpoint_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved to: {checkpoint_path}")

    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()