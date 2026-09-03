"""Flow-matching training loop, shared by every geometry.

Nothing below is specific to a manifold. The representation, the noise prior, the
probability path, the loss and the ODE step are all supplied by the
:class:`~cfm.manifolds.base.Manifold` selected in ``conf/manifold/``, so the
cylindrical and Euclidean experiments are the *same* run with one config value
changed - same data, same architecture, same optimizer, same schedule.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from cfm.data.dataset import SKMTEADataset
from cfm.data.splits import load_split_file_names, select_indices
from cfm.manifolds import build_manifold
from cfm.utils.inference import build_model

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

    # --- Reproducibility ---
    # One seed puts both geometries on element-wise identical weights everywhere
    # except init_conv, so initialisation is not a confound. Data order, the noise
    # draw and the time sample still vary with the seed - run several seeds per
    # arm and report the spread.
    seed = cfg.get("training", {}).get("seed", 0)
    loader_generator = None
    if seed is not None:
        seed = int(seed)
        torch.manual_seed(seed)
        loader_generator = torch.Generator()
        loader_generator.manual_seed(seed)
        print(f"Seeded torch with {seed}")
    else:
        print("Unseeded run (training.seed=null): not reproducible.")

    # --- Geometry ---
    # The single switch between the cylindrical model and the Euclidean baseline.
    manifold = build_manifold(cfg).to(device)

    # --- W&B Configuration ---
    use_wandb = cfg.get("logging", {}).get("use_wandb", False)
    if use_wandb and HAS_WANDB:
        print("Weights & Biases logging enabled.")
        output_dir = cfg.get("paths", {}).get("output_dir", ".")
        os.makedirs(output_dir, exist_ok=True)
        config_dict = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
        wandb.init(
            project=cfg.get("logging", {}).get("project_name", "Cylindrical-Flow-Matching"),
            name=cfg.get("logging", {}).get("experiment_name", "uniform_noise_run"),
            dir=output_dir,
            config=config_dict,
        )
    else:
        use_wandb = False
        print("Local logging only.")

    # --- Data Pipeline ---
    data_dir = cfg.get("dataset", {}).get("data_dir", "../data/skm-tea-mini/v1-release")
    num_slices = cfg.get("dataset", {}).get("num_slices", 1)
    model_name = cfg.get("model", {}).get("name", "c_unet")

    # Model and dataloader must agree on the slice layout. Checked before the
    # dataset is opened so a config mistake fails immediately, rather than after
    # every .h5 in data_dir has been scanned; the mismatch would otherwise surface
    # as an opaque shape error deep inside a convolution.
    needs_slice_window = model_name == "c_unet_cross_slice"
    if needs_slice_window and (num_slices == 1 or num_slices % 2 == 0):
        raise ValueError(
            f"model={model_name} consumes slice windows but dataset.num_slices={num_slices}. "
            "Set dataset.num_slices to an odd value > 1, e.g. "
            f"'uv run src/cfm/train.py model={model_name} dataset.num_slices=3'."
        )
    if not needs_slice_window and num_slices > 1:
        raise ValueError(
            f"dataset.num_slices={num_slices} produces 5D slice windows, but "
            f"model={model_name} is a 2D model expecting [B, C, H, W]. "
            "Use model=c_unet_cross_slice, or set dataset.num_slices=1."
        )

    # The manifold owns the transform either way, so x_1 arrives in whatever
    # representation the selected geometry trains on and everything downstream is
    # shape-agnostic. 2.5D splits the pipeline in two: normalisation needs the whole
    # stacked window at once (one peak for the window, not one per slice), so it
    # moves post-stack, still ahead of the crop.
    slice_pipeline: Callable[[torch.Tensor], torch.Tensor]
    if num_slices > 1:
        slice_pipeline, window_pipeline = manifold.build_window_transforms(crop_base=16)
    else:
        slice_pipeline = manifold.build_transform(crop_base=16)
        window_pipeline = None

    dataset_cfg = cfg.get("dataset", {})
    echo_idx = dataset_cfg.get("echo_idx", 0)
    coil_idx = dataset_cfg.get("coil_idx", 0)
    use_cache = dataset_cfg.get("use_cache", True)
    cache_dir = dataset_cfg.get("cache_dir", ".cache")
    acceleration = dataset_cfg.get("acceleration", 4)
    mask_cfg = dataset_cfg.get("mask")
    if mask_cfg is not None:
        if isinstance(mask_cfg, DictConfig):
            container = OmegaConf.to_container(mask_cfg, resolve=True)
            mask_cfg = dict(cast(dict[str, Any], container))
        elif isinstance(mask_cfg, dict):
            mask_cfg = dict(mask_cfg)

    dataset = SKMTEADataset(
        data_dir=data_dir,
        transform=slice_pipeline,
        window_transform=window_pipeline,
        num_slices=num_slices,
        acceleration=acceleration,
        mask=mask_cfg,
        echo_idx=echo_idx,
        coil_idx=coil_idx,
        use_cache=use_cache,
        cache_dir=cache_dir,
    )

    # Train only on the volumes the split manifest lists, through the same two
    # functions evaluate.py uses. Without this the loader globs every .h5 and
    # `evaluate.split=test` scores volumes that were trained on, which makes every
    # absolute number reconstruction fidelity on seen data rather than a
    # generalization result. Set dataset.split=null for a directory with no
    # annotations/ (and then say so when reporting).
    split = cfg.get("dataset", {}).get("split", "train")
    file_names = load_split_file_names(data_dir, split, config_key="dataset.split")
    indices = select_indices(dataset.slice_map, file_names, config_key="dataset.split")
    num_files = len({os.path.basename(dataset.slice_map[i][0]) for i in indices})
    print(
        f"Split '{split}': {len(indices)} of {len(dataset.slice_map)} slices "
        f"from {num_files} volume(s)."
    )
    dataset_subset: SKMTEADataset | Subset[Any] = (
        Subset(dataset, indices) if len(indices) < len(dataset.slice_map) else dataset
    )

    batch_size = cfg.get("training", {}).get("batch_size", 4)
    num_workers = cfg.get("training", {}).get("num_workers", 4)
    dataloader = DataLoader(
        dataset_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=loader_generator,
    )

    # --- Model Initialization ---
    # Identical trunk for both geometries; only the input width follows the state.
    model: torch.nn.Module = build_model(
        cfg,
        device,
        in_channels=manifold.state_channels,
        out_channels=manifold.velocity_channels,
    )

    if cfg.get("training", {}).get("compile", True):
        print("Compiling model via Triton (this may take a minute during the first epoch)...")
        compiled_model = torch.compile(model)
        model = cast(torch.nn.Module, compiled_model)
    else:
        print("torch.compile disabled (training.compile=false).")

    # --- Optimizer & Scheduler ---
    lr = cfg.get("training", {}).get("learning_rate", 2e-4)
    weight_decay = cfg.get("training", {}).get("weight_decay", 1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler_cfg = cfg.get("training", {}).get("scheduler", {})
    epochs = cfg.get("training", {}).get("epochs", 100)
    eta_min = scheduler_cfg.get("eta_min", 1e-6)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)

    # Clipping is NOT scale-invariant, and the two losses differ ~4x in magnitude
    # (the cylindrical total carries an O(pi) angular term). AdamW is otherwise
    # scale-invariant, so this is the one place the difference leaks in: at a
    # threshold that binds for one arm and not the other, they become different
    # optimizers. Pre-clip norm is logged as `grad_norm` so this can be checked.
    grad_clip = cfg.get("training", {}).get("grad_clip", 1.0)
    if grad_clip is not None:
        grad_clip = float(grad_clip)

    # Retrieves checkpoint directory from yaml config or falls back to current directory
    checkpoint_dir = cfg.get("paths", {}).get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Main Training Loop ---
    for epoch in range(epochs):
        model.train()
        epoch_loss_total = 0.0
        # Component names come from the manifold, so a geometry's own breakdown
        # reaches the logs without any geometry-specific code in this loop.
        epoch_components: dict[str, float] = {}

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        for _batch_idx, batch in enumerate(pbar):
            x_1 = batch.to(device)

            # 2.5D batches are [B, S, C, H, W]; plain 2D batches are [B, C, H, W].
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

            x_0 = manifold.sample_noise(flat, h, w, device)

            # --- Time Sampling ---
            # One time per sample: every slice of a window is the same example, so
            # they share t. repeat_interleave keeps a sample's slices adjacent,
            # matching how the flattened tensors are unfolded again below.
            t_model = torch.rand(b, device=device)
            t_flat = t_model.repeat_interleave(s) if is_window else t_model
            t_bridge = t_flat.view(flat, 1, 1, 1)

            # --- Bridge: Interpolation and target velocity ---
            x_1_flat = x_1.reshape(flat, c, h, w) if is_window else x_1
            x_t, target_v = manifold.bridge(x_0, x_1_flat, t_bridge)

            if is_window:
                # Model takes the whole window but supervises the center slice only.
                x_t = x_t.view(b, s, c, h, w)
                target_v = target_v.view(b, s, manifold.velocity_channels, h, w)[:, center]
                x_1_sup = x_1[:, center]
            else:
                x_1_sup = x_1

            # --- Forward Pass ---
            optimizer.zero_grad()
            pred_v = model(x_t, t_model)

            # --- Loss ---
            loss, components = manifold.loss(pred_v, target_v, target_x1=x_1_sup)

            # --- Backprop ---
            loss.backward()
            if grad_clip is not None:
                # clip_grad_norm_ returns the total norm BEFORE clipping.
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
            else:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")))
            optimizer.step()

            # --- Update metrics ---
            epoch_loss_total += loss.item()
            for name, value in components.items():
                epoch_components[name] = epoch_components.get(name, 0.0) + value.item()

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    **{name: f"{value.item():.4f}" for name, value in components.items()},
                }
            )

            # Step-level logging
            if use_wandb:
                wandb.log(
                    {
                        "step_loss": loss.item(),
                        **{f"step_loss_{name}": value.item() for name, value in components.items()},
                        "grad_norm": grad_norm,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                )

        # --- Epoch Summary ---
        avg_loss = epoch_loss_total / len(dataloader)
        avg_components = {name: total / len(dataloader) for name, total in epoch_components.items()}

        current_lr = optimizer.param_groups[0]["lr"]
        breakdown = ", ".join(f"{name}: {value:.5f}" for name, value in avg_components.items())
        print(f"Epoch {epoch + 1} | Avg Loss: {avg_loss:.5f} ({breakdown}) | LR: {current_lr:.6f}")

        scheduler.step()

        # Epoch-level logging and Sanity Check of the image
        if use_wandb:
            log_dict = {
                "epoch": epoch + 1,
                "epoch_avg_loss": avg_loss,
                **{f"epoch_avg_loss_{name}": value for name, value in avg_components.items()},
                "learning_rate_epoch": current_lr,
            }
            # Once every 10 epochs, log the realistic amplitude target from the dataset.
            # Taken through the manifold so the picture is a modulus in both
            # geometries, rather than channel 0 of whatever the state happens to be.
            if (epoch + 1) % 10 == 0:
                # x_1_sup is the center slice for 2.5D windows, so this stays a 2D
                # image either way.
                gt_amp_img = (
                    torch.abs(manifold.to_complex(x_1_sup[0:1]))[0, 0].detach().cpu().numpy()
                )
                log_dict["ground_truth_sample"] = wandb.Image(
                    gt_amp_img, caption=f"Epoch {epoch + 1} Target Amp"
                )

            wandb.log(log_dict)

        # ---------------------------------------------------------
        # FIX: Cleanly save model without torch.compile artifacts
        # ---------------------------------------------------------
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")

            unwrapped_model = getattr(model, "_orig_mod", model)
            state_dict = unwrapped_model.state_dict()

            torch.save(state_dict, checkpoint_path)
            print(f"Model saved cleanly to: {checkpoint_path}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
