"""Flow-matching training loop, shared by every geometry.

Nothing below is specific to a manifold. The representation, the noise prior, the
probability path, the loss and the ODE step are all supplied by the
:class:`~cfm.manifolds.base.Manifold` selected in ``conf/manifold/``, so the
cylindrical and Euclidean experiments are the *same* run with one config value
changed - same data, same architecture, same optimizer, same schedule.

The loop is also shared by every scale. Under ``torchrun`` the model is wrapped in
DistributedDataParallel and the split is sharded by a DistributedSampler; without it
``world_size`` is 1, no process group exists and the same statements run unchanged.
``training.batch_size`` is therefore **per GPU**, and the effective batch is
``batch_size * world_size`` - the learning rate is not rescaled for you.
"""

from __future__ import annotations

import os
import signal
from collections.abc import Callable
from types import FrameType
from typing import Any, cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset
from tqdm import tqdm

from cfm.core.dataset import BaseComplexDataset
from cfm.data import build_dataset, build_geometry_transform
from cfm.data.splits import load_split_file_names, select_indices
from cfm.data.transforms import Compose
from cfm.flow import BRIDGE_ENDPOINTS
from cfm.flow.coupling import build_coupling
from cfm.manifolds import build_manifold
from cfm.utils.checkpoint import load_training_state, save_training_state, state_path
from cfm.utils.distributed import (
    any_across_ranks,
    cleanup_distributed,
    print_main,
    setup_distributed,
    sum_across_ranks,
    unwrap_model,
)
from cfm.utils.inference import build_model

# Optional Weights & Biases logging
try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# Set by the handler below, read at the end of every epoch. Slurm's grace signal
# arrives some minutes before the wall clock, which is the window in which the run
# has to stop of its own accord and leave a loadable resume file behind.
_STOP_REQUESTED = False


def _request_stop(signum: int, _frame: FrameType | None) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print_main(f"\nSignal {signal.Signals(signum).name} received: stopping after this epoch.")


def _stop_requested(preempt_file: str | None) -> bool:
    """Whether this rank has been asked to stop and leave a resumable state behind.

    Two channels, because on a cluster the signal has to survive being relayed from
    the batch script through ``srun``, ``uv`` and the ``torchrun`` agent before it
    reaches this process, and it does not: the agent has no SIGUSR1 handler, so the
    workers are killed outright instead of being allowed to finish the epoch. The
    batch script therefore touches a file that no launcher can swallow, and the
    signal handler stays for the runs that have no batch script in the way.
    """
    if _STOP_REQUESTED:
        return True
    return preempt_file is not None and os.path.exists(preempt_file)


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # First statement of the run: it calls torch.cuda.set_device, which every later
    # allocation depends on, and it decides which rank is allowed to print.
    ctx = setup_distributed()
    device = ctx.device

    # SIGTERM is what scancel and most preemption policies send; SIGUSR1 is the
    # convention for a grace warning. Neither survives the cluster launcher chain -
    # see _stop_requested - so these cover direct runs and training.preempt_file
    # covers the batch job.
    signal.signal(signal.SIGUSR1, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    preempt_file = cfg.get("training", {}).get("preempt_file")

    print_main(OmegaConf.to_yaml(cfg))

    # --- Performance Optimization ---
    precision = cfg.get("training", {}).get("matmul_precision", "highest")
    torch.set_float32_matmul_precision(precision)
    print_main(f"Set matmul precision to: {precision}")

    print_main(
        f"Starting training on: {device} "
        f"(rank {ctx.rank + 1}/{ctx.world_size}, local rank {ctx.local_rank})"
    )

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
        print_main(f"Seeded torch with {seed}")
    else:
        print_main("Unseeded run (training.seed=null): not reproducible.")

    # --- Geometry ---
    # The single switch between the cylindrical model and the Euclidean baseline.
    manifold = build_manifold(cfg).to(device)

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

    # Validated here rather than at first use: the alternative is discovering an
    # unusable combination after the dataset has been indexed and the model built.
    bridge_endpoint = str(cfg.get("training", {}).get("bridge", "noise"))
    if bridge_endpoint not in BRIDGE_ENDPOINTS:
        raise ValueError(
            f"training.bridge must be one of {list(BRIDGE_ENDPOINTS)}, got {bridge_endpoint!r}."
        )
    conditional_bridge = bridge_endpoint == "aliased"
    if conditional_bridge and num_slices > 1:
        raise ValueError(
            "training.bridge='aliased' needs the dataset's reconstruction mode, which is "
            f"single-slice only, but dataset.num_slices={num_slices}. Set "
            "dataset.num_slices=1, or train the 2.5D model with training.bridge=noise."
        )

    # --- Coupling ---
    # Which prior sample is paired with which datum. The default reproduces plain
    # flow matching; an optimal-transport coupling removes the crossings that make
    # the regression target an average of conflicting velocities.
    coupling_name = str(cfg.get("training", {}).get("coupling", "independent"))
    coupling = build_coupling(coupling_name)
    reorders = coupling_name not in ("independent", "none")
    if reorders and conditional_bridge:
        raise ValueError(
            f"training.coupling={coupling_name!r} reorders the data batch, but "
            "training.bridge='aliased' fixes each pairing by construction: the alias "
            "belongs to its own target. Use training.bridge=noise."
        )
    if reorders and num_slices > 1:
        raise ValueError(
            f"training.coupling={coupling_name!r} pairs whole samples, but "
            f"dataset.num_slices={num_slices} spreads one sample across a window. "
            "Set dataset.num_slices=1."
        )
    print_main(f"Coupling: {coupling_name}")

    # The manifold owns the transform either way, so x_1 arrives in whatever
    # representation the selected geometry trains on and everything downstream is
    # shape-agnostic. 2.5D splits the pipeline in two: normalisation needs the whole
    # stacked window at once (one peak for the window, not one per slice), so it
    # moves post-stack, still ahead of the crop.
    #
    # dataset.crop_size runs first, on the complex slice: cohorts whose volumes do
    # not share a matrix size cannot be collated without it, and cropping before
    # normalisation is what keeps the peak modulus a property of the field of view
    # actually trained on rather than of the oversampled readout around it.
    dataset_cfg = cfg.get("dataset", {})
    geometry = build_geometry_transform(dataset_cfg.get("crop_size"), crop_base=16)

    slice_pipeline: Callable[[torch.Tensor], torch.Tensor]
    if num_slices > 1:
        slice_pipeline, window_pipeline = manifold.build_window_transforms(crop_base=16)
    else:
        slice_pipeline = manifold.build_transform(crop_base=16)
        window_pipeline = None
    slice_pipeline = Compose([geometry, slice_pipeline])

    # Rank 0 downloads the dataset if needed, and all other ranks wait at the barrier,
    # preventing race conditions on concurrent downloads or extractions under DDP.
    if ctx.is_distributed:
        if ctx.is_main:
            from cfm.data.download import ensure_dataset_exists

            data_dir_str = str(data_dir)
            if "skm-tea-mini" in data_dir_str:
                ensure_dataset_exists("skm_tea", data_dir)
            elif "fastmri_local" in data_dir_str:
                ensure_dataset_exists("fastmri", data_dir, mode="local")
            elif "fastmri_full" in data_dir_str:
                ensure_dataset_exists("fastmri", data_dir, mode="full")
        torch.distributed.barrier()

    # Two dataset modes for two bridges. Generation mode hands back a state tensor
    # already in the manifold's representation, because the geometry pipeline runs
    # inside the dataset. Reconstruction mode hands back complex `input`/`target`
    # tensors sharing one amplitude scale, and the loop maps both through
    # `from_complex` itself - the same arrangement evaluate.py uses, which is the
    # point: the conditional bridge only helps if training and evaluation construct
    # their states identically.
    dataset: BaseComplexDataset
    if conditional_bridge:
        dataset = build_dataset(
            dataset_cfg,
            data_dir=data_dir,
            mode="reconstruction",
            pre_transform=geometry,
            num_slices=1,
        )
    else:
        dataset = build_dataset(
            dataset_cfg,
            data_dir=data_dir,
            transform=slice_pipeline,
            window_transform=window_pipeline,
            num_slices=num_slices,
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
    print_main(
        f"Split '{split}': {len(indices)} of {len(dataset.slice_map)} slices "
        f"from {num_files} volume(s)."
    )
    if conditional_bridge:
        acceleration = dataset_cfg.get("acceleration", "?")
        print_main(
            f"Bridge: aliased -> clean (conditional, R={acceleration}). The model sees "
            "the undersampled reconstruction at t=0, matching evaluate.py's warm start."
        )
    else:
        print_main("Bridge: noise -> clean (unconditional prior).")
    dataset_subset: BaseComplexDataset | Subset[Any] = (
        Subset(dataset, indices) if len(indices) < len(dataset.slice_map) else dataset
    )

    batch_size = cfg.get("training", {}).get("batch_size", 4)
    num_workers = cfg.get("training", {}).get("num_workers", 4)

    # Single-process runs keep the plain shuffle so their data order is unchanged by
    # this file; DistributedSampler is only introduced where it is actually needed,
    # to shard the split disjointly across ranks. Its default padding repeats a few
    # samples to give every rank the same number of batches, which is what keeps the
    # ranks in lockstep at the last step of the epoch.
    sampler: DistributedSampler[Any] | None = None
    if ctx.is_distributed:
        sampler = DistributedSampler(
            dataset_subset,
            num_replicas=ctx.world_size,
            rank=ctx.rank,
            shuffle=True,
            seed=seed if seed is not None else 0,
        )
        print_main(
            f"DDP over {ctx.world_size} ranks: batch_size={batch_size} per GPU, "
            f"effective batch {batch_size * ctx.world_size}. The learning rate is "
            "NOT rescaled - set training.learning_rate yourself when comparing to a "
            "single-GPU run."
        )

    dataloader = DataLoader(
        dataset_subset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        generator=loader_generator if sampler is None else None,
    )

    # --- Model Initialization ---
    # Identical trunk for both geometries; only the input width follows the state.
    model: torch.nn.Module = build_model(
        cfg,
        device,
        in_channels=manifold.state_channels,
        out_channels=manifold.velocity_channels,
    )

    # --- Optimizer & Scheduler ---
    # Built on the bare module, before any wrapper: neither DDP nor torch.compile
    # replaces the parameter tensors, so the optimizer state stays loadable by a run
    # that wraps them differently (one GPU vs eight, compiled vs not).
    lr = cfg.get("training", {}).get("learning_rate", 2e-4)
    weight_decay = cfg.get("training", {}).get("weight_decay", 1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler_cfg = cfg.get("training", {}).get("scheduler", {})
    epochs = cfg.get("training", {}).get("epochs", 100)
    eta_min = scheduler_cfg.get("eta_min", 1e-6)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)

    # --- Auto-resume ---
    # Every rank reads the same file and restores the same state, so no broadcast is
    # needed and a rank that somehow missed it would fail loudly on the first
    # gradient comparison rather than train against stale weights.
    state_dir = cfg.get("paths", {}).get("state_dir", "outputs/state")
    auto_resume = cfg.get("training", {}).get("auto_resume", False)
    start_epoch = 0
    resumed_run_id: str | None = None
    if auto_resume:
        resume_state = load_training_state(state_dir, device)
        if resume_state is None:
            print_main(f"Auto-resume on, no state at {state_path(state_dir)}: starting fresh.")
        else:
            model.load_state_dict(resume_state["model"])
            optimizer.load_state_dict(resume_state["optimizer"])
            scheduler.load_state_dict(resume_state["scheduler"])
            # T_max travels inside the scheduler state, so extending training.epochs
            # across a resume would otherwise keep annealing on the old horizon and
            # send the LR back up the far side of the cosine.
            if scheduler.T_max != epochs:
                print_main(
                    f"training.epochs changed {scheduler.T_max} -> {epochs} since the "
                    "resume state was written; re-anneal on the new horizon."
                )
                scheduler.T_max = epochs
            start_epoch = int(resume_state["epochs_completed"])
            resumed_run_id = resume_state["wandb_run_id"]
            print_main(
                f"Resumed from {state_path(state_dir)} at epoch {start_epoch}/{epochs}. "
                "Weights, optimizer moments and LR schedule restored; the noise and "
                "time draws are not replayed, so the run is not bit-identical to an "
                "uninterrupted one."
            )

    # --- W&B Configuration ---
    # Rank 0 only: N ranks logging the same step would interleave into N runs.
    use_wandb = cfg.get("logging", {}).get("use_wandb", False) and ctx.is_main
    wandb_run_id: str | None = None
    if use_wandb and HAS_WANDB:
        print_main("Weights & Biases logging enabled.")
        output_dir = cfg.get("paths", {}).get("output_dir", ".")
        os.makedirs(output_dir, exist_ok=True)
        config_dict = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
        wandb.init(
            project=cfg.get("logging", {}).get("project_name", "Cylindrical-Flow-Matching"),
            name=cfg.get("logging", {}).get("experiment_name", "uniform_noise_run"),
            dir=output_dir,
            config=config_dict,
            # A requeued job reattaches to the run it was writing before the
            # interruption, so one training curve is one line rather than one line
            # per scheduler decision.
            id=resumed_run_id,
            resume="allow",
        )
        wandb_run_id = wandb.run.id if wandb.run is not None else None
    else:
        use_wandb = False
        print_main("Local logging only.")

    # --- Parallel wrappers ---
    # DDP first, then compile: DDPOptimizer needs to see the bucket boundaries, which
    # it can only do when the DDP module is the thing being traced.
    if ctx.is_distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[ctx.local_rank] if device.type == "cuda" else None,
        )

    if cfg.get("training", {}).get("compile", True):
        print_main("Compiling model via Triton (this may take a minute during the first epoch)...")
        compiled_model = torch.compile(model)
        model = cast(torch.nn.Module, compiled_model)
    else:
        print_main("torch.compile disabled (training.compile=false).")

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
    if ctx.is_main:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Main Training Loop ---
    if start_epoch >= epochs:
        print_main(
            f"Resume state is already at epoch {start_epoch} of {epochs}: nothing to train. "
            "Raise training.epochs to continue, or training.auto_resume=false to restart."
        )

    for epoch in range(start_epoch, epochs):
        model.train()
        if sampler is not None:
            # Without this every epoch reuses one permutation, and a resumed run
            # would replay the orders it had already seen.
            sampler.set_epoch(epoch)

        epoch_loss_total = 0.0
        # Component names come from the manifold, so a geometry's own breakdown
        # reaches the logs without any geometry-specific code in this loop.
        epoch_components: dict[str, float] = {}

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", disable=not ctx.is_main)

        for _batch_idx, batch in enumerate(pbar):
            # Reconstruction mode yields a dict of complex tensors; generation mode a
            # single state tensor. Both ends of the conditional bridge are mapped
            # through the manifold here, so the geometry still owns the
            # representation and nothing below this point is mode-specific.
            x_alias: torch.Tensor | None = None
            if conditional_bridge:
                x_1 = manifold.from_complex(batch["target"].to(device))
                x_alias = manifold.from_complex(batch["input"].to(device))
            else:
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

            # The only difference between the two bridges: where the path starts.
            # Everything downstream - the time draw, the bridge, the target velocity,
            # the loss - is identical, which is what keeps the two training regimes
            # comparable and keeps the geometry the only variable under study.
            if x_alias is not None:
                x_0 = x_alias
            else:
                x_0 = manifold.sample_noise(flat, h, w, device)

            # --- Time Sampling ---
            # One time per sample: every slice of a window is the same example, so
            # they share t. repeat_interleave keeps a sample's slices adjacent,
            # matching how the flattened tensors are unfolded again below.
            t_model = torch.rand(b, device=device)
            t_flat = t_model.repeat_interleave(s) if is_window else t_model
            t_bridge = t_flat.view(flat, 1, 1, 1)

            # --- Coupling: re-pair the batch before the bridge sees it ---
            # x_1 is rebound too, because x_1_sup below feeds the loss mask and
            # would otherwise be the un-permuted batch: the endpoint the bridge
            # integrates towards and the endpoint the loss weights by have to be
            # the same sample. Guarded to the unwindowed, unconditional case at
            # startup, so this is a plain rebind rather than a reshape.
            x_1_flat = x_1.reshape(flat, c, h, w) if is_window else x_1
            if reorders:
                x_1_flat = coupling(x_0, x_1_flat, manifold)
                x_1 = x_1_flat

            # --- Bridge: Interpolation and target velocity ---
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
            loss, components = manifold.loss(pred_v, target_v, target_x1=x_1_sup, t=t_model)

            # --- Backprop ---
            # DDP all-reduces the gradients inside backward(), so the norm below is
            # already the global norm and clipping means the same thing at any scale.
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
        # Reduced in one collective so the reported average is over the whole epoch
        # rather than over this rank's shard. Keys are sorted because the reduction
        # is positional and the ranks have to agree on the order.
        component_names = sorted(epoch_components)
        totals = sum_across_ranks(
            [
                epoch_loss_total,
                *(epoch_components[name] for name in component_names),
                len(dataloader),
            ],
            device,
        )
        num_batches = totals[-1]
        avg_loss = totals[0] / num_batches
        avg_components = {
            name: total / num_batches
            for name, total in zip(component_names, totals[1:-1], strict=True)
        }

        current_lr = optimizer.param_groups[0]["lr"]
        breakdown = ", ".join(f"{name}: {value:.5f}" for name, value in avg_components.items())
        print_main(
            f"Epoch {epoch + 1} | Avg Loss: {avg_loss:.5f} ({breakdown}) | LR: {current_lr:.6f}"
        )

        scheduler.step()

        # Epoch-level logging and Sanity Check of the image
        if use_wandb:
            log_dict: dict[str, Any] = {
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

        if ctx.is_main:
            # Written every epoch, whatever the inference checkpoint interval is:
            # this is the file a requeued job comes back from, so its staleness is
            # the amount of GPU time an interruption costs.
            save_training_state(
                state_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epochs_completed=epoch + 1,
                wandb_run_id=wandb_run_id,
            )

            # ---------------------------------------------------------
            # FIX: Cleanly save model without torch.compile artifacts
            # ---------------------------------------------------------
            if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                torch.save(unwrap_model(model).state_dict(), checkpoint_path)
                print_main(f"Model saved cleanly to: {checkpoint_path}")

        # Agreed on across ranks: a rank stopping on a signal its peers never
        # received would leave them blocked in the next epoch's collectives.
        if any_across_ranks(_stop_requested(preempt_file), device):
            print_main(
                f"Stopping at epoch {epoch + 1}/{epochs}. Resume state is at "
                f"{state_path(state_dir)}."
            )
            break

    if use_wandb:
        wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()
