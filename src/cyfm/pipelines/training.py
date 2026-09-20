"""The flow-matching training run, as a pipeline.

Nothing here is specific to a manifold. The representation, the noise prior, the
probability path, the loss and the ODE step all come from the
:class:`~cyfm.core.manifold.BaseManifold` selected in ``conf/manifold/``, so the
cylindrical and Euclidean experiments are the *same* run with one config value
changed: same data, same architecture, same optimizer, same schedule.

Nothing here is specific to a scale either. Under ``torchrun`` the model is
wrapped in DistributedDataParallel and the split is sharded by a
DistributedSampler; without it ``world_size`` is 1, no process group exists and
the same statements run unchanged. ``training.batch_size`` is therefore **per
GPU**, and the effective batch is ``batch_size * world_size`` -- the learning
rate is not rescaled for you.

What is *not* factored out of :meth:`TrainingPipeline.run_epoch` is deliberate.
The five lines that are the algorithm -- sample the prior, sample a time, couple,
bridge, regress -- stay together and in order, because reading them in one place
is how anyone checks this implements the paper. The 2.5D window contract stays
with them for the same reason: the ``repeat_interleave`` that builds ``t_flat``
has to match the ``view`` that unfolds the prediction, and separating the two
would put a silent shape bug one refactor away.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any, Self

import torch
from torch.utils.data import Subset
from tqdm import tqdm

from cyfm.config.adapters import manifold_from_config
from cyfm.config.resolve import as_plain_dict
from cyfm.config.schema import RunConfig
from cyfm.core.dataset import BaseComplexDataset
from cyfm.core.manifold import BaseManifold
from cyfm.core.pipeline import BasePipeline
from cyfm.data import build_dataset, build_geometry_transform
from cyfm.data.splits import load_split_file_names, select_indices
from cyfm.data.transforms import Compose, slice_transform
from cyfm.flow import BRIDGE_ENDPOINTS
from cyfm.flow.couplings import build_coupling
from cyfm.pipelines.reporting import RunLogger, build_logger
from cyfm.pipelines.runtime import (
    CheckpointWriter,
    EpochAccumulator,
    StopController,
    build_dataloader,
    build_optimizer_and_scheduler,
    maybe_resume,
    wrap_for_execution,
)
from cyfm.utils.distributed import cleanup_distributed, print_main, setup_distributed
from cyfm.utils.inference import build_model
from cyfm.utils.seeding import seed_everything

__all__ = ["TrainingPipeline", "ensure_local_dataset"]

# Epochs between inference checkpoints. The resume state is written every epoch
# regardless; this is the cadence of the bare state dicts evaluate.py reads.
CHECKPOINT_INTERVAL = 10

# Divisibility the U-Net's downsampling depth requires of the field.
CROP_BASE = 16


def ensure_local_dataset(data_dir: str | None) -> None:
    """Fetch the fastMRI subset if the configured directory does not hold it.

    Known defect, carried over unchanged and deliberately not fixed here: the
    dispatch matches a *substring of the path* rather than asking the dataset
    which cohort it is. ``conf/dataset/fastmri_local.yaml`` happens to point at
    ``data/fastmri_local``, so the first branch fires; no shipped config produces
    a path containing ``fastmri_full``, so the second is unreachable. Pointing
    the local config at a differently named directory silently skips the
    download.

    Fixing it means asking the dataset registry, which is where the knowledge
    belongs. That is left out of this refactor because there is no fastMRI data
    on this machine to verify a change against, and a download path that cannot
    be exercised is the wrong thing to alter on a branch that must not move a
    number.

    Args:
        data_dir: ``dataset.data_dir``.
    """
    from cyfm.data.download import ensure_dataset_exists

    data_dir_str = str(data_dir)
    if "fastmri_local" in data_dir_str:
        ensure_dataset_exists("fastmri", data_dir_str, mode="local")
    elif "fastmri_full" in data_dir_str:
        ensure_dataset_exists("fastmri", data_dir_str, mode="full")


class TrainingPipeline(BasePipeline[None]):
    """One training run, from a composed config to checkpoints on disk.

    Args:
        cfg: The whole config tree, as a ``DictConfig`` or a plain mapping.
    """

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self.config = RunConfig.from_config(cfg)
        self.dataset_cfg = as_plain_dict(cfg.get("dataset"))

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> Self:
        """Build from a composed config tree.

        Args:
            cfg: The whole config.

        Returns:
            The pipeline, not yet set up.
        """
        return cls(cfg)

    # ---------------------------------------------------------------- setup ---

    def setup(self) -> None:
        """Build everything the loop needs, and reject what it cannot run.

        Ordered so the cheapest rejections happen first: a config that cannot
        train fails before a cohort is indexed or a model is built.
        """
        # First statement of the run: it calls torch.cuda.set_device, which every
        # later allocation depends on, and it decides which rank may print.
        self.ctx = setup_distributed()
        self.device = self.ctx.device

        self.stopper = StopController(self.config.training.preempt_file)
        self.stopper.install()

        self._validate()

        torch.set_float32_matmul_precision(self.config.training.matmul_precision)
        print_main(f"Set matmul precision to: {self.config.training.matmul_precision}")
        print_main(
            f"Starting training on: {self.device} (rank {self.ctx.rank + 1}/"
            f"{self.ctx.world_size}, local rank {self.ctx.local_rank})"
        )

        loader_generator = seed_everything(self.config.training.seed)
        if self.config.training.seed is None:
            print_main("Unseeded run (training.seed=null): not reproducible.")
        else:
            print_main(f"Seeded torch with {self.config.training.seed}")

        # The single switch between the cylindrical model and the Euclidean baseline.
        self.manifold: BaseManifold = manifold_from_config(self.cfg).to(self.device)

        self.coupling = build_coupling(self.config.training.coupling)
        # Whether the coupling permutes the batch, which the loop reports once.
        self.reorders = self.config.training.coupling not in ("independent", "none")
        print_main(f"Coupling: {self.config.training.coupling}")

        dataset = self._build_dataset()
        self.dataloader, self.sampler = build_dataloader(
            dataset, self.config.training, self.ctx, loader_generator
        )

        # Identical trunk for both geometries; only the input width follows the state.
        model = build_model(
            self.cfg,
            self.device,
            in_channels=self.manifold.state_channels,
            out_channels=self.manifold.velocity_channels,
            velocity_bound=(
                self.manifold.velocity_bound
                if bool(as_plain_dict(self.cfg.get("model")).get("bounded_velocity", False))
                else None
            ),
        )
        self.optimizer, self.scheduler = build_optimizer_and_scheduler(model, self.config.training)
        resume = maybe_resume(
            state_dir=self.config.paths.state_dir,
            device=self.device,
            model=model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epochs=self.config.training.epochs,
            enabled=self.config.training.auto_resume,
        )
        self.start_epoch = resume.start_epoch

        self.logger: RunLogger = build_logger(
            use_wandb=self.config.logging.use_wandb,
            is_main=self.ctx.is_main,
            project=self.config.logging.project_name,
            name=self.config.logging.experiment_name,
            directory=self.config.paths.output_dir,
            config=as_plain_dict(self.cfg),
            resume_id=resume.run_id,
        )

        self.model = wrap_for_execution(model, self.ctx, self.config.training.compile)
        # After DDP and compile, so the wrapper sees the module that actually
        # runs, and after the optimiser, which stays bound to the bare parameters.
        self.score_model = self.manifold.wrap_model(self.model)

        self.checkpoints = CheckpointWriter(
            state_dir=self.config.paths.state_dir,
            checkpoint_dir=self.config.paths.checkpoint_dir,
            interval=CHECKPOINT_INTERVAL,
            enabled=self.ctx.is_main,
        )

    def _validate(self) -> None:
        """Reject configurations this loop cannot run, before anything is built.

        Raises:
            ValueError: On a 5D slice window, or an unknown bridge endpoint.
        """
        num_slices = int(self.dataset_cfg.get("num_slices", 1))
        # Slice windows were consumed by exactly one architecture,
        # c_unet_cross_slice, which could train but never sample and went with
        # the other reconstruction-era models. Nothing in the package takes a 5D
        # input now, so a window is a config error rather than a mode.
        if num_slices > 1:
            raise ValueError(
                f"dataset.num_slices={num_slices} produces 5D slice windows, which no "
                "model in this package consumes. Set dataset.num_slices=1."
            )
        if self.config.training.bridge not in BRIDGE_ENDPOINTS:
            raise ValueError(
                f"training.bridge must be one of {list(BRIDGE_ENDPOINTS)}, "
                f"got {self.config.training.bridge!r}."
            )

    def _build_dataset(self) -> BaseComplexDataset | Subset[Any]:
        """Open the cohort and gate it down to the configured split.

        The manifold owns the representation, so ``x_1`` arrives in whatever the
        selected geometry trains on and everything downstream is shape-agnostic.
        ``dataset.crop_size`` runs first, on the complex slice: cohorts whose
        volumes do not share a matrix size cannot be collated without it, and
        cropping before normalisation is what keeps the peak modulus a property
        of the field of view actually trained on rather than of the oversampled
        readout around it.

        Returns:
            The dataset, or a ``Subset`` of it when the split is a strict subset.
        """
        data_dir = self.dataset_cfg.get("data_dir")
        pipeline: Callable[[torch.Tensor], torch.Tensor] = Compose(
            [
                build_geometry_transform(self.dataset_cfg.get("crop_size"), crop_base=CROP_BASE),
                slice_transform(self.manifold, crop_base=CROP_BASE),
            ]
        )

        # Rank 0 fetches, every other rank waits, so concurrent downloads cannot race.
        if self.ctx.is_distributed:
            if self.ctx.is_main:
                ensure_local_dataset(data_dir)
            torch.distributed.barrier()

        dataset = build_dataset(
            self.dataset_cfg, data_dir=data_dir, transform=pipeline, num_slices=1
        )

        # Train only on the volumes the split manifest lists, through the same two
        # functions evaluate.py uses. Without this the loader globs every .h5 and
        # `evaluate.split=test` scores volumes that were trained on, which makes
        # every absolute number fidelity on seen data rather than a generalisation
        # result.
        split = self.dataset_cfg.get("split", "train")
        file_names = load_split_file_names(data_dir, split, config_key="dataset.split")
        indices = select_indices(dataset.slice_map, file_names, config_key="dataset.split")
        num_files = len({str(dataset.slice_map[i][0]).rsplit("/", 1)[-1] for i in indices})
        print_main(
            f"Split '{split}': {len(indices)} of {len(dataset.slice_map)} slices "
            f"from {num_files} volume(s)."
        )
        print_main("Bridge: noise -> clean (unconditional prior).")
        if len(indices) < len(dataset.slice_map):
            return Subset(dataset, indices)
        return dataset

    # -------------------------------------------------------------- execute ---

    def execute(self) -> None:
        """Run every epoch, checkpointing and checking for preemption after each."""
        epochs = self.config.training.epochs
        if self.start_epoch >= epochs:
            print_main(
                f"Resume state is already at epoch {self.start_epoch} of {epochs}: nothing "
                "to train. Raise training.epochs to continue, or "
                "training.auto_resume=false to restart."
            )

        for epoch in range(self.start_epoch, epochs):
            avg_loss, avg_components, last_target = self.run_epoch(epoch)
            self.scheduler.step()
            self._log_epoch(epoch, avg_loss, avg_components, last_target)
            self.checkpoints.write(
                epoch=epoch,
                total_epochs=epochs,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                run_id=self.logger.run_id,
            )
            if self.stopper.should_stop(self.device):
                print_main(f"Stopping at epoch {epoch + 1}/{epochs}.")
                break

    def run_epoch(self, epoch: int) -> tuple[float, dict[str, float], torch.Tensor]:
        """One pass over the split.

        Args:
            epoch: Zero-based epoch index.

        Returns:
            ``(average_loss, average_components, last_target_batch)``. The last
            target is returned rather than kept as loop state because the
            epoch-level image log needs it: in the original loop it leaked out of
            the batch loop as a live local, which is exactly the kind of thing
            that breaks silently when a loop is extracted.
        """
        self.model.train()
        if self.sampler is not None:
            # Without this every epoch reuses one permutation, and a resumed run
            # would replay the orders it had already seen.
            self.sampler.set_epoch(epoch)

        # Timed from here, so the first batch includes worker start-up and the
        # store's first open: that is what "seconds to first batch" costs a new
        # cohort.
        started = time.perf_counter()
        first_batch_seconds: float | None = None
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        accumulator = EpochAccumulator()
        x_1_sup = torch.empty(0)
        progress = tqdm(
            self.dataloader,
            desc=f"Epoch {epoch + 1}/{self.config.training.epochs}",
            disable=not self.ctx.is_main,
        )

        for batch in progress:
            x_1 = batch.to(self.device)
            b, c, h, w = x_1.shape

            x_0 = self.manifold.sample_noise(b, h, w, self.device)
            t_model = torch.rand(b, device=self.device)
            t_bridge = t_model.view(b, 1, 1, 1)

            # x_1 is rebound too: x_1_sup below feeds the loss mask and would
            # otherwise be the un-permuted batch. The endpoint the bridge
            # integrates towards and the endpoint the loss weights by have to be
            # the same sample.
            if self.reorders:
                x_1 = self.coupling(x_0, x_1, self.manifold)
            x_t, target_v = self.manifold.bridge(x_0, x_1, t_bridge)
            x_1_sup = x_1

            self.optimizer.zero_grad()
            pred_v = self.score_model(x_t, t_model)
            loss, components = self.manifold.loss(pred_v, target_v, target_x1=x_1_sup, t=t_model)

            # DDP all-reduces the gradients inside backward(), so the norm below
            # is already the global norm and clipping means the same thing at any
            # scale. With clipping off the same call still measures it: the
            # infinite threshold never binds, and `grad_norm` stays comparable
            # between a clipped arm and an unclipped one.
            loss.backward()
            clip = self.config.training.grad_clip
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float("inf") if clip is None else clip
                )
            )
            self.optimizer.step()

            if first_batch_seconds is None:
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                first_batch_seconds = time.perf_counter() - started

            accumulator.add(loss.item(), components)
            progress.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    **{name: f"{value.item():.4f}" for name, value in components.items()},
                }
            )
            self.logger.log(
                {
                    "step_loss": loss.item(),
                    **{f"step_loss_{n}": v.item() for n, v in components.items()},
                    "grad_norm": grad_norm,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                }
            )

        avg_loss, avg_components, batches = accumulator.reduce(self.device)
        self._print_epoch_line(
            epoch, avg_loss, avg_components, started, first_batch_seconds, batches
        )
        return avg_loss, avg_components, x_1_sup

    def _print_epoch_line(
        self,
        epoch: int,
        avg_loss: float,
        avg_components: dict[str, float],
        started: float,
        first_batch_seconds: float | None,
        batches: int,
    ) -> None:
        """Print the two lines per epoch, one human and one parseable.

        The second is read by the fastMRI gate, and a table's GPU budget is
        extrapolated from it, so its shape is a contract.
        """
        current_lr = self.optimizer.param_groups[0]["lr"]
        breakdown = ", ".join(f"{name}: {value:.5f}" for name, value in avg_components.items())
        print_main(
            f"Epoch {epoch + 1} | Avg Loss: {avg_loss:.5f} ({breakdown}) | LR: {current_lr:.6f}"
        )
        first = float("nan") if first_batch_seconds is None else first_batch_seconds
        peak_vram_gib = float("nan")
        if self.device.type == "cuda":
            peak_vram_gib = torch.cuda.max_memory_allocated(self.device) / 2**30
        print_main(
            f"epoch_seconds={time.perf_counter() - started:.1f} "
            f"first_batch_seconds={first:.1f} peak_vram_gib={peak_vram_gib:.2f} "
            f"batches={batches} batch_size={self.config.training.batch_size}"
        )

    def _log_epoch(
        self,
        epoch: int,
        avg_loss: float,
        avg_components: dict[str, float],
        last_target: torch.Tensor,
    ) -> None:
        """Send the epoch's scalars, and every tenth epoch a target image."""
        self.logger.log(
            {
                "epoch": epoch + 1,
                "epoch_avg_loss": avg_loss,
                **{f"epoch_avg_loss_{n}": v for n, v in avg_components.items()},
                "learning_rate_epoch": self.optimizer.param_groups[0]["lr"],
            }
        )
        # Taken through the manifold so the picture is a modulus in both
        # geometries, rather than channel 0 of whatever the state happens to be.
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0 and last_target.numel():
            image = torch.abs(self.manifold.to_complex(last_target[0:1]))[0, 0].detach().cpu()
            self.logger.log_image("ground_truth_sample", image, f"Epoch {epoch + 1} Target Amp")

    # ------------------------------------------------------------- teardown ---

    def teardown(self) -> None:
        """Close the logger and the process group, even if the run raised.

        The original loop did neither on an exception: a failure mid-epoch left
        the W&B run open and the process group up, which on a cluster means the
        job's remaining ranks hang instead of exiting.
        """
        logger = getattr(self, "logger", None)
        if logger is not None:
            logger.finish()
        cleanup_distributed()
