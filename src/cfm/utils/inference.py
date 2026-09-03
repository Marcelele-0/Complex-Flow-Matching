"""Model construction and checkpoint loading shared by the inference entry points.

``generate.py`` and ``evaluate.py`` both need to build the configured architecture
and load a trained checkpoint. Keeping that in one place means a new architecture
is registered once in :func:`build_model` rather than in every entry point, and it
makes the logic unit-testable without standing up a Hydra ``main()``.
"""

from __future__ import annotations

import glob
import os

import torch
from omegaconf import DictConfig

from cfm.core.registry import MODELS

# Models that train but cannot yet drive an ODE solver, mapped to why.
SAMPLING_UNSUPPORTED_MODELS = {
    "c_unet_cross_slice": (
        "predicts a center-slice velocity [B, 2, H, W] from a slice window "
        "[B, S, C, H, W]. A solver advances a state with a velocity of matching "
        "shape, so it cannot step a multi-slice state from this output. Training "
        "is supported on either geometry; sampling needs all-slice decoding, "
        "which is not implemented yet."
    ),
}


def reject_unsupported_sampling_model(cfg: DictConfig) -> None:
    """Fail early when the configured model cannot be sampled from.

    Called by the sampling entry points before any checkpoint or data work, so the
    limitation is stated plainly instead of surfacing as a shape error inside the
    solver.

    Args:
        cfg: Full Hydra config; reads ``model.name``.

    Raises:
        NotImplementedError: If the configured model cannot drive the solver.
    """
    model_name = cfg.get("model", {}).get("name", "c_unet")
    reason = SAMPLING_UNSUPPORTED_MODELS.get(model_name)
    if reason is not None:
        raise NotImplementedError(f"model={model_name} {reason}")


def build_model(
    cfg: DictConfig,
    device: torch.device,
    in_channels: int = 3,
    out_channels: int = 2,
) -> torch.nn.Module:
    """Instantiate the architecture named by ``cfg.model.name``, on ``device``.

    Uses the MODELS registry to resolve and instantiate architectures.

    Args:
        cfg: Full Hydra config. Reads ``model.name``, ``model.base_channels`` and,
            for the attention variant, ``channel_mults`` / ``use_attention`` /
            ``attn_heads``.
        device: Device to move the instantiated model to.
        in_channels: Width of the state the model consumes, normally
            ``Manifold.state_channels``. Defaults to the cylindrical 3 so a
            caller predating the manifold group is unaffected. This is the *only*
            architectural difference between the two geometries: pass 2 and the
            same trunk consumes ``(Re, Im)`` instead of ``(m, cos, sin)``.
        out_channels: Width of the velocity the model emits, normally
            ``Manifold.velocity_channels``. 2 for both geometries.

    Returns:
        The model, on ``device``, in whatever mode ``torch.nn.Module`` defaults to
        (:func:`load_weights` switches it to eval).

    Raises:
        ValueError: If ``cfg.model.name`` is not a known architecture in MODELS.
    """

    model_name = cfg.get("model", {}).get("name", "c_unet")
    base_channels = cfg.get("model", {}).get("base_channels", 64)

    if not MODELS.contains(model_name):
        raise ValueError(
            f"Unknown config model_name: {model_name}. Available models: {MODELS.list()}"
        )

    match model_name:
        case "c_unet_attention" | "cylindrical_unet_attention":
            channel_mults = cfg.get("model", {}).get("channel_mults", [1, 2, 4, 8, 8])
            use_attention = cfg.get("model", {}).get("use_attention", True)
            attn_heads = cfg.get("model", {}).get("attn_heads", 4)

            model: torch.nn.Module = MODELS.build(
                model_name,
                base_channels=base_channels,
                channel_mults=list(channel_mults),
                use_attention=use_attention,
                attn_heads=attn_heads,
                in_channels=in_channels,
                out_channels=out_channels,
            ).to(device)
            print(f"Instantiated CylindricalUNetAttention with base_channels={base_channels}")

        case "c_unet_cross_slice" | "cylindrical_unet_cross_slice":
            channel_mults = cfg.get("model", {}).get("channel_mults", [1, 2, 4, 8, 8])
            attn_heads = cfg.get("model", {}).get("attn_heads", 4)

            model = MODELS.build(
                model_name,
                base_channels=base_channels,
                channel_mults=list(channel_mults),
                attn_heads=attn_heads,
                in_channels=in_channels,
            ).to(device)
            print(f"Instantiated CylindricalUNetCrossSlice with base_channels={base_channels}")

        case "c_unet" | "cylindrical_unet":
            model = MODELS.build(
                model_name,
                base_channels=base_channels,
                in_channels=in_channels,
                out_channels=out_channels,
            ).to(device)
            print(f"Instantiated standard CylindricalUNet with base_channels={base_channels}")

        case _:
            model_kwargs = {k: v for k, v in cfg.get("model", {}).items() if k != "name"}
            model = MODELS.build(
                model_name,
                in_channels=in_channels,
                out_channels=out_channels,
                **model_kwargs,
            ).to(device)
            print(f"Instantiated registered model {model_name}")

    print(f"  state channels in: {in_channels}   velocity channels out: {out_channels}")
    return model


def find_latest_checkpoint(base_dir: str = "outputs/train") -> str | None:
    """Return the most recently written ``.pt`` under ``base_dir``, or ``None``.

    Args:
        base_dir: Directory searched recursively.

    Returns:
        Path to the newest checkpoint by mtime, which for the layout ``train.py``
        writes is the highest epoch of the newest run. ``None`` if there is none.
    """
    checkpoints = glob.glob(os.path.join(base_dir, "**", "*.pt"), recursive=True)
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)


def resolve_checkpoint(cfg: DictConfig, section: str, orig_cwd: str) -> str:
    """Locate the checkpoint for the run named in ``cfg[section].run_name``.

    Falls back to ``cfg.logging.experiment_name`` when the section sets no
    ``run_name``, then searches ``outputs/train/{run_name}/`` where ``train.py``
    writes.

    Args:
        cfg: Full Hydra config.
        section: Config section holding ``run_name``, e.g. ``"generate"`` or
            ``"evaluate"``. Only used to read the key and to name it in errors.
        orig_cwd: ``hydra.utils.get_original_cwd()``; Hydra moves the working
            directory to the run output dir, so paths must be anchored to this.

    Returns:
        Path to the checkpoint.

    Raises:
        ValueError: If no run name can be resolved from either place.
        FileNotFoundError: If the run directory holds no ``.pt`` file.
    """
    run_name = cfg.get(section, {}).get("run_name")
    if not run_name:
        run_name = cfg.get("logging", {}).get("experiment_name")
    if not run_name:
        raise ValueError(
            f"No run_name in {section} config and no experiment_name in logging config"
        )

    run_dir = os.path.join(orig_cwd, "outputs", "train", run_name)
    print(f"Searching for best checkpoint in: {run_dir}")

    checkpoint_path = find_latest_checkpoint(run_dir)
    if checkpoint_path is None:
        raise FileNotFoundError(f"No .pt checkpoints found in '{run_dir}'")

    return checkpoint_path


def load_weights(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    """Load a checkpoint into ``model`` in place and switch it to eval mode.

    Strips the ``_orig_mod.`` prefix that ``torch.compile`` adds to state dict
    keys, so a checkpoint saved from a compiled model loads into an uncompiled one.

    Args:
        model: Model to load into.
        checkpoint_path: Path to a ``.pt`` state dict.
        device: Device to map the stored tensors onto.
    """
    print(f"Loading weights from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    # eval() here rather than at the call sites: forgetting it silently corrupts
    # every metric through dropout/norm behaviour.
    model.eval()
    print(f"Successfully loaded weights from {checkpoint_path}")
