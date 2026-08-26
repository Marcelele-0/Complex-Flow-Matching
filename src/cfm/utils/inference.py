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


def build_model(cfg: DictConfig, device: torch.device) -> torch.nn.Module:
    """Instantiate the architecture named by ``cfg.model.name``, on ``device``.

    Register new architectures here: this is the single place both entry points
    resolve a model name through.

    Args:
        cfg: Full Hydra config. Reads ``model.name``, ``model.base_channels`` and,
            for the attention variant, ``channel_mults`` / ``use_attention`` /
            ``attn_heads``.
        device: Device to move the instantiated model to.

    Returns:
        The model, on ``device``, in whatever mode ``torch.nn.Module`` defaults to
        (:func:`load_weights` switches it to eval).

    Raises:
        ValueError: If ``cfg.model.name`` is not a known architecture.
    """
    model_name = cfg.get("model", {}).get("name", "c_unet")
    base_channels = cfg.get("model", {}).get("base_channels", 64)

    match model_name:
        case "c_unet_attention":
            # Imported lazily so the attention model is only pulled in when asked for.
            from cfm.models.cylindrical_unet_attention import CylindricalUNetAttention

            channel_mults = cfg.get("model", {}).get("channel_mults", [1, 2, 4, 8, 8])
            use_attention = cfg.get("model", {}).get("use_attention", True)
            attn_heads = cfg.get("model", {}).get("attn_heads", 4)

            model: torch.nn.Module = CylindricalUNetAttention(
                base_channels=base_channels,
                channel_mults=list(channel_mults),
                use_attention=use_attention,
                attn_heads=attn_heads,
            ).to(device)
            print(f"Instantiated CylindricalUNetAttention with base_channels={base_channels}")

        case "c_unet":
            from cfm.models.cylindrical_unet import CylindricalUNet

            model = CylindricalUNet(base_channels=base_channels).to(device)
            print(f"Instantiated standard CylindricalUNet with base_channels={base_channels}")

        case _:
            raise ValueError(f"Unknown config model_name: {model_name}")

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
