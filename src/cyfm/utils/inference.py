"""Model construction and checkpoint loading shared by the entry points.

``train.py`` and ``evaluate.py`` both need to build the configured architecture,
and ``evaluate.py`` additionally has to find and load a trained checkpoint.
Keeping that in one place means a new architecture is reached once through the
MODELS registry rather than in every entry point, and it makes the logic
unit-testable without standing up a Hydra ``main()``.

This module used to carry ``import cyfm.models`` for its registration side
effect, because it was the only thing guaranteeing ``MODELS`` was non-empty.
That is now :mod:`cyfm`'s job, which runs before any submodule of it can be
imported, so the crutch is gone.
"""

from __future__ import annotations

import glob
import inspect
import os
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from cyfm.config.resolve import as_plain_dict
from cyfm.core.registry import MODELS

#: Architecture used when the config names none, matching ``conf/model/``'s root
#: default. Stated once so the entry point and this builder cannot disagree.
DEFAULT_MODEL = "c_unet"


def build_model(
    cfg: Mapping[str, Any],
    device: torch.device,
    in_channels: int = 3,
    out_channels: int = 2,
    velocity_bound: Sequence[float] | None = None,
) -> torch.nn.Module:
    """Instantiate the architecture named by ``cfg.model.name``, on ``device``.

    Every key of the ``model`` group except ``name`` is forwarded to the
    constructor, filtered by what that constructor accepts. There is no branch
    per architecture: registering a class is what makes it reachable.

    ``velocity_bound`` is forwarded on the same terms. It is a property of the
    geometry rather than of the config, so it cannot come from the group, and it
    reaches only an architecture that declares it -- the U-Net does, the
    pointwise MLP does not.

    Args:
        cfg: Full config. Reads the ``model`` group.
        device: Device to move the instantiated model to.
        in_channels: Width of the state the model consumes, normally
            ``Manifold.state_channels``. Defaults to the cylindrical 3 so a
            caller predating the manifold group is unaffected. This is the *only*
            architectural difference between the two geometries: pass 2 and the
            same trunk consumes ``(Re, Im)`` instead of ``(m, cos, sin)``.
        out_channels: Width of the velocity the model emits, normally
            ``Manifold.velocity_channels``. 2 for both geometries.
        velocity_bound: Per-channel ceiling on the emitted velocity, from the
            geometry. Ignored by an architecture that does not take one.

    Returns:
        The model, on ``device``, in whatever mode ``torch.nn.Module`` defaults to
        (:func:`load_weights` switches it to eval).

    Raises:
        ValueError: If ``cfg.model.name`` is not a known architecture in MODELS,
            or the group carries a key the architecture cannot accept.
    """
    model_cfg = as_plain_dict(cfg.get("model"))
    model_name = str(model_cfg.pop("name", DEFAULT_MODEL))

    if not MODELS.contains(model_name):
        raise ValueError(
            f"Unknown config model_name: {model_name}. Available models: {MODELS.list()}"
        )

    model_cls = MODELS.get(model_name)
    accepted = {
        name
        for name, parameter in inspect.signature(model_cls).parameters.items()
        if parameter.kind is not inspect.Parameter.VAR_KEYWORD
    }
    unknown = sorted(set(model_cfg) - accepted)
    if unknown:
        raise ValueError(
            f"model={model_name!r} does not accept {', '.join(unknown)}. "
            f"Accepted keys: {', '.join(sorted(accepted))}."
        )

    settings: dict[str, Any] = dict(model_cfg)
    settings["in_channels"] = in_channels
    settings["out_channels"] = out_channels
    if "velocity_bound" in accepted:
        settings["velocity_bound"] = velocity_bound

    model = MODELS.build(model_name, **settings).to(device)
    print(f"Instantiated {type(model).__name__} ({model_name})")
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


def resolve_checkpoint(cfg: Mapping[str, Any], section: str, orig_cwd: str) -> str:
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
