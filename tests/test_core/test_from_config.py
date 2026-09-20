"""Each registered class reads its own config; the factory no longer branches.

``build_manifold`` used to hold a ``match`` with one arm per geometry, and two of
those arms re-typed the constructors' defaults -- nine values for the cylinder,
five for the plane. That is how defaults drift: the copy in the factory silently
wins whenever a config omits the key, and nothing compares the two. The third arm
already avoided it and said why in a comment; these tests pin the rule the other
two now follow.
"""

from __future__ import annotations

import pytest
import torch

from cyfm.config.adapters import manifold_from_config
from cyfm.manifolds import build_manifold
from cyfm.manifolds.complex_diffusion import ComplexDiffusionManifold
from cyfm.manifolds.cylindrical import CylindricalManifold
from cyfm.manifolds.euclidean import EuclideanManifold
from cyfm.utils.inference import build_model


class TestDefaultsComeFromTheConstructor:
    @pytest.mark.parametrize(
        ("manifold_cls", "attribute"),
        [
            (CylindricalManifold, "phase_weight"),
            (CylindricalManifold, "phase_spread"),
            (CylindricalManifold, "spatial_correlation"),
            (EuclideanManifold, "noise_prior"),
            (EuclideanManifold, "spatial_correlation"),
            (ComplexDiffusionManifold, "sigma_max"),
            (ComplexDiffusionManifold, "nfe_mode"),
        ],
    )
    def test_an_empty_config_gives_the_constructor_default(
        self, manifold_cls: type, attribute: str
    ) -> None:
        """One source of truth per default: the constructor signature."""
        from_empty = build_manifold({"name": manifold_cls.name})
        direct = manifold_cls()
        assert getattr(from_empty, attribute) == getattr(direct, attribute)

    def test_the_loss_group_reaches_the_geometry_that_reads_it(self) -> None:
        cfg = {
            "manifold": {"name": "cylindrical"},
            "training": {"loss": {"phase_loss_type": "cosine", "lambda_phase": 0.25}},
        }
        manifold = manifold_from_config(cfg)
        assert manifold._loss.phase_loss_type == "cosine"
        assert manifold._loss.lambda_phase == 0.25

    def test_the_same_group_reaches_the_other_geometry_under_its_own_key(self) -> None:
        """``vel_loss_type`` in the config, ``loss_type`` in the constructor.

        One training config serves both geometries; each takes the keys that
        apply to it. That mapping is declared on the geometry as
        ``config_loss_keys``, which is what lets the factory be branch-free.
        """
        cfg = {
            "manifold": {"name": "euclidean"},
            "training": {"loss": {"vel_loss_type": "l2", "phase_loss_type": "cosine"}},
        }
        manifold = manifold_from_config(cfg)
        assert manifold._loss.loss_type == "l2"

    def test_a_geometry_ignores_loss_keys_it_does_not_declare(self) -> None:
        """Score matching has no velocity loss type, so the group is not read."""
        cfg = {
            "manifold": {"name": "complex_diffusion"},
            "training": {"loss": {"vel_loss_type": "l2", "amp_loss_type": "l2"}},
        }
        assert manifold_from_config(cfg).nfe_mode == ComplexDiffusionManifold().nfe_mode


class TestUnknownKeysAreRejected:
    """A typo in an override used to be a successful run with the wrong settings."""

    def test_an_unknown_manifold_key_names_itself_and_the_alternatives(self) -> None:
        with pytest.raises(ValueError, match="does not accept phase_wieght"):
            build_manifold({"name": "cylindrical", "phase_wieght": 2.0})

    def test_a_key_belonging_to_another_geometry_is_rejected(self) -> None:
        """``phase_weight`` is real, but not on the plane."""
        with pytest.raises(ValueError, match="does not accept phase_weight"):
            build_manifold({"name": "euclidean", "phase_weight": 2.0})

    def test_name_is_not_mistaken_for_an_unknown_key(self) -> None:
        build_manifold({"name": "cylindrical"})  # would raise if `name` leaked through

    def test_an_unknown_model_key_is_rejected(self, torch_cpu_device) -> None:
        """The deleted attention model's keys were being silently swallowed."""
        with pytest.raises(ValueError, match="does not accept use_attention"):
            build_model(
                {"model": {"name": "c_unet", "base_channels": 8, "use_attention": True}},
                torch_cpu_device,
            )


class TestVelocityBoundReachesOnlyWhatDeclaresIt:
    def test_the_unet_receives_the_geometry_bound(self, torch_cpu_device) -> None:
        model = build_model(
            {"model": {"name": "c_unet", "base_channels": 8}},
            torch_cpu_device,
            in_channels=3,
            velocity_bound=CylindricalManifold().velocity_bound,
        )
        assert model._bound is not None

    def test_the_pointwise_control_does_not_and_is_not_an_error(self, torch_cpu_device) -> None:
        """The MLP takes no bound; passing one must not make the build fail.

        This was the only difference between the two arms of the ``match`` that
        was deleted, so it is the one thing worth pinning about its removal.
        """
        model = build_model(
            {"model": {"name": "mlp", "base_channels": 16, "depth": 2}},
            torch_cpu_device,
            in_channels=3,
            velocity_bound=CylindricalManifold().velocity_bound,
        )
        assert not hasattr(model, "_bound")


@pytest.fixture
def torch_cpu_device() -> torch.device:
    """The device every test here builds on; models are never moved off it."""
    return torch.device("cpu")
