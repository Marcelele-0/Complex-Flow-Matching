"""Tests for generic Registry and pre-instantiated core registries."""

from typing import cast

import pytest
import torch
import torch.nn as nn

from cfm.core.dataset import BaseComplexDataset
from cfm.core.loss import BaseLoss
from cfm.core.manifold import BaseManifold
from cfm.core.registry import (
    DATASETS,
    LOSSES,
    MANIFOLDS,
    MASKS,
    MODELS,
    SOLVERS,
    Registry,
)
from cfm.core.solver import BaseODESolver
from cfm.data.masks import BaseMaskGenerator


def test_registry_basic_operations() -> None:
    """Test register, get, build, contains, and list on a custom Registry."""
    test_reg: Registry[nn.Module] = Registry("test_modules")

    @test_reg.register("linear_block")
    class DummyLinear(nn.Module):
        def __init__(self, in_features: int = 10, out_features: int = 5):
            super().__init__()
            self.fc = nn.Linear(in_features, out_features)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    # Test contains
    assert test_reg.contains("linear_block")
    assert "linear_block" in test_reg
    assert not test_reg.contains("non_existent")
    assert "non_existent" not in test_reg

    # Test get
    cls_retrieved = test_reg.get("linear_block")
    assert cls_retrieved is DummyLinear

    # Test build
    instance = test_reg.build("linear_block", in_features=8, out_features=4)
    assert isinstance(instance, DummyLinear)
    assert instance.fc.in_features == 8
    assert instance.fc.out_features == 4

    # Test list & len & repr
    assert test_reg.list() == ["linear_block"]
    assert len(test_reg) == 1
    assert "linear_block" in repr(test_reg)
    assert list(iter(test_reg)) == ["linear_block"]


def test_registry_bare_decorator() -> None:
    """Test @REGISTRY.register without arguments using class name."""

    class AutoNamedClass:
        def __init__(self, val: int = 42):
            self.val = val

    test_reg: Registry[AutoNamedClass] = Registry("bare_decorator_test")
    test_reg.register(AutoNamedClass)

    assert test_reg.contains("AutoNamedClass")
    obj = test_reg.build("AutoNamedClass", val=100)
    assert obj.val == 100


def test_registry_duplicate_registration_guards() -> None:
    """Test duplicate registration raising ValueError unless force=True."""
    test_reg: Registry[object] = Registry("duplicate_guard_test")

    @test_reg.register("item")
    class Item1:
        pass

    with pytest.raises(ValueError, match="already registered"):

        @test_reg.register("item")
        class Item2:
            pass

    # Overwrite with force=True
    @test_reg.register("item", force=True)
    class Item3:
        pass

    assert test_reg.get("item") is Item3


def test_registry_missing_key_error() -> None:
    """Test that get and build raise KeyError with available keys listed on missing key."""
    test_reg: Registry[object] = Registry("missing_key_test")
    test_reg.register("alpha")(lambda: "alpha")

    with pytest.raises(KeyError, match="not found in registry 'missing_key_test'"):
        test_reg.get("beta")

    with pytest.raises(KeyError, match="not found in registry 'missing_key_test'"):
        test_reg.build("beta")


def test_core_registries_populated() -> None:
    """Verify built-in CFM components are registered in their respective registries."""
    import cfm.data.dataset  # noqa: F401
    import cfm.data.fastmri  # noqa: F401
    import cfm.data.masks  # noqa: F401
    import cfm.flow.euclidean_solver  # noqa: F401
    import cfm.flow.solver  # noqa: F401
    import cfm.manifolds.cylindrical  # noqa: F401
    import cfm.manifolds.euclidean  # noqa: F401
    import cfm.models  # noqa: F401

    # MANIFOLDS
    assert "cylindrical" in MANIFOLDS
    assert "euclidean" in MANIFOLDS
    assert issubclass(cast(type, MANIFOLDS.get("cylindrical")), BaseManifold)
    assert issubclass(cast(type, MANIFOLDS.get("euclidean")), BaseManifold)

    # MODELS
    assert "c_unet" in MODELS
    assert "c_unet_attention" in MODELS
    assert "c_unet_cross_slice" in MODELS

    # SOLVERS
    assert "cylindrical" in SOLVERS
    assert "euclidean" in SOLVERS
    assert issubclass(cast(type, SOLVERS.get("cylindrical")), BaseODESolver)
    assert issubclass(cast(type, SOLVERS.get("euclidean")), BaseODESolver)

    # LOSSES
    assert "cylindrical" in LOSSES
    assert "euclidean" in LOSSES
    assert issubclass(cast(type, LOSSES.get("cylindrical")), BaseLoss)
    assert issubclass(cast(type, LOSSES.get("euclidean")), BaseLoss)

    # DATASETS
    assert "skm_tea" in DATASETS
    assert issubclass(cast(type, DATASETS.get("skm_tea")), BaseComplexDataset)
    assert "fastmri" in DATASETS
    assert issubclass(cast(type, DATASETS.get("fastmri")), BaseComplexDataset)

    # MASKS
    assert "cartesian" in MASKS
    assert "poisson_disc" in MASKS
    assert issubclass(cast(type, MASKS.get("cartesian")), BaseMaskGenerator)
    assert issubclass(cast(type, MASKS.get("poisson_disc")), BaseMaskGenerator)
