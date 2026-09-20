"""Unit tests for the pointwise velocity MLP.

Two properties carry weight. It must genuinely have no spatial mixing, or it is
not the control it claims to be; and under ``shared_encoding`` it must hand both
geometries the same view of the same point, which is the confound control the
synthetic gates are built on.
"""

from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from cyfm.core.registry import MODELS
from cyfm.models.pointwise_mlp import PointwiseVelocityMLP, SinusoidalPositionEmbeddings
from cyfm.utils.inference import build_model


def _cylindrical(samples: torch.Tensor) -> torch.Tensor:
    """Lift complex samples of shape [B, H, W] into (m, cos phi, sin phi)."""
    amplitude, phase = samples.abs(), samples.angle()
    return torch.stack([amplitude, torch.cos(phase), torch.sin(phase)], dim=1)


def _euclidean(samples: torch.Tensor) -> torch.Tensor:
    """Lift complex samples of shape [B, H, W] into (re, im)."""
    return torch.stack([samples.real, samples.imag], dim=1)


# --- Registry and construction ---


def test_model_is_registered_under_both_names() -> None:
    """Registration is what makes model=<name> resolve."""
    assert MODELS.get("mlp") is PointwiseVelocityMLP
    assert MODELS.get("pointwise_mlp") is PointwiseVelocityMLP


@pytest.mark.parametrize("name", ["mlp", "mlp_gate"])
def test_shipped_config_builds_through_the_entry_point_helper(name: str) -> None:
    """The config goes through build_model, the path train.py actually uses."""
    cfg = OmegaConf.create({"model": OmegaConf.load(f"conf/model/{name}.yaml")})
    model = build_model(cfg, torch.device("cpu"), in_channels=3, out_channels=2)
    assert isinstance(model, PointwiseVelocityMLP)


# --- Shape ---


@pytest.mark.parametrize(("channels", "height", "width"), [(3, 16, 16), (2, 8, 12), (3, 1, 1)])
def test_forward_shape(channels: int, height: int, width: int) -> None:
    """A velocity is emitted at every coefficient."""
    model = PointwiseVelocityMLP(in_channels=channels, base_channels=16, depth=1)
    output = model(torch.randn(4, channels, height, width), torch.rand(4))
    assert output.shape == (4, 2, height, width)


def test_forward_rejects_wrong_rank_and_width() -> None:
    """Shape mistakes surface here rather than as an opaque matmul error."""
    model = PointwiseVelocityMLP(in_channels=3, base_channels=8, depth=1)
    with pytest.raises(ValueError, match=r"expected \[B, C, H, W\]"):
        model(torch.randn(4, 3, 8), torch.rand(4))
    with pytest.raises(ValueError, match="expected 3 channels"):
        model(torch.randn(4, 2, 8, 8), torch.rand(4))


# --- No spatial mixing ---


def test_output_is_pointwise_under_a_spatial_permutation() -> None:
    """Shuffling coefficients shuffles the output identically: no mixing."""
    torch.manual_seed(0)
    model = PointwiseVelocityMLP(in_channels=3, base_channels=32, depth=2).eval()
    state = torch.randn(2, 3, 6, 6)
    time = torch.rand(2)

    permutation = torch.randperm(36)
    shuffled = state.reshape(2, 3, 36)[:, :, permutation].reshape(2, 3, 6, 6)

    with torch.no_grad():
        direct = model(state, time).reshape(2, 2, 36)[:, :, permutation]
        via_shuffle = model(shuffled, time).reshape(2, 2, 36)
    assert torch.allclose(direct, via_shuffle, atol=1e-6)


def test_each_sample_gets_its_own_time() -> None:
    """Times are per sample, not per coefficient, and must not be transposed."""
    torch.manual_seed(1)
    model = PointwiseVelocityMLP(in_channels=3, base_channels=32, depth=2).eval()
    state = torch.randn(1, 3, 4, 4).repeat(2, 1, 1, 1)

    with torch.no_grad():
        output = model(state, torch.tensor([0.0, 1.0]))
    assert not torch.allclose(output[0], output[1], atol=1e-4)


# --- The gates' confound control ---


def test_shared_encoding_gives_both_geometries_the_same_view() -> None:
    """Same point, same features, whichever state representation carries it."""
    torch.manual_seed(2)
    samples = torch.randn(3, 4, 4, dtype=torch.complex64) + 0.5
    cylindrical = PointwiseVelocityMLP(in_channels=3, shared_encoding=True)
    euclidean = PointwiseVelocityMLP(in_channels=2, shared_encoding=True)

    from_cylinder = cylindrical.encode(_cylindrical(samples).permute(0, 2, 3, 1).reshape(-1, 3))
    from_plane = euclidean.encode(_euclidean(samples).permute(0, 2, 3, 1).reshape(-1, 2))
    assert torch.allclose(from_cylinder, from_plane, atol=1e-5)


def test_shared_encoding_equalises_capacity_across_geometries() -> None:
    """A capacity difference between arms would be a confound; this removes it."""
    cylindrical = PointwiseVelocityMLP(in_channels=3, shared_encoding=True)
    euclidean = PointwiseVelocityMLP(in_channels=2, shared_encoding=True)
    assert sum(p.numel() for p in cylindrical.parameters()) == sum(
        p.numel() for p in euclidean.parameters()
    )


def test_without_shared_encoding_the_geometries_differ_in_width() -> None:
    """The package's default: each geometry hands the trunk its own state."""
    cylindrical = PointwiseVelocityMLP(in_channels=3, shared_encoding=False)
    euclidean = PointwiseVelocityMLP(in_channels=2, shared_encoding=False)
    assert sum(p.numel() for p in cylindrical.parameters()) != sum(
        p.numel() for p in euclidean.parameters()
    )


def test_shared_encoding_is_finite_at_the_origin() -> None:
    """The cylinder chart is singular at A = 0; the encoding must not be."""
    model = PointwiseVelocityMLP(in_channels=2, shared_encoding=True)
    encoded = model.encode(torch.zeros(8, 2))
    assert bool(torch.isfinite(encoded).all())


def test_shared_encoding_rejects_a_state_it_cannot_decode() -> None:
    """Only a plane or a cylinder state carries a complex number to decode."""
    with pytest.raises(ValueError, match="shared_encoding needs a plane"):
        PointwiseVelocityMLP(in_channels=5, shared_encoding=True)


# --- Time embedding ---


def test_sinusoidal_embedding_shape_and_range() -> None:
    """The embedding is bounded and of the requested width."""
    embedding = SinusoidalPositionEmbeddings(8)(torch.rand(5))
    assert embedding.shape == (5, 8)
    assert bool((embedding.abs() <= 1.0 + 1e-6).all())


@pytest.mark.parametrize("dim", [0, 3, -2])
def test_sinusoidal_embedding_rejects_bad_width(dim: int) -> None:
    """An odd or non-positive width cannot be split into sin and cos halves."""
    with pytest.raises(ValueError, match="positive and even"):
        SinusoidalPositionEmbeddings(dim)


def test_embedded_time_changes_the_parameter_count() -> None:
    """time_embedding_dim widens the first layer rather than being ignored."""
    raw = PointwiseVelocityMLP(in_channels=3, time_embedding_dim=0)
    embedded = PointwiseVelocityMLP(in_channels=3, time_embedding_dim=16)
    assert sum(p.numel() for p in embedded.parameters()) > sum(p.numel() for p in raw.parameters())


# --- Training ---


def test_gradients_reach_every_parameter() -> None:
    """The model is trainable, which the entry points assume without checking."""
    model = PointwiseVelocityMLP(in_channels=3, base_channels=16, depth=2)
    model(torch.randn(2, 3, 4, 4), torch.rand(2)).square().mean().backward()
    assert all(p.grad is not None for p in model.parameters())


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"in_channels": 0}, "in_channels must be positive"),
        ({"out_channels": 0}, "out_channels must be positive"),
        ({"base_channels": 0}, "base_channels must be positive"),
        ({"depth": 0}, "depth must be positive"),
    ],
)
def test_constructor_validation(kwargs: dict[str, Any], match: str) -> None:
    """Bad configuration fails at build time."""
    with pytest.raises(ValueError, match=match):
        PointwiseVelocityMLP(**kwargs)
