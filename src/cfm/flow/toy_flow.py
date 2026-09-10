"""Two-dimensional flow matching on the cylinder and in the plane.

A gate needs a setting where the geometry can be varied and *nothing else* can.
On images that is impossible in practice -- the two arms differ in loss, in input
representation and in prior, which is what invalidates the current comparison --
so this module builds the same experiment on single complex numbers, where every
one of those differences can be removed by construction:

* **Same prior.** Both arms start from the same circularly symmetric complex
  Gaussian, drawn once per batch.
* **Same input.** :class:`ToyVelocityField` decodes whichever state it is given
  into one shared encoding ``(re, im, m, cos phi, sin phi)`` before the first
  layer, so neither geometry is handed a representation the other lacks. The
  network, its width and its parameter count are identical across arms.
* **Same loss.** Plain unweighted regression of the bridge's target velocity in
  whatever tangent space that bridge lives in. No amplitude weighting anywhere.

What is left varying is the bridge (which path connects a pair), the coupling
(which pairs are formed) and the integrator (how a velocity advances a state).
Those are the three things under test.

The bridges and solvers are the production ones from :mod:`cfm.flow`, driven at
``H = W = 1``. That is deliberate: a gate that reimplemented the geodesic would
measure the reimplementation rather than the method.

Both bridges have a conditional velocity that is **constant along the path**, so
the converged regression residual *is* the straightness statistic of the
rectified-flow literature, ``E ||(x_1 - x_0) - v(x_t, t)||^2``, and needs no
separate estimator. :func:`straightness` reports it normalised by the mean
squared displacement so the two geometries can be compared on one scale.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from cfm.flow.bridge import GeodesicFlowBridge
from cfm.flow.euclidean_bridge import LinearFlowBridge
from cfm.flow.euclidean_solver import EuclideanODESolver
from cfm.flow.solver import CylindricalODESolver

__all__ = [
    "Geometry",
    "ToyVelocityField",
    "build_bridge",
    "build_solver",
    "from_state",
    "straightness",
    "to_state",
]

Geometry = Literal["euclidean", "cylindrical"]

# Below this amplitude the phase of a complex number is numerically meaningless,
# so the shared encoding reports a fixed direction rather than atan2 noise. The
# cylinder chart is genuinely singular at the origin; this is where that shows.
_AMPLITUDE_FLOOR = 1e-8


def to_state(samples: torch.Tensor, geometry: Geometry) -> torch.Tensor:
    """Lift complex samples into a geometry's state representation.

    Args:
        samples: Complex tensor of shape ``[n]``.
        geometry: ``"euclidean"`` for ``(re, im)``, ``"cylindrical"`` for
            ``(m, cos phi, sin phi)``.

    Returns:
        State tensor of shape ``[n, 2, 1, 1]`` or ``[n, 3, 1, 1]``.

    Raises:
        ValueError: If ``samples`` is not a 1D complex tensor, or the geometry
            is unknown.
    """
    if samples.ndim != 1:
        raise ValueError(f"samples must be 1D, got shape {tuple(samples.shape)}")
    if not samples.is_complex():
        raise ValueError("samples must be a complex tensor")

    if geometry == "euclidean":
        channels = torch.stack([samples.real, samples.imag], dim=1)
    elif geometry == "cylindrical":
        amplitude = samples.abs()
        phase = samples.angle()
        channels = torch.stack([amplitude, torch.cos(phase), torch.sin(phase)], dim=1)
    else:
        raise ValueError(f"unknown geometry {geometry!r}")

    return channels.unsqueeze(-1).unsqueeze(-1)


def from_state(state: torch.Tensor, geometry: Geometry) -> torch.Tensor:
    """Project a state back to complex samples.

    Args:
        state: State tensor of shape ``[n, C, 1, 1]``.
        geometry: The geometry the state is written in.

    Returns:
        Complex tensor of shape ``[n]``.

    Raises:
        ValueError: If the channel count does not match the geometry, or the
            geometry is unknown.
    """
    flat = state.reshape(state.shape[0], state.shape[1])
    if geometry == "euclidean":
        if flat.shape[1] != 2:
            raise ValueError(f"euclidean state needs 2 channels, got {flat.shape[1]}")
        return torch.complex(flat[:, 0], flat[:, 1])
    if geometry == "cylindrical":
        if flat.shape[1] != 3:
            raise ValueError(f"cylindrical state needs 3 channels, got {flat.shape[1]}")
        return torch.polar(flat[:, 0].clamp_min(0.0), torch.atan2(flat[:, 2], flat[:, 1]))
    raise ValueError(f"unknown geometry {geometry!r}")


def build_bridge(geometry: Geometry) -> LinearFlowBridge | GeodesicFlowBridge:
    """Return the production bridge for a geometry.

    Args:
        geometry: Which geometry to build for.

    Returns:
        The bridge instance.

    Raises:
        ValueError: If the geometry is unknown.
    """
    if geometry == "euclidean":
        return LinearFlowBridge()
    if geometry == "cylindrical":
        return GeodesicFlowBridge()
    raise ValueError(f"unknown geometry {geometry!r}")


def build_solver(geometry: Geometry, num_steps: int) -> EuclideanODESolver | CylindricalODESolver:
    """Return the production Heun solver for a geometry.

    Args:
        geometry: Which geometry to build for.
        num_steps: Integration steps over ``[0, 1]``.

    Returns:
        The solver instance.

    Raises:
        ValueError: If the geometry is unknown or ``num_steps`` is not positive.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    if geometry == "euclidean":
        return EuclideanODESolver(num_steps=num_steps)
    if geometry == "cylindrical":
        return CylindricalODESolver(num_steps=num_steps)
    raise ValueError(f"unknown geometry {geometry!r}")


class ToyVelocityField(nn.Module):
    """A small velocity field over single complex numbers.

    The input encoding is shared across geometries on purpose: the network is
    handed ``(re, im, m, cos phi, sin phi, t)`` regardless of which state
    representation it was called with, so a difference between the arms cannot be
    a difference in what the network was allowed to see. Only the *output* is
    geometry specific -- ``(v_re, v_im)`` in the plane, ``(u_m, u_phi)`` on the
    cylinder -- because those tangent spaces genuinely differ.

    Args:
        geometry: Which state representation ``forward`` will receive.
        width: Hidden width of the MLP.
        depth: Number of hidden layers, at least one.

    Raises:
        ValueError: If the geometry is unknown, or width or depth is not
            positive.
    """

    def __init__(self, geometry: Geometry, width: int = 128, depth: int = 3) -> None:
        super().__init__()
        if geometry not in ("euclidean", "cylindrical"):
            raise ValueError(f"unknown geometry {geometry!r}")
        if width < 1:
            raise ValueError(f"width must be positive, got {width}")
        if depth < 1:
            raise ValueError(f"depth must be positive, got {depth}")

        self.geometry: Geometry = geometry
        layers: list[nn.Module] = [nn.Linear(6, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.SiLU()])
        layers.append(nn.Linear(width, 2))
        self.net = nn.Sequential(*layers)

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """Decode a geometry-specific state into the shared encoding.

        Args:
            state: State tensor of shape ``[n, C, 1, 1]``.

        Returns:
            Tensor of shape ``[n, 5]``: ``(re, im, m, cos phi, sin phi)``.
        """
        flat = state.reshape(state.shape[0], state.shape[1])
        if self.geometry == "euclidean":
            real, imag = flat[:, 0], flat[:, 1]
            amplitude = torch.sqrt(real * real + imag * imag).clamp_min(_AMPLITUDE_FLOOR)
            cosine, sine = real / amplitude, imag / amplitude
        else:
            amplitude = flat[:, 0].clamp_min(0.0)
            cosine, sine = flat[:, 1], flat[:, 2]
            real, imag = amplitude * cosine, amplitude * sine
        return torch.stack([real, imag, amplitude, cosine, sine], dim=1)

    def forward(self, state: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the tangent velocity at a state and a time.

        Args:
            state: State tensor of shape ``[n, C, 1, 1]``.
            t: Times of shape ``[n]`` or broadcastable to it, in ``[0, 1]``.

        Returns:
            Velocity of shape ``[n, 2, 1, 1]``, in this geometry's tangent space.
        """
        features = self.encode(state)
        time = t.reshape(-1, 1).to(features.dtype).expand(features.shape[0], 1)
        velocity = self.net(torch.cat([features, time], dim=1))
        return velocity.unsqueeze(-1).unsqueeze(-1)


def straightness(residual: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
    """Regression residual normalised by the displacement it had to explain.

    Because both bridges carry a conditional velocity that is constant in ``t``,
    this is the rectified-flow straightness statistic and not merely the training
    loss under another name. Normalising makes the plane and the cylinder
    comparable despite living in different tangent spaces: ``0`` is a perfectly
    straight field, ``1`` is a field that explains none of the displacement.

    Args:
        residual: Per-sample squared error, shape ``[n]``.
        displacement: Per-sample squared target velocity, shape ``[n]``.

    Returns:
        Scalar tensor.

    Raises:
        ValueError: If the shapes disagree or the displacement is degenerate.
    """
    if residual.shape != displacement.shape:
        raise ValueError(
            f"residual and displacement must match, got {tuple(residual.shape)} "
            f"and {tuple(displacement.shape)}"
        )
    total = displacement.mean()
    if float(total) <= 0.0:
        raise ValueError("displacement is degenerate; nothing to normalise against")
    return residual.mean() / total
