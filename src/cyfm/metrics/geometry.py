"""What the path did, measured along the trajectories the solver actually took.

Everything else in this package scores the *endpoint*: a batch of generated
fields against a batch of references. The two probes here score the *path*, and
they are what the paper's central argument rests on.

Both are defined only for an arm whose network emits a velocity. A score-based
baseline regresses something else, for which a straightness residual and an
angular velocity would still evaluate to numbers, and those numbers would mean
nothing. The evaluation sweep therefore omits the keys entirely for such an arm
rather than writing zero or null -- a reader must not be able to mistake "not
applicable" for "measured, and small".

These lived in :mod:`cyfm.evaluate` until this package existed, which is part of
why that module ran to 562 lines and could not be exercised without standing up
a Hydra ``main``.
"""

from __future__ import annotations

import torch

from cyfm.core.manifold import BaseManifold
from cyfm.core.protocols import Coupling, VelocityField

__all__ = ["AngularVelocityProbe", "straightness"]


class AngularVelocityProbe:
    """Wraps a velocity field and records what the solver actually meets.

    Theorem 3 is a statement about the path, not the endpoint: under a Cartesian
    field the induced angular velocity

        theta_dot = (x v_y - y v_x) / A^2

    diverges as ``A -> 0``, while on the cylinder the angular velocity *is* a
    coordinate of the prediction and is bounded by ``pi`` from the range of
    ``atan2``. Recording both along the same trajectories is what turns that from
    an argument into a number.

    Args:
        network: The velocity field to wrap, as any callable of ``(state, time)``.
        manifold: The geometry, which supplies the read-off. This was a name
            matched against a tuple in this module; a geometry now states for
            itself whether the quantity exists and how it is computed, so a new
            one cannot be silently left out of a table it belongs in.
        mid_window: Half-width of the band around ``t = 0.5`` reported separately,
            which is where a chordal path passes closest to the origin.
        enabled: Whether to record at all. False for an arm whose output is not
            a velocity, where the numbers would mean nothing.
    """

    def __init__(
        self,
        network: VelocityField,
        manifold: BaseManifold,
        mid_window: float = 0.1,
        enabled: bool = True,
    ) -> None:
        self.network = network
        self.manifold = manifold
        self.mid_window = mid_window
        # Two different questions, and both must hold. `enabled` asks whether the
        # network emits a velocity at all; the manifold's capability asks whether
        # an angular velocity can be read off this geometry's path. An arm can
        # satisfy the second and not the first -- the score-based baseline carries
        # the flat representation, so it has the formula -- and recording nothing
        # is correct there.
        self.enabled = enabled and manifold.reports_induced_angular_velocity
        # Per-sample extrema of the batch in flight, and the batches already
        # finished. Kept apart because the reduction is over a trajectory, within
        # one batch: a final short batch has a different sample count, and
        # reducing it against the previous one elementwise would either raise or,
        # worse, broadcast two unrelated samples together.
        self._min_amplitude: torch.Tensor | None = None
        self._peak_angular: torch.Tensor | None = None
        self._finished_amplitude: list[torch.Tensor] = []
        self._finished_angular: list[torch.Tensor] = []
        self.peak_angular_mid = 0.0

    def start_batch(self) -> None:
        """Close the batch in flight, so the next one accumulates on its own."""
        if self._min_amplitude is not None:
            self._finished_amplitude.append(self._min_amplitude)
            self._min_amplitude = None
        if self._peak_angular is not None:
            self._finished_angular.append(self._peak_angular)
            self._peak_angular = None

    @property
    def min_amplitude(self) -> torch.Tensor | None:
        """Per-sample minimum amplitude over every trajectory seen so far."""
        batches = [*self._finished_amplitude]
        if self._min_amplitude is not None:
            batches.append(self._min_amplitude)
        return torch.cat(batches, dim=0) if batches else None

    @property
    def peak_angular(self) -> torch.Tensor | None:
        """Per-sample peak angular velocity over every trajectory seen so far."""
        batches = [*self._finished_angular]
        if self._peak_angular is not None:
            batches.append(self._peak_angular)
        return torch.cat(batches, dim=0) if batches else None

    def __call__(self, state: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Record, then delegate.

        Args:
            state: State tensor ``[B, C, H, W]``.
            t: Times ``[B]``.

        Returns:
            The wrapped field's velocity.
        """
        velocity = self.network(state, t)
        if not self.enabled:
            return velocity
        amplitude, angular = self.manifold.induced_angular_velocity(state, velocity)

        magnitude = angular.abs()
        self._min_amplitude = (
            amplitude
            if self._min_amplitude is None
            else torch.minimum(self._min_amplitude, amplitude)
        )
        self._peak_angular = (
            magnitude
            if self._peak_angular is None
            else torch.maximum(self._peak_angular, magnitude)
        )
        if abs(float(t.reshape(-1)[0]) - 0.5) <= self.mid_window:
            self.peak_angular_mid = max(self.peak_angular_mid, float(magnitude.max()))
        return velocity


@torch.no_grad()
def straightness(
    model: torch.nn.Module,
    manifold: BaseManifold,
    data_states: torch.Tensor,
    device: torch.device,
    generator: torch.Generator,
    coupling: Coupling | None = None,
    chunk: int | None = None,
) -> float:
    """Regression residual of the conditional velocity, normalised.

    Computed under the coupling the model was trained with. A model regressed
    toward optimal-transport pairs has to be scored against optimal-transport
    pairs: scoring it against independent ones measures a different regression,
    and once made an OT arm look exactly as unlearnable as its baseline. A
    coupling that removes crossings should lower this, and lowering it is the
    mechanism behind any reduction in solver steps.

    Args:
        model: The trained velocity field.
        manifold: The geometry, supplying the prior and the bridge.
        data_states: Data in the manifold's state representation,
            ``[B, state_channels, H, W]``.
        device: Device to compute on.
        generator: RNG for the prior draw and the time sample. Must live on
            ``device``: torch refuses a CPU generator for a CUDA draw.
        coupling: The pairing training used. ``None`` pairs independently.
        chunk: Fields per forward pass. ``None`` sends the whole batch at once,
            which is what every synthetic run did and what keeps their numbers
            reproducible; a 320x320 cohort does not fit that way, so the fastMRI
            experiments set it. Chunking changes how the generator is consumed, so
            two chunk sizes give slightly different draws and must not be mixed
            within one table.

    Returns:
        ``0`` for a perfectly straight field, ``1`` for one that explains none of
        the displacement.
    """
    batch, _, height, width = data_states.shape
    step = batch if chunk is None else min(chunk, batch)

    residuals, displacements = [], []
    for start in range(0, batch, step):
        block = data_states[start : start + step]
        size = block.shape[0]
        prior = manifold.sample_noise(size, height, width, device, generator=generator)
        times = torch.rand(size, 1, 1, 1, device=device, generator=generator)
        if coupling is not None:
            block = coupling(prior, block, manifold)
        state, target = manifold.bridge(prior, block, times)
        residuals.append((model(state, times.reshape(-1)) - target).square().flatten(1).sum(1))
        displacements.append(target.square().flatten(1).sum(1))

    displacement = torch.cat(displacements)
    total = displacement.mean()
    if float(total) <= 0.0:
        return 0.0
    return float(torch.cat(residuals).mean() / total)
