"""Structural contracts: what a component must *do*, not what it must inherit.

Three roles in this package are filled by objects that have no reason to share an
ancestor. A velocity field is a :class:`torch.nn.Module`, so a second base class
would put it under multiple inheritance with ``nn.Module`` for nothing. A sampler
is either an ODE integrator or an SDE predictor-corrector, and those two have
genuinely different steps. A coupling is one method. For all three the useful
question is whether an object has the right shape, which is what
:class:`typing.Protocol` asks and what an abstract base class cannot: an ABC also
demands a particular lineage.

Abstract base classes remain where they carry *behaviour* --
:class:`~cyfm.core.manifold.BaseManifold` fixes what the two arms of the paper's
comparison share, and :class:`~cyfm.core.solver.BaseODESolver` supplies the Heun
loop its subclasses reuse. The rule this module follows: a base class when there
is something to inherit, a protocol when there is only something to satisfy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    # Type-checking only, so the runtime edge between these two modules points one
    # way: cyfm.core.manifold imports this module for Sampler, and this module
    # borrows BaseManifold back only for an annotation. isinstance against a
    # runtime_checkable protocol matches on method names alone and never resolves it.
    from cyfm.core.manifold import BaseManifold


@runtime_checkable
class VelocityField(Protocol):
    """A network that maps a manifold state and a time to a tangent vector.

    The entire contract the solvers and the training loop rely on. Deliberately
    no ``in_channels`` / ``out_channels``: those are *arguments* to
    :func:`~cyfm.utils.inference.build_model`, chosen from the manifold, and
    nothing reads them back off the model. Requiring them here would invent a
    contract no caller uses and force every architecture to carry two attributes
    for the type checker's benefit.

    Satisfied by :class:`~cyfm.models.unet.CylindricalUNet` and
    :class:`~cyfm.models.mlp.PointwiseVelocityMLP` as written, and also by the
    plain callable :meth:`~cyfm.core.manifold.BaseManifold.wrap_model` returns --
    which is the point, since the score-based arm hands the solver a closure
    rather than a module.
    """

    def __call__(self, state: torch.Tensor, time: torch.Tensor, /) -> torch.Tensor:
        """Evaluate the field.

        Positional-only, because that is how every solver and probe in this
        package calls one. Naming the parameters in the protocol would bind every
        implementation to spell them the same way, which is a constraint on the
        implementations and not on the contract.

        Args:
            state: Manifold state ``[B, state_channels, H, W]``.
            time: Time in ``[0, 1]``, shape ``[B]``.

        Returns:
            Tangent vector ``[B, velocity_channels, H, W]``.
        """
        ...


@runtime_checkable
class Sampler(Protocol):
    """All the evaluation sweep requires: a prior in, a sample out.

    Deliberately weaker than :class:`~cyfm.core.solver.BaseODESolver`. A flow arm
    integrates a velocity with a fixed ``dt`` and so has a meaningful ``step``;
    the diffusion baseline's predictor moves between noise levels and has no
    velocity and no ``dt`` to take. Typing the manifold's factory to this protocol
    lets both be returned without pretending the second is an ODE solver.
    """

    num_steps: int

    @property
    def evaluations(self) -> int:
        """Model calls one :meth:`sample` makes.

        Required, not optional. Probing for it and falling back to Heun's count
        would report a wrong cost for any sampler that forgot to declare one,
        silently, which is the mislabelling the reported figure exists to stop.
        """
        ...

    def sample(
        self,
        model: VelocityField,
        noise: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Integrate from the prior at t=0 to data at t=1.

        ``generator`` is accepted by every sampler so the caller can seed one
        without knowing which it holds; a deterministic solver ignores it.
        """
        ...


@runtime_checkable
class Coupling(Protocol):
    """Reorders a batch of data endpoints against a batch of prior samples.

    One method, and nothing in this package ever asks a coupling what it *is* --
    :mod:`cyfm.evaluate` names the type only in an annotation. A protocol is
    therefore the whole contract, and it lets a caller pass a bare function where
    the two classes in :mod:`cyfm.flow.couplings` would otherwise have to be
    subclassed.

    Conformance is not left to chance: :mod:`cyfm.flow.couplings` asserts each
    implementation against this protocol at module level, so ``mypy`` rejects a
    wrong signature at the definition site exactly as an abstract base class did.
    """

    def __call__(
        self,
        prior: torch.Tensor,
        data: torch.Tensor,
        manifold: BaseManifold,
    ) -> torch.Tensor:
        """Return the data batch reordered to pair with the prior batch.

        Args:
            prior: Prior states ``[B, state_channels, H, W]``.
            data: Data states ``[B, state_channels, H, W]``.
            manifold: The geometry whose metric the pairing is judged in.

        Returns:
            The data batch, permuted along the batch dimension.
        """
        ...


__all__ = ["Coupling", "Sampler", "VelocityField"]
