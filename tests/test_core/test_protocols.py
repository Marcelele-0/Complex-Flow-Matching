"""The structural contracts are only worth something if something checks them.

A ``Protocol`` is enforced by the type checker at every call site, which is where
its value is. These tests add the half ``mypy`` cannot give: that the objects the
registries actually hand out satisfy the contracts at runtime, including the
members an abstract base class would have forced and a protocol only declares.
"""

from __future__ import annotations

import inspect
from typing import cast

import pytest
import torch

from cyfm.core.protocols import Coupling, Sampler, VelocityField
from cyfm.core.registry import COUPLINGS, MODELS, SOLVERS
from cyfm.flow.couplings import COUPLING_IMPLEMENTATIONS
from cyfm.utils.inference import build_model  # noqa: F401  populates MODELS


class TestVelocityField:
    @pytest.mark.parametrize("name", ["c_unet", "mlp"])
    def test_registered_models_are_velocity_fields(self, name: str) -> None:
        model = MODELS.build(name, in_channels=3, out_channels=2)
        assert isinstance(model, VelocityField)

    @pytest.mark.parametrize("name", ["c_unet", "mlp"])
    def test_registered_models_honour_the_shape_contract(self, name: str) -> None:
        """The protocol states shapes; nothing but a call can confirm them."""
        field: VelocityField = MODELS.build(name, in_channels=3, out_channels=2)
        out = field(torch.rand(2, 3, 16, 16), torch.rand(2))
        assert out.shape == (2, 2, 16, 16)

    def test_a_wrapped_model_is_still_a_velocity_field(self) -> None:
        """``wrap_model`` returns a closure, not a module: the protocol covers both.

        This is the reason the contract is a protocol rather than a base class.
        The score-based arm hands the solver a plain function, which no
        ``nn.Module`` subclass could be.
        """
        from cyfm.manifolds.complex_diffusion import ComplexDiffusionManifold

        manifold = ComplexDiffusionManifold()
        wrapped = manifold.wrap_model(MODELS.build("mlp", in_channels=2, out_channels=2))
        assert isinstance(wrapped, VelocityField)


class TestSampler:
    @pytest.mark.parametrize("name", ["cylindrical", "euclidean", "pc_diffusion"])
    def test_registered_solvers_are_samplers(self, name: str) -> None:
        solver = SOLVERS.build(name, num_steps=2)
        assert isinstance(solver, Sampler)

    @pytest.mark.parametrize("name", ["cylindrical", "euclidean", "pc_diffusion"])
    def test_every_solver_declares_its_own_cost(self, name: str) -> None:
        """``evaluations`` is what the few-step tables report as the budget.

        Abstract on both solver base classes rather than defaulting to Heun's
        ``2n-1``: a default would let a sampler with a different budget quote
        someone else's number into a published table.
        """
        solver = SOLVERS.build(name, num_steps=3)
        assert isinstance(solver.evaluations, int)
        assert solver.evaluations >= 1

    @pytest.mark.parametrize("name", ["cylindrical", "euclidean", "pc_diffusion"])
    def test_every_solver_accepts_a_generator(self, name: str) -> None:
        """Declared on the base classes, not left to each subclass to remember.

        ``evaluate.py`` passes a seeded generator to whichever sampler the
        geometry returns. A solver that omitted the parameter used to type-check
        and then fail there; the base classes now declare it.
        """
        solver = SOLVERS.build(name, num_steps=2)
        parameters = inspect.signature(solver.sample).parameters
        assert "generator" in parameters


class TestCoupling:
    @pytest.mark.parametrize("name", sorted(COUPLINGS.list()))
    def test_every_registered_coupling_satisfies_the_protocol(self, name: str) -> None:
        coupling = COUPLINGS.build(name)
        assert isinstance(coupling, Coupling)

    def test_the_registry_resolves_only_to_checked_implementations(self) -> None:
        """Every key maps to a class in the statically checked tuple.

        ``COUPLING_IMPLEMENTATIONS`` is annotated ``tuple[type[Coupling], ...]``,
        so ``mypy`` rejects a wrong signature at the definition site -- the one
        guarantee the abstract base class used to provide. This asserts the tuple
        is not merely decorative: nothing reaches callers that bypassed it.
        """
        registered = {cast(type, COUPLINGS.get(name)) for name in COUPLINGS.list()}
        assert registered == set(COUPLING_IMPLEMENTATIONS)
