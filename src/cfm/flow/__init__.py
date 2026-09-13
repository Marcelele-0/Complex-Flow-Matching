"""Probability paths and bridges for Complex Flow Matching (CFM)."""

from __future__ import annotations

# The distribution the probability path starts from at ``t = 0``, selected by
# ``training.bridge``. Only the manifold's own prior remains: this paper generates
# complex fields, it does not reconstruct them from measurements, so there is no
# measurement to start from. The tuple is kept rather than inlined so an invalid
# value still fails with a list of what is allowed.
BRIDGE_ENDPOINTS = ("noise",)

__all__ = [
    "BaseCoupling",
    "IndependentCoupling",
    "OptimalTransportCoupling",
    "build_coupling",
    "BRIDGE_ENDPOINTS",
]
