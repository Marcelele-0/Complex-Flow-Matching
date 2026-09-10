"""Probability paths and bridges for Complex Flow Matching (CFM)."""

from __future__ import annotations

# Distributions the probability path may start from at ``t = 0``, selected by
# ``training.bridge`` and mirrored by ``evaluate.bridge``:
#
#   "noise"    the manifold's own prior. The model learns an unconditional
#              generator and the measurement enters only at inference time.
#   "aliased"  the zero-filled reconstruction of the same slice. The model
#              learns a conditional map from measurement to clean image, so
#              training and inference start from the same distribution.
#
# The two endpoints are not interchangeable at inference: a checkpoint trained
# from "aliased" expects the measurement itself at ``t = 0``, not a partially
# noised target, which is why both scripts read the same vocabulary.
BRIDGE_ENDPOINTS = ("noise", "aliased")

__all__ = [
    "BaseCoupling",
    "IndependentCoupling",
    "OptimalTransportCoupling",
    "build_coupling",
    "BRIDGE_ENDPOINTS",
]
