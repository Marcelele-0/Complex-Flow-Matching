"""Readers for the compact HDF5 stores the real-data experiments train on.

Nothing here touches a raw acquisition. The knee and speech cohorts are turned
into one field per sample ahead of time -- coil-combined and cropped for MRI, cut
into segments for speech -- by the builders in ``scripts/data/``, and these
modules only read the result. That separation is what keeps ESPIRiT, NYU
credentials and hours of preprocessing out of a training run.
"""

from __future__ import annotations
