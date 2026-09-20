"""Synthetic complex-valued distributions on the cylinder with tunable
amplitude-phase dependence.

The cylinder ``R+ x S^1`` is the geometry the project claims; a synthetic
target on it is the only place where a claim about that geometry can be
measured without a network, a loss weighting, or a reconstruction pipeline in
the way. This module builds such a target, with one knob that no MRI or audio
cohort exposes: **how strongly amplitude and phase depend on each other**.

Why the knob matters
--------------------
A coupling that factorises the transport cost across amplitude and phase,

    min_pi Int (c_A + c_theta) dpi
        >= min_{pi_A} Int c_A dpi_A  +  min_{pi_theta} Int c_theta dpi_theta

holds with equality only when both the source and the target are product
measures, ``p(A, theta) = p(A) p(theta)``. The inequality is otherwise strict,
because independently optimal marginal couplings need not be realisable by a
single joint coupling. Real k-space and real spectrograms are firmly on the
strict side. This module lets that side be dialled in and measured rather than
argued about.

Construction
------------
Dependence is imposed with a **Gaussian copula**, which is what makes the sweep
a controlled experiment: for every value of ``coupling`` the amplitude marginal
and the phase marginal are *exactly* the same, and only the joint changes. A
difference measured across the sweep therefore cannot be a marginal effect.

Two dependence structures are provided, and they differ in whether the circle's
topology is respected:

* ``"spiral"`` -- amplitude is comonotone with the angle through the copula.
  At ``coupling = 1`` amplitude is a strictly increasing function of ``theta``
  on ``(-pi, pi]``, so it jumps at the branch cut. This is the Archimedean
  spiral, and the cut is real: an Archimedean spiral genuinely has one.
* ``"cardioid"`` -- the copula is driven by ``cos(theta - phase_offset)``
  instead, which is periodic. There is no cut, and the joint is mirror
  symmetric about ``phase_offset``. This is the structure to reach for when a
  reader might object that the spiral's discontinuity is an artefact, and it is
  the closer analogue of the conjugate symmetry that real-image k-space carries.

Presets
-------
``COUPLING_PRESETS`` names the three settings the gate experiments use:
``"independent"`` (0.0), ``"partial"`` (0.5) and ``"comonotone"`` (1.0). Note
that ``coupling`` is the *latent* Gaussian correlation, not an observable
correlation of the samples; the observable one is lower, and
:func:`cyfm.metrics.circular_linear_correlation` reports it.

Example:
    >>> toy = CylinderToy(coupling=1.0, structure="spiral")
    >>> amplitude, phase = toy.sample_polar(4096)
    >>> float(circular_linear_correlation(amplitude, phase)) > 0.9
    True
"""

from __future__ import annotations

import math
from typing import Final, Literal

import torch

__all__ = [
    "COUPLING_PRESETS",
    "AmplitudeMode",
    "CylinderToy",
    "Structure",
    "PhaseMode",
    "cylinder_prior",
]

# Guards against division by a zero-width CDF cell and against feeding an exact
# 0 or 1 to the normal quantile function, which is infinite there.
_EPS: Final[float] = 1e-12
_PROB_EPS: Final[float] = 1e-7

Structure = Literal["spiral", "cardioid"]

#: ``(mean, concentration, weight)`` of one von Mises component of the phase
#: marginal. Concentration is the von Mises ``kappa``; larger is tighter.
PhaseMode = tuple[float, float, float]

#: ``(mean, standard deviation, weight)`` of one Gaussian component of the
#: amplitude marginal, truncated to the positive half line by the sampling grid.
AmplitudeMode = tuple[float, float, float]

COUPLING_PRESETS: Final[dict[str, float]] = {
    "independent": 0.0,
    "partial": 0.5,
    "comonotone": 1.0,
}

_DEFAULT_PHASE_MODES: Final[tuple[PhaseMode, ...]] = (
    (-2.0, 4.0, 1.0),
    (0.6, 6.0, 1.0),
    (2.4, 3.0, 1.0),
)

_DEFAULT_AMPLITUDE_MODES: Final[tuple[AmplitudeMode, ...]] = (
    (0.35, 0.06, 1.0),
    (0.90, 0.08, 1.0),
)


class _GridDistribution:
    """A 1D distribution held as a piecewise-constant density on a fixed grid.

    Sampling and CDF evaluation both go through the piecewise-linear CDF built
    from that density, so ``cdf`` and ``icdf`` are exact inverses of each other
    up to grid resolution. The density is normalised on construction, which is
    what lets callers pass unnormalised kernels (a von Mises without its Bessel
    factor, a Gaussian without its ``sqrt(2 pi)``).

    Args:
        edges: Bin edges, shape ``[G + 1]``, strictly increasing.
        density: Non-negative unnormalised density per bin, shape ``[G]``.
    """

    def __init__(self, edges: torch.Tensor, density: torch.Tensor) -> None:
        if edges.ndim != 1 or density.ndim != 1:
            raise ValueError("edges and density must be 1D")
        if edges.numel() != density.numel() + 1:
            raise ValueError(
                f"edges must have one more element than density, got "
                f"{edges.numel()} and {density.numel()}"
            )
        if torch.any(density < 0):
            raise ValueError("density must be non-negative")

        widths = edges[1:] - edges[:-1]
        if torch.any(widths <= 0):
            raise ValueError("edges must be strictly increasing")

        mass = density * widths
        total = mass.sum()
        if float(total) <= 0.0:
            raise ValueError("density integrates to zero")

        cdf = torch.cat([torch.zeros(1, dtype=mass.dtype, device=mass.device), mass.cumsum(0)])
        cdf = cdf / total
        cdf[-1] = 1.0

        self.edges = edges
        self.mass = mass / total
        self.cdf = cdf

    @property
    def centers(self) -> torch.Tensor:
        """Bin centres, shape ``[G]``."""
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    def icdf(self, u: torch.Tensor) -> torch.Tensor:
        """Map uniform variates to samples by inverting the CDF.

        Args:
            u: Values in ``[0, 1]``, any shape.

        Returns:
            Samples with the same shape as ``u``.
        """
        idx = torch.searchsorted(self.cdf, u.contiguous().reshape(-1), right=True) - 1
        idx = idx.clamp(0, self.edges.numel() - 2)
        lo, hi = self.cdf[idx], self.cdf[idx + 1]
        left, right = self.edges[idx], self.edges[idx + 1]
        frac = (u.reshape(-1) - lo) / (hi - lo).clamp_min(_EPS)
        return (left + frac * (right - left)).reshape(u.shape)

    def cdf_at(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the CDF, the inverse direction of :meth:`icdf`.

        Args:
            x: Points at which to evaluate, any shape.

        Returns:
            Values in ``[0, 1]`` with the same shape as ``x``.
        """
        idx = torch.searchsorted(self.edges, x.contiguous().reshape(-1), right=True) - 1
        idx = idx.clamp(0, self.edges.numel() - 2)
        left, right = self.edges[idx], self.edges[idx + 1]
        lo, hi = self.cdf[idx], self.cdf[idx + 1]
        frac = (x.reshape(-1) - left) / (right - left).clamp_min(_EPS)
        return (lo + frac * (hi - lo)).clamp(0.0, 1.0).reshape(x.shape)


class _PeriodicUniformiser:
    """Maps an angle to a uniform variate through a periodic statistic.

    Uses ``cos(theta - offset)``, whose distribution under the phase marginal is
    computed on the same grid the marginal lives on. The result is uniform on
    ``[0, 1]`` by construction, so it can drive a Gaussian copula without
    disturbing any marginal -- and unlike the angle itself it has no branch cut.

    Args:
        phase: The phase marginal, as a grid distribution over ``(-pi, pi]``.
        offset: The angle about which the statistic is symmetric, in radians.
    """

    def __init__(self, phase: _GridDistribution, offset: float) -> None:
        statistic = torch.cos(phase.centers - offset)
        order = torch.argsort(statistic)
        mass = phase.mass[order]
        # Midpoint convention: a cell's cumulative probability is taken at its
        # centre, which keeps the image strictly inside (0, 1) and therefore
        # safe for the normal quantile function.
        self._values = statistic[order]
        self._cumulative = mass.cumsum(0) - 0.5 * mass
        self._offset = offset

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        """Uniformise angles.

        Args:
            theta: Angles in radians, any shape.

        Returns:
            Values in ``(0, 1)`` with the same shape as ``theta``.
        """
        statistic = torch.cos(theta - self._offset).contiguous().reshape(-1)
        idx = torch.searchsorted(self._values, statistic).clamp(1, self._values.numel() - 1)
        left, right = self._values[idx - 1], self._values[idx]
        lo, hi = self._cumulative[idx - 1], self._cumulative[idx]
        frac = (statistic - left) / (right - left).clamp_min(_EPS)
        out = lo + frac * (hi - lo)
        return out.clamp(_PROB_EPS, 1.0 - _PROB_EPS).reshape(theta.shape)


class CylinderToy:
    """A synthetic complex distribution on ``R+ x S^1`` with tunable dependence.

    Both marginals are held fixed across the whole ``coupling`` sweep, so the
    only quantity that varies is the dependence between amplitude and phase.
    See the module docstring for why that invariance is the point.

    Args:
        coupling: Latent Gaussian correlation in ``[0, 1]``. ``0`` makes
            amplitude and phase independent; ``1`` makes amplitude a
            deterministic function of phase. Not an observable correlation --
            use :func:`cyfm.metrics.circular_linear_correlation` for that.
        structure: ``"spiral"`` for comonotone dependence on the angle itself
            (with a branch cut), ``"cardioid"`` for periodic dependence through
            ``cos(theta - phase_offset)`` (no cut, mirror symmetric).
        phase_modes: von Mises components ``(mean, kappa, weight)`` of the phase
            marginal. A non-uniform phase marginal is deliberate: between two
            uniform phase marginals circular optimal transport is degenerate and
            has no work to do.
        amplitude_modes: Gaussian components ``(mean, std, weight)`` of the
            amplitude marginal, truncated to the positive half line.
        phase_offset: Symmetry angle for ``structure="cardioid"``, in radians.
            Unused by ``"spiral"``.
        grid_size: Number of bins per marginal. Controls the resolution of the
            inverse-CDF sampler.
        device: Device the grids and samples live on.
        dtype: Floating point dtype of the grids and of the polar samples.

    Raises:
        ValueError: If ``coupling`` is outside ``[0, 1]``, ``structure`` is
            unknown, ``grid_size`` is too small, a component weight is
            non-positive, or an amplitude standard deviation is non-positive.
    """

    def __init__(
        self,
        coupling: float = 0.0,
        structure: Structure = "spiral",
        phase_modes: tuple[PhaseMode, ...] = _DEFAULT_PHASE_MODES,
        amplitude_modes: tuple[AmplitudeMode, ...] = _DEFAULT_AMPLITUDE_MODES,
        phase_offset: float = 0.0,
        grid_size: int = 4096,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if not 0.0 <= coupling <= 1.0:
            raise ValueError(f"coupling must be in [0, 1], got {coupling}")
        if structure not in ("spiral", "cardioid"):
            raise ValueError(f"structure must be 'spiral' or 'cardioid', got {structure!r}")
        if grid_size < 16:
            raise ValueError(f"grid_size must be >= 16, got {grid_size}")
        if not phase_modes:
            raise ValueError("phase_modes must not be empty")
        if not amplitude_modes:
            raise ValueError("amplitude_modes must not be empty")

        self.coupling = float(coupling)
        self.structure: Structure = structure
        self.phase_offset = float(phase_offset)
        self.grid_size = int(grid_size)
        self.device = torch.device(device)
        self.dtype = dtype

        self._phase = self._build_phase(phase_modes)
        self._amplitude = self._build_amplitude(amplitude_modes)
        self._periodic = (
            _PeriodicUniformiser(self._phase, self.phase_offset)
            if structure == "cardioid"
            else None
        )

    def _build_phase(self, modes: tuple[PhaseMode, ...]) -> _GridDistribution:
        """Grid the von Mises mixture over ``(-pi, pi]``."""
        edges = torch.linspace(
            -math.pi, math.pi, self.grid_size + 1, device=self.device, dtype=self.dtype
        )
        centers = 0.5 * (edges[:-1] + edges[1:])
        density = torch.zeros_like(centers)
        for mean, kappa, weight in modes:
            if weight <= 0.0:
                raise ValueError(f"phase mode weight must be positive, got {weight}")
            if kappa < 0.0:
                raise ValueError(f"phase mode concentration must be >= 0, got {kappa}")
            # The von Mises normaliser 1 / (2 pi I0(kappa)) is constant in theta
            # and is absorbed by the grid normalisation, so no Bessel is needed.
            # kappa is subtracted first to keep the exponent bounded above by 0.
            density = density + weight * torch.exp(kappa * (torch.cos(centers - mean) - 1.0))
        return _GridDistribution(edges, density)

    def _build_amplitude(self, modes: tuple[AmplitudeMode, ...]) -> _GridDistribution:
        """Grid the truncated Gaussian mixture over the positive half line."""
        upper = max(mean + 5.0 * std for mean, std, _ in modes)
        edges = torch.linspace(
            0.0, float(upper), self.grid_size + 1, device=self.device, dtype=self.dtype
        )
        centers = 0.5 * (edges[:-1] + edges[1:])
        density = torch.zeros_like(centers)
        for mean, std, weight in modes:
            if weight <= 0.0:
                raise ValueError(f"amplitude mode weight must be positive, got {weight}")
            if std <= 0.0:
                raise ValueError(f"amplitude mode std must be positive, got {std}")
            density = density + weight * torch.exp(-0.5 * ((centers - mean) / std) ** 2)
        return _GridDistribution(edges, density)

    def sample_polar(
        self, n: int, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw ``n`` samples in polar coordinates.

        Args:
            n: Number of samples, must be positive.
            generator: Optional RNG for reproducibility.

        Returns:
            ``(amplitude, phase)``, both shape ``[n]``. Amplitude is positive,
            phase lies in ``(-pi, pi]``.

        Raises:
            ValueError: If ``n`` is not positive.
        """
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")

        angle_normal = torch.randn(n, generator=generator, device=self.device, dtype=self.dtype)
        residual_normal = torch.randn(n, generator=generator, device=self.device, dtype=self.dtype)
        return self.polar_from_latents(angle_normal, residual_normal)

    def polar_from_latents(
        self, angle_normal: torch.Tensor, residual_normal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the copula and the inverse CDFs to caller-supplied latents.

        Exposed so a caller can impose structure the sampler does not know about
        -- spatial correlation, most usefully -- while keeping this class's
        guarantee that both marginals are exactly the configured ones for every
        value of ``coupling``. The latents must be standard normal *marginally*;
        any dependence between neighbouring entries passes straight through to
        the field and changes nothing about the pointwise distribution.

        Args:
            angle_normal: Standard normal latents driving the phase, any shape.
            residual_normal: Independent standard normal latents of the same
                shape, driving the part of the amplitude the phase does not
                explain.

        Returns:
            ``(amplitude, phase)``, both shaped like the latents.

        Raises:
            ValueError: If the two latent tensors differ in shape.
        """
        if angle_normal.shape != residual_normal.shape:
            raise ValueError(
                f"latents must share a shape, got {tuple(angle_normal.shape)} and "
                f"{tuple(residual_normal.shape)}"
            )

        phase = self._phase.icdf(torch.special.ndtr(angle_normal))

        if self._periodic is None:
            driver = angle_normal
        else:
            driver = torch.special.ndtri(self._periodic(phase))

        rho = self.coupling
        latent = rho * driver + math.sqrt(max(0.0, 1.0 - rho * rho)) * residual_normal
        amplitude = self._amplitude.icdf(torch.special.ndtr(latent))
        return amplitude, phase

    def sample(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw ``n`` samples as complex numbers.

        Args:
            n: Number of samples, must be positive.
            generator: Optional RNG for reproducibility.

        Returns:
            Complex tensor of shape ``[n]``, ``A * exp(i * theta)``.
        """
        amplitude, phase = self.sample_polar(n, generator=generator)
        return torch.polar(amplitude, phase)

    @classmethod
    def from_preset(cls, preset: str, **kwargs: object) -> CylinderToy:
        """Build a toy from a named entry of :data:`COUPLING_PRESETS`.

        Args:
            preset: One of ``"independent"``, ``"partial"``, ``"comonotone"``.
            **kwargs: Forwarded to :class:`CylinderToy`, except ``coupling``.

        Returns:
            The configured toy.

        Raises:
            ValueError: If ``preset`` is unknown or ``coupling`` is passed.
        """
        if preset not in COUPLING_PRESETS:
            raise ValueError(
                f"unknown preset {preset!r}, expected one of {sorted(COUPLING_PRESETS)}"
            )
        if "coupling" in kwargs:
            raise ValueError("coupling is set by the preset and must not be passed")
        return cls(coupling=COUPLING_PRESETS[preset], **kwargs)  # type: ignore[arg-type]

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(coupling={self.coupling}, "
            f"structure={self.structure!r}, grid_size={self.grid_size})"
        )


def cylinder_prior(
    n: int,
    scale: float = 1.0,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Draw the reference prior: circularly symmetric complex Gaussian noise.

    Amplitude is Rayleigh, phase is uniform on the circle, and the two are
    independent -- which is both the canonical complex noise model and the
    regime in which the product-measure factorisation of the transport cost is
    exact on the prior side. It is also what thermal noise in k-space is.

    Args:
        n: Number of samples, must be positive.
        scale: Sets ``E|z|^2 = scale^2``.
        generator: Optional RNG for reproducibility.
        device: Device to sample on.
        dtype: Floating point dtype of the real and imaginary parts.

    Returns:
        Complex tensor of shape ``[n]``.

    Raises:
        ValueError: If ``n`` is not positive or ``scale`` is not positive.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if scale <= 0.0:
        raise ValueError(f"scale must be positive, got {scale}")
    sigma = scale / math.sqrt(2.0)
    real = sigma * torch.randn(n, generator=generator, device=device, dtype=dtype)
    imag = sigma * torch.randn(n, generator=generator, device=device, dtype=dtype)
    return torch.complex(real, imag)


