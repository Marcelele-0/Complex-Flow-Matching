"""The short-time Fourier transform this repository trains on, in one place.

Audio is the paper's second domain and the one where phase is audible rather than
theoretical: a spectrogram's modulus is what a mel-based model keeps and its phase is
what such a model throws away and then guesses back with Griffin-Lim. Representing a
frame as amplitude and phase on the cylinder is therefore the same construction as the
MRI path, applied to a signal where the reconstruction error can be heard.

Both directions live here so that the analysis used to build a store and the synthesis
used to listen to a sample cannot drift apart.

Band convention. ``n_fft=128`` yields 65 bins, and a network wants 64. The bin dropped
is Nyquist, chosen by measurement rather than by symmetry. Both DC and Nyquist are real
by construction, so neither has a free phase, which argues for dropping either; what
separates them is energy. On a speech-like ``1/f`` spectrum the round-trip residual is

    drop DC       0.724
    drop Nyquist  0.0008

because low frequencies carry most of the energy in speech and 8 kHz carries almost
none. On white noise the two are indistinguishable (0.069 against 0.072), which is why
this has to be checked on a realistic spectrum. The cost of the choice is that bin 0
keeps a degenerate phase, constant at 0 or pi; that is one channel of sixty-four, against
losing most of the signal. :func:`inverse_stft` restores the dropped bin as zero.
"""

from __future__ import annotations

import torch

__all__ = [
    "DEFAULT_STFT",
    "StftProtocol",
    "forward_stft",
    "inverse_stft",
    "segment_frames",
]


class StftProtocol:
    """The analysis settings a store is built with, carried as one object.

    Args:
        sample_rate: Sampling rate the audio is decoded at, in hertz.
        n_fft: Transform length. ``n_fft // 2 + 1`` bins are produced and one is
            dropped, so ``n_fft`` should be twice the wanted bin count.
        hop: Samples between consecutive frames. Half of ``n_fft`` satisfies the
            constant-overlap-add condition for a Hann window, which is what makes the
            inverse exact.
        frames: Frames per stored example.

    Raises:
        ValueError: If any field is non-positive, or if ``hop`` does not divide
            ``n_fft`` evenly, which would break constant overlap-add.
    """

    def __init__(
        self, sample_rate: int = 16_000, n_fft: int = 128, hop: int = 64, frames: int = 64
    ) -> None:
        for name, value in (
            ("sample_rate", sample_rate),
            ("n_fft", n_fft),
            ("hop", hop),
            ("frames", frames),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if n_fft % hop:
            raise ValueError(f"hop must divide n_fft, got n_fft={n_fft}, hop={hop}")
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop = int(hop)
        self.frames = int(frames)

    @property
    def bins(self) -> int:
        """Bins kept per frame, after the Nyquist term is dropped."""
        return self.n_fft // 2

    @property
    def segment_samples(self) -> int:
        """Samples one stored example spans."""
        return self.frames * self.hop

    @property
    def seconds(self) -> float:
        """Duration one stored example spans."""
        return self.segment_samples / self.sample_rate

    def as_dict(self) -> dict[str, int | float]:
        """The protocol as plain data, for a manifest or an HDF5 attribute block."""
        return {
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "hop": self.hop,
            "frames": self.frames,
            "bins": self.bins,
            "seconds": round(self.seconds, 6),
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StftProtocol):
            return NotImplemented
        return self.as_dict() == other.as_dict()

    def __repr__(self) -> str:
        return (
            f"StftProtocol(sample_rate={self.sample_rate}, n_fft={self.n_fft}, "
            f"hop={self.hop}, frames={self.frames})"
        )


DEFAULT_STFT = StftProtocol()


def _window(protocol: StftProtocol, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.hann_window(protocol.n_fft, periodic=True, device=device, dtype=dtype)


def forward_stft(waveform: torch.Tensor, protocol: StftProtocol = DEFAULT_STFT) -> torch.Tensor:
    """Complex spectrogram of one waveform, with the Nyquist bin dropped.

    Args:
        waveform: Real samples ``[T]``.
        protocol: Analysis settings.

    Returns:
        Complex tensor ``[bins, n_frames]``.

    Raises:
        ValueError: If ``waveform`` is not one-dimensional.
    """
    if waveform.ndim != 1:
        raise ValueError(f"waveform must be 1-D, got shape {tuple(waveform.shape)}")
    spectrum = torch.stft(
        waveform,
        n_fft=protocol.n_fft,
        hop_length=protocol.hop,
        win_length=protocol.n_fft,
        window=_window(protocol, waveform.device, waveform.dtype),
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    return spectrum[:-1].to(torch.complex64)


def inverse_stft(
    spectrogram: torch.Tensor, protocol: StftProtocol = DEFAULT_STFT, length: int | None = None
) -> torch.Tensor:
    """Waveform of a complex spectrogram, restoring the dropped Nyquist bin as zero.

    Args:
        spectrogram: Complex tensor ``[bins, n_frames]`` as produced by
            :func:`forward_stft`.
        protocol: The settings the spectrogram was produced with.
        length: Samples to return; the inverse is trimmed or zero-padded to it.

    Returns:
        Real tensor ``[T]``.

    Raises:
        ValueError: If the bin count does not match the protocol.
    """
    if spectrogram.ndim != 2:
        raise ValueError(f"spectrogram must be 2-D, got shape {tuple(spectrogram.shape)}")
    if spectrogram.shape[0] != protocol.bins:
        raise ValueError(
            f"spectrogram has {spectrogram.shape[0]} bins, protocol expects {protocol.bins}"
        )
    nyquist = torch.zeros(
        (1, spectrogram.shape[1]), dtype=spectrogram.dtype, device=spectrogram.device
    )
    full = torch.cat([spectrogram, nyquist], dim=0)
    return torch.istft(
        full,
        n_fft=protocol.n_fft,
        hop_length=protocol.hop,
        win_length=protocol.n_fft,
        window=_window(protocol, spectrogram.device, torch.float32),
        center=True,
        normalized=False,
        onesided=True,
        length=length,
    )


def segment_frames(spectrogram: torch.Tensor, protocol: StftProtocol = DEFAULT_STFT) -> torch.Tensor:
    """Cut a spectrogram into non-overlapping examples of ``protocol.frames`` frames.

    Segments do not overlap, so two examples never share a frame and the split by
    speaker cannot leak through a shared window. A tail shorter than one full segment
    is dropped rather than padded, because a padded example is partly silence and the
    amplitude marginal is one of the quantities being measured.

    Args:
        spectrogram: Complex tensor ``[bins, n_frames]``.
        protocol: The settings it was produced with.

    Returns:
        Complex tensor ``[n_segments, bins, frames]``; empty when the spectrogram is
        shorter than one segment.
    """
    bins, available = spectrogram.shape
    count = available // protocol.frames
    if count == 0:
        return spectrogram.new_zeros((0, bins, protocol.frames))
    usable = spectrogram[:, : count * protocol.frames]
    return usable.reshape(bins, count, protocol.frames).permute(1, 0, 2).contiguous()
