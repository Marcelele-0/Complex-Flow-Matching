"""The audio intake: transform, segmentation, store and the speaker-level split."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import soundfile
import torch

from cfm.data.stft import DEFAULT_STFT, StftProtocol, forward_stft, inverse_stft, segment_frames
from cfm.data.stft_store import StftStoreDataset

BUILDER = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "data" / "build_stft_store.py"


def a_waveform(
    seconds: float, protocol: StftProtocol = DEFAULT_STFT, seed: int = 0
) -> torch.Tensor:
    """Band-limited noise, which exercises every bin rather than a single tone."""
    generator = torch.Generator().manual_seed(seed)
    length = int(seconds * protocol.sample_rate)
    return torch.randn(length, generator=generator, dtype=torch.float32) * 0.1


def test_the_transform_keeps_the_bin_count_the_network_expects() -> None:
    spectrogram = forward_stft(a_waveform(1.0))
    assert spectrogram.shape[0] == DEFAULT_STFT.bins == 64
    assert spectrogram.dtype == torch.complex64


def test_the_full_band_round_trip_is_exact() -> None:
    """The analysis settings satisfy constant overlap-add, so nothing is lost.

    This is the contract the band convention is then allowed to bend: hop = n_fft / 2
    with a periodic Hann window reconstructs to float precision, so any later error is
    attributable to the dropped bin and not to the transform.
    """
    protocol = DEFAULT_STFT
    waveform = a_waveform(0.75, protocol)
    window = torch.hann_window(protocol.n_fft, periodic=True)
    spectrum = torch.stft(
        waveform,
        n_fft=protocol.n_fft,
        hop_length=protocol.hop,
        win_length=protocol.n_fft,
        window=window,
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )
    recovered = torch.istft(
        spectrum,
        n_fft=protocol.n_fft,
        hop_length=protocol.hop,
        win_length=protocol.n_fft,
        window=window,
        center=True,
        length=waveform.numel(),
    )
    assert torch.allclose(waveform, recovered, atol=1e-5)


def a_speechlike_waveform(seconds: float = 0.75, seed: int = 0) -> torch.Tensor:
    """Noise shaped to a ``1/f`` spectrum, which is where speech puts its energy.

    White noise is the wrong probe for the band convention: it spreads energy evenly
    over the bins, so dropping any one of them costs the same and the choice looks
    arbitrary. On a realistic spectrum it is not remotely arbitrary.
    """
    generator = torch.Generator().manual_seed(seed)
    samples = int(seconds * DEFAULT_STFT.sample_rate)
    spectrum = torch.fft.rfft(torch.randn(samples, generator=generator))
    decay = torch.arange(spectrum.numel()).clamp(min=1).float() ** 0.9
    shaped = torch.fft.irfft(spectrum / decay, n=samples)
    return shaped / shaped.abs().max() * 0.5


def test_dropping_nyquist_costs_almost_nothing_on_a_speech_like_spectrum() -> None:
    """The measurement the band convention rests on.

    Dropping DC instead would discard most of the signal, which is why the convention
    is asserted here rather than left to whoever next edits the transform.
    """
    protocol = DEFAULT_STFT
    waveform = a_speechlike_waveform()
    recovered = inverse_stft(forward_stft(waveform, protocol), protocol, length=waveform.numel())
    kept = ((waveform - recovered).norm() / waveform.norm()).item()

    window = torch.hann_window(protocol.n_fft, periodic=True)
    spectrum = torch.stft(
        waveform,
        n_fft=protocol.n_fft,
        hop_length=protocol.hop,
        win_length=protocol.n_fft,
        window=window,
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )
    without_dc = spectrum.clone()
    without_dc[0] = 0
    dropped_dc = torch.istft(
        without_dc,
        n_fft=protocol.n_fft,
        hop_length=protocol.hop,
        win_length=protocol.n_fft,
        window=window,
        center=True,
        length=waveform.numel(),
    )
    cost_of_dc = ((waveform - dropped_dc).norm() / waveform.norm()).item()

    assert kept < 0.01
    assert cost_of_dc > 0.5
    assert kept < cost_of_dc / 50


def test_segments_do_not_overlap_and_a_short_tail_is_dropped() -> None:
    protocol = DEFAULT_STFT
    spectrogram = forward_stft(a_waveform(1.0, protocol), protocol)
    segments = segment_frames(spectrogram, protocol)
    assert segments.shape[1:] == (protocol.bins, protocol.frames)
    assert segments.shape[0] == spectrogram.shape[1] // protocol.frames
    # Segment k is exactly the k-th non-overlapping block of frames, nothing re-used.
    for k in range(segments.shape[0]):
        block = spectrogram[:, k * protocol.frames : (k + 1) * protocol.frames]
        assert torch.equal(segments[k], block)


def test_a_clip_shorter_than_one_segment_yields_nothing() -> None:
    protocol = DEFAULT_STFT
    tiny = forward_stft(a_waveform(protocol.seconds / 4, protocol), protocol)
    assert segment_frames(tiny, protocol).shape[0] == 0


def test_the_protocol_rejects_a_hop_that_breaks_overlap_add() -> None:
    with pytest.raises(ValueError, match="hop must divide n_fft"):
        StftProtocol(n_fft=128, hop=48)


def test_the_transform_is_deterministic_for_one_waveform() -> None:
    waveform = a_waveform(0.5)
    assert torch.equal(forward_stft(waveform), forward_stft(waveform))


def _a_corpus(root: pathlib.Path, speakers: int = 6, per_speaker: int = 2) -> None:
    """A LibriSpeech-shaped tree: <speaker>/<chapter>/<utterance>.flac."""
    for speaker in range(speakers):
        chapter = root / f"{100 + speaker}" / "1"
        chapter.mkdir(parents=True)
        for utterance in range(per_speaker):
            wave = a_waveform(1.0, seed=speaker * 10 + utterance).numpy()
            path = chapter / f"{100 + speaker}-1-{utterance:04d}.flac"
            soundfile.write(str(path), wave, DEFAULT_STFT.sample_rate, format="FLAC")


def _build(tmp_path: pathlib.Path, **extra: str) -> pathlib.Path:
    raw = tmp_path / "raw"
    _a_corpus(raw)
    out = tmp_path / "store" / "dev-clean.h5"
    command = [
        sys.executable,
        str(BUILDER),
        "--raw-dir",
        str(raw),
        "--out",
        str(out),
        "--split",
        "dev-clean",
        *[part for key, value in extra.items() for part in (f"--{key}", value)],
    ]
    done = subprocess.run(command, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return out


def test_the_builder_writes_a_store_the_dataset_reads(tmp_path: pathlib.Path) -> None:
    out = _build(tmp_path)
    manifest = json.loads(out.with_suffix(".json").read_text())
    assert len(manifest["speakers"]) == 6
    assert manifest["protocol"]["bins"] == 64
    assert manifest["sha256"]

    dataset = StftStoreDataset(data_dir=out.parent, store=out.name, role="all")
    assert len(dataset) == len(manifest["segments"])
    sample = dataset[0]
    assert sample.shape == (1, 64, 64)
    assert sample.dtype == torch.complex64


def test_the_split_is_by_speaker_and_never_shares_a_voice(tmp_path: pathlib.Path) -> None:
    out = _build(tmp_path)
    fit = StftStoreDataset(data_dir=out.parent, store=out.name, role="fit", holdout_fraction=0.34)
    held = StftStoreDataset(
        data_dir=out.parent, store=out.name, role="holdout", holdout_fraction=0.34
    )
    assert set(fit.speakers).isdisjoint(held.speakers)
    everything = StftStoreDataset(data_dir=out.parent, store=out.name, role="all")
    assert len(fit) + len(held) == len(everything)


def test_the_speaker_assignment_does_not_move_when_the_cohort_grows(
    tmp_path: pathlib.Path,
) -> None:
    """Hashing the id, not the file order, is what makes a rebuild reproducible."""
    from cfm.data.stft_store import _speaker_role

    before = {str(s): _speaker_role(str(s), 0.2, 0) for s in range(100, 106)}
    after = {str(s): _speaker_role(str(s), 0.2, 0) for s in range(100, 120)}
    assert all(after[k] == v for k, v in before.items())


def test_the_store_is_unnormalised_so_one_transform_owns_normalisation(
    tmp_path: pathlib.Path,
) -> None:
    """The manifold transform divides by the per-field peak, as it does for MRI."""
    out = _build(tmp_path)
    dataset = StftStoreDataset(data_dir=out.parent, store=out.name, role="all")
    peaks = torch.tensor([dataset[i].abs().max() for i in range(min(8, len(dataset)))])
    assert not torch.allclose(peaks, torch.ones_like(peaks))

    normalised = StftStoreDataset(
        data_dir=out.parent,
        store=out.name,
        role="all",
        transform=lambda field: field / field.abs().max().clamp_min(1e-12),
    )
    assert pytest.approx(1.0, abs=1e-5) == normalised[0].abs().max().item()


def test_a_missing_store_names_the_builder(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError, match="build_stft_store.py"):
        StftStoreDataset(data_dir=tmp_path, store="absent.h5")


def test_the_builder_can_delete_each_utterance_as_it_is_consumed(
    tmp_path: pathlib.Path,
) -> None:
    raw = tmp_path / "raw"
    _a_corpus(raw, speakers=2, per_speaker=1)
    out = tmp_path / "store" / "dev-clean.h5"
    done = subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--raw-dir",
            str(raw),
            "--out",
            str(out),
            "--delete-consumed",
        ],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    assert list(raw.rglob("*.flac")) == []
    assert np.asarray(json.loads(out.with_suffix(".json").read_text())["segments"]).size
