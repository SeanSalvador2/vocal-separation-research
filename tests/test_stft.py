"""STFT/iSTFT round-trip and Nyquist-row handling (MASTER_PLAN §5, G0)."""

from __future__ import annotations

import torch

from singnet.audio import (
    N_FRAMES,
    N_FREQ_BINS,
    N_MASK_BINS,
    STFT,
    analyze_chunk,
    append_nyquist,
    apply_mask,
    crop_or_pad_frames,
    drop_nyquist,
)


def _reconstruction_db(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(10 * torch.log10(((x - y) ** 2).mean() / (x**2).mean()))


def test_roundtrip_below_minus_60_db(stft: STFT) -> None:
    torch.manual_seed(0)
    x = 0.1 * torch.randn(1, 264600)
    y = stft.inverse(stft.transform(x), length=x.shape[-1])
    assert _reconstruction_db(x, y) < -60.0


def test_roundtrip_stereo_batch(stft: STFT) -> None:
    torch.manual_seed(1)
    x = 0.1 * torch.randn(3, 200000)
    y = stft.inverse(stft.transform(x), length=x.shape[-1])
    assert _reconstruction_db(x, y) < -60.0


def test_spectrogram_shapes(stft: STFT) -> None:
    x = 0.1 * torch.randn(2, 264600)
    spec = stft.transform(x)
    assert spec.shape[-2] == N_FREQ_BINS  # 2049 incl. Nyquist
    assert torch.is_complex(spec)


def test_analyze_chunk_crops_to_network_grid(stft: STFT) -> None:
    x = 0.1 * torch.randn(2, 264600)
    spec, mag = analyze_chunk(stft, x)
    assert spec.shape[-2:] == (N_FREQ_BINS, N_FRAMES)  # complex keeps all bins
    assert mag.shape[-2:] == (N_MASK_BINS, N_FRAMES)  # magnitude drops Nyquist


def test_nyquist_drop_and_reappend_roundtrip() -> None:
    mag = torch.randn(2, N_FREQ_BINS, N_FRAMES)
    dropped = drop_nyquist(mag)
    assert dropped.shape[-2] == N_MASK_BINS
    reappended = append_nyquist(dropped, value=1.0)
    assert reappended.shape[-2] == N_FREQ_BINS
    assert torch.allclose(reappended[..., :N_MASK_BINS, :], dropped)
    assert torch.allclose(reappended[..., N_MASK_BINS, :], torch.ones(2, N_FRAMES))


def test_apply_mask_preserves_phase(stft: STFT) -> None:
    x = 0.1 * torch.randn(1, 264600)
    spec, _ = analyze_chunk(stft, x)
    mask = torch.ones(1, N_MASK_BINS, N_FRAMES)  # identity mask
    masked = apply_mask(mask, spec)
    # Identity mask must leave the complex spectrogram untouched (phase intact).
    assert torch.allclose(masked, spec, atol=1e-5)


def test_crop_or_pad_frames_center() -> None:
    x = torch.arange(259).float().reshape(1, 1, 259)
    cropped = crop_or_pad_frames(x, 256)
    assert cropped.shape[-1] == 256
    # short input pads symmetrically to the target length
    short = torch.ones(1, 1, 100)
    padded = crop_or_pad_frames(short, 256)
    assert padded.shape[-1] == 256
