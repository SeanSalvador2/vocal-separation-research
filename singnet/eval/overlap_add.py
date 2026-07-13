r"""Chunked overlap-add inference with a raised-cosine crossfade (MASTER_PLAN §7.2).

The model consumes exactly ``N_FRAMES`` (256) frames, so full-track inference
slides a fixed-length window over the mixture, separates each chunk, and
crossfades the results. To avoid any frame crop/pad on the inference path we use
a chunk of exactly ``(N_FRAMES - 1) * hop = 261120`` samples (~5.92 s), whose STFT
is exactly 256 frames — its iSTFT returns the identical length, so identity-mask
inference reconstructs the input **exactly** (unit-tested).

Reconstruction is a weighted overlap-add normalised by the accumulated window,
i.e. a partition of unity: with any strictly-positive window the identity mask
reconstructs the signal regardless of window shape; the raised-cosine taper just
makes real (non-identity) chunk seams inaudible.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from ..audio.stft import DEFAULT_HOP, N_FRAMES, STFT, apply_mask

#: Inference chunk length whose STFT is exactly N_FRAMES frames (no crop needed).
EVAL_CHUNK_SAMPLES = (N_FRAMES - 1) * DEFAULT_HOP  # 261120
DEFAULT_OVERLAP = 0.25

# A chunk separator maps a mono chunk waveform (chunk_len,) -> estimate (chunk_len,).
ChunkSeparator = Callable[[Tensor], Tensor]


def raised_cosine_window(chunk_len: int, overlap: float = DEFAULT_OVERLAP) -> Tensor:
    """Window that ramps up/down over the overlap region and is flat 1.0 inside.

    The ramps are raised-cosine (Hann half-cycles); the flat interior guarantees
    the taper only touches the crossfade seams.
    """
    overlap_len = int(round(chunk_len * overlap))
    window = torch.ones(chunk_len, dtype=torch.float32)
    if overlap_len > 0:
        idx = torch.arange(overlap_len, dtype=torch.float32)
        ramp = 0.5 * (1.0 - torch.cos(torch.pi * (idx + 0.5) / overlap_len))
        window[:overlap_len] = ramp
        window[-overlap_len:] = torch.flip(ramp, dims=[0])
    return window


def overlap_add(
    mix_wave: Tensor,
    separate_chunk: ChunkSeparator,
    chunk_len: int = EVAL_CHUNK_SAMPLES,
    overlap: float = DEFAULT_OVERLAP,
) -> Tensor:
    """Slide ``separate_chunk`` over ``mix_wave`` (mono, 1-D) and crossfade.

    Chunk start positions step by ``chunk_len * (1 - overlap)``; a final chunk is
    right-aligned so the tail is always covered. Short inputs are zero-padded to
    one chunk. The output is normalised by the accumulated window weights.
    """
    if mix_wave.dim() != 1:
        raise ValueError("overlap_add expects a mono 1-D waveform")
    total = int(mix_wave.shape[0])
    device = mix_wave.device
    window = raised_cosine_window(chunk_len, overlap).to(device)
    overlap_len = int(round(chunk_len * overlap))
    hop = max(1, int(round(chunk_len * (1.0 - overlap))))

    padded_len = max(total, chunk_len)
    out = torch.zeros(padded_len, dtype=torch.float32, device=device)
    weight = torch.zeros(padded_len, dtype=torch.float32, device=device)
    mix = mix_wave
    if padded_len > total:
        mix = torch.nn.functional.pad(mix_wave, (0, padded_len - total))

    starts = list(range(0, padded_len - chunk_len + 1, hop))
    if not starts or starts[-1] != padded_len - chunk_len:
        starts.append(padded_len - chunk_len)
    for idx, start in enumerate(starts):
        # A true signal boundary has no neighbour to crossfade with, so the
        # first chunk keeps a flat (1.0) left edge and the last chunk a flat
        # right edge — otherwise the taper underflows to 0 and zeroes samples.
        chunk_window = window.clone()
        if idx == 0:
            chunk_window[:overlap_len] = 1.0
        if idx == len(starts) - 1:
            chunk_window[-overlap_len:] = 1.0
        chunk = mix[start : start + chunk_len]
        est = separate_chunk(chunk).to(torch.float32)
        out[start : start + chunk_len] += est * chunk_window
        weight[start : start + chunk_len] += chunk_window

    out = out / weight.clamp_min(1e-8)
    return out[:total]


def model_chunk_separator(model, stft: STFT, device: torch.device | None = None) -> ChunkSeparator:
    """Build a chunk separator that runs the SingNet model (mask -> masked iSTFT)."""
    device = device or next(model.parameters()).device

    @torch.no_grad()
    def separate(chunk: Tensor) -> Tensor:
        wave = chunk.to(device).unsqueeze(0)  # (1, L)
        spec = stft.transform(wave)
        mask = model(spec.abs()[:, : spec.shape[-2] - 1, :])  # drop Nyquist -> (1, 2048, T)
        est = stft.inverse(apply_mask(mask, spec), length=chunk.shape[0])
        return est.squeeze(0).cpu()

    return separate


def identity_chunk_separator(stft: STFT) -> ChunkSeparator:
    """Chunk separator with an all-ones mask (STFT round-trip) — for tests/demos."""

    def separate(chunk: Tensor) -> Tensor:
        wave = chunk.unsqueeze(0)
        spec = stft.transform(wave)
        mask = torch.ones(1, spec.shape[-2] - 1, spec.shape[-1])
        est = stft.inverse(apply_mask(mask, spec), length=chunk.shape[0])
        return est.squeeze(0)

    return separate


def separate_track(
    model,
    mix_wave: Tensor,
    stft: STFT,
    device: torch.device | None = None,
    overlap: float = DEFAULT_OVERLAP,
) -> Tensor:
    """Full-track vocal estimate for a mono mixture via model overlap-add."""
    return overlap_add(mix_wave, model_chunk_separator(model, stft, device), EVAL_CHUNK_SAMPLES, overlap)
