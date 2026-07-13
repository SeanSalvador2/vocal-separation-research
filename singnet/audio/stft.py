"""Differentiable STFT / iSTFT front-end and mask/reconstruction helpers.

Design pins (MASTER_PLAN §5):

* ``n_fft = 4096``, ``hop = 1024`` (= n_fft/4), Hann window, ``center=True``.
  A 6 s / 44.1 kHz chunk -> 2049 freq bins x 259 frames.
* The network consumes the **first 2048 bins**. Bin 2048 (Nyquist) is dropped
  from the network input and the mask, and re-appended with mask ``1.0`` at
  reconstruction — negligible energy, keeps every feature-map dimension a power
  of two.
* Frames are centre-cropped/padded to **256** so the 5-stride-2 encoder reaches
  a clean 64 x 8 bottleneck.

Reconstruction happens *in the cropped (2049 bins x 256 frames) domain*: the
estimate ``v̂`` and the target ``v`` are both ``iSTFT`` of a 256-frame
spectrogram, so they share an identical analysis grid and equal length. The
<=3 dropped edge frames (~3 ms of a 6 s chunk) are immaterial and, being
loss-arm-independent, do not threaten the controlled comparison.

Everything here is differentiable (``torch.stft`` / ``torch.istft``), so the
waveform-domain losses (sisdr, l1mrstft) get gradients back to the mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

# --- pinned front-end constants (MASTER_PLAN §5) ----------------------------
DEFAULT_N_FFT = 4096
DEFAULT_HOP = 1024
N_FREQ_BINS = DEFAULT_N_FFT // 2 + 1  # 2049, includes Nyquist
N_MASK_BINS = DEFAULT_N_FFT // 2  # 2048, the bins the network sees / masks
N_FRAMES = 256  # cropped frame count feeding the network


@dataclass(frozen=True)
class STFTConfig:
    """Immutable STFT settings; the project default is the no-arg instance."""

    n_fft: int = DEFAULT_N_FFT
    hop: int = DEFAULT_HOP
    win_length: int | None = None  # defaults to n_fft
    center: bool = True

    @property
    def effective_win_length(self) -> int:
        return self.win_length if self.win_length is not None else self.n_fft


class STFT(nn.Module):
    """A thin, differentiable STFT/iSTFT wrapper holding a Hann window buffer.

    The window is registered as a (non-persistent) buffer so it follows the
    module's device/dtype under ``.to(...)`` without being saved in checkpoints.
    """

    def __init__(
        self,
        n_fft: int = DEFAULT_N_FFT,
        hop: int = DEFAULT_HOP,
        win_length: int | None = None,
        center: bool = True,
    ) -> None:
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop = int(hop)
        self.win_length = int(win_length) if win_length is not None else int(n_fft)
        self.center = bool(center)
        window = torch.hann_window(self.win_length, periodic=True)
        self.register_buffer("window", window, persistent=False)

    def transform(self, wave: Tensor) -> Tensor:
        """Waveform ``(..., L)`` -> complex spectrogram ``(..., F, T)``."""
        lead_shape = wave.shape[:-1]
        flat = wave.reshape(-1, wave.shape[-1])
        spec = torch.stft(
            flat,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win_length,
            window=self.window.to(flat.device, flat.dtype),
            center=self.center,
            return_complex=True,
        )
        return spec.reshape(*lead_shape, spec.shape[-2], spec.shape[-1])

    def inverse(self, spec: Tensor, length: int | None = None) -> Tensor:
        """Complex spectrogram ``(..., F, T)`` -> waveform ``(..., L)``."""
        lead_shape = spec.shape[:-2]
        flat = spec.reshape(-1, spec.shape[-2], spec.shape[-1])
        if length is None:
            length = self.hop * (flat.shape[-1] - 1)
        wave = torch.istft(
            flat,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win_length,
            window=self.window.to(flat.device, torch.float32),
            center=self.center,
            length=length,
        )
        return wave.reshape(*lead_shape, wave.shape[-1])

    def forward(self, wave: Tensor) -> Tensor:  # noqa: D401 - alias
        """Alias for :meth:`transform` so ``STFT()(wave)`` works."""
        return self.transform(wave)


# --- frame cropping ---------------------------------------------------------

def crop_or_pad_frames(spec: Tensor, n_frames: int = N_FRAMES) -> Tensor:
    """Centre-crop (or symmetric zero-pad) the frame axis to ``n_frames``.

    Operates on the last dimension (``T``). Cropping keeps the central frames;
    padding is symmetric with zeros. Used to turn the 259-frame chunk STFT into
    the network's 256-frame grid.
    """
    length = spec.shape[-1]
    if length == n_frames:
        return spec
    if length > n_frames:
        start = (length - n_frames) // 2
        return spec[..., start : start + n_frames]
    total = n_frames - length
    left = total // 2
    right = total - left
    return torch.nn.functional.pad(spec, (left, right))


# --- Nyquist / mask helpers -------------------------------------------------

def drop_nyquist(mag: Tensor) -> Tensor:
    """Keep the first ``N_MASK_BINS`` (2048) freq bins, dropping the Nyquist row."""
    return mag[..., :N_MASK_BINS, :]


def append_nyquist(mask: Tensor, value: float = 1.0) -> Tensor:
    """Re-append one Nyquist freq row of constant ``value`` to a 2048-bin mask.

    Produces a ``(..., 2049, T)`` mask so it aligns with the full mixture
    spectrogram at reconstruction (MASTER_PLAN §5: Nyquist passed through with
    mask 1.0).
    """
    pad_shape = (*mask.shape[:-2], 1, mask.shape[-1])
    nyquist = mask.new_full(pad_shape, value)
    return torch.cat([mask, nyquist], dim=-2)


def apply_mask(mask: Tensor, mix_spec: Tensor, nyquist_value: float = 1.0) -> Tensor:
    """Apply a real 2048-bin ``mask`` to the complex 2049-bin ``mix_spec``.

    Multiplying a real mask by the complex mixture spectrogram scales the
    magnitude while preserving the **mixture phase** — exactly the mask-family
    reconstruction ``V̂ = M ⊙ |X| e^{i∠X}`` (MASTER_PLAN §1.3, §5).
    """
    mask_full = append_nyquist(mask, nyquist_value)
    return mask_full.to(mix_spec.real.dtype) * mix_spec


# --- chunk-level front-end used by the train loop and evaluator -------------

def analyze_chunk(stft: STFT, wave: Tensor) -> tuple[Tensor, Tensor]:
    """Waveform chunk -> (cropped complex spec ``(..,2049,256)``, mask-bin magnitude
    ``(..,2048,256)``).

    The complex spectrogram keeps all 2049 bins (needed for phase-correct
    reconstruction); the returned magnitude keeps only the 2048 network bins.
    """
    spec = crop_or_pad_frames(stft.transform(wave))
    mag = drop_nyquist(spec.abs())
    return spec, mag


def reconstruct(stft: STFT, mask: Tensor, mix_spec: Tensor) -> Tensor:
    """Reconstruct the estimated waveform from a mask and the mixture spectrogram."""
    return stft.inverse(apply_mask(mask, mix_spec))
