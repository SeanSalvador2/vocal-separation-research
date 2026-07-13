r"""Per-band mechanism metrics for the band-split study (Direction 03 §6.3).

The mechanism figure asks *where* an arm's SI-SDR advantage lives across
frequency. For each system, track, and analysis band :math:`b` we report two
band-restricted metrics, both well-defined (no mask/phase asymmetry) and
unit-tested on synthetic band-limited signals:

* **Band-limited SI-SDR** — zero every STFT bin outside :math:`b` in *both* the
  reference and the estimate (each signal's own STFT), iSTFT back to the
  waveform, and score SI-SDR on the two band-limited signals. A tone inside the
  band scores :math:`\gg 0`; energy outside the band is removed from both and so
  contributes nothing.
* **Band magnitude error** — mean L1 between reference and estimate magnitudes
  within :math:`b`, normalized by the reference band's mean magnitude, NaN-guarded
  for (near-)empty bands (the AAC ~16 kHz top end).

Analysis happens on a **fixed 6-band grid** — the union of both variants' interior
edges plus a 100 Hz floor split — so the figure is layout-neutral (neither the mel
nor the uniform arm is scored on its own favorable grid). See
:func:`analysis_grid_hz`.

Reuses the project SI-SDR (:mod:`singnet.metrics.si_sdr`) so there is one
implementation of the arithmetic, and the same STFT pins as the front end
(:mod:`singnet.audio.stft`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from ..audio.stft import DEFAULT_HOP, DEFAULT_N_FFT
from ..metrics.si_sdr import si_sdr

DEFAULT_SR = 44100
#: Level below which a band's reference magnitude counts as empty (NaN-guarded).
EMPTY_BAND_EPS = 1e-8


def _as_wave(x: "np.ndarray | Tensor") -> Tensor:
    return torch.as_tensor(np.asarray(x), dtype=torch.float32).reshape(-1)


def _bin_frequencies(n_fft: int, sr: int) -> Tensor:
    """Centre frequency (Hz) of each of the ``n_fft // 2 + 1`` rFFT bins."""
    return torch.arange(n_fft // 2 + 1, dtype=torch.float32) * sr / n_fft


def _band_bin_mask(n_fft: int, sr: int, lo_hz: float, hi_hz: float) -> Tensor:
    """Boolean ``(F,)`` mask of bins whose centre frequency is in ``[lo, hi)``."""
    freqs = _bin_frequencies(n_fft, sr)
    return (freqs >= lo_hz) & (freqs < hi_hz)


def _stft(wave: Tensor, n_fft: int, hop: int) -> tuple[Tensor, Tensor]:
    window = torch.hann_window(n_fft, dtype=wave.dtype)
    spec = torch.stft(wave, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                      window=window, center=True, return_complex=True)
    return spec, window


def band_limited_sisdr(
    ref: "np.ndarray | Tensor",
    est: "np.ndarray | Tensor",
    band_hz: tuple[float, float],
    sr: int = DEFAULT_SR,
    n_fft: int = DEFAULT_N_FFT,
    hop: int = DEFAULT_HOP,
) -> float:
    """SI-SDR after zeroing all bins outside ``band_hz`` in **both** signals' STFTs.

    Both signals are band-limited by their *own* STFT (so there is no mask/phase
    asymmetry), inverted to the waveform, and scored with the project SI-SDR.
    Returns NaN if the band contains no STFT bins or the reference is silent
    within it (the empty-band guard).
    """
    lo, hi = float(band_hz[0]), float(band_hz[1])
    ref_w, est_w = _as_wave(ref), _as_wave(est)
    length = ref_w.shape[0]
    mask = _band_bin_mask(n_fft, sr, lo, hi)
    if not bool(mask.any()):
        return float("nan")

    ref_spec, window = _stft(ref_w, n_fft, hop)
    est_spec, _ = _stft(est_w, n_fft, hop)
    band = mask[:, None].to(ref_spec.dtype)
    ref_band = torch.istft(ref_spec * band, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                           window=window, center=True, length=length)
    est_band = torch.istft(est_spec * band, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                           window=window, center=True, length=length)
    if float((ref_band**2).sum()) < EMPTY_BAND_EPS:
        return float("nan")
    return si_sdr(est_band.numpy(), ref_band.numpy())


def band_mag_error(
    ref: "np.ndarray | Tensor",
    est: "np.ndarray | Tensor",
    band_hz: tuple[float, float],
    sr: int = DEFAULT_SR,
    n_fft: int = DEFAULT_N_FFT,
    hop: int = DEFAULT_HOP,
) -> float:
    """Mean L1 magnitude error within ``band_hz``, normalized by the reference level.

    ``mean|R - E|`` over the band's time-frequency bins divided by the reference
    band's mean magnitude (a dimensionless relative error). Returns NaN when the
    band has no bins or the reference band is (near-)empty — the AAC-dead-top-band
    guard.
    """
    lo, hi = float(band_hz[0]), float(band_hz[1])
    ref_w, est_w = _as_wave(ref), _as_wave(est)
    mask = _band_bin_mask(n_fft, sr, lo, hi)
    if not bool(mask.any()):
        return float("nan")
    ref_mag = _stft(ref_w, n_fft, hop)[0].abs()[mask]
    est_mag = _stft(est_w, n_fft, hop)[0].abs()[mask]
    ref_level = float(ref_mag.mean())
    if ref_level < EMPTY_BAND_EPS:
        return float("nan")
    return float((ref_mag - est_mag).abs().mean()) / ref_level


def analysis_grid_hz(sr: int = DEFAULT_SR, n_fft: int = DEFAULT_N_FFT) -> list[float]:
    """The fixed 6-band analysis grid edges in Hz (MASTER_PLAN §6.3).

    Union of both variants' interior band edges (mel: ~1.53, 6.43 kHz; uniform:
    ~7.35, 14.70 kHz) with a 100 Hz floor split and the ``[0, sr/2]`` outer edges —
    7 edges, 6 bands — so the mechanism figure is neutral to either layout.
    """
    from ..models.bandsplit_unet import bin_to_hz, mel_edges, uniform_edges

    mel_interior = [bin_to_hz(b, sr, n_fft) for b in mel_edges(3)[1:-1]]
    uni_interior = [bin_to_hz(b, sr, n_fft) for b in uniform_edges(3)[1:-1]]
    edges = {0.0, 100.0, float(sr) / 2.0, *mel_interior, *uni_interior}
    return sorted(edges)


def banded_report(
    ref: "np.ndarray | Tensor",
    est: "np.ndarray | Tensor",
    grid_hz: list[float] | None = None,
    sr: int = DEFAULT_SR,
    n_fft: int = DEFAULT_N_FFT,
    hop: int = DEFAULT_HOP,
) -> pd.DataFrame:
    """Both per-band metrics over the analysis grid (one system, one track).

    Returns a DataFrame with columns ``band, lo_hz, hi_hz, band_sisdr,
    band_mag_error`` — the per-band rows the notebook aggregates into the
    mechanism figure (variant − baseline deltas per band).
    """
    edges = analysis_grid_hz(sr, n_fft) if grid_hz is None else list(grid_hz)
    rows = []
    for i, (lo, hi) in enumerate(zip(edges, edges[1:])):
        rows.append({
            "band": i,
            "lo_hz": round(lo, 2),
            "hi_hz": round(hi, 2),
            "band_sisdr": band_limited_sisdr(ref, est, (lo, hi), sr, n_fft, hop),
            "band_mag_error": band_mag_error(ref, est, (lo, hi), sr, n_fft, hop),
        })
    return pd.DataFrame(rows)
