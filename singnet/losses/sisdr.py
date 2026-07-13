r"""``sisdr`` — negative time-domain SI-SDR (metric-as-loss).

.. math::

    \mathcal L = -\,\overline{\text{SI-SDR}}(\hat v, v)
    \quad\text{over non-skipped chunks},

with SI-SDR as in :mod:`singnet.metrics.si_sdr` (Le Roux et al., 2019),
:math:`\varepsilon = 10^{-8}`. The estimate waveform :math:`\hat v` is obtained
by a **differentiable iSTFT** of the masked mixture spectrogram, so gradients
flow from the SI-SDR value back to the mask (and thus the model).

**Silent-target guard (MASTER_PLAN §4.4, pre-registered).** The SI-SDR
denominator :math:`\lVert v\rVert^2` is singular on a silent target. Chunks whose
target-vocal RMS is below −60 dBFS (amplitude ``1e-3``) are *excluded from the
loss for this arm only*. Batch composition is unchanged across arms — the skip
is a per-chunk mask inside the loss. The skip **rate** is returned in ``aux`` and
logged per run (it is itself a finding feeding Direction 08).
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..audio.stft import DEFAULT_HOP, DEFAULT_N_FFT, STFT, apply_mask
from ..metrics.si_sdr import DEFAULT_EPS, si_sdr_torch
from ._base import LossOutput, SeparationLoss

#: −60 dBFS as a linear-amplitude RMS threshold (10 ** (-60/20)).
SILENCE_RMS_THRESHOLD = 10.0 ** (-60.0 / 20.0)


class SiSdrLoss(SeparationLoss):
    """Negative mean SI-SDR with the pre-registered silent-target guard."""

    needs_waveform = True

    def __init__(
        self,
        n_fft: int = DEFAULT_N_FFT,
        hop: int = DEFAULT_HOP,
        eps: float = DEFAULT_EPS,
        silence_rms: float = SILENCE_RMS_THRESHOLD,
    ) -> None:
        super().__init__()
        self.stft = STFT(n_fft=n_fft, hop=hop)
        self.eps = float(eps)
        self.silence_rms = float(silence_rms)

    def _compute(self, mask, mix_mag, tgt_mag, mix_stft, tgt_wave, mix_wave) -> LossOutput:  # type: ignore[override]
        if mix_stft is None or tgt_wave is None:
            raise ValueError("SiSdrLoss requires mix_stft and tgt_wave (waveform path).")

        est_wave: Tensor = self.stft.inverse(apply_mask(mask, mix_stft), length=tgt_wave.shape[-1])

        # Silent-target guard: keep only chunks whose target RMS >= threshold.
        rms = tgt_wave.pow(2).mean(dim=-1).clamp_min(0.0).sqrt()  # (B,)
        valid = rms >= self.silence_rms
        n_total = int(valid.numel())
        n_valid = int(valid.sum().item())
        skip_rate = 1.0 - (n_valid / n_total if n_total else 0.0)

        if n_valid > 0:
            # Index BEFORE computing SI-SDR so degenerate silent chunks never
            # enter the arithmetic (0 * inf would poison the mean).
            sisdr = si_sdr_torch(est_wave[valid], tgt_wave[valid], eps=self.eps)
            loss = -sisdr.mean()
        else:
            # Whole batch silent: return a graph-connected zero (no update).
            loss = est_wave.sum() * 0.0

        aux = {
            "skip_rate": skip_rate,
            "n_skipped": float(n_total - n_valid),
            "n_total": float(n_total),
        }
        return loss, aux
