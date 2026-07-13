r"""``msemag`` — MSE on masked magnitude (Open-Unmix's choice).

.. math::  \mathcal L = \operatorname{mean}\bigl(\hat M\odot|X| - |S|\bigr)^2

Provenance: Open-Unmix (Stöter et al., 2019). Relative to :mod:`l1mag`, the
quadratic penalty weights large-magnitude (loud) time-frequency errors more
heavily — it chases spectral peaks and tolerates small errors in quiet bins.
"""

from __future__ import annotations

from torch import Tensor

from ._base import LossOutput, SeparationLoss, masked_magnitude


class MseMagLoss(SeparationLoss):
    """Mean squared error between masked-mixture and target magnitudes."""

    needs_waveform = False

    def _compute(self, mask, mix_mag, tgt_mag, mix_stft, tgt_wave, mix_wave) -> LossOutput:  # type: ignore[override]
        est_mag: Tensor = masked_magnitude(mask, mix_mag)
        loss = (est_mag - tgt_mag).pow(2).mean()
        return loss, {}
