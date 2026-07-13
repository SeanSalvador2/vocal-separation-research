r"""``l1mag`` — L1 on masked magnitude (Spleeter's choice).

.. math::  \mathcal L = \operatorname{mean}\bigl|\,\hat M\odot|X| - |S|\,\bigr|

Provenance: Spleeter (Hennequin et al., 2020). Analytic properties (THEORY §3):
zero iff the masked magnitude equals the target magnitude everywhere; gradient
``sign(Ŝ_mag - |S|) ⊙ |X|`` w.r.t. the mask — robust to magnitude outliers,
indifferent to phase.
"""

from __future__ import annotations

from torch import Tensor

from ._base import LossOutput, SeparationLoss, masked_magnitude


class L1MagLoss(SeparationLoss):
    """Mean absolute error between masked-mixture and target magnitudes."""

    needs_waveform = False

    def _compute(self, mask, mix_mag, tgt_mag, mix_stft, tgt_wave, mix_wave) -> LossOutput:  # type: ignore[override]
        est_mag: Tensor = masked_magnitude(mask, mix_mag)
        loss = (est_mag - tgt_mag).abs().mean()
        return loss, {}
