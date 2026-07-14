r"""``l1mag`` — L1 on masked magnitude (Spleeter's choice).

.. math::  \mathcal L = \operatorname{mean}\bigl|\,\hat M\odot|X| - |S|\,\bigr|

Provenance: Spleeter (Hennequin et al., 2020). Analytic properties (THEORY §3):
zero iff the masked magnitude equals the target magnitude everywhere; gradient
``sign(Ŝ_mag - |S|) ⊙ |X|`` w.r.t. the mask — robust to magnitude outliers,
indifferent to phase.
"""

from __future__ import annotations

from torch import Tensor

from ._base import LossOutput, SeparationLoss, masked_magnitude, per_chunk_mean


class L1MagLoss(SeparationLoss):
    """Mean absolute error between masked-mixture and target magnitudes."""

    needs_waveform = False

    def _compute(self, mask, mix_mag, tgt_mag, mix_stft, tgt_wave, mix_wave, *, reduce=True) -> LossOutput:  # type: ignore[override]
        est_mag: Tensor = masked_magnitude(mask, mix_mag)
        err = (est_mag - tgt_mag).abs()
        if reduce:
            return err.mean(), {}
        per_chunk = per_chunk_mean(err)
        return per_chunk.mean(), {"per_chunk": per_chunk}
