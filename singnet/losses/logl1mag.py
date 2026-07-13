r"""``logl1mag`` — L1 on log-magnitudes (Gusó-family ``LOGL1freq``).

.. math::

    \mathcal L = \operatorname{mean}
        \bigl|\,\log(\hat M\odot|X| + \varepsilon_{\log})
              - \log(|S| + \varepsilon_{\log})\,\bigr|,
    \quad \varepsilon_{\log} = 10^{-5}.

Log compression shrinks the dynamic range so that quiet time-frequency
structure (reverb tails, breath noise, low-energy partials) contributes to the
loss on a par with loud peaks — closer to perceptual loudness than the linear
:mod:`l1mag`. The ``ε_log`` floor keeps ``log`` finite where the masked
magnitude is zero and is the reason the loss is computed in fp32 (§6, §12).
"""

from __future__ import annotations

import torch
from torch import Tensor

from ._base import LossOutput, SeparationLoss, masked_magnitude

EPS_LOG = 1e-5


class LogL1MagLoss(SeparationLoss):
    """Mean absolute error between log-compressed magnitudes."""

    needs_waveform = False

    def __init__(self, eps_log: float = EPS_LOG) -> None:
        super().__init__()
        self.eps_log = float(eps_log)

    def _compute(self, mask, mix_mag, tgt_mag, mix_stft, tgt_wave, mix_wave) -> LossOutput:  # type: ignore[override]
        est_mag: Tensor = masked_magnitude(mask, mix_mag)
        log_est = torch.log(est_mag + self.eps_log)
        log_tgt = torch.log(tgt_mag + self.eps_log)
        loss = (log_est - log_tgt).abs().mean()
        return loss, {}
