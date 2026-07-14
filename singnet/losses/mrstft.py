r"""``l1mrstft`` — L1-magnitude backbone + multi-resolution STFT auxiliary.

.. math::

    \mathcal L = \operatorname{mean}\bigl|\hat M\odot|X| - |S|\bigr|
        \;+\; \lambda \sum_{m=1}^{3}
        \Bigl(\mathcal L_{\text{sc}}^{(m)} + \mathcal L_{\text{mag}}^{(m)}\Bigr),
    \qquad \lambda = 0.5,

with, per resolution :math:`m` on waveforms :math:`(\hat v, v)` (Yamamoto et al.,
2020 / auraloss):

.. math::

    \mathcal L_{\text{sc}} = \frac{\lVert\,|S_m| - |\hat S_m|\,\rVert_F}
                                  {\lVert\,|S_m|\,\rVert_F}, \qquad
    \mathcal L_{\text{mag}} = \operatorname{mean}
        \bigl|\log(|S_m| + \varepsilon_{\log}) - \log(|\hat S_m| + \varepsilon_{\log})\bigr|.

Resolutions (auraloss/PWG defaults, code-verified): ``fft = [1024, 2048, 512]``,
``hop = [120, 240, 50]``, ``win = [600, 1200, 240]`` (Hann).

**Note on the sum vs. mean.** MASTER_PLAN §6 pins the aggregation as a *sum*
over the three resolutions (auraloss averages); the two differ only by a
constant folded into ``λ``. We follow the contract exactly — see THEORY §3.5.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..audio.stft import DEFAULT_HOP, DEFAULT_N_FFT, STFT, apply_mask
from ._base import LossOutput, SeparationLoss, masked_magnitude, per_chunk_mean

EPS_LOG = 1e-5
DEFAULT_FFT_SIZES = (1024, 2048, 512)
DEFAULT_HOP_SIZES = (120, 240, 50)
DEFAULT_WIN_SIZES = (600, 1200, 240)


class _STFTResolution(nn.Module):
    """One STFT analysis resolution; returns waveform magnitude spectrograms."""

    def __init__(self, fft_size: int, hop_size: int, win_size: int) -> None:
        super().__init__()
        self.fft_size = int(fft_size)
        self.hop_size = int(hop_size)
        self.win_size = int(win_size)
        self.register_buffer("window", torch.hann_window(self.win_size), persistent=False)

    def magnitude(self, wave: Tensor) -> Tensor:
        spec = torch.stft(
            wave,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.window.to(wave.device, wave.dtype),
            center=True,
            return_complex=True,
        )
        return spec.abs()


class MultiResolutionSTFTLoss(nn.Module):
    r"""Sum over resolutions of (spectral convergence + log-magnitude L1)."""

    def __init__(
        self,
        fft_sizes=DEFAULT_FFT_SIZES,
        hop_sizes=DEFAULT_HOP_SIZES,
        win_sizes=DEFAULT_WIN_SIZES,
        eps_log: float = EPS_LOG,
    ) -> None:
        super().__init__()
        if not (len(fft_sizes) == len(hop_sizes) == len(win_sizes)):
            raise ValueError("fft_sizes, hop_sizes, win_sizes must have equal length")
        self.resolutions = nn.ModuleList(
            _STFTResolution(f, h, w) for f, h, w in zip(fft_sizes, hop_sizes, win_sizes)
        )
        self.eps_log = float(eps_log)

    @staticmethod
    def _spectral_convergence(
        tgt_mag: Tensor, est_mag: Tensor, eps: float = 1e-8, *, reduce: bool = True
    ) -> Tensor:
        batch = tgt_mag.shape[0]
        diff = (tgt_mag - est_mag).reshape(batch, -1).norm(dim=1)
        ref = tgt_mag.reshape(batch, -1).norm(dim=1) + eps
        ratio = diff / ref  # (B,)
        return ratio.mean() if reduce else ratio

    def _log_magnitude(self, tgt_mag: Tensor, est_mag: Tensor, *, reduce: bool = True) -> Tensor:
        per_element = (
            torch.log(tgt_mag + self.eps_log) - torch.log(est_mag + self.eps_log)
        ).abs()
        return per_element.mean() if reduce else per_element.flatten(1).mean(dim=1)

    def forward(self, est_wave: Tensor, tgt_wave: Tensor, *, reduce: bool = True) -> Tensor:
        total = est_wave.new_zeros(()) if reduce else est_wave.new_zeros(est_wave.shape[0])
        for res in self.resolutions:
            est_mag = res.magnitude(est_wave)
            tgt_mag = res.magnitude(tgt_wave)
            total = (
                total
                + self._spectral_convergence(tgt_mag, est_mag, reduce=reduce)
                + self._log_magnitude(tgt_mag, est_mag, reduce=reduce)
            )
        return total


class L1MrStftLoss(SeparationLoss):
    """L1-magnitude backbone plus the ``λ``-weighted MR-STFT auxiliary."""

    needs_waveform = True

    def __init__(
        self,
        lam: float = 0.5,
        n_fft: int = DEFAULT_N_FFT,
        hop: int = DEFAULT_HOP,
        fft_sizes=DEFAULT_FFT_SIZES,
        hop_sizes=DEFAULT_HOP_SIZES,
        win_sizes=DEFAULT_WIN_SIZES,
    ) -> None:
        super().__init__()
        self.lam = float(lam)
        self.stft = STFT(n_fft=n_fft, hop=hop)
        self.mrstft = MultiResolutionSTFTLoss(fft_sizes, hop_sizes, win_sizes)

    def _compute(self, mask, mix_mag, tgt_mag, mix_stft, tgt_wave, mix_wave, *, reduce=True) -> LossOutput:  # type: ignore[override]
        if mix_stft is None or tgt_wave is None:
            raise ValueError("L1MrStftLoss requires mix_stft and tgt_wave (waveform path).")

        est_mag = masked_magnitude(mask, mix_mag)
        est_wave: Tensor = self.stft.inverse(apply_mask(mask, mix_stft), length=tgt_wave.shape[-1])
        if reduce:
            l1 = (est_mag - tgt_mag).abs().mean()
            mr = self.mrstft(est_wave, tgt_wave)
            loss = l1 + self.lam * mr
            aux = {"l1_mag": float(l1.detach()), "mrstft": float(mr.detach())}
            return loss, aux

        # reduce=False: the same L1 + λ·MR-STFT combination, unreduced over chunks.
        l1_pc = per_chunk_mean((est_mag - tgt_mag).abs())              # (B,)
        mr_pc = self.mrstft(est_wave, tgt_wave, reduce=False)          # (B,)
        per_chunk = l1_pc + self.lam * mr_pc
        aux = {
            "l1_mag": float(l1_pc.mean().detach()),
            "mrstft": float(mr_pc.mean().detach()),
            "per_chunk": per_chunk,
        }
        return per_chunk.mean(), aux
