"""Shared loss contract and the fp32 / autocast-disabled compute region.

Every loss implements the *one uniform signature* (MASTER_PLAN §10):

    forward(mask, mix_mag, tgt_mag, mix_stft, tgt_wave, mix_wave, *, reduce=True)
        -> (loss, aux)

* ``mask``     : predicted soft mask, ``(B, 2048, 256)`` in ``[0, 1]``.
* ``mix_mag``  : mixture magnitude on the 2048 network bins, ``(B, 2048, 256)``.
* ``tgt_mag``  : target-vocal magnitude on the same bins.
* ``mix_stft`` : complex mixture spectrogram ``(B, 2049, 256)`` (all bins,
                 carries the mixture phase) — used by waveform losses to
                 reconstruct the estimate.
* ``tgt_wave`` : target-vocal waveform ``(B, L')`` (STFT-consistent length).
* ``mix_wave`` : mixture waveform ``(B, L')``.

Magnitude-only losses ignore the waveform arguments; waveform losses ignore
none. ``aux`` is a dict of logging values (e.g. ``skip_rate``).

**Per-chunk contract extension (Direction 06 §3.2, backward-compatible).** The
keyword-only ``reduce`` flag defaults to ``True`` — the historical behaviour, in
which ``forward`` returns the batch-reduced scalar and the *same* ``aux`` it
always did (so every Direction 01–05 loss test passes unmodified). With
``reduce=False`` the loss additionally exposes an **unreduced, per-chunk**
``(B,)`` tensor of losses in ``aux["per_chunk"]`` — differentiable and computed
in the same fp32 region as the scalar — which :class:`singnet.losses.trimmed.
TrimmedLoss` ranks to keep the lowest-loss chunks. The scalar returned under
``reduce=False`` is ``per_chunk.mean()``; the default ``reduce=True`` path is
byte-for-byte the pre-extension computation.

All losses run their arithmetic in **float32 inside an autocast-disabled
region** (MASTER_PLAN §6, §12): ``log`` and division near zero are unstable in
fp16, so we pin fp32 regardless of the surrounding AMP context. The per-chunk
vector is fp32 for the same reason (AMP interaction, MASTER_PLAN §11).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch import Tensor, nn

# Type alias for the uniform loss return. ``aux`` carries logging scalars and,
# under ``reduce=False``, the differentiable ``per_chunk`` (B,) tensor.
LossOutput = tuple[Tensor, dict[str, Any]]


@contextmanager
def fp32_autocast_disabled() -> Iterator[None]:
    """Context manager that disables autocast on all device types.

    Defensive across torch builds/devices: entering ``torch.autocast`` with a
    device type that is unavailable can raise, so each is guarded. Inside this
    region callers additionally cast their tensors to float32.
    """
    managers = []
    for device_type in ("cuda", "cpu"):
        try:
            mgr = torch.autocast(device_type=device_type, enabled=False)
            mgr.__enter__()
            managers.append(mgr)
        except (RuntimeError, ValueError):
            pass
    try:
        yield
    finally:
        for mgr in reversed(managers):
            mgr.__exit__(None, None, None)


class SeparationLoss(nn.Module):
    """Base class fixing the uniform forward signature.

    Subclasses implement :meth:`_compute`, which runs already inside the fp32 /
    autocast-disabled region with float32 inputs.
    """

    #: whether this loss needs the differentiable-iSTFT waveform path.
    needs_waveform: bool = False

    def forward(
        self,
        mask: Tensor,
        mix_mag: Tensor,
        tgt_mag: Tensor,
        mix_stft: Tensor | None = None,
        tgt_wave: Tensor | None = None,
        mix_wave: Tensor | None = None,
        *,
        reduce: bool = True,
    ) -> LossOutput:
        with fp32_autocast_disabled():
            return self._compute(
                mask.float(),
                mix_mag.float(),
                tgt_mag.float(),
                None if mix_stft is None else mix_stft.to(torch.complex64),
                None if tgt_wave is None else tgt_wave.float(),
                None if mix_wave is None else mix_wave.float(),
                reduce=reduce,
            )

    def _compute(
        self,
        mask: Tensor,
        mix_mag: Tensor,
        tgt_mag: Tensor,
        mix_stft: Tensor | None,
        tgt_wave: Tensor | None,
        mix_wave: Tensor | None,
        *,
        reduce: bool = True,
    ) -> LossOutput:
        raise NotImplementedError


def masked_magnitude(mask: Tensor, mix_mag: Tensor) -> Tensor:
    """Estimated source magnitude ``Ŝ_mag = M ⊙ |X|`` on the network bins."""
    return mask * mix_mag


def per_chunk_mean(per_element: Tensor) -> Tensor:
    """Mean a per-element error tensor over everything but the batch axis.

    ``per_element`` is ``(B, …)``; the result is the ``(B,)`` per-chunk loss. Used
    only on the ``reduce=False`` path, so the ``reduce=True`` scalar computation is
    left byte-identical to each loss's historical expression.
    """
    return per_element.flatten(1).mean(dim=1)
