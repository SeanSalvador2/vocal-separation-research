r"""``trimmed`` — Iterative Trimmed Loss Minimization (ITLM) as a loss wrapper.

Direction 06 §3.2. Per optimizer step, compute the per-chunk losses
:math:`\ell_i` of a wrapped base loss over the batch of :math:`B` chunks, keep
the :math:`\lceil(1-q)B\rceil` **smallest**, and average **only those** — so
gradients flow only through the kept chunks (Shen & Sanghavi, arXiv 1810.11874;
``research/papers/noisy-label-canon.md`` §B):

.. math::

    S = \{\text{indices of the }\lceil(1-q)B\rceil\text{ smallest }\ell_i\},
    \qquad \mathcal L_{\text{trim}} = \frac{1}{|S|}\sum_{i\in S}\ell_i .

The wrapper is **loss-agnostic**: it consumes any project loss's per-chunk path
(``base(..., reduce=False)`` → ``aux["per_chunk"]``, a differentiable ``(B,)``
fp32 tensor — MASTER_PLAN §3.2, "a small, backward-compatible extension of the
loss contract"). Selection uses the **detached** per-chunk values (ranking must
not depend on the gradient) but the averaged loss indexes the *differentiable*
per-chunk tensor, so ``autograd`` never sees the dropped chunks.

**Why this is a stress-test, not the classic setting.** The small-loss trick
assumes some samples are clean. Under Direction 06's *uniform* ε-bleed every
target is corrupted; trimming instead selects the chunks whose accompaniment
happens to be quiet — the *effectively cleanest* targets (THEORY §5,
curriculum-by-cleanliness). To make that mechanism testable the wrapper logs, per
step, the kept/dropped indices and — when the training step threads it in — the
per-chunk **accompaniment energy** of kept vs dropped chunks into ``aux`` (the
every-500-steps telemetry CSV, MASTER_PLAN §5).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ._base import LossOutput, SeparationLoss, fp32_autocast_disabled


class TrimmedLoss(SeparationLoss):
    """Keep-the-lowest-``(1-q)`` trimmed average of a base loss's per-chunk values.

    Args:
        base: any :class:`SeparationLoss` (its ``reduce=False`` path supplies the
            per-chunk vector — every project loss implements it).
        q: trim fraction in ``[0, 1)``. ``q = 0`` keeps every chunk (≡ the base
            loss). Direction 06 uses ``q ∈ {0.10, 0.30}``; ``q = 0.30`` on a batch
            of 16 keeps ``⌈0.7·16⌉ = 12`` chunks.

    The wrapper inherits :attr:`needs_waveform` from ``base`` so the train loop
    still supplies the waveform tensors for e.g. an ``sisdr`` base.
    """

    def __init__(self, base: SeparationLoss, q: float) -> None:
        super().__init__()
        q = float(q)
        if not (0.0 <= q < 1.0):
            raise ValueError(f"trim fraction q must be in [0, 1); got {q!r}")
        self.base = base
        self.q = q
        self.needs_waveform = bool(getattr(base, "needs_waveform", False))

    @staticmethod
    def keep_count(batch: int, q: float) -> int:
        r"""``⌈(1-q)·B⌉`` clamped to ``[1, B]`` — the number of chunks kept."""
        keep = math.ceil((1.0 - float(q)) * int(batch))
        return max(1, min(int(batch), keep))

    def forward(  # type: ignore[override]
        self,
        mask: Tensor,
        mix_mag: Tensor,
        tgt_mag: Tensor,
        mix_stft: Tensor | None = None,
        tgt_wave: Tensor | None = None,
        mix_wave: Tensor | None = None,
        *,
        reduce: bool = True,
        chunk_energy: Tensor | None = None,
    ) -> LossOutput:
        # The base's own forward pins fp32; we additionally pin it around the
        # ranking + trimmed mean so AMP cannot reorder the selection (§11).
        with fp32_autocast_disabled():
            _, base_aux = self.base(
                mask, mix_mag, tgt_mag, mix_stft, tgt_wave, mix_wave, reduce=False
            )
            per_chunk: Tensor = base_aux["per_chunk"].float()  # (B,), differentiable
            batch = int(per_chunk.shape[0])
            keep = self.keep_count(batch, self.q)

            # Rank by DETACHED loss (ascending); keep the lowest, drop the rest.
            order = torch.argsort(per_chunk.detach(), descending=False, stable=True)
            kept_idx = order[:keep]
            dropped_idx = order[keep:]

            # Average only the kept chunks — gradients never reach the dropped ones.
            loss = per_chunk[kept_idx].mean()

            aux: dict[str, object] = {
                "trim_q": self.q,
                "batch_size": float(batch),
                "n_kept": float(keep),
                "n_dropped": float(batch - keep),
                "kept_fraction": keep / batch,
                "kept_idx": kept_idx.detach().cpu().tolist(),
                "dropped_idx": dropped_idx.detach().cpu().tolist(),
                "per_chunk_loss": per_chunk.detach(),
            }
            # Carry through any base telemetry (e.g. sisdr skip_rate), namespaced.
            for key, value in base_aux.items():
                if key != "per_chunk":
                    aux[f"base_{key}"] = value

            if chunk_energy is not None:
                # §5 mechanism telemetry: accompaniment energy of kept vs dropped.
                energy = chunk_energy.detach().float().reshape(-1)
                kept_e = energy[kept_idx]
                dropped_e = energy[dropped_idx]
                aux.update(
                    {
                        "kept_energy_mean": float(kept_e.mean()) if kept_e.numel() else float("nan"),
                        "dropped_energy_mean": float(dropped_e.mean())
                        if dropped_e.numel()
                        else float("nan"),
                        "kept_energy_median": float(kept_e.median()) if kept_e.numel() else float("nan"),
                        "dropped_energy_median": float(dropped_e.median())
                        if dropped_e.numel()
                        else float("nan"),
                        "chunk_energy": energy,
                    }
                )

            if not reduce:
                # Contract-completeness: expose the (untrimmed) per-chunk vector too.
                aux["per_chunk"] = per_chunk
            return loss, aux
