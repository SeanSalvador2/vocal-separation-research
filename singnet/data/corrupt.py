r"""``StemBleed`` — the controlled ε-bleed target corruption (Direction 06 §3.1).

Real training stems are dirty: microphone bleed leaves accompaniment content in
the "vocals" stem (SDX'23's ``SDXDB23_Bleeding``, ``research/papers/
fabbro2023-sdx23.md``). Direction 06 simulates a *single-parameter* version by
redistributing a fraction ε of a track's accompaniment into its vocal **target**,
applied at the **stem level, at load time, before augmentation**, to
**training-split tracks only**:

.. math::

    \tilde v = v + \varepsilon\,a, \qquad \tilde a = (1-\varepsilon)\,a,

with :math:`a` the same track's accompaniment. Two properties, both unit-tested:

* **Mixture invariance** — :math:`\tilde v + \tilde a = v + a` *exactly*: the
  corrupted stems still sum to the true mixture (the defining property of real
  bleed, and of ``SDXDB23_Bleeding``'s redistribution). The network input
  :math:`x = v + a` is therefore untouched; only the *target* is corrupted, so
  the model learns :math:`x \mapsto \tilde v`, i.e. to leave ε·a in its estimate.
* **Determinism** — no RNG: :class:`StemBleed` is a pure affine map on the stem
  pair, so wiring it into the dataset cannot perturb any augmentation RNG stream
  (MASTER_PLAN §3.4; the kept-stream-equality test).

**Loudness (pre-registered, §3.1).** The corrupted target's energy grows with ε
— real bleed adds energy — and we do **not** renormalize: the per-chunk input
standardization is mixture-side (unaffected) and the random-gain augmentation
already randomizes source levels, so renormalizing would silently convert "bleed"
into "bleed + attenuation."

**The eval-split guard (structural, §3.1 / §11 "the fatal bug class").**
Corruption must never touch validation/test targets. :func:`build_corruption`
and :class:`singnet.data.MusdbChunks` **both** refuse a non-``train`` split by
construction (raising :class:`EvalSplitCorruptionError`), so a corrupting dataset
on an eval split cannot be built at all — not a runtime warning, an object that
does not exist.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

Sources = Mapping[str, np.ndarray]

#: Splits on which target corruption is permitted (training targets only).
CORRUPTIBLE_SPLITS: tuple[str, ...] = ("train",)


class EvalSplitCorruptionError(ValueError):
    """Raised when stem-bleed corruption is requested on a non-``train`` split.

    The eval-split guard (MASTER_PLAN §3.1): validation and test targets are
    *never* corrupted. This is raised at construction time — the corrupting
    object cannot exist for ``valid``/``test`` — so the guard is structural, not
    advisory.
    """


class StemBleed:
    """Deterministic ε-bleed of accompaniment into the vocal target (§3.1).

    Args:
        epsilon: bleed fraction ε ∈ [0, 1]. ε = 0 is the identity (the clean
            cell); Direction 06 uses ε ∈ {0.05, 0.15, 0.30}.

    Calling the transform maps a ``{"vocals", "accompaniment"}`` stem dict to the
    corrupted pair ``{ṽ, ã}``; any other keys pass through unchanged.
    """

    def __init__(self, epsilon: float) -> None:
        epsilon = float(epsilon)
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError(f"epsilon must be in [0, 1]; got {epsilon!r}")
        self.epsilon = epsilon

    def __call__(self, sources: Sources) -> dict[str, np.ndarray]:
        if "vocals" not in sources or "accompaniment" not in sources:
            raise KeyError(
                "StemBleed needs 'vocals' and 'accompaniment' stems; got "
                f"{sorted(sources)}"
            )
        eps = self.epsilon
        vocals = np.asarray(sources["vocals"], dtype=np.float32)
        accompaniment = np.asarray(sources["accompaniment"], dtype=np.float32)
        out: dict[str, np.ndarray] = dict(sources)
        # ṽ = v + ε·a, ã = (1 − ε)·a — computed in float64 then cast, so the
        # mixture-invariance residual is at float32 round-off, not accumulated.
        out["vocals"] = (vocals.astype(np.float64) + eps * accompaniment.astype(np.float64)).astype(
            np.float32
        )
        out["accompaniment"] = ((1.0 - eps) * accompaniment.astype(np.float64)).astype(np.float32)
        return out

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"StemBleed(epsilon={self.epsilon!r})"


def corruption_epsilon(config_block: Any) -> float:
    """Read ε from an optional ``corrupt:`` config block (``0.0`` if absent).

    Accepts the canonical ``{"epsilon": <float>}`` dict; a missing block, an empty
    dict, or ``None`` all mean "no corruption" (ε = 0). Mirrors
    :func:`singnet.utils.config.corruption_epsilon` so callers can pass either the
    whole config or just its ``corrupt`` sub-block.
    """
    if isinstance(config_block, Mapping):
        if "corrupt" in config_block:  # a full config was passed
            return corruption_epsilon(config_block.get("corrupt"))
        return float(config_block.get("epsilon", 0.0) or 0.0)
    return 0.0


def build_corruption(config_block: Any, split: str) -> StemBleed | None:
    """Factory: build the :class:`StemBleed` for a ``corrupt:`` block and split.

    Returns ``None`` when no corruption is requested (block absent or ε = 0) — so
    the clean cell is byte-identical to an uncorrupted run. When ε > 0 the split
    **must** be ``train``: a non-train split raises :class:`EvalSplitCorruptionError`
    here, before any dataset is built (the structural eval-split guard, §3.1).
    """
    eps = corruption_epsilon(config_block)
    if eps == 0.0:
        return None
    if split not in CORRUPTIBLE_SPLITS:
        raise EvalSplitCorruptionError(
            f"stem-bleed corruption (ε={eps}) refuses split {split!r}; corruption is "
            f"permitted on {CORRUPTIBLE_SPLITS} targets only (MASTER_PLAN §3.1). "
            "Validation/test stems are never corrupted."
        )
    return StemBleed(eps)
