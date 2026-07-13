"""On-the-fly, seeded, waveform-domain augmentation (MASTER_PLAN §5).

The verified standard MSS recipe, applied *in this fixed order* — remix, then
gain, then sign flip:

1. **Random source remixing** — vocals from track *i* + accompaniment from a
   different track *j*. This is a cross-track operation the dataset performs (it
   needs two tracks); the pipeline owns the *switch* and the *RNG stream* for it,
   and :func:`remix` documents the combine step.
2. **Random per-source gain** — amplitude ``U(0.25, 1.25)`` per source (the exact
   UMX/Demucs code range, *not* dB-symmetric).
3. **Random sign flip** — polarity inversion with ``p = 0.5`` per source
   (Demucs ``FlipSign``). Exactly magnitude-invariant, so it is the design's
   built-in negative control (Direction 02 THEORY §3).

**Per-transform RNG streams (Direction 02 §3.5).** Each transform — and the
always-on chunk sampler — owns an *independent* stream keyed by
``(seed, stream-id, step)`` via :func:`singnet.utils.seed.derive_rng`. Because
the streams are independent, disabling one transform does **not** consume the
draws of, or reshuffle, the others: the leave-one-out arms differ *only* in the
transform under test. This is unit-tested (gate G0). Determinism given
``(seed, step)`` is preserved; nothing has been trained, so bit-exact equality
with any earlier single-stream layout is neither required nor claimed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..utils.seed import derive_rng

Sources = dict[str, np.ndarray]

GAIN_LOW = 0.25
GAIN_HIGH = 1.25
FLIP_PROB = 0.5

#: Stable integer ids per RNG stream (built-in ``hash`` is *not* process-stable,
#: so transform names are mapped to fixed integers instead). ``sample`` is the
#: always-on chunk sampler (which track / which window); it is a sampling policy,
#: not a factorized transform (MASTER_PLAN §3.1), and keeps its own stream so the
#: three switchable transforms are cleanly independent of it.
STREAM_IDS: dict[str, int] = {"sample": 0, "remix": 1, "gain": 2, "flip": 3}


def remix(vocals: np.ndarray, accompaniment: np.ndarray) -> Sources:
    """Assemble a remixed source pair: vocals from one track, accomp from another.

    The mixture is later formed as ``vocals + accompaniment``; because MUSDB
    stems are linearly additive, this cross-track mixture is exact (no
    estimation) — the single biggest small-data lever (Direction 02 H-02a).
    """
    return {"vocals": np.asarray(vocals), "accompaniment": np.asarray(accompaniment)}


def random_gain(
    sources: Sources, rng: np.random.Generator, low: float = GAIN_LOW, high: float = GAIN_HIGH
) -> Sources:
    """Scale each source by an independent amplitude ``U(low, high)``."""
    return {name: sig * float(rng.uniform(low, high)) for name, sig in sources.items()}


def random_sign_flip(sources: Sources, rng: np.random.Generator, p: float = FLIP_PROB) -> Sources:
    """Invert the polarity of each source independently with probability ``p``.

    Exactly magnitude-invariant: ``|STFT(-s)| = |STFT(s)|`` (Direction 02
    THEORY §3), so for a magnitude-masking model this transform changes *nothing*
    the network sees — the pre-registered negative control.
    """
    out: Sources = {}
    for name, sig in sources.items():
        out[name] = -sig if rng.random() < p else np.asarray(sig)
    return out


@dataclass
class AugmentPipeline:
    """The augmentation switchboard (MASTER_PLAN §5; Direction 02 §3.1, §9).

    ``AugmentPipeline(remix, gain, flip, seed)``. The three booleans toggle the
    factorized transforms; ``AugmentPipeline(True, True, True)`` is exactly the
    Direction-01 full recipe (remix -> gain -> flip, constants below). Each
    transform draws from an independent ``(seed, stream-id, step)`` stream
    (:data:`STREAM_IDS`), so a leave-one-out ablation perturbs only the transform
    under test.

    The pipeline owns the remix *switch* and its RNG stream but not the
    two-track combine itself: :class:`singnet.data.MusdbChunks` calls
    :meth:`stream` to draw the remix partner (it is the object holding the
    tracks). Gain and flip are applied here by :meth:`__call__`.
    """

    remix: bool = True
    gain: bool = True
    flip: bool = True
    seed: int = 0
    gain_low: float = GAIN_LOW
    gain_high: float = GAIN_HIGH
    flip_prob: float = FLIP_PROB

    def stream(self, name: str, step: int) -> np.random.Generator:
        """Return the independent RNG generator for ``name`` at ``step``.

        ``name`` is one of :data:`STREAM_IDS` (``sample``/``remix``/``gain``/
        ``flip``). The generator is a pure function of ``(seed, stream-id, step)``
        and nothing else — crucially not the state of the other transforms.
        """
        if name not in STREAM_IDS:
            raise KeyError(f"unknown augmentation stream {name!r}; expected {tuple(STREAM_IDS)}")
        return derive_rng(self.seed, STREAM_IDS[name], int(step))

    def with_seed(self, seed: int) -> "AugmentPipeline":
        """Return a copy bound to ``seed`` (used to sync the pipeline to a run)."""
        return replace(self, seed=int(seed))

    def apply_gain(self, sources: Sources, step: int) -> Sources:
        """Per-source gain on the ``gain`` stream (identity if ``gain`` is off)."""
        if not self.gain:
            return dict(sources)
        return random_gain(sources, self.stream("gain", step), self.gain_low, self.gain_high)

    def apply_flip(self, sources: Sources, step: int) -> Sources:
        """Per-source sign flip on the ``flip`` stream (identity if ``flip`` off)."""
        if not self.flip:
            return dict(sources)
        return random_sign_flip(sources, self.stream("flip", step), self.flip_prob)

    def __call__(self, sources: Sources, step: int) -> Sources:
        """Apply the enabled per-source transforms (gain then flip) at ``step``.

        Remix is handled upstream by the dataset (it needs two tracks); this
        applies steps 2-3 of the recipe in order, each on its own stream.
        """
        return self.apply_flip(self.apply_gain(sources, step), step)
