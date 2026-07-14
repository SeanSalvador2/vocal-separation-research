r"""Chunk-start sampling policies (MASTER_PLAN §4.1) — Direction 08's second deliverable.

Four policies decide *where in a track a training chunk starts*, holding everything
else fixed. With per-track windowed vocal-energy profiles ``E`` (:mod:`singnet.data.profiles`,
one RMS per 1-s grid start over the 6-s window), the chunk-start distribution is:

======================  ==================================================================
``uniform``             uniform over valid starts — the UMX-lore baseline, **bit-identical
                        to the shared project baseline cell** (it draws the sample-resolution
                        ``rng.integers`` exactly as ``MusdbChunks`` always did).
``energy``              :math:`p(s)\propto(1-\lambda)\,E(s)/\sum E + \lambda/N` — energy-
                        proportional with a uniform floor ``λ = 0.1``; silent starts are
                        down-weighted, **never excluded** (mass ``≥ λ/N`` per start).
``drop``                uniform over starts whose window vocal RMS ``≥ θ`` (``θ = −60`` dBFS);
                        silent-window starts excluded entirely (the policy hypothesized to
                        *worsen* leakage, H-08b).
``curriculum``          the ``energy`` form with ``λ`` annealed ``1.0 → 0.1`` linearly over
                        the first 50 % of steps, then held (starts uniform, ends energy).
======================  ==================================================================

The non-uniform policies draw their weighted start from a **dedicated** ``(seed,
"sampling", step)`` RNG stream (:data:`singnet.data.augment.STREAM_IDS` id 5), so
switching policy cannot perturb the augmentation streams (remix/gain/flip) or, for the
``uniform`` arm, the sample-stream draw itself — the shared-cell guarantee (unit-tested,
gate G0). The weighted draw picks a 1-s grid start; the returned value is that grid
point in samples (clamped into the valid range).
"""

from __future__ import annotations

import numpy as np

from .profiles import DEFAULT_GRID_S

POLICIES: tuple[str, ...] = ("uniform", "energy", "drop", "curriculum")
DEFAULT_THETA_DB = -60.0     #: drop threshold, shared with the sisdr silent guard (§2.1)
DEFAULT_FLOOR_LAMBDA = 0.1   #: energy/curriculum uniform-floor mass λ (§4.1)
DEFAULT_SR = 44100


class ChunkSampler:
    """A chunk-start policy (MASTER_PLAN §4.1).

    Args:
        policy: one of :data:`POLICIES`.
        theta_db: ``drop`` window-RMS threshold in dBFS (unused by other policies).
        floor_lambda: ``energy``/``curriculum`` uniform-floor mass λ.
        total_steps: training budget, only consulted by ``curriculum``'s λ(t) schedule
            (``None`` -> ``curriculum`` degenerates to a fixed-λ ``energy`` draw).
        grid_s: profile start-grid spacing in seconds (1-s, matching the prep pass).
        sr: sample rate, mapping a grid index to a sample-domain start.
    """

    def __init__(
        self,
        policy: str = "uniform",
        *,
        theta_db: float = DEFAULT_THETA_DB,
        floor_lambda: float = DEFAULT_FLOOR_LAMBDA,
        total_steps: int | None = None,
        grid_s: float = DEFAULT_GRID_S,
        sr: int = DEFAULT_SR,
    ) -> None:
        if policy not in POLICIES:
            raise ValueError(f"unknown sampling policy {policy!r}; expected one of {POLICIES}")
        self.policy = policy
        self.theta_db = float(theta_db)
        self.floor_lambda = float(floor_lambda)
        self.total_steps = None if total_steps is None else int(total_steps)
        self.grid_s = float(grid_s)
        self.sr = int(sr)

    def is_uniform(self) -> bool:
        """True for the baseline policy (which takes the untouched sample-stream path)."""
        return self.policy == "uniform"

    @property
    def grid_hop_samples(self) -> int:
        return int(round(self.grid_s * self.sr))

    def lambda_at(self, step: int) -> float:
        r"""The floor mass λ in effect at ``step``.

        ``curriculum``: linear ``1.0 → floor_lambda`` over the first half of training,
        held at ``floor_lambda`` after (``λ(0)=1`` = pure uniform, ``λ(T/2..T)=0.1`` =
        energy). ``energy``: constant ``floor_lambda``. Other policies: unused.
        """
        if self.policy == "curriculum" and self.total_steps and self.total_steps > 0:
            half = 0.5 * self.total_steps
            if step >= half:
                return self.floor_lambda
            frac = step / half
            return 1.0 + (self.floor_lambda - 1.0) * frac
        return self.floor_lambda

    def weights(self, profile: np.ndarray, step: int = 0) -> np.ndarray:
        r"""The chunk-start probability vector over the ``N`` grid starts of ``profile``.

        Sums to 1. For ``energy``/``curriculum`` every entry is ``≥ λ/N`` (the floor
        lower bound — silence is down-weighted, never starved). ``drop`` puts uniform
        mass on the supported (``RMS ≥ θ``) starts and ``0`` elsewhere.
        """
        profile = np.asarray(profile, dtype=np.float64).reshape(-1)
        n = profile.size
        if n == 0:
            raise ValueError(
                "empty energy profile — the track is shorter than one chunk window, or the "
                "profile pass has not run (`scripts/prepare_data.py --write-energy-profiles`)."
            )
        uniform = np.full(n, 1.0 / n)
        if self.policy == "uniform":
            return uniform
        if self.policy == "drop":
            thresh = 10.0 ** (self.theta_db / 20.0)
            support = profile >= thresh
            if not support.any():
                return uniform  # all windows sub-θ: fall back to uniform (documented, §12)
            return support.astype(np.float64) / float(support.sum())
        # energy / curriculum: (1−λ)·E/ΣE + λ/N
        lam = self.lambda_at(step)
        total = float(profile.sum())
        energy_term = profile / total if total > 0.0 else uniform
        return (1.0 - lam) * energy_term + lam * uniform

    def start(
        self,
        n_sample_starts: int,
        profile: np.ndarray | None,
        rng: np.random.Generator,
        step: int = 0,
    ) -> int:
        """A chunk start in ``[0, n_sample_starts)`` under this policy, drawn from ``rng``.

        ``uniform`` draws the sample-resolution ``rng.integers`` (bit-identical to the
        legacy ``MusdbChunks`` behaviour); the others draw a 1-s grid start weighted by
        ``profile`` and map it to samples, clamped into range.
        """
        if self.policy == "uniform":
            return int(rng.integers(0, n_sample_starts))
        if profile is None:
            raise ValueError(f"policy {self.policy!r} requires an energy profile")
        weights = self.weights(profile, step)
        grid_index = int(rng.choice(weights.size, p=weights))
        return int(min(grid_index * self.grid_hop_samples, n_sample_starts - 1))


def build_chunk_sampler(config: dict) -> ChunkSampler:
    """Construct the :class:`ChunkSampler` a config asks for (MASTER_PLAN §4.1, §6).

    A config with **no** ``sampling:`` block (every Direction 01–06 config) builds the
    ``uniform`` sampler — the no-op that reproduces the shared baseline cell bit-for-bit
    (regression-tested). A ``sampling:`` block selects the policy and its constants;
    ``total_steps`` (for curriculum) and ``sr`` are read from the config.
    """
    from ..utils.config import sampling_policy

    spec = sampling_policy(config)
    steps = int(config.get("steps", 0)) or None
    return ChunkSampler(
        spec["policy"],
        theta_db=spec["theta_db"],
        floor_lambda=spec["floor_lambda"],
        total_steps=steps,
        sr=int(config.get("sample_rate", DEFAULT_SR)),
    )
