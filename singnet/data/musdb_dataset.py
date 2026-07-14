"""``MusdbChunks`` — the training/tuning dataset (MASTER_PLAN §4.4, §10 contract).

Deterministic per ``(seed, sample_index)`` — *never* a function of the loss arm
or of which augmentations are enabled beyond the transform under test, so a
controlled comparison changes exactly one thing. Each item is drawn as:

1. pick a vocals track and a uniform-random 6 s window (the ``sample`` stream),
2. **remix** (if on): draw a *separate* accompaniment track and window (the
   ``remix`` stream). **Without remix**, the accompaniment is the *same track's*
   accompaniment at the *same window* — the true track mixture (§3.1),
3. apply per-source **gain** then **sign flip** (:mod:`singnet.data.augment`),
   each on its own independent stream,
4. form the mixture as ``vocals + accompaniment`` (additivity exact by design).

Because vocals come from the always-on ``sample`` stream and gain/flip from their
own streams, toggling **remix** changes *only* the accompaniment source; toggling
**flip** changes *only* the polarities; toggling **gain** changes *only* the
scales. This per-transform independence is the leave-one-out invariant (§3.5,
gate G0) and is unit-tested.

The dataset reads decoded WAV shards via a :class:`TrackStore`; for tests it is
constructed from an :class:`InMemoryStore` of synthetic arrays (no MUSDB, no
network). The map-style ``__getitem__(i)`` is seeded by ``(seed, i)`` so global
step ``s`` (pulling indices ``[s*B : (s+1)*B]``) is reproducible and arm-agnostic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .augment import AugmentPipeline
from .corrupt import CORRUPTIBLE_SPLITS, EvalSplitCorruptionError, StemBleed
from .manifest import Manifest
from .sampling import ChunkSampler

DEFAULT_SR = 44100
DEFAULT_CHUNK_S = 6.0
SOURCE_NAMES = ("vocals", "accompaniment")


def load_track_allowlist(path: str | Path | None) -> list[str] | None:
    """Read a subset allowlist CSV (a ``track`` column) into a list of names.

    ``None`` (or an empty path) means "no restriction — train on the full split"
    (MASTER_PLAN §3.2). The CSV is the names-only artifact written by
    ``scripts/make_subsets.py``; ``#`` provenance-header lines are skipped.
    Returns the track names in file order (the nested-draw order fixed at G0b).
    """
    if path is None:
        return None
    frame = pd.read_csv(path, dtype=str, comment="#")
    if "track" not in frame.columns:
        raise ValueError(f"allowlist {path} missing a 'track' column (got {list(frame.columns)})")
    return [t.strip() for t in frame["track"].tolist()]


@runtime_checkable
class TrackStore(Protocol):
    """Provides mono float32 sources per track (``vocals``/``accompaniment``)."""

    sample_rate: int

    def track_names(self) -> list[str]:
        ...

    def load_sources(self, track: str) -> dict[str, np.ndarray]:
        ...


class InMemoryStore:
    """Synthetic in-memory store for tests and notebook demos (no I/O)."""

    def __init__(self, tracks: dict[str, dict[str, np.ndarray]], sample_rate: int = DEFAULT_SR) -> None:
        self.sample_rate = int(sample_rate)
        self._tracks = {
            name: {src: _to_mono_f32(sig) for src, sig in stems.items()}
            for name, stems in tracks.items()
        }

    def track_names(self) -> list[str]:
        return list(self._tracks)

    def load_sources(self, track: str) -> dict[str, np.ndarray]:
        return self._tracks[track]


class WavShardStore:
    """Real store reading decoded WAV shards produced by ``scripts/prepare_data.py``.

    Layout: ``<shard_root>/<track>/{vocals,accompaniment}.wav`` (falling back to
    summing ``drums+bass+other`` if a precomputed accompaniment shard is absent).
    Uses ``soundfile``; RUN LATER — requires the decoded dataset on disk.
    """

    def __init__(self, shard_root: str | Path, sample_rate: int = DEFAULT_SR) -> None:
        self.shard_root = Path(shard_root)
        self.sample_rate = int(sample_rate)

    def track_names(self) -> list[str]:
        return sorted(p.name for p in self.shard_root.iterdir() if p.is_dir())

    def load_sources(self, track: str) -> dict[str, np.ndarray]:
        import soundfile as sf  # local import: only needed with real data

        track_dir = self.shard_root / track
        vocals, sr = sf.read(track_dir / "vocals.wav", dtype="float32", always_2d=True)
        if sr != self.sample_rate:
            raise ValueError(f"{track}: sample rate {sr} != expected {self.sample_rate}")
        acc_path = track_dir / "accompaniment.wav"
        if acc_path.exists():
            accompaniment, _ = sf.read(acc_path, dtype="float32", always_2d=True)
        else:
            accompaniment = sum(
                sf.read(track_dir / f"{stem}.wav", dtype="float32", always_2d=True)[0]
                for stem in ("drums", "bass", "other")
            )
        return {"vocals": _to_mono_f32(vocals), "accompaniment": _to_mono_f32(accompaniment)}


class MusdbChunks(Dataset):
    """Chunk sampler with the augmentation switchboard; deterministic per ``(seed, index)``.

    Args:
        store: the :class:`TrackStore` providing sources.
        manifest: the split manifest; ``split`` selects trainable rows.
        split: ``"train"`` or ``"valid"`` (``"test"`` is refused by the manifest).
        seed: the run seed (also bound onto the augmentation pipeline's streams).
        chunk_s: chunk length in seconds (6.0 pinned).
        augment: when no explicit ``pipeline`` is given, this master switch builds
            ``AugmentPipeline(remix=remix, gain=augment, flip=augment)`` — the
            Direction-01 semantics where ``augment`` toggled gain + flip together.
        remix: cross-track remixing (only used when ``pipeline`` is ``None``).
        track_allowlist: subset of track names to train on (``None`` = full split;
            MASTER_PLAN §3.2). Restricts both the sampling pool and the remix pool.
        length: number of samples the map exposes (``steps * batch``). Defaults
            to a small multiple of the track count for demos.
        pipeline: an explicit :class:`AugmentPipeline` switchboard; when given it
            is the authority on remix/gain/flip (the ``augment``/``remix`` bools
            are ignored) and it is re-bound to this dataset's ``seed``.
        corrupt: optional :class:`singnet.data.StemBleed` applied at load time,
            **before** augmentation, to every loaded track's stems (Direction 06
            §3.1). Refused on non-``train`` splits (the structural eval-split
            guard). ``None`` (the default) is the byte-identical uncorrupted path.
        sampler: optional :class:`singnet.data.ChunkSampler` chunk-start policy
            (Direction 08 §4.1). ``None`` / a ``uniform`` sampler both take the
            byte-identical legacy path (sample-resolution ``rng.integers`` on the
            ``sample``/``remix`` streams). A non-uniform policy draws the weighted
            start from the dedicated ``sampling`` stream and **requires**
            ``energy_profiles``.
        energy_profiles: per-track windowed vocal-RMS profiles (:mod:`singnet.data.profiles`),
            consumed only by a non-uniform ``sampler``; a missing track's profile is a
            loud error pointing at the ``--write-energy-profiles`` prep pass.
        batch_size: batch size, used only to map an item ``index`` to a training
            ``step`` (``index // batch_size``) for ``curriculum``'s λ(t) schedule.
    """

    def __init__(
        self,
        store: TrackStore,
        manifest: Manifest,
        split: str = "train",
        seed: int = 0,
        chunk_s: float = DEFAULT_CHUNK_S,
        augment: bool = True,
        remix: bool = True,
        track_allowlist: list[str] | None = None,
        length: int | None = None,
        pipeline: AugmentPipeline | None = None,
        corrupt: StemBleed | None = None,
        sampler: ChunkSampler | None = None,
        energy_profiles: dict[str, np.ndarray] | None = None,
        batch_size: int = 16,
    ) -> None:
        self.store = store
        self.seed = int(seed)
        self.sample_rate = int(store.sample_rate)
        self.chunk_len = int(round(chunk_s * self.sample_rate))
        # Direction 08 §4.1: the chunk-start policy. Default `uniform` is the shared
        # baseline cell — its draw path below is byte-identical to the pre-D08 loader.
        self.sampler = sampler if sampler is not None else ChunkSampler("uniform")
        self.energy_profiles = energy_profiles
        self.batch_size = max(1, int(batch_size))

        # Structural eval-split guard (§3.1): a corrupting dataset simply cannot be
        # constructed on a non-train split — raised here, before any row is read.
        if corrupt is not None and split not in CORRUPTIBLE_SPLITS:
            raise EvalSplitCorruptionError(
                f"stem-bleed corruption refuses split {split!r}; training targets "
                f"only ({CORRUPTIBLE_SPLITS}). Validation/test stems are never corrupted."
            )
        self.corrupt = corrupt

        if pipeline is None:
            pipeline = AugmentPipeline(remix=bool(remix), gain=bool(augment), flip=bool(augment))
        self.pipeline = pipeline.with_seed(self.seed)
        self.remix = self.pipeline.remix

        # Manifest guard: only trainable rows, none flagged test, intersected with
        # the store and (optionally) the subset allowlist. Manifest order is the
        # canonical order so the remix pool does not depend on allowlist ordering.
        available = set(store.track_names())
        tracks = [t for t in manifest.trainable_tracks(split) if t in available]
        if track_allowlist is not None:
            allow = set(track_allowlist)
            tracks = [t for t in tracks if t in allow]
        manifest.assert_no_test_rows(tracks)
        if not tracks:
            raise ValueError(f"no {split} tracks available in the store")
        self.tracks = tracks

        self._length = int(length) if length is not None else max(64, 8 * len(self.tracks))

    def __len__(self) -> int:
        return self._length

    @property
    def n_songs(self) -> int:
        """Number of tracks actually sampled (the subset size for the scaling curve)."""
        return len(self.tracks)

    def _prepare(self, samples: np.ndarray) -> np.ndarray:
        """Loop-pad a track to at least one chunk length (fixed-size sampling)."""
        n = len(samples)
        if n < self.chunk_len:
            reps = int(np.ceil(self.chunk_len / max(n, 1)))
            samples = np.tile(samples, reps)
        return samples

    def _start(self, padded: np.ndarray, rng: np.random.Generator) -> int:
        return int(rng.integers(0, len(padded) - self.chunk_len + 1))

    def _profile_for(self, track: str) -> np.ndarray:
        """The energy profile for ``track`` (a loud error if the prep pass is missing)."""
        if self.energy_profiles is None or track not in self.energy_profiles:
            raise ValueError(
                f"sampling policy {self.sampler.policy!r} needs an energy profile for "
                f"{track!r}, but none was loaded — run `scripts/prepare_data.py "
                f"--write-energy-profiles --out <shard_root>` first (MASTER_PLAN §5)."
            )
        return self.energy_profiles[track]

    def _policy_start(
        self, padded: np.ndarray, track: str, rng: np.random.Generator, step: int
    ) -> int:
        """A non-uniform policy chunk start (weighted grid draw on the ``sampling`` stream)."""
        n_starts = len(padded) - self.chunk_len + 1
        return self.sampler.start(n_starts, self._profile_for(track), rng, step=step)

    def _slice(self, padded: np.ndarray, start: int) -> np.ndarray:
        return padded[start : start + self.chunk_len].astype(np.float32)

    def _sources(self, track: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Return ``(raw, model)`` stems for ``track``.

        ``raw`` is the clean stem dict from the store; ``model`` is its ε-bleed
        corrupted view (:attr:`corrupt`) when configured, else ``raw`` itself. The
        corruption is deterministic (no RNG), so returning both is free and the
        augmentation streams are untouched. ``raw`` supplies the §5 accompaniment-
        energy telemetry (the *uncorrupted* ⟨a⟩-energy); ``model`` supplies the
        stems the network actually trains on.
        """
        raw = self.store.load_sources(track)
        if self.corrupt is None:
            return raw, raw
        return raw, self.corrupt(raw)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        p = self.pipeline
        # Direction 08 §4.1: `uniform` keeps the exact legacy draw (sample-resolution
        # `rng.integers` on the sample/remix streams — bit-identical to the shared
        # baseline cell). A non-uniform policy draws its weighted grid start from the
        # dedicated `sampling` stream, so it perturbs no other stream; the curriculum
        # schedule reads the training `step` (index // batch_size).
        uniform = self.sampler.is_uniform()
        policy_rng = None if uniform else p.stream("sampling", index)
        step = index // self.batch_size

        # 1. Vocals: always-on sampler picks the track and the window. The stems
        #    are ε-bleed corrupted at load (before augmentation, §3.1) when set.
        sample_rng = p.stream("sample", index)
        voc_track = str(sample_rng.choice(self.tracks))
        voc_raw, voc_sources = self._sources(voc_track)
        voc_padded = self._prepare(voc_sources["vocals"])
        voc_start = (
            self._start(voc_padded, sample_rng)
            if uniform
            else self._policy_start(voc_padded, voc_track, policy_rng, step)
        )
        vocals = self._slice(voc_padded, voc_start)

        # §5 telemetry: the RAW same-track accompaniment energy at the vocal window
        # — the ⟨a⟩-energy that sets how much ε·a bleeds into this chunk's target
        # (ε- and augmentation-invariant, so a clean per-chunk cleanliness covariate
        # the trimmer's selection is ranked against; MASTER_PLAN §5, §11 telemetry).
        voc_acc_padded = self._prepare(voc_raw["accompaniment"])
        voc_acc_start = min(voc_start, len(voc_acc_padded) - self.chunk_len)
        acc_energy = float(np.mean(self._slice(voc_acc_padded, voc_acc_start).astype(np.float64) ** 2))

        # 2. Accompaniment: remixed from a separate track (remix stream), or the
        #    same track's accompaniment at the same window (true track mixture).
        if p.remix:
            remix_rng = p.stream("remix", index)
            acc_track = str(remix_rng.choice(self.tracks))
            _, acc_sources = self._sources(acc_track)
            acc_padded = self._prepare(acc_sources["accompaniment"])
            # §4.1: the remix partner's chunk is drawn by the *same* policy (a second
            # draw off the shared `sampling` stream for the non-uniform arms).
            acc_start = (
                self._start(acc_padded, remix_rng)
                if uniform
                else self._policy_start(acc_padded, acc_track, policy_rng, step)
            )
        else:
            acc_padded = self._prepare(voc_sources["accompaniment"])
            acc_start = min(voc_start, len(acc_padded) - self.chunk_len)
        accompaniment = self._slice(acc_padded, acc_start)

        # 3. Per-source gain then sign flip, each on its own stream.
        sources = p({"vocals": vocals, "accompaniment": accompaniment}, index)

        # 4. Mixture is the sum of the (possibly transformed) sources.
        mixture = sources["vocals"] + sources["accompaniment"]
        return {
            "mixture": torch.from_numpy(np.ascontiguousarray(mixture)).float(),
            "vocals": torch.from_numpy(np.ascontiguousarray(sources["vocals"])).float(),
            "accompaniment": torch.from_numpy(np.ascontiguousarray(sources["accompaniment"])).float(),
            "acc_energy": torch.tensor(acc_energy, dtype=torch.float32),
        }


def _to_mono_f32(signal: np.ndarray) -> np.ndarray:
    """Mono mixdown (mean of channels) as contiguous float32, shape ``(n,)``."""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim == 2:
        # accept (n, ch) or (ch, n); the longer axis is time.
        if signal.shape[0] < signal.shape[1]:
            signal = signal.T
        signal = signal.mean(axis=1)
    return np.ascontiguousarray(signal, dtype=np.float32)
