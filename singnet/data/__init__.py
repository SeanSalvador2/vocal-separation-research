"""Data layer: split manifest, augmentation recipe, and the chunk dataset."""

from __future__ import annotations

from .augment import (
    STREAM_IDS,
    AugmentPipeline,
    random_channel_swap,
    random_gain,
    random_sign_flip,
    remix,
)
from .corrupt import (
    CORRUPTIBLE_SPLITS,
    EvalSplitCorruptionError,
    StemBleed,
    build_corruption,
)
from .manifest import Manifest, TestRowError, load_manifest
from .musdb_dataset import (
    DEFAULT_CHUNK_S,
    DEFAULT_SR,
    SOURCE_NAMES,
    InMemoryStore,
    MusdbChunks,
    TrackStore,
    WavShardStore,
    load_track_allowlist,
)
from .profiles import (
    compute_energy_profiles,
    load_energy_profiles,
    windowed_vocal_rms,
    write_energy_profiles,
)
from .pseudo import (
    MixedPools,
    MusdbShardLeak,
    PseudoLabeledShards,
    assert_not_under_musdb,
    build_pseudo_dataset,
    read_teacher_provenance,
)
from .sampling import POLICIES, ChunkSampler, build_chunk_sampler

__all__ = [
    "AugmentPipeline",
    "STREAM_IDS",
    "random_gain",
    "random_sign_flip",
    "random_channel_swap",
    "remix",
    "StemBleed",
    "build_corruption",
    "EvalSplitCorruptionError",
    "CORRUPTIBLE_SPLITS",
    "Manifest",
    "TestRowError",
    "load_manifest",
    "DEFAULT_CHUNK_S",
    "DEFAULT_SR",
    "SOURCE_NAMES",
    "InMemoryStore",
    "MusdbChunks",
    "TrackStore",
    "WavShardStore",
    "load_track_allowlist",
    "ChunkSampler",
    "POLICIES",
    "build_chunk_sampler",
    "windowed_vocal_rms",
    "compute_energy_profiles",
    "write_energy_profiles",
    "load_energy_profiles",
    "PseudoLabeledShards",
    "MixedPools",
    "MusdbShardLeak",
    "assert_not_under_musdb",
    "build_pseudo_dataset",
    "read_teacher_provenance",
]
