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
]
