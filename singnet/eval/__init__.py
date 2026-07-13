"""Evaluation: overlap-add inference, oracles/floor, and the frozen protocol."""

from __future__ import annotations

from .evaluate import (
    TrackScores,
    evaluate,
    oracle_ibm,
    oracle_irm,
    score_system,
    validation_sisdr,
)
from .overlap_add import (
    DEFAULT_OVERLAP,
    EVAL_CHUNK_SAMPLES,
    identity_chunk_separator,
    model_chunk_separator,
    overlap_add,
    raised_cosine_window,
    separate_track,
)

__all__ = [
    "TrackScores",
    "evaluate",
    "oracle_ibm",
    "oracle_irm",
    "score_system",
    "validation_sisdr",
    "DEFAULT_OVERLAP",
    "EVAL_CHUNK_SAMPLES",
    "identity_chunk_separator",
    "model_chunk_separator",
    "overlap_add",
    "raised_cosine_window",
    "separate_track",
]
