"""Evaluation: overlap-add inference, oracles/floor, banded metrics, and the protocol."""

from __future__ import annotations

from .banded import (
    analysis_grid_hz,
    band_limited_sisdr,
    band_mag_error,
    banded_report,
)
from .evaluate import (
    TrackScores,
    evaluate,
    oracle_ibm,
    oracle_irm,
    score_system,
    validation_report,
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
    "analysis_grid_hz",
    "band_limited_sisdr",
    "band_mag_error",
    "banded_report",
    "TrackScores",
    "evaluate",
    "oracle_ibm",
    "oracle_irm",
    "score_system",
    "validation_report",
    "validation_sisdr",
    "DEFAULT_OVERLAP",
    "EVAL_CHUNK_SAMPLES",
    "identity_chunk_separator",
    "model_chunk_separator",
    "overlap_add",
    "raised_cosine_window",
    "separate_track",
]
