"""Shared synthetic fixtures — CPU only, no MUSDB, no network.

Every fixture builds small, deterministic signals so the whole suite stays well
under the 5-minute CPU budget (MASTER_PLAN G0). Nothing here touches disk or the
real dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from singnet.audio import STFT
from singnet.data import InMemoryStore, Manifest

SR = 44100


def _tone(freq: float, seconds: float, sr: int = SR, amp: float = 0.2) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture(scope="session")
def stft() -> STFT:
    return STFT()


@pytest.fixture()
def synthetic_store() -> InMemoryStore:
    """Four ~7 s tracks with distinct vocal/accompaniment tones."""
    rng = np.random.default_rng(1234)
    tracks: dict[str, dict[str, np.ndarray]] = {}
    for i in range(4):
        secs = 7.0
        vocals = _tone(220 + 40 * i, secs) + 0.01 * rng.standard_normal(int(secs * SR)).astype(np.float32)
        accompaniment = _tone(110 + 30 * i, secs) + 0.01 * rng.standard_normal(int(secs * SR)).astype(np.float32)
        tracks[f"train_{i:02d}"] = {"vocals": vocals, "accompaniment": accompaniment}
    return InMemoryStore(tracks, SR)


@pytest.fixture()
def synthetic_manifest() -> Manifest:
    rows = [
        {"track": "train_00", "split": "train"},
        {"track": "train_01", "split": "train"},
        {"track": "train_02", "split": "valid"},
        {"track": "train_03", "split": "valid"},
        {"track": "held_out_00", "split": "test"},
    ]
    return Manifest(pd.DataFrame(rows))


@pytest.fixture()
def small_waveform_batch() -> torch.Tensor:
    """A tiny (B=2) waveform batch of one eval chunk length."""
    from singnet.eval import EVAL_CHUNK_SAMPLES

    torch.manual_seed(0)
    return 0.1 * torch.randn(2, EVAL_CHUNK_SAMPLES)
