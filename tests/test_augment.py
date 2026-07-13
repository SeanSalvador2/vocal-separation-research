"""Augmentation determinism and remix additivity (MASTER_PLAN §4.5, G0)."""

from __future__ import annotations

import numpy as np
import torch

from singnet.data import AugmentPipeline, MusdbChunks, random_gain, random_sign_flip
from singnet.utils.seed import derive_rng


def test_gain_range_within_bounds() -> None:
    rng = derive_rng(0, 0)
    src = {"vocals": np.ones(100, dtype=np.float32), "accompaniment": np.ones(100, dtype=np.float32)}
    for _ in range(200):
        out = random_gain(src, rng, low=0.25, high=1.25)
        for sig in out.values():
            peak = float(np.max(np.abs(sig)))
            assert 0.25 - 1e-6 <= peak <= 1.25 + 1e-6


def test_sign_flip_only_negates() -> None:
    rng = derive_rng(0, 1)
    src = {"vocals": np.array([1.0, -2.0, 3.0], dtype=np.float32)}
    flipped = 0
    for _ in range(50):
        out = random_sign_flip(src, rng, p=0.5)["vocals"]
        assert np.allclose(np.abs(out), np.abs(src["vocals"]))
        if np.allclose(out, -src["vocals"]):
            flipped += 1
    assert 0 < flipped < 50  # both branches exercised


def test_pipeline_deterministic_given_seed_and_step() -> None:
    # The pipeline now owns its per-transform (seed, name, step) streams, so it
    # is called with a step index rather than a pre-built generator (Direction 02
    # §3.5). Same guarantee as before: draws are a pure function of (seed, step).
    pipe = AugmentPipeline(seed=7)
    src = {
        "vocals": np.linspace(-1, 1, 256, dtype=np.float32),
        "accompaniment": np.linspace(1, -1, 256, dtype=np.float32),
    }
    out_a = pipe(src, 3)
    out_b = pipe(src, 3)
    for key in src:
        assert np.array_equal(out_a[key], out_b[key])
    out_c = pipe(src, 4)
    assert not np.array_equal(out_a["vocals"], out_c["vocals"])
    # The seed also keys the stream: a different seed gives a different draw.
    out_d = AugmentPipeline(seed=8)(src, 3)
    assert not np.array_equal(out_a["vocals"], out_d["vocals"])


def test_dataset_deterministic_and_arm_independent(synthetic_store, synthetic_manifest) -> None:
    ds = MusdbChunks(synthetic_store, synthetic_manifest, "train", seed=0, length=32)
    # same index -> identical item (no loss arm ever enters the RNG stream)
    a, b = ds[5], ds[5]
    assert torch.equal(a["mixture"], b["mixture"])
    assert torch.equal(a["vocals"], b["vocals"])
    # different index -> different item
    assert not torch.equal(ds[5]["mixture"], ds[6]["mixture"])


def test_dataset_mixture_equals_sum_of_sources(synthetic_store, synthetic_manifest) -> None:
    ds = MusdbChunks(synthetic_store, synthetic_manifest, "train", seed=0, length=32)
    for idx in (0, 3, 9):
        item = ds[idx]
        assert torch.allclose(item["mixture"], item["vocals"] + item["accompaniment"], atol=1e-6)


def test_two_seeds_differ(synthetic_store, synthetic_manifest) -> None:
    ds0 = MusdbChunks(synthetic_store, synthetic_manifest, "train", seed=0, length=32)
    ds1 = MusdbChunks(synthetic_store, synthetic_manifest, "train", seed=1, length=32)
    assert not torch.equal(ds0[0]["mixture"], ds1[0]["mixture"])
