"""Direction 02 augmentation switchboard + per-transform RNG streams (G0, §3.1/§3.5).

Covers the MASTER_PLAN §8 gate-G0 items:

* per-transform RNG-stream independence — disabling one transform leaves the
  others' draws unchanged (streams are keyed by `(seed, name, step)`),
* the switchboard builds the declared pipeline for each Direction-02 config,
* `AugmentPipeline(True, True, True)` is exactly the Direction-01 full recipe
  (remix -> gain -> flip, constants per §5) — the no-regression guarantee,
* "no remix" is the true track mixture (same track, same window).
"""

from __future__ import annotations

import numpy as np
import torch

from singnet.data import AugmentPipeline, MusdbChunks, load_track_allowlist
from singnet.data.augment import FLIP_PROB, GAIN_HIGH, GAIN_LOW
from singnet.utils.config import augment_switches, resolve_config

CONFIG_DIR = "02-augmentation-data-scaling/configs"


# --- per-transform stream independence (the §3.5 leave-one-out invariant) -----

def test_streams_are_independent_of_the_other_switches() -> None:
    # The gain and remix streams are byte-identical whether or not flip is on:
    # each is a pure function of (seed, name, step), nothing else.
    on = AugmentPipeline(remix=True, gain=True, flip=True, seed=0)
    no_flip = AugmentPipeline(remix=True, gain=True, flip=False, seed=0)
    no_gain = AugmentPipeline(remix=True, gain=False, flip=True, seed=0)
    for step in range(6):
        assert np.array_equal(on.stream("gain", step).uniform(size=4),
                              no_flip.stream("gain", step).uniform(size=4))
        assert np.array_equal(on.stream("remix", step).integers(0, 10_000, size=4),
                              no_flip.stream("remix", step).integers(0, 10_000, size=4))
        # and toggling gain does not touch the flip or remix streams
        assert np.array_equal(on.stream("flip", step).random(size=4),
                              no_gain.stream("flip", step).random(size=4))
        assert np.array_equal(on.stream("remix", step).integers(0, 10_000, size=4),
                              no_gain.stream("remix", step).integers(0, 10_000, size=4))


def test_disabling_flip_leaves_remix_and_gain_draws_unchanged(synthetic_store, synthetic_manifest) -> None:
    # At the dataset level: with flip on vs off (remix+gain identical streams), the
    # magnitudes of both sources are identical — only the sign pattern differs.
    common = dict(seed=0, length=24)
    ds_flip = MusdbChunks(synthetic_store, synthetic_manifest, "train",
                          pipeline=AugmentPipeline(remix=True, gain=True, flip=True), **common)
    ds_noflip = MusdbChunks(synthetic_store, synthetic_manifest, "train",
                            pipeline=AugmentPipeline(remix=True, gain=True, flip=False), **common)
    for i in (0, 5, 11, 17):
        a, b = ds_flip[i], ds_noflip[i]
        assert torch.equal(a["vocals"].abs(), b["vocals"].abs())
        assert torch.equal(a["accompaniment"].abs(), b["accompaniment"].abs())


def test_toggling_remix_leaves_vocals_bit_identical(synthetic_store, synthetic_manifest) -> None:
    # Remix changes ONLY the accompaniment source; the vocals chunk (sample stream)
    # and its gain/flip (own streams) are untouched — the controlled comparison.
    common = dict(seed=0, length=24)
    ds_remix = MusdbChunks(synthetic_store, synthetic_manifest, "train",
                           pipeline=AugmentPipeline(remix=True, gain=True, flip=True), **common)
    ds_noremix = MusdbChunks(synthetic_store, synthetic_manifest, "train",
                             pipeline=AugmentPipeline(remix=False, gain=True, flip=True), **common)
    for i in (0, 3, 9, 20):
        assert torch.equal(ds_remix[i]["vocals"], ds_noremix[i]["vocals"])


def test_no_remix_is_true_track_mixture(synthetic_manifest) -> None:
    # Same track, same window: with remix off and no gain/flip, the accompaniment
    # is the same track's accompaniment at the same start as the vocals.
    from singnet.data import InMemoryStore

    n = 44100 * 7
    ramp = np.arange(n, dtype=np.float32)
    tracks = {
        "train_00": {"vocals": ramp, "accompaniment": -ramp},
        "train_01": {"vocals": np.ones(n, dtype=np.float32), "accompaniment": np.full(n, 2.0, np.float32)},
    }
    store = InMemoryStore(tracks, 44100)
    ds = MusdbChunks(store, synthetic_manifest, "train", seed=1, length=16,
                     pipeline=AugmentPipeline(remix=False, gain=False, flip=False))
    item = ds[4]
    # vocals[k] = start + k and accompaniment[k] = -(start + k) for the SAME start.
    assert torch.allclose(item["accompaniment"], -item["vocals"], atol=1e-4)


# --- the switchboard builds the declared pipeline for each config -------------

def test_configs_build_declared_switchboard() -> None:
    expected = {
        "base.yaml": (True, True, True),
        "loo_no_remix_seed0.yaml": (False, True, True),
        "loo_no_gain_seed0.yaml": (True, False, True),
        "loo_no_flip_seed0.yaml": (True, True, False),
        "loo_none_seed0.yaml": (False, False, False),
        "scale_n21_seed0.yaml": (True, True, True),
        "scale_n43_seed0.yaml": (True, True, True),
        "contingency_full_sisdr_seed0.yaml": (True, True, True),
        "contingency_none_sisdr_seed0.yaml": (False, False, False),
    }
    for name, (remix, gain, flip) in expected.items():
        cfg = resolve_config(f"{CONFIG_DIR}/{name}")
        sw = augment_switches(cfg)
        assert (sw["remix"], sw["gain"], sw["flip"]) == (remix, gain, flip), name


def test_scaling_configs_declare_their_subset() -> None:
    for name, expect in {"scale_n21_seed0.yaml": "n21.csv", "scale_n43_seed0.yaml": "n43.csv",
                         "scale_n64_seed0.yaml": "n64.csv"}.items():
        cfg = resolve_config(f"{CONFIG_DIR}/{name}")
        assert cfg["data"]["track_allowlist_csv"].endswith(expect), name
    # the full-recipe / LOO / contingency configs carry no allowlist (full split)
    for name in ("base.yaml", "loo_no_remix_seed0.yaml", "contingency_full_sisdr_seed0.yaml"):
        cfg = resolve_config(f"{CONFIG_DIR}/{name}")
        assert cfg.get("data", {}).get("track_allowlist_csv") is None, name


# --- AugmentPipeline(True, True, True) == the Direction-01 full recipe ---------

def test_default_pipeline_is_the_full_recipe() -> None:
    pipe = AugmentPipeline()
    assert (pipe.remix, pipe.gain, pipe.flip) == (True, True, True)
    assert (pipe.gain_low, pipe.gain_high, pipe.flip_prob) == (GAIN_LOW, GAIN_HIGH, FLIP_PROB)
    assert AugmentPipeline(True, True, True) == pipe  # positional (remix, gain, flip)


def test_full_recipe_applies_gain_then_flip_in_range() -> None:
    pipe = AugmentPipeline(remix=True, gain=True, flip=True, seed=3)
    src = {"vocals": np.ones(2048, np.float32), "accompaniment": np.ones(2048, np.float32)}
    seen_flip = False
    for step in range(60):
        out = pipe(src, step)
        for sig in out.values():
            peak = float(np.max(np.abs(sig)))
            assert GAIN_LOW - 1e-6 <= peak <= GAIN_HIGH + 1e-6  # gain scaled the ones
            if np.all(sig < 0):
                seen_flip = True
    assert seen_flip  # sign flip does fire at p=0.5


def test_gain_only_and_flip_only_isolate_their_effect() -> None:
    ones = {"vocals": np.ones(512, np.float32), "accompaniment": np.ones(512, np.float32)}
    # gain only: values are a positive scale of ones (never negative)
    gain_only = AugmentPipeline(remix=True, gain=True, flip=False, seed=0)
    for step in range(20):
        for sig in gain_only(ones, step).values():
            assert np.all(sig > 0)
    # flip only: values are exactly +1 or -1 (no scaling)
    flip_only = AugmentPipeline(remix=True, gain=False, flip=True, seed=0)
    for step in range(20):
        for sig in flip_only(ones, step).values():
            assert set(np.unique(sig)).issubset({-1.0, 1.0})


# --- track allowlist restricts the sampling pool (§3.2) -----------------------

def test_track_allowlist_restricts_pool(synthetic_store, synthetic_manifest) -> None:
    full = MusdbChunks(synthetic_store, synthetic_manifest, "train", seed=0, length=16)
    assert full.n_songs == 2  # train_00, train_01
    subset = MusdbChunks(synthetic_store, synthetic_manifest, "train", seed=0, length=16,
                         track_allowlist=["train_00"])
    assert subset.n_songs == 1 and subset.tracks == ["train_00"]


def test_load_track_allowlist_none_is_full_split() -> None:
    assert load_track_allowlist(None) is None
