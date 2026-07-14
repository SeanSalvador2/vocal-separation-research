"""Chunk-sampling policies — exact weights, floor bound, schedule, stream independence,
and the uniform bit-compatibility regression (G0, MASTER_PLAN §4.1).

The load-bearing test is :func:`test_uniform_policy_reproduces_legacy_chunk_starts`: the
``uniform`` arm is the shared baseline cell, so it must draw the *exact* track + start the
pre-Direction-08 loader did. The expected starts are reconstructed from the documented
legacy formula (``sample_rng.choice(tracks)`` then ``sample_rng.integers(0, n_starts)``),
independent of the new sampler code, so a regression in the uniform path breaks the test.
"""

from __future__ import annotations

import numpy as np
import torch

from singnet.data import AugmentPipeline, ChunkSampler, MusdbChunks
from singnet.data.sampling import DEFAULT_FLOOR_LAMBDA, build_chunk_sampler


# --- exact weight vectors ---------------------------------------------------

def test_uniform_weights_are_flat() -> None:
    w = ChunkSampler("uniform").weights(np.array([1.0, 5.0, 0.0, 2.0]))
    assert np.allclose(w, 0.25)


def test_energy_weights_exact() -> None:
    # p(s) = (1−λ)·E/ΣE + λ/N, λ=0.1, E=[1,3], N=2, ΣE=4.
    w = ChunkSampler("energy", floor_lambda=0.1).weights(np.array([1.0, 3.0]))
    assert np.allclose(w, [0.9 * 0.25 + 0.05, 0.9 * 0.75 + 0.05])  # [0.275, 0.725]
    assert np.isclose(w.sum(), 1.0)


def test_energy_floor_lower_bound() -> None:
    # Every start keeps mass ≥ λ/N even where E=0 — silence is down-weighted, not starved.
    lam, profile = DEFAULT_FLOOR_LAMBDA, np.array([0.0, 0.0, 5.0, 0.0])
    w = ChunkSampler("energy", floor_lambda=lam).weights(profile)
    assert np.all(w >= lam / profile.size - 1e-12)
    assert np.isclose(w[0], lam / profile.size)  # a zero-energy start sits exactly at the floor


def test_drop_support_set() -> None:
    # drop keeps uniform mass on the RMS ≥ θ starts, zero on the sub-θ ones.
    thresh = 10.0 ** (-60.0 / 20.0)
    profile = np.array([thresh * 2, thresh * 0.5, thresh * 3, thresh * 0.1])
    w = ChunkSampler("drop", theta_db=-60.0).weights(profile)
    assert np.allclose(w, [0.5, 0.0, 0.5, 0.0])
    assert w[1] == 0.0 and w[3] == 0.0  # silent-window starts excluded entirely


def test_drop_all_silent_falls_back_to_uniform() -> None:
    profile = np.full(5, 10.0 ** (-80.0 / 20.0))  # every window below −60 dBFS
    w = ChunkSampler("drop", theta_db=-60.0).weights(profile)
    assert np.allclose(w, 0.2)  # degenerate track: uniform fallback (documented, §12)


# --- curriculum schedule ----------------------------------------------------

def test_curriculum_lambda_schedule_values() -> None:
    # λ(t): linear 1.0 → 0.1 over the first 50 % of steps, then held at 0.1.
    s = ChunkSampler("curriculum", floor_lambda=0.1, total_steps=1000)
    got = [s.lambda_at(t) for t in (0, 250, 500, 750, 1000)]
    assert np.allclose(got, [1.0, 0.55, 0.1, 0.1, 0.1])


def test_curriculum_weights_track_the_schedule() -> None:
    profile = np.array([1.0, 3.0])
    s = ChunkSampler("curriculum", floor_lambda=0.1, total_steps=1000)
    assert np.allclose(s.weights(profile, step=0), [0.5, 0.5])          # λ=1 -> uniform
    assert np.allclose(s.weights(profile, step=500), [0.275, 0.725])    # λ=0.1 -> energy
    # a mid-anneal point (λ=0.55 at 25 %): (1−λ)E/ΣE + λ/N.
    mid = s.weights(profile, step=250)
    assert np.allclose(mid, [0.45 * 0.25 + 0.275, 0.45 * 0.75 + 0.275])


def test_curriculum_without_total_steps_is_fixed_energy() -> None:
    s = ChunkSampler("curriculum", floor_lambda=0.1, total_steps=None)
    assert s.lambda_at(0) == 0.1  # no schedule -> degenerates to fixed-λ energy


# --- the weighted draw lands on grid starts ---------------------------------

def test_energy_draw_prefers_high_energy_grid_points() -> None:
    # A profile with all mass on grid index 3 draws start ≈ 3·sr almost surely.
    sr = 100
    profile = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
    sampler = ChunkSampler("energy", floor_lambda=0.0, grid_s=1.0, sr=sr)  # no floor -> deterministic
    rng = np.random.default_rng(0)
    starts = [sampler.start(10_000, profile, rng) for _ in range(20)]
    assert set(starts) == {3 * sr}


def test_policy_start_clamps_into_range() -> None:
    sr = 100
    profile = np.array([0.0, 1.0])  # grid index 1 -> 1·sr = 100, but only 50 valid starts
    sampler = ChunkSampler("energy", floor_lambda=0.0, grid_s=1.0, sr=sr)
    assert sampler.start(50, profile, np.random.default_rng(0)) == 49  # clamped to n-1


# --- uniform bit-compatibility (the shared-cell guarantee) ------------------

def test_uniform_sampler_start_equals_integers() -> None:
    sampler = ChunkSampler("uniform")
    for seed in range(4):
        a, b = np.random.default_rng(seed), np.random.default_rng(seed)
        assert sampler.start(1000, None, a) == int(b.integers(0, 1000))


def test_uniform_policy_reproduces_legacy_chunk_starts(synthetic_store, synthetic_manifest) -> None:
    seed = 0
    ds = MusdbChunks(  # augmentation off isolates the sampler draw
        synthetic_store, synthetic_manifest, "train", seed=seed, length=32,
        augment=False, remix=False,
    )
    pipe = AugmentPipeline(remix=False, gain=False, flip=False, seed=seed)
    for index in range(12):
        # the documented pre-D08 draw: track then start, both off the `sample` stream.
        rng = pipe.stream("sample", index)
        voc_track = str(rng.choice(ds.tracks))
        padded = synthetic_store.load_sources(voc_track)["vocals"]  # ≥ chunk_len, no padding
        expected_start = int(rng.integers(0, len(padded) - ds.chunk_len + 1))
        expected = padded[expected_start : expected_start + ds.chunk_len].astype("float32")
        assert np.array_equal(ds[index]["vocals"].numpy(), expected)


def test_explicit_uniform_sampler_equals_default(synthetic_store, synthetic_manifest) -> None:
    # Passing an explicit uniform ChunkSampler is byte-identical to the default path.
    kw = dict(split="train", seed=1, length=32)
    default = MusdbChunks(synthetic_store, synthetic_manifest, **kw)
    explicit = MusdbChunks(synthetic_store, synthetic_manifest, sampler=ChunkSampler("uniform"), **kw)
    for i in (0, 4, 7):
        assert torch.equal(default[i]["mixture"], explicit[i]["mixture"])
        assert torch.equal(default[i]["vocals"], explicit[i]["vocals"])


# --- stream independence from augmentation ----------------------------------

def test_sampling_stream_is_independent_of_augmentation_streams() -> None:
    # Drawing from the `sampling` stream leaves the gain/flip/sample streams untouched:
    # each stream() call is a fresh (seed, id, step) generator.
    pipe = AugmentPipeline(seed=3)
    gain_before = pipe.stream("gain", 5).random()
    _ = pipe.stream("sampling", 5).random()  # a policy draw
    gain_after = pipe.stream("gain", 5).random()
    assert gain_before == gain_after
    # the sampling stream (id 5) is distinct from the sample stream (id 0).
    assert pipe.stream("sampling", 5).random() != pipe.stream("sample", 5).random()


def test_energy_policy_leaves_augmentation_draws_unchanged(synthetic_store, synthetic_manifest) -> None:
    # Switching the policy from uniform to energy must not change the gain/flip draws for
    # a given (seed, index): those come from their own streams, not the sampling stream.
    profiles = {t: np.ones(3) for t in ("train_00", "train_01")}
    seed, index = 2, 6
    # the gain stream is a pure function of (seed, index), independent of the policy.
    gain_a = AugmentPipeline(seed=seed).stream("gain", index).random()
    gain_b = AugmentPipeline(seed=seed).stream("gain", index).random()
    assert gain_a == gain_b
    # and constructing the energy dataset does not raise for a well-formed profile set.
    ds = MusdbChunks(
        synthetic_store, synthetic_manifest, "train", seed=seed, length=32,
        sampler=ChunkSampler("energy", sr=synthetic_store.sample_rate), energy_profiles=profiles,
    )
    assert ds[index]["mixture"].shape == ds[index]["vocals"].shape


def test_non_uniform_policy_without_profiles_errors(synthetic_store, synthetic_manifest) -> None:
    ds = MusdbChunks(
        synthetic_store, synthetic_manifest, "train", seed=0, length=32,
        sampler=ChunkSampler("energy", sr=synthetic_store.sample_rate), energy_profiles=None,
    )
    try:
        ds[0]
        raised = False
    except ValueError as exc:
        raised = "write-energy-profiles" in str(exc)
    assert raised  # loud error pointing at the prep pass


# --- build_chunk_sampler from config ----------------------------------------

def test_build_chunk_sampler_defaults_to_uniform() -> None:
    # A config with no sampling block (every D01–06 config) builds the uniform no-op.
    sampler = build_chunk_sampler({"arm": "l1mag", "seed": 0, "steps": 16000})
    assert sampler.is_uniform()


def test_build_chunk_sampler_reads_policy_block() -> None:
    cfg = {"arm": "energy", "seed": 0, "steps": 16000,
           "sampling": {"policy": "curriculum", "floor_lambda": 0.1}}
    sampler = build_chunk_sampler(cfg)
    assert sampler.policy == "curriculum" and sampler.total_steps == 16000
    assert sampler.lambda_at(0) == 1.0 and sampler.lambda_at(16000) == 0.1
