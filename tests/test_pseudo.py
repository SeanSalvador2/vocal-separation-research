"""Pseudo-label pool: MUSDB-path guard + chunk-level pool mixing (G0, §3.3/§4.1).

Direction-10 stage-D items (MASTER_PLAN §3.3, §4.1, §8):
* :class:`PseudoLabeledShards` **structurally refuses** any path under a MUSDB shard root
  (path containment) or a MUSDB *decode* root (content sniff) — mirroring D06's guard;
* :class:`MixedPools` selects the pool per example from a **dedicated** ``(seed,"pool",index)``
  stream — the empirical pool frequencies match the stream exactly (not by statistics), the
  pool draw does not perturb either pool's augmentation streams (independence), and a remix
  partner never crosses pools (asserted by construction/type — disjoint pools, item == the
  chosen pool's own chunk).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from singnet.data import (
    AugmentPipeline,
    InMemoryStore,
    Manifest,
    MixedPools,
    MusdbChunks,
    MusdbShardLeak,
    PseudoLabeledShards,
    assert_not_under_musdb,
)
from singnet.data.augment import STREAM_IDS
from singnet.data.pseudo import POOL_STREAM
from singnet.utils.seed import derive_rng

SR = 44100


# --- the dedicated pool stream is id 6 (appended, not reused) ---------------

def test_pool_stream_is_a_new_dedicated_id() -> None:
    assert STREAM_IDS[POOL_STREAM] == 6
    # the five pre-D10 streams keep their ids (append never reorders) -> no data order changes.
    assert STREAM_IDS["sample"] == 0 and STREAM_IDS["remix"] == 1 and STREAM_IDS["gain"] == 2
    assert STREAM_IDS["flip"] == 3 and STREAM_IDS["channelswap"] == 4 and STREAM_IDS["sampling"] == 5


# --- the structural MUSDB-path guard ----------------------------------------

def test_guard_refuses_root_under_musdb(tmp_path) -> None:
    musdb = tmp_path / "musdb_shards"
    musdb.mkdir()
    pseudo_under = musdb / "pseudo"          # nested under the MUSDB root
    pseudo_under.mkdir()
    with pytest.raises(MusdbShardLeak):
        PseudoLabeledShards(pseudo_under, musdb_roots=(musdb,))
    with pytest.raises(MusdbShardLeak):       # equal to the MUSDB root
        PseudoLabeledShards(musdb, musdb_roots=(musdb,))


def test_guard_refuses_musdb_decode_root_by_content(tmp_path) -> None:
    # A MUSDB decode root: a track dir with the drums+bass+other 4-stem layout (a pseudo root
    # is 2-stem only). The content sniff refuses it even with NO explicit musdb_roots passed.
    root = tmp_path / "looks_like_musdb"
    track = root / "SomeTrack"
    track.mkdir(parents=True)
    for stem in ("mixture.wav", "vocals.wav", "drums.wav", "bass.wav", "other.wav"):
        (track / stem).write_bytes(b"")
    with pytest.raises(MusdbShardLeak):
        PseudoLabeledShards(root)


def test_guard_refuses_musdb_index_json(tmp_path) -> None:
    root = tmp_path / "decode_root"
    root.mkdir()
    (root / "index.json").write_text(json.dumps({"T1": {"subset": "train", "n_samples": 10}}))
    with pytest.raises(MusdbShardLeak):
        PseudoLabeledShards(root)


def test_guard_accepts_a_clean_pseudo_root(tmp_path) -> None:
    # A 2-stem pseudo root (no drums/bass/other, no subset index) not under any MUSDB root.
    root = tmp_path / "pseudo_shards"
    clip = root / "fma_000010"
    clip.mkdir(parents=True)
    for stem in ("mixture.wav", "vocals.wav", "accompaniment.wav"):
        (clip / stem).write_bytes(b"")
    store = PseudoLabeledShards(root, musdb_roots=(tmp_path / "musdb_elsewhere",))
    assert store.track_names() == ["fma_000010"]


def test_assert_not_under_musdb_pure_function(tmp_path) -> None:
    musdb = tmp_path / "m"
    musdb.mkdir()
    (tmp_path / "p").mkdir()
    # a clean sibling passes and returns the resolved path
    resolved = assert_not_under_musdb(tmp_path / "p", musdb_roots=(musdb,))
    assert resolved == (tmp_path / "p").resolve()
    with pytest.raises(MusdbShardLeak):
        assert_not_under_musdb(musdb / "sub", musdb_roots=(musdb,))


def test_build_manifest_is_train_only(tmp_path) -> None:
    root = tmp_path / "pseudo"
    for cid in ("a", "b", "c"):
        (root / cid).mkdir(parents=True)
    store = PseudoLabeledShards(root)
    manifest = store.build_manifest()
    assert isinstance(manifest, Manifest)
    assert sorted(manifest.tracks_for("train")) == ["a", "b", "c"]
    assert manifest.tracks_for("valid") == [] and manifest.tracks_for("test") == []


# --- pool-mixing fixtures ---------------------------------------------------

def _pool(tracks_prefix: str, seed: int = 0, length: int = 200) -> MusdbChunks:
    """A MusdbChunks over two synthetic tracks named ``<prefix>_00/01`` (an isolated pool)."""
    rng = np.random.default_rng(hash(tracks_prefix) % (2**31))
    n = SR * 7
    tracks = {
        f"{tracks_prefix}_00": {"vocals": rng.standard_normal(n).astype(np.float32),
                                "accompaniment": rng.standard_normal(n).astype(np.float32)},
        f"{tracks_prefix}_01": {"vocals": rng.standard_normal(n).astype(np.float32),
                                "accompaniment": rng.standard_normal(n).astype(np.float32)},
    }
    store = InMemoryStore(tracks, SR)
    manifest = Manifest(pd.DataFrame(
        [{"track": t, "split": "train"} for t in tracks]
    ))
    pipe = AugmentPipeline(remix=True, gain=True, flip=True)
    return MusdbChunks(store, manifest, "train", seed=seed, length=length, pipeline=pipe)


# --- empirical pool frequencies match the dedicated stream EXACTLY ----------

def test_pool_selection_is_exactly_the_dedicated_stream() -> None:
    musdb, pseudo = _pool("musdb"), _pool("fma")
    mp = MixedPools(musdb, pseudo, p_fma=0.5, seed=7, length=400)
    # recompute the pool sequence directly from the (seed,"pool",index) stream — the pool
    # selection must equal it example-for-example (via the stream, not by statistics).
    expected = [
        "fma" if float(derive_rng(7, STREAM_IDS[POOL_STREAM], i).random()) < 0.5 else "musdb"
        for i in range(400)
    ]
    assert [mp.pool_of(i) for i in range(400)] == expected
    # the realized frequency is then a FIXED, reproducible number for this seed (a determinism
    # anchor, not a random draw): close to p_fma over many draws.
    frac = sum(p == "fma" for p in expected) / 400
    assert abs(frac - 0.5) < 0.05


def test_pool_counts_match_ratio_deterministically() -> None:
    musdb, pseudo = _pool("musdb"), _pool("fma")
    counts_a = MixedPools(musdb, pseudo, p_fma=0.25, seed=0, length=800).pool_counts()
    counts_b = MixedPools(musdb, pseudo, p_fma=0.25, seed=0, length=800).pool_counts()
    assert counts_a == counts_b                       # reproducible
    assert counts_a["fma"] + counts_a["musdb"] == 800
    assert abs(counts_a["fma"] / 800 - 0.25) < 0.05   # tracks p_fma


def test_p_fma_extremes_route_entirely_to_one_pool() -> None:
    musdb, pseudo = _pool("musdb"), _pool("fma")
    all_fma = MixedPools(musdb, pseudo, p_fma=1.0, seed=1, length=64)
    all_musdb = MixedPools(musdb, pseudo, p_fma=0.0, seed=1, length=64)
    assert all(all_fma.pool_of(i) == "fma" for i in range(64))
    assert all(all_musdb.pool_of(i) == "musdb" for i in range(64))


# --- within-pool remix guarantee: by construction, not by sampling ----------

def test_within_pool_remix_by_construction() -> None:
    musdb, pseudo = _pool("musdb"), _pool("fma")
    mp = MixedPools(musdb, pseudo, p_fma=0.5, seed=3, length=200)
    # (1) the two pools are DISJOINT track sets, so a remix partner (drawn from a pool's own
    #     `tracks`) can never be from the other pool — the structural guarantee.
    assert set(musdb.tracks).isdisjoint(set(pseudo.tracks))
    # (2) each returned example equals EXACTLY the chosen pool's own chunk at that index — the
    #     whole (vocals + remixed accompaniment + mixture) is composed within one pool.
    for i in (0, 1, 5, 13, 40, 199):
        expected_ds = musdb if mp.pool_of(i) == "musdb" else pseudo
        item = mp[i]
        ref = expected_ds[i]
        assert torch.equal(item["vocals"], ref["vocals"])
        assert torch.equal(item["accompaniment"], ref["accompaniment"])
        assert torch.equal(item["mixture"], ref["mixture"])
    # the pool tag rides along for telemetry.
    assert "pool_is_fma" in mp[0]


# --- the pool draw does NOT perturb either pool's augmentation streams -------

def test_pool_draw_is_independent_of_within_pool_streams() -> None:
    musdb, pseudo = _pool("musdb"), _pool("fma")
    # wrapping in MixedPools (any p_fma/seed) does not change what each pool yields at index i:
    # the pool stream is id 6, disjoint from sample/remix/gain/flip. So MixedPools[i] for a
    # musdb-pool i is byte-identical to the bare musdb_ds[i].
    for p_fma, seed in ((0.5, 0), (0.3, 9), (0.7, 21)):
        mp = MixedPools(musdb, pseudo, p_fma=p_fma, seed=seed, length=120)
        for i in range(120):
            ref = (musdb if mp.pool_of(i) == "musdb" else pseudo)[i]
            item = mp[i]
            assert torch.equal(item["vocals"], ref["vocals"])
            assert torch.equal(item["accompaniment"], ref["accompaniment"])
    # and the bare pool datasets are themselves unchanged object-for-object (sanity).
    assert torch.equal(musdb[4]["vocals"], _pool("musdb")[4]["vocals"])


def test_mixedpools_validates_p_fma() -> None:
    musdb, pseudo = _pool("musdb"), _pool("fma")
    for bad in (-0.1, 1.5, 2.0):
        with pytest.raises(ValueError):
            MixedPools(musdb, pseudo, p_fma=bad, seed=0)
