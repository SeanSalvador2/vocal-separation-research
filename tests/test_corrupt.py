"""Direction 06 stem-bleed corruption (MASTER_PLAN §3.1, gate G0).

Exact ṽ/ã formulas; mixture invariance to float tolerance; determinism (no RNG);
the structural eval-split guard (constructor + dataset wiring refuse valid/test);
the corruption does not perturb any augmentation RNG stream; and the per-chunk
accompaniment-energy telemetry is threaded into the batch.
"""

from __future__ import annotations

import numpy as np
import pytest

from singnet.data import (
    AugmentPipeline,
    CORRUPTIBLE_SPLITS,
    EvalSplitCorruptionError,
    MusdbChunks,
    StemBleed,
    build_corruption,
)
from singnet.data.corrupt import corruption_epsilon


# --- exact formulas + invariance + determinism ------------------------------

def test_stembleed_exact_formulas() -> None:
    sb = StemBleed(0.3)
    v = np.array([1.0, 2.0, -3.0], dtype=np.float32)
    a = np.array([10.0, -20.0, 30.0], dtype=np.float32)
    out = sb({"vocals": v, "accompaniment": a})
    assert np.allclose(out["vocals"], v + 0.3 * a, atol=1e-6)          # ṽ = v + εa
    assert np.allclose(out["accompaniment"], 0.7 * a, atol=1e-6)        # ã = (1−ε)a


@pytest.mark.parametrize("eps", [0.0, 0.05, 0.15, 0.30, 1.0])
def test_mixture_invariance_exact(eps: float) -> None:
    rng = np.random.default_rng(0)
    v = rng.standard_normal(2048).astype(np.float32)
    a = rng.standard_normal(2048).astype(np.float32)
    out = StemBleed(eps)({"vocals": v, "accompaniment": a})
    # ṽ + ã = v + a exactly (the defining property of real bleed / SDXDB23).
    assert np.allclose(out["vocals"] + out["accompaniment"], v + a, atol=1e-5)


def test_determinism_no_rng() -> None:
    sb = StemBleed(0.15)
    src = {"vocals": np.arange(16, dtype=np.float32), "accompaniment": np.ones(16, np.float32)}
    a1 = sb(src)
    a2 = sb(src)
    assert np.array_equal(a1["vocals"], a2["vocals"])
    assert np.array_equal(a1["accompaniment"], a2["accompaniment"])


def test_epsilon_zero_is_identity() -> None:
    v = np.array([1.0, 2.0, 3.0], np.float32)
    a = np.array([4.0, 5.0, 6.0], np.float32)
    out = StemBleed(0.0)({"vocals": v, "accompaniment": a})
    assert np.allclose(out["vocals"], v) and np.allclose(out["accompaniment"], a)


def test_constructor_rejects_out_of_range_epsilon() -> None:
    for bad in (-0.1, 1.5, 2.0):
        with pytest.raises(ValueError):
            StemBleed(bad)


def test_missing_stem_key_raises() -> None:
    with pytest.raises(KeyError):
        StemBleed(0.3)({"vocals": np.zeros(4, np.float32)})


# --- the structural eval-split guard ----------------------------------------

@pytest.mark.parametrize("split", ["valid", "test"])
def test_build_corruption_refuses_eval_splits(split: str) -> None:
    with pytest.raises(EvalSplitCorruptionError):
        build_corruption({"epsilon": 0.30}, split)


def test_build_corruption_none_when_clean() -> None:
    # ε = 0 / absent -> None on ANY split (no corruption requested, so no guard trip).
    assert build_corruption({"epsilon": 0.0}, "valid") is None
    assert build_corruption(None, "test") is None
    assert build_corruption({}, "valid") is None


def test_build_corruption_trains() -> None:
    sb = build_corruption({"epsilon": 0.30}, "train")
    assert isinstance(sb, StemBleed) and sb.epsilon == 0.30
    assert "train" in CORRUPTIBLE_SPLITS


@pytest.mark.parametrize("split", ["valid", "test"])
def test_dataset_refuses_corruption_on_eval_split(synthetic_store, synthetic_manifest, split: str) -> None:
    # Even handed a StemBleed directly, MusdbChunks refuses a non-train split —
    # the guard is structural at the dataset boundary too, not only in the factory.
    with pytest.raises(EvalSplitCorruptionError):
        MusdbChunks(synthetic_store, synthetic_manifest, split, seed=0, corrupt=StemBleed(0.3))


def test_dataset_allows_corruption_on_train(synthetic_store, synthetic_manifest) -> None:
    ds = MusdbChunks(synthetic_store, synthetic_manifest, "train", seed=0, corrupt=StemBleed(0.3), length=8)
    item = ds[0]
    assert {"mixture", "vocals", "accompaniment", "acc_energy"} <= set(item)


def test_config_epsilon_reader() -> None:
    assert corruption_epsilon({"corrupt": {"epsilon": 0.15}}) == 0.15
    assert corruption_epsilon({}) == 0.0
    assert corruption_epsilon({"corrupt": {}}) == 0.0


# --- corruption does not perturb any augmentation RNG stream -----------------

def _dataset(store, manifest, *, corrupt, remix, augment, seed=0, length=16):
    pipe = AugmentPipeline(remix=remix, gain=augment, flip=augment, seed=seed)
    return MusdbChunks(store, manifest, "train", seed=seed, length=length, pipeline=pipe, corrupt=corrupt)


def test_invariance_through_dataset_no_augment(synthetic_store, synthetic_manifest) -> None:
    # augment + remix OFF -> the batch mixture is the true track mixture ṽ+ã = v+a,
    # identical with/without corruption. Also proves the sample stream (which track,
    # which window) is unperturbed (same v, same a were drawn).
    clean = _dataset(synthetic_store, synthetic_manifest, corrupt=None, remix=False, augment=False)
    dirty = _dataset(synthetic_store, synthetic_manifest, corrupt=StemBleed(0.30), remix=False, augment=False)
    for i in (0, 3, 7, 11):
        assert np.allclose(clean[i]["mixture"], dirty[i]["mixture"], atol=1e-5)
        # the vocal target IS corrupted, though (ṽ ≠ v for ε>0).
        assert not np.allclose(clean[i]["vocals"], dirty[i]["vocals"], atol=1e-3)


def test_kept_stream_equality_with_augmentation(synthetic_store, synthetic_manifest) -> None:
    # augment + remix ON. The remix/gain/flip streams must be byte-identical with and
    # without corruption: the corrupted accompaniment is exactly (1−ε)× the clean one,
    # elementwise (same remix partner + window, same gain factor, same flip sign).
    eps = 0.30
    clean = _dataset(synthetic_store, synthetic_manifest, corrupt=None, remix=True, augment=True)
    dirty = _dataset(synthetic_store, synthetic_manifest, corrupt=StemBleed(eps), remix=True, augment=True)
    for i in (0, 2, 5, 9, 13):
        c = clean[i]["accompaniment"].numpy()
        d = dirty[i]["accompaniment"].numpy()
        assert np.allclose(d, (1.0 - eps) * c, atol=1e-5)


def test_acc_energy_threaded_and_deterministic(synthetic_store, synthetic_manifest) -> None:
    ds = _dataset(synthetic_store, synthetic_manifest, corrupt=StemBleed(0.30), remix=True, augment=True)
    e0 = ds[4]["acc_energy"]
    assert e0.ndim == 0 and float(e0) >= 0.0
    # deterministic per (seed, index); ε- and augmentation-invariant (raw ⟨a⟩-energy).
    assert float(ds[4]["acc_energy"]) == float(ds[4]["acc_energy"])
    clean = _dataset(synthetic_store, synthetic_manifest, corrupt=None, remix=True, augment=True)
    assert float(clean[4]["acc_energy"]) == pytest.approx(float(e0), abs=1e-6)
