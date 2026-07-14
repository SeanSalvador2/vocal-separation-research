"""Teacher labeling post-processing: 2-stem consistency, residual, activity screen (G0, §3.2).

Direction-10 stage-D items (MASTER_PLAN §3.2, §8): ``accompaniment = mixture - vocals`` is
**exact**; the raw 4-stem sum-residual is computed in dB (recorded, not used); the vocal
activity ratio comes from Direction 08's windowed-RMS profile; the activity screen keeps
``ratio ≥ 0.20`` with the pre-registered one-shot 0.10 fallback; the provenance JSON
round-trips. The demucs invocation itself is RUN LATER (lazy import) — never executed here.
The core lives in ``scripts/teacher_label.py`` and is pure/synthetic-array-tested.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import teacher_label as tl  # noqa: E402

SR = 44100


# --- 2-stem consistency (â = x - v̂ exact) ----------------------------------

def test_two_stem_consistency_is_exact() -> None:
    rng = np.random.default_rng(0)
    mixture = rng.standard_normal(4096).astype(np.float32)
    teacher_vocals = 0.4 * rng.standard_normal(4096).astype(np.float32)
    out = tl.two_stem_consistency(mixture, teacher_vocals)
    assert np.allclose(out["vocals"], teacher_vocals, atol=1e-6)
    # vocals + accompaniment == mixture EXACTLY (float32 round-off only)
    assert np.allclose(out["vocals"] + out["accompaniment"], mixture, atol=1e-5)
    assert np.allclose(out["accompaniment"], mixture - teacher_vocals, atol=1e-6)


def test_two_stem_consistency_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        tl.two_stem_consistency(np.zeros(10, np.float32), np.zeros(8, np.float32))


# --- raw 4-stem sum residual in dB (recorded, not used) ---------------------

def test_consistency_residual_zero_when_stems_sum_to_mixture() -> None:
    rng = np.random.default_rng(1)
    drums = rng.standard_normal(2048)
    bass = rng.standard_normal(2048)
    other = rng.standard_normal(2048)
    vocals = rng.standard_normal(2048)
    mixture = drums + bass + other + vocals  # perfectly consistent
    stems = {"drums": drums, "bass": bass, "other": other, "vocals": vocals}
    db = tl.consistency_residual_db(stems, mixture)
    assert db < -100.0  # essentially the eps floor: residual ~ 0


def test_consistency_residual_grows_with_inconsistency() -> None:
    rng = np.random.default_rng(2)
    mixture = rng.standard_normal(2048)
    # stems that do NOT sum to the mixture -> a large (near 0 dB) residual ratio.
    stems = {"drums": np.zeros(2048), "bass": np.zeros(2048),
             "other": np.zeros(2048), "vocals": 0.5 * mixture}
    db = tl.consistency_residual_db(stems, mixture)
    assert -8.0 < db < 1.0  # residual is a large fraction of the mixture energy


def test_consistency_residual_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        tl.consistency_residual_db({"vocals": np.zeros(4)}, np.zeros(8))


# --- vocal activity ratio (Direction-08 profile machinery) ------------------

def test_activity_ratio_all_active() -> None:
    # a loud, sustained tone -> every 6-s window is active -> ratio 1.0
    t = np.arange(int(12 * SR)) / SR
    loud = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    assert tl.vocal_activity_ratio(loud, SR) == pytest.approx(1.0)


def test_activity_ratio_silent_is_zero() -> None:
    silent = np.zeros(int(12 * SR), dtype=np.float32)
    assert tl.vocal_activity_ratio(silent, SR) == 0.0


def test_activity_ratio_partial() -> None:
    # first half loud, second half silent -> a fraction (not 0, not 1) of active windows.
    t = np.arange(int(10 * SR)) / SR
    loud = 0.3 * np.sin(2 * np.pi * 220 * t)
    sig = np.concatenate([loud, np.zeros(int(10 * SR))]).astype(np.float32)
    ratio = tl.vocal_activity_ratio(sig, SR)
    assert 0.0 < ratio < 1.0


def test_activity_ratio_clip_shorter_than_window_is_zero() -> None:
    assert tl.vocal_activity_ratio(np.ones(SR, np.float32), SR) == 0.0  # < one 6-s window


# --- the activity screen (20% keep, 10% pre-registered fallback) ------------

def test_screen_keeps_above_threshold_in_order() -> None:
    # keep == the survivor count at 0.20, so no fallback: exactly the >= 0.20 clips, in order.
    ratios = {"a": 0.5, "b": 0.1, "c": 0.25, "d": 0.05, "e": 0.9}
    res = tl.screen_by_activity(ratios, keep=3, threshold=0.20)
    assert res.kept == ["a", "c", "e"]          # >= 0.20, insertion order (0.1/0.05 excluded)
    assert res.used_threshold == 0.20 and not res.fell_back
    assert res.n_survivors == 3


def test_screen_falls_back_to_ten_percent_once() -> None:
    # only 1 clip clears 0.20 but we need 3 -> fallback to 0.10 pulls in the 0.12/0.15 clips.
    ratios = {"a": 0.5, "b": 0.12, "c": 0.15, "d": 0.05}
    res = tl.screen_by_activity(ratios, keep=3, threshold=0.20, fallback=0.10)
    assert res.fell_back is True and res.used_threshold == 0.10
    assert res.kept == ["a", "b", "c"]          # 0.05 still excluded
    assert res.exhausted is False


def test_screen_exhausted_uses_all_survivors_and_records_count() -> None:
    ratios = {"a": 0.5, "b": 0.02, "c": 0.03}   # only 1 survives even at 0.10
    res = tl.screen_by_activity(ratios, keep=800, threshold=0.20, fallback=0.10)
    assert res.fell_back is True and res.exhausted is True
    assert res.kept == ["a"] and res.n_survivors == 1


def test_screen_takes_first_keep_in_order() -> None:
    ratios = {f"t{i}": 0.9 for i in range(10)}
    res = tl.screen_by_activity(ratios, keep=4, threshold=0.20)
    assert res.kept == ["t0", "t1", "t2", "t3"]  # deterministic first-N order
    assert res.exhausted is False


# --- provenance writer ------------------------------------------------------

def test_write_provenance_roundtrip(tmp_path) -> None:
    prov = tl.TeacherProvenance(
        demucs_version="4.0.1", model="htdemucs",
        settings={"sample_rate": 44100, "two_stem": "vocals; accompaniment = mixture - vocals"},
        n_labeled=1200, n_kept=800, activity_threshold=0.20, used_threshold=0.20,
        consistency_residual_db_mean=-42.3,
    )
    path = tmp_path / "sub" / "provenance.json"
    tl.write_provenance(path, prov)
    payload = json.loads(path.read_text())
    assert payload["model"] == "htdemucs" and payload["demucs_version"] == "4.0.1"
    assert payload["n_kept"] == 800 and payload["used_threshold"] == 0.20
    assert payload["settings"]["two_stem"].startswith("vocals")
    assert abs(payload["consistency_residual_db_mean"] + 42.3) < 1e-9


# --- the demucs invocation is lazy: importing the module needs no heavy deps -

def test_module_imports_without_demucs_or_torch() -> None:
    # The module is import-clean (demucs/torch imports live inside RUN-LATER functions).
    assert hasattr(tl, "load_teacher") and hasattr(tl, "run")
    src = Path(tl.__file__).read_text()
    assert "def load_teacher" in src
    # the demucs import is inside a function body, never at module top level.
    top_level = [ln for ln in src.splitlines() if ln.startswith("import ") or ln.startswith("from ")]
    assert not any("demucs" in ln for ln in top_level)
