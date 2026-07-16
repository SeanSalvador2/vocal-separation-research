"""Domain tooling: T1 ffmpeg command construction, T2 exact-SNR noise, T2 rule (G0).

Direction-05 stage-D domain items (MASTER_PLAN §3.2, §8): the T1 re-encode commands
are deterministic and carry the pinned AAC settings; the T2 pink-noise op hits the
12 dB SNR **exactly** (the tested invariant, since ffmpeg is unavailable here); and the
§3.2 resolution rule (genre if materializable, else noise) is deterministic and writes
a decision file. The core functions live in ``scripts/make_domains.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import make_domains as md  # noqa: E402


# --- T1: AAC 64 kbps re-encode command construction -------------------------

def test_t1_commands_count_and_pinned_settings() -> None:
    cmds = md.aac64_commands("/shards/trackA")
    assert len(cmds) == 2 * len(md.STEMS)  # encode + decode per stem
    encode = cmds[0]
    assert encode[0] == "ffmpeg"
    for flag in ("-c:a", "aac", "-b:a", "64k", "-ar", "44100", "-ac", "2", "+bitexact"):
        assert flag in encode
    # the decode step reads the .m4a and writes the domain-suffixed wav.
    decode = cmds[1]
    assert any(a.endswith("vocals.t1_aac64.m4a") for a in decode)
    assert any(a.endswith("vocals.t1_aac64.wav") for a in decode)


def test_t1_commands_are_deterministic() -> None:
    assert md.aac64_commands("/shards/trackA") == md.aac64_commands("/shards/trackA")
    assert md.aac64_commands("/shards/trackA") != md.aac64_commands("/shards/trackB")


def test_t1_plan_over_shard_dirs(tmp_path) -> None:
    for t in ("song1", "song2"):
        (tmp_path / t).mkdir()
    plan = md.t1_plan(tmp_path)
    assert [e["track"] for e in plan] == ["song1", "song2"]  # sorted, deterministic
    assert all(len(e["commands"]) == 2 * len(md.STEMS) for e in plan)


def test_ffmpeg_available_returns_bool() -> None:
    assert isinstance(md.ffmpeg_available(), bool)


def test_run_t1_execute_requires_ffmpeg(tmp_path, monkeypatch) -> None:
    (tmp_path / "song1").mkdir()
    monkeypatch.setattr(md, "ffmpeg_available", lambda: False)
    # planning never needs ffmpeg...
    assert md.run_t1_aac64(tmp_path, execute=False)
    # ...but executing without it fails loud.
    try:
        md.run_t1_aac64(tmp_path, execute=True)
        raised = False
    except RuntimeError:
        raised = True
    assert raised


# --- T2: exact-SNR pink noise ----------------------------------------------

def test_pink_noise_is_seeded_and_zero_mean() -> None:
    a = md.pink_noise(4096, np.random.default_rng(0))
    b = md.pink_noise(4096, np.random.default_rng(0))
    assert np.array_equal(a, b)                       # deterministic given the rng seed
    assert not np.array_equal(a, md.pink_noise(4096, np.random.default_rng(1)))
    assert abs(float(a.mean())) < 1e-6


def test_scale_noise_hits_exact_snr() -> None:
    sig = (0.2 * np.sin(2 * np.pi * 220 * np.arange(44100) / 44100)).astype(np.float32)
    for target in (12.0, 6.0, 20.0):
        noise = md.pink_noise(len(sig), np.random.default_rng(3))[0]
        scaled = md.scale_noise_to_snr(sig, noise, target)
        assert abs(md.measured_snr_db(sig, scaled) - target) < 1e-6


def test_add_noise_at_snr_mono_and_stereo() -> None:
    sig = (0.2 * np.sin(2 * np.pi * 220 * np.arange(22050) / 22050)).astype(np.float32)
    noisy = md.add_noise_at_snr(sig, np.random.default_rng(4), snr_db=12.0)
    assert abs(md.measured_snr_db(sig, noisy.astype(np.float64) - sig) - 12.0) < 1e-6
    stereo = np.stack([sig, 0.9 * sig])
    noisy2 = md.add_noise_at_snr(stereo, np.random.default_rng(5), snr_db=12.0)
    assert noisy2.shape == stereo.shape and noisy2.dtype == np.float32
    assert abs(md.measured_snr_db(stereo, noisy2.astype(np.float64) - stereo) - 12.0) < 1e-6


# --- T2 resolution rule (§3.2) ---------------------------------------------

def test_resolve_t2_noise_fallback_when_no_genres() -> None:
    dec = md.resolve_t2(None)
    assert dec["domain"] == "t2_noise12db" and dec["op"] == "pink_noise_12db"
    assert md.resolve_t2(None) == dec  # deterministic


def test_resolve_t2_genre_when_materializable() -> None:
    gm = {}
    for i in range(16):
        gm[f"tr{i}"] = ("rock", "train")
    for i in range(6):
        gm[f"te{i}"] = ("rock", "test")
    for i in range(3):
        gm[f"po{i}"] = ("pop", "train")
    dec = md.resolve_t2(gm)
    assert dec["domain"] == "t2_genre" and dec["genre"] == "rock"
    assert dec["counts"] == {"train": 16, "valid": 0, "test": 6}


def test_resolve_t2_insufficient_cluster_falls_back_to_noise() -> None:
    gm = {f"tr{i}": ("jazz", "train") for i in range(10)}  # only 10 train, 0 test
    assert md.resolve_t2(gm)["domain"] == "t2_noise12db"


def test_largest_genre_cluster_thresholds_and_ties() -> None:
    assert md.largest_genre_cluster({}) is None
    gm = {}
    for g, (ntr, nte) in {"a": (14, 5), "b": (20, 8)}.items():
        for i in range(ntr):
            gm[f"{g}_tr{i}"] = (g, "train")
        for i in range(nte):
            gm[f"{g}_te{i}"] = (g, "test")
    assert md.largest_genre_cluster(gm)["genre"] == "b"  # the bigger eligible cluster


def test_write_t2_decision(tmp_path) -> None:
    dec = md.resolve_t2(None)
    path = md.write_t2_decision(tmp_path / "t2_decision.json", dec)
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["domain"] == "t2_noise12db" and "rule" in loaded
