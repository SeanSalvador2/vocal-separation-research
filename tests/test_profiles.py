"""Energy-profile computation — exact windowed vocal RMS + JSON round-trip (G0, §5)."""

from __future__ import annotations

import numpy as np

from singnet.data import (
    InMemoryStore,
    compute_energy_profiles,
    load_energy_profiles,
    windowed_vocal_rms,
    write_energy_profiles,
)


def test_windowed_vocal_rms_constant_signal() -> None:
    sr = 10
    vocals = np.full(30, 0.5, dtype=np.float64)  # constant amplitude everywhere
    prof = windowed_vocal_rms(vocals, sr, window_s=1.0, grid_s=0.5)  # win=10, hop=5
    # windows start at 0,5,10,15,20 (5 windows), every one all-0.5 -> RMS 0.5.
    assert len(prof) == 5
    assert np.allclose(prof, 0.5)


def test_windowed_vocal_rms_step_signal_exact() -> None:
    sr = 10
    vocals = np.concatenate([np.full(15, 0.5), np.zeros(15)]).astype(np.float64)
    prof = windowed_vocal_rms(vocals, sr, window_s=1.0, grid_s=0.5)  # win=10, hop=5
    # start 0: [0:10] all 0.5 -> 0.5; start 5: [5:15] all 0.5 -> 0.5;
    # start 10: [10:20] = five 0.5 + five 0 -> sqrt(0.125); start 15,20: all 0 -> 0.
    expected = [0.5, 0.5, float(np.sqrt(0.125)), 0.0, 0.0]
    assert np.allclose(prof, expected)


def test_windowed_vocal_rms_track_shorter_than_window_is_empty() -> None:
    assert windowed_vocal_rms(np.zeros(5), sr=10, window_s=1.0).size == 0


def test_compute_and_roundtrip_profiles(tmp_path) -> None:
    sr = 10
    store = InMemoryStore(
        {
            "a": {"vocals": np.full(30, 0.5, dtype=np.float64), "accompaniment": np.zeros(30)},
            "b": {"vocals": np.zeros(30, dtype=np.float64), "accompaniment": np.zeros(30)},
        },
        sample_rate=sr,
    )
    profiles = compute_energy_profiles(store, ["a", "b"], window_s=1.0, grid_s=0.5)
    assert np.allclose(profiles["a"], 0.5) and np.allclose(profiles["b"], 0.0)

    path = tmp_path / "energy_profiles.json"
    write_energy_profiles(path, profiles, sr=sr, window_s=1.0, grid_s=0.5)
    assert path.exists() and path.with_suffix(".csv").exists()
    loaded = load_energy_profiles(path)
    assert set(loaded) == {"a", "b"}
    assert np.allclose(loaded["a"], profiles["a"]) and np.allclose(loaded["b"], profiles["b"])


def test_load_missing_profiles_is_loud(tmp_path) -> None:
    try:
        load_energy_profiles(tmp_path / "nope.json")
        raised = False
    except FileNotFoundError as exc:
        raised = "write-energy-profiles" in str(exc)
    assert raised
