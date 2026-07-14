"""Per-track windowed vocal-energy profiles (MASTER_PLAN §4.1, §5).

A profile is the vocal-activity signal the non-uniform sampling policies
(:mod:`singnet.data.sampling`) steer on: the RMS of the **ground-truth vocal stem**
over the 6-s chunk window starting at each 1-s grid point. Profiles are a *prep-time*
artifact — written once by ``scripts/prepare_data.py --write-energy-profiles``
alongside the WAV shards (CPU-minutes, RUN LATER) — because they depend only on the
stems, never on the run. The ``uniform`` policy (and every Direction 01–06 config)
needs no profile at all; a non-uniform policy without one is a **loud** error telling
the user to run the profile pass.

The core :func:`windowed_vocal_rms` is pure and unit-tested on constructed stems to
exact values; :func:`load_energy_profiles` reads the JSON the prep pass writes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_WINDOW_S = 6.0  #: chunk window the profile RMS is taken over (the 6-s chunk)
DEFAULT_GRID_S = 1.0    #: spacing of candidate chunk starts (the 1-s start grid)


def windowed_vocal_rms(
    vocals: np.ndarray,
    sr: int,
    window_s: float = DEFAULT_WINDOW_S,
    grid_s: float = DEFAULT_GRID_S,
) -> np.ndarray:
    r"""RMS of ``vocals`` over each ``window_s`` window on a ``grid_s`` start grid.

    Window ``g`` covers samples ``[g·hop, g·hop + win)`` with ``hop = grid_s·sr`` and
    ``win = window_s·sr``; the profile has one value per window that fully fits in the
    track. A track shorter than one window yields an **empty** profile (the sampler
    then reports the profile is missing/degenerate). Returns float64, shape ``(N,)``.
    """
    wave = np.asarray(vocals, dtype=np.float64).reshape(-1)
    n = wave.size
    win = max(1, int(round(window_s * sr)))
    hop = max(1, int(round(grid_s * sr)))
    if n < win:
        return np.empty(0, dtype=np.float64)
    starts = range(0, n - win + 1, hop)
    return np.array([float(np.sqrt(np.mean(wave[s : s + win] ** 2))) for s in starts])


def compute_energy_profiles(
    store,
    tracks: list[str],
    window_s: float = DEFAULT_WINDOW_S,
    grid_s: float = DEFAULT_GRID_S,
) -> dict[str, np.ndarray]:
    """Windowed vocal RMS for each track in ``tracks`` (reads each stem via ``store``)."""
    return {
        t: windowed_vocal_rms(store.load_sources(t)["vocals"], store.sample_rate, window_s, grid_s)
        for t in tracks
    }


def write_energy_profiles(
    path: str | Path,
    profiles: dict[str, np.ndarray],
    sr: int,
    window_s: float = DEFAULT_WINDOW_S,
    grid_s: float = DEFAULT_GRID_S,
) -> None:
    """Write ``<path>`` (JSON) + a sibling ``<path>.csv`` long-form table.

    JSON carries a ``meta`` block (sr / window / grid) and per-track RMS lists; the
    CSV has ``track, grid_index, start_s, vocal_rms`` rows for spreadsheet spot-checks.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {"sr": int(sr), "window_s": float(window_s), "grid_s": float(grid_s)},
        "profiles": {t: [float(x) for x in prof] for t, prof in profiles.items()},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    import pandas as pd

    rows = [
        {"track": t, "grid_index": g, "start_s": g * grid_s, "vocal_rms": float(prof[g])}
        for t, prof in profiles.items()
        for g in range(len(prof))
    ]
    pd.DataFrame(rows, columns=["track", "grid_index", "start_s", "vocal_rms"]).to_csv(
        path.with_suffix(".csv"), index=False
    )


def load_energy_profiles(path: str | Path) -> dict[str, np.ndarray]:
    """Read the JSON written by :func:`write_energy_profiles` into ``{track: rms[]}``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"energy-profile file {path} missing — run "
            "`scripts/prepare_data.py --write-energy-profiles --out <shard_root>` first (RUN LATER)."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {t: np.asarray(vals, dtype=np.float64) for t, vals in payload["profiles"].items()}
