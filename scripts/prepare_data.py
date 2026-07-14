#!/usr/bin/env python3
r"""Decode MUSDB18 to WAV shards, index them, and build/verify the split manifest.

**RUN LATER — CPU + network only, ~30-60 min once (MASTER_PLAN §4.3).** Nothing
here runs during the CPU test suite; it needs the MUSDB18 archive on disk.

Typical use (from the run book, §8):

    # 1. decode STEMS -> per-track WAV shards (+ mono mixdowns + index.json)
    python scripts/prepare_data.py --musdb-root $MUSDB_ROOT --out $SHARD_ROOT

    # 2. write/refresh the real split manifest from the decoded folders
    python scripts/prepare_data.py --write-manifest --out $SHARD_ROOT \
        --splits-csv 01-loss-function-study/configs/splits.csv

    # 3. verify shard counts, sample rates, and mixture ≈ sum(stems)
    python scripts/prepare_data.py --verify --out $SHARD_ROOT

Decode is the slow step; do it once, not per session. Expected output ≈ 12-15 GB.
Uses the optional `musdb`/`stempeg` deps (import-guarded); install them from the
data-time section of requirements.txt first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from singnet.data.manifest import Manifest  # noqa: E402

# The 14 verified validation names (sigsep mus.yaml). See splits.csv header.
VALIDATION_TRACKS = [
    "Actions - One Minute Smile",
    "Clara Berry And Wooldog - Waltz For My Victims",
    "Johnny Lokke - Promises & Lies",
    "Patrick Talbot - A Reason To Leave",
    "Triviul - Angelsaint",
    "Alexander Ross - Goodbye Bolero",
    "Fergessen - Nos Palpitants",
    "Leaf - Summerghost",
    "Skelpolu - Human Mistakes",
    "Young Griffo - Pennies",
    "ANiMAL - Rockshow",
    "James May - On The Line",
    "Meaxic - Take A Step",
    "Traffic Experiment - Sirens",
]
STEMS = ("vocals", "drums", "bass", "other")
MIX_TOLERANCE = 1e-3  # max abs error for mixture ≈ sum(stems) on MUSDB18 AAC


def decode(musdb_root: str, out: str, sample_rate: int = 44100) -> None:
    """Decode every MUSDB track to ``<out>/<track>/{stem}.wav`` + mono mixdowns."""
    import musdb  # optional dep
    import soundfile as sf

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    db = musdb.DB(root=musdb_root, is_wav=False)
    if len(db) == 0:
        raise SystemExit(f"no MUSDB tracks found under {musdb_root!r} (RUN LATER: fetch Zenodo 1117372)")

    index: dict[str, dict] = {}
    for track in db:
        track_dir = out_dir / track.name
        track_dir.mkdir(parents=True, exist_ok=True)
        stems = {name: track.targets[name].audio.astype("float32") for name in STEMS}
        mixture = track.audio.astype("float32")
        accompaniment = stems["drums"] + stems["bass"] + stems["other"]
        for name, audio in {**stems, "mixture": mixture, "accompaniment": accompaniment}.items():
            sf.write(track_dir / f"{name}.wav", audio, sample_rate)
        index[track.name] = {
            "n_samples": int(mixture.shape[0]),
            "duration_s": float(mixture.shape[0] / sample_rate),
            "subset": track.subset,  # 'train' or 'test'
        }
        print(f"decoded {track.name} ({index[track.name]['duration_s']:.1f}s)")
    (out_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"wrote {len(index)} tracks + index.json to {out_dir}")


def write_manifest(out: str, splits_csv: str) -> None:
    """Rewrite splits.csv from the decoded folders, failing loud on any mismatch."""
    index_path = Path(out) / "index.json"
    if not index_path.exists():
        raise SystemExit(f"{index_path} missing — run decode first (RUN LATER)")
    index = json.loads(index_path.read_text(encoding="utf-8"))

    train_pool = sorted(t for t, meta in index.items() if meta["subset"] == "train")
    test_pool = sorted(t for t, meta in index.items() if meta["subset"] == "test")

    missing = [t for t in VALIDATION_TRACKS if t not in set(train_pool)]
    if missing:
        raise SystemExit(f"FAIL: validation tracks absent from the train folder: {missing}")
    if len(train_pool) != 100 or len(test_pool) != 50:
        raise SystemExit(f"FAIL: expected 100 train / 50 test, got {len(train_pool)}/{len(test_pool)}")

    import pandas as pd

    rows = [{"track": t, "split": "valid"} for t in VALIDATION_TRACKS]
    rows += [{"track": t, "split": "train"} for t in train_pool if t not in set(VALIDATION_TRACKS)]
    rows += [{"track": t, "split": "test"} for t in test_pool]
    frame = pd.DataFrame(rows)
    Manifest(frame)  # validates columns/splits
    header = (
        "# SingNet split manifest — regenerated from the decoded MUSDB18 folders\n"
        "# by scripts/prepare_data.py --write-manifest. 86 train / 14 valid / 50 test.\n"
        "# The 14 valid names are the verified sigsep mus.yaml validation_tracks.\n"
    )
    Path(splits_csv).write_text(header + frame.to_csv(index=False), encoding="utf-8")
    print(f"wrote real manifest ({len(frame)} tracks) to {splits_csv}")


def write_energy_profiles(
    out: str, sample_rate: int = 44100, window_s: float = 6.0, grid_s: float = 1.0
) -> None:
    """Write per-track windowed vocal-RMS profiles alongside the shards (Direction 08 §5).

    The vocal-activity signal the non-uniform sampling policies steer on: for each decoded
    track, the RMS of its vocal stem over the 6-s chunk window at each 1-s grid start.
    Writes ``<out>/energy_profiles.json`` (+ a sibling ``.csv``). CPU-minutes, RUN LATER —
    needs the decoded shards on disk; fails loud without them.
    """
    import soundfile as sf

    from singnet.data.profiles import windowed_vocal_rms
    from singnet.data.profiles import write_energy_profiles as _write

    out_dir = Path(out)
    track_dirs = sorted(p for p in out_dir.iterdir() if p.is_dir()) if out_dir.exists() else []
    if not track_dirs:
        raise SystemExit(f"no shards under {out_dir} — run decode first (RUN LATER)")

    profiles: dict = {}
    for track_dir in track_dirs:
        voc_path = track_dir / "vocals.wav"
        if not voc_path.exists():
            raise SystemExit(f"FAIL: {track_dir.name} has no vocals.wav — run decode first")
        vocals, sr = sf.read(voc_path, dtype="float32", always_2d=True)
        if sr != sample_rate:
            raise SystemExit(f"FAIL: {track_dir.name} sample rate {sr} != {sample_rate}")
        mono = vocals.mean(axis=1)
        profiles[track_dir.name] = windowed_vocal_rms(mono, sample_rate, window_s, grid_s)

    path = out_dir / "energy_profiles.json"
    _write(path, profiles, sr=sample_rate, window_s=window_s, grid_s=grid_s)
    total = sum(len(p) for p in profiles.values())
    print(f"wrote {len(profiles)} track profiles ({total} grid windows) to {path} (+ .csv)")


def verify(out: str, sample_rate: int = 44100) -> None:
    """Re-check shard counts, sample rates, and mixture ≈ sum(stems)."""
    import numpy as np
    import soundfile as sf

    out_dir = Path(out)
    track_dirs = [p for p in out_dir.iterdir() if p.is_dir()]
    if not track_dirs:
        raise SystemExit(f"no shards under {out_dir} — run decode first (RUN LATER)")
    max_err = 0.0
    for track_dir in track_dirs:
        mixture, sr = sf.read(track_dir / "mixture.wav", dtype="float32")
        if sr != sample_rate:
            raise SystemExit(f"FAIL: {track_dir.name} sample rate {sr} != {sample_rate}")
        recon = sum(sf.read(track_dir / f"{s}.wav", dtype="float32")[0] for s in STEMS)
        err = float(np.max(np.abs(mixture - recon)))
        max_err = max(max_err, err)
    status = "OK" if max_err < MIX_TOLERANCE else "FAIL"
    print(f"{status}: {len(track_dirs)} tracks; max |mixture - sum(stems)| = {max_err:.2e} (tol {MIX_TOLERANCE})")
    if max_err >= MIX_TOLERANCE:
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--musdb-root", help="MUSDB18 archive root (Zenodo 1117372, decoded from STEMS)")
    parser.add_argument("--out", required=True, help="output shard root (SHARD_ROOT)")
    parser.add_argument("--splits-csv", default="01-loss-function-study/configs/splits.csv")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--verify", action="store_true", help="verify existing shards (no decode)")
    parser.add_argument("--write-manifest", action="store_true", help="rewrite splits.csv from decoded folders")
    parser.add_argument("--write-energy-profiles", action="store_true",
                        help="Direction 08: per-track vocal-RMS profiles (6-s window, 1-s grid)")
    args = parser.parse_args(argv)

    if args.verify:
        verify(args.out, args.sample_rate)
    elif args.write_manifest:
        write_manifest(args.out, args.splits_csv)
    elif args.write_energy_profiles:
        write_energy_profiles(args.out, args.sample_rate)
    else:
        if not args.musdb_root:
            parser.error("decoding requires --musdb-root")
        decode(args.musdb_root, args.out, args.sample_rate)


if __name__ == "__main__":
    main()
