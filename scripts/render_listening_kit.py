#!/usr/bin/env python3
r"""Render the blind listening-check kit (MASTER_PLAN §7.5).

**RUN LATER — CPU, human time, zero GPU.** For each selected test track, find the
loudest 12 s vocal window and render **isolated vocals** and **karaoke**
(mixture − vocals) for the 3 full-budget systems, level-matched to −23 LUFS, into
a per-track folder ready for a blind A/B session.

    python scripts/render_listening_kit.py \
        --checkpoints checkpoints/A/best.pt checkpoints/B/best.pt checkpoints/C/best.pt \
        --labels l1 l1mrstft sisdr \
        --shard-root $SHARD_ROOT --splits-csv 01-loss-function-study/configs/splits.csv \
        --out 01-loss-function-study/results/listening_kit

Track selection: ``--tracks`` if given, else the 5 median-vocals-SI-SDR test
tracks read from ``results/test_per_track.csv`` (fixed before listening, §7.5).
Level matching uses ``pyloudnorm`` if installed, else an RMS fallback (a warning
is printed). No audio is redistributed beyond the rating session (license, §4.1).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

WINDOW_S = 12.0
TARGET_LUFS = -23.0


def loudest_vocal_window(vocals: np.ndarray, sr: int, window_s: float = WINDOW_S) -> tuple[int, int]:
    """Return (start, end) sample indices of the highest-energy vocal window."""
    win = int(window_s * sr)
    if len(vocals) <= win:
        return 0, len(vocals)
    energy = np.convolve(vocals.astype(np.float64) ** 2, np.ones(win), mode="valid")
    start = int(np.argmax(energy))
    return start, start + win


def match_loudness(audio: np.ndarray, sr: int, target_lufs: float = TARGET_LUFS) -> np.ndarray:
    """Normalise to target LUFS (pyloudnorm) or an RMS approximation (fallback)."""
    try:
        import pyloudnorm as pyln  # optional

        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio)
        return pyln.normalize.loudness(audio, loudness, target_lufs)
    except ImportError:
        print("  [warn] pyloudnorm not installed — using RMS level-match approximation")
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) + 1e-12
        target_rms = 10 ** ((target_lufs) / 20.0)  # rough amplitude proxy
        return (audio * (target_rms / rms)).astype(np.float32)


def select_tracks(per_track_csv: Path, n: int = 5) -> list[str]:
    """The n median-vocals-SI-SDR test tracks (fixed before listening)."""
    import pandas as pd

    frame = pd.read_csv(per_track_csv)
    model = frame[frame["system"] == "singnet"].sort_values("sisdr_vocals")
    if model.empty:
        raise SystemExit(f"no 'singnet' rows in {per_track_csv} — run the test pass first")
    mid = len(model) // 2
    lo = max(0, mid - n // 2)
    return model.iloc[lo : lo + n]["track"].tolist()


def main(argv: list[str] | None = None) -> None:
    import soundfile as sf
    import torch

    from singnet.audio import STFT
    from singnet.eval import separate_track
    from singnet.models import build_model
    from singnet.train.loop import load_checkpoint

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="3 full-budget checkpoints")
    parser.add_argument("--labels", nargs="+", required=True, help="system labels (align with checkpoints)")
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--splits-csv", required=True)
    parser.add_argument("--out", default="01-loss-function-study/results/listening_kit")
    parser.add_argument("--tracks", nargs="*", default=None, help="explicit track names (else median-5)")
    parser.add_argument("--per-track-csv", default="01-loss-function-study/results/test_per_track.csv")
    parser.add_argument("--sample-rate", type=int, default=44100)
    args = parser.parse_args(argv)

    if len(args.checkpoints) != len(args.labels):
        parser.error("--checkpoints and --labels must have equal length")

    from singnet.data.musdb_dataset import WavShardStore

    store = WavShardStore(args.shard_root, sample_rate=args.sample_rate)
    if not Path(args.shard_root).exists():
        raise SystemExit(f"shard_root {args.shard_root} missing — run prepare_data.py (RUN LATER)")
    tracks = args.tracks or select_tracks(Path(args.per_track_csv))
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    stft = STFT()
    models = []
    for ckpt in args.checkpoints:
        model = build_model()
        load_checkpoint(ckpt, model=model, restore_rng=False)
        model.eval()
        models.append(model)

    for track in tracks:
        sources = store.load_sources(track)
        vocals, accompaniment = sources["vocals"], sources["accompaniment"]
        mixture = vocals + accompaniment
        start, end = loudest_vocal_window(vocals, args.sample_rate)
        clip_mix = mixture[start:end]
        track_dir = out_root / track.replace(" ", "_")
        track_dir.mkdir(parents=True, exist_ok=True)
        sf.write(track_dir / "reference_vocals.wav",
                 match_loudness(vocals[start:end], args.sample_rate), args.sample_rate)
        for label, model in zip(args.labels, models):
            voc_hat = separate_track(model, torch.from_numpy(clip_mix).float(), stft).cpu().numpy()
            karaoke = clip_mix - voc_hat
            sf.write(track_dir / f"{label}_vocals.wav", match_loudness(voc_hat, args.sample_rate), args.sample_rate)
            sf.write(track_dir / f"{label}_karaoke.wav", match_loudness(karaoke, args.sample_rate), args.sample_rate)
        print(f"rendered {track} -> {track_dir}")


if __name__ == "__main__":
    main()
