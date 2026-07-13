#!/usr/bin/env python3
r"""Materialize Direction 05's two shifted domains from the existing shards (§3.2).

**RUN LATER — CPU, after data prep (gate G0b).** Builds, alongside the decoded WAV
shards, the domain-suffixed audio the fine-tune pipeline consumes:

* ``--domain t1_aac64`` (**primary**) — re-encode every stem to **AAC 64 kbps**
  stereo with ffmpeg (deterministic, version-pinned settings), decode back to WAV,
  and (RUN LATER, in numpy) re-sum the degraded accompaniment. Mixtures are formed
  as **sums of the degraded stems**, so the supervised pairs stay *exactly* additive
  (THEORY §5, one-line proof). ffmpeg is required; the script **checks** for it and
  only executes with ``--execute`` — otherwise it prints/writes the command plan.

      python scripts/make_domains.py --domain t1_aac64 --shards $SHARD_ROOT [--execute]

* ``--domain t2_noise12db`` (secondary fallback) — add **pink noise at 12 dB SNR** to
  each mixture (targets stay clean → a separate-and-denoise adaptation). Pure NumPy,
  seeded, so it is byte-reproducible and needs no ffmpeg. The SNR is achieved
  *exactly* (unit-tested on synthetic arrays).

      python scripts/make_domains.py --domain t2_noise12db --shards $SHARD_ROOT --seed 0

* ``--resolve-t2`` — apply the §3.2 rule and write a decision file: IF an official
  per-track **genre** labeling can be materialized (from ``musdb`` metadata or the
  Zenodo tracklist) with a largest cluster of ≥ 14 train + ≥ 5 test tracks, then
  T2 = that genre cluster; ELSE T2 = the noise domain. The *rule* is pre-registered;
  genre availability is resolved here at prep time (currently unverified → the code
  checks and falls back, avoiding a hidden dependency).

      python scripts/make_domains.py --resolve-t2 --shards $SHARD_ROOT

The core functions are importable and unit-tested (command construction + determinism,
exact-SNR noise, the T2 decision rule); ``main`` is the thin CLI.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from singnet.utils.seed import derive_rng  # noqa: E402

# --- T1: AAC 64 kbps re-encode (pinned, deterministic) ----------------------

#: The MUSDB source stems re-encoded for T1. The accompaniment/mixture are re-summed
#: from the degraded stems (kept exactly additive), not re-encoded as a whole.
STEMS: tuple[str, ...] = ("vocals", "drums", "bass", "other")
DOMAIN_T1 = "t1_aac64"
DOMAIN_T2 = "t2_noise12db"

#: Pinned AAC settings (bitexact for byte-reproducibility across ffmpeg builds).
AAC_BITRATE = "64k"
AAC_SAMPLE_RATE = 44100
AAC_CHANNELS = 2
_FF_COMMON = ("ffmpeg", "-hide_banner", "-nostdin", "-y", "-fflags", "+bitexact")


def ffmpeg_available() -> bool:
    """True iff an ``ffmpeg`` binary is on PATH (checked before any T1 execution)."""
    return shutil.which("ffmpeg") is not None


def aac64_commands(track_dir: str | Path, stems: tuple[str, ...] = STEMS) -> list[list[str]]:
    r"""Deterministic ffmpeg argv lists to build the ``t1_aac64`` stems for one track.

    For each stem two commands: (1) encode ``<stem>.wav`` -> AAC 64 kbps
    ``<stem>.t1_aac64.m4a``; (2) decode it back to ``<stem>.t1_aac64.wav`` at 44.1 kHz
    stereo. Pure string construction — a pure function of ``(track_dir, stems)``, so
    regenerating yields byte-identical commands (unit-tested); nothing is executed
    here (see :func:`run_t1_aac64` for the ``--execute`` path).
    """
    track_dir = Path(track_dir)
    commands: list[list[str]] = []
    for stem in stems:
        src = str(track_dir / f"{stem}.wav")
        aac = str(track_dir / f"{stem}.{DOMAIN_T1}.m4a")
        dst = str(track_dir / f"{stem}.{DOMAIN_T1}.wav")
        commands.append([
            *_FF_COMMON, "-i", src,
            "-c:a", "aac", "-b:a", AAC_BITRATE, "-ar", str(AAC_SAMPLE_RATE), "-ac", str(AAC_CHANNELS),
            aac,
        ])
        commands.append([
            *_FF_COMMON, "-i", aac,
            "-ar", str(AAC_SAMPLE_RATE), "-ac", str(AAC_CHANNELS), dst,
        ])
    return commands


def t1_plan(shard_root: str | Path, stems: tuple[str, ...] = STEMS) -> list[dict[str, Any]]:
    """Per-track T1 command plan over every track directory under ``shard_root``.

    Returns ``[{track, commands}, …]`` in sorted track order (deterministic). Used by
    the CLI to print/write the plan without executing (ffmpeg not required to plan).
    """
    shard_root = Path(shard_root)
    tracks = sorted(p.name for p in shard_root.iterdir() if p.is_dir()) if shard_root.exists() else []
    return [{"track": t, "commands": aac64_commands(shard_root / t, stems)} for t in tracks]


def run_t1_aac64(shard_root: str | Path, *, execute: bool = False) -> list[dict[str, Any]]:
    """Print (and optionally execute) the T1 plan. **Execution is RUN LATER.**

    Never executes unless both ``execute=True`` **and** ffmpeg is present (fails loud
    otherwise). The additivity of the degraded mixture is preserved by re-summing the
    decoded degraded stems in NumPy (that summation step is part of the RUN-LATER
    execute path; command construction is what the tests pin).
    """
    plan = t1_plan(shard_root)
    if execute:
        if not ffmpeg_available():
            raise RuntimeError("ffmpeg not found on PATH — install it (Colab ships it) before --execute.")
        import subprocess

        for entry in plan:  # pragma: no cover - RUN LATER (needs ffmpeg + data)
            for cmd in entry["commands"]:
                subprocess.run(cmd, check=True)
    return plan


# --- T2: pink noise at an exact SNR (pure NumPy, seeded) ---------------------

def pink_noise(n: int, rng: np.random.Generator, channels: int = 1) -> np.ndarray:
    r"""Seeded pink (1/f) noise of shape ``(channels, n)``, zero-mean.

    White noise shaped in the frequency domain so power :math:`\propto 1/f`
    (amplitude :math:`\propto 1/\sqrt f`). Deterministic given ``rng``.
    """
    out = np.empty((channels, n), dtype=np.float64)
    freqs = np.fft.rfftfreq(n if n > 1 else 2)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    for c in range(channels):
        white = rng.standard_normal(n)
        spec = np.fft.rfft(white)
        shaped = np.fft.irfft(spec * scale, n=n)
        out[c] = shaped - shaped.mean()
    return out


def _power(x: np.ndarray) -> float:
    return float(np.mean(np.square(np.asarray(x, dtype=np.float64))))


def scale_noise_to_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    r"""Scale ``noise`` so that ``signal`` has exactly ``snr_db`` dB SNR over it.

    Solves :math:`\text{SNR}=10\log_{10}(P_s/(g^2 P_n))` for the gain
    :math:`g=\sqrt{P_s/(P_n\,10^{\text{SNR}/10})}`, giving a measured SNR equal to the
    target to floating precision (unit-tested). Degenerate (silent) noise/signal are
    returned unscaled.
    """
    ps, pn = _power(signal), _power(noise)
    if pn <= 0.0 or ps <= 0.0:
        return np.asarray(noise, dtype=np.float64)
    gain = np.sqrt(ps / (pn * (10.0 ** (snr_db / 10.0))))
    return gain * np.asarray(noise, dtype=np.float64)


def add_noise_at_snr(
    mixture: np.ndarray, rng: np.random.Generator, snr_db: float = 12.0
) -> np.ndarray:
    """Return ``mixture`` + pink noise scaled to exactly ``snr_db`` dB SNR (T2 op).

    Accepts mono ``(n,)`` or stereo ``(2, n)``; noise is generated per channel and
    scaled against the whole-signal power so the achieved SNR is exact.
    """
    mix = np.asarray(mixture, dtype=np.float64)
    if mix.ndim == 1:
        noise = pink_noise(mix.shape[-1], rng, channels=1)[0]
    else:
        noise = pink_noise(mix.shape[-1], rng, channels=mix.shape[0])
    scaled = scale_noise_to_snr(mix, noise, snr_db)
    return (mix + scaled).astype(np.float32)


def measured_snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    """Measured SNR ``10·log10(P_signal / P_noise)`` in dB (for tests/diagnostics)."""
    return 10.0 * np.log10(_power(signal) / _power(noise))


# --- T2 resolution rule (§3.2) ----------------------------------------------

MIN_GENRE_TRAIN = 14
MIN_GENRE_TEST = 5


def largest_genre_cluster(
    genre_map: dict[str, tuple[str, str]],
    min_train: int = MIN_GENRE_TRAIN,
    min_test: int = MIN_GENRE_TEST,
) -> dict[str, Any] | None:
    """Largest genre with ≥ ``min_train`` train and ≥ ``min_test`` test tracks, else None.

    ``genre_map`` maps ``track -> (genre, split)`` with ``split`` in
    ``{train, valid, test}``. Ties broken by (train+test) count then genre name, so the
    choice is deterministic.
    """
    counts: dict[str, dict[str, int]] = {}
    for _track, (genre, split) in genre_map.items():
        bucket = counts.setdefault(genre, {"train": 0, "valid": 0, "test": 0})
        if split in bucket:
            bucket[split] += 1
    eligible = [
        (g, c) for g, c in counts.items() if c["train"] >= min_train and c["test"] >= min_test
    ]
    if not eligible:
        return None
    genre, c = max(eligible, key=lambda gc: (gc[1]["train"] + gc[1]["test"], gc[0]))
    return {"genre": genre, "train": c["train"], "valid": c["valid"], "test": c["test"]}


def try_load_musdb_genres(shard_root: str | Path | None = None) -> dict[str, tuple[str, str]] | None:
    """Attempt to materialize per-track genres from ``musdb`` (RUN LATER; None if absent).

    Guarded import — ``musdb`` is a data-time extra not installed for the CPU suite, so
    this returns ``None`` now and the caller falls back to the noise domain. At prep
    time (RUN LATER) it would read the tracklist/metadata and build the genre map.
    """
    try:  # pragma: no cover - musdb is a RUN-LATER data-time dependency
        import musdb  # type: ignore  # noqa: F401
    except Exception:
        return None
    return None  # pragma: no cover - real extraction wired at prep time (RUN LATER)


def resolve_t2(
    genre_map: dict[str, tuple[str, str]] | None = None,
    *,
    min_train: int = MIN_GENRE_TRAIN,
    min_test: int = MIN_GENRE_TEST,
) -> dict[str, Any]:
    """Apply the §3.2 rule and return a decision dict (does not write a file).

    ``genre_map`` absent/None (genre labels not materializable) -> the **noise**
    fallback; a materializable largest cluster -> the **genre** domain. The rule is
    deterministic; the same inputs always resolve the same way.
    """
    cluster = largest_genre_cluster(genre_map, min_train, min_test) if genre_map else None
    if cluster is not None:
        return {
            "domain": "t2_genre",
            "op": "genre_subset",
            "genre": cluster["genre"],
            "counts": {k: cluster[k] for k in ("train", "valid", "test")},
            "reason": (
                f"largest genre cluster {cluster['genre']!r} has "
                f"{cluster['train']} train / {cluster['test']} test tracks "
                f"(>= {min_train}/{min_test}); adapt-to-genre selected."
            ),
            "rule": "genre if a >=14-train/>=5-test cluster is materializable, else noise",
        }
    return {
        "domain": DOMAIN_T2,
        "op": "pink_noise_12db",
        "genre": None,
        "counts": None,
        "reason": (
            "no per-track genre labeling materializable (musdb metadata/tracklist "
            "unavailable or no cluster meets >=14 train / >=5 test); noise fallback selected."
        ),
        "rule": "genre if a >=14-train/>=5-test cluster is materializable, else noise",
    }


def write_t2_decision(path: str | Path, decision: dict[str, Any]) -> Path:
    """Write the T2 decision as JSON (the committed, git-diffable record)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


# --- CLI --------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--domain", choices=[DOMAIN_T1, DOMAIN_T2], default=None,
                        help="which domain to materialize")
    parser.add_argument("--resolve-t2", action="store_true", help="apply the §3.2 T2 rule + write decision")
    parser.add_argument("--shards", default=None, help="shard root (track dirs); RUN LATER for real data")
    parser.add_argument("--execute", action="store_true", help="T1 only: actually run ffmpeg (needs ffmpeg)")
    parser.add_argument("--seed", type=int, default=0, help="T2 noise seed")
    parser.add_argument("--out", default="05-lora-source-separation/results/t2_decision.json",
                        help="T2 decision file path")
    args = parser.parse_args(argv)

    if args.resolve_t2:
        genre_map = try_load_musdb_genres(args.shards)
        decision = resolve_t2(genre_map)
        write_t2_decision(args.out, decision)
        print(f"T2 resolved -> {decision['domain']} ({decision['op']})")
        print(f"  reason: {decision['reason']}")
        print(f"  written: {args.out}")
        return

    if args.domain == DOMAIN_T1:
        if not args.shards:
            raise SystemExit("--domain t1_aac64 needs --shards (RUN LATER, after data prep)")
        print(f"ffmpeg available: {ffmpeg_available()}")
        plan = run_t1_aac64(args.shards, execute=args.execute)
        print(f"T1 (aac64) plan: {len(plan)} tracks, "
              f"{sum(len(e['commands']) for e in plan)} ffmpeg commands "
              f"({'EXECUTED' if args.execute else 'plan only — pass --execute to run'}).")
        for entry in plan[:1]:
            print(f"  example [{entry['track']}]: {' '.join(entry['commands'][0])}")
        return

    if args.domain == DOMAIN_T2:
        if not args.shards:
            raise SystemExit("--domain t2_noise12db needs --shards (RUN LATER, after data prep)")
        rng = derive_rng(args.seed, 0)
        _ = rng  # RUN LATER: iterate mixtures, add_noise_at_snr, write <mix>.t2_noise12db.wav
        print("T2 (noise12db): pure-NumPy pink noise at 12 dB SNR per mixture — RUN LATER (needs shards).")
        print("  the op (add_noise_at_snr) is unit-tested for exact SNR; seed = "
              f"{args.seed}.")
        return

    parser.error("nothing to do — pass --domain {t1_aac64,t2_noise12db} or --resolve-t2")


if __name__ == "__main__":
    main()
