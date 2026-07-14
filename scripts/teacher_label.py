#!/usr/bin/env python3
r"""Teacher (htdemucs) labeling + 2-stem consistency + activity screen (MASTER_PLAN §3.2).

**RUN LATER — GPU, ~2-4 h once. Never executed in the test suite.** Runs the teacher over
the screened FMA clips, derives the 2-stem pseudo-labels the student trains on, screens by
teacher-vocal activity, and writes a provenance file. The demucs invocation is **constructed
here but lazily imported inside a function and RUN-LATER-guarded** — importing this module
pulls in no heavy deps. The post-processing functions are pure and unit-tested on synthetic
arrays (``tests/test_teacher_label.py``); the orchestration (:func:`run`) is RUN LATER.

Pinned choices (MASTER_PLAN §3.2):

* **2-stem consistency by construction:** ``vocals = teacher_vocals``; ``accompaniment :=
  mixture - teacher_vocals`` (exact additivity, mirroring the student's task). The teacher's
  raw 4-stem sum-residual is **measured and recorded** (:func:`consistency_residual_db`) but
  **not used**.
* **Vocal-activity screen (Direction-08 machinery):** the teacher vocals' windowed RMS
  profile (:func:`singnet.data.profiles.windowed_vocal_rms`) gives a per-clip activity ratio;
  keep clips with ratio ``>= 0.20``; take the first ``N_train = 800`` in deterministic order.
  **Pre-registered fallback:** if fewer than 800 survive, lower the threshold to ``0.10``
  once; if still short, use all survivors and record the count.

Run book (§6; RUN LATER)::

    python scripts/teacher_label.py --manifest $PSEUDO_ROOT/manifest.csv \
           --keep 800 --activity-threshold 0.20 --out $PSEUDO_ROOT
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from singnet.data.profiles import DEFAULT_GRID_S, DEFAULT_WINDOW_S, windowed_vocal_rms  # noqa: E402

#: Project-wide "silent" amplitude threshold in dBFS (== sisdr guard / SLR θ); a profile
#: window counts as *active* when its vocal RMS is at or above it (MASTER_PLAN §3.2).
DEFAULT_ACTIVITY_THETA_DB = -60.0
#: Pre-registered activity-ratio keep threshold and its one-shot fallback (§3.2).
DEFAULT_ACTIVITY_THRESHOLD = 0.20
FALLBACK_ACTIVITY_THRESHOLD = 0.10
DEFAULT_KEEP = 800
DEFAULT_TEACHER_MODEL = "htdemucs"
EPS = 1e-8


# --- 2-stem consistency (pure, unit-tested) ---------------------------------

def two_stem_consistency(mixture: np.ndarray, teacher_vocals: np.ndarray) -> dict[str, np.ndarray]:
    r"""Derive the exact 2-stem pseudo-labels: ``vocals = v̂``, ``accompaniment = x - v̂``.

    Mirrors the student's own additive task (MASTER_PLAN §3.2): the accompaniment target is
    defined as the residual so that ``vocals + accompaniment == mixture`` **exactly** (float
    round-off only), whatever the teacher output. Computed in float64 then cast, so the
    additivity residual is at float32 round-off, not accumulated.
    """
    mix = np.asarray(mixture, dtype=np.float64).reshape(-1)
    voc = np.asarray(teacher_vocals, dtype=np.float64).reshape(-1)
    if mix.shape != voc.shape:
        raise ValueError(f"mixture {mix.shape} and teacher_vocals {voc.shape} must match")
    accompaniment = (mix - voc).astype(np.float32)
    return {"vocals": voc.astype(np.float32), "accompaniment": accompaniment}


def consistency_residual_db(
    teacher_stems: Mapping[str, np.ndarray], mixture: np.ndarray, eps: float = EPS
) -> float:
    r"""Raw 4-stem sum-residual in dB: ``10·log10(‖x - Σ ŝ‖² / ‖x‖²)`` (recorded, not used).

    The teacher (htdemucs) emits four stems whose sum need not equal the mixture; this
    measures that inconsistency as an energy ratio in dB (MASTER_PLAN §3.2, §11) — a per-clip
    provenance number, **not** a training target. Very negative = the teacher's stems sum back
    to the mixture; near 0 = a large residual. Lower (more negative) is more self-consistent.
    """
    mix = np.asarray(mixture, dtype=np.float64).reshape(-1)
    total = np.zeros_like(mix)
    for name, stem in teacher_stems.items():
        arr = np.asarray(stem, dtype=np.float64).reshape(-1)
        if arr.shape != mix.shape:
            raise ValueError(f"stem {name!r} shape {arr.shape} != mixture {mix.shape}")
        total = total + arr
    residual = mix - total
    num = float(np.sum(residual**2))
    den = float(np.sum(mix**2))
    return float(10.0 * np.log10((num + eps) / (den + eps)))


# --- activity screen (Direction-08 profile machinery; pure, unit-tested) ----

def vocal_activity_ratio(
    teacher_vocals: np.ndarray,
    sr: int,
    theta_db: float = DEFAULT_ACTIVITY_THETA_DB,
    window_s: float = DEFAULT_WINDOW_S,
    grid_s: float = DEFAULT_GRID_S,
) -> float:
    r"""Fraction of the teacher-vocals' windowed-RMS profile that is *active* (RMS ≥ θ).

    Uses Direction 08's :func:`windowed_vocal_rms` (6-s window on a 1-s grid) on the teacher
    vocals, then reports the share of windows at or above ``10**(theta_db/20)`` — the clip's
    **activity ratio** in ``[0, 1]`` (MASTER_PLAN §3.2). A clip shorter than one window
    (empty profile) has ratio ``0.0``.
    """
    profile = windowed_vocal_rms(teacher_vocals, sr, window_s, grid_s)
    if profile.size == 0:
        return 0.0
    thresh = 10.0 ** (float(theta_db) / 20.0)
    return float(np.mean(profile >= thresh))


@dataclass
class ActivityScreenResult:
    """Outcome of the activity screen (MASTER_PLAN §3.2, gate G1)."""

    kept: list[str]
    used_threshold: float
    n_survivors: int
    fell_back: bool
    exhausted: bool  #: True iff fewer than ``keep`` survived even after the fallback


def screen_by_activity(
    ratios: Mapping[str, float],
    keep: int = DEFAULT_KEEP,
    threshold: float = DEFAULT_ACTIVITY_THRESHOLD,
    fallback: float = FALLBACK_ACTIVITY_THRESHOLD,
) -> ActivityScreenResult:
    r"""Keep clips whose activity ratio ``≥ threshold``; the pre-registered 10 % fallback (§3.2).

    Iterates ``ratios`` in insertion order (deterministic). Survivors are those at or above
    ``threshold``; if fewer than ``keep`` survive, the threshold is lowered to ``fallback``
    **once** and survivors recomputed. The first ``keep`` survivors are returned; ``exhausted``
    flags the "still short after fallback — use all survivors and record the count" branch.
    No RNG: a pure function of the ratios and the thresholds.
    """
    items = list(ratios.items())
    survivors = [tid for tid, r in items if float(r) >= threshold]
    used, fell_back = float(threshold), False
    if len(survivors) < keep:
        survivors = [tid for tid, r in items if float(r) >= fallback]
        used, fell_back = float(fallback), True
    kept = survivors[: int(keep)]
    return ActivityScreenResult(
        kept=kept,
        used_threshold=used,
        n_survivors=len(survivors),
        fell_back=fell_back,
        exhausted=len(survivors) < int(keep),
    )


# --- provenance writer (pure, unit-tested) ----------------------------------

@dataclass
class TeacherProvenance:
    """The committed teacher-labeling provenance record (MASTER_PLAN §3.2, §7 G1)."""

    demucs_version: str
    model: str
    settings: dict[str, object] = field(default_factory=dict)
    n_labeled: int = 0
    n_kept: int = 0
    activity_threshold: float = DEFAULT_ACTIVITY_THRESHOLD
    used_threshold: float = DEFAULT_ACTIVITY_THRESHOLD
    consistency_residual_db_mean: float = float("nan")


def write_provenance(path: str | Path, provenance: TeacherProvenance) -> None:
    """Write the teacher provenance JSON (demucs version / model / settings; committed).

    The audit trail for "which teacher, which settings, how self-consistent" (MASTER_PLAN
    §3.2, THEORY §5). Committed alongside the manifest; **no audio**.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "demucs_version": provenance.demucs_version,
        "model": provenance.model,
        "settings": provenance.settings,
        "n_labeled": int(provenance.n_labeled),
        "n_kept": int(provenance.n_kept),
        "activity_threshold": float(provenance.activity_threshold),
        "used_threshold": float(provenance.used_threshold),
        "consistency_residual_db_mean": float(provenance.consistency_residual_db_mean),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# --- the demucs invocation (constructed here; lazy import; RUN LATER) -------

def load_teacher(model_name: str = DEFAULT_TEACHER_MODEL):  # pragma: no cover - RUN LATER
    """Lazily import demucs and load the teacher model (**RUN LATER — needs demucs + weights**).

    The import lives inside the function so importing ``scripts/teacher_label`` (and running
    the unit tests) pulls in no heavy deps and downloads nothing. Raises a clear, actionable
    error if demucs is not installed.
    """
    try:
        from demucs.pretrained import get_model  # type: ignore
    except ImportError as exc:  # pragma: no cover - RUN LATER
        raise RuntimeError(
            "demucs is not installed — `pip install demucs` (data-time dep) before teacher "
            "labeling (RUN LATER; MASTER_PLAN §3.2). Nothing is downloaded during tests."
        ) from exc
    return get_model(model_name)


def teacher_separate_vocals(model, mixture: np.ndarray, sr: int) -> dict[str, np.ndarray]:  # pragma: no cover - RUN LATER
    """Run the teacher and return its 4 stems as ``{name: mono float32}`` (**RUN LATER**).

    Constructs the demucs ``apply_model`` invocation; lazily imports torch + demucs.apply.
    Never executed by the CPU test suite (the post-processing that consumes its output is
    tested on synthetic arrays instead).
    """
    import torch  # pragma: no cover
    from demucs.apply import apply_model  # type: ignore  # pragma: no cover

    wav = torch.as_tensor(np.asarray(mixture, dtype=np.float32))
    if wav.ndim == 1:
        wav = wav.unsqueeze(0).repeat(2, 1)  # demucs expects stereo (channels, time)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (wav.std() + EPS)
    sources = apply_model(model, wav[None], device="cuda" if torch.cuda.is_available() else "cpu")[0]
    sources = sources * wav.std() + ref.mean()
    names = list(getattr(model, "sources", ["drums", "bass", "other", "vocals"]))
    return {name: sources[i].mean(0).cpu().numpy().astype(np.float32) for i, name in enumerate(names)}


def run(  # pragma: no cover - RUN LATER (GPU, needs manifest audio + demucs)
    manifest_csv: str | Path,
    out: str | Path,
    *,
    keep: int = DEFAULT_KEEP,
    activity_threshold: float = DEFAULT_ACTIVITY_THRESHOLD,
    model_name: str = DEFAULT_TEACHER_MODEL,
    audio_dir: str | Path | None = None,
    sample_rate: int = 44100,
) -> TeacherProvenance:
    """Label the screened clips, screen by activity, write shards + profiles + provenance.

    **RUN LATER — GPU, ~2-4 h once.** Fails loud without the manifest / audio / demucs. Writes
    2-stem WAV shards (mixture + pseudo vocals + pseudo accompaniment) + activity profiles +
    ``provenance.json`` under ``out`` (never committed; audio stays on Drive). The tested
    post-processing (:func:`two_stem_consistency`, :func:`consistency_residual_db`,
    :func:`vocal_activity_ratio`, :func:`screen_by_activity`, :func:`write_provenance`) is
    called here; this orchestration is the only untested (RUN LATER) part.
    """
    import soundfile as sf  # pragma: no cover

    import pandas as pd  # pragma: no cover

    manifest_csv = Path(manifest_csv)
    if not manifest_csv.exists():
        raise FileNotFoundError(
            f"manifest {manifest_csv} missing — run scripts/prepare_fma.py first (RUN LATER)."
        )
    frame = pd.read_csv(manifest_csv, comment="#")
    screened = frame[frame.get("screened", False).astype(bool)] if "screened" in frame else frame
    audio_root = Path(audio_dir) if audio_dir is not None else manifest_csv.parent / "audio"
    if not audio_root.exists():
        raise FileNotFoundError(
            f"FMA audio dir {audio_root} missing — teacher labeling is RUN LATER (needs audio)."
        )

    model = load_teacher(model_name)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ratios: dict[str, float] = {}
    residuals: list[float] = []
    for track_id in screened["track_id"].astype(str):
        wav_path = audio_root / f"{track_id}.wav"
        if not wav_path.exists():
            continue
        mixture, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        mono = mixture.mean(axis=1)
        stems = teacher_separate_vocals(model, mono, sr)
        two = two_stem_consistency(mono, stems["vocals"])
        residuals.append(consistency_residual_db(stems, mono))
        ratios[track_id] = vocal_activity_ratio(two["vocals"], sr)
        clip_dir = out_dir / track_id
        clip_dir.mkdir(parents=True, exist_ok=True)
        sf.write(clip_dir / "mixture.wav", mono, sr)
        sf.write(clip_dir / "vocals.wav", two["vocals"], sr)
        sf.write(clip_dir / "accompaniment.wav", two["accompaniment"], sr)

    screen = screen_by_activity(ratios, keep=keep, threshold=activity_threshold)
    provenance = TeacherProvenance(
        demucs_version=_demucs_version(),
        model=model_name,
        settings={"sample_rate": sample_rate, "two_stem": "vocals; accompaniment = mixture - vocals"},
        n_labeled=len(ratios),
        n_kept=len(screen.kept),
        activity_threshold=activity_threshold,
        used_threshold=screen.used_threshold,
        consistency_residual_db_mean=float(np.mean(residuals)) if residuals else float("nan"),
    )
    write_provenance(out_dir / "provenance.json", provenance)
    (out_dir / "kept_clips.csv").write_text(
        "track_id\n" + "\n".join(screen.kept) + "\n", encoding="utf-8"
    )
    return provenance


def _demucs_version() -> str:  # pragma: no cover - RUN LATER
    try:
        import demucs  # type: ignore

        return str(getattr(demucs, "__version__", "unknown"))
    except ImportError:
        return "not-installed"


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - RUN LATER
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", required=True, help="the committed FMA manifest.csv (screened clips)")
    parser.add_argument("--keep", type=int, default=DEFAULT_KEEP, help="N_train survivors to keep (§3.2)")
    parser.add_argument("--activity-threshold", type=float, default=DEFAULT_ACTIVITY_THRESHOLD)
    parser.add_argument("--model", default=DEFAULT_TEACHER_MODEL, help="teacher model signature (htdemucs)")
    parser.add_argument("--audio-dir", default=None, help="FMA audio dir (RUN LATER)")
    parser.add_argument("--out", required=True, help="pseudo root (shards + profiles + provenance)")
    args = parser.parse_args(argv)
    prov = run(args.manifest, args.out, keep=args.keep, activity_threshold=args.activity_threshold,
               model_name=args.model, audio_dir=args.audio_dir)
    print(f"teacher labeling done: {prov.n_kept}/{prov.n_labeled} kept at threshold "
          f"{prov.used_threshold} (demucs {prov.demucs_version}).")


if __name__ == "__main__":
    main()
