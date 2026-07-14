r"""The Direction-10 consolidated test session (MASTER_PLAN §5, §7 G3).

**RUN LATER — needs the decoded MUSDB shards, every trained student checkpoint, and (for the
teacher anchor) demucs + htdemucs weights.** One pass, one CSV
(``results/test_session.csv``), no reruns: on the **50 MUSDB test tracks** (which the
teacher never labels — no leakage by construction) it scores

* the **student checkpoints** — the 6 seed-cell rows (``musdb_only``, ``mixed``) + the 3
  single-seed arms (``distill_only``, ``mixed25``, ``mixed_trim``) — read from the registry,
* **the teacher itself** (``htdemucs``) through a **lazy demucs wrapper** (:class:`LazyDemucsTeacher`;
  the import is guarded and RUN LATER) — the ``s_T`` gap-closure anchor,
* the **do-nothing** floor and the **oracle IRM** ceiling,

each with vocals/accompaniment SI-SDR, SI-SDRi, and — the §5 battery — **SLR at θ ∈ {−50,
−60, −70}**. The teacher's 2 stems are derived by the same 2-stem consistency the student
learns (``accompaniment = mixture − teacher_vocals``), so student and teacher are scored on
one footing. The function is a **skeleton that fails loud** when the shards, registry, or
checkpoints are absent (exactly the CPU test environment), so the unit test asserts the
clean error and never runs the heavy pass; the teacher import is reached only RUN LATER.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

#: Columns of the consolidated test-session CSV (§5).
TEST_SESSION_COLUMNS: tuple[str, ...] = (
    "system", "arm", "seed", "data_source",
    "sisdr_vocals_mean", "sisdr_vocals_median", "sisdr_accomp_mean", "sisdr_i_vocals_mean",
    "slr_m50", "slr_m60", "slr_m70",
    "n_tracks", "is_teacher", "checkpoint_path",
)

#: The student arms scored in the session (their registry rows carry the checkpoints).
STUDENT_ARMS: tuple[str, ...] = ("musdb_only", "mixed", "distill_only", "mixed25", "mixed_trim")


class LazyDemucsTeacher:
    """The teacher (htdemucs) as a lazy, RUN-LATER demucs wrapper (the ``s_T`` anchor, §5).

    Importing this class pulls in **no** heavy deps: demucs + torch are imported inside
    :meth:`separate_vocals`, which is reached only during the RUN-LATER test pass. Its
    accompaniment is derived by the student's own 2-stem consistency (``mixture − vocals``),
    so the teacher is scored on the same 2-stem task.
    """

    def __init__(self, model_name: str = "htdemucs", device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

    def separate_vocals(self, mixture, sr: int):  # pragma: no cover - RUN LATER (needs demucs)
        """Return the teacher's mono vocals estimate for a mixture (**RUN LATER**)."""
        import sys

        import numpy as np

        # scripts/ is not a package; add it to the path like the unit tests do, then import
        # the demucs-invocation helpers (their demucs import is itself lazy + RUN LATER).
        scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from teacher_label import load_teacher, teacher_separate_vocals

        if self._model is None:
            self._model = load_teacher(self.model_name)
        stems = teacher_separate_vocals(self._model, np.asarray(mixture), sr)
        return stems["vocals"]


@dataclass
class TeacherSessionSpec:
    """Resolved inputs for the consolidated session (all paths, no computation yet)."""

    registry_path: Path
    shard_root: Path
    splits_csv: Path
    output_dir: Path
    include_teacher: bool = True
    slr: bool = True
    device: str = "cpu"
    teacher_model: str = "htdemucs"


def _require(path: Path, what: str) -> Path:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{what} missing at {path!r} — the Direction-10 test session is RUN LATER "
            "(needs decoded MUSDB shards + trained checkpoints + demucs; MASTER_PLAN §6 run book)."
        )
    return Path(path)


def build_teacher_session(
    registry_path: str | Path,
    shard_root: str | Path,
    splits_csv: str | Path,
    output_dir: str | Path = "10-demucs-distillation/results",
    *,
    include_teacher: bool = True,
    slr: bool = True,
    device: str = "cpu",
    teacher_model: str = "htdemucs",
) -> pd.DataFrame:
    """Score the student checkpoints + teacher + anchors on the 50 MUSDB test tracks (§5).

    **RUN LATER.** Fails loud (``FileNotFoundError``) if the shards, registry, or splits are
    absent — the state of the CPU test environment, so the unit test exercises the clean-error
    path. When the inputs exist (RUN LATER) it writes ``<output_dir>/test_session.csv`` with
    one row per (system) carrying vocals/accomp SI-SDR, SI-SDRi, and SLR at all three θ, plus
    the ``htdemucs`` teacher anchor (``include_teacher``) and the do-nothing / oracle-IRM lines.
    """
    spec = TeacherSessionSpec(
        registry_path=Path(registry_path), shard_root=Path(shard_root),
        splits_csv=Path(splits_csv), output_dir=Path(output_dir),
        include_teacher=include_teacher, slr=slr, device=device, teacher_model=teacher_model,
    )
    _require(spec.shard_root, "shard_root (decoded MUSDB shards)")
    _require(spec.splits_csv, "splits_csv (the frozen 86/14/50 split)")
    _require(spec.registry_path, "registry.csv (the completed student runs)")

    # --- RUN LATER: the heavy scoring pass (needs checkpoints + data + demucs) -----
    import numpy as np  # pragma: no cover - RUN LATER
    import torch  # pragma: no cover

    from ..audio.stft import STFT  # pragma: no cover
    from ..data.manifest import Manifest  # pragma: no cover
    from ..data.musdb_dataset import WavShardStore  # pragma: no cover
    from ..train.loop import load_checkpoint  # pragma: no cover
    from ..models import build_model_from_config  # pragma: no cover
    from .evaluate import _oracle_vocals, score_system  # pragma: no cover
    from .overlap_add import separate_track  # pragma: no cover

    registry = pd.read_csv(spec.registry_path)  # pragma: no cover
    manifest = Manifest.from_csv(spec.splits_csv)  # pragma: no cover
    store = WavShardStore(spec.shard_root)  # pragma: no cover
    dev = torch.device(spec.device)  # pragma: no cover
    stft = STFT().to(dev)  # pragma: no cover
    test_tracks = [t for t in manifest.tracks_for("test") if t in set(store.track_names())]  # pragma: no cover
    rows: list[dict[str, object]] = []  # pragma: no cover

    def _aggregate(system, arm, seed, data_source, scored, is_teacher, checkpoint):  # pragma: no cover
        voc = [s.sisdr_vocals for s in scored]
        return {
            "system": system, "arm": arm, "seed": seed, "data_source": data_source,
            "sisdr_vocals_mean": float(np.mean(voc)) if voc else float("nan"),
            "sisdr_vocals_median": float(np.median(voc)) if voc else float("nan"),
            "sisdr_accomp_mean": float(np.mean([s.sisdr_accomp for s in scored])) if scored else float("nan"),
            "sisdr_i_vocals_mean": float(np.mean([s.sisdr_i_vocals for s in scored])) if scored else float("nan"),
            "slr_m50": float(np.nanmean([s.slr_m50 for s in scored])) if scored else float("nan"),
            "slr_m60": float(np.nanmean([s.slr_m60 for s in scored])) if scored else float("nan"),
            "slr_m70": float(np.nanmean([s.slr_m70 for s in scored])) if scored else float("nan"),
            "n_tracks": len(scored), "is_teacher": bool(is_teacher), "checkpoint_path": checkpoint,
        }

    # do-nothing floor + oracle-IRM ceiling (one pass; anchors for the (SI-SDR, SLR) plane).
    for system in ("do_nothing", "oracle_irm"):  # pragma: no cover
        scored = []
        for track in test_tracks:
            src = store.load_sources(track)
            vocals = src["vocals"].astype(np.float64)
            accompaniment = src["accompaniment"].astype(np.float64)
            mixture = vocals + accompaniment
            if system == "do_nothing":
                voc_hat = mixture
            else:
                voc_hat = _oracle_vocals(
                    torch.from_numpy(mixture).float(), torch.from_numpy(vocals).float(),
                    torch.from_numpy(accompaniment).float(), stft.cpu(), "irm",
                ).numpy()
            scored.append(score_system(track, system, voc_hat, mixture - voc_hat,
                                       vocals, accompaniment, mixture, sr=int(store.sample_rate), slr=slr))
        rows.append(_aggregate(system, system, "", "anchor", scored, False, ""))

    # every completed student checkpoint from the registry.
    for _idx, run in registry.iterrows():  # pragma: no cover
        checkpoint = run.get("checkpoint_path")
        if not (isinstance(checkpoint, str) and checkpoint and Path(checkpoint).exists()):
            continue
        payload = load_checkpoint(checkpoint, restore_rng=False)
        model = build_model_from_config(payload.get("config") or {}).to(dev)
        model.load_state_dict(payload["model"])
        model.eval()
        scored = []
        for track in test_tracks:
            src = store.load_sources(track)
            vocals = src["vocals"].astype(np.float64)
            accompaniment = src["accompaniment"].astype(np.float64)
            mixture = vocals + accompaniment
            voc_hat = separate_track(model, torch.from_numpy(mixture).float().to(dev), stft, dev).cpu().numpy()
            scored.append(score_system(track, "singnet", voc_hat, mixture - voc_hat,
                                       vocals, accompaniment, mixture, sr=int(store.sample_rate), slr=slr))
        rows.append(_aggregate(run.get("arm"), run.get("arm"), run.get("seed"),
                               run.get("data_source"), scored, False, checkpoint))

    # the teacher anchor (s_T) — the lazy demucs wrapper, scored on the 2-stem consistency.
    if include_teacher:  # pragma: no cover
        teacher = LazyDemucsTeacher(teacher_model, device=spec.device)
        scored = []
        for track in test_tracks:
            src = store.load_sources(track)
            vocals = src["vocals"].astype(np.float64)
            accompaniment = src["accompaniment"].astype(np.float64)
            mixture = vocals + accompaniment
            voc_hat = np.asarray(teacher.separate_vocals(mixture, int(store.sample_rate)), dtype=np.float64)
            scored.append(score_system(track, "teacher", voc_hat, mixture - voc_hat,
                                       vocals, accompaniment, mixture, sr=int(store.sample_rate), slr=slr))
        rows.append(_aggregate("teacher", "teacher", "", "teacher", scored, True, ""))

    frame = pd.DataFrame(rows, columns=list(TEST_SESSION_COLUMNS))  # pragma: no cover
    spec.output_dir.mkdir(parents=True, exist_ok=True)  # pragma: no cover
    frame.to_csv(spec.output_dir / "test_session.csv", index=False)  # pragma: no cover
    return frame  # pragma: no cover


__all__ = [
    "build_teacher_session",
    "LazyDemucsTeacher",
    "TeacherSessionSpec",
    "TEST_SESSION_COLUMNS",
    "STUDENT_ARMS",
]
