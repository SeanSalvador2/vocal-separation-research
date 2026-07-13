"""The experiment run registry (MASTER_PLAN §7.1) — a CSV, git-diffable.

Every run appends/updates one row keyed by a **config hash** so an interrupted
sweep resumes exactly and no two runs collide. The registry and the eval CSVs
are committed; checkpoints and audio are not.

Schema (exact column order)::

    run_id, arm, seed, budget, config_hash, git_commit, gpu, wall_clock_h,
    steps_done, best_val_sisdr, final_val_sisdr, sisdr_skip_rate, checkpoint_path,
    aug_remix, aug_gain, aug_flip, n_songs

The last four columns are the Direction-02 additions (MASTER_PLAN §6): the
augmentation switchboard state and the subset size for the scaling curve. They
are **appended** so the schema stays backward-compatible — an older Direction-01
registry (without them) reads back with those columns NaN-filled, and Direction-01
runs populate them with the full-recipe defaults (all True, 86 songs).
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import pandas as pd

REGISTRY_COLUMNS: tuple[str, ...] = (
    "run_id",
    "arm",
    "seed",
    "budget",
    "config_hash",
    "git_commit",
    "gpu",
    "wall_clock_h",
    "steps_done",
    "best_val_sisdr",
    "final_val_sisdr",
    "sisdr_skip_rate",
    "checkpoint_path",
    "aug_remix",
    "aug_gain",
    "aug_flip",
    "n_songs",
)


@dataclass
class RunRecord:
    """One registry row. Numeric metrics default to NaN until measured."""

    run_id: str
    arm: str
    seed: int
    budget: int
    config_hash: str
    git_commit: str = ""
    gpu: str = ""
    wall_clock_h: float = float("nan")
    steps_done: int = 0
    best_val_sisdr: float = float("nan")
    final_val_sisdr: float = float("nan")
    sisdr_skip_rate: float = float("nan")
    checkpoint_path: str = ""
    # Direction-02 additions (defaults = the Direction-01 full recipe on 86 songs).
    aug_remix: bool = True
    aug_gain: bool = True
    aug_flip: bool = True
    n_songs: int = 86

    def as_row(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def make_run_id(arm: str, seed: int, budget: str | int, config_hash: str) -> str:
    """Human-readable, collision-free run id (``arm_seedS_budget_hash``)."""
    return f"{arm}_seed{seed}_{budget}_{config_hash}"


def current_git_commit(default: str = "") -> str:
    """Return the current short git commit, or ``default`` if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return default


def read_registry(path: str | Path) -> pd.DataFrame:
    """Read the registry (returns an empty, correctly-typed frame if absent)."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=list(REGISTRY_COLUMNS))
    return pd.read_csv(path)


def write_registry(path: str | Path, frame: pd.DataFrame) -> None:
    """Write the registry with the canonical column order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.reindex(columns=list(REGISTRY_COLUMNS)).to_csv(path, index=False)


def upsert_run(path: str | Path, record: RunRecord) -> pd.DataFrame:
    """Insert ``record`` or update the existing row with the same ``run_id``.

    Idempotent: appending the same run twice leaves a single row (the second
    call overwrites metrics) — exactly what a resumed run needs.
    """
    frame = read_registry(path)
    row = record.as_row()
    if len(frame):  # drop any existing row for this run_id (resume overwrites)
        frame = frame[frame["run_id"] != record.run_id]
    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    write_registry(path, frame)
    return frame


def get_run(path: str | Path, run_id: str) -> dict[str, Any] | None:
    """Return the row for ``run_id`` as a dict, or None if absent."""
    frame = read_registry(path)
    match = frame[frame["run_id"] == run_id] if len(frame) else frame
    if not len(match):
        return None
    return match.iloc[0].to_dict()
