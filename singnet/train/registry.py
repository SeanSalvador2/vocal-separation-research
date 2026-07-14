"""The experiment run registry (MASTER_PLAN §7.1) — a CSV, git-diffable.

Every run appends/updates one row keyed by a **config hash** so an interrupted
sweep resumes exactly and no two runs collide. The registry and the eval CSVs
are committed; checkpoints and audio are not.

Schema (exact column order)::

    run_id, arm, seed, budget, config_hash, git_commit, gpu, wall_clock_h,
    steps_done, best_val_sisdr, final_val_sisdr, sisdr_skip_rate, checkpoint_path,
    aug_remix, aug_gain, aug_flip, n_songs, base_width,
    domain, recipe, rank, lr, trainable_params, trainable_share, peak_vram_gb,
    epsilon, trim_q, kept_fraction_observed, trim_energy_stats_path

The ``aug_*``/``n_songs`` block is the Direction-02 addition (MASTER_PLAN §6). The
``base_width`` is the Direction-03 addition (§5, §9): the chosen tower/decoder base
width ``c`` recorded per run. The seven columns (``domain`` … ``peak_vram_gb``)
are the **Direction-05** addition (D05 MASTER_PLAN §5): the domain, PEFT recipe, LoRA
rank, (probe-frozen) learning rate, and the trained parameter count / share / peak
VRAM. The final four (``epsilon``, ``trim_q``, ``kept_fraction_observed``,
``trim_energy_stats_path``) are the **Direction-06** addition (D06 MASTER_PLAN §5): the
stem-bleed level ε, the trimmed-loss fraction q, the observed mean kept-fraction, and
the per-run trim-telemetry CSV path. Every new column is **appended** so the schema
stays backward-compatible: an older registry (without them) reads back NaN-filled, and
runs that predate a column populate it with the identity default (the full recipe on 86
songs; ``base_width`` = 32; the D05 fields empty/NaN for the Directions 01-03 rows; the
D06 fields empty/NaN for non-bleed runs).
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
    "base_width",
    # Direction-05 (PEFT / LoRA fine-tuning) additions — appended block, §5.
    "domain",
    "recipe",
    "rank",
    "lr",
    "trainable_params",
    "trainable_share",
    "peak_vram_gb",
    # Direction-06 (robust training under stem bleed) additions — appended block, §5.
    "epsilon",
    "trim_q",
    "kept_fraction_observed",
    "trim_energy_stats_path",
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
    # Direction-03 addition (default = the baseline SingNet-C1 width).
    base_width: int = 32
    # Direction-05 additions (PEFT run metadata; defaults = "not a PEFT run").
    domain: str = ""
    recipe: str = ""
    rank: int = 0
    lr: float = float("nan")
    trainable_params: int = 0
    trainable_share: float = float("nan")
    peak_vram_gb: float = float("nan")
    # Direction-06 additions (robust-training metadata; defaults = "not a bleed run").
    epsilon: float = float("nan")
    trim_q: float = float("nan")
    kept_fraction_observed: float = float("nan")
    trim_energy_stats_path: str = ""

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
