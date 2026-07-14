"""Training loop, LR schedule, checkpointing, and the run registry."""

from __future__ import annotations

from .loop import (
    RunResult,
    build_optimizer,
    build_scheduler,
    build_training_loss,
    chunk_silent_fraction,
    is_new_best,
    load_checkpoint,
    make_lr_lambda,
    prepare_batch,
    run,
    save_checkpoint,
    write_exposure_telemetry,
    write_trim_telemetry,
)
from .registry import (
    REGISTRY_COLUMNS,
    RunRecord,
    current_git_commit,
    get_run,
    make_run_id,
    read_registry,
    upsert_run,
)

__all__ = [
    "RunResult",
    "build_optimizer",
    "build_scheduler",
    "build_training_loss",
    "chunk_silent_fraction",
    "is_new_best",
    "load_checkpoint",
    "make_lr_lambda",
    "prepare_batch",
    "run",
    "save_checkpoint",
    "write_exposure_telemetry",
    "write_trim_telemetry",
    "REGISTRY_COLUMNS",
    "RunRecord",
    "current_git_commit",
    "get_run",
    "make_run_id",
    "read_registry",
    "upsert_run",
]
