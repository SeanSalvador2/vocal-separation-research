"""Training loop, LR schedule, checkpointing, and the run registry."""

from __future__ import annotations

from .loop import (
    RunResult,
    build_optimizer,
    build_scheduler,
    load_checkpoint,
    make_lr_lambda,
    prepare_batch,
    run,
    save_checkpoint,
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
    "load_checkpoint",
    "make_lr_lambda",
    "prepare_batch",
    "run",
    "save_checkpoint",
    "REGISTRY_COLUMNS",
    "RunRecord",
    "current_git_commit",
    "get_run",
    "make_run_id",
    "read_registry",
    "upsert_run",
]
