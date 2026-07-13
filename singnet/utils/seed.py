"""Reproducible seeding.

Two distinct notions of "seed" live here and must not be confused:

* **Run seed** — the single integer (``0``, ``1``, ``2`` in the sweep) that
  fixes model init and the whole training stream. :func:`seed_everything`
  applies it to Python, NumPy and torch.
* **Per-sample RNG** — the dataloader draws each training example from a stream
  that is a pure function of ``(run_seed, sample_index)`` and *nothing else*
  (crucially, not the loss arm). :func:`derive_rng` builds that stream. This is
  what makes the five loss arms see a bit-identical data order (MASTER_PLAN
  §3.1's controlled-comparison invariant).

:func:`capture_rng_state` / :func:`restore_rng_state` snapshot and restore the
global RNGs so a training run is resumable to the step (MASTER_PLAN §7.1).
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy and torch (CPU + CUDA) from a single integer.

    Args:
        seed: the run seed.
        deterministic: if True, request deterministic torch algorithms. We use
            ``warn_only=True`` so that ops without a deterministic kernel warn
            rather than crash a long training run (deviation documented in the
            reproducibility appendix).
    """
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def derive_rng(seed: int, *coords: int) -> np.random.Generator:
    """Return an independent NumPy generator keyed by ``(seed, *coords)``.

    Uses :class:`numpy.random.SeedSequence` so distinct coordinate tuples yield
    statistically independent streams. The dataloader calls
    ``derive_rng(run_seed, sample_index)`` — the loss arm never enters, which is
    exactly the invariant a controlled loss comparison needs.
    """
    entropy = [int(seed), *(int(c) for c in coords)]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def worker_init_fn(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` that gives each worker a distinct seed.

    Derives the worker seed from the current torch seed so it still tracks the
    run seed set by :func:`seed_everything`.
    """
    base = torch.initial_seed() % (2**32)
    seed = (base + worker_id) % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def capture_rng_state() -> dict[str, Any]:
    """Snapshot Python/NumPy/torch RNG state for a resumable checkpoint."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG state captured by :func:`capture_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_state = state["torch"]
    if not isinstance(torch_state, torch.Tensor):
        torch_state = torch.as_tensor(torch_state, dtype=torch.uint8)
    torch.set_rng_state(torch_state.to(torch.uint8))
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
