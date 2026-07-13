#!/usr/bin/env python3
r"""Score checkpoints (+ do-nothing floor + oracle IRM/IBM) on a split.

**RUN LATER — needs decoded shards and trained checkpoints (MASTER_PLAN §7.3-7.4).**
Thin CLI over :func:`singnet.eval.evaluate`; writes ``<out>/<split>_per_track.csv``
and (with ``--museval``) the secondary ``<split>_museval.csv``.

The **single test pass** for the whole direction (§7.4) is exactly one invocation
with ``--split test`` on the 3 full-budget checkpoints — run only after the val
analysis is frozen (gate G3). Example (from the run book, §8):

    python scripts/evaluate.py --checkpoint checkpoints/<hash>/best.pt \
        --split test --shard-root $SHARD_ROOT \
        --splits-csv 01-loss-function-study/configs/splits.csv \
        --output-dir 01-loss-function-study/results --oracles --museval
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from singnet.eval.evaluate import _cli  # noqa: E402

if __name__ == "__main__":
    _cli()
