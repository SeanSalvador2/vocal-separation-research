#!/usr/bin/env python3
r"""Draw the nested, deterministic training subsets for the scaling curve (§3.2, §4).

**RUN LATER — CPU, once, after data prep (gate G0b).** From the *materialized*
split manifest (real track names, not placeholders), draw nested subsets
``n21 ⊂ n43 ⊂ n64 ⊂ n86`` with a single subset-seed and commit the names-only
CSVs. Nesting means "less data" strictly removes songs; the 14-track validation
split never changes (MASTER_PLAN §3.2).

    python scripts/make_subsets.py \
        --manifest 01-loss-function-study/configs/splits.csv \
        --sizes 21 43 64 --seed 0 \
        --out 02-augmentation-data-scaling/configs/subsets/ \
        [--index $SHARD_ROOT/index.json]

The draw is a pure function of ``(manifest train order, --seed)`` — regenerating
with the same inputs yields byte-identical files (unit-tested). The script
**fails loud** if the manifest still carries ``__PLACEHOLDER`` rows (it must be
materialized first), and prints a per-subset balance report (duration and
vocal-activity if a prepare-data index is supplied, else song counts only) so a
pathological draw is visible *before* training. Pre-registered rule: the seed-0
draw is used regardless — pathologies are reported, never re-rolled (§4).

The core functions are importable and unit-tested; ``main`` is the thin CLI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from singnet.data.manifest import Manifest  # noqa: E402

PLACEHOLDER_PREFIX = "__PLACEHOLDER"


def assert_materialized(tracks: list[str]) -> None:
    """Raise if any track name is a documented placeholder (manifest not built yet)."""
    placeholders = [t for t in tracks if t.startswith(PLACEHOLDER_PREFIX)]
    if placeholders:
        raise ValueError(
            f"manifest still has {len(placeholders)} placeholder rows "
            f"(e.g. {placeholders[0]!r}); run prepare_data.py --write-manifest first "
            "(MASTER_PLAN §4, gate G0b)."
        )


def make_nested_subsets(
    train_tracks: list[str], sizes: list[int], seed: int
) -> dict[int, list[str]]:
    """Return ``{size: tracks}`` nested subsets drawn once with ``seed``.

    The full train order is shuffled once (a pure function of ``seed``); each
    subset is the first ``size`` names of that shuffle, so the subsets are nested
    by construction (``subset[s1] ⊂ subset[s2]`` for ``s1 < s2``) and drawing the
    full split (``size == len(train_tracks)``) recovers the whole split.
    """
    n = len(train_tracks)
    sizes = sorted(int(s) for s in sizes)
    if sizes and sizes[-1] > n:
        raise ValueError(f"requested subset size {sizes[-1]} exceeds the {n}-track train split")
    if any(s <= 0 for s in sizes):
        raise ValueError(f"subset sizes must be positive; got {sizes}")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(n)
    shuffled = [train_tracks[i] for i in order]
    return {s: shuffled[:s] for s in sizes}


def subset_balance(
    subsets: dict[int, list[str]], index: dict | None = None
) -> pd.DataFrame:
    """Per-subset balance report: song count, and duration/vocal-activity if known.

    ``index`` is the optional prepare-data metadata map ``{track: {...}}`` (keys
    ``duration_s`` and/or ``vocal_activity`` tolerated under a few spellings). With
    no index only ``n_songs`` is reported (the rest is pending data prep).
    """
    dur_keys = ("duration_s", "duration", "duration_sec")
    voc_keys = ("vocal_activity", "vocal_activity_ratio", "vocal_ratio")

    def _get(meta: dict, keys: tuple[str, ...]) -> float | None:
        for k in keys:
            if k in meta:
                return float(meta[k])
        return None

    rows = []
    for size in sorted(subsets):
        tracks = subsets[size]
        row: dict[str, object] = {"subset": f"n{size}", "n_songs": len(tracks)}
        if index is not None:
            durs = [d for t in tracks if (d := _get(index.get(t, {}), dur_keys)) is not None]
            vocs = [v for t in tracks if (v := _get(index.get(t, {}), voc_keys)) is not None]
            if durs:
                row["total_duration_min"] = round(sum(durs) / 60.0, 2)
                row["mean_duration_s"] = round(float(np.mean(durs)), 2)
            if vocs:
                row["mean_vocal_activity"] = round(float(np.mean(vocs)), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def write_subsets(subsets: dict[int, list[str]], out_dir: str | Path) -> list[Path]:
    """Write each subset as ``n{size}.csv`` (a names-only ``track`` column)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for size in sorted(subsets):
        path = out_dir / f"n{size}.csv"
        header = (
            f"# Nested training subset n{size} (MASTER_PLAN §3.2). Names only, no audio.\n"
            f"# Drawn once by scripts/make_subsets.py; nested n21 ⊂ n43 ⊂ n64 ⊂ n86.\n"
        )
        body = "track\n" + "".join(f"{t}\n" for t in subsets[size])
        path.write_text(header + body, encoding="utf-8")
        paths.append(path)
    return paths


def load_index(index_path: str | Path | None) -> dict | None:
    if index_path is None:
        return None
    with open(index_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", required=True, help="materialized splits.csv (track,split)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[21, 43, 64], help="subset sizes")
    parser.add_argument("--seed", type=int, default=0, help="subset draw seed (pre-registered: 0)")
    parser.add_argument("--out", required=True, help="output dir for n{size}.csv")
    parser.add_argument("--index", default=None, help="optional prepare-data index.json for balance")
    args = parser.parse_args(argv)

    manifest = Manifest.from_csv(args.manifest)
    train_tracks = manifest.tracks_for("train")
    assert_materialized(train_tracks)
    print(f"train split: {len(train_tracks)} materialized tracks; sizes={args.sizes} seed={args.seed}")

    subsets = make_nested_subsets(train_tracks, args.sizes, args.seed)
    paths = write_subsets(subsets, args.out)
    for path in paths:
        print(f"  wrote {path}")

    print("\nSubset-balance report (§4 — reviewed at G0b, never re-rolled):")
    report = subset_balance(subsets, load_index(args.index))
    print(report.to_string(index=False))
    if args.index is None:
        print("(duration / vocal-activity columns appear once --index points at the "
              "prepare_data metadata; song counts shown for now.)")


if __name__ == "__main__":
    main()
