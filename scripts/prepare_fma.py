#!/usr/bin/env python3
r"""FMA metadata filter + license audit + deterministic screen sample (MASTER_PLAN §3.1).

**Metadata-only. No audio is ever read, downloaded, or committed.** This script turns
the FMA ``tracks.csv`` metadata into a committed, auditable **track-ID manifest** of the
license-safe clips the teacher may later label. It is **RUN LATER** on the real FMA
metadata (CPU + network, ~minutes); its pure functions — the license allow/deny decision,
the deterministic screen sample, and the manifest writer — are unit-tested on synthetic
metadata fixtures (``tests/test_prepare_fma.py``), so gate G0 exercises them with no data.

The license call is an **allowlist, DENY-by-default** (MASTER_PLAN §3.1, THEORY §5):

* **Allowed (pinned):** CC0 / Public Domain, CC-BY, CC-BY-SA, CC-BY-NC, CC-BY-NC-SA.
* **Denied:** *any* ``-ND`` (NoDerivatives) variant — separated stems are derivative works,
  so ND is the load-bearing legal exclusion — **and** any non-CC / unclear / empty /
  ``All Rights Reserved`` string. Unclear licenses are denied, never guessed (the
  allowlist, not a blocklist).

Run book (§6; RUN LATER — nothing here touches audio)::

    python scripts/prepare_fma.py --metadata-dir $FMA_META --audio-dir $FMA_AUDIO \
           --screen 1200 --seed 0 --out $PSEUDO_ROOT     # writes $PSEUDO_ROOT/manifest.csv

The manifest carries, per allowlisted track: the FMA track id, the raw license string, the
canonical license token, and a ``screened`` flag (in the deterministic screen sample or
not). **Never audio.**
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: The pinned canonical license allowlist (MASTER_PLAN §3.1). CC0 covers Public Domain.
ALLOWED_LICENSES: frozenset[str] = frozenset(
    {"CC0", "CC-BY", "CC-BY-SA", "CC-BY-NC", "CC-BY-NC-SA"}
)


def canonicalize_license(raw: object) -> str | None:
    r"""Map a raw FMA license string to a canonical allowlist token, or ``None`` to deny.

    Handles both human-readable names ("Attribution-NonCommercial-ShareAlike 3.0
    International") and Creative-Commons URL forms
    ("http://creativecommons.org/licenses/by-nc-sa/3.0/"). The decision is an
    **allowlist**: a string is denied unless it is recognizably a CC0/Public-Domain or a
    CC-BY(-NC)(-SA) license *without* a NoDerivatives clause. In particular:

    * empty / non-string / ``All Rights Reserved`` / arbitrary garbage -> ``None`` (deny),
    * **any** ``-ND`` / NoDerivatives variant -> ``None`` (deny; the §5 load-bearing call),
    * CC0 / Public Domain / CC0-Universal / "Creative Commons Zero" -> ``"CC0"``,
    * otherwise the ``CC-BY[-NC][-SA]`` token iff it lands in :data:`ALLOWED_LICENSES`.
    """
    if not isinstance(raw, str):
        return None
    s = raw.strip().lower()
    if not s:
        return None

    # Public-domain / CC0 bucket (checked first — it carries no BY clause).
    if (
        "cc0" in s
        or "publicdomain" in s
        or "public domain" in s
        or "public-domain" in s
        or "/zero/" in s
        or ("creative commons" in s and "zero" in s)
    ):
        return "CC0"

    # NoDerivatives is denied outright, whatever else the string says (§5).
    if (
        "noderiv" in s
        or "no deriv" in s
        or "-nd" in s
        or "/nd" in s
        or re.search(r"\bnd\b", s)
    ):
        return None

    # In scope only if it is a CC Attribution (BY) license.
    has_by = "attribution" in s or "/by" in s or "by-" in s or bool(re.search(r"\bby\b", s))
    if not has_by:
        return None

    nc = (
        "noncommercial" in s
        or "non-commercial" in s
        or "-nc" in s
        or "/nc" in s
        or bool(re.search(r"\bnc\b", s))
    )
    sa = (
        "sharealike" in s
        or "share-alike" in s
        or "-sa" in s
        or "/sa" in s
        or bool(re.search(r"\bsa\b", s))
    )
    code = "CC-BY" + ("-NC" if nc else "") + ("-SA" if sa else "")
    return code if code in ALLOWED_LICENSES else None


def license_allowed(raw: object) -> bool:
    """True iff ``raw`` canonicalizes to an allowlisted license (deny-by-default)."""
    return canonicalize_license(raw) is not None


def filter_licenses(frame: pd.DataFrame) -> pd.DataFrame:
    """Annotate the license table with ``license_canonical`` + ``allowed`` and keep allowed rows.

    Expects columns ``track_id`` and ``license`` (see :func:`read_license_table`). Returns a
    copy of the **allowlisted** rows only, with the canonical token added; the full annotated
    table (for an audit render) is available via :func:`license_audit`.
    """
    annotated = license_audit(frame)
    return annotated[annotated["allowed"]].drop(columns=["allowed"]).reset_index(drop=True)


def license_audit(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` with ``license_canonical`` (or ``''``) + ``allowed`` bool columns.

    The full allow/deny table the license-audit notebook renders (MASTER_PLAN §10;
    THEORY §5). Denied rows keep ``license_canonical == ''`` so the reason (unclear / ND /
    non-CC) stays inspectable next to the raw string.
    """
    _require_columns(frame, ("track_id", "license"))
    out = frame.copy()
    canonical = out["license"].map(canonicalize_license)
    out["license_canonical"] = canonical.fillna("")
    out["allowed"] = canonical.notna()
    return out


def screen_sample(track_ids: list[str], n: int, seed: int) -> list[str]:
    """Draw ``n`` track ids deterministically (seed) from ``track_ids`` (MASTER_PLAN §3.1).

    Input-order independent (the ids are de-duplicated and sorted before the draw) and
    fully reproducible: the same ``(track_ids, n, seed)`` always yields the same list, in
    the permutation draw order. ``n`` larger than the pool returns the whole (shuffled)
    pool. Uses a dedicated :class:`numpy.random.default_rng(seed)`; no global RNG touched.
    """
    ids = sorted(dict.fromkeys(str(t) for t in track_ids))
    take = min(int(n), len(ids))
    if take <= 0:
        return []
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(ids))[:take]
    return [ids[i] for i in order]


def build_manifest_frame(
    allowed: pd.DataFrame, screened_ids: list[str]
) -> pd.DataFrame:
    """Assemble the committed manifest (track ids + licenses + screen flags; never audio).

    ``allowed`` is the :func:`filter_licenses` output (track_id, license, license_canonical);
    ``screened_ids`` is the :func:`screen_sample` draw. Rows are sorted by track id for a
    stable, diff-friendly manifest; the ``screened`` column flags the deterministic sample.
    """
    screened = set(str(t) for t in screened_ids)
    out = allowed.copy()
    out["track_id"] = out["track_id"].astype(str)
    out = out.rename(columns={"license": "license_raw"})
    out["screened"] = out["track_id"].isin(screened)
    cols = ["track_id", "license_raw", "license_canonical", "screened"]
    return out[cols].sort_values("track_id").reset_index(drop=True)


def write_manifest(
    path: str | Path, manifest: pd.DataFrame, *, provenance: dict[str, object] | None = None
) -> None:
    """Write the committed track-ID manifest CSV with a ``#`` provenance header (no audio).

    The header records the license allowlist, the screen parameters, and the counts so the
    manifest is self-describing for a public audit (MASTER_PLAN §3.3, §11). Mirrors the
    ``#``-comment provenance style of ``splits.csv``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prov = provenance or {}
    n_screened = int(manifest["screened"].sum()) if len(manifest) else 0
    header_lines = [
        "# FMA license-safe track manifest — metadata only, NO AUDIO (MASTER_PLAN §3.1).",
        f"# allowlist: {sorted(ALLOWED_LICENSES)} (DENY-by-default; every -ND variant excluded, §5).",
        f"# allowlisted_tracks: {len(manifest)}  screened: {n_screened}",
    ]
    for key in ("source", "metadata_dir", "audio_dir", "screen", "seed"):
        if key in prov:
            header_lines.append(f"# {key}: {prov[key]}")
    header = "\n".join(header_lines) + "\n"
    path.write_text(header + manifest.to_csv(index=False), encoding="utf-8")


def read_license_table(csv_path: str | Path) -> pd.DataFrame:
    """Load a simplified ``track_id, license`` metadata table (fail-loud).

    The canonical schema the filter operates on: a CSV with (at least) ``track_id`` and
    ``license`` columns. The real FMA ``tracks.csv`` (a two-level-header monster) is flattened
    to this schema by :func:`flatten_fma_tracks_csv` at RUN-LATER time; every tested code path
    uses this simplified table. Raises ``FileNotFoundError``/``ValueError`` loudly.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"metadata table {csv_path} missing — fetch the FMA metadata and point "
            "--metadata-dir at it (RUN LATER; MASTER_PLAN §6 run book)."
        )
    frame = pd.read_csv(csv_path, dtype=str, comment="#")
    _require_columns(frame, ("track_id", "license"))
    return frame[["track_id", "license"]].copy()


def flatten_fma_tracks_csv(tracks_csv: str | Path) -> pd.DataFrame:  # pragma: no cover - RUN LATER
    """Flatten the real FMA ``tracks.csv`` (two-level header) to ``track_id, license``.

    **RUN LATER — needs the real FMA metadata.** FMA's ``tracks.csv`` uses a two-row column
    header with the track id as the index; the per-track license lives under the ``track``
    group's ``license`` field. This adapter emits the simplified table
    :func:`read_license_table` consumes. Untested in the CPU suite (no metadata present).
    """
    tracks_csv = Path(tracks_csv)
    if not tracks_csv.exists():
        raise FileNotFoundError(f"FMA tracks.csv missing at {tracks_csv} (RUN LATER)")
    frame = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])
    license_col = None
    for col in frame.columns:
        if isinstance(col, tuple) and col[0] == "track" and "license" in str(col[1]).lower():
            license_col = col
            break
    if license_col is None:
        raise ValueError(f"no ('track','license*') column found in {tracks_csv}")
    out = pd.DataFrame(
        {"track_id": frame.index.astype(str), "license": frame[license_col].astype(str)}
    )
    return out.reset_index(drop=True)


def prepare(
    metadata_table: pd.DataFrame, *, screen: int, seed: int, provenance: dict[str, object] | None = None
) -> tuple[pd.DataFrame, dict[str, int]]:
    """License-filter, screen-sample, and assemble the manifest frame + audit counts (pure).

    Returns ``(manifest_frame, counts)`` where ``counts`` has ``{n_total, n_allowed,
    n_denied, n_screened}``. This is the tested core of the CLI (no I/O).
    """
    audit = license_audit(metadata_table)
    allowed = audit[audit["allowed"]].drop(columns=["allowed"]).reset_index(drop=True)
    screened_ids = screen_sample(allowed["track_id"].tolist(), screen, seed)
    manifest = build_manifest_frame(allowed, screened_ids)
    counts = {
        "n_total": int(len(audit)),
        "n_allowed": int(len(allowed)),
        "n_denied": int(len(audit) - len(allowed)),
        "n_screened": int(len(screened_ids)),
    }
    return manifest, counts


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"metadata table missing columns {missing} (have {list(frame.columns)})")


def main(argv: list[str] | None = None) -> None:
    """CLI (RUN LATER — needs real FMA metadata; writes ``<out>/manifest.csv``, no audio)."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--metadata-dir", required=True, help="dir holding the FMA tracks.csv metadata")
    parser.add_argument("--audio-dir", default=None,
                        help="FMA audio dir (recorded in provenance; NEVER read here — teacher step, RUN LATER)")
    parser.add_argument("--screen", type=int, default=1200, help="N_screen deterministic sample size (§3.1)")
    parser.add_argument("--seed", type=int, default=0, help="screen-sample seed (§3.1)")
    parser.add_argument("--out", required=True, help="output pseudo root; manifest written to <out>/manifest.csv")
    parser.add_argument("--tracks-csv", default=None,
                        help="explicit path to a simplified track_id,license table (else <metadata-dir>/tracks.csv)")
    args = parser.parse_args(argv)

    # Fail loud without real metadata (the whole point — no audio, no guessing).
    if args.tracks_csv is not None:
        table = read_license_table(args.tracks_csv)
    else:
        real = Path(args.metadata_dir) / "tracks.csv"
        if not real.exists():
            raise SystemExit(
                f"no tracks.csv under {args.metadata_dir!r} — fetch the FMA metadata first "
                "(RUN LATER; MASTER_PLAN §6). Nothing was written."
            )
        try:
            table = read_license_table(real)  # simplified schema
        except ValueError:
            table = flatten_fma_tracks_csv(real)  # real two-level-header FMA metadata

    manifest, counts = prepare(table, screen=args.screen, seed=args.seed,
                               provenance={"source": "mdeff/fma", "metadata_dir": args.metadata_dir,
                                           "audio_dir": args.audio_dir, "screen": args.screen, "seed": args.seed})
    out_path = Path(args.out) / "manifest.csv"
    write_manifest(out_path, manifest,
                   provenance={"source": "mdeff/fma", "metadata_dir": args.metadata_dir,
                               "audio_dir": args.audio_dir, "screen": args.screen, "seed": args.seed})
    print(
        f"wrote {out_path}: {counts['n_allowed']} allowlisted / {counts['n_total']} total "
        f"({counts['n_denied']} denied), {counts['n_screened']} screened (seed {args.seed}). NO AUDIO."
    )


if __name__ == "__main__":
    main()
