"""Direction 10 registry schema additions (G0, §5).

The five appended columns (``data_source, p_fma, n_pseudo_clips, teacher_version,
teacher_consistency_db``) round-trip; an older Direction-01..08 registry written without them
still reads back with the full schema (defaults-filled) — the backward-compatibility
guarantee. No existing column moved (the D08 block still ends where it did, now followed by
the D10 block, located by index).
"""

from __future__ import annotations

import math

import pandas as pd

from singnet.train import REGISTRY_COLUMNS, RunRecord, read_registry, upsert_run

D10_BLOCK = ("data_source", "p_fma", "n_pseudo_clips", "teacher_version", "teacher_consistency_db")


def test_d10_columns_are_the_appended_tail() -> None:
    for col in D10_BLOCK:
        assert col in REGISTRY_COLUMNS
    assert REGISTRY_COLUMNS[-5:] == D10_BLOCK               # the D10 block ends the schema
    assert REGISTRY_COLUMNS[-6] == "best_val_slr"           # the D08 block still precedes it
    # base_width (D03) < domain (D05) < epsilon (D06) < policy (D08) < data_source (D10).
    assert REGISTRY_COLUMNS.index("policy") < REGISTRY_COLUMNS.index("data_source")
    assert REGISTRY_COLUMNS.index("epsilon") < REGISTRY_COLUMNS.index("policy")


def test_runrecord_roundtrips_pseudo_metadata(tmp_path) -> None:
    path = tmp_path / "registry.csv"
    rec = RunRecord(
        run_id="mixed_seed0_reduced_h", arm="mixed", seed=0, budget=16000, config_hash="h",
        data_source="mixed", p_fma=0.5, n_pseudo_clips=800,
        teacher_version="htdemucs", teacher_consistency_db=-38.4, best_val_sisdr=5.7,
    )
    upsert_run(path, rec)
    frame = read_registry(path)
    assert list(frame.columns) == list(REGISTRY_COLUMNS)
    row = frame.iloc[0]
    assert row["data_source"] == "mixed"
    assert abs(float(row["p_fma"]) - 0.5) < 1e-9
    assert int(row["n_pseudo_clips"]) == 800
    assert row["teacher_version"] == "htdemucs"
    assert abs(float(row["teacher_consistency_db"]) + 38.4) < 1e-9


def test_defaults_are_musdb_only() -> None:
    rec = RunRecord(run_id="r", arm="l1mag", seed=0, budget=16000, config_hash="h")
    assert rec.data_source == "musdb"
    assert math.isnan(rec.p_fma) and rec.n_pseudo_clips == 0
    assert rec.teacher_version == "" and math.isnan(rec.teacher_consistency_db)


def test_old_registry_without_d10_columns_reads_back(tmp_path) -> None:
    # A legacy Direction-01..08 CSV lacking the five D10 columns.
    path = tmp_path / "legacy.csv"
    legacy_cols = [c for c in REGISTRY_COLUMNS if c not in D10_BLOCK]
    pd.DataFrame([{c: "" for c in legacy_cols}]).to_csv(path, index=False)
    rec = RunRecord(run_id="new", arm="mixed", seed=0, budget=16000, config_hash="h2",
                    data_source="mixed", p_fma=0.5, n_pseudo_clips=800)
    upsert_run(path, rec)
    out = read_registry(path)
    assert list(out.columns) == list(REGISTRY_COLUMNS)
    new_row = out[out["run_id"] == "new"].iloc[0]
    assert new_row["data_source"] == "mixed"
    # the legacy row's D10 columns are absent/NaN — honest, not fabricated.
    legacy_row = out[out["run_id"] != "new"]
    if len(legacy_row):
        assert math.isnan(pd.to_numeric(legacy_row.iloc[0]["p_fma"], errors="coerce"))
