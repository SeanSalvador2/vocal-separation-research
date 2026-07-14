"""Direction 06 registry schema additions (G0, §5).

The four appended columns (``epsilon, trim_q, kept_fraction_observed,
trim_energy_stats_path``) round-trip; an older Direction-01..05 registry written
without them still reads back with the full schema (NaN-filled) — the
backward-compatibility guarantee. No existing column moved.
"""

from __future__ import annotations

import math

import pandas as pd

from singnet.train import REGISTRY_COLUMNS, RunRecord, read_registry, upsert_run

D06_BLOCK = ("epsilon", "trim_q", "kept_fraction_observed", "trim_energy_stats_path")


def test_d06_columns_are_the_appended_tail() -> None:
    for col in D06_BLOCK:
        assert col in REGISTRY_COLUMNS
    assert REGISTRY_COLUMNS[-4:] == D06_BLOCK
    assert REGISTRY_COLUMNS[-5] == "peak_vram_gb"  # the D05 block still precedes them
    # base_width (D03) precedes the D05 block, which precedes the D06 block.
    assert REGISTRY_COLUMNS.index("base_width") < REGISTRY_COLUMNS.index("domain")
    assert REGISTRY_COLUMNS.index("domain") < REGISTRY_COLUMNS.index("epsilon")


def test_runrecord_roundtrips_bleed_metadata(tmp_path) -> None:
    path = tmp_path / "registry.csv"
    rec = RunRecord(
        run_id="trim30_seed0_reduced_h", arm="trim30", seed=0, budget=16000, config_hash="h",
        epsilon=0.30, trim_q=0.30, kept_fraction_observed=0.6875,
        trim_energy_stats_path="checkpoints/h/trim_energy_stats.csv",
        best_val_sisdr=5.1,
    )
    upsert_run(path, rec)
    frame = read_registry(path)
    assert list(frame.columns) == list(REGISTRY_COLUMNS)
    row = frame.iloc[0]
    assert abs(float(row["epsilon"]) - 0.30) < 1e-9
    assert abs(float(row["trim_q"]) - 0.30) < 1e-9
    assert abs(float(row["kept_fraction_observed"]) - 0.6875) < 1e-9
    assert row["trim_energy_stats_path"] == "checkpoints/h/trim_energy_stats.csv"


def test_defaults_are_not_a_bleed_run() -> None:
    rec = RunRecord(run_id="r", arm="l1mag", seed=0, budget=16000, config_hash="h")
    assert math.isnan(rec.epsilon) and math.isnan(rec.trim_q)
    assert math.isnan(rec.kept_fraction_observed) and rec.trim_energy_stats_path == ""


def test_old_registry_without_d06_columns_reads_back(tmp_path) -> None:
    # A legacy Direction-01..05 CSV lacking the four D06 columns.
    path = tmp_path / "legacy.csv"
    legacy_cols = [c for c in REGISTRY_COLUMNS if c not in D06_BLOCK]
    pd.DataFrame([{c: "" for c in legacy_cols}]).to_csv(path, index=False)
    rec = RunRecord(run_id="new", arm="bleed30", seed=0, budget=16000, config_hash="h2", epsilon=0.30)
    upsert_run(path, rec)
    out = read_registry(path)
    assert list(out.columns) == list(REGISTRY_COLUMNS)
    new_row = out[out["run_id"] == "new"].iloc[0]
    assert abs(float(new_row["epsilon"]) - 0.30) < 1e-9
    # the legacy row's D06 columns are absent/NaN — honest, not fabricated.
    legacy_row = out[out["run_id"] != "new"]
    if len(legacy_row):
        assert math.isnan(pd.to_numeric(legacy_row.iloc[0]["epsilon"], errors="coerce"))
