"""Direction 08 registry schema additions (G0, §6).

The five appended columns (``policy, theta_db, floor_lambda, silent_exposure_observed,
best_val_slr``) round-trip; an older Direction-01..06 registry written without them still
reads back with the full schema (NaN/uniform-filled) — the backward-compatibility
guarantee. No existing column moved (the D06 block still ends where it did, now followed
by the D08 block).
"""

from __future__ import annotations

import math

import pandas as pd

from singnet.train import REGISTRY_COLUMNS, RunRecord, read_registry, upsert_run

D08_BLOCK = ("policy", "theta_db", "floor_lambda", "silent_exposure_observed", "best_val_slr")


def test_d08_columns_are_the_appended_tail() -> None:
    for col in D08_BLOCK:
        assert col in REGISTRY_COLUMNS
    # Direction 10 later appends a further five columns, so the D08 block is located by index
    # rather than as the literal tail (no D08 column moved — the same no-column-moved
    # guarantee, now aware of the D10 append; the D06->D08 precedent applied again).
    start = REGISTRY_COLUMNS.index("policy")
    assert REGISTRY_COLUMNS[start : start + 5] == D08_BLOCK
    assert REGISTRY_COLUMNS[start - 1] == "trim_energy_stats_path"  # the D06 block still precedes it
    # base_width (D03) < domain (D05) < epsilon (D06) < policy (D08): appends never reorder.
    assert REGISTRY_COLUMNS.index("epsilon") < REGISTRY_COLUMNS.index("policy")


def test_runrecord_roundtrips_sampling_metadata(tmp_path) -> None:
    path = tmp_path / "registry.csv"
    rec = RunRecord(
        run_id="energy_seed0_reduced_h", arm="energy", seed=0, budget=16000, config_hash="h",
        policy="energy", theta_db=-60.0, floor_lambda=0.1,
        silent_exposure_observed=0.083, best_val_slr=-14.2, best_val_sisdr=5.4,
    )
    upsert_run(path, rec)
    frame = read_registry(path)
    assert list(frame.columns) == list(REGISTRY_COLUMNS)
    row = frame.iloc[0]
    assert row["policy"] == "energy"
    assert abs(float(row["theta_db"]) + 60.0) < 1e-9
    assert abs(float(row["floor_lambda"]) - 0.1) < 1e-9
    assert abs(float(row["silent_exposure_observed"]) - 0.083) < 1e-9
    assert abs(float(row["best_val_slr"]) + 14.2) < 1e-9


def test_defaults_are_the_uniform_baseline() -> None:
    rec = RunRecord(run_id="r", arm="l1mag", seed=0, budget=16000, config_hash="h")
    assert rec.policy == "uniform"
    assert math.isnan(rec.theta_db) and math.isnan(rec.floor_lambda)
    assert math.isnan(rec.silent_exposure_observed) and math.isnan(rec.best_val_slr)


def test_old_registry_without_d08_columns_reads_back(tmp_path) -> None:
    # A legacy Direction-01..06 CSV lacking the five D08 columns.
    path = tmp_path / "legacy.csv"
    legacy_cols = [c for c in REGISTRY_COLUMNS if c not in D08_BLOCK]
    pd.DataFrame([{c: "" for c in legacy_cols}]).to_csv(path, index=False)
    rec = RunRecord(run_id="new", arm="drop", seed=0, budget=16000, config_hash="h2",
                    policy="drop", theta_db=-60.0)
    upsert_run(path, rec)
    out = read_registry(path)
    assert list(out.columns) == list(REGISTRY_COLUMNS)
    new_row = out[out["run_id"] == "new"].iloc[0]
    assert new_row["policy"] == "drop"
    # the legacy row's D08 columns are absent/NaN — honest, not fabricated.
    legacy_row = out[out["run_id"] != "new"]
    if len(legacy_row):
        assert math.isnan(pd.to_numeric(legacy_row.iloc[0]["best_val_slr"], errors="coerce"))
