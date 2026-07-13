"""Direction 05 registry schema additions (G0, §5).

The seven appended columns (``domain, recipe, rank, lr, trainable_params,
trainable_share, peak_vram_gb``) round-trip; an older Direction-01/02/03 registry
written without them still reads back with the full schema (NaN-filled) — the
backward-compatibility guarantee. No existing column moved.
"""

from __future__ import annotations

import math

import pandas as pd

from singnet.train import REGISTRY_COLUMNS, RunRecord, read_registry, upsert_run

D05_BLOCK = ("domain", "recipe", "rank", "lr", "trainable_params", "trainable_share", "peak_vram_gb")


def test_d05_columns_are_the_appended_tail() -> None:
    for col in D05_BLOCK:
        assert col in REGISTRY_COLUMNS
    assert REGISTRY_COLUMNS[-7:] == D05_BLOCK
    assert REGISTRY_COLUMNS[-8] == "base_width"  # the D03 append still precedes them


def test_runrecord_roundtrips_peft_metadata(tmp_path) -> None:
    path = tmp_path / "registry.csv"
    rec = RunRecord(
        run_id="t1_aac64_lora16_seed0_ft6k_h", arm="lora16", seed=0, budget=6000, config_hash="h",
        domain="t1_aac64", recipe="lora16", rank=16, lr=1e-3,
        trainable_params=431_520, trainable_share=0.048522, peak_vram_gb=2.3,
        best_val_sisdr=5.7,
    )
    upsert_run(path, rec)
    frame = read_registry(path)
    assert list(frame.columns) == list(REGISTRY_COLUMNS)
    row = frame.iloc[0]
    assert row["domain"] == "t1_aac64" and row["recipe"] == "lora16"
    assert int(row["rank"]) == 16 and int(row["trainable_params"]) == 431_520
    assert abs(float(row["trainable_share"]) - 0.048522) < 1e-9
    assert abs(float(row["peak_vram_gb"]) - 2.3) < 1e-9


def test_defaults_are_not_a_peft_run() -> None:
    rec = RunRecord(run_id="r", arm="l1mag", seed=0, budget=16000, config_hash="h")
    assert rec.domain == "" and rec.recipe == "" and rec.rank == 0
    assert math.isnan(rec.lr) and math.isnan(rec.trainable_share) and math.isnan(rec.peak_vram_gb)
    assert rec.trainable_params == 0


def test_old_registry_without_d05_columns_reads_back(tmp_path) -> None:
    # A legacy Direction-01/02/03 CSV lacking the seven D05 columns.
    path = tmp_path / "legacy.csv"
    legacy_cols = [c for c in REGISTRY_COLUMNS if c not in D05_BLOCK]
    pd.DataFrame([{c: "" for c in legacy_cols}]).to_csv(path, index=False)
    rec = RunRecord(run_id="new", arm="lora4", seed=0, budget=6000, config_hash="h2",
                    domain="t2", recipe="lora4", rank=4)
    upsert_run(path, rec)
    out = read_registry(path)
    assert list(out.columns) == list(REGISTRY_COLUMNS)
    new_row = out[out["run_id"] == "new"].iloc[0]
    assert new_row["recipe"] == "lora4" and int(new_row["rank"]) == 4
    # the legacy row's D05 columns are absent/NaN — honest, not fabricated.
    legacy_row = out[out["run_id"] != "new"]
    if len(legacy_row):
        assert math.isnan(pd.to_numeric(legacy_row.iloc[0]["trainable_share"], errors="coerce"))
