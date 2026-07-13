"""Direction 02 registry schema additions (G0, §6).

The four appended columns (`aug_remix, aug_gain, aug_flip, n_songs`) round-trip,
and an older Direction-01 registry written without them still reads back with the
full schema (NaN-filled) — the backward-compatibility guarantee.
"""

from __future__ import annotations

import math

import pandas as pd

from singnet.train import REGISTRY_COLUMNS, RunRecord, read_registry, upsert_run


def test_new_columns_are_in_the_schema() -> None:
    for col in ("aug_remix", "aug_gain", "aug_flip", "n_songs"):
        assert col in REGISTRY_COLUMNS
    # The four D02 columns are appended as a contiguous block after the original
    # Direction-01 schema (no existing column position changed). Direction 03
    # appends base_width after them, and Direction 05 appends a further seven
    # (domain … peak_vram_gb) — each still a backward-compatible append.
    d02_block = ("aug_remix", "aug_gain", "aug_flip", "n_songs")
    start = REGISTRY_COLUMNS.index("aug_remix")
    assert REGISTRY_COLUMNS[start : start + 4] == d02_block
    assert REGISTRY_COLUMNS[start + 4] == "base_width"  # the Direction-03 append
    # the Direction-05 block follows base_width, ending the schema.
    assert REGISTRY_COLUMNS[start + 5 :] == (
        "domain", "recipe", "rank", "lr", "trainable_params", "trainable_share", "peak_vram_gb",
    )
    assert REGISTRY_COLUMNS[-1] == "peak_vram_gb"


def test_runrecord_roundtrips_switchboard(tmp_path) -> None:
    path = tmp_path / "registry.csv"
    rec = RunRecord(run_id="loo_no_remix", arm="l1mag", seed=0, budget=16000, config_hash="h",
                    aug_remix=False, aug_gain=True, aug_flip=True, n_songs=86)
    upsert_run(path, rec)
    frame = read_registry(path)
    assert list(frame.columns) == list(REGISTRY_COLUMNS)
    row = frame.iloc[0]
    assert bool(row["aug_remix"]) is False and bool(row["aug_gain"]) is True
    assert int(row["n_songs"]) == 86


def test_defaults_are_the_full_recipe() -> None:
    rec = RunRecord(run_id="r", arm="l1mag", seed=0, budget=16000, config_hash="h")
    assert (rec.aug_remix, rec.aug_gain, rec.aug_flip, rec.n_songs) == (True, True, True, 86)


def test_old_registry_without_new_columns_reads_back(tmp_path) -> None:
    # Simulate a legacy Direction-01 CSV lacking the four new columns.
    path = tmp_path / "legacy.csv"
    legacy_cols = [c for c in REGISTRY_COLUMNS if c not in ("aug_remix", "aug_gain", "aug_flip", "n_songs")]
    pd.DataFrame([{c: "" for c in legacy_cols}]).to_csv(path, index=False)
    frame = read_registry(path)
    # read_registry does not force the schema, but upsert/write reindex does:
    rec = RunRecord(run_id="new", arm="sisdr", seed=1, budget=16000, config_hash="h2", n_songs=21)
    upsert_run(path, rec)
    out = read_registry(path)
    assert list(out.columns) == list(REGISTRY_COLUMNS)
    new_row = out[out["run_id"] == "new"].iloc[0]
    assert int(new_row["n_songs"]) == 21
    # the legacy row's new columns are absent/NaN — honest, not fabricated
    legacy_row = out[out["run_id"] != "new"]
    if len(legacy_row):
        assert math.isnan(pd.to_numeric(legacy_row.iloc[0]["n_songs"], errors="coerce"))
