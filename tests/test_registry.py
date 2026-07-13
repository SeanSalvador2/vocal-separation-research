"""Run registry read/write and config-hash stability (G0, §7.1)."""

from __future__ import annotations

from singnet.train import REGISTRY_COLUMNS, RunRecord, get_run, read_registry, upsert_run
from singnet.utils.config import hash_config


def test_config_hash_is_order_independent() -> None:
    a = {"arm": "l1mag", "seed": 0, "steps": 16000, "loss_kwargs": {"lam": 0.5}}
    b = {"loss_kwargs": {"lam": 0.5}, "steps": 16000, "seed": 0, "arm": "l1mag"}
    assert hash_config(a) == hash_config(b)


def test_config_hash_ignores_non_identity_keys() -> None:
    base = {"arm": "l1mag", "seed": 0, "steps": 16000}
    with_env = {**base, "gpu": "T4", "output_dir": "/drive/run1", "shard_root": "/x"}
    assert hash_config(base) == hash_config(with_env)


def test_config_hash_changes_with_content() -> None:
    a = {"arm": "l1mag", "seed": 0}
    b = {"arm": "sisdr", "seed": 0}
    c = {"arm": "l1mag", "seed": 1}
    assert hash_config(a) != hash_config(b)
    assert hash_config(a) != hash_config(c)


def test_registry_roundtrip(tmp_path) -> None:
    path = tmp_path / "registry.csv"
    rec = RunRecord(run_id="l1mag_seed0_reduced_abc123", arm="l1mag", seed=0, budget=16000,
                    config_hash="abc123", best_val_sisdr=4.2)
    upsert_run(path, rec)
    frame = read_registry(path)
    assert list(frame.columns) == list(REGISTRY_COLUMNS)
    assert len(frame) == 1
    row = get_run(path, rec.run_id)
    assert row is not None and row["arm"] == "l1mag" and float(row["best_val_sisdr"]) == 4.2


def test_registry_upsert_is_idempotent(tmp_path) -> None:
    path = tmp_path / "registry.csv"
    rec = RunRecord(run_id="r1", arm="sisdr", seed=1, budget=16000, config_hash="h", steps_done=1000)
    upsert_run(path, rec)
    rec.steps_done = 16000  # resumed run updates in place
    rec.best_val_sisdr = 5.1
    upsert_run(path, rec)
    frame = read_registry(path)
    assert len(frame) == 1
    assert int(frame.iloc[0]["steps_done"]) == 16000
    assert float(frame.iloc[0]["best_val_sisdr"]) == 5.1


def test_empty_registry_has_schema(tmp_path) -> None:
    frame = read_registry(tmp_path / "absent.csv")
    assert list(frame.columns) == list(REGISTRY_COLUMNS)
    assert len(frame) == 0
