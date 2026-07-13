"""Split manifest guard: the training loader refuses test rows (G0, §4.2)."""

from __future__ import annotations

import pytest

from singnet.data import Manifest, MusdbChunks, TestRowError


def test_counts_and_tracks(synthetic_manifest: Manifest) -> None:
    assert synthetic_manifest.counts == {"train": 2, "valid": 2, "test": 1}
    assert set(synthetic_manifest.tracks_for("valid")) == {"train_02", "train_03"}


def test_trainable_tracks_refuses_test_split(synthetic_manifest: Manifest) -> None:
    with pytest.raises(TestRowError):
        synthetic_manifest.trainable_tracks("test")


def test_assert_no_test_rows_flags_test_track(synthetic_manifest: Manifest) -> None:
    with pytest.raises(TestRowError):
        synthetic_manifest.assert_no_test_rows(["train_00", "held_out_00"])
    # a purely trainable list passes silently
    synthetic_manifest.assert_no_test_rows(["train_00", "train_02"])


def test_dataset_refuses_test_split(synthetic_store, synthetic_manifest) -> None:
    with pytest.raises(TestRowError):
        MusdbChunks(synthetic_store, synthetic_manifest, "test", seed=0)


def test_is_test_predicate(synthetic_manifest: Manifest) -> None:
    assert synthetic_manifest.is_test("held_out_00") is True
    assert synthetic_manifest.is_test("train_00") is False


def test_unknown_split_rejected() -> None:
    import pandas as pd

    with pytest.raises(ValueError):
        Manifest(pd.DataFrame([{"track": "x", "split": "bogus"}]))
