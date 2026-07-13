"""Direction 02 nested-subset tooling (G0, §3.2, §4).

Determinism (same manifest + seed -> identical draw), nesting
(n21 ⊂ n43 ⊂ n64), fail-loud on placeholder manifests, and the balance report.
The core functions live in ``scripts/make_subsets.py`` and are imported here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import make_subsets as ms  # noqa: E402

TRAIN = [f"Artist{i:02d} - Song {i}" for i in range(86)]


def test_make_nested_subsets_is_deterministic() -> None:
    a = ms.make_nested_subsets(TRAIN, [21, 43, 64], seed=0)
    b = ms.make_nested_subsets(TRAIN, [21, 43, 64], seed=0)
    assert a == b
    assert {k: len(v) for k, v in a.items()} == {21: 21, 43: 43, 64: 64}


def test_subsets_are_nested() -> None:
    s = ms.make_nested_subsets(TRAIN, [21, 43, 64], seed=0)
    # nested as sets, and as prefixes of one shuffle (the strong form)
    assert set(s[21]) <= set(s[43]) <= set(s[64]) <= set(TRAIN)
    assert s[43][:21] == s[21]
    assert s[64][:43] == s[43]


def test_different_seed_changes_the_draw() -> None:
    assert ms.make_nested_subsets(TRAIN, [21], seed=0) != ms.make_nested_subsets(TRAIN, [21], seed=1)


def test_no_duplicates_within_a_subset() -> None:
    s = ms.make_nested_subsets(TRAIN, [21, 43, 64], seed=7)
    for tracks in s.values():
        assert len(tracks) == len(set(tracks))


def test_oversized_request_fails_loud() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        ms.make_nested_subsets(TRAIN, [21, 200], seed=0)


def test_placeholder_manifest_fails_loud() -> None:
    with pytest.raises(ValueError, match="placeholder"):
        ms.assert_materialized(["Real Artist - Song", "__PLACEHOLDER_train_003"])
    ms.assert_materialized(["Real Artist - Song"])  # a materialized list does not raise


def test_write_subsets_roundtrips_names_only(tmp_path) -> None:
    s = ms.make_nested_subsets(TRAIN, [21, 43], seed=0)
    paths = ms.write_subsets(s, tmp_path)
    assert {p.name for p in paths} == {"n21.csv", "n43.csv"}
    # the committed CSV is a names-only 'track' column readable by the loader
    from singnet.data import load_track_allowlist

    loaded = load_track_allowlist(tmp_path / "n21.csv")
    assert loaded == s[21]


def test_balance_report_columns_depend_on_index() -> None:
    s = ms.make_nested_subsets(TRAIN, [21, 43], seed=0)
    # no index -> song counts only
    counts_only = ms.subset_balance(s, index=None)
    assert list(counts_only.columns) == ["subset", "n_songs"]
    # index with duration + vocal activity -> those columns appear
    index = {t: {"duration_s": 120.0 + i, "vocal_activity": 0.3} for i, t in enumerate(TRAIN)}
    rich = ms.subset_balance(s, index=index)
    assert {"total_duration_min", "mean_vocal_activity"} <= set(rich.columns)
    assert (rich["n_songs"] == pd.Series([21, 43])).all()
