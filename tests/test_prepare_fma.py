"""FMA metadata filter, license allow/deny table, screen determinism (G0, §3.1).

Direction-10 stage-D items (MASTER_PLAN §3.1, §8): the license decision is an
allowlist, DENY-by-default — every allowlisted license passes; every ``-ND`` variant,
empty/garbage string, and ``All Rights Reserved`` is denied; the screen sample is
deterministic in ``(ids, n, seed)``; the manifest carries IDs + licenses + screen flags
and **never** audio. The core lives in ``scripts/prepare_fma.py`` and is pure/fixture-tested.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import prepare_fma as pf  # noqa: E402


# --- the allow/deny license table (the load-bearing legal call) -------------

ALLOWED_CASES = {
    "CC0 1.0 Universal": "CC0",
    "Public Domain Mark 1.0": "CC0",
    "Creative Commons Zero": "CC0",
    "Attribution 3.0 United States": "CC-BY",
    "CC BY 4.0": "CC-BY",
    "http://creativecommons.org/licenses/by/3.0/": "CC-BY",
    "Attribution-ShareAlike 3.0": "CC-BY-SA",
    "Attribution-NonCommercial 3.0 International": "CC-BY-NC",
    "Attribution-NonCommercial-ShareAlike 3.0": "CC-BY-NC-SA",
    "http://creativecommons.org/licenses/by-nc-sa/3.0/": "CC-BY-NC-SA",
}

DENIED_CASES = [
    "Attribution-NoDerivatives 4.0 International",       # -ND: the load-bearing exclusion
    "Attribution-NonCommercial-NoDerivatives 4.0",      # -NC-ND
    "http://creativecommons.org/licenses/by-nc-nd/4.0/",  # URL -ND
    "CC BY-ND 4.0",
    "All rights reserved",
    "All Rights Reserved",
    "",                                                 # empty
    "   ",                                              # whitespace
    "totally made up license",                          # garbage / non-CC
    "Some Proprietary EULA v2",
]


@pytest.mark.parametrize("raw,expected", list(ALLOWED_CASES.items()))
def test_allowlisted_licenses_pass(raw: str, expected: str) -> None:
    assert pf.canonicalize_license(raw) == expected
    assert pf.license_allowed(raw) is True
    assert expected in pf.ALLOWED_LICENSES


@pytest.mark.parametrize("raw", DENIED_CASES)
def test_denied_licenses_are_refused(raw: str) -> None:
    assert pf.canonicalize_license(raw) is None
    assert pf.license_allowed(raw) is False


def test_non_string_licenses_denied() -> None:
    for bad in (None, float("nan"), 42, ["cc-by"]):
        assert pf.canonicalize_license(bad) is None


def test_nd_is_denied_even_with_attribution() -> None:
    # ND dominates: an otherwise-allowlisted BY/NC/SA string with a NoDerivatives clause
    # is still denied (separated stems are derivative works, §5).
    assert pf.canonicalize_license("Attribution-ShareAlike-NoDerivatives") is None
    assert pf.canonicalize_license("by-sa-nd") is None


# --- filter + audit ---------------------------------------------------------

def _metadata() -> pd.DataFrame:
    rows = [
        {"track_id": "10", "license": "Attribution-NonCommercial-ShareAlike 3.0"},  # allow
        {"track_id": "11", "license": "CC0 1.0 Universal"},                          # allow
        {"track_id": "12", "license": "Attribution-NoDerivatives 4.0 International"},  # deny (ND)
        {"track_id": "13", "license": "All Rights Reserved"},                        # deny
        {"track_id": "14", "license": ""},                                           # deny (empty)
        {"track_id": "15", "license": "CC BY 4.0"},                                  # allow
    ]
    return pd.DataFrame(rows)


def test_filter_keeps_only_allowlisted() -> None:
    allowed = pf.filter_licenses(_metadata())
    assert set(allowed["track_id"]) == {"10", "11", "15"}
    assert "license_canonical" in allowed.columns
    assert set(allowed["license_canonical"]) == {"CC-BY-NC-SA", "CC0", "CC-BY"}


def test_audit_table_flags_every_row() -> None:
    audit = pf.license_audit(_metadata())
    assert len(audit) == 6
    assert int(audit["allowed"].sum()) == 3
    # denied rows keep an empty canonical token (reason stays inspectable next to the raw)
    denied = audit[~audit["allowed"]]
    assert set(denied["track_id"]) == {"12", "13", "14"}
    assert (denied["license_canonical"] == "").all()


def test_filter_missing_columns_raises() -> None:
    with pytest.raises(ValueError):
        pf.filter_licenses(pd.DataFrame([{"id": "1", "lic": "CC0"}]))


# --- deterministic screen sample --------------------------------------------

def test_screen_sample_is_deterministic() -> None:
    ids = [str(i) for i in range(50)]
    a = pf.screen_sample(ids, 12, seed=0)
    b = pf.screen_sample(ids, 12, seed=0)
    assert a == b                       # reproducible in (ids, n, seed)
    assert len(a) == 12
    assert set(a) <= set(ids)           # a genuine subset
    assert len(set(a)) == 12            # no duplicates


def test_screen_sample_is_input_order_independent() -> None:
    ids = [str(i) for i in range(30)]
    assert pf.screen_sample(ids, 8, seed=3) == pf.screen_sample(list(reversed(ids)), 8, seed=3)


def test_screen_sample_seed_changes_the_draw() -> None:
    ids = [str(i) for i in range(100)]
    assert pf.screen_sample(ids, 20, seed=0) != pf.screen_sample(ids, 20, seed=1)


def test_screen_sample_caps_at_pool_size() -> None:
    ids = ["a", "b", "c"]
    drawn = pf.screen_sample(ids, 10, seed=0)
    assert sorted(drawn) == ["a", "b", "c"]


# --- prepare (filter + screen + manifest) + writer --------------------------

def test_prepare_assembles_manifest_and_counts() -> None:
    manifest, counts = pf.prepare(_metadata(), screen=2, seed=0)
    assert counts == {"n_total": 6, "n_allowed": 3, "n_denied": 3, "n_screened": 2}
    assert list(manifest.columns) == ["track_id", "license_raw", "license_canonical", "screened"]
    assert len(manifest) == 3                        # only allowlisted rows
    assert int(manifest["screened"].sum()) == 2      # exactly the screen sample flagged


def test_write_manifest_roundtrip_no_audio(tmp_path) -> None:
    manifest, _ = pf.prepare(_metadata(), screen=2, seed=1)
    out = tmp_path / "pseudo" / "manifest.csv"
    pf.write_manifest(out, manifest, provenance={"source": "mdeff/fma", "seed": 1, "screen": 2})
    text = out.read_text()
    assert text.startswith("#")                       # provenance header present
    assert "NO AUDIO" in text
    back = pd.read_csv(out, comment="#", dtype=str)
    assert list(back.columns) == ["track_id", "license_raw", "license_canonical", "screened"]
    # no audio bytes: the manifest is a small text table of ids + licenses only
    assert ".wav" not in text and ".mp3" not in text


def test_read_license_table_fails_loud_without_metadata(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        pf.read_license_table(tmp_path / "does_not_exist.csv")


def test_read_license_table_reads_simplified_schema(tmp_path) -> None:
    path = tmp_path / "tracks.csv"
    _metadata().to_csv(path, index=False)
    table = pf.read_license_table(path)
    assert list(table.columns) == ["track_id", "license"]
    assert len(table) == 6
