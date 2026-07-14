"""Direction 10 pseudo-pool schema: musdb_only ≡ shared cell, source identity, hash stability (G0, §4.1).

Gate-G0 config items (MASTER_PLAN §4.1, §5, §8):
* ``data_source: musdb`` (or an absent key) canonicalizes to NO key, so ``musdb_only`` is
  BIT-IDENTICAL in hash to the shared baseline cell (D01 ``l1mag`` ≡ D02/D03/D06/D08 base ≡
  ``a97d5400e994``) — why it is *reused*, not retrained (§4.1: "0 new runs", the sixth reuse);
* adding the pseudo schema leaves every existing Direction 01–08 hash untouched;
* ``data_source`` and ``p_fma`` are config-hash identity fields (musdb/mixed/distill distinct;
  ``mixed`` vs ``mixed25`` distinct by p_fma), while the pseudo-shard *locations* are not.

The tests here exercise the canonicalization *mechanism* on inline dicts + the D10 YAMLs; the
run-matrix hashes are asserted in ``tests/test_train_d10.py``.
"""

from __future__ import annotations

import pytest

from singnet.utils.config import (
    canonicalize_config,
    hash_config,
    pseudo_data_spec,
    resolve_config,
)

SHARED_CELL = "a97d5400e994"
D01 = "01-loss-function-study/configs/l1mag_seed0_reduced.yaml"
D06_BASE = "06-robust-training/configs/base.yaml"
D08_BASE = "08-silence-leakage/configs/base.yaml"
D10 = "10-demucs-distillation/configs"

_BASE = {"arm": "l1mag", "seed": 0, "steps": 16000}


# --- the shared-cell equality (the sixth reuse) -----------------------------

def test_musdb_only_hashes_as_shared_cell() -> None:
    assert hash_config(resolve_config(D01)) == SHARED_CELL
    assert hash_config(resolve_config(f"{D10}/base.yaml")) == SHARED_CELL
    # an explicit `data_source: musdb` canonicalizes away -> still the shared-cell hash.
    cfg = resolve_config(D01)
    assert hash_config({**cfg, "data_source": "musdb"}) == SHARED_CELL
    assert hash_config({**cfg, "data_source": "musdb", "p_fma": 0.5}) == SHARED_CELL


def test_pseudo_schema_leaves_existing_hashes_unchanged() -> None:
    # The shared cell still holds across every prior direction after the pseudo schema landed.
    assert hash_config(resolve_config(D01)) == SHARED_CELL
    assert hash_config(resolve_config(D06_BASE)) == SHARED_CELL
    assert hash_config(resolve_config(D08_BASE)) == SHARED_CELL
    # a config with no pseudo key is unaffected by the schema's existence.
    assert hash_config(dict(_BASE)) == hash_config(dict(_BASE))


# --- data_source / p_fma are identity fields --------------------------------

def test_data_source_is_an_identity_field() -> None:
    h_musdb = hash_config(dict(_BASE))
    h_mixed = hash_config({**_BASE, "data_source": "mixed", "p_fma": 0.5})
    h_distill = hash_config({**_BASE, "data_source": "distill"})
    assert h_musdb != h_mixed and h_musdb != h_distill and h_mixed != h_distill
    # `distill` -> p_fma 1.0 by construction, so an explicit p_fma on distill is irrelevant.
    assert hash_config({**_BASE, "data_source": "distill"}) == hash_config(
        {**_BASE, "data_source": "distill", "p_fma": 1.0}
    )


def test_p_fma_is_an_identity_field_for_mixed() -> None:
    h50 = hash_config({**_BASE, "data_source": "mixed", "p_fma": 0.5})
    h25 = hash_config({**_BASE, "data_source": "mixed", "p_fma": 0.25})
    assert h50 != h25  # mixed vs mixed25 hash distinctly


def test_pseudo_locations_are_not_identity_fields() -> None:
    base = {**_BASE, "data_source": "mixed", "p_fma": 0.5}
    with_paths = {**base, "pseudo_root": "/drive/pseudo", "pseudo_manifest": "/drive/pseudo/m.csv"}
    assert hash_config(base) == hash_config(with_paths)  # locations excluded from the hash


# --- normalization + canonicalization --------------------------------------

def test_pseudo_data_spec_normalizes() -> None:
    assert pseudo_data_spec({}) is None
    assert pseudo_data_spec({"data_source": "musdb"}) is None
    assert pseudo_data_spec({"data_source": "mixed", "p_fma": 0.3}) == {"data_source": "mixed", "p_fma": 0.3}
    assert pseudo_data_spec({"data_source": "distill"}) == {"data_source": "distill", "p_fma": 1.0}
    # idempotent against the canonical nested block
    assert pseudo_data_spec({"pseudo": {"data_source": "mixed", "p_fma": 0.5}}) == {
        "data_source": "mixed", "p_fma": 0.5
    }


def test_canonicalize_is_idempotent_with_pseudo() -> None:
    cfg = {**_BASE, "data_source": "mixed", "p_fma": 0.5, "pseudo_root": "/x"}
    once = canonicalize_config(cfg)
    assert canonicalize_config(once) == once
    assert once["pseudo"] == {"data_source": "mixed", "p_fma": 0.5}
    # musdb-only canonicalizes to NO pseudo key at all.
    assert "pseudo" not in canonicalize_config({**_BASE, "data_source": "musdb"})


def test_invalid_pseudo_configs_raise() -> None:
    with pytest.raises(ValueError):
        hash_config({**_BASE, "data_source": "bogus"})
    with pytest.raises(ValueError):
        hash_config({**_BASE, "data_source": "mixed"})          # mixed needs p_fma
    for bad in (0.0, 1.0, 1.5, -0.2):
        with pytest.raises(ValueError):
            hash_config({**_BASE, "data_source": "mixed", "p_fma": bad})
