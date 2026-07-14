"""Direction 08 sampling schema: uniform ≡ shared cell, policy identity, hash stability (G0, §4.1).

Gate-G0 config items (MASTER_PLAN §4.1, §4.2, §8):
* the ``uniform`` policy (or an absent ``sampling:`` block) canonicalizes to NO key, so a
  uniform config is BIT-IDENTICAL in hash to the shared baseline cell (D01 ``l1mag`` ≡
  D02/D03 base) — why the uniform arm is *reused*, not retrained (§4.2: "0 new runs");
* adding the sampling schema leaves every existing Direction 01–06 hash untouched;
* ``policy`` is a config-hash identity field (uniform/energy/drop/curriculum distinct), and
  only the *relevant* constant enters each policy's hash (θ for ``drop``, λ for
  ``energy``/``curriculum``).

The tests here exercise the canonicalization *mechanism* on inline dicts + the existing
shared-cell configs; the Direction-08 YAML files are asserted in ``tests/test_train_d08.py``.
"""

from __future__ import annotations

import pytest

from singnet.utils.config import (
    canonicalize_config,
    hash_config,
    resolve_config,
    sampling_policy,
)

D01 = "01-loss-function-study/configs/l1mag_seed0_reduced.yaml"
D02_BASE = "02-augmentation-data-scaling/configs/base.yaml"
D03_BASE = "03-mini-band-split/configs/base.yaml"

_BASE = {"arm": "x", "seed": 0, "steps": 16000}


# --- the shared-cell equality ----------------------------------------------

def test_uniform_sampling_hashes_as_shared_cell() -> None:
    shared = hash_config(resolve_config(D01))
    cfg = resolve_config(D01)
    # an explicit uniform block (with defaults) canonicalizes away -> shared-cell hash.
    with_uniform = {**cfg, "sampling": {"policy": "uniform", "theta_db": -60.0, "floor_lambda": 0.1}}
    assert hash_config(with_uniform) == shared
    # and an absent block is trivially the shared cell.
    assert hash_config(cfg) == shared


def test_sampling_schema_leaves_existing_hashes_unchanged() -> None:
    # The D01 ≡ D02 ≡ D03 shared cell still holds after the sampling schema was added.
    assert hash_config(resolve_config(D01)) == hash_config(resolve_config(D02_BASE))
    assert hash_config(resolve_config(D03_BASE)) == hash_config(resolve_config(D01))
    # a config with no sampling key is unaffected by the schema's existence.
    assert hash_config(dict(_BASE)) == hash_config(dict(_BASE))


# --- policy is an identity field -------------------------------------------

def test_policy_is_an_identity_field() -> None:
    h_uniform = hash_config(dict(_BASE))
    hashes = {
        p: hash_config({**_BASE, "sampling": {"policy": p}})
        for p in ("energy", "drop", "curriculum")
    }
    hashes["uniform_block"] = hash_config({**_BASE, "sampling": {"policy": "uniform"}})
    assert hashes["uniform_block"] == h_uniform  # uniform block == no block
    assert len({h_uniform, hashes["energy"], hashes["drop"], hashes["curriculum"]}) == 4


def test_only_relevant_constant_enters_each_policy_hash() -> None:
    # θ steers `drop`; λ steers `energy`/`curriculum`. Each is an identity field only for
    # the policy it affects, so a spurious constant does not mint a false-distinct run.
    def h(block):
        return hash_config({**_BASE, "sampling": block})

    assert h({"policy": "drop", "theta_db": -60}) != h({"policy": "drop", "theta_db": -50})
    assert h({"policy": "energy", "floor_lambda": 0.1}) != h({"policy": "energy", "floor_lambda": 0.2})
    # θ is irrelevant to `energy`; λ is irrelevant to `drop` — neither changes the hash.
    assert h({"policy": "energy", "theta_db": -60}) == h({"policy": "energy", "theta_db": -50})
    assert h({"policy": "drop", "floor_lambda": 0.1}) == h({"policy": "drop", "floor_lambda": 0.2})


# --- normalization + canonicalization --------------------------------------

def test_sampling_policy_fills_defaults() -> None:
    assert sampling_policy({}) == {"policy": "uniform", "theta_db": -60.0, "floor_lambda": 0.1}
    spec = sampling_policy({"sampling": {"policy": "drop", "theta_db": -70}})
    assert spec["policy"] == "drop" and spec["theta_db"] == -70.0


def test_canonicalize_is_idempotent_with_sampling() -> None:
    cfg = {**_BASE, "sampling": {"policy": "energy", "floor_lambda": 0.1}}
    once = canonicalize_config(cfg)
    assert canonicalize_config(once) == once
    assert once["sampling"] == {"policy": "energy", "floor_lambda": 0.1}  # only λ, not θ
    dropped = canonicalize_config({**_BASE, "sampling": {"policy": "drop", "theta_db": -60.0}})
    assert dropped["sampling"] == {"policy": "drop", "theta_db": -60.0}  # only θ, not λ


def test_unknown_policy_raises() -> None:
    with pytest.raises(ValueError):
        hash_config({**_BASE, "sampling": {"policy": "bogus"}})
