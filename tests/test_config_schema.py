"""Direction 02 config schema: canonicalization + hash equality (G0, §1, §8).

The gate-G0 config-hash items:

* the default-fill makes "no augment block" == "explicit defaults" (and the
  legacy Direction-01 bool form) hash identically,
* the real Direction-01 ``l1mag`` cell and the Direction-02 ``full`` cell
  (base.yaml) share a config hash -> the FULL-86 anchor is reused, not retrained,
* the switchboard and the subset allowlist DO change the hash (distinct runs),
* canonicalization is idempotent.
"""

from __future__ import annotations

from singnet.utils.config import canonicalize_config, hash_config, resolve_config

D01 = "01-loss-function-study/configs/l1mag_seed0_reduced.yaml"
D02 = "02-augmentation-data-scaling/configs"


def test_no_augment_block_equals_explicit_defaults() -> None:
    old = {"arm": "l1mag", "seed": 0, "steps": 16000}
    new = {"arm": "l1mag", "seed": 0, "steps": 16000,
           "augment": {"remix": True, "gain": True, "flip": True},
           "data": {"track_allowlist_csv": None}}
    assert hash_config(old) == hash_config(new)


def test_legacy_bool_form_equals_dict_form() -> None:
    legacy = {"arm": "l1mag", "seed": 0, "steps": 16000, "augment": True, "remix": True}
    explicit = {"arm": "l1mag", "seed": 0, "steps": 16000,
                "augment": {"remix": True, "gain": True, "flip": True}}
    assert hash_config(legacy) == hash_config(explicit)


def test_d01_l1mag_equals_d02_full_hash() -> None:
    # The headline G0 equality: Direction 01's l1mag sweep cell IS Direction 02's
    # full / n86 cell (§3.3), so it is reused rather than retrained.
    assert hash_config(resolve_config(D01)) == hash_config(resolve_config(f"{D02}/base.yaml"))


def test_switchboard_changes_the_hash() -> None:
    full = resolve_config(f"{D02}/base.yaml")
    for name in ("loo_no_remix_seed0", "loo_no_gain_seed0", "loo_no_flip_seed0", "loo_none_seed0"):
        assert hash_config(resolve_config(f"{D02}/{name}.yaml")) != hash_config(full)


def test_subset_allowlist_changes_the_hash() -> None:
    full = hash_config(resolve_config(f"{D02}/base.yaml"))
    n21 = hash_config(resolve_config(f"{D02}/scale_n21_seed0.yaml"))
    n43 = hash_config(resolve_config(f"{D02}/scale_n43_seed0.yaml"))
    assert len({full, n21, n43}) == 3  # full / n21 / n43 are three distinct runs


def test_all_reduced_configs_have_distinct_hashes() -> None:
    names = [
        "loo_no_remix_seed0", "loo_no_gain_seed0", "loo_no_flip_seed0", "loo_none_seed0",
        "scale_n21_seed0", "scale_n21_seed1", "scale_n21_seed2", "scale_n43_seed0", "scale_n64_seed0",
    ]
    hashes = {hash_config(resolve_config(f"{D02}/{n}.yaml")) for n in names}
    assert len(hashes) == len(names)


def test_seed_changes_hash_but_not_switchboard() -> None:
    h0 = hash_config(resolve_config(f"{D02}/scale_n21_seed0.yaml"))
    h1 = hash_config(resolve_config(f"{D02}/scale_n21_seed1.yaml"))
    assert h0 != h1  # seed is part of identity


def test_canonicalize_is_idempotent() -> None:
    cfg = resolve_config(f"{D02}/scale_n21_seed0.yaml")
    once = canonicalize_config(cfg)
    assert canonicalize_config(once) == once
    assert once["augment"] == {"remix": True, "gain": True, "flip": True}
    assert once["track_allowlist_csv"].endswith("n21.csv")
    assert "data" not in once and "remix" not in once  # folded / flattened
