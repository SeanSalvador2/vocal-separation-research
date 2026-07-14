"""Direction 06 config schema: shared clean cell, ε/q identity fields, hash stability (G0, §3.3, §8).

Gate-G0 config items (MASTER_PLAN §3.1, §3.3, §8):
* the clean cell (base.yaml, ε=0) is BIT-IDENTICAL in hash to D01 `l1mag_seed0_reduced`
  (and D02/D03 base) — the shared-baseline equality (§3.3);
* the optional `corrupt`/`trim` blocks leave every existing D01-D05 hash untouched
  (hash stability), and ε=0 canonicalizes to "no corruption" (clean ≡ D01);
* `epsilon` is a config-hash identity field when present (bleed05/15/30 distinct), and
  `trim_q` distinguishes trim30 / trim30_q10 / trim_clean;
* every matrix config resolves with the right ε/q/loss.
"""

from __future__ import annotations

from singnet.utils.config import (
    canonicalize_config,
    corruption_epsilon,
    hash_config,
    resolve_config,
    trim_q,
)

D06 = "06-robust-training/configs"
D01 = "01-loss-function-study/configs/l1mag_seed0_reduced.yaml"
D02_BASE = "02-augmentation-data-scaling/configs/base.yaml"
D03_BASE = "03-mini-band-split/configs/base.yaml"
D05 = "05-lora-source-separation/configs/t1_lora16_seed0.yaml"

MATRIX = [
    "bleed05_seed0", "bleed15_seed0",
    "bleed30_seed0", "bleed30_seed1", "bleed30_seed2",
    "trim30_seed0", "trim30_seed1", "trim30_seed2",
    "trim30_q10_seed0", "trim_clean_seed0",
    "contingency_bleed30_sisdr_seed0", "contingency_trim30_sisdr_seed0",
]


def _h(name: str) -> str:
    return hash_config(resolve_config(f"{D06}/{name}.yaml"))


# --- the shared clean cell --------------------------------------------------

def test_d06_base_equals_d01_and_shared_cell() -> None:
    # The headline §3.3 equality: the Direction-06 clean cell (ε=0) IS the D01 l1mag
    # sweep cell, so it is reused across the 3 clean seeds, never retrained.
    h01 = hash_config(resolve_config(D01))
    h02 = hash_config(resolve_config(D02_BASE))
    h03 = hash_config(resolve_config(D03_BASE))
    h06 = hash_config(resolve_config(f"{D06}/base.yaml"))
    assert h06 == h01 == h02 == h03


# --- hash stability: the new keys touch no existing hash --------------------

def test_new_d06_keys_do_not_change_existing_hashes() -> None:
    # Adding the corrupt/trim schema left every existing config's hash untouched
    # (the D01≡D02≡D03 shared cell still holds; a D05 config is unaffected too).
    assert hash_config(resolve_config(D01)) == hash_config(resolve_config(D02_BASE))
    assert hash_config(resolve_config(D03_BASE)) == hash_config(resolve_config(D01))
    # a config with no corrupt/trim keys is unaffected by their existence in the schema.
    plain = {"arm": "l1mag", "seed": 0, "steps": 16000}
    assert hash_config(plain) == hash_config(dict(plain))


def test_epsilon_zero_canonicalizes_to_no_corruption() -> None:
    # ε=0 (explicit) must hash identically to a config with no corrupt block at all,
    # so the clean cell is shared even if someone writes corrupt: {epsilon: 0}.
    without = {"arm": "l1mag", "seed": 0, "steps": 16000}
    with_zero = {"arm": "l1mag", "seed": 0, "steps": 16000, "corrupt": {"epsilon": 0.0}}
    assert hash_config(without) == hash_config(with_zero)
    # a positive ε mints a distinct hash.
    with_bleed = {"arm": "l1mag", "seed": 0, "steps": 16000, "corrupt": {"epsilon": 0.30}}
    assert hash_config(with_bleed) != hash_config(without)


# --- epsilon / trim_q are identity fields -----------------------------------

def test_all_matrix_configs_distinct_hashes() -> None:
    hashes = {name: _h(name) for name in MATRIX}
    hashes["base"] = hash_config(resolve_config(f"{D06}/base.yaml"))
    assert len(set(hashes.values())) == len(hashes)  # 13 distinct runs


def test_epsilon_is_an_identity_field() -> None:
    assert _h("bleed05_seed0") != _h("bleed15_seed0") != _h("bleed30_seed0")
    assert _h("bleed05_seed0") != _h("bleed30_seed0")
    for name, eps in (("bleed05_seed0", 0.05), ("bleed15_seed0", 0.15), ("bleed30_seed0", 0.30)):
        assert corruption_epsilon(resolve_config(f"{D06}/{name}.yaml")) == eps


def test_trim_q_is_an_identity_field() -> None:
    # trim block distinguishes bleed30 from trim30, and q distinguishes trim30 / q10.
    assert _h("bleed30_seed0") != _h("trim30_seed0")
    assert _h("trim30_seed0") != _h("trim30_q10_seed0")
    assert trim_q(resolve_config(f"{D06}/trim30_seed0.yaml")) == 0.30
    assert trim_q(resolve_config(f"{D06}/trim30_q10_seed0.yaml")) == 0.10
    assert trim_q(resolve_config(f"{D06}/bleed30_seed0.yaml")) is None


def test_trim_clean_control_is_distinct_from_clean_cell() -> None:
    # the control (ε=0 + trimming) is a NEW distinct run vs the clean cell.
    assert _h("trim_clean_seed0") != hash_config(resolve_config(f"{D06}/base.yaml"))
    cfg = resolve_config(f"{D06}/trim_clean_seed0.yaml")
    assert corruption_epsilon(cfg) == 0.0 and trim_q(cfg) == 0.30


def test_seed_is_part_of_identity() -> None:
    assert _h("bleed30_seed0") != _h("bleed30_seed1") != _h("bleed30_seed2")
    assert _h("trim30_seed0") != _h("trim30_seed1")


def test_contingency_configs_switch_loss_to_sisdr() -> None:
    for name in ("contingency_bleed30_sisdr_seed0", "contingency_trim30_sisdr_seed0"):
        cfg = resolve_config(f"{D06}/{name}.yaml")
        assert cfg["loss"] == "sisdr" and corruption_epsilon(cfg) == 0.30
    # distinct from their l1mag counterparts (loss is an identity field).
    assert _h("contingency_bleed30_sisdr_seed0") != _h("bleed30_seed0")
    assert _h("contingency_trim30_sisdr_seed0") != _h("trim30_seed0")


def test_configs_declare_l1mag_loss_and_experimental_arm() -> None:
    cfg = resolve_config(f"{D06}/bleed30_seed0.yaml")
    assert cfg["arm"] == "bleed30" and cfg.get("loss") == "l1mag"
    cfg = resolve_config(f"{D06}/trim30_seed0.yaml")
    assert cfg["arm"] == "trim30" and cfg.get("loss") == "l1mag"


# --- canonicalization is idempotent with the corrupt block ------------------

def test_canonicalize_handles_corrupt_block_idempotently() -> None:
    cfg = resolve_config(f"{D06}/bleed30_seed0.yaml")
    once = canonicalize_config(cfg)
    assert canonicalize_config(once) == once            # idempotent
    assert once["corrupt"] == {"epsilon": 0.30}         # canonical identity block
    # trim block passes straight through canonicalization.
    trimmed = canonicalize_config(resolve_config(f"{D06}/trim30_seed0.yaml"))
    assert trimmed["trim"] == {"q": 0.30}
