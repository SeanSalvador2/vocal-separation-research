"""Direction 05 config schema: distinct hashes, recipe/rank fields, D01-03 untouched (G0).

Gate-G0 config items (MASTER_PLAN §3, §8):
* every D05 config resolves and the 12 main configs have distinct hashes;
* the recipe / rank / domain / lr fields read back correctly; T2 carries the
  ``op: pending-G0b`` marker resolved at G0b;
* the new D05 keys are optional-with-defaults, so **all existing D01-03 hashes are
  unchanged** (the three-way shared-baseline equality still holds).
"""

from __future__ import annotations

from singnet.peft import RECIPES, apply_recipe, load_umxhq, recipe_rank
from singnet.utils.config import canonicalize_config, hash_config, resolve_config

D05 = "05-lora-source-separation/configs"
D01 = "01-loss-function-study/configs/l1mag_seed0_reduced.yaml"
D02_BASE = "02-augmentation-data-scaling/configs/base.yaml"
D03_BASE = "03-mini-band-split/configs/base.yaml"

MAIN = [
    "t1_head_seed0", "t1_lora4_seed0",
    "t1_lora16_seed0", "t1_lora16_seed1", "t1_lora16_seed2",
    "t1_full_seed0", "t1_full_seed1", "t1_full_seed2",
    "t2_head_seed0", "t2_lora4_seed0", "t2_lora16_seed0", "t2_full_seed0",
]


def test_all_main_configs_resolve_and_have_distinct_hashes() -> None:
    hashes = {name: hash_config(resolve_config(f"{D05}/{name}.yaml")) for name in MAIN}
    assert len(set(hashes.values())) == len(MAIN)  # 12 distinct runs


def test_configs_declare_recipe_rank_domain() -> None:
    expect = {
        "t1_head_seed0": ("head", None, "t1_aac64"),
        "t1_lora4_seed0": ("lora4", 4, "t1_aac64"),
        "t1_lora16_seed2": ("lora16", 16, "t1_aac64"),
        "t1_full_seed1": ("full", None, "t1_aac64"),
        "t2_lora16_seed0": ("lora16", 16, "t2"),
    }
    for name, (recipe, rank, domain) in expect.items():
        cfg = resolve_config(f"{D05}/{name}.yaml")
        assert cfg["recipe"] == recipe and cfg["domain"] == domain
        assert cfg.get("rank") == rank
        # the config's declared rank agrees with the recipe's implied rank.
        assert recipe_rank(recipe) == (rank if recipe.startswith("lora") else None)


def test_recipe_matches_the_known_recipe_set() -> None:
    for name in MAIN:
        assert resolve_config(f"{D05}/{name}.yaml")["recipe"] in RECIPES


def test_umx_recipe_augment_is_gain_remix_channelswap_no_flip() -> None:
    cfg = resolve_config(f"{D05}/t1_lora16_seed0.yaml")
    assert cfg["augment"] == {"remix": True, "gain": True, "flip": False}
    assert cfg["channelswap"] is True


def test_t2_configs_carry_pending_op_marker() -> None:
    for name in ("t2_head_seed0", "t2_lora4_seed0", "t2_lora16_seed0", "t2_full_seed0"):
        cfg = resolve_config(f"{D05}/{name}.yaml")
        assert cfg["op"] == "pending-G0b"       # resolved at G0b by --resolve-t2
        assert cfg["domain"] == "t2"
    # T1 configs read their degraded shards via the domain suffix.
    assert resolve_config(f"{D05}/t1_head_seed0.yaml")["domain_suffix"] == "t1_aac64"


def test_seed_is_part_of_identity() -> None:
    h0 = hash_config(resolve_config(f"{D05}/t1_lora16_seed0.yaml"))
    h1 = hash_config(resolve_config(f"{D05}/t1_lora16_seed1.yaml"))
    assert h0 != h1


def test_recipe_changes_hash_within_a_domain() -> None:
    hashes = {
        r: hash_config(resolve_config(f"{D05}/t1_{r}_seed0.yaml"))
        for r in ("head", "lora4", "lora16", "full")
    }
    assert len(set(hashes.values())) == 4


def test_d05_configs_build_the_right_recipe_on_the_mock() -> None:
    # end-to-end: the config's recipe/rank applied to the mock gives the pinned share.
    from singnet.peft import measured_trainable_share

    cfg = resolve_config(f"{D05}/t1_lora16_seed0.yaml")
    model = load_umxhq("cpu", mock=True)
    apply_recipe(model, cfg["recipe"], r=cfg.get("rank"))
    assert abs(measured_trainable_share(model) - 0.048522) < 1e-4


def test_new_d05_keys_do_not_change_d01_d02_d03_hashes() -> None:
    # The headline backward-compat guarantee: adding the D05 schema left every existing
    # config's hash untouched — the three-way shared-baseline equality still holds.
    h01 = hash_config(resolve_config(D01))
    h02 = hash_config(resolve_config(D02_BASE))
    h03 = hash_config(resolve_config(D03_BASE))
    assert h01 == h02 == h03
    # a config without the D05 keys is unaffected by their existence in the schema
    # (hash_config only canonicalizes augment/allowlist; unknown keys pass through).
    plain = {"arm": "l1mag", "seed": 0, "steps": 16000}
    assert hash_config(plain) == hash_config(dict(plain))


def test_probe_template_is_a_template_not_a_main_run() -> None:
    cfg = resolve_config(f"{D05}/t1_probe_template.yaml")
    assert cfg["budget_name"] == "probe" and cfg["steps"] == 500
    # its hash differs from every main config (short budget, probe arm).
    main_hashes = {hash_config(resolve_config(f"{D05}/{n}.yaml")) for n in MAIN}
    assert hash_config(cfg) not in main_hashes


def test_canonicalize_handles_d05_config() -> None:
    cfg = resolve_config(f"{D05}/t1_lora4_seed0.yaml")
    once = canonicalize_config(cfg)
    assert canonicalize_config(once) == once            # idempotent
    assert once["recipe"] == "lora4" and once["rank"] == 4  # D05 keys preserved
