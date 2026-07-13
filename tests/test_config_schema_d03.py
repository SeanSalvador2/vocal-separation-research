"""Direction 03 config schema: shared-baseline hash, split identity, base_width (G0, §3.2, §8).

Gate-G0 config items:
* the baseline cell (base.yaml) is BIT-IDENTICAL in hash to D01 `l1mag_seed0_reduced`
  and D02 `base` — the three-way shared-baseline equality (§3.2),
* the optional `model:` block leaves old configs' hashes untouched but makes each
  split arm a distinct run,
* every split config builds the right architecture within ±2 %,
* the appended `base_width` registry column round-trips.
"""

from __future__ import annotations

from singnet.models import (
    BandSplitUNet,
    SingNetC1,
    build_model_from_config,
    mel_edges,
    uniform_edges,
)
from singnet.train import REGISTRY_COLUMNS, RunRecord, read_registry, upsert_run
from singnet.utils.config import hash_config, resolve_config

D01 = "01-loss-function-study/configs/l1mag_seed0_reduced.yaml"
D02_BASE = "02-augmentation-data-scaling/configs/base.yaml"
D03 = "03-mini-band-split/configs"
SPLITS = [f"split_{b}_seed{s}" for b in ("mel", "uniform") for s in (0, 1, 2)]
BASELINE = 9_835_745


def test_d03_base_equals_d01_and_d02_shared_cell() -> None:
    # The headline §3.2 equality: the Direction-03 baseline cell IS the D01 l1mag
    # sweep cell (and the D02 base cell), so it is reused, never retrained.
    h01 = hash_config(resolve_config(D01))
    h02 = hash_config(resolve_config(D02_BASE))
    h03 = hash_config(resolve_config(f"{D03}/base.yaml"))
    assert h01 == h02 == h03


def test_baseline_config_builds_singnet_c1() -> None:
    model = build_model_from_config(resolve_config(f"{D03}/base.yaml"))
    assert isinstance(model, SingNetC1)
    assert model.num_parameters == BASELINE


def test_model_block_changes_hash_vs_baseline() -> None:
    base = hash_config(resolve_config(f"{D03}/base.yaml"))
    for name in SPLITS:
        assert hash_config(resolve_config(f"{D03}/{name}.yaml")) != base  # a distinct run


def test_all_split_configs_have_distinct_hashes() -> None:
    hashes = {hash_config(resolve_config(f"{D03}/{n}.yaml")) for n in SPLITS}
    assert len(hashes) == len(SPLITS)


def test_split_configs_build_matched_within_2pct() -> None:
    for name in SPLITS:
        model = build_model_from_config(resolve_config(f"{D03}/{name}.yaml"))
        assert isinstance(model, BandSplitUNet)
        assert model.num_parameters == 9_841_896
        assert abs(model.num_parameters - BASELINE) / BASELINE <= 0.02


def test_split_configs_select_correct_band_layout() -> None:
    mel = build_model_from_config(resolve_config(f"{D03}/split_mel_seed0.yaml"))
    uni = build_model_from_config(resolve_config(f"{D03}/split_uniform_seed0.yaml"))
    assert mel.edges_bins == mel_edges(3) == [0, 142, 597, 2048]
    assert uni.edges_bins == uniform_edges(3) == [0, 683, 1365, 2048]


def test_loss_key_decouples_from_arm() -> None:
    # split arms carry the experimental arm id but train on l1mag (the loss key);
    # the contingency configs override the loss to sisdr.
    cfg = resolve_config(f"{D03}/split_mel_seed0.yaml")
    assert cfg["arm"] == "split_mel" and cfg.get("loss") == "l1mag"
    cont = resolve_config(f"{D03}/contingency_mel_sisdr_seed0.yaml")
    assert cont["arm"] == "split_mel" and cont["loss"] == "sisdr"


def test_contingency_baseline_is_singnet_with_sisdr() -> None:
    cfg = resolve_config(f"{D03}/contingency_baseline_sisdr_seed0.yaml")
    assert isinstance(build_model_from_config(cfg), SingNetC1)
    assert cfg["loss"] == "sisdr"
    # distinct from the shared baseline cell (different loss)
    assert hash_config(cfg) != hash_config(resolve_config(f"{D03}/base.yaml"))


def test_smoke_configs_build_and_overfit_setup() -> None:
    for bands, name in (("mel", "smoke_mel"), ("uniform", "smoke_uniform")):
        cfg = resolve_config(f"{D03}/{name}.yaml")
        assert cfg["augment"] is False and cfg["remix"] is False  # single-chunk overfit
        model = build_model_from_config(cfg)
        assert isinstance(model, BandSplitUNet) and model.edges_bins[1:-1] == (
            mel_edges(3)[1:-1] if bands == "mel" else uniform_edges(3)[1:-1]
        )


def test_registry_base_width_roundtrips(tmp_path) -> None:
    path = tmp_path / "registry.csv"
    rec = RunRecord(run_id="split_mel_seed0_reduced_h", arm="split_mel", seed=0, budget=16000,
                    config_hash="h", base_width=23)
    upsert_run(path, rec)
    frame = read_registry(path)
    assert list(frame.columns) == list(REGISTRY_COLUMNS)
    assert int(frame.iloc[0]["base_width"]) == 23


def test_registry_base_width_defaults_to_baseline_width() -> None:
    rec = RunRecord(run_id="r", arm="l1mag", seed=0, budget=16000, config_hash="h")
    assert rec.base_width == 32  # the baseline SingNet-C1 width
