"""Deterministic width search + closed-form == built model (G0, §3.1, §8).

The credibility crux (MASTER_PLAN §3.1, §12): the ±2 % parameter match is exact
and reproducible, not hand-waved. These tests pin the search output, assert the
closed-form accounting equals the real ``nn.Module`` param count, and check the
committed per-module table sums.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import match_params as mp  # noqa: E402
from singnet.models import BandSplitUNet, SingNetC1  # noqa: E402

TARGET = 9_835_745


def test_closed_form_reproduces_baseline() -> None:
    assert mp.baseline_params(32) == TARGET
    assert mp.baseline_params(32) == SingNetC1().num_parameters


def test_conv_bn_params_formula() -> None:
    # Conv2d(1,32,5) = 25*1*32 + 32 (bias) = 832; BN(32) = 64; total 896 (baseline enc1).
    assert mp.conv_bn_params(1, 32) == 896
    # Conv2d(32,64,5) = 25*32*64 + 64 = 51264; BN 128; total 51392 (baseline enc2).
    assert mp.conv_bn_params(32, 64) == 51_392


def test_match_width_is_deterministic_and_pinned() -> None:
    r = mp.match_width(TARGET, tol=0.02)
    assert (r.base_width, r.bottleneck_width, r.delta) == (23, 382, 14)
    assert r.params == 9_841_896
    assert r.within_tol and abs(r.rel_error) <= 0.02
    # re-running is byte-identical (deterministic search)
    assert mp.match_width(TARGET, tol=0.02) == r


def test_search_picks_largest_width_under_budget() -> None:
    # c=23 pure is under budget; c=24 pure overshoots -> 23 is the largest admissible base.
    assert mp.variant_params(23, 16 * 23) <= TARGET < mp.variant_params(24, 16 * 24)
    assert mp.match_width(TARGET).base_width == 23


def test_bottleneck_bump_minimizes_absolute_error() -> None:
    r = mp.match_width(TARGET)
    err = abs(r.params - TARGET)
    for delta in range(0, 40):
        cand = abs(mp.variant_params(23, 16 * 23 + delta) - TARGET)
        assert err <= cand  # chosen delta is the argmin


def test_closed_form_equals_built_model() -> None:
    r = mp.match_width(TARGET)
    mel = BandSplitUNet.from_mel_bands(base_width=r.base_width, bottleneck_width=r.bottleneck_width)
    uni = BandSplitUNet.from_uniform_bands(base_width=r.base_width, bottleneck_width=r.bottleneck_width)
    assert mp.variant_params(r.base_width, r.bottleneck_width) == mel.num_parameters == uni.num_parameters


def test_verify_against_model_agrees() -> None:
    r = mp.match_width(TARGET)
    counts = mp.verify_against_model(r)
    assert counts == {"split_mel": r.params, "split_uniform": r.params}


def test_per_module_table_sums_to_totals() -> None:
    r = mp.match_width(TARGET)
    rows = mp.per_module_rows(r)
    assert sum(base for _, base, _ in rows) == TARGET
    assert sum(var for _, _, var in rows) == r.params
    # encoder rows are the 3-tower total: enc1 = 3 * (conv_bn(1,23)) = 3 * 644 = 1932
    enc1 = dict((name, var) for name, _, var in rows)["enc1"]
    assert enc1 == 3 * mp.conv_bn_params(1, 23) == 1_932


def test_render_markdown_has_key_numbers() -> None:
    r = mp.match_width(TARGET)
    md = mp.render_markdown(r, mp.verify_against_model(r))
    assert "9,841,896" in md and "9,835,745" in md
    assert "c = 23" in md and "b5 = 382" in md
    assert "[0, 142, 597, 2048]" in md and "[0, 683, 1365, 2048]" in md
