"""Three-arm ordered-chain verdict logic H-03 (G0, §2, §13).

Each pre-registered outcome cell is exercised on constructed per-seed data whose
between-seed spread (pooled σ_seed ~ 0.1 dB) is small enough that the ~1 dB arm
gaps are unambiguous.
"""

from __future__ import annotations

import math

from singnet.analysis import ordered_chain_verdict, pooled_seed_sigma, three_arm_sigma

# Tight per-seed clusters (sample std ~0.1) reused across the cells.
BASE = [5.0, 5.1, 4.9]        # mean 5.0
UNI_HIGH = [6.0, 6.1, 5.9]    # mean 6.0  (splitting helps)
MEL_HIGH = [7.0, 7.1, 6.9]    # mean 7.0  (mel helps beyond splitting)


def test_three_arm_sigma_matches_pooled_formula() -> None:
    assert three_arm_sigma(BASE, UNI_HIGH, MEL_HIGH) == pooled_seed_sigma(BASE, UNI_HIGH, MEL_HIGH)
    # sqrt of the mean of the three (equal) sample variances
    assert abs(three_arm_sigma(BASE, UNI_HIGH, MEL_HIGH) - 0.1) < 1e-9


def test_fully_supported_ordered_chain() -> None:
    v = ordered_chain_verdict(BASE, UNI_HIGH, MEL_HIGH)
    assert v.e1_holds and v.e2_holds
    assert v.verdict == "fully_supported"
    assert abs(v.e1 - 1.0) < 1e-9 and abs(v.e2 - 1.0) < 1e-9


def test_partial_e1_splitting_only() -> None:
    mel_flat = [6.05, 6.15, 5.95]   # mel ~ uniform
    v = ordered_chain_verdict(BASE, UNI_HIGH, mel_flat)
    assert v.e1_holds and not v.e2_holds
    assert v.verdict == "partial_E1"


def test_partial_e2_mel_only() -> None:
    uni_flat = [5.05, 5.15, 4.95]   # uniform ~ baseline
    v = ordered_chain_verdict(BASE, uni_flat, MEL_HIGH)
    assert v.e2_holds and not v.e1_holds
    assert v.verdict == "partial_E2"


def test_refuted_null_all_within_band() -> None:
    uni = [5.03, 5.13, 4.93]
    mel = [5.06, 5.16, 4.96]
    v = ordered_chain_verdict(BASE, uni, mel)
    assert not v.e1_holds and not v.e2_holds
    assert v.verdict == "refuted_null"


def test_refuted_negative_baseline_wins() -> None:
    base_high = [7.0, 7.1, 6.9]     # baseline mean 7.0
    uni = [5.0, 5.1, 4.9]           # mean 5.0
    mel = [5.05, 5.15, 4.95]        # mean 5.05 (~ uniform, so E2 does not hold)
    v = ordered_chain_verdict(base_high, uni, mel)
    assert not v.e1_holds and not v.e2_holds
    assert v.verdict == "refuted_negative"


def test_verdict_fields_are_populated() -> None:
    v = ordered_chain_verdict(BASE, UNI_HIGH, MEL_HIGH)
    assert not math.isnan(v.sigma_seed)
    assert (v.mean_baseline, v.mean_uniform, v.mean_mel) == (5.0, 6.0, 7.0)
