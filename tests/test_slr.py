"""SLR silence-leakage metric — anchors, injected leakage, edges (G0, MASTER_PLAN §2).

The metric is the direction's core deliverable, so the tests are exhaustive: exact dB
values for known-injected leakage, the three anchors (do-nothing = 0 dB, perfect = ε
floor, 10 % leak = −10 dB), NaN on empty regions, threshold-boundary + min-run + frame
edge behaviour of the silent-region identifier, and joint-gain invariance.
"""

from __future__ import annotations

import numpy as np

from singnet.metrics import silent_regions, slr, slr_report
from singnet.metrics.slr import DEFAULT_EPS


# --- silent-region identification ------------------------------------------

def test_silent_regions_all_silent_track() -> None:
    sr = 1000
    wave = np.zeros(sr, dtype=np.float32)  # 1 s of pure silence
    regions = silent_regions(wave, sr, theta_db=-60.0)
    assert len(regions) == 1
    start, end = regions[0]
    assert start == 0 and end == sr  # covers the whole track, clamped to the end


def test_silent_regions_no_silence_when_loud() -> None:
    sr = 1000
    wave = np.full(sr, 0.5, dtype=np.float32)  # −6 dBFS everywhere, well above −60
    assert silent_regions(wave, sr, theta_db=-60.0) == []


def test_silent_regions_threshold_boundary_is_strict() -> None:
    # A frame whose RMS sits exactly at the threshold is NOT silent (strict <).
    sr = 1000
    thresh = 10.0 ** (-60.0 / 20.0)
    at = np.full(sr, thresh, dtype=np.float64)
    below = np.full(sr, thresh * 0.5, dtype=np.float64)
    assert silent_regions(at, sr, theta_db=-60.0) == []          # == threshold -> not silent
    assert len(silent_regions(below, sr, theta_db=-60.0)) == 1   # below -> silent


def test_silent_regions_min_run_filters_short_gaps() -> None:
    # A 0.3 s silent gap (< L_min = 0.5 s) between loud regions is discarded.
    sr = 1000
    loud = np.full(int(0.6 * sr), 0.5, dtype=np.float64)
    gap = np.zeros(int(0.3 * sr), dtype=np.float64)
    wave = np.concatenate([loud, gap, loud])
    assert silent_regions(wave, sr, theta_db=-60.0, min_run_s=0.5) == []
    # the same gap survives when L_min is dropped to 0.2 s.
    assert len(silent_regions(wave, sr, theta_db=-60.0, min_run_s=0.2)) == 1


def test_silent_regions_merge_contiguous_frames_into_one_run() -> None:
    # A 2 s silent stretch is one merged run, not many per-frame regions.
    sr = 1000
    wave = np.concatenate([
        np.full(sr, 0.5, dtype=np.float64),
        np.zeros(2 * sr, dtype=np.float64),
        np.full(sr, 0.5, dtype=np.float64),
    ])
    regions = silent_regions(wave, sr, theta_db=-60.0)
    assert len(regions) == 1
    start, end = regions[0]
    assert (end - start) >= int(0.5 * sr)  # the merged silent span


def test_silent_regions_track_shorter_than_frame_is_empty() -> None:
    sr = 1000
    wave = np.zeros(50, dtype=np.float64)  # 50 ms < one 100 ms frame -> no frames
    assert silent_regions(wave, sr, frame_s=0.1) == []


def test_silent_regions_run_at_track_end_clamped() -> None:
    sr = 1000
    wave = np.concatenate([np.full(sr, 0.5, dtype=np.float64), np.zeros(sr, dtype=np.float64)])
    regions = silent_regions(wave, sr, theta_db=-60.0)
    assert len(regions) == 1
    _, end = regions[0]
    assert end <= wave.size  # never overruns the track


def test_silent_regions_theta_sensitivity_monotone() -> None:
    # A quiet-but-not-silent passage counts as silent under a laxer (higher) θ and
    # not under a stricter (lower) θ — the §2.1 sensitivity axis.
    sr = 1000
    quiet = np.full(sr, 10.0 ** (-65.0 / 20.0), dtype=np.float64)  # −65 dBFS
    wave = np.concatenate([np.full(sr, 0.5, dtype=np.float64), quiet])
    assert len(silent_regions(wave, sr, theta_db=-60.0)) == 1  # −65 < −60 -> silent
    assert silent_regions(wave, sr, theta_db=-70.0) == []      # −65 > −70 -> not silent


# --- SLR anchors ------------------------------------------------------------

def _one_region(n: int) -> list[tuple[int, int]]:
    return [(0, n)]


def test_slr_do_nothing_anchor_is_exactly_zero() -> None:
    rng = np.random.default_rng(0)
    mix = rng.standard_normal(4096)
    # do-nothing separator: v̂ = x -> numerator == denominator -> exactly 0 dB.
    assert slr(mix, mix, _one_region(4096)) == 0.0


def test_slr_perfect_separator_at_eps_floor() -> None:
    rng = np.random.default_rng(1)
    mix = rng.standard_normal(4096)
    est = np.zeros(4096)  # perfect: no vocal energy in silence
    mix_energy = float(np.sum(mix**2))
    expected = 10.0 * np.log10(DEFAULT_EPS / (mix_energy + DEFAULT_EPS))
    assert np.isclose(slr(est, mix, _one_region(4096)), expected, atol=1e-9)


def test_slr_ten_percent_energy_leak_is_minus_ten_db() -> None:
    rng = np.random.default_rng(2)
    mix = rng.standard_normal(8192)
    est = np.sqrt(0.1) * mix  # exactly 10 % of the mixture energy leaks through
    assert np.isclose(slr(est, mix, _one_region(8192)), -10.0, atol=1e-6)


def test_slr_injected_leakage_exact_db() -> None:
    # A constructed estimate carrying a known energy fraction f scores 10·log10(f).
    rng = np.random.default_rng(3)
    mix = rng.standard_normal(8192)
    for f, expected_db in ((0.01, -20.0), (0.5, 10.0 * np.log10(0.5)), (1.0, 0.0)):
        est = np.sqrt(f) * mix
        assert np.isclose(slr(est, mix, _one_region(8192)), expected_db, atol=1e-6)


def test_slr_nan_on_empty_regions() -> None:
    mix = np.random.default_rng(4).standard_normal(1000)
    assert np.isnan(slr(mix, mix, []))


def test_slr_joint_gain_invariance() -> None:
    # Scaling est AND mix by the same c leaves SLR unchanged (eps=0, exact).
    rng = np.random.default_rng(5)
    mix = rng.standard_normal(4096)
    est = 0.3 * mix + 0.05 * rng.standard_normal(4096)
    base = slr(est, mix, _one_region(4096), eps=0.0)
    for c in (0.25, 2.0, 10.0, 1e-3):
        assert np.isclose(slr(c * est, c * mix, _one_region(4096), eps=0.0), base, atol=1e-9)


def test_slr_monotone_in_leaked_energy() -> None:
    rng = np.random.default_rng(6)
    mix = rng.standard_normal(4096)
    vals = [slr(np.sqrt(f) * mix, mix, _one_region(4096)) for f in (0.001, 0.01, 0.1, 0.5)]
    assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))  # more leak -> higher SLR


def test_slr_regions_restrict_the_support() -> None:
    # Energy outside the silent regions is ignored: a loud leak placed only in the
    # ACTIVE half does not change SLR measured on the silent half.
    n = 4000
    mix = np.ones(n)
    est_a = np.zeros(n)
    est_b = np.zeros(n)
    est_b[:2000] = 5.0  # a huge leak, but only in the active (first) half
    silent = [(2000, 4000)]
    assert slr(est_a, mix, silent) == slr(est_b, mix, silent)


# --- slr_report -------------------------------------------------------------

def test_slr_report_covers_all_thetas() -> None:
    sr = 1000
    vocal_gt = np.concatenate([np.full(sr, 0.5), np.zeros(2 * sr)]).astype(np.float64)
    mix = np.ones(3 * sr)
    est = np.zeros(3 * sr)  # perfect separator
    report = slr_report(est, mix, vocal_gt, sr, thetas=(-50.0, -60.0, -70.0))
    assert set(report) == {-50.0, -60.0, -70.0}
    for theta, entry in report.items():
        assert set(entry) == {"slr", "n_regions", "silent_samples"}
        assert entry["n_regions"] == 1.0  # the silent second-half run
        assert entry["slr"] < -30.0        # perfect separator scores far below 0 dB


def test_slr_report_nan_when_never_silent() -> None:
    sr = 1000
    vocal_gt = np.full(3 * sr, 0.5, dtype=np.float64)  # never silent
    mix = np.ones(3 * sr)
    report = slr_report(mix, mix, vocal_gt, sr)
    for entry in report.values():
        assert np.isnan(entry["slr"]) and entry["n_regions"] == 0.0
