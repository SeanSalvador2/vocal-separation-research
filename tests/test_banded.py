"""Per-band mechanism metrics on synthetic band-limited signals (G0, §6.3, §8).

Gate-G0 banded-metric items (MASTER_PLAN §6.3): an in-band tone scores high;
out-of-band energy is ~ignored; empty bands are NaN-guarded; the fixed analysis
grid is 6 bands.
"""

from __future__ import annotations

import math

import numpy as np

from singnet.eval import analysis_grid_hz, band_limited_sisdr, band_mag_error, banded_report
from singnet.metrics.si_sdr import si_sdr

SR = 44100
# mel/uniform interior edges land these analysis bands (Hz):
MID_BAND = (1528.86, 6427.66)   # a mel-interior band (bins ~142..596)
EMPTY_BAND = (22040.0, 22050.0)  # no bin centre falls here -> empty


def _tone(freq: float, seconds: float = 1.0, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float64)


def test_in_band_tone_scores_high() -> None:
    ref = _tone(3000.0)  # 3 kHz is inside MID_BAND
    score = band_limited_sisdr(ref, ref.copy(), MID_BAND, SR)
    assert score > 40.0  # a perfect in-band estimate scores very high


def test_out_of_band_energy_is_ignored() -> None:
    ref = _tone(3000.0)                       # in-band content
    est = ref + _tone(10000.0)                # estimate adds a 10 kHz out-of-band tone
    band = band_limited_sisdr(ref, est, MID_BAND, SR)
    full = si_sdr(est, ref)                    # full-band SI-SDR is dragged down by the 10 kHz
    assert band > 40.0                         # the band metric ignores the out-of-band energy
    assert band > full + 20.0                  # ... and is far better than the full-band score


def test_missing_in_band_content_is_penalized() -> None:
    ref = _tone(3000.0)                        # energy only in MID_BAND
    est = _tone(10000.0)                        # estimate has NO in-band content
    score = band_limited_sisdr(ref, est, MID_BAND, SR)
    assert score < 5.0                          # the band metric sees the missing content


def test_empty_band_returns_nan() -> None:
    ref = _tone(3000.0)
    assert math.isnan(band_limited_sisdr(ref, ref.copy(), EMPTY_BAND, SR))
    assert math.isnan(band_mag_error(ref, ref.copy(), EMPTY_BAND, SR))


def test_band_with_silent_reference_returns_nan() -> None:
    # A reference with no energy in a populated band (AAC-dead-top-band analogue,
    # in the limit) -> the NaN guard fires rather than a 0/0.
    ref = np.zeros(SR, dtype=np.float64)       # silent reference
    est = _tone(3000.0)                        # estimate has in-band content
    assert math.isnan(band_limited_sisdr(ref, est, MID_BAND, SR))
    assert math.isnan(band_mag_error(ref, est, MID_BAND, SR))


def test_band_mag_error_zero_at_perfect_prediction() -> None:
    ref = _tone(3000.0)
    assert band_mag_error(ref, ref.copy(), MID_BAND, SR) < 1e-4


def test_band_mag_error_positive_and_relative() -> None:
    ref = _tone(3000.0)
    est = 0.5 * ref                            # halved magnitude -> a real relative error
    err = band_mag_error(ref, est, MID_BAND, SR)
    assert 0.0 < err < 2.0                      # normalized, so O(1)


def test_analysis_grid_is_six_bands() -> None:
    edges = analysis_grid_hz(SR)
    assert edges == sorted(edges) and len(edges) == 7    # 7 edges -> 6 bands
    assert edges[0] == 0.0 and abs(edges[-1] - SR / 2) < 1e-6
    assert 100.0 in edges                                # the floor split
    # both variants' interior edges are present (layout-neutral grid)
    assert any(abs(e - 1528.86) < 1.0 for e in edges)    # mel interior
    assert any(abs(e - 7353.6) < 1.0 for e in edges)     # uniform interior


def test_banded_report_covers_grid() -> None:
    ref = _tone(3000.0)
    est = ref + 0.1 * _tone(9000.0)
    report = banded_report(ref, est, sr=SR)
    assert list(report.columns) == ["band", "lo_hz", "hi_hz", "band_sisdr", "band_mag_error"]
    assert len(report) == 6                              # 6 analysis bands
    # the band containing 3 kHz is scored (finite), a dead band is NaN-guarded
    mid = report[(report.lo_hz <= 3000) & (report.hi_hz > 3000)].iloc[0]
    assert not math.isnan(mid["band_sisdr"])
