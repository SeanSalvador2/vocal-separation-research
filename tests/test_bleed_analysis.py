"""Direction 06 bleed analysis (MASTER_PLAN §5, THEORY §3/§6, gate G0).

Prediction line on synthetic stems with known energy ratios: the orthogonal case
matches the closed form 10·log10(‖v‖²/(ε²‖a‖²)) exactly, the non-orthogonal case
matches the projection definition SI-SDR(v+εa, v); the measured-vs-predicted gap;
and the recovery fraction ρ with its delta-method CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from singnet.analysis import (
    measured_vs_predicted,
    predicted_curve,
    prediction_line,
    recovery_fraction,
)
from singnet.metrics.si_sdr import si_sdr


# --- prediction line: orthogonal case matches the closed form ---------------

def test_orthogonal_case_exact_equals_closed_form() -> None:
    # v ⊥ a exactly: v = [1,1,1,1], a = [1,-1,1,-1], <v,a> = 0, ‖v‖²=‖a‖²=4.
    v = np.array([1.0, 1.0, 1.0, 1.0])
    a = np.array([1.0, -1.0, 1.0, -1.0])
    eps_list = [0.05, 0.15, 0.30]
    df = prediction_line([("t", v, a)], eps_list)
    for _, row in df.iterrows():
        # orthogonal -> the exact projection value equals the closed form exactly.
        assert row["si_sdr_exact"] == pytest.approx(row["si_sdr_orth"], abs=1e-9)
        assert abs(row["ortho_gap_db"]) < 1e-9
        # and both equal -20·log10(ε) since ‖v‖² = ‖a‖² (energy ratio 0 dB).
        assert row["si_sdr_exact"] == pytest.approx(-20.0 * np.log10(row["epsilon"]), abs=1e-9)
    assert abs(df["cos_va"].iloc[0]) < 1e-12


def test_energy_ratio_anchors_the_line() -> None:
    # v louder than a: ‖v‖² = 4·‖a‖² -> +6.02 dB anchor added to -20log10(ε).
    v = np.array([2.0, 2.0, 2.0, 2.0])   # ‖v‖² = 16
    a = np.array([1.0, -1.0, 1.0, -1.0])  # ‖a‖² = 4, orthogonal to v
    df = prediction_line([("t", v, a)], [0.1])
    row = df.iloc[0]
    expected = 10 * np.log10(16.0 / (0.01 * 4.0))  # = -20log10(0.1) + 10log10(4)
    assert row["si_sdr_exact"] == pytest.approx(expected, abs=1e-9)
    assert row["energy_ratio_db"] == pytest.approx(10 * np.log10(4.0), abs=1e-9)


# --- prediction line: non-orthogonal case matches the projection definition -

def test_nonorthogonal_case_matches_projection_definition() -> None:
    v = np.array([1.0, 0.0])
    a = np.array([1.0, 1.0])   # <v,a> = 1 != 0
    df = prediction_line([("t", v, a)], [0.5])
    row = df.iloc[0]
    # hand computation (THEORY §3): est=[1.5,0.5], α=1.5, target=[1.5,0], noise=[0,-0.5]
    # -> SI-SDR = 10log10(2.25/0.25) = 10log10(9) ≈ 9.5424 dB.
    assert row["si_sdr_exact"] == pytest.approx(10 * np.log10(9.0), abs=1e-6)
    # the orthogonal approximation is 10log10(1/(0.25·2)) = 10log10(2) ≈ 3.0103 dB.
    assert row["si_sdr_orth"] == pytest.approx(10 * np.log10(2.0), abs=1e-6)
    assert row["ortho_gap_db"] > 0  # the projection value exceeds the ⊥ approximation here
    # and it equals SI-SDR(v+εa, v) computed independently.
    assert row["si_sdr_exact"] == pytest.approx(si_sdr(v + 0.5 * a, v, eps=0.0), abs=1e-9)


def test_prediction_line_shape_and_curve() -> None:
    rng = np.random.default_rng(0)
    stems = [("t%d" % i, rng.standard_normal(2048), rng.standard_normal(2048)) for i in range(3)]
    eps_list = [0.05, 0.15, 0.30]
    df = prediction_line(stems, eps_list)
    assert len(df) == 3 * len(eps_list)
    assert set(df["epsilon"]) == set(eps_list)
    curve = predicted_curve(df)
    # P(ε) is decreasing in ε (more bleed -> worse achievable SI-SDR).
    vals = curve.sort_index().to_numpy()
    assert np.all(np.diff(vals) < 0)


def test_prediction_line_accepts_pairs_and_dicts() -> None:
    v = np.array([1.0, 1.0, 1.0, 1.0])
    a = np.array([1.0, -1.0, 1.0, -1.0])
    from_pair = prediction_line([(v, a)], [0.1]).iloc[0]["si_sdr_exact"]
    from_dict = prediction_line([{"track": "x", "vocals": v, "accompaniment": a}], [0.1]).iloc[0]["si_sdr_exact"]
    assert from_pair == pytest.approx(from_dict, abs=1e-9)


# --- measured vs predicted --------------------------------------------------

def test_measured_vs_predicted_gap() -> None:
    predicted = {0.05: 20.0, 0.15: 12.0, 0.30: 6.0}
    measured = {0.05: 21.0, 0.15: 12.0, 0.30: 4.0}
    out = measured_vs_predicted(measured, predicted)
    assert list(out["epsilon"]) == [0.05, 0.15, 0.30]
    gaps = dict(zip(out["epsilon"], out["gap_db"]))
    assert gaps[0.05] == pytest.approx(1.0)    # above -> implicit robustness
    assert gaps[0.15] == pytest.approx(0.0)    # on the line -> faithful
    assert gaps[0.30] == pytest.approx(-2.0)   # below -> optimization damage


def test_measured_vs_predicted_from_curve_series() -> None:
    rng = np.random.default_rng(1)
    stems = [("t%d" % i, rng.standard_normal(4096), rng.standard_normal(4096)) for i in range(4)]
    df = prediction_line(stems, [0.05, 0.30])
    curve = predicted_curve(df)
    measured = {0.05: float(curve.loc[0.05]) + 0.5, 0.30: float(curve.loc[0.30]) - 0.5}
    out = measured_vs_predicted(measured, curve)
    assert out.loc[out["epsilon"] == 0.05, "gap_db"].iloc[0] == pytest.approx(0.5, abs=1e-6)


# --- recovery fraction + delta-method CI ------------------------------------

def test_recovery_fraction_point_estimate() -> None:
    r = recovery_fraction(s_clean=6.0, s_bleed=3.0, s_trim=4.5)
    assert r.evaluable and r.rho == pytest.approx(0.5)
    assert r.recovered_db == pytest.approx(1.5) and r.degradation_db == pytest.approx(3.0)


def test_recovery_fraction_zero_variance_ci_collapses() -> None:
    r = recovery_fraction(6.0, 3.0, 4.5, sigma_clean=0.0, sigma_bleed=0.0, sigma_trim=0.0)
    assert r.se == pytest.approx(0.0)
    assert r.ci_lo == pytest.approx(r.rho) == pytest.approx(r.ci_hi)


def test_recovery_fraction_ci_brackets_rho() -> None:
    r = recovery_fraction(6.0, 3.0, 4.5, sigma_clean=0.4, sigma_bleed=0.4, sigma_trim=0.4, n=3)
    assert r.se > 0.0
    assert r.ci_lo < r.rho < r.ci_hi


def test_recovery_fraction_not_evaluable_when_no_degradation() -> None:
    # H-06a unsupported (the model was not hurt): den <= 0 -> ρ not evaluable.
    r = recovery_fraction(s_clean=5.0, s_bleed=5.2, s_trim=5.3)
    assert not r.evaluable and np.isnan(r.rho)


def test_recovery_fraction_full_recovery() -> None:
    r = recovery_fraction(s_clean=6.0, s_bleed=3.0, s_trim=6.0)
    assert r.rho == pytest.approx(1.0)  # trimming recovers the whole gap
