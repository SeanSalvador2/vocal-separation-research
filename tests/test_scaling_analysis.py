"""Direction 02 scaling / LOO analysis functions (G0, §6).

Covers the gate-G0 analysis items: recovery of a known log2 slope, bootstrap-CI
sanity, per-doubling secant slopes, the pooled σ_seed, and the leave-one-out
Δ table (with its interaction gap) on a synthetic registry.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from singnet.analysis import fit_log2, loo_table, pooled_seed_sigma, secant_slopes

NS = [21, 43, 64, 86]


def test_fit_log2_recovers_known_slope() -> None:
    a0, b0 = 3.0, 1.7
    scores = [a0 + b0 * math.log2(n) for n in NS]
    fit = fit_log2(NS, scores, sigma=1e-9)
    assert abs(fit.a - a0) < 1e-6
    assert abs(fit.b - b0) < 1e-6
    # a (near-)noiseless fit has a (near-)degenerate CI around the true slope.
    assert fit.b_ci[0] <= fit.b <= fit.b_ci[1]
    assert fit.b_ci[1] - fit.b_ci[0] < 1e-3


def test_bootstrap_ci_brackets_slope_and_widens_with_sigma() -> None:
    scores = [3.0 + 1.7 * math.log2(n) for n in NS]
    tight = fit_log2(NS, scores, sigma=0.3, seed=0)
    wide = fit_log2(NS, scores, sigma=1.0, seed=0)
    for fit in (tight, wide):
        assert fit.b_ci[0] < fit.b < fit.b_ci[1]
    assert (wide.b_ci[1] - wide.b_ci[0]) > (tight.b_ci[1] - tight.b_ci[0])
    assert wide.b_se > tight.b_se


def test_fit_log2_defaults_sigma_to_residual() -> None:
    # Non-collinear points: a CI is still produced without an explicit sigma.
    scores = [1.0, 2.5, 2.7, 4.0]
    fit = fit_log2(NS, scores)  # sigma=None -> residual SE
    assert fit.b_ci[0] < fit.b < fit.b_ci[1]
    assert not math.isnan(fit.sigma)


def test_secant_slopes_constant_for_log_linear() -> None:
    scores = [2.0 + 1.25 * math.log2(n) for n in NS]
    slopes = secant_slopes(NS, scores)["slope_db_per_doubling"].to_numpy()
    assert np.allclose(slopes, 1.25)
    assert len(slopes) == len(NS) - 1


def test_secant_slope_is_per_doubling() -> None:
    # A jump of +2 dB across an exact doubling (21 -> 42) is a slope of +2/doubling.
    df = secant_slopes([21, 42], [5.0, 7.0])
    assert abs(float(df["slope_db_per_doubling"].iloc[0]) - 2.0) < 1e-9


def test_pooled_seed_sigma_matches_formula() -> None:
    full = [10.0, 11.0, 12.0]   # sample variance 1.0
    n21 = [4.0, 7.0, 10.0]      # sample variance 9.0
    assert abs(pooled_seed_sigma(full, n21) - math.sqrt((1.0 + 9.0) / 2.0)) < 1e-9
    assert math.isnan(pooled_seed_sigma([5.0]))  # <2 seeds -> undefined


def _loo_registry() -> pd.DataFrame:
    rows = []
    for seed, val in zip((0, 1, 2), (5.0, 5.1, 5.2)):  # full mean = 5.1
        rows.append(dict(aug_remix=True, aug_gain=True, aug_flip=True, n_songs=86,
                         best_val_sisdr=val, seed=seed))
    rows += [
        dict(aug_remix=False, aug_gain=True, aug_flip=True, n_songs=86, best_val_sisdr=3.1, seed=0),
        dict(aug_remix=True, aug_gain=False, aug_flip=True, n_songs=86, best_val_sisdr=4.6, seed=0),
        dict(aug_remix=True, aug_gain=True, aug_flip=False, n_songs=86, best_val_sisdr=5.05, seed=0),
        dict(aug_remix=False, aug_gain=False, aug_flip=False, n_songs=86, best_val_sisdr=2.6, seed=0),
    ]
    return pd.DataFrame(rows)


def test_loo_table_deltas_and_interaction() -> None:
    table = loo_table(_loo_registry()).set_index("contrast")
    assert abs(table.loc["full", "val_score"] - 5.1) < 1e-9
    assert abs(table.loc["no_remix", "delta"] - 2.0) < 1e-9
    assert abs(table.loc["no_gain", "delta"] - 0.5) < 1e-9
    assert abs(table.loc["no_flip", "delta"] - 0.05) < 1e-9
    assert abs(table.loc["none", "delta"] - 2.5) < 1e-9  # Δ_total
    # interaction gap = Δ_total - ΣΔ_t = 2.5 - (2.0 + 0.5 + 0.05) = -0.05
    assert abs(table.loc["interaction_gap", "delta"] - (-0.05)) < 1e-9
    assert int(table.loc["full", "n_runs"]) == 3  # averaged over three seeds


def test_loo_table_handles_missing_cell() -> None:
    reg = _loo_registry()
    reg = reg[~((reg.aug_remix) & (~reg.aug_gain) & (reg.aug_flip))]  # drop no_gain
    table = loo_table(reg).set_index("contrast")
    assert math.isnan(table.loc["no_gain", "delta"])
    assert math.isnan(table.loc["interaction_gap", "delta"])  # NaN propagates honestly
    assert abs(table.loc["no_remix", "delta"] - 2.0) < 1e-9   # others still computed


def test_loo_table_accepts_string_booleans() -> None:
    # A registry round-tripped through CSV yields 'True'/'False' strings.
    reg = _loo_registry().astype({"aug_remix": str, "aug_gain": str, "aug_flip": str})
    table = loo_table(reg).set_index("contrast")
    assert abs(table.loc["no_remix", "delta"] - 2.0) < 1e-9
