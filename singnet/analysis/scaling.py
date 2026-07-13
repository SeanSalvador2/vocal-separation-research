"""Scaling-curve and leave-one-out analysis (MASTER_PLAN §6, §9).

Pure, dependency-light functions the notebooks *call* (never re-implement): the
log2 scaling fit with a parametric-bootstrap CI on the slope, the per-doubling
secant slopes, the leave-one-out Δ table with its interaction gap, and the pooled
between-seed noise σ_seed. Everything is NumPy/pandas only (no SciPy) and
unit-tested (gate G0): known-slope recovery, bootstrap-CI sanity, and the LOO
table on a synthetic registry.

Conventions.

* The scaling functions take **one score per distinct song-count N** (the seed
  *mean* at that N); seed spread enters only through the ``sigma`` noise argument
  and the endpoint bands drawn in the notebook.
* Slopes are in **dB per doubling** of songs — i.e. per unit of ``log2(N)`` — so
  the number is directly comparable across the {21, 43, 64, 86} points whether or
  not a given pair is an exact doubling (MASTER_PLAN §6; H-02b decision rule).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ArrayLike = "np.ndarray | list[float]"


@dataclass
class FitResult:
    """A log2 scaling fit ``SISDR(N) ≈ a + b·log2(N)`` with a bootstrap CI on ``b``.

    Attributes:
        a: intercept (dB at ``N = 1``; extrapolated, not physical).
        b: slope in **dB per doubling** of songs — the H-02b headline number.
        b_ci: ``(lo, hi)`` percentile CI on ``b`` from the parametric bootstrap.
        b_se: bootstrap standard error of ``b`` (the CI's spread as one number).
        sigma: the per-point noise used (scalar summary; ``σ_seed`` in practice).
    """

    a: float
    b: float
    b_ci: tuple[float, float]
    b_se: float = float("nan")
    sigma: float = float("nan")


def _as_float_arrays(ns: ArrayLike, scores: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    ns = np.asarray(ns, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if ns.shape != scores.shape:
        raise ValueError(f"ns and scores must align; got {ns.shape} vs {scores.shape}")
    if ns.ndim != 1 or ns.size < 2:
        raise ValueError("need at least two (N, score) points for a scaling fit")
    if np.any(ns <= 0):
        raise ValueError("song counts N must be positive (log2 is taken)")
    return ns, scores


def _ols_slope_intercept(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Closed-form simple-linear-regression slope/intercept of ``y`` on ``x``."""
    xbar = x.mean()
    sxx = np.sum((x - xbar) ** 2)
    if sxx == 0:
        raise ValueError("all N are identical — the slope is unidentifiable")
    b = float(np.sum((x - xbar) * (y - y.mean())) / sxx)
    a = float(y.mean() - b * xbar)
    return a, b


def fit_log2(
    ns: ArrayLike,
    scores: ArrayLike,
    sigma: "float | np.ndarray | None" = None,
    *,
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> FitResult:
    """Least-squares fit of ``score = a + b·log2(N)`` with a parametric-bootstrap CI.

    The point estimate is ordinary least squares on ``x = log2(N)``. The CI on the
    slope ``b`` comes from a **parametric bootstrap** (MASTER_PLAN §6): draw
    ``n_boot`` synthetic datasets ``y* = y + N(0, sigma)`` with the given *per-point*
    noise, refit each, and take percentiles of the ``b*`` distribution.

    Args:
        ns: distinct song counts N (e.g. ``[21, 43, 64, 86]``).
        scores: the val-SI-SDR *mean* at each N (one per N).
        sigma: per-point noise — a scalar (broadcast) or an array aligned to ``ns``.
            Pass ``σ_seed`` here. If ``None``, the fit's residual standard error
            (dof ``= n − 2``) is used so a CI is still returned.
        n_boot: bootstrap draws (10,000 pre-registered).
        ci: central interval mass (0.95 → the 2.5/97.5 percentiles).
        seed: RNG seed for the bootstrap (reproducible).

    Returns:
        :class:`FitResult` with ``a``, ``b``, the ``b`` CI, its bootstrap SE, and
        the scalar noise used.
    """
    ns, scores = _as_float_arrays(ns, scores)
    x = np.log2(ns)
    a, b = _ols_slope_intercept(x, scores)

    resid = scores - (a + b * x)
    if sigma is None:
        dof = max(1, ns.size - 2)
        sig = np.full(ns.size, float(np.sqrt(np.sum(resid**2) / dof)))
    else:
        sig = np.broadcast_to(np.asarray(sigma, dtype=float), ns.shape).astype(float)

    xbar = x.mean()
    sxx = np.sum((x - xbar) ** 2)
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(n_boot, ns.size)) * sig[None, :]
    y_star = scores[None, :] + noise
    # Vectorized simple-regression slope for every bootstrap replicate.
    b_star = ((x - xbar)[None, :] * (y_star - y_star.mean(axis=1, keepdims=True))).sum(axis=1) / sxx

    lo_q = 100.0 * (1.0 - ci) / 2.0
    hi_q = 100.0 - lo_q
    b_ci = (float(np.percentile(b_star, lo_q)), float(np.percentile(b_star, hi_q)))
    return FitResult(a=a, b=b, b_ci=b_ci, b_se=float(b_star.std(ddof=1)), sigma=float(np.mean(sig)))


def secant_slopes(ns: ArrayLike, scores: ArrayLike) -> pd.DataFrame:
    """Per-doubling secant slopes between consecutive points (MASTER_PLAN §6).

    Answers "is the curve bending?" without leaning on the 2-parameter fit: each
    row is the slope ``Δscore / Δlog2(N)`` (dB per doubling) between adjacent N.
    Points are sorted by N first.

    Returns a DataFrame with columns
    ``n_from, n_to, score_from, score_to, delta_songs, slope_db_per_doubling``.
    """
    ns, scores = _as_float_arrays(ns, scores)
    order = np.argsort(ns)
    ns, scores = ns[order], scores[order]
    rows = []
    for i in range(ns.size - 1):
        dlog = np.log2(ns[i + 1]) - np.log2(ns[i])
        slope = float((scores[i + 1] - scores[i]) / dlog) if dlog != 0 else float("nan")
        rows.append(
            {
                "n_from": float(ns[i]),
                "n_to": float(ns[i + 1]),
                "score_from": float(scores[i]),
                "score_to": float(scores[i + 1]),
                "delta_songs": float(ns[i + 1] - ns[i]),
                "slope_db_per_doubling": slope,
            }
        )
    return pd.DataFrame(rows)


def pooled_seed_sigma(*groups: ArrayLike) -> float:
    """Pooled between-seed std σ_seed = sqrt(mean of per-group sample variances).

    For the two 3-seed anchor cells this is exactly
    ``sqrt((s²_FULL + s²_n21) / 2)`` (MASTER_PLAN §2). Groups with fewer than two
    seeds contribute no variance estimate (they are skipped).
    """
    variances = [np.var(np.asarray(g, dtype=float), ddof=1) for g in groups if np.size(g) >= 2]
    if not variances:
        return float("nan")
    return float(np.sqrt(np.mean(variances)))


def _as_bool(value: object) -> bool:
    """Coerce registry cell values (bool / 0-1 / 'True'/'False'/'1'/'0') to bool."""
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def loo_table(
    registry_df: pd.DataFrame,
    *,
    score_col: str = "best_val_sisdr",
    n_full: int | None = None,
) -> pd.DataFrame:
    """Leave-one-out Δ table with the interaction gap (MASTER_PLAN §2, §6).

    Reads a registry frame carrying ``aug_remix, aug_gain, aug_flip, n_songs`` and
    ``score_col``. On the full split (``n_songs == n_full``, default the max N):

    * ``full`` = mean score with all three transforms on (averaged over seeds),
    * ``no_t`` rows give ``Δ_t = val(full) − val(full∖t)`` for t in remix/gain/flip,
    * ``none`` gives ``Δ_total = val(full) − val(none)``,
    * ``interaction_gap`` = ``Δ_total − Σ_t Δ_t`` (transform synergy/redundancy).

    Missing cells yield NaN deltas (so a partially-populated registry degrades
    gracefully rather than raising). Returns a DataFrame with columns
    ``contrast, remix, gain, flip, n_runs, val_score, delta``.
    """
    df = registry_df.copy()
    for col in ("aug_remix", "aug_gain", "aug_flip"):
        if col not in df.columns:
            raise KeyError(f"registry missing switchboard column {col!r}")
        df[col] = df[col].map(_as_bool)
    if "n_songs" not in df.columns:
        raise KeyError("registry missing 'n_songs' column")

    n_full = int(pd.to_numeric(df["n_songs"]).max()) if n_full is None else int(n_full)
    at_full = df[pd.to_numeric(df["n_songs"]) == n_full]

    def cell(remix: bool, gain: bool, flip: bool) -> tuple[float, int]:
        sub = at_full[
            (at_full["aug_remix"] == remix)
            & (at_full["aug_gain"] == gain)
            & (at_full["aug_flip"] == flip)
        ]
        return (float(sub[score_col].mean()), int(len(sub))) if len(sub) else (float("nan"), 0)

    val_full, n_full_runs = cell(True, True, True)
    rows = [
        {"contrast": "full", "remix": True, "gain": True, "flip": True,
         "n_runs": n_full_runs, "val_score": val_full, "delta": 0.0}
    ]
    deltas = []
    for name, (remix, gain, flip) in {
        "no_remix": (False, True, True),
        "no_gain": (True, False, True),
        "no_flip": (True, True, False),
    }.items():
        val, n_runs = cell(remix, gain, flip)
        delta = val_full - val
        deltas.append(delta)
        rows.append({"contrast": name, "remix": remix, "gain": gain, "flip": flip,
                     "n_runs": n_runs, "val_score": val, "delta": delta})

    val_none, n_none = cell(False, False, False)
    delta_total = val_full - val_none
    rows.append({"contrast": "none", "remix": False, "gain": False, "flip": False,
                 "n_runs": n_none, "val_score": val_none, "delta": delta_total})
    rows.append({"contrast": "interaction_gap", "remix": None, "gain": None, "flip": None,
                 "n_runs": 0, "val_score": float("nan"), "delta": delta_total - float(np.sum(deltas))})
    return pd.DataFrame(rows)
