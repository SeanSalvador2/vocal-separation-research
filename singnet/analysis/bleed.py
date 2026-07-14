r"""Bleed dose–response analysis (Direction 06 §5, THEORY §3, §6).

Pure, dependency-light functions the notebooks *call* (never re-implement): the
**prediction line** :math:`P(\varepsilon)` from clean stems, the measured-vs-
predicted gap, and the recovery fraction :math:`\rho` with a delta-method CI.
NumPy/pandas + the one SI-SDR implementation of record
(:func:`singnet.metrics.si_sdr.si_sdr`); no SciPy, no torch, no I/O in the pure
functions (a thin RUN-LATER CLI at the bottom reads real stems).

**The prediction line (THEORY §3).** A model that perfectly learns the corrupted
conditional outputs :math:`\hat v = \tilde v = v + \varepsilon a`, so its clean
SI-SDR against the true :math:`v` is, *exactly* (the projection definition of
SI-SDR),

.. math::

    P_{\text{track}}(\varepsilon) = \text{SI-SDR}(v + \varepsilon a,\; v),

computed numerically from the stems. When :math:`v \perp a` this collapses to the
closed form

.. math::

    P_{\text{track}}(\varepsilon) = 10\log_{10}\frac{\lVert v\rVert^2}
        {\varepsilon^2 \lVert a\rVert^2}
        = -20\log_{10}\varepsilon + 10\log_{10}\frac{\lVert v\rVert^2}{\lVert a\rVert^2},

a :math:`-20\log_{10}\varepsilon` line anchored by each track's vocal/accompaniment
energy ratio. :func:`prediction_line` returns **both** columns per (track, ε);
their difference is the non-orthogonality gap. The measured curve sitting *above*
the prediction is implicit robustness; *below* is optimization damage (§2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from ..metrics.si_sdr import si_sdr


def _iter_stems(stems_iterable: Iterable[Any]) -> Iterable[tuple[str, np.ndarray, np.ndarray]]:
    """Normalize a stems iterable to ``(track, vocals, accompaniment)`` triples.

    Accepts, per item: a ``(track, v, a)`` triple, a ``(v, a)`` pair (auto-named by
    index), or a mapping with ``vocals``/``accompaniment`` (and optional
    ``track``/``name``) keys.
    """
    for i, item in enumerate(stems_iterable):
        if isinstance(item, Mapping):
            name = str(item.get("track", item.get("name", i)))
            v = np.asarray(item["vocals"], dtype=np.float64).reshape(-1)
            a = np.asarray(item["accompaniment"], dtype=np.float64).reshape(-1)
        elif len(item) == 3:
            name, v, a = item
            name = str(name)
            v = np.asarray(v, dtype=np.float64).reshape(-1)
            a = np.asarray(a, dtype=np.float64).reshape(-1)
        elif len(item) == 2:
            v, a = item
            name = str(i)
            v = np.asarray(v, dtype=np.float64).reshape(-1)
            a = np.asarray(a, dtype=np.float64).reshape(-1)
        else:  # pragma: no cover - defensive
            raise ValueError(f"cannot interpret stems item {item!r}")
        yield name, v, a


def _orthogonal_prediction_db(vocal_energy: float, acc_energy: float, epsilon: float) -> float:
    r"""``10·log10(‖v‖² / (ε²‖a‖²))`` — the ``v ⊥ a`` closed form (THEORY §3)."""
    if epsilon == 0.0:
        return float("inf")  # a perfect estimate at ε = 0
    denom = (epsilon**2) * acc_energy
    if denom <= 0.0:
        return float("inf")
    return float(10.0 * np.log10(vocal_energy / denom))


def prediction_line(
    stems_iterable: Iterable[Any], epsilons: Iterable[float], *, eps_sisdr: float = 0.0
) -> pd.DataFrame:
    r"""Per-(track, ε) prediction line: exact projection SI-SDR **and** the ``v⊥a`` form.

    For each clean-stem pair ``(v, a)`` and each ε, computes

    * ``si_sdr_exact`` — ``SI-SDR(v + εa, v)`` numerically from the stems (the exact
      projection form; ``eps_sisdr = 0`` for the eps-free THEORY definition on
      non-degenerate stems), and
    * ``si_sdr_orth`` — ``10·log10(‖v‖²/(ε²‖a‖²))`` (equal to the exact value iff
      ``v ⊥ a``; the two coincide in the orthogonal case and diverge otherwise).

    Returns a tidy DataFrame with columns ``track, epsilon, si_sdr_exact,
    si_sdr_orth, ortho_gap_db, vocal_energy, acc_energy, energy_ratio_db, cos_va``.
    ``ortho_gap_db = si_sdr_exact − si_sdr_orth`` and ``cos_va`` is the stems'
    cosine (0 ⇔ orthogonal). The curve ``P(ε)`` is the per-ε mean over tracks
    (see :func:`predicted_curve`).
    """
    eps_list = [float(e) for e in epsilons]
    rows: list[dict[str, Any]] = []
    for name, v, a in _iter_stems(stems_iterable):
        vocal_energy = float(v @ v)
        acc_energy = float(a @ a)
        cross = float(v @ a)
        cos_va = cross / (np.sqrt(vocal_energy * acc_energy) + 1e-20)
        energy_ratio_db = (
            float(10.0 * np.log10(vocal_energy / acc_energy)) if acc_energy > 0 else float("inf")
        )
        for eps in eps_list:
            estimate = v + eps * a
            exact = si_sdr(estimate, v, eps=eps_sisdr)
            orth = _orthogonal_prediction_db(vocal_energy, acc_energy, eps)
            rows.append(
                {
                    "track": name,
                    "epsilon": eps,
                    "si_sdr_exact": exact,
                    "si_sdr_orth": orth,
                    "ortho_gap_db": exact - orth,
                    "vocal_energy": vocal_energy,
                    "acc_energy": acc_energy,
                    "energy_ratio_db": energy_ratio_db,
                    "cos_va": cos_va,
                }
            )
    return pd.DataFrame(rows)


def predicted_curve(prediction_df: pd.DataFrame, *, column: str = "si_sdr_exact") -> pd.Series:
    """The prediction line ``P(ε)`` — the per-ε mean over tracks of ``column``.

    ``column`` is ``si_sdr_exact`` (default; the projection form used for the
    headline overlay) or ``si_sdr_orth`` (the orthogonal approximation).
    """
    return prediction_df.groupby("epsilon")[column].mean()


def _as_epsilon_series(obj: Mapping[float, float] | pd.Series | pd.DataFrame, value_col: str) -> pd.Series:
    """Coerce a ε→value mapping / Series / two-column frame to a Series indexed by ε."""
    if isinstance(obj, pd.Series):
        return obj.sort_index()
    if isinstance(obj, pd.DataFrame):
        return obj.set_index("epsilon")[value_col].sort_index()
    return pd.Series({float(k): float(v) for k, v in obj.items()}).sort_index()


def measured_vs_predicted(
    measured: Mapping[float, float] | pd.Series | pd.DataFrame,
    predicted: Mapping[float, float] | pd.Series | pd.DataFrame,
) -> pd.DataFrame:
    r"""The measured-vs-predicted gap per ε (the §2 scientific payload).

    ``measured`` is the arm's measured clean SI-SDR at each ε (3-seed mean for the
    seeded cells); ``predicted`` is ``P(ε)`` (e.g. from :func:`predicted_curve`).
    Both may be a ``{ε: value}`` mapping, a Series indexed by ε, or a frame with an
    ``epsilon`` column (plus ``measured``/``predicted``). Returns a DataFrame with
    ``epsilon, measured, predicted, gap_db`` on the shared ε grid, where
    ``gap_db = measured − predicted``: ``> 0`` implicit robustness, ``< 0``
    optimization damage, ``≈ 0`` faithful learning of the corrupted conditional.
    """
    m = _as_epsilon_series(measured, "measured")
    p = _as_epsilon_series(predicted, "predicted")
    grid = sorted(set(m.index) & set(p.index))
    rows = [
        {
            "epsilon": eps,
            "measured": float(m.loc[eps]),
            "predicted": float(p.loc[eps]),
            "gap_db": float(m.loc[eps] - p.loc[eps]),
        }
        for eps in grid
    ]
    return pd.DataFrame(rows)


@dataclass
class RecoveryResult:
    r"""Recovery fraction ρ with its delta-method CI (H-06b, THEORY §6).

    Attributes:
        rho: :math:`\rho = (s_{\text{trim}} - s_{\text{bleed}}) /
            (s_{\text{clean}} - s_{\text{bleed}})` — the fraction of the bleed
            degradation that trimming buys back. ``nan`` when the degradation
            ``den ≤ 0`` (H-06a unsupported ⇒ "not evaluable", §2).
        se: delta-method standard error of ρ.
        ci_lo, ci_hi: the ``z``-scaled CI on ρ.
        recovered_db: numerator :math:`s_{\text{trim}} - s_{\text{bleed}}`.
        degradation_db: denominator :math:`s_{\text{clean}} - s_{\text{bleed}}`.
        evaluable: ``False`` when ``den ≤ 0`` (ρ is nan).
    """

    rho: float
    se: float
    ci_lo: float
    ci_hi: float
    recovered_db: float
    degradation_db: float
    evaluable: bool


def recovery_fraction(
    s_clean: float,
    s_bleed: float,
    s_trim: float,
    *,
    sigma_clean: float = 0.0,
    sigma_bleed: float = 0.0,
    sigma_trim: float = 0.0,
    n: int = 3,
    z: float = 1.96,
) -> RecoveryResult:
    r"""Recovery fraction ρ and its delta-method CI (H-06b).

    ρ is the share of the bleed degradation ``s_clean − s_bleed`` recovered by
    trimming, ``ρ = (s_trim − s_bleed)/(s_clean − s_bleed)``. Treating the three
    3-seed cell means as independent with per-mean variances ``σ²/n`` (``σ`` the
    between-seed std of each cell), the **delta method** linearizes ρ:

    .. math::

        \widehat{\mathrm{Var}}(\rho) \approx
            \Big(\tfrac{1}{d}\Big)^2\tfrac{\sigma_{\text{trim}}^2}{n}
          + \Big(\tfrac{u}{d^2}\Big)^2\tfrac{\sigma_{\text{clean}}^2}{n}
          + \Big(\tfrac{u-d}{d^2}\Big)^2\tfrac{\sigma_{\text{bleed}}^2}{n},

    with ``u = s_trim − s_bleed`` and ``d = s_clean − s_bleed``. The CI is
    ``ρ ± z·SE``. When ``d ≤ 0`` (the model was *not* hurt, H-06a unsupported) ρ is
    not evaluable and returned as ``nan`` (MASTER_PLAN §2).
    """
    u = float(s_trim - s_bleed)
    d = float(s_clean - s_bleed)
    if d <= 0.0:
        return RecoveryResult(float("nan"), float("nan"), float("nan"), float("nan"), u, d, False)

    rho = u / d
    var = (
        (1.0 / d) ** 2 * (float(sigma_trim) ** 2 / n)
        + (u / d**2) ** 2 * (float(sigma_clean) ** 2 / n)
        + ((u - d) / d**2) ** 2 * (float(sigma_bleed) ** 2 / n)
    )
    se = float(np.sqrt(var))
    return RecoveryResult(float(rho), se, float(rho - z * se), float(rho + z * se), u, d, True)


# --- RUN LATER: prediction line from real clean stems -----------------------

def _cli(argv: list[str] | None = None) -> None:  # pragma: no cover - RUN LATER, needs data
    r"""``python -m singnet.analysis.bleed --split valid --epsilons 0.05 0.15 0.30``.

    RUN LATER (CPU-minutes, needs the decoded WAV shards). Reads the *clean* stems
    of a split via :class:`singnet.data.WavShardStore`, computes the prediction
    line, and writes/prints ``P(ε)``. Independent of any run output — the
    prediction is a property of the clean stems alone (the §7 G3 audit relies on
    this: the prediction never touches a corrupted or eval-leaked target).
    """
    import argparse
    from pathlib import Path

    from ..data import Manifest, WavShardStore

    parser = argparse.ArgumentParser(description="Direction 06 prediction line (RUN LATER; needs shards).")
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--epsilons", nargs="+", type=float, default=[0.05, 0.15, 0.30])
    parser.add_argument("--shard-root", default=None, help="decoded WAV shard root")
    parser.add_argument("--splits-csv", default="01-loss-function-study/configs/splits.csv")
    parser.add_argument("--out", default=None, help="optional CSV output path")
    args = parser.parse_args(argv)

    shard_root = args.shard_root
    if not shard_root or not Path(shard_root).exists():
        raise FileNotFoundError(
            f"shard_root {shard_root!r} missing — run scripts/prepare_data.py first (RUN LATER)."
        )
    manifest = Manifest.from_csv(args.splits_csv)
    store = WavShardStore(shard_root)
    tracks = manifest.tracks_for(args.split)

    def _stems() -> Iterable[tuple[str, np.ndarray, np.ndarray]]:
        for track in tracks:
            src = store.load_sources(track)
            yield track, src["vocals"], src["accompaniment"]

    frame = prediction_line(_stems(), args.epsilons)
    curve = predicted_curve(frame)
    print(f"prediction line P(ε) on split={args.split} ({len(tracks)} tracks):")
    print(curve.to_string())
    if args.out:
        frame.to_csv(args.out, index=False)
        print(f"wrote per-track prediction line -> {args.out}")


if __name__ == "__main__":  # pragma: no cover
    _cli()
