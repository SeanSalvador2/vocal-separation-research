r"""Silence-Leakage Ratio (SLR) — Direction 08's core deliverable (MASTER_PLAN §2).

The field cannot *measure* ghost vocals — energy leaking into the vocal estimate
during instrumental passages. museval **drops** silent-reference frames (NaN,
code-verified in ``00-shared-research/papers/bsseval-museval-sisec2018.md``) and
SI-SDR is **singular** on a silent target (``α = ŝᵀs / ‖s‖² `` divides by 0). SLR
measures exactly that blind spot, on the silent regions the standard metrics throw
away.

Silent regions are read from the **ground-truth vocal stem**: frame-wise RMS over
100 ms frames / 50 ms hop; a frame is *silent* iff its RMS is below ``θ`` dBFS
(``θ = −60`` primary — the same amplitude threshold as Direction 01's ``sisdr``
silent-target guard, ``singnet.losses.sisdr.SILENCE_RMS_THRESHOLD``, one
project-wide notion of "silent"); contiguous silent frames form runs, and only runs
lasting ``L_min = 0.5 s`` or more (not breaths/rests) enter the silent-region set.

Per track, over the samples of the silent regions :math:`R_{\text{sil}}`, with
:math:`\hat v` the estimate and :math:`x` the mixture,

.. math::

    \text{SLR} = 10\log_{10}
        \frac{\sum_{t\in R_{\text{sil}}}\hat v(t)^2 + \varepsilon}
             {\sum_{t\in R_{\text{sil}}} x(t)^2 + \varepsilon},
        \qquad \varepsilon = 10^{-8}.

Lower is better. Anchors (derived in ``THEORY.md`` §3, asserted in
``tests/test_slr.py``): a do-nothing separator (:math:`\hat v = x`) scores **exactly
0 dB**; a perfect one (:math:`\hat v = 0` in silence) scores at the ε floor
(:math:`10\log_{10}\varepsilon/(E_x+\varepsilon)`); one passing 10 % of the mixture
energy scores **−10 dB**. Referencing :math:`x` (not the silent :math:`v=0`) is what
avoids the SI-SDR singularity — ε floors the *log*, not a projection. Tracks whose
:math:`R_{\text{sil}}` is empty are ``NaN`` and excluded from aggregation (mirroring
museval's own convention); the valid-n is always reported.
"""

from __future__ import annotations

import numpy as np

DEFAULT_EPS = 1e-8
DEFAULT_THETA_DB = -60.0
DEFAULT_FRAME_S = 0.1
DEFAULT_HOP_S = 0.05
DEFAULT_MIN_RUN_S = 0.5
#: Sensitivity axis reported on every conclusion (MASTER_PLAN §2.1).
DEFAULT_THETAS: tuple[float, ...] = (-50.0, -60.0, -70.0)

Region = tuple[int, int]


def _frame_starts(n: int, frame_len: int, hop_len: int) -> list[int]:
    """Frame start samples: ``0, hop, 2·hop, …`` while a full frame fits in ``n``.

    A track shorter than one frame yields **no** frames (empty list) — the metric
    then has no silent regions and returns ``NaN`` (edge case, unit-tested).
    """
    if n < frame_len:
        return []
    return list(range(0, n - frame_len + 1, hop_len))


def silent_regions(
    vocal_wave: np.ndarray,
    sr: int,
    theta_db: float = DEFAULT_THETA_DB,
    frame_s: float = DEFAULT_FRAME_S,
    hop_s: float = DEFAULT_HOP_S,
    min_run_s: float = DEFAULT_MIN_RUN_S,
) -> list[Region]:
    r"""Sample-index silent regions of a GT vocal stem (MASTER_PLAN §2.1).

    A frame (``frame_s`` long, ``hop_s`` hop) is *silent* iff its RMS is strictly
    below ``10**(theta_db/20)`` (θ in dBFS; the boundary frame at exactly the
    threshold is **not** silent). Consecutive silent frames merge into a run; a run
    is kept iff the sample span it covers lasts ``≥ min_run_s``. Returns half-open
    ``[start, end)`` sample intervals, clamped to the track length (a run at the
    very end does not overrun).
    """
    wave = np.asarray(vocal_wave, dtype=np.float64).reshape(-1)
    n = wave.size
    frame_len = max(1, int(round(frame_s * sr)))
    hop_len = max(1, int(round(hop_s * sr)))
    min_run_len = int(round(min_run_s * sr))
    thresh = 10.0 ** (theta_db / 20.0)  # dBFS amplitude threshold (== sisdr guard at −60)

    starts = _frame_starts(n, frame_len, hop_len)
    silent = [
        bool(np.sqrt(np.mean(wave[s : s + frame_len] ** 2)) < thresh) for s in starts
    ]

    regions: list[Region] = []
    i, n_frames = 0, len(starts)
    while i < n_frames:
        if not silent[i]:
            i += 1
            continue
        j = i
        while j + 1 < n_frames and silent[j + 1]:
            j += 1
        region_start = starts[i]
        region_end = min(starts[j] + frame_len, n)  # clamp the tail frame to the track
        if region_end - region_start >= min_run_len:
            regions.append((region_start, region_end))
        i = j + 1
    return regions


def slr(
    est_wave: np.ndarray,
    mix_wave: np.ndarray,
    regions: list[Region],
    eps: float = DEFAULT_EPS,
) -> float:
    r"""Silence-Leakage Ratio in dB over ``regions`` (``NaN`` if ``regions`` empty).

    ``NaN`` on empty regions is the museval-mirroring convention (§2.2). Exactly
    ``0.0`` when ``est_wave == mix_wave`` on the regions (do-nothing anchor), and
    invariant to a gain applied jointly to ``est`` and ``mix`` when ``eps = 0``.
    """
    if not regions:
        return float("nan")
    est = np.asarray(est_wave, dtype=np.float64).reshape(-1)
    mix = np.asarray(mix_wave, dtype=np.float64).reshape(-1)
    est_energy = float(sum(np.sum(est[s:e] ** 2) for s, e in regions))
    mix_energy = float(sum(np.sum(mix[s:e] ** 2) for s, e in regions))
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(10.0 * np.log10((est_energy + eps) / (mix_energy + eps)))


def slr_report(
    est: np.ndarray,
    mix: np.ndarray,
    vocal_gt: np.ndarray,
    sr: int,
    thetas: tuple[float, ...] = DEFAULT_THETAS,
    *,
    frame_s: float = DEFAULT_FRAME_S,
    hop_s: float = DEFAULT_HOP_S,
    min_run_s: float = DEFAULT_MIN_RUN_S,
    eps: float = DEFAULT_EPS,
) -> dict[float, dict[str, float]]:
    """SLR + valid-n at each θ in ``thetas`` (the §2.1 sensitivity report).

    Returns ``{theta: {"slr", "n_regions", "silent_samples"}}``. ``slr`` is ``NaN``
    for a θ under which the track has no ≥ ``min_run_s`` silent run (excluded from
    aggregation upstream; the valid-n is ``n_regions``).
    """
    report: dict[float, dict[str, float]] = {}
    for theta in thetas:
        regions = silent_regions(
            vocal_gt, sr, theta_db=theta, frame_s=frame_s, hop_s=hop_s, min_run_s=min_run_s
        )
        report[float(theta)] = {
            "slr": slr(est, mix, regions, eps=eps),
            "n_regions": float(len(regions)),
            "silent_samples": float(sum(e - s for s, e in regions)),
        }
    return report
