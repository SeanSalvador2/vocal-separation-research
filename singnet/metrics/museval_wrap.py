"""Thin wrapper around ``museval`` BSS-Eval SDR — the *secondary* metric.

``museval`` (and its ``musdb`` companion) are heavy, data-time dependencies. They
are intentionally **not** imported at module load: the import happens inside the
functions so the default test suite never needs them (tests guard with
``pytest.importorskip('museval')``). MASTER_PLAN §7.4 keeps these BSS-Eval SDR
numbers in their own clearly-labelled table, never mixed with SI-SDR.

museval reports the median-over-frames / median-over-tracks SDR convention; the
museval version is recorded in the eval CSV for reproducibility (§12 version
drift risk).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def museval_version() -> str:
    """Return the installed museval version (raises if museval is absent)."""
    import museval  # local import: optional dependency

    return getattr(museval, "__version__", "unknown")


def bss_eval_sdr(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int = 44100,
    win: float = 1.0,
    hop: float = 1.0,
) -> dict[str, Any]:
    """Compute framewise BSS-Eval v4 metrics for one source via ``museval``.

    Args:
        reference: ``(n_samples,)`` or ``(n_samples, n_channels)`` target.
        estimate:  same shape as ``reference``.
        sample_rate: sampling rate (Hz).
        win, hop: BSS-Eval evaluation window / hop in seconds.

    Returns:
        Dict with ``sdr``/``isr``/``sir``/``sar`` arrays (per frame) plus their
        nan-medians (the museum-standard median-of-frames aggregation) and the
        museval version string.
    """
    import museval  # local import: optional dependency

    ref = _as_2d(reference)
    est = _as_2d(estimate)
    sdr, isr, sir, sar = museval.evaluate(
        ref[np.newaxis, ...], est[np.newaxis, ...], win=int(win * sample_rate), hop=int(hop * sample_rate)
    )
    return {
        "sdr": sdr[0],
        "isr": isr[0],
        "sir": sir[0],
        "sar": sar[0],
        "sdr_median": float(np.nanmedian(sdr[0])),
        "sir_median": float(np.nanmedian(sir[0])),
        "sar_median": float(np.nanmedian(sar[0])),
        "museval_version": museval_version(),
    }


def _as_2d(x: np.ndarray) -> np.ndarray:
    """Coerce to ``(n_samples, n_channels)`` float64 as museval expects."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    return x
