"""Analysis helpers for Direction 02 (augmentation factorization + data scaling).

Pure functions the notebooks call to turn the run registry into the pre-registered
figures and verdicts: the log2 scaling fit with a bootstrap slope CI, per-doubling
secant slopes, the leave-one-out Δ table with its interaction gap, and the pooled
between-seed noise σ_seed. See :mod:`singnet.analysis.scaling`.
"""

from __future__ import annotations

from .bandsplit import (
    OrderedChainVerdict,
    ordered_chain_verdict,
    three_arm_sigma,
)
from .bleed import (
    RecoveryResult,
    measured_vs_predicted,
    predicted_curve,
    prediction_line,
    recovery_fraction,
)
from .scaling import (
    FitResult,
    fit_log2,
    loo_table,
    pooled_seed_sigma,
    secant_slopes,
)

__all__ = [
    "FitResult",
    "fit_log2",
    "loo_table",
    "pooled_seed_sigma",
    "secant_slopes",
    "OrderedChainVerdict",
    "ordered_chain_verdict",
    "three_arm_sigma",
    "prediction_line",
    "predicted_curve",
    "measured_vs_predicted",
    "recovery_fraction",
    "RecoveryResult",
]
