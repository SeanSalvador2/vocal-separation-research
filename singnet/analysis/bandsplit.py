r"""Three-arm ordered-chain statistics for the band-split verdict (Direction 03 §2).

Implements the pre-registered H-03 decision logic so the notebook stays thin and
the verdict is unit-tested (gate G0). Given per-seed validation vocals SI-SDR for
the three arms (baseline, uniform-split, mel-split):

* pooled between-seed noise
  :math:`\sigma_{\text{seed}} = \sqrt{(s^2_{\text{base}} + s^2_{\text{uni}} +
  s^2_{\text{mel}})/3}` (reuses :func:`singnet.analysis.scaling.pooled_seed_sigma`),
* the two ordered effects
  :math:`E_1 = \overline{v}(\text{uni}) - \overline{v}(\text{base})` (splitting
  helps) and :math:`E_2 = \overline{v}(\text{mel}) - \overline{v}(\text{uni})`
  (mel spacing helps beyond splitting),
* the verdict cell (§2, §13): fully supported (both links > σ), partial (exactly
  one), refuted-null (neither, all three within ±σ), or refuted-negative (baseline
  beats both by > σ).

The two links are **jointly** pre-registered as one ordered claim, so they are
*not* corrected as multiple comparisons (THEORY §6).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scaling import pooled_seed_sigma

ArrayLike = "np.ndarray | list[float]"


@dataclass
class OrderedChainVerdict:
    """The H-03 readout (means, σ_seed, the two effects, and the verdict cell)."""

    mean_baseline: float
    mean_uniform: float
    mean_mel: float
    sigma_seed: float
    e1: float                 # uniform − baseline (splitting)
    e2: float                 # mel − uniform (mel spacing beyond splitting)
    e1_holds: bool            # E1 > σ_seed
    e2_holds: bool            # E2 > σ_seed
    verdict: str              # one of the §13 cells


def three_arm_sigma(baseline: ArrayLike, uniform: ArrayLike, mel: ArrayLike) -> float:
    """Pooled between-seed σ over the three arms (√ of the mean of the 3 variances)."""
    return pooled_seed_sigma(baseline, uniform, mel)


def ordered_chain_verdict(
    baseline: ArrayLike, uniform: ArrayLike, mel: ArrayLike
) -> OrderedChainVerdict:
    """Evaluate the pre-registered ordered-chain hypothesis H-03 (§2).

    Args:
        baseline / uniform / mel: per-seed best-checkpoint validation vocals
            SI-SDR for the three arms (each a length-``n_seeds`` sequence).

    Returns:
        :class:`OrderedChainVerdict` with the arm means, pooled σ_seed, the two
        ordered effects and whether each clears the noise band, and the verdict
        cell (``fully_supported`` | ``partial_E1`` | ``partial_E2`` |
        ``refuted_null`` | ``refuted_negative``).
    """
    mean_b = float(np.mean(baseline))
    mean_u = float(np.mean(uniform))
    mean_m = float(np.mean(mel))
    sigma = three_arm_sigma(baseline, uniform, mel)

    e1 = mean_u - mean_b
    e2 = mean_m - mean_u
    e1_holds = e1 > sigma
    e2_holds = e2 > sigma

    if e1_holds and e2_holds:
        verdict = "fully_supported"                       # mel > uniform > baseline
    elif e1_holds:
        verdict = "partial_E1"                            # splitting helps, mel ≈ uniform
    elif e2_holds:
        verdict = "partial_E2"                            # mel > uniform, but uniform ≤ baseline
    else:
        baseline_beats_both = (mean_b - mean_u > sigma) and (mean_b - mean_m > sigma)
        verdict = "refuted_negative" if baseline_beats_both else "refuted_null"

    return OrderedChainVerdict(
        mean_baseline=mean_b, mean_uniform=mean_u, mean_mel=mean_m, sigma_seed=sigma,
        e1=e1, e2=e2, e1_holds=e1_holds, e2_holds=e2_holds, verdict=verdict,
    )
