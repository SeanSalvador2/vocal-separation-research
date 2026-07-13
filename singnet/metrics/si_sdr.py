r"""Scale-invariant signal-to-distortion ratio (Le Roux et al., 2019).

The definition of record for this project's primary metric (arXiv 1811.02508;
see ``00-shared-research/papers/leroux2019-si-sdr.md``). For reference source
:math:`s` and estimate :math:`\hat s`,

.. math::

    \alpha = \frac{\hat s^\top s}{\lVert s\rVert^2 + \varepsilon}, \qquad
    \text{SI-SDR} = 10\log_{10}
        \frac{\lVert \alpha s\rVert^2 + \varepsilon}
             {\lVert \alpha s - \hat s\rVert^2 + \varepsilon}.

:math:`\alpha s` is the orthogonal projection of :math:`\hat s` onto the line
spanned by :math:`s`; the residual :math:`\alpha s - \hat s` is orthogonal to
it. The value is invariant to rescaling **either** argument (proof and
derivation in ``THEORY.md`` §3-4).

Both a NumPy variant (evaluation) and a torch variant (training loss and torch
evaluation) live here so there is exactly one implementation of the arithmetic.
The negative-SI-SDR training loss (:mod:`singnet.losses.sisdr`) and the
evaluator (:mod:`singnet.eval.evaluate`) both call in here.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

DEFAULT_EPS = 1e-8


def si_sdr(estimate: np.ndarray, reference: np.ndarray, eps: float = DEFAULT_EPS) -> float:
    """SI-SDR in dB for a single 1-D NumPy pair ``(estimate, reference)``.

    Returns a Python float. With ``eps=0`` the value is exactly scale-invariant
    in both arguments (used by the invariance unit test on non-degenerate
    signals); the default ``eps`` guards the silent-reference singularity.
    """
    estimate = np.asarray(estimate, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    ref_energy = np.float64(reference @ reference) + eps
    alpha = np.float64(estimate @ reference) / ref_energy
    target = alpha * reference
    noise = target - estimate
    num = np.float64(target @ target) + eps
    den = np.float64(noise @ noise) + eps
    # NumPy division so a zero residual (perfect estimate, eps=0) yields +inf
    # rather than a Python ZeroDivisionError.
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(10.0 * np.log10(num / den))


def si_sdr_torch(
    estimate: Tensor, reference: Tensor, eps: float = DEFAULT_EPS
) -> Tensor:
    """Batched SI-SDR in dB for torch tensors, shape ``(..., L)`` -> ``(...)``.

    Differentiable in ``estimate`` (and ``reference``); computed in float32.
    The last axis is the time axis reduced over.
    """
    estimate = estimate.to(torch.float32)
    reference = reference.to(torch.float32)
    ref_energy = (reference * reference).sum(dim=-1, keepdim=True) + eps
    alpha = (estimate * reference).sum(dim=-1, keepdim=True) / ref_energy
    target = alpha * reference
    noise = target - estimate
    num = (target * target).sum(dim=-1) + eps
    den = (noise * noise).sum(dim=-1) + eps
    return 10.0 * torch.log10(num / den)


def si_sdr_improvement(
    estimate: np.ndarray, reference: np.ndarray, mixture: np.ndarray, eps: float = DEFAULT_EPS
) -> float:
    """SI-SDRi = SI-SDR(estimate, ref) - SI-SDR(mixture, ref) — the karaoke delta."""
    return si_sdr(estimate, reference, eps) - si_sdr(mixture, reference, eps)
