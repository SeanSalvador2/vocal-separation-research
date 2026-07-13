"""SI-SDR metric: analytic cases, scale invariance in both arguments (G0, §6)."""

from __future__ import annotations

import numpy as np
import torch

from singnet.metrics import si_sdr, si_sdr_improvement, si_sdr_torch


def test_perfect_estimate_is_large() -> None:
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    # exact match with eps=0 -> +inf (0 residual)
    assert np.isinf(si_sdr(ref, ref, eps=0.0))
    # perfect up to a scale is also perfect (scale invariance)
    assert np.isinf(si_sdr(5.0 * ref, ref, eps=0.0))


def test_scale_invariance_estimate_argument() -> None:
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(500)
    est = ref + 0.3 * rng.standard_normal(500)
    base = si_sdr(est, ref, eps=0.0)
    for c in (0.5, 2.0, -3.0, 100.0):
        assert np.isclose(si_sdr(c * est, ref, eps=0.0), base, atol=1e-6)


def test_scale_invariance_reference_argument() -> None:
    rng = np.random.default_rng(1)
    ref = rng.standard_normal(500)
    est = ref + 0.3 * rng.standard_normal(500)
    base = si_sdr(est, ref, eps=0.0)
    for c in (0.5, 2.0, -3.0, 100.0):
        assert np.isclose(si_sdr(est, c * ref, eps=0.0), base, atol=1e-6)


def test_numpy_and_torch_agree() -> None:
    rng = np.random.default_rng(2)
    ref = rng.standard_normal(1000)
    est = 0.8 * ref + 0.2 * rng.standard_normal(1000)
    np_val = si_sdr(est, ref)
    torch_val = float(si_sdr_torch(torch.tensor(est), torch.tensor(ref)))
    assert np.isclose(np_val, torch_val, atol=1e-4)


def test_torch_batched_shape() -> None:
    est = torch.randn(4, 1000)
    ref = torch.randn(4, 1000)
    out = si_sdr_torch(est, ref)
    assert out.shape == (4,)


def test_improvement_is_delta() -> None:
    rng = np.random.default_rng(3)
    ref = rng.standard_normal(1000)
    mix = ref + rng.standard_normal(1000)
    est = ref + 0.1 * rng.standard_normal(1000)
    expected = si_sdr(est, ref) - si_sdr(mix, ref)
    assert np.isclose(si_sdr_improvement(est, ref, mix), expected)
