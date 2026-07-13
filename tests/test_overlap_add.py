"""Overlap-add crossfade: identity-mask reconstruction (G0, §7.2)."""

from __future__ import annotations

import pytest
import torch

from singnet.audio import STFT
from singnet.eval import (
    EVAL_CHUNK_SAMPLES,
    identity_chunk_separator,
    overlap_add,
    raised_cosine_window,
)


def _recon_db(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(10 * torch.log10(((x - y) ** 2).mean() / (x**2).mean()))


@pytest.mark.parametrize(
    "n",
    [EVAL_CHUNK_SAMPLES // 2, EVAL_CHUNK_SAMPLES, 2 * EVAL_CHUNK_SAMPLES, 3 * EVAL_CHUNK_SAMPLES + 12345],
)
def test_identity_mask_reconstructs_signal(stft: STFT, n: int) -> None:
    torch.manual_seed(0)
    x = 0.1 * torch.randn(n)
    y = overlap_add(x, identity_chunk_separator(stft))
    assert y.shape == x.shape
    assert _recon_db(x, y) < -60.0


def test_window_is_flat_in_the_interior() -> None:
    w = raised_cosine_window(1000, 0.25)
    assert torch.allclose(w[400:600], torch.ones(200))
    assert float(w[0]) < 1e-3 and float(w[-1]) < 1e-3


def test_overlap_add_requires_mono() -> None:
    stft = STFT()
    with pytest.raises(ValueError):
        overlap_add(torch.randn(2, 1000), identity_chunk_separator(stft))
