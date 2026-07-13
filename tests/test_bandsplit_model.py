"""BandSplitUNet: band partition, mel edges, pad/crop routing, shapes, grads (G0, §8).

Gate-G0 items for the band-split front-end (MASTER_PLAN §8):
partition exact/disjoint/exhaustive over the 2048 bins for both layouts; the mel
edge formula is self-consistent with an independent recompute and close to the
plan's estimate; the pad/crop routing round-trips; forward shapes + gradient flow
for both variants; exact pinned parameter counts.
"""

from __future__ import annotations

import math

import torch

from singnet.audio import N_FRAMES, N_MASK_BINS
from singnet.models import BandSplitUNet, SingNetC1, mel_edges, uniform_edges
from singnet.models.bandsplit_unet import (
    MATCHED_BASE_WIDTH,
    MATCHED_BOTTLENECK_WIDTH,
    MATCHED_VARIANT_PARAM_COUNT,
    hz_to_bin,
)

# The plan's estimate (MASTER_PLAN §3.1, §9); the code derives the edges from the
# HTK formula and the test asserts self-consistency + proximity to these.
MEL_ESTIMATE = [0, 142, 597, 2048]
UNIFORM_EXPECTED = [0, 683, 1365, 2048]


def _partition_is_valid(edges: list[int], n_bins: int = N_MASK_BINS) -> None:
    assert edges[0] == 0 and edges[-1] == n_bins            # spans [0, 2048]
    assert all(hi > lo for lo, hi in zip(edges, edges[1:]))  # strictly increasing => disjoint, non-empty
    widths = [hi - lo for lo, hi in zip(edges, edges[1:])]
    assert sum(widths) == n_bins                            # exhaustive, no overlap


def test_mel_partition_exact_disjoint_exhaustive() -> None:
    edges = mel_edges(3)
    _partition_is_valid(edges)
    assert edges == MEL_ESTIMATE  # the exact derived values (reported by the agent)


def test_uniform_partition_exact_disjoint_exhaustive() -> None:
    edges = uniform_edges(3)
    _partition_is_valid(edges)
    assert edges == UNIFORM_EXPECTED


def test_mel_edges_formula_self_consistency() -> None:
    """Recompute the HTK mel partition independently; assert code == formula, ±3 of estimate."""
    sr, n_fft = 44100, 4096

    def hz_to_mel(f: float) -> float:
        return 2595.0 * math.log10(1.0 + f / 700.0)

    def mel_to_hz(m: float) -> float:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_hi = hz_to_mel(sr / 2)
    interior_hz = [mel_to_hz(i / 3 * mel_hi) for i in (1, 2)]
    recomputed = [0, *(hz_to_bin(f, sr, n_fft) for f in interior_hz), N_MASK_BINS]

    assert mel_edges(3, sr=sr, n_fft=n_fft) == recomputed          # self-consistent with the formula
    for got, estimate in zip(recomputed[1:-1], MEL_ESTIMATE[1:-1]):
        assert abs(got - estimate) <= 3                            # ±3-bin proximity to the plan


def test_mel_and_uniform_interior_edges_differ() -> None:
    # The two layouts genuinely place capacity differently (the whole point of the control).
    assert mel_edges(3)[1:-1] != uniform_edges(3)[1:-1]
    # mel concentrates bins low: its interior edges sit below the uniform ones.
    assert mel_edges(3)[1] < uniform_edges(3)[1]


def test_padded_widths_are_multiples_of_32() -> None:
    for model in (BandSplitUNet.from_mel_bands(), BandSplitUNet.from_uniform_bands()):
        for w, p in zip(model.band_widths, model.padded_widths):
            assert p % 32 == 0 and p >= w and p - w < 32
        assert model.padded_total == sum(model.padded_widths)


def test_routing_roundtrip_mel() -> None:
    model = BandSplitUNet.from_mel_bands()
    x = torch.randn(2, 1, N_MASK_BINS, N_FRAMES)
    recovered = model.crop_back(torch.cat(model.split_and_pad(x), dim=-2))
    assert torch.equal(recovered, x)  # pad then gather-back is exact


def test_routing_roundtrip_uniform() -> None:
    model = BandSplitUNet.from_uniform_bands()
    x = torch.randn(1, 3, N_MASK_BINS, N_FRAMES)  # multi-channel, still round-trips
    recovered = model.crop_back(torch.cat(model.split_and_pad(x), dim=-2))
    assert torch.equal(recovered, x)


def test_forward_shapes_and_mask_range_both_variants() -> None:
    for model in (BandSplitUNet.from_mel_bands().eval(), BandSplitUNet.from_uniform_bands().eval()):
        mix_mag = torch.rand(2, N_MASK_BINS, N_FRAMES)
        with torch.no_grad():
            mask = model(mix_mag)
        assert mask.shape == (2, N_MASK_BINS, N_FRAMES)          # baseline-identical interface
        assert float(mask.min()) >= 0.0 and float(mask.max()) <= 1.0


def test_matches_baseline_output_interface() -> None:
    # Drop-in: same in/out shape as SingNetC1, so train loop / evaluator are unchanged.
    baseline = SingNetC1().eval()
    variant = BandSplitUNet.from_mel_bands().eval()
    mix_mag = torch.rand(1, N_MASK_BINS, N_FRAMES)
    with torch.no_grad():
        assert baseline(mix_mag).shape == variant(mix_mag).shape


def test_accepts_channel_dim_input() -> None:
    model = BandSplitUNet.from_uniform_bands().eval()
    mix_mag = torch.rand(1, 1, N_MASK_BINS, N_FRAMES)  # (B,1,F,T) also accepted (via _featurize)
    assert model(mix_mag).shape == (1, N_MASK_BINS, N_FRAMES)


def test_gradient_flow_both_variants() -> None:
    for ctor in (BandSplitUNet.from_mel_bands, BandSplitUNet.from_uniform_bands):
        model = ctor()
        mix_mag = torch.rand(1, N_MASK_BINS, N_FRAMES)
        model(mix_mag).sum().backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert all(g is not None for g in grads)
        assert all(torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)
        # every tower receives gradient (no dead band)
        for tower in model.towers:
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in tower.parameters())


def test_exact_param_counts_pinned() -> None:
    mel = BandSplitUNet.from_mel_bands()
    uni = BandSplitUNet.from_uniform_bands()
    # both variants identical by construction (edges change compute, not params)
    assert mel.num_parameters == uni.num_parameters == MATCHED_VARIANT_PARAM_COUNT == 9_841_896
    # within the pre-registered ±2 % of the 9,835,745 baseline
    baseline = SingNetC1().num_parameters
    assert abs(mel.num_parameters - baseline) / baseline <= 0.02
    # the matched width/bottleneck are the pinned search result
    assert (mel.base_width, mel.bottleneck_width) == (MATCHED_BASE_WIDTH, MATCHED_BOTTLENECK_WIDTH)
    assert (mel.base_width, mel.bottleneck_width) == (23, 382)


def test_pure_variant_is_below_baseline_and_out_of_tol() -> None:
    # c=23 without the bottleneck bump is 9,584,170 (-2.56 %), which motivates the bump.
    pure = BandSplitUNet.from_mel_bands(bottleneck_width=None).num_parameters
    assert pure == 9_584_170
    assert (SingNetC1().num_parameters - pure) / SingNetC1().num_parameters > 0.02


def test_bad_edges_raise() -> None:
    import pytest

    with pytest.raises(ValueError):
        BandSplitUNet([0, 100, 100, 2048], base_width=8)  # empty band (non-increasing)
    with pytest.raises(ValueError):
        BandSplitUNet([0, 100, 2000], base_width=8)  # last edge != 2048
    with pytest.raises(ValueError):
        BandSplitUNet([10, 100, 2048], base_width=8)  # first edge != 0
