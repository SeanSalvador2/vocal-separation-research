"""SingNet-C1 shapes, exact parameter count, and gradient flow (G0, §5)."""

from __future__ import annotations

import torch

from singnet.audio import N_FRAMES, N_MASK_BINS
from singnet.models import SingNetC1, build_model

# The exact count derived by hand in THEORY.md §5. The code is authoritative;
# if the architecture ever changes, update THEORY.md §5 to match this number.
EXPECTED_PARAM_COUNT = 9_835_745


def test_exact_parameter_count() -> None:
    model = SingNetC1()
    assert model.num_parameters == EXPECTED_PARAM_COUNT


def test_forward_shapes_and_mask_range() -> None:
    model = build_model().eval()
    mix_mag = torch.rand(2, N_MASK_BINS, N_FRAMES)
    with torch.no_grad():
        mask = model(mix_mag)
    assert mask.shape == (2, N_MASK_BINS, N_FRAMES)
    assert float(mask.min()) >= 0.0 and float(mask.max()) <= 1.0


def test_accepts_channel_dim_input() -> None:
    model = build_model().eval()
    mix_mag = torch.rand(1, 1, N_MASK_BINS, N_FRAMES)
    mask = model(mix_mag)
    assert mask.shape == (1, N_MASK_BINS, N_FRAMES)


def test_gradient_flow_smoke() -> None:
    model = build_model()
    mix_mag = torch.rand(1, N_MASK_BINS, N_FRAMES)
    mask = model(mix_mag)
    mask.sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)
    # at least some parameters receive a non-zero gradient
    assert any(g.abs().sum() > 0 for g in grads)


def test_bottleneck_channels() -> None:
    """The 5th encoder block reaches the 512-channel 64x8 bottleneck (§5)."""
    model = build_model().eval()
    x = torch.rand(1, 1, N_MASK_BINS, N_FRAMES)
    from singnet.models.unet import _featurize

    feat = _featurize(x)
    e1 = model.enc1(feat)
    e2 = model.enc2(e1)
    e3 = model.enc3(e2)
    e4 = model.enc4(e3)
    e5 = model.enc5(e4)
    assert e5.shape[1:] == (512, 64, 8)
