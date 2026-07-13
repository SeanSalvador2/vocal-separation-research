"""Loss unit tests (MASTER_PLAN §6): analytic values, perfect-prediction bounds,
gradient flow through iSTFT, the silent-target guard, and fp16-range finiteness."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from singnet.audio import N_FRAMES, N_MASK_BINS, STFT, analyze_chunk
from singnet.losses import ARM_NAMES, MultiResolutionSTFTLoss, build
from singnet.losses.sisdr import SILENCE_RMS_THRESHOLD

F, T = 32, 16  # small magnitude grids for the analytic (magnitude-only) checks


def _mag_inputs(mask_val, mix_val, tgt_val):
    mask = torch.full((2, F, T), float(mask_val), requires_grad=True)
    mix_mag = torch.full((2, F, T), float(mix_val))
    tgt_mag = torch.full((2, F, T), float(tgt_val))
    return mask, mix_mag, tgt_mag


# --- analytic values --------------------------------------------------------

def test_l1mag_analytic_and_perfect() -> None:
    loss = build("l1mag")
    # est = 0.5 * 2 = 1 == tgt -> exactly 0
    mask, mix, tgt = _mag_inputs(0.5, 2.0, 1.0)
    val, _ = loss(mask, mix, tgt)
    assert torch.allclose(val, torch.zeros(()), atol=1e-7)
    # est = 1, tgt = 0 -> mean |1| = 1
    mask, mix, tgt = _mag_inputs(0.5, 2.0, 0.0)
    val, _ = loss(mask, mix, tgt)
    assert torch.allclose(val, torch.ones(()), atol=1e-6)


def test_msemag_analytic() -> None:
    loss = build("msemag")
    mask, mix, tgt = _mag_inputs(0.5, 2.0, 0.0)  # est=1, err^2=1
    val, _ = loss(mask, mix, tgt)
    assert torch.allclose(val, torch.ones(()), atol=1e-6)
    mask, mix, tgt = _mag_inputs(0.25, 4.0, 3.0)  # est=1, err=-2, ^2=4
    val, _ = loss(mask, mix, tgt)
    assert torch.allclose(val, torch.full((), 4.0), atol=1e-6)


def test_logl1mag_perfect_is_zero() -> None:
    loss = build("logl1mag")
    mask, mix, tgt = _mag_inputs(0.5, 2.0, 1.0)  # est == tgt == 1
    val, _ = loss(mask, mix, tgt)
    assert torch.allclose(val, torch.zeros(()), atol=1e-7)


def test_mrstft_zero_on_identical_waveforms() -> None:
    mr = MultiResolutionSTFTLoss()
    torch.manual_seed(0)
    wave = 0.1 * torch.randn(2, 60000)
    val = mr(wave, wave)
    assert torch.allclose(val, torch.zeros(()), atol=1e-6)


# --- waveform-path fixtures -------------------------------------------------

def _waveform_case(silent: bool = False, scale: float = 1.0):
    """Build (mask, mix_mag, tgt_mag, mix_stft, tgt_wave, mix_wave) via a real STFT."""
    stft = STFT()
    torch.manual_seed(0)
    from singnet.eval import EVAL_CHUNK_SAMPLES

    mix_wave = scale * 0.1 * torch.randn(2, EVAL_CHUNK_SAMPLES)
    voc_wave = torch.zeros_like(mix_wave) if silent else scale * 0.05 * torch.randn(2, EVAL_CHUNK_SAMPLES)
    mix_spec, mix_mag = analyze_chunk(stft, mix_wave)
    tgt_spec, tgt_mag = analyze_chunk(stft, voc_wave)
    tgt_wave = stft.inverse(tgt_spec)
    mix_wave_r = stft.inverse(mix_spec)
    mask = torch.rand(2, N_MASK_BINS, N_FRAMES, requires_grad=True)
    return mask, mix_mag, tgt_mag, mix_spec, tgt_wave, mix_wave_r


@pytest.mark.parametrize("arm", ["sisdr", "l1mrstft"])
def test_gradient_flows_to_mask_through_istft(arm: str) -> None:
    loss = build(arm)
    mask, mix_mag, tgt_mag, mix_stft, tgt_wave, mix_wave = _waveform_case()
    val, _ = loss(mask, mix_mag, tgt_mag, mix_stft, tgt_wave, mix_wave)
    val.backward()
    assert mask.grad is not None
    assert torch.isfinite(mask.grad).all()
    assert mask.grad.abs().sum() > 0  # non-trivial gradient


@pytest.mark.parametrize("arm", list(ARM_NAMES))
def test_all_losses_finite_at_extreme_magnitudes(arm: str) -> None:
    """fp16-range magnitudes (up to ~65504) must not overflow — losses run fp32."""
    loss = build(arm)
    mask, mix_mag, tgt_mag, mix_stft, tgt_wave, mix_wave = _waveform_case(scale=1.0)
    big = 6.0e4
    mix_mag = mix_mag * big
    tgt_mag = tgt_mag * big
    mix_stft = mix_stft * big
    tgt_wave = tgt_wave * big
    mix_wave = mix_wave * big
    val, _ = loss(mask, mix_mag, tgt_mag, mix_stft, tgt_wave, mix_wave)
    assert torch.isfinite(val).all()


def test_sisdr_matches_negative_metric() -> None:
    from singnet.metrics import si_sdr_torch

    loss = build("sisdr")
    mask, mix_mag, tgt_mag, mix_stft, tgt_wave, mix_wave = _waveform_case()
    val, aux = loss(mask, mix_mag, tgt_mag, mix_stft, tgt_wave, mix_wave)
    # reconstruct the estimate the same way the loss does and compare
    from singnet.audio import apply_mask

    est_wave = loss.stft.inverse(apply_mask(mask.detach(), mix_stft), length=tgt_wave.shape[-1])
    expected = -si_sdr_torch(est_wave, tgt_wave).mean()
    assert torch.allclose(val.detach(), expected, atol=1e-4)
    assert aux["skip_rate"] == 0.0


def test_sisdr_silent_guard_skips_and_reports_rate() -> None:
    loss = build("sisdr")
    # one silent target chunk, one voiced -> skip_rate == 0.5
    stft = STFT()
    from singnet.eval import EVAL_CHUNK_SAMPLES

    torch.manual_seed(1)
    mix_wave = 0.1 * torch.randn(2, EVAL_CHUNK_SAMPLES)
    voc = 0.05 * torch.randn(2, EVAL_CHUNK_SAMPLES)
    voc[0] = 0.0  # chunk 0 target is silent
    mix_spec, mix_mag = analyze_chunk(stft, mix_wave)
    tgt_spec, tgt_mag = analyze_chunk(stft, voc)
    tgt_wave = stft.inverse(tgt_spec)
    mask = torch.rand(2, N_MASK_BINS, N_FRAMES, requires_grad=True)
    val, aux = loss(mask, mix_mag, tgt_mag, mix_spec, tgt_wave, stft.inverse(mix_spec))
    assert aux["skip_rate"] == 0.5
    assert torch.isfinite(val)
    val.backward()  # still differentiable on the kept chunk
    assert mask.grad is not None and torch.isfinite(mask.grad).all()


def test_sisdr_all_silent_returns_finite_zero_grad() -> None:
    loss = build("sisdr")
    mask, mix_mag, tgt_mag, mix_stft, tgt_wave, mix_wave = _waveform_case(silent=True)
    val, aux = loss(mask, mix_mag, tgt_mag, mix_stft, tgt_wave, mix_wave)
    assert aux["skip_rate"] == 1.0
    assert torch.isfinite(val)
    val.backward()
    assert mask.grad is not None and torch.isfinite(mask.grad).all()


def test_silence_threshold_is_minus_60_dbfs() -> None:
    assert np.isclose(SILENCE_RMS_THRESHOLD, 10 ** (-60 / 20))


def test_uniform_contract_magnitude_losses_ignore_waveforms() -> None:
    """Magnitude losses accept the full signature but need no waveform args."""
    for arm in ("l1mag", "msemag", "logl1mag"):
        loss = build(arm)
        mask, mix, tgt = _mag_inputs(0.5, 2.0, 1.0)
        val, aux = loss(mask, mix, tgt)  # no waveform args supplied
        assert torch.isfinite(val) and aux == {}


def test_l1mrstft_lambda_configurable() -> None:
    """The exploratory λ configs (0.25, 1.0) change the loss weighting."""
    mask, mix_mag, tgt_mag, mix_stft, tgt_wave, mix_wave = _waveform_case()
    v_half, _ = build("l1mrstft", lam=0.5)(mask, mix_mag, tgt_mag, mix_stft, tgt_wave, mix_wave)
    v_one, _ = build("l1mrstft", lam=1.0)(mask, mix_mag, tgt_mag, mix_stft, tgt_wave, mix_wave)
    assert not torch.allclose(v_half, v_one)
