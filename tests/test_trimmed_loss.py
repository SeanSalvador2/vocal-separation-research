"""Direction 06 trimmed loss (ITLM) + the per-chunk loss-contract extension (§3.2, G0).

Analytic trimming on a constructed batch (known ranking -> known kept set -> known
scalar); q=0 ≡ base loss; gradients flow only through kept chunks; ranking stable at
fp16-range magnitudes (fp32 path); the wrapper is loss-agnostic (l1mag + sisdr); and
the per-chunk path is backward-compatible (reduce=True aux unchanged, reduce=False
adds a differentiable (B,) vector).
"""

from __future__ import annotations

import pytest
import torch

from singnet.audio import N_FRAMES, N_MASK_BINS, STFT, analyze_chunk
from singnet.losses import ARM_NAMES, TrimmedLoss, build


# --- fixtures ---------------------------------------------------------------

def _const_chunk_batch(per_chunk_targets, *, mask_val=0.0, mix_val=1.0, F=4, T=4, requires_grad=False):
    """A magnitude batch whose per-chunk L1 loss equals ``|mask*mix - tgt|`` per chunk.

    With ``mask_val=0`` and ``mix_val=1`` the estimate is 0, so the per-chunk L1 loss
    is exactly ``per_chunk_targets`` — a batch with an analytically known ranking.
    """
    b = len(per_chunk_targets)
    mask = torch.full((b, F, T), float(mask_val), requires_grad=requires_grad)
    mix = torch.full((b, F, T), float(mix_val))
    tgt = torch.zeros((b, F, T))
    for i, c in enumerate(per_chunk_targets):
        tgt[i] = float(c)
    return mask, mix, tgt


def _waveform_batch(b=4, seed=0):
    from singnet.eval import EVAL_CHUNK_SAMPLES

    stft = STFT()
    torch.manual_seed(seed)
    mix_wave = 0.1 * torch.randn(b, EVAL_CHUNK_SAMPLES)
    voc_wave = 0.05 * torch.randn(b, EVAL_CHUNK_SAMPLES)
    mix_spec, mix_mag = analyze_chunk(stft, mix_wave)
    tgt_spec, tgt_mag = analyze_chunk(stft, voc_wave)
    tgt_wave = stft.inverse(tgt_spec)
    mask = torch.rand(b, N_MASK_BINS, N_FRAMES, requires_grad=True)
    return stft, mask, mix_mag, tgt_mag, mix_spec, tgt_wave, stft.inverse(mix_spec)


# --- keep-count arithmetic --------------------------------------------------

def test_keep_count_ceil_rule() -> None:
    assert TrimmedLoss.keep_count(16, 0.30) == 12   # ⌈0.7·16⌉ (MASTER_PLAN §11: keep 11/16? -> 12)
    assert TrimmedLoss.keep_count(16, 0.10) == 15   # ⌈0.9·16⌉
    assert TrimmedLoss.keep_count(4, 0.5) == 2
    assert TrimmedLoss.keep_count(8, 0.0) == 8      # keep all
    assert TrimmedLoss.keep_count(3, 0.99) == 1     # clamped to >= 1


# --- analytic trimming ------------------------------------------------------

def test_analytic_trimming_known_kept_set_and_scalar() -> None:
    trimmed = TrimmedLoss(build("l1mag"), q=0.5)
    mask, mix, tgt = _const_chunk_batch([0.1, 0.4, 0.2, 0.3])
    loss, aux = trimmed(mask, mix, tgt)
    # keep the 2 smallest (0.1 @ idx0, 0.2 @ idx2); drop {1, 3}. Mean = 0.15.
    assert set(aux["kept_idx"]) == {0, 2}
    assert set(aux["dropped_idx"]) == {1, 3}
    assert aux["n_kept"] == 2 and aux["n_dropped"] == 2
    assert aux["kept_fraction"] == 0.5
    assert torch.allclose(loss, torch.tensor(0.15), atol=1e-6)


def test_q0_equals_base_loss() -> None:
    base = build("l1mag")
    trimmed = TrimmedLoss(base, q=0.0)
    mask, mix, tgt = _const_chunk_batch([0.1, 0.4, 0.2, 0.3])
    base_val, _ = base(mask, mix, tgt)
    trim_val, aux = trimmed(mask, mix, tgt)
    assert aux["n_kept"] == 4  # keeps every chunk
    assert torch.allclose(trim_val, base_val, atol=1e-6)


@pytest.mark.parametrize("q", [0.10, 0.30, 0.50])
def test_kept_are_the_lowest_loss_chunks(q: float) -> None:
    trimmed = TrimmedLoss(build("l1mag"), q=q)
    targets = [0.9, 0.1, 0.7, 0.2, 0.5, 0.3, 0.8, 0.4]
    mask, mix, tgt = _const_chunk_batch(targets)
    _, aux = trimmed(mask, mix, tgt)
    kept = set(aux["kept_idx"])
    keep = TrimmedLoss.keep_count(len(targets), q)
    expected = set(sorted(range(len(targets)), key=lambda i: targets[i])[:keep])
    assert kept == expected


# --- gradient masking -------------------------------------------------------

def test_gradients_flow_only_through_kept_chunks() -> None:
    trimmed = TrimmedLoss(build("l1mag"), q=0.5)
    mask, mix, tgt = _const_chunk_batch([0.1, 0.4, 0.2, 0.3], requires_grad=True)
    loss, aux = trimmed(mask, mix, tgt)
    loss.backward()
    assert mask.grad is not None and torch.isfinite(mask.grad).all()
    for i in aux["dropped_idx"]:
        assert torch.count_nonzero(mask.grad[i]) == 0      # dropped -> zero gradient
    for i in aux["kept_idx"]:
        assert mask.grad[i].abs().sum() > 0                # kept -> real gradient


# --- ranking stability at fp16-range magnitudes -----------------------------

def test_ranking_stable_at_extreme_magnitudes() -> None:
    trimmed = TrimmedLoss(build("l1mag"), q=0.5)
    base_targets = [0.1, 0.4, 0.2, 0.3]
    big = 6.0e4  # near the fp16 ceiling; the per-chunk path runs fp32
    mask, mix, tgt = _const_chunk_batch([t * big for t in base_targets], mix_val=big)
    loss, aux = trimmed(mask, mix, tgt)
    assert torch.isfinite(loss)
    assert set(aux["kept_idx"]) == {0, 2}                  # same ranking as the small-magnitude case


# --- loss-agnostic: wraps l1mag and sisdr -----------------------------------

def test_wraps_l1mag_end_to_end() -> None:
    trimmed = TrimmedLoss(build("l1mag"), q=0.30)
    stft, mask, mix_mag, tgt_mag, mix_stft, tgt_wave, mix_wave = _waveform_batch(b=8)
    loss, aux = trimmed(mask, mix_mag, tgt_mag)
    assert torch.isfinite(loss) and aux["n_kept"] == 6    # ⌈0.7·8⌉
    loss.backward()
    assert mask.grad is not None and torch.isfinite(mask.grad).all()


def test_wraps_sisdr_end_to_end() -> None:
    base = build("sisdr")
    trimmed = TrimmedLoss(base, q=0.30)
    assert trimmed.needs_waveform is True                 # inherited from the base
    stft, mask, mix_mag, tgt_mag, mix_stft, tgt_wave, mix_wave = _waveform_batch(b=8)
    loss, aux = trimmed(mask, mix_mag, tgt_mag, mix_stft, tgt_wave, mix_wave)
    assert torch.isfinite(loss) and aux["n_kept"] == 6
    assert "base_skip_rate" in aux                        # base telemetry carried through
    loss.backward()
    assert mask.grad is not None and torch.isfinite(mask.grad).all()


# --- accompaniment-energy telemetry (§5 mechanism) --------------------------

def test_energy_telemetry_split_by_kept_dropped() -> None:
    trimmed = TrimmedLoss(build("l1mag"), q=0.5)
    targets = [0.1, 0.4, 0.2, 0.3]
    mask, mix, tgt = _const_chunk_batch(targets)
    # energy correlated with loss: chunks with high loss carry high accompaniment energy.
    energy = torch.tensor(targets)
    _, aux = trimmed(mask, mix, tgt, chunk_energy=energy)
    assert aux["kept_energy_mean"] < aux["dropped_energy_mean"]   # the §5 signature
    assert set(aux["kept_idx"]) == {0, 2}


def test_invalid_q_rejected() -> None:
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError):
            TrimmedLoss(build("l1mag"), q=bad)


# --- the per-chunk loss contract is backward-compatible ---------------------

@pytest.mark.parametrize("arm", list(ARM_NAMES))
def test_reduce_flag_contract(arm: str) -> None:
    loss = build(arm)
    stft, mask, mix_mag, tgt_mag, mix_stft, tgt_wave, mix_wave = _waveform_batch(b=5)
    # reduce=True (default): historical behaviour, no per_chunk key in aux.
    scal, aux = loss(mask, mix_mag, tgt_mag, mix_stft, tgt_wave, mix_wave)
    assert "per_chunk" not in aux
    # reduce=False: adds a differentiable (B,) vector; its mean is the returned scalar.
    scal2, aux2 = loss(mask, mix_mag, tgt_mag, mix_stft, tgt_wave, mix_wave, reduce=False)
    pc = aux2["per_chunk"]
    assert pc.shape == (5,) and pc.dtype == torch.float32 and pc.requires_grad
    assert torch.allclose(scal2, pc.mean(), atol=1e-6)


def test_magnitude_reduce_true_is_unchanged() -> None:
    # The exact analytic values the Direction-01 suite pins must be identical.
    loss = build("l1mag")
    mask = torch.full((2, 32, 16), 0.5)
    mix = torch.full((2, 32, 16), 2.0)
    tgt = torch.full((2, 32, 16), 0.0)  # est = 1, |1| = 1
    val, aux = loss(mask, mix, tgt)
    assert torch.allclose(val, torch.ones(())) and aux == {}
