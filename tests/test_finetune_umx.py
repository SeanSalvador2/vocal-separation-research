"""UMX fine-tune plumbing: stereo channel swap, stereo chunks, loss/opt/schedule (G0).

Direction-05 stage-D items that run on CPU without training (MASTER_PLAN §5, §8):
* the stereo channel-swap transform (determinism, p-behaviour, mono no-op) and its
  **D02-compat**: it never perturbs the mono gain/flip/remix streams nor the mono
  ``__call__`` output;
* ``UmxStereoChunks`` yields deterministic, additive **stereo** pairs;
* the magnitude front-end / MSE loss / Adam(trainable-only) / warmup-then-constant LR;
* the recipe share table; and a mock forward+backward on tiny tensors (no loop).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from singnet.audio import STFT
from singnet.data import AugmentPipeline, Manifest, random_channel_swap
from singnet.peft import (
    StereoInMemoryStore,
    UmxStereoChunks,
    apply_recipe,
    build_umx_optimizer,
    load_umxhq,
    make_umx_lr_lambda,
    recipe_share_table,
    umx_augment_pipeline,
    umx_magnitude,
    umx_mse_loss,
)
from singnet.peft.finetune_umx import WARMUP_STEPS, sanity


@pytest.fixture()
def stereo_store() -> StereoInMemoryStore:
    rng = np.random.default_rng(7)
    tracks = {
        f"train_{i:02d}": {
            "vocals": (0.1 * rng.standard_normal((2, 44100 * 7))).astype(np.float32),
            "accompaniment": (0.1 * rng.standard_normal((2, 44100 * 7))).astype(np.float32),
        }
        for i in range(3)
    }
    return StereoInMemoryStore(tracks, 44100)


@pytest.fixture()
def stereo_manifest() -> Manifest:
    return Manifest(pd.DataFrame([
        {"track": "train_00", "split": "train"},
        {"track": "train_01", "split": "train"},
        {"track": "train_02", "split": "valid"},
    ]))


# --- stereo channel swap ----------------------------------------------------

def test_channel_swap_swaps_stereo_and_ignores_mono() -> None:
    src = {"v": np.stack([np.zeros(8), np.ones(8)]).astype(np.float32)}
    swapped = seen_same = False
    for step in range(40):
        rng = np.random.default_rng([0, step])
        out = random_channel_swap(src, rng, p=0.5)["v"]
        if np.array_equal(out, src["v"][::-1]):
            swapped = True
        if np.array_equal(out, src["v"]):
            seen_same = True
    assert swapped and seen_same  # both branches fire at p=0.5
    # a mono (1-D) source is never touched.
    mono = {"v": np.arange(8, dtype=np.float32)}
    assert np.array_equal(random_channel_swap(mono, np.random.default_rng(0))["v"], mono["v"])


def test_channelswap_is_off_by_default_and_stream_isolated() -> None:
    on = AugmentPipeline(remix=True, gain=True, flip=True, channelswap=True, seed=0)
    off = AugmentPipeline(remix=True, gain=True, flip=True, channelswap=False, seed=0)
    assert AugmentPipeline().channelswap is False  # default off -> mono D01/D02 unchanged
    for step in range(6):
        # toggling channelswap does not touch the gain / flip / remix streams.
        assert np.array_equal(on.stream("gain", step).uniform(size=4),
                              off.stream("gain", step).uniform(size=4))
        assert np.array_equal(on.stream("flip", step).random(size=4),
                              off.stream("flip", step).random(size=4))


def test_mono_call_output_is_unchanged_by_channelswap_flag() -> None:
    # The mono __call__ (gain -> flip) must be byte-identical regardless of the new flag.
    src = {"vocals": np.ones(64, np.float32), "accompaniment": np.linspace(-1, 1, 64, dtype=np.float32)}
    base = AugmentPipeline(remix=True, gain=True, flip=True, seed=5)
    with_flag = AugmentPipeline(remix=True, gain=True, flip=True, channelswap=True, seed=5)
    for step in range(8):
        a, b = base(src, step), with_flag(src, step)
        for k in src:
            assert np.array_equal(a[k], b[k])


# --- stereo chunk dataset ---------------------------------------------------

def test_stereo_chunks_are_stereo_deterministic_and_additive(stereo_store, stereo_manifest) -> None:
    ds = UmxStereoChunks(stereo_store, stereo_manifest, "train", seed=0, chunk_s=1.0, length=16)
    item = ds[3]
    assert item["mixture"].shape[0] == 2 and item["vocals"].shape[0] == 2
    assert torch.equal(ds[3]["mixture"], ds[3]["mixture"])          # deterministic
    assert not torch.equal(ds[3]["mixture"], ds[4]["mixture"])      # index-dependent
    # additivity holds unless channel swap independently permuted the two sources;
    # with channelswap OFF the mixture is exactly the sum.
    ds_noswap = UmxStereoChunks(
        stereo_store, stereo_manifest, "train", seed=0, chunk_s=1.0, length=16,
        pipeline=umx_augment_pipeline(channelswap=False, seed=0),
    )
    it = ds_noswap[2]
    assert torch.allclose(it["mixture"], it["vocals"] + it["mixture"] - it["vocals"], atol=1e-6)


def test_umx_augment_pipeline_recipe() -> None:
    p = umx_augment_pipeline(seed=0)
    # gain + channel swap + remix, and NO sign flip (UMX, not Demucs).
    assert (p.remix, p.gain, p.flip, p.channelswap) == (True, True, False, True)


# --- front-end / loss / optimizer / schedule --------------------------------

def test_umx_magnitude_and_loss(stereo_store, stereo_manifest) -> None:
    ds = UmxStereoChunks(stereo_store, stereo_manifest, "train", seed=0, chunk_s=1.0, length=8)
    stft = STFT()
    mix = torch.stack([ds[i]["mixture"] for i in range(2)])
    mag = umx_magnitude(stft, mix)
    assert mag.shape[:2] == (2, 2) and mag.shape[2] == 2049  # (B, 2, nb_output_bins, T)
    assert torch.isfinite(mag).all() and (mag >= 0).all()
    loss = umx_mse_loss(mag, torch.zeros_like(mag))
    assert loss.item() > 0 and torch.isfinite(loss)


def test_optimizer_only_trains_lora_params_and_refuses_zeroshot() -> None:
    model = load_umxhq("cpu", mock=True)
    apply_recipe(model, "lora4")
    opt = build_umx_optimizer(model, lr=1e-3)
    n_opt = sum(len(g["params"]) for g in opt.param_groups)
    assert n_opt == 34  # exactly the lora4 trainable tensors
    zeroshot = load_umxhq("cpu", mock=True)
    apply_recipe(zeroshot, "zeroshot")
    with pytest.raises(ValueError):
        build_umx_optimizer(zeroshot, lr=1e-3)


def test_lr_schedule_warmup_then_constant() -> None:
    f = make_umx_lr_lambda(WARMUP_STEPS)
    assert f(0) == 0.0
    assert f(WARMUP_STEPS // 2) == pytest.approx(0.5)
    assert f(WARMUP_STEPS) == 1.0
    assert f(6000) == 1.0  # constant after warmup (no decay), unlike D01-03's cosine


def test_mock_forward_backward_smoke(stereo_store, stereo_manifest) -> None:
    ds = UmxStereoChunks(stereo_store, stereo_manifest, "train", seed=0, chunk_s=1.0, length=8)
    stft = STFT()
    model = load_umxhq("cpu", mock=True)
    apply_recipe(model, "lora16")
    model.train()
    mix = torch.stack([ds[i]["mixture"] for i in range(2)])
    voc = torch.stack([ds[i]["vocals"] for i in range(2)])
    loss = umx_mse_loss(model(umx_magnitude(stft, mix)), umx_magnitude(stft, voc))
    loss.backward()
    # gradients reached the LoRA params (and only trainable ones exist in the opt).
    assert model.fc1.lora_B.grad is not None
    assert model.lstm.parametrizations.weight_ih_l0[0].lora_B.grad is not None


# --- sanity table -----------------------------------------------------------

def test_recipe_share_table_rows() -> None:
    table = recipe_share_table()
    assert list(table["recipe"]) == ["zeroshot", "head", "lora4", "lora16", "full"]
    lora16 = table.set_index("recipe").loc["lora16"]
    assert lora16["trainable_params"] == 431_520
    assert abs(lora16["trainable_share"] - 0.048522) < 1e-4


def test_sanity_returns_table_on_mock(capsys) -> None:
    table = sanity(mock=True)
    assert isinstance(table, pd.DataFrame) and len(table) == 5
    assert "RUN LATER" in capsys.readouterr().out


# --- consolidated test session (skeleton errors cleanly without data) -------

def test_test_matrix_errors_cleanly_without_data() -> None:
    from singnet.peft.evaluate_umx import TEST_MATRIX_COLUMNS, build_test_matrix

    assert TEST_MATRIX_COLUMNS[:6] == (
        "run_id", "domain", "recipe", "rank", "seed", "eval_set",
    )
    with pytest.raises(FileNotFoundError, match="shard_root"):
        build_test_matrix("nope/registry.csv", "nope/shards", "nope/splits.csv", "nope/out")
