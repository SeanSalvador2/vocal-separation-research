"""LR schedule and resumable checkpoint round-trip (G0, §7.1).

Resume determinism is tested via a **state-dict round-trip** (no optimization
loop): save -> load -> assert every tensor of state is bit-identical.
"""

from __future__ import annotations

import copy

import torch

from singnet.models import build_model
from singnet.train import (
    build_optimizer,
    build_scheduler,
    load_checkpoint,
    make_lr_lambda,
    save_checkpoint,
)
from singnet.train.loop import BASE_LR, MIN_LR, WARMUP_STEPS


def test_lr_lambda_warmup_and_cosine() -> None:
    total = 16000
    fn = make_lr_lambda(WARMUP_STEPS, total, BASE_LR, MIN_LR)
    assert fn(0) == 0.0  # warmup starts at 0
    assert abs(fn(WARMUP_STEPS) - 1.0) < 1e-9  # peak at end of warmup
    # monotonic decay after warmup
    mid, late = fn(WARMUP_STEPS + 1000), fn(total - 1000)
    assert 1.0 > mid > late
    # ends near min ratio
    assert abs(fn(total) - MIN_LR / BASE_LR) < 1e-6


def test_scheduler_applies_factor() -> None:
    model = build_model()
    opt = build_optimizer(model)
    sched = build_scheduler(opt, total_steps=16000)
    # step 0 lr is ~0 during warmup
    assert opt.param_groups[0]["lr"] < 1e-6
    # a real optimizer step precedes each scheduler step (order avoids the
    # "scheduler before optimizer" warning and the skipped-first-value pitfall)
    for _ in range(WARMUP_STEPS):
        opt.step()
        sched.step()
    assert abs(opt.param_groups[0]["lr"] - BASE_LR) < 1e-6


def test_checkpoint_state_dict_roundtrip(tmp_path) -> None:
    torch.manual_seed(0)
    model = build_model()
    opt = build_optimizer(model)
    sched = build_scheduler(opt, total_steps=1000)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    # take one fake optimizer step so optimizer state is non-empty
    mix = torch.rand(1, 2048, 256)
    loss = model(mix).sum()
    loss.backward()
    opt.step()
    sched.step()

    original_model = copy.deepcopy(model.state_dict())
    original_opt = copy.deepcopy(opt.state_dict())

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=model, optimizer=opt, scheduler=sched, scaler=scaler,
                    step=1234, best_metric=3.3, config={"arm": "l1mag"})

    # fresh objects, then restore
    model2 = build_model()
    opt2 = build_optimizer(model2)
    sched2 = build_scheduler(opt2, total_steps=1000)
    payload = load_checkpoint(path, model=model2, optimizer=opt2, scheduler=sched2)

    assert payload["step"] == 1234
    assert payload["best_metric"] == 3.3
    for key, tensor in original_model.items():
        assert torch.equal(tensor, model2.state_dict()[key])
    # optimizer exp_avg / step buffers restored identically
    st1 = original_opt["state"]
    st2 = opt2.state_dict()["state"]
    assert st1.keys() == st2.keys()
    for pid in st1:
        for buf_key, val in st1[pid].items():
            if torch.is_tensor(val):
                assert torch.equal(val, st2[pid][buf_key])


def test_prepare_batch_shapes(stft) -> None:
    from singnet.eval import EVAL_CHUNK_SAMPLES
    from singnet.train import prepare_batch

    mixture = 0.1 * torch.randn(2, EVAL_CHUNK_SAMPLES)
    vocals = 0.05 * torch.randn(2, EVAL_CHUNK_SAMPLES)
    out = prepare_batch(stft, mixture, vocals)
    assert out["mix_mag"].shape == (2, 2048, 256)
    assert out["tgt_mag"].shape == (2, 2048, 256)
    assert out["mix_stft"].shape[-2:] == (2049, 256)
    assert out["tgt_wave"].shape[0] == 2
