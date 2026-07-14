"""Direction 06 train-loop wiring (MASTER_PLAN §5, gate G0).

The loop changes must be **no-ops for every existing config**: a D01-style config
builds an identical training setup (plain base loss, no corruption) as before the
Direction-06 wiring existed. A D06 config builds the TrimmedLoss + StemBleed, and the
trim telemetry (kept-vs-dropped accompaniment energy) round-trips to a CSV.
"""

from __future__ import annotations

import torch

import pandas as pd

from singnet.data import StemBleed
from singnet.data.corrupt import build_corruption
from singnet.losses import L1MagLoss, SiSdrLoss, TrimmedLoss, build
from singnet.train import build_training_loss, write_trim_telemetry
from singnet.train.loop import TRIM_TELEMETRY_COLUMNS, _trim_telemetry_row
from singnet.utils.config import resolve_config

D06 = "06-robust-training/configs"
D01 = "01-loss-function-study/configs/l1mag_seed0_reduced.yaml"


def _mag_batch(b=6):
    torch.manual_seed(0)
    mask = torch.rand(b, 8, 8)
    mix = torch.rand(b, 8, 8) + 0.5
    tgt = torch.rand(b, 8, 8)
    return mask, mix, tgt


# --- the no-op guarantee for existing configs -------------------------------

def test_d01_config_builds_plain_base_loss() -> None:
    cfg = resolve_config(D01)
    loss_fn = build_training_loss(cfg)
    # identical to the pre-D06 construction: a plain L1MagLoss, NOT wrapped.
    assert isinstance(loss_fn, L1MagLoss) and not isinstance(loss_fn, TrimmedLoss)
    # and its forward matches the bare registry-built loss byte-for-byte.
    mask, mix, tgt = _mag_batch()
    ref, _ = build("l1mag")(mask, mix, tgt)
    got, _ = loss_fn(mask, mix, tgt)
    assert torch.equal(ref, got)


def test_d01_config_builds_no_corruption() -> None:
    cfg = resolve_config(D01)
    assert build_corruption(cfg.get("corrupt"), "train") is None


def test_d06_bleed_config_builds_corruption_but_plain_loss() -> None:
    cfg = resolve_config(f"{D06}/bleed30_seed0.yaml")
    corruption = build_corruption(cfg.get("corrupt"), "train")
    assert isinstance(corruption, StemBleed) and corruption.epsilon == 0.30
    # a pure bleed arm (no trim block) still builds the plain base loss.
    assert isinstance(build_training_loss(cfg), L1MagLoss)


def test_d06_trim_config_builds_trimmed_loss() -> None:
    cfg = resolve_config(f"{D06}/trim30_seed0.yaml")
    loss_fn = build_training_loss(cfg)
    assert isinstance(loss_fn, TrimmedLoss) and loss_fn.q == 0.30
    assert isinstance(loss_fn.base, L1MagLoss)
    assert isinstance(build_corruption(cfg.get("corrupt"), "train"), StemBleed)


def test_d06_contingency_builds_trimmed_sisdr() -> None:
    cfg = resolve_config(f"{D06}/contingency_trim30_sisdr_seed0.yaml")
    loss_fn = build_training_loss(cfg)
    assert isinstance(loss_fn, TrimmedLoss) and isinstance(loss_fn.base, SiSdrLoss)
    assert loss_fn.needs_waveform is True


def test_trim_clean_control_builds_trim_without_corruption() -> None:
    cfg = resolve_config(f"{D06}/trim_clean_seed0.yaml")
    assert isinstance(build_training_loss(cfg), TrimmedLoss)
    assert build_corruption(cfg.get("corrupt"), "train") is None  # ε=0 control


# --- the trim telemetry round-trips -----------------------------------------

def test_trim_telemetry_csv_roundtrip(tmp_path) -> None:
    trimmed = TrimmedLoss(build("l1mag"), q=0.5)
    mask, mix, tgt = _mag_batch(b=8)
    energy = torch.arange(8, dtype=torch.float32)  # chunk energy per index
    _, aux = trimmed(mask, mix, tgt, chunk_energy=energy)
    row = _trim_telemetry_row(500, aux)
    assert set(row) == set(TRIM_TELEMETRY_COLUMNS)
    assert row["step"] == 500 and row["n_kept"] == 4.0

    path = tmp_path / "sub" / "trim_energy_stats.csv"
    write_trim_telemetry(path, [row, _trim_telemetry_row(1000, aux)])
    frame = pd.read_csv(path)
    assert list(frame.columns) == list(TRIM_TELEMETRY_COLUMNS)
    assert len(frame) == 2 and int(frame.iloc[0]["step"]) == 500


def test_trim_step_produces_energy_split(tmp_path) -> None:
    # the §5 telemetry: with energy correlated to loss, kept chunks have lower ⟨a⟩-energy.
    trimmed = TrimmedLoss(build("l1mag"), q=0.5)
    b = 8
    mask = torch.zeros(b, 4, 4, requires_grad=True)
    mix = torch.ones(b, 4, 4)
    tgt = torch.zeros(b, 4, 4)
    losses = [0.9, 0.1, 0.7, 0.2, 0.5, 0.3, 0.8, 0.4]
    for i, c in enumerate(losses):
        tgt[i] = c
    energy = torch.tensor(losses)  # energy tracks loss
    loss, aux = trimmed(mask, mix, tgt, chunk_energy=energy)
    assert aux["kept_energy_mean"] < aux["dropped_energy_mean"]
    loss.backward()
    for i in aux["dropped_idx"]:
        assert torch.count_nonzero(mask.grad[i]) == 0
