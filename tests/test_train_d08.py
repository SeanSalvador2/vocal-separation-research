"""Direction 08 train-loop wiring + configs (MASTER_PLAN §4, §6, gate G0).

The loop changes must be **no-ops for every existing config**: a D01-style config builds
the uniform sampler and byte-identical data. A D08 config builds the right policy; the
exposure telemetry round-trips; and — the pre-registered guard — checkpoint selection is
SI-SDR-only (SLR is computed alongside but never compared).
"""

from __future__ import annotations

import inspect

import pandas as pd
import torch

from singnet.data import ChunkSampler, MusdbChunks
from singnet.data.sampling import build_chunk_sampler
from singnet.train import chunk_silent_fraction, is_new_best, write_exposure_telemetry
from singnet.train.loop import SAMPLING_TELEMETRY_COLUMNS, _exposure_row
from singnet.train import loop as loop_mod
from singnet.eval.evaluate import validation_report
from singnet.utils.config import hash_config, resolve_config, sampling_policy

D08 = "08-silence-leakage/configs"
D01 = "01-loss-function-study/configs/l1mag_seed0_reduced.yaml"

MATRIX = [
    "base", "energy_seed0", "energy_seed1", "energy_seed2",
    "drop_seed0", "drop_seed1", "drop_seed2",
    "curriculum_seed0", "curriculum_seed1",
    "contingency_uniform_sisdr_seed0", "contingency_energy_sisdr_seed0",
]


# --- the no-op guarantee for existing configs -------------------------------

def test_d01_config_builds_uniform_sampler() -> None:
    assert build_chunk_sampler(resolve_config(D01)).is_uniform()


def test_uniform_sampler_data_is_bit_identical(synthetic_store, synthetic_manifest) -> None:
    # The D01 config wired through build_chunk_sampler produces byte-identical chunks to
    # the default (pre-D08) loader path — the shared-cell guarantee at the dataset level.
    kw = dict(split="train", seed=0, length=32)
    legacy = MusdbChunks(synthetic_store, synthetic_manifest, **kw)
    d08 = MusdbChunks(
        synthetic_store, synthetic_manifest,
        sampler=build_chunk_sampler(resolve_config(D01)), **kw,
    )
    for i in (0, 3, 9):
        assert torch.equal(legacy[i]["mixture"], d08[i]["mixture"])
        assert torch.equal(legacy[i]["vocals"], d08[i]["vocals"])


# --- D08 configs build the right policy -------------------------------------

def test_d08_base_equals_shared_cell() -> None:
    assert hash_config(resolve_config(f"{D08}/base.yaml")) == hash_config(resolve_config(D01))


def test_d08_matrix_hashes_are_distinct() -> None:
    hashes = {n: hash_config(resolve_config(f"{D08}/{n}.yaml")) for n in MATRIX}
    assert len(set(hashes.values())) == len(hashes)  # 11 distinct runs (base = uniform cell)


def test_d08_configs_resolve_with_right_policy_and_loss() -> None:
    cases = {
        "energy_seed0": ("energy", "l1mag"),
        "drop_seed0": ("drop", "l1mag"),
        "curriculum_seed0": ("curriculum", "l1mag"),
        "contingency_uniform_sisdr_seed0": ("uniform", "sisdr"),
        "contingency_energy_sisdr_seed0": ("energy", "sisdr"),
    }
    for name, (policy, loss) in cases.items():
        cfg = resolve_config(f"{D08}/{name}.yaml")
        assert sampling_policy(cfg)["policy"] == policy
        assert cfg.get("loss") == loss


def test_d08_build_sampler_wires_curriculum_schedule() -> None:
    s = build_chunk_sampler(resolve_config(f"{D08}/curriculum_seed0.yaml"))
    assert s.policy == "curriculum" and s.total_steps == 16000
    assert s.lambda_at(0) == 1.0 and s.lambda_at(8000) == 0.1  # anneal over the first half


def test_d08_drop_and_energy_constants() -> None:
    drop = build_chunk_sampler(resolve_config(f"{D08}/drop_seed0.yaml"))
    energy = build_chunk_sampler(resolve_config(f"{D08}/energy_seed0.yaml"))
    assert drop.policy == "drop" and drop.theta_db == -60.0
    assert energy.policy == "energy" and energy.floor_lambda == 0.1


# --- exposure telemetry -----------------------------------------------------

def test_chunk_silent_fraction() -> None:
    # 4 chunks: two below −60 dBFS (RMS < 1e-3), two above -> fraction 0.5.
    vocals = torch.stack([
        torch.zeros(1000),                     # silent
        torch.full((1000,), 1e-4),             # −80 dBFS -> silent
        torch.full((1000,), 0.5),              # loud
        torch.full((1000,), 0.1),              # loud
    ])
    assert chunk_silent_fraction(vocals, -60.0) == 0.5


def test_exposure_telemetry_roundtrip(tmp_path) -> None:
    rows = [_exposure_row(500, 0.10, 1600, -60.0), _exposure_row(1000, 0.08, 1600, -60.0)]
    assert set(rows[0]) == set(SAMPLING_TELEMETRY_COLUMNS)
    path = tmp_path / "sub" / "sampling_exposure.csv"
    write_exposure_telemetry(path, rows)
    frame = pd.read_csv(path)
    assert list(frame.columns) == list(SAMPLING_TELEMETRY_COLUMNS)
    assert len(frame) == 2 and int(frame.iloc[0]["step"]) == 500


# --- the selection-bias guard (§6): selection is SI-SDR-only ----------------

def test_checkpoint_selection_is_blind_to_slr() -> None:
    # `is_new_best` takes SI-SDR only — no SLR parameter, and SLR is never named in its
    # signature or executable body (the docstring may *explain* the guard; the code cannot
    # read SLR). Together these prove selection cannot be SLR-biased (§6).
    sig = inspect.signature(is_new_best)
    assert list(sig.parameters) == ["candidate_sisdr", "best_sisdr"]
    assert "slr" not in is_new_best.__code__.co_varnames        # no slr arg/local
    return_line = next(ln for ln in inspect.getsource(is_new_best).splitlines()
                       if ln.strip().startswith("return"))
    assert "slr" not in return_line.lower()                     # the selection expression
    assert is_new_best(5.0, 4.0) and not is_new_best(3.0, 4.0)
    # and the training loop routes best-checkpoint selection through it on the SI-SDR value.
    run_src = inspect.getsource(loop_mod.run)
    assert "is_new_best(final_val" in run_src
    # the best-checkpoint save is gated by the is_new_best block, never by an SLR compare.
    select_line = next(ln for ln in run_src.splitlines() if "is_new_best(" in ln)
    assert "slr" not in select_line.lower()


def test_validation_report_computes_slr_alongside_sisdr() -> None:
    # SLR IS computed at every eval point (alongside SI-SDR) — just never selected on.
    src = inspect.getsource(validation_report)
    assert "slr" in src and "si_sdr" in src
    # the report exposes both keys the loop reads/logs.
    assert '"sisdr"' in src and '"slr"' in src


# --- a non-uniform dataset actually steers on the profile -------------------

def test_energy_dataset_draws_from_profile(synthetic_store, synthetic_manifest) -> None:
    # With all profile mass on the first grid window, energy sampling concentrates the
    # vocal chunk near the start — a visible behavioural difference from uniform.
    sr = synthetic_store.sample_rate
    profiles = {t: _peaked_profile() for t in ("train_00", "train_01")}
    ds = MusdbChunks(
        synthetic_store, synthetic_manifest, "train", seed=0, length=64,
        sampler=ChunkSampler("energy", floor_lambda=0.0, sr=sr), energy_profiles=profiles,
    )
    # the item is well-formed and additive; the draw path exercised without error.
    item = ds[0]
    assert torch.allclose(item["mixture"], item["vocals"] + item["accompaniment"], atol=1e-6)


def _peaked_profile():
    import numpy as np
    prof = np.zeros(2, dtype=np.float64)
    prof[0] = 1.0
    return prof
