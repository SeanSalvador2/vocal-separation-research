"""Direction 10 wiring: configs, loss decoupling, loop branch, dry-run, test session (G0, §4-5).

The loop changes must be **no-ops for every existing config**: a musdb-only config has no
pseudo spec and builds byte-identical data. A D10 config resolves the right data_source/p_fma;
``mixed_trim`` wraps the base loss in the (D06) TrimmedLoss; ``build_pseudo_dataset`` fails
loud without labeled shards; the dry-run and the RUN-LATER test session both behave. The
config-hash accounting (musdb_only ≡ shared cell; the 6 runs distinct) is asserted here too.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from singnet.losses import TrimmedLoss
from singnet.train import build_training_loss
from singnet.train import loop as loop_mod
from singnet.utils.config import hash_config, pseudo_data_spec, resolve_config

import run_sweep  # noqa: E402

D10 = "10-demucs-distillation/configs"
D01 = "01-loss-function-study/configs/l1mag_seed0_reduced.yaml"
SHARED_CELL = "a97d5400e994"

MATRIX = [
    "base", "mixed_seed0", "mixed_seed1", "mixed_seed2",
    "distill_only_seed0", "mixed25_seed0", "mixed_trim_seed0",
    "contingency_musdb_only_sisdr_seed0", "contingency_mixed_sisdr_seed0",
]


# --- configs resolve with the right source / ratio / loss -------------------

def test_musdb_only_base_equals_shared_cell() -> None:
    assert hash_config(resolve_config(f"{D10}/base.yaml")) == hash_config(resolve_config(D01))
    assert hash_config(resolve_config(f"{D10}/base.yaml")) == SHARED_CELL
    assert pseudo_data_spec(resolve_config(f"{D10}/base.yaml")) is None  # musdb-only


def test_matrix_hashes_are_distinct() -> None:
    hashes = {n: hash_config(resolve_config(f"{D10}/{n}.yaml")) for n in MATRIX}
    assert len(set(hashes.values())) == len(hashes)  # 9 distinct (base = the shared cell)


def test_configs_resolve_with_right_source_and_loss() -> None:
    cases = {
        "mixed_seed0": (("mixed", 0.5), "l1mag"),
        "mixed25_seed0": (("mixed", 0.25), "l1mag"),
        "distill_only_seed0": (("distill", 1.0), "l1mag"),
        "mixed_trim_seed0": (("mixed", 0.5), "l1mag"),
        "contingency_musdb_only_sisdr_seed0": (None, "sisdr"),
        "contingency_mixed_sisdr_seed0": (("mixed", 0.5), "sisdr"),
    }
    for name, (expect_spec, loss) in cases.items():
        cfg = resolve_config(f"{D10}/{name}.yaml")
        spec = pseudo_data_spec(cfg)
        if expect_spec is None:
            assert spec is None, name
        else:
            assert (spec["data_source"], spec["p_fma"]) == expect_spec, name
        assert str(cfg.get("loss", cfg.get("arm"))) == loss, name


# --- mixed_trim reuses the D06 TrimmedLoss via config -----------------------

def test_mixed_trim_wires_trimmed_loss() -> None:
    cfg = resolve_config(f"{D10}/mixed_trim_seed0.yaml")
    loss = build_training_loss(cfg)
    assert isinstance(loss, TrimmedLoss) and loss.q == 0.3
    # the plain mixed arm is NOT trimmed (only mixed_trim adds the trim block).
    assert not isinstance(build_training_loss(resolve_config(f"{D10}/mixed_seed0.yaml")), TrimmedLoss)


# --- the loop branch is present and no-op for musdb-only --------------------

def test_loop_routes_data_source_through_pseudo_spec() -> None:
    src = inspect.getsource(loop_mod.run)
    # the mixed/distill branch reads pseudo_data_spec and builds the pool datasets.
    assert "pseudo_data_spec(config)" in src
    assert "MixedPools(" in src and "build_pseudo_dataset(" in src
    # and the musdb path (pseudo is None) leaves train_ds untouched (branch guarded by `if
    # pseudo is not None`), so every Direction 01–08 config builds byte-identical data.
    assert "if pseudo is not None:" in src


def test_build_pseudo_dataset_fails_loud_without_shards() -> None:
    from singnet.data import build_pseudo_dataset

    cfg = resolve_config(f"{D10}/mixed_seed0.yaml")  # pseudo_root: null
    with pytest.raises(FileNotFoundError):
        build_pseudo_dataset(cfg, seed=0, length=64)


# --- the dry-run prints per-config source/ratio/guard (no GPU) --------------

def test_dry_run_d10_smoke(capsys) -> None:
    paths = [Path(D10) / n for n in run_sweep.D10_REDUCED]  # names already carry .yaml
    run_sweep.dry_run_d10(paths)
    out = capsys.readouterr().out
    assert "MUSDB-path guard active" in out and "True" in out
    assert "refuses MUSDB roots" in out          # the guard status per pseudo config
    assert "data_source" in out and "p_fma" in out
    assert SHARED_CELL in out                     # the shared musdb_only cell line
    # every reduced config appears with its arm.
    for arm in ("mixed", "distill_only", "mixed25", "mixed_trim"):
        assert arm in out


def test_configs_for_direction_10() -> None:
    reduced = run_sweep.configs_for("10", "reduced", Path(D10),
                                    "10-demucs-distillation/results/registry.csv",
                                    exploratory=False, halve=False)
    assert [p.name for p in reduced] == run_sweep.D10_REDUCED
    assert len(reduced) == 6  # the 6 new runs; musdb_only is the reused shared cell
    conting = run_sweep.configs_for("10", "contingency", Path(D10),
                                    "10-demucs-distillation/results/registry.csv",
                                    exploratory=False, halve=False)
    assert len(conting) == 2


# --- the RUN-LATER test session errors cleanly without data -----------------

def test_teacher_session_fails_loud_without_data(tmp_path) -> None:
    from singnet.eval.teacher_session import build_teacher_session

    # no shards / registry / splits present -> a clean FileNotFoundError, never a heavy pass.
    with pytest.raises(FileNotFoundError):
        build_teacher_session(
            registry_path=tmp_path / "registry.csv",
            shard_root=tmp_path / "shards",
            splits_csv=tmp_path / "splits.csv",
            output_dir=tmp_path / "out",
            include_teacher=True, slr=True,
        )


def test_lazy_teacher_imports_no_demucs() -> None:
    # Importing the teacher wrapper pulls in no demucs/torch (the import is inside a method).
    from singnet.eval.teacher_session import LazyDemucsTeacher

    teacher = LazyDemucsTeacher("htdemucs")
    assert teacher.model_name == "htdemucs" and teacher._model is None
    src = Path(inspect.getfile(LazyDemucsTeacher)).read_text()
    top_level = [ln for ln in src.splitlines() if ln.startswith("import ") or ln.startswith("from ")]
    assert not any("demucs" in ln for ln in top_level)


def test_evaluate_cli_routes_direction_10(tmp_path) -> None:
    # `evaluate.py --direction 10 --test-session` routes to build_teacher_session and errors
    # cleanly (no data) — the CLI wiring is exercised without any heavy pass.
    from singnet.eval.evaluate import _cli

    with pytest.raises(FileNotFoundError):
        _cli([
            "--direction", "10", "--test-session", "--include-teacher", "--slr",
            "--shard-root", str(tmp_path / "shards"),
            "--splits-csv", str(tmp_path / "splits.csv"),
            "--output-dir", str(tmp_path / "out"),
        ])
