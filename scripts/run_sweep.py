#!/usr/bin/env python3
r"""Run a direction's sweep / confirmation / contingency runs (MASTER_PLAN §3, §8).

**RUN LATER — GPU (per-run runtime in each direction's §7/§8).** Iterates a
direction's configs and trains each via ``singnet.train.run``. Fully resumable: a
run whose registry row already shows ``steps_done >= steps`` is skipped, and an
interrupted run resumes from its last checkpoint (a Colab disconnect costs
minutes).

    # Direction 01 — the 15-run loss sweep (default direction), REDUCED budget
    python scripts/run_sweep.py --stage reduced
    python scripts/run_sweep.py --stage reduced --exploratory   # + the 2 lambda configs
    python scripts/run_sweep.py --stage full                    # 3 confirmation runs

    # Direction 02 — augmentation factorization + data scaling
    python scripts/run_sweep.py --direction 02 --dry-run        # CPU: switchboard + subset sizes
    python scripts/run_sweep.py --direction 02 --stage reduced  # the 9 new runs
    python scripts/run_sweep.py --direction 02 --stage contingency  # 2 sisdr sensitivity runs

    # Direction 03 — param-matched mini band-split
    python scripts/run_sweep.py --direction 03 --dry-run        # CPU: band edges, c, param count
    python scripts/run_sweep.py --direction 03 --stage reduced  # the 6 split runs (mel x3, uniform x3)
    python scripts/run_sweep.py --direction 03 --stage full     # generate + run the confirmations

    # Direction 05 — LoRA / PEFT fine-tuning of umxhq
    python scripts/run_sweep.py --direction 05 --dry-run        # CPU: recipe/rank/LR/trainable-share (mock)
    python scripts/run_sweep.py --direction 05 --stage probes   # generate + run the 12 LR probes (RUN LATER)
    python scripts/run_sweep.py --direction 05 --stage main     # the 12 fine-tune runs (RUN LATER)

    # Direction 06 — robust training under stem bleed
    python scripts/run_sweep.py --direction 06 --dry-run        # CPU: ε/q/eval-guard per config
    python scripts/run_sweep.py --direction 06 --stage reduced  # the 10 new bleed/trim runs (RUN LATER)
    python scripts/run_sweep.py --direction 06 --stage contingency  # 2 sisdr contingency runs (RUN LATER)

    # Direction 08 — silence-leakage metric + chunk-sampling policies
    python scripts/run_sweep.py --direction 08 --dry-run        # CPU: policy/θ/λ per config
    python scripts/run_sweep.py --direction 08 --stage reduced  # the 8 new sampling-policy runs (RUN LATER)
    python scripts/run_sweep.py --direction 08 --stage contingency  # 2 sisdr contingency runs (RUN LATER)

    # Direction 10 — Demucs-as-teacher pseudo-label distillation
    python scripts/run_sweep.py --direction 10 --dry-run        # CPU: data_source/p_fma/n_pseudo/guards
    python scripts/run_sweep.py --direction 10 --stage reduced  # the 6 new mixed/distill runs (RUN LATER)
    python scripts/run_sweep.py --direction 10 --stage contingency  # 2 sisdr contingency runs (RUN LATER)

``--dry-run`` (CPU, no GPU, no data) instantiates each config and prints the
per-config summary that matters for that direction — the augmentation switchboard
+ subset size (Directions 01/02), the band edges, chosen width ``c`` and exact
parameter count (Direction 03), or the recipe / rank / LR / trainable-share on the
mock host (Direction 05) — so a mismatch is caught before any GPU spend.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from singnet.data import AugmentPipeline, load_track_allowlist  # noqa: E402
from singnet.train import get_run, make_run_id, read_registry, run  # noqa: E402
from singnet.utils.config import (  # noqa: E402
    augment_switches,
    hash_config,
    resolve_config,
    track_allowlist_path,
)

DIRECTIONS: dict[str, dict[str, str]] = {
    "01": {
        "config_dir": "01-loss-function-study/configs",
        "registry": "01-loss-function-study/results/registry.csv",
    },
    "02": {
        "config_dir": "02-augmentation-data-scaling/configs",
        "registry": "02-augmentation-data-scaling/results/registry.csv",
    },
    "03": {
        "config_dir": "03-mini-band-split/configs",
        "registry": "03-mini-band-split/results/registry.csv",
    },
    "05": {
        "config_dir": "05-lora-source-separation/configs",
        "registry": "05-lora-source-separation/results/registry.csv",
    },
    "06": {
        "config_dir": "06-robust-training/configs",
        "registry": "06-robust-training/results/registry.csv",
    },
    "08": {
        "config_dir": "08-silence-leakage/configs",
        "registry": "08-silence-leakage/results/registry.csv",
    },
    "10": {
        "config_dir": "10-demucs-distillation/configs",
        "registry": "10-demucs-distillation/results/registry.csv",
    },
}

ARMS = ["l1mag", "msemag", "logl1mag", "sisdr", "l1mrstft"]  # Direction 01
SEEDS = [0, 1, 2]
REDUCED_STEPS, FULL_STEPS = 16000, 40000

# Direction 02 launch lists (MASTER_PLAN §3.1, §3.2, §3.4). The FULL-86 / n86 cell
# is *not* here: it is bit-identical to Direction 01's l1mag sweep cell and is
# reused, not retrained (§3.3, asserted by the config-hash-equality test).
D02_REDUCED = [
    "loo_no_remix_seed0.yaml", "loo_no_gain_seed0.yaml", "loo_no_flip_seed0.yaml",
    "loo_none_seed0.yaml",
    "scale_n21_seed0.yaml", "scale_n21_seed1.yaml", "scale_n21_seed2.yaml",
    "scale_n43_seed0.yaml", "scale_n64_seed0.yaml",
]
D02_CONTINGENCY = ["contingency_full_sisdr_seed0.yaml", "contingency_none_sisdr_seed0.yaml"]

# Direction 03 launch lists (MASTER_PLAN §3.2). The `baseline` arm is NOT here:
# it is the shared Direction-01 l1mag cell (config-hash-equal, reused not
# retrained). Confirmations (confirm_{best,baseline}_full.yaml) are generated at
# analysis freeze — see generate_full_configs_d03 — and are NOT pre-committed.
D03_REDUCED = [
    "split_mel_seed0.yaml", "split_mel_seed1.yaml", "split_mel_seed2.yaml",
    "split_uniform_seed0.yaml", "split_uniform_seed1.yaml", "split_uniform_seed2.yaml",
]
D03_CONTINGENCY = ["contingency_baseline_sisdr_seed0.yaml", "contingency_mel_sisdr_seed0.yaml"]

# Direction 05 launch list (MASTER_PLAN §3.3 run matrix): 8 T1 + 4 T2 = 12 runs.
# `zeroshot` is not a run (it is the untuned host, scored in the test session), and
# the LoRA B=0 start reproduces it exactly (§2). The probe configs are GENERATED at
# RUN LATER by generate_probe_configs_d05 (not pre-committed — they depend on the §3.4
# grids and are expanded from configs/t1_probe_template.yaml).
D05_MAIN = [
    "t1_head_seed0.yaml", "t1_lora4_seed0.yaml",
    "t1_lora16_seed0.yaml", "t1_lora16_seed1.yaml", "t1_lora16_seed2.yaml",
    "t1_full_seed0.yaml", "t1_full_seed1.yaml", "t1_full_seed2.yaml",
    "t2_head_seed0.yaml", "t2_lora4_seed0.yaml", "t2_lora16_seed0.yaml", "t2_full_seed0.yaml",
]
# Per-recipe LR probe grids (§3.4): PEFT wants larger LRs than full-FT (THEORY §5).
D05_PROBE_GRIDS = {
    "head": ("3.0e-4", "1.0e-3", "3.0e-3"),
    "lora4": ("3.0e-4", "1.0e-3", "3.0e-3"),
    "lora16": ("3.0e-4", "1.0e-3", "3.0e-3"),
    "full": ("3.0e-5", "1.0e-4", "3.0e-4"),
}
D05_PROBE_RANK = {"head": "null", "lora4": "4", "lora16": "16", "full": "null"}

# Direction 06 launch lists (MASTER_PLAN §3.3 run matrix): 10 new runs. The clean
# (ε=0) cell is NOT here — it is the shared Direction-01 l1mag cell across all three
# clean seeds (config-hash-equal to base.yaml; reused, not retrained, §3.3). The two
# contingency runs are budget-gated (run only if Direction 01 flips the default loss).
D06_REDUCED = [
    "bleed05_seed0.yaml", "bleed15_seed0.yaml",
    "bleed30_seed0.yaml", "bleed30_seed1.yaml", "bleed30_seed2.yaml",
    "trim30_seed0.yaml", "trim30_seed1.yaml", "trim30_seed2.yaml",
    "trim30_q10_seed0.yaml", "trim_clean_seed0.yaml",
]
D06_CONTINGENCY = ["contingency_bleed30_sisdr_seed0.yaml", "contingency_trim30_sisdr_seed0.yaml"]

# Direction 08 launch lists (MASTER_PLAN §4.2 run matrix): 8 new runs. The `uniform` cell
# is NOT here — it is the shared Direction-01 l1mag cell (config-hash-equal to base.yaml;
# reused, not retrained, §4.2). The two contingency runs are budget-gated (run only if
# Direction 01 flips the default loss).
D08_REDUCED = [
    "energy_seed0.yaml", "energy_seed1.yaml", "energy_seed2.yaml",
    "drop_seed0.yaml", "drop_seed1.yaml", "drop_seed2.yaml",
    "curriculum_seed0.yaml", "curriculum_seed1.yaml",
]
D08_CONTINGENCY = ["contingency_uniform_sisdr_seed0.yaml", "contingency_energy_sisdr_seed0.yaml"]

# Direction 10 launch lists (MASTER_PLAN §4.1 run matrix): 6 new runs. The `musdb_only` cell
# is NOT here — it is the shared Direction-01 l1mag cell (config-hash-equal to base.yaml;
# reused, not retrained, §4.1: "0 new runs" — the sixth reuse of a97d5400e994). The two
# contingency runs are budget-gated (run only if Direction 01 flips the default loss).
D10_REDUCED = [
    "mixed_seed0.yaml", "mixed_seed1.yaml", "mixed_seed2.yaml",
    "distill_only_seed0.yaml", "mixed25_seed0.yaml", "mixed_trim_seed0.yaml",
]
D10_CONTINGENCY = ["contingency_musdb_only_sisdr_seed0.yaml", "contingency_mixed_sisdr_seed0.yaml"]


def _already_done(config_path: Path, registry: str) -> bool:
    """True if the registry shows this config completed to its step budget."""
    cfg = resolve_config(config_path)
    run_id = make_run_id(cfg["arm"], cfg["seed"], cfg.get("budget_name", "custom"), hash_config(cfg))
    row = get_run(registry, run_id)
    return bool(row) and int(row.get("steps_done", 0)) >= int(cfg["steps"])


def _run_config(config_path: Path, registry: str) -> None:
    if _already_done(config_path, registry):
        print(f"skip (already complete): {config_path.name}")
        return
    print(f"running {config_path.name} (RUN LATER: GPU)")
    result = run(config_path, registry_path=registry)
    print(f"  -> {result}")


def _run_config_d05(config_path: Path, registry: str) -> None:
    """Direction 05: fine-tune one recipe/domain via the dedicated UMX loop (RUN LATER)."""
    from singnet.peft.finetune_umx import finetune  # lazy: needs torch + (RUN LATER) data

    cfg = resolve_config(config_path)
    domain, recipe = str(cfg.get("domain", "standard")), str(cfg["recipe"])
    run_id = make_run_id(f"{domain}_{recipe}", int(cfg.get("seed", 0)),
                         cfg.get("budget_name", "ft6k"), hash_config(cfg))
    row = get_run(registry, run_id)
    if row and int(row.get("steps_done", 0)) >= int(cfg.get("steps", 6000)):
        print(f"skip (already complete): {config_path.name}")
        return
    print(f"running {config_path.name} (RUN LATER: GPU + umxhq weights)")
    result = finetune(config_path, registry_path=registry)
    print(f"  -> {result}")


def d01_reduced(config_dir: Path, exploratory: bool) -> list[Path]:
    configs = [config_dir / f"{arm}_seed{seed}_reduced.yaml" for arm in ARMS for seed in SEEDS]
    if exploratory:
        configs += [config_dir / "l1mrstft_lam025_seed0_reduced.yaml",
                    config_dir / "l1mrstft_lam100_seed0_reduced.yaml"]
    return configs


def generate_full_configs(config_dir: Path, registry: str, steps: int) -> list[Path]:
    """Direction 01: pick top-2 arms by mean val SI-SDR (+ l1mag), write FULL configs."""
    frame = read_registry(registry)
    if frame.empty:
        raise SystemExit("registry empty — run the reduced sweep first (RUN LATER)")
    reduced_rows = frame[frame["budget"] == REDUCED_STEPS]
    means = reduced_rows.groupby("arm")["best_val_sisdr"].mean().sort_values(ascending=False)
    top2 = list(means.head(2).index)
    chosen = list(dict.fromkeys([*top2, "l1mag"]))[:3]
    print(f"confirmation arms (top-2 by mean val SI-SDR + l1mag): {chosen}")

    paths: list[Path] = []
    for arm in chosen:
        body = (
            f"# CONFIRMATION run (MASTER_PLAN §3.2): {arm}, seed 0, FULL budget.\n"
            f"# Generated by run_sweep.py from the frozen reduced-sweep ranking.\n"
            f"base: base.yaml\narm: {arm}\nseed: 0\nbudget_name: full\nsteps: {steps}\n"
        )
        path = config_dir / f"{arm}_seed0_full.yaml"
        path.write_text(body, encoding="utf-8")
        paths.append(path)
    return paths


def generate_full_configs_d03(config_dir: Path, registry: str, steps: int) -> list[Path]:
    """Direction 03: pick the best split variant by mean val SI-SDR, write the two
    FULL confirmation configs (best variant + baseline), generated at analysis freeze.

    MASTER_PLAN §3.2/§3.3: the confirmation pair guards budget-dependence. The
    baseline FULL cell is the Direction-01 l1mag FULL seed-0 run (config-hash-equal
    when generated from base.yaml); it is written here for a self-contained run book
    but is a *reuse* target, not a new architecture.
    """
    frame = read_registry(registry)
    if frame.empty:
        raise SystemExit("registry empty — run the reduced split sweep first (RUN LATER)")
    reduced = frame[(frame["budget"] == REDUCED_STEPS) & (frame["arm"].isin(["split_mel", "split_uniform"]))]
    if reduced.empty:
        raise SystemExit("no split_* reduced rows in the registry — run --stage reduced first")
    means = reduced.groupby("arm")["best_val_sisdr"].mean().sort_values(ascending=False)
    best = str(means.index[0])
    bands = "mel" if best == "split_mel" else "uniform"
    print(f"confirmation: best variant by mean val SI-SDR = {best} ({means.to_dict()})")

    paths: list[Path] = []
    variant_body = (
        f"# CONFIRMATION run (MASTER_PLAN §3.2): {best}, seed 0, FULL budget.\n"
        f"# Generated by run_sweep.py from the frozen reduced-sweep ranking.\n"
        f"base: base.yaml\narm: {best}\nloss: l1mag\nseed: 0\nbudget_name: full\nsteps: {steps}\n"
        f"model:\n  arch: bandsplit\n  bands: {bands}\n  base_width: 23\n"
        f"  bottleneck_width: 382\n  n_bands: 3\n"
    )
    baseline_body = (
        "# CONFIRMATION run (MASTER_PLAN §3.2): baseline, seed 0, FULL budget.\n"
        "# NOTE: identity-equal to Direction 01's l1mag seed-0 FULL cell — reuse it if\n"
        "# already trained (config-hash check) rather than retraining.\n"
        f"base: base.yaml\narm: l1mag\nseed: 0\nbudget_name: full\nsteps: {steps}\n"
    )
    for name, body in ((f"confirm_{best}_full.yaml", variant_body),
                       ("confirm_baseline_full.yaml", baseline_body)):
        path = config_dir / name
        path.write_text(body, encoding="utf-8")
        paths.append(path)
    return paths


def generate_probe_configs_d05(config_dir: Path) -> list[Path]:
    """Direction 05: expand the probe template into 4 recipes x 3 LRs x 500 steps (§3.4).

    RUN LATER (writes ``probes_<recipe>_lr<lr>.yaml`` in the config dir). Each probe
    inherits ``base.yaml`` on T1, sets the recipe/rank and one grid LR, and runs 500
    steps on T1 val; the chosen LRs are frozen into the main configs at G2. Generated
    (not pre-committed) so the §3.4 grids are the single source of truth.
    """
    paths: list[Path] = []
    for recipe, grid in D05_PROBE_GRIDS.items():
        rank = D05_PROBE_RANK[recipe]
        for lr in grid:
            body = (
                f"# GENERATED LR probe (MASTER_PLAN §3.4): {recipe}, lr={lr}, 500 steps on T1 val.\n"
                f"# Expanded by run_sweep.py from configs/t1_probe_template.yaml — do not hand-edit.\n"
                f"base: base.yaml\ndomain: t1_aac64\ndomain_suffix: t1_aac64\n"
                f"budget_name: probe\nsteps: 500\n"
                f"recipe: {recipe}\narm: probe_{recipe}\nrank: {rank}\nseed: 0\nlr: {lr}\n"
            )
            name = f"probes_{recipe}_lr{lr.replace('.', 'p').replace('-', 'm')}.yaml"
            path = config_dir / name
            path.write_text(body, encoding="utf-8")
            paths.append(path)
    return paths


def configs_for(
    direction: str, stage: str, config_dir: Path, registry: str, *, exploratory: bool, halve: bool
) -> list[Path]:
    if direction == "05":
        if stage == "probes":
            return generate_probe_configs_d05(config_dir)
        if stage == "main":
            return [config_dir / name for name in D05_MAIN]
        raise SystemExit("direction 05 stages are 'probes' or 'main' (see --help)")
    if direction == "01":
        if stage == "reduced":
            return d01_reduced(config_dir, exploratory)
        if stage == "full":
            return generate_full_configs(config_dir, registry, FULL_STEPS // 2 if halve else FULL_STEPS)
        raise SystemExit("direction 01 has no 'contingency' stage (see --help)")
    if direction == "03":
        if stage == "reduced":
            return [config_dir / name for name in D03_REDUCED]
        if stage == "full":
            return generate_full_configs_d03(config_dir, registry, FULL_STEPS // 2 if halve else FULL_STEPS)
        if stage == "contingency":
            return [config_dir / name for name in D03_CONTINGENCY]
        raise SystemExit("unknown stage for direction 03 (reduced|full|contingency)")
    if direction == "06":
        if stage == "reduced":
            return [config_dir / name for name in D06_REDUCED]
        if stage == "contingency":
            return [config_dir / name for name in D06_CONTINGENCY]
        raise SystemExit(
            "direction 06 'full'/clean cell is the shared Direction-01 l1mag cell (reused, "
            "not retrained, §3.3) — use --stage reduced or contingency."
        )
    if direction == "08":
        if stage == "reduced":
            return [config_dir / name for name in D08_REDUCED]
        if stage == "contingency":
            return [config_dir / name for name in D08_CONTINGENCY]
        raise SystemExit(
            "direction 08 'full'/uniform cell is the shared Direction-01 l1mag cell (reused, "
            "not retrained, §4.2) — use --stage reduced or contingency."
        )
    if direction == "10":
        if stage == "reduced":
            return [config_dir / name for name in D10_REDUCED]
        if stage == "contingency":
            return [config_dir / name for name in D10_CONTINGENCY]
        raise SystemExit(
            "direction 10 'full'/musdb_only cell is the shared Direction-01 l1mag cell (reused, "
            "not retrained, §4.1) — use --stage reduced or contingency."
        )
    # direction 02
    if stage == "reduced":
        return [config_dir / name for name in D02_REDUCED]
    if stage == "contingency":
        return [config_dir / name for name in D02_CONTINGENCY]
    raise SystemExit(
        "direction 02 'full'/n86 is the shared Direction-01 l1mag cell (reused, not "
        "retrained, §3.3) — use --stage reduced or contingency."
    )


def _subset_size(cfg: dict) -> str:
    """Human label for a config's training-set size (full split or a subset CSV)."""
    path = track_allowlist_path(cfg)
    if path is None:
        return "full (86)"
    tracks = None
    if Path(path).exists():
        try:
            tracks = load_track_allowlist(path)
        except Exception:  # noqa: BLE001 — dry-run must never crash on a bad/absent CSV
            tracks = None
    if tracks is not None:
        return f"{len(tracks)} ({Path(path).name})"
    return f"pending G0b ({Path(path).name})"


def dry_run_bandsplit(config_paths: list[Path]) -> None:
    """Direction 03 dry-run: band edges, chosen width c, and exact param count (no GPU).

    Instantiates each config's model on CPU and prints the band partition (bins +
    Hz), the base width / bottleneck, and the parameter count with its Δ vs the
    9,835,745 baseline — so the ±2 % match and the partition are verified before
    any GPU spend (MASTER_PLAN §7 run book step 1).
    """
    from singnet.models import BASELINE_PARAM_COUNT, build_model_from_config
    from singnet.models.bandsplit_unet import bin_to_hz

    print(f"DRY RUN — direction 03: {len(config_paths)} configs (no training)\n")
    header = (f"{'config':<26} {'arm':<14} {'loss':<6} {'edges(bins)':<20} "
              f"{'c':<4} {'b5':<5} {'params':<11} {'Δ%':<8} hash")
    print(header)
    print("-" * len(header))
    for path in config_paths:
        if not path.exists():
            print(f"{path.name:<26} MISSING")
            continue
        cfg = resolve_config(path)
        model = build_model_from_config(cfg)
        loss = str(cfg.get("loss", cfg.get("arm", "?")))
        mcfg = cfg.get("model") or {}
        if hasattr(model, "edges_bins"):
            edges = str(model.edges_bins)
            c, b5 = model.base_width, model.bottleneck_width
        else:
            bc = int(cfg.get("base_channels", 32))
            edges, c, b5 = "— (full spectrum)", bc, 16 * bc
        params = model.num_parameters
        delta = 100.0 * (params - BASELINE_PARAM_COUNT) / BASELINE_PARAM_COUNT
        print(f"{path.name:<26} {str(cfg.get('arm','?')):<14} {loss:<6} {edges:<20} "
              f"{c:<4} {b5:<5} {params:<11,} {delta:<+8.3f} {hash_config(cfg)}")
    # show the mel/uniform interior edges in Hz for the layout figure
    from singnet.models import mel_edges, uniform_edges
    mel_hz = [round(bin_to_hz(b)) for b in mel_edges()[1:-1]]
    uni_hz = [round(bin_to_hz(b)) for b in uniform_edges()[1:-1]]
    print(f"\nmel interior edges:     bins {mel_edges()[1:-1]}  ≈ {mel_hz} Hz")
    print(f"uniform interior edges: bins {uniform_edges()[1:-1]}  ≈ {uni_hz} Hz")
    print("baseline arm = shared D01 l1mag cell (0 new runs). Verify the ±2 % match above.")


def dry_run_d05(config_paths: list[Path]) -> None:
    """Direction 05 dry-run: recipe / rank / LR / trainable-share on the mock host (no GPU).

    Builds the shape-faithful mock umxhq, applies each config's recipe, and prints the
    trained parameter count + host-share — the exact numbers that gate G0 pins ({head
    ~23.7 %, lora4 ~1.27 %, lora16 ~4.85 %} of the 8,893,348-param host) — so a recipe
    mismatch is caught before any GPU spend. No weights are downloaded.
    """
    from singnet.peft import (  # lazy: peft is only needed for the D05 dry-run
        apply_recipe,
        load_umxhq,
        measured_trainable_share,
        recipe_rank,
    )

    print(f"DRY RUN — direction 05: {len(config_paths)} configs (no training, mock host)\n")
    header = (f"{'config':<26} {'domain':<10} {'recipe':<8} {'rank':<5} {'lr':<9} "
              f"{'trainable':<11} {'share':<9} hash")
    print(header)
    print("-" * len(header))
    for path in config_paths:
        if not path.exists():
            print(f"{path.name:<26} MISSING")
            continue
        cfg = resolve_config(path)
        recipe = str(cfg["recipe"])
        rank = cfg.get("rank") if cfg.get("rank") is not None else recipe_rank(recipe)
        model = load_umxhq("cpu", mock=True)
        apply_recipe(model, recipe, r=rank)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        share = measured_trainable_share(model)
        print(f"{path.name:<26} {str(cfg.get('domain','?')):<10} {recipe:<8} "
              f"{str(rank):<5} {str(cfg.get('lr','?')):<9} {trainable:<11,} "
              f"{share*100:<8.4f}% {hash_config(cfg)}")
    print("\nShares are of the 8,893,348-param host (H-05a denominator). "
          "Verify recipe/rank/LR match MASTER_PLAN §3.1/§3.4 before GPU spend.")


def dry_run_d06(config_paths: list[Path]) -> None:
    """Direction 06 dry-run: ε / q / eval-guard status per config (no GPU, no data).

    For each config prints the bleed level ε (data path), the trim fraction q (loss
    path), and the **eval-guard status** — that ``build_corruption`` structurally
    refuses ``valid``/``test`` for every ε > 0 arm — plus the trimmed loss the loop
    would build and the config hash, so a mislabelled ε/q or a leaked guard is caught
    before any GPU spend (MASTER_PLAN §7 run book step 1).
    """
    from singnet.data.corrupt import EvalSplitCorruptionError, build_corruption
    from singnet.losses import TrimmedLoss
    from singnet.train import build_training_loss
    from singnet.utils.config import corruption_epsilon, trim_q

    print(f"DRY RUN — direction 06: {len(config_paths)} configs (no training)\n")
    header = (f"{'config':<34} {'arm':<12} {'loss':<6} {'seed':<4} {'eps':<5} {'q':<5} "
              f"{'trimmed':<8} {'eval-guard':<22} hash")
    print(header)
    print("-" * len(header))
    for path in config_paths:
        if not path.exists():
            print(f"{path.name:<34} MISSING")
            continue
        cfg = resolve_config(path)
        eps = corruption_epsilon(cfg)
        q = trim_q(cfg)
        trimmed = isinstance(build_training_loss(cfg), TrimmedLoss)
        if eps > 0.0:
            try:  # the guard MUST refuse a non-train split when ε > 0
                build_corruption(cfg.get("corrupt"), "valid")
                guard = "LEAK! (no refusal)"
            except EvalSplitCorruptionError:
                guard = "refuses valid/test"
        else:
            guard = "n/a (clean ε=0)"
        print(f"{path.name:<34} {str(cfg.get('arm','?')):<12} "
              f"{str(cfg.get('loss', cfg.get('arm'))):<6} {str(cfg.get('seed','?')):<4} "
              f"{str(eps):<5} {str(q):<5} {str(trimmed):<8} {guard:<22} {hash_config(cfg)}")
    base = Path(DIRECTIONS['06']['config_dir']) / "base.yaml"
    if base.exists():
        print(f"\nshared clean cell (ε=0, = D01 l1mag) config-hash: {hash_config(resolve_config(base))}")
    print("Verify ε/q per config and that every ε>0 arm refuses the eval splits "
          "(MASTER_PLAN §3.1/§3.2) before GPU spend.")


def dry_run_d08(config_paths: list[Path]) -> None:
    """Direction 08 dry-run: policy / θ / λ per config, with curriculum's λ(t) endpoints (no GPU).

    For each config prints the chunk-sampling policy and the constant that steers it (θ for
    ``drop``, λ for ``energy``/``curriculum``), plus — for ``curriculum`` — the pinned
    schedule endpoints λ(0)=1.0 → λ(T/2..T)=floor and the annealing budget, so a mislabelled
    policy or a broken schedule is caught before any GPU spend (MASTER_PLAN §7 run book).
    """
    from singnet.data.sampling import build_chunk_sampler
    from singnet.utils.config import sampling_policy

    print(f"DRY RUN — direction 08: {len(config_paths)} configs (no training)\n")
    header = (f"{'config':<38} {'arm':<12} {'loss':<6} {'seed':<4} {'policy':<11} "
              f"{'θ(dB)':<7} {'λ':<6} {'λ(t) schedule':<26} hash")
    print(header)
    print("-" * len(header))
    for path in config_paths:
        if not path.exists():
            print(f"{path.name:<38} MISSING")
            continue
        cfg = resolve_config(path)
        spec = sampling_policy(cfg)
        sampler = build_chunk_sampler(cfg)
        policy = spec["policy"]
        theta = f"{spec['theta_db']:.0f}" if policy == "drop" else "—"
        lam = f"{spec['floor_lambda']:.2f}" if policy in ("energy", "curriculum") else "—"
        if policy == "curriculum":
            steps = int(cfg.get("steps", 0))
            schedule = f"1.0→{spec['floor_lambda']:.1f} over 0..{steps // 2}, hold"
        else:
            schedule = "n/a"
        print(f"{path.name:<38} {str(cfg.get('arm','?')):<12} "
              f"{str(cfg.get('loss', cfg.get('arm'))):<6} {str(cfg.get('seed','?')):<4} "
              f"{policy:<11} {theta:<7} {lam:<6} {schedule:<26} {hash_config(cfg)}")
    base = Path(DIRECTIONS['08']['config_dir']) / "base.yaml"
    if base.exists():
        print(f"\nshared uniform cell (= D01 l1mag) config-hash: {hash_config(resolve_config(base))}")
    print("Verify policy/θ/λ per config; the uniform arm reuses the shared cell "
          "(MASTER_PLAN §4.1/§4.2) before GPU spend.")


def dry_run_d10(config_paths: list[Path]) -> None:
    """Direction 10 dry-run: data_source / p_fma / n_pseudo / MUSDB-path guard per config (no GPU).

    For each config prints the training-data source (musdb/mixed/distill), the FMA pool
    probability p_fma, the pseudo clip count (from the manifest if present, else RUN LATER),
    and the **MUSDB-path guard status** — that :class:`PseudoLabeledShards` structurally refuses
    a MUSDB shard root — plus the config hash, so a mislabelled source/ratio or a leaked guard
    is caught before any GPU spend (MASTER_PLAN §7 run book step 2). ``musdb_only`` is the
    shared cell (0 new runs); the guard is exercised live on a MUSDB-like root.
    """
    from singnet.data.pseudo import MusdbShardLeak, PseudoLabeledShards
    from singnet.utils.config import pseudo_data_spec, pseudo_paths

    # exercise the guard once so the printed status is a real refusal, not a claim.
    guard_ok = False
    try:
        PseudoLabeledShards("/tmp/fake_musdb_root", musdb_roots=("/tmp/fake_musdb_root",))
    except MusdbShardLeak:
        guard_ok = True

    print(f"DRY RUN — direction 10: {len(config_paths)} configs (no training)\n")
    print(f"MUSDB-path guard active (PseudoLabeledShards refuses a MUSDB root): {guard_ok}\n")
    header = (f"{'config':<38} {'arm':<12} {'loss':<6} {'seed':<4} {'data_source':<12} "
              f"{'p_fma':<6} {'n_pseudo':<20} {'guard':<22} hash")
    print(header)
    print("-" * len(header))
    for path in config_paths:
        if not path.exists():
            print(f"{path.name:<38} MISSING")
            continue
        cfg = resolve_config(path)
        spec = pseudo_data_spec(cfg)
        data_source = "musdb" if spec is None else spec["data_source"]
        p_fma = "—" if spec is None else f"{spec['p_fma']:.2f}"
        root, manifest = pseudo_paths(cfg)
        if spec is None:
            n_pseudo = "n/a (musdb)"
            guard = "n/a (no pseudo)"
        else:
            n_pseudo = _pseudo_count(manifest)
            guard = "refuses MUSDB roots" if guard_ok else "LEAK! (no refusal)"
        print(f"{path.name:<38} {str(cfg.get('arm','?')):<12} "
              f"{str(cfg.get('loss', cfg.get('arm'))):<6} {str(cfg.get('seed','?')):<4} "
              f"{data_source:<12} {p_fma:<6} {n_pseudo:<20} {guard:<22} {hash_config(cfg)}")
    base = Path(DIRECTIONS['10']['config_dir']) / "base.yaml"
    if base.exists():
        print(f"\nshared musdb_only cell (= D01 l1mag) config-hash: {hash_config(resolve_config(base))}")
    print("Verify data_source/p_fma per config; the musdb_only arm reuses the shared cell "
          "and the pseudo pool refuses MUSDB shard paths (MASTER_PLAN §3.3/§4.1) before GPU spend.")


def _pseudo_count(manifest_path: str | None) -> str:
    """Human label for the pseudo clip count (from the manifest CSV if present, else RUN LATER)."""
    if manifest_path is None or not Path(manifest_path).exists():
        return "RUN LATER (label)"
    try:
        import pandas as pd

        frame = pd.read_csv(manifest_path, comment="#")
        screened = int(frame.get("screened", pd.Series(dtype=bool)).astype(bool).sum())
        return f"{screened} screened"
    except Exception:  # noqa: BLE001 — dry-run must never crash on a bad/absent manifest
        return "pending (manifest?)"


def dry_run(config_paths: list[Path], direction: str) -> None:
    """Instantiate each config's pipeline; print the switchboard + subset size (no GPU)."""
    if direction == "03":
        dry_run_bandsplit(config_paths)
        return
    if direction == "05":
        dry_run_d05(config_paths)
        return
    if direction == "06":
        dry_run_d06(config_paths)
        return
    if direction == "08":
        dry_run_d08(config_paths)
        return
    if direction == "10":
        dry_run_d10(config_paths)
        return
    print(f"DRY RUN — direction {direction}: {len(config_paths)} configs (no training)\n")
    header = (f"{'config':<32} {'loss':<9} {'seed':<4} {'remix':<6} {'gain':<6} "
              f"{'flip':<6} {'n_songs':<18} hash")
    print(header)
    print("-" * len(header))
    for path in config_paths:
        if not path.exists():
            print(f"{path.name:<32} MISSING")
            continue
        cfg = resolve_config(path)
        sw = augment_switches(cfg)
        pipe = AugmentPipeline(
            remix=sw["remix"], gain=sw["gain"], flip=sw["flip"], seed=int(cfg.get("seed", 0))
        )
        print(
            f"{path.name:<32} {str(cfg.get('arm','?')):<9} {str(cfg.get('seed','?')):<4} "
            f"{str(pipe.remix):<6} {str(pipe.gain):<6} {str(pipe.flip):<6} "
            f"{_subset_size(cfg):<18} {hash_config(cfg)}"
        )
    if direction == "02":
        base = Path(DIRECTIONS['02']['config_dir']) / "base.yaml"
        if base.exists():
            print(f"\nshared FULL-86 cell (= D01 l1mag) config-hash: {hash_config(resolve_config(base))}")
        print("Verify the switchboard/subset columns match MASTER_PLAN §3.1/§3.2 before GPU spend.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--direction", choices=["01", "02", "03", "05", "06", "08", "10"], default="01",
                        help="which study to run")
    parser.add_argument("--stage", default="reduced",
                        choices=["reduced", "full", "contingency", "probes", "main"])
    parser.add_argument("--dry-run", action="store_true",
                        help="CPU: print switchboard + subset sizes, no training")
    parser.add_argument("--exploratory", action="store_true", help="D01 only: also run the 2 lambda configs")
    parser.add_argument("--halve", action="store_true", help="D01 full: apply the §3.2 budget-halving rule")
    args = parser.parse_args(argv)

    config_dir = Path(DIRECTIONS[args.direction]["config_dir"])
    registry = DIRECTIONS[args.direction]["registry"]
    configs = configs_for(
        args.direction, args.stage, config_dir, registry, exploratory=args.exploratory, halve=args.halve
    )

    if args.dry_run:
        dry_run(configs, args.direction)
        return

    print(f"direction={args.direction} stage={args.stage}: {len(configs)} configs")
    runner = _run_config_d05 if args.direction == "05" else _run_config
    for config_path in configs:
        if not config_path.exists():
            raise SystemExit(f"missing config {config_path} (did config generation run?)")
        runner(config_path, registry)


if __name__ == "__main__":
    main()
