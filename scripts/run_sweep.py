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

``--dry-run`` (CPU, no GPU, no data) instantiates each config and prints the
per-config summary that matters for that direction — the augmentation switchboard
+ subset size (Directions 01/02), or the band edges, chosen width ``c`` and exact
parameter count (Direction 03) — so a mismatch is caught before any GPU spend.
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


def configs_for(
    direction: str, stage: str, config_dir: Path, registry: str, *, exploratory: bool, halve: bool
) -> list[Path]:
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


def dry_run(config_paths: list[Path], direction: str) -> None:
    """Instantiate each config's pipeline; print the switchboard + subset size (no GPU)."""
    if direction == "03":
        dry_run_bandsplit(config_paths)
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
    parser.add_argument("--direction", choices=["01", "02", "03"], default="01", help="which study to run")
    parser.add_argument("--stage", default="reduced", choices=["reduced", "full", "contingency"])
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
    for config_path in configs:
        if not config_path.exists():
            raise SystemExit(f"missing config {config_path} (did config generation run?)")
        _run_config(config_path, registry)


if __name__ == "__main__":
    main()
