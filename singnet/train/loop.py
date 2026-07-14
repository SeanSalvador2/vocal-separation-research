r"""Training loop, LR schedule, and resumable checkpointing (MASTER_PLAN §7.1).

Pinned configuration (identical for every run — only the loss module changes):

* AdamW, ``lr = 1e-3``, ``betas = (0.9, 0.999)``, ``weight_decay = 1e-4``.
* Linear warmup 500 steps, then cosine decay to ``1e-5`` at budget end.
* Batch 16 x 6 s chunks, AMP (fp16 + ``GradScaler``), global-L2 grad clip 5.0.
* Checkpoint (model + optimizer + scaler + scheduler + RNG) every 1000 steps.
* Full-track val SI-SDR every 2000 steps; best checkpoint = highest val SI-SDR.

**The heavy** :func:`run` **is RUN LATER** — it needs the decoded dataset and a
GPU. It is import-clean and syntactically runnable, and fails loud with a clear
message when the shards are missing. The unit tests exercise the pieces that run
on CPU without training: the LR schedule and the checkpoint state-dict
round-trip (resume equivalence).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor, nn

from ..audio.stft import STFT, analyze_chunk
from ..losses import SeparationLoss, TrimmedLoss
from ..losses import build as build_loss
from ..models import build_model_from_config
from ..utils.config import (
    augment_switches,
    corruption_epsilon,
    hash_config,
    resolve_config,
    track_allowlist_path,
    trim_q,
)
from ..utils.seed import capture_rng_state, restore_rng_state, seed_everything
from .registry import RunRecord, current_git_commit, make_run_id, upsert_run

# --- pinned hyper-parameters ------------------------------------------------
BASE_LR = 1e-3
MIN_LR = 1e-5
WEIGHT_DECAY = 1e-4
BETAS = (0.9, 0.999)
WARMUP_STEPS = 500
GRAD_CLIP = 5.0
CHECKPOINT_EVERY = 1000
VAL_EVERY = 2000
#: Direction 06 §5: cadence of the kept-vs-dropped accompaniment-energy telemetry.
TRIM_TELEMETRY_EVERY = 500
TRIM_TELEMETRY_COLUMNS = (
    "step", "n_kept", "n_dropped", "kept_fraction",
    "kept_energy_mean", "dropped_energy_mean",
    "kept_energy_median", "dropped_energy_median",
)
#: Direction 08 §6: cadence + schema of the realized silent-chunk exposure telemetry.
SAMPLING_TELEMETRY_EVERY = 500
SAMPLING_TELEMETRY_COLUMNS = ("step", "silent_fraction", "n_chunks", "theta_db")


def build_training_loss(config: dict[str, Any]) -> SeparationLoss:
    """Build the (optionally trimmed) training loss for a config (Direction 06 §3.2).

    Returns the plain base loss when no ``trim:`` block is present — so every
    Direction 01–05 config builds **byte-identically** to before this wiring
    existed (the no-op guarantee, unit-tested). A ``trim.q`` wraps the base loss in
    a loss-agnostic :class:`singnet.losses.TrimmedLoss`.
    """
    arm = str(config["arm"])
    loss_name = str(config.get("loss", arm))
    base = build_loss(loss_name, **config.get("loss_kwargs", {}))
    q = trim_q(config)
    if q is None:
        return base
    return TrimmedLoss(base, q)


def _trim_telemetry_row(step: int, aux: dict[str, Any]) -> dict[str, Any]:
    """One kept-vs-dropped accompaniment-energy telemetry row from a TrimmedLoss aux."""
    return {
        "step": int(step),
        "n_kept": aux.get("n_kept"),
        "n_dropped": aux.get("n_dropped"),
        "kept_fraction": aux.get("kept_fraction"),
        "kept_energy_mean": aux.get("kept_energy_mean", float("nan")),
        "dropped_energy_mean": aux.get("dropped_energy_mean", float("nan")),
        "kept_energy_median": aux.get("kept_energy_median", float("nan")),
        "dropped_energy_median": aux.get("dropped_energy_median", float("nan")),
    }


def write_trim_telemetry(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write the per-run trim-telemetry CSV (the §2 mechanism evidence, §5)."""
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(TRIM_TELEMETRY_COLUMNS)).to_csv(path, index=False)


# --- Direction 08: sampling exposure telemetry + the selection-bias guard ---

def chunk_silent_fraction(vocals: Tensor, theta_db: float) -> float:
    """Fraction of chunks in a ``(B, L)`` batch whose vocal-target RMS is below θ dBFS.

    The policy's *realized* silent-chunk exposure (MASTER_PLAN §6) — closes the loop on
    the THEORY §4 per-policy exposure predictions. Pure read, no RNG: logging it cannot
    perturb training or the data stream.
    """
    flat = vocals.reshape(vocals.shape[0], -1).to(torch.float64)
    rms = flat.pow(2).mean(dim=-1).sqrt()
    return float((rms < 10.0 ** (theta_db / 20.0)).to(torch.float64).mean())


def _exposure_row(step: int, silent_fraction: float, n_chunks: int, theta_db: float) -> dict[str, Any]:
    """One realized-exposure telemetry row (§6)."""
    return {
        "step": int(step),
        "silent_fraction": float(silent_fraction),
        "n_chunks": int(n_chunks),
        "theta_db": float(theta_db),
    }


def write_exposure_telemetry(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write the per-run silent-chunk exposure CSV (§6 policy-realized-behavior evidence)."""
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(SAMPLING_TELEMETRY_COLUMNS)).to_csv(path, index=False)


def is_new_best(candidate_sisdr: float, best_sisdr: float) -> bool:
    """Best-checkpoint selection test — **SI-SDR only** (MASTER_PLAN §6 selection guard).

    Checkpoint selection must never see SLR, so the leakage comparison (H-08b) is not
    selection-biased: the SLR reported for an arm is that of its *SI-SDR*-best checkpoint.
    This function takes no SLR argument by construction (asserted in ``tests/test_train_d08.py``).
    """
    return candidate_sisdr > best_sisdr


@dataclass
class RunResult:
    """Summary returned by :func:`run` and mirrored into the registry."""

    run_id: str
    config_hash: str
    steps_done: int
    best_val_sisdr: float
    final_val_sisdr: float
    sisdr_skip_rate: float
    checkpoint_path: str


# --- LR schedule ------------------------------------------------------------

def make_lr_lambda(
    warmup_steps: int, total_steps: int, base_lr: float = BASE_LR, min_lr: float = MIN_LR
) -> Callable[[int], float]:
    r"""Multiplicative LR factor: linear warmup then cosine decay to ``min_lr``.

    Returns ``f(step)`` such that ``lr(step) = base_lr * f(step)``:

    * ``step < warmup``: ``step / warmup`` (linear 0 -> 1),
    * else: cosine from 1 down to ``min_lr / base_lr`` over the remaining steps.
    """
    min_ratio = min_lr / base_lr

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return lr_lambda


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=BASE_LR, betas=BETAS, weight_decay=WEIGHT_DECAY)


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(WARMUP_STEPS, total_steps))


# --- front-end prep ---------------------------------------------------------

def prepare_batch(stft: STFT, mixture: Tensor, vocals: Tensor) -> dict[str, Tensor]:
    """Turn raw waveform batches into the tensors the loss contract expects.

    Target/mixture waveforms are the STFT-consistent reconstructions (iSTFT of
    the cropped spectrogram) so estimate and target share an identical grid and
    length (see :mod:`singnet.audio.stft`).
    """
    mix_spec, mix_mag = analyze_chunk(stft, mixture)
    tgt_spec, tgt_mag = analyze_chunk(stft, vocals)
    return {
        "mix_mag": mix_mag,
        "tgt_mag": tgt_mag,
        "mix_stft": mix_spec,
        "tgt_wave": stft.inverse(tgt_spec),
        "mix_wave": stft.inverse(mix_spec),
    }


# --- checkpointing ----------------------------------------------------------

def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: "torch.amp.GradScaler | None" = None,
    step: int = 0,
    best_metric: float = float("-inf"),
    config: dict[str, Any] | None = None,
    capture_rng: bool = True,
) -> None:
    """Persist model + optimizer + scaler + scheduler + RNG + bookkeeping."""
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": int(step),
        "best_metric": float(best_metric),
        "config": config or {},
        "rng_state": capture_rng_state() if capture_rng else None,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: "torch.amp.GradScaler | None" = None,
    map_location: str = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore state into the provided objects; returns the raw payload.

    Any object left as ``None`` is skipped, so the same function serves full
    resume and metrics-only inspection.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None:
        model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if restore_rng and payload.get("rng_state") is not None:
        restore_rng_state(payload["rng_state"])
    return payload


# --- the orchestration (RUN LATER) ------------------------------------------

def run(config_path: str | Path, *, registry_path: str | Path | None = None) -> RunResult:
    """Train one arm to its budget from a resolved YAML config. **RUN LATER.**

    Expected runtime (T4): ~1.3-1.8 h at REDUCED (16k steps), ~3.5-4.5 h at FULL
    (40k). Requires the decoded WAV shards on disk and (for AMP) a GPU. The
    function is resumable to the step via the checkpoint written under
    ``config['output_dir']``.

    This is deliberately not exercised by the CPU test suite; the schedule and
    checkpoint round-trip it relies on are unit-tested separately.
    """
    # Local imports: only needed when actually training, keeps import graph light.
    from torch.utils.data import DataLoader

    from ..data.augment import AugmentPipeline
    from ..data.corrupt import build_corruption
    from ..data.manifest import Manifest
    from ..data.musdb_dataset import MusdbChunks, WavShardStore, load_track_allowlist
    from ..data.profiles import load_energy_profiles
    from ..data.sampling import build_chunk_sampler
    from ..eval.evaluate import validation_report
    from ..utils.config import sampling_policy

    config = resolve_config(config_path)
    config_hash = hash_config(config)
    seed = int(config.get("seed", 0))
    arm = str(config["arm"])
    # The loss is usually named by `arm` (Directions 01/02). Direction 03's arms
    # (baseline/split_mel/split_uniform) all train on `l1mag`, so a `loss` key
    # decouples the loss from the experimental arm id; absent -> loss == arm
    # (byte-identical for the old configs, hash unchanged).
    loss_name = str(config.get("loss", arm))
    budget_name = str(config.get("budget_name", "custom"))
    total_steps = int(config["steps"])
    batch_size = int(config.get("batch_size", 16))
    seed_everything(seed, deterministic=config.get("deterministic", False))

    shard_root = config.get("shard_root")
    if not shard_root or not Path(shard_root).exists():
        raise FileNotFoundError(
            "shard_root missing — run scripts/prepare_data.py first (RUN LATER). "
            f"Got shard_root={shard_root!r}."
        )
    manifest = Manifest.from_csv(config["splits_csv"])
    store = WavShardStore(shard_root, sample_rate=config.get("sample_rate", 44100))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # The augmentation switchboard (§3.1) and the subset allowlist (§3.2) are the
    # only two things that vary across this direction's runs; both are read from
    # the config's canonical form so D01 (augment: bool) and D02 (augment: dict)
    # configs build the same pipeline object.
    switches = augment_switches(config)
    pipeline = AugmentPipeline(remix=switches["remix"], gain=switches["gain"], flip=switches["flip"])
    allowlist = load_track_allowlist(track_allowlist_path(config))
    # Direction 06 §3.1: optional load-time ε-bleed of the TRAIN targets (None for
    # every uncorrupted run, so the dataset is byte-identical). The eval-split guard
    # is structural — build_corruption refuses a non-train split by construction.
    corruption = build_corruption(config.get("corrupt"), "train")
    # Direction 08 §4.1: the chunk-start policy. `build_chunk_sampler` returns the uniform
    # no-op for every Direction 01–06 config (bit-identical data); a non-uniform policy
    # needs the prep-time energy profiles (loud error if the profile pass has not run).
    sampler = build_chunk_sampler(config)
    energy_profiles = None
    if not sampler.is_uniform():
        energy_profiles = load_energy_profiles(Path(shard_root) / "energy_profiles.json")
    train_ds = MusdbChunks(
        store, manifest, "train", seed=seed, chunk_s=config.get("chunk_s", 6.0),
        track_allowlist=allowlist, pipeline=pipeline, length=total_steps * batch_size,
        corrupt=corruption, sampler=sampler, energy_profiles=energy_profiles, batch_size=batch_size,
    )
    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, drop_last=True,
        num_workers=config.get("num_workers", 4),
    )

    stft = STFT().to(device)
    model = build_model_from_config(config).to(device)
    # Direction 06 §3.2: TrimmedLoss wrap when a `trim:` block is present; the plain
    # base loss otherwise (identical to the pre-D06 construction for every D01–D05 run).
    loss_fn = build_training_loss(config).to(device)
    is_trimmed = isinstance(loss_fn, TrimmedLoss)
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer, total_steps)
    use_amp = bool(config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    output_dir = Path(config.get("output_dir", "checkpoints")) / config_hash
    ckpt_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    registry_path = registry_path or config.get("registry_path", "results/registry.csv")

    start_step, best_val = 0, float("-inf")
    if ckpt_path.exists():  # resume
        payload = load_checkpoint(ckpt_path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler)
        start_step, best_val = int(payload["step"]), float(payload["best_metric"])

    started = time.time()
    skip_rate, final_val = float("nan"), float("nan")
    step = start_step
    trim_rows: list[dict[str, Any]] = []
    kept_fraction_sum, kept_fraction_n = 0.0, 0
    # Direction 08 §6: realized silent-chunk exposure telemetry + the SLR of the SI-SDR-best
    # checkpoint (descriptive; SLR never drives selection — `is_new_best` is SI-SDR only).
    exposure_rows: list[dict[str, Any]] = []
    exposure_sum, exposure_n = 0.0, 0
    window_sum, window_n = 0.0, 0
    theta_db = sampler.theta_db
    best_slr = float("nan")
    data_iter = iter(loader)
    while step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        prepared = prepare_batch(stft, batch["mixture"].to(device), batch["vocals"].to(device))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            mask = model(prepared["mix_mag"])
        # For a TrimmedLoss the per-chunk accompaniment energy is threaded in for the
        # §5 telemetry; for every other loss the call is byte-identical to before.
        extra = {"chunk_energy": batch["acc_energy"].to(device)} if is_trimmed else {}
        loss, aux = loss_fn(mask, **prepared, **extra)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        skip_rate = aux.get("skip_rate", skip_rate)
        step += 1

        if is_trimmed:
            kept_fraction_sum += float(aux["kept_fraction"])
            kept_fraction_n += 1
            if step % TRIM_TELEMETRY_EVERY == 0:
                trim_rows.append(_trim_telemetry_row(step, aux))

        # §6 realized exposure: fraction of drawn chunks whose vocal target is silent (θ).
        frac = chunk_silent_fraction(batch["vocals"], theta_db)
        exposure_sum += frac
        exposure_n += 1
        window_sum += frac
        window_n += 1
        if step % SAMPLING_TELEMETRY_EVERY == 0:
            exposure_rows.append(_exposure_row(step, window_sum / window_n, window_n * batch_size, theta_db))
            window_sum, window_n = 0.0, 0

        if step % VAL_EVERY == 0:
            # SLR is computed alongside SI-SDR every eval point, but checkpoint SELECTION
            # is SI-SDR-only (`is_new_best` takes no SLR argument) — the §6 bias guard.
            report = validation_report(model, stft, store, manifest, device)
            final_val = report["sisdr"]
            if is_new_best(final_val, best_val):
                best_val, best_slr = final_val, report["slr"]
                save_checkpoint(best_path, model=model, optimizer=optimizer, scheduler=scheduler,
                                scaler=scaler, step=step, best_metric=best_val, config=config)
        if step % CHECKPOINT_EVERY == 0:
            save_checkpoint(ckpt_path, model=model, optimizer=optimizer, scheduler=scheduler,
                            scaler=scaler, step=step, best_metric=best_val, config=config)

    save_checkpoint(ckpt_path, model=model, optimizer=optimizer, scheduler=scheduler,
                    scaler=scaler, step=step, best_metric=best_val, config=config)
    run_id = make_run_id(arm, seed, budget_name, config_hash)
    model_cfg = config.get("model") or {}
    base_width = int(
        model_cfg.get("base_width", config.get("base_channels", 32))
        if model_cfg.get("arch") == "bandsplit"
        else config.get("base_channels", 32)
    )
    # Direction 06 §5 registry fields: ε (data path), q (loss path), the observed mean
    # kept-fraction, and the trim-telemetry CSV path (written only for trimmed runs).
    trim_energy_stats_path = ""
    kept_fraction_observed = float("nan")
    if is_trimmed:
        trim_energy_stats_path = str(output_dir / "trim_energy_stats.csv")
        write_trim_telemetry(trim_energy_stats_path, trim_rows)
        kept_fraction_observed = kept_fraction_sum / max(1, kept_fraction_n)
    q_value = trim_q(config)
    # Direction 08 §6: write the realized-exposure CSV and record the sampling metadata
    # (policy + its relevant constant; the realized silent exposure; and the SLR of the
    # SI-SDR-best checkpoint — descriptive, never a selection input).
    exposure_stats_path = str(output_dir / "sampling_exposure.csv")
    write_exposure_telemetry(exposure_stats_path, exposure_rows)
    silent_exposure_observed = exposure_sum / max(1, exposure_n)
    spec = sampling_policy(config)
    policy_theta = spec["theta_db"] if spec["policy"] == "drop" else float("nan")
    policy_lambda = spec["floor_lambda"] if spec["policy"] in ("energy", "curriculum") else float("nan")
    upsert_run(
        registry_path,
        RunRecord(
            run_id=run_id, arm=arm, seed=seed, budget=total_steps, config_hash=config_hash,
            git_commit=current_git_commit(), gpu=torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
            wall_clock_h=(time.time() - started) / 3600.0, steps_done=step,
            best_val_sisdr=best_val, final_val_sisdr=final_val, sisdr_skip_rate=skip_rate,
            checkpoint_path=str(best_path),
            aug_remix=switches["remix"], aug_gain=switches["gain"], aug_flip=switches["flip"],
            n_songs=train_ds.n_songs, base_width=base_width,
            epsilon=corruption_epsilon(config),
            trim_q=float("nan") if q_value is None else q_value,
            kept_fraction_observed=kept_fraction_observed,
            trim_energy_stats_path=trim_energy_stats_path,
            policy=spec["policy"], theta_db=policy_theta, floor_lambda=policy_lambda,
            silent_exposure_observed=silent_exposure_observed, best_val_slr=best_slr,
        ),
    )
    return RunResult(run_id, config_hash, step, best_val, final_val, skip_rate, str(best_path))


def _cli(argv: list[str] | None = None) -> None:
    """``python -m singnet.train --config …`` entry point (RUN LATER)."""
    import argparse

    parser = argparse.ArgumentParser(description="Train one SingNet loss arm (RUN LATER; needs data + GPU).")
    parser.add_argument("--config", required=True, help="path to a resolved YAML arm config")
    parser.add_argument("--registry", default=None, help="registry CSV path (overrides config)")
    args = parser.parse_args(argv)
    result = run(args.config, registry_path=args.registry)
    print(result)
