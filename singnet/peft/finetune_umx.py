r"""The dedicated Open-Unmix fine-tune loop (Direction 05 MASTER_PLAN §5, §6, §8).

A **separate** trainer for the UMX family — it deliberately does *not* generalize
``singnet/train/loop.py`` (keeping Directions 01-03 risk-free) but **reuses** the
project's seed, checkpoint, and registry utilities. The pipeline is UMX-native:

* **stereo** 6-s chunks (the host is stereo; ``UmxStereoChunks`` keeps both channels,
  unlike the mono ``MusdbChunks``), with the **code-verified UMX augmentation recipe**
  — per-source gain ``U(0.25, 1.25)``, stereo **channel swap** ``p = 0.5``, and
  cross-track **remix** — reusing :class:`~singnet.data.AugmentPipeline`'s
  per-transform ``(seed, stream, step)`` streams (no sign flip; that is Demucs, not
  UMX);
* loss = **MSE on magnitude** (UMX-native, identical for every recipe — this is a
  recipe study, not a loss study);
* **Adam** with a **200-step linear warmup then constant LR** (§3.5); the LR is the
  probe-frozen, per-recipe value (§3.4);
* the recipe's freezing/LoRA-wrapping via :func:`~singnet.peft.umx_wrapper.apply_recipe`;
  BatchNorm stays in **train mode** for every trained recipe (§3.1 running-stats policy);
* resumable checkpointing and a one-row-per-run registry with the §5 columns
  (``domain, recipe, rank, lr, trainable_params, trainable_share, peak_vram_gb``).

**The heavy** :func:`finetune` **is RUN LATER** — it needs decoded stereo shards, the
umxhq weights (G1), and a GPU; it fails loud when they are absent. The CPU test suite
exercises the pieces that run without training: the stereo dataset + augmentation, the
magnitude/loss/optimizer/schedule builders, and a mock forward on tiny tensors.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from ..audio.stft import STFT
from ..data.augment import AugmentPipeline
from ..data.manifest import Manifest
from ..data.musdb_dataset import DEFAULT_CHUNK_S, DEFAULT_SR
from ..metrics.si_sdr import si_sdr
from ..train.loop import load_checkpoint, save_checkpoint
from ..train.registry import RunRecord, current_git_commit, make_run_id, upsert_run
from ..utils.config import augment_switches, hash_config, resolve_config
from ..utils.seed import seed_everything
from .lora import count_trainable_params
from .umx_wrapper import (
    RECIPES,
    apply_recipe,
    load_umxhq,
    measured_trainable_share,
    recipe_rank,
    recipe_trainable_share,
)

# --- pinned fine-tune budget (MASTER_PLAN §3.3, §3.5) -----------------------
FT_STEPS = 6000
FT_BATCH = 16
FT_CHUNK_S = DEFAULT_CHUNK_S       # 6.0 s
WARMUP_STEPS = 200                 # linear warmup, then constant LR
GRAD_CLIP = 5.0
VAL_EVERY = 500                    # validation cadence (best-checkpoint by val SI-SDR)
CHECKPOINT_EVERY = 500
DEFAULT_HOST = "umxhq"


# --- stereo stores ----------------------------------------------------------

def _to_stereo_f32(signal: np.ndarray) -> np.ndarray:
    """Coerce a source to contiguous ``(2, n)`` float32 (duplicate mono to stereo)."""
    arr = np.asarray(signal, dtype=np.float32)
    if arr.ndim == 1:
        arr = np.stack([arr, arr], axis=0)
    elif arr.ndim == 2:
        # accept (n, 2) or (2, n); the axis of length 2 is the channel axis.
        if arr.shape[0] != 2 and arr.shape[1] == 2:
            arr = arr.T
        if arr.shape[0] == 1:
            arr = np.repeat(arr, 2, axis=0)
        elif arr.shape[0] > 2:
            arr = arr[:2]
    return np.ascontiguousarray(arr, dtype=np.float32)


@runtime_checkable
class StereoTrackStore(Protocol):
    """Provides **stereo** ``(2, n)`` float32 sources per track."""

    sample_rate: int

    def track_names(self) -> list[str]:
        ...

    def load_sources_stereo(self, track: str) -> dict[str, np.ndarray]:
        ...


class StereoInMemoryStore:
    """In-memory stereo store for tests and notebook demos (no I/O)."""

    def __init__(self, tracks: dict[str, dict[str, np.ndarray]], sample_rate: int = DEFAULT_SR) -> None:
        self.sample_rate = int(sample_rate)
        self._tracks = {
            name: {src: _to_stereo_f32(sig) for src, sig in stems.items()}
            for name, stems in tracks.items()
        }

    def track_names(self) -> list[str]:
        return list(self._tracks)

    def load_sources_stereo(self, track: str) -> dict[str, np.ndarray]:
        return self._tracks[track]


class StereoWavShardStore:
    """Real stereo store reading decoded WAV shards (RUN LATER; needs data on disk).

    Same layout as :class:`~singnet.data.WavShardStore` but keeps both channels
    (``(2, n)``) — the UMX host is stereo. Optionally reads a domain-suffixed shard
    root (e.g. ``<root>/<track>/vocals.t1_aac64.wav``) written by
    ``scripts/make_domains.py`` (``domain_suffix``).
    """

    def __init__(
        self, shard_root: str | Path, sample_rate: int = DEFAULT_SR, domain_suffix: str | None = None
    ) -> None:
        self.shard_root = Path(shard_root)
        self.sample_rate = int(sample_rate)
        self.domain_suffix = domain_suffix

    def _name(self, stem: str) -> str:
        return f"{stem}.{self.domain_suffix}.wav" if self.domain_suffix else f"{stem}.wav"

    def track_names(self) -> list[str]:
        return sorted(p.name for p in self.shard_root.iterdir() if p.is_dir())

    def load_sources_stereo(self, track: str) -> dict[str, np.ndarray]:
        import soundfile as sf  # local import: only needed with real data

        track_dir = self.shard_root / track
        vocals, sr = sf.read(track_dir / self._name("vocals"), dtype="float32", always_2d=True)
        if sr != self.sample_rate:
            raise ValueError(f"{track}: sample rate {sr} != expected {self.sample_rate}")
        acc_path = track_dir / self._name("accompaniment")
        if acc_path.exists():
            accompaniment, _ = sf.read(acc_path, dtype="float32", always_2d=True)
        else:
            accompaniment = sum(
                sf.read(track_dir / self._name(stem), dtype="float32", always_2d=True)[0]
                for stem in ("drums", "bass", "other")
            )
        return {"vocals": _to_stereo_f32(vocals), "accompaniment": _to_stereo_f32(accompaniment)}


# --- the stereo chunk dataset ----------------------------------------------

def umx_augment_pipeline(
    remix: bool = True, gain: bool = True, channelswap: bool = True, seed: int = 0
) -> AugmentPipeline:
    """The UMX augmentation switchboard (gain + channel swap + remix; **no** sign flip)."""
    return AugmentPipeline(
        remix=remix, gain=gain, flip=False, channelswap=channelswap, seed=seed
    )


class UmxStereoChunks(Dataset):
    """Stereo 6-s chunk sampler with the UMX augmentation recipe (§3.3).

    Deterministic per ``(seed, index)`` — the same per-transform stream design as
    :class:`~singnet.data.MusdbChunks` (§3.5), reusing the ``sample``/``remix``/
    ``gain``/``channelswap`` streams. Each item: pick vocals (track+window) on the
    ``sample`` stream; draw the accompaniment cross-track on the ``remix`` stream
    (or the same track+window without remix); apply per-source gain then stereo
    channel swap; sum to the mixture. Returns stereo tensors ``(2, chunk)``.
    """

    def __init__(
        self,
        store: StereoTrackStore,
        manifest: Manifest,
        split: str = "train",
        seed: int = 0,
        chunk_s: float = FT_CHUNK_S,
        track_allowlist: list[str] | None = None,
        length: int | None = None,
        pipeline: AugmentPipeline | None = None,
    ) -> None:
        self.store = store
        self.seed = int(seed)
        self.sample_rate = int(store.sample_rate)
        self.chunk_len = int(round(chunk_s * self.sample_rate))
        if pipeline is None:
            pipeline = umx_augment_pipeline(seed=self.seed)
        self.pipeline = pipeline.with_seed(self.seed)
        self.remix = self.pipeline.remix

        available = set(store.track_names())
        tracks = [t for t in manifest.trainable_tracks(split) if t in available]
        if track_allowlist is not None:
            allow = set(track_allowlist)
            tracks = [t for t in tracks if t in allow]
        manifest.assert_no_test_rows(tracks)
        if not tracks:
            raise ValueError(f"no {split} tracks available in the stereo store")
        self.tracks = tracks
        self._length = int(length) if length is not None else max(64, 8 * len(self.tracks))

    def __len__(self) -> int:
        return self._length

    @property
    def n_songs(self) -> int:
        return len(self.tracks)

    def _prepare(self, samples: np.ndarray) -> np.ndarray:
        """Loop-pad a ``(2, n)`` source to at least one chunk length."""
        n = samples.shape[-1]
        if n < self.chunk_len:
            reps = int(np.ceil(self.chunk_len / max(n, 1)))
            samples = np.tile(samples, (1, reps))
        return samples

    def _start(self, padded: np.ndarray, rng: np.random.Generator) -> int:
        return int(rng.integers(0, padded.shape[-1] - self.chunk_len + 1))

    def _slice(self, padded: np.ndarray, start: int) -> np.ndarray:
        return np.ascontiguousarray(padded[:, start : start + self.chunk_len], dtype=np.float32)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        p = self.pipeline
        sample_rng = p.stream("sample", index)
        voc_track = str(sample_rng.choice(self.tracks))
        voc_padded = self._prepare(_to_stereo_f32(self.store.load_sources_stereo(voc_track)["vocals"]))
        voc_start = self._start(voc_padded, sample_rng)
        vocals = self._slice(voc_padded, voc_start)

        if p.remix:
            remix_rng = p.stream("remix", index)
            acc_track = str(remix_rng.choice(self.tracks))
            acc_padded = self._prepare(
                _to_stereo_f32(self.store.load_sources_stereo(acc_track)["accompaniment"])
            )
            acc_start = self._start(acc_padded, remix_rng)
        else:
            acc_padded = self._prepare(
                _to_stereo_f32(self.store.load_sources_stereo(voc_track)["accompaniment"])
            )
            acc_start = min(voc_start, acc_padded.shape[-1] - self.chunk_len)
        accompaniment = self._slice(acc_padded, acc_start)

        sources = {"vocals": vocals, "accompaniment": accompaniment}
        sources = p.apply_gain(sources, index)
        sources = p.apply_channelswap(sources, index)
        mixture = sources["vocals"] + sources["accompaniment"]
        return {
            "mixture": torch.from_numpy(np.ascontiguousarray(mixture)).float(),
            "vocals": torch.from_numpy(np.ascontiguousarray(sources["vocals"])).float(),
        }


# --- UMX front-end / loss / optimizer / schedule ----------------------------

def umx_magnitude(stft: STFT, wave: Tensor) -> Tensor:
    """Stereo waveform ``(B, 2, L)`` -> UMX magnitude ``(B, 2, nb_output_bins, T)``."""
    return stft.transform(wave).abs()


def umx_mse_loss(estimate_mag: Tensor, target_mag: Tensor) -> Tensor:
    """UMX-native training objective: mean-squared error on magnitude (§1.3)."""
    return torch.mean((estimate_mag - target_mag) ** 2)


def make_umx_lr_lambda(warmup_steps: int = WARMUP_STEPS) -> Callable[[int], float]:
    """Multiplicative LR factor: linear warmup to 1.0 over ``warmup_steps``, then constant.

    Unlike the Directions 01-03 cosine schedule, UMX fine-tuning uses a **constant** LR
    after warmup (§3.5) — the PEFT-fairness comparison holds the schedule fixed and lets
    only the (probe-chosen) base LR differ per recipe.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0

    return lr_lambda


def build_umx_optimizer(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    """Adam over the model's **trainable** parameters only (frozen base excluded)."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("no trainable parameters — apply a non-zeroshot recipe first")
    return torch.optim.Adam(trainable, lr=lr)


def peak_vram_gb() -> float:
    """Peak CUDA memory this process has allocated, in GB (NaN on CPU)."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return float("nan")


# --- sanity report (recipe share table; RUN-LATER val is guarded) -----------

def recipe_share_table(base_params: int | None = None) -> "Any":
    """A DataFrame of every recipe's trained parameter count and host-share (§3.1).

    Pure/closed-form on the verified shapes (no model needed) — this is the G1
    "recompute the trainable-share table" artifact, runnable on CPU now. ``head``
    ≈ 23.7 %, ``lora4`` ≈ 1.27 %, ``lora16`` ≈ 4.85 % of the 8,893,348-param host.
    """
    import pandas as pd

    from .umx_wrapper import BASE_PARAM_COUNT, _lora_ab_cost

    base = base_params if base_params is not None else BASE_PARAM_COUNT
    rows = []
    for recipe in RECIPES:
        share = recipe_trainable_share(recipe)
        trained = int(round(share * base))
        rows.append(
            {
                "recipe": recipe,
                "rank": recipe_rank(recipe) or 0,
                "trainable_params": trained,
                "trainable_share": share,
                "added_lora_params": _lora_ab_cost(recipe_rank(recipe)) if recipe_rank(recipe) else 0,
            }
        )
    return pd.DataFrame(rows)


def sanity(mock: bool = False, device: str = "cpu") -> "Any":
    """G1 checkpoint sanity (RUN LATER for the real weights; the table runs now).

    Prints and returns the recipe trainable-share table. With real weights
    (``mock=False``, RUN LATER) this is where zero-shot val SI-SDR on the standard +
    T1 val splits would be measured (must beat do-nothing by > 3 dB on standard, §7
    G1); that measurement needs the decoded shards and is left to the run book.
    """
    model = load_umxhq(device, mock=mock)
    table = recipe_share_table(count_trainable_params(model) if _all_trainable(model) else None)
    print("Direction 05 — recipe trainable-share table (host params "
          f"= {sum(p.numel() for p in model.parameters()):,}):")
    print(table.to_string(index=False))
    print("\nzero-shot val SI-SDR sanity (standard + T1 val): RUN LATER — needs shards + weights.")
    return table


def _all_trainable(model: nn.Module) -> bool:
    return all(p.requires_grad for p in model.parameters())


# --- the fine-tune loop (RUN LATER) -----------------------------------------

@dataclass
class FinetuneResult:
    """Summary returned by :func:`finetune` and mirrored into the registry."""

    run_id: str
    domain: str
    recipe: str
    rank: int
    config_hash: str
    steps_done: int
    trainable_params: int
    trainable_share: float
    best_val_sisdr: float
    checkpoint_path: str


def finetune(config_path: str | Path, *, registry_path: str | Path | None = None) -> FinetuneResult:
    """Fine-tune one recipe on one domain to the 6 k-step budget. **RUN LATER.**

    Needs decoded stereo shards, the umxhq weights (G1), and a GPU; resumable to the
    step via the checkpoint under ``config['output_dir']``. Not exercised by the CPU
    suite — the schedule/loss/optimizer/dataset it composes are unit-tested separately.
    """
    from torch.utils.data import DataLoader

    config = resolve_config(config_path)
    config_hash = hash_config(config)
    seed = int(config.get("seed", 0))
    domain = str(config.get("domain", "standard"))
    recipe = str(config["recipe"])
    if recipe not in RECIPES:
        raise ValueError(f"unknown recipe {recipe!r}; expected {RECIPES}")
    rank = int(config.get("rank") or recipe_rank(recipe) or 0)
    lr = float(config.get("lr", 1e-3))
    total_steps = int(config.get("steps", FT_STEPS))
    batch_size = int(config.get("batch_size", FT_BATCH))
    warmup = int(config.get("warmup_steps", WARMUP_STEPS))
    seed_everything(seed, deterministic=config.get("deterministic", False))

    shard_root = config.get("shard_root")
    if not shard_root or not Path(shard_root).exists():
        raise FileNotFoundError(
            "shard_root missing — run scripts/prepare_data.py + scripts/make_domains.py first "
            f"(RUN LATER). Got shard_root={shard_root!r}."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = Manifest.from_csv(config["splits_csv"])
    store = StereoWavShardStore(
        shard_root, sample_rate=config.get("sample_rate", DEFAULT_SR),
        domain_suffix=config.get("domain_suffix"),
    )

    switches = augment_switches(config)
    pipeline = umx_augment_pipeline(
        remix=switches["remix"], gain=switches["gain"],
        channelswap=bool(config.get("channelswap", True)), seed=seed,
    )
    train_ds = UmxStereoChunks(
        store, manifest, "train", seed=seed, chunk_s=config.get("chunk_s", FT_CHUNK_S),
        pipeline=pipeline, length=total_steps * batch_size,
    )
    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, drop_last=True,
        num_workers=config.get("num_workers", 4),
    )

    stft = STFT().to(device)
    model = load_umxhq(device, mock=bool(config.get("mock", False)))
    apply_recipe(model, recipe, r=rank or None)
    model.train()  # BN stays in train mode for every trained recipe (§3.1)
    optimizer = build_umx_optimizer(model, lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_umx_lr_lambda(warmup))

    output_dir = Path(config.get("output_dir", "checkpoints")) / f"d05_{config_hash}"
    ckpt_path, best_path = output_dir / "last.pt", output_dir / "best.pt"
    registry_path = registry_path or config.get("registry_path", "05-lora-source-separation/results/registry.csv")

    start_step, best_val = 0, float("-inf")
    if ckpt_path.exists():
        payload = load_checkpoint(ckpt_path, model=model, optimizer=optimizer, scheduler=scheduler)
        start_step, best_val = int(payload["step"]), float(payload["best_metric"])

    started = time.time()
    step = start_step
    data_iter = iter(loader)
    while step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        mix_mag = umx_magnitude(stft, batch["mixture"].to(device))
        tgt_mag = umx_magnitude(stft, batch["vocals"].to(device))
        optimizer.zero_grad(set_to_none=True)
        est_mag = model(mix_mag)
        loss = umx_mse_loss(est_mag, tgt_mag)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % VAL_EVERY == 0:
            val = umx_validation_sisdr(model, stft, store, manifest, device)
            if val > best_val:
                best_val = val
                save_checkpoint(best_path, model=model, optimizer=optimizer, scheduler=scheduler,
                                step=step, best_metric=best_val, config=config)
            model.train()
        if step % CHECKPOINT_EVERY == 0:
            save_checkpoint(ckpt_path, model=model, optimizer=optimizer, scheduler=scheduler,
                            step=step, best_metric=best_val, config=config)

    save_checkpoint(ckpt_path, model=model, optimizer=optimizer, scheduler=scheduler,
                    step=step, best_metric=best_val, config=config)
    trainable = count_trainable_params(model)
    share = measured_trainable_share(model)
    run_id = make_run_id(f"{domain}_{recipe}", seed, "ft6k", config_hash)
    upsert_run(
        registry_path,
        RunRecord(
            run_id=run_id, arm=recipe, seed=seed, budget=total_steps, config_hash=config_hash,
            git_commit=current_git_commit(),
            gpu=torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
            wall_clock_h=(time.time() - started) / 3600.0, steps_done=step,
            best_val_sisdr=best_val, checkpoint_path=str(best_path),
            domain=domain, recipe=recipe, rank=rank, lr=lr,
            trainable_params=trainable, trainable_share=share, peak_vram_gb=peak_vram_gb(),
        ),
    )
    return FinetuneResult(run_id, domain, recipe, rank, config_hash, step, trainable, share,
                          best_val, str(best_path))


def umx_validation_sisdr(model, stft: STFT, store, manifest, device: torch.device) -> float:
    """Mean full-track vocals SI-SDR over the domain val tracks (RUN LATER).

    UMX-native stereo separation (mask x mixture, mixture-phase iSTFT, no Wiener),
    scored per channel and averaged. Needs decoded stereo shards.
    """
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for track in manifest.tracks_for("valid"):
            if track not in set(store.track_names()):
                continue
            sources = store.load_sources_stereo(track)
            vocals = _to_stereo_f32(sources["vocals"])
            accompaniment = _to_stereo_f32(sources["accompaniment"])
            mixture = torch.from_numpy(vocals + accompaniment).float().to(device)
            est = umx_separate(model, mixture, stft)
            scores.append(si_sdr(est.cpu().numpy().reshape(-1), vocals.reshape(-1)))
    model.train()
    return float(np.mean(scores)) if scores else float("nan")


def umx_separate(model, mixture: Tensor, stft: STFT) -> Tensor:
    """Full-track stereo vocals estimate: mask x mixture spectrogram, iSTFT (RUN LATER)."""
    spec = stft.transform(mixture.unsqueeze(0))          # (1, 2, F, T) complex
    est_mag = model(spec.abs())                           # (1, 2, F, T) magnitude
    ratio = est_mag / (spec.abs() + 1e-8)
    est_spec = ratio.to(spec.dtype) * spec
    return stft.inverse(est_spec, length=mixture.shape[-1]).squeeze(0)


# --- CLI --------------------------------------------------------------------

def _cli(argv: list[str] | None = None) -> None:
    """``python -m singnet.peft.finetune_umx`` entry point (RUN LATER; needs data + GPU)."""
    parser = argparse.ArgumentParser(
        description="Fine-tune umxhq with a PEFT recipe (RUN LATER; needs stereo shards + GPU)."
    )
    parser.add_argument("--config", default=None, help="path to a resolved D05 YAML config")
    parser.add_argument("--registry", default=None, help="registry CSV path (overrides config)")
    parser.add_argument("--sanity", action="store_true",
                        help="G1: print the recipe trainable-share table (+ zero-shot val, RUN LATER)")
    parser.add_argument("--mock", action="store_true", help="use the mock host (no weight download)")
    args = parser.parse_args(argv)
    if args.sanity:
        sanity(mock=args.mock)
        return
    if not args.config:
        parser.error("--config is required unless --sanity is given")
    result = finetune(args.config, registry_path=args.registry)
    print(result)


if __name__ == "__main__":
    _cli()
