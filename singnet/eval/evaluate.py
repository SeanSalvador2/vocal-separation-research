r"""Full-track evaluation: SI-SDR, oracle/floor anchors, and the CSV writers.

Implements the frozen protocol (MASTER_PLAN §7.2-7.4):

* **Model** vocals + accompaniment SI-SDR per track (full-track, chunked
  overlap-add inference, mixture-phase iSTFT), plus SI-SDRi.
* **Do-nothing floor** — mixture as "vocals".
* **Oracle IRM** :math:`M = |S|/(|S|+|A|+\varepsilon)` and **oracle IBM**
  (:math:`1` iff :math:`|S|>|A|`) — the mask-family upper bounds at this STFT
  resolution. These lines appear on every results figure (headroom context).
* **museval BSS-Eval SDR** — the clearly-labelled *secondary* table.

``evaluate`` and the module CLI are **RUN LATER** (they need decoded shards and a
trained checkpoint). The oracle/do-nothing mask arithmetic and full-track SI-SDR
are pure and unit-tested on synthetic signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from ..audio.stft import STFT
from ..metrics.si_sdr import si_sdr, si_sdr_improvement
from ..models import build_model_from_config
from .overlap_add import separate_track

EPS = 1e-8


# --- oracle / floor masks (pure, unit-tested) -------------------------------

def oracle_irm(voc_mag: Tensor, acc_mag: Tensor, eps: float = EPS) -> Tensor:
    r"""Ideal ratio mask ``|S| / (|S| + |A| + eps)`` in ``[0, 1)``."""
    return voc_mag / (voc_mag + acc_mag + eps)


def oracle_ibm(voc_mag: Tensor, acc_mag: Tensor) -> Tensor:
    r"""Ideal binary mask: ``1`` where ``|S| > |A|`` else ``0``."""
    return (voc_mag > acc_mag).to(voc_mag.dtype)


def _oracle_vocals(mix_wave: Tensor, voc_wave: Tensor, acc_wave: Tensor, stft: STFT, kind: str) -> Tensor:
    """Analytic full-track oracle vocals estimate (no chunking needed)."""
    length = mix_wave.shape[-1]
    mix_spec = stft.transform(mix_wave)
    voc_mag = stft.transform(voc_wave).abs()
    acc_mag = stft.transform(acc_wave).abs()
    mask = oracle_irm(voc_mag, acc_mag) if kind == "irm" else oracle_ibm(voc_mag, acc_mag)
    return stft.inverse(mask.to(mix_spec.dtype) * mix_spec, length=length)


# --- per-track scoring ------------------------------------------------------

@dataclass
class TrackScores:
    """SI-SDR scores for one system on one track."""

    track: str
    system: str
    sisdr_vocals: float
    sisdr_accomp: float
    sisdr_i_vocals: float


def score_system(
    track: str,
    system: str,
    voc_estimate: np.ndarray,
    acc_estimate: np.ndarray,
    vocals: np.ndarray,
    accompaniment: np.ndarray,
    mixture: np.ndarray,
) -> TrackScores:
    """Compute vocals/accompaniment SI-SDR (+ vocals SI-SDRi) for one system."""
    return TrackScores(
        track=track,
        system=system,
        sisdr_vocals=si_sdr(voc_estimate, vocals),
        sisdr_accomp=si_sdr(acc_estimate, accompaniment),
        sisdr_i_vocals=si_sdr_improvement(voc_estimate, vocals, mixture),
    )


def validation_sisdr(model, stft: STFT, store, manifest, device: torch.device) -> float:
    """Mean full-track vocals SI-SDR over the validation tracks (RUN LATER).

    The selection metric used every 2000 steps by the train loop. Requires the
    decoded validation shards; on synthetic fixtures it works too but is not part
    of the CPU test budget.
    """
    model.eval()
    scores: list[float] = []
    for track in manifest.tracks_for("valid"):
        if track not in set(store.track_names()):
            continue
        sources = store.load_sources(track)
        vocals = torch.from_numpy(sources["vocals"]).float()
        accompaniment = torch.from_numpy(sources["accompaniment"]).float()
        mixture = vocals + accompaniment
        est = separate_track(model, mixture.to(device), stft, device)
        scores.append(si_sdr(est.cpu().numpy(), vocals.numpy()))
    model.train()
    return float(np.mean(scores)) if scores else float("nan")


def evaluate(
    checkpoint: str | Path | None,
    split: str,
    *,
    shard_root: str | Path,
    splits_csv: str | Path,
    output_dir: str | Path = "results",
    oracles: bool = True,
    museval: bool = False,
    sample_rate: int = 44100,
    device: str = "cpu",
) -> pd.DataFrame:
    """Score a checkpoint (+ floor/oracles) on a split and write the §7.4 CSVs.

    **RUN LATER** — needs decoded shards on disk and, for a model system, a
    trained checkpoint. Writes ``<output_dir>/{split}_per_track.csv`` and, when
    ``museval=True``, ``<output_dir>/{split}_museval.csv`` (secondary).
    """
    from ..data.manifest import Manifest
    from ..data.musdb_dataset import WavShardStore

    shard_root = Path(shard_root)
    if not shard_root.exists():
        raise FileNotFoundError(f"shard_root {shard_root} missing — run prepare_data.py (RUN LATER)")

    manifest = Manifest.from_csv(splits_csv)
    store = WavShardStore(shard_root, sample_rate=sample_rate)
    dev = torch.device(device)
    stft = STFT().to(dev)

    model = None
    if checkpoint is not None:
        from ..train.loop import load_checkpoint

        # Build the architecture the checkpoint was trained with (baseline or a
        # Direction-03 band-split variant) from its saved config, then load weights.
        payload = load_checkpoint(checkpoint, restore_rng=False)
        model = build_model_from_config(payload.get("config") or {}).to(dev)
        model.load_state_dict(payload["model"])
        model.eval()

    rows: list[TrackScores] = []
    museval_rows: list[dict[str, Any]] = []
    for track in manifest.tracks_for(split):
        if track not in set(store.track_names()):
            continue
        sources = store.load_sources(track)
        vocals = sources["vocals"].astype(np.float64)
        accompaniment = sources["accompaniment"].astype(np.float64)
        mixture = vocals + accompaniment

        # do-nothing floor: the mixture itself is the estimate for each target.
        rows.append(score_system(track, "do_nothing", mixture, mixture,
                                 vocals, accompaniment, mixture))
        if oracles:
            for kind in ("irm", "ibm"):
                voc_hat = _oracle_vocals(
                    torch.from_numpy(mixture).float(), torch.from_numpy(vocals).float(),
                    torch.from_numpy(accompaniment).float(), stft.cpu(), kind,
                ).numpy()
                rows.append(score_system(track, f"oracle_{kind}", voc_hat, mixture - voc_hat,
                                         vocals, accompaniment, mixture))
        if model is not None:
            voc_hat = separate_track(model, torch.from_numpy(mixture).float().to(dev), stft, dev).cpu().numpy()
            acc_hat = mixture - voc_hat
            rows.append(score_system(track, "singnet", voc_hat, acc_hat, vocals, accompaniment, mixture))
            if museval:
                from ..metrics.museval_wrap import bss_eval_sdr

                res = bss_eval_sdr(vocals, voc_hat, sample_rate)
                museval_rows.append({"track": track, "system": "singnet", "sdr_median": res["sdr_median"],
                                     "museval_version": res["museval_version"]})

    frame = pd.DataFrame([r.__dict__ for r in rows])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / f"{split}_per_track.csv", index=False)
    if museval and museval_rows:
        pd.DataFrame(museval_rows).to_csv(output_dir / f"{split}_museval.csv", index=False)
    return frame


def _cli(argv: list[str] | None = None) -> None:
    """``python -m singnet.eval`` / ``scripts/evaluate.py`` entry point (RUN LATER).

    Directions 01-03: score a checkpoint (+ floor/oracles) on one split. Direction 05:
    ``--direction 05 --test-matrix`` builds the consolidated PEFT test session CSV
    (:func:`singnet.peft.evaluate_umx.build_test_matrix`) instead.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Score checkpoints + oracles on a split (RUN LATER; needs data).")
    parser.add_argument("--checkpoint", default=None, help="model checkpoint (omit to score only floor/oracles)")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--splits-csv", required=True)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--oracles", action="store_true")
    parser.add_argument("--museval", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--direction", default=None, choices=[None, "01", "02", "03", "05"],
                        help="Direction 05 routes --test-matrix to the PEFT test session")
    parser.add_argument("--test-matrix", action="store_true",
                        help="Direction 05: build the consolidated test-matrix CSV (RUN LATER)")
    parser.add_argument("--registry", default="05-lora-source-separation/results/registry.csv",
                        help="Direction 05 test-matrix: the completed-runs registry")
    args = parser.parse_args(argv)

    if args.direction == "05" and args.test_matrix:
        from ..peft.evaluate_umx import build_test_matrix

        frame = build_test_matrix(
            args.registry, shard_root=args.shard_root, splits_csv=args.splits_csv,
            output_dir=args.output_dir, device=args.device,
        )
        print(frame.groupby(["recipe", "eval_set"])["sisdr_vocals_mean"].mean())
        return

    frame = evaluate(
        args.checkpoint, args.split, shard_root=args.shard_root, splits_csv=args.splits_csv,
        output_dir=args.output_dir, oracles=args.oracles, museval=args.museval, device=args.device,
    )
    print(frame.groupby("system")[["sisdr_vocals", "sisdr_accomp"]].mean())
