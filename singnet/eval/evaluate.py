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
from ..metrics.slr import silent_regions, slr, slr_report
from ..models import build_model_from_config
from .overlap_add import separate_track

EPS = 1e-8
#: Primary SLR threshold used by the validation hook (MASTER_PLAN §2.1, §6).
VAL_SLR_THETA_DB = -60.0


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
    """SI-SDR scores for one system on one track (+ Direction-08 SLR when requested).

    The three ``slr_m{50,60,70}`` columns (θ = −50/−60/−70 dBFS) and the primary-θ
    region count are ``NaN``/``0`` unless ``score_system(slr=True)`` computed them; a
    track with no ≥ ``L_min`` silent run is ``NaN`` (excluded at aggregation, §2.2).
    """

    track: str
    system: str
    sisdr_vocals: float
    sisdr_accomp: float
    sisdr_i_vocals: float
    # Direction-08 SLR columns (θ = −50/−60/−70 dBFS); NaN unless slr=True was requested.
    slr_m50: float = float("nan")
    slr_m60: float = float("nan")
    slr_m70: float = float("nan")
    slr_n_regions_60: float = float("nan")


def score_system(
    track: str,
    system: str,
    voc_estimate: np.ndarray,
    acc_estimate: np.ndarray,
    vocals: np.ndarray,
    accompaniment: np.ndarray,
    mixture: np.ndarray,
    *,
    sr: int | None = None,
    slr: bool = False,
) -> TrackScores:
    """Compute vocals/accompaniment SI-SDR (+ vocals SI-SDRi) for one system.

    With ``slr=True`` (and ``sr`` given) it also computes SLR at all three θ against the
    GT vocal + mixture — including for the do-nothing (0 dB) and oracle anchors, so the
    (SI-SDR, SLR) plane's anchor points come from the same call path (MASTER_PLAN §3, §6).
    """
    scores = TrackScores(
        track=track,
        system=system,
        sisdr_vocals=si_sdr(voc_estimate, vocals),
        sisdr_accomp=si_sdr(acc_estimate, accompaniment),
        sisdr_i_vocals=si_sdr_improvement(voc_estimate, vocals, mixture),
    )
    if slr:
        if sr is None:
            raise ValueError("score_system(slr=True) requires the sample rate `sr`")
        report = slr_report(voc_estimate, mixture, vocals, sr)
        scores.slr_m50 = report[-50.0]["slr"]
        scores.slr_m60 = report[-60.0]["slr"]
        scores.slr_m70 = report[-70.0]["slr"]
        scores.slr_n_regions_60 = report[-60.0]["n_regions"]
    return scores


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


def validation_report(
    model, stft: STFT, store, manifest, device: torch.device, theta_db: float = VAL_SLR_THETA_DB
) -> dict[str, float]:
    """Mean val vocals SI-SDR **and** SLR (θ) over the validation tracks (RUN LATER).

    One separation pass per track feeds both metrics. SI-SDR is the **selection** metric
    (:func:`is_new_best`); SLR is computed *alongside* purely for logging — checkpoint
    selection never reads it (the §6 selection-bias guard, asserted in tests). Tracks with
    no ≥ ``L_min`` silent run contribute to ``slr_valid_n`` only when valid (museval-style).
    """
    model.eval()
    sr = int(store.sample_rate)
    sisdrs: list[float] = []
    slrs: list[float] = []
    for track in manifest.tracks_for("valid"):
        if track not in set(store.track_names()):
            continue
        sources = store.load_sources(track)
        vocals = sources["vocals"].astype(np.float64)
        mixture = (sources["vocals"] + sources["accompaniment"]).astype(np.float64)
        est = separate_track(
            model, torch.from_numpy(mixture).float().to(device), stft, device
        ).cpu().numpy()
        sisdrs.append(si_sdr(est, vocals))
        regions = silent_regions(vocals, sr, theta_db=theta_db)
        val = slr(est, mixture, regions)
        if not np.isnan(val):
            slrs.append(val)
    model.train()
    return {
        "sisdr": float(np.mean(sisdrs)) if sisdrs else float("nan"),
        "slr": float(np.mean(slrs)) if slrs else float("nan"),
        "slr_valid_n": float(len(slrs)),
    }


def evaluate(
    checkpoint: str | Path | None,
    split: str,
    *,
    shard_root: str | Path,
    splits_csv: str | Path,
    output_dir: str | Path = "results",
    oracles: bool = True,
    museval: bool = False,
    slr: bool = False,
    sample_rate: int = 44100,
    device: str = "cpu",
) -> pd.DataFrame:
    """Score a checkpoint (+ floor/oracles) on a split and write the §7.4 CSVs.

    **RUN LATER** — needs decoded shards on disk and, for a model system, a
    trained checkpoint. Writes ``<output_dir>/{split}_per_track.csv`` and, when
    ``museval=True``, ``<output_dir>/{split}_museval.csv`` (secondary). With
    ``slr=True`` (Direction 08 §6) every system — including the do-nothing and oracle
    anchors — also gets SLR at all three θ written into the per-track CSV.
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
                                 vocals, accompaniment, mixture, sr=sample_rate, slr=slr))
        if oracles:
            for kind in ("irm", "ibm"):
                voc_hat = _oracle_vocals(
                    torch.from_numpy(mixture).float(), torch.from_numpy(vocals).float(),
                    torch.from_numpy(accompaniment).float(), stft.cpu(), kind,
                ).numpy()
                rows.append(score_system(track, f"oracle_{kind}", voc_hat, mixture - voc_hat,
                                         vocals, accompaniment, mixture, sr=sample_rate, slr=slr))
        if model is not None:
            voc_hat = separate_track(model, torch.from_numpy(mixture).float().to(dev), stft, dev).cpu().numpy()
            acc_hat = mixture - voc_hat
            rows.append(score_system(track, "singnet", voc_hat, acc_hat, vocals, accompaniment, mixture,
                                     sr=sample_rate, slr=slr))
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
    parser.add_argument("--slr", action="store_true",
                        help="Direction 08: also score SLR at θ ∈ {−50,−60,−70} for every system")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--direction", default=None, choices=[None, "01", "02", "03", "05", "08", "10"],
                        help="Direction 05 routes --test-matrix to the PEFT test session; "
                             "Direction 10 routes --test-session to the teacher/student session")
    parser.add_argument("--test-matrix", action="store_true",
                        help="Direction 05: build the consolidated test-matrix CSV (RUN LATER)")
    parser.add_argument("--test-session", action="store_true",
                        help="Direction 10: build the consolidated teacher/student test session (RUN LATER)")
    parser.add_argument("--include-teacher", action="store_true",
                        help="Direction 10: score the htdemucs teacher anchor (lazy demucs, RUN LATER)")
    parser.add_argument("--registry", default="05-lora-source-separation/results/registry.csv",
                        help="Direction 05/10 test session: the completed-runs registry")
    args = parser.parse_args(argv)

    if args.direction == "05" and args.test_matrix:
        from ..peft.evaluate_umx import build_test_matrix

        frame = build_test_matrix(
            args.registry, shard_root=args.shard_root, splits_csv=args.splits_csv,
            output_dir=args.output_dir, device=args.device,
        )
        print(frame.groupby(["recipe", "eval_set"])["sisdr_vocals_mean"].mean())
        return

    if args.direction == "10" and args.test_session:
        from .teacher_session import build_teacher_session

        registry = args.registry
        if registry == "05-lora-source-separation/results/registry.csv":  # default -> D10 registry
            registry = "10-demucs-distillation/results/registry.csv"
        frame = build_teacher_session(
            registry, shard_root=args.shard_root, splits_csv=args.splits_csv,
            output_dir=args.output_dir, include_teacher=args.include_teacher, slr=args.slr,
            device=args.device,
        )
        print(frame[["system", "sisdr_vocals_mean", "slr_m60", "is_teacher"]])
        return

    frame = evaluate(
        args.checkpoint, args.split, shard_root=args.shard_root, splits_csv=args.splits_csv,
        output_dir=args.output_dir, oracles=args.oracles, museval=args.museval, slr=args.slr,
        device=args.device,
    )
    cols = ["sisdr_vocals", "sisdr_accomp"] + (["slr_m60"] if args.slr else [])
    print(frame.groupby("system")[cols].mean())
