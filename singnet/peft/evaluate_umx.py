r"""The Direction-05 consolidated test session (MASTER_PLAN §5, §7 G3).

**RUN LATER — needs the decoded stereo shards and every trained checkpoint.** One
pass, one CSV (``results/test_matrix.csv``), no reruns: each recipe checkpoint **plus
zero-shot** (the untuned host) is scored on

* **(a) the domain's transformed 50-track test set** — the *adapted-domain gain*, and
* **(b) the standard 50-track test set** — the *forgetting* (source-domain regression).

Vocals SI-SDR per track via the project's tested metric (UMX-native stereo separation,
mixture-phase iSTFT, **no Wiener** — §1.3); this is the only place the test set is
touched, and only after every decision (LRs, checkpoints) is frozen on validation
(gate G3). The function is a **skeleton that fails loud** when the shards, registry, or
checkpoints are absent — the CPU suite asserts the clean error, never the heavy pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

#: Columns of the consolidated test-matrix CSV (§5).
TEST_MATRIX_COLUMNS: tuple[str, ...] = (
    "run_id", "domain", "recipe", "rank", "seed", "eval_set",
    "sisdr_vocals_mean", "sisdr_vocals_median", "n_tracks", "checkpoint_path",
)

#: The two evaluation sets each system is scored on (§5).
EVAL_SETS: tuple[str, ...] = ("adapted", "standard")


@dataclass
class TestMatrixSpec:
    """Resolved inputs for the consolidated session (all paths, no computation yet)."""

    registry_path: Path
    shard_root: Path
    splits_csv: Path
    output_dir: Path
    device: str = "cpu"


def _require(path: Path, what: str) -> Path:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{what} missing at {path!r} — the Direction-05 test session is RUN LATER "
            "(needs decoded stereo shards + trained checkpoints; see MASTER_PLAN §6 run book)."
        )
    return Path(path)


def build_test_matrix(
    registry_path: str | Path,
    shard_root: str | Path,
    splits_csv: str | Path,
    output_dir: str | Path = "05-lora-source-separation/results",
    *,
    device: str = "cpu",
) -> pd.DataFrame:
    """Score every recipe checkpoint + zero-shot on the adapted + standard test sets.

    **RUN LATER.** Fails loud (``FileNotFoundError``) if the shards, registry, or
    splits are absent — which is exactly the state of the CPU test environment, so the
    unit test exercises the clean-error path. When the inputs exist (RUN LATER) it
    writes ``<output_dir>/test_matrix.csv`` with one row per (system, eval_set).
    """
    spec = TestMatrixSpec(
        registry_path=Path(registry_path), shard_root=Path(shard_root),
        splits_csv=Path(splits_csv), output_dir=Path(output_dir), device=device,
    )
    _require(spec.shard_root, "shard_root (decoded stereo shards)")
    _require(spec.splits_csv, "splits_csv (the frozen 86/14/50 split)")
    _require(spec.registry_path, "registry.csv (the 12 completed runs)")

    # --- RUN LATER: the heavy scoring pass (needs weights + data + GPU) ------
    from ..audio.stft import STFT  # pragma: no cover - RUN LATER
    from ..data.manifest import Manifest  # pragma: no cover
    from ..metrics.si_sdr import si_sdr  # pragma: no cover
    from ..train.loop import load_checkpoint  # pragma: no cover
    from .finetune_umx import StereoWavShardStore, umx_separate  # pragma: no cover
    from .umx_wrapper import load_umxhq  # pragma: no cover

    registry = pd.read_csv(spec.registry_path)  # pragma: no cover
    manifest = Manifest.from_csv(spec.splits_csv)  # pragma: no cover
    stft = STFT()  # pragma: no cover
    rows: list[dict[str, object]] = []  # pragma: no cover

    # zero-shot + each trained recipe row; each scored on adapted (domain-transformed
    # test set) and standard (untransformed) test tracks — the gain/forgetting pair.
    for _idx, run in registry.iterrows():  # pragma: no cover - RUN LATER
        checkpoint = run.get("checkpoint_path")
        model = load_umxhq(device, mock=False)
        if isinstance(checkpoint, str) and checkpoint and Path(checkpoint).exists():
            load_checkpoint(checkpoint, model=model, restore_rng=False)
        for eval_set in EVAL_SETS:
            suffix = run.get("domain_suffix") if eval_set == "adapted" else None
            store = StereoWavShardStore(spec.shard_root, domain_suffix=suffix)
            scores = []
            for track in manifest.tracks_for("test"):
                if track not in set(store.track_names()):
                    continue
                src = store.load_sources_stereo(track)
                import numpy as np

                mix = torch_from(src["vocals"] + src["accompaniment"])
                est = umx_separate(model, mix, stft)
                scores.append(si_sdr(est.cpu().numpy().reshape(-1),
                                     np.asarray(src["vocals"]).reshape(-1)))
            rows.append({
                "run_id": run.get("run_id"), "domain": run.get("domain"),
                "recipe": run.get("recipe"), "rank": run.get("rank"), "seed": run.get("seed"),
                "eval_set": eval_set,
                "sisdr_vocals_mean": float(pd.Series(scores).mean()) if scores else float("nan"),
                "sisdr_vocals_median": float(pd.Series(scores).median()) if scores else float("nan"),
                "n_tracks": len(scores), "checkpoint_path": checkpoint,
            })

    frame = pd.DataFrame(rows, columns=list(TEST_MATRIX_COLUMNS))  # pragma: no cover
    spec.output_dir.mkdir(parents=True, exist_ok=True)  # pragma: no cover
    frame.to_csv(spec.output_dir / "test_matrix.csv", index=False)  # pragma: no cover
    return frame  # pragma: no cover


def torch_from(array):  # pragma: no cover - RUN LATER helper
    import torch

    return torch.from_numpy(__import__("numpy").ascontiguousarray(array)).float()


__all__ = ["build_test_matrix", "TEST_MATRIX_COLUMNS", "EVAL_SETS", "TestMatrixSpec"]
