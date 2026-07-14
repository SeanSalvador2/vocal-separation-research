r"""Pseudo-labeled FMA shards + chunk-level pool mixing (Direction 10, MASTER_PLAN §3, §4.1).

Two objects, both mirroring established Direction-06/08 patterns:

* :class:`PseudoLabeledShards` — a :class:`singnet.data.musdb_dataset.TrackStore` over the
  teacher-labeled 2-stem FMA shards (30-s clips → 6-s chunks; same shard interface as the
  MUSDB store) that **structurally refuses any path under a MUSDB shard root**. This is the
  leakage guard of §3.3: pseudo-labeled FMA data must never be read from, or nested under, a
  MUSDB decode root (which would risk MUSDB audio entering the pseudo pool). The refusal is a
  dedicated exception raised at construction — the offending store *cannot exist* — mirroring
  Direction 06's :class:`EvalSplitCorruptionError` structural guard, not a runtime warning.

* :class:`MixedPools` — the chunk-level pool mix (§4.1): each training example draws its
  source pool (Bernoulli ``p_fma``) from a **dedicated** ``(seed, "pool", index)`` RNG stream
  (:data:`singnet.data.augment.STREAM_IDS` id 6), then returns that pool's own chunk. Because
  each pool is a *separate* :class:`~singnet.data.musdb_dataset.MusdbChunks` over its own store
  with its own :class:`~singnet.data.augment.AugmentPipeline`, remix stays **within-pool by
  construction** — a remix partner is drawn from that pool's ``tracks`` and can never cross
  pools. The pool draw pulls only from stream 6, so it perturbs neither pool's augmentation/
  sampling streams (the pool-independence guarantee, unit-tested).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..utils.seed import derive_rng
from .augment import STREAM_IDS
from .manifest import Manifest
from .musdb_dataset import DEFAULT_SR, MusdbChunks, _to_mono_f32

#: Name of the dedicated pool-selection RNG stream (id 6 in :data:`STREAM_IDS`).
POOL_STREAM = "pool"
#: Per-track-dir stem files that mark a MUSDB *decode* root (prepare_data.py output). A pseudo
#: root has only the 2-stem {mixture, vocals, accompaniment} layout — never drums/bass/other.
MUSDB_STEM_MARKERS: tuple[str, ...] = ("drums.wav", "bass.wav", "other.wav")


class MusdbShardLeak(ValueError):
    """Raised when a pseudo store is rooted at, under, or inside a MUSDB shard root.

    The §3.3 leakage guard: FMA pseudo-labels must live on a **separate** root from the MUSDB
    shards. This is raised at :class:`PseudoLabeledShards` construction — the offending store
    cannot be built — so the guard is structural, mirroring Direction 06's eval-split guard.
    """


def _looks_like_musdb_shards(root: Path) -> bool:
    """Content sniff: does ``root`` hold a MUSDB *decode* (4-stem / ``index.json`` w/ subsets)?

    True iff the root carries an ``index.json`` with per-track ``subset`` fields (the
    prepare_data.py decode index) **or** any track dir with the full drums+bass+other stem
    set. Pseudo roots (teacher_label.py output) never do — they are 2-stem only.
    """
    if not root.exists():
        return False
    index = root / "index.json"
    if index.exists():
        try:
            meta = json.loads(index.read_text(encoding="utf-8"))
            if isinstance(meta, dict) and any(
                isinstance(v, dict) and "subset" in v for v in meta.values()
            ):
                return True
        except (json.JSONDecodeError, OSError):
            pass
    for sub in root.iterdir():
        if sub.is_dir() and all((sub / marker).exists() for marker in MUSDB_STEM_MARKERS):
            return True
    return False


def assert_not_under_musdb(root: str | Path, musdb_roots: tuple[str | Path, ...] = ()) -> Path:
    r"""Raise :class:`MusdbShardLeak` if ``root`` is at/under a MUSDB shard root (§3.3).

    Two structural checks, both refusing at construction: (i) **path containment** — ``root``
    equals or is nested under any of ``musdb_roots`` (the MUSDB shard roots the caller knows,
    e.g. the training ``shard_root``); (ii) a **content sniff** (:func:`_looks_like_musdb_shards`)
    that catches a MUSDB decode root even when the caller passed no explicit roots. Returns the
    resolved ``root`` on success.
    """
    resolved = Path(root).resolve()
    for musdb in musdb_roots:
        if musdb is None:
            continue
        m = Path(musdb).resolve()
        if resolved == m or m in resolved.parents or resolved in m.parents:
            raise MusdbShardLeak(
                f"pseudo shard root {resolved} collides with the MUSDB shard root {m} — FMA "
                "pseudo-labels must live on a SEPARATE root (MASTER_PLAN §3.3 leakage guard). "
                "No FMA audio may enter a MUSDB split, and no pseudo data may nest under MUSDB."
            )
    if _looks_like_musdb_shards(resolved):
        raise MusdbShardLeak(
            f"pseudo shard root {resolved} looks like a MUSDB decode root (4-stem layout or an "
            "index.json with subset fields) — refusing to read pseudo-labels from it (§3.3)."
        )
    return resolved


class PseudoLabeledShards:
    """Store over teacher-labeled 2-stem FMA shards; **refuses MUSDB shard roots** (§3.3).

    Layout (teacher_label.py output, RUN LATER): ``<root>/<clip_id>/{mixture,vocals,
    accompaniment}.wav`` — the same per-track shard interface as
    :class:`~singnet.data.musdb_dataset.WavShardStore`, so a
    :class:`~singnet.data.musdb_dataset.MusdbChunks` reads it unchanged (30-s clips → 6-s
    chunks). Every pseudo clip belongs to the ``train`` split (:meth:`build_manifest`);
    pseudo data never enters validation/test (§4.2). Reading WAVs is RUN LATER (soundfile);
    the MUSDB-path guard and the manifest are pure and unit-tested.

    Args:
        root: the pseudo shard root (**must not be a MUSDB shard root**).
        musdb_roots: known MUSDB shard roots to refuse containment under (e.g. the training
            ``shard_root``); the content sniff catches a MUSDB decode root even if empty.
        sample_rate: shard sample rate (44100 pinned).
    """

    def __init__(
        self,
        root: str | Path,
        *,
        musdb_roots: tuple[str | Path, ...] = (),
        sample_rate: int = DEFAULT_SR,
    ) -> None:
        self.root = assert_not_under_musdb(root, musdb_roots)
        self.sample_rate = int(sample_rate)

    def track_names(self) -> list[str]:
        """Clip ids = the sorted shard sub-directories (empty if the root has none yet)."""
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def load_sources(self, clip: str) -> dict[str, np.ndarray]:
        """Read a pseudo clip's ``{vocals, accompaniment}`` mono stems (**RUN LATER**, soundfile)."""
        import soundfile as sf  # local import: only needed with real pseudo shards

        clip_dir = self.root / clip
        vocals, sr = sf.read(clip_dir / "vocals.wav", dtype="float32", always_2d=True)
        if sr != self.sample_rate:
            raise ValueError(f"{clip}: sample rate {sr} != expected {self.sample_rate}")
        acc_path = clip_dir / "accompaniment.wav"
        if acc_path.exists():
            accompaniment, _ = sf.read(acc_path, dtype="float32", always_2d=True)
        else:  # derive from the exact 2-stem consistency: accompaniment = mixture - vocals
            mixture, _ = sf.read(clip_dir / "mixture.wav", dtype="float32", always_2d=True)
            accompaniment = mixture - vocals
        return {"vocals": _to_mono_f32(vocals), "accompaniment": _to_mono_f32(accompaniment)}

    def build_manifest(self) -> Manifest:
        """A ``train``-only :class:`Manifest` over every pseudo clip (pseudo never enters val/test)."""
        names = self.track_names()
        frame = pd.DataFrame({"track": names, "split": ["train"] * len(names)})
        return Manifest(frame)

    @property
    def n_clips(self) -> int:
        return len(self.track_names())


class MixedPools(Dataset):
    r"""Chunk-level MUSDB↔FMA pool mix (MASTER_PLAN §4.1): per-example Bernoulli pool draw.

    Each ``__getitem__(index)`` draws the source pool from a **dedicated** ``(seed, "pool",
    index)`` stream — ``"fma"`` with probability ``p_fma``, else ``"musdb"`` — then returns
    that pool's own ``[index]`` chunk. The two pools are distinct
    :class:`~singnet.data.musdb_dataset.MusdbChunks`, each remixing **within its own store**,
    so a remix partner never crosses pools (the within-pool guarantee, by construction). The
    pool draw consumes only the id-6 stream, so it perturbs neither pool's augmentation/
    sampling streams (the pool-independence guarantee).

    Args:
        musdb_ds: the MUSDB pool (a :class:`MusdbChunks` over the MUSDB store).
        pseudo_ds: the FMA pseudo pool (a :class:`MusdbChunks` over :class:`PseudoLabeledShards`).
        p_fma: Bernoulli probability of drawing the FMA pool per example (``∈ [0, 1]``).
        seed: run seed keying the dedicated pool stream.
        length: number of examples the map exposes (``steps * batch``); defaults to the MUSDB
            pool's length.
    """

    def __init__(
        self,
        musdb_ds: MusdbChunks,
        pseudo_ds: MusdbChunks,
        p_fma: float,
        seed: int,
        *,
        length: int | None = None,
    ) -> None:
        p_fma = float(p_fma)
        if not (0.0 <= p_fma <= 1.0):
            raise ValueError(f"p_fma must be in [0, 1]; got {p_fma!r}")
        self.musdb_ds = musdb_ds
        self.pseudo_ds = pseudo_ds
        self.p_fma = p_fma
        self.seed = int(seed)
        self._length = int(length) if length is not None else len(musdb_ds)

    def __len__(self) -> int:
        return self._length

    def _pool_rng(self, index: int) -> np.random.Generator:
        """The dedicated pool-selection generator for ``index`` (id-6 stream, nothing else)."""
        return derive_rng(self.seed, STREAM_IDS[POOL_STREAM], int(index))

    def pool_of(self, index: int) -> str:
        """``"fma"`` (prob ``p_fma``) or ``"musdb"`` for example ``index`` — a pure stream draw."""
        return "fma" if float(self._pool_rng(index).random()) < self.p_fma else "musdb"

    def pool_dataset(self, index: int) -> MusdbChunks:
        """The pool :class:`MusdbChunks` example ``index`` routes to (by construction)."""
        return self.pseudo_ds if self.pool_of(index) == "fma" else self.musdb_ds

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        pool = self.pool_of(index)
        # The selected pool draws its OWN chunk at `index` from its OWN streams (remix stays
        # within that pool's tracks); the unused pool is never touched. Indexing the pool with
        # the SAME `index` keeps its stream coordinate intact -> pool draw is independent of it.
        source = self.musdb_ds if pool == "musdb" else self.pseudo_ds
        item = dict(source[index])
        item["pool_is_fma"] = torch.tensor(1.0 if pool == "fma" else 0.0, dtype=torch.float32)
        return item

    def pool_counts(self, n: int | None = None) -> dict[str, int]:
        """Deterministic ``{"fma", "musdb"}`` counts over the first ``n`` (or all) examples."""
        n = self._length if n is None else int(n)
        fma = sum(self.pool_of(i) == "fma" for i in range(n))
        return {"fma": int(fma), "musdb": int(n - fma)}


def build_pseudo_dataset(
    config: dict,
    *,
    seed: int,
    chunk_s: float = 6.0,
    length: int | None = None,
    batch_size: int = 16,
    musdb_shard_root: str | Path | None = None,
) -> MusdbChunks:
    """Build the FMA pseudo pool :class:`MusdbChunks` from a config (**RUN LATER**, needs shards).

    Reads ``pseudo_root``/``pseudo_manifest`` (non-identity locations), constructs the
    MUSDB-refusing :class:`PseudoLabeledShards` store (passing the MUSDB ``shard_root`` as a
    forbidden root), and wraps it in a :class:`MusdbChunks` with **its own** augmentation
    pipeline sharing the run's ``(seed, name, step)`` streams — so remix stays within the FMA
    pool. Fails loud if ``pseudo_root`` is unset (teacher labeling has not run).
    """
    from .augment import AugmentPipeline
    from ..utils.config import augment_switches, pseudo_paths

    root, _manifest_csv = pseudo_paths(config)
    if not root:
        raise FileNotFoundError(
            "pseudo_root missing — run scripts/prepare_fma.py + scripts/teacher_label.py first "
            "(RUN LATER; MASTER_PLAN §6 run book). The pseudo pool needs teacher-labeled shards."
        )
    forbidden = (musdb_shard_root,) if musdb_shard_root else ()
    store = PseudoLabeledShards(
        root, musdb_roots=forbidden, sample_rate=int(config.get("sample_rate", DEFAULT_SR))
    )
    manifest = store.build_manifest()
    switches = augment_switches(config)
    pipeline = AugmentPipeline(remix=switches["remix"], gain=switches["gain"], flip=switches["flip"])
    return MusdbChunks(
        store, manifest, "train", seed=int(seed), chunk_s=chunk_s,
        pipeline=pipeline, length=length, batch_size=batch_size,
    )


def read_teacher_provenance(root: str | Path) -> dict[str, object]:
    """Read ``<root>/provenance.json`` (teacher version/model/consistency), or ``{}`` if absent.

    RUN LATER helper for the registry fields (``teacher_version``,
    ``teacher_consistency_db``); returns an empty dict pre-labeling so the loop records the
    identity defaults.
    """
    path = Path(root) / "provenance.json"
    if not path.exists():
        return {}
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return {}
