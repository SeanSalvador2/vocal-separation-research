"""Split manifest loader and the train-loader-refuses-test-rows guard.

The frozen 86/14/50 split (MASTER_PLAN §4.2) lives in
``01-loss-function-study/configs/splits.csv`` as ``track,split`` rows — **names
only, no audio**. Two guarantees this module enforces:

* training/tuning code may only touch ``train``/``valid`` rows; ``test`` rows are
  refused (grep-auditable discipline of §4.2 / gate G3), and
* the manifest is the single authority on which split a track belongs to.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

#: Splits a training/tuning loader is allowed to read.
TRAINABLE_SPLITS: tuple[str, ...] = ("train", "valid")
TEST_SPLIT = "test"
_REQUIRED_COLUMNS = frozenset({"track", "split"})


class TestRowError(ValueError):
    """Raised when training/tuning code tries to load a ``test`` row."""

    # tell pytest this is not a test class (its name starts with "Test").
    __test__ = False


@dataclass(frozen=True)
class Manifest:
    """A thin, validated wrapper over the ``track,split`` table."""

    frame: pd.DataFrame

    def __post_init__(self) -> None:
        missing = _REQUIRED_COLUMNS - set(self.frame.columns)
        if missing:
            raise ValueError(f"manifest missing columns: {sorted(missing)}")
        unknown = set(self.frame["split"]) - {*TRAINABLE_SPLITS, TEST_SPLIT}
        if unknown:
            raise ValueError(f"manifest has unknown splits: {sorted(unknown)}")

    @classmethod
    def from_csv(cls, path: str | Path) -> "Manifest":
        # ``comment="#"`` skips the provenance header (source URL + placeholder
        # scheme) that splits.csv carries; track names never contain ``#``.
        frame = pd.read_csv(path, dtype={"track": str, "split": str}, comment="#")
        frame["track"] = frame["track"].str.strip()
        frame["split"] = frame["split"].str.strip()
        return cls(frame.reset_index(drop=True))

    def tracks_for(self, split: str) -> list[str]:
        """Track names in a split (raises for unknown split names)."""
        if split not in {*TRAINABLE_SPLITS, TEST_SPLIT}:
            raise ValueError(f"unknown split {split!r}")
        return self.frame.loc[self.frame["split"] == split, "track"].tolist()

    def trainable_tracks(self, split: str) -> list[str]:
        """Track names for a *trainable* split; refuses ``test`` (the guard)."""
        if split == TEST_SPLIT:
            raise TestRowError(
                "training/tuning loader refuses the 'test' split; the 50 test tracks "
                "are read exactly once by the evaluator (MASTER_PLAN §7.4)."
            )
        return self.tracks_for(split)

    def is_test(self, track: str) -> bool:
        rows = self.frame.loc[self.frame["track"] == track, "split"]
        return bool(len(rows)) and bool((rows == TEST_SPLIT).all())

    def assert_no_test_rows(self, tracks: list[str]) -> None:
        """Raise :class:`TestRowError` if any of ``tracks`` is a test track."""
        offending = [t for t in tracks if self.is_test(t)]
        if offending:
            raise TestRowError(f"refusing test tracks in a training context: {offending}")

    @property
    def counts(self) -> dict[str, int]:
        return self.frame["split"].value_counts().to_dict()

    def to_csv(self, path: str | Path) -> None:
        self.frame.to_csv(path, index=False)


def load_manifest(path: str | Path) -> Manifest:
    """Convenience loader mirroring :meth:`Manifest.from_csv`."""
    return Manifest.from_csv(path)
