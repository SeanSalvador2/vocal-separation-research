"""verify() gates catch real decode bugs and tolerate expected AAC codec noise.

Grounded by the first real-data run (2026-07-16): MUSDB18's five streams are
AAC-coded independently and the mixture stream is peak-limited relative to the raw
stem sum, so pointwise max |mixture − sum| reached 3.0 on healthy shards. The gate
is therefore relative-RMS + correlation (decode-bug signatures), with max-abs
reported as information. Deviation logged in
01-loss-function-study/results/DEVIATIONS.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import prepare_data  # noqa: E402

SR = 44100


def _write_shards(root: Path, mixture: np.ndarray, stems: dict[str, np.ndarray]) -> None:
    track = root / "Synth Artist - Synth Track"
    track.mkdir(parents=True)
    sf.write(track / "mixture.wav", mixture, SR)
    for name, wave in stems.items():
        sf.write(track / f"{name}.wav", wave, SR)


def _stems(rng: np.random.Generator, n: int = SR) -> dict[str, np.ndarray]:
    return {s: 0.2 * rng.standard_normal((n, 2)).astype(np.float32)
            for s in prepare_data.STEMS}


def test_codec_noise_and_peak_limited_transients_pass(tmp_path: Path, capsys) -> None:
    rng = np.random.default_rng(0)
    stems = _stems(rng)
    total = sum(stems.values())
    mixture = total + 0.001 * rng.standard_normal(total.shape).astype(np.float32)
    mixture[1000:1010] = np.clip(total[1000:1010] * 4.0, -1.0, 1.0)  # limited transient
    _write_shards(tmp_path, mixture, stems)
    prepare_data.verify(str(tmp_path), SR)  # must not raise
    out = capsys.readouterr().out
    assert "OK" in out and "informative" in out


def test_misaligned_mixture_fails_on_correlation(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    stems = _stems(rng)
    mixture = np.roll(sum(stems.values()), 1000, axis=0)  # decode misalignment
    _write_shards(tmp_path, mixture, stems)
    with pytest.raises(SystemExit):
        prepare_data.verify(str(tmp_path), SR)


def test_gain_bug_fails_on_relative_rms(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    stems = _stems(rng)
    mixture = 0.5 * sum(stems.values())  # stems decoded at wrong gain
    _write_shards(tmp_path, mixture, stems)
    with pytest.raises(SystemExit):
        prepare_data.verify(str(tmp_path), SR)


def test_shape_mismatch_fails_hard(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    stems = _stems(rng)
    mixture = sum(stems.values())[:-100]  # truncated mixture stream
    _write_shards(tmp_path, mixture, stems)
    with pytest.raises(SystemExit):
        prepare_data.verify(str(tmp_path), SR)
