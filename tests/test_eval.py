"""Oracle/floor masks and full-track scoring (G0, §7.3; Direction 08 SLR columns §6)."""

from __future__ import annotations

import numpy as np
import torch

from singnet.eval import oracle_ibm, oracle_irm, score_system


def test_oracle_irm_bounds_and_values() -> None:
    voc = torch.tensor([[1.0, 3.0], [0.0, 2.0]])
    acc = torch.tensor([[1.0, 1.0], [4.0, 0.0]])
    irm = oracle_irm(voc, acc)
    assert torch.all((irm >= 0.0) & (irm <= 1.0))
    assert torch.allclose(irm[0, 0], torch.tensor(0.5), atol=1e-6)
    assert torch.allclose(irm[0, 1], torch.tensor(0.75), atol=1e-6)


def test_oracle_ibm_is_binary() -> None:
    voc = torch.tensor([[1.0, 3.0], [0.0, 2.0]])
    acc = torch.tensor([[1.0, 1.0], [4.0, 0.0]])
    ibm = oracle_ibm(voc, acc)
    assert set(ibm.unique().tolist()) <= {0.0, 1.0}
    assert ibm[0, 1] == 1.0  # 3 > 1
    assert ibm[1, 0] == 0.0  # 0 < 4


def test_irm_recovers_vocals_when_no_overlap() -> None:
    """Where accompaniment is silent, the IRM is ~1 and vocals pass through."""
    voc = torch.tensor([[2.0, 5.0]])
    acc = torch.zeros_like(voc)
    irm = oracle_irm(voc, acc)
    assert torch.allclose(irm, torch.ones_like(irm), atol=1e-6)


def test_score_system_structure() -> None:
    rng = np.random.default_rng(0)
    vocals = rng.standard_normal(4000)
    accompaniment = rng.standard_normal(4000)
    mixture = vocals + accompaniment
    scores = score_system("trackA", "do_nothing", mixture, mixture, vocals, accompaniment, mixture)
    assert scores.track == "trackA" and scores.system == "do_nothing"
    # a perfect vocal estimate scores far above the do-nothing floor
    perfect = score_system("trackA", "oracle", vocals, accompaniment, vocals, accompaniment, mixture)
    assert perfect.sisdr_vocals > scores.sisdr_vocals
    # without slr=True the SLR columns stay NaN (backward-compatible default).
    assert np.isnan(scores.slr_m60)


def test_score_system_slr_columns() -> None:
    # A GT vocal that is silent for its second half (≥ L_min), so SLR is defined there.
    sr = 1000
    vocals = np.concatenate([np.full(2 * sr, 0.5), np.zeros(2 * sr)])
    accompaniment = np.full(4 * sr, 0.3)
    mixture = vocals + accompaniment
    # do-nothing (v̂ = x) scores exactly 0 dB SLR at every θ — the readable anchor (§2.2).
    do_nothing = score_system("t", "do_nothing", mixture, mixture, vocals, accompaniment, mixture,
                              sr=sr, slr=True)
    assert do_nothing.slr_m50 == 0.0 and do_nothing.slr_m60 == 0.0 and do_nothing.slr_m70 == 0.0
    assert do_nothing.slr_n_regions_60 == 1.0  # the silent second half
    # a perfect separator (v̂ = 0) leaks nothing -> SLR far below 0 dB.
    perfect = score_system("t", "singnet", np.zeros_like(mixture), mixture, vocals,
                           accompaniment, mixture, sr=sr, slr=True)
    assert perfect.slr_m60 < -30.0


def test_score_system_slr_requires_sr() -> None:
    x = np.ones(1000)
    try:
        score_system("t", "s", x, x, x, x, x, slr=True)  # no sr
        raised = False
    except ValueError:
        raised = True
    assert raised
