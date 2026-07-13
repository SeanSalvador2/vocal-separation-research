"""Oracle/floor masks and full-track scoring (G0, §7.3)."""

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
