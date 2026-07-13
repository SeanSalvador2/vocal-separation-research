"""SingNet models: the fixed SingNet-C1 U-Net and the band-split front-end.

:func:`build_model_from_config` dispatches on an **optional** ``model`` config
block so Directions 01/02 configs (no block) build the baseline unchanged and
Direction 03's split configs build :class:`BandSplitUNet`. Absence of the block
is exactly the baseline, so old configs hash unchanged (Direction 03 §3.2).
"""

from __future__ import annotations

from typing import Any

from .bandsplit_unet import (
    BASELINE_PARAM_COUNT,
    MATCHED_BASE_WIDTH,
    MATCHED_BOTTLENECK_WIDTH,
    MATCHED_VARIANT_PARAM_COUNT,
    BandSplitUNet,
    build_bandsplit,
    mel_edges,
    uniform_edges,
)
from .unet import SingNetC1, build_model

__all__ = [
    "SingNetC1",
    "build_model",
    "build_model_from_config",
    "BandSplitUNet",
    "build_bandsplit",
    "mel_edges",
    "uniform_edges",
    "BASELINE_PARAM_COUNT",
    "MATCHED_BASE_WIDTH",
    "MATCHED_BOTTLENECK_WIDTH",
    "MATCHED_VARIANT_PARAM_COUNT",
]


def build_model_from_config(config: dict[str, Any]):
    """Build the model an experiment config asks for (baseline or band-split).

    The ``model`` block is optional; its absence (Directions 01/02, and the
    Direction 03 baseline cell) yields :class:`SingNetC1` at ``base_channels`` —
    so those configs are byte-identical and hash unchanged. A
    ``model: {arch: bandsplit, bands: mel|uniform, base_width, bottleneck_width,
    n_bands}`` block builds the matched :class:`BandSplitUNet`.
    """
    model_cfg = config.get("model") or {}
    arch = str(model_cfg.get("arch", "baseline"))
    if arch == "baseline":
        return build_model(int(config.get("base_channels", 32)))
    if arch == "bandsplit":
        return build_bandsplit(
            bands=str(model_cfg.get("bands", "mel")),
            base_width=int(model_cfg.get("base_width", MATCHED_BASE_WIDTH)),
            bottleneck_width=int(model_cfg.get("bottleneck_width", MATCHED_BOTTLENECK_WIDTH)),
            n_bands=int(model_cfg.get("n_bands", 3)),
        )
    raise ValueError(f"unknown model arch {arch!r}; expected 'baseline' or 'bandsplit'")
