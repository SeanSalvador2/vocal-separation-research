"""The five training losses and the ``build`` registry (MASTER_PLAN §6)."""

from __future__ import annotations

from ._base import SeparationLoss, fp32_autocast_disabled, masked_magnitude
from .l1mag import L1MagLoss
from .logl1mag import LogL1MagLoss
from .mrstft import L1MrStftLoss, MultiResolutionSTFTLoss
from .msemag import MseMagLoss
from .registry import ARM_NAMES, build
from .sisdr import SILENCE_RMS_THRESHOLD, SiSdrLoss

__all__ = [
    "SeparationLoss",
    "fp32_autocast_disabled",
    "masked_magnitude",
    "L1MagLoss",
    "MseMagLoss",
    "LogL1MagLoss",
    "SiSdrLoss",
    "L1MrStftLoss",
    "MultiResolutionSTFTLoss",
    "SILENCE_RMS_THRESHOLD",
    "ARM_NAMES",
    "build",
]
