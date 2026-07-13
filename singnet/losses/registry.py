"""Loss registry — ``build(name, **kw)`` returns the loss module for an arm.

The five arms of Direction 01 (MASTER_PLAN §3.1) map to loss classes here. The
uniform contract (:class:`singnet.losses._base.SeparationLoss`) means the train
loop is *identical* across arms: it always calls ``loss(mask, mix_mag, tgt_mag,
mix_stft, tgt_wave, mix_wave)`` and only the module identity changes.
"""

from __future__ import annotations

from typing import Any, Callable

from ._base import SeparationLoss
from .l1mag import L1MagLoss
from .logl1mag import LogL1MagLoss
from .mrstft import L1MrStftLoss
from .msemag import MseMagLoss
from .sisdr import SiSdrLoss

#: The five arm names, in MASTER_PLAN §3.1 order.
ARM_NAMES: tuple[str, ...] = ("l1mag", "msemag", "logl1mag", "sisdr", "l1mrstft")

_REGISTRY: dict[str, Callable[..., SeparationLoss]] = {
    "l1mag": L1MagLoss,
    "msemag": MseMagLoss,
    "logl1mag": LogL1MagLoss,
    "sisdr": SiSdrLoss,
    "l1mrstft": L1MrStftLoss,
}


def build(name: str, **kwargs: Any) -> SeparationLoss:
    """Construct the loss module for arm ``name``.

    Args:
        name: one of :data:`ARM_NAMES`.
        **kwargs: forwarded to the loss constructor (e.g. ``lam=0.25`` for the
            exploratory ``l1mrstft`` configs, ``eps``/``silence_rms`` for
            ``sisdr``). Magnitude losses accept no extra kwargs.

    Raises:
        KeyError: if ``name`` is not a registered arm.
    """
    if name not in _REGISTRY:
        raise KeyError(f"unknown loss arm {name!r}; expected one of {ARM_NAMES}")
    return _REGISTRY[name](**kwargs)
