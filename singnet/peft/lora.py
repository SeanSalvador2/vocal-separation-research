r"""LoRA adapters for linear and (Bi)LSTM weights (Direction 05 MASTER_PLAN §3.1, THEORY §2-3).

Low-Rank Adaptation (Hu et al., arXiv 2106.09685) freezes a pretrained weight
:math:`W_0\in\mathbb R^{d\times k}` and learns a rank-:math:`r` additive update

.. math::

    h = W_0 x + \Delta W\,x = W_0 x + \tfrac{\alpha}{r}\,B A\,x,\qquad
    B\in\mathbb R^{d\times r},\ A\in\mathbb R^{r\times k},

with :math:`A\sim\mathcal N(0,\sigma^2)` and **:math:`B=0`** so that :math:`\Delta W=0`
at initialization — the adapted module reproduces the frozen one **exactly** (the
B=0-identity invariant, unit-tested for both a ``Linear`` and a wrapped ``LSTM``).
Trainable parameters per wrapped matrix are :math:`r(d+k)` versus the full
:math:`dk` (THEORY §2). After training, :func:`merge_lora` bakes
:math:`W=W_0+\tfrac{\alpha}{r}BA` back into a plain module (zero inference latency;
merged forward == wrapped forward, round-trip-tested).

Two wrapping mechanisms, one math:

* :class:`LoRALinear` **replaces** an ``nn.Linear`` with a module that holds the
  frozen base plus ``lora_A``/``lora_B`` (used for UMX's ``fc1``/``fc2``/``fc3``).
* :func:`wrap_lstm_lora` uses ``torch.nn.utils.parametrize`` on the gate-stacked
  ``weight_ih_l{k}``/``weight_hh_l{k}`` (+ ``_reverse``) matrices of an ``nn.LSTM``
  (THEORY §3 derives why a rank-:math:`r` update on the stacked :math:`4h\times d`
  matrix is valid). A forward pre-hook refreshes the LSTM's cached ``_flat_weights``
  so the parametrized weights flow into every forward and gradients reach A/B; this
  is the piece that de-optimizes cuDNN's fused kernel (MASTER_PLAN §11 — acceptable
  at 6 k steps, wall-clock recorded per run). On CPU (the whole test suite) the
  non-fused path is used and the B=0 identity is bit-exact.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from torch import Tensor, nn

#: LoRA weight-matrix name prefixes inside an ``nn.LSTM`` (both directions, all
#: layers). ``weight_ih_l{k}`` is the gate-stacked input projection (4h x d_in),
#: ``weight_hh_l{k}`` the recurrent one (4h x h); ``_reverse`` is the backward
#: direction of a bidirectional layer (THEORY §3.4).
LSTM_LORA_PREFIXES: tuple[str, str] = ("weight_ih_l", "weight_hh_l")


def _freeze(params: Iterable[nn.Parameter]) -> None:
    for p in params:
        p.requires_grad_(False)


# --- the parametrization (used for the LSTM; also the merge math) -----------

class LoRAParametrization(nn.Module):
    r"""A ``torch.nn.utils.parametrize`` module implementing ``W -> W + (α/r)·BA``.

    Registered on a weight tensor of shape ``(d_out, d_in)``; holds ``lora_A``
    (``r x d_in``, init :math:`\mathcal N(0,\sigma^2)`) and ``lora_B``
    (``d_out x r``, init ``0``). :meth:`forward` returns the effective weight and
    :meth:`right_inverse` maps a plain weight back to itself, so the frozen base
    (parametrization ``.original``) is preserved unchanged.
    """

    def __init__(self, d_out: int, d_in: int, r: int, alpha: float, sigma: float = 0.02) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {r}")
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(r)
        self.lora_A = nn.Parameter(torch.empty(r, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))
        nn.init.normal_(self.lora_A, mean=0.0, std=sigma)

    @property
    def delta(self) -> Tensor:
        """The current low-rank update ``(α/r)·B A`` (``d_out x d_in``)."""
        return self.scaling * (self.lora_B @ self.lora_A)

    def forward(self, weight: Tensor) -> Tensor:
        return weight + self.scaling * (self.lora_B @ self.lora_A)

    def right_inverse(self, weight: Tensor) -> Tensor:
        # The stored "original" is the plain base weight; B=0 => forward is identity.
        return weight


# --- the Linear wrapper -----------------------------------------------------

class LoRALinear(nn.Module):
    r"""A frozen ``nn.Linear`` plus a trainable rank-:math:`r` LoRA update.

    ``LoRALinear(base, r, alpha)`` computes ``h = W0 x + (α/r)·B A x`` with the base
    weight/bias frozen. With ``B=0`` at init the forward equals ``base(x)`` exactly
    (unit-tested). :meth:`merge` returns a plain ``nn.Linear`` whose weight is
    ``W0 + (α/r)·B A`` (round-trip-tested against the wrapped forward).
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float, sigma: float = 0.02) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects an nn.Linear base, got {type(base).__name__}")
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {r}")
        self.base = base
        _freeze(self.base.parameters())
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(r)
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.normal_(self.lora_A, mean=0.0, std=sigma)

    @property
    def delta_weight(self) -> Tensor:
        """The low-rank weight update ``(α/r)·B A`` (``out_features x in_features``)."""
        return self.scaling * (self.lora_B @ self.lora_A)

    def forward(self, x: Tensor) -> Tensor:
        # (α/r)·B A x computed as two small matmuls: x·Aᵀ then ·Bᵀ (never forms d×k).
        lora = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return self.base(x) + self.scaling * lora

    def merge(self) -> nn.Linear:
        """Return a plain ``nn.Linear`` with ``W = W0 + (α/r)·B A`` (bias copied)."""
        merged = nn.Linear(self.in_features, self.out_features, bias=self.base.bias is not None)
        with torch.no_grad():
            merged.weight.copy_(self.base.weight + self.delta_weight)
            if self.base.bias is not None:
                merged.bias.copy_(self.base.bias)
        return merged.to(self.base.weight.device, self.base.weight.dtype)


# --- LSTM wrapping via parametrization --------------------------------------

def _lstm_lora_weight_names(lstm: nn.LSTM) -> list[str]:
    """The gate-stacked weight names LoRA wraps (ih/hh, both directions, all layers)."""
    return [
        name
        for name in lstm._flat_weights_names  # type: ignore[attr-defined]
        if name.startswith(LSTM_LORA_PREFIXES)
    ]


def _refresh_flat_weights(module: nn.LSTM, _inputs) -> None:
    """Forward pre-hook: re-pull ``_flat_weights`` from the (parametrized) attributes.

    ``nn.LSTM`` caches its weights in ``_flat_weights`` at construction; after a
    parametrization is registered the attribute ``weight_ih_l0`` becomes a computed
    tensor ``W0 + (α/r)·B A``. Re-reading it here before every forward makes the
    LoRA update flow through the (CPU/non-fused) LSTM kernel and keeps A/B in the
    autograd graph (de-risked: B=0 gives a bit-exact identity, B≠0 reaches A and B).
    """
    module._flat_weights = [  # type: ignore[attr-defined]
        getattr(module, name) if hasattr(module, name) else None
        for name in module._flat_weights_names  # type: ignore[attr-defined]
    ]


def wrap_lstm_lora(
    lstm: nn.LSTM, r: int, alpha: float, sigma: float = 0.02
) -> nn.LSTM:
    r"""Attach LoRA to every ``weight_ih_l{k}``/``weight_hh_l{k}`` (+ ``_reverse``).

    Wraps the gate-stacked :math:`4h\times d` matrices of a (bi)LSTM in place via
    ``torch.nn.utils.parametrize`` (THEORY §3: a rank-:math:`r` update on the stacked
    matrix is a shared low-rank structure across the i/f/g/o gates). The frozen base
    weights (the parametrization ``.original`` tensors) receive no gradient; only the
    per-matrix ``lora_A``/``lora_B`` train. Idempotent-safe: already-parametrized
    weights are skipped. Returns the same module.
    """
    if r <= 0:
        raise ValueError(f"LoRA rank must be positive, got {r}")
    names = _lstm_lora_weight_names(lstm)
    for name in names:
        if parametrize.is_parametrized(lstm, name):
            continue
        weight = getattr(lstm, name)
        d_out, d_in = weight.shape
        parametrize.register_parametrization(
            lstm, name, LoRAParametrization(d_out, d_in, r=r, alpha=alpha, sigma=sigma)
        )
    # Freeze the stored base weights (parametrization keeps them as `.original`).
    for pname, p in lstm.named_parameters():
        if pname.endswith(".original"):
            p.requires_grad_(False)
    # Make the parametrized weights flow into every forward (see _refresh_flat_weights).
    if not getattr(lstm, "_lora_refresh_handle", None):
        handle = lstm.register_forward_pre_hook(_refresh_flat_weights)
        lstm._lora_refresh_handle = handle  # type: ignore[attr-defined]
    _refresh_flat_weights(lstm, None)
    return lstm


# --- merge-back -------------------------------------------------------------

def merge_lora(module: nn.Module) -> nn.Module:
    """Bake every LoRA update into plain weights, in place; return the module.

    * each :class:`LoRALinear` child is replaced by ``child.merge()`` (an
      ``nn.Linear`` with ``W = W0 + (α/r)·B A``);
    * each parametrized ``nn.LSTM`` has its LoRA parametrizations removed with
      ``leave_parametrized=True`` (the effective weight is written back into a plain
      ``Parameter``), the refresh hook detached, and ``_flat_weights`` rebuilt.

    The merged module has no LoRA parameters and its forward equals the wrapped
    forward (round-trip-tested for both a ``Linear`` and an ``LSTM``).
    """
    # Replace LoRALinear children (do this bottom-up over named children).
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            setattr(module, name, child.merge())
        else:
            merge_lora(child)

    # Un-parametrize any LSTM (or RNN base) carrying LoRA parametrizations.
    if isinstance(module, nn.LSTM) and parametrize.is_parametrized(module):
        for wname in list(module.parametrizations.keys()):  # type: ignore[attr-defined]
            parametrize.remove_parametrizations(module, wname, leave_parametrized=True)
        handle = getattr(module, "_lora_refresh_handle", None)
        if handle is not None:
            handle.remove()
            module._lora_refresh_handle = None  # type: ignore[attr-defined]
        module._flat_weights = [  # type: ignore[attr-defined]
            getattr(module, n) if hasattr(module, n) else None
            for n in module._flat_weights_names  # type: ignore[attr-defined]
        ]
        try:  # rebuild the cuDNN-flat buffer where possible (no-op/guarded on CPU)
            module.flatten_parameters()
        except Exception:  # noqa: BLE001 - flatten is a best-effort optimization
            pass
    return module


# --- reporting --------------------------------------------------------------

def count_total_params(model: nn.Module) -> int:
    """Total number of parameters (trainable + frozen)."""
    return sum(p.numel() for p in model.parameters())


def count_trainable_params(model: nn.Module) -> int:
    """Number of parameters with ``requires_grad=True``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def trainable_share(model: nn.Module, denom: int | None = None) -> float:
    """Trainable fraction. ``denom`` defaults to the model's own total params.

    Pass the **host (base) parameter count** as ``denom`` to get the "share of the
    host's parameters" that H-05a is stated against (MASTER_PLAN §2, §3.1) — for a
    LoRA model that denominator excludes the added A/B params.
    """
    total = denom if denom is not None else count_total_params(model)
    if total == 0:
        return float("nan")
    return count_trainable_params(model) / float(total)


def trainable_report(model: nn.Module) -> pd.DataFrame:
    """Per-parameter breakdown: ``name, shape, n_params, trainable, share``.

    ``share`` is each tensor's fraction of the model's **own** total parameters (the
    column sums to 1.0). Frozen LoRA bases appear with ``trainable=False``; the
    ``lora_A``/``lora_B`` rows carry the trained capacity.
    """
    total = count_total_params(model) or 1
    rows = [
        {
            "name": name,
            "shape": tuple(p.shape),
            "n_params": int(p.numel()),
            "trainable": bool(p.requires_grad),
            "share": p.numel() / float(total),
        }
        for name, p in model.named_parameters()
    ]
    return pd.DataFrame(rows, columns=["name", "shape", "n_params", "trainable", "share"])


def lora_param_cost(d_out: int, d_in: int, r: int) -> int:
    """Trainable-parameter cost of a rank-:math:`r` LoRA wrap of a ``d_out x d_in``
    matrix: :math:`r(d_{out}+d_{in})` (THEORY §2)."""
    return int(r) * (int(d_out) + int(d_in))


# convenience: keep math import used by docs/consumers tidy
__all__ = [
    "LoRAParametrization",
    "LoRALinear",
    "wrap_lstm_lora",
    "merge_lora",
    "trainable_report",
    "trainable_share",
    "count_total_params",
    "count_trainable_params",
    "lora_param_cost",
    "LSTM_LORA_PREFIXES",
]
