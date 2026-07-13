r"""The Open-Unmix host: a shape-faithful mock + the five-recipe freezing/wrapping.

Direction 05 adapts the pretrained **umxhq vocals** separator; every CPU test runs
on a **mock** built from the verified code-default shapes (no weight download,
MASTER_PLAN §4). :class:`MockOpenUnmix` mirrors ``openunmix.model.OpenUnmix`` exactly
(``../../00-shared-research/papers/openunmix2019.md`` §2), so parameter counts, the
recipe attachment points, and the trainable-share table are the *real* numbers:

* input crop to ``nb_bins = 1487`` (umxhq's 16 kHz bandwidth), standardized by the
  learnable ``input_mean``/``input_scale`` (per cropped bin);
* ``fc1`` ``Linear(2·1487 -> 512, bias=False)`` -> ``bn1`` -> ``tanh``;
* ``lstm`` ``LSTM(512, 256, 3 layers, bidirectional, dropout 0.4)`` (output 512);
* skip-concat of the ``fc1`` and ``lstm`` outputs -> 1024;
* ``fc2`` ``Linear(1024 -> 512, bias=False)`` -> ``bn2`` -> ``relu``;
* ``fc3`` ``Linear(512 -> 2·2049, bias=False)`` -> ``bn3``;
* learnable ``output_scale``/``output_mean`` (per full output bin), ``relu``, then
  multiply the (uncropped) input magnitude — a non-negative mask (``nb_output_bins
  = 2049``).

**Total base parameters = 8,893,348** (pinned; recomputed against the real
checkpoint at G1). :func:`apply_recipe` implements the §3.1 recipe table exactly;
:func:`recipe_trainable_share` returns the trained fraction of the host's params.

.. note::
   The head-only share is **≈ 23.7 %**, not the MASTER_PLAN §3.1 estimate of
   "≈ 18 %". That estimate assumed a symmetric spectrum (``fc3`` output ``2·nb_bins``);
   the **verified** shapes have ``fc3`` output ``2·nb_output_bins`` (2049 > 1487), so
   the head is larger. The LoRA shares (r=4 → 1.27 %, r=16 → 4.85 % of the host) are
   robust to that approximation and match the plan within ±0.3 pp. See
   ``results/DEVIATIONS.md``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .lora import (
    LoRALinear,
    count_total_params,
    lora_param_cost,
    trainable_share,
    wrap_lstm_lora,
)

# --- verified umxhq code-default shapes (openunmix2019.md §2) ----------------
NB_BINS = 1487            #: bandwidth-cropped input bins (umxhq, 16 kHz @ n_fft=4096)
NB_OUTPUT_BINS = 2049     #: full output bins (n_fft//2 + 1)
NB_CHANNELS = 2           #: stereo host
HIDDEN_SIZE = 512         #: fc1 output / lstm input
LSTM_HIDDEN = HIDDEN_SIZE // 2  #: 256 (bidirectional => 512 out)
NB_LAYERS = 3
LSTM_DROPOUT = 0.4
N_FFT = 4096
SAMPLE_RATE = 44100
BANDWIDTH_HZ = 16000

#: Pinned host parameter count (recomputed against the checkpoint at G1).
BASE_PARAM_COUNT = 8_893_348

#: The five recipes (MASTER_PLAN §3.1), and the LoRA matrices lora* wraps.
RECIPES: tuple[str, ...] = ("zeroshot", "head", "lora4", "lora16", "full")
#: The head recipe's trained modules/tensors (§3.1).
HEAD_MODULES: tuple[str, ...] = ("fc3", "bn3", "output_scale", "output_mean")
#: The scale/mean tensors trained *in addition to* the LoRA A/B (§3.1 lora rows).
LORA_TRAINED_SCALARS: tuple[str, ...] = (
    "input_mean", "input_scale", "output_scale", "output_mean",
)
#: The Linear submodules LoRA wraps in the lora recipes (§3.1).
LORA_LINEAR_MODULES: tuple[str, ...] = ("fc1", "fc2", "fc3")


def recipe_rank(recipe: str) -> int | None:
    """The LoRA rank a recipe implies (``lora4`` -> 4, ``lora16`` -> 16, else None)."""
    if recipe == "lora4":
        return 4
    if recipe == "lora16":
        return 16
    return None


class MockOpenUnmix(nn.Module):
    """A random-weight, shape-faithful stand-in for ``openunmix.model.OpenUnmix``.

    Same submodules, shapes, and forward as the real umxhq vocals model, so the
    parameter accounting and recipe wrapping are the genuine numbers. Built with
    random weights (no download); every unit test uses it. The forward consumes a
    magnitude spectrogram ``(nb_samples, nb_channels, nb_output_bins, nb_frames)``
    and returns a non-negative magnitude estimate of the same shape.
    """

    def __init__(
        self,
        nb_bins: int = NB_BINS,
        nb_output_bins: int = NB_OUTPUT_BINS,
        nb_channels: int = NB_CHANNELS,
        hidden_size: int = HIDDEN_SIZE,
        nb_layers: int = NB_LAYERS,
    ) -> None:
        super().__init__()
        self.nb_bins = int(nb_bins)
        self.nb_output_bins = int(nb_output_bins)
        self.nb_channels = int(nb_channels)
        self.hidden_size = int(hidden_size)

        self.fc1 = nn.Linear(self.nb_bins * self.nb_channels, hidden_size, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=nb_layers,
            bidirectional=True,
            batch_first=False,
            dropout=LSTM_DROPOUT if nb_layers > 1 else 0.0,
        )
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, self.nb_output_bins * self.nb_channels, bias=False)
        self.bn3 = nn.BatchNorm1d(self.nb_output_bins * self.nb_channels)

        # Learnable standardization (per cropped input bin) and output affine
        # (per full output bin) — Parameters, matching the real model.
        self.input_mean = nn.Parameter(torch.zeros(self.nb_bins))
        self.input_scale = nn.Parameter(torch.ones(self.nb_bins))
        self.output_scale = nn.Parameter(torch.ones(self.nb_output_bins))
        self.output_mean = nn.Parameter(torch.ones(self.nb_output_bins))

        # Random-but-reasonable init so a mock forward produces finite values.
        for lin in (self.fc1, self.fc2, self.fc3):
            nn.init.xavier_uniform_(lin.weight)

    def forward(self, x: Tensor) -> Tensor:
        """``(nb_samples, nb_channels, nb_output_bins, nb_frames)`` -> same shape."""
        mix = x.detach().clone()
        # (samples, channels, bins, frames) -> (frames, samples, channels, bins)
        x = x.permute(3, 0, 1, 2)
        nb_frames, nb_samples, nb_channels, _ = x.shape
        x = x[..., : self.nb_bins]                     # crop to the bandwidth
        x = x + self.input_mean
        x = x * self.input_scale
        x = self.fc1(x.reshape(-1, nb_channels * self.nb_bins))
        x = self.bn1(x)
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        x = torch.tanh(x)
        lstm_out = self.lstm(x)[0]
        x = torch.cat([x, lstm_out], dim=-1)            # skip-concat -> 1024
        x = self.fc2(x.reshape(-1, x.shape[-1]))
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)
        x = x * self.output_scale + self.output_mean
        x = F.relu(x)                                   # non-negative mask
        x = x.permute(1, 2, 3, 0)                        # -> (samples, ch, bins, frames)
        return x * mix                                   # apply to the input magnitude


def load_umxhq(device: str | torch.device = "cpu", mock: bool = False) -> nn.Module:
    """Return the umxhq vocals ``OpenUnmix`` on ``device``.

    ``mock=True`` builds :class:`MockOpenUnmix` (random weights, no network) — this
    is what every unit test uses. ``mock=False`` is **RUN LATER (G1)**: it loads the
    published MIT-licensed weights via ``torch.hub`` (a few network MB) and extracts
    the ``vocals`` target. The hub import is lazy and lives inside this branch, so it
    is never touched by the CPU suite.
    """
    device = torch.device(device)
    if mock:
        return MockOpenUnmix().to(device)

    # --- RUN LATER: real checkpoint (never executed in tests) ---------------
    from torch import hub  # local, lazy: only when actually downloading weights

    bundle = hub.load(  # pragma: no cover - network + GPU, RUN LATER
        "sigsep/open-unmix-pytorch", "umxhq", targets=["vocals"], device=str(device), pretrained=True
    )
    # `umxhq(...)` returns a Separator; the per-target OpenUnmix lives in its
    # `target_models` dict. Fall back to the object itself if already a bare model.
    target_models = getattr(bundle, "target_models", None)  # pragma: no cover
    if isinstance(target_models, dict) and "vocals" in target_models:  # pragma: no cover
        return target_models["vocals"].to(device)
    return bundle.to(device)  # pragma: no cover


# --- recipe application (freezing + LoRA wrapping) --------------------------

def _set_all_requires_grad(model: nn.Module, flag: bool) -> None:
    for p in model.parameters():
        p.requires_grad_(flag)


def apply_recipe(model: nn.Module, recipe: str, r: int | None = None) -> nn.Module:
    r"""Freeze/unfreeze (and LoRA-wrap) ``model`` for a recipe (MASTER_PLAN §3.1).

    In place; returns the (possibly re-wrapped) model.

    * ``zeroshot`` — freeze everything (no training; the B=0 LoRA start is this too).
    * ``head`` — train ``fc3``, ``bn3``, ``output_scale``, ``output_mean``; freeze rest.
    * ``lora4`` / ``lora16`` — freeze all base weights; wrap ``fc1``/``fc2``/``fc3``
      with :class:`LoRALinear` and every LSTM ``weight_ih``/``weight_hh`` (both
      directions, 3 layers) with :func:`wrap_lstm_lora`; additionally train
      ``input_mean``/``input_scale``/``output_scale``/``output_mean``. ``r`` defaults
      to the rank implied by the name (4 or 16).
    * ``full`` — train everything.

    **BatchNorm policy (pinned, §3.1/§11):** for every *trained* recipe the BN
    layers stay in **train mode**, so their running statistics adapt to the shifted
    domain (this is a uniform choice applied to head/lora/full alike; zero-shot never
    updates BN). ``apply_recipe`` sets ``requires_grad`` only — the train/eval mode is
    set by the fine-tune loop, which keeps BN in train mode (see ``finetune_umx.py``).
    """
    if recipe not in RECIPES:
        raise ValueError(f"unknown recipe {recipe!r}; expected one of {RECIPES}")

    if recipe == "zeroshot":
        _set_all_requires_grad(model, False)
        return model

    if recipe == "full":
        _set_all_requires_grad(model, True)
        return model

    if recipe == "head":
        _set_all_requires_grad(model, False)
        for mod_name in HEAD_MODULES:
            obj = getattr(model, mod_name)
            if isinstance(obj, nn.Parameter):
                obj.requires_grad_(True)
            else:
                for p in obj.parameters():
                    p.requires_grad_(True)
        return model

    # lora4 / lora16
    rank = r if r is not None else recipe_rank(recipe)
    if rank is None:
        raise ValueError(f"recipe {recipe!r} needs an explicit rank r")
    alpha = 2 * rank  # α = 2r fixed (no α tuning; §3.1)

    _set_all_requires_grad(model, False)
    for lin_name in LORA_LINEAR_MODULES:
        base = getattr(model, lin_name)
        setattr(model, lin_name, LoRALinear(base, r=rank, alpha=alpha))
    wrap_lstm_lora(model.lstm, r=rank, alpha=alpha)
    for scalar_name in LORA_TRAINED_SCALARS:
        getattr(model, scalar_name).requires_grad_(True)
    return model


def trainable_param_names(model: nn.Module) -> set[str]:
    """The set of ``named_parameters`` currently trainable (requires_grad)."""
    return {name for name, p in model.named_parameters() if p.requires_grad}


def recipe_trainable_share(recipe: str, r: int | None = None) -> float:
    r"""Trained fraction of the **host's** parameters for a recipe (§3.1, closed form).

    Denominator is :data:`BASE_PARAM_COUNT` (the host base, excluding any added LoRA
    A/B) — the quantity H-05a's "< 5 % of the host's parameters" is stated against.
    Returns the exact pinned values: ``head ≈ 0.2373``, ``lora4 ≈ 0.01273``,
    ``lora16 ≈ 0.04852`` (``zeroshot`` 0, ``full`` 1).
    """
    if recipe == "zeroshot":
        return 0.0
    if recipe == "full":
        return 1.0
    if recipe == "head":
        head = (
            NB_OUTPUT_BINS * NB_CHANNELS * HIDDEN_SIZE     # fc3.weight
            + 2 * NB_OUTPUT_BINS * NB_CHANNELS             # bn3 weight+bias
            + NB_OUTPUT_BINS                               # output_scale
            + NB_OUTPUT_BINS                               # output_mean
        )
        return head / BASE_PARAM_COUNT

    rank = r if r is not None else recipe_rank(recipe)
    if rank is None:
        raise ValueError(f"recipe {recipe!r} needs an explicit rank r")
    ab = _lora_ab_cost(rank)
    scalars = 2 * NB_BINS + 2 * NB_OUTPUT_BINS
    return (ab + scalars) / BASE_PARAM_COUNT


def _lora_ab_cost(r: int) -> int:
    """Sum of LoRA A/B parameters over the wrapped matrices at rank ``r`` (§3.1)."""
    cost = 0
    cost += lora_param_cost(HIDDEN_SIZE, NB_BINS * NB_CHANNELS, r)          # fc1
    cost += lora_param_cost(HIDDEN_SIZE, HIDDEN_SIZE * 2, r)                # fc2
    cost += lora_param_cost(NB_OUTPUT_BINS * NB_CHANNELS, HIDDEN_SIZE, r)   # fc3
    for _layer in range(NB_LAYERS):
        for _direction in range(2):
            cost += lora_param_cost(4 * LSTM_HIDDEN, HIDDEN_SIZE, r)        # weight_ih
            cost += lora_param_cost(4 * LSTM_HIDDEN, LSTM_HIDDEN, r)        # weight_hh
    return cost


def measured_trainable_share(model: nn.Module, base_params: int = BASE_PARAM_COUNT) -> float:
    """Trained fraction measured on a live (possibly wrapped) model, host-denominated."""
    return trainable_share(model, denom=base_params)


__all__ = [
    "MockOpenUnmix",
    "load_umxhq",
    "apply_recipe",
    "trainable_param_names",
    "recipe_trainable_share",
    "measured_trainable_share",
    "recipe_rank",
    "count_total_params",
    "RECIPES",
    "HEAD_MODULES",
    "LORA_LINEAR_MODULES",
    "LORA_TRAINED_SCALARS",
    "BASE_PARAM_COUNT",
    "NB_BINS",
    "NB_OUTPUT_BINS",
    "NB_CHANNELS",
    "HIDDEN_SIZE",
    "LSTM_HIDDEN",
    "NB_LAYERS",
    "N_FFT",
    "SAMPLE_RATE",
    "BANDWIDTH_HZ",
]
