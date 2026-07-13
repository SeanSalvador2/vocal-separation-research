"""Parameter-efficient fine-tuning for the Open-Unmix host (Direction 05).

* :mod:`singnet.peft.lora` — ``LoRALinear``, ``wrap_lstm_lora``, ``merge_lora``,
  ``trainable_report`` (the adapter math; B=0 identity + merge round-trip tested).
* :mod:`singnet.peft.umx_wrapper` — the shape-faithful ``MockOpenUnmix``,
  ``load_umxhq(mock=…)``, and ``apply_recipe`` (the five §3.1 recipes).
* :mod:`singnet.peft.finetune_umx` — the dedicated UMX fine-tune loop (RUN LATER).
"""

from __future__ import annotations

from .lora import (
    LoRALinear,
    LoRAParametrization,
    count_total_params,
    count_trainable_params,
    lora_param_cost,
    merge_lora,
    trainable_report,
    trainable_share,
    wrap_lstm_lora,
)
from .umx_wrapper import (
    BASE_PARAM_COUNT,
    RECIPES,
    MockOpenUnmix,
    apply_recipe,
    load_umxhq,
    measured_trainable_share,
    recipe_rank,
    recipe_trainable_share,
    trainable_param_names,
)
from .finetune_umx import (
    FT_STEPS,
    WARMUP_STEPS,
    StereoInMemoryStore,
    StereoWavShardStore,
    UmxStereoChunks,
    build_umx_optimizer,
    finetune,
    make_umx_lr_lambda,
    recipe_share_table,
    sanity,
    umx_augment_pipeline,
    umx_magnitude,
    umx_mse_loss,
)

__all__ = [
    "LoRALinear",
    "LoRAParametrization",
    "wrap_lstm_lora",
    "merge_lora",
    "trainable_report",
    "trainable_share",
    "count_total_params",
    "count_trainable_params",
    "lora_param_cost",
    "MockOpenUnmix",
    "load_umxhq",
    "apply_recipe",
    "trainable_param_names",
    "recipe_trainable_share",
    "measured_trainable_share",
    "recipe_rank",
    "RECIPES",
    "BASE_PARAM_COUNT",
    "FT_STEPS",
    "WARMUP_STEPS",
    "StereoInMemoryStore",
    "StereoWavShardStore",
    "UmxStereoChunks",
    "build_umx_optimizer",
    "finetune",
    "make_umx_lr_lambda",
    "recipe_share_table",
    "sanity",
    "umx_augment_pipeline",
    "umx_magnitude",
    "umx_mse_loss",
]
