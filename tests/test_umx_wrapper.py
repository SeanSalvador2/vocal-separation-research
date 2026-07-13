"""Mock umxhq shapes, the five recipes' trainable-name sets, and the share table (G0).

The Direction-05 gate-G0 host items (MASTER_PLAN §7):
* the mock host has the **verified** shapes and the pinned 8,893,348 param count;
* each recipe freezes/trains **exactly** the §3.1 parameter set (asserted by name);
* the trainable-share table matches the plan within ±0.3 pp for the LoRA recipes and
  is pinned exactly for all five (head is 23.7 %, not the plan's symmetric-spectrum
  estimate of 18 % — see results/DEVIATIONS.md);
* a mock forward runs on tiny tensors and preserves the UMX I/O shape.
"""

from __future__ import annotations

import torch

from singnet.peft import (
    BASE_PARAM_COUNT,
    RECIPES,
    apply_recipe,
    count_total_params,
    load_umxhq,
    measured_trainable_share,
    recipe_trainable_share,
    trainable_param_names,
)
from singnet.peft.umx_wrapper import (
    HIDDEN_SIZE,
    NB_BINS,
    NB_CHANNELS,
    NB_OUTPUT_BINS,
    MockOpenUnmix,
)

# Pinned exact host-denominated shares (computed on the verified shapes).
PINNED_SHARE = {
    "zeroshot": 0.0,
    "head": 2_110_470 / BASE_PARAM_COUNT,     # 0.237309
    "lora4": 113_184 / BASE_PARAM_COUNT,      # 0.012727
    "lora16": 431_520 / BASE_PARAM_COUNT,     # 0.048522
    "full": 1.0,
}
# The plan's ("≈") targets; the LoRA rows must sit within ±0.3 pp of these.
PLAN_TARGET = {"head": 0.18, "lora4": 0.012, "lora16": 0.049}


def test_mock_has_verified_shapes_and_param_count() -> None:
    model = load_umxhq("cpu", mock=True)
    assert isinstance(model, MockOpenUnmix)
    assert count_total_params(model) == BASE_PARAM_COUNT == 8_893_348
    # the load-bearing shapes (fc1 in = 2*nb_bins; fc3 out = 2*nb_output_bins; ...).
    assert model.fc1.weight.shape == (HIDDEN_SIZE, NB_BINS * NB_CHANNELS)
    assert model.fc3.weight.shape == (NB_OUTPUT_BINS * NB_CHANNELS, HIDDEN_SIZE)
    assert model.input_mean.shape == (NB_BINS,) and model.output_scale.shape == (NB_OUTPUT_BINS,)
    # LSTM: 3 layers, bidirectional, hidden 256 => weight_ih_l0 is (4*256, 512).
    assert model.lstm.num_layers == 3 and model.lstm.bidirectional
    assert model.lstm.weight_ih_l0.shape == (4 * 256, HIDDEN_SIZE)


def test_mock_forward_preserves_umx_shape() -> None:
    torch.manual_seed(0)
    model = load_umxhq("cpu", mock=True).eval()
    # tiny magnitude spectrogram (nb_samples, nb_channels, nb_output_bins, nb_frames)
    x = torch.rand(2, NB_CHANNELS, NB_OUTPUT_BINS, 4)
    out = model(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert (out >= 0).all()  # a non-negative magnitude estimate (relu * mix)


def test_recipe_names_are_the_known_set() -> None:
    assert RECIPES == ("zeroshot", "head", "lora4", "lora16", "full")


def test_zeroshot_trains_nothing() -> None:
    model = load_umxhq("cpu", mock=True)
    apply_recipe(model, "zeroshot")
    assert trainable_param_names(model) == set()
    assert measured_trainable_share(model) == 0.0


def test_full_trains_everything() -> None:
    model = load_umxhq("cpu", mock=True)
    apply_recipe(model, "full")
    assert trainable_param_names(model) == {n for n, _ in model.named_parameters()}
    assert measured_trainable_share(model) == 1.0


def test_head_recipe_exact_trainable_set() -> None:
    model = load_umxhq("cpu", mock=True)
    apply_recipe(model, "head")
    assert trainable_param_names(model) == {
        "fc3.weight", "bn3.weight", "bn3.bias", "output_scale", "output_mean",
    }


def _expected_lora_names() -> set[str]:
    names = {
        "fc1.lora_A", "fc1.lora_B", "fc2.lora_A", "fc2.lora_B", "fc3.lora_A", "fc3.lora_B",
        "input_mean", "input_scale", "output_scale", "output_mean",
    }
    for k in range(3):
        for suffix in ("", "_reverse"):
            for w in ("weight_ih", "weight_hh"):
                base = f"lstm.parametrizations.{w}_l{k}{suffix}.0"
                names.add(f"{base}.lora_A")
                names.add(f"{base}.lora_B")
    return names


def test_lora_recipes_exact_trainable_set() -> None:
    expected = _expected_lora_names()
    for recipe in ("lora4", "lora16"):
        model = load_umxhq("cpu", mock=True)
        apply_recipe(model, recipe)
        got = trainable_param_names(model)
        assert got == expected, recipe
        assert len(got) == 34  # 6 fc + 4 scale/mean + 24 lstm (12 matrices x A/B)
        # the frozen bases carry no gradient.
        assert not model.fc1.base.weight.requires_grad


def test_trainable_share_table_pinned_and_measured() -> None:
    for recipe in RECIPES:
        closed = recipe_trainable_share(recipe)
        assert abs(closed - PINNED_SHARE[recipe]) < 1e-9, recipe
        model = load_umxhq("cpu", mock=True)
        apply_recipe(model, recipe)
        assert abs(measured_trainable_share(model) - PINNED_SHARE[recipe]) < 1e-9, recipe


def test_lora_shares_match_plan_within_0p3pp() -> None:
    # The H-05a-relevant LoRA shares land within ±0.3 pp of the plan's estimates.
    assert abs(recipe_trainable_share("lora4") - PLAN_TARGET["lora4"]) < 0.003
    assert abs(recipe_trainable_share("lora16") - PLAN_TARGET["lora16"]) < 0.003
    # lora16 is comfortably under the H-05a < 5 % budget.
    assert recipe_trainable_share("lora16") < 0.05


def test_head_share_is_23_7_not_18() -> None:
    # Pinned: the verified shapes give 23.7 %, not the plan's symmetric-spectrum 18 %
    # (fc3 output is 2*nb_output_bins=4098, not 2*nb_bins). Documented deviation.
    assert abs(recipe_trainable_share("head") - 0.237309) < 1e-4
    assert recipe_trainable_share("head") > 0.20  # decisively not ~18 %
