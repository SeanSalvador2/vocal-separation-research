"""LoRA adapter math: B=0 identity, merge round-trip, α/r scaling, r(d+k) count (G0).

The Direction-05 gate-G0 adapter items (MASTER_PLAN §7):
* ``LoRALinear`` with ``B=0`` reproduces the frozen ``nn.Linear`` **exactly**;
* a parametrized ``nn.LSTM`` with ``B=0`` reproduces the frozen LSTM **exactly**
  (the #1-risk plumbing, de-risked on CPU);
* merge-back == wrapped forward for both, and un-parametrizing leaves a plain module;
* the ``α/r`` scaling behaves as stated; trainable count is ``r(d_in + d_out)``.
"""

from __future__ import annotations

import torch
from torch import nn

from singnet.peft import (
    LoRALinear,
    count_trainable_params,
    merge_lora,
    trainable_report,
    wrap_lstm_lora,
)
from singnet.peft.lora import lora_param_cost


def test_lora_linear_b0_is_exact_identity() -> None:
    torch.manual_seed(0)
    base = nn.Linear(16, 12, bias=True)
    x = torch.randn(5, 16)
    wrapped = LoRALinear(base, r=4, alpha=8)
    # B = 0 at init => the wrapped forward equals the base forward bit-for-bit.
    assert torch.allclose(wrapped(x), base(x), atol=0.0, rtol=0.0)


def test_lora_linear_base_is_frozen_and_ab_trainable() -> None:
    base = nn.Linear(16, 12, bias=False)
    wrapped = LoRALinear(base, r=4, alpha=8)
    assert not wrapped.base.weight.requires_grad
    assert wrapped.lora_A.requires_grad and wrapped.lora_B.requires_grad


def test_lora_trainable_count_is_r_times_din_plus_dout() -> None:
    d_out, d_in, r = 12, 16, 4
    wrapped = LoRALinear(nn.Linear(d_in, d_out, bias=False), r=r, alpha=8)
    # only lora_A (r x d_in) + lora_B (d_out x r) train: r(d_in + d_out).
    assert count_trainable_params(wrapped) == r * (d_in + d_out)
    assert count_trainable_params(wrapped) == lora_param_cost(d_out, d_in, r)


def test_lora_linear_merge_roundtrip() -> None:
    torch.manual_seed(1)
    base = nn.Linear(16, 12, bias=True)
    x = torch.randn(4, 16)
    wrapped = LoRALinear(base, r=4, alpha=8)
    nn.init.normal_(wrapped.lora_B, std=0.3)  # move B off zero -> a real update
    wrapped_out = wrapped(x)
    merged = wrapped.merge()
    assert isinstance(merged, nn.Linear)
    assert torch.allclose(merged(x), wrapped_out, atol=1e-5)
    # W_merged = W0 + (alpha/r) B A
    expected = base.weight + (8 / 4) * (wrapped.lora_B @ wrapped.lora_A)
    assert torch.allclose(merged.weight, expected, atol=1e-6)


def test_lora_alpha_over_r_scaling() -> None:
    torch.manual_seed(2)
    base = nn.Linear(8, 8, bias=False)
    a = LoRALinear(base, r=4, alpha=4)   # scaling 1.0
    b = LoRALinear(nn.Linear(8, 8, bias=False), r=4, alpha=8)  # scaling 2.0
    # give both the same A, B; the delta must scale exactly with alpha/r.
    with torch.no_grad():
        b.lora_A.copy_(a.lora_A)
        b.lora_B.copy_(a.lora_B)
        a.lora_B.normal_()
        b.lora_B.copy_(a.lora_B)
    assert torch.allclose(b.delta_weight, 2.0 * a.delta_weight, atol=1e-6)


def _small_lstm() -> nn.LSTM:
    return nn.LSTM(input_size=8, hidden_size=4, num_layers=2, bidirectional=True, dropout=0.0)


def test_lstm_lora_b0_is_exact_identity() -> None:
    torch.manual_seed(3)
    lstm = _small_lstm().eval()
    x = torch.randn(6, 3, 8)  # (T, B, in)
    with torch.no_grad():
        ref, _ = lstm(x)
    wrap_lstm_lora(lstm, r=2, alpha=4)
    lstm.eval()
    out, _ = lstm(x)
    # B = 0 at init => bit-exact identity through the (parametrized) LSTM.
    assert torch.allclose(out, ref, atol=0.0, rtol=0.0)


def test_lstm_lora_wraps_all_directions_and_layers() -> None:
    lstm = _small_lstm()
    wrap_lstm_lora(lstm, r=2, alpha=4)
    trainable = {n for n, p in lstm.named_parameters() if p.requires_grad}
    # 2 layers x 2 directions x {ih, hh} = 8 wrapped matrices, each with lora_A + lora_B.
    lora_names = {n for n in trainable if "lora_" in n}
    assert len(lora_names) == 8 * 2
    # every base weight is frozen (kept as the parametrization `.original`).
    originals = {n for n, p in lstm.named_parameters() if n.endswith(".original")}
    assert originals and all(not p.requires_grad for n, p in lstm.named_parameters()
                             if n.endswith(".original"))


def test_lstm_lora_update_flows_and_merges() -> None:
    torch.manual_seed(4)
    lstm = _small_lstm().eval()
    x = torch.randn(6, 3, 8)
    with torch.no_grad():
        ref, _ = lstm(x)
    wrap_lstm_lora(lstm, r=2, alpha=4)
    for m in lstm.modules():  # push B off zero on every parametrization
        if hasattr(m, "lora_B") and hasattr(m, "lora_A"):
            nn.init.normal_(m.lora_B, std=0.2)
    lstm.eval()
    wrapped_out, _ = lstm(x)
    assert not torch.allclose(wrapped_out, ref)  # the LoRA update actually changed the output
    merge_lora(lstm)
    lstm.eval()
    merged_out, _ = lstm(x)
    # merged (plain) LSTM == wrapped forward; parametrization removed.
    assert torch.allclose(merged_out, wrapped_out, atol=1e-5)
    from torch.nn.utils import parametrize

    assert not parametrize.is_parametrized(lstm)


def test_merge_lora_on_a_module_tree() -> None:
    torch.manual_seed(5)

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = LoRALinear(nn.Linear(8, 8, bias=False), r=2, alpha=4)
            self.lstm = _small_lstm()

    net = Net().eval()
    wrap_lstm_lora(net.lstm, r=2, alpha=4)
    nn.init.normal_(net.lin.lora_B, std=0.3)
    x = torch.randn(4, 8)
    with torch.no_grad():
        lin_before = net.lin(x)
    merge_lora(net)
    assert isinstance(net.lin, nn.Linear)  # LoRALinear child replaced by a plain Linear
    assert torch.allclose(net.lin(x), lin_before, atol=1e-5)


def test_trainable_report_columns_and_shares_sum_to_one() -> None:
    lstm = _small_lstm()
    wrap_lstm_lora(lstm, r=2, alpha=4)
    report = trainable_report(lstm)
    assert list(report.columns) == ["name", "shape", "n_params", "trainable", "share"]
    assert abs(report["share"].sum() - 1.0) < 1e-9
    # the frozen bases show trainable=False; the lora_* rows show trainable=True.
    lora_rows = report[report["name"].str.contains("lora_")]
    assert bool(lora_rows["trainable"].all())
