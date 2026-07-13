# Deviations log — Direction 05

Per MASTER_PLAN footer: *any deviation during execution goes here with date +
reason.*

## Experimental / execution deviations

**None.** Nothing has been trained and no weights have been downloaded; the frozen
plan (MASTER_PLAN, 2026-07-13) is intact. This section accumulates entries only if
execution departs from the pre-registered design (e.g. the G2 LR-grid escalation, the
G0b genre/noise resolution, or the G1 "shift too mild" branch).

## Build-time implementation clarifications

Not deviations from any explicit spec value except where called out — these record
choices made where the plan left an implementation detail open, plus **one substantive
correction** to a plan estimate. All code is in `singnet/peft/` + `scripts/` and is
unit-tested (gate G0). Reported values are the exact derived numbers.

- **2026-07-13 — HEAD trainable share is 23.73 %, not the plan's "≈ 18 %"
  (a corrected estimate; SUBSTANTIVE).** The MASTER_PLAN §3.1 table and LITERATURE §5
  estimate the head-only recipe at ≈ 18 % of the host. That estimate assumed a
  **symmetric spectrum** — `fc3` output = `2·nb_bins` (2·1487 = 2974) and a ~8.3 M
  total. The **verified** Open-Unmix shapes (00-shared-research/papers/openunmix2019.md
  §2; the mock is byte-shape-identical) have `fc3` output = `2·nb_output_bins`
  (2·2049 = 4098) — the input is bandwidth-cropped to 1487 bins but the mask is
  emitted over the **full** 2049-bin spectrum. Recomputed exactly against the built
  mock (`tests/test_umx_wrapper.py`):
  - **host base params = 8,893,348** (matches "≈ 8–9 M", "~8.9 M/target");
  - `head` = `fc3.weight (2,098,176) + bn3 (8,196) + output_scale (2,049) +
    output_mean (2,049)` = **2,110,470 → 23.7309 %**.
  The plan's "≈ 18 %" is therefore an artifact of the symmetric-spectrum
  approximation, and I honor the **verified shapes** (a hard constraint: "your mock
  must match its shapes") and pin the true 23.73 %. Crucially, the **LoRA shares that
  H-05a depends on are robust** to the approximation and match the plan within ±0.3 pp:
  - `lora4`  = **113,184 → 1.2727 %** (plan ≈ 1.2 %),
  - `lora16` = **431,520 → 4.8522 %** (plan ≈ 4.9 %; comfortably < 5 %, H-05a intact).
  The share denominator is the **host base** (8,893,348), i.e. "% of the host's
  parameters" exactly as H-05a is stated (§2); the added LoRA A/B params are excluded
  from that denominator. The G1 recompute against the real checkpoint is expected to
  reproduce these to the parameter (the mock is shape-exact). Head is now the
  *inefficient* baseline the LoRA recipes should beat by an even larger margin than the
  plan anticipated.

- **2026-07-13 — LSTM LoRA via `torch.nn.utils.parametrize` + a forward pre-hook.**
  `wrap_lstm_lora` registers a `LoRAParametrization` on every gate-stacked
  `weight_ih_l{k}` / `weight_hh_l{k}` (+ `_reverse`), both directions, all 3 layers
  (THEORY §3). Because `nn.LSTM` caches its weights in `_flat_weights` at construction,
  a forward **pre-hook** (`_refresh_flat_weights`) re-pulls the parametrized weights
  before every forward so the `W0 + (α/r)·BA` update flows into the (CPU/non-fused)
  kernel and gradients reach A/B. De-risked on CPU: **B = 0 gives a bit-exact identity
  (max abs diff 0.0)**, B ≠ 0 changes the output and both A and B receive gradients,
  and merge-back reproduces the wrapped forward exactly (0.0). **cuDNN caveat**
  (MASTER_PLAN §11): the parametrized weights are not a single contiguous buffer, so on
  GPU cuDNN's fused LSTM kernel de-optimizes to the slower unfused path — acceptable at
  6 k steps; wall-clock is recorded per run. `merge_lora` removes the parametrizations
  with `leave_parametrized=True`, detaches the hook, and rebuilds `_flat_weights`,
  yielding a plain `nn.LSTM`.

- **2026-07-13 — `LoRALinear` is a module wrapper; the LSTM uses parametrization.** Per
  the §8 contract, `LoRALinear(base, r, alpha)` is a small `nn.Module` holding the
  frozen base plus `lora_A`/`lora_B` (used for `fc1`/`fc2`/`fc3`), while the recurrent
  weights use `parametrize`. Both merge to plain modules and both satisfy the B = 0
  identity + merge round-trip tests. `α = 2r` is fixed (no α tuning, §3.1); A ~ N(0,
  σ²) with σ = 0.02 default, B = 0.

- **2026-07-13 — Mock UMX total = 8,893,348 params; a faithful `MockOpenUnmix`.**
  Rather than `pip install openunmix`, the CPU host is a self-contained
  `nn.Module` with the **exact** submodules/shapes/forward of the real model
  (crop→`fc1`/`bn1`/tanh→BiLSTM→skip-concat→`fc2`/`bn2`/relu→`fc3`/`bn3`→
  output affine→relu→×mix). The real path (`load_umxhq(mock=False)`) is code-pathed
  (lazy `torch.hub` import, guarded, `# pragma: no cover`) but never executed in tests
  (MASTER_PLAN §4). Input/output I/O convention matches the real model
  `(nb_samples, nb_channels, nb_output_bins, nb_frames)`.

- **2026-07-13 — Stereo channel-swap added to `AugmentPipeline`; D02-compatible.** UMX's
  augmentation recipe is gain + **channel swap** + cross-track remix, with **no sign
  flip** (Demucs, not UMX — `data.py` verified). I extended
  `singnet/data/augment.py` with `random_channel_swap`, a `channelswap` stream (id 4),
  a `channelswap: bool = False` switch, and `apply_channelswap`. The addition is
  **byte-for-byte backward-compatible** with Directions 01-02:
  - `channelswap` defaults **off**, so `AugmentPipeline()` and the positional
    `AugmentPipeline(True, True, True)` are unchanged (still `== AugmentPipeline()`);
  - the mono `__call__` (gain → flip) is **never** touched — channel swap is applied
    only by the UMX stereo dataset via the explicit `apply_channelswap`;
  - the new stream id does not perturb the `sample`/`remix`/`gain`/`flip` streams (each
    keyed by its own fixed id) — asserted in `tests/test_finetune_umx.py`.
  All D01/D02 augment tests (`tests/test_augment*.py`) pass unchanged. Stereo shards are
  read by a new `StereoWavShardStore` / `UmxStereoChunks` (the mono `MusdbChunks` and
  `WavShardStore` are untouched, so Directions 01-03 stay risk-free).

- **2026-07-13 — Registry gains seven appended columns
  (`domain, recipe, rank, lr, trainable_params, trainable_share, peak_vram_gb`).**
  `REGISTRY_COLUMNS`/`RunRecord` append the D05 block after `base_width` (§5).
  Backward-compatible: an older registry reads back NaN-filled; the D05 fields default
  to "not a PEFT run" (`domain=""`, `rank=0`, NaN metrics). No existing column moved.
  Following the D03 precedent (which updated the same position-guard), the
  `tests/test_registry_schema_d02.py` `[-1]` assertion was updated to check the D02
  block, then `base_width`, then the D05 block ending the schema — the same
  no-column-moved guarantee, now aware of the D05 append. The Directions 01-03 registry
  rows carry empty/NaN D05 fields; the shared registry machinery
  (`read/write/upsert`) is reused verbatim (§5 "reuses the registry utilities").

- **2026-07-13 — New D05 config keys are optional; every existing hash is unchanged.**
  The D05 keys (`host`, `domain`, `recipe`, `rank`, `lr`, `warmup_steps`,
  `channelswap`, `domain_suffix`, `op`, …) appear **only** in Direction-05 configs.
  `hash_config` canonicalizes just the augment/allowlist schema and passes all other
  keys through, so the Directions 01-03 configs — which never carry the D05 keys — hash
  identically to before (the D01 ≡ D02 ≡ D03-base shared-cell equality still holds,
  `tests/test_config_schema_d05.py`). The 12 main D05 configs have distinct hashes;
  T2 configs carry `op: pending-G0b` until `--resolve-t2` fixes the domain at G0b; the
  probe configs are **generated** by `run_sweep.py --stage probes` (not pre-committed,
  from `configs/t1_probe_template.yaml`), mirroring the D01/D03 confirmation-config
  pattern.

- **2026-07-13 — Dedicated fine-tune loop; `train/loop.py` untouched.** `finetune_umx.py`
  is a separate trainer (Adam, 200-step warmup then **constant** LR, MSE-on-magnitude,
  stereo pipeline) per the §5/§8 contract; it reuses `save/load_checkpoint`,
  `seed_everything`, the registry, `STFT`, and `si_sdr` but does **not** modify or
  generalize `singnet/train/loop.py`, so Directions 01-03 training semantics are
  bit-unchanged. The heavy `finetune()` and the consolidated `build_test_matrix()`
  (`evaluate_umx.py`) are RUN LATER — they fail loud (`FileNotFoundError`) when shards /
  checkpoints are absent, which the CPU suite asserts.
