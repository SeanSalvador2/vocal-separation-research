# Deviations log — Direction 03

Per MASTER_PLAN footer: *any deviation during execution goes here with date +
reason.*

## Experimental / execution deviations

**None.** Nothing has been trained; the frozen plan (MASTER_PLAN, 2026-07-13) is
intact. This section accumulates entries only if execution departs from the
pre-registered design (e.g. the G2 σ_seed escalation adds a seed, the §3.2
hash-mismatch forces the baseline cell to be retrained here, or a post-hoc width
change is needed).

## Build-time implementation clarifications

Not deviations from any explicit spec value — these record choices made where the
plan left an implementation detail open, logged for full transparency. All code
is in `singnet/` + `scripts/` and is unit-tested (gate G0). Reported values are
the exact derived numbers.

- **2026-07-13 — Width search: `c = 23`, bottleneck bump `Δ = +14` → `b5 = 382`,
  `9,841,896` params (`+0.0625 %`).** MASTER_PLAN §3.1 allows "a single ±Δ
  adjustment at the bottleneck level only, if needed". It *is* needed: the pure
  3-tower variant closed form is `P(c) = 18100 c² + 403 c + 1`, and neither pure
  width lands within ±2 % — `c = 23` is `9,584,170` (`-2.56 %`) and `c = 24` is
  `10,435,273` (`+6.10 %`). The deterministic rule (`scripts/match_params.py`,
  documented there and in THEORY §4): (1) pick the **largest base width whose pure
  variant stays under the baseline** → `c = 23` (keeps the base structure from
  overshooting and the bottleneck ≥ 16c, matching the plan's `c ≈ 23` estimate);
  (2) apply the **single positive bottleneck bump** `Δ` (level-5 width
  `b5 = 16c + Δ`) that **minimizes `|P − 9,835,745|`** while staying within 2 % →
  `Δ = 14`, `b5 = 382`, `P = 9,841,896` (`+0.0625 %`, the tightest achievable
  match). Both variants (mel and uniform) have this **identical** count — conv
  params are spatial-size-independent, so the band edges change compute, not
  parameters. Pinned in `singnet/models/bandsplit_unet.py`
  (`MATCHED_BASE_WIDTH=23`, `MATCHED_BOTTLENECK_WIDTH=382`), in the variant
  configs, and asserted in `tests/test_match_params.py` +
  `tests/test_bandsplit_model.py`.

- **2026-07-13 — Mel band edges derived from the HTK formula: bins
  `[0, 142, 597, 2048]`.** `mel_edges()` implements HTK
  `m = 2595·log10(1 + f/700)`, equal-mel partition of `[0, 22050]`, and
  `bin = round(f·n_fft/sr)` at `sr = 44100, n_fft = 4096`. The interior edges are
  bins **142** (≈ 1.53 kHz) and **597** (≈ 6.43 kHz) — matching the plan's
  estimate `{0, 142, 597, 2048}` exactly (0-bin deviation, well within the ±3
  proximity the test allows). Uniform (equal-bin) edges: `[0, 683, 1365, 2048]`
  (≈ 7.35 kHz, 14.70 kHz). Derived in code; `tests/test_bandsplit_model.py`
  recomputes the formula independently and asserts self-consistency.

- **2026-07-13 — Band-split architecture (interpretation of §3.1).** Each of the
  3 towers is the **full 5-level** baseline encoder (`1 → c → 2c → 4c → 8c → b5`),
  so the towers' level-5 outputs, concatenated along the **frequency axis**, ARE
  the bottleneck; the baseline-identical decoder (`dec1` consuming the `b5`
  bottleneck, `dec2…dec5`/`head` at width `c`) consumes the per-level frequency
  concatenations exactly as the baseline consumes its encoder skips. This is the
  reading that yields the plan's `c ≈ 23` (a 4-level-tower + shared-bottleneck
  reading would give `c ≈ 29`). "No cross-band mixing before the bottleneck" holds
  because the towers are independent until the concatenation; convolutions only
  span bands in the decoder.

- **2026-07-13 — Padding carried through the decoder; per-band crop-back at the
  mask.** Each band is zero-padded on its **high-frequency side** up to the next
  multiple of 32 (so five stride-2 layers halve exactly), giving a padded total
  `P = 2112` for both layouts (mel padded widths `[160, 480, 1472]`, uniform
  `[704, 704, 704]`). `P` is carried through the concatenated skips, bottleneck
  and decoder (all spatial dims stay aligned by construction), and the final
  `(B, 1, 2112, 256)` mask is cropped back to `(B, 1, 2048, 256)` by gathering
  each band's real bins (a fixed index map; the U-Net's input↔output spatial
  alignment makes this meaningful). The pad→crop routing round-trips exactly
  (`tests/test_bandsplit_model.py::test_routing_roundtrip_*`). Boundary effects at
  the band seams are exactly the pre-registered "refuted (negative)" diagnostic
  (MASTER_PLAN §3 boundary analysis, §12), localized by the per-band figure.

- **2026-07-13 — `singnet/models/unet.py` NOT structurally refactored; blocks
  shared by import.** `BandSplitUNet` reuses `unet.py`'s `_EncoderBlock`,
  `_DecoderBlock`, `_featurize`, and `_cat_skip` directly (import), so the block
  design is provably baseline-identical. `SingNetC1` itself is **unchanged** — its
  `enc*/dec*/head` attributes, forward semantics, and **state-dict keys** are
  untouched, and `tests/test_model.py::test_exact_parameter_count` (9,835,745) and
  `::test_bottleneck_channels` still pass verbatim. The plan permitted a heavier
  refactor with documented key changes; the lighter share-by-import was chosen to
  avoid any state-dict churn or baseline-test edits. **No state-dict keys changed.**

- **2026-07-13 — Optional `model:` config block + `loss` key (both default to the
  baseline behavior, so old configs hash unchanged).** A config's optional
  `model: {arch: bandsplit, bands, base_width, bottleneck_width, n_bands}` block
  selects `BandSplitUNet` via `build_model_from_config`; absent (Directions 01/02
  and the D03 baseline cell) it builds `SingNetC1` at `base_channels`. An optional
  `loss` key decouples the training loss from the experimental `arm` id (D03's
  arms — `split_mel`/`split_uniform` — all train on `l1mag`); absent, `loss == arm`
  (the Directions 01/02 behavior). **Neither `canonicalize_config` nor the
  Directions 01/02 YAMLs were touched**, so `hash_config` is unchanged for every
  existing config and the D01≡D02≡D03-base shared-cell equality holds
  (`tests/test_config_schema_d03.py`). The baseline arm therefore appears in the
  registry as `arm = l1mag` (it *is* the D01 l1mag run); the notebook maps it to
  the `baseline` label for display.

- **2026-07-13 — Registry gains one appended column, `base_width`.**
  `REGISTRY_COLUMNS` appends `base_width` after the Direction-02 block (`arm`
  already existed and now carries the band-split arm id). Backward-compatible: an
  older registry reads back NaN-filled; runs default `base_width = 32` (the
  baseline width). `tests/test_registry_schema_d02.py`'s position assertion was
  updated to check the D02 columns remain a contiguous appended block *and* that
  `base_width` is the new last column — the same backward-compat guarantee, now
  aware of the D03 append. No column moved.

- **2026-07-13 — `scripts/match_params.py` cross-checks the closed form against
  the built `nn.Module`.** The per-module table (`results/param_match_table.md`,
  committed) and THEORY §4 are generated from the same closed-form accounting the
  test asserts equals `BandSplitUNet(...).num_parameters` — so a reader sees the
  match is exact, not hand-waved. The confirmation configs (`confirm_*_full.yaml`)
  are generated at analysis freeze by `run_sweep.py --direction 03 --stage full`,
  **not pre-committed** (they depend on the frozen reduced-sweep ranking).
