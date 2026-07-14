# Deviations log — Direction 06

Per MASTER_PLAN footer: *any deviation during execution goes here with date +
reason.*

## Experimental / execution deviations

**None.** Nothing has been trained and no GPU work has run; the frozen plan
(MASTER_PLAN, 2026-07-13) is intact. This section accumulates entries only if
execution departs from the pre-registered design (e.g. the G2 +1-seed escalation,
or an H-06a "robust/flat" branch).

## Build-time implementation clarifications

Not deviations from any explicit spec value **except the one flagged SUBSTANTIVE
item** — these record choices made where the plan left an implementation detail
open. All code is in `singnet/{data,losses,analysis,train}/` + `scripts/` and is
unit-tested (gate G0). Reported values are the exact derived numbers.

- **2026-07-14 — `keep = ⌈(1−q)·B⌉` per the §3.2 pinned formula (= 12/16 at
  q=0.30), NOT the §11 parenthetical "keep 11/16" (SUBSTANTIVE, plan-internal
  inconsistency).** MASTER_PLAN §3.2 pins the trim rule as "keep the
  `⌈(1−q)B⌉` smallest," and the Direction-06 stage-D contract repeats it verbatim.
  For q = 0.30, B = 16 this is `⌈0.70·16⌉ = ⌈11.2⌉ = 12` kept / 4 dropped. The §11
  risk-table parenthetical "q = 0.30 ⇒ keep 11/16 (stated)" is arithmetically the
  *drop*-ceiling convention (`drop = ⌈q·B⌉ = ⌈4.8⌉ = 5 ⇒ keep 11`), which
  contradicts §3.2's *keep*-ceiling formula. I implement the **pinned §3.2 formula**
  (the explicit, repeated contract): `TrimmedLoss.keep_count(16, 0.30) == 12`,
  asserted in `tests/test_trimmed_loss.py::test_keep_count_ceil_rule`. The observed
  kept-fraction logged to the registry (`kept_fraction_observed`) makes the actual
  12/16 = 0.75 transparent in every run, so the analysis reads the true value
  regardless of which parenthetical a reader recalls. (q = 0.10 keeps
  `⌈0.90·16⌉ = 15`.)

- **2026-07-14 — Loss-contract per-chunk path via a keyword-only `reduce=True`
  flag.** The plan (§3.2 / §8) offered two shapes for the extension — "a `reduce`
  flag OR a per-chunk vector in `aux`". I chose the **`reduce` flag** because it
  keeps every `reduce=True` code path **byte-identical** to the pre-extension
  computation (each loss returns its historical scalar and the *same* `aux`, so
  `aux == {}` for the magnitude losses still holds), while `reduce=False` adds the
  differentiable `(B,)` `aux["per_chunk"]` that `TrimmedLoss` ranks. **All Direction
  01–05 loss tests pass unmodified** (`tests/test_losses.py`, verified). The
  per-chunk vector is fp32 (the existing autocast-disabled region), so AMP cannot
  reorder the trim ranking (§11). `singnet/losses/_base.py` adds a shared
  `per_chunk_mean` helper; the five losses gained only the `reduce=False` branch.

- **2026-07-14 — the §5 trim-telemetry ⟨a⟩-energy is the RAW same-track
  accompaniment energy at the vocal window (pre-corruption, pre-augmentation).**
  §5's mechanism is "high-⟨a⟩-energy chunks carry higher irreducible loss ⇒ trimming
  ranks by accompaniment energy." The quantity that sets a chunk's *irreducible*
  loss is `‖ε·a_voc‖²`, where `a_voc` is the vocals-track's **own** accompaniment
  (the component that bleeds into `ṽ = v + ε·a_voc`) — not the remixed accompaniment
  `ã_j` that ends up in the mixture. I therefore thread `acc_energy` =
  `mean(a_voc[window]²)` computed from the **raw** (uncorrupted, un-gained) stem:
  it is ε- and augmentation-invariant, so it is a clean per-chunk cleanliness
  covariate against which the trimmer's selection is ranked (kept ⟨a⟩-energy <
  dropped ⟨a⟩-energy is the testable §5 signature). Ranking by raw `‖a_voc‖²` ≡
  ranking by the bleed energy `ε²‖a_voc‖²` (ε constant per arm). Threaded
  dataset → `batch["acc_energy"]` → the training step → `TrimmedLoss(chunk_energy=…)`
  → the every-500-steps CSV. `singnet/data/musdb_dataset.py`,
  `tests/test_corrupt.py::test_acc_energy_threaded_and_deterministic`.

- **2026-07-14 — the eval-split guard is enforced at TWO structural choke points.**
  §3.1 requires the guard be structural, not advisory. (i) `build_corruption(block,
  split)` **raises** `EvalSplitCorruptionError` for any ε > 0 on a non-`train` split
  — the transform for an eval split cannot be constructed; (ii) `MusdbChunks.__init__`
  **raises** the same if handed a `StemBleed` with `split != "train"` — a corrupting
  dataset object cannot exist for valid/test. Either path alone suffices; both are
  unit-tested (`tests/test_corrupt.py`). The prediction line
  (`singnet/analysis/bleed.py`) is computed from **clean** stems only, independent of
  any corrupted or eval target (the §7 G3 audit relies on this).

- **2026-07-14 — sisdr per-chunk path: silent-target chunks surface as high loss
  and are trimmed; the `reduce=True` silence *skip* is untouched.** On the
  `reduce=False` path `SiSdrLoss` returns per-chunk −SI-SDR over **all** chunks (a
  silent target has ‖v‖²≈eps ⇒ large −SI-SDR ⇒ ranked worst ⇒ dropped by
  construction). The pre-registered `reduce=True` silence *skip* (§4.4) is a separate
  mechanism, left byte-identical for the untrimmed arms, so `test_sisdr_*` pass
  unmodified. The trimmed wrapper works around both `l1mag` (primary) and `sisdr`
  (contingency), `tests/test_trimmed_loss.py`.

- **2026-07-14 — Registry gains four appended columns
  (`epsilon, trim_q, kept_fraction_observed, trim_energy_stats_path`).**
  `REGISTRY_COLUMNS`/`RunRecord` append the D06 block after the D05 block (§5).
  Backward-compatible: an older registry reads back NaN-filled; the D06 fields
  default to "not a bleed run" (NaN metrics, empty path). No existing column moved.
  Following the **D05→D02 precedent** (D05 updated the D02 position-guard to be aware
  of its append), the position-guard assertions in
  `tests/test_registry_schema_d02.py` and `tests/test_registry_schema_d05.py` were
  updated to locate the D05 block by index (rather than as the literal tail) and to
  assert the D06 block ends the schema — the same no-column-moved guarantee, now
  aware of the D06 append. The Directions 01–05 registry rows carry empty/NaN D06
  fields; the shared registry machinery (`read/write/upsert`) is reused verbatim.

- **2026-07-14 — the `corrupt` config block is optional; ε = 0 canonicalizes to
  "no corruption".** `hash_config` normalizes `corrupt` so an absent block or
  `epsilon: 0` leaves **no** corruption key — hash-identical to an uncorrupted
  Directions 01–05 config — while ε > 0 becomes the canonical `{"epsilon": <float>}`
  identity field. The `trim` block passes straight through (already an identity
  field, so `trim30` / `trim30_q10` / `trim_clean` hash distinctly). Consequence: the
  D06 clean cell (`base.yaml`) hashes **bit-identically** to D01 `l1mag_seed0_reduced`
  (and D02/D03 base) — the shared clean cell is reused, not retrained — and every
  prior config's hash is unchanged (`tests/test_config_schema_d06.py`).

- **2026-07-14 — arm vs loss decoupling mirrors Direction 03.** Each D06 config
  carries an experimental `arm` id (`bleed30`, `trim30`, `trim_clean`, …) and a
  separate `loss` key (`l1mag`, or `sisdr` for the contingencies), exactly as D03's
  `split_mel`/`loss: l1mag` split. `run_id = arm_seedS_budget_hash` stays readable;
  the training loss is `loss`, optionally wrapped by `TrimmedLoss` when a `trim:`
  block is present. The train loop is a **no-op for every D01–05 config**
  (`build_training_loss` returns the plain base loss; `build_corruption` returns
  `None`) — guarded by `tests/test_train_d06.py`.
