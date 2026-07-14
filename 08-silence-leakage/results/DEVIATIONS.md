# Deviations log — Direction 08

Per MASTER_PLAN footer: *any deviation during execution goes here with date +
reason.*

## Experimental / execution deviations

**None.** Nothing has been trained and no GPU work has run; the frozen plan
(MASTER_PLAN, 2026-07-13) is intact. This section accumulates entries only if
execution departs from the pre-registered design (e.g. the G2 +1-seed escalation, a
θ/L_min revision on the G2 valid-n fail path, or an H-08b "silence for free" branch).

## Build-time implementation clarifications

Not deviations from any explicit spec value **except the one flagged SUBSTANTIVE
item** — these record choices made where the plan left an implementation detail open.
All code is in `singnet/{metrics,data,eval,train}/` + `scripts/` and is unit-tested
(gate G0). Reported values are the exact derived numbers.

- **2026-07-14 — the non-uniform policies draw at 1-s grid resolution; `uniform`
  stays at sample resolution (SUBSTANTIVE, plan left the resolution open).** The energy
  profile `E(s)` is defined on the 1-s start grid (§4.1, §5), but the legacy chunk
  sampler picks a start at *sample* resolution (`rng.integers`). To honour both, the
  `energy`/`drop`/`curriculum` policies draw a grid index weighted by `E` and map it to
  the sample-domain start `grid_index · sr` (clamped into the valid range), while
  `uniform` keeps the exact sample-resolution `rng.integers` draw. This makes `uniform`
  bit-identical to the shared cell (the §4.2 guarantee) and keeps the weighted policies
  faithful to the profile grid. `singnet/data/sampling.py::ChunkSampler.start`,
  `tests/test_sampling.py::test_uniform_policy_reproduces_legacy_chunk_starts`.

- **2026-07-14 — a dedicated `sampling` RNG stream (id 5) carries the policy draw.**
  `STREAM_IDS` gains `"sampling": 5` (`singnet/data/augment.py`). The non-uniform
  weighted start (for both the vocal window and, under remix, the partner window) is
  drawn from `(seed, "sampling", step)`, so switching policy perturbs neither the
  augmentation streams (remix/gain/flip) nor — for `uniform`, which never touches this
  stream — the `sample` stream. The `sample` stream still chooses *which track*
  (uniform over tracks) for every policy, so the track sequence is shared across arms and
  only the within-track window differs. Stream independence is unit-tested
  (`tests/test_sampling.py`).

- **2026-07-14 — the `sampling` config block is optional; `uniform` canonicalizes to
  "no sampling key".** `hash_config` normalizes `sampling` so an absent block or
  `policy: uniform` leaves **no** sampling key — hash-identical to the shared baseline
  cell — while a non-uniform policy becomes a canonical identity block carrying **only
  the constant that steers it** (`theta_db` for `drop`, `floor_lambda` for
  `energy`/`curriculum`). Consequence: the D08 `base.yaml` hashes **bit-identically** to
  D01 `l1mag_seed0_reduced` (and D02/D03/D06 base), so the uniform arm is reused, not
  retrained; every prior config's hash is unchanged (`tests/test_config_schema_d08.py`).

- **2026-07-14 — `curriculum` λ(t) reads the training step as `index // batch_size`.**
  The map-style dataset derives the step from the item index (the loop pulls indices
  `[s·B:(s+1)·B]` for step `s`, shuffle off), so `MusdbChunks` takes an optional
  `batch_size` (default 16) used **only** to compute the curriculum step; it does not
  touch the uniform path or any hash. λ(t) is linear `1.0 → floor_lambda` over the first
  50 % of steps, then held — values `[1.0, 0.55, 0.1, 0.1, 0.1]` at `{0,25,50,75,100}%`
  (`tests/test_sampling.py::test_curriculum_lambda_schedule_values`).

- **2026-07-14 — `drop` on an all-silent track falls back to uniform (the §12 float-
  underflow guard, analog).** If *every* window of a track is sub-θ (a degenerate,
  near-silent track), `drop`'s support set is empty; rather than produce a zero weight
  vector, the sampler falls back to uniform over that track's starts. The `energy`
  floor makes this impossible for `energy`/`curriculum` (mass ≥ λ/N per start, unit-
  tested). Documented so the fallback is a known, tested behaviour, not a surprise.

- **2026-07-14 — exposure telemetry is computed for ALL arms, every step, on the
  *augmented* vocal target.** The §6 realized silent-chunk exposure (fraction of drawn
  chunks with vocal-target RMS < θ) is accumulated every step and a windowed-mean row is
  written every 500 steps (`sampling_exposure.csv`); the run mean is the registry's
  `silent_exposure_observed`. It is measured on `batch["vocals"]` — the target the model
  actually sees (post gain/flip) — is a pure read (no RNG), and is logged for the uniform
  baseline too, since §13 relies on exposure to say whether the policies differed in
  practice. `singnet/train/loop.py::chunk_silent_fraction`, `tests/test_train_d08.py`.

- **2026-07-14 — checkpoint selection is SI-SDR-only via `is_new_best`; SLR is computed
  alongside but never compared (§6 selection-bias guard).** Validation runs
  `validation_report`, which separates each val track once and returns both mean SI-SDR
  and mean SLR (θ = −60). Selection calls `is_new_best(candidate_sisdr, best_sisdr)` —
  a function that takes **no SLR argument** by construction; `best_val_slr` records the
  SLR *of the SI-SDR-best checkpoint* (descriptive). A test asserts the selection helper
  has no `slr` parameter and never names `slr` in its source
  (`tests/test_train_d08.py::test_checkpoint_selection_is_blind_to_slr`).

- **2026-07-14 — Registry gains five appended columns
  (`policy, theta_db, floor_lambda, silent_exposure_observed, best_val_slr`).**
  `REGISTRY_COLUMNS`/`RunRecord` append the D08 block after the D06 block (§6).
  Backward-compatible: an older registry reads back NaN-filled; the D08 fields default to
  the uniform baseline (`policy="uniform"`, NaN telemetry). No existing column moved.
  Following the **D06→D05→D02 precedent**, the position-guard assertion in
  `tests/test_registry_schema_d06.py` was updated to locate the D06 block by index
  (rather than as the literal tail) and the D08 block is asserted to end the schema
  (`tests/test_registry_schema_d08.py`). `theta_db`/`floor_lambda` are recorded only for
  the policy that uses each (NaN otherwise), mirroring the hash-identity choice.

- **2026-07-14 — `TrackScores` gains four optional SLR columns; `score_system` gains
  keyword-only `sr`/`slr`.** `evaluate(..., slr=True)` (CLI `--slr`) writes
  `slr_m50/slr_m60/slr_m70` (θ = −50/−60/−70) + `slr_n_regions_60` into the per-track
  CSV for **every** system, including the do-nothing (0 dB) and oracle anchors — so the
  (SI-SDR, SLR) plane's anchors come from one call path. The new `score_system` args are
  keyword-only with defaults, so the existing positional call sites (and
  `tests/test_eval.py`) are unaffected. `singnet/eval/evaluate.py`.

- **2026-07-14 — arm vs loss decoupling mirrors Directions 03/06.** Each D08 config
  carries an experimental `arm` id (`energy`, `drop`, `curriculum`, `uniform`) and a
  separate `loss` key (`l1mag`, or `sisdr` for the contingencies). `run_id =
  arm_seedS_budget_hash` stays readable; the base `uniform` cell keeps `arm: l1mag` so it
  hashes as — and is reused as — the shared D01 `l1mag` cell.
