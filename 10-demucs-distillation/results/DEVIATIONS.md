# Deviations log — Direction 10

Per MASTER_PLAN footer: *any deviation during execution goes here with date +
reason.*

## Experimental / execution deviations

**None.** Nothing has been trained, no data has been downloaded (neither FMA audio/metadata
nor demucs weights), and no GPU work has run; the frozen plan (MASTER_PLAN, 2026-07-13) is
intact. This section accumulates entries only if execution departs from the pre-registered
design (e.g. the G1 activity-threshold fallback firing, the G2 +1-seed escalation, or a
pre-registered outcome branch — negative transfer, distill-only ≥ musdb_only, SLR
degradation).

## Build-time implementation clarifications

Not deviations from any explicit spec value — these record choices made where the plan left
an implementation detail open. All code is in `singnet/{data,eval,train,utils}/` + `scripts/`
and is unit-tested (gate G0). Reported values are the exact derived numbers.

- **2026-07-14 — the license decision is an allowlist canonicalizer, DENY-by-default.**
  `scripts/prepare_fma.py::canonicalize_license` maps a raw FMA license string (human-readable
  name *or* CC URL form) to one of the pinned canonical tokens {CC0, CC-BY, CC-BY-SA, CC-BY-NC,
  CC-BY-NC-SA} or to `None` (deny). The `-ND`/NoDerivatives check runs FIRST and dominates
  (separated stems are derivative works, §5/THEORY §5), so an otherwise-allowlisted BY/NC/SA
  string with a NoDerivatives clause is still denied. Empty / non-string / `All Rights
  Reserved` / non-CC garbage all deny. The full allow/deny table is unit-tested
  (`tests/test_prepare_fma.py`).

- **2026-07-14 — the screen sample is `numpy.random.default_rng(seed)` over the SORTED,
  de-duplicated allowlisted ids.** `screen_sample(ids, n, seed)` sorts + de-dups the ids
  (input-order independence), draws a `rng.permutation(len(ids))[:n]`, and returns the drawn
  ids in permutation order — deterministic and reproducible in `(ids, n, seed)`; a different
  seed changes the draw; `n > pool` returns the whole (shuffled) pool
  (`tests/test_prepare_fma.py`). Distinct from the training RNG (`derive_rng`), which keys the
  per-example data streams.

- **2026-07-14 — 2-stem consistency is computed in float64 then cast (â = x − v̂ exact).**
  `scripts/teacher_label.py::two_stem_consistency` mirrors `StemBleed`'s float64→float32 cast so
  the additivity residual `‖(v̂ + â) − x‖` is at float32 round-off, not accumulated. The raw
  4-stem sum-residual `consistency_residual_db` (recorded in the provenance, NOT used as a
  target) is the honest measure of teacher self-inconsistency (`tests/test_teacher_label.py`).

- **2026-07-14 — the activity ratio uses Direction 08's `windowed_vocal_rms`; a window is
  active at the project-wide −60 dBFS silence threshold.** `vocal_activity_ratio` reports the
  fraction of the teacher-vocals' 6-s/1-s-grid RMS profile at or above `10**(−60/20)` — the
  same "silent" notion as the `sisdr` guard and the SLR metric (one project-wide θ). The screen
  keeps clips with ratio ≥ 0.20; the pre-registered fallback lowers the threshold to 0.10 ONCE
  and, if still short of `keep`, uses all survivors and records the count (`ActivityScreenResult`,
  `tests/test_teacher_label.py`). The threshold is on the clip-level ratio, matching the §3.2
  wording ("activity ratio ≥ 20 %").

- **2026-07-14 — a dedicated `pool` RNG stream (id 6) carries the pool draw.** `STREAM_IDS`
  gains `"pool": 6` (`singnet/data/augment.py`). `MixedPools.pool_of(index)` draws
  Bernoulli(p_fma) from `(seed, "pool", index)`, so switching the pool mix perturbs neither
  pool's augmentation streams (remix/gain/flip/sample/sampling) — each keyed by its own fixed
  id. Appending id 6 leaves every existing stream byte-identical (no prior run's data order
  changes). Pool selection is keyed by the per-example item index (the map-style coordinate),
  so the mix is per-example, and indexing the chosen pool with that same index keeps its stream
  coordinate intact — the pool-independence guarantee (`tests/test_pseudo.py`).

- **2026-07-14 — within-pool remix is guaranteed by construction (two distinct datasets).**
  `MixedPools` holds two separate `MusdbChunks` (one per store) with their own
  `AugmentPipeline` instances; `__getitem__` routes each example to exactly one and returns its
  chunk. Because each `MusdbChunks` draws its remix partner from its OWN `tracks`, a partner can
  never cross pools — asserted by construction/type (the two pools' track sets are disjoint and
  the returned item equals the chosen pool's own `[index]`), not by sampling
  (`tests/test_pseudo.py`).

- **2026-07-14 — the MUSDB-path guard is structural (path containment + a content sniff).**
  `PseudoLabeledShards` refuses at construction (`MusdbShardLeak`) if its root equals/nests
  with any known MUSDB shard root (the training `shard_root` is passed in) OR if the root looks
  like a MUSDB *decode* root (an `index.json` with per-track `subset` fields, or a track dir
  with the drums+bass+other 4-stem layout a pseudo root never has). This mirrors Direction 06's
  structural eval-split guard — the offending store cannot be built (`tests/test_pseudo.py`).

- **2026-07-14 — the `data_source`/`p_fma` schema is optional; `musdb` canonicalizes to "no
  pseudo key".** `hash_config` normalizes so `data_source: musdb` (or an absent key) leaves
  **no** pseudo key — hash-identical to the shared baseline cell — while `mixed`/`distill`
  becomes a canonical `{data_source, p_fma}` identity block (`distill` -> p_fma 1.0). The
  pseudo-shard *locations* (`pseudo_root`, `pseudo_manifest`) are non-identity (excluded from
  the hash, like `shard_root`). Consequence: the D10 `base.yaml` hashes **bit-identically** to
  D01 `l1mag_seed0_reduced` (`a97d5400e994`) — the sixth reuse of the shared cell — so
  `musdb_only` is reused, not retrained; every prior config's hash is unchanged
  (`tests/test_config_schema_d10.py`).

- **2026-07-14 — Registry gains five appended columns
  (`data_source, p_fma, n_pseudo_clips, teacher_version, teacher_consistency_db`).**
  `REGISTRY_COLUMNS`/`RunRecord` append the D10 block after the D08 block (§5).
  Backward-compatible: an older registry reads back NaN-filled; the D10 fields default to
  `data_source="musdb"` with NaN pool/teacher telemetry. No existing column moved. Following the
  **D06→D08 precedent**, the position-guard assertion in `tests/test_registry_schema_d08.py` was
  updated to locate the D08 block by index (rather than as the literal tail) and the D10 block
  is asserted to end the schema (`tests/test_registry_schema_d10.py`).

- **2026-07-14 — the test session routes through `evaluate.py --direction 10 --test-session
  --include-teacher --slr`.** `singnet/eval/teacher_session.py::build_teacher_session` is the §5
  skeleton (student checkpoints from the registry + the lazy `htdemucs` teacher wrapper + the
  do-nothing / oracle-IRM anchors + SLR at θ ∈ {−50,−60,−70}), mirroring Direction 05's
  `build_test_matrix`. It fails loud (`FileNotFoundError`) without shards/registry/checkpoints,
  so the CPU suite asserts the clean-error path; the demucs import is lazy and reached only RUN
  LATER (`tests/test_train_d10.py`).
