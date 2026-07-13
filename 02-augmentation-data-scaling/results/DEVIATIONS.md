# Deviations log — Direction 02

Per MASTER_PLAN footer: *any deviation during execution goes here with date +
reason.*

## Experimental / execution deviations

**None.** Nothing has been trained; the frozen plan (MASTER_PLAN, 2026-07-13) is
intact. This section accumulates entries only if execution departs from the
pre-registered design (e.g. the G2 σ_seed escalation adds seeds to `no_remix`, or
the §3.3 hash-mismatch forces the FULL-86 cell to be retrained here).

## Build-time implementation clarifications

Not deviations from any explicit spec value — these record choices made where the
plan left an implementation detail open, logged for full transparency. All are in
`singnet/` and are unit-tested (gate G0).

- **2026-07-13 — Per-transform RNG streams replace Direction 01's single stream.**
  MASTER_PLAN §3.5 requires each transform to own an independent
  `(seed, name, step)` stream so a leave-one-out ablation leaves the other
  transforms' draws unchanged. `AugmentPipeline` now derives four independent
  streams — `sample` (the always-on chunk sampler), `remix`, `gain`, `flip`
  (`singnet/data/augment.py::STREAM_IDS`) — instead of the previous single
  `derive_rng(seed, index)` stream shared by all draws. Because **nothing has been
  trained**, bit-exact equality with the old stream layout is neither required nor
  claimed (Direction 01 §3.1 only guarantees determinism given `(seed, step)` and
  arm-independence, both preserved). The Direction-01 pipeline test
  (`tests/test_augment.py::test_pipeline_deterministic_given_seed_and_step`) was
  updated from the old `pipe(sources, rng)` call to the new `pipe(sources, step)`
  signature; its guarantee is unchanged (determinism in `(seed, step)`) and was
  *strengthened* with an added seed-sensitivity assertion. No other Direction-01
  test changed; all still pass. `AugmentPipeline(True, True, True)` remains exactly
  the Direction-01 full recipe (remix → gain → flip, constants per §5), and the
  Direction-01 YAML configs are byte-identical.

- **2026-07-13 — "No remix" = the true track mixture (same track, same window).**
  MASTER_PLAN §3.1 defines the remix-off example as "the true track mixture (sum
  of that track's own stems)". The dataset therefore takes the accompaniment from
  the *same* track and the *same* 6 s window as the vocals when remix is off
  (previously the two stems were drawn from independent windows of the same track).
  This makes the remix on/off contrast maximally controlled: the vocals chunk is
  bit-identical either way (same `sample` stream), and only the accompaniment
  source changes — verified in `tests/test_augment_switchboard.py`.

- **2026-07-13 — Remix partner is an independent draw within the active pool.**
  MASTER_PLAN §5 specifies "independent draws within the active split/subset"
  (matching UMX `random_track_mix` / Demucs `Remix`); the §3.1 table's "j ≠ i" is
  the typical case. The code draws the accompaniment track independently from the
  active pool, so a self-pair can occur with probability ≈ 1/n (≈ 1.2 % at n = 86);
  this is the code-verified behavior of both reference repos and is not rejected.

- **2026-07-13 — Config canonicalization default-fills the augment/allowlist
  schema before hashing.** `singnet/utils/config.py::canonicalize_config` folds
  three spellings of the same experiment — a legacy `augment: true` + `remix: true`
  config, a config with no augmentation block, and an explicit
  `augment: {remix, gain, flip}` + `data: {track_allowlist_csv: null}` block — to
  one canonical form, so they hash equal. This is what makes Direction 01's `l1mag`
  cell provably the same run as this direction's `full`/`n86` cell (§3.3;
  `tests/test_config_schema.py`, gate G0). The subset allowlist enters the hash by
  its **path**, so each nested subset is a distinct experiment.

- **2026-07-13 — Registry schema gains four appended columns.**
  `aug_remix, aug_gain, aug_flip, n_songs` are appended to `REGISTRY_COLUMNS`
  (`singnet/train/registry.py`) so an older Direction-01 registry reads back with
  them NaN-filled and Direction-01 runs populate them with the full-recipe defaults
  (all True, 86 songs). No existing column moved.

- **2026-07-13 — Fixed a repo-wide `.gitignore` bug that untracked the entire
  `singnet/data/` source package.** The audio-hygiene rule `data/` (unanchored)
  matched *any* directory named `data` at any depth, so the `singnet/data/`
  **source** package (`augment.py`, `musdb_dataset.py`, `manifest.py`,
  `__init__.py`) was never committed by Direction 01 — a fresh clone of the branch
  could not `import singnet.data`, and this direction's core delta (the
  augmentation switchboard) lives there. Anchored the rules to the repo root
  (`/data/`, `/shards/`) so top-level MUSDB dirs stay ignored while the package is
  tracked. Those four data-layer files are therefore committed here for the first
  time (the three I did not modify — `manifest.py` and the two stores' host file —
  are the unchanged Direction-01 versions). Flagged to the orchestrator: this
  affects every direction, not just this one.

- **2026-07-13 — Subset helpers live in `scripts/make_subsets.py`; scaling analysis
  in `singnet/analysis/scaling.py`.** The analysis functions (the contracted
  `fit_log2`, `secant_slopes`, `loo_table`, plus `pooled_seed_sigma`) are pure
  NumPy/pandas — **SciPy is not a project dependency**, so the log2 fit and its
  parametric-bootstrap slope CI are implemented in closed form. The subset draw /
  balance / write functions are importable from the script and unit-tested
  (`tests/test_subsets.py`).
