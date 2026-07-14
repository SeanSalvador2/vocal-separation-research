# MASTER PLAN — Direction 08: The silence problem — chunk-sampling policies and the cost of quiet, measured by a metric the field doesn't have

**Status: pre-registered plan. Nothing trained. All GPU work is deferred ("RUN LATER").**
Self-contained by design; cross-references ([`research/LITERATURE.md`](research/LITERATURE.md),
[`research/papers/silence-metrics-grounding.md`](research/papers/silence-metrics-grounding.md),
[`../01-loss-function-study/MASTER_PLAN.md`](../01-loss-function-study/MASTER_PLAN.md))
add depth, not required context.

---

## 0. TL;DR

Karaoke's most audible failure is **ghost vocals**: energy leaking into the vocal
estimate during instrumental passages. The field cannot even *measure* this failure:
museval **drops silent-reference frames** (NaN — verified from its source), and SI-SDR
is **singular** on a silent target. We (1) define and ship a ~50-line, unit-tested
metric — **SLR, the Silence-Leakage Ratio** — that measures exactly this blind spot;
(2) run the controlled experiment the training-lore never got: **four chunk-sampling
policies** (uniform / energy-weighted / drop-silent / curriculum), identical everything
else, scored on **both** overall vocals SI-SDR **and** SLR. Pre-registered bets:
energy-weighted sampling buys overall quality (H-08a), and fully dropping silent chunks
poisons silence behavior (H-08b) — a real **quality-vs-leakage tradeoff**, drawn as a
two-metric plane with anchors (do-nothing = 0 dB SLR; oracle ≈ −∞). Novelty framing is
honest and pre-audited: BSMamba2 (arXiv 2508.14556) attacks the same silence problem
via *architecture*; **the metric + the policy-vs-leakage tradeoff are ours** (gap-check
logged 2026-07-13). The uniform arm is the project's shared baseline cell, so this
direction adds only **8 new GPU runs ≈ 11–15 T4-hours**.

---

## 1. Problem framing

### 1.1 Two entangled questions, separated by design
1. **Sampling:** MUSDB tracks have long vocal-silent stretches. Training chunks are
   drawn *somehow* — UMX draws uniformly at random (code-verified; that lore is our
   baseline). Should sampling favor vocal-active regions? The intuition says yes
   (more signal per step); the karaoke intuition says careful (a model that never sees
   silence never learns to output it).
2. **Measurement:** whatever the policy does to *silence behavior* is invisible to the
   standard metrics: museval NaNs silent-reference frames out of the median; SI-SDR's
   projection divides by ‖v‖² = 0. A model can ace SDR while haunting every
   instrumental break. Direction 08's product motivation (StemCraft karaoke) makes
   this blind spot the headline.

### 1.2 Novelty, honestly bounded (pre-audited in Phase 0)
Adjacent prior exists and is cited: BSMamba2 ("Mamba2 Meets Silence", 2508.14556,
Aug 2025) targets sparse-vocal robustness via long-range architecture; Demucs
augmentation *injects* silence; BSRNN uses an activity detector for pseudo-label
mining (not for labeled-data sampling). **No source defines a silence-leakage metric
or reports a sampling-policy-vs-leakage tradeoff** — that is the defensible gap
(gap-check queries + results logged in the shared VERIFICATION_LOG). The claim we will
publish is exactly that, no more.

### 1.3 Fixed infrastructure (inherited, not re-decided)
Model = SingNet-C1, `l1mag`, full augmentation recipe (remix + gain + flip, stream-
keyed), STFT 4096/1024, 6-s mono chunks, AdamW + warmup/cosine, AMP, batch 16,
REDUCED = 16 k steps, best-checkpoint by validation vocals SI-SDR, 86/14/50 split.
The ONLY thing that varies here: **where chunks come from** (the sampling policy).

---

## 2. The SLR metric (pre-registered definition — the direction's core deliverable)

### 2.1 Silent-region identification (from ground-truth vocal stems)
Frame-wise RMS of the GT vocal stem: frames of **100 ms** with **50 ms hop**; a frame
is *silent* iff RMS < θ, **θ = −60 dBFS** (primary; sensitivity report at
θ ∈ {−50, −60, −70}); silent frames form **runs**, and only runs of length
**L_min = 0.5 s** or more enter the silent-region set $R_{\text{sil}}$ (short gaps are
breaths/rests, not instrumental passages). The −60 dBFS constant is deliberately shared
with Direction 01's `sisdr` silent-target guard (one project-wide notion of "silent").

### 2.2 Definition
Per track, with $\hat v$ = estimated vocals, $x$ = mixture, over the samples of
$R_{\text{sil}}$:

$$\text{SLR} = 10\log_{10}\frac{\sum_{t \in R_{\text{sil}}}\hat v(t)^2 + \epsilon}{\sum_{t \in R_{\text{sil}}} x(t)^2 + \epsilon},\qquad \epsilon = 10^{-8}.$$

Lower is better. **Anchors (what makes it readable):** a do-nothing separator
($\hat v = x$) scores **exactly 0 dB**; a perfect separator ($\hat v = 0$ in silence)
scores at the ε floor; a separator passing 10 % of mixture energy scores −10 dB.
Tracks with empty $R_{\text{sil}}$ are NaN and excluded (mirrors museval's convention);
the **valid-n is always reported**. Aggregate = mean over valid tracks (+ per-track
distribution in figures).

### 2.3 Properties (derived in THEORY §3, tested in code)
Monotone in leaked silent-region energy; invariant to global gain applied to both
$\hat v$ and $x$; well-defined for every estimator including silence (no singularity —
the ε guards the log, not a projection); θ/L_min sensitivity is an explicit report
axis, not a hidden constant. Unit tests construct synthetic tracks with known injected
leakage and assert SLR to numerical precision, plus edge cases (no silent regions →
NaN; all-silent; threshold boundary; run-merging at L_min).

---

## 3. Pre-registered hypotheses & decision rules

Notation: $q(\text{policy})$ = mean best-checkpoint validation vocals SI-SDR;
$\ell(\text{policy})$ = mean validation SLR (θ = −60, valid tracks). Two pooled seed
noises from the three 3-seed cells (uniform, energy, drop):
$\sigma_{\text{seed}}^{q}$ for SI-SDR and $\sigma_{\text{seed}}^{\ell}$ for SLR.

> **H-08a (quality).** Energy-weighted sampling beats uniform on overall quality:
> $q(\text{energy}) - q(\text{uniform}) > \sigma_{\text{seed}}^{q}$.
> Refuted iff ≤; each direction pre-written (§13).

> **H-08b (the cost of dropping silence).** Discarding silent chunks poisons silence
> behavior: $\ell(\text{drop}) - \ell(\text{uniform}) > \sigma_{\text{seed}}^{\ell}$
> (drop leaks detectably more). Refuted iff ≤ — i.e. the model outputs silence "for
> free" without ever training on it (a genuinely interesting negative about mask
> models; pre-written).

**The headline exhibit (pre-registered format):** the **(SI-SDR, SLR) plane** — four
policy points with per-seed scatter, plus the do-nothing (0 dB SLR) and oracle-IRM
anchors. **We pre-commit to NOT scalarizing** the two axes into one score; the
tradeoff, if it exists, is the finding. Curriculum's pre-registered role: does an
anneal capture energy's quality gain *without* drop's leakage cost (dominates both on
the plane / sits between / dominates neither — three pre-written readings).

**Descriptive secondaries:** θ-sensitivity of every conclusion ({−50, −60, −70});
per-policy exposure statistics (fraction of silent chunks actually seen during
training — logged, closing the loop on the THEORY §4 exposure predictions); SLR of
Direction 01's arms at ε... (cross-feed: the `sisdr`-loss guard's skip rate vs SLR,
descriptive); museval SDR secondary table (expected ≈ blind to the policy differences
on silence — itself worth one line if confirmed).

---

## 4. Experimental design

### 4.1 The four policies (pinned exactly; `singnet/data/sampling.py`)

With per-track windowed vocal-energy profiles $E(s)$ (RMS of the GT vocal stem over
the 6-s window starting at $s$, computed on a 1-s start grid at data-prep time):

| Policy | Chunk-start distribution | Constants |
|---|---|---|
| `uniform` | uniform over valid starts (the UMX-lore baseline; **bit-identical to the shared project baseline cell**) | — |
| `energy` | $p(s) \propto (1-\lambda)\frac{E(s)}{\sum E} + \lambda\frac{1}{N}$ — energy-proportional with a uniform floor (silent chunks are down-weighted, **never excluded**) | λ = 0.1 |
| `drop` | uniform over starts whose window vocal RMS ≥ θ; silent-window starts excluded entirely | θ = −60 dBFS |
| `curriculum` | the `energy` form with λ annealed 1.0 → 0.1 linearly over the first 50 % of steps, then held (starts uniform, ends energy) | λ(t) pinned |

Policy draws use a dedicated (seed, "sampling", step) RNG stream (the established
stream design), so switching policies cannot perturb augmentation streams
(unit-tested). Remix partners' chunks are drawn by the **same** policy (the policy
defines "what the model sees", uniformly applied to all sources; stated).

### 4.2 Run matrix

| Arm | Seeds | New runs |
|---|---|---|
| `uniform` | {0,1,2} | **0** (shared cell ≡ D01 `l1mag` ≡ D02 `full` ≡ D03 `baseline` ≡ D06 `clean`; hash-asserted) |
| `energy` | {0,1,2} | 3 |
| `drop` | {0,1,2} | 3 |
| `curriculum` | {0,1} | 2 |
| **Total new** | | **8** |

Contingency (pre-registered, budget-gated): if Direction 01 flips the project default
loss, re-run {uniform, energy} × 1 seed under the new loss (+2) — noting the known
interaction (the `sisdr` loss's silent-target guard already performs a soft `drop`
inside the loss; the policy × guard interplay is then part of the sensitivity note).

### 4.3 Invariants
Identical across arms: architecture & init per seed, augmentation streams per seed,
loss, optimizer/schedule/steps/batch, validation protocol. ONLY the chunk-start
distribution differs (and its schedule, for curriculum).

---

## 5. Data

MUSDB18 as in Direction 01 (research-only, nothing committed). New prep artifact:
**per-track vocal-energy profiles** on the 1-s start grid (extends the existing
prep-time energy index; `scripts/prepare_data.py --write-energy-profiles`, CPU-minutes,
RUN LATER; synthetic-fixture tests now). Silent-region sets for val/test SLR are
computed from GT stems at evaluation time by the tested `slr` module (no caching
needed; deterministic).

---

## 6. Training & evaluation protocol

Training config: D01 §7.1 verbatim; registry at
`08-silence-leakage/results/registry.csv` with appended columns
(`policy`, `theta_db`, `floor_lambda`, `silent_exposure_observed`, `best_val_slr`).
Per-run exposure telemetry: every 500 steps, the fraction of drawn chunks whose target
vocal is silent (θ) — the policy's realized behavior, not just its definition.

**Validation (decisions):** D01 §7.2 (mean vocals SI-SDR) **plus SLR** (θ = −60) on
the 14 val tracks at every eval point; best-checkpoint selection stays on SI-SDR
(pre-registered: selection must not see SLR, so the leakage comparison is not
selection-biased; SLR of the SI-SDR-best checkpoint is what H-08b judges).

**Test (exactly one pass):** best checkpoints of the three 3-seed cells (9 ckpts) +
curriculum's 2 + do-nothing + oracle IRM on the 50 test tracks: vocals/accomp SI-SDR,
**SLR at all three θ**, museval SDR secondary. Committed CSVs; paired bootstrap +
Wilcoxon on the two pre-registered pairs ((energy − uniform) on SI-SDR;
(drop − uniform) on SLR).

**Analysis:** the (SI-SDR, SLR) plane figure; θ-sensitivity grid; exposure-vs-outcome
table (policy exposure → SLR, the mechanism link); valid-n reporting throughout.

---

## 7. Compute budget & run book

| Item | Cost (T4 est.) |
|---|---|
| 8 runs × REDUCED | 10–14 h |
| Test pass (11 ckpts + anchors, incl. SLR) | ~1 h |
| Energy-profile prep | CPU-minutes |
| Contingency | 0 or ~3 h |
| **Ceiling** | **≈ 11–15 (18) T4-h** |

**Run book (all RUN LATER; prerequisites: D01 data prep + G1 smoke done):**
```bash
# 0. energy profiles (CPU, once):
python scripts/prepare_data.py --write-energy-profiles --out $SHARD_ROOT   # [RUN LATER]
# 1. dry-run (CPU): policies instantiate; sampling-weight sanity printed per config
python scripts/run_sweep.py --direction 08 --dry-run                       # [RUN LATER]
# 2. the 8 runs (GPU):
python scripts/run_sweep.py --direction 08 --stage reduced                 # [RUN LATER]
# 3. analysis freeze (CPU): notebook 02 §1–5 (plane figure on val).
# 4. single test pass (GPU minutes):
python scripts/evaluate.py --direction 08 --split test --slr               # [RUN LATER]
```

---

## 8. Go / no-go gates

| Gate | When | Criterion | On fail |
|---|---|---|---|
| **G0** | now (CPU) | full-repo `pytest` green incl.: SLR analytics (known-leakage constructions exact; NaN/valid-n; θ/L_min edges; anchors: v̂=x → 0 dB, v̂=0 → ε-floor); silent-region identification (runs, L_min, merging); policy distributions (exact weights for constructed profiles; drop's support; curriculum schedule values; floor never zero for `energy`); stream independence (policy ↔ augmentation); config hash-stability + uniform ≡ shared-cell hash | fix before GPU |
| **G1** | first GPU session | inherited (D01 smoke); D08 dry-run passes; energy profiles written & spot-checked | debug prep/policies |
| **G2** | after the two new 3-seed cells | both σ_seed's computed; σ_seed^q < 1.0 dB (else pre-registered escalation: +1 seed to the closest quality pair, +2 runs max); at least 10/14 val tracks have valid SLR (else θ sensitivity check before proceeding) | escalate per rule |
| **G3** | before test pass | registry + exposure telemetry complete (8 runs); analysis frozen on val; selection never used SLR (audit the registry's best-checkpoint fields) | fix, re-freeze |

---

## 9. Deliverables & layout (this direction)

```
08-silence-leakage/
  MASTER_PLAN.md            ← this file
  THEORY.md + theory/theory.tex   ← stage B (§10)
  research/                 ← Phase-0 literature (done)
  notebooks/
    01_silence_anatomy.ipynb        ← stage C (§11)
    02_sampling_experiments.ipynb
  configs/
    base.yaml; energy_seed{0,1,2}.yaml; drop_seed{0,1,2}.yaml;
    curriculum_seed{0,1}.yaml; contingency_{uniform,energy}_sisdr_seed0.yaml
  results/                  ← registry.csv, exposure CSVs, test CSVs, DEVIATIONS.md (empty now)
  paper/PAPER.md, SEAN-README.md  ← stage E
singnet/metrics/slr.py     ← stage D (tested; ~the famous 50 lines + tests)
singnet/data/sampling.py   ← stage D (tested)
scripts/prepare_data.py    ← stage D extension (--write-energy-profiles)
```

**Code contracts (stage-D deltas only; Directions 01–06 untouched semantically):**
- `singnet.metrics.slr.silent_regions(vocal_wave, sr, theta_db=-60, frame_s=0.1,
  hop_s=0.05, min_run_s=0.5) -> list[(start, end)]` and
  `slr(est_wave, mix_wave, regions, eps=1e-8) -> float | nan`, plus
  `slr_report(est, mix, vocal_gt, sr, thetas=(-50,-60,-70)) -> dict` — pure,
  ~50 lines, exhaustively tested.
- `singnet.data.sampling.ChunkSampler(policy, energy_profile, seed, *, theta_db,
  floor_lambda, schedule)` with the four §4.1 policies; consumed by `MusdbChunks`
  via an optional `sampling:` config block (optional-with-defaults = uniform, so
  every existing config hash is unchanged — tested; uniform arm hash ≡ shared cell).
- Eval integration: `evaluate.py --slr` computes SLR (all θ) alongside SI-SDR;
  oracle/do-nothing anchors included.
- `run_sweep.py --direction 08`; registry/telemetry appends per §6.

---

## 10. THEORY.md outline (stage-B spec)

1. **The silence problem formalized**: vocal-activity statistics of MUSDB (from the
   prep index; expected activity ratios per EDA); training exposure as a distribution
   over (chunk, activity) induced by each policy; the karaoke failure mode (why
   leakage in silence is perceptually catastrophic while metric-invisible).
2. **The measurement gap, from primary sources**: museval's silent-frame NaN handling
   (code-verified quote), SI-SDR's singularity (one-line derivation from the
   projection form), why SAR/SIR inherit the same blindness — the precise sense in
   which no standard metric can rank two systems by ghost-vocal behavior.
3. **SLR**: definition, anchors, invariances (joint-gain), monotonicity in leaked
   energy, ε's role (floor, not projection guard), θ/L_min sensitivity analysis
   (what moves when they move), estimator properties (valid-n, per-track vs pooled
   aggregation), and honest limitations (blind to *timbre* of the leakage; sample-
   domain, so phase-invisible; a 12 dB-down hi-hat and 12 dB-down vocal ghost score
   identically — the listening check's remaining role).
4. **The four policies as distributions**: exact forms; expected silent-chunk exposure
   per policy given an activity profile (closed-form for uniform/drop, computable for
   energy/curriculum); the mechanism hypotheses — drop ⇒ zero exposure ⇒ the model
   never fits the "output silence" mode; energy ⇒ reduced-but-nonzero exposure; what
   the sigmoid-mask parameterization does with unseen regimes (why leakage, not
   random noise, is the predicted failure).
5. **Curriculum**: the λ(t) schedule; the exposure integral over training; why
   "uniform early, energy late" is the theoretically-motivated order (fit the easy
   silence mode first, then concentrate on the hard active mode) — stated as
   motivation, not established fact.
6. **Statistics**: two metrics, two pooled σ_seed's, two pre-registered pairs (no
   scalarization; no correction across the two hypotheses — jointly pre-registered);
   the plane-figure semantics; selection-bias guard (checkpoint selection blind to
   SLR).
7. `theory/theory.tex`: compilable standalone mirror.

## 11. Notebook specs (stage-C spec)

Global rules identical to prior directions (un-run; RUN-LATER banners with §7
runtimes; logic in `singnet/`; Colab bootstrap; cross-link D01 data prep).
- **01_silence_anatomy.ipynb**: how much silence MUSDB vocals actually contain
  (activity-profile figures; RUN LATER on real data, synthetic demo cells now); the
  silent-region identifier walked through (θ, L_min, runs — on constructed audio);
  SLR anchors demonstrated numerically (do-nothing → 0 dB; perfect → floor; 10 %
  leak → −10 dB); the four policies' sampling-weight profiles drawn over one track;
  the measurement-gap story (museval NaN quote + SI-SDR singularity) told visually.
- **02_sampling_experiments.ipynb**: pre-registration recap verbatim (§3); run matrix
  + shared-cell accounting; launch cells (RUN LATER); analysis: **the (SI-SDR, SLR)
  plane** with seed scatter + anchors (headline), θ-sensitivity grid, exposure
  telemetry vs outcome, per-policy curves, test pass + paired stats; §13
  interpretation branches as labeled stubs; conclusions + product recommendation
  cell for StemCraft (which policy ships in SingNet training).

---

## 12. Risks & mitigations (direction-specific)

- **θ/L_min define the result** → pre-registered defaults + mandatory three-θ
  sensitivity on every conclusion; the shared −60 dBFS constant with D01 avoids
  intra-project inconsistency.
- **Val tracks with no valid silent regions** shrink SLR's n → valid-n gate in G2;
  test-set n reported; per-track distributions shown, not just means.
- **`energy` accidentally excluding silence** (float underflow of the floor) → the
  λ-mixture form guarantees mass ≥ λ/N per start; unit-tested lower bound.
- **Selection bias via SLR** → checkpoint selection is SI-SDR-only (G3 audit).
- **Curriculum confound** (schedule interacts with LR decay) → stated; curriculum is
  a secondary arm with 2 seeds; no primary hypothesis rests on it.
- **Leakage ≠ audibility** (THEORY §3 limitation) → the umbrella plan's listening
  check covers the perceptual claim; SLR claims are energy claims, worded as such.
- **Selection-on-quality could mask late silence regressions** (best-SI-SDR checkpoint
  might precede SLR drift) → best-vs-final checkpoint SLR both logged (descriptive).

---

## 13. Outcome interpretation matrix (pre-registered; full prose in paper/PAPER.md)

| Outcome | Reading | Consequence |
|---|---|---|
| H-08a + H-08b supported (the predicted tradeoff) | sampling is a real lever with a real price; the plane shows the frontier | ship `energy` (quality) unless karaoke-critical, then `uniform`/`curriculum`; SLR enters the project's standard eval battery |
| H-08a supported, H-08b refuted | free lunch: focus sampling on vocals, silence behavior survives anyway (mask prior suffices) | ship `energy` everywhere; the "never seen silence" fear is retired for mask models at this scale |
| H-08a refuted, H-08b supported | sampling doesn't buy quality but dropping still poisons silence — pure downside risk | ship `uniform`; publish the caution against activity filtering |
| Both refuted | chunk sampling is a non-lever at this scale/budget (16 k steps may see enough of everything) | honest null; exposure telemetry says whether policies even differed in practice (the diagnostic) |
| Curriculum dominates the plane | annealed exposure captures both goods | ship `curriculum`; flag schedule sensitivity as future work |
| Curriculum dominated | added complexity, no win | one-line negative |
| SLR invalid on too many tracks (G2 fail path) | MUSDB's silences are rarer/shorter than assumed | θ/L_min sensitivity becomes primary; metric definition revised only via DEVIATIONS.md with reasons |

---

## 14. Timeline (Sean's part-time weeks, once GPU exists; after D01 data prep)

| Week | Work | Gate |
|---|---|---|
| 1 | G0 green (now); profiles + dry-run; the two new 3-seed cells (6 runs) | G0, G1, G2 |
| 2 | curriculum runs; freeze analysis; test pass | G3 |
| 3 | fill paper; verdicts; StemCraft recommendation | — |

---

*Frozen on 2026-07-13. Deviations during execution go to `results/DEVIATIONS.md` with
date + reason.*
