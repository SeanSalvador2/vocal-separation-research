# MASTER PLAN — Direction 06: Robust training — how much do bleeding targets actually hurt a compact separator, and what does a cheap defense recover?

**Status: pre-registered plan. Nothing trained. All GPU work is deferred ("RUN LATER").**
Self-contained by design; cross-references ([`research/LITERATURE.md`](research/LITERATURE.md),
[`../01-loss-function-study/MASTER_PLAN.md`](../01-loss-function-study/MASTER_PLAN.md),
[`../PLAN.md`](../PLAN.md)) add depth, not required context.

---

## 0. TL;DR

Real training stems are dirty: microphone bleed and mislabeled content contaminate the
"ground-truth" targets (the SDX'23 organizers built two whole datasets —
SDXDB23_LabelNoise / SDXDB23_Bleeding — to make the point, and flagged robust MSS as
under-studied). We corrupt MUSDB18's **training targets only** with a controlled
stem-bleed knob — vocals target ṽ = v + ε·a, accompaniment ã = (1−ε)·a, mixture exactly
unchanged — at ε ∈ {0, 5, 15, 30 %}, train the standard compact U-Net at each level,
and compare the measured clean-test degradation curve against a **closed-form
prediction**: a model that perfectly learns the corrupted conditional must leak exactly
ε·a, so its clean SI-SDR is analytically computable per track. Then we test a cheap,
classical defense — **trimmed loss** (train each step on the lowest-loss chunks;
ITLM/small-loss-trick lineage) — at the worst corruption level, plus a
does-it-hurt-clean-data control. **10 new GPU runs ≈ 14–19 T4-hours** (the ε = 0 cell
is the shared baseline cell from Directions 01/02/03, hash-asserted). The curve cannot
fail to exist; every branch — including "the model is implicitly robust" and "trimming
does nothing" — is pre-registered as a finding.

---

## 1. Problem framing

### 1.1 The question
Supervised separation assumes clean isolated targets. In practice stems come from real
sessions: drum mics pick up vocals, "vocals" stems carry instrument bleed, label errors
abound. SDX'23 (Fabbro et al., TISMIR 2024) made corrupted-training-data a first-class
research axis with simulated bleeding/label-noise datasets built from MoisesDB, but
published no *dose–response curve*: how much does a given bleed level actually cost a
compact model, is the damage shaped like the theory says, and does the oldest trick in
the noisy-label book (train on your smallest losses) buy anything back? (Gap verified
2026-07-13; recent adjacent work — blind data cleaning for MSS, arXiv 2510.15409 — is
cited, and does not report a controlled dose–response + mitigation study at compact
scale.)

### 1.2 Why this is a strong design
- **The treatment is a single continuous knob (ε)** applied by construction, so the
  curve is exact science on exactly the data we already have — no new datasets.
- **A falsifiable quantitative null model exists** (§2, "prediction line"): the
  perfectly-corruption-fitting model's clean SI-SDR is computable in closed form. The
  interesting result is the *gap* between measured and predicted — evidence for
  implicit robustness (measured above prediction) or optimization damage (below).
- **The mitigation has 40 years of theory** (trimmed estimators; ITLM's provable
  recovery; Arpit et al.'s memorization timing) but no MSS data point.

### 1.3 Fixed infrastructure (inherited, not re-decided)
Model = SingNet-C1 (9,835,745 params), `l1mag` loss, full augmentation recipe (remix +
gain + flip), STFT 4096/1024, 6-s mono chunks, AdamW + warmup/cosine, AMP, batch 16,
REDUCED = 16 k steps for every run, best-checkpoint by clean validation SI-SDR,
86/14/50 split. Only two new mechanisms enter: the corruption transform and the
trimmed-loss wrapper.

---

## 2. Pre-registered hypotheses & decision rules

Notation: $s(\varepsilon)$ = mean best-checkpoint **clean** validation vocals SI-SDR of
the arm trained at bleed level ε. σ_seed = pooled between-seed std over the three
3-seed cells (ε = 0, ε = 0.30, trimmed@0.30). $P(\varepsilon)$ = the **prediction
line**: per validation track, SI-SDR(v + εa, v) computed analytically from the clean
stems (the score of the estimator that leaks exactly ε of the accompaniment), averaged
over the 14 tracks; $P(\varepsilon) \approx 10\log_{10}\frac{\lVert v\rVert^2}{\varepsilon^2\lVert a\rVert^2}$
per track — a −20 log₁₀ε line anchored by each track's vocal/accompaniment energy
ratio (exact form in THEORY §3).

> **H-06a (dose–response).** Bleed hurts, monotonically and detectably:
> - **Supported** iff $s(0) - s(0.30) > \max(\sigma_{\text{seed}}, 0.5\text{ dB})$
>   and the four points are non-increasing in ε within ±σ_seed tolerance.
> - **Refuted (robust)** iff $s(0) - s(0.30) \le \sigma_{\text{seed}}$ — the model
>   shrugs off 30 % bleed (pre-registered as a *surprising positive* about mask-model
>   robustness, immediately checked against the prediction line, which would then sit
>   far below the measured curve).
> - **Mixed** otherwise (non-monotone middle beyond tolerance → investigate before
>   interpreting).
> **Shape read-out (descriptive, pre-registered):** the measured curve is compared to
> $P(\varepsilon)$: measured ≈ prediction (±σ_seed band) ⇒ the model faithfully learns
> the corrupted conditional (no implicit denoising); measured **above** ⇒ implicit
> robustness (architecture/loss/augmentation partially reject bleed); measured
> **below** ⇒ corruption additionally destabilizes optimization. This comparison — not
> the bare slope — is the scientific payload.

> **H-06b (cheap defense).** At the worst level, trimmed loss recovers a detectable
> fraction:
> - **Supported** iff $s_{\text{trim}}(0.30) - s(0.30) > \sigma_{\text{seed}}$.
>   Report the **recovery fraction**
>   $\rho = \frac{s_{\text{trim}}(0.30) - s(0.30)}{s(0) - s(0.30)}$ with a
>   seed-propagated CI (defined only under H-06a support; else "not evaluable").
> - **Refuted** iff the delta ≤ σ_seed (trimming buys nothing here — a clean negative
>   against the small-loss trick's transfer to *uniform* corruption).
> **Control (pre-registered):** trimming at ε = 0 must not cost more than σ_seed on
> clean data; if it does, the defense's price is part of the headline.

**Descriptive secondaries:** q-sensitivity (10 % vs 30 % trim); which chunks get
trimmed (logged accompaniment-energy statistics of kept vs dropped chunks — THEORY §5
predicts trimming selects low-⟨a⟩-energy chunks under uniform bleed, the mechanism by
which small-loss can work at all here); training-curve shapes (Arpit-style late
divergence between clean and corrupted arms).

---

## 3. Experimental design

### 3.1 The corruption model (pinned exactly; `singnet/data/corrupt.py`)

Applied **at the stem level, at load time, before augmentation**, to **training-split
tracks only** (an eval-split guard refuses to corrupt val/test rows — unit-tested):

$$\tilde v = v + \varepsilon\, a, \qquad \tilde a = (1 - \varepsilon)\, a,$$

with $a$ = the same track's accompaniment (sum of its non-vocal stems), ε constant per
arm. Properties (each unit-tested): mixture invariance ($\tilde v + \tilde a = v + a$
exactly — the corrupted stems still sum to the true mixture, the defining property of
real bleed and of SDXDB23_Bleeding's redistribution); determinism (no RNG); the
augmentation pipeline then operates on the *corrupted* stems (so a remixed training
example is $\tilde v_i + \tilde a_j$ — exactly what a practitioner remixing a
corrupted dataset would produce).

**Loudness decision (pre-registered):** the corrupted target's energy grows with ε —
real bleed adds energy; we do **not** renormalize (documenting the Phase-0 note's
concern): the per-chunk input standardization is mixture-side and unaffected, the
random-gain augmentation already randomizes source levels, and renormalizing would
silently convert "bleed" into "bleed + attenuation." The prediction line uses the same
un-normalized construction, so the comparison is internally consistent.

**Relation to SDX'23 (honest):** this is a controlled *simplification* of
SDXDB23_Bleeding (constant ε, accompaniment→vocals direction only) — a monotone knob
in exchange for realism; stated in every artifact.

### 3.2 The mitigation (pinned exactly; `singnet/losses/trimmed.py`)

**Trimmed loss** (ITLM instantiation): per optimizer step, compute per-chunk losses
$\ell_i$ over the batch, keep the $\lceil (1-q) B \rceil$ smallest, average only those
(gradients flow only through kept chunks). $q \in \{0.10, 0.30\}$; primary q = 0.30
(matched to the corruption's uniform presence — under uniform bleed, trimming selects
the chunks whose accompaniment happens to be quiet, i.e. the *effectively cleanest*
targets; THEORY §5 derives this selection mechanism, which is NOT the classic
"some samples are clean" setting — a deliberate, stated stress-test of the small-loss
trick). The wrapper is loss-agnostic (wraps any project loss exposing per-chunk
values — a small, backward-compatible extension of the loss contract).

### 3.3 Run matrix

| Arm | ε | trim q | Seeds | New runs |
|---|---|---|---|---|
| `clean` | 0 | — | {0,1,2} | **0** (shared cell ≡ D01 `l1mag` ≡ D02 `full` ≡ D03 `baseline`; hash-asserted) |
| `bleed05` | 0.05 | — | {0} | 1 |
| `bleed15` | 0.15 | — | {0} | 1 |
| `bleed30` | 0.30 | — | {0,1,2} | 3 |
| `trim30` | 0.30 | 0.30 | {0,1,2} | 3 |
| `trim30_q10` | 0.30 | 0.10 | {0} | 1 |
| `trim_clean` (control) | 0 | 0.30 | {0} | 1 |
| **Total new** | | | | **10** |

Contingency (pre-registered, budget-gated): if Direction 01's final verdict flips the
project default loss, re-run {`bleed30`, `trim30`} × 1 seed under the new loss (+2
runs) — the trimmed wrapper is loss-agnostic by construction.

### 3.4 Invariants
Identical across arms: architecture & init per seed, data order & augmentation streams
per seed ((seed, transform, step)-keyed; corruption is deterministic so it cannot
perturb any RNG stream — unit-tested), optimizer/schedule/steps/batch, clean
validation protocol. The ONLY differences: ε (data path) and q (loss path).

---

## 4. Data

MUSDB18 exactly as in Direction 01 (research-only, nothing committed; decoded WAV
shards; 86/14/50 with the verified validation list). **No new data.** Corruption is a
load-time transform on training-split stems only; **validation and test stems are
never corrupted** (guard unit-tested). The prediction line $P(\varepsilon)$ is computed
once from clean validation (and, at the end, test) stems — CPU-minutes, RUN LATER.

---

## 5. Training & evaluation protocol

Training config: D01 §7.1 verbatim; registry at
`06-robust-training/results/registry.csv` with appended columns
(`epsilon`, `trim_q`, `kept_fraction_observed`, `trim_energy_stats_path`).
Per-run trim telemetry: the loop logs, every 500 steps, the accompaniment-energy
distribution of kept vs dropped chunks (the §2 mechanism evidence; written as a small
CSV per run).

**Validation (decisions):** D01 §7.2 verbatim on **clean** stems.

**Test (exactly one pass):** the three 3-seed cells' best checkpoints (`clean`,
`bleed30`, `trim30` — 9 checkpoints) on the 50 **clean** test tracks; per-track CSV;
paired stats on the two pre-registered pairs (bleed30 − clean; trim30 − bleed30):
bootstrap 95 % CI + Wilcoxon. The single-seed curve/control arms are validation-scoped
(never tested).

**Analysis (in `singnet/analysis/bleed.py`, unit-tested):** prediction line
$P(\varepsilon)$ from stems (synthetic-fixture tests now: constructed v, a with known
energy ratios ⇒ exact expected values); curve assembly with σ_seed band; measured-vs-
predicted gap per ε; recovery fraction ρ with delta-method CI; trim-telemetry readers.

---

## 6. Compute budget & run book

| Item | Cost (T4 est.) |
|---|---|
| 10 runs × REDUCED | 13–18 h |
| Test pass (9 checkpoints) + prediction lines | ~1 h |
| Contingency | 0 or ~3 h |
| **Ceiling** | **≈ 14–19 (22) T4-h** |

**Run book (all RUN LATER; prerequisites: D01 data prep + G1 smoke done):**
```bash
# 1. dry-run (CPU): pipelines instantiate; ε/q per config printed; eval-guard checked
python scripts/run_sweep.py --direction 06 --dry-run          # [RUN LATER]
# 2. the 10 runs (GPU):
python scripts/run_sweep.py --direction 06 --stage reduced    # [RUN LATER]
# 3. prediction lines (CPU-minutes):
python -m singnet.analysis.bleed --split valid --epsilons 0.05 0.15 0.30  # [RUN LATER]
# 4. analysis freeze: notebook 02 §1–5 (curve + gap + mitigation on val).
# 5. single test pass (GPU minutes):
python scripts/evaluate.py --direction 06 --split test        # [RUN LATER]
```

---

## 7. Go / no-go gates

| Gate | When | Criterion | On fail |
|---|---|---|---|
| **G0** | now (CPU) | full-repo `pytest` green incl.: corruption exactness (ṽ, ã formulas; mixture invariance to float tolerance; determinism; **eval-split guard**); trimmed-loss analytics (constructed batch → exact kept set; q = 0 ≡ base; gradients only via kept chunks; loss-contract per-chunk path backward-compatible); prediction-line function on synthetic stems; config schema/hash (shared ε=0 cell ≡ D01 cell); registry appends | fix before GPU |
| **G1** | first GPU session | inherited (D01 smoke); D06 dry-run passes | debug configs |
| **G2** | after the three 3-seed cells | σ_seed < 1.0 dB; `bleed30` seeds did not diverge | pre-registered: +1 seed on the closest pair (+2 runs max) |
| **G3** | before test pass | registry complete (10 runs + telemetry CSVs); analysis frozen on val; corruption never touched val/test (grep + guard-log audit) | fix, re-freeze |

---

## 8. Deliverables & layout (this direction)

```
06-robust-training/
  MASTER_PLAN.md            ← this file
  THEORY.md + theory/theory.tex   ← stage B (§9)
  research/                 ← Phase-0 literature (done)
  notebooks/
    01_bleed_anatomy.ipynb           ← stage C (§10)
    02_robustness_experiments.ipynb
  configs/
    base.yaml; bleed{05,15,30}_seed*.yaml; trim30_seed{0,1,2}.yaml;
    trim30_q10_seed0.yaml; trim_clean_seed0.yaml;
    contingency_{bleed30,trim30}_sisdr_seed0.yaml
  results/                  ← registry.csv, telemetry CSVs, test CSVs, DEVIATIONS.md (empty now)
  paper/PAPER.md, SEAN-README.md  ← stage E
singnet/data/corrupt.py    ← stage D (tested)
singnet/losses/trimmed.py  ← stage D (tested; loss-contract per-chunk extension)
singnet/analysis/bleed.py  ← stage D (tested)
```

**Code contracts (stage-D deltas only; Directions 01–05 untouched semantically):**
- `singnet.data.corrupt.StemBleed(epsilon: float)` applied in the dataset between shard
  load and augmentation; constructor refuses non-train splits; `epsilon` recorded into
  the config hash (identity field).
- Loss contract extension: every project loss gains an optional per-chunk-values path
  (e.g. `forward(..., reduce: bool = True)` or per-chunk vector in `aux`) —
  backward-compatible (default behavior unchanged; existing tests must pass
  unmodified). `singnet.losses.trimmed.TrimmedLoss(base: SeparationLoss, q: float)`
  implements §3.2 and logs kept/dropped indices + per-chunk target-accompaniment
  energies to `aux` for telemetry.
- `singnet.analysis.bleed.prediction_line(stems, epsilons) -> DataFrame`,
  `measured_vs_predicted(...)`, `recovery_fraction(...)` (delta-method CI) — pure,
  tested.
- `run_sweep.py --direction 06`; registry appends per §5.

---

## 9. THEORY.md outline (stage-B spec)

1. **Corruption as continuous-target label noise**: the redistribution model, mixture
   invariance, relation to SDXDB23_Bleeding's construction (what we simplify and why);
   the loudness decision.
2. **What the corrupted-optimal model does**: for L1-magnitude training, derive the
   optimal mask under target ṽ = v + εa (per TF bin, mask* ≈ (|V + εA|)/(|X|)-flavored
   — derive properly with phase caveats); conclude the perfectly-fit model **leaks ε·a
   by construction**.
3. **The prediction line**: SI-SDR(v + εa, v) in closed form —
   $10\log_{10}\bigl(\lVert v\rVert^2 / (\varepsilon^2 \lVert a\rVert^2)\bigr)$ exactly
   when $v \perp a$ (state the orthogonality caveat and the exact projection form
   without it; both implemented); per-track anchoring; why the −20 log₁₀ε slope is the
   "no implicit robustness" null and what above/below gaps mean.
4. **Trimmed estimators**: ITLM's objective and recovery guarantee (sketch, with its
   assumptions honestly compared to ours — *sample-level* noise there vs *uniform*
   bleed here); Arpit et al.'s memorization timing as the reason small-loss selection
   contains signal early.
5. **The selection mechanism under uniform bleed**: per-chunk corrupted-target loss
   decomposes so that high-⟨a⟩-energy chunks carry systematically higher irreducible
   loss ⇒ trimming ranks by accompaniment energy ⇒ the defense is effectively
   *curriculum-by-cleanliness*; derive, and state the testable telemetry signature
   (kept chunks' ⟨a⟩-energy < dropped chunks').
6. **Statistics**: pooled σ_seed across three cells; recovery-fraction CI (delta
   method); the two pre-registered paired tests; multiple-comparison posture (two
   primary pairs only).
7. `theory/theory.tex`: compilable standalone mirror.

## 10. Notebook specs (stage-C spec)

Global rules identical to prior directions (un-run; RUN-LATER banners with §6
runtimes; logic in `singnet/`; Colab bootstrap; cross-link D01 data prep).
- **01_bleed_anatomy.ipynb**: what ε-bleed looks and sounds like (waveform/spectrogram
  overlays at each ε; audio cells RUN LATER); the mixture-invariance property
  demonstrated; the corrupted-optimal-model argument in pictures; the prediction line
  rendered on synthetic stems (CPU-runnable later); the trimming mechanism preview
  (constructed batch, which chunks get dropped).
- **02_robustness_experiments.ipynb**: pre-registration recap verbatim (§2); run
  matrix + shared-cell accounting; launch cells (RUN LATER); analysis: the
  dose–response curve with σ_seed band **and the prediction line overlaid** (the
  headline figure), measured-vs-predicted gap per ε, mitigation deltas + recovery
  fraction ρ with CI, trim-cost-on-clean control, q-sensitivity, trim telemetry
  (kept-vs-dropped accompaniment-energy distributions vs the THEORY §5 signature);
  §12 interpretation branches as labeled stubs; conclusions + cross-links (Direction
  05's LoRA-noise-robustness bridge 2602.00084 as future work; Direction 08's
  silence-energy machinery reused by the telemetry).

---

## 11. Risks & mitigations (direction-specific)

- **Trimming under uniform corruption may genuinely do nothing** (the classic trick
  assumes clean samples exist) → this is a pre-registered honest negative with a
  mechanism story; the telemetry decides *why* (no energy-separation in kept/dropped ⇒
  the selection signal never existed).
- **Batch-level trimming interacts with batch size** (q of 16 chunks = coarse) →
  documented; kept-fraction observed is logged; q = 0.30 ⇒ keep 11/16 (stated).
- **Curve too flat to interpret** (model robust) → prediction-line comparison turns
  "flat" into a *positive, quantified* robustness claim rather than a null.
- **Non-monotone middle points** (single-seed ε = 0.05/0.15) → tolerance band in the
  decision rule; G2 escalation bounded.
- **Corruption leaking into eval** (the fatal bug class) → constructor guard +
  grep-audit at G3 + the prediction line's independence (computed from clean stems
  only).
- **AMP interaction with per-chunk losses** (fp16 underflow reordering the trim
  ranking) → per-chunk losses computed in fp32 (existing loss-contract rule);
  ranking-stability unit test at extreme magnitudes.

---

## 12. Outcome interpretation matrix (pre-registered; full prose in paper/PAPER.md)

| Outcome | Reading | Consequence |
|---|---|---|
| H-06a supported, measured ≈ prediction | the model faithfully learns whatever target you give it — data quality is a hard constraint, quantified in dB per ε | data-cleaning (and Direction 10's teacher-quality question) gains a measured stake; the curve becomes a calibration chart for "how clean must stems be" |
| H-06a supported, measured **above** prediction | implicit robustness: architecture/loss/augmentation partially reject bleed | localize the source (remix? sigmoid mask cap?) as named future work; good news for practitioners with mildly dirty data |
| H-06a supported, measured **below** prediction | corruption also destabilizes training (optimization damage beyond the information limit) | check curves/grad norms; robust-loss (GCE-style) becomes the follow-up mitigation |
| H-06a refuted (robust/flat) | 30 % bleed within noise — mask models at this scale are surprisingly corruption-tolerant | verify against prediction line (which must sit well below); strong, surprising, publishable |
| H-06b supported (ρ > 0 detectably) | the small-loss trick transfers to uniform bleed via energy-selection — a cheap, general defense | adopt trimming as an optional training flag project-wide; report ρ and the telemetry mechanism |
| H-06b refuted | trimming fails without sample-level cleanliness — an honest boundary for the classic trick | recommend robust losses / cleaning instead; telemetry explains the failure |
| Control fires (trim hurts clean > σ_seed) | the defense has a price on clean data | report the tradeoff; recommend trimming only under suspected corruption |

---

## 13. Timeline (Sean's part-time weeks, once GPU exists; after D01 data prep)

| Week | Work | Gate |
|---|---|---|
| 1 | G0 green (now); dry-run; the three 3-seed cells (6 new runs) | G0, G1, G2 |
| 2 | remaining 4 runs; prediction lines; freeze analysis; test pass | G3 |
| 3 | fill paper; verdicts | — |

---

*Frozen on 2026-07-13. Deviations during execution go to `results/DEVIATIONS.md` with
date + reason. (One divergence from the Phase-0 sketch is pre-declared here rather
than deviated later: the trim grid is q ∈ {10, 30}% — two points chosen for budget —
instead of the sketch's {5, 10, 20}%.)*
