# MASTER PLAN — Direction 02: How far do 100 songs actually go? Augmentation factorization + data-scaling curves

**Status: pre-registered plan. Nothing trained. All GPU work is deferred ("RUN LATER").**
Self-contained by design: an independent reader or agent can execute this study from this
file alone. Cross-references ([`research/LITERATURE.md`](research/LITERATURE.md),
[`../PLAN.md`](../PLAN.md), [`../01-loss-function-study/MASTER_PLAN.md`](../01-loss-function-study/MASTER_PLAN.md))
add depth, not required context. This direction is the umbrella plan's **H2, deepened**.

---

## 0. TL;DR

Two controlled question blocks sharing one training stack (the Direction-01 U-Net and
pipeline, loss fixed to `l1mag`):

1. **Factorization (leave-one-out):** train with the full standard augmentation recipe
   (cross-song **remix**, per-source **gain** U(0.25, 1.25), **sign flip** p = 0.5), then
   remove exactly one transform at a time, then all — 5 configurations. Pre-registered
   bet: **remix is the single highest-leverage transform, by more than seed noise.**
2. **Data scaling:** train with the full recipe on nested subsets of
   **{21, 43, 64, 86} songs** at identical step budget. Pre-registered bet: **the curve
   is still rising at 86 songs** (MUSDB-only training is data-starved, not saturated) —
   measured as "the last doubling (43 → 86) gains more than seed noise."

**12 training runs** at the Direction-01 REDUCED budget (16 k steps), 3 seeds on the two
cells that anchor the noise band (FULL-86 and n=21), 1 seed elsewhere; **one** test-set
pass (the FULL-86 seeds only). Budget ceiling **≈ 16–25 T4-GPU-hours**. Every outcome —
"remix dominates," "augmentation barely matters," "the curve plateaus" — is a
pre-registered, reportable finding.

---

## 1. Problem framing

### 1.1 The question
The standard MSS augmentation recipe (remix / gain / channel-swap / flip) is copied from
repo to repo — we code-verified the exact transforms and constants in both Open-Unmix
(`data.py`) and Demucs (`augment.py`) — yet **no published work factorizes what each
transform is worth** for a compact model on MUSDB-only data, and none pairs that with a
**data-scaling curve** answering the question every small-data practitioner actually has:
*would more songs help more than more augmentation?* The closest prior, "Singing voice
separation: a study on training data" (arXiv 1906.02618, 2019), studies data amount/type
but not a per-transform factorization with seed bands at compact scale (gap-check logged
2026-07-13 in [`../00-shared-research/VERIFICATION_LOG.md`](../00-shared-research/VERIFICATION_LOG.md)).

### 1.2 Why it matters here
- **For the project:** every remaining direction trains on this recipe; knowing which
  transforms carry the value (and how data-starved we are) calibrates all of them.
  The umbrella plan pre-registered this as **H2**; this direction is its full treatment.
- **For the portfolio:** "I measured the value of data vs augmentation before asking for
  more of either" is data-centric ML as practiced in industry.

### 1.3 Fixed infrastructure (inherited, not re-decided)
Model = SingNet-C1 (9,835,745 params, [`../01-loss-function-study/MASTER_PLAN.md §5`](../01-loss-function-study/MASTER_PLAN.md)).
Loss = `l1mag` (the project default; contingency in §3.4 if Direction 01 later
overturns it). STFT 4096/1024 Hann; 6-s mono chunks, uniform random start; AdamW 1e-3,
warmup 500 + cosine; AMP with fp32 losses; batch 16; REDUCED = 16,000 steps for every
run in this direction (identical budget ⇒ the *data*, not compute, varies).

---

## 2. Pre-registered hypotheses & decision rules

Let val-SI-SDR(c) = mean full-track vocals SI-SDR over the 14 validation tracks for
config c (final-vs-best checkpoint both logged; decisions use **best**, as in D01).
Let **σ_seed** = pooled between-seed standard deviation from the two 3-seed cells
(FULL-86 and n21), pooled as $\sigma_{\text{seed}}=\sqrt{(s^2_{\text{FULL}}+s^2_{\text{n21}})/2}$.

> **H-02a (remix dominance).** Define per-transform contributions
> $\Delta_t = \overline{\text{val}}(\text{FULL}) - \text{val}(\text{FULL} \setminus t)$
> for $t \in$ {remix, gain, flip}.
> - **Supported** iff $\Delta_{\text{remix}} > \max(\Delta_{\text{gain}}, \Delta_{\text{flip}}) + \sigma_{\text{seed}}$
>   **and** $\Delta_{\text{remix}} > \sigma_{\text{seed}}$.
> - **Refuted** iff $\Delta_{\text{remix}} \le \max(\Delta_{\text{gain}}, \Delta_{\text{flip}})$
>   (remix is not the top lever) **or** $\Delta_{\text{remix}} \le \sigma_{\text{seed}} \wedge \Delta_{\text{total}} \le \sigma_{\text{seed}}$
>   (augmentation as a whole is worthless at this budget — see descriptive quantities).
> - **Mixed** otherwise (e.g. remix on top but within the band).
> This is a **single pre-registered contrast** (remix vs the best of the others), so no
> multiple-comparison correction is applied to the primary verdict; the full Δ table is
> descriptive.

> **H-02b (unsaturated scaling).** With the full recipe and nested subsets
> $N \in \{21, 43, 64, 86\}$:
> - **Supported** iff $\overline{\text{val}}(86) - \text{val}(43) > \sigma_{\text{seed}}$
>   (the last measured **doubling** of data still buys more than run-to-run noise) and
>   the four-point sequence is non-decreasing within $\pm\sigma_{\text{seed}}$ tolerance.
> - **Refuted (plateau)** iff $\overline{\text{val}}(86) - \text{val}(43) \le \sigma_{\text{seed}}/2$
>   — at this capacity/budget, MUSDB's data volume is no longer the binding constraint.
> - **Mixed** otherwise (borderline gain, or a non-monotone middle worse than the
>   tolerance — see §12 subset-sensitivity risk).

**Descriptive quantities (no hypotheses, reported with CIs):**
$\Delta_{\text{total}} = \overline{\text{val}}(\text{FULL}) - \text{val}(\text{NONE})$
(total recipe value); the interaction gap
$\Delta_{\text{total}} - \sum_t \Delta_t$ (transform synergy/redundancy); the log-fit
slope $b$ in $\widehat{\text{SISDR}}(N) = a + b \log_2 N$ (dB per doubling of songs,
least-squares on the four means, parametric-bootstrap CI using σ_seed as per-point
noise); the same-loss D01 FULL-86 arm (`l1mag_seed*_reduced`) cross-check — Direction
01's `l1mag` sweep cell **is** this direction's FULL-86 cell (§3.3).

**Scope note (pre-registered deviation from the RESEARCH_DIRECTIONS #2 sketch):** the
menu sketch mentioned pitch/tempo among candidate transforms. Our frozen recipe
(D01 §4.5, PLAN Phase 2) deliberately excludes pitch/tempo (heavyweight, separate code
path even in Demucs) and channel-swap (inapplicable to the mono pipeline — documented,
not silently dropped). The factorization covers exactly the three transforms the
project actually trains with.

---

## 3. Experimental design

### 3.1 Factorization block (LOO) — all on the full 86-song train split

| Config | remix | gain | flip | Seeds | Runs |
|---|---|---|---|---|---|
| `full` | ✓ | ✓ | ✓ | {0, 1, 2} | **0 new** (≡ D01 `l1mag` sweep cell, §3.3) |
| `no_remix` | ✗ | ✓ | ✓ | {0} | 1 |
| `no_gain` | ✓ | ✗ | ✓ | {0} | 1 |
| `no_flip` | ✓ | ✓ | ✗ | {0} | 1 |
| `none` | ✗ | ✗ | ✗ | {0} | 1 |

Without remix, a training example is the **true track mixture** (sum of that track's own
stems — identical additivity path, so mixture construction stays bit-comparable);
uniform random chunk start stays on everywhere (it is the sampling policy, not a
factorized transform).

### 3.2 Scaling block — full recipe, nested subsets, identical budget

| Config | Songs | Seeds | Runs |
|---|---|---|---|
| `n21` | 21 | {0, 1, 2} | 3 |
| `n43` | 43 | {0} | 1 |
| `n64` | 64 | {0} | 1 |
| `n86` ≡ `full` | 86 | {0, 1, 2} | **0 new** (shared) |

Subsets are **nested** (n21 ⊂ n43 ⊂ n64 ⊂ n86 = the 86-song train split), drawn once
with subset-seed 0 by `scripts/make_subsets.py` from the materialized manifest
(fail-loud if the manifest still contains placeholder rows), committed as
`configs/subsets/n{21,43,64}.csv` **before any training run** (gate G0b). The 14-track
validation split never changes. Remix draws partners **within the active subset** (the
shrinking remix pool is part of what "less data" means — stated, not hidden).

### 3.3 Run accounting & the shared cell

Direction 01's `l1mag` REDUCED sweep cell (3 seeds) used: full 86-song split, full
recipe, identical model/optimizer/budget — **bit-identical configuration** to this
direction's `full`/`n86` cell. It is therefore **reused, not retrained** (config-hash
equality is asserted by a unit test before analysis). New GPU runs in this direction:
**4 (LOO) + 5 (scaling) = 9**; with the two 3-seed anchor cells shared/new as above,
the direction's own new-run total is **9**; worst case (if D01's cell is unavailable or
hash-mismatched) it re-runs as +3 → 12.

### 3.4 Contingency (pre-registered, budget-gated)
If Direction 01's final verdict replaces the project default loss (H-01a refuted →
`sisdr`), re-run `full` and `none` under the new loss (2 runs, seed 0) to check that the
factorization's *sign structure* is loss-stable; report as a sensitivity appendix. This
contingency does not alter H-02a/b verdicts (which are defined under `l1mag`).

### 3.5 Controlled-comparison invariants
Identical: architecture & init (per seed), optimizer/schedule/steps/batch, STFT, loss,
validation protocol, chunk-sampling policy. The ONLY differences: the augmentation
switchboard (§3.1) or the track allowlist (§3.2). The dataloader RNG remains a function
of (seed, step) — with a transform disabled, its random draws are **not consumed**
(each transform owns an independent RNG stream derived from (seed, transform-name,
step)), so disabling one transform does not reshuffle the others. This invariant is
unit-tested (G0).

---

## 4. Data

Dataset, license, acquisition, shard preparation: identical to Direction 01
([`../01-loss-function-study/MASTER_PLAN.md §4`](../01-loss-function-study/MASTER_PLAN.md)) —
MUSDB18 (Zenodo 1117372, research-only, nothing committed), 86/14/50 split with the
verified 14-track validation list, decoded WAV shards on Drive.

New for this direction:
- `scripts/make_subsets.py --manifest <materialized splits.csv> --sizes 21 43 64 --seed 0`
  → nested subset CSVs. Determinism unit-tested (same manifest + seed ⇒ identical
  files); nesting asserted; committed once generated (names only — no audio).
- Subset-balance report (printed by the script, included in the notebook): per-subset
  total duration, vocal-activity ratio, genre spread if metadata available — so a
  pathological draw (e.g. n21 all-instrumental-heavy) is visible *before* training.
  **Pre-registered rule:** the seed-0 draw is used regardless (no re-rolling — that
  would be selection); pathologies are reported and, if present, addressed by the §12
  sensitivity note.

---

## 5. The transforms (exact, code-pinned)

Applied per chunk, waveform domain, order fixed (remix → gain → flip), each with an
independent (seed, name, step)-keyed RNG stream (`singnet/data/augment.py`):

| Transform | Operation | Constants | Provenance (code-verified) |
|---|---|---|---|
| **remix** | vocals from track *i*, accompaniment stems from track *j ≠ i* (independent draws within the active split/subset) | — | UMX `random_track_mix`; Demucs `Remix(group_size=4)` |
| **gain** | per-source amplitude × U(0.25, 1.25) | 0.25–1.25 linear | UMX `_augment_gain`; Demucs `Scale` |
| **flip** | per-source $s \to -s$ with p = 0.5 | p = 0.5 | Demucs `FlipSign` |

Mixture is always constructed as the sum of the (possibly transformed) sources —
additivity is exact by construction in every arm.

---

## 6. Training & evaluation protocol

Training config: §1.3 (bit-identical to D01 REDUCED runs). Checkpoint/resume/registry
conventions identical to D01 §7.1; this direction's registry lives at
`02-augmentation-data-scaling/results/registry.csv` (same schema + columns
`aug_remix, aug_gain, aug_flip, n_songs`).

**Validation (decisions):** D01 §7.2 protocol verbatim — full-track overlap-add on the
14 validation tracks, mean vocals SI-SDR, every 2,000 steps, best-checkpoint selection.

**Test (exactly one pass for this direction):** the 3 FULL-86 checkpoints (seeds 0–2)
on the 50 test tracks — anchors the scaling curve's endpoint on truly held-out data and
measures the val→test generalization gap of the headline cell. LOO/subset arms are
**never** tested (their questions are validation-scoped by design; avoids 9 extra
test-set exposures). Committed: `results/test_per_track.csv`.

**Analysis (all in `singnet/analysis/scaling.py`, unit-tested; notebook only calls it):**
- LOO Δ-table + bar chart with the σ_seed band; interaction gap.
- Scaling curve: 4 points, endpoint seed bands, log₂ fit (a, b) with
  parametric-bootstrap 95 % CI on b (10 k draws, per-point noise = σ_seed); secant
  slopes per doubling (21→43, 43→86) reported alongside the fit so the "is it bending?"
  question is answered without leaning on the 2-parameter form.
- Extrapolation: the fitted curve is drawn only over [21, 86]; a dashed segment to 200
  songs may be shown **only** with the caption "descriptive extrapolation — no data
  beyond 86 songs" (pre-registered wording; Saijo & Bando 2025 motivates but does not
  license extrapolation).

---

## 7. Compute budget & run book

| Item | Runs × time (T4 est.) | Subtotal |
|---|---|---|
| LOO block (new) | 4 × ~1.3–1.8 h | 5–7 h |
| Scaling block (new) | 5 × ~1.3–1.8 h | 7–9 h |
| FULL-86 anchor | 0 (shared with D01; +3 runs ≈ 4–5.5 h only if hash-mismatch) | 0 (–5.5) h |
| Test pass + analysis | < 1 h | < 1 h |
| Contingency (§3.4) | 0 or 2 × ~1.5 h | 0–3 h |
| **Ceiling** | | **≈ 13–25 T4-h** |

**Run book (all RUN LATER; prerequisites: D01 data prep done, D01 G1 smoke passed):**
```bash
# 0. subsets (CPU, once, after data prep):            [RUN LATER]
python scripts/make_subsets.py --manifest 01-loss-function-study/configs/splits.csv \
       --sizes 21 43 64 --seed 0 --out 02-augmentation-data-scaling/configs/subsets/
# commit the generated subset CSVs before any training (gate G0b).
# 1. config dry-run (CPU, minutes):                    [RUN LATER]
python scripts/run_sweep.py --direction 02 --dry-run   # instantiates all 9 pipelines,
#    prints the augmentation switchboard + subset sizes per config; abort on mismatch.
# 2. the 9 runs (GPU):                                 [RUN LATER]
python scripts/run_sweep.py --direction 02 --stage reduced
# 3. analysis freeze (CPU): notebook 02 sections 1–5; verify D01 hash-share of FULL-86.
# 4. single test pass (GPU minutes):                   [RUN LATER]
python scripts/evaluate.py --checkpoints <3 FULL-86 ckpts> --split test
# 5. (only if D01 flipped the default loss) contingency: [RUN LATER]
python scripts/run_sweep.py --direction 02 --stage contingency
```

---

## 8. Go / no-go gates

| Gate | When | Criterion | On fail |
|---|---|---|---|
| **G0** | now (CPU) | full repo `pytest` green including new tests: per-transform RNG-stream independence (disabling one transform leaves the others' draws unchanged); switchboard configs build the declared pipelines; subset tool determinism + nesting; scaling-fit recovery of a known slope; config-hash equality test D01-`l1mag`↔D02-`full` | fix before GPU |
| **G0b** | after data prep | subset CSVs generated, balance report reviewed, committed | regenerate only on placeholder-manifest errors (never on unfavorable balance) |
| **G1** | first GPU session | inherited D01 G1 (smoke_overfit) already passed; D02 dry-run passes | debug configs |
| **G2** | after the 6 anchor-cell runs (n21×3 + verification of FULL-86×3) | σ_seed computed; σ_seed < 1.0 dB (else the LOO single-seed design cannot resolve plausible deltas → pre-registered escalation: add seeds 1–2 to `no_remix` only, +2 runs) | escalate per rule |
| **G3** | before test pass | registry complete (9–14 runs); analysis frozen & committed; no subset/LOO config ever touched test rows (grep-audited) | fix, re-freeze |

---

## 9. Deliverables & layout (this direction)

```
02-augmentation-data-scaling/
  MASTER_PLAN.md            ← this file
  THEORY.md + theory/theory.tex   ← stage B (spec §10)
  research/                 ← Phase-0 literature (done)
  notebooks/
    01_augmentation_anatomy.ipynb        ← stage C (spec §11)
    02_factorization_scaling_experiments.ipynb
  configs/
    base.yaml               ← inherits D01 base; loss=l1mag, budget=reduced
    loo_{no_remix,no_gain,no_flip,none}_seed0.yaml       (4)
    scale_n21_seed{0,1,2}.yaml, scale_n{43,64}_seed0.yaml (5)
    contingency_{full,none}_sisdr_seed0.yaml              (2, §3.4)
    subsets/                ← generated + committed at G0b (names only)
  results/                  ← registry.csv, eval CSVs, DEVIATIONS.md (empty now)
  paper/PAPER.md            ← stage E
  SEAN-README.md            ← stage E
singnet/analysis/scaling.py ← shared package addition (stage D, tested)
singnet/data/augment.py     ← extended: per-transform streams + switchboard (stage D)
```

**Code contracts (stage-D deltas only — nothing else in `singnet/` changes semantics):**
- `augment.py`: `AugmentPipeline(remix: bool, gain: bool, flip: bool, seed: int)`;
  per-transform RNG streams keyed (seed, name, step); existing D01 behavior ==
  `AugmentPipeline(True, True, True)` (regression-tested so D01 runs are unaffected).
- `MusdbChunks(..., track_allowlist: list[str] | None)` for subsets (None = full split).
- `analysis/scaling.py`: `fit_log2(ns, scores) -> FitResult(a, b, b_ci)`,
  `loo_table(registry_df) -> DataFrame`, `secant_slopes(...)`; all pure, unit-tested.
- Config schema gains `augment: {remix, gain, flip}` and `data.track_allowlist_csv`;
  `run_sweep.py` gains `--direction 02 --stage reduced|contingency --dry-run`.

---

## 10. THEORY.md outline (stage-B spec)

1. **Augmentation as distribution design**: the training distribution under remix as a
   product of per-source marginals vs the joint (real-song) distribution; the
   independence approximation and where it breaks (key/tempo coherence, stem bleed);
   why this changes the *mixture manifold* rather than just adding noise.
2. **Effective sample size combinatorics**: chunk-level pool sizes with/without remix
   (n tracks × chunks/track vs the product space); why "12× more songs" ≠ "12× more
   effective data" (correlation structure); honest bounds not fake precision.
3. **Gain and flip as group actions**: what invariances U(0.25, 1.25) gain and sign flip
   encode in a magnitude-masking model (flip is *exactly* magnitude-invariant → predict
   its Δ ≈ 0 within noise — a built-in sanity probe for the whole design, state it);
   gain interacts with the per-chunk input standardization (derive what actually changes:
   the target scale, not the input statistics).
4. **Scaling-curve models at 4 points**: log-linear vs saturating exponential vs power
   law; what 4 points can and cannot identify; why we pre-registered the secant-based
   decision rule instead of a functional-form claim; parametric bootstrap for the fit CI.
5. **Statistics of the design**: pooled σ_seed (why pooling FULL-86 and n21 is
   conservative), single-contrast pre-registration vs multiple comparisons, the
   interaction gap Δ_total − ΣΔ_t and what non-zero values mean.
6. `theory/theory.tex`: compilable standalone mirror.

## 11. Notebook specs (stage-C spec)

Global rules identical to D01 §14 (un-run; RUN-LATER banners with §7 runtimes; logic in
`singnet/`, notebooks orchestrate; Colab bootstrap cell; cross-link rather than
duplicate D01's data/EDA notebook).
- **01_augmentation_anatomy.ipynb**: what each transform *does* — waveform/spectrogram
  before-after galleries per transform; a remix-mixture gallery ("musically incoherent
  but statistically rich"); the flip-invariance sanity argument rendered visually;
  combinatorics table from THEORY §2; subset-balance report rendering (post-G0b);
  what to listen for (RUN LATER audio cells).
- **02_factorization_scaling_experiments.ipynb**: pre-registration recap verbatim from
  §2; run matrix + shared-cell accounting (§3.3); dry-run + launch cells (RUN LATER);
  registry-driven analysis: LOO Δ bar chart with σ_seed band, interaction gap, scaling
  curve with endpoint bands + log₂ fit + secant slopes, val→test generalization of the
  FULL-86 cell; **interpretation section with the §13 pre-written branches**;
  conclusions + "what this changes for directions 03–10" (data-vs-capacity reading
  feeding Direction 03; remix-value feeding Direction 10's pseudo-label pool sizing).

---

## 12. Risks & mitigations (direction-specific)

- **Flip Δ ≈ 0 is expected, not a failure** (THEORY §3 predicts magnitude-invariance):
  it doubles as a negative control — if $\Delta_{\text{flip}} > \sigma_{\text{seed}}$
  something is wrong with the design (investigate before interpreting anything else).
  Pre-registered as the design's built-in placebo.
- **Subset draw luck**: n21 could be unrepresentative → 3 seeds *of training* on the
  same draw quantify training noise but not draw noise; pre-registered honesty: report
  the balance table, never re-roll, and label the curve "one nested draw" (a
  multi-draw study is out of budget and noted as future work).
- **Remix pool shrinkage at small n** is part of the treatment (less data = fewer remix
  partners), not a confound — but state it so readers don't misread the curve as
  "songs-only."
- **Shared-cell drift**: if D01's `l1mag` cell was trained under any config drift, the
  hash test (G0) catches it and the cell re-runs here (+3 runs, budgeted).
- **Single-seed LOO arms**: deltas smaller than σ_seed are *by design* unresolvable —
  the G2 escalation (seeds on `no_remix`) protects only the primary contrast; secondary
  deltas (gain) may stay ambiguous and are reported as such.
- **AAC additivity error**: MUSDB18's compressed stems sum to the mixture only to
  ~1e-3; both remix-on and remix-off arms construct mixtures as stem sums (§5), so the
  comparison is internally exact; the (tiny) mismatch to the *original* mixture is a
  dataset-wide constant documented in D01's prep verification.

---

## 13. Outcome interpretation matrix (pre-registered; full prose in paper/PAPER.md)

| Outcome | Reading | Consequence |
|---|---|---|
| H-02a supported | Remix is the small-data lever; folklore confirmed with an effect size | remix stays mandatory everywhere; D10's unlabeled-pool sizing gains a measured anchor |
| H-02a refuted (remix ≤ others) | The mixture-manifold story is wrong at this scale — gain/flip-style perturbations suffice | re-examine remix's role before D10; flag for replication at FULL budget |
| H-02a refuted (nothing matters, Δ_total ≤ σ_seed) | At 16 k steps the model is budget-bound, not data-bound | rerun `full` vs `none` once at FULL budget before accepting (pre-registered +2 runs) |
| H-02b supported | MUSDB-only training is data-starved; slope b quantifies the price of data | strengthens D10 (distillation adds effective data); motivates extra-data lines in any future work |
| H-02b refuted (plateau) | 86 songs saturate this capacity — the constraint is the model, not the data | strengthens D03 (capacity/architecture is the lever); D10's premise weakens (state so) |
| H-02b mixed / non-monotone | subset-draw sensitivity dominates | report per-subset balance; downgrade curve to descriptive |
| Flip control fires (Δ_flip > σ_seed) | design fault (RNG streams, normalization, or a bug) | halt interpretation; debug; document in DEVIATIONS.md |

---

## 14. Timeline (Sean's part-time weeks, once GPU exists; assumes D01 weeks 1–2 done)

| Week | Work | Gate |
|---|---|---|
| 1 | G0 tests green (now, CPU); after D01 data prep: subsets + G0b; dry-run; start 9 runs | G0, G0b, G1 |
| 2 | finish runs; G2 check; freeze analysis; test pass; (contingency if D01 flipped) | G2, G3 |
| 3 | fill paper placeholders; figures; verdicts | — |

---

*Frozen on 2026-07-13. Deviations during execution go to `results/DEVIATIONS.md` with
date + reason.*
