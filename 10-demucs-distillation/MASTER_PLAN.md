# MASTER PLAN — Direction 10: Demucs as teacher — pseudo-label distillation to close the small-model gap

**Status: pre-registered plan. Nothing trained, no data downloaded. All GPU/network
work is deferred ("RUN LATER").** Self-contained by design; cross-references
([`research/LITERATURE.md`](research/LITERATURE.md),
[`../00-shared-research/papers/demucs-hybrid-family.md`](../00-shared-research/papers/demucs-hybrid-family.md),
[`../06-robust-training/MASTER_PLAN.md`](../06-robust-training/MASTER_PLAN.md)) add
depth, not required context.

---

## 0. TL;DR

Our compact student (SingNet-C1) is capped by 86 labeled songs; the big teacher
(HT-Demucs, the engine StemCraft already ships) is ~5 dB better but 10× the model. The
classic industrial move: **let the teacher label unlabeled music, and train the student
on the pseudo-labels.** We build a **license-audited FMA pipeline** (~800 CC-licensed
30-s clips ≈ 6.7 h, filtered to exclude no-derivative licenses, vocal-screened using
the teacher's own outputs + Direction 08's activity machinery), label it **once** with
`htdemucs` (2–4 GPU-h, provenance-pinned), and train three data arms with the loss and
recipe held fixed: **MUSDB-only** (the shared baseline cell — its sixth reuse, 0 new
runs), **mixed** (MUSDB + pseudo-labeled FMA, 3 seeds), and **distilled-only** (1
seed). Pre-registered bet **H-10: mixed training closes ≥ 25 % of the student→teacher
SI-SDR gap on the MUSDB test set** (which the teacher never labels — no leakage by
construction). Cross-feeds are wired in: Direction 06's dose–response chart prices
teacher error as corrupted-target training (and its trimmed loss ships here as an
exploratory defense arm), Direction 08's SLR joins the eval battery, and Direction
02's scaling verdict pre-frames whether "more data" was ever the right lever. **6 new
training runs + one teacher-labeling pass ≈ 11–16 T4-hours** — the menu's heaviest
direction made affordable by the shared cell and tight arms.

---

## 1. Problem framing

### 1.1 The question
Semi-supervised separation works at the high end — BSRNN mined unlabeled songs with an
activity detector for pseudo-label fine-tuning; MixIT pre-training helps modern MSS
(Saijo & Bando 2025) — but nobody reports the **compact-student** version: can a
~10 M-param from-scratch model close a meaningful fraction of its gap to a SOTA
teacher purely from teacher-labeled, license-safe public audio, at hobbyist compute?
That is *the* practical question for anyone who owns a small model and a GPU weekend
(gap and adjacent work verified in Phase 0; the honest framing is "first careful
compact-scale look", not "first ever pseudo-labeling").

### 1.2 Why this design isolates the data effect
The three arms differ **only in the training-data source** — same architecture, same
`l1mag` loss (pseudo-labels enter as ordinary targets), same augmentation recipe
(remix **within-pool** so real stems never mix with pseudo-stems), same budget. We
deliberately do NOT introduce a special distillation objective (feature matching,
soft-mask temperature analogues): THEORY §2 surveys those and states why the
data-effect isolation comes first — a fancier objective would confound "more data"
with "new loss." (The regression setting also lacks Hinton-style temperature; that
analogy's limits are derived, not hand-waved.)

### 1.3 Fixed infrastructure (inherited, not re-decided)
SingNet-C1 (9,835,745 params), `l1mag`, remix+gain+flip (stream-keyed), STFT
4096/1024, 6-s mono chunks, uniform sampling policy, AdamW + warmup/cosine, AMP,
batch 16, REDUCED = 16 k steps, best-checkpoint by MUSDB validation SI-SDR, 86/14/50
split. The teacher is `htdemucs` — the exact engine StemCraft ships, making the
student→teacher gap the product-relevant one.

---

## 2. Pre-registered hypothesis & decision rules

Notation: on the MUSDB **test** set (one pass, end of study), let
$s_{\text{base}}$ = mean vocals SI-SDR of `musdb_only` (3 seeds),
$s_{\text{mix}}$ = mean of `mixed` (3 seeds),
$s_{\text{T}}$ = the teacher's own score (evaluated in the same session, same metric).
Gap $G = s_{\text{T}} - s_{\text{base}}$; closure
$\hat C = (s_{\text{mix}} - s_{\text{base}}) / G$. σ_seed = pooled between-seed std of
the two 3-seed cells (validation-based decisions use the val analogue).

> **H-10 (gap closure).**
> - **Supported** iff $s_{\text{mix}} - s_{\text{base}} > \sigma_{\text{seed}}$ (the
>   improvement is real) **and** $\hat C \ge 0.25$ (it is big enough to matter).
> - **Partially supported** iff the improvement is real but $\hat C < 0.25$ — pseudo-
>   labels help, less than hoped; the measured $\hat C$ with CI is the finding.
> - **Refuted (null)** iff $|s_{\text{mix}} - s_{\text{base}}| \le \sigma_{\text{seed}}$.
> - **Refuted (negative transfer)** iff $s_{\text{mix}} < s_{\text{base}} - \sigma_{\text{seed}}$
>   — teacher errors/domain shift poison the student (the Direction-06 bridge branch).
> Precondition (sanity, not hypothesis): $G > 2$ dB on the frozen protocol — else the
> premise ("a large gap exists") failed upstream and the study reframes to documenting
> that (pre-registered).

**Pre-registered secondaries (descriptive, pre-written readings):**
- `distill_only` vs `musdb_only` (1 seed): can ~6.7 h of pseudo-labeled CC audio
  substitute for 86 real songs end-to-end?
- `mixed25` (p_FMA = 0.25, 1 seed): mixing-ratio sensitivity.
- `mixed_trim` (1 seed, exploratory, Direction-06 cross-feed): trimmed loss (q = 0.30)
  as a teacher-error defense — motivated by D06's uniform-bleed mechanism (teacher
  error is *structured* corruption; stated).
- **SLR** (Direction 08's metric) for every arm — does pseudo-label training change
  silence behavior? (Teacher leakage in quiet passages is a plausible transfer.)
- Conditional framing hook: the paper cross-references Direction 02's scaling verdict
  (⟪data-starved / saturated⟫) — if D02 found a plateau, H-10's premise was weak and
  the write-up must say so *before* reporting the result.

---

## 3. The pseudo-label pipeline (the direction's engineering core)

### 3.1 Source data & license audit (`scripts/prepare_fma.py`)
- **Corpus:** FMA-small (Defferrard et al., 1612.01840; mdeff/fma): 8,000 × 30-s
  clips, 7.2 GB, genre-balanced; metadata CC BY 4.0 (per-track license field).
- **License allowlist (pinned):** {CC0/Public Domain, CC-BY, CC-BY-SA, CC-BY-NC,
  CC-BY-NC-SA}. **Excluded:** any **-ND** variant (separated stems are derivative
  works — the ND exclusion is the load-bearing legal call, derived in THEORY §5) and
  any non-CC/unclear license strings. Filter runs on the metadata CSV; the resulting
  **track-ID manifest is committed** (IDs + licenses only — never audio).
- **Screen sample:** N_screen = 1,200 clips drawn deterministically (seed 0) from the
  allowlisted pool.
- **Nothing is downloaded now**; the script constructs and documents every step,
  fail-loud without data, tested on synthetic metadata fixtures.

### 3.2 Teacher labeling (`scripts/teacher_label.py`, RUN LATER, 2–4 GPU-h once)
- Run `htdemucs` over the screened clips; **record demucs package version, model
  signature/hash, and inference settings** in a provenance file (committed).
- **2-stem consistency by construction (pinned):** vocals = teacher vocals output;
  accompaniment := mixture − teacher vocals (exact additivity, mirroring the student's
  task; the teacher's raw 4-stem sum-residual is *measured and recorded* but not
  used).
- **Vocal-activity screen (BSRNN-style, using Direction 08's machinery):** keep clips
  whose teacher-vocal activity ratio ≥ 20 % (energy profiles via the D08 profile
  tooling on teacher vocals); take the first **N_train = 800** surviving clips
  (deterministic order). Pre-registered fallback: if fewer than 800 survive, lower the
  threshold to 10 % once; if still short, use all survivors and record the count.
- Output: WAV shards (mixture + 2 pseudo-stems) + activity profiles on Drive; never
  committed.

### 3.3 Leakage & provenance guards
- The MUSDB test set is **never teacher-labeled** and no FMA audio enters any MUSDB
  split: the pseudo-pool dataset class **structurally refuses MUSDB shard paths**
  (unit-tested), mirroring Direction 06's guard pattern.
- Teacher trained on MUSDB train (+800 private songs) — legitimate for a
  teacher/ceiling; the student never sees MUSDB test through any path (stated in the
  paper's setup).
- FMA↔MUSDB catalog overlap is implausible (disjoint sources) and noted, not assumed
  away silently.

---

## 4. Experimental design

### 4.1 Run matrix

| Arm | Training data | Seeds | New runs |
|---|---|---|---|
| `musdb_only` | MUSDB 86 | {0,1,2} | **0** (shared cell, hash-asserted — sixth direction on `a97d5400e994`) |
| `mixed` | 50/50 chunk-level pool mix: MUSDB 86 + FMA-pseudo 800 | {0,1,2} | 3 |
| `distill_only` | FMA-pseudo 800 only | {0} | 1 |
| `mixed25` | 75/25 MUSDB/FMA | {0} | 1 |
| `mixed_trim` | as `mixed` + TrimmedLoss(q=0.30) | {0} | 1 |
| **Total new** | | | **6** |

Chunk-level pool mixing (pinned): each training example draws its source pool
(Bernoulli p_FMA from a dedicated (seed, "pool", step) stream), then chunks/remixes
**within** that pool (real stems never remix with pseudo-stems; augmentation streams
untouched — tested). Contingency (pre-registered): if Direction 01 flips the default
loss, re-run {`musdb_only`, `mixed`} × 1 seed under it (+2).

### 4.2 Invariants
Identical across arms: architecture & init per seed, loss (except the labeled
`mixed_trim` arm), optimizer/schedule/steps/batch, augmentation recipe, sampling
policy (uniform), validation protocol (MUSDB 14-track val — **pseudo-data never enters
validation**). ONLY the training-data source (and p_FMA) differs.

---

## 5. Training & evaluation protocol

Training: D01 §7.1 verbatim; registry at
`10-demucs-distillation/results/registry.csv` with appended columns
(`data_source`, `p_fma`, `n_pseudo_clips`, `teacher_version`, `teacher_consistency_db`).

**Validation (decisions):** MUSDB 14-track val, vocals SI-SDR (best-checkpoint
selection), SLR recorded alongside (selection stays SI-SDR-only, per the D08 rule).

**Test (exactly one session at the end):** on the 50 MUSDB test tracks — the 6
student checkpoints from 3-seed cells (`musdb_only`, `mixed`) + the three 1-seed arms
+ **the teacher itself** (the $s_T$ anchor) + do-nothing + oracle IRM. Metrics:
vocals/accomp SI-SDR, SI-SDRi, **SLR (θ = −60)**, museval SDR secondary. Paired
bootstrap + Wilcoxon on the one pre-registered pair (`mixed` − `musdb_only`);
gap-closure $\hat C$ with a seed+paired-track propagated CI (delta method, THEORY §4).

---

## 6. Compute budget & run book

| Item | Cost (T4 est.) |
|---|---|
| FMA download + license filter + screen (CPU/network) | ~1 h, no GPU |
| Teacher labeling (1,200 × 30 s through htdemucs) | 2–4 GPU-h, once |
| 6 training runs × REDUCED | 8–11 h |
| Test session (10 systems + teacher + anchors, incl. SLR) | ~1–1.5 h |
| Contingency | 0 or ~3 h |
| **Ceiling** | **≈ 12–17 (20) T4-h** |

**Run book (all RUN LATER; prerequisites: D01 data prep + G1 smoke done):**
```bash
# 0. FMA metadata + license manifest + screen sample (CPU+network):   [RUN LATER]
python scripts/prepare_fma.py --metadata-dir $FMA_META --audio-dir $FMA_AUDIO \
       --screen 1200 --seed 0 --out $PSEUDO_ROOT     # commits nothing; writes manifest
# 1. teacher labeling + activity screen (GPU 2–4 h, once):            [RUN LATER]
python scripts/teacher_label.py --manifest $PSEUDO_ROOT/manifest.csv \
       --keep 800 --activity-threshold 0.20 --out $PSEUDO_ROOT
# 2. dry-run (CPU): pool mixing + guards printed per config:          [RUN LATER]
python scripts/run_sweep.py --direction 10 --dry-run
# 3. the 6 runs (GPU):                                                [RUN LATER]
python scripts/run_sweep.py --direction 10 --stage reduced
# 4. analysis freeze on val; then the one test session (GPU ~1.5 h):  [RUN LATER]
python scripts/evaluate.py --direction 10 --test-session --include-teacher --slr
```

---

## 7. Go / no-go gates

| Gate | When | Criterion | On fail |
|---|---|---|---|
| **G0** | now (CPU) | full-repo `pytest` green incl.: license filter on fixture metadata (allow/deny table incl. ND and unclear strings); deterministic screen selection; pool-mixing probabilities + stream independence (pool draws don't perturb augmentation/sampling streams); consistency post-processing math (â = x − v̂ exact); MUSDB-path guard on the pseudo pool; config hash-stability + `musdb_only` ≡ shared cell | fix before GPU |
| **G0b** | after FMA metadata fetch | license manifest committed (IDs + licenses only); screen sample frozen | re-run filter only on metadata errors |
| **G1** | teacher session | labeling runs end-to-end; provenance file written; consistency residual recorded; ≥ 800 clips survive the activity screen (else the pre-registered threshold fallback); spot-listen 3 clips (teacher output sane) | debug before training |
| **G2** | after the `mixed` 3-seed cell | σ_seed < 1.0 dB; no diverged run | +1 seed on `mixed` only (+1 run max) |
| **G3** | before test session | registry complete (6 runs); analysis frozen on val; pseudo-data provably absent from val/test paths (guard log + grep audit) | fix, re-freeze |

---

## 8. Deliverables & layout (this direction)

```
10-demucs-distillation/
  MASTER_PLAN.md            ← this file
  THEORY.md + theory/theory.tex   ← stage B (§9)
  research/                 ← Phase-0 literature (done)
  notebooks/
    01_teacher_and_data.ipynb        ← stage C (§10)
    02_distillation_experiments.ipynb
  configs/
    base.yaml; mixed_seed{0,1,2}.yaml; distill_only_seed0.yaml;
    mixed25_seed0.yaml; mixed_trim_seed0.yaml;
    contingency_{musdb_only,mixed}_sisdr_seed0.yaml
  results/                  ← registry.csv, license manifest, provenance file,
                              test CSVs, DEVIATIONS.md (empty now)
  paper/PAPER.md, SEAN-README.md  ← stage E
singnet/data/pseudo.py     ← stage D (tested: pseudo shards, pool mixing, guards)
scripts/prepare_fma.py     ← stage D (tested on fixtures)
scripts/teacher_label.py   ← stage D (CLI + post-processing tested; execution RUN LATER)
```

**Code contracts (stage-D deltas only; Directions 01–08 untouched semantically):**
- `singnet.data.pseudo.PseudoLabeledShards(root, manifest)` (30-s clips → 6-s chunks;
  same shard interface as MUSDB shards; **raises on MUSDB paths**) and
  `MixedPools(musdb_ds, pseudo_ds, p_fma, seed)` (Bernoulli pool draw on a dedicated
  stream; within-pool remix by construction — each pool keeps its own AugmentPipeline
  instance sharing the same (seed, name, step) streams).
- `prepare_fma.py`: `--filter-licenses` (pinned allowlist), `--screen N --seed`,
  manifest writer (IDs, licenses, screen flags); pure-metadata operation, fixture-
  tested.
- `teacher_label.py`: demucs invocation construction (lazy import, RUN-LATER guard),
  post-processing (â = x − v̂; consistency residual in dB; activity profiles via the
  D08 profile tooling), provenance writer. Post-processing functions unit-tested on
  synthetic arrays.
- `run_sweep.py --direction 10`; `evaluate.py --direction 10 --test-session
  --include-teacher --slr` (teacher evaluated through a lazy demucs wrapper, RUN
  LATER; skeleton errors cleanly without it); registry appends per §5.

---

## 9. THEORY.md outline (stage-B spec)

1. **Distillation & pseudo-labeling, honestly taxonomized**: Hinton KD (soft targets,
   temperature) and the derivation of why the temperature mechanism has no regression
   analogue (logit scaling vs continuous targets — what survives: the teacher's
   outputs as targets, i.e. self-training/pseudo-labeling à la BSRNN); the design
   space we deliberately don't enter (feature matching, mask-space KD, waveform-L1
   distillation) and the data-effect-isolation argument for hard pseudo-labels with
   the unchanged loss.
2. **Teacher error as structured corruption — the Direction-06 bridge**: teacher
   residual $r = \hat v_T - v$ as target corruption; how it differs from D06's ε-bleed
   (content-correlated, level-varying, both-directions); what D06's dose–response
   chart does and does not license us to predict here; why `mixed_trim` is the
   transferred defense and what its telemetry would show if teacher error behaves
   like bleed.
3. **Domain shift**: FMA (CC indie, 30-s clips, production diversity) vs MUSDB
   (full tracks, studio stems); covariate shift on mixtures AND label shift via
   teacher error; why within-pool remixing preserves each pool's mixture statistics;
   the 30-s-clip boundary effect on 6-s chunking (negligible, quantified).
- 4. **The gap-closure estimand**: $\hat C$'s definition; why the teacher's own
   test-set score is the right denominator (product-relevant ceiling; teacher's
   MUSDB-train exposure stated); seed + paired-track CI via the delta method; what
   $\hat C$ does not mean (no claim the student approaches the teacher off-MUSDB).
5. **License formalism**: the FMA license taxonomy; the ND-exclusion argument
   (separated stems as derivative works); NC-inclusion rationale (research use, no
   redistribution); the audit trail (committed manifest, provenance file).
6. **Statistics**: one pre-registered pair + descriptive secondaries; σ_seed pooling;
   the precondition gate (G > 2 dB) and its rationale.
7. `theory/theory.tex`: compilable standalone mirror.

## 10. Notebook specs (stage-C spec)

Global rules identical to prior directions (un-run; RUN-LATER banners with §6
runtimes; logic in `singnet/`; Colab bootstrap; cross-link D01 data prep).
- **01_teacher_and_data.ipynb**: the pipeline walked end-to-end — license filter demo
  on fixture metadata (allow/deny table rendered), screen sampling, teacher labeling
  cells (RUN LATER, 2–4 GPU-h), consistency residual + activity screen, pseudo-stem
  gallery (what teacher labels look like vs real stems — spectrogram pairs, RUN
  LATER), provenance file rendered; the leakage-guard story.
- **02_distillation_experiments.ipynb**: pre-registration recap verbatim (§2); run
  matrix + shared-cell accounting; launch cells (RUN LATER); analysis: the
  gap-closure headline ($s_{\text{base}}$, $s_{\text{mix}}$, $s_T$, $\hat C$ with CI,
  drawn as a ladder), distill-only reading, p_FMA sensitivity, trim cross-feed
  reading, SLR battery table, test session + paired stats; §12 interpretation
  branches as labeled stubs; conclusions + the program-level synthesis cell (how
  D02's scaling verdict, D06's corruption chart, and this result compose into the
  project's data-vs-capacity story).

---

## 11. Risks & mitigations (direction-specific)

- **Teacher-error transfer is the central risk** (htdemucs bleeds vocals↔other on
  hard content) → measured, not feared: the consistency residual is recorded, SLR
  catches silence-side transfer, `mixed_trim` is the defense probe, and negative
  transfer is a pre-registered branch with the D06 bridge as its reading.
- **Domain mismatch mutes gains** → pre-registered partial/null branches; the FMA
  genre spread is reported next to MUSDB's; within-pool remix avoids compounding.
- **License audit gaps** (metadata errors, ambiguous strings) → unclear licenses are
  DENIED by default (allowlist, not blocklist); the manifest is committed for public
  audit; no audio ever committed.
- **FMA availability drift** (mirror URLs change) → the script takes explicit paths;
  instructions name the official mdeff/fma release assets; nothing hardcodes a URL
  that can rot silently.
- **30-s clips ≠ full songs** (less within-track diversity per clip) → counted as
  part of the treatment ("cheap public audio"), quantified in the data-stats table.
- **Precondition failure (G ≤ 2 dB)** → upstream anchor problem (student unexpectedly
  strong or teacher weak on the frozen protocol); pre-registered reframe documented.
- **Budget creep** (the menu's warned scope risk) → the pipeline is one script + one
  labeling session; arms are capped at 6; everything else is inherited machinery.

---

## 12. Outcome interpretation matrix (pre-registered; full prose in paper/PAPER.md)

| Outcome | Reading | Consequence |
|---|---|---|
| H-10 supported ($\hat C \ge 25$ %) | pseudo-labeling closes real gap at hobbyist compute — the industrial playbook works at compact scale | StemCraft's from-scratch engine gains a cheap upgrade path; scales with more CC audio (named future work, not run) |
| Partial ($>σ$, $\hat C < 25$ %) | help is real but modest — likely domain-shift-limited | report $\hat C$ + CI; the p_FMA and data-volume axes become the named follow-ups |
| Refuted (null) | 6.7 h of shifted pseudo-data adds nothing MUSDB's 86 songs + remix didn't already give | connect to D02's verdict (if data-starved, the *kind* of data mattered; if saturated, this null was predicted) — the program-level synthesis carries the value |
| Negative transfer | teacher errors poison the student — corrupted-target training in the wild | the D06 bridge quantifies; `mixed_trim`'s result says whether the cheap defense helps; a strong cautionary result |
| `distill_only` ≥ `musdb_only` | pseudo-labels rival real labels end-to-end | provocative headline (1 seed — flagged for replication before any strong claim) |
| `distill_only` ≪ | real labels carry irreplaceable signal | grounds the "labels matter" story with a number |
| SLR degrades under pseudo-training | teacher silence-leakage transfers | Direction 08's metric earns its keep; karaoke products filter pseudo-data by teacher SLR (named follow-up) |

---

## 13. Timeline (Sean's part-time weeks, once GPU exists; after D01 data prep)

| Week | Work | Gate |
|---|---|---|
| 1 | G0 green (now); FMA metadata + manifest (G0b); teacher session (G1) | G0, G0b, G1 |
| 2 | the 6 runs; G2; freeze analysis | G2 |
| 3 | test session; fill paper; program-level synthesis | G3 |

---

*Frozen on 2026-07-13. Deviations during execution go to `results/DEVIATIONS.md` with
date + reason.*
