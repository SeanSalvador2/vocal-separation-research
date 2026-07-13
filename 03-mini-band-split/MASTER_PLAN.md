# MASTER PLAN — Direction 03: A mini band-split front-end — does the band-partition idea survive at ~10 M params?

**Status: pre-registered plan. Nothing trained. All GPU work is deferred ("RUN LATER").**
Self-contained by design; cross-references ([`research/LITERATURE.md`](research/LITERATURE.md),
[`../01-loss-function-study/MASTER_PLAN.md`](../01-loss-function-study/MASTER_PLAN.md),
[`../PLAN.md`](../PLAN.md)) add depth, not required context.

---

## 0. TL;DR

The current SOTA family in music source separation (BSRNN → BS-RoFormer → Mel-RoFormer →
SCNet → Moises-Light) is built on one shared idea: **split the spectrogram into frequency
bands and give each band dedicated capacity**. Every published demonstration is at large
scale, usually with extra training data, and always confounded with many simultaneous
design changes. We isolate the idea itself at compact scale: **three models at matched
parameter count (±2 % of the 9,835,745-param baseline)** — (i) the uniform-encoder U-Net
baseline, (ii) a **3-band mel-spaced** band-split encoder, (iii) a **3-band
uniform-width** band-split encoder (the control that separates "splitting" from "mel
spacing") — 3 seeds each at the standard reduced budget, one ordered pre-registered
hypothesis, a per-band error breakdown as the mechanism figure, and 2 full-budget
confirmation runs. **The baseline cell is shared with Directions 01/02** (bit-identical
config, hash-asserted), so this direction adds only **8 new GPU runs ≈ 16–21 T4-hours**.
A within-noise null is pre-registered as a *finding* ("band-split is a large-model
phenomenon"), not a failure — and the March-2026 BSRNN replication study (arXiv
2603.09187, professionals failing to reproduce published band-split numbers) makes the
modest, controlled framing the defensible one.

---

## 1. Problem framing

### 1.1 The question
Band-splitting is the load-bearing idea of the current SOTA family, but the literature
never isolates it: BSRNN changed bands *and* sequence modeling *and* training recipe;
Mel-RoFormer showed mel-spaced bands beat heuristic bands but only inside a full
transformer at full scale; Moises-Light engineered an efficient band-split U-Net but
reports an efficiency frontier, not a controlled split-vs-no-split comparison; SCNet
varies band compression alongside everything else. **No published work answers: at
~10 M parameters on MUSDB-only data, does giving frequency bands dedicated encoder
capacity — everything else fixed — improve vocal separation?** (Gap verified 2026-07-13;
[`research/LITERATURE.md §4`](research/LITERATURE.md).)

### 1.2 Two effects, deliberately separated
The design isolates two nested claims:
1. **Splitting**: parallel per-band encoders (no cross-band mixing until the bottleneck)
   vs one full-spectrum encoder — *dedicated capacity and band-local statistics*.
2. **Mel spacing**: perceptually-motivated band edges (narrow low bands, wide high
   bands) vs equal-width edges — *where* the capacity is concentrated.
The uniform-split arm is the control making these separable; without it, a mel-split win
would confound the two (this is exactly the confound Mel-RoFormer's comparison to BSRNN
carries at large scale).

### 1.3 Fixed infrastructure (inherited, not re-decided)
Everything except the encoder front-end is Direction 01's frozen stack: STFT 4096/1024
Hann (2048 network bins), 6-s mono chunks, uniform random chunk start, full augmentation
recipe (remix + gain + flip), `l1mag` loss, AdamW 1e-3 + warmup/cosine, AMP (fp32
losses), batch 16, REDUCED = 16 k steps / FULL = 40 k steps, best-checkpoint by
validation SI-SDR, 86/14/50 split with the verified validation list.

---

## 2. Pre-registered hypothesis & decision rules

Let $\overline{v}(\text{arm})$ = mean best-checkpoint validation vocals SI-SDR over the
arm's 3 seeds; σ_seed = pooled between-seed std over the three arms
($\sigma_{\text{seed}} = \sqrt{(s^2_{\text{base}} + s^2_{\text{uni}} + s^2_{\text{mel}})/3}$).

> **H-03 (ordered band-split benefit).** At matched parameters:
> - **Effect 1 (splitting helps):** $E_1 := \overline{v}(\text{uniform}) - \overline{v}(\text{baseline}) > \sigma_{\text{seed}}$.
> - **Effect 2 (mel spacing helps beyond splitting):** $E_2 := \overline{v}(\text{mel}) - \overline{v}(\text{uniform}) > \sigma_{\text{seed}}$.
> - **Fully supported** iff E1 **and** E2 (the ordered chain mel > uniform > baseline,
>   each link outside the noise band).
> - **Partially supported** iff exactly one of E1/E2 holds (each partial case has a
>   distinct pre-written interpretation, §13).
> - **Refuted (null)** iff neither holds with all three arms within ±σ_seed of each
>   other — pre-registered headline: *"the band-split advantage is not detectable at
>   ~10 M params / MUSDB-only / 16 k steps"*.
> - **Refuted (negative)** iff $\overline{v}(\text{baseline})$ exceeds both variants by
>   > σ_seed — band isolation actively hurts at this scale (boundary-artifact reading,
>   §13).
> Confirmation guard: the FULL-budget pair (§3.3) must not reverse the REDUCED-budget
> verdict's sign on its comparison; if it does, the verdict is downgraded to **mixed
> (budget-dependent)** — itself a pre-registered finding about short-training
> architecture comparisons.

**Mechanism evidence (pre-registered, not a hypothesis):** the per-band error breakdown
(§6.3). If H-03 holds, gains should concentrate in the vocal-energy-dense bands
(~100 Hz–4 kHz); if gains are uniform across bands (or concentrated where vocals are
absent), the dedicated-capacity mechanism is questioned even under a positive headline —
we commit to saying so.

**Descriptive secondaries (no hypotheses):** measured FLOPs/inference RTF per arm (at
matched *parameters*, the band variants are analytically ~2× cheaper in encoder FLOPs —
THEORY §4 derives why conv parameter count is spatial-size-independent but compute is
not; an incidental efficiency result reported honestly as a bonus, never as the claim);
train-loss curves per arm (optimization-difficulty comparison).

---

## 3. Experimental design

### 3.1 The three arms

| Arm | Encoder front-end | Band edges (Hz, HTK mel; exact values computed in code) | Params |
|---|---|---|---|
| `baseline` | single full-spectrum encoder (SingNet-C1 verbatim) | — | 9,835,745 |
| `split_uniform` | 3 parallel band towers, equal **bin** widths | ≈ {0–7.35 k, 7.35–14.7 k, 14.7–22.05 k} | 9,835,745 ± 2 % |
| `split_mel` | 3 parallel band towers, equal **mel** widths | ≈ {0–1.53 k, 1.53–6.43 k, 6.43–22.05 k} | identical to `split_uniform` by construction |

**Band-split architecture (both variants; only the edge list differs):**
- The 2048 network bins are partitioned into 3 contiguous bands (exact, disjoint,
  exhaustive — asserted).
- Each band gets its own encoder tower: the same 5-level Conv(5×5, stride 2)–BN–LReLU
  design as the baseline, base width **c** (common to all towers), channels
  c→2c→4c→8c→16c. Towers pad their band's bin extent up to the next multiple of 32
  internally and crop back on output (edges therefore need no divisibility).
- **No cross-band mixing before the bottleneck** (the treatment). At each level, tower
  outputs are concatenated along the **frequency axis** (channel counts match by shared
  width), forming full-spectrum skip tensors; the bottleneck and decoder are the
  baseline design at base width c and consume these exactly as the baseline consumes
  its encoder features.
- **Width selection:** c is chosen by a deterministic search
  (`scripts/match_params.py`) as the largest width (with a single ±Δ adjustment allowed
  at the bottleneck level only, if needed) satisfying
  |P(variant) − 9,835,745| / 9,835,745 ≤ **2 %**. THEORY §4 derives the closed-form
  P(c); the chosen c and the per-module parameter table for all three arms are
  committed and **asserted in unit tests** (the credibility crux — a reader must be
  able to see the match is exact, not hand-waved). Expected c ≈ 23 (≈ 0.53× baseline
  width; derivation in THEORY §4).
- Mask head, featurization (log1p + per-chunk standardization), Nyquist handling:
  baseline-identical.

### 3.2 Run matrix

| Stage | Runs | Budget | Seeds | Purpose |
|---|---|---|---|---|
| Sweep | `split_mel` × 3, `split_uniform` × 3 | REDUCED (16 k) | {0,1,2} | H-03 primary evidence |
| `baseline` | **0 new** — shared cell ≡ D01 `l1mag` sweep ≡ D02 `full` (config-hash equality asserted before analysis) | REDUCED | {0,1,2} | third arm |
| Confirmation | best variant + `baseline` | FULL (40 k) | {0} | budget-dependence guard |
| Contingency (only if D01 flips the default loss) | `baseline` + `split_mel` | REDUCED | {0} | loss-sensitivity of the comparison |

New GPU runs: **6 + 2 = 8** (+2 contingency; +3 only if the shared-cell hash check
fails and the baseline must be retrained).

### 3.3 Controlled-comparison invariants
Identical across arms: data order & augmentation streams per seed ((seed, transform,
step)-keyed, arm-independent — inherited invariant, already unit-tested), loss,
optimizer, schedule, steps, batch, STFT, mask head, validation protocol. The ONLY
difference: the encoder front-end module (and, between the two variants, only the
band-edge list). Init: Kaiming per seed; note (pre-registered honesty): matched seeds
do **not** produce matched initial weights across arms (different shapes) — seed
matching here controls data order, not init; the 3-seed spread absorbs init variance.

---

## 4. Data

Identical to Directions 01/02 (MUSDB18, Zenodo 1117372, research-only, nothing
committed; 86/14/50 split; decoded WAV shards on Drive; prep via
`scripts/prepare_data.py` — see [`../01-loss-function-study/MASTER_PLAN.md §4`](../01-loss-function-study/MASTER_PLAN.md)).
No new data work beyond one artifact: the **EDA vocal-energy-by-band table** (computed
from the prep-time energy index; RUN LATER, CPU-minutes) that quantifies where vocal
energy actually lives relative to the two band layouts — motivating figure for the
mechanism analysis (and a design-honesty check: MUSDB18's AAC content is band-limited
≈ 16 kHz, so the top mel band is partly dead spectrum; stated up front).

---

## 5. Model & training

Baseline: SingNet-C1 verbatim (D01 §5). Variants: §3.1. Training config: D01 §7.1
verbatim (this direction's registry at `03-mini-band-split/results/registry.csv`, with
an added `arm` column and the chosen width `c` recorded per run).

---

## 6. Evaluation protocol

### 6.1 Validation (decisions)
D01 §7.2 verbatim: full-track overlap-add on the 14 validation tracks, mean vocals
SI-SDR, every 2 k steps, best checkpoint per run.

### 6.2 Test (exactly one pass for this direction)
The 2 FULL-budget checkpoints (best variant + baseline) on the 50 test tracks,
per-track CSV committed (`results/test_per_track.csv`), paired per-track delta with
bootstrap 95 % CI + Wilcoxon (one pre-registered pair). Sweep arms never touch test.

### 6.3 Per-band mechanism analysis (`singnet/eval/banded.py`, new, unit-tested)
For each system and track, and each analysis band $b$ (we analyze on a **fixed 6-band
grid** — the union of both variants' edges plus a 100 Hz floor split — so the figure is
layout-neutral):
- **Band-limited SI-SDR:** zero all STFT bins outside $b$ in *both* reference and
  estimate (each signal's own STFT), iSTFT, compute SI-SDR on the band-limited
  waveforms. Well-defined (no mask/phase asymmetry), tested on synthetic band-limited
  signals (a tone inside the band scores ≫ 0; energy outside contributes ≈ nothing).
- **Band magnitude error:** mean L1 between reference and estimate magnitudes within
  $b$, normalized by reference band energy; NaN-guarded for empty bands (AAC top end).
The mechanism figure: per-band deltas (each variant − baseline) with per-track spread;
pre-registered reading rule in §2.

### 6.4 Efficiency table (descriptive)
Params (from the committed match table), measured MACs (via a counting hook on one
6-s chunk, CPU), inference RTF on CPU for one validation track — all three arms.

---

## 7. Compute budget & run book

| Item | Runs × time (T4 est.) | Subtotal |
|---|---|---|
| Variant sweep | 6 × ~1.3–1.8 h | 8–11 h |
| Confirmations (FULL) | 2 × ~3.5–4.5 h | 7–9 h |
| Smoke (G1: variants overfit-1-chunk) | 2 × ~0.2 h | ~0.5 h |
| Test pass + banded analysis | < 1 h | < 1 h |
| Baseline cell | 0 (shared; +4–5.5 h only on hash mismatch) | 0 h |
| Contingency (§3.2) | 0 or 2 × ~1.5 h | 0–3 h |
| **Ceiling** | | **≈ 16–25 T4-h** |

**Run book (all RUN LATER; prerequisites: D01 data prep + G1 done):**
```bash
# 0. width selection + param tables (CPU, seconds; already committed by stage D,
#    re-runnable):                                   [safe to run now]
python scripts/match_params.py --report
# 1. dry-run (CPU): all 03 configs instantiate; band partitions asserted [RUN LATER]
python scripts/run_sweep.py --direction 03 --dry-run
# 2. G1 smokes (GPU, ~30 min):                       [RUN LATER]
python -m singnet.train --config 03-mini-band-split/configs/smoke_mel.yaml
python -m singnet.train --config 03-mini-band-split/configs/smoke_uniform.yaml
#    PASS: single-chunk overfit > +20 dB within 2k steps (same G1 bar as baseline).
# 3. the 6 sweep runs (GPU):                         [RUN LATER]
python scripts/run_sweep.py --direction 03 --stage reduced
# 4. analysis freeze (CPU): notebook 02 §1–5; verify baseline hash-share.
# 5. confirmations (GPU):                            [RUN LATER]
python scripts/run_sweep.py --direction 03 --stage full
# 6. single test pass + banded + efficiency (GPU minutes / CPU): [RUN LATER]
python scripts/evaluate.py --checkpoints <2 ckpts> --split test --banded --efficiency
```

---

## 8. Go / no-go gates

| Gate | When | Criterion | On fail |
|---|---|---|---|
| **G0** | now (CPU) | full-repo `pytest` green incl. new tests: band partition exact/disjoint/exhaustive; pad-crop routing round-trip; **param-match assertions (all three arms, ±2 %, exact counts pinned)**; mel-edge closed-form vs code; banded metrics on synthetic signals; baseline hash-equality (D01 cell); forward-shape + gradient-flow for both variants | fix before GPU |
| **G1** | first GPU session | both variant smokes pass the D01 overfit bar | debug the towers, not the sweep |
| **G2** | after sweep | σ_seed < 1.0 dB and no diverged run | pre-registered escalation: +1 seed on the two closest arms (+2 runs) |
| **G3** | before test pass | registry complete; analysis frozen & committed; test paths never in training configs (grep-audit) | fix, re-freeze |

---

## 9. Deliverables & layout (this direction)

```
03-mini-band-split/
  MASTER_PLAN.md            ← this file
  THEORY.md + theory/theory.tex   ← stage B (§10)
  research/                 ← Phase-0 literature (done)
  notebooks/
    01_bandsplit_architecture.ipynb      ← stage C (§11)
    02_bandsplit_experiments.ipynb
  configs/
    base.yaml                        (inherits D01 base; loss l1mag)
    split_mel_seed{0,1,2}.yaml, split_uniform_seed{0,1,2}.yaml
    smoke_mel.yaml, smoke_uniform.yaml
    confirm_{best,baseline}_full.yaml   (best generated at analysis freeze)
    contingency_{baseline,mel}_sisdr_seed0.yaml
  results/                  ← registry.csv, eval CSVs, DEVIATIONS.md (empty now)
  paper/PAPER.md, SEAN-README.md  ← stage E
singnet/models/bandsplit_unet.py  ← stage D (tested)
singnet/eval/banded.py            ← stage D (tested)
scripts/match_params.py           ← stage D (deterministic width search + report)
```

**Code contracts (stage-D deltas only):**
- `singnet.models.bandsplit_unet.BandSplitUNet(edges_bins: list[int], base_width: int)`
  — edges as bin boundaries; classmethods `from_mel_bands(n_bands=3)` /
  `from_uniform_bands(n_bands=3)` computing edges via
  `mel_edges(n_bands, sr=44100, n_fft=4096, fmin=0)` (HTK: $m = 2595\log_{10}(1+f/700)$,
  equal-mel partition, bin = round(f / (sr/n_fft))) — closed-form values asserted in
  tests (expected ≈ bins {0, 142, 597, 2048}, i.e. ≈ {1.53, 6.43} kHz interior edges).
- `scripts/match_params.py`: deterministic; prints/writes the per-module table
  (markdown) for all three arms + chosen c; test asserts the committed c and the ±2 %
  bound; the same table is reproduced in THEORY §4.
- `singnet.eval.banded.band_limited_sisdr(ref, est, edges_hz, ...)` and
  `band_mag_error(...)`; fixed 6-band analysis grid helper.
- `run_sweep.py --direction 03`; registry gains `arm`, `base_width` columns
  (backward-compatible append, as in D02).
- Existing directions must not regress: full `pytest` green at every commit; D01/D02
  configs untouched.

---

## 10. THEORY.md outline (stage-B spec)

1. **The band-split idea, formalized**: what per-band dedicated capacity means for a
   conv encoder (band-local weight sharing vs full-spectrum weight sharing); relation
   to BSRNN's band feature MLPs and Mel-RoFormer's mel bands (what we keep: the
   partition; what we drop: sequence models — and why that isolates the partition).
2. **Mel scale**: HTK formula, equal-mel partition derivation for our (sr, n_fft),
   exact edge computation matching the code; where vocal energy lives (cite the EDA
   artifact); the AAC ~16 kHz band-limit caveat for the top band.
3. **Boundary analysis**: zero-padding at band edges, per-tower receptive fields,
   what artifacts band isolation can introduce (the mechanism behind a possible
   "refuted (negative)" outcome).
4. **Parameter and compute accounting (the crux)**: conv parameter count is
   independent of spatial extent — derive; closed-form P(c) for the variant
   (3 towers + shared decoder), the width-selection equation 3·E(c) + D(c) ≈ P₀ and
   its solution c ≈ 23; the per-module tables for all three arms (numbers must match
   `match_params.py` output and the unit tests); FLOPs(c) derivation showing the
   variants are ~0.5× encoder-FLOPs at matched params — capacity ≠ compute, and what
   each licenses us to claim.
5. **Per-band metrics**: definitions (§6.3), well-definedness, what band-limited
   SI-SDR does and does not measure (energy-weighting caveats near-empty bands).
6. **Statistics**: 3-arm pooled σ_seed; the ordered-chain hypothesis (two one-sided
   effects, why we do NOT correct the two links as multiple comparisons — they are
   jointly pre-registered as a single ordered claim — but do treat partial outcomes as
   distinct pre-registered cells); budget-dependence guard logic.
7. `theory/theory.tex`: compilable standalone mirror.

## 11. Notebook specs (stage-C spec)

Global rules identical to D01 §14 (un-run; RUN-LATER banners with §7 runtimes; logic in
`singnet/`; Colab bootstrap; cross-link D01 notebooks for data prep).
- **01_bandsplit_architecture.ipynb**: the idea in pictures (spectrogram with both
  band layouts overlaid; vocal-energy-by-band table); architecture walkthrough
  (towers, freq-concat skips, pad/crop); the param-match table rendered from
  `match_params.py` with the ±2 % assertion shown; the capacity-vs-compute distinction
  (FLOPs table); routing round-trip demo cell (CPU-runnable later); the boundary-
  artifact question visualized (what the towers cannot see).
- **02_bandsplit_experiments.ipynb**: pre-registration recap verbatim from §2; run
  matrix + shared-baseline accounting; smoke/launch cells (RUN LATER); analysis:
  3-arm ranking with σ_seed band, E1/E2 effect readout with the ordered-chain verdict
  logic implemented, per-band mechanism figure (6-band grid, variant−baseline deltas),
  budget-dependence check (REDUCED vs FULL), efficiency table, test-pass cells;
  **§13 interpretation branches** as labeled markdown stubs; conclusions + what the
  verdict changes for the project (feeds the capacity-vs-data narrative with D02).

---

## 12. Risks & mitigations (direction-specific)

- **Null is the modal outcome** (pre-registered as a finding; the field's own
  replication study 2603.09187 shows even full-scale BSRNN numbers resist
  reproduction). The write-up leads with the controlled-comparison contribution, not
  the direction of the effect.
- **Param-match subtleties**: BN parameters, mask-head 1×1, and pad-crop bookkeeping
  can silently break the ±2 % claim → exact counts pinned in tests; the table is
  committed; any post-hoc width change is a DEVIATIONS.md entry.
- **Boundary artifacts** could penalize variants for reasons unrelated to capacity
  allocation → the per-band figure localizes damage at edges (bands adjacent to
  boundaries) vs band interiors; pre-registered diagnostic.
- **Top-band emptiness** (AAC ≈ 16 kHz ceiling): the mel top band covers partly dead
  spectrum — capacity "wasted" by design; this is a *property of mel spacing on this
  data*, hence part of the treatment, stated in advance (and visible in the EDA table).
- **Optimization-difficulty confound**: variants might train slower per step (deeper
  effective graph via towers) — train-loss curves reported; the FULL-budget
  confirmation guards the "needed longer" objection at 2.5× budget.
- **Init-variance asymmetry** (§3.3 note): absorbed by 3 seeds; no claim rests on a
  single-seed comparison.

---

## 13. Outcome interpretation matrix (pre-registered; full prose in paper/PAPER.md)

| Outcome | Reading | Consequence |
|---|---|---|
| Fully supported (mel > uniform > baseline) | the partition idea itself transfers to compact scale, and *where* capacity goes matters — the SOTA family's core idea is scale-robust | adopt mel-split front-end as an option for the shipped model; efficiency bonus (≈½ encoder FLOPs) strengthens deployment story |
| Partial: E1 only (splitting helps, mel ≈ uniform) | dedicated capacity is the active ingredient; perceptual spacing is not (at 3 bands) | uniform split preferred (simpler); note 3-band granularity as the suspect for mel's null |
| Partial: E2 only (mel > uniform, but uniform ≤ baseline) | splitting alone hurts/does nothing; mel spacing *rescues* it — capacity concentration, not isolation, is the mechanism | intriguing; per-band figure adjudicates; flag for a band-count follow-up |
| Refuted (null, all within band) | **headline:** the band-split advantage is a large-model/large-data phenomenon — not detectable at 10 M params on MUSDB at this budget | baseline stays; the project's capacity-vs-data narrative (with D02's verdict) sharpens; honest boost to the field's replication discourse (cite 2603.09187) |
| Refuted (negative, baseline wins) | band isolation costs more than dedicated capacity buys at this scale (boundary reading via per-band figure) | baseline stays; document the failure mode the SOTA family engineered around (overlaps, band MLPs) |
| Budget-dependent (FULL reverses REDUCED) | short-budget architecture comparisons mislead — a methodological finding | report both; future architecture comparisons at FULL only |
| Mechanism mismatch (positive headline, gains in wrong bands) | effect real but not the dedicated-capacity story | say so; propose the follow-up (band-count/overlap sweep) without running it |

---

## 14. Timeline (Sean's part-time weeks, once GPU exists; after D01 data prep)

| Week | Work | Gate |
|---|---|---|
| 1 | G0 green (now, CPU); dry-run; 2 smokes; start 6-run sweep | G0, G1 |
| 2 | finish sweep; G2; freeze analysis; 2 FULL confirmations | G2 |
| 3 | test pass + banded + efficiency; fill paper; verdicts | G3 |

---

*Frozen on 2026-07-13. Deviations during execution go to `results/DEVIATIONS.md` with
date + reason.*
