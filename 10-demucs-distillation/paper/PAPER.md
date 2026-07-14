# Closing the Gap with a Teacher: Pseudo-Label Distillation from HT-Demucs to a Compact Separator on License-Audited Public Audio

**Draft scaffold — placeholders ⟪like this⟫ are filled only from committed result CSVs.
All interpretations were written before any training (pre-registered 2026-07-13); the
applicable branch is selected once results exist, the rest move to Appendix C.**

---

## Abstract

A compact from-scratch separator is capped by its 86 labeled training songs; the large
teacher it will never beat (HT-Demucs) can label unlimited unlabeled music. We test the
industrial playbook at hobbyist scale: label ~800 license-audited 30-second Creative
Commons clips (FMA; no-derivative licenses excluded by an allowlist filter, manifest
committed for audit) **once** with `htdemucs` (2–4 GPU-hours), and train a fixed
9.8 M-parameter student on three data diets with everything else held constant:
MUSDB-only, MUSDB+pseudo (50/50), and pseudo-only. Pre-registered hypothesis: mixing
pseudo-labeled data closes **≥ 25 %** of the student→teacher SI-SDR gap on the MUSDB
test set, which the teacher never labels. We find gap G = ⟪⟫ dB and closure
Ĉ = ⟪⟫ (CI ⟪⟫) → **⟪supported / partial / null / negative transfer⟫**. Pseudo-only
training reaches ⟪⟫ dB vs real-label training's ⟪⟫ dB. Teacher-error side effects are
audited with the project's silence-leakage metric (SLR ⟪moved / unmoved⟫) and probed
with a trimmed-loss defense arm (⟪⟫). Cross-referenced against the program's sibling
studies — the data-scaling verdict (⟪D02⟫) and the corrupted-target dose–response
chart (D06) — the result composes into a practical answer to the small-model
practitioner's triage: ⟪data / architecture / cleanliness⟫ is where the next dB lives.

## 1. Introduction

Every small separation model has the same ceiling: public labeled data. The high-end
literature routes around it — BSRNN mined unlabeled songs with an activity detector
for semi-supervised fine-tuning; MixIT pre-training helps modern architectures — but
those recipes live at SOTA scale. The compact-student version, using only a shipped
teacher and license-safe public audio at weekend-GPU budgets, is unreported. It is
also the version practitioners actually need: if it works, every small model gains a
free upgrade path; if teacher errors poison the student, that is a warning the
distillation folklore does not currently carry.

Our design isolates the **data effect**: pseudo-labels enter as ordinary training
targets under the unchanged loss and recipe, so the three arms differ only in what
the data is. The teacher's imperfection is treated as a first-class object — this
project's Direction 06 measured, with a closed-form null model, what corrupted
targets cost a student; teacher residuals are *structured* corruption, and the bridge
between the two studies (what transfers, what doesn't) is derived rather than assumed.

**Contributions.** (1) The first controlled compact-scale pseudo-label distillation
measurement for MSS, with a pre-registered gap-closure estimand; (2) a reproducible,
license-audited public-audio pipeline (allowlist filter with no-derivatives excluded,
committed manifest, teacher provenance file, structural anti-leakage guards);
(3) a teacher-error audit combining the SLR silence metric and a trimmed-loss defense
arm; (4) the program-level synthesis: data-scaling (D02) + corruption pricing (D06) +
distillation (this study) as one coherent answer to data-vs-capacity-vs-cleanliness.

## 2. Related Work

**Semi-supervised MSS.** BSRNN's pseudo-label fine-tuning (activity-detector-selected
segments from unlabeled catalogs); Saijo & Bando's MixIT pre-training for MSS;
self-refining label approaches (Koo et al.). All at large scale with bespoke recipes.

**Distillation.** Hinton et al.'s soft-target KD is classification-native; its
temperature mechanism has no regression analogue (derived in THEORY §1) — what
survives for separation is self-training on teacher outputs, which is what we do,
deliberately without a new objective.

**Teachers and ensembles.** HT-Demucs is the ceiling family; MVSep documents
ensembling/teacher selection. Our teacher is the exact engine the downstream
application (StemCraft) ships, making the measured gap the product-relevant one.

**Data.** FMA (Defferrard et al.): 8,000 × 30-s clips with per-track artist-chosen CC
licenses and CC-BY metadata — the licensing nuance (audio licenses vary per track;
ND variants prohibit derivatives) drives our allowlist design.

## 3. Method

**Pipeline.** (i) License filter on FMA metadata: allowlist {CC0/PD, CC-BY, CC-BY-SA,
CC-BY-NC, CC-BY-NC-SA}, deny-by-default, ND hard-vetoed (separated stems are
derivative works); committed ID+license manifest. (ii) Deterministic screen sample
(1,200 clips, seed 0). (iii) One teacher pass (`htdemucs`, version/settings pinned in
a provenance file); 2-stem consistency by construction (accompaniment := mixture −
teacher vocals; the raw 4-stem residual is recorded: ⟪⟫ dB). (iv) Activity screen
(teacher-vocal activity ≥ 20 %, via the project's profile tooling) → the first 800
survivors. (v) Structural guards: the pseudo-data class refuses MUSDB paths at
construction; MUSDB test is never labeled; validation stays MUSDB-only.

**Arms.** `musdb_only` (the program's shared baseline cell, 3 seeds);
`mixed` (chunk-level 50/50 pool draw from a dedicated RNG stream; remix strictly
within-pool, guaranteed by construction, 3 seeds); `distill_only` (1 seed);
`mixed25` (p_FMA = 0.25, 1 seed); `mixed_trim` (TrimmedLoss q = 0.30, 1 seed — the
Direction-06 defense probed against structured teacher error).

**Fixed everything else:** SingNet-C1, `l1mag`, remix+gain+flip, 16 k steps, batch 16,
best checkpoint by MUSDB validation SI-SDR.

## 4. Experimental Setup

Gap G = s_T − s_base and closure Ĉ = (s_mix − s_base)/G on the 50-track MUSDB test
set, evaluated in one session (students + teacher + do-nothing + oracle IRM; SI-SDR,
SI-SDRi, SLR at −60 dBFS, museval secondary). **H-10:** supported iff
s_mix − s_base > σ_seed and Ĉ ≥ 0.25; partial iff real but smaller; refuted-null iff
within σ_seed; negative transfer iff below −σ_seed. Precondition: G > 2 dB (else the
premise failed and the study reframes, pre-registered). σ_seed pooled from the two
3-seed cells; paired-track bootstrap + Wilcoxon on the one pre-registered pair;
delta-method CI on Ĉ.

## 5. Results

> Generated from `results/registry.csv`, `results/test_session.csv`, the license
> manifest, and the teacher provenance file; figures regenerate from CSVs.

### 5.1 Pipeline facts
License filter: ⟪n_allowlisted⟫/8,000 clips passed; screen sample 1,200; activity
screen kept ⟪⟫ (threshold ⟪20 % / 10 % fallback⟫) → 800 used. Teacher: demucs ⟪ver⟫,
consistency residual ⟪⟫ dB. Pseudo-pool total ⟪⟫ h vs MUSDB train ⟪5.5⟫ h.

### 5.2 The gap-closure ladder (test set)

| System | vocals SI-SDR (dB) | SLR (dB) |
|---|---|---|
| do-nothing | ⟪⟫ | 0 (def.) |
| `musdb_only` (3 seeds) | ⟪ ± ⟫ | ⟪⟫ |
| `distill_only` | ⟪⟫ | ⟪⟫ |
| `mixed25` | ⟪⟫ | ⟪⟫ |
| **`mixed`** (3 seeds) | ⟪ ± ⟫ | ⟪⟫ |
| `mixed_trim` | ⟪⟫ | ⟪⟫ |
| teacher (htdemucs) | ⟪⟫ | ⟪⟫ |
| oracle IRM | ⟪⟫ | ⟪⟫ |

G = ⟪⟫ dB (precondition ⟪met/failed⟫). Δ(mixed − musdb_only) = ⟪⟫ dB
(paired CI ⟪⟫, Wilcoxon p = ⟪⟫). **Ĉ = ⟪⟫ (CI ⟪⟫) → H-10: ⟪verdict⟫.**
⟪HEADLINE FIGURE: the ladder with the closure bracket drawn.⟫

### 5.3 Secondaries
`distill_only` vs `musdb_only`: ⟪⟫. p_FMA sensitivity: ⟪⟫. Trim arm: ⟪⟫.
SLR across arms: ⟪table⟫ — pseudo-training ⟪did / did not⟫ transfer teacher silence
behavior. Program cross-reference: D02's scaling verdict was ⟪⟫; D06's chart prices
30 % unstructured bleed at ⟪⟫ dB — the teacher's structured error produced ⟪⟫.

## 6. Discussion — pre-registered interpretation branches

**⟪SELECT applicable; unselected → Appendix C.⟫**

### Branch S (supported: Ĉ ≥ 25 %)
The industrial playbook works at hobbyist scale: ⟪⟫ GPU-hours of teacher labeling on
free, license-safe audio bought ⟪⟫ dB — ⟪Ĉ⟫ of the way to a teacher 10× the size.
Composed with D02 ⟪if data-starved: exactly as the scaling curve predicted — the
constraint was data, and pseudo-data is data⟫, the compact-model roadmap is clear:
scale the pseudo-pool before touching the architecture (the named, not-run, next
step: does Ĉ grow with pool size?). StemCraft's from-scratch engine gains a concrete
upgrade path that never touches restricted data.

### Branch P (partial: real gain, Ĉ < 25 %)
Pseudo-data helps (⟪⟫ dB) but the bar was ambitious: ⟪Ĉ⟫ against the pre-registered
25 %. The two named suspects, distinguishable by the secondaries: domain shift (FMA's
production/genre spread vs MUSDB — if `mixed25` ≈ `mixed`, the marginal pseudo-value
saturates fast) and teacher-error ceiling (if `mixed_trim` > `mixed`, error was
biting). Follow-ups named: larger/genre-matched pools; teacher-SLR-filtered clips.

### Branch N (null)
Six-plus hours of shifted pseudo-audio added nothing detectable over MUSDB + remix.
The program synthesis carries the value: ⟪if D02 said saturated: this null was
*predicted* by the scaling verdict — the constraint at this capacity is the model,
not data, and both studies now say so independently⟫ ⟪if D02 said data-starved: the
*kind* of data mattered — remixed real stems ≠ teacher-labeled shifted audio; the gap
between the two is the finding⟫.

### Branch NT (negative transfer)
Teacher errors poisoned the student (−⟪⟫ dB): corrupted-target training in the wild.
The D06 bridge quantifies the reading — structured residuals hurt ⟪more/less⟫ than
the equivalent-energy unstructured bleed priced by the dose–response chart —
and the trim arm ⟪recovered ⟪⟫ / did not defend⟫. The cautionary tale most
distillation write-ups lack, with the mechanism instrumented.

### Branch D+ (`distill_only` ≥ `musdb_only`)
Pseudo-labels rival real labels end-to-end (⟪⟫ vs ⟪⟫ dB, 1 seed — flagged for
replication before any strong claim). If it holds, labeled data's moat at compact
scale is thinner than assumed; the license-audited pipeline becomes the story.

### Branch SLR (silence transfer detected)
Pseudo-training moved SLR by ⟪⟫ dB — the teacher's silence behavior is heritable.
Direction 08's metric earns its program-level keep; teacher-SLR-filtering of pseudo
clips is the named fix.

### 6.x Threats to validity (written before results)
One teacher (results may be htdemucs-specific); 30-s clips (less within-track
diversity; part of the "cheap public audio" treatment); FMA↔MUSDB domain gap is
uncontrolled by design (it *is* the realistic condition); 1-seed secondaries are
directional only; the teacher saw MUSDB train during its own training (legitimate for
a ceiling anchor; the student never sees MUSDB test through any path — structural
guards + audit); Ĉ is protocol-relative (frozen SI-SDR protocol; museval secondary
never mixed); pool mixing at chunk level entangles data source with effective epoch
count over each pool (stated; p_FMA sensitivity partially probes it).

## 7. Conclusion

⟪Two paragraphs from the selected branches: (1) the verdict with G, Ĉ, and the
side-effect audit; (2) the program-level synthesis — what Directions 02 + 06 + 10
jointly tell a small-model practitioner about where the next dB lives, and what
StemCraft does with it.⟫

## References

1. Hinton, Vinyals, Dean. *Distilling the Knowledge in a Neural Network.*
   arXiv:1503.02531.
2. Rouard, Massa, Défossez. *Hybrid Transformers for Music Source Separation.*
   (HT-Demucs.) arXiv:2211.08553.
3. Luo, Yu. *Music Source Separation with Band-Split RNN.* arXiv:2209.15174.
4. Saijo, Bando. *(MixIT pre-training for MSS.)* arXiv:2505.07631.
5. Defferrard, Benzi, Vandergheynst, Bresson. *FMA: A Dataset for Music Analysis.*
   ISMIR 2017. arXiv:1612.01840.
6. Solovyev, Stempkovskiy, Habruseva. *Benchmarks and Leaderboards for Sound
   Demixing Tasks.* arXiv:2305.07489.
7. Koo et al. *(Self-refining pseudo labels for MSS.)* arXiv:2307.12576.
8. Le Roux, Wisdom, Erdogan, Hershey. *SDR — Half-Baked or Well Done?* ICASSP 2019.
   arXiv:1811.02508.
9. Rafii, Liutkus, Stöter, Mimilakis, Bittner. *MUSDB18.* Zenodo 1117372.

## Appendix A — Reproducibility

Commands: MASTER_PLAN.md §6 run book. License filter verified in review (allow/deny
table incl. ND hard-veto and URL forms); MUSDB-path guard verified live (nested path
refused at construction). Committed audit artifacts: license manifest (IDs +
licenses), teacher provenance file. Shared baseline cell hash `a97d5400e994` (sixth
direction). Suite at scaffold time: 442 passed, 1 skipped. GPU spend: ⟪actual⟫ vs
≈12–17 T4-h ceiling. Deviations: `results/DEVIATIONS.md`.

## Appendix B — Per-track tables, license manifest summary, SLR grids ⟪generated⟫

## Appendix C — Pre-registered alternative readings (unselected branches) ⟪moved here⟫
