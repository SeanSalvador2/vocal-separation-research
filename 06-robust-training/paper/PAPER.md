# How Much Does Dirty Ground Truth Cost? A Dose–Response Study of Stem Bleed for Compact Music Source Separation, with a Closed-Form Null Model and a Classical Defense

**Draft scaffold — placeholders ⟪like this⟫ are filled only from committed result CSVs.
All interpretations were written before any training (pre-registered 2026-07-13); the
applicable branch is selected once results exist, the rest move to Appendix C.**

---

## Abstract

Supervised source separation trusts its targets, but real stems carry bleed: the SDX'23
challenge introduced entire corrupted-training datasets (SDXDB23_Bleeding /
LabelNoise) and flagged robustness to dirty training data as under-studied. We measure
the dose–response directly: MUSDB18 training targets are corrupted with a controlled,
mixture-invariant bleed knob — vocals ṽ = v + ε·a, accompaniment ã = (1−ε)·a, mixture
unchanged — at ε ∈ {0, 5, 15, 30 %}, and a fixed compact mask U-Net (9.8 M params) is
trained at each level and evaluated on *clean* data. Uniquely, the curve is judged
against a **closed-form null model**: a network that perfectly fits the corrupted
conditional must leak exactly ε·a, so its clean SI-SDR is analytically computable per
track (≈ −20 log₁₀ ε anchored by each track's vocal/accompaniment energy ratio). We
find the measured curve sits ⟪at / above / below⟫ the prediction (⟪gap⟫ dB at
ε = 30 %), i.e. ⟪the model faithfully learns its corruption / exhibits implicit
robustness / suffers additional optimization damage⟫. A classical cheap defense —
trimmed loss (train on the lowest-loss chunks; ITLM lineage) — recovers ⟪ρ⟫ of the
damage at ε = 30 % ⟪at a clean-data cost of ⟪⟫ dB⟫, with per-chunk telemetry
⟪confirming / refuting⟫ the predicted selection mechanism (trimming ranks chunks by
accompaniment energy under uniform bleed). ⟪One sentence: the practical
data-cleanliness guidance in dB per ε.⟫

## 1. Introduction

Every supervised separator is trained on stems someone promised were isolated. The
promise is routinely broken — microphone bleed, session compromises, mislabeled
content — and the SDX'23 organizers elevated this from anecdote to benchmark by
shipping corrupted training sets. What the literature still lacks is quantitative
guidance: *how many dB does a given contamination level cost*, is the damage
information-theoretic or optimization-driven, and does the oldest noisy-label defense
transfer?

Our design answers with three unusual properties. First, the corruption is a **single
continuous knob** applied by construction to data we already have, preserving the
mixture exactly (the redistribution form real bleed takes). Second — the piece we have
not seen elsewhere — the experiment has a **quantitative null model**: if training
converges to the corrupted conditional, the clean-eval score is *computable in closed
form* from the stems, so the measured-minus-predicted gap cleanly separates "the model
learns what you feed it" from implicit robustness or added optimization damage. Third,
the defense test is honest about its own theory: trimmed loss assumes some samples are
clean, whereas our bleed is uniform — we derive the only mechanism by which it could
still work (chunks with quiet accompaniment are *effectively* cleaner, so small-loss
selection becomes curriculum-by-cleanliness) and instrument the training loop to test
that mechanism directly.

**Contributions.** (1) The first dose–response curve for stem-bleed corruption of a
compact MSS model, with seed bands; (2) the closed-form prediction-line methodology
for corrupted-target training; (3) a mechanism-instrumented evaluation of trimmed loss
under uniform corruption, including its clean-data cost; (4) reproducible tooling in
which corruption structurally cannot touch evaluation data.

## 2. Related Work

**Robust MSS.** Fabbro et al. (SDX'23, TISMIR 2024) introduced the corrupted-data
tracks; the winning system used loss-masking against noisy examples (TFC-TDF-UNet v3,
arXiv 2306.09382); Koo et al. (2307.12576) refine labels by self-training; recent
"blind data cleaning" for MSS (2510.15409) filters training data post hoc. None
reports a controlled ε-curve or a closed-form null.

**Noisy-label learning.** The small-loss principle: DNNs fit clean structure before
memorizing noise (Arpit et al., 1706.05394); iterative trimmed loss has recovery
guarantees under *sample-level* contamination (ITLM — Shen & Sanghavi, 1810.11874);
Co-teaching (1804.06872) and GCE (1805.07836) are the standard alternatives. Our
setting deliberately violates the sample-level assumption (uniform bleed) — the
transfer question is the point. LoRA's rank-limited noise resistance (2602.00084)
connects this direction to the project's Direction 05 as future work.

## 3. Method

**Corruption (the treatment).** At load time, training-split stems only:
ṽ = v + ε·a, ã = (1−ε)·a (same-track accompaniment; ε constant per arm; no
renormalization — real bleed adds energy, and the null model uses the same
construction). Mixture invariance ṽ + ã = v + a holds exactly; augmentation (remix,
gain, flip) then operates on corrupted stems, exactly as it would for a practitioner
with a corrupted dataset. Corruption of validation or test splits is structurally
impossible (constructor raises; dataset wiring raises; both unit-tested).

**The null model (prediction line).** If the trained model outputs ṽ for the true
conditional, its clean-eval residual is ε·a, giving per track
SI-SDR(v + εa, v) — computed exactly (projection form) and in the orthogonal
approximation 10 log₁₀(‖v‖² / (ε²‖a‖²)); the two agree to ⟪ortho_gap⟫ dB on MUSDB
(cos similarity of stems ≈ 0). Verified numerically in review: exact = closed form to
3 decimals on synthetic orthogonal stems; slope spacing = −20 log₁₀ ε.

**The defense.** TrimmedLoss(q): per step, rank the batch's per-chunk losses (fp32),
keep the ⌈(1−q)·B⌉ smallest (q = 0.30 ⇒ keep 12/16), average those; gradients flow
only through kept chunks. Telemetry logs kept-vs-dropped accompaniment energies every
500 steps — the §2 mechanism's testable signature.

**Fixed everything else:** SingNet-C1 (9,835,745 params), `l1mag`, full augmentation,
REDUCED = 16 k steps, seeds {0,1,2} on the three anchor cells, best-checkpoint by
clean validation SI-SDR.

## 4. Experimental Setup

Run matrix: clean ε=0 (3 seeds; the project's shared baseline cell, hash-verified) /
ε = 0.05, 0.15 (1 seed) / ε = 0.30 (3 seeds) / trim q=0.30 @ ε=0.30 (3 seeds) /
trim q=0.10 @ ε=0.30 (1 seed) / trim q=0.30 @ ε=0 (clean-cost control, 1 seed).
**H-06a** supported iff s(0) − s(0.30) > max(σ_seed, 0.5 dB) with a monotone curve
(±σ_seed tolerance); the shape verdict is the measured-vs-prediction gap.
**H-06b** supported iff trim recovers > σ_seed at ε = 0.30; recovery fraction
ρ = (s_trim(0.30) − s(0.30)) / (s(0) − s(0.30)) with delta-method CI.
One test pass (clean 50-track set): the nine 3-seed-cell checkpoints; paired bootstrap
+ Wilcoxon on (bleed30 − clean) and (trim30 − bleed30).

## 5. Results

> Generated from `results/registry.csv`, `results/test_per_track.csv`,
> `results/trim_energy_stats*.csv`, and `singnet.analysis.bleed` outputs.

### 5.1 The dose–response curve vs the null model

| ε | val SI-SDR (dB) | prediction P(ε) (dB) | gap (meas − pred) |
|---|---|---|---|
| 0 | ⟪ ± seed⟫ | — (clean ceiling n/a) | — |
| 0.05 | ⟪⟫ | ⟪⟫ | ⟪⟫ |
| 0.15 | ⟪⟫ | ⟪⟫ | ⟪⟫ |
| 0.30 | ⟪ ± seed⟫ | ⟪⟫ | ⟪⟫ |

σ_seed = ⟪⟫ dB. **H-06a: ⟪supported / refuted-robust / mixed⟫**; shape verdict:
**⟪at / above / below⟫ prediction** (⟪⟫ dB at ε = 0.30).
⟪Headline figure: measured curve + σ_seed band + prediction line.⟫
Orthogonality check on real stems: mean |cos(v, a)| = ⟪⟫; exact-vs-orth gap ⟪⟫ dB.

### 5.2 The defense

| Arm | val SI-SDR (dB) |
|---|---|
| bleed30 | ⟪ ± seed⟫ |
| trim30 (q=0.30) | ⟪ ± seed⟫ |
| trim30_q10 | ⟪⟫ |
| trim_clean control (q=0.30 @ ε=0) | ⟪⟫ vs clean ⟪⟫ |

Recovery ρ = ⟪⟫ (CI ⟪⟫) → **H-06b: ⟪supported / refuted / not evaluable⟫**.
Clean-data cost of trimming: ⟪⟫ dB (⟪within / beyond⟫ σ_seed).
**Mechanism telemetry:** kept-chunk accompaniment energy ⟪< / ≈⟫ dropped-chunk energy
(⟪figure: kept-vs-dropped energy distributions over training⟫) →
⟪confirms / refutes⟫ the curriculum-by-cleanliness mechanism.

### 5.3 Test pass (clean, 9 checkpoints)
⟪Paired table with CIs + Wilcoxon for the two pre-registered pairs; per-track CSV
committed.⟫ Training-curve exhibit: ⟪Arpit-style late clean/corrupted divergence
present / absent⟫.

## 6. Discussion — pre-registered interpretation branches

**⟪SELECT applicable; unselected → Appendix C.⟫**

### 6.1 The curve (H-06a × shape)

**Branch A/P (supported, ≈ prediction).** The model learns what it is fed, to within
seed noise — no free robustness. The curve *is* the practical calibration chart:
ε % bleed costs ⟪table⟫ dB, so stems must be cleaner than ε ≈ ⟪⟫ for a target quality
of ⟪⟫ dB. Direction 10's teacher-quality concern (pseudo-labels are corrupted targets
by construction) inherits this chart directly — we can now price teacher error in dB.

**Branch A/A (supported, above prediction).** Training rejects part of the bleed:
the measured curve beats the perfectly-corrupted-fit bound by ⟪⟫ dB. Candidate
mechanisms (pre-declared, to be distinguished in future work, not claimed): remix
augmentation decorrelates v from its bleed partner across examples (the εa term is
inconsistent with the remixed mixture's accompaniment ⟪check: bleed is applied before
remix, so the leaked a_i ≠ mixture's a_j — the inconsistency the model may exploit⟫);
the sigmoid mask cap limits expressible leakage in bins where εa exceeds the mixture.
This is the happiest outcome for practitioners: mildly dirty data is partly
self-correcting under the standard recipe.

**Branch A/B (supported, below prediction).** Corruption costs more than the
information limit — it also degrades optimization (consistent with ⟪curve/grad-norm
exhibits⟫). Robust losses (GCE-style), not just data cleaning, become the recommended
follow-up.

**Branch R (refuted-robust).** 30 % bleed within seed noise while the prediction line
sits ⟪⟫ dB below: strong implicit robustness — the surprising headline, immediately
qualified by the mechanism candidates above and flagged for replication at FULL
budget (pre-registered follow-up before any strong claim).

### 6.2 The defense (H-06b × control × telemetry)

**Branch D-yes.** Trimming recovers ρ = ⟪⟫ of the damage for one line of code, and the
telemetry shows the predicted energy-ranking (kept ⟨a⟩-energy ⟪⟫ vs dropped ⟪⟫) — the
small-loss trick generalizes beyond its sample-level home turf via effective-
cleanliness selection. Adopt as an optional flag project-wide; recommend for suspected-
bleed datasets ⟪with the clean-cost caveat if the control fired⟫.

**Branch D-no.** No recovery beyond noise. If telemetry shows no energy separation,
the selection signal simply doesn't exist under uniform bleed at this loss surface —
an honest boundary for the classic trick (its assumptions matter); if separation
exists but quality didn't move, selection wasn't the bottleneck. Recommend robust
losses / cleaning (2510.15409) instead.

**Branch D-cost.** The clean-data control fired (trimming costs ⟪⟫ dB at ε = 0):
trimming discards hard-but-clean chunks (dense mixes, quiet vocals) as THEORY §5
predicted it might. The defense is conditional: apply only under suspected corruption.

### 6.3 Threats to validity (written before results)

The corruption is a controlled simplification of real bleed (constant ε, one
direction, same-track partner; SDX'23's is per-song, all-directions); single
architecture/loss/budget (contingency pair re-checks under the alternative loss ⟪run?
result⟫); middle ε points are single-seed (tolerance-banded); trimming granularity is
batch-level (16 chunks; keep = 12/16 at q = 0.30); the prediction line assumes
convergence to the conditional — at 16 k steps undertraining could masquerade as
"below prediction" (checked against the ε = 0 arm's own convergence ⟪exhibit⟫);
telemetry's ⟨a⟩-energy is windowed mean-square, blind to spectral overlap structure.

## 7. Conclusion

⟪Two paragraphs from the selected branches: (1) the dB-per-ε chart, the null-model
verdict, and the defense's recovery/cost; (2) the practical data-cleanliness guidance
for small-scale MSS training and what it changes in this project — Direction 10's
teacher-error budget, and whether trimming ships as a training flag.⟫

## References

1. Fabbro et al. *The Sound Demixing Challenge 2023 — Music Demixing Track.* TISMIR
   2024. arXiv:2308.06979.
2. Shen, Sanghavi. *Learning with Bad Training Data via Iterative Trimmed Loss
   Minimization.* ICML 2019. arXiv:1810.11874.
3. Arpit et al. *A Closer Look at Memorization in Deep Networks.* ICML 2017.
   arXiv:1706.05394.
4. Han et al. *Co-teaching: Robust Training of Deep Neural Networks with Extremely
   Noisy Labels.* NeurIPS 2018. arXiv:1804.06872.
5. Zhang, Sabuncu. *Generalized Cross Entropy Loss for Training Deep Neural Networks
   with Noisy Labels.* NeurIPS 2018. arXiv:1805.07836.
6. Kim, Lee, Jung. *(TFC-TDF-UNet v3 / SDX'23 loss-masking.)* arXiv:2306.09382.
7. Koo et al. *(Self-refining pseudo labels for MSS.)* arXiv:2307.12576.
8. *(Blind data cleaning for MSS.)* arXiv:2510.15409.
9. Steele. *Why LoRA Resists Label Noise.* arXiv:2602.00084.
10. Le Roux, Wisdom, Erdogan, Hershey. *SDR — Half-Baked or Well Done?* ICASSP 2019.
    arXiv:1811.02508.
11. Rafii, Liutkus, Stöter, Mimilakis, Bittner. *MUSDB18.* Zenodo 1117372.

## Appendix A — Reproducibility

Commands: MASTER_PLAN.md §6 run book. Structural eval-split guard:
`singnet/data/corrupt.py` (`EvalSplitCorruptionError`, two choke points, unit-tested).
Prediction line verified in review: exact = closed form to 3 decimals on orthogonal
synthetic stems; −20 log₁₀ ε spacing confirmed. Suite at scaffold time: 292 passed,
1 skipped. Shared ε = 0 cell hash: `a97d5400e994` (≡ Directions 01/02/03 baseline).
GPU spend: ⟪actual⟫ vs ≈14–19 T4-h ceiling. Deviations: `results/DEVIATIONS.md`
(notably: keep-count convention pinned to ⌈(1−q)B⌉ = 12/16, correcting a plan-internal
parenthetical that said 11/16).

## Appendix B — Per-track tables, telemetry distributions, training curves ⟪generated⟫

## Appendix C — Pre-registered alternative readings (unselected branches) ⟪moved here⟫
