# The Cost of Quiet: A Silence-Leakage Metric and a Controlled Chunk-Sampling Study for Karaoke Source Separation

**Draft scaffold — placeholders ⟪like this⟫ are filled only from committed result CSVs.
All interpretations were written before any training (pre-registered 2026-07-13); the
applicable branch is selected once results exist, the rest move to Appendix C.**

---

## Abstract

The most audible failure of a karaoke separator is ghost vocals — energy leaking into
the vocal estimate during instrumental passages — and the field's standard metrics
cannot see it: museval drops silent-reference frames from its median (verified from
source), and SI-SDR's projection is singular on a silent target. We contribute (1) the
**Silence-Leakage Ratio (SLR)**: a ~80-line, exhaustively unit-tested metric with
interpretable anchors — a do-nothing separator scores exactly 0 dB, a 10 % energy leak
scores exactly −10 dB, perfect silence hits the ε floor (all three verified
numerically); and (2) the controlled experiment the training lore never received: four
chunk-sampling policies — uniform (the code-verified Open-Unmix baseline),
energy-weighted with a uniform floor, drop-silent, and an annealed curriculum — trained
otherwise bit-identically (3 seeds on anchor arms) and scored on **both** overall
vocals SI-SDR and SLR. Pre-registered bets: energy-weighting buys overall quality
(H-08a) and dropping silent chunks poisons silence behavior (H-08b), yielding a real
quality-vs-leakage tradeoff. We find ⟪H-08a verdict + Δ dB⟫ and ⟪H-08b verdict + Δ SLR
dB⟫; the (SI-SDR, SLR) plane shows ⟪the tradeoff / a free lunch / a non-lever⟫, and
the shipped recommendation for our karaoke engine is ⟪policy⟫. Novelty is bounded
honestly: recent work attacks sparse-vocal robustness via architecture (BSMamba2,
arXiv 2508.14556); the metric and the policy-vs-leakage tradeoff are, to our
knowledge, new.

## 1. Introduction

Vocal separation models are trained on randomly cropped chunks, and music is mostly
*not* singing: MUSDB18 tracks contain long instrumental stretches (⟪measured activity
statistics⟫). Practitioners have always handled this informally — Open-Unmix crops
uniformly at random (we verified the code), community lore suggests biasing toward
vocal-active regions, and some pipelines silently drop silent chunks as "wasted
compute." Two things are wrong with this picture. First, nobody has measured what
these policies do, holding everything else fixed. Second — and worse — the standard
evaluation stack is structurally incapable of catching the most user-salient failure
a policy could cause: a model that never trained on silence has never been taught to
output it, and if it hums along during instrumental breaks, museval's silent-frame
NaN convention and SI-SDR's silent-target singularity both look away.

This paper fixes the measurement first and then runs the experiment.

**Contributions.** (1) SLR: a defined, anchored, unit-tested silence-leakage metric
that fills a verified blind spot in BSS evaluation practice; (2) the first controlled
sampling-policy comparison for MSS training (uniform / energy / drop / curriculum,
bit-identical otherwise), evaluated on the two-metric plane; (3) exposure telemetry
linking each policy's *realized* silent-chunk diet to its silence behavior (mechanism,
not just outcome); (4) a shipped recommendation for karaoke products, and the metric
itself as reusable ~80-line code.

## 2. Related Work

**Metrics and silence.** BSS-Eval/museval compute frame-wise metrics and set frames
with silent references to NaN, excluding them from the median (verified from
`museval` source in our Phase-0 review); SI-SDR (Le Roux et al., 2019) divides by the
reference energy and is undefined on silence. Consequently, published SDR tables carry
no information about silent-region behavior. Our SLR is deliberately minimal — an
energy ratio over ground-truth-silent regions with pinned identification constants —
rather than a perceptual model; its limitations (energy-blind to timbre) are stated
and delegated to listening checks.

**The silence problem.** BSMamba2 ("Mamba2 Meets Silence", 2508.14556) targets
sparse-vocal robustness via long-range state-space modeling — the architecture lever,
reporting 11.03 dB cSDR; Demucs's augmentation *injects* silence; BSRNN's activity
detector mines pseudo-labels (not labeled-data sampling). None defines a leakage
metric or compares sampling policies controlled; our gap-check (logged 2026-07-13)
found no such study through mid-2026.

**This project.** The uniform arm is the SingNet project's shared baseline cell
(config-hash-identical across five directions), inheriting its tested stack; the
−60 dBFS silence threshold is shared project-wide (Direction 01's SI-SDR-loss guard
uses the same constant).

## 3. The Silence-Leakage Ratio

**Identification.** Ground-truth vocal RMS over 100 ms frames (50 ms hop); frames
below θ = −60 dBFS form runs; runs ≥ 0.5 s constitute the silent-region set R_sil.
θ ∈ {−50, −60, −70} is a mandatory sensitivity axis on every conclusion.

**Definition.** SLR = 10·log₁₀[(Σ_{R_sil} v̂² + ε) / (Σ_{R_sil} x² + ε)], ε = 10⁻⁸;
per track, NaN if R_sil is empty (valid-n always reported); aggregate = mean over
valid tracks.

**Anchors and properties** (derived in THEORY §3; verified numerically in review):
v̂ = x ⇒ exactly 0 dB; a 10 % energy leak ⇒ exactly −10 dB; v̂ = 0 ⇒ ε floor
(≈ −123 dB on our constructions); invariant to joint gain; monotone in leaked energy;
defined for every estimator (no singularity). Limitations: an energy metric — blind to
*what* leaks (a −12 dB hi-hat and a −12 dB vocal ghost score alike); sample-domain.

## 4. Experimental Setup

Fixed stack: SingNet-C1 (9,835,745 params), `l1mag` loss, full augmentation, 16 k
steps, batch 16, seeds {0,1,2} on uniform/energy/drop (curriculum 2 seeds), best
checkpoint by validation SI-SDR — **selection is blind to SLR by construction**
(enforced in code and asserted by a test), so leakage comparisons carry no selection
bias. Policies (exact forms in MASTER_PLAN §4.1): uniform (legacy path, bit-identical
— regression-tested); energy (weights ∝ (1−λ)·E/ΣE + λ/N, λ = 0.1: down-weight, never
exclude); drop (support excludes windows with vocal RMS < θ); curriculum (λ: 1.0 → 0.1
over the first half of training). Non-uniform policies draw on the 1-s profile grid
(a logged implementation clarification); a dedicated RNG stream isolates sampling from
augmentation. Hypotheses: **H-08a** q(energy) − q(uniform) > σ_seed^q; **H-08b**
ℓ(drop) − ℓ(uniform) > σ_seed^ℓ. One test pass at the end (11 checkpoints +
do-nothing + oracle IRM; SI-SDR, SLR at three θ, museval secondary).

## 5. Results

> Generated from `results/registry.csv`, exposure CSVs, and
> `results/test_per_track.csv`; figures regenerate from CSVs.

### 5.1 Validation outcomes

| Policy | val SI-SDR (dB) | val SLR (dB, θ=−60) | silent exposure (realized) |
|---|---|---|---|
| uniform | ⟪ ± seed⟫ | ⟪ ± seed⟫ | ⟪⟫ % |
| energy | ⟪ ± seed⟫ | ⟪ ± seed⟫ | ⟪⟫ % |
| drop | ⟪ ± seed⟫ | ⟪ ± seed⟫ | ≈ 0 % (by construction; observed ⟪⟫) |
| curriculum | ⟪2 seeds⟫ | ⟪⟫ | ⟪⟫ % |

σ_seed^q = ⟪⟫ dB; σ_seed^ℓ = ⟪⟫ dB. **H-08a: ⟪…⟫** (Δ = ⟪⟫).
**H-08b: ⟪…⟫** (Δ = ⟪⟫). Valid-SLR n = ⟪⟫/14 val tracks.
⟪HEADLINE FIGURE: the (SI-SDR, SLR) plane — four policies with per-seed scatter,
do-nothing at 0 dB SLR, oracle IRM at ⟪⟫.⟫

### 5.2 Sensitivity and mechanism
θ-sensitivity: conclusions ⟪stable / flip at θ=⟪⟫⟫ (grid table). Exposure → SLR:
⟪the mechanism plot: realized silent-chunk fraction vs SLR across arms/seeds⟫ —
⟪supports / complicates⟫ the never-learned-silence account. Best-vs-final checkpoint
SLR drift: ⟪⟫.

### 5.3 Test pass
⟪Paired table with CIs + Wilcoxon for the two pre-registered pairs; SLR at three θ;
museval secondary — expected near-blind to policy differences on silence: ⟪confirmed?
one line⟫.⟫

## 6. Discussion — pre-registered interpretation branches

**⟪SELECT applicable; unselected → Appendix C.⟫**

### Branch T (both supported — the predicted tradeoff)
Sampling is a real lever with a real price: energy-weighting buys ⟪⟫ dB of overall
quality while dropping silence costs ⟪⟫ dB of SLR. The plane's frontier is the
deliverable: quality-focused training should ship `energy`; karaoke-critical products
should ship ⟪uniform / curriculum⟫, or `energy` with its floor λ raised — the knob is
now measured, not folkloric. SLR joins the project's standard evaluation battery, and
the shipped SingNet engine documents its policy choice.

### Branch F (H-08a yes, H-08b no — free lunch)
Concentrating training on vocal-active regions pays ⟪⟫ dB and silence behavior
survives anyway — even `drop`'s SLR stays within ⟪⟫ of uniform's. The mechanism
reading: the sigmoid mask's multiplicative structure gives silence "for free" (a mask
can't hallucinate energy absent from the mixture bins with low vocal evidence), and
16 k steps of remixed audio still expose the model to enough quiet vocals
incidentally (exposure telemetry: ⟪⟫ %). Ship `energy` everywhere; retire the fear.

### Branch P (H-08a no, H-08b yes — pure downside)
Activity-weighted sampling buys nothing here, but discarding silence still poisons
leakage by ⟪⟫ dB — sampling cleverness is all risk, no reward at this scale. Ship
`uniform`; publish the caution (this is the branch most useful to practitioners
copying lore).

### Branch N (both refuted — a non-lever)
All four arms coincide within both noise bands. Exposure telemetry adjudicates: if
realized exposures genuinely differed (⟪⟫ % vs ⟪⟫ %) yet outcomes didn't, chunk
sampling is a non-lever for compact mask models at this budget — an honest null that
saves the community tuning effort; if exposures barely differed, MUSDB's activity
structure made the policies near-equivalent in practice (a dataset statement, not a
training one), and the study's transferability note says so.

### Branch C+ / C− (curriculum dominates / is dominated)
⟪If it dominates both axes: annealed exposure captures both goods; ship it; schedule
sensitivity named as future work. If dominated: complexity buys nothing; one line.⟫

### Branch V (SLR validity issue: valid-n too small)
⟪Fires only if G2's valid-n gate failed: conclusions rest on ⟪n⟫ tracks; θ/L_min
sensitivity becomes the primary exhibit and the metric's constants are revisited via
the deviation log — reported as a finding about MUSDB's silence structure.⟫

### 6.x Threats to validity (written before results)
SLR is an energy metric (audibility claims deferred to the umbrella listening check);
θ/L_min constants define the measurand (three-θ sensitivity mandatory); non-uniform
policies sample on a 1-s grid vs uniform's sample-resolution (logged; expected
immaterial at 6-s windows, stated not proven); the drop arm interacts with remix
(silent-vocal chunks excluded before remixing — the policy applies to all sources
uniformly, stated); single scale/budget/loss (contingency pair re-checks under the
alternative loss ⟪run? result⟫); MUSDB's silence statistics may not transfer to other
catalogs.

## 7. Conclusion

⟪Two paragraphs from the selected branches: (1) verdicts with effect sizes, the plane,
and the mechanism telemetry; (2) the shipped recommendation for karaoke training, SLR
as the reusable artifact — the field can rank ghost-vocal behavior for the first time
in one ~80-line file — and what the umbrella project adopts (SLR in every later
evaluation).⟫

## References

1. Le Roux, Wisdom, Erdogan, Hershey. *SDR — Half-Baked or Well Done?* ICASSP 2019.
   arXiv:1811.02508.
2. Stöter, Liutkus, Ito. *The 2018 Signal Separation Evaluation Campaign.* (museval /
   BSS-Eval v4.) arXiv:1804.06267.
3. Stöter, Uhlich, Liutkus, Mitsufuji. *Open-Unmix.* JOSS 2019. 10.21105/joss.01667.
4. Kim, Choi. *Mamba2 Meets Silence: Robust Vocal Source Separation for Sparse
   Regions.* arXiv:2508.14556.
5. Luo, Yu. *Music Source Separation with Band-Split RNN.* arXiv:2209.15174.
6. Défossez, Usunier, Bottou, Bach. *Music Source Separation in the Waveform Domain.*
   arXiv:1911.13254.
7. Rafii, Liutkus, Stöter, Mimilakis, Bittner. *MUSDB18.* Zenodo 1117372.
8. Jaffe, Burgoyne. *Musical Source Separation Bake-Off.* WASPAA 2025.
   arXiv:2507.06917.

## Appendix A — Reproducibility

Commands: MASTER_PLAN.md §7 run book. SLR anchors verified in review: do-nothing =
0.0 dB, 10 % energy leak = −10.0 dB (exact), perfect = ε floor, empty regions = NaN.
Uniform-arm bit-compatibility with the shared baseline cell: regression-tested against
the legacy formula independently of the new sampler code; shared-cell hash
`a97d5400e994`. Selection blindness to SLR: test-asserted. Suite at scaffold time:
358 passed, 1 skipped. GPU spend: ⟪actual⟫ vs ≈11–15 T4-h ceiling. Deviations:
`results/DEVIATIONS.md` (10 build-time clarifications; substantive: non-uniform
policies draw on the 1-s profile grid).

## Appendix B — Per-track SLR distributions, θ grids, exposure curves ⟪generated⟫

## Appendix C — Pre-registered alternative readings (unselected branches) ⟪moved here⟫
