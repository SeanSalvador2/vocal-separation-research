# What Are 100 Songs Worth? Factorizing Augmentation and Measuring Data Scaling for Compact Music Source Separation

**Draft scaffold — placeholders ⟪like this⟫ are filled only from committed result CSVs.
All interpretations below were written before any training (pre-registered 2026-07-13);
the applicable branch is selected once results exist, the rest move to Appendix C.**

---

## Abstract

Every modern music source separation system trains with the same folk recipe on the same
100-song dataset: cross-song source remixing, random per-source gain, random sign flip.
The recipe is copied verbatim between codebases — we verify the exact transforms and
constants in Open-Unmix and Demucs — yet no published study factorizes what each
transform contributes, nor pairs that with a data-scaling curve for a compact model.
We train a fixed 9.8 M-parameter magnitude-mask U-Net on MUSDB18 under a leave-one-out
augmentation design (full recipe / minus-remix / minus-gain / minus-flip / none) and on
nested subsets of {21, 43, 64, 86} songs at identical step budget, with seed-noise bands
from three-seed anchor cells and one held-out test pass. Pre-registered hypotheses:
(H-02a) remixing is the single dominant transform; (H-02b) the data-scaling curve has
not saturated at 86 songs. We find ⟪H-02a verdict + Δ_remix vs Δ_next in dB⟫ and
⟪H-02b verdict + slope b dB/doubling with CI⟫. The sign-flip arm — provably inert for a
magnitude-masking model — serves as a built-in negative control and measured
⟪Δ_flip⟫ dB, ⟪validating / invalidating⟫ the design. ⟪One sentence: the data-vs-
augmentation bottom line.⟫

## 1. Introduction

Small-data training is the norm in music source separation: the standard public corpus,
MUSDB18, offers 86 usable training songs, and the community's answer is a fixed
augmentation recipe inherited from Spleeter/Open-Unmix/Demucs practice. Two questions a
practitioner immediately asks are unanswered in the literature: **which part of the
recipe carries the value**, and **would more songs still help a small model, or is it
capacity-bound?** Both are cheap to answer properly at compact scale — with controls,
seeds, and a frozen protocol — and both matter beyond this project: the first calibrates
every augmentation decision downstream, the second decides whether effort goes to data
acquisition (or pseudo-labeling; cf. our Direction 10) or to architecture (Direction 03).

**Contributions.** (1) The first per-transform leave-one-out factorization of the
standard MSS augmentation recipe at compact scale, with seed-noise bands; (2) a
data-scaling curve on nested MUSDB subsets at fixed compute, with a pre-registered
"has it saturated?" decision rule; (3) a built-in negative control (sign flip, provably
inert under magnitude masking — derived in THEORY.md §3) that validates the
experimental machinery itself; (4) verified code-level provenance of the recipe
(exact constants from the Open-Unmix and Demucs training code), correcting loose
paraphrases in circulation (e.g. "±3–6 dB gain" — the code says amplitude U(0.25, 1.25)).

## 2. Related Work

**The recipe.** Demucs's `augment.py` implements Shift, FlipChannels, FlipSign,
Scale(0.25–1.25), and cross-track Remix (group size 4); Open-Unmix's `data.py`
implements the same gain range, channel swap, and `random_track_mix`. Papers state the
recipe's presence — Défossez et al. attribute Demucs's MUSDB-only viability to "proper
data augmentation" — but publish no per-transform decomposition.

**Data quantity.** Prétet et al. (arXiv 1906.02618) study training-data amount and type
for singing-voice separation — the closest prior; they do not factorize the augmentation
recipe or use a param-matched compact model with seed bands. Saijo & Bando (arXiv
2505.07631) show unlabeled-data pre-training helps modern MSS, implying data (not
architecture) binds at MUSDB scale — motivation, but not a measurement, of the scaling
slope we report. MixIT (Wisdom et al., arXiv 2006.12701) is the unsupervised antecedent.

**Scale caveat.** Our findings are claims about a ~10 M-parameter magnitude-mask U-Net
at a 16 k-step budget on MUSDB18 — the regime where small teams actually operate — and
are not extrapolations to SOTA band-split transformers.

## 3. Method

Fixed infrastructure (inherited from the project's Direction 01, normative spec in
MASTER_PLAN.md §1.3): SingNet-C1 U-Net (9,835,745 params), STFT 4096/1024, 6-s mono
chunks, `l1mag` loss, AdamW + warmup/cosine, AMP, batch 16, REDUCED = 16 k steps for
every run. Only two things ever vary: the **augmentation switchboard** and the
**track allowlist**.

**Factorization arms** (86 songs): full / no-remix / no-gain / no-flip / none. Without
remix, a training example is the true track mixture (same-track stems, same window),
so the contrast isolates the remix distribution shift with bit-identical everything
else. Each transform owns an independent RNG stream keyed (seed, transform, step), so
disabling one cannot reshuffle the others — unit-tested.

**Scaling arms** (full recipe): nested subsets 21 ⊂ 43 ⊂ 64 ⊂ 86 songs, drawn once with
a fixed seed, committed before training, never re-rolled; the validation split never
changes; remix partners come from the active subset (pool shrinkage is part of the
treatment, stated).

**Anchors:** the full-86 cell is shared with Direction 01's `l1mag` sweep cell
(config-hash equality asserted in code); three seeds there and at n = 21 give the pooled
noise band σ_seed.

## 4. Experimental Setup

- Runs: 4 (LOO) + 5 (scaling) new at REDUCED budget; 3-seed cells at FULL-86 (shared)
  and n21; single seed elsewhere. Optional pre-registered contingency (loss flip from
  Direction 01) and G2 escalation (+2 seeds on no-remix) — ⟪used / not used⟫.
- Decisions on the 14-track validation split; **one** test pass (50 tracks, FULL-86
  seeds only) anchors generalization. Metrics: vocals SI-SDR (primary; Le Roux 2019).
- **H-02a decision rule (verbatim, pre-registered):** supported iff
  Δ_remix > max(Δ_gain, Δ_flip) + σ_seed and Δ_remix > σ_seed.
- **H-02b decision rule:** supported iff mean(86) − val(43) > σ_seed with a
  non-decreasing 4-point sequence (±σ_seed tolerance); refuted (plateau) iff that last
  doubling gains ≤ σ_seed/2.
- Descriptive: total recipe value Δ_total, interaction gap Δ_total − ΣΔ_t, log₂ fit
  slope b (dB/doubling) with parametric-bootstrap 95 % CI, secant slopes per doubling.

## 5. Results

> Numbers generated by `singnet.analysis.scaling` from
> `results/registry.csv` / `results/test_per_track.csv`; figures regenerate from CSVs.

### 5.1 Factorization (leave-one-out)

| Config | val SI-SDR (dB) | Δ from FULL (dB) |
|---|---|---|
| full (3 seeds) | ⟪mean ± std⟫ | — |
| no-remix | ⟪⟫ | Δ_remix = ⟪⟫ |
| no-gain | ⟪⟫ | Δ_gain = ⟪⟫ |
| no-flip (negative control) | ⟪⟫ | Δ_flip = ⟪⟫ |
| none | ⟪⟫ | Δ_total = ⟪⟫ |

σ_seed (pooled) = ⟪⟫ dB. Interaction gap Δ_total − ΣΔ_t = ⟪⟫ dB.
**Negative control:** |Δ_flip| = ⟪⟫ ⟪within / OUTSIDE⟫ σ_seed → design
⟪validated / halted, see DEVIATIONS⟫. **H-02a: ⟪supported / refuted / mixed⟫.**
⟪Figure: Δ bar chart with σ_seed band.⟫

### 5.2 Data scaling

| N songs | val SI-SDR (dB) |
|---|---|
| 21 (3 seeds) | ⟪mean ± std⟫ |
| 43 | ⟪⟫ |
| 64 | ⟪⟫ |
| 86 (3 seeds) | ⟪mean ± std⟫ |

Last-doubling gain (43→86): ⟪⟫ dB vs σ_seed = ⟪⟫ → **H-02b: ⟪supported / refuted /
mixed⟫.** Log₂ fit: b = ⟪⟫ dB/doubling, 95 % CI ⟪⟫; secants 21→43: ⟪⟫, 43→86: ⟪⟫
(curvature: ⟪bending / straight⟫). ⟪Figure: scaling curve, endpoint bands, fit line
over [21, 86] only.⟫

### 5.3 Generalization anchor (single test pass)

FULL-86, 3 seeds, 50 test tracks: ⟪mean ± seed-std⟫ dB (val→test gap ⟪⟫ dB).
⟪Table: per-seed; per-track CSV committed.⟫

## 6. Discussion — pre-registered interpretation branches

**⟪SELECT applicable branches; unselected move to Appendix C.⟫**

### 6.1 H-02a — what carries the recipe

**Branch S (remix dominates).** The folk belief is now a measurement: Δ_remix = ⟪⟫ dB
of the recipe's Δ_total = ⟪⟫ dB — the mixture-manifold enrichment (combinatorial
cross-song mixtures) is what small-data MSS training actually buys, while per-source
gain contributes ⟪⟫ and flip, as theory demands, ⟪≈0⟫. *Consequences:* remix is
non-negotiable in every later direction; Direction 10's premise (a teacher can
manufacture *more* mixtures) gains a measured anchor; augmentation-tuning effort
elsewhere is deprioritized.

**Branch R-order (remix not on top).** Gain ⟪or flip — flag control first⟫ matched or
beat remix (Δ_remix = ⟪⟫ ≤ Δ_⟪⟫ = ⟪⟫). Two candidate mechanisms we pre-commit to
distinguishing: (i) at 16 k steps the model never exploits the enlarged mixture
manifold (budget-bound) — testable by the FULL-budget contingency rerun ⟪run? result⟫;
(ii) the per-chunk standardization already provides the invariance remix adds at this
scale. Either way the community default is not self-evident at compact scale — worth
stating loudly.

**Branch R-null (nothing matters: Δ_total ≤ σ_seed).** Augmentation as a whole moved
nothing detectable at this budget. Pre-registered follow-up executed: full-vs-none at
FULL budget → ⟪result⟫. If the null persists, the honest conclusion is that at ~10 M
params / MUSDB scale, the binding constraint is not training-distribution richness —
connect to H-02b's verdict for the data-vs-capacity story.

**Branch M (remix on top, within band).** Ranked as expected but unresolvable at n = 1
seed for middle arms; G2 escalation ⟪ran: result / not triggered⟫. Report point
estimates with the band; no strong claim.

### 6.2 H-02b — is 86 songs enough?

**Branch S (still rising).** The last doubling paid ⟪⟫ dB (> σ_seed): a compact model
remains **data-starved at the full MUSDB training set** — b = ⟪⟫ dB/doubling means a
hypothetical 172-song corpus would be worth ≈ ⟪b⟫ dB more *if the trend held* (stated
as descriptive extrapolation only). *Consequences:* data-side levers (Direction 10's
pseudo-labels; future extra corpora) are rationally prioritized over architecture
tweaks at this scale; Direction 03's null-result prior strengthens.

**Branch R (plateau).** The curve flattened (43→86 ≤ σ_seed/2): at this
capacity/budget, more songs stop helping before the corpus runs out — the model, not
MUSDB, is the constraint. *Consequences:* re-weights the project toward capacity/
architecture directions (03) and away from data-acquisition narratives; Direction 10's
"more data via teacher" premise weakens and its plan's honest-negative branch becomes
the expected outcome (we say so *now*, before Direction 10 runs).

**Branch M (borderline / non-monotone).** ⟪Which point misbehaves⟫; per-subset balance
table ⟪shows / doesn't show⟫ a compositional cause. The curve is reported as
descriptive; the one-draw limitation (pre-registered) is the leading explanation, and a
multi-draw replication is named as the concrete follow-up.

### 6.3 Cross-cutting readings

- **Interaction gap** ⟪≈0: transforms near-additive / >0: synergy / <0: redundancy⟫ —
  ⟪one-paragraph reading tied to THEORY §5⟫.
- **Flip control fired** (only if applicable): interpretation halts; the fault analysis
  in DEVIATIONS.md ⟪summary⟫ takes precedence over any substantive claim above.
- **Val→test gap** ⟪⟫ dB on the anchor cell: ⟪ordinary / concerning⟫ for the
  validation-scoped claims of this study.

### 6.4 Threats to validity (written before results)

One nested subset draw (draw-luck bounded by honesty, not by replication); single-seed
middle cells; REDUCED budget only (mitigated by the pre-registered FULL-budget
contingency for the null case); mono pipeline (channel-swap untestable — stated, not
generalized about); remix pool shrinkage is part of the "less data" treatment by
design; MUSDB's AAC additivity error (~1e-3) is uniform across arms; findings are
regime-specific (compact model, 16 k steps, MUSDB18).

## 7. Conclusion

⟪Two paragraphs from the selected branches: (1) the measured anatomy of the standard
recipe + the scaling verdict, with numbers; (2) what a practitioner with 100 songs and
a small model should actually do — the data-vs-augmentation-vs-capacity triage this
study exists to inform, and what it changes for Directions 03 and 10.⟫

## References

1. Défossez, Usunier, Bottou, Bach. *Music Source Separation in the Waveform Domain.*
   arXiv:1911.13254 (+ `facebookresearch/demucs` `augment.py`).
2. Stöter, Uhlich, Liutkus, Mitsufuji. *Open-Unmix.* JOSS 2019. 10.21105/joss.01667
   (+ `sigsep/open-unmix-pytorch` `data.py`).
3. Prétet, Hennequin, Royo-Letelier, Vaglio. *Singing Voice Separation: A Study on
   Training Data.* ICASSP 2019. arXiv:1906.02618.
4. Saijo, Bando. *(MixIT pre-training for MSS.)* arXiv:2505.07631.
5. Wisdom, Tzinis, Erdogan, Weiss, Wilson, Hershey. *Unsupervised Sound Separation
   Using Mixture Invariant Training.* NeurIPS 2020. arXiv:2006.12701.
6. Le Roux, Wisdom, Erdogan, Hershey. *SDR — Half-Baked or Well Done?* ICASSP 2019.
   arXiv:1811.02508.
7. Rafii, Liutkus, Stöter, Mimilakis, Bittner. *MUSDB18.* Zenodo 1117372.
8. Hennequin, Khlif, Voituret, Moussallam. *Spleeter.* JOSS 2020. 10.21105/joss.02154.

## Appendix A — Reproducibility

Commands: MASTER_PLAN.md §7 run book. Environment: repo-root `requirements.txt`
(pinned). Seeds {0,1,2}; subset seed 0; registry keyed by config hash (the FULL-86
cell's hash-identity with Direction 01's `l1mag` cell is test-asserted:
`tests/test_config_schema.py`). Test suite at scaffold time: 108 passed, 1 skipped.
GPU spend: ⟪actual⟫ vs the ≈13–25 T4-h pre-registered ceiling. Deviations:
`results/DEVIATIONS.md` (currently: the six build-time clarifications logged there,
including the per-transform RNG streams and the "no-remix = true track mixture"
contrast definition).

## Appendix B — Per-track and per-seed tables ⟪generated⟫

## Appendix C — Pre-registered alternative readings (unselected branches) ⟪moved here⟫
