# Direction 10 — Demucs as teacher: pseudo-label distillation to close the small-model gap

Deep-dives: [`papers/hinton2015-kd.md`](papers/hinton2015-kd.md),
[`papers/fma-dataset.md`](papers/fma-dataset.md) (CORE — licensing),
[`papers/solovyev2023-mvsep.md`](papers/solovyev2023-mvsep.md). Teacher: shared Demucs-family note.

## 1. Direction recap (from RESEARCH_DIRECTIONS #10)
- **Hypothesis:** distilling **HT-Demucs (teacher)** outputs on **~100–200 unlabeled, license-safe FMA tracks** as **soft targets** for SingNet **closes ≥ 25% of the SingNet→Demucs SI-SDR gap** on MUSDB test, at **zero extra labeled data**.
- **Minimal experiment:** teacher-label an FMA subset once (GPU-cheap); train SingNet on **(a) MUSDB only, (b) MUSDB + distilled, (c) distilled only**; 3 seeds on (a)/(b); **evaluate on MUSDB test (never distilled)**.
- **Falsification / success test:** **gap-closure % with CIs**. Supported iff (b) closes ≥25% of the (a)→Demucs gap, CI-separated from (a). The honest negative — "**distillation transferred the teacher's errors / domain mismatch muted gains**" — is a valid, reportable outcome.
- Difficulty/Risk: **High / Medium-High** (most moving parts: new dataset, teacher pipeline, domain shift).

## 2. Paper map
| Paper | Role | Specific claim we rely on | Verified | Deep-dive |
|---|---|---|---|---|
| Hinton et al. 2015 (1503.02531) | **method** | soft targets transfer "dark knowledge"; $\lambda$-mix soft/hard | CONFIRMED | [papers/hinton2015-kd.md](papers/hinton2015-kd.md) |
| FMA (1612.01840) + `mdeff/fma` | **infrastructure (CORE)** | per-track CC-licensed audio; research-use; filterable subsets | CONFIRMED (README) | [papers/fma-dataset.md](papers/fma-dataset.md) |
| HT-Demucs / Hybrid Demucs | **teacher** | strong MUSDB-HQ+extra-data teacher; also its bleeding errors | CONFIRMED | [shared](../../00-shared-research/papers/demucs-hybrid-family.md) |
| BSRNN (2209.15174) | **precedent (semi-sup)** | pseudo-label fine-tuning on unlabeled songs via activity detector | CONFIRMED | [Dir 03](../../03-mini-band-split/research/papers/luo2022-bsrnn.md) |
| MixIT-for-MSS (2505.07631) | **precedent (unlabeled)** | unlabeled FMA helps MSS (self-supervised) | CONFIRMED | [Dir 02](../../02-augmentation-data-scaling/research/papers/saijo2025-mixit-mss.md) |
| MVSep (2305.07489) | **context** | teacher-selection / ensembling benchmark | CONFIRMED | [papers/solovyev2023-mvsep.md](papers/solovyev2023-mvsep.md) |

## 3. Replicate-vs-extend
- **Replicate:** the semi-supervised-on-unlabeled-music idea (BSRNN pseudo-labels; MixIT-for-MSS) and knowledge distillation (Hinton).
- **Extend (our delta):** **cross-architecture teacher→student distillation** — a big *hybrid-waveform* teacher (`htdemucs`) → a compact *spectrogram-mask* student (SingNet) — on **license-safe unlabeled FMA**, measuring **gap-closure on held-out MUSDB test**. BSRNN self-distills (same model); MixIT is self-supervised; **teacher-student KD across architectures for MSS with a quantified gap-closure target is the specific, less-charted combination.**

## 4. The gap
- Semi-supervision for MSS exists (BSRNN pseudo-labels, MixIT-for-MSS) but is **self-supervised / self-distillation**. **Cross-architecture KD (large hybrid teacher → small spectrogram student) on CC-unlabeled data, with a gap-closure metric on MUSDB test, is not a standard published result** (representation-distillation like 2506.07237 is a different task). Novelty = the specific teacher/student pairing + the honest gap-closure measurement, not "distillation" in the abstract.

## 5. Direction-specific technical notes (distillation objectives + temperature analogue — for MASTER_PLAN)
**Regression, not classification** (`hinton2015-kd.md §4`): the teacher's separated output *is* the soft target; there is no softmax to temperature-soften. Candidate objectives, teacher $=$ `htdemucs`, output $\hat s^{T}=\text{teacher}(x)$ (vocal), student $\hat s^{S}$, on unlabeled $x\in$ FMA:

| Objective | Loss | Notes |
|---|---|---|
| **Waveform L1 KD** | $\lVert \hat s^{S}-\hat s^{T}\rVert_1$ | matches Demucs's own loss; teacher output in time domain (needs student iSTFT) |
| **Spectrogram (mag) KD** | $\big\lVert\,|\hat S^{S}|-|\hat S^{T}|\,\big\rVert_1$ | matches SingNet's magnitude-domain head; cheapest |
| **Mask-space KD** | $\lVert M^{S}-M^{T}\rVert$, $M^{T}=\text{clamp}(|\hat S^{T}|/|X|)$ | teacher mask derived post-hoc (Demucs is waveform-native); clamp for $>1$ |
| **Combined (Hinton $\lambda$)** | $(1{-}\lambda)\mathcal L_{\text{GT}}^{\text{MUSDB}} + \lambda\,\mathcal L_{\text{KD}}^{\text{FMA}}$ | GT loss on labeled MUSDB, KD loss on unlabeled FMA |

**Temperature analogue (OUR framing, marked as analogy):** no softmax temperature; the closest cousins are (i) the continuous teacher output itself (softer than absent GT), (ii) **spectral dynamic-range compression** (log or power-law on magnitudes) that de-emphasizes loud peaks and surfaces low-energy structure — a loose "softening" — and (iii) the $\lambda$ soft/hard mix. Recommend **Spectrogram-mag KD** as the primary objective (matches the student's head, cheapest, no iSTFT in the loss), with waveform-L1 KD as a secondary.

**Practical pipeline:** filter `fma_small` metadata → permissive-license ~100–200 tracks → teacher-label once with `htdemucs` (cache soft targets) → train (a)/(b)/(c) → evaluate on MUSDB test only.

## 6. Risks this literature implies
- **Teacher-error transfer (central):** `htdemucs` "suffers from bleeding, especially vocals↔other" (shared Demucs note) — distillation copies these errors; the student can inherit the teacher's leakage. This is the mechanism behind the honest-negative outcome.
- **Domain shift (central):** FMA production/genre ≠ MUSDB; the teacher may separate FMA worse, producing noisy soft targets that don't transfer to MUSDB test — muting or reversing gains. Report teacher quality-on-FMA caveats; (c) "distilled only" isolates this.
- **License discipline:** FMA audio licenses vary per track (`fma-dataset.md §3`) — filter to permissive licenses; commit no audio, only soft targets/weights.
- **Scope creep (High difficulty):** new dataset + teacher inference pipeline + FMA hygiene is the heaviest option; budget accordingly (the base plan flags this).
- **KD-is-classification caveat:** the "temperature" story is an analogy, not Hinton's mechanism — frame it honestly in THEORY/REPORT; the real transfer is the teacher's continuous output as a target.
