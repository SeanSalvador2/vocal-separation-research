# Hybrid Demucs & Hybrid Transformer Demucs (the `htdemucs` family)

Two papers, one lineage — the family StemCraft already ships (`htdemucs_6s`) and the **teacher for Direction 10**.
- **Hybrid Demucs (v3):** Alexandre Défossez, *"Hybrid Spectrogram and Waveform Source Separation,"* MDX 2021 workshop / arXiv **2111.03600**. | Verified: HF paper_search (abstract), 2026-07-13.
- **Hybrid Transformer Demucs (HT-Demucs):** Simon Rouard, Francisco Massa, Alexandre Défossez, ICASSP 2023 / arXiv **2211.08553**. | Verified: HF paper_search (abstract), 2026-07-13.

## 1. Problem & context
Demucs v1 was pure-waveform. Hybrid Demucs adds a **parallel spectrogram branch**, letting the model choose the domain per source. HT-Demucs replaces the innermost layers with a **cross-domain Transformer**, asking whether long-range context helps — with a blunt honest answer about data.

## 2. Method
- **Hybrid Demucs:** a **temporal/spectral bi-U-Net** — a waveform U-Net branch and a spectrogram U-Net branch that merge in a shared bottleneck, so representations can cross domains. Adds compressed residual branches, local attention, singular-value regularization.
- **HT-Demucs:** same bi-U-Net skeleton, but the innermost encoder/decoder layers → a **cross-domain Transformer Encoder** using **self-attention within a domain and cross-attention across domains** (spectral↔temporal). Sparse-attention kernels extend the receptive field; per-source fine-tuning for the SOTA entry.

## 3. Key results (exact, with convention — all museval SDR on MUSDB-HQ)
- **Hybrid Demucs:** **won the Sony Music Demixing Challenge (MDX) 2021**; **+1.4 dB SDR across all sources** on MUSDB-HQ, plus subjective wins (overall quality 2.83/5 vs 2.36 for non-hybrid Demucs; contamination-absence 3.04 vs 2.37). CONFIRMED (abstract). *Note:* the +1.4 dB baseline in the abstract is **vs the non-hybrid (v2) Demucs**; RESEARCH_NOTES/RESEARCH_DIRECTIONS phrase it "over prior/previous SOTA," which is defensible (it won MDX'21) but the abstract's explicit contrast is non-hybrid Demucs — noted, not edited.
- **HT-Demucs:** the honest quote (verbatim): *"While it performs poorly when trained only on MUSDB,"* it beats Hybrid Demucs (same data) **by 0.45 dB** with **800 extra songs**; with extra data + sparse attention + per-source fine-tuning it reaches **state-of-the-art 9.20 dB avg SDR**. CONFIRMED (abstract). The "performs poorly when trained only on MUSDB" phrase — quoted throughout our docs — is exact.

## 4. Limitations & caveats
- Transformer capacity is **data-bound**: HT-Demucs needs hundreds of extra songs to pay off — the central reason PLAN rejects "band-split/transformer from scratch" for the headline and why these are cited as *ceilings*, not reproduction targets.
- 9.20 dB is museval SDR **with 800 extra songs** — never comparable to StemCraft's +9.55 dB window SI-SDR for `htdemucs_6s` (different estimator, different protocol).

## 5. Relevance to this project
- **Anchor/ceiling (PLAN §2.2):** the field ceiling this project explicitly does not chase from scratch.
- **Direction 10 (CORE — teacher):** `htdemucs`/`htdemucs_6s` (already in StemCraft, +9.55 dB window SI-SDR measured) is the **distillation teacher**. Its "performs poorly on MUSDB-only / needs extra data" property is *why* distillation on unlabeled FMA is plausible: the teacher already encodes what 800+ songs taught it, and distillation transfers that to SingNet without new labels. **Teacher-error risk:** HT-Demucs "suffers from some bleeding, especially vocals↔other" (inherited from v1) — distillation will transfer those errors (Direction 10's honest failure mode).
- **H4 / oracle discussion:** Demucs beating the IRM oracle on bass (v1) is the concrete demonstration that waveform models escape the magnitude-mask ceiling — the framing for THEORY.md §2.

## 6. Verification notes
- Both titles/authors/venues/arXiv IDs and all quoted numbers (MDX'21 win, +1.4 dB, 2.83/2.36, "performs poorly when trained only on MUSDB," 0.45 dB / 800 songs, 9.20 dB): CONFIRMED from HF paper_search abstracts.
- `htdemucs_6s` as StemCraft's shipped engine and its +9.55 dB window SI-SDR: from StemCraft's own `REAL_RESULTS.md` (project-internal, not re-verified here).
