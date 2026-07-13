# Mel-Band RoFormer for Music Source Separation

Ju-Chiang Wang, Wei-Tsung Lu, Minz Won (ByteDance). arXiv **2310.01809** (3 Oct 2023). | Verified: HF paper_search (title/authors/date/abstract), 2026-07-13. **Context** for Direction 03 — the direct evidence that **the band partition itself matters**, which is Direction 03's central question.

> Not to be confused with **arXiv 2409.04702** ("Mel-RoFormer for Vocal Separation and Vocal Melody Transcription," Wang/Lu/Chen, Sep 2024) — a *later, different* paper. Direction 03 cites the **MSS** one (2310.01809).

## 1. Problem & context
BS-RoFormer's band-split is **empirically designed, non-overlapping, heuristic**. Mel-RoFormer replaces it with a **psychoacoustically motivated Mel-band scheme** and tests whether principled band spacing beats heuristic band spacing — everything else held fixed.

## 2. Method
- **Mel-band Projection front-end:** map STFT frequency bins into **overlapped subbands according to the mel scale** (contrast: BSRNN/BS-RoFormer use **non-overlapping**, heuristic bands). Overlap gives redundant, more reliable per-band features.
- Same **interleaved RoPE Transformers** modeling frequency and time as two sequences.

## 3. Key results (exact)
- On **MUSDB18-HQ**, **Mel-RoFormer outperforms BS-RoFormer on vocals, drums, and other stems**. CONFIRMED (abstract). (Relative, same-backbone comparison — the cleanest published "band scheme matters" result.)

## 4. Limitations & caveats
- Still a heavy RoFormer; the finding is about the **front-end band scheme**, transferable to small models in principle but demonstrated only at scale.
- "Outperforms on vocals/drums/other" is stem-wise; no isolated ablation of overlap vs mel-spacing separately.

## 5. Relevance to this project (Direction 03 — the load-bearing citation)
- **This is the paper that justifies Direction 03's hypothesis** that *mel-split > uniform-split*. If, at full scale, mel-spaced bands beat heuristic bands with the backbone fixed, the question "does mel-spacing help at 5–10 M params?" is well-posed and non-trivial.
- Direction 03's 3-way control (baseline U-Net / **mel-split** / **uniform-split**) is a direct miniaturization of the BS-RoFormer→Mel-RoFormer comparison: it isolates *splitting* (uniform-split vs baseline) from *mel spacing* (mel-split vs uniform-split).
- **Honest caveat for the report:** Mel-RoFormer's gain is at large scale with transformers; a *null* at 5–10 M params (band scheme washes out) is a plausible and reportable Direction-03 outcome.

## 6. Verification notes
- Title/authors/arXiv ID + claim (overlapped mel bands vs non-overlapping heuristic; beats BS-RoFormer on vocals/drums/other on MUSDB18-HQ): CONFIRMED verbatim from HF abstract. RESEARCH_DIRECTIONS "Wang et al." correct (Ju-Chiang Wang first author).
</content>
