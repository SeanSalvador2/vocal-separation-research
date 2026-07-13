# SCNet: Sparse Compression Network for Music Source Separation

Weinan Tong, Jiaxu Zhu, Jun Chen, Shiyin Kang, Tao Jiang, Yang Li, Zhiyong Wu, Helen Meng. ICASSP 2024 (pp. 1276–1280) / arXiv **2401.13276** (24 Jan 2024). | Verified: HF paper_search (abstract) + WebSearch (ICASSP 2024 venue), 2026-07-13. **Context** for Direction 03 — the "unequal band compression" idea.

## 1. Problem & context
Good MSS at low complexity is hard in super-wide-band audio; prior work either ignores subband differences or loses information making subband features. SCNet splits the spectrogram and **compresses low-information bands more aggressively**, spending capacity where it matters.

## 2. Method
- Split the mixture spectrogram into **several subbands**; a **sparsity-based encoder** models each band with a **higher compression ratio on low-information bands** (typically high frequencies), concentrating modeling on information-dense bands (typically low/mid, where vocals/bass live).
- Frequency-domain network; encoder + decoder + dual-path bottleneck (the family Moises-Light later builds on).

## 3. Key results (exact, with convention)
- **9.0 dB SDR on MUSDB18-HQ, no extra data** — outperforms then-SOTA at lower complexity. CONFIRMED (abstract).
- **CPU inference time = 48% of HT-Demucs.** CONFIRMED (abstract). (Matches RESEARCH_NOTES/RESEARCH_DIRECTIONS "~48%.")

## 4. Limitations & caveats
- Still a full-scale model; the "unequal compression" principle is the transferable idea, not the exact net.

## 5. Relevance to this project (Direction 03)
- **The design principle Direction 03 can borrow cheaply:** give bands *unequal* capacity/compression rather than a uniform conv encoder — vocal-relevant bands get more, high bands get less. This is the mechanism behind the hypothesis that a band-split front-end helps at fixed parameter count (bands get *dedicated, unequal* capacity).
- Reinforces the "few, unequal subbands (low/mid/high)" choice adopted from Moises-Light for the compact band modules.
- Not reproduced; cited as efficiency-frontier evidence and the 48%-CPU-cost data point.

## 6. Verification notes
- Title/authors/arXiv ID + numbers (9.0 dB MUSDB18-HQ no extra data; 48% of HT-Demucs CPU): CONFIRMED verbatim from HF abstract. **ICASSP 2024 venue CONFIRMED** (WebSearch: cmsworkshops ICASSP2024, pp. 1276–1280).
</content>
