# Music Source Separation with Band-Split RoPE Transformer (BS-RoFormer)

Wei-Tsung Lu, Ju-Chiang Wang, Qiuqiang Kong, Yun-Ning Hung (ByteDance). arXiv **2309.02612** (5 Sep 2023). | Verified: HF paper_search (title/authors/date/abstract), 2026-07-13. **Context** for Direction 03 (the band-split + transformer ceiling; the current SOTA family).

## 1. Problem & context
Takes BSRNN's band-split front-end and replaces the RNN sequence modeling with **hierarchical Transformers**, asking whether attention over bands/time raises the ceiling. It did — this is the SDX'23-winning family.

## 2. Method
- **Band-split module** projects the input **complex spectrogram** into **subband-level representations** (BSRNN-style, non-overlapping, heuristic bandwidths).
- A stack of **hierarchical Transformers** models **inner-band** and **inter-band** sequences for multi-band mask estimation (the band/sequence duality of BSRNN, now with self-attention).
- **Rotary Position Embedding (RoPE)** is the key training enabler for the transformer over these sequences.

## 3. Key results (exact, with convention)
- Trained on **MUSDB18-HQ + 500 extra songs → ranked 1st in the MSS track of SDX'23**. CONFIRMED (abstract).
- A **smaller BS-RoFormer on MUSDB18-HQ, no extra data → 9.80 dB average SDR** (SOTA without extra data). CONFIRMED (abstract). Metric is average SDR over stems on MUSDB18-HQ.

## 4. Limitations & caveats
- Transformer + band-split is **heavy** and the winning entry used 500 extra songs; out of scope to train from scratch. Cited as the field ceiling and the reference point for Moises-Light's "13× fewer parameters."
- Band scheme is **empirical/heuristic** ("defined without analytic support") — the explicit motivation for Mel-RoFormer.

## 5. Relevance to this project (Direction 03)
- The **ceiling** the band-split idea reaches at full scale, and the **parameter reference** for the efficiency story (Moises-Light is 13× smaller than this).
- Its admission that the band partition is heuristic is exactly why Direction 03's **mel-split vs uniform-split control** is a real question.
- Not reproduced; cited only.

## 6. Verification notes
- Title/authors/arXiv ID + all numbers (SDX'23 1st with 500 extra; 9.80 dB avg SDR no extra data; RoPE; hierarchical transformers; band-split of complex spectrogram): CONFIRMED verbatim from HF abstract. RESEARCH_DIRECTIONS "Lu et al." correct (Wei-Tsung Lu first author).
