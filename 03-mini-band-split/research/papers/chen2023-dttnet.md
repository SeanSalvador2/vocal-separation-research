# DTTNet: Dual-Path TFC-TDF UNet — a lightweight MSS framework

Junyu Chen, Susmitha Vekkot, Pancham Shukla. ICASSP 2024 / arXiv **2309.08684** (Sep 2023; rev. 19 Mar 2024). | Verified: HF paper_search (title/authors/date/abstract), 2026-07-13. **Context** for Direction 03 — "lightweight ≠ weak," and the direct base of Moises-Light.

## 1. Problem & context
MSS trends toward ever-larger models. DTTNet asks whether a **carefully designed lightweight** architecture can match a heavyweight (BSRNN) — the thesis that most matters for a notebook-budget project.

## 2. Method
- **TFC-TDF U-Net** (Time-Frequency Convolution + Time-Distributed Fully-connected UNet) with a **Dual-Path module** for sequence modeling — a symmetric encoder/decoder U-Net with dual-path (frequency/time) modeling in the bottleneck. This is the **base architecture Moises-Light extends** (`hung2025-moises-light.md`).

## 3. Key results (exact, with convention)
- **10.12 dB cSDR on vocals**, vs **10.01 dB reported for BSRNN**, with **86.7% fewer parameters** than BSRNN. CONFIRMED verbatim (abstract). Metric is **cSDR** (chunk SDR, MDX convention).
- **This abstract is the source of the "BSRNN vocals ≈ 10.01 dB" number** used to ground `luo2022-bsrnn.md` §4.

## 4. Limitations & caveats
- cSDR on MUSDB18-HQ; not comparable to SI-SDR/museval-median. "Lightweight" is relative (still millions of params, MUSDB-HQ scale) — Direction 03's ~5–10 M compact target is smaller still.

## 5. Relevance to this project (Direction 03)
- The **existence proof** that a small, well-designed model can reach big-model quality — the optimistic prior behind Direction 03 (and behind the whole "compact from-scratch" project posture).
- As Moises-Light's base, it defines the design family (TFC-TDF + dual-path) whose *band-split* variant Direction 03 miniaturizes.
- Supplies the **BSRNN 10.01 dB vocals cSDR** cross-reference (BSRNN's own abstract omits the single number).
- Not reproduced; cited as the lightweight-competitive anchor.

## 6. Verification notes
- Title/authors/arXiv ID + numbers (10.12 dB vocals cSDR; 10.01 dB BSRNN; 86.7% fewer params): CONFIRMED verbatim from HF abstract.
