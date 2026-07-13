# Direction 03 — A mini band-split front-end: does the band-partition idea survive at 5 M params?

Deep-dives: [`papers/luo2022-bsrnn.md`](papers/luo2022-bsrnn.md) (CORE),
[`papers/hung2025-moises-light.md`](papers/hung2025-moises-light.md) (CORE),
[`papers/lu2023-bs-roformer.md`](papers/lu2023-bs-roformer.md),
[`papers/wang2023-mel-roformer.md`](papers/wang2023-mel-roformer.md),
[`papers/scnet2024.md`](papers/scnet2024.md),
[`papers/chen2023-dttnet.md`](papers/chen2023-dttnet.md).

## 1. Direction recap (from RESEARCH_DIRECTIONS #3)
- **Hypothesis:** replacing SingNet's uniform 2-D conv encoder with a **band-split front-end** (separate low/mid/high sub-encoders on **mel-spaced** bands, merged in the bottleneck) **improves vocals SI-SDR at equal parameter count**, because vocal energy is concentrated and bands get dedicated capacity.
- **Minimal experiment:** **3 models at matched ~5–10 M params** — (i) baseline U-Net, (ii) **3-band mel-split**, (iii) **3-band uniform-split** (isolates "splitting" from "mel spacing"); **3 seeds each**.
- **Falsification / success test:** supported iff **mel-split > uniform-split > baseline** outside the seed noise band; a **per-frequency-band error breakdown** explains *why*. A within-noise null (band scheme washes out at tiny scale) is plausible and **reportable**.
- Difficulty/Risk: **Medium/Medium** (null result plausible — still reportable, less flashy).

## 2. Paper map
| Paper | Role | Specific claim we rely on | Verified | Deep-dive |
|---|---|---|---|---|
| BSRNN (2209.15174) | **replicate (idea)** | band-split + dedicated per-band capacity beats uniform front-end | CONFIRMED (10.01 dB via DTTNet) | [papers/luo2022-bsrnn.md](papers/luo2022-bsrnn.md) |
| Mel-RoFormer (2310.01809) | **replicate (the twist)** | **mel/overlapped bands beat heuristic bands** on vocals/drums/other, same backbone | CONFIRMED | [papers/wang2023-mel-roformer.md](papers/wang2023-mel-roformer.md) |
| Moises-Light (2510.06785) | **precedent** | band-split U-Net competitive at 13× fewer params than BS-RoFormer / ½ SCNet | CONFIRMED | [papers/hung2025-moises-light.md](papers/hung2025-moises-light.md) |
| SCNet (2401.13276) | **context** | unequal band compression (more capacity to info-dense bands) | CONFIRMED | [papers/scnet2024.md](papers/scnet2024.md) |
| DTTNet (2309.08684) | **context** | lightweight ≠ weak (10.12 dB vocals, 86.7% fewer params than BSRNN) | CONFIRMED | [papers/chen2023-dttnet.md](papers/chen2023-dttnet.md) |
| BS-RoFormer (2309.02612) | **context (ceiling)** | band-split+transformer SOTA (9.80 dB no extra data) | CONFIRMED | [papers/lu2023-bs-roformer.md](papers/lu2023-bs-roformer.md) |
| BSRNN replication (2603.09187) | **context (cautionary)** | replicating BSRNN's published numbers is hard even at full scale; optimized variant + public code released | CONFIRMED (WS, orchestrator re-check) | §4 / §6 |
| Generalized Bandsplit (2309.02539) | **context** | common-encoder BSRNN generalization; SNR + 1-norm loss; overcomplete/psychoacoustic bands | CONFIRMED (HF; shared) | [shared cross-ref](../../00-shared-research/papers/kong2021-cirm-resunet.md) (band analysis) |
| cIRM ResUNet (2109.05418) | **infrastructure** | complex mask + per-TF-bin analysis; oracle IRM ceiling | CONFIRMED | [shared](../../00-shared-research/papers/kong2021-cirm-resunet.md) |

## 3. Replicate-vs-extend
- **Replicate:** the band-split front-end (BSRNN) and the mel-vs-heuristic band comparison (Mel-RoFormer), at small scale.
- **Extend (our delta):** a **param-matched, seeded, 3-way controlled ablation** (baseline / uniform-split / mel-split) at **5–10 M params on MUSDB-only** — the one comparison the SOTA papers *don't* run (they change many things at once and use extra data). We isolate two effects: *does splitting help?* (uniform-split vs baseline) and *does mel spacing help beyond splitting?* (mel-split vs uniform-split), with a per-band error breakdown for mechanism.

## 4. The gap
- Every band-split result in the literature is at **large scale, often with extra data, and confounds** band scheme with architecture, capacity, and training recipe. **No published work isolates the band-partition effect in a param-matched compact model on MUSDB-only.** Moises-Light comes closest (efficient band-split) but reports an *efficiency frontier*, not a controlled split-vs-uniform-vs-baseline ablation. Mel-RoFormer isolates band scheme but only at full transformer scale.
- Consequently a **clean null at 5–10 M params is a genuine, novel finding** ("the band-split advantage is a large-model phenomenon"), and a positive is a strong one.
- **Reproducibility warning (Mar 2026):** *"The Costs of Reproducibility in Music Separation Research: a Replication of Band-Split RNN"* (Magron, Douwes & Serizel, arXiv **2603.09187**) reports that an experienced team could **not** fully reproduce BSRNN's published numbers from the paper alone, and releases an optimized BSRNN variant with public code. Two implications: (i) it independently validates this direction's modest, controlled, param-matched framing over chasing headline numbers; (ii) their public code is a reference implementation for band-split details (band edges, per-band normalization) worth consulting during implementation.

## 5. Direction-specific technical notes (param-matching methodology — for MASTER_PLAN)
The whole experiment stands or falls on **honest parameter matching**. Method:

1. **Fix a budget** $P$ (e.g. 8.0 M) and a **target tolerance** (say ±2%); every config's param count is **derived by hand in THEORY.md and verified programmatically** (PLAN Phase 3 rule).
2. **Baseline:** uniform 2-D conv U-Net; set base channels $c_0\in\{16,32,64\}$ and depth so params $\approx P$.
3. **Uniform-split (3 equal bands):** partition the $F$ STFT bins into 3 **equal-width** bands $[0,F/3),[F/3,2F/3),[2F/3,F)$; three parallel sub-encoders (channel-grouped, as in Moises-Light's split modules) feed a shared bottleneck + shared decoder. Because 3 sub-encoders each see $F/3$ bins, per-band channel width is tuned so the **sum** of sub-encoder + bottleneck + decoder params $\approx P$.
4. **Mel-split (3 mel-spaced bands):** identical structure, but band edges are **mel-spaced** (more bins in low/mid Hz where vocals concentrate), e.g. edges at mel-uniform points → roughly $[0,\sim500),[\sim500,\sim2000),[\sim2000,f_{\max}]$ Hz. Per-band bin counts differ, so per-band channel widths are re-tuned to hit $P$ again.
5. **Matching discipline:** keep the **bottleneck and decoder identical** across (3)/(4); only the **front-end band split (edges + per-band widths)** differs. Baseline (2) matches $P$ with a uniform encoder. Report a param-count table (per-module) for all three so a reader sees the match is exact, not hand-waved.
6. **Per-frequency-band error breakdown (mechanism figure):** after training, compute vocals SI-SDR (or magnitude error) **restricted to each frequency band** on the test set; the hypothesis predicts mel-split's advantage concentrates in the **vocal-dominant mid bands**. This is the "why" figure.

**Band-design priors from the literature:** few, **unequal** bands (SCNet/Moises-Light use 3: low/mid/high), **heavier encoder than decoder** (Moises-Light), and **mel/overlapped** spacing beating heuristic (Mel-RoFormer). BSRNN's non-uniform hand-designed bands are the "expert-knowledge" alternative if 3 equal/mel bands underperform.

## 6. Risks this literature implies
- **Null at small scale is likely** — the band-split gains are demonstrated with transformers + extra data; at 5–10 M params the effect may be within seed noise. Pre-register that a clean null is the finding ("band-split is a large-model phenomenon"), not a failure.
- **Param-matching is the credibility crux** — if the three configs aren't truly matched, any SI-SDR gap is a capacity artifact. Verify counts programmatically and publish the per-module table.
- **Implementation/test burden** — a new band-split encoder module needs unit tests (round-trip, correct band routing/merge); budget for it (Medium difficulty). Band-split implementations are demonstrably error-prone — even a professional BSRNN replication fell short of published numbers (2603.09187) — so treat their public code as a cross-check.
- **Confound with mask/loss/STFT** — hold STFT settings, mask type, loss, optimizer, and the 14-track valid protocol fixed across all three; only the front-end changes.
