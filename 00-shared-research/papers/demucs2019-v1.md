# Music Source Separation in the Waveform Domain (Demucs v1)

Alexandre Défossez, Nicolas Usunier, Léon Bottou, Francis Bach. ISMIR 2019 / arXiv **1911.13254** (an earlier version is arXiv **1909.01174**, *"Demucs: Deep Extractor for Music Sources with extra unlabeled data remixed"*). | Verified: HF paper_search (title/authors/date/abstract) + WebSearch (loss/augmentation, second arXiv ID), 2026-07-13. Paper *body* not directly readable (arXiv/ar5iv/honu.io all blocked or 403) — body-level claims are marked accordingly.

The waveform-domain reference model, the **source of the standard augmentation recipe**, and the paper RESEARCH_DIRECTIONS cites for "L1 > SI-SNR." That last attribution needs the care documented in §2.

## 1. Problem & context
State-of-the-art MSS in 2019 was magnitude masking (Spleeter/UMX). Demucs asks whether a **waveform-to-waveform** model can beat it. It adapts Conv-TasNet to music (beats spectrogram methods but with audible artifacts) and proposes **Demucs**: a time-domain **U-Net encoder/decoder with a bidirectional LSTM bottleneck**. Abstract (verified): *"with proper data augmentation, Demucs beats all existing state-of-the-art architectures… with 6.3 SDR on average, (and up to 6.8 with 150 extra training songs, even surpassing the IRM oracle for the bass source)."*

## 2. Method — loss (the Direction-01 crux) and the honest attribution
- **Training loss:** Demucs is trained with an **L1 (mean absolute error) loss directly on the output waveform**, $\mathcal{L}=\lVert \hat s - s\rVert_1$. CONFIRMED (WebSearch + it is the well-known Demucs choice).
- **What the paper actually ablates:** the reported loss ablation compares **L1 vs L2/MSE on the waveform**, and finds **L1 better** ("improved performance quite a bit"). This L1-vs-L2 comparison is the verifiable ablation.
- **The "L1 > SI-SNR" claim — precise status.** RESEARCH_DIRECTIONS §1.2 and #1 attribute to 1911.13254 that "L1 waveform loss outperforms SI-SNR-style losses for MSS." The **clean, same-architecture L1-vs-SI-SNR ablation does not appear to be in the paper** (WebSearch summary: "multi-scale spectral loss and SI-SDR were **not tested** in the original Demucs ablation studies"). The SI-SNR comparison is **indirect**: Conv-TasNet — trained with SI-SNR, per its speech-separation origin — produces more audible artifacts (confirmed by the paper's human eval) and Demucs (L1) beats it, but that confounds architecture with loss.
  - **Resolution (do not overstate):** the defensible reading is *"the Demucs line favors waveform L1/regression over the SI-SNR objective inherited from Conv-TasNet; Demucs's own controlled ablation is L1-vs-L2."* The **controlled** loss benchmark that genuinely ranks L1 vs SI-SDR vs spectral losses is **Gusó et al. 2022** (`../../01-loss-function-study/research/papers/guso2022-loss-metrics.md`), which is where Direction 01's tension properly lives. **Marked [PARTIALLY VERIFIED]**; RESEARCH_DIRECTIONS is *not* edited (the spirit is defensible and the body was unreadable), but Direction 01 states the nuance explicitly and leans on Gusó for the controlled claim.

## 3. Method — augmentation recipe (from `demucs/augment.py`, verified) — the standard MSS recipe
Exact classes and defaults read from `facebookresearch/demucs` `augment.py`:
| Class | Transform | Defaults |
|---|---|---|
| `Shift` | random temporal shift up to N samples | `shift=8192, same=False` |
| `FlipChannels` | random L/R stereo swap | — |
| `FlipSign` | random polarity inversion $s\to -s$ | equal prob |
| `Remix` | **shuffle sources within a batch group** to build new mixtures (cross-track remix) | `proba=1, group_size=4` |
| `Scale` | random per-source gain | `proba=1, min=0.25, max=1.25` |

Pitch/tempo shift is a **separate, heavier** augmentation (applied at data-loading, not in `augment.py`) — optional in this project. This table (plus UMX's `data.py`) is the code provenance for Direction 02's factorized ablation; note **`Remix` (source shuffling) and `Scale` (gain 0.25–1.25×) match UMX exactly**, and Demucs adds `FlipSign`.

## 4. Key results (exact, with convention)
- **6.3 dB average SDR** (all 4 sources, **museval**, MUSDB, no extra data); **6.8 dB with +150 extra songs** — and at that point *beats the IRM oracle on bass*. CONFIRMED (abstract). These are museval SDR, not SI-SDR — do not compare to StemCraft's window SI-SDR.
- Human eval: Demucs rated more natural than Conv-TasNet, but "suffers from some bleeding, especially between vocals and other."
- Quantizable to 120 MB without accuracy loss.

## 5. Relevance to this project
- **Alternatives (PLAN §2.2):** the "waveform from scratch" option — cited but not the headline (compute/data-hungry).
- **Direction 01:** the L1-champion anchor and the honest source of the L1-vs-SI-SNR tension (see §2; controlled claim → Gusó).
- **Direction 02 (CORE):** `augment.py` is the canonical recipe factorized (Remix is H2's single highest-leverage lever).
- **Direction 10:** the Demucs **family** is the distillation teacher (see `demucs-hybrid-family.md` for the actual `htdemucs` teacher).

## 6. Verification notes
- Title/authors/ISMIR-2019/abstract numbers (6.3, 6.8, +150, IRM-bass, Conv-TasNet, bleeding): CONFIRMED (HF paper_search abstract).
- Second arXiv ID 1909.01174: CONFIRMED (WebSearch).
- Augmentation classes/defaults: CONFIRMED from `augment.py` source.
- L1 training loss: CONFIRMED (WebSearch + known). **L1-vs-L2 ablation: CONFIRMED (WebSearch).** **L1-vs-SI-SNR as a clean ablation: NOT confirmed — [PARTIALLY VERIFIED], indirect via Conv-TasNet; paper body unreadable via available routes.** This is the single flagged nuance in the shared library.
</content>
