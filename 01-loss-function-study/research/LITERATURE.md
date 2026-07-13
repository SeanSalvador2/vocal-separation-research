# Direction 01 — "Train on what you test?" A controlled loss-function study at small scale

Literature guide for the loss study. Deep-dives: [`papers/guso2022-loss-metrics.md`](papers/guso2022-loss-metrics.md) (CORE),
[`papers/yamamoto2020-mrstft.md`](papers/yamamoto2020-mrstft.md). Shared dependencies linked below.

## 1. Direction recap (from RESEARCH_DIRECTIONS #1)
- **Hypothesis (H5 candidate):** training a compact mask U-Net directly on (negative) **SI-SDR does *not* beat L1-magnitude** on SI-SDR evaluation — the literature's tension holds at small scale — **but a multi-resolution STFT auxiliary term reduces audible artifacts**.
- **Minimal experiment:** fixed architecture/data/seed policy; **5 losses** — L1-mag, MSE-mag, log-L1-mag, time-domain SI-SDR, (L1 + MR-STFT) — × **3 seeds**; evaluate **SI-SDR**, **museval SDR**, and a **5-clip artifact listening check**.
- **Falsification / success test:** a loss-ranking table with seed error bars. **H5 supported iff the SI-SDR-trained model is ≤ the L1-magnitude model on SI-SDR *within the seed noise band*** (i.e. training on the eval metric buys nothing at this scale). A clean win for SI-SDR-training *refutes* H5 (equally reportable). The MR-STFT sub-claim is supported iff it wins the listening check at **equal** SI-SDR.
- Difficulty/Risk: **Low/Low** — every outcome is reportable.

## 2. Paper map
| Paper | Role | Specific claim we rely on | Verified | Deep-dive |
|---|---|---|---|---|
| Gusó et al. 2022 (2202.07968) | **replicate** | controlled loss benchmark exists; SDR can mislead; spectrogram/phase-sensitive losses competitive | CONFIRMED scope; specific rankings `[UNVERIFIED]` | [papers/guso2022-loss-metrics.md](papers/guso2022-loss-metrics.md) |
| Yamamoto et al. 2020 (1910.11480) | **infrastructure** | MR-STFT = spectral-convergence + log-mag over resolutions | CONFIRMED (+ auraloss code) | [papers/yamamoto2020-mrstft.md](papers/yamamoto2020-mrstft.md) |
| Demucs v1 (1911.13254) | **context / motivation** | waveform L1 preferred over SI-SNR-inherited Conv-TasNet | **PARTIAL** (L1-vs-L2 in paper; L1-vs-SI-SNR indirect) | [shared](../../00-shared-research/papers/demucs2019-v1.md) |
| Spleeter (JOSS) | **control (loss)** | L1-on-magnitude is the from-scratch default | CONFIRMED | [shared](../../00-shared-research/papers/spleeter2020.md) |
| Open-Unmix (JOSS) | **control (loss)** | MSE-on-magnitude alternative | CONFIRMED (code) | [shared](../../00-shared-research/papers/openunmix2019.md) |
| Le Roux 2019 (1811.02508) | **infrastructure** | SI-SDR definition + scale-invariance + silence singularity | CONFIRMED | [shared](../../00-shared-research/papers/leroux2019-si-sdr.md) |
| Bake-Off 2025 (2507.06917) | **context (metric policy)** | for vocals, SDR is the best perceptual proxy → expect MR-STFT to help artifacts, not SDR | CONFIRMED | [shared](../../00-shared-research/papers/bakeoff2025-metrics-perception.md) |

## 3. Replicate-vs-extend
- **Replicate:** Gusó et al.'s *controlled loss comparison* methodology (fixed model/data, swap only the loss) and its headline that training-metric ≠ eval-metric is subtle.
- **Extend (our delta):** run it in the **compact-magnitude-mask-U-Net-on-MUSDB18-only** regime (Gusó used a larger/different setup) under the project's frozen 86/14/50 protocol and 3-seed noise-band policy, and add the **MR-STFT-as-artifact-reducer** test with a listening check. The novel, defensible sentence: *"Papers disagree on whether to train on your eval metric; I ran the controlled comparison at my scale and measured it with seeds + a listening check."*

## 4. The gap
- Gusó's ranking is at one (non-compact) scale; **nobody reports the L1-vs-SI-SDR-vs-MR-STFT comparison for a ~5–15 M-param mask U-Net trained on MUSDB-only** with seed noise bands and a paired listening check. The Demucs "L1 beats SI-SNR" line is **not a clean same-architecture ablation** (it is confounded via Conv-TasNet — see the shared Demucs note §2), so the tension is genuinely *open* at small scale, not settled.
- **This gap is not closed by a mid-2026 paper** (no controlled compact-MSS loss study surfaced in verification).

## 5. Direction-specific technical notes (the loss zoo, for MASTER_PLAN / THEORY.md §4)
With mixture $x$, target $s$, mask $M\in[0,1]^{F\times T}$, $\hat S=M\odot|X|e^{i\angle X}$, $\hat s=\text{iSTFT}(\hat S)$:

| # | Loss | Formula | Domain | Needs diff. iSTFT? | Provenance |
|---|---|---|---|---|---|
| 1 | **L1-mag** | $\lVert M\odot|X|-|S|\rVert_1$ | magnitude | no | Spleeter |
| 2 | **MSE-mag** | $\lVert M\odot|X|-|S|\rVert_2^2$ | magnitude | no | UMX |
| 3 | **log-L1-mag** | $\lVert \log(|\hat S|+\epsilon)-\log(|S|+\epsilon)\rVert_1$ | log-magnitude | no | Gusó `LOGL1freq` |
| 4 | **−SI-SDR (time)** | $-10\log_{10}\frac{\lVert\alpha s\rVert^2}{\lVert\alpha s-\hat s\rVert^2},\ \alpha=\frac{\hat s^\top s}{\lVert s\rVert^2}$ | time | **yes** | Le Roux 2019 |
| 5 | **L1 + MR-STFT** | $\lVert M\odot|X|-|S|\rVert_1 + \lambda\!\sum_m(\mathcal L_{\text{sc}}^{(m)}+\mathcal L_{\text{mag}}^{(m)})$ | multi-res | **yes** (for MR term) | PWG / auraloss |

with $\mathcal L_{\text{sc}}=\lVert|S|-|\hat S|\rVert_F/\lVert|S|\rVert_F$, $\mathcal L_{\text{mag}}=\lVert\log(|S|+\epsilon)-\log(|\hat S|+\epsilon)\rVert_1$, resolutions `fft=[1024,2048,512]`, `hop=[120,240,50]`, `win=[600,1200,240]`.

**Implementation notes.** Losses 4–5 require a differentiable iSTFT (unit-test STFT→iSTFT round-trip < −60 dB per PLAN Phase 2). SI-SDR's $\lVert s\rVert^2$ denominator is singular on silent target chunks → guard with an $\epsilon$ and/or exclude fully-silent chunks from the SI-SDR loss term (ties to Direction 08). Report all five on **both** SI-SDR and museval SDR so a loss that games one estimator is visible.

## 6. Risks this literature implies
- **SI-SDR loss instability on silent/near-silent chunks** (Le Roux §3 singularity) can make loss 4 train worse for reasons unrelated to the hypothesis — mitigate with the $\epsilon$-guard and chunk-sampling policy; document it.
- **MR-STFT may not move SDR** (Bake-Off says SDR already tracks vocal perception) — so pre-register that the MR-STFT claim is judged on the *listening check at equal SDR*, not on SDR, to avoid a false "no effect" conclusion.
- **Gusó's specific ranking is `[UNVERIFIED]`** — do not state "Gusó found loss X best" in the report without a primary read; cite only the verified scope (SDR can mislead; spectrogram losses competitive).
- **Confound control:** identical architecture/data/optimizer/seeds across the 5 losses is essential (this is the whole point); only the loss and, where required, the presence of a differentiable iSTFT path may change.
