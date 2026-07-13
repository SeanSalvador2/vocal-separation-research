# On loss functions and evaluation metrics for music source separation

Enric Gusó, Jordi Pons, Santiago Pascual, Joan Serrà (Dolby). ICASSP 2022 / arXiv **2202.07968** (submitted 16 Feb 2022). | Verified: WebSearch (arXiv abs, ICASSP 2022, author list) — not surfaced by HF paper_search; **paper body not readable** (arxiv/ar5iv/arxiv-vanity all blocked, jordipons.me 403). Scope CONFIRMED via abstract; specific loss rankings marked `[UNVERIFIED]`. Verified 2026-07-13.

**The CORE paper for Direction 01** — the only *controlled* benchmark that ranks MSS losses head-to-head, and the paper that turns Demucs's informal "L1 works" into a real question.

## 1. Problem & context
Practitioners pick a loss by folklore (Spleeter→L1-mag, UMX→MSE-mag, Demucs→L1-waveform) and then evaluate on SDR. Two things are unexamined: (i) *which loss actually produces the best separations*, and (ii) *whether SDR is even the right yardstick*. Gusó et al. benchmark an extensive set of MSS losses **in one controlled experimental setup** (fixed model/data/protocol), then re-use those same losses as **candidate evaluation metrics** by cross-correlating each against a **subjective listening test**. Motivating observation (abstract, verified): the standard SDR "can be misleading in some scenarios."

## 2. Method — the loss zoo (definitions the MASTER_PLAN needs)
The paper spans time-domain and frequency-domain families, with plain and log-compressed and phase-sensitive variants. The definitions below are the standard forms Direction 01 will implement; the paper's own notation (e.g. `L2freq`, `SISDRfreq`, `LOGL2freq`, `LOGL1freq`) denotes the frequency-domain versions.

Notation: mixture $x$, target source $s$, estimate $\hat s$; STFT $X,S,\hat S$ (complex); magnitudes $|X|,|S|,|\hat S|$; predicted mask $M$, so $\hat S = M\odot|X|\cdot e^{i\angle X}$ (magnitude mask + mixture phase) and $\hat s=\text{iSTFT}(\hat S)$.

**Time-domain losses**
- **L1 (waveform):** $\mathcal L_{1}= \lVert \hat s - s\rVert_1$ — Demucs's choice.
- **L2/MSE (waveform):** $\mathcal L_{2}= \lVert \hat s - s\rVert_2^2$.
- **(negative) SI-SDR:** $\mathcal L_{\text{SISDR}} = -10\log_{10}\dfrac{\lVert \alpha s\rVert^2}{\lVert \alpha s-\hat s\rVert^2},\ \alpha=\hat s^\top s/\lVert s\rVert^2$ (see `../../../00-shared-research/papers/leroux2019-si-sdr.md`). Differentiable through iSTFT; directly optimizes the eval metric — the arm this project is most curious about.

**Frequency-domain (spectrogram) losses**
- **L1-mag / L2-mag:** $\lVert |\hat S|-|S|\rVert_{1}$ (Spleeter), $\lVert |\hat S|-|S|\rVert_{2}^2$ (UMX). Equivalent to a masked-magnitude regression.
- **log-L1-mag / log-L2-mag (`LOGL1freq`/`LOGL2freq`):** $\lVert \log(|\hat S|+\epsilon)-\log(|S|+\epsilon)\rVert_{1\text{ or }2}$ — compresses dynamic range, weighting quiet TF structure more (closer to perceptual loudness).
- **phase-sensitive / complex (`L2freq` on complex $S$, `SISDRfreq`):** losses computed on complex $S$ or with a phase-sensitive target, penalizing phase error the magnitude losses ignore — the paper's argument that phase-aware frequency losses can beat magnitude-only ones.

**The metric–loss duality (the paper's twist).** Each loss $\mathcal L$ is also scored as a *metric* by computing $\mathcal L(\hat s,s)$ on the test set and correlating with human ratings. A loss that correlates well with perception is both a good training objective *and* a better yardstick than SDR.

## 3. Key results
- **CONFIRMED (abstract):** an extensive controlled benchmark of losses; losses re-used as metrics via cross-correlation with a subjective test; **SDR can mislead**; frequency-domain / spectrogram losses (the L2/log family) are competitive and phase-sensitive losses are considered.
- **`[UNVERIFIED]` (from prior search snippet; body inaccessible):** the specific recommendation that spectrogram-domain / phase-sensitive frequency losses — reported as **`L2freq`, `SISDRfreq`, `LOGL2freq`, `LOGL1freq`** — are the best-performing and best-correlated. Treat the *ranking of specific named losses* as unverified; the *direction* (spectrogram/phase-sensitive losses competitive-to-better, SDR imperfect) is verified. RESEARCH_DIRECTIONS §1.2's characterization ("spectrogram-domain losses (L2/log-L1 family) competitive… SDR can mislead") is consistent with the verified scope and is retained.

## 4. Limitations & caveats
- Single controlled setup (their model/data); not necessarily the *compact-U-Net-on-MUSDB-only* regime this project runs — which is exactly why replicating at small scale is a real (not redundant) contribution.
- Subjective test is finite-N; correlations are indicative.

## 5. Relevance to this project (Direction 01 specifically)
- **This is the paper Direction 01 replicates-and-narrows.** Gusó benchmarks many losses in one (mid-scale) setup; Direction 01 asks the same question in the **compact-mask-U-Net-on-MUSDB-only** regime with the project's frozen protocol and seed policy.
- The **five losses** in Direction 01's minimal experiment (L1-mag, MSE-mag, log-L1-mag, time-domain SI-SDR, L1+MR-STFT) are a deliberate subset of Gusó's zoo chosen to (a) span the magnitude vs time-domain vs multi-resolution axes and (b) directly test the "train on your eval metric (SI-SDR)?" tension.
- **What we replicate:** the controlled loss-comparison methodology and the SDR-can-mislead finding.
- **What we extend:** small-scale regime + a MR-STFT auxiliary specifically as an *artifact reducer* (measured by the 5-clip listening check), not an SDR chaser — the Bake-Off (`../../../00-shared-research/papers/bakeoff2025-metrics-perception.md`) says SDR is the best perceptual proxy *for vocals*, so we do not expect embedding/spectral losses to raise SDR; we expect them to cut artifacts at equal SDR.
- **What we deliberately ignore:** the full metric-battery; we report SI-SDR + museval SDR + the small listening check only.

## 6. Verification notes
- Title/authors/venue/arXiv ID: CONFIRMED (WebSearch).
- Scope (controlled loss benchmark; losses-as-metrics; SDR misleading; phase-sensitive/spectrogram losses considered): CONFIRMED (abstract via WebSearch).
- **Specific recommended loss names + rankings: `[UNVERIFIED]`** — paper body unreadable via every working route; do not assert the exact winning loss without a primary read. This is the one Direction-01 fact that a future full-text read should close.
