# Decoupling Magnitude and Phase Estimation with Deep ResUNet for Music Source Separation

Qiuqiang Kong, Yin Cao, Haohe Liu, Keunwoo Choi, Yuxuan Wang (ByteDance). ISMIR 2021 / arXiv **2109.05418**. Code: `bytedance/music_source_separation`. | Verified: HF paper_search (title/authors/date/abstract), 2026-07-13.

The paper that **quantifies the magnitude-mask ceiling** — the theoretical backbone of PLAN's phase discussion (THEORY.md §2, H4) and the reference for why a soft ratio mask caps out.

## 1. Problem & context
Magnitude-mask + mixture-phase models (Spleeter/UMX) have three stated limits, verbatim from the abstract: (1) **incorrect phase reconstruction degrades performance**; (2) a bounded mask $M\in[0,1]$ is wrong because **"22% of time-frequency bins have ideal ratio mask values of over 1"** on MUSDB18; (3) very deep architectures are under-explored. Kong et al. fix all three.

## 2. Method — the math
Work in the complex STFT domain. Mixture $X=S+N$ (complex, per TF bin), target $S$.

**Ideal ratio mask (magnitude) and why it exceeds 1.** The magnitude IRM is $M_{\text{IRM}} = |S|/|X|$. Because $X=S+N$ and interferers can partially cancel the target, $|X|$ can be *smaller* than $|S|$, so $M_{\text{IRM}}>1$ for a substantial fraction of bins — measured at **22% on MUSDB18**. A sigmoid mask ($\in[0,1]$) *cannot represent* those bins; hence "allow the mask magnitude to exceed 1."

**Complex ideal ratio mask (cIRM).** Define a complex mask $M=M_r+iM_i$ with $\hat S = M \odot X$ (complex product). The exact target is
$$M_{\text{cIRM}} = \frac{S\,\overline{X}}{|X|^2} = \frac{S_r X_r + S_i X_i}{X_r^2+X_i^2} + i\,\frac{S_i X_r - S_r X_i}{X_r^2+X_i^2}.$$

**The decoupling trick (the contribution).** Rather than regress $(M_r,M_i)$ directly, they **decouple magnitude and phase**: the network predicts (a) a **bounded mask magnitude** $|M|$ (via a learned, unbounded-capable parameterization) and (b) a **phase** $\angle M$ (as $(\cos,\sin)$ or an angle), and reconstructs
$$\hat S = \underbrace{|M|\,|X|}_{\text{corrected magnitude}}\;\cdot\;\exp\!\big(i(\angle X + \angle M)\big).$$
So the estimated **phase** is the mixture phase $\angle X$ *rotated* by a learned $\angle M$ — this is what recovers phase the mixture-phase baseline throws away. The magnitude branch is allowed to exceed 1 (fixing limit 2).

**Backbone.** A **residual U-Net up to 143 layers** (ResUNet) — deep enough to test limit 3.

## 3. Key results (exact, with convention)
- **Vocals SDR (museval, MUSDB18): 7.24 → 8.98 dB** — the cIRM ResUNet beats the previous best (7.24 dB) by **+1.74 dB**, a then-SOTA vocals result. CONFIRMED (abstract).
- The **22% of TF bins with IRM > 1** statistic is stated in the abstract (not just body) — CONFIRMED verbatim.

## 4. Limitations & caveats
- 143 layers is **heavy**; the gain conflates *cIRM target* with *massive capacity*. The paper does not isolate "how much is phase modeling vs how much is depth" at small scale — **precisely the open question** a param-matched study (the retired Direction #4, and the framing for H4) would answer.
- Complex-valued targets are harder to train/debug (unit tests on synthetic signals recommended).

## 5. Relevance to this project
- **THEORY.md §2 (CORE):** the IRM>1 fact, the cIRM decoupling equations above, and the Wiener/MMSE connection are the math for the "why magnitude+mixture-phase caps quality" section.
- **PLAN H4 (oracle headroom):** motivates scoring oracle IRM/cIRM to bound how much of SingNet's gap is *phase* vs *capacity/data*. The 22%-of-bins-need-M>1 fact is why the sweep includes an **unbounded (ReLU) mask** alongside the sigmoid ratio mask (PLAN Phase 5 mask-type knob).
- **Direction 08 / 03:** the per-TF-bin analysis style (which bins need M>1, band structure) is the analysis template for band-wise error breakdowns.
- **Deliberately not reproduced:** 143-layer cIRM from scratch is out of scope; cited as the ceiling-of-the-mask-family and the phase-limit quantifier.

## 6. Verification notes
- Title/authors/ISMIR-2021/arXiv ID and all numbers (22% bins IRM>1, 7.24→8.98 dB vocals, 143 layers, cIRM): CONFIRMED verbatim from HF paper_search abstract.
- The cIRM decoupling equations ($M_{\text{cIRM}}=S\bar X/|X|^2$; $\hat S=|M||X|e^{i(\angle X+\angle M)}$): standard complex-mask algebra, consistent with the abstract's description of decoupling magnitude/phase estimation [equations reproduced from method knowledge; abstract confirms the decoupling and the >1 allowance].
