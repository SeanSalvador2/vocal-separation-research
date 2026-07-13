# BSS-Eval, museval, and SiSEC 2018 — the literature metric and its silence behavior

Three bound-together sources. Verified 2026-07-13:
- **BSS-Eval** (SDR/ISR/SIR/SAR): Vincent, Gribonval, Févotte, *"Performance measurement in blind audio source separation,"* IEEE TASLP 14(4):1462–1469, **2006** — no arXiv; venue/definition CONFIRMED via WebSearch + reproduced in SiSEC 2018.
- **SiSEC 2018**: Stöter, Liutkus, Ito, *"The 2018 Signal Separation Evaluation Campaign,"* arXiv **1804.06267** — route: WebSearch (title/authors/April-2018/introduces MUSDB18 + Python BSSEval + 3 oracle masks).
- **museval** silent-frame handling: read directly from `sigsep/sigsep-mus-eval` `museval/metrics.py` — route: WebFetch raw.githubusercontent.com. **This code behavior is load-bearing for Direction 08.**

## 1. Problem & context
BSS-Eval is the field-standard MUSDB scoring used by essentially every paper's headline "X dB SDR." SiSEC 2018 released the MUSDB18 database (`musdb18-dataset.md`) and the official Python `museval` toolbox, so all post-2018 MUSDB numbers use this estimator. Understanding *exactly* what it computes — and what it silently ignores — is what keeps this project's metric comparisons honest.

## 2. Method — the math and the framing
BSS-Eval decomposes an estimate $\hat s$ into four orthogonal parts by projecting onto subspaces spanned by (allowed) filtered versions of the true source and interferers:

$$\hat s \;=\; s_{\text{target}} \;+\; e_{\text{interf}} \;+\; e_{\text{noise}} \;+\; e_{\text{artif}},$$

with (all in dB):
$$\text{SDR}=10\log_{10}\frac{\lVert s_{\text{target}}\rVert^{2}}{\lVert e_{\text{interf}}+e_{\text{noise}}+e_{\text{artif}}\rVert^{2}},\quad
\text{SIR}=10\log_{10}\frac{\lVert s_{\text{target}}\rVert^{2}}{\lVert e_{\text{interf}}\rVert^{2}},\quad
\text{SAR}=10\log_{10}\frac{\lVert s_{\text{target}}+e_{\text{interf}}+e_{\text{noise}}\rVert^{2}}{\lVert e_{\text{artif}}\rVert^{2}}.$$

Crucially, $s_{\text{target}}$ is allowed to be a **distortion-filtered** version of $s$ (up to a 512-tap FIR in the standard toolbox) — this is the "free filter" Le Roux et al. (`leroux2019-si-sdr.md`) object to. ISR (source Image-to-Spatial-distortion Ratio) measures spatial/channel error.

**Aggregation convention (the one everyone quotes).** museval computes metrics on **1 s windows** (framewise), then reports the **median over frames within a track**, then the **median over the 50 test tracks**. "museval SDR" = median-of-frames, median-of-tracks. This is a *different estimator* from StemCraft's SI-SDR-on-a-12 s-window; the two cannot be subtracted (RESEARCH_NOTES §0).

## 3. Silent-frame handling — verified from `metrics.py` (Direction 08's foundation)
Read directly from the museval source:
- **Whole-source guard:** museval asserts every reference source is non-silent; `_any_source_silent()` detects all-zero references and the evaluator raises rather than score an underdetermined case ("at least one of the reference sources is all 0s… introduces ambiguity").
- **Per-frame guard (the key behavior):** framewise, for each 1 s window,
  ```
  if not _any_source_silent(ref_slice) and not _any_source_silent(est_slice):
      # compute SDR/ISR/SIR/SAR normally
  else:
      a = np.empty((4, nsrc, nsrc)); a[:] = np.nan   # SDR,ISR,SIR,SAR
      s_r[:, :, :, t] = a
  ```
  **Any frame in which the reference (or estimate) source is silent is assigned `NaN` for all four metrics and dropped from the median.**

**Consequence (state this in Direction 08 and THEORY.md §5):** museval SDR is computed *only over frames where the vocal is active*. Energy the model leaks into `v̂` during a **truly silent vocal region contributes nothing to the SDR** — those frames are `NaN`ed away. Combined with SI-SDR's $s=0$ singularity (`leroux2019-si-sdr.md` §3), **no standard MSS metric measures vocal leakage into silence**. That is precisely the "ghost vocals in the instrumental break" failure a karaoke product cares about, and the gap Direction 08's `silence-leakage` metric is defined to fill.

## 4. Oracle masks (SiSEC 2018 → PLAN H4)
SiSEC 2018 shipped reference implementations of **IBM, IRM, and MWF** oracles on MUSDB. These are the honest ceilings for the mask family and are reused by Directions 03/04-lineage and the H4 oracle study.

## 5. Relevance to this project
- **All directions:** museval BSS-Eval SDR is the literature-comparability secondary; report it in its own table, median-of-frames/median-of-tracks, never mixed with SI-SDR.
- **Direction 08 (CORE dependency):** the `NaN`-on-silence behavior above is *the* motivation and must be cited from code (function `_any_source_silent`, the framewise `NaN` insertion).
- **Bake-Off (`bakeoff2025-metrics-perception.md`):** shows which of these BSS-Eval sub-metrics (SDR vs SI-SAR) best track human perception per stem.

## 6. Verification notes
- BSS-Eval definitions/decomposition and the 512-tap filter: CONFIRMED via WebSearch (Vincent 2006 TASLP); the four-term decomposition is textbook/standard.
- SiSEC 2018 (MUSDB18 introduction, Python BSSEval, 3 oracle masks): CONFIRMED via WebSearch of 1804.06267.
- **Silent-frame `NaN` behavior: CONFIRMED directly from museval `metrics.py` source** (quoted logic above) — the single most important verified fact for Direction 08.
- 1 s window / median-of-frames/median-of-tracks aggregation: standard museval convention [CONFIRMED as museval default; exact window length is the toolbox default].
