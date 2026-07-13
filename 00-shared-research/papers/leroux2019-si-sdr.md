# SDR – half-baked or well done?

Jonathan Le Roux, Scott Wisdom, Hakan Erdogan, John R. Hershey. ICASSP 2019. arXiv **1811.02508**. | Verified: HF paper_search (title/authors/date "6 Nov 2018") + WebSearch (arXiv ID, ICASSP 2019), 2026-07-13.

This is the definition-of-record for the project's **primary metric (SI-SDR)** and the honest source for its failure modes. StemCraft's `bench/metrics.py:si_sdr` implements exactly this.

## 1. Problem & context
BSS-Eval SDR (Vincent et al. 2006; see `bsseval-museval-sisec2018.md`) allows an arbitrary *distortion filter* (up to 512-tap FIR) when matching estimate to reference, so an algorithm can look better simply by implicitly rescaling/refiltering. Le Roux et al. argue this has been "improperly used and abused, especially in the case of single-channel separation," where hundreds of papers claimed 0.1 dB wins. They propose a simpler, filter-free, **scale-invariant** measure. (Abstract verbatim confirms: *"various examples of critical failure of the original SDR that SI-SDR overcomes."*)

## 2. Method — the math
Let $s\in\mathbb{R}^{L}$ be the reference source and $\hat s\in\mathbb{R}^{L}$ the estimate. SI-SDR allows only a single global scale $\alpha$ on the reference (equivalently, on the estimate), chosen by least squares to be optimal, then measures the residual:

$$\alpha \;=\; \frac{\hat s^{\top} s}{\lVert s\rVert^{2}}, \qquad
s_{\text{target}} \;=\; \alpha\, s, \qquad
e_{\text{noise}} \;=\; \hat s - s_{\text{target}},$$

$$\boxed{\;\text{SI-SDR}(\hat s, s) \;=\; 10\log_{10}\frac{\lVert s_{\text{target}}\rVert^{2}}{\lVert e_{\text{noise}}\rVert^{2}}
\;=\; 10\log_{10}\frac{\lVert \alpha s\rVert^{2}}{\lVert \alpha s - \hat s\rVert^{2}}\;}$$

$s_{\text{target}}=\alpha s$ is the orthogonal projection of $\hat s$ onto the line spanned by $s$; $e_{\text{noise}}\perp s_{\text{target}}$ by construction.

**Scale invariance (the defining property).** For any $c\neq 0$, replacing $\hat s\to c\hat s$ scales both $s_{\text{target}}$ and $e_{\text{noise}}$ by $c$, so the ratio — and SI-SDR — is unchanged. This removes the "free" gain that BSS-Eval's filter granted. **SI-SDRi** ("improvement") = SI-SDR$(\hat s,s)-$SI-SDR$(x,s)$ where $x$ is the input mixture; it is the number StemCraft reports and the karaoke-relevant delta.

Relation to (negative) SI-SDR as a **training loss**: minimizing $-\text{SI-SDR}$ is a differentiable objective on the time-domain waveform (requires a differentiable iSTFT if the model predicts spectrogram masks). This is the objective Direction 01 tests against L1-magnitude.

## 3. Failure modes / pathologies (the reason §4 exists in THEORY.md, and the reason Direction 08 exists)
1. **Silent reference ($s=0$).** $\alpha=\hat s^{\top}s/\lVert s\rVert^{2}$ divides by zero → SI-SDR **undefined**. Any nonzero leakage $\hat s$ during a truly silent vocal region has *no* well-defined SI-SDR. This is exactly why museval inserts `NaN` for silent frames (`bsseval-museval-sisec2018.md` §3) and why energy leaked into silent regions is **invisible** to SDR-family metrics — the gap Direction 08's custom metric fills.
2. **Silent estimate ($\hat s=0$), $s\neq0$.** $\alpha=0\Rightarrow s_{\text{target}}=0,\;e_{\text{noise}}=0$ → $0/0$, undefined; approached as a limit SI-SDR $\to-\infty$.
3. **Orthogonal error.** As $\hat s\to$ orthogonal to $s$, $\alpha\to 0$ and SI-SDR $\to -\infty$; StemCraft's unit test uses a constructed orthogonal-error case scoring a finite reference value (~20 dB for its specific construction).
4. **Perfect-gain reference.** $\hat s=cs$ gives $e_{\text{noise}}=0$ → $+\infty$; StemCraft tests this returns `+inf`.

## 4. Key results
No large benchmark; the contribution is definitional + counterexamples showing BSS-Eval SDR can rank systems differently from SI-SDR on constructed pathological pairs. Adopted field-wide as the single-channel default post-2019.

## 5. Relevance to this project
- **Metric (all directions):** SI-SDR (StemCraft convention) is the primary axis; museval BSS-Eval SDR is the clearly-labeled literature secondary. **Never conflated** (RESEARCH_NOTES §0).
- **Direction 01:** time-domain $-$SI-SDR is one of the five losses; its scale invariance and silence degeneracy are the theoretical reasons it may *not* dominate L1-magnitude on music (cf. Gusó et al.).
- **Direction 08:** pathologies (1)–(2) are the formal justification that a *separate* silence-leakage metric is needed — SDR literally cannot score silent regions.
- **THEORY.md §4/§5:** the boxed derivation and scale-invariance proof go here directly.

## 6. Verification notes
- Title/authors/venue/arXiv ID: CONFIRMED (HF paper_search + WebSearch).
- SI-SDR formula and scale-invariance: standard, matches the paper's definition; the projection form is reproduced from the paper's equations [core equations CONFIRMED against abstract's claim of a "slightly modified definition… simpler, more robust measure"; algebraic detail is textbook].
- Zero-signal pathologies: derived from the formula (division by $\lVert s\rVert^2$); consistent with the paper's "critical failure" framing and with museval's silent-frame handling (independently verified in code).
</content>
