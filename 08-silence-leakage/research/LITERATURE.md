# Direction 08 — The silence problem: vocal-activity-aware sampling and the cost of quiet

Deep-dive: [`papers/silence-metrics-grounding.md`](papers/silence-metrics-grounding.md) (metric pathologies,
training-code behavior, and the leakage-metric draft). Shared metric notes linked below.

## 1. Direction recap (from RESEARCH_DIRECTIONS #8)
- **Hypothesis:** (a) **energy-weighted chunk sampling beats uniform sampling** on overall vocals SI-SDR; but (b) **fully *dropping* silent chunks hurts false-positive behavior** — measured as energy leaked into $\hat v$ during truly-silent vocal regions (a "silence-leakage" metric we define).
- **Minimal experiment:** **4 sampling policies** — uniform / energy-weighted / drop-silent / curriculum (silent→active) — same budget, **2–3 seeds**; evaluate **overall SI-SDR *and* the silence-leakage metric** on test-set silent regions (from GT stems).
- **Falsification / success test:** a **two-axis result (quality vs silence leakage)** showing a real tradeoff. Supported iff energy-weighted ≥ uniform on SI-SDR *and* drop-silent worsens leakage (SLR) outside the seed band. Any coherent tradeoff plot is reportable.
- Difficulty/Risk: **Low-Medium / Low.**

## 2. Paper map
| Paper / source | Role | Specific claim we rely on | Verified | Where |
|---|---|---|---|---|
| museval `metrics.py` | **infrastructure** | silent frames → `NaN`, dropped from SDR | CONFIRMED (code) | [shared](../../00-shared-research/papers/bsseval-museval-sisec2018.md) |
| Le Roux 2019 (1811.02508) | **infrastructure** | SI-SDR singular on silent target | CONFIRMED | [shared](../../00-shared-research/papers/leroux2019-si-sdr.md) |
| Open-Unmix `data.py` | **baseline (control)** | uniform random-start chunking, no activity filter | CONFIRMED (code) | [papers/silence-metrics-grounding.md](papers/silence-metrics-grounding.md) |
| BSRNN (2209.15174) | **context** | source-activity detector (for pseudo-labels, not sampling) | CONFIRMED | [Dir 03](../../03-mini-band-split/research/papers/luo2022-bsrnn.md) |
| Demucs (1911.13254) | **context** | chunk-drop p=0.1 simulates silence | CONFIRMED (aug) | [shared](../../00-shared-research/papers/demucs2019-v1.md) |
| BSMamba2 "Mamba2 Meets Silence" (2508.14556) | **recent related** | sparse-vocal robustness via Mamba2 (11.03 dB cSDR) | CONFIRMED (WS) | §4 |
| Continual SVS (2512.02432) | **cross-ref** | human marks false positives = leakage | CONFIRMED | [Dir 05](../../05-lora-source-separation/research/papers/continual-svs-2025.md) |

## 3. Replicate-vs-extend
- **Replicate:** the standard uniform chunk-sampling baseline (UMX, code-verified) and the general idea of activity-aware sampling (BSRNN's activity detector, applied here to *labeled-data sampling* rather than pseudo-labeling).
- **Extend (our delta):** a **custom silence-leakage metric (SLR)** that measures exactly the karaoke failure the standard metrics ignore, plus a **controlled 4-policy comparison** producing a **quality-vs-leakage tradeoff** — and the metric ships in `singnet/` (~50 lines + tests) and improves the shipped product (no ghost vocals in instrumental breaks).

## 4. The gap — honest, novelty-affecting (READ THIS)
RESEARCH_DIRECTIONS #8 states "**no paper reports a controlled chunk-sampling-policy ablation for MSS training**." The **gap-check (2026-07-13, WebSearch: "chunk sampling policy ablation MSS vocal activity weighted silent segments")** shows this is **somewhat too strong** — adjacent prior exists:
- an SVS data-sampling ablation forcing a **% of chunks to contain singing voice ∈ [0,25,50,75,100]%** (an activity-weighted-sampling ablation in spirit; exact citation not pinned from the snippet — `[unverified citation]`);
- **Demucs** drops chunks with p=0.1 to simulate silence;
- **"Mamba2 Meets Silence" (BSMamba2, 2508.14556, Aug 2025)** targets the *same silence/sparse-vocal problem* but via **architecture** (Mamba2 long-range modeling, 11.03 dB cSDR SOTA), not sampling policy or a leakage metric.

**What remains genuinely novel:** (i) the **custom silence-leakage metric (SLR)** — no source defines a silence-leakage/precision metric; the field either ignores silent frames (museval `NaN`) or notes SDR's silence-sensitivity without a dedicated measure; (ii) the **explicit quality-vs-leakage two-axis tradeoff** across 4 policies for a compact model; (iii) the **karaoke-product framing**. **Recommendation to orchestrator:** soften the "nobody reports a chunk-sampling ablation" claim to "**the silence problem in MSS is studied via architecture (BSMamba2) and content-forcing, but no one defines a silence-leakage metric or runs the policy-vs-leakage tradeoff**." The metric is the defensible novelty, not the sampling ablation alone.

## 5. Direction-specific technical notes (the metric + policies — for MASTER_PLAN)
**Silence-leakage metric (SLR), OUR construction** — full formal draft with edge cases (thresholds, $\epsilon$, $L_{\min}$, empty-region `NaN`) is in [`papers/silence-metrics-grounding.md §3`](papers/silence-metrics-grounding.md):
$$\text{SLR}=10\log_{10}\frac{\sum_{t\in R_{\text{sil}}}\hat v(t)^2+\epsilon}{\sum_{t\in R_{\text{sil}}}x(t)^2+\epsilon},\quad R_{\text{sil}}=\{\text{GT-vocal-silent frames, }\ge L_{\min}=0.5\text{ s, }{-60}\text{ dB threshold}\}.$$
Primary threshold $-60$ dB (report $\{-50,-60,-70\}$); $\epsilon\!\sim\!10^{-8}$; empty $R_{\text{sil}}\to$`NaN`, excluded (mirrors museval).

**The 4 sampling policies (hold everything else fixed):**
1. **uniform** — `random.uniform` start (UMX baseline);
2. **energy-weighted** — sample chunk start with probability ∝ windowed GT-vocal energy (never zero — keep a floor so silent chunks retain nonzero mass, per PLAN Phase 2);
3. **drop-silent** — exclude chunks whose GT-vocal RMS < threshold (the policy hypothesized to *worsen* SLR);
4. **curriculum** — start silent-inclusive, anneal toward energy-weighted (or vice-versa).

Report each on **(overall vocals SI-SDR, museval SDR, SLR)**; the headline figure is **SI-SDR vs SLR** with per-policy points + seed CIs.

## 6. Risks this literature implies
- **Novelty is the metric, not the ablation** — frame the report around SLR + the tradeoff; cite BSMamba2 and the forced-% prior so the sampling-ablation claim isn't over-stated (see §4).
- **Threshold/$L_{\min}$ define the result** — a poorly chosen silence threshold changes SLR sign; pre-register defaults and report the $\{-50,-60,-70\}$ dB sweep.
- **Energy-weighted must keep silent-chunk mass nonzero** — fully starving the model of silence is the `drop-silent` arm; the `energy-weighted` arm must *down-weight*, not exclude, or (a) and (b) collapse.
- **Test-set silent-region identification is itself a small pipeline** — needs unit tests (GT-based, deterministic); a buggy $R_{\text{sil}}$ invalidates SLR.
- **SLR excludes empty-region tracks** — report the valid-$n$; some MUSDB tracks may have near-continuous vocals.
