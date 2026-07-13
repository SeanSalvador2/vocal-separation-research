# Direction 06 — Robust training: how much do label noise and bleeding hurt a small separator?

Deep-dives: [`papers/fabbro2023-sdx23.md`](papers/fabbro2023-sdx23.md) (CORE),
[`papers/noisy-label-canon.md`](papers/noisy-label-canon.md) (the mitigation lineage).

## 1. Direction recap (from RESEARCH_DIRECTIONS #6)
- **Hypothesis:** simulated stem **bleeding** (mixing $\varepsilon$ of accompaniment into the vocal *target*) degrades SingNet's vocals SI-SDR **roughly linearly in $\varepsilon$**, and a simple **loss-side mitigation** (discard the top-$k\%$ highest-loss chunks per epoch — a noisy-label heuristic) **recovers a measurable fraction** of the loss.
- **Minimal experiment:** corrupt MUSDB training targets at $\varepsilon\in\{0,0.05,0.15,0.30\}$ (built from the clean stems we already have — no new data); train at each level; re-train the worst level with the mitigation; **1 seed per level + 3 at the endpoints**.
- **Falsification / success test:** a **degradation curve (SI-SDR vs $\varepsilon$)** + a **mitigation delta**. Supported iff degradation is monotone/roughly linear *and* the trimmed-loss defense recovers a measurable, CI-separated fraction at the worst $\varepsilon$. (A *non-linear* curve or a null mitigation are still reportable — "the curve cannot fail to exist.")
- Difficulty/Risk: **Medium / Low.**

## 2. Paper map
| Paper | Role | Specific claim we rely on | Verified | Deep-dive |
|---|---|---|---|---|
| SDX'23 (2308.06979) | **replicate (concept)** | robust-MSS task; Bleeding = cross-stem partial content; +1.6 dB over MDX'21 | CONFIRMED | [papers/fabbro2023-sdx23.md](papers/fabbro2023-sdx23.md) |
| ITLM (1810.11874) | **method (mitigation)** | iteratively train on lowest-loss subset; provable recovery | CONFIRMED | [papers/noisy-label-canon.md](papers/noisy-label-canon.md) |
| Co-teaching (1804.06872) | **method (mitigation)** | small-loss selection is noise-robust | CONFIRMED | [noisy-label-canon.md](papers/noisy-label-canon.md) |
| GCE (1805.07836) | **method (alt)** | robust loss between CCE and MAE | CONFIRMED | [noisy-label-canon.md](papers/noisy-label-canon.md) |
| Arpit 2017 (1706.05394) | **foundation** | DNNs learn clean patterns before memorizing noise | CONFIRMED | [noisy-label-canon.md](papers/noisy-label-canon.md) |
| TFC-TDF-UNet v3 (2306.09382) | **precedent (MSS)** | SDX'23-winning loss-masking for noise-robust MSS | CONFIRMED | [noisy-label-canon.md](papers/noisy-label-canon.md) |
| Self-refining labels (2307.12576) | **context (MSS)** | label refinement recovers clean-label quality | CONFIRMED | [noisy-label-canon.md](papers/noisy-label-canon.md) |
| Blind data cleaning (2510.15409) | **recent related** | data cleaning for MSS (2025) | title/existence (WS) | [noisy-label-canon.md](papers/noisy-label-canon.md) |
| MUSDB18 | **infrastructure** | clean stems we corrupt | CONFIRMED | [shared](../../00-shared-research/papers/musdb18-dataset.md) |

## 3. Replicate-vs-extend
- **Replicate:** SDX'23's *bleeding* corruption concept and the small-loss/trimmed-loss mitigation principle (ITLM/Co-teaching).
- **Extend (our delta):** SDX'23 studied robustness at SOTA scale with gated datasets; **nobody reports a controlled degradation curve (SI-SDR vs $\varepsilon$) + a cheap loss-side mitigation for a *compact* model on MUSDB-only**. Direction 06 does, at notebook budget, from MUSDB's own stems.

## 4. The gap
- SDX'23 *created the task* but the community results are at large scale on the gated LabelNoise/Bleeding sets. The **compact-model, single-parameter, curve-plus-cheap-defense** study is unfilled. **Recent related work to acknowledge:** *Towards Blind Data Cleaning: A Case Study in MSS* (2510.15409, Oct 2025) and *Self-refining Pseudo Labels* (2307.12576) address *label noise / cleaning*, not a **controlled bleed degradation curve for a compact model** — so novelty stands, but these must be cited. (No dedicated ε-bleed-curve paper surfaced.)

## 5. Direction-specific technical notes (for MASTER_PLAN / THEORY.md)
**Formal ε-bleed corruption model (OUR construction).** With clean vocal $v$, accompaniment $a=$ drums+bass+other, mixture $x=v+a$:
$$\boxed{\ \tilde v \;=\; v \;+\; \varepsilon\,a,\qquad \varepsilon\in\{0,0.05,0.15,0.30\}\ }$$
- **Only the target is corrupted; $x$ is untouched.** The model learns $x\mapsto \tilde v$, i.e. to *leave $\varepsilon a$ in its vocal estimate* — a continuous-target label noise. (Optionally gain-normalize $\tilde v$ to keep loudness fixed so $\varepsilon$ isn't confounded with level; document the choice.)
- **Relation to SDX'23:** a controlled simplification of `SDXDB23_Bleeding` (constant $\varepsilon$, accompaniment→vocal only, vs SDX'23's realistic per-song all-directions bleed). Trades realism for a clean monotone knob.
- **Expected geometry:** a perfectly-fit model outputs $\hat v=\tilde v=v+\varepsilon a$; the residual vs clean $v$ is $\varepsilon a$, so the *achievable* SI-SDR is bounded by $10\log_{10}(\lVert v\rVert^2/\lVert\varepsilon a\rVert^2)$ → SI-SDR degrades ~$-20\log_{10}\varepsilon$ + const, i.e. **monotone in $\varepsilon$** (roughly linear in dB over the tested range) — the hypothesis's mechanism, derivable in closed form (good THEORY.md content).

**Trimmed-loss mitigation (ITLM instantiated).** Per epoch, with per-chunk losses $\ell_i$ and keep-fraction $\alpha=1-k$:
$$\text{train on } S=\{\text{chunks with the }\lceil\alpha N\rceil\text{ smallest }\ell_i\},\quad k\in\{5,10,20\}\%.$$
Chunks whose target carries large $\varepsilon a$ have persistently high loss (Arpit: after the model has learned clean structure) → trimmed. Sweep $k$; report recovered-dB vs $k$ at the worst $\varepsilon$. **GCE-style robust regression loss** is the alternative mitigation if trimming underperforms.

## 6. Risks this literature implies
- **ε confounded with loudness** — corrupting $\tilde v=v+\varepsilon a$ changes target energy; normalize or the curve conflates bleed with gain. State the normalization.
- **Trimming can discard hard-but-clean chunks** (dense mixes, quiet vocals) — over-trimming hurts on clean data too; sweep $k$ and report the clean-data cost of trimming.
- **Classification→regression transfer** — canon A–D is classification; the "high-loss = corrupted" premise is weaker for regression where legitimately hard chunks also have high loss. ITLM (general estimator) is the safest citation; frame results as an empirical test of the transfer.
- **Small ε may be within noise** — ε=0.05 might not separate from ε=0 at this scale; the 3-seed endpoints and CIs are essential to claim a curve.
