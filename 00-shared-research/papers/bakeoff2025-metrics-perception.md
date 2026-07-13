# Musical Source Separation Bake-Off: Comparing Objective Metrics with Human Perception

Noah Jaffe, John Ashley Burgoyne. WASPAA 2025 (Tahoe City, CA, Oct 12–15 2025). arXiv **2507.06917**. | Verified: WebSearch (arXiv abs + dblp WASPAA 2025), 2026-07-13. (Not surfaced by HF paper_search; WebSearch is the verifying route.)

The most recent (mid-2025) word on whether our metrics mean what we think. Motivates PLAN's listening check and hedges the SDR↔perception gap.

## 1. Problem & context
Post-2019 MSS reports SI-SDR / BSS-Eval SDR as if they proxy perceived quality. The Bake-Off runs a **large listener study on the MUSDB18 test set** (~30 ratings per track across **seven distinct listener groups**) and correlates human ratings against a battery of objective metrics — both legacy energy-ratio metrics (BSS-Eval v4 SDR/ISR/SIR/SAR, SI-SDR variants incl. SI-SAR) and embedding/reference-free metrics (Fréchet Audio Distance with CLAP-LAION-music, EnCodec, VGGish, Wav2Vec2, HuBERT embeddings).

## 2. Method
Per-stem, per-system human ratings vs objective scores; rank correlation (Kendall's $\tau$) reported per stem. The design deliberately spans stems (vocals/drums/bass) because the authors hypothesize — and find — that the best metric is **stem-dependent**.

## 3. Key results (exact, WebSearch-confirmed)
- **Vocals:** BSS-Eval **SDR is the best-performing metric** — it tracks perceived vocal quality better than the alternatives.
- **Drums / bass:** **SI-SAR** (scale-invariant signal-to-artifacts ratio) predicts listener ratings **better than SDR**.
- **FAD** with the **CLAP-LAION-music** embedding is competitive *for drums/bass* (Kendall's $\tau \approx 0.25$ drums, $0.19$ bass) — but **none of the embedding-based metrics (including CLAP) correlate positively with human perception for vocals** (uncorrelated or negatively correlated).
- **Recommendation:** stem-specific evaluation; **no single metric** reliably reflects perceptual quality across all source types.

## 4. Limitations & caveats
- MUSDB18 test only; specific systems; listener panels are of finite size (~30/track). Correlations are rank-order, not calibrated MOS predictions.
- Says nothing about the *karaoke/accompaniment* direction specifically (vocal-bleed salience) — which is exactly the open sub-question Direction 07 (not on this project's 7) would pilot, and which this project's listening check touches qualitatively.

## 5. Relevance to this project
- **Metric policy (all directions):** vindicates using **SDR/SI-SDR as the primary axis for the vocals target** (our headline stem) — for vocals, SDR *is* the best available perceptual proxy. This is a genuine green light for the project's metric choice, not just a hedge.
- **Direction 01 (loss study):** independent confirmation that "optimize the eval metric" is subtle — embedding losses that look attractive can be perceptually anti-correlated for vocals; supports testing MR-STFT as an *artifact* reducer rather than an SDR chaser.
- **Honest reporting / PLAN §1.2, §7 listening check:** the citation that justifies a (small, pre-registered) human check and forbids over-claiming from FAD-style numbers on vocals.

## 6. Verification notes
- Title/authors/venue/arXiv ID: CONFIRMED (WebSearch: arXiv 2507.06917 abs, dblp WASPAA 2025 listing).
- Findings (SDR best for vocals; SI-SAR better for drums/bass; FAD Kendall $\tau$ 0.25/0.19; embeddings not positively correlated for vocals; stem-specific recommendation): CONFIRMED via WebSearch summary of the abstract/paper. These match RESEARCH_NOTES §6 and RESEARCH_DIRECTIONS §1.4 exactly. Authorship (Jaffe & Burgoyne) newly pinned — our docs did not name authors.
</content>
