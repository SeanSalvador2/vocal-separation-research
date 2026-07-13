# Noisy-label canon — the mitigation lineage for Direction 06

The theory behind Direction 06's loss-side defense ("discard top-k% highest-loss chunks per epoch").
Four classic papers + the MSS-specific noisy-label work. All verified 2026-07-13 (WebSearch; not on HF paper_search).

## A. Why loss can separate clean from corrupted — the memorization effect
**Arpit et al., "A Closer Look at Memorization in Deep Networks," ICML 2017, arXiv 1706.05394** (Arpit, Jastrzębski, Ballas, Krueger, E. Bengio, Kanwal, Maharaj, Fischer, Courville, Y. Bengio, Lacoste-Julien). CONFIRMED.
- Finding: **DNNs learn simple, general patterns *first* and memorize noise *later*.** Consequently, **early in training, high per-example loss ≈ likely-corrupted** and low loss ≈ likely-clean. This is the assumption every small-loss/trimmed-loss method rests on, including Direction 06's.

## B. The trimmed-loss estimator (Direction 06's exact mechanism)
**Shen & Sanghavi, "Learning with Bad Training Data via Iterative Trimmed Loss Minimization" (ITLM), ICML 2019 (PMLR v97:5739–5748), arXiv 1810.11874.** CONFIRMED (arXiv ID + PMLR).
- **Method:** alternate between (i) **selecting the fraction $\alpha$ of samples with the lowest current loss** and (ii) **retraining on only those**. Formally, with per-sample losses $\ell_i(\theta)$ and keep-fraction $\alpha$:
  $$S_t = \{\text{indices of the }\lceil\alpha N\rceil\text{ smallest }\ell_i(\theta_t)\},\qquad \theta_{t+1}=\arg\min_\theta \sum_{i\in S_t}\ell_i(\theta).$$
  The top $(1-\alpha)N$ highest-loss samples are **trimmed** each round.
- **Guarantee:** provably recovers the ground truth with **linear convergence** in generalized linear models under standard assumptions; empirically effective on deep classifiers with label errors, GANs with bad images, and backdoor attacks.
- **Direction 06 mapping:** per epoch, drop the top-$k\%$ highest-loss **chunks** (keep-fraction $\alpha = 1-k$) and train on the rest — ITLM with the chunk as the unit. Heavily-bled target chunks ($\tilde v = v+\varepsilon a$ with large residual $\varepsilon a$) have persistently high loss → trimmed away.

## C. Co-teaching — small-loss selection with two networks
**Han et al., "Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels," NeurIPS 2018, arXiv 1804.06872.** CONFIRMED.
- Two networks train simultaneously; each **selects the small-loss instances** in every mini-batch (the $R(t)$ fraction with lowest loss = "probably clean") and **feeds them to its peer** for the update. $R(t)$ **decreases over training** (trim more as memorization sets in — following Arpit's schedule). Cross-teaching prevents a single net's confirmation bias.
- **Direction 06 relevance:** the single-network trimmed-loss variant (ITLM) is the cheap version we run; Co-teaching is the citation for *why the small-loss trick is robust* and the natural extension if the single-net defense underperforms.

## D. Robust loss functions (an alternative axis)
**Zhang & Sabuncu, "Generalized Cross Entropy Loss for Training DNNs with Noisy Labels," NeurIPS 2018, arXiv 1805.07836.** CONFIRMED.
- $\mathcal L_q(f(x),y) = \frac{1 - f_y(x)^q}{q}$ interpolates **CCE ($q\to0$)** and **MAE ($q=1$)**; a **noise-robust generalization** of both (MAE is robust but trains poorly; GCE keeps robustness with better optimization).
- **Direction 06 relevance:** a *loss-shape* mitigation (make the loss itself robust) as an alternative/complement to *sample-trimming*. For MSS regression the analogue is a robust regression loss (e.g. down-weighting high-residual chunks) — cited as the second mitigation family if the trimmed-loss defense is chosen for the headline.

## E. MSS-specific noisy-label work (the closest prior)
- **Kim, Lee, Jung, "SDX'23 MDX Technical Report: TFC-TDF-UNet v3," arXiv 2306.09382.** CONFIRMED (HF). The **SDX'23-winning** noise-robust system uses a **loss-masking** approach for noise-robust training — the direct MSS precedent for a loss-side defense.
- **Koo, Chae, Jeon, Lee, "Self-refining of Pseudo Labels for MSS with Noisy Labeled Data," arXiv 2307.12576.** CONFIRMED (HF). Automated **label refinement** on partially mislabeled MSS data; models trained on the self-refined set match clean-label training. (Addresses SDX'23-style *label noise*, complementary to Direction 06's *bleed*.)
- **"Towards Blind Data Cleaning: A Case Study in MSS," arXiv 2510.15409 (Oct 2025).** Surfaced in gap-check; a **mid-2025 data-cleaning-for-MSS** paper — recent related work Direction 06 must acknowledge (it post-dates the base plan).
- **Cross-cut: "Why LoRA Resists Label Noise," arXiv 2602.00084 (Feb 2026)** — LoRA as inherently noise-robust; ties Direction 06 to Direction 05.

## Domain-transfer caveat (state in the report)
Canon A–D is from **classification** (discrete label noise). Direction 06's task is **regression** with a **continuous, corrupted target** ($\tilde v=v+\varepsilon a$). The transferable principle is "**persistently high-loss chunks are likely-corrupted**," which holds for per-chunk regression loss; ITLM (a general trimmed-loss estimator, proven beyond classification) is the most directly applicable. The novelty is precisely that **nobody has run this for compact MSS with a controlled ε-bleed curve** — see `../LITERATURE.md §4`.

## Verification notes
- All four classic IDs (1706.05394, 1810.11874, 1804.06872, 1805.07836) + venues + methods: CONFIRMED (WebSearch: arXiv abs + PMLR/NeurIPS proceedings).
- MSS-specific (2306.09382, 2307.12576 CONFIRMED via HF; 2510.15409, 2602.00084 via WebSearch gap-checks).
- ITLM/Co-teaching/GCE equations reproduced from the standard formulations; abstracts/summaries confirm the mechanisms.
