# Is MixIT Really Unsuitable for Correlated Sources? Exploring MixIT for Unsupervised Pre-training in Music Source Separation

Kohei Saijo, Yoshiaki Bando. arXiv **2505.07631** (12 May 2025). | Verified: HF paper_search (title/authors/date/abstract), 2026-07-13. **CORE** for Direction 02's "data is the constraint" claim; also cited by Direction 10.

## 1. Problem & context
MixIT (`wisdom2020-mixit.md`) was assumed unsuitable for MSS because its mixture-of-mixtures trick presumes source independence, while music stems are correlated. Saijo & Bando **reframe the failure**: the hard part of MSS is not inter-source correlation but the **ill-posed, application-dependent definition of a "stem"** — the model doesn't know *what* to separate. Under that reframing, MixIT is worth revisiting as an **unsupervised pre-training** signal on unlabeled music. This is the most direct published evidence that, at MUSDB scale, **data — not architecture — is the binding constraint** (Direction 02's thesis (b)).

## 2. Method
- **Stage 1 (pre-train):** train a separator on **in-the-wild, unlabeled audio from the Free Music Archive (FMA)** using the **MixIT** objective (no isolated stems). This teaches general instrument-separation structure from cheap, license-safe data (FMA licensing: `../../10-demucs-distillation/research/papers/fma-dataset.md`).
- **Stage 2 (fine-tune):** fine-tune on **MUSDB18 with supervision** (standard stem targets).
- **Backbone:** the **band-split TF-Locoformer**, a state-of-the-art MSS model — so the gain is measured on a strong, modern architecture, not a toy.

## 3. Key results
- **MixIT-based pre-training on FMA improves MSS performance over training from scratch** on MUSDB18. CONFIRMED (abstract). (The abstract does not quote a single headline dB delta; the qualitative result — pre-training helps — is the verified claim.)
- Preliminary experiments show MixIT "can still separate instruments to some extent," refuting the blanket "MixIT is unsuitable for correlated sources."

## 4. Limitations & caveats
- Uses a heavy SOTA backbone (TF-Locoformer) and full FMA-scale pre-training — **not** a compact-model, notebook-budget recipe. Direction 02 borrows the *thesis* (unlabeled music helps; data is the constraint), not the exact pipeline.
- No exact per-dB scaling curve is published — leaving the "how does quality scale with #songs?" question (Direction 02's (b)) genuinely open.

## 5. Relevance to this project (Direction 02)
- **Motivates thesis (b):** if unsupervised FMA pre-training measurably helps a SOTA model, then MUSDB's 86 training songs are on the steep part of the data curve — so Direction 02's scaling figure (val SI-SDR vs {21,43,64,86} songs) should be **rising, not saturated**. Confirming the curve has not plateaued is the concrete, quantified version of this paper's qualitative claim, at small scale.
- **Bridges to Direction 10:** the same "unlabeled FMA → pre-train/augment → fine-tune on MUSDB" structure is what Direction 10 does with *distillation* (soft targets from a Demucs teacher) instead of MixIT (self-supervised remixing). Direction 02 = "how far does data alone go"; Direction 10 = "borrow a teacher's knowledge on that data."
- **What we ignore:** the MixIT objective itself and TF-Locoformer — out of scope; cited as the data-centric ceiling.

## 6. Verification notes
- Title/authors/date/arXiv ID + method (FMA MixIT pre-train → MUSDB fine-tune, band-split TF-Locoformer, improves over scratch): CONFIRMED verbatim from HF paper_search abstract.
- No fabricated dB numbers — the abstract reports a qualitative improvement only; recorded as such.
</content>
