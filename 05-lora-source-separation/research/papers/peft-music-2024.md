# Parameter-Efficient Transfer Learning for Music Foundation Models

Yiwei Ding et al. arXiv **2411.19371** (28 Nov 2024). | Verified: WebSearch (arXiv abs/html, method, numbers) — not on HF paper_search. Verified 2026-07-13. **Context** for Direction 05 — the "PEFT works in the *music* domain" evidence, and the precise scope boundary that leaves MSS open.

> **Title correction (logged):** our docs call it "PEFT for music foundation models." The actual title is **"Parameter-Efficient *Transfer Learning* for Music Foundation Models"** (PETL). Authors led by Yiwei Ding.

## 1. Problem & context
Adapting a music foundation model (e.g. MusicFM) to a downstream task is usually done by **probing** (frozen backbone → suboptimal) or **fine-tuning** (expensive, overfits). This paper systematically tests **parameter-efficient transfer learning** (PETL) instead — the same idea Direction 05 applies, but on **classification/tagging** tasks, not separation.

## 2. Method
Three PETL families evaluated on frozen music foundation models: **adapter-based**, **prompt-based**, and **reparameterization-based (LoRA)**. Only a small number of parameters train; downstream tasks include music tagging (MTG-Jamendo) and other MIR classification/regression tasks.

## 3. Key results (exact)
- On the **large** MTG-Jamendo dataset, fine-tuning MusicFM beats some PETL methods, but **Adapter and LoRA still show superior performance**, at **0.36%** (Adapter) and **0.22%** (LoRA) trainable parameters, with **3×** and **2.5×** training speedups vs fine-tuning. CONFIRMED (WebSearch).
- Take-away: **adapters/LoRA reach ≥ full-fine-tune quality at <1% trainable params** on music foundation-model *downstream* tasks — matching RESEARCH_DIRECTIONS' "<1% trainable params" claim.

## 4. Limitations & caveats
- **Scope is classification/tagging on foundation-model *embeddings*, not source separation.** MSS is a dense regression / mask-prediction task with a different architecture (BiLSTM/U-Net, not a frozen SSL backbone). The paper's success does **not** transfer automatically — it establishes that PEFT *can* work in music, leaving "does LoRA work for MSS models?" genuinely open.

## 5. Relevance to this project (Direction 05)
- **The strongest published "PEFT works in music" data point** — supports the plausibility half of Direction 05's hypothesis (LoRA recovers most of the fine-tune gain at <5% params).
- **Defines the gap precisely:** it does PEFT for music *tagging*, not *separation* — so Direction 05's "first careful small-scale LoRA-for-MSS" framing is honest (not "first PEFT in music," but "first LoRA-for-MSS").
- Its 0.36%/0.22% figures are the reference the Direction-05 "quality-vs-trainable-params" curve is benchmarked against qualitatively (different task, so not a numeric target).

## 6. Verification notes
- arXiv ID + method (adapter/prompt/LoRA on music foundation models) + numbers (0.36%/0.22% params, 3×/2.5× speedup, MTG-Jamendo): CONFIRMED (WebSearch of abs + html).
- **Exact title** ("…Transfer Learning…") pinned; RESEARCH_NOTES/DIRECTIONS paraphrase noted (not a wrong citation — arXiv ID is correct).
</content>
