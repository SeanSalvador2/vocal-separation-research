# Continual Learning for Singing Voice Separation with Human in the Loop Adaptation

arXiv **2512.02432** (Dec 2025). | Verified: WebSearch (arXiv abs/html — title/scope) — not on HF paper_search. Verified 2026-07-13. **Context** for Direction 05 (domain-adaptation motivation); also cross-refs Direction 08 (false-positive/leakage feedback).

> **Existence/date note:** arXiv ID 2512.02432 places this in **December 2025**. A WebSearch summary attached a "26th Symposium… 2021" venue line that is inconsistent with the ID and is **not relied upon** — only the arXiv record (title/scope) is treated as verified. If a future check finds no such record, downgrade to UNVERIFIABLE.

## 1. Problem & context
Deployed separators meet songs whose genre/instrumentation differ from training. This paper proposes an **interactive continual-learning** framework that lets a user **adapt** a singing-voice separator to new target songs with minimal effort/expertise — signaling active interest in *adapting separators to specific content*, which is the practical motivation behind Direction 05's domain-shift framing.

## 2. Method
- **U-Net base** producing a spectrogram mask for vocals.
- **Human-in-the-loop step:** the user provides feedback by **marking false positives** (regions where the model wrongly output vocal energy); the model fine-tunes on this feedback.
- Continual: adaptation from a few corrected songs.

## 3. Key results
- Adapting via the human-in-the-loop feedback **improves performance not only on the corrected songs but also on new, previously unseen songs**. CONFIRMED scope (WebSearch). (Qualitative; no headline dB captured.)

## 4. Limitations & caveats
- Requires human interaction (marking false positives) — heavier than an automatic adapter; and it is a *personalization/continual* framing, not a PEFT parameter-efficiency study.
- Details (exact adaptation objective, dataset) not read from the (blocked) PDF.

## 5. Relevance to this project
- **Direction 05:** evidence that *adapting a pretrained separator to shifted content is an active, valued problem* — the motivation for testing LoRA as a **cheap, forgetting-resistant** adapter (vs this paper's interactive full-model fine-tune).
- **Direction 08 cross-ref:** the feedback signal is **false positives = vocal energy where there should be none** — i.e. exactly the **silence leakage** Direction 08 defines a metric for. A separator that leaks less in silence needs fewer such corrections; the two directions reinforce each other.

## 6. Verification notes
- arXiv ID + title + scope (U-Net mask, human marks false positives, continual, generalizes to unseen songs): CONFIRMED (WebSearch of abs/html). The mismatched "2021 symposium" venue string is explicitly **not** trusted (see date note).
