# Distilling the Knowledge in a Neural Network

Geoffrey Hinton, Oriol Vinyals, Jeff Dean. NeurIPS 2014 Deep Learning Workshop / arXiv **1503.02531** (2015). | Verified: WebSearch (arXiv abs, method, temperature) — not surfaced by HF paper_search. Verified 2026-07-13. **Context/method** for Direction 10 — the distillation principle, with the caveat that MSS is regression, not classification.

## 1. Problem & context
A large "teacher" (or ensemble) encodes knowledge a small "student" can inherit by training on the teacher's **soft outputs** rather than hard labels. The soft outputs carry **dark knowledge** — the relative structure among non-target responses — which is a richer signal than one-hot labels. Direction 10 applies this idea to close the SingNet→Demucs gap using a Demucs teacher on unlabeled audio.

## 2. Method — the math (classification form)
Teacher logits $z_i$ → softened probabilities via **temperature $T$**:
$$p_i(T) = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}.$$
Higher $T$ softens the distribution, exposing the teacher's relative confidences ("dark knowledge"). The student minimizes a weighted sum of the **distillation loss** (cross-entropy to the teacher's soft targets at temperature $T$, scaled by $T^2$) and the **student loss** (cross-entropy to true hard labels, when available):
$$\mathcal L = (1-\lambda)\,\mathcal L_{\text{hard}} + \lambda\,T^2\,\mathcal L_{\text{soft}}(p^{\text{student}}(T),\,p^{\text{teacher}}(T)).$$
Experiments used $T\in[1,20]$; **lower $T$ works better when the student is very small** relative to the teacher.

## 3. Key results
- A distilled student recovers much of an ensemble's/large model's accuracy at a fraction of the cost; dark knowledge transfers the teacher's generalization. CONFIRMED (WebSearch).

## 4. Limitations & caveats (critical for Direction 10)
- **KD is defined for classification** (softmax + temperature). **MSS is dense regression** — the teacher's output is a **separated waveform/spectrogram**, not a probability vector; **there is no softmax to temperature-soften.** The temperature mechanism does *not* transfer directly. Direction 10 must adopt a **regression analogue** (see §5).
- Distillation transfers the teacher's **errors** as faithfully as its knowledge — a central Direction-10 risk (HT-Demucs's vocal↔other bleeding).

## 5. Relevance to this project (Direction 10 — the regression analogue)
Since there is no softmax, the "soft target" is simply the **teacher's continuous separated output**, which is already softer/richer than the (absent) ground truth on unlabeled data. Candidate objectives and the temperature analogue are developed in `../LITERATURE.md §5`; in brief:
- **soft target = teacher's separated vocal** (waveform or magnitude), used as the regression target on unlabeled FMA;
- **temperature analogue** = spectral/dynamic-range compression (log or power-law on magnitudes) that de-emphasizes loud peaks and exposes low-energy structure — the closest regression cousin to softmax temperature (mark as **analogy, not identity**);
- combine with GT loss (Hinton's $\lambda$ mixing) on the labeled MUSDB portion.

## 6. Verification notes
- Title/authors/arXiv ID + temperature/dark-knowledge/$T\in[1,20]$/small-student-lower-T: CONFIRMED (WebSearch).
- The softmax-temperature equations are the standard KD formulation; the **regression caveat and analogue are OUR framing** (clearly marked) — Hinton's method itself is classification-only.
</content>
