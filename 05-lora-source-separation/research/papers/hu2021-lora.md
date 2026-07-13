# LoRA: Low-Rank Adaptation of Large Language Models

Edward Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (Microsoft). ICLR 2022 / arXiv **2106.09685** (17 Jun 2021). | Verified: WebSearch (arXiv abs + HF paper page, author list, core claims) — not surfaced by HF paper_search (returns newer LoRA variants). Verified 2026-07-13. **CORE** for Direction 05 — the method, with the math the MASTER_PLAN needs to wrap UMX.

## 1. Problem & context
Full fine-tuning updates all $|\Phi|$ parameters of a pretrained model — a fresh copy per task, and for large models, infeasible. LoRA freezes the pretrained weights and learns a **low-rank additive update** per weight matrix, exploiting the empirical finding that the *change* needed to adapt has **low intrinsic rank**. This is the single most job-relevant fine-tuning technique of the decade, and Direction 05 is the first careful small-scale application to **music source separation**.

## 2. Method — the math (full)
For a pretrained linear map with weight $W_0\in\mathbb R^{d\times k}$ and input $x\in\mathbb R^{k}$, the fine-tuned weight is $W = W_0 + \Delta W$. LoRA constrains $\Delta W$ to **rank $r\ll\min(d,k)$**:
$$\Delta W = B A,\qquad B\in\mathbb R^{d\times r},\; A\in\mathbb R^{r\times k},$$
so the adapted forward pass is
$$\boxed{\,h = W_0 x + \Delta W x = W_0 x + \tfrac{\alpha}{r}\,B A x\,}$$
where $\alpha$ is a constant scaling (tuning $\alpha$ ≈ tuning a learning rate; the $\alpha/r$ factor keeps scale roughly constant as $r$ varies).

**Initialization.** $A\sim\mathcal N(0,\sigma^2)$, **$B=0$**, so $\Delta W = 0$ at the start of training — adaptation begins exactly at the pretrained function (no shock).

**What trains.** Only $A,B$ (the frozen $W_0$ receives no gradient). **Trainable parameters per wrapped matrix:**
$$\#\text{LoRA}(d,k,r) = r\,(d+k)\qquad\text{vs. full } d\,k.$$
For $d=k=512, r=4$: $4\cdot1024=4096$ vs $262{,}144$ — a **64× reduction** for that matrix.

**Zero inference latency.** After training, merge $W = W_0 + \tfrac{\alpha}{r}BA$ into a single matrix → identical inference cost to the original (unlike adapters, which add layers). Swapping tasks = swapping the merged delta.

**Where applied.** In the paper, LoRA wraps the **Transformer self-attention projection matrices** ($W_q,W_k,W_v,W_o$); ablations show adapting $W_q,W_v$ is often enough. The principle generalizes to *any* dense/linear weight — including an LSTM's input/recurrent projections (the Direction-05 target; see §5).

## 3. Key results (exact)
- vs **GPT-3 175B** full fine-tuning: **10,000× fewer trainable parameters** and **3× less GPU memory**, with **on-par or better** quality on RoBERTa, DeBERTa, GPT-2, GPT-3. CONFIRMED (WebSearch).
- No additional inference latency (merge property).

## 4. Limitations & caveats
- Gains assume the adaptation is genuinely low-rank; for a large *domain shift* a small $r$ may under-fit (Direction 05 sweeps $r\in\{4,16\}$).
- LoRA on **recurrent** layers (LSTM) is less standard than on attention — the input-to-hidden and hidden-to-hidden projections must be wrapped explicitly; this is Direction 05's main engineering risk.

## 5. Relevance to this project (Direction 05 — the exact plumbing)
UMX (`../../../00-shared-research/papers/openunmix2019.md`) is the host. LoRA-wrappable weight matrices and their per-matrix trainable-param cost $r(d_{out}+d_{in})$:
- **`fc1`** $512\times(2\cdot\texttt{nb\_bins})$ — a big dense matrix; LoRA here is a textbook linear wrap.
- **3× BiLSTM layers** — wrap `weight_ih` ($4\cdot256\times \text{in}$) and `weight_hh` ($4\cdot256\times256$) **per direction** (the 4 gates are concatenated in `weight_ih/hh`; LoRA on the full gate-stacked matrix is valid). This is the contained-but-nontrivial part.
- **`fc2`** $512\times1024$, **`fc3`** $(2\cdot\texttt{nb\_output\_bins})\times512$.

Direction 05 tests LoRA (rank 4 and 16) against zero-shot, head-only, and full fine-tune on shifted domains (compressed-AAC vs HQ; a genre subset), measuring adapted-domain gain **and** source-domain regression (catastrophic-forgetting resistance). The $B=0$ init means LoRA starts as exactly zero-shot UMX — a clean baseline continuity. Full rank-4/16 budget table is in this direction's LITERATURE.md §5.

## 6. Verification notes
- Title/authors/venue/arXiv ID + core claims (freeze $W_0$; $\Delta W=BA$; 10,000×/3× vs GPT-3; on-par/better): CONFIRMED (WebSearch: arXiv 2106.09685 abs + HF paper page).
- The $h=W_0x+\frac{\alpha}{r}BAx$ form, $A\sim\mathcal N$/$B=0$ init, $r(d+k)$ count, merge-for-zero-latency: standard LoRA formulation, matches the paper [equations reproduced from the well-known method; abstract/summary confirm the low-rank $\Delta W=BA$ mechanism and parameter savings].
