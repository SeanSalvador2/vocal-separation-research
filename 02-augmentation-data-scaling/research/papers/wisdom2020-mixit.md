# Unsupervised Sound Separation Using Mixture Invariant Training (MixIT)

Scott Wisdom, Efthymios Tzinis, Hakan Erdogan, Ron J. Weiss, Kevin Wilson, John R. Hershey (Google). NeurIPS 2020 / arXiv **2006.12701** (24 Oct 2020). | Verified: HF paper_search (title/authors/date/abstract), 2026-07-13. **Context** paper for Direction 02 (the "data is the binding constraint" thesis).

## 1. Problem & context
Supervised separation needs isolated ground-truth sources — expensive and distribution-mismatched to real audio. MixIT is a **fully unsupervised** objective needing only **single-channel acoustic mixtures**, enabling training/adaptation on in-the-wild data. It is the ancestor of the MSS-specific method Direction 02 actually leans on (Saijo & Bando 2025).

## 2. Method — the math
Take two independent mixtures $x_1, x_2$ (each an unknown sum of sources) and form a **mixture of mixtures** $\bar x = x_1 + x_2$. A network separates $\bar x$ into $M$ estimated sources $\hat{\mathbf s}=[\hat s_1,\dots,\hat s_M]$. MixIT then finds the assignment of estimates back to the two original mixtures that best reconstructs them:
$$\mathcal L_{\text{MixIT}} = \min_{A\in\mathcal A}\;\Big[\,\ell\big(x_1,\ [A\hat{\mathbf s}]_1\big) + \ell\big(x_2,\ [A\hat{\mathbf s}]_2\big)\Big],$$
where $A\in\{0,1\}^{2\times M}$ is constrained so **each estimated source is assigned to exactly one mixture** (columns of $A$ sum to 1), and $\ell$ is typically negative SNR. The min over $A$ is the "mixture-invariant" permutation-style search. No isolated sources are ever required.

Semi-supervised use: mix labeled synthetic data (with PIT) and unlabeled real data (with MixIT) → unsupervised domain adaptation.

## 3. Key results
- Competitive with supervised methods on **speech** separation; improves reverberant speech separation, trains a speech enhancer from noisy mixtures, and improves **universal sound separation** by adding in-the-wild data. CONFIRMED (abstract).
- **Not an MSS/music result** — music comes later (Saijo & Bando 2025).

## 4. Limitations & caveats
- **Over-separation:** with $M$ larger than the true source count, MixIT can split a source across output channels.
- **Source-independence assumption:** the mixture-of-mixtures trick implicitly treats sources as independent — the exact property music (correlated stems: bass/drums/harmony) violates, which is why MixIT was long assumed unsuitable for MSS (the assumption Saijo & Bando challenge).

## 5. Relevance to this project (Direction 02)
- **Framing, not method:** MixIT is the evidence that *unlabeled data can substitute for labels*, supporting Direction 02's core thesis that MUSDB-only training is **data-starved** and that the data-scaling curve (val SI-SDR vs #songs) is the real story.
- Direction 02 does **not** implement MixIT (it stays supervised on MUSDB with augmentation); MixIT is the "what if we had more data" ceiling the scaling curve gestures toward, and the mechanism Direction 10 (distillation) and the MSS-MixIT paper operationalize.

## 6. Verification notes
- Title/authors/venue/arXiv ID + scope (unsupervised, mixture-of-mixtures, semi-supervised adaptation, speech/universal): CONFIRMED (HF abstract).
- MixIT objective + constraint on $A$: standard formulation consistent with the abstract's "separate… such that separated sources can be remixed to approximate the original mixtures" [equation reproduced from method knowledge; abstract confirms the mechanism].
