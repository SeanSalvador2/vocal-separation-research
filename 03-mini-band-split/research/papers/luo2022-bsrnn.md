# Music Source Separation with Band-split RNN (BSRNN)

Yi Luo, Jianwei Yu. IEEE/ACM TASLP 31:1893–1901 (2023) / arXiv **2209.15174** (30 Sep 2022). | Verified: HF paper_search (title/authors/date/abstract) + WebSearch (semi-supervised detail, TASLP venue), 2026-07-13. **CORE** for Direction 03 (the band-split idea) and cross-referenced by Directions 08/10 (source-activity detector, pseudo-labels).

## 1. Problem & context
Prior MSS architectures were borrowed from speech/other fields and ignored **music-specific structure**. BSRNN's thesis: a target instrument's energy occupies characteristic frequency sub-bands, so **explicitly splitting the spectrogram into hand-chosen subbands** and modeling them with dedicated capacity beats a uniform front-end. This is the exact idea Direction 03 asks about at 5–10 M params.

## 2. Method — the math
Input: complex STFT $X\in\mathbb C^{F\times T}$ of the mixture.

**(a) Band split.** Partition the $F$ frequency bins into $K$ **non-overlapping subbands** with **hand-designed, non-uniform bandwidths** (narrow where the target's energy concentrates, wide elsewhere — a per-target design choice, "expert knowledge"). Subband $k$ has $F_k$ bins, $\sum_k F_k=F$. Each subband's real+imag features (dim $2F_k$) pass through a per-band **fully-connected + norm** to a common feature dim $N$:
$$Z_k \in \mathbb R^{N\times T},\quad k=1..K \;\Rightarrow\; Z\in\mathbb R^{N\times K\times T}.$$

**(b) Interleaved band- and sequence-level modeling (dual-path).** Alternate two BLSTM stacks with residual connections:
- **Sequence-level RNN** across time $T$, applied per band (models temporal dynamics within a band);
- **Band-level RNN** across the $K$ bands, applied per time frame (models cross-band dependencies).

This is the "band-split" analogue of dual-path RNN (`intra`/`inter` chunks) but over (time, band) instead of (intra-chunk, inter-chunk).

**(c) Mask estimation.** Per band, an MLP (with a GLU-style output) predicts a **complex mask** $M_k\in\mathbb C^{F_k\times T}$ for that subband; concatenate across bands → full complex mask $M\in\mathbb C^{F\times T}$; estimate $\hat S = M\odot X$ (complex product → recovers phase). Loss combines L1 on the complex STFT (real+imag) and on magnitude [exact loss weighting = training knowledge; the complex-mask + frequency-domain L1 structure is confirmed by the "multi-band mask estimation" abstract description].

## 3. Semi-supervised fine-tuning (the pseudo-label pipeline — Directions 08/10 cross-ref)
BSRNN adds a semi-supervised stage using **unlabeled songs** (WebSearch-confirmed mechanism):
- Use a **strong pre-trained BSRNN as its *own* source-activity detector**: run it on unlabeled tracks and keep only **segments where the target source is detected active** (energy of the separated target above a threshold) — this "digs valid segments" and avoids training on segments where the target is absent.
- Use the same pre-trained model as a **pseudo-label generator**: its separated output on the unlabeled segment is the training target.
- Fine-tune on these pseudo-labeled active segments (as data augmentation), improving **all four** instrument tracks. The paper notes this bypasses a *separately trained* activity detector by reusing the model itself.

## 4. Key results (exact, with convention)
- **BSRNN trained on MUSDB18-HQ only significantly outperforms top MDX-2021 models**; semi-supervised fine-tuning improves all four stems. CONFIRMED (abstract).
- **Vocals cSDR ≈ 10.01 dB** (MUSDB18-HQ) — this specific number is **from DTTNet's comparison table** (`chen2023-dttnet.md`), which reports "10.01 dB reported for BSRNN"; consistent with RESEARCH_NOTES' "~10.0 dB cSDR vocals." The BSRNN abstract itself does not quote the single number — flagged.
- Metric is **cSDR** (chunk SDR, MDX convention), *not* museval median-SDR nor SI-SDR — do not cross-compare.

## 5. Relevance to this project (Direction 03)
- **This is the architecture whose core idea Direction 03 miniaturizes.** BSRNN establishes: (i) band-splitting with *dedicated per-band capacity* helps; (ii) the *bandwidth design* is a deliberate, per-target choice (the knob Mel-RoFormer later questions).
- **What we replicate:** a band-split front-end (separate sub-encoders per band, merged in the bottleneck) grafted onto the compact SingNet U-Net, at matched ~5–10 M params.
- **What we extend:** BSRNN is large and MUSDB-HQ-scale; **does the band-split advantage survive at 5–10 M params, and is it the *split* or the *mel spacing* that matters** (Direction 03's 3-way control: baseline / mel-split / uniform-split).
- **Cross-refs:** the source-activity-detector idea grounds Direction 08's activity-aware sampling and Direction 10's pseudo-label option.

## 6. Verification notes
- Title/authors/venue/arXiv ID + abstract claims (band-split, interleaved band/sequence RNN, expert-knowledge bandwidths, semi-supervised fine-tuning beats MDX-2021 top models): CONFIRMED (HF abstract).
- Semi-supervised mechanism (model-as-own-activity-detector + pseudo-label generator, keep active segments): CONFIRMED (WebSearch).
- **Vocals 10.01 dB cSDR: CONFIRMED via DTTNet cross-ref, not BSRNN's own abstract** — flagged.
- Exact band scheme and loss weighting: `[training knowledge]` — architecture structure confirmed, numeric band edges/loss coefficients not read from source.
</content>
