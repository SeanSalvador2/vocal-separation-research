# Moises-Light: Resource-efficient Band-split U-Net For Music Source Separation

Yun-Ning (Amy) Hung, Igor Pereira, Filip Korzeniowski (Moises/Music.AI). WASPAA 2025 / arXiv **2510.06785** (8 Oct 2025). | Verified: WebSearch (arXiv abs, IEEE Xplore/WASPAA, architecture + param claims) — not on HF paper_search. Verified 2026-07-13. **CORE** for Direction 03: the closest professional analogue to "band-split U-Net at small scale."

## 1. Problem & context
SOTA MSS (BS-RoFormer, SCNet) is accurate but heavy — a problem for on-device/resource-limited deployment, while lightweight models usually lose accuracy. Moises-Light targets the **efficiency frontier**: a **band-split U-Net** that matches much larger systems at a fraction of the parameters. It is the paper that makes "band-split + efficient + competitive" a publishable axis — exactly Direction 03's territory, done professionally.

## 2. Method — the architecture (WebSearch-confirmed)
Built on the **TFC-TDF V3 / DTTNet lineage** (Dual-path TFC-TDF U-Net — see `chen2023-dttnet.md`), a symmetric U-Net of **encoder + decoder + dual-path sequence modeling in the bottleneck**. Moises-Light's specific changes:
- **Band-split into TFC-TDF blocks:** all TDF (Time-Distributed Fully-connected) conv layers in the original TFC-TDF V3 blocks (except skip-connection ones) are **replaced with split modules featuring $N_{\text{band}}$ channel groups**, so each subband is processed **independently** within the block. Uses **3 subbands (low/mid/high)** like SCNet (vs BS-RoFormer's 62) — an *unequal, small* partition.
- **Asymmetric encoder/decoder capacity:** number of split-module layers in the **decoder decreased ~3×**, while the encoder is made **heavier** (channel width $G$ raised **48 → 56**); split factor $N_{\text{split}}=3$ in the encoder, $1$ in the decoder.
- **Dual-path RoPE transformers:** DTTNet's dual-path **RNN** replaced with **dual-path RoPE Transformers** for sequence modeling along frequency and time — improves performance "without significantly increasing parameters."

## 3. Key results (exact, WebSearch-confirmed)
- Competitive SDR on **MUSDB18-HQ** while using **13× fewer parameters than BS-RoFormer** and **~half the parameters of SCNet**. **Both** comparisons CONFIRMED. (No single headline dB captured in the snippet; the *efficiency-at-competitive-accuracy* claim is the verified result.)

## 4. Limitations & caveats
- "Competitive," not SOTA-beating — the contribution is the **params/accuracy trade-off**, not peak SDR.
- Still a WASPAA-grade model (encoder heavy, RoPE transformers) — its "light" is relative to BS-RoFormer, not to a 5–10 M notebook model. Direction 03 borrows the *design principles* (band-split into an efficient U-Net; unequal small partition; asymmetric enc/dec), not the exact net.

## 5. Relevance to this project (Direction 03)
- **The professional precedent for Direction 03's exact move:** graft a band-split front-end onto an efficient U-Net and show it competes at low parameter count. Direction 03 does the tiny-scale, param-matched *controlled* version.
- **Design cues we adopt:** (i) **few, unequal subbands** (3: low/mid/high) rather than many — feasible at 5–10 M params; (ii) **heavier encoder than decoder**; (iii) split = **channel-grouped processing per band**, a cheap way to give bands dedicated capacity without multiplying parameters. These directly inform the mel-split / uniform-split encoder modules and the param-matching methodology.
- **What we ignore:** RoPE transformers (too heavy for the SingNet budget); TFC-TDF V3 specifics. Direction 03's band modules sit on the compact conv U-Net.
- **Honest framing:** "I took the core idea behind an efficiency-frontier paper and tested whether it transfers to tiny models with a param-matched control" — replication with a real twist, close to what Moises-Light did professionally.

## 6. Verification notes
- Title/authors/venue/arXiv ID: CONFIRMED (WebSearch: arXiv 2510.06785 abs, IEEE Xplore/WASPAA 2025).
- **13× fewer than BS-RoFormer AND ~half of SCNet: CONFIRMED** (WebSearch) — both match RESEARCH_DIRECTIONS.
- **"built on DTTNet": CONFIRMED** as building on the TFC-TDF V3 / DTTNet family (WebSearch architecture description), with the dual-path RNN→RoPE-transformer swap.
- Architecture specifics ($N_{\text{band}}$ groups, $G$ 48→56, $N_{\text{split}}$ 3/1, decoder ÷3, 3 subbands): CONFIRMED via WebSearch of the paper; exact block diagrams not read from the (blocked) PDF.
