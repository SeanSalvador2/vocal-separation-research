# Spleeter: a fast and efficient music source separation tool with pre-trained models

Hennequin, Khlif, Voituret, Moussallam (Deezer). JOSS 5(50):2154, 2020. DOI **10.21105/joss.02154**. Repo: `deezer/spleeter`. | Verified: WebSearch (JOSS DOI, architecture, loss, training) + WebFetch repo README, 2026-07-13.

The canonical "from-scratch magnitude-mask U-Net" template — the architecture family PLAN Track A ("SingNet") deliberately mirrors.

## 1. Problem & context
Deezer released Spleeter as an open, fast, pretrained separator. It is the reference point for "a plain spectrogram U-Net with an L1 mask loss, scaled up on a large private corpus" — the design SingNet reproduces at small scale on MUSDB.

## 2. Method
- **Architecture:** spectrogram-domain **U-Net**, **12 layers = 6-down encoder + 6-up decoder**, skip connections, soft mask on magnitude, mixture-phase iSTFT. Models for 2 stems (vocals/accompaniment), 4 stems (v/d/b/other), 5 stems (adds piano).
- **Loss:** **L1 norm between the masked input-mix magnitude spectrogram and the source-target magnitude spectrogram** — i.e. $\mathcal{L}=\lVert M\odot|X| - |S|\rVert_1$. This is the exact loss SingNet starts from (PLAN Phase 4) and one arm of Direction 01.
- **Training data:** ~**25,000** 30-second song excerpts from Deezer's **internal** catalogue (not MUSDB); Adam; ~1 week on a single GPU.
- **Speed:** separates 4 stems ~**100× faster than real-time** on a single GPU.

## 3. What is and isn't published (be precise)
- The **JOSS paper is a ~2-page software paper**; it documents the U-Net, the L1 magnitude loss, and the tool — it does **not** contain MUSDB SDR tables.
- The repo **README does not report SDR numbers either**; it links to a wiki "Separation-Performances" page and states the 2- and 4-stem models "have high performances on the musdb dataset."
- The commonly cited **"~6.6 dB vocals SDR"** is therefore **not from the JOSS paper**; it comes from the **Spleeter wiki / third-party museval benchmarks** on MUSDB. Quote it as such, never as a paper result.

## 4. Limitations & caveats
- Magnitude mask + mixture phase → phase-limited (see `kong2021-cirm-resunet.md`).
- Trained on a large private set, so its quality is **not** a from-scratch-on-MUSDB reference; it is an architecture and loss template, not a data-matched baseline.

## 5. Relevance to this project
- **Track A / SingNet:** the 6-down/6-up magnitude-mask U-Net + L1 loss is the direct blueprint.
- **Direction 01:** L1-magnitude is the "champion to beat" loss; the tension is whether time-domain SI-SDR or MR-STFT improves on it at small scale.
- **Direction 02:** Spleeter's success is largely a *data-scale* story (25k songs) — motivating the "how far do 100 songs go?" question.
- **transfer-dialog precedent (`../../05-lora-source-separation`)** fine-tunes Spleeter among others.

## 6. Verification notes
- JOSS DOI, U-Net 6+6, L1 magnitude loss, 25k excerpts, 100× RT: CONFIRMED (WebSearch of JOSS/Deezer-research + repo README).
- "6.6 dB not in the paper": CONFIRMED by reading the README (links out to a wiki; no in-README SDR table). RESEARCH_NOTES' "~6.6 dB vocals SDR (SiSEC/MUSDB18)" is retained but should be attributed to the wiki/third-party eval, not the JOSS paper — flagged here rather than edited (the number itself is not asserted to be from the paper in our docs).
