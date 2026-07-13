# Verification Log — SingNet Phase 0

Claim-by-claim source verification for the seven research directions. Every citation used
anywhere in this library has a row here. All verification done **2026-07-13**.

**Routes** (pre-probed): `HF` = `mcp__Hugging_Face__paper_search` (arXiv-backed abstracts);
`WS` = WebSearch (US-only); `GH` = WebFetch on github.com / raw.githubusercontent.com;
`ZEN` = WebFetch on zenodo.org. BLOCKED (unusable): arxiv.org, ar5iv, export.arxiv,
semanticscholar API, sigsep.github.io, paperswithcode, joss.theoj.org, hf.co paper pages
(WebFetch 403 — use HF tool instead), ai.honu.io PDF (403).

**Verdicts:** CONFIRMED / CORRECTED (right value given; source doc edited) /
PARTIAL (spirit holds, exact claim not fully groundable) / UNVERIFIABLE.

Rows are grouped: **Shared library** (this step), then one block appended per direction
(Steps 2–8), then **gap-check queries**.

---

## Shared library rows

| # | Source / ID | Claim checked | Route | Verdict | Notes |
|---|---|---|---|---|---|
| S1 | MUSDB18, Zenodo **1117372** | 150 tracks (100/50), 4 stems+mix, STEMS/AAC, ~4.4 GB, non-commercial | ZEN | **CORRECTED** | Size is **4.7 GB** (not ~4.4); 5 stereo streams (mix + vocals/drums/bass/other); license "educational purposes only… no commercial." RESEARCH_NOTES §3 size corrected in-place. |
| S2 | MUSDB18-HQ, Zenodo **3338373** | Uncompressed WAV, access-restricted | ZEN + WS | **CONFIRMED** | 22.7 GB; **request-access, manually approved (<1 day)** — CONFIRMED (one WebFetch summary wrongly said "open"; SigSep/WS confirms restricted). RESEARCH_NOTES correct. |
| S3 | `musdb` `split="valid"` 14-track list | The frozen-protocol validation split | GH (sigsep-mus-db `mus.yaml`) | **CONFIRMED** | 14 names captured verbatim in `papers/musdb18-dataset.md §3`. Load-bearing for PLAN §1.5.1. |
| S4 | Le Roux et al., **1811.02508**, ICASSP 2019 | Title/authors; SI-SDR def; SDR pathologies incl. silence | HF + WS | **CONFIRMED** | arXiv ID CONFIRMED; abstract confirms "critical failure examples." Zero-signal singularity derived from formula. |
| S5 | Vincent et al. **2006** TASLP, BSS-Eval | SDR/ISR/SIR/SAR decomposition, 512-tap filter | WS | **CONFIRMED** | No arXiv (pre-arXiv-era); venue/def CONFIRMED. |
| S6 | SiSEC 2018, **1804.06267** | Introduced MUSDB18 + Python BSSEval (museval) + IBM/IRM/MWF oracles | WS | **CONFIRMED** | Stöter/Liutkus/Ito, April 2018. |
| S7 | museval `metrics.py` silent-frame behavior | Silent frames → NaN, dropped from median | GH (sigsep-mus-eval) | **CONFIRMED** | Quoted logic in `papers/bsseval-museval-sisec2018.md §3`. **Foundation of Direction 08.** |
| S8 | Bake-Off, **2507.06917**, WASPAA 2025 | SDR best for vocals; SI-SAR better drums/bass; FAD/embeddings not +corr for vocals | WS | **CONFIRMED** | Authors **Jaffe & Burgoyne** (newly pinned). Kendall τ 0.25 drums / 0.19 bass for CLAP-FAD. Not on HF; WS-verified. |
| S9 | Spleeter, JOSS **10.21105/joss.02154** | U-Net 6+6, L1 magnitude loss, 25k songs, ~6.6 dB vocals | WS + GH (README) | **PARTIAL** | Architecture/L1/25k/100×RT CONFIRMED. **"~6.6 dB" is NOT in the JOSS paper or README** (README links to a wiki perf page) — attribute to wiki/third-party, not the paper. |
| S10 | Open-Unmix, JOSS **10.21105/joss.01667** + repo | umx 6.32 / umxhq 6.25 vocals; 3-layer BiLSTM; Wiener; MIT; exact `model.py`/`data.py` | GH | **CONFIRMED** | Full architecture + augmentation shapes captured in `papers/openunmix2019.md`. Gain range is **U(0.25,1.25)**, not "±3–6 dB" (RESEARCH_NOTES loose wording noted). |
| S11 | Demucs v1, **1911.13254** (also **1909.01174**) | 6.3 avg SDR / 6.8 with +150; aug recipe; L1>SI-SNR | HF + WS + GH (`augment.py`) | **PARTIAL** | Numbers + aug (Shift/FlipChannels/FlipSign/Remix/Scale) CONFIRMED. **L1-vs-L2 ablation CONFIRMED; clean L1-vs-SI-SNR ablation NOT in paper** (indirect via Conv-TasNet). See `demucs2019-v1.md §2`. RESEARCH_DIRECTIONS not edited; Direction 01 states the nuance and leans on Gusó. |
| S12 | Hybrid Demucs, **2111.03600** | Won MDX 2021; +1.4 dB across sources (MUSDB-HQ) | HF | **CONFIRMED** | +1.4 dB baseline in abstract is vs **non-hybrid Demucs**; "over previous SOTA" phrasing defensible (won MDX'21). |
| S13 | HT-Demucs, **2211.08553** | 9.20 dB avg SDR w/ 800 extra; "performs poorly when trained only on MUSDB" | HF | **CONFIRMED** | Quote is **exact**; +0.45 dB over Hybrid Demucs @800 songs; 9.20 dB needs extra data + sparse attn + per-source FT. |
| S14 | cIRM ResUNet, **2109.05418**, ISMIR 2021 | 7.24→8.98 dB vocals; 22% bins IRM>1; 143 layers | HF | **CONFIRMED** | All three numbers verbatim in abstract. Kong/Cao/Liu/Choi/Wang. |

**Corrections made to source docs in Step 1:** RESEARCH_NOTES §3 MUSDB18 size ~4.4 GB → **4.7 GB** (Zenodo). (Other loose phrasings — UMX gain "±3–6 dB", Spleeter "6.6 dB", Demucs "L1>SI-SNR" — are flagged in the relevant deep-dives and appended-direction rows rather than rewritten, per protocol: they are imprecise, not wrong numbers/authors/venues, and some source bodies were unreadable.)

---

## Direction rows (appended per step 2–8)

_(01 loss-study, 02 aug/scaling, 03 band-split, 05 LoRA, 06 robust, 08 silence, 10 distillation — appended below as each direction is written.)_

### 01 — Loss-function study

| # | Source / ID | Claim checked | Route | Verdict | Notes |
|---|---|---|---|---|---|
| 01-1 | Gusó et al., **2202.07968**, ICASSP 2022 | Controlled MSS loss benchmark; SDR can mislead; recommends spectrogram/phase-sensitive losses (`L2freq`/`SISDRfreq`/`LOGL2freq`/`LOGL1freq`) | WS | **PARTIAL** | Title/authors/venue/scope CONFIRMED. **Specific recommended loss names + ranking `[UNVERIFIED]`** — body unreadable (arxiv/ar5iv/arxiv-vanity/jordipons all blocked/403). Verified direction (SDR misleading; spectrogram losses competitive) matches RESEARCH_DIRECTIONS §1.2. |
| 01-2 | Parallel WaveGAN, **1910.11480**, ICASSP 2020 | MR-STFT loss = spectral convergence + log-mag; 1.44 M params, 4.16 MOS | HF | **CONFIRMED** | Yamamoto/Song/Kim. |
| 01-3 | `auraloss/freq.py` (MR-STFT impl) | $\mathcal L_{sc}=\lVert|S|-|\hat S|\rVert_F/\lVert|S|\rVert_F$; log-mag L1; `fft=[1024,2048,512]`, `hop=[120,240,50]`, `win=[600,1200,240]` | GH | **CONFIRMED** | Code-exact defaults. |
| 01-4 | Demucs v1 loss claim (L1 > SI-SNR) | Attribution to 1911.13254 | WS + GH | **PARTIAL** | L1-vs-L2 ablation CONFIRMED; **clean L1-vs-SI-SNR ablation NOT in paper** (indirect via Conv-TasNet). Direction 01 leans on Gusó for the controlled claim; RESEARCH_DIRECTIONS not edited. |

### 02 — Augmentation & data-scaling

| # | Source / ID | Claim checked | Route | Verdict | Notes |
|---|---|---|---|---|---|
| 02-1 | MixIT, **2006.12701**, NeurIPS 2020 | Unsupervised mixture-invariant training; semi-supervised adaptation | HF | **CONFIRMED** | Wisdom/Tzinis/Erdogan/Weiss/Wilson/Hershey. Speech/universal, not MSS. |
| 02-2 | MixIT-for-MSS, **2505.07631** | FMA MixIT pre-train → MUSDB fine-tune (band-split TF-Locoformer) beats scratch | HF | **CONFIRMED** | Saijo & Bando. No headline dB in abstract (qualitative "improves"); recorded as such. |
| 02-3 | UMX/Demucs aug recipes | gain U(0.25,1.25), channelswap p=0.5, source remixing, sign flip, shift | GH (`data.py`,`augment.py`) | **CONFIRMED** | Code-exact; see shared notes. UMX gain is 0.25–1.25 (not "±3–6 dB" of RESEARCH_NOTES). |
| 02-GAP | Gap-check: factorized aug ablation + scaling curve for compact MSS | Does it exist? | WS | **NULL (open)** | Query 2026-07-13. Transforms well-documented; Demucs discusses aug *impact*; **one adjacent prior "SVS: a study on training data" 1906.02618 (2019)**. No dedicated factorized+scaling study at compact scale → novelty stands; cite 1906.02618 as closest prior. |

### 03 — Mini band-split

| # | Source / ID | Claim checked | Route | Verdict | Notes |
|---|---|---|---|---|---|
| 03-1 | BSRNN, **2209.15174**, TASLP 2023 | band-split + interleaved band/sequence RNN; beats MDX-2021; semi-sup pseudo-labels via activity detector | HF + WS | **CONFIRMED** | Luo & Yu. Vocals **10.01 dB cSDR via DTTNet cross-ref** (not BSRNN's own abstract). |
| 03-2 | Moises-Light, **2510.06785**, WASPAA 2025 | band-split U-Net on DTTNet lineage; 13× fewer than BS-RoFormer, ½ SCNet | WS | **CONFIRMED** | Hung/Pereira/Korzeniowski. "built on DTTNet" = TFC-TDF V3/DTTNet family + dual-path RoPE; both param claims CONFIRMED. |
| 03-3 | Mel-RoFormer, **2310.01809** | overlapped mel bands beat heuristic bands on vocals/drums/other (MUSDB18-HQ) | HF | **CONFIRMED** | Wang/Lu/Won. Distinct from 2409.04702 (later vocal-sep paper). Load-bearing for "mel-split > uniform-split." |
| 03-4 | SCNet, **2401.13276**, ICASSP 2024 | 9.0 dB MUSDB18-HQ no extra data; 48% of HT-Demucs CPU; unequal band compression | HF + WS | **CONFIRMED** | Venue ICASSP 2024 pp.1276–1280 confirmed (WS). |
| 03-5 | DTTNet, **2309.08684**, ICASSP 2024 | 10.12 dB vocals cSDR; 86.7% fewer params than BSRNN | HF | **CONFIRMED** | Chen/Vekkot/Shukla. Source of the BSRNN 10.01 dB number. |
| 03-6 | BS-RoFormer, **2309.02612** | SDX'23 1st (500 extra); 9.80 dB no extra data; RoPE | HF | **CONFIRMED** | Lu et al. Ceiling + param reference. |
| 03-7 | Generalized Bandsplit, **2309.02539** | common-encoder BSRNN generalization; SNR+1-norm loss; psychoacoustic bands | HF | **CONFIRMED** | Watcharasupat et al.; cinematic (DnR), context only. |

### 05 — LoRA source separation

| # | Source / ID | Claim checked | Route | Verdict | Notes |
|---|---|---|---|---|---|
| 05-1 | LoRA, **2106.09685**, ICLR 2022 | $\Delta W=BA$; 10,000× fewer params / 3× less mem vs GPT-3; on-par/better | WS | **CONFIRMED** | Hu/Shen/Wallis/Allen-Zhu/Li/Wang/Wang/Chen. Not on HF search; WS-verified. |
| 05-2 | PETL for music, **2411.19371** | adapters/LoRA ≥ full-FT at <1% params (0.36%/0.22%) on music tagging | WS | **CONFIRMED** | **Title = "Parameter-Efficient *Transfer Learning* for Music Foundation Models"** (Ding et al.); RESEARCH_NOTES paraphrase noted. Scope = tagging, NOT MSS. |
| 05-3 | Transfer→dialog, **2106.09093** | UMX/Spleeter/Conv-TasNet full-FT transfer to dialog separation | WS | **CONFIRMED** | Strauss/Paulus/Torcoli/Edler. **Full FT, not PEFT** — sharpens the gap. |
| 05-4 | Continual SVS, **2512.02432** | U-Net + human-marks-false-positives continual adaptation | WS | **CONFIRMED (existence/scope)** | Dec-2025 arXiv record. A WS-summary "2021 symposium" line is inconsistent with the ID and NOT relied on. |
| 05-5 | UMX LoRA target shapes | fc1/fc2/fc3 + 3-layer BiLSTM weight_ih/hh; ~8.3 M total; r4≈1.2%, r16≈4.9% | GH (`model.py`) | **CONFIRMED** | Param budget computed from code shapes (nb_bins≈1487 assumed; verify programmatically). |
| 05-GAP | Gap-check: PEFT/LoRA-for-MSS by mid-2026 | Does any published LoRA-for-MSS study exist? | WS | **NULL (open) — KEY** | Query 2026-07-13. LoRA in music = generation, beat-tracking (2503.10086), deepfake — **none for source separation**. Direction 05 novelty VERIFIED. Cross-cut: "Why LoRA Resists Label Noise" 2602.00084 (→ Dir 06). |

---

## Gap-check queries (logged per Step)

_(PEFT-for-MSS; controlled chunk-sampling ablation for MSS; factorized augmentation ablation — appended with dates + null/positive findings as each relevant direction is written.)_
</content>
