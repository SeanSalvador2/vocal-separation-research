# Research Notes — Karaoke Vocal Separation (cited)

These are the grounded findings behind `PLAN.md`. Every architecture, dataset, and
benchmark number below is attributed to a real, dated source (arXiv/HF paper, an
official repo, or the Zenodo/SigSep dataset pages). Numbers are quoted as the
sources report them; where a metric convention differs (SI-SDR vs museval BSS-Eval
SDR vs MDX cSDR/uSDR) it is flagged, because those are **not** directly comparable.

> Metric caution used throughout: StemCraft's own numbers (spectral **+4.37 dB**,
> Demucs **+9.55 dB** on vocals) are **SI-SDR on the loudest 12 s vocal window of 5
> MUSDB18 test tracks** (see `stemcraft/REAL_RESULTS.md`). The literature usually
> reports **museval BSS-Eval SDR** (median over 1 s frames, all 50 test tracks) or
> MDX **cSDR/uSDR**. These are different estimators; a "6.3 dB SDR" paper number and
> a "+9.55 dB SI-SDR window" StemCraft number cannot be subtracted. The plan
> re-scores everything with StemCraft's own `bench/metrics.py` so all comparisons in
> the report are apples-to-apples.

---

## 1. Architecture landscape (music source separation)

| Model | Domain / core idea | Vocals metric (as reported) | Train data | Notes / source |
|---|---|---|---|---|
| **Spleeter** (Deezer, 2019) | Spectrogram-masking **U-Net** (6-down/6-up), soft mask, **L1 on masked magnitude** | ~6.6 dB vocals SDR (SiSEC/MUSDB18) | 25k internal songs (not MUSDB) | Fast, the classic "from-scratch U-Net" template. JOSS paper `10.21105/joss.02154`; architecture summary via Deezer/JOSS. |
| **Open-Unmix (UMX / UMX-HQ)** (SigSep, 2019) | **3-layer BiLSTM on magnitude spectrogram**, ratio mask, multichannel **Wiener** post-filter (norbert) | **umxhq 6.25 dB**, **umx 6.32 dB** vocals | MUSDB18 / MUSDB18-HQ only | MIT license (umxl weights CC-BY-NC-SA). The canonical *reproducible on MUSDB* baseline. Source: sigsep/open-unmix-pytorch README. |
| **Demucs (v1, waveform)** (Défossez 2019) | **Waveform** U-Net + BiLSTM (encoder/decoder in time domain) | 6.3 dB avg SDR (all sources), up to 6.8 with +150 songs | MUSDB (+extra) | arXiv **1911.13254**. First waveform model to beat spectrogram masking on MUSDB. |
| **Hybrid Demucs (v3)** (Défossez 2021) | Parallel **spectrogram + waveform** branches | +1.4 dB SDR over prior across sources (MUSDB-HQ) | MUSDB-HQ (+extra) | arXiv **2111.03600**. Won MDX 2021. |
| **Hybrid Transformer Demucs (HT-Demucs)** (Rouard 2022) | Hybrid bi-U-Net with a **cross-domain Transformer** in the bottleneck | **9.20 dB avg SDR** (with 800 extra songs); *"performs poorly when trained only on MUSDB"* | MUSDB-HQ + 800 songs | arXiv **2211.08553**. This is the `htdemucs` family StemCraft already ships (`htdemucs_6s`). Key honest quote: transformers need lots of extra data. |
| **Decoupled Mag/Phase ResUNet** (Kong/ByteDance 2021) | 143-layer **ResUNet**, **complex ideal ratio mask (cIRM)**, allows mask magnitude > 1 | **8.98 dB vocals** (beat prior 7.24) | MUSDB18 | arXiv **2109.05418**. Shows why magnitude-only + Griffin/noisy-phase caps out: 22% of TF bins have IRM > 1. |
| **KUIELab-MDX-Net** (Kim 2021) | **Two-stream**: TF branch + time-domain branch, blended | MDX-2021 2nd place | MUSDB18 | arXiv **2111.12203**. Good "balanced perf/compute" reference. |
| **BSRNN (Band-Split RNN)** (Luo & Yu 2022) | Split spectrogram into **subbands**, interleaved band-level + sequence-level RNN | ~**10.0 dB cSDR vocals** (per DTTNet's comparison) | MUSDB18-HQ only | arXiv **2209.15174**. Beats MDX-2021 top models trained on MUSDB alone; adds semi-supervised finetuning. |
| **DTTNet (TFC-TDF UNet, dual-path)** (Chen 2024) | Lightweight TFC-TDF U-Net + dual-path | **10.12 dB cSDR vocals** with **86.7% fewer params** than BSRNN | MUSDB18-HQ | arXiv **2309.08684**. Evidence that lightweight ≠ weak if the design is good. |
| **BS-RoFormer** (Lu 2023) | Band-split + hierarchical **Transformer with RoPE** | **9.80 dB avg SDR** (no extra data); SOTA | MUSDB18-HQ (+500 for the winning run) | arXiv **2309.02612**. Won SDX23. Current SOTA family. |
| **Mel-RoFormer** (Wang 2023) | RoFormer with **mel-band** split | Beats BS-RoFormer on vocals/drums/other | MUSDB18-HQ | arXiv **2310.01809**. |

**MDX / SDX challenge context**: the Sony Music Demixing (MDX 2021) and Sound
Demixing (SDX 2023) challenges are the SOTA driver; `mvsep.com` hosts a public
model leaderboard (arXiv **2305.07489**, Solovyev 2023). Challenge metrics are
**cSDR** (chunk/segment SDR) and **uSDR** (utterance SDR).

### Tradeoffs distilled
- **Spectrogram magnitude-mask U-Net / BiLSTM** (Spleeter, UMX): simplest to build
  and train from scratch, small compute, but **phase is taken from the mixture**
  (noisy-phase reconstruction) which caps quality (~6 dB vocals) — the ResUNet
  cIRM paper quantifies why (22% of bins need mask > 1). **This is the right target
  for the "I built it from scratch" piece.**
- **Waveform / hybrid (Demucs)**: higher quality, models phase implicitly, but far
  more compute and data-hungry; HT-Demucs explicitly *"performs poorly when trained
  only on MUSDB."* Not realistic to reproduce from scratch in a notebook.
- **Band-split + Transformer (BSRNN, RoFormers)**: SOTA (~10 dB) but large,
  long training, and the RoFormers used extra data. Out of scope to train; useful
  only as a cited "ceiling of the field."

---

## 2. Losses, masks, reconstruction, metrics

**Masking strategies**
- **Soft / ratio mask (IRM-style)**: predict `M ∈ [0,1]` (sigmoid) per TF bin;
  `Ŝ = M ⊙ |X|`, reconstruct with the **mixture phase**. Used by Spleeter (L1) and
  UMX. Simple, stable, phase-limited.
- **Complex ideal ratio mask (cIRM)**: predict real+imag mask so phase is corrected;
  ResUNet paper (2109.05418) shows a big gain (7.24 → 8.98 dB vocals) and that masks
  should be allowed > 1.
- **Direct magnitude regression** then mixture-phase iSTFT (UMX predicts a target
  magnitude, effectively an unbounded mask).
- **Post-filter**: **multichannel Wiener filtering** (norbert) on the network's
  magnitude estimates — UMX's standard inference step, worth ~0.3–0.5 dB.

**Loss functions**
- **L1 / MSE on (masked) magnitude spectrogram** — Spleeter uses L1, UMX uses MSE on
  magnitude. Cheap, stable, the default for a from-scratch spectrogram model.
- **SI-SDR loss** (time-domain, scale-invariant) — Le Roux et al. 2019 "SDR — half-
  baked or well done?"; requires differentiable iSTFT; directly optimizes the
  eval metric StemCraft uses.
- **Multi-resolution STFT loss** (sum of spectral-convergence + log-magnitude L1 at
  several FFT sizes) — from neural-vocoder literature (Parallel WaveGAN); reduces
  artifacts when training in/through the waveform domain.
- A **generalized band-split SNR + L1 loss** is proposed in the cinematic band-split
  paper (2309.02539) — motivation for combining SI-SDR-like and sparsity terms.

**Evaluation metrics**
- **SI-SDR** (scale-invariant SDR, dB) — StemCraft's `bench/metrics.py:si_sdr`
  already implements Le Roux 2019 and is unit-tested (+inf for gain-scaled ref, 20 dB
  for a constructed orthogonal-error case). **Primary metric for this project.**
- **BSS-Eval SDR/SIR/SAR** via **museval** — the field-standard MUSDB scoring
  (median over 1 s frames). Optional secondary for literature comparability.
- **MDX cSDR / uSDR** — challenge conventions; note but not required.

---

## 3. Datasets

**MUSDB18 / MUSDB18-HQ** (SigSep — Rafii et al.)
- **150 full tracks**, **100 train / 50 test**, 44.1 kHz stereo, **4 stems**:
  `vocals`, `drums`, `bass`, `other` (+ the mixture). This exactly matches
  StemCraft's existing `collapse_to_musdb4` mapping (guitar+piano+other → other).
- **MUSDB18** (compressed **STEMS**/MP4, AAC — effective bandwidth ~16 kHz):
  ~4.7 GB per Zenodo (StemCraft's downloader measured **4.68 GB**). Zenodo record **1117372**.
- **MUSDB18-HQ** (uncompressed WAV, full bandwidth): Zenodo record **3338373**;
  **access-restricted** ("educational purposes only … request access, manually
  checked"). Larger (tens of GB uncompressed).
- **License**: research/education only, **not redistributable, non-commercial** —
  StemCraft already treats it this way (downloads, extracts test split, deletes,
  commits nothing). The plan keeps that policy.

**Hugging Face availability** (the user has an HF account; searched via HF tools):
- `danjacobellis/musdb18HQ` — parquet mirror; splits `train` (128 rows, ~4.8 GB) +
  `validation` (118 rows, ~4.6 GB). Convenient `datasets`-loadable HQ mirror.
- `CLAPv2/MUSDB18-HQ`, `sebchw/musdb18` (uses `stempeg`), `jxie/musdb18` — other
  mirrors.
- `cs229-audio-ml-project/musdb18-processed` — **pre-segmented "active stem"
  segments** (MIT-tagged, derived) — handy for fast iteration, but verify license
  provenance before leaning on it.
- Extra data (optional, for the fine-tune/aug track): `kwatcharasupat/musdb25`
  (multitrack MUSDB18 recreation, alpha), `choihy/musdb_and_moisesdb`, **MoisesDB**.
- ⚠️ These HF mirrors are community re-uploads of a restricted dataset. **Primary
  recommended path = official Zenodo via the `musdb`/`stempeg` loader** (matches
  StemCraft's existing `experiments/realdata_download.py`); treat HF mirrors as a
  convenience fallback and note their unofficial status in the report.

**Augmentation (critical on 100 training songs)** — the standard MSS recipe, used by
Spleeter, UMX, and Demucs:
- **Random source remixing / "shuffling"**: build training mixtures by summing
  `vocals` from one song with `drums/bass/other` from *other* songs. This is the
  single biggest lever — it turns 100 songs into combinatorially many mixtures and is
  why MSS models don't overfit. (Demucs paper: *"with proper data augmentation"*.)
- **Random gain** per source (e.g. ±3–6 dB), **random channel swap**, **sign flip**.
- **Pitch/tempo shift** (Demucs uses ±semitone / ±tempo%) — heavier, optional.
- **Random crop** to fixed-length chunks (e.g. 3–6 s) each step.
- **MixIT-style unsupervised pre-training** on unlabeled audio (FMA) then fine-tune
  on MUSDB improves over from-scratch (arXiv **2505.07631**) — a stretch/bonus idea.

---

## 4. Feasibility (from-scratch, notebook, consumer hardware)

- **Realistic quality, from scratch on MUSDB18**: a compact magnitude-mask U-Net /
  BiLSTM lands around **UMX territory (~6 dB museval SDR vocals)** *at best*, and a
  genuinely small/short-trained model more like **~4–6 dB**. Matching Demucs
  (~9 dB) or SOTA (~10 dB) from scratch in a notebook is **not realistic** — those
  need waveform/transformer models and/or hundreds of extra songs (HT-Demucs
  explicitly underperforms when trained on MUSDB only).
- **Fine-tuning path is the higher-quality comparison**: start from **pretrained
  Open-Unmix (umxhq)** — already ~6.25 dB vocals — and fine-tune; this reaches good
  quality in far less compute and gives the report a legitimate "fine-tuning"
  storyline distinct from the from-scratch model.
- **Compute reality**:
  - **GPU (single consumer card / Colab T4/L4, or an RTX-class)**: from-scratch U-Net
    to a decent checkpoint in a handful of hours to ~1 day; UMX-style full training
    historically **~1–2 days/GPU**. Fine-tuning UMX: hours.
  - **CPU-only** (StemCraft's real-data suite ran on 4 CPU cores, no GPU): fine for
    **inference/eval** and a **tiny "proof it trains" run** on a few-song subset, but
    **full training is impractical** on CPU. The plan makes a GPU the assumed
    training environment and CPU the inference/eval environment.
- **Inference cost in-app**: a small spectrogram U-Net separates a song in
  seconds–tens of seconds on CPU with chunked overlap-add (comparable to StemCraft's
  Demucs CPU cost: ~84 s for 5 excerpts). Export as TorchScript or ONNX.

---

## 5. Source list (for citations in REPORT/THEORY)

- Spleeter — JOSS `10.21105/joss.02154`; deezer/spleeter.
- Open-Unmix — Stöter et al. 2019; sigsep/open-unmix-pytorch (README reports umxhq
  6.25 / umx 6.32 dB vocals).
- Demucs (waveform) — arXiv 1911.13254.
- Hybrid Demucs — arXiv 2111.03600.
- HT-Demucs — arXiv 2211.08553.
- Decoupled Mag/Phase ResUNet (cIRM) — arXiv 2109.05418.
- KUIELab-MDX-Net — arXiv 2111.12203.
- BSRNN — arXiv 2209.15174.
- DTTNet — arXiv 2309.08684.
- BS-RoFormer — arXiv 2309.02612; Mel-RoFormer — arXiv 2310.01809.
- Generalized Band-Split (loss) — arXiv 2309.02539.
- MDX/SDX benchmark & mvsep leaderboard — arXiv 2305.07489.
- MixIT pre-training for MSS — arXiv 2505.07631.
- SI-SDR definition — Le Roux et al., "SDR — half-baked or well done?", 2019.
- MUSDB18 — Zenodo 1117372; MUSDB18-HQ — Zenodo 3338373; SigSep dataset page.
- StemCraft internal baselines — `stemcraft/REAL_RESULTS.md`, `stemcraft/ABLATIONS.md`,
  `stemcraft/src/stemcraft/bench/metrics.py`, `.../separation.py`.

---

## 6. Addendum (2026-07 literature refresh, for `RESEARCH_DIRECTIONS.md`)

New sources verified during the Phase-R direction review (full context and how each
grounds a candidate direction is in `RESEARCH_DIRECTIONS.md`):

**Architectures / efficiency**
- **SCNet** (arXiv 2401.13276, ICASSP 2024) — subband splitting with sparsity-based
  unequal compression; **9.0 dB SDR on MUSDB18-HQ, no extra data**, ~48 % of
  HT-Demucs CPU inference time.
- **Moises-Light** (Hung, Pereira & Korzeniowski, WASPAA 2025, arXiv 2510.06785) —
  resource-efficient **band-split U-Net** built on DTTNet; competitive MUSDB-HQ SDR
  with ~**13× fewer params than BS-RoFormer**, ~half of SCNet's; scales with
  MoisesDB extra data. Evidence the efficiency frontier is a first-class result.
- **HS-TasNet** (Venkatesh et al., ICASSP 2024, arXiv 2402.17701) — real-time MSS at
  **23 ms latency**: 4.65 overall SDR on MUSDB (5.55 with extra data); follow-up
  arXiv 2511.13146 ("Towards Practical Real-Time Low-Latency MSS").
- **Banquet** (Watcharasupat & Lerch, arXiv 2406.18747) — query-based single-decoder
  separation beyond 4 stems.

**Losses / evaluation**
- **Gusó, Pons, Pascual & Serrà** (ICASSP 2022, arXiv 2202.07968) — controlled
  benchmark of MSS loss functions + cross-correlation of metrics with a subjective
  test; notes SDR can mislead. Tension with Demucs's report that L1 beats
  SI-SNR-style losses for MSS training (arXiv 1911.13254).
- **"Musical Source Separation Bake-Off"** (WASPAA 2025, arXiv 2507.06917) —
  large listener study on MUSDB18: SDR best predicts perception **for vocals**,
  SI-SAR better for drums/bass, embedding/FAD metrics uncorrelated or negatively
  correlated with perceived vocal quality; recommends stem-specific evaluation.

**Data / robustness / semi-supervision**
- **SDX'23 music track** (Fabbro et al., arXiv 2308.06979, TISMIR 2024) — introduces
  **robust MSS**: training under simulated label noise and bleeding
  (SDXDB23_LabelNoise / SDXDB23_Bleeding); flags corrupted training data as an
  under-studied axis. Best SDX'23 system: +1.6 dB SDR over the MDX'21 winner.
- **MixIT** (Wisdom et al., arXiv 2006.12701) — the original mixture-invariant
  unsupervised training (complements the already-cited MSS application 2505.07631).

**Fine-tuning / adaptation**
- **LoRA** (Hu et al., arXiv 2106.09685) — low-rank adaptation, generic PEFT.
- **PEFT for music foundation models** (arXiv 2411.19371) — adapters/LoRA reach
  near-full-fine-tune quality at <1 % trainable params on music tagging tasks; no
  published equivalent for MSS models (a genuine gap).
- **Transfer from MSS to dialog separation** (arXiv 2106.09093) — hands-on
  cross-domain fine-tuning precedent for separation models.
- **Continual SVS adaptation with human-in-the-loop** (arXiv 2512.02432).

**Generative separation (context only — out of Colab scope)**
- **MSDM** (Mariani et al., arXiv 2302.02257, ICLR 2024) — joint diffusion
  generation + separation; latent variant arXiv 2409.06190; diffusion/consistency
  refinement of separator outputs arXiv 2412.06965.

**Adjacent evaluation/benchmarks**
- **MedleyVox** (Jeon et al., arXiv 2211.07302) — benchmark for multiple-singing-
  voice separation (duet/unison/main-vs-rest); adjacent to StemCraft's measured
  1-voice/2-voice harmony cliff.
- **SepACap** (arXiv 2509.26580) — a-cappella multi-singer separation with periodic
  activations + composite loss.
- **Separated vocals improve Whisper lyrics transcription** (arXiv 2506.15514) — a
  downstream-utility evaluation axis for separators.
</content>
</invoke>
