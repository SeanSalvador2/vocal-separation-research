# Project Plan — "SingNet": A From-Scratch Karaoke Vocal Separator

**A deep-learning / data-science portfolio project.** Train a lightweight, from-scratch
vocal-separation model through the full DS lifecycle, then ship it into
[StemCraft](https://github.com/SeanSalvador2/stemcraft) as a new `SeparationEngine` alongside the classical
`SpectralSeparator` and `DemucsSeparator`.

> **Status: PLAN (v2, optimized) — approved direction, nothing built or trained yet.**
> No datasets are downloaded. Grounding for every claim is in
> [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md); the candidate research-contribution menu is
> in [`RESEARCH_DIRECTIONS.md`](RESEARCH_DIRECTIONS.md).

**What this project is optimized for (read this first).** This is a portfolio project
for DS/ML/AI job applications. The deliverable a hiring manager actually evaluates is
not the final SI-SDR number — it is the *process*: pre-registered hypotheses, controlled
experiments, honest variance reporting, a frozen evaluation protocol, one-command
reproducibility, and a report that reads like competent research writing. Every phase
below is designed so that even a "we didn't beat X" outcome is a strong, defensible
result. Beating Demucs is explicitly a non-goal.

---

## 0. TL;DR

- **Build a compact spectrogram-masking U-Net from scratch** on MUSDB18, targeting a
  **2-stem split: vocals vs accompaniment** (the exact primitive Apple Music Sing /
  karaoke needs). This is the headline "I built it from scratch" piece.
- **Run a second track that fine-tunes pretrained Open-Unmix (umxhq)** so the report
  can compare **classical spectral → from-scratch U-Net → fine-tuned UMX → Demucs**
  on one honest axis (StemCraft's own SI-SDR bench).
- **Pre-registered hypotheses (§1.3)** with falsification criteria, a **frozen
  evaluation protocol (§1.5)** defined before any training, **multi-seed variance
  reporting** on headline configs, and a **"reproduce all figures" one-command path**.
- **A research contribution phase (Phase R, §3)** — a small replicate-and-extend study
  chosen from the menu in `RESEARCH_DIRECTIONS.md` — turns the project from "I trained
  a model" into "I asked and answered a question the literature doesn't quite answer."
- **Honest quality expectation** (StemCraft window SI-SDR, vocals): from-scratch U-Net
  **~+4 to +7 dB** (beat the +4.37 spectral baseline, fall short of Demucs's +9.55);
  fine-tuned UMX **~+6 to +8 dB**.
- **Compute**: designed for **Google Colab Pro** (T4/L4/A100) as the primary training
  environment; local AMD RX 9060 XT 16 GB (ROCm) as a secondary/optional runner; CPU
  for inference/eval inside StemCraft.
- **Deliverables**: 4–5 Jupyter notebooks backed by a tested `singnet/` package,
  `REPORT.md`, `THEORY.md` (math), a figure suite, an experiment registry, and a
  `SingNetSeparator` implementing StemCraft's `SeparationEngine`.

---

## 1. Problem framing, hypotheses & success criteria

### 1.1 Task definition
Given a stereo music mixture `x(t)`, estimate the **lead-vocals** signal `v̂(t)` and
the **accompaniment** `â(t) = x(t) − v̂(t)`. We deliberately start with **2 stems
(vocals / accompaniment)**, not 6, because:
1. It is the exact operation karaoke / Apple Music Sing performs (duck or remove the
   vocal).
2. It is the one stem StemCraft's classical fallback already recovers usefully
   (+4.37 dB), so it is the cleanest place to demonstrate a learned improvement.
3. MUSDB18 provides an isolated `vocals` stem and `accompaniment = drums+bass+other`,
   so ground truth is exact.

### 1.2 Metric
Primary: **SI-SDR (dB)** using StemCraft's existing, unit-tested
[`bench/metrics.py:si_sdr`](https://github.com/SeanSalvador2/stemcraft/blob/main/src/stemcraft/bench/metrics.py) — so results
sit on the **same axis** as the numbers already in `REAL_RESULTS.md`. We also report
**SI-SDR improvement (SI-SDRi)** over the input mixture, and **museval BSS-Eval SDR**
as a clearly-labeled secondary for literature comparability. The two estimators are
**never conflated** (see `RESEARCH_NOTES.md §0`); recent evidence that SDR-family
metrics imperfectly track perception for some stems (WASPAA 2025 "Bake-Off",
arXiv 2507.06917) is acknowledged and motivates the listening check in §3 Phase 7.

### 1.3 Pre-registered hypotheses (written before any training; the report scores each)

| ID | Hypothesis | Prediction (falsifiable) | Where tested |
|---|---|---|---|
| **H1** | A compact learned mask model beats a tuned classical DSP separator on real music | SingNet vocals SI-SDR > +4.37 dB (spectral fallback) on the frozen protocol, by ≥ +1 dB | Phase 4/7 |
| **H2** | Source remixing is the single highest-leverage augmentation on 100 songs | Removing remix augmentation costs more val SI-SDR than removing any other single augmentation | Phase 5 |
| **H3** | Transfer beats from-scratch at equal (small) compute | Fine-tuned umxhq > SingNet on vocals SI-SDR for < 25 % of SingNet's GPU-hours | Phase 6 |
| **H4** | The mixture-phase ceiling is real but not dominant at small capacity | Oracle IRM (magnitude, mixture phase) scores several dB above SingNet — i.e. capacity/data, not phase, is our binding constraint | Phase 7 (oracle study) |
| **H5** | *(Phase R placeholder — set when a research direction is chosen from `RESEARCH_DIRECTIONS.md`)* | — | Phase R |

Each hypothesis gets an explicit verdict in `REPORT.md` (supported / refuted / mixed),
including the refuted ones. The **oracle-mask study in H4** (score the ideal ratio
mask and ideal binary mask on the test set) is cheap, standard, and gives the report a
principled "how much headroom did our model family even have?" upper bound — a detail
that strongly signals experimental maturity.

### 1.4 Anchors and targets (all SI-SDR on vocals, StemCraft convention)

| Reference | Vocals SI-SDR | Role |
|---|---|---|
| Do-nothing (mixture as "vocals") | ~0 dB (negative on windows) | floor |
| **StemCraft `SpectralSeparator`** | **+4.37 dB** (measured, `REAL_RESULTS.md`) | **the number to beat (H1)** |
| Oracle IRM / IBM (computed in Phase 7) | measured upper bound | headroom of the mask family (H4) |
| **From-scratch SingNet U-Net** | **target +4 to +7 dB** | headline deliverable |
| **Fine-tuned Open-Unmix (umxhq)** | **target +6 to +8 dB** | transfer-learning track (H3) |
| **StemCraft `DemucsSeparator` (htdemucs_6s)** | **+9.55 dB** (measured) | ceiling / not expected to beat |

> Caveat carried over from `REAL_RESULTS.md`: those two measured anchors are means on
> the loudest 12 s vocal window of **5** MUSDB test tracks. Phase 7 **re-scores the
> spectral fallback and Demucs on the full frozen protocol** (all 50 test tracks) so
> the final ladder is apples-to-apples; the 5-track window numbers are kept as a
> continuity cross-check with StemCraft's published docs.

### 1.5 Frozen evaluation protocol (defined now, before training — no re-negotiation)

1. **Splits.** MUSDB18: 86 train / **14 validation** (the standard `musdb`
   `split="valid"` track list, same one Open-Unmix uses) / 50 test. The test set is
   **never touched** until Phase 7; all tuning decisions use the 14-track validation
   split only.
2. **Primary score.** Vocals + accompaniment SI-SDR per track (full-track, computed
   on overlapping chunks and length-weighted-averaged, mono-summed like StemCraft's
   bench), reported **per-song and aggregate (mean, median, std, min)** over all 50
   test tracks.
3. **Continuity score.** The exact `REAL_RESULTS.md` protocol (loudest 12 s vocal
   window, same 5 tracks) so the new engines drop into StemCraft's existing tables.
4. **Literature score (secondary).** museval BSS-Eval SDR (median-of-frames,
   median-of-tracks) on the full test set — reported in its own clearly-labeled table.
5. **Statistics policy (honest for small n).** Engines are compared **paired by
   track**: report paired per-song deltas, a **bootstrap 95 % CI over the 50 tracks**
   for mean deltas, and a **Wilcoxon signed-rank test** for headline pairwise claims.
   Training stochasticity is reported separately: the **final SingNet config is
   trained with 3 seeds** (mean ± std across seeds); one-factor sweep points run 1
   seed with the seed-noise band from the 3-seed run drawn on every sweep figure, so
   readers can see which sweep differences are within noise. No significance claims
   are made from the 5-track window protocol.
6. **Listening check.** A small blind test (protocol in Phase 7) — reported as
   qualitative, preference-count evidence, never as "proof."

---

## 2. Recommended approach + alternatives

### 2.1 Recommended (two complementary tracks + one research phase)

**Track A — From-scratch spectrogram U-Net ("SingNet")** — the headline.
A compact magnitude-mask U-Net (Spleeter-family design: ~6 encoder / 6 decoder
conv blocks, skip connections, sigmoid soft mask on `|STFT|`, mixture-phase iSTFT).
Small enough to train on a Colab GPU in hours, and the clearest possible
demonstration of "I implemented the architecture, loss, and training loop myself."
Expected **~+4 to +7 dB** vocals SI-SDR (§1.4).

**Track B — Fine-tune pretrained Open-Unmix (umxhq)** — the transfer-learning
comparison. UMX is MIT-licensed, MUSDB-trained, ~6.25 dB museval vocals out of the
box. Fine-tune its vocals model on our augmented pipeline. This gives the report a
genuine fine-tuning narrative (distinct from Track A) and a stronger candidate engine
at a fraction of the compute — and it directly tests H3. **Scope is deliberately
bounded** (§3 Phase 6): one recipe comparison, not a second research program.

**Phase R — Research contribution** — one small replicate-and-extend study chosen
from the 10-option menu in `RESEARCH_DIRECTIONS.md`, filled in as H5 once picked.
This is what elevates the project above "yet another U-Net on MUSDB."

### 2.2 Alternatives considered (unchanged conclusions, updated citations)

| Alt | What | Why not the headline |
|---|---|---|
| **Waveform / hybrid from scratch** (Demucs-style) | Time-domain U-Net + LSTM/Transformer | Compute/data-hungry; HT-Demucs *"performs poorly when trained only on MUSDB"* (arXiv 2211.08553). |
| **Band-split RNN / RoFormer from scratch** | SOTA (~9.8–10 dB SDR: BS-RoFormer 2309.02612, SCNet 2401.13276) | Weeks of GPU training even for the "small" versions. Cited as the field ceiling; the *band-split idea* survives as a candidate Phase R direction at tiny scale (see `RESEARCH_DIRECTIONS.md` #3). |
| **Complex-mask (cIRM) U-Net from scratch** | Predict phase too (arXiv 2109.05418: 7.24→8.98 dB) | Trickier to train/debug; kept as a candidate Phase R direction (#4) rather than a vague "stretch goal," so it only happens with a real hypothesis attached. |
| **Diffusion/generative separation** (MSDM, arXiv 2302.02257) | Generative posterior sampling | Research-grade training cost and unstable metrics; cited as context only. |

**Recommendation: Track A core + Track B bounded comparison + one Phase R direction.**

---

## 3. Full data-science lifecycle

Each phase lists concrete deliverables and an exit criterion. Notebook mapping in §4.

### Phase 1 — Data acquisition, licensing & EDA
- **Acquire MUSDB18** via the official Zenodo route using the `musdb`/`stempeg`
  loaders — mirroring StemCraft's existing `experiments/realdata_download.py` pattern
  (resumable, research-only, nothing committed). Start on **MUSDB18 (compressed
  STEMS, ~4.4 GB)**; optionally re-run final numbers on **MUSDB18-HQ**. On Colab:
  download once to Google Drive, decode stems to per-track `.npy`/`.wav` shards
  (decode is the slow step; do it once, not per-session).
- **License hygiene**: MUSDB is research/education-only, non-redistributable. Commit
  no audio; keep data on Drive/local scratch only; weights and report contain no
  copyrighted audio.
- **EDA deliverables** (each tied to a later design decision, not decoration):
  per-track duration/loudness stats; **vocal activity ratio** per track (→ justifies
  the chunk-sampling policy in Phase 2); vocal spectral energy distribution
  (→ justifies n_fft/band choices); genre/tempo spread (→ defines the Phase 7 error-
  analysis strata); train/valid/test split audit.
- **Visuals**: mixture vs stems waveform + log-mel spectrogram; vocal-activity
  timeline; per-track vocal-energy histogram.
- **Exit criterion**: split manifest (86/14/50 track lists) committed as CSV; EDA
  notebook runs top-to-bottom on Colab.

### Phase 2 — Preprocessing pipeline
- **Resample** to a working rate (default **44.1 kHz**; 22.05 kHz kept as a compute-
  saving ablation knob, not the default — MUSDB18's AAC already caps ~16 kHz).
- **STFT**: default **n_fft = 4096, hop = 1024, Hann** (UMX's convention), with
  n_fft ∈ {1024, 2048, 4096} as a Phase 5 sweep. Compute STFT **on-the-fly on GPU**
  (`torch.stft`) rather than caching spectrograms — cheaper than Drive I/O and keeps
  augmentation in the waveform domain.
- **Chunking**: fixed **6 s** segments (3 s as a sweep point). **Chunk-sampling
  policy**: sample chunks with probability weighted by vocal activity (from the
  Phase 1 activity mask), never fully discarding silent chunks — the model must also
  learn to output silence. The policy itself is an ablation knob (and candidate
  Phase R direction #8).
- **Normalization**: per-chunk mixture-statistics standardization; log-magnitude
  option.
- **Augmentation (the key lever on 100 songs)** — the standard Spleeter/UMX/Demucs
  recipe (`RESEARCH_NOTES.md §3`): **random source remixing** (vocals from song *i* +
  accompaniment from song *j*), **random per-source gain** (±6 dB), **channel swap**,
  **sign flip**; optional pitch/tempo shift. Implemented as composable, individually
  switchable transforms so Phase 5 (and H2) can ablate them factor-by-factor.
- **Deliverable**: a deterministic, **seeded** `Dataset`/`DataLoader` in the
  `singnet/` package with **unit tests** on: STFT→iSTFT round-trip error (< −60 dB),
  remix correctness (mixture == sum of sources), augmentation determinism given a
  seed, and mask-target correctness. Reuse StemCraft's `audio_io` conventions.
- **Exit criterion**: tests green; one batch visualized end-to-end (waveform → chunks
  → augmented spectrogram → target mask).

### Phase 3 — Model architecture(s) + the math
- **Track A — SingNet U-Net**: encoder of strided 2-D conv blocks (Conv→BN→ReLU),
  symmetric decoder with transposed convs + skips, final **sigmoid** → soft ratio
  mask `M ∈ [0,1]^{F×T}`; `V̂ = M ⊙ |X|`; mixture-phase iSTFT. Width/depth are
  Phase 5 knobs. Target **≈ 5–15 M params** (fits any Colab GPU with batch 8–16 at
  6 s / 44.1 kHz in mixed precision).
- **Track B — Open-Unmix**: load `umxhq`, fine-tune the vocals target model; UMX's
  BiLSTM + Wiener post-filter used as-is.
- **The math (goes into `THEORY.md`, outline in §4.3)** — written *while building*,
  not retro-fitted: every equation in THEORY.md must map to a line of code in
  `singnet/` (the report cross-references them).
- **Exit criterion**: model summary (param count derived by hand in THEORY.md §4 and
  verified programmatically); forward pass shape test; gradient-flow smoke test.

### Phase 4 — Training loop + experiment tracking
- **Optimizer**: AdamW, LR 1e-3 with cosine or plateau schedule (Phase 5 knob),
  gradient clipping, early stopping on **validation SI-SDR** (not loss), best-
  checkpoint saving, **mixed precision (bf16/fp16)**.
- **Loss**: start **L1 on masked magnitude** (Spleeter's choice); the loss comparison
  itself is a Phase 5 experiment (and candidate Phase R direction #1), informed by
  Gusó et al. (ICASSP 2022, arXiv 2202.07968).
- **Colab resilience (required, not optional)**: checkpoint + optimizer + RNG state
  to Drive every N steps; training script fully resumable; every run identified by a
  config hash so an interrupted sweep continues where it stopped.
- **Tracking**: **CSV/JSON-first experiment registry** (`results/registry.csv`: run
  id, config hash, seed, GPU type, wall-clock, best val SI-SDR, checkpoint path) +
  matplotlib curves — dependency-safe and diff-able in git. Weights & Biases optional
  on top, never load-bearing. Log train/val loss, val SI-SDR per epoch, LR, grad
  norm, and sample audio every N epochs.
- **Exit criterion**: the M3 proof-it-trains gate (§7); registry populated by ≥ 1
  full run with resumption tested at least once.

### Phase 5 — Hyperparameter tuning & controlled ablations (from-scratch track)
One factor at a time (mirroring `stemcraft/ABLATIONS.md` house style), on the fixed
14-track validation split, at a **declared reduced budget** (e.g. 40 % of full
training steps) with the top-2 configs re-verified at full budget (the budget caveat
is stated in the report — sweep rankings at reduced budget are themselves a finding):

| Knob | Candidate values | Hypothesis |
|---|---|---|
| **STFT n_fft / hop** | 1024/2048/4096, hop = n_fft/4 | freq-vs-time resolution tradeoff |
| **Mask type** | sigmoid ratio vs unbounded (ReLU) | ratio-mask cap vs mask > 1 (arXiv 2109.05418: 22 % of bins need M > 1) |
| **Loss** | L1-mag vs MSE-mag vs time-domain SI-SDR vs multi-res STFT | which optimizes eval SI-SDR (cf. arXiv 2202.07968) |
| **Width/depth** | base channels {16,32,64}, depth {4,5,6} | capacity vs overfit on 100 songs |
| **LR schedule** | cosine vs plateau vs constant; LR {3e-4, 1e-3, 3e-3} | convergence/stability |
| **Augmentation** | none / +gain+swap / +remix / +pitch-tempo (leave-one-out for H2) | quantify the remix lever |
| **Chunk length / sample rate** | 3 s vs 6 s; 22.05 k vs 44.1 k | context vs compute |

- **Search strategy**: coarse one-factor sweeps → small random search over the 2–3
  most sensitive knobs. **Seed policy per §1.5.5** (sweeps 1 seed + noise band;
  final config 3 seeds). Held-out test untouched. **Negative results are reported
  with the same prominence as wins** (house style: "maj7 actively hurts").
- **Compute budget**: the sweep table above is ~20–25 runs at reduced budget ≈
  **25–40 T4-hours total** (less on L4/A100) — feasible on Colab Pro across a week
  of sessions; the registry + resumability make interruptions cheap.

### Phase 6 — Fine-tuning track (bounded scope)
- Load pretrained **umxhq**; fine-tune the vocals model on the Phase 2 augmented
  pipeline. Exactly **three recipe comparisons** (no open-ended search):
  (a) full fine-tune vs (b) frozen-BiLSTM head-only vs (c) LR-warmup full fine-tune;
  each ~1–3 GPU-hours.
- Deliverable: fine-tuned checkpoint + from-scratch-vs-fine-tuned comparison table
  with GPU-hours column (this is the H3 evidence — quality *per unit compute* is
  the story, not just quality).
- If Phase R direction #5 (LoRA/PEFT) is chosen, this phase merges into it.

### Phase 7 — Evaluation & error analysis
- **Score per the frozen protocol (§1.5)**: all 50 test tracks, per-song + aggregate,
  4-way ladder (spectral / SingNet / fine-tuned UMX / Demucs) on identical audio;
  plus the 5-track continuity windows; plus museval secondary table.
- **Oracle study (H4)**: score IRM and IBM oracle masks (and oracle-magnitude +
  mixture-phase resynthesis) on the test set — the honest headroom line every mask-
  based paper should show.
- **Error analysis** (strata fixed in Phase 1 EDA): SI-SDR by genre, male/female
  lead, solo vs harmony vocals, dense vs sparse mix; per-song scatter SingNet-vs-
  Demucs; worst-5 tracks dissected with mask/spectrogram figures; link back to
  StemCraft's known 1-voice/2-voice cliff.
- **Listening test (small, honest, pre-registered)**: 8 clips × {spectral, SingNet,
  Demucs} karaoke mixes; blind, randomized, level-matched; n ≥ 8 raters; one fixed
  question ("which version has less audible vocal residue?") + MUSHRA-style 0–100
  quality slider; report preference counts, per-clip medians, and rater agreement;
  explicitly framed as qualitative support, not statistical proof. Motivated by the
  documented SDR↔perception gap (arXiv 2507.06917).
- **Deliverable**: results CSVs + figures + before/after audio examples + a
  hypothesis-scoreboard table (H1–H5 verdicts).

### Phase R — Research contribution (placeholder until direction chosen)
> **To be filled from [`RESEARCH_DIRECTIONS.md`](RESEARCH_DIRECTIONS.md)** — a menu
> of 10 replicate-and-slightly-extend studies, each with hypothesis, minimal
> experiment, Colab feasibility, and risk rating. Once the user picks one:
> 1. Its hypothesis becomes **H5** in §1.3 (pre-registered before running).
> 2. Its experiment plan becomes this phase's checklist, budgeted at **≤ 25–35
>    additional GPU-hours** and ~1–2 weeks of part-time work.
> 3. Its writeup becomes `REPORT.md §7` ("Research study") with its own related-work
>    paragraph, method, results (with seeds/CIs per §1.5.5), and limitations.
> Placement: after Phase 5 (it typically reuses the trained SingNet + sweep
> infrastructure), before final Phase 7 scoring, so its best outcome can ship.

### Phase 8 — Integration into StemCraft
- Implement `SingNetSeparator(SeparationEngine)` with the same interface:
  `separate(audio: AudioData) -> Dict[str, np.ndarray]` returning all 6 canonical
  `STEMS`, mono float32, input length. 2-stem mapping: `vocals` = learned estimate;
  accompaniment → `other` with `drums/bass/guitar/piano` zeroed (documented,
  honest), or optional hybrid spectral banding fill. Karaoke needs only
  vocals + accompaniment.
- **Export**: **ONNX** preferred (runs without a torch dependency; mirrors the
  "runs-anywhere" spectral fallback), TorchScript fallback. Numerical parity test
  exported-vs-eager (< 1e-4 max abs diff on a fixed input).
- **Inference**: chunked STFT with 25 % overlap-add, mixture-phase iSTFT, optional
  Wiener post-filter. Budget: CPU seconds–tens of seconds per song; measured RTF
  goes into the report's cost table next to Demucs's 84 s / 5 excerpts.
- **Engine picker**: add `"singnet"` to `get_separator(...)` with lazy import
  (mirror `_demucs_available()`); cache via existing `StemCache`.
- **Deliverable**: engine class + load/inference test + README note on when to pick
  SingNet vs spectral vs Demucs.

---

## 4. Deliverables

### 4.0 Code architecture rule (portfolio signal)
Notebooks **orchestrate and visualize**; all load-bearing logic (data, transforms,
model, losses, metrics glue, train loop) lives in the importable **`singnet/`
package with unit tests**. Hiring managers read this as "can structure real ML code,"
and it is what makes the sweeps and Phase R re-runnable.

### 4.1 Jupyter notebooks
1. **`01_data_and_eda.ipynb`** — download/loader, license notes, split manifest, EDA
   stats + the decision-linked visuals.
2. **`02_preprocessing_and_augmentation.ipynb`** — STFT/iSTFT, chunk sampling,
   augmentation gallery, round-trip + determinism checks, mask-target visualization.
3. **`03_model_train_tune.ipynb`** — SingNet architecture walkthrough, training
   runs, Phase 5 sweeps with curves/tables (reads/writes the experiment registry).
4. **`04_finetune_eval_integration.ipynb`** — UMX fine-tuning, frozen-protocol
   4-way evaluation + oracle study + error analysis, ONNX export + parity test,
   `SingNetSeparator` smoke test.
5. **`05_research_study.ipynb`** — Phase R (created once the direction is chosen).

### 4.2 `REPORT.md` (narrative structure — reads like a short paper, not a diary)
1. Abstract with the headline ladder table and one-sentence H1–H5 verdicts.
2. Introduction: problem, why 2-stem karaoke, relation to StemCraft.
3. Related work (compressed from RESEARCH_NOTES: mask U-Nets → hybrid → band-split
   SOTA; where this project sits and why).
4. Data & licensing; EDA findings that drove design decisions.
5. Method: preprocessing, architecture, losses, training (equations referenced from
   THEORY.md).
6. Experiments: pre-registered hypotheses, protocol (§1.5 verbatim), sweeps with
   noise bands, negative results.
7. Research study (Phase R).
8. Results: 4-way ladder + oracle headroom + error analysis + listening check.
9. Integration & cost (RTF, memory, export path).
10. Limitations & honest gap-to-Demucs discussion; future work.
11. Reproducibility appendix: exact commands, config hashes, seeds, env lockfile,
    hardware, total GPU-hours.

### 4.3 `THEORY.md` (math/architecture, LaTeX) — outline
1. **STFT / iSTFT**: definitions, windowing, COLA condition (proof sketch), why
   magnitude + mixture phase.
2. **Mask family**: IBM, IRM, ratio mask, why the oracle IRM is an upper bound;
   why `M` can exceed 1 and cIRM (arXiv 2109.05418), with the Wiener-filter
   derivation connecting IRM to MMSE under Gaussian source assumptions.
3. **U-Net**: conv/transpose-conv shapes, receptive-field and parameter-count
   derivations (verified against code).
4. **Losses**: L1/MSE magnitude; **SI-SDR derived** (projection form,
   `SI-SDR = 10 log10(‖αs‖²/‖αs−ŝ‖²)`, `α = ŝᵀs/‖s‖²`), its scale invariance and
   failure modes (Le Roux 2019); multi-resolution STFT loss.
5. **Metrics**: SI-SDR vs BSS-Eval SDR/SIR/SAR vs MDX cSDR/uSDR — definitions, why
   they differ, and what each rewards (ties to RESEARCH_NOTES §0 and arXiv
   2507.06917).
6. **Augmentation as distribution design**: why source remixing changes the mixture
   distribution the model sees (and its independence assumption caveat).

### 4.4 Visuals / figures (each produced by the reproduce-figures path, §6)
- Mixture vs stems spectrograms; predicted-mask heatmaps; oracle-vs-predicted mask
  side-by-side.
- Training/val curves per experiment; sweep charts **with the 3-seed noise band**.
- 4-way ladder chart with per-song scatter + bootstrap CI whiskers; oracle line.
- Error-analysis bars (genre / vocal type); worst-case dissections.
- Latency/RTF vs quality table across engines.
- Before/after audio examples (small WAVs, no MUSDB redistribution — short excerpts
  only within research-use norms, or synthesized/CC examples for the public repo).

---

## 5. Compute & data plan (sized for the hardware the user actually has)

**Primary: Google Colab Pro** (T4 16 GB / L4 24 GB / A100 40 GB).
- SingNet full training run (~10 M params, mixed precision, batch ~12 × 6 s):
  **~6–12 h on T4, ~3–6 h on L4, ~1.5–3 h on A100** to a converged checkpoint.
- Phase 5 sweep block: **~25–40 T4-hours** total at reduced budget (§3 Phase 5).
- 3-seed final training: ~3 × one full run.
- UMX fine-tuning: **1–3 GPU-hours per recipe**.
- Full-test-set inference for the 4-way ladder: minutes on GPU (Demucs included).
- Working style: **Drive-backed checkpoints + resumable runs + config-hash registry**
  (§3 Phase 4) so session disconnects cost minutes, not runs. Estimated total
  project GPU budget **≈ 60–100 GPU-hours** — comfortably within Colab Pro over
  4–8 weeks of part-time work.

**Secondary: local AMD RX 9060 XT 16 GB (RDNA4, ROCm on Linux/WSL2).**
- Treated as optional overflow (long unattended sweeps, listening-clip rendering).
- Known friction: ROCm wheel availability for RDNA4 and occasional op gaps; the plan
  never *depends* on it. Anything that runs locally must run identically on Colab
  (same pinned env, same configs). If ROCm setup exceeds ~half a day, drop it.

**CPU (StemCraft's env)**: inference/eval of the shipped engine and the tiny
proof-it-trains run only.

**Data**: MUSDB18 compressed (~4.4 GB download; ~12–15 GB after decoding to wav
shards on Drive) for development; MUSDB18-HQ optional for final numbers (Zenodo
access-restricted; tens of GB — only if Drive quota allows). Nothing committed.

**Proof-it-trains milestone (before any long run)**: overfit **one 6 s chunk** to
near-zero loss; then 3–5 songs for a few epochs, confirm val SI-SDR beats do-nothing.
Only then launch full training (runs on T4 in < 1 h; partially CPU-feasible).

---

## 6. Rigor & reproducibility standards (project-wide contract)

1. **Environment pinning**: `requirements.txt` with exact pins + `pip freeze`
   lockfile committed per results-generation date; Python/torch/CUDA versions and
   GPU type recorded in the registry for every run. One bootstrap cell at the top of
   each notebook installs the pinned env on Colab.
2. **Seeds**: every run takes an explicit seed controlling Python/NumPy/torch +
   DataLoader workers; seed recorded in the registry; `torch.use_deterministic_algorithms`
   where feasible (deviations documented).
3. **Configs**: every experiment is a YAML config; the registry keys runs by config
   hash; no "I changed a constant in the notebook" experiments.
4. **Reproduce-all-figures path**: `python -m singnet.reproduce --figures` (or
   `make figures`) regenerates every figure and table in REPORT.md from the committed
   CSVs; a separate `--from-checkpoints` tier re-runs evaluation from checkpoints.
   The README states exactly which tier a reader can run without MUSDB access.
5. **Test-set discipline**: test tracks read exactly twice — once in Phase 7, once
   for final report figures. Enforced by keeping test-set paths out of all training/
   tuning configs.
6. **Honest reporting**: negative results and refuted hypotheses appear in the
   abstract-level summary, not buried; every number carries its protocol label
   (window SI-SDR / full-track SI-SDR / museval SDR) so nothing is cross-compared
   dishonestly.

---

## 7. Honest risks & expectations

- **Quality ceiling**: from-scratch on MUSDB18, expect **~+4 to +7 dB** vocals
  SI-SDR (StemCraft convention) — beat +4.37 spectral, likely not Demucs's +9.55.
  Framed as a rigor showcase, not a Demucs-beater (same posture as StemCraft docs).
- **Small data / overfitting**: 100 songs → remix augmentation is essential (H2);
  without it the model overfits. Headline ablation.
- **Phase limitation**: magnitude mask + mixture phase caps quality; H4's oracle
  study quantifies exactly how much of our gap is phase vs capacity — turning a
  risk into a result.
- **Metric confusion**: museval SDR ≠ StemCraft window SI-SDR; never conflated
  (§1.5); SDR↔perception gap acknowledged (arXiv 2507.06917) and hedged with the
  listening check.
- **Compute risk**: Colab session limits / GPU lottery → mitigated by resumable
  runs, config-hash registry, reduced-budget sweeps, and the cheap UMX track as the
  shippable fallback if from-scratch training stalls.
- **ROCm risk**: local AMD path may not work smoothly for training — it is optional
  by design; timebox to half a day.
- **Statistical honesty**: 50 test songs and small seed counts → paired
  bootstrap/Wilcoxon over tracks, seed-noise bands on sweeps, and no p-value theater
  on 5-track windows.
- **Integration mismatch**: 2-stem model behind a 6-stem interface → documented
  mapping; karaoke needs only vocals + accompaniment.
- **Dataset licensing**: research-only, non-redistributable — no audio committed;
  public audio examples use short research-context excerpts or CC material.

---

## 8. Phased milestone timeline with go/no-go checkpoints

| Phase | Work | Deliverable | Budget (approx) | Go/No-Go |
|---|---|---|---|---|
| **M0 Setup** | Env pinning, Drive layout, dataset download, license notes | pinned env + loader stub | 0 GPU-h | data loads; a stem plays |
| **M1 EDA** | Phase 1 | `01_…ipynb` + split manifest + figures | 0 GPU-h | activity/spectrogram visuals sane |
| **M2 Pipeline** | Phase 2 | `02_…ipynb`; tested Dataset | 0 GPU-h | round-trip + determinism tests green |
| **M3 Proof-it-trains** | tiny overfit + 3–5 song run | loss→0 on one chunk; val SI-SDR > do-nothing | < 2 GPU-h | **GO gate** — fix pipeline before scaling |
| **M4 Full train (A)** | Phase 4 | trained SingNet + curves + registry | 6–12 GPU-h | val SI-SDR > +4.37 (H1 on val) |
| **M5 Sweeps** | Phase 5 | ablation tables/figures with noise bands | 25–40 GPU-h | best config; H2 verdict |
| **M6 Fine-tune (B)** | Phase 6 | UMX recipes + compute-normalized table | 3–9 GPU-h | H3 verdict on val |
| **MR Research study** | Phase R | `05_…ipynb` + REPORT §7 | ≤ 25–35 GPU-h | H5 verdict (either direction is a result) |
| **M7 Eval** | Phase 7 | frozen-protocol ladder, oracle, error analysis, listening test | ~2 GPU-h + human time | test numbers final; hypothesis scoreboard done |
| **M8 Integration** | Phase 8 | `SingNetSeparator` + ONNX + parity test | 0 GPU-h | runs on CPU behind interface; beats spectral in-app |
| **M9 Docs** | finalize | REPORT, THEORY, notebooks, reproduce-figures path | 0 GPU-h | figures regenerate from CSVs |

**Primary go/no-go is M3** (prove learning before spending GPU-days). Secondary is
**M4** (clear the +4.37 bar). If M4 stalls, the fine-tuned-UMX track becomes the
shippable engine and the from-scratch model remains the rigor centerpiece. Phase R
is scheduled after M5 so it reuses the tuned model + infrastructure.

---

## 9. Repository layout (proposed, for the build phase — not created yet)

```
vocal-separation-lab/
  PLAN.md                      # this file
  RESEARCH_NOTES.md            # cited findings
  RESEARCH_DIRECTIONS.md       # lit review + Phase R menu (choose one)
  README.md                    # (build) how to reproduce, incl. figure tiers
  REPORT.md                    # (build) the research write-up
  THEORY.md                    # (build) math/architecture
  requirements.txt / lock      # (build) pinned env
  notebooks/                   # 01_..05_ ipynb (orchestration only)
  singnet/                     # tested package: data, model, losses, train, eval, reproduce
  configs/                     # YAML per experiment
  results/
    registry.csv               # run registry (id, config hash, seed, GPU, scores)
    csv/  figures/  audio/     # committed outputs (no MUSDB audio)
  checkpoints/                 # weights + exported onnx/torchscript (or Drive links)
```

The eventual `SingNetSeparator` lands in StemCraft (e.g.
`stemcraft/src/stemcraft/separation.py` or a submodule) so it deploys with the app.
