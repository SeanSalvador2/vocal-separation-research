# Research Directions — Literature Review & Phase-R Menu

**Purpose.** `PLAN.md` reserves a "Phase R — Research contribution": one small,
honest, *replicate-and-slightly-extend* study appropriate for a recent DS/ML/AI
undergrad, feasible on Colab Pro, that turns the project from "trained a U-Net" into
"asked and answered a question the literature doesn't quite answer." This document is
(1) a current literature review of music/singing-voice separation, and (2) a menu of
**10 candidate directions** with hypotheses, minimal experiments, feasibility, and a
recommendation ranking. **The user picks one; its hypothesis becomes H5 in PLAN.md.**

All citations are real papers/repos (arXiv IDs or official pages); numbers are quoted
as the sources report them. Metric caution from `RESEARCH_NOTES.md §0` applies
throughout: museval SDR ≠ MDX cSDR/uSDR ≠ StemCraft's window SI-SDR.

---

## Part 1 — Literature review (state of the field, mid-2026 view)

### 1.1 Where SOTA is now

The MDX'21 → SDX'23 challenge era moved the field from hybrid waveform/spectrogram
U-Nets to **band-split time-frequency models with sequence modeling**:

- **HT-Demucs** (Rouard et al., arXiv 2211.08553): hybrid bi-U-Net + cross-domain
  Transformer; ~9.2 dB avg SDR *with 800 extra songs*; explicitly *"performs poorly
  when trained only on MUSDB."*
- **BSRNN** (Luo & Yu, arXiv 2209.15174): splits the spectrogram into hand-chosen
  subbands, interleaves band-level and time-level RNNs; ~10 dB cSDR vocals trained
  on MUSDB18-HQ alone; also introduced **semi-supervised fine-tuning with
  pseudo-labels** on unlabeled songs selected by a source-activity detector.
- **BS-RoFormer** (Lu et al., arXiv 2309.02612): band-split + hierarchical
  Transformer with rotary embeddings; won SDX'23 (trained with 500 extra songs);
  9.80 dB avg SDR *without* extra data. **Mel-RoFormer** (Wang et al., arXiv
  2310.01809) replaces the ad-hoc band scheme with a mel-scale mapping and beats it
  on vocals — evidence that the *band partition itself* matters.
- **SCNet** (arXiv 2401.13276, ICASSP 2024): splits subbands and compresses
  low-information bands more aggressively; 9.0 dB SDR on MUSDB18-HQ with no extra
  data at ~48 % of HT-Demucs's CPU inference time.
- **DTTNet** (Chen et al., arXiv 2309.08684): dual-path TFC-TDF U-Net; 10.12 dB
  cSDR vocals with **86.7 % fewer parameters than BSRNN** — lightweight ≠ weak.
- **Moises-Light** (Hung, Pereira & Korzeniowski, WASPAA 2025, arXiv 2510.06785):
  a **resource-efficient band-split U-Net** built on DTTNet; competitive SDR on
  MUSDB18-HQ with ~**13× fewer parameters than BS-RoFormer** and about half of
  SCNet's. The efficiency frontier is an active, publishable axis — not just peak SDR.

**Reading for this project**: peak SDR now belongs to band-split transformers with
extra data, but the *ideas* (band-splitting, complex-spectrogram targets, efficiency
engineering) transfer to small models — which is exactly the replicate-at-small-scale
territory Phase R lives in.

### 1.2 Masks, phase, and losses

- **Magnitude masks + mixture phase** (Spleeter, UMX) cap quality: Kong et al.
  (arXiv 2109.05418) show ~22 % of TF bins need mask > 1 and that predicting a
  **complex ideal ratio mask (cIRM)** with a deep ResUNet lifts vocals 7.24 → 8.98 dB
  SDR on MUSDB18. Open question at *small* scale: how much of that gain is the cIRM
  target vs the 143-layer capacity? (Direction #4.)
- **Loss functions**: Gusó, Pons, Pascual & Serrà, *"On loss functions and evaluation
  metrics for music source separation"* (ICASSP 2022, arXiv 2202.07968) benchmark a
  large set of losses in a controlled setup and cross-correlate candidate metrics
  with a subjective test; they find spectrogram-domain losses (L2/log-L1 family)
  competitive and note SDR can mislead. Défossez et al. (arXiv 1911.13254) train
  Demucs on L1 waveform loss — their in-paper ablation is L1-vs-L2, and the contrast
  with SI-SNR-style training is indirect (Conv-TasNet, trained on SI-SNR, shows
  audible artifacts in their human evaluations). So the "obvious" move of
  training on your eval metric (SI-SDR) is *not* clearly right for music — a genuine,
  cheap-to-test tension. (Direction #1.)
- **Multi-resolution STFT losses** come from the vocoder literature and are commonly
  added when training through the waveform.

### 1.3 Data: augmentation, scale, and corruption

- The standard recipe (Spleeter/UMX/Demucs; see arXiv 1911.13254) is **random source
  remixing across songs, per-source gain, channel swap**, optional pitch/tempo — but
  published work rarely *factorizes* how much each transform contributes for a small
  model on MUSDB-only data. (Direction #2.)
- **MixIT-style unsupervised pre-training** on unlabeled music then supervised
  fine-tuning helps MSS (Saijo & Bando, arXiv 2505.07631) — evidence that data, not
  architecture, is the binding constraint at MUSDB scale.
- **SDX'23** (Fabbro et al., arXiv 2308.06979, TISMIR 2024) introduced **robust MSS**:
  training with simulated *label noise* and *bleeding* (SDXDB23_LabelNoise /
  SDXDB23_Bleeding) — the organizers themselves flag corrupted training data as an
  under-studied, practically important axis. (Direction #6.)

### 1.4 Evaluation is itself under fire

- Le Roux et al. 2019 ("SDR — half-baked or well done?") motivated SI-SDR.
- The **"Musical Source Separation Bake-Off"** (WASPAA 2025, arXiv 2507.06917) ran a
  large listener study on MUSDB18: BSS-Eval SDR predicts perception best **for
  vocals**, SI-SAR predicts better for drums/bass, and **embedding metrics (FAD
  variants) are uncorrelated or negatively correlated with perceived vocal quality**.
  Authors recommend stem-specific evaluation. Small careful perception studies are
  therefore publishable-grade contributions, not garnish. (Direction #7.)

### 1.5 Efficiency, latency, and deployment

- **HS-TasNet** (Venkatesh et al., ICASSP 2024, arXiv 2402.17701): real-time,
  23 ms-latency separation at 4.65 overall SDR on MUSDB (5.55 with extra data) —
  quantifying the price of causality/low latency. Follow-up work continues (arXiv
  2511.13146, *"Towards Practical Real-Time Low-Latency MSS"*). Offline models' SDR
  ≠ deployable quality; the latency/quality Pareto is a real research axis that fits
  StemCraft's "engines with different cost profiles" story. (Direction #9.)
- SCNet and Moises-Light (above) make parameter/inference efficiency a first-class
  reported result.

### 1.6 Adjacent threads (context for the menu)

- **Generative/diffusion separation**: MSDM (Mariani et al., arXiv 2302.02257, ICLR
  2024) does joint generation+separation; latent variants (arXiv 2409.06190) and
  consistency-refinement hybrids (arXiv 2412.06965) exist. Compute-heavy and
  metric-unstable — cited as context, deliberately **not** on the menu.
- **Multiple singing voices**: MedleyVox (Jeon et al., arXiv 2211.07302) is the
  evaluation benchmark for lead-vs-rest / duet / unison separation — directly
  adjacent to StemCraft's measured 1-voice/2-voice harmony cliff.
- **Query/stem-agnostic models**: Banquet (Watcharasupat & Lerch, arXiv 2406.18747)
  separates beyond 4 stems with one query-conditioned decoder.
- **PEFT in music**: LoRA/adapters give near-full-fine-tune quality at < 1 %
  trainable parameters for music *classification* foundation models (arXiv
  2411.19371); Hu et al.'s LoRA (arXiv 2106.09685) is generic. A hands-on transfer
  study fine-tuning MSS models cross-domain exists for dialog separation (arXiv
  2106.09093) — but **published PEFT-for-MSS results are scarce**, which makes a
  careful small study genuinely useful. (Direction #5.)
- **Continual/personalized separation**: human-in-the-loop adaptation for SVS
  (arXiv 2512.02432) signals interest in adapting separators to specific content —
  supporting the domain-adaptation angle of #5.
- **Downstream utility**: separated vocals measurably improve lyrics transcription
  with Whisper (arXiv 2506.15514) — an alternative *utility-based* evaluation axis.

---

## Part 2 — The menu: 10 candidate Phase-R directions

Common constraints: Colab Pro (T4/L4/A100), MUSDB18(-HQ) only unless stated, the
SingNet U-Net + pipeline from PLAN.md Phases 1–5 as infrastructure, ≤ ~25–35 GPU-hours,
evaluated on the frozen protocol (PLAN §1.5) with the seeds/CI policy. Difficulty is
{Low, Medium, High} for an undergrad; Risk = chance the study fails to produce a
defensible result (note: a clean negative *is* a defensible result in every option).

---

### #1 — "Train on what you test?" A controlled loss-function study at small scale
- **Builds on**: Gusó et al., ICASSP 2022 (arXiv 2202.07968); Défossez et al. (arXiv
  1911.13254), whose Demucs trains on L1 waveform loss (in-paper ablation: L1 vs L2;
  the L1-vs-SI-SNR contrast is indirect, via SI-SNR-trained Conv-TasNet's artifacts).
- **Hypothesis**: training a compact mask U-Net directly on (negative) SI-SDR does
  **not** beat L1-magnitude on SI-SDR evaluation — the literature's tension holds at
  small scale — but a multi-resolution STFT auxiliary term reduces audible artifacts.
- **Minimal experiment**: fixed architecture/data/seed policy; 5 losses (L1-mag,
  MSE-mag, log-L1-mag, time-domain SI-SDR, L1 + multi-res STFT) × 3 seeds; evaluate
  SI-SDR, museval SDR, and a 5-clip informal artifact listening check.
- **Feasibility**: 15 runs ≈ 15–25 T4-hours at reduced budget + 3 full-budget
  confirmations. Entirely within the Phase 5 machinery.
- **Expected outcome / success test**: a loss-ranking table with seed error bars; H5
  supported if the SI-SDR-trained model is ≤ the L1 model within noise band.
- **Portfolio story**: "I found conflicting claims in two well-known papers and ran
  the controlled comparison at my scale" — textbook scientific process.
- **Difficulty/Risk**: **Low / Low** (any outcome is reportable).

### #2 — How far do 100 songs actually go? Augmentation factorization + data-scaling curves
- **Builds on**: the standard augmentation recipe (Demucs, arXiv 1911.13254; UMX);
  MixIT-for-MSS motivation that data is the constraint (arXiv 2505.07631).
- **Hypothesis**: (a) source remixing accounts for the majority of the augmentation
  gain (PLAN H2, deepened); (b) with remixing, val SI-SDR vs number-of-training-songs
  follows a log-like curve that has *not* saturated at 86 songs — quantifying how
  data-starved MUSDB-only training is.
- **Minimal experiment**: leave-one-out augmentation ablation (5 configs) + training
  on {21, 43, 64, 86} songs with full augmentation (4 configs), 1 seed each + 3 seeds
  at the two endpoints; plot the scaling curve with extrapolation caveats.
- **Feasibility**: ~9–12 runs at reduced budget ≈ 15–25 T4-hours.
- **Expected outcome / success test**: an augmentation-contribution bar chart and a
  data-scaling figure; success = clean monotone curves with error bars (the *shape*
  is the finding, whichever way it bends).
- **Portfolio story**: data-centric ML is exactly what industry teams do daily;
  "I measured the value of data vs augmentation before asking for more of either"
  is a hiring-manager-native narrative.
- **Difficulty/Risk**: **Low / Low**. Deepest synergy with the existing plan (it *is*
  an expanded H2).

### #3 — A mini band-split front-end: does the band-partition idea survive at 5 M params?
- **Builds on**: BSRNN (arXiv 2209.15174), Mel-RoFormer's finding that mel-scale
  bands beat ad-hoc bands (arXiv 2310.01809), Moises-Light's band-split U-Net
  (arXiv 2510.06785), SCNet's unequal band compression (arXiv 2401.13276).
- **Hypothesis**: replacing SingNet's uniform 2-D conv encoder with a band-split
  front-end (separate low/mid/high sub-encoders on mel-spaced bands, merged in the
  bottleneck) improves vocals SI-SDR at **equal parameter count**, because vocal
  energy is concentrated and bands get dedicated capacity.
- **Minimal experiment**: 3 models at matched ~5–10 M params: baseline U-Net,
  3-band mel-split variant, 3-band *uniform*-split variant (to isolate "splitting"
  from "mel spacing"); 3 seeds each on the final configs.
- **Feasibility**: ~9 full-ish runs ≈ 20–35 T4-hours; moderate implementation work
  (a new encoder module + tests).
- **Expected outcome / success test**: supported if mel-split > uniform-split >
  baseline outside the seed noise band; a per-frequency-band error breakdown figure
  explains *why*.
- **Portfolio story**: "I took the core idea behind the current SOTA family and
  tested whether it transfers to tiny models" — replication with a real twist
  (param-matched control), close to what Moises-Light did professionally.
- **Difficulty/Risk**: **Medium / Medium** (null result plausible — still reportable,
  but less flashy).

### #4 — Does phase matter when you're small? Magnitude mask vs complex mask at fixed capacity
- **Builds on**: Kong et al.'s cIRM ResUNet result, 7.24 → 8.98 dB vocals on a
  143-layer network (arXiv 2109.05418).
- **Hypothesis**: at ~5–10 M params on MUSDB-only data, switching the head from
  sigmoid magnitude mask to a bounded complex mask yields **less than half** the
  relative gain reported at 143-layer scale — i.e. small models are capacity-bound,
  not phase-bound. (Directly quantifies PLAN H4's oracle-headroom question with a
  trained model.)
- **Minimal experiment**: same backbone, 3 heads: (a) sigmoid magnitude mask,
  (b) unbounded magnitude mask, (c) cIRM head (real+imag, bounded via tanh
  compression as in the paper); 3 seeds each; plus the oracle IRM/cIRM lines from
  Phase 7 for context.
- **Feasibility**: ~9 runs ≈ 20–30 T4-hours; head implementation is contained but
  the cIRM target/loss needs care (unit tests on synthetic signals).
- **Expected outcome / success test**: a "gain from phase modeling vs model scale"
  discussion anchored by oracle bounds; success = a clear measured gap between the
  three heads with CIs, whichever direction.
- **Portfolio story**: "Papers show X helps at 143 layers; nobody says whether it
  helps at 10 M params — I measured it, with oracle upper bounds." Strong
  theory-meets-experiment flavor (pairs beautifully with THEORY.md §2).
- **Difficulty/Risk**: **Medium / Medium** (debugging complex-valued targets is the
  main risk; mitigated by the oracle-mask unit tests).

### #5 — LoRA for source separation: parameter-efficient fine-tuning of a pretrained separator
- **Builds on**: LoRA (arXiv 2106.09685); PEFT for music foundation models — adapters
  /LoRA reach full-fine-tune quality with < 1 % trainable params on music tagging
  (arXiv 2411.19371); transfer-learning-from-MSS precedent (arXiv 2106.09093).
  **Gap**: essentially no published LoRA-for-MSS numbers.
- **Hypothesis**: LoRA on Open-Unmix's BiLSTM/fc layers recovers ≥ 90 % of the
  full-fine-tune SI-SDR gain of Phase 6 at < 5 % trainable parameters, and is *less
  prone to catastrophic forgetting* when adapting to a shifted domain (e.g.
  MUSDB18's compressed AAC vs HQ audio, or a genre subset).
- **Minimal experiment**: 4 recipes × 2 target domains: zero-shot umxhq, head-only,
  LoRA (rank ∈ {4, 16}), full fine-tune; measure adapted-domain gain **and**
  source-domain regression; 2–3 seeds on the headline pair.
- **Feasibility**: fine-tunes are 1–3 GPU-hours each → ~15–25 GPU-hours total; UMX
  is small and MIT-licensed; LoRA on an LSTM requires wrapping input/recurrent
  projections (contained, testable).
- **Expected outcome / success test**: a quality-vs-trainable-params curve + a
  forgetting table; success = the curve exists with CIs (any shape is a finding,
  and even a "LoRA underperforms on LSTMs" result is a useful negative).
- **Portfolio story**: applies the single most job-relevant 2020s fine-tuning
  technique to a domain where it's unreported — modern, practical, and honest about
  novelty ("first *careful small-scale* look," not "first ever").
- **Difficulty/Risk**: **Medium / Medium-Low** (compute is cheap; main risk is
  LSTM-LoRA plumbing).

### #6 — Robust training: how much do label noise and bleeding actually hurt a small separator?
- **Builds on**: SDX'23's robust-MSS track and its LabelNoise/Bleeding datasets
  (Fabbro et al., arXiv 2308.06979) — the organizers explicitly flag corrupted
  training data as an under-explored, industry-relevant problem.
- **Hypothesis**: simulated stem bleeding (mixing ε of accompaniment into the vocal
  target) degrades SingNet's vocals SI-SDR roughly linearly in ε, and a simple
  mitigation (loss-side: discarding the top-k% highest-loss chunks per epoch, a
  noisy-label heuristic) recovers a measurable fraction of the loss.
- **Minimal experiment**: corrupt MUSDB training targets at ε ∈ {0, 5, 15, 30 %}
  bleed (constructed from the stems we already have — no new data needed); train at
  each level; re-train the worst level with the mitigation; 1 seed per level + 3 at
  the endpoints.
- **Feasibility**: ~6–8 runs ≈ 15–25 T4-hours; corruption is a data-pipeline
  transform we control end-to-end.
- **Expected outcome / success test**: a degradation curve (dB vs ε) + mitigation
  delta; success = the curve, full stop — nobody publishes this for compact models.
- **Portfolio story**: "real-world training data is dirty; I measured the damage and
  a cheap defense" — the data-quality narrative every applied-ML team cares about.
- **Difficulty/Risk**: **Medium / Low** (the curve cannot fail to exist).

### #7 — Do our metrics hear what people hear? A small perception-vs-metric study on the 4-way ladder
- **Builds on**: the WASPAA 2025 Bake-Off (arXiv 2507.06917: SDR best for vocals,
  SI-SAR better for drums/bass, FAD-style metrics anti-correlated on vocals); Le Roux
  2019; Gusó et al. 2022's metric cross-correlation.
- **Hypothesis**: on karaoke/vocal-isolation outputs from the project's own 4-way
  ladder (spectral / SingNet / UMX-ft / Demucs), per-clip SI-SDR rank-correlates with
  blind listener preference at ρ ≥ 0.6 for isolated vocals but **worse for the
  karaoke (accompaniment) direction**, where vocal *bleed salience* dominates
  perception.
- **Minimal experiment**: 10 test-set clips × 4 systems, blind randomized MUSHRA-lite
  (hidden reference + anchor) with two questions (overall quality; audible vocal
  bleed), n ≈ 10–15 raters (friends/classmates, self-hosted webMUSHRA or a simple
  form); compute Spearman ρ between ratings and {SI-SDR, museval SDR, SI-SAR} per
  stem direction; report rater agreement (Krippendorff's α).
- **Feasibility**: ~0 additional GPU-hours (reuses Phase 7 outputs); the cost is
  human-organizational. Sample-size honesty is mandatory (report CIs on ρ; frame as
  a pilot replication).
- **Expected outcome / success test**: a metric-vs-perception correlation table for
  *this project's own systems* + a concrete recommendation for StemCraft's bench
  (e.g. "add SI-SAR for accompaniment"); success = coherent correlations with honest
  uncertainty.
- **Portfolio story**: evaluation literacy is rare and prized; "I didn't just report
  a metric — I checked whether the metric means anything for my use case, replicating
  a WASPAA 2025 methodology at pilot scale."
- **Difficulty/Risk**: **Low-Medium / Medium** (risk is recruiting raters and noisy
  small-n correlations; mitigated by pre-registering the pilot framing).

### #8 — The silence problem: vocal-activity-aware sampling and the cost of quiet
- **Builds on**: MUSDB's long vocal-silent stretches (documented in our own Phase-1
  EDA); informal silent-chunk handling in UMX/Spleeter training lore; BSRNN's
  source-activity detector for pseudo-labeling (arXiv 2209.15174). Adjacent work
  exists — BSMamba2 (arXiv 2508.14556) attacks sparse-vocal robustness via
  architecture, and Demucs augments by injecting silence — but none defines a
  silence-leakage metric or reports a sampling-policy-vs-leakage tradeoff; that
  metric + tradeoff is the defensible gap (see 08-silence-leakage/research/LITERATURE.md §4).
- **Hypothesis**: (a) energy-weighted chunk sampling beats uniform sampling on
  overall vocals SI-SDR; but (b) fully *dropping* silent chunks hurts the model's
  false-positive behavior — measured as energy leaked into `v̂` during truly silent
  vocal regions (a "silence precision" metric we define and justify).
- **Minimal experiment**: 4 sampling policies (uniform / energy-weighted / drop-silent
  / curriculum silent→active), same budget, 2–3 seeds; evaluate overall SI-SDR *and*
  the silence-leakage metric on test-set silent regions (identified from GT stems).
- **Feasibility**: ~8–12 runs ≈ 15–25 T4-hours; the new metric is ~50 lines + tests.
- **Expected outcome / success test**: a two-axis result (quality vs silence
  leakage) showing a real tradeoff; success = the tradeoff plot with CIs.
- **Portfolio story**: an original-but-modest ablation nobody quite reports, plus a
  small, well-motivated custom metric — shows taste for finding real questions in
  plain sight, and it directly improves the shipped karaoke feature (no ghost vocals
  in instrumental breaks).
- **Difficulty/Risk**: **Low-Medium / Low**.

### #9 — The price of real-time: a latency/quality Pareto study for the shipped engine
- **Builds on**: HS-TasNet (Venkatesh et al., ICASSP 2024, arXiv 2402.17701: 23 ms
  latency at 4.65 SDR) and its follow-up (arXiv 2511.13146); SCNet's efficiency
  reporting (arXiv 2401.13276).
- **Hypothesis**: SingNet's offline quality degrades gracefully down to ~1 s
  effective latency (chunk + lookahead) but falls off a cliff below the STFT
  window scale; a quantified Pareto curve lets StemCraft offer an honest "live
  preview" mode.
- **Minimal experiment**: take the trained SingNet; evaluate inference-side variants
  (chunk length {12, 6, 3, 1, 0.5 s} × overlap {0, 25, 50 %} × with/without future
  context) — mostly *inference-only* sweeps; optionally one causal-retrained variant
  (masked convs) as the single training run; measure SI-SDR + real-time factor +
  latency on CPU and on the local AMD GPU.
- **Feasibility**: inference sweeps are cheap (< 5 GPU-hours); 1 optional retrain
  (~8 T4-hours). The RX 9060 XT gets a legitimate role (deployment benchmarking,
  where ROCm friction is low).
- **Expected outcome / success test**: a latency-vs-SI-SDR Pareto figure + an RTF
  table across devices; success = the curve + a shipped `latency_mode` flag in
  `SingNetSeparator`.
- **Portfolio story**: MLOps/deployment thinking — "I characterized the
  quality/latency tradeoff and shipped the knob," which is exactly the systems
  maturity most junior-DS portfolios lack. Weakest on *scientific* novelty of the ten.
- **Difficulty/Risk**: **Low-Medium / Low**.

### #10 — Demucs as teacher: pseudo-label distillation to close the small-model gap
- **Builds on**: BSRNN's semi-supervised fine-tuning via pseudo-labels on unlabeled
  songs (arXiv 2209.15174); MixIT-for-MSS pre-training (arXiv 2505.07631); ensemble
  practice from the MVSep benchmark (arXiv 2305.07489).
- **Hypothesis**: distilling HT-Demucs (teacher) outputs on ~100–200 *unlabeled,
  license-safe* tracks (e.g. FMA CC-licensed subset) as soft targets for SingNet
  closes ≥ 25 % of the SingNet→Demucs SI-SDR gap on MUSDB test, at zero extra
  labeled data.
- **Minimal experiment**: teacher-label an FMA subset once (GPU-cheap); train
  SingNet on (a) MUSDB only, (b) MUSDB + distilled data, (c) distilled only;
  3 seeds on (a)/(b); evaluate on MUSDB test (never distilled).
- **Feasibility**: teacher labeling ~2–4 GPU-hours; 7–9 training runs ≈ 20–30
  T4-hours; extra dataset handling (FMA download/licensing hygiene) adds real
  engineering work — the heaviest option on the menu.
- **Expected outcome / success test**: gap-closure percentage with CIs; success =
  any statistically visible movement (positive or the honest "distillation
  transferred the teacher's errors" negative).
- **Portfolio story**: distillation + semi-supervision are high-demand industrial
  techniques; "I made a small model meaningfully better without new labels" is a
  compelling arc — but it stretches scope (new dataset, teacher pipeline).
- **Difficulty/Risk**: **High / Medium-High** (most moving parts; domain mismatch
  between FMA and MUSDB may mute gains).

---

## Part 3 — Recommendation ranking (decision stays with the user)

| Rank | Option | Why |
|---|---|---|
| **1** | **#2 Augmentation + data-scaling** | Lowest risk, deepest synergy (it *is* PLAN H2 expanded), guaranteed-reportable curves, and the most hiring-manager-native narrative (data-centric ML). The safest excellent choice. |
| **2** | **#5 LoRA for separation** | Best novelty-per-GPU-hour on the menu: a real literature gap, tiny compute, and the most job-relevant technique of the decade. Slightly more plumbing risk than #2. |
| **3** | **#8 Silence/sampling study** | Genuinely original micro-ablation + a custom metric, directly improves the shipped karaoke feature. Modest wow-factor, very sound science. |
| **4** | **#4 Phase-vs-capacity** | The most intellectually satisfying (theory + oracle bounds + a scale question papers skip); medium debugging risk. |
| **5** | **#7 Metric–perception pilot** | Cheapest compute, highest "evaluation literacy" signal; depends on recruiting ~10 raters. **Note: a slim version of #7 pairs well as a free add-on to any other pick**, since Phase 7 already includes a small listening test. |
| 6 | #1 Loss study | Solid and safe, but the most "expected" of the menu. |
| 7 | #6 Robustness | Underrated and industry-flavored; slightly less connected to the karaoke headline. |
| 8 | #3 Mini band-split | Great story if positive; realistic chance of a within-noise null at this scale. |
| 9 | #9 Latency Pareto | Excellent engineering signal, thinnest scientific question. |
| 10 | #10 Distillation | Most exciting ceiling, most scope creep — pick only if timeline is generous. |

**Bottom line**: pick **#2** for maximum safety and coherence, **#5** for maximum
novelty at low compute, or **#4** for maximum depth. Whichever is chosen, its
hypothesis is registered as **H5 in PLAN.md §1.3** before any Phase-R run starts.

---

## Part 4 — Source list (this document)

- BSRNN — Luo & Yu, arXiv 2209.15174. BS-RoFormer — Lu et al., arXiv 2309.02612.
  Mel-RoFormer — Wang et al., arXiv 2310.01809. HT-Demucs — Rouard et al., arXiv
  2211.08553. Hybrid Demucs — arXiv 2111.03600. Demucs v1 — Défossez et al., arXiv
  1911.13254.
- SCNet — arXiv 2401.13276 (ICASSP 2024). DTTNet — Chen et al., arXiv 2309.08684.
  Moises-Light — Hung, Pereira & Korzeniowski, WASPAA 2025, arXiv 2510.06785.
  KUIELab-MDX-Net — arXiv 2111.12203.
- cIRM ResUNet — Kong et al., arXiv 2109.05418.
- Loss/metric study — Gusó, Pons, Pascual, Serrà, ICASSP 2022, arXiv 2202.07968.
  SI-SDR — Le Roux et al. 2019. Bake-Off perception study — WASPAA 2025, arXiv
  2507.06917.
- SDX'23 music track (robust MSS, LabelNoise/Bleeding) — Fabbro et al., arXiv
  2308.06979 (TISMIR 2024). MVSep benchmark — Solovyev et al., arXiv 2305.07489.
- MixIT for MSS — Saijo & Bando, arXiv 2505.07631. MixIT — Wisdom et al., arXiv
  2006.12701.
- HS-TasNet — Venkatesh et al., ICASSP 2024, arXiv 2402.17701; follow-up arXiv
  2511.13146.
- LoRA — Hu et al., arXiv 2106.09685. PEFT for music foundation models — arXiv
  2411.19371. MSS transfer to dialog separation — arXiv 2106.09093. Continual SVS
  adaptation — arXiv 2512.02432.
- MSDM diffusion separation — Mariani et al., arXiv 2302.02257; latent variant arXiv
  2409.06190; diffusion refinement arXiv 2412.06965.
- MedleyVox — Jeon et al., arXiv 2211.07302. Banquet — arXiv 2406.18747. SepACap —
  arXiv 2509.26580. Separation→lyrics-transcription utility — arXiv 2506.15514.
- Open-Unmix — Stöter et al., JOSS 2019 (sigsep/open-unmix-pytorch). Spleeter — JOSS
  10.21105/joss.02154. MUSDB18 — Zenodo 1117372; MUSDB18-HQ — Zenodo 3338373.
