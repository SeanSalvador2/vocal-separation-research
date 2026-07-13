# MASTER PLAN — Direction 01: "Train on what you test?" A controlled loss-function study at small scale

**Status: pre-registered plan. Nothing trained. All GPU work is deferred ("RUN LATER").**
This document is deliberately self-contained: a reader with no other context — or an agent
executing independently — can run the entire study from this file alone. Where it
summarizes project-wide rules it restates them rather than assuming them. Cross-references
([`research/LITERATURE.md`](research/LITERATURE.md), [`../PLAN.md`](../PLAN.md)) are for
depth, not required context.

---

## 0. TL;DR

Train the **same compact magnitude-mask U-Net** on MUSDB18 **five times, changing only the
training loss** — L1-magnitude, MSE-magnitude, log-L1-magnitude, negative time-domain
SI-SDR, and L1 + multi-resolution-STFT auxiliary — 3 seeds each at a reduced step budget
(15 runs), then confirm the top arms at full budget (3 runs), and score everything on a
frozen evaluation protocol (SI-SDR primary, museval SDR secondary, small blind listening
check for artifacts). The pre-registered bet: **training directly on the eval metric
(SI-SDR) does *not* beat plain L1-magnitude — but the MR-STFT term reduces audible
artifacts at equal SI-SDR.** Every outcome (supported / refuted / mixed) is a reportable
finding. Budget ceiling: **≈ 25–40 T4-GPU-hours**, all on Colab Pro.

---

## 1. Problem framing

### 1.1 The question
Music source separation papers disagree about what to optimize. Spleeter trains on
**L1-magnitude**; Open-Unmix on **MSE-magnitude**; Demucs on **waveform L1** (its in-paper
ablation is L1-vs-L2 — the popular "L1 beats SI-SNR" reading is *indirect*, via
SI-SNR-trained Conv-TasNet's audible artifacts); Gusó et al. (ICASSP 2022, arXiv
2202.07968) ran the controlled comparison at their scale and found spectrogram-domain
losses competitive and SDR sometimes misleading (their *specific* ranking is
`[UNVERIFIED]` in this project — the paper body was unreachable; we rely only on the
verified scope). Meanwhile the "obvious" ML move — *train on the metric you report* —
would say: optimize SI-SDR directly.

Nobody has published this comparison for the regime we care about: a **compact (~10 M
param) magnitude-mask U-Net trained on MUSDB18 only**, with seed-noise bands and a
listening check (gap verified 2026-07-13; see [`research/LITERATURE.md §4`](research/LITERATURE.md)).

### 1.2 Why it matters here
Every later SingNet direction (02, 03, 05, 06, 08, 10) trains this U-Net with *some* loss.
Direction 01 decides that default **empirically instead of by folklore**, and produces the
shared training/eval infrastructure every other direction reuses.

### 1.3 Task definition (fixed for the whole study)
Given a mono mixture waveform $x(t)$ (44.1 kHz), estimate the lead-vocals waveform
$\hat v(t)$ and accompaniment $\hat a(t) = x(t) - \hat v(t)$ (2-stem karaoke primitive).
The model predicts a soft ratio mask $M \in [0,1]^{F \times T}$ on the mixture magnitude
spectrogram; reconstruction uses the **mixture phase**:
$\hat V = M \odot |X| e^{i\angle X}$, $\hat v = \text{iSTFT}(\hat V)$.

---

## 2. Pre-registered hypothesis & falsification criteria

> **H-01a (primary).** Training the fixed U-Net on negative time-domain SI-SDR does
> **not** improve vocals SI-SDR at evaluation relative to L1-magnitude training.
> **Decision rule:** on the primary protocol (reduced budget, 3 seeds, 14-track
> validation, §7.2) let $\Delta$ = mean val SI-SDR(sisdr-arm) − mean val SI-SDR(l1-arm),
> and let $\sigma_{\text{seed}}$ = the pooled between-seed std of the two arms.
> - **Supported** if $\Delta \le +\sigma_{\text{seed}}$ (SI-SDR training buys nothing
>   beyond seed noise) **and** the full-budget test-set pair does not reverse this by
>   more than the paired bootstrap 95 % CI.
> - **Refuted** if $\Delta > +\sigma_{\text{seed}}$ on validation **and** the full-budget
>   test pair confirms (paired per-track delta > 0 with 95 % bootstrap CI excluding 0).
> - **Mixed** otherwise (val and test disagree, or seed band overlaps ambiguously) —
>   reported as mixed, with variance discussion.

> **H-01b (secondary).** Adding a multi-resolution STFT auxiliary term (λ = 0.5) to L1
> reduces **audible artifacts** at approximately equal SI-SDR.
> **Decision rule:** "approximately equal SI-SDR" = |SI-SDR(l1mrstft) − SI-SDR(l1)| ≤
> σ_seed on validation. Given that, **supported** if the blind listening check (§7.5)
> prefers the MR-STFT arm on the artifact question in ≥ 60 % of clip-level pairwise
> judgments; **refuted** if ≤ 40 %; **inconclusive** between. If the SI-SDR-equality
> precondition fails, H-01b is scored **not evaluable as posed** and we report the
> quality/artifact tradeoff instead. (Pre-registered because the Bake-Off paper, arXiv
> 2507.06917, found BSS-Eval SDR is already the *best* perceptual proxy for vocals — so
> we deliberately do not expect MR-STFT to move SDR itself.)

Secondary (exploratory, no hypothesis): the relative ordering of MSE-mag vs L1-mag vs
log-L1-mag; the SI-SDR-loss silent-chunk skip rate (feeds Direction 08).

---

## 3. Experimental design

### 3.1 The five arms (only the loss changes — everything else is bit-identical)

| Arm id | Training loss (exact form in §6) | Provenance |
|---|---|---|
| `l1mag` | L1 on masked magnitude | Spleeter's choice |
| `msemag` | MSE on masked magnitude | Open-Unmix's choice |
| `logl1mag` | L1 on log-magnitudes | Gusó-family "LOG" losses |
| `sisdr` | negative time-domain SI-SDR (differentiable iSTFT) | Le Roux 2019 metric-as-loss |
| `l1mrstft` | L1-mag + 0.5 × multi-res STFT (waveform) | Parallel WaveGAN / auraloss |

**Controlled-comparison invariants (test-enforced where possible):** identical
architecture & init (per seed), identical data order & augmentation stream (the
dataloader RNG is a function of (seed, step) only — *not* of the loss), identical
optimizer/schedule/steps/batch, identical eval. The ONLY differences: the loss module,
and (for `sisdr`/`l1mrstft`) a differentiable iSTFT in the training graph.

### 3.2 Run matrix

| Stage | Runs | Budget (steps) | Seeds | Purpose |
|---|---|---|---|---|
| **Sweep (primary evidence)** | 5 arms × 3 seeds = **15** | REDUCED = 16,000 | {0, 1, 2} | H-01a decision on val; seed-noise band |
| **Confirmation** | top-2 arms by mean val SI-SDR (+ `l1mag` if not already included) = **3** | FULL = 40,000 | {0} | stability of ranking; the checkpoints that go to test + listening |
| **Exploratory (optional, budget-gated)** | `l1mrstft` λ ∈ {0.25, 1.0} = 2 | REDUCED | {0} | λ sensitivity; clearly labeled exploratory |

Budget-scaling decision rule (pre-registered, applied once after gate G1 measures real
throughput): if a FULL run projects to **> 12 h on the available GPU**, halve both budgets
(REDUCED 8,000 / FULL 20,000) **for all arms simultaneously** and record the change in the
registry and report. Comparability is preserved because every arm always shares the same
budget.

### 3.3 What "primary evidence" means (honest-stats policy)
- H-01a is decided on the **15-run sweep** (3 seeds → between-seed std per arm; pooled
  σ_seed) using **validation** scores. The 3 confirmation runs guard against
  "ranking flips at full budget"; the single **test pass** (§7.4) provides final honest
  numbers with **paired-by-track bootstrap 95 % CIs and a Wilcoxon signed-rank test** on
  the headline pair. No p-value theater on 5-track windows; no test-set tuning.
- Sweep figures always draw the **seed-noise band** so readers see which differences are
  within noise (project house style).

---

## 4. Data

### 4.1 Dataset & license
**MUSDB18 (compressed STEMS, Zenodo record 1117372, 4.7 GB).** 150 stereo tracks at
44.1 kHz: 100 train / 50 test, stems = vocals, drums, bass, other (+ mixture).
License: research/education only, **non-redistributable** — *no audio is ever committed*;
audio lives on Colab Drive (or local scratch) only. Optional upgrade: MUSDB18-HQ (Zenodo
3338373, access-restricted WAV) — not required for this direction.

### 4.2 Splits (frozen; manifest committed as CSV — no audio, names only)
- **Train: 86 tracks** = the 100 MUSDB train tracks minus the 14 below.
- **Validation: 14 tracks** — the standard `musdb` `split="valid"` list (verbatim,
  verified from the `sigsep-mus-db` source; also in
  [`../00-shared-research/papers/musdb18-dataset.md §3`](../00-shared-research/papers/musdb18-dataset.md)):
  Actions - One Minute Smile; Clara Berry And Wooldog - Waltz For My Victims; Johnny
  Lokke - Promises & Lies; Patrick Talbot - A Reason To Leave; Triviul - Angelsaint;
  Alexander Ross - Goodbye Bolero; Fergessen - Nos Palpitants; Leaf - Summerghost;
  Skelpolu - Human Mistakes; Young Griffo - Pennies; ANiMAL - Rockshow; James May - On
  The Line; Meaxic - Take A Step; Traffic Experiment - Sirens.
- **Test: 50 tracks** (MUSDB test split) — read **exactly once** for this direction (§7.4).
- Manifest file: `01-loss-function-study/configs/splits.csv` (columns: track, split).
  Training/tuning configs must reference train/valid rows only (test rows carry a
  `test` flag the training loader refuses to load — enforced in code + test).

### 4.3 Acquisition & preparation (RUN LATER — CPU/network only, ~30–60 min once)
1. On Colab (or locally): `pip install musdb` then download MUSDB18 via the official
   Zenodo route (`musdb.DB(root=..., download=True)` fetches the 7-second preview set —
   **do not use it**; fetch the full 4.7 GB archive from Zenodo record 1117372, unzip to
   Drive at `MUSDB_ROOT`).
2. Decode stems once to per-track WAV shards (44.1 kHz stereo float32) with
   `scripts/prepare_data.py --musdb-root $MUSDB_ROOT --out $SHARD_ROOT`
   (uses `musdb`/`stempeg`; also writes per-track mono mixdowns and a duration/energy
   index JSON used by the sampler). Decode is the slow step (~30 min); do it once, not
   per-session. Expected output size ≈ 12–15 GB on Drive.
3. `scripts/prepare_data.py --verify` re-checks shard counts, sample rates, and that
   `mixture ≈ vocals + drums + bass + other` (max abs error < 1e-3 for MUSDB18's AAC;
   report the measured value).

### 4.4 Chunking & sampling policy (fixed across arms)
- **Chunk length 6.0 s** (264,600 samples), mono mixdown of the stereo stems
  (mean of channels), sampled with **uniform random start** within a track
  (the UMX-verified baseline policy — deliberately NOT activity-weighted; that question
  belongs to Direction 08).
- **SI-SDR-loss silent-target guard (pre-registered):** chunks whose target-vocal RMS
  < −60 dBFS are **excluded from the batch loss for the `sisdr` arm only** (the SI-SDR
  denominator is singular on silence — Le Roux 2019). The skip is implemented as a
  per-chunk mask inside the loss (batch composition stays identical across arms); the
  skip **rate** is logged per run and reported (it is a finding for Direction 08).
  All magnitude losses process silent chunks normally.

### 4.5 Augmentation (fixed across arms — the verified standard recipe)
Applied on-the-fly, waveform domain, seeded, in this order:
1. **Random source remixing**: vocals from track *i* + accompaniment stems from track
   *j* (per-chunk, within the train split) — the single biggest small-data lever.
2. **Random per-source gain**: amplitude scale U(0.25, 1.25) per source (the exact
   UMX/Demucs code range — *not* dB-symmetric; provenance verified from both repos).
3. **Random sign flip** (p = 0.5, per source).
4. Channel swap is inapplicable (mono mixdown) — documented, not silently dropped.
No pitch/tempo augmentation (heavier; out of scope for this direction).
Augmentation determinism given (seed, step) is unit-tested.

---

## 5. Model (fixed architecture — "SingNet-C1")

Single-channel magnitude-mask U-Net, Spleeter-family, ≈ 10 M params (exact count derived
symbolically in THEORY.md §5 and asserted programmatically in `tests/test_model.py`).

- **STFT front-end** (in-graph, `torch.stft`): n_fft = 4096, hop = 1024, Hann window,
  center=True → 2049 freq bins × 259 frames for a 6 s chunk. The network consumes the
  first **2048** bins (bin 2048 = Nyquist, dropped from the *network input/mask* and
  passed through with mask 1.0 at reconstruction — negligible energy, keeps shapes
  power-of-two) and center-crops/pads frames to **256**.
- **Input featurization**: $\log(1 + |X|)$, standardized per chunk (scalar mean/std over
  the chunk's featurized bins). Identical across arms.
- **Encoder** (5 blocks): Conv2d 5×5 stride 2 pad 2 → BatchNorm → LeakyReLU(0.2);
  channels 1→32→64→128→256→512 (feature map 2048×256 → 64×8 at bottleneck).
- **Decoder** (5 blocks): ConvTranspose2d 5×5 stride 2 → BatchNorm → ReLU, skip
  concatenation from the mirrored encoder level; Dropout2d(0.5) on the first three
  decoder blocks (Spleeter convention); channels mirror the encoder.
- **Head**: 1×1 Conv2d → **sigmoid** → mask $M \in [0,1]^{2048 \times 256}$.
- **Reconstruction**: $\hat V = M' \odot |X| e^{i\angle X}$ (M′ = M with the Nyquist row
  re-appended as 1.0), $\hat v = \text{iSTFT}(\hat V)$ (differentiable; in the training
  graph only for `sisdr`/`l1mrstft`).
- **Init**: Kaiming-normal (fan-in) for convs, per-run seed; BN default init.

Stereo at inference: process L/R independently through the mono model (documented
limitation; primary metric is mono-summed anyway).

---

## 6. Losses (exact, implementation-pinned)

Notation: $|X|, |S|$ = mixture/target-vocal magnitudes (raw, not log); $\hat M$ = predicted
mask; $\hat S_{\text{mag}} = \hat M \odot |X|$; $\hat v, v$ = estimated/target waveforms;
all means are over (bins × frames) or samples of the valid (non-guard-skipped) batch.

1. **`l1mag`**: $\mathcal L = \operatorname{mean}\bigl|\hat S_{\text{mag}} - |S|\bigr|$
2. **`msemag`**: $\mathcal L = \operatorname{mean}\bigl(\hat S_{\text{mag}} - |S|\bigr)^2$
3. **`logl1mag`**: $\mathcal L = \operatorname{mean}\bigl|\log(\hat S_{\text{mag}} + \varepsilon_{\log}) - \log(|S| + \varepsilon_{\log})\bigr|$, $\varepsilon_{\log} = 10^{-5}$
4. **`sisdr`**: $\mathcal L = -\,\overline{\text{SI-SDR}}(\hat v, v)$ over non-skipped chunks, with
   $\alpha = \frac{\hat v^\top v}{\lVert v\rVert^2 + \varepsilon}$,
   $\text{SI-SDR} = 10\log_{10}\frac{\lVert\alpha v\rVert^2 + \varepsilon}{\lVert\alpha v - \hat v\rVert^2 + \varepsilon}$,
   $\varepsilon = 10^{-8}$; silent-target guard per §4.4.
5. **`l1mrstft`**: $\mathcal L = \operatorname{mean}|\hat S_{\text{mag}} - |S|| + \lambda \sum_{m=1}^{3}\bigl(\mathcal L_{\text{sc}}^{(m)} + \mathcal L_{\text{mag}}^{(m)}\bigr)$, $\lambda = 0.5$, computed on waveforms $(\hat v, v)$ with
   $\mathcal L_{\text{sc}} = \frac{\lVert |S_m| - |\hat S_m| \rVert_F}{\lVert |S_m| \rVert_F}$,
   $\mathcal L_{\text{mag}} = \operatorname{mean}\bigl|\log(|S_m| + \varepsilon_{\log}) - \log(|\hat S_m| + \varepsilon_{\log})\bigr|$,
   resolutions (auraloss defaults, code-verified): fft = [1024, 2048, 512],
   hop = [120, 240, 50], win = [600, 1200, 240] (Hann).

**Loss unit tests (CPU, no training):** each loss = its analytic value on constructed
cases; = 0 (or the documented bound) at perfect prediction; SI-SDR is **invariant to
scaling either argument** ($\hat v \to c\hat v$ and $v \to cv$ each leave the value
unchanged — both cancel in the projection form; derived in THEORY.md §3–4 and asserted
numerically); gradients flow to mask parameters through iSTFT for losses 4–5; the
silent-guard skips exactly the constructed silent chunks and the skip rate is reported.

---

## 7. Training, evaluation & analysis protocol

### 7.1 Training configuration (identical for every run)
- Optimizer **AdamW** (lr 1e-3, betas (0.9, 0.999), weight_decay 1e-4).
- Schedule: linear warmup 500 steps → cosine decay to 1e-5 at budget end.
- Batch 16 × 6 s mono chunks, **AMP mixed precision** (fp16 with GradScaler; bf16 if the
  GPU supports it), grad-clip (global L2 norm) 5.0.
- OOM fallback (pre-registered, applies to all arms together): batch 8 + gradient
  accumulation 2 (identical effective batch 16).
- Checkpoint + optimizer + scaler + RNG state every 1,000 steps to Drive; runs are
  resumable to the step (resume determinism unit-tested with mocked state on CPU).
- Validation SI-SDR (full-track, §7.2) every 2,000 steps; cheap proxy (batch val loss)
  every 500. **Best-checkpoint = highest val SI-SDR**; sweeps additionally keep the
  final-step checkpoint (both reported; selection uses best).
- Registry: every run appends to `01-loss-function-study/results/registry.csv`
  (run_id, arm, seed, budget, config_hash, git_commit, gpu, wall_clock_h, steps_done,
  best_val_sisdr, final_val_sisdr, sisdr_skip_rate, checkpoint_path). The registry and
  eval CSVs are committed; checkpoints are not.

### 7.2 Validation protocol (the selection metric; no test contact)
Full-track inference on the **14 validation tracks**: mono mixdown, chunked STFT
inference with 25 % overlap-add (raised-cosine crossfade), mixture-phase iSTFT →
per-track vocals SI-SDR (Le Roux definition; accompaniment SI-SDR also logged) → report
**mean over the 14 tracks**. Runtime ≈ 1–2 min/checkpoint on T4 — cheap enough for
every-2k-steps use.

### 7.3 Oracle & floor anchors (computed once, CPU, RUN LATER ~30 min)
- **Do-nothing floor**: mixture-as-vocals SI-SDR on val + test.
- **Oracle IRM** ($M = |S| / (|S| + |A| + \varepsilon)$) and **oracle IBM**
  (1 if $|S| > |A|$): upper bounds of the mask family at our STFT resolution, val + test.
  These two lines appear on every results figure (headroom context; PLAN H4 anchor).

### 7.4 Test pass (**exactly one** for the whole direction)
After confirmation runs finish and analysis of val results is frozen in the notebook:
score the **3 full-budget checkpoints** + do-nothing + IRM/IBM oracles on all **50 test
tracks** (same §7.2 inference). Outputs, committed:
`results/test_per_track.csv` (track × system × {SI-SDR vocals, SI-SDR accomp, SI-SDRi}),
`results/test_museval.csv` (museval BSS-Eval SDR, median-of-frames/median-of-tracks —
clearly labeled secondary, never mixed with SI-SDR numbers).
Stats: paired per-track deltas for (sisdr-arm − l1-arm) and (l1mrstft − l1): bootstrap
95 % CI (10,000 resamples over tracks) + Wilcoxon signed-rank p (two-sided). Anything
else is descriptive.

### 7.5 Listening check (H-01b evidence; human time, zero GPU; RUN LATER)
- Material: 5 test tracks (the 5-track StemCraft continuity list if available in the
  test split — else the 5 median-SI-SDR test tracks; fixed before listening), loudest
  12 s vocal window each; render **isolated vocals** and **karaoke** (mix − vocals) for
  the 3 full-budget systems (level-matched to −23 LUFS).
- Design: blind, randomized A/B pairs (l1 vs l1mrstft primary; l1 vs sisdr secondary),
  n ≥ 5 raters, one fixed question: *"Which version has fewer artifacts (musical noise,
  gurgling, phasiness)?"* + optional 0–100 quality slider.
- Analysis: preference counts + per-clip majorities; report rater agreement; framed as
  qualitative pilot evidence (n is small), never as statistical proof.
- Ethics/practicality: raters are friends/classmates; 12 s research-context excerpts;
  no audio redistributed beyond the rating session.

---

## 8. Compute budget & run book (Colab Pro)

| Item | Runs × time (T4 est.) | Subtotal |
|---|---|---|
| Sweep (REDUCED 16k steps) | 15 × ~1.3–1.8 h | 20–27 h |
| Confirmation (FULL 40k) | 3 × ~3.5–4.5 h | 10–14 h |
| Val evals during training | amortized in the above | — |
| Oracles + test pass + renders | ~1–2 h total | 1–2 h |
| **Ceiling** | | **≈ 31–43 T4-h** (less on L4/A100) |

If gate-G1 throughput shows the ceiling exceeding ~45 T4-h, the §3.2 halving rule
applies (→ ≈ 16–22 T4-h). Sessions are disposable: resumable checkpoints + registry
mean a disconnect costs minutes.

**Run book (exact commands; every step marked RUN LATER until Sean has Colab Pro):**
```bash
# 0. one-time env (Colab cell provided in notebooks/; local equivalent):
pip install -r requirements.txt          # pinned; CPU-ok
# 1. one-time data prep (CPU + network, ~1 h):  [RUN LATER]
python scripts/prepare_data.py --musdb-root $MUSDB_ROOT --out $SHARD_ROOT
python scripts/prepare_data.py --verify --out $SHARD_ROOT
# 2. gate G1 — proof-it-trains (GPU, < 1 h):    [RUN LATER]
python -m singnet.train --config 01-loss-function-study/configs/smoke_overfit.yaml
#    PASS iff: single-chunk overfit reaches train SI-SDR > +20 dB within 2k steps
#    AND a 5-track mini-run beats the do-nothing floor on val-mini. Record steps/s.
# 3. the 15-run sweep (GPU):                    [RUN LATER]
python scripts/run_sweep.py --stage reduced     # iterates the 15 configs, resumable
# 4. analysis freeze: run notebook 03 sections 1–6 (CPU) → picks top-2 arms
# 5. confirmation runs (GPU):                   [RUN LATER]
python scripts/run_sweep.py --stage full        # 3 configs
# 6. oracles + single test pass (GPU minutes / CPU ok):  [RUN LATER]
python scripts/evaluate.py --checkpoints <3 ckpts> --split test --oracles --museval
# 7. listening kit render (CPU):                [RUN LATER]
python scripts/render_listening_kit.py --checkpoints <3 ckpts>
```

---

## 9. Go / no-go gates

| Gate | When | Criterion | On fail |
|---|---|---|---|
| **G0** | now (no GPU) | `pytest` green on CPU: STFT round-trip < −60 dB; loss analytic tests; augmentation determinism; model shapes + param count; registry/resume logic; split-manifest guard (train loader refuses test rows) | fix before any GPU spend |
| **G1** | first GPU session | smoke_overfit passes (§8 step 2); measured steps/s recorded; budget rule applied | debug pipeline; do not start sweep |
| **G2** | after sweep | pooled σ_seed < the between-arm spread of the top-3 arms (i.e., the experiment can resolve differences); no run diverged unexplained | add seeds 3,4 to the two closest arms only; re-assess |
| **G3** | before test pass | registry complete for 18 runs; analysis of val frozen & committed; test paths never appeared in any training config (grep-audited) | fix, re-freeze |

---

## 10. Deliverables & repository layout (this direction)

```
01-loss-function-study/
  MASTER_PLAN.md            ← this file
  THEORY.md                 ← stage B (math: STFT/COLA, masks, the 5 losses derived,
                               SI-SDR projection + pathologies, MR-STFT, U-Net param
                               count derivation, seed/CI statistics)
  theory/theory.tex         ← LaTeX mirror of THEORY.md (compilable article)
  research/                 ← Phase-0 literature (done)
  notebooks/
    01_data_and_eda.ipynb           (un-run; acquisition, license, splits, EDA that
                                     motivates loss choices: magnitude/dynamic-range
                                     distributions, vocal activity, silence stats)
    02_pipeline_and_model.ipynb     (un-run; STFT round-trip, chunking, augmentation
                                     gallery, mask targets, architecture walkthrough,
                                     param table, smoke-overfit cell [RUN LATER])
    03_loss_study_experiments.ipynb (un-run; pre-registration recap, run matrix,
                                     training cells [RUN LATER], registry-driven
                                     analysis: ranking table + seed bands, curves,
                                     mask/spectrogram galleries, oracle lines, test
                                     pass, pre-written interpretation branches,
                                     conclusions)
  configs/
    splits.csv, base.yaml, smoke_overfit.yaml,
    {l1mag,msemag,logl1mag,sisdr,l1mrstft}_seed{0,1,2}_reduced.yaml   (15)
    {top2 + l1mag}_seed0_full.yaml                                     (3, generated)
    l1mrstft_lam{025,100}_seed0_reduced.yaml                           (2, exploratory)
  results/                  ← registry.csv + eval CSVs (committed; no audio/ckpts)
  paper/PAPER.md            ← stage E (scaffold with placeholders + pre-written
                               interpretations for every outcome)
  SEAN-README.md            ← stage E (plain-language guide)
singnet/                    ← shared package (repo root; reused by directions 02–10)
  audio/stft.py  data/{musdb_dataset,augment,manifest}.py  models/unet.py
  losses/{l1mag,msemag,logl1mag,sisdr,mrstft,registry}.py
  metrics/{si_sdr,museval_wrap}.py  train/{loop,registry}.py
  eval/{overlap_add,evaluate}.py  utils/{seed,config}.py
tests/                      ← CPU-only, synthetic fixtures, no MUSDB, no training
requirements.txt            ← pinned (CPU-installable; torch CPU wheel ok)
```

**Code contracts (so THEORY/notebooks/scripts agree):**
- `singnet.losses.build(name: str, **kw) -> nn.Module` with
  `forward(mask, mix_mag, tgt_mag, mix_stft, tgt_wave, mix_wave) -> (loss, aux: dict)`
  — one uniform signature; magnitude-only losses ignore waveform args; `aux` carries
  e.g. `skip_rate`.
- `singnet.data.MusdbChunks(manifest, split, seed, chunk_s=6.0, augment=True)` —
  deterministic per (seed, step); synthetic-fixture constructible for tests.
- `singnet.train.run(config_path) -> RunResult` and CLI `python -m singnet.train
  --config …` — config-hash keyed, resumable.
- `singnet.eval.evaluate(checkpoint, split, protocol=...) -> DataFrame` and
  `scripts/evaluate.py` CLI.
- Every equation in THEORY.md cross-references the implementing function
  (`singnet/losses/sisdr.py::SiSdrLoss`, etc.).

---

## 11. Timeline (Sean's part-time weeks, once GPU exists)

| Week | Work | Gate |
|---|---|---|
| 1 | data prep; G0 already green; run G1 smoke; start sweep | G1 |
| 2 | finish 15 sweep runs (2–3 Colab sessions); freeze val analysis | G2 |
| 3 | 3 confirmation runs; oracles; the one test pass; render listening kit; run listening check | G3 |
| 4 | fill paper placeholders from CSVs; finalize figures; write verdicts | — |

---

## 12. Risks & mitigations (direction-specific)

- **`sisdr` arm trains poorly for silence-related reasons, not loss-quality reasons** →
  the −60 dBFS guard (§4.4) + logged skip rate; if skip rate > 20 % the finding is
  reframed as "SI-SDR loss is fragile on real music data" (still a result).
- **MR-STFT slows training** (extra iSTFT + 3 STFTs) → measured steps/s per arm recorded
  in the registry; comparisons are at equal *steps* (pre-registered; equal-wall-clock is
  reported as a secondary sensitivity note).
- **Seed noise swamps arm differences (G2 fail)** → escalation rule fixed in G2: 2 extra
  seeds on the two closest arms only (bounded +4 runs ≈ +6 T4-h).
- **AMP instability for `logl1mag`/`sisdr`** (log/div near zero in fp16) → losses always
  computed in fp32 (autocast-disabled region) — pinned in the loss contract; unit-tested
  for finiteness at extreme inputs.
- **museval version drift** → museval pinned in requirements; version recorded in eval CSVs.
- **Colab disconnects** → 1k-step checkpointing + resumable sweep runner; registry marks
  partial runs; nothing is ever restarted from scratch.

---

## 13. THEORY.md outline (stage-B spec — Opus writes to this)

1. Signals, STFT/iSTFT, windowing, COLA condition (proof sketch for Hann, hop = n_fft/4);
   why magnitude + mixture phase; the Nyquist-row handling used by the code.
2. Mask family: ratio mask, sigmoid bound, IRM/IBM oracles (definitions used in §7.3);
   why oracle IRM upper-bounds this model family; where mask > 1 would matter (cIRM
   context, arXiv 2109.05418).
3. The five losses, each: definition → gradient sketch → what it rewards/ignores →
   provenance. For SI-SDR: full projection derivation (α as least-squares projection),
   scale-invariance proof, silence singularity, ε-guard behavior.
4. Train-metric vs eval-metric: why arg-max of one loss need not arg-max SI-SDR under a
   bounded-mask, mixture-phase parameterization (the reachable-set argument); what the
   Bake-Off correlation results imply for interpreting H-01b.
5. U-Net parameter-count derivation (per-layer table summing to the exact count) +
   receptive field; verified against `tests/test_model.py`.
6. Statistics of the design: seed noise pooling, paired bootstrap over tracks, Wilcoxon;
   why sweeps use 3 seeds and what σ_seed does/doesn't license.
7. `theory/theory.tex`: same content, compilable standalone (documentclass article,
   amsmath; no external figures required to compile).

## 14. Notebook specs (stage-C spec — Opus writes to these)

Global rules: **no cell is executed**; every GPU/network cell opens with a bold
**⚠️ RUN THIS LATER** banner + expected runtime/GPU (from §8); markdown explains *what
will happen and why* at each step in plain words; every figure cell states what the
figure will show and how to read it; notebooks import from `singnet/` (no load-bearing
logic inline); a Colab bootstrap cell (pip install pinned reqs + Drive mount) tops each
notebook; a "state of this notebook" cell links MASTER_PLAN sections.
- **01_data_and_eda**: license & acquisition (RUN LATER), split manifest render, EDA:
  track duration/loudness tables, vocal-activity ratio per track, magnitude & log-mag
  histograms (→ motivates log-domain losses), silent-region stats (→ motivates the
  sisdr guard; feeds Direction 08), spectrogram gallery. Interpretation stubs tied to
  design decisions.
- **02_pipeline_and_model**: STFT/iSTFT round-trip demo (synthetic, CPU-runnable later),
  chunk + augmentation gallery (before/after waveforms & spectrograms), mask-target
  visualization, architecture walkthrough with the THEORY §5 param table, gradient-flow
  smoke description, smoke_overfit cell (RUN LATER, < 1 GPU-h, G1 criteria stated).
- **03_loss_study_experiments**: H-01a/H-01b verbatim from §2 (pre-registration recap),
  run-matrix table, sweep launch cells (RUN LATER, per-stage runtime), registry-driven
  analysis: ranking table with seed bands, training curves, per-track val scatter,
  mask/spectrogram error galleries per arm, oracle/floor lines, confirmation + single
  test pass (RUN LATER), paired-stats cells, listening-kit render + result entry form,
  **interpretation section with pre-written branches** (supported/refuted/mixed for
  H-01a and H-01b, verbatim-adaptable from PAPER.md), conclusions + "what this changes
  for directions 02–10".

---

## 15. Outcome interpretation matrix (pre-registered readings; full prose in paper/PAPER.md)

| Outcome | Reading | Consequence for the project |
|---|---|---|
| H-01a supported | Training on the eval metric buys nothing at compact scale; consistent with spectrogram-loss competitiveness (Gusó, verified scope) and Demucs practice | keep `l1mag` as the project default loss |
| H-01a refuted | Direct metric optimization helps at small capacity; the literature tension is scale-dependent | adopt `sisdr` (with guard) as default; flag re-run implications for later directions |
| H-01a mixed | val/test or seed instability | keep `l1mag` (simplicity prior); report variance honestly |
| H-01b supported | SDR-equal artifact gains are real; metric blind spot confirmed at small scale (Bake-Off-consistent) | adopt MR-STFT aux for shipped-model training |
| H-01b refuted/inconclusive | artifact differences inaudible at this scale or swamped by rater noise | drop the aux term (simplicity); document |
| `sisdr` skip rate high (> 20 %) | SI-SDR loss fragility on real music is itself a finding | feeds Direction 08's silence framing |

---

*Frozen on 2026-07-13. Any deviation during execution must be recorded in
`results/DEVIATIONS.md` with date + reason (empty file created with the repo).*
