# MASTER PLAN — Direction 05: LoRA for source separation — parameter-efficient fine-tuning of a pretrained separator

**Status: pre-registered plan. Nothing trained, no weights downloaded. All GPU/network
work is deferred ("RUN LATER").** Self-contained by design; cross-references
([`research/LITERATURE.md`](research/LITERATURE.md),
[`../00-shared-research/papers/openunmix2019.md`](../00-shared-research/papers/openunmix2019.md),
[`../PLAN.md`](../PLAN.md)) add depth, not required context.

---

## 0. TL;DR

Take the pretrained **Open-Unmix vocals model (`umxhq`)** — the canonical MIT-licensed,
MUSDB-trained separator — and adapt it to two shifted domains with **five recipes**:
zero-shot (no training), head-only, **LoRA rank 4**, **LoRA rank 16**, and full
fine-tune. Measure (a) **adapted-domain gain** and (b) **source-domain regression**
(catastrophic forgetting). Pre-registered bets: **LoRA r=16 recovers ≥ 90 % of the
full-fine-tune gain at < 5 % trainable parameters (H-05a), and forgets less (H-05b)**.
As of the 2026-07-13 verification sweep, **no published LoRA/PEFT-for-source-separation
study exists** — this is the project's strongest novelty claim, framed honestly as
"first *careful small-scale* look," not "first ever." 12 fine-tuning runs (1–1.5 GPU-h
each) + LR probes ≈ **18–24 T4-hours**. The engineering risk is LoRA on an LSTM
(non-standard); it is de-risked by a B=0-identity unit test and a mock-UMX test suite
that runs with **no weight download**.

---

## 1. Problem framing

### 1.1 The question
LoRA is the default adaptation technique for large language and vision models, and PEFT
reaches near-full-fine-tune quality at < 1 % trainable parameters for music *tagging*
foundation models (arXiv 2411.19371). For music *source separation* there are no
published PEFT numbers at all (gap-check logged 2026-07-13; the nearest works are
full-fine-tune transfer, arXiv 2106.09093, and LoRA for beat tracking, 2503.10086).
Yet the practical use case is obvious: a user's library is *shifted* from the training
distribution (low-bitrate rips, specific genres, live recordings), full fine-tuning of
even a small separator per-domain is wasteful, and forgetting the source domain is a
real product regression. Does the low-intrinsic-rank story hold for a **recurrent**
separation model?

### 1.2 Why Open-Unmix as the host
- MIT license, published weights (`umxhq`), architecture verified from code in Phase 0.
- Small (≈ 8–9 M params: exact count computed programmatically at G1 and recorded),
  fine-tunes in ~1 GPU-hour — the cheapest credible host.
- Its BiLSTM core makes the study *more* interesting: LoRA on recurrent weights is
  under-documented, so even "LoRA underperforms on LSTMs" is a useful negative.
- Direct project relevance: umxhq is the umbrella plan's Track-B engine candidate.

### 1.3 What we deliberately do NOT vary
One host model, one loss (UMX's native **MSE on magnitudes** for every recipe — this is
a *recipe* comparison, not a loss study; Direction 01 owns losses), one FT budget, the
UMX-native stereo data pipeline with its published augmentation recipe (gain
U(0.25, 1.25), channel swap p = 0.5, cross-track remix — code-verified). The Wiener
post-filter is OFF for all recipe comparisons (constant offset; one with-Wiener line is
reported once, descriptively, at the end).

---

## 2. Pre-registered hypotheses & decision rules

Notation: for domain D and recipe R, let $g_D(R)$ = mean vocals SI-SDR on D's
**target-domain test tracks** minus zero-shot's score on the same tracks
("adapted-domain gain"), and $f(R)$ = zero-shot SI-SDR on the **standard** test set
minus R's SI-SDR on the standard test set after adapting ("forgetting"; positive =
regression). σ_seed = between-seed std pooled over the two 3-seed cells (LoRA-16 and
full-FT on T1).

> **H-05a (efficiency).** On the primary domain T1:
> $g_{T1}(\text{LoRA-16}) \ge 0.9 \cdot g_{T1}(\text{full-FT})$, with LoRA-16 training
> < 5 % of the host's parameters.
> - **Precondition:** $g_{T1}(\text{full-FT}) > \max(2\sigma_{\text{seed}}, 0.3\text{ dB})$
>   — the domain shift must be big enough that adaptation does something; otherwise
>   H-05a is **not evaluable** (pre-registered branch: "the shift was too mild"), and
>   the secondary domain T2 becomes primary for this hypothesis.
> - **Supported / refuted** by the 3-seed means with a seed-propagated CI on the gain
>   ratio (delta method in THEORY §6); ratio ≥ 0.9 with CI excluding < 0.75 → clean
>   support; ratio < 0.9 with CI excluding ≥ 0.9 → refuted; else mixed.
> **H-05b (forgetting).** $f(\text{LoRA-16}) < f(\text{full-FT}) - \sigma_{\text{seed}}$
> on T1 (LoRA regresses less on the source domain), with the same-direction check on T2
> reported descriptively.

**Descriptive (no hypotheses):** the **quality-vs-trainable-params curve** (zero-shot →
head → LoRA-4 → LoRA-16 → full; the headline figure); LoRA-4 vs LoRA-16 (rank
sensitivity); per-recipe wall-clock and peak-VRAM (the practical PEFT sell); the
engineering claim that LoRA-wrapped UMX with B = 0 reproduces zero-shot *exactly*
(unit-tested — this is also each LoRA run's guaranteed starting point).

---

## 3. Experimental design

### 3.1 The five recipes (what trains, exactly)

| Recipe | Trainable | Frozen | Expected trainable share* |
|---|---|---|---|
| `zeroshot` | nothing | all | 0 % |
| `head` | `fc3`, `bn3`, `output_scale`, `output_mean` | rest | ≈ 18 % |
| `lora4` | LoRA A/B (r = 4, α = 2r) on fc1, fc2, fc3 + all LSTM `weight_ih/hh` (both directions, 3 layers); plus input/output scale+mean | all base weights | ≈ 1.2 % |
| `lora16` | as above with r = 16 | as above | ≈ 4.9 % |
| `full` | everything | — | 100 % |

\* Shares from the Phase-0 accounting on the code-default shapes (fc1: 2·nb_bins→512;
LSTM 3×2 directions, weight_ih 1024×512 / weight_hh 1024×256; fc2 1024→512; fc3
512→2·nb_output_bins; nb_bins ≈ 1487 for umxhq's 16 kHz bandwidth crop) — **recomputed
programmatically against the real checkpoint at G1 and pinned in the report**; the
< 5 % claim of H-05a uses the recomputed number. BatchNorm layers stay in train mode
for all trained recipes (running stats adapt; pre-registered, applies uniformly).

**LoRA mechanics (pinned):** $h = W_0 x + \frac{\alpha}{r} BAx$, $A \sim \mathcal N(0,
\sigma^2)$, $B = 0$, α = 2r fixed (no α tuning). LSTM wrapping via PyTorch weight
**parametrization** on the flat `weight_ih_l{k}`/`weight_hh_l{k}` (+`_reverse`)
matrices — LoRA on the gate-stacked (4h × d) matrix, valid and derived in THEORY §3.
Merge-back (`W = W_0 + (α/r)BA`) implemented + round-trip-tested (merged model ==
wrapped model outputs).

### 3.2 The two domains (constructed from data we have; no new licenses)

- **T1 — codec-degraded audio (primary).** Every stem re-encoded AAC **64 kbps**
  stereo (ffmpeg, deterministic settings pinned in `scripts/make_domains.py`), then
  mixtures formed as sums of degraded stems — supervised pairs stay exactly additive.
  This is the "user's low-bitrate rip" scenario; umxhq never saw such artifacts
  (HQ-trained). Applies to train/val/test alike (T1-test = degraded 50-track test set).
- **T2 — secondary, resolved at G0b by a deterministic rule:** IF an official
  per-track genre labeling can be materialized at prep time (from `musdb` track
  metadata or the Zenodo record's tracklist), T2 = **largest genre cluster** with
  ≥ 14 train + ≥ 5 test tracks (adapt-to-genre; source regression = the other genres'
  test tracks). ELSE T2 = **live-noise domain**: pink noise added to mixtures at
  12 dB SNR (targets stay clean; a separate+denoise adaptation). The rule, not the
  choice, is pre-registered; genre availability is currently **unverified** — the
  fallback avoids a hidden dependency.

### 3.3 Run matrix

| Domain | Recipes × seeds | Runs |
|---|---|---|
| T1 | head × 1, lora4 × 1, **lora16 × {0,1,2}**, **full × {0,1,2}** | 8 |
| T2 | head, lora4, lora16, full × 1 (seed 0) | 4 |
| LR probes (T1 only; §3.4) | 4 recipes × 3 LRs × 500 steps | ~12 short probes |

Fine-tune budget: **6,000 steps**, batch 16 × 6-s stereo chunks, identical for every
trained recipe; validation (on the domain's transformed 14-track val split) every 500
steps; best-checkpoint by val SI-SDR. Seeds control data order/augmentation and LoRA-A
init. T2 reuses T1's chosen LRs (pre-registered — T2 is directional evidence, not an
independent tuning exercise).

### 3.4 LR fairness protocol (pre-registered)
Each trained recipe gets its own LR, chosen by a 500-step probe on T1 val loss over a
recipe-appropriate grid: full-FT ∈ {3e-5, 1e-4, 3e-4}; head/LoRA ∈ {3e-4, 1e-3, 3e-3}.
Rationale (THEORY §5): PEFT methods systematically want larger LRs; a single shared LR
would bias the comparison against one side. Probes are logged in the registry; chosen
LRs frozen before the 12 main runs.

### 3.5 Invariants
Identical across recipes within a domain: pretrained init (bit-identical checkpoint),
data order & augmentation stream per seed, loss (UMX-native MSE), optimizer (Adam),
schedule (constant LR after 200-step warmup), steps, batch, eval protocol. The ONLY
differences: which parameters receive gradients (and the LoRA reparametrization).

---

## 4. Data & the pretrained checkpoint

- Base data: MUSDB18 (Zenodo 1117372, research-only, nothing committed), the project's
  86/14/50 split, **stereo** WAV shards (the D01 prep already writes stereo; the UMX
  pipeline consumes stereo directly).
- Domain materialization: `scripts/make_domains.py --domain t1_aac64` (ffmpeg required;
  availability checked at G0b, RUN LATER) and `--domain t2_*` per the §3.2 rule;
  deterministic, writes alongside the shards with a domain suffix + a manifest;
  ~30 min CPU each.
- Checkpoint: `umxhq` vocals target via `torch.hub.load('sigsep/open-unmix-pytorch',
  'umxhq')` at G1 (RUN LATER; ~network MBs). **No tests depend on the download** — the
  CPU test suite uses a **mock UMX** built from the verified code-default shapes.
- License hygiene: umxhq weights are MIT (verified Phase 0); MUSDB audio never
  committed; degraded-domain audio lives with the shards on Drive only.

---

## 5. Training & evaluation protocol

- **Trainer:** `singnet/peft/finetune_umx.py` — a dedicated loop for the UMX family
  (reuses the project's registry/checkpoint/seed utilities; deliberately does NOT
  generalize `train/loop.py`, to keep Directions 01–03 risk-free). Registry:
  `05-lora-source-separation/results/registry.csv` with columns
  (run_id, domain, recipe, rank, lr, seed, trainable_params, trainable_share,
  steps, wall_clock_h, peak_vram_gb, best_val_sisdr, checkpoint_path).
- **Inference/eval:** UMX's native full-track spectrogram path with mixture-phase
  iSTFT, **no Wiener** (§1.3); vocals SI-SDR per track via the project's tested
  metric; museval SDR recorded as the secondary table.
- **Validation** (all decisions): the domain-transformed 14-track val split.
- **Test (one consolidated session at the very end):** all recipe checkpoints +
  zero-shot evaluated on (a) their domain's transformed 50-track test set (adapted
  gain) and (b) the standard test set (forgetting) — a single pass writing one CSV
  (`results/test_matrix.csv`); no tuning after it. This is more test exposure than
  Directions 01–03 (15 cheap evals), pre-registered here because *adaptation quality
  on held-out tracks IS the object of study*, and every decision (LRs, checkpoints)
  freezes on validation beforehand.
- **Stats:** 3-seed cells give σ_seed; H-05a's gain ratio gets a delta-method CI
  (THEORY §6); per-track paired deltas + bootstrap CIs for the two pre-registered
  comparisons (lora16 vs full: gain on T1, forgetting on standard).

---

## 6. Compute budget & run book

| Item | Cost (T4 est.) |
|---|---|
| LR probes: ~12 × 500 steps | ~1.5–2 h |
| T1 runs: 8 × 6 k steps | ~8–12 h |
| T2 runs: 4 × 6 k steps | ~4–6 h |
| Consolidated eval session (15 evals) | ~1–1.5 h |
| Domain materialization (CPU) | ~1 h, no GPU |
| **Ceiling** | **≈ 18–24 T4-h** |

**Run book (all RUN LATER; prerequisites: D01 data prep done):**
```bash
# 0. domains + T2 rule resolution (CPU):                       [RUN LATER]
python scripts/make_domains.py --domain t1_aac64 --shards $SHARD_ROOT
python scripts/make_domains.py --resolve-t2 --shards $SHARD_ROOT   # prints genre|noise decision + manifest
# 1. G1 — checkpoint sanity (GPU minutes):                     [RUN LATER]
python -m singnet.peft.finetune_umx --sanity   # loads umxhq, recomputes trainable-share table,
#    zero-shot val SI-SDR on standard + T1 val (must beat do-nothing by > 3 dB on standard)
# 2. LR probes (GPU ~2 h):                                     [RUN LATER]
python scripts/run_sweep.py --direction 05 --stage probes
# 3. freeze LRs (writes configs), then the 12 runs:            [RUN LATER]
python scripts/run_sweep.py --direction 05 --stage main
# 4. consolidated test session (GPU ~1 h):                     [RUN LATER]
python scripts/evaluate.py --direction 05 --test-matrix
# 5. analysis: notebook 02 (CPU).
```

---

## 7. Go / no-go gates

| Gate | When | Criterion | On fail |
|---|---|---|---|
| **G0** | now (CPU) | full-repo `pytest` green incl.: LoRA linear math (B=0 identity; merge round-trip; α/r scaling; r(d_in+d_out) counting); **parametrized-LSTM B=0 identity** on a small random LSTM; freezing masks per recipe (exact trainable-name sets); mock-UMX trainable-share table ≈ {head ~18 %, lora4 ~1.2 %, lora16 ~4.9 %} within ±0.3 pp; domain-op determinism (noise path); config schema/hash | fix before GPU |
| **G0b** | after data prep | domains materialized; T2 rule resolved & logged; additivity re-verified on degraded stems | fix pipeline |
| **G1** | first GPU session | umxhq loads; recomputed trainable shares pinned; zero-shot sanity (> 3 dB over do-nothing on standard val); **H-05a precondition probe**: zero-shot on T1 val vs standard val gap recorded | if T1 gap ≈ 0, invoke the pre-registered "shift too mild" path (T2 promotes to primary) |
| **G2** | after probes | LRs chosen & frozen; no probe diverged | widen grid one notch (pre-registered: one retry) |
| **G3** | before test session | registry complete (12 runs); analysis frozen on val; test untouched | fix, re-freeze |

---

## 8. Deliverables & layout (this direction)

```
05-lora-source-separation/
  MASTER_PLAN.md            ← this file
  THEORY.md + theory/theory.tex   ← stage B (§9)
  research/                 ← Phase-0 literature (done)
  notebooks/
    01_lora_anatomy.ipynb            ← stage C (§10)
    02_lora_adaptation_experiments.ipynb
  configs/
    base.yaml; probes_*.yaml (generated); t1_{head,lora4,lora16,full}_seed*.yaml (8)
    t2_{head,lora4,lora16,full}_seed0.yaml (4)   # t2 op field filled at G0b
  results/                  ← registry.csv, test_matrix.csv, DEVIATIONS.md (empty now)
  paper/PAPER.md, SEAN-README.md  ← stage E
singnet/peft/__init__.py, lora.py, umx_wrapper.py, finetune_umx.py   ← stage D (tested)
scripts/make_domains.py    ← stage D
```

**Code contracts (stage-D deltas only; Directions 01–03 untouched semantically):**
- `singnet.peft.lora.LoRALinear(base: nn.Linear, r, alpha)` and
  `wrap_lstm_lora(lstm: nn.LSTM, r, alpha)` (parametrization-based; both directions,
  all layers), `merge_lora(module) -> plain module`, `trainable_report(model) ->
  DataFrame` (name, shape, trainable, share).
- `singnet.peft.umx_wrapper.load_umxhq(device, mock: bool = False)` — `mock=True`
  builds the shape-faithful random-weight stand-in used by every unit test (no
  network); `apply_recipe(model, recipe, r) -> model` (freezing + wrapping; exact
  trainable-name sets asserted in tests).
- `finetune_umx.py`: CLI per §6; UMX-native stereo chunk pipeline with the verified
  UMX augmentation recipe; registry/checkpoint/resume via existing utilities.
- `make_domains.py`: `--domain t1_aac64` (ffmpeg CLI constructed + version-pinned
  settings; execution RUN LATER), `--domain t2_noise12db` (pure numpy, seeded),
  `--resolve-t2` (the §3.2 rule; prints + writes decision file).
- `run_sweep.py --direction 05 --stage probes|main`; `evaluate.py --direction 05
  --test-matrix`.

---

## 9. THEORY.md outline (stage-B spec)

1. **Transfer & PEFT background**: full-FT vs head vs LoRA; the low-intrinsic-rank
   hypothesis; what "adaptation" means for a regression (mask-estimation) model vs the
   classification settings PEFT is validated on.
2. **LoRA math, complete**: $\Delta W = \frac{\alpha}{r}BA$; init; merge; trainable
   counting $r(d_{in}+d_{out})$; why α = 2r removes a tuning axis (scale argument).
3. **LoRA on a BiLSTM (the novel plumbing)**: LSTM gate equations with the stacked
   (4h × d) weight layout; what a rank-r update on the *stacked* matrix means (shared
   low-rank structure across gates — derive; contrast with per-gate wrapping, and why
   we pre-register stacked as the primary and note per-gate as future work);
   bidirectionality and layer indexing; the parametrization mechanism and its
   correctness condition (B = 0 ⇒ exact identity, testable).
4. **The UMX host, formalized**: the §3.1 architecture equations; where each recipe's
   trainable set sits in the computation graph; the head-only ≈ 18 % accounting; the
   full trainable-share table (mock-derived now, checkpoint-recomputed at G1).
5. **Domain shift & fairness**: codec degradation as covariate shift on both inputs
   and targets (with stem-wise re-encode preserving additivity — proof one line);
   noise domain as input-only shift; the per-recipe-LR fairness argument (why one
   shared LR biases; probe protocol).
6. **Statistics**: gain-ratio delta-method CI; forgetting metric properties (why we
   anchor to zero-shot, not to published numbers); seed policy; what the 15-eval test
   session does/doesn't license.
7. `theory/theory.tex`: compilable standalone mirror.

## 10. Notebook specs (stage-C spec)

Global rules identical to prior directions (un-run; RUN-LATER banners with §6 runtimes;
logic in `singnet/`; Colab bootstrap; cross-link D01 data prep).
- **01_lora_anatomy.ipynb**: UMX architecture tour on the **mock** model (shape table,
  where recipes attach); LoRA visualized (the B=0 start, rank as a dial); the
  trainable-params table + log-scale params-vs-recipe chart; the two domains
  illustrated (spectrograms of a stem before/after 64 kbps re-encode; the T2 rule
  explained); the merge round-trip demo (CPU-runnable later).
- **02_lora_adaptation_experiments.ipynb**: pre-registration recap verbatim (§2);
  LR-probe cells + freeze step; run matrix + launch cells (RUN LATER); analysis:
  the quality-vs-trainable-params curve (per domain), forgetting table + paired
  deltas, rank-4-vs-16 note, wall-clock/VRAM table, with-Wiener descriptive line;
  §12 interpretation branches as labeled stubs; conclusions + what the verdict means
  for StemCraft (per-user/per-library adapters as a product feature) and for
  Direction 06's noisy-label cross-cut (2602.00084).

---

## 11. Risks & mitigations (direction-specific)

- **LSTM-LoRA plumbing is the #1 risk** → parametrization approach with B=0-identity
  and merge round-trip tests; the mock host makes every failure reproducible on CPU
  now, not on GPU later. cuDNN may de-optimize parametrized LSTMs (slower training) —
  acceptable at 6 k steps; wall-clock recorded per run.
- **Shift too mild (H-05a precondition fails)** → pre-registered: T2 promotes to
  primary; if both domains are mild, the honest finding is "umxhq is codec/genre-robust
  zero-shot" (reportable, ties to the Bake-Off metric discussion).
- **BN running stats confound** (they adapt even in "frozen" recipes if left in train
  mode) → pinned: BN train-mode for all trained recipes uniformly; zero-shot never
  runs BN updates; sensitivity note in THEORY §4.
- **Genre labels unavailable** → the G0b fallback rule (noise domain) is pre-registered;
  no mid-study improvisation.
- **ffmpeg absence on Colab** → it ships on Colab by default; the script checks and
  fails loud; the noise fallback never needs it.
- **Checkpoint drift upstream** (torch.hub model updates) → record the hub commit hash
  + weight file checksum in the registry at G1; pin in the report.
- **Test-session breadth** (15 evals) → all decisions freeze on val first (G3);
  the session is one script, one CSV, no reruns.

---

## 12. Outcome interpretation matrix (pre-registered; full prose in paper/PAPER.md)

| Outcome | Reading | Consequence |
|---|---|---|
| H-05a + H-05b supported | the low-rank adaptation story transfers to recurrent separators; PEFT is the right default for per-domain separator adaptation | StemCraft gains a "per-library adapter" product story at ~1–5 % weight cost; write up as the headline |
| H-05a supported, H-05b refuted | LoRA is efficient but not forgetting-protective here | adapters still win on cost; forgetting claims dropped, discussed vs 2602.00084's theory |
| H-05a refuted (ratio < 0.9) | separation adaptation is NOT low-rank at this scale — a genuine negative for the PEFT-transfers-everywhere narrative | report prominently with the rank-4/16 curve; propose (not run) higher ranks / per-gate wrapping |
| Precondition failed both domains | umxhq is robust to realistic consumer shifts zero-shot | reframe: the practical answer is "you don't need adaptation" — a useful product finding |
| head ≥ LoRA at equal budget | the adaptation lives in the output remapping, not the representation | intriguing; per-layer ablation proposed as future work |
| LSTM-LoRA trains poorly/unstably (vs fc-only) | recurrent LoRA is the hard part — a real engineering finding for the PEFT literature gap | document failure modes; fc-only LoRA becomes the recommendation |

---

## 13. Timeline (Sean's part-time weeks, once GPU exists; after D01 data prep)

| Week | Work | Gate |
|---|---|---|
| 1 | G0 green (now); domains + T2 rule; G1 checkpoint sanity + share table; LR probes | G0, G0b, G1, G2 |
| 2 | 12 main runs (2–3 sessions) | — |
| 3 | consolidated test session; analysis; fill paper; verdicts | G3 |

---

*Frozen on 2026-07-13. Deviations during execution go to `results/DEVIATIONS.md` with
date + reason.*
