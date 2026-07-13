# Train on What You Test? A Controlled Loss-Function Study for Compact Music Source Separation

**Draft scaffold — placeholders marked ⟪like this⟫ are filled only from committed result
CSVs after the runs execute. Every interpretation below was written *before* any
training run (pre-registered on 2026-07-13); the applicable branch is selected, the
others are kept in an appendix for transparency.**

---

## Abstract

Music source separation systems are trained with a surprising variety of losses —
L1-magnitude (Spleeter), MSE-magnitude (Open-Unmix), waveform L1 (Demucs) — while being
evaluated on SDR-family metrics, and the "obvious" alternative of training directly on
the evaluation metric is rarely used. We train a fixed compact (9.8 M-parameter)
magnitude-mask U-Net on MUSDB18 five times, changing **only the training loss**
(L1-magnitude, MSE-magnitude, log-L1-magnitude, negative SI-SDR, and L1 + multi-resolution
STFT), three seeds each, under a frozen evaluation protocol with paired statistics and a
small blind listening check. Pre-registered hypotheses: (H-01a) SI-SDR training does not
beat L1-magnitude on SI-SDR evaluation at this scale; (H-01b) the multi-resolution STFT
auxiliary reduces audible artifacts at equal SI-SDR. We find ⟪one-sentence headline:
H-01a verdict + effect size in dB with CI⟫ and ⟪one-sentence H-01b verdict with
preference percentage⟫. ⟪One sentence on the strongest secondary finding, e.g. the
log-L1 ordering or the SI-SDR silent-chunk skip rate.⟫ All code, configs, seeds, and
per-track results are released; every figure regenerates from committed CSVs.

## 1. Introduction

Practitioners building source-separation models face an immediate, under-documented
choice: what loss to train on. The field's most-copied systems disagree — Spleeter uses
L1 on masked magnitude, Open-Unmix uses MSE on magnitude, Demucs uses waveform L1 — and
none of them trains on the SDR-family metrics they report. The tension is real rather
than rhetorical: Gusó et al. (ICASSP 2022) benchmarked losses in a controlled setup and
found spectrogram-domain losses competitive while cautioning that SDR can mislead;
Défossez et al.'s Demucs paper ablates L1 against L2 (choosing L1) but never runs a
clean same-architecture comparison against SI-SNR-style training — the popular "L1 beats
SI-SDR training" reading is indirect. And the 2025 "Bake-Off" listener study found that
for *vocals* specifically, BSS-Eval SDR is the best perceptual proxy among common
metrics — which cuts both ways: optimizing SDR-like objectives might be exactly right
(the metric is meaningful) or unnecessary (magnitude losses already achieve it).

Nobody has published this comparison for the regime hobbyists and product teams actually
occupy: a compact (~10 M parameter) magnitude-mask U-Net trained on MUSDB18 only. We run
it, pre-registered, with the loss as the *only* moving part.

**Contributions.** (1) A controlled five-loss comparison at compact scale with seed-noise
bands and paired per-track statistics; (2) a pre-registered test of the
train-on-your-metric intuition (H-01a) and of multi-resolution STFT as an artifact
reducer (H-01b); (3) a documented, reusable, unit-tested training/eval stack
(`singnet/`); (4) honest reporting of ⟪negative/positive⟫ results either way, including
a measured SI-SDR-loss silent-chunk fragility rate that motivates a companion study on
silence (Direction 08).

## 2. Related Work

**Losses for separation.** Spleeter trains a 6-down/6-up magnitude U-Net with L1 on the
masked spectrogram; Open-Unmix (UMX) trains a BiLSTM magnitude model with MSE; Demucs
trains waveform models with L1 and reports an L1-vs-L2 ablation. Gusó et al. (2022)
compare a large set of waveform- and spectrogram-domain losses in a controlled MSS setup
and cross-correlate metrics with a listening test; their verified conclusions are that
spectrogram-domain losses are competitive and SDR can mislead (we do not rely on their
exact ranking, which we could not re-verify from the paper body; see the project
verification log). Multi-resolution STFT losses originate in neural vocoding (Parallel
WaveGAN) as an artifact-suppression device and are implemented in the `auraloss` library
whose default resolutions we adopt.

**Metrics.** SI-SDR (Le Roux et al., 2019) fixes SDR's scale-sensitivity via a
least-squares projection; it is our primary metric. museval BSS-Eval SDR remains the
literature convention and is reported secondarily, never conflated. The WASPAA 2025
Bake-Off study grounds our metric policy: SDR-family metrics track vocal perception
best, which is why H-01b is judged by a listening check *at equal SI-SDR* rather than by
SI-SDR movement.

**Scale caveat.** SOTA systems (band-split transformer families, ~10 dB SDR) train far
larger models, often with extra data; our question is deliberately about the compact,
MUSDB-only regime, and our results should not be extrapolated beyond it.

## 3. Method

*(Summarized from MASTER_PLAN.md §5–§6, which is normative; THEORY.md derives every
equation and cross-references the implementing code.)*

Fixed model: "SingNet-C1", a single-channel magnitude-mask U-Net (5 conv-down / 5
conv-up, 5×5 kernels, stride 2, BN, sigmoid mask head; 9,835,745 parameters — derived
per-layer in THEORY.md §5 and asserted by a unit test). STFT: n_fft 4096, hop 1024,
Hann. Reconstruction: predicted ratio mask on mixture magnitude, mixture phase, iSTFT
(differentiable when the loss needs the waveform).

The five arms (only the loss changes; data order, augmentation stream, init, optimizer,
schedule, steps are bit-identical per seed):

| Arm | Loss | Domain |
|---|---|---|
| `l1mag` | $\operatorname{mean}\lvert M\odot\lvert X\rvert-\lvert S\rvert\rvert$ | magnitude |
| `msemag` | $\operatorname{mean}(M\odot\lvert X\rvert-\lvert S\rvert)^2$ | magnitude |
| `logl1mag` | L1 between $\log(\cdot+10^{-5})$ magnitudes | log-magnitude |
| `sisdr` | $-$SI-SDR of iSTFT output vs target, with a −60 dBFS silent-target guard (skip rate logged) | time |
| `l1mrstft` | `l1mag` $+\;0.5\sum_{m=1}^{3}(\mathcal L^{(m)}_{\text{sc}}+\mathcal L^{(m)}_{\text{mag}})$ at resolutions {512, 1024, 2048} | mixed |

Data: MUSDB18 (86 train / 14 val / 50 test, the standard `musdb` validation list;
research-only license, no audio redistributed). 6-s mono chunks, uniform random start;
augmentation = cross-track source remixing, per-source gain U(0.25, 1.25), sign flip —
the code-verified UMX/Demucs recipe. Training: AdamW 1e-3, warmup+cosine, AMP with
losses computed in fp32, batch 16, REDUCED = 16 k steps (sweep) / FULL = 40 k steps
(confirmation), checkpoints + RNG state every 1 k steps.

## 4. Experimental Setup

- **Run matrix:** 5 arms × 3 seeds at REDUCED (15 runs, the primary evidence) → top-2
  arms + `l1mag` at FULL, seed 0 (3 runs) → one test pass.
- **Protocol:** selection and all tuning on the 14-track validation split only;
  the 50-track test set is read exactly once, at the end, for the 3 FULL checkpoints +
  do-nothing floor + oracle IRM/IBM anchors.
- **Statistics:** between-seed std pooled across arms (σ_seed) drawn as a band on every
  sweep figure; test-set claims use paired-by-track deltas with bootstrap 95 % CIs
  (10 k resamples) and a two-sided Wilcoxon signed-rank test — for exactly two
  pre-registered pairs: (`sisdr` − `l1mag`) and (`l1mrstft` − `l1mag`).
- **Listening check:** 5 test clips × 3 systems, level-matched, blind randomized A/B on
  the artifact question, n ≥ 5 raters; qualitative pilot evidence by design.
- **Hypotheses (pre-registered verbatim; decision rules in MASTER_PLAN §2):** H-01a and
  H-01b as stated in the abstract.

## 5. Results

> Every number below is generated by `scripts/evaluate.py` /
> `notebooks/03_loss_study_experiments.ipynb` from committed CSVs
> (`results/registry.csv`, `results/test_per_track.csv`, `results/test_museval.csv`).

### 5.1 Sweep (REDUCED budget, 15 runs, validation)

| Arm | val SI-SDR mean ± seed-std (dB) | best seed | worst seed |
|---|---|---|---|
| `l1mag` | ⟪⟫ | ⟪⟫ | ⟪⟫ |
| `msemag` | ⟪⟫ | ⟪⟫ | ⟪⟫ |
| `logl1mag` | ⟪⟫ | ⟪⟫ | ⟪⟫ |
| `sisdr` | ⟪⟫ (skip rate ⟪⟫ %) | ⟪⟫ | ⟪⟫ |
| `l1mrstft` | ⟪⟫ | ⟪⟫ | ⟪⟫ |

Pooled σ_seed = ⟪⟫ dB. Δ(`sisdr` − `l1mag`) = ⟪⟫ dB → H-01a on validation:
⟪supported / refuted / ambiguous⟫. ⟪Figure: ranking chart with seed band.⟫
⟪Figure: training curves.⟫

### 5.2 Confirmation (FULL budget) and the single test pass

| System | test SI-SDR vocals (mean [95 % CI]) | median | museval SDR (median) |
|---|---|---|---|
| do-nothing floor | ⟪⟫ | ⟪⟫ | ⟪⟫ |
| ⟪arm A⟫ | ⟪⟫ | ⟪⟫ | ⟪⟫ |
| ⟪arm B⟫ | ⟪⟫ | ⟪⟫ | ⟪⟫ |
| `l1mag` (if not A/B) | ⟪⟫ | ⟪⟫ | ⟪⟫ |
| oracle IRM | ⟪⟫ | ⟪⟫ | ⟪⟫ |
| oracle IBM | ⟪⟫ | ⟪⟫ | ⟪⟫ |

Paired deltas: `sisdr` − `l1mag` = ⟪⟫ dB, 95 % CI ⟪⟫, Wilcoxon p = ⟪⟫;
`l1mrstft` − `l1mag` = ⟪⟫ dB, 95 % CI ⟪⟫, Wilcoxon p = ⟪⟫.
⟪Figure: per-track paired scatter.⟫ ⟪Figure: oracle-headroom ladder.⟫

### 5.3 Listening check (H-01b)

⟪n⟫ raters × ⟪k⟫ clip-pairs. Preference for `l1mrstft` over `l1mag` on the artifact
question: ⟪⟫ % (per-clip majorities: ⟪⟫/5). Rater agreement: ⟪⟫.
SI-SDR-equality precondition |Δ| ≤ σ_seed: ⟪met / not met⟫.

### 5.4 Secondary observations

- MSE vs L1 vs log-L1 ordering: ⟪⟫.
- `sisdr` silent-chunk skip rate: ⟪⟫ % (per-run range ⟪⟫).
- Steps/s per arm (wall-clock cost of waveform-path losses): ⟪⟫.

## 6. Discussion — pre-registered interpretation branches

**⟪SELECT the applicable branch per outcome; the unselected branches move to
Appendix C ("Pre-registered alternative readings") — they are part of the record.⟫**

### 6.1 H-01a — training on the eval metric

**Branch S (supported: Δ ≤ +σ_seed, test does not reverse).**
Training directly on SI-SDR bought nothing over plain L1-magnitude at compact scale —
consistent with the practitioner folklore embodied by Spleeter/UMX/Demucs and with the
competitiveness of spectrogram losses in Gusó et al.'s controlled setup. The mechanism
reading (THEORY.md §4): under a bounded sigmoid mask with mixture phase, all losses
select from the same reachable set of estimates, and near that set's SI-SDR-optimal
region the magnitude-L1 geometry appears to be as good a guide as the metric itself —
while being cheaper (no iSTFT in the graph, measured ⟪⟫ % faster) and free of the
silence singularity that forced a guard (skip rate ⟪⟫ %). *Practical consequence:*
`l1mag` stays the project default for Directions 02–10; SI-SDR remains an evaluation
metric, not a training loss, at this scale.

**Branch R (refuted: Δ > +σ_seed on val, confirmed on test with CI excluding 0).**
At this scale, direct metric optimization genuinely helps (by ⟪⟫ dB, a ⟪⟫× multiple of
seed noise). The literature tension resolves scale-dependently: the indirect
Demucs-era preference for L1 does not transfer to a compact bounded-mask model, and the
correct reading is that within a *small* reachable set, pointing the optimizer exactly
at the target metric matters more, not less. *Practical consequence:* `sisdr` (with the
−60 dBFS guard, whose skip rate of ⟪⟫ % remains a real operational cost) becomes the
default loss for the remaining directions; we re-examine the Direction 02/03 baselines
for sensitivity before reusing older checkpoints.

**Branch M (mixed: val and test disagree, or the seed band makes the call ambiguous).**
The honest summary is that at compact scale the loss choice between `l1mag` and `sisdr`
is within (or at the edge of) run-to-run noise: an effect this small — |Δ| ≈ ⟪⟫ dB
against σ_seed ≈ ⟪⟫ dB — would need ⟪⟫+ seeds to resolve, which we pre-registered as out
of scope (G2 escalation adds only two seeds to the two closest arms; result: ⟪⟫). We
keep `l1mag` on a simplicity prior and report the ambiguity as a finding: loss-function
papers claiming small gains at single seeds are, at this scale, likely reporting noise.

### 6.2 H-01b — MR-STFT as an artifact reducer

**Branch S (equal SI-SDR + listening preference ≥ 60 %).**
The auxiliary term changed *what the errors sound like* without changing what SI-SDR
measures — raters preferred `l1mrstft` in ⟪⟫ % of judgments at a Δ SI-SDR of only ⟪⟫ dB.
This is exactly the metric blind spot the Bake-Off warns about, observed at small scale:
SDR-family metrics are the best available vocal proxy yet still miss a texture
dimension the multi-resolution spectral terms regularize. *Practical consequence:* ship
`l1mag + 0.5·MR-STFT` for user-facing checkpoints; keep plain `l1mag` for
metric-comparability studies.

**Branch R (equal SI-SDR, preference ≤ 40 %).**
Raters did not hear what the vocoder literature predicts. Either the compact model's
dominant artifacts (mask-induced musical noise / mixture-phase smearing) are upstream of
what MR-STFT regularizes, or the effect exists below our pilot's sensitivity (n = ⟪⟫).
We drop the auxiliary term (simplicity, ⟪⟫ % training-cost saving) and record that at
compact scale, artifact claims for auxiliary spectral losses should not be assumed
transferable from vocoders.

**Branch I (inconclusive: 40–60 % or poor rater agreement α = ⟪⟫).**
The pilot cannot distinguish the hypotheses; we report preference counts without a
verdict and flag the follow-up (larger n, MUSHRA-style anchors) as future work rather
than claiming either direction.

**Branch N (precondition failed: |SI-SDR(l1mrstft) − SI-SDR(l1mag)| > σ_seed).**
H-01b is not evaluable as posed. We instead report the observed quality/artifact
tradeoff: ⟪which side moved and by how much⟫, with the listening data as descriptive
context. Pre-registered note: if `l1mrstft` *gained* SI-SDR, that is itself evidence
against the "SDR is already saturated for vocals" reading and is discussed under §6.4.

### 6.3 Secondary pre-registered readings

- **log-L1 ≥ L1 ≥ MSE:** consistent with the dynamic-range argument (THEORY.md §3.3) —
  log compression up-weights the quiet TF bins where vocal texture lives. If instead
  **MSE ≥ L1**: the energy-weighted geometry suits SI-SDR's quadratic residual —
  and UMX's choice was not an accident. If **log-L1 last**: the ε-floor and fp32-guarded
  small-magnitude gradients likely dominated; we check the training curves for the
  characteristic early-plateau signature before interpreting further.
- **High `sisdr` skip rate (> 20 %):** SI-SDR training is operationally fragile on real
  music regardless of its ranking — a data-shaped, not model-shaped, weakness, feeding
  Direction 08's silence study directly.
- **Sweep→FULL ranking flip:** if the top-2 order changes at full budget, reduced-budget
  sweeps (a standard community practice we pre-declared) are themselves unreliable at
  this scale — reported prominently, not buried.

### 6.4 Threats to validity (written before results)

Single architecture and dataset (findings are claims about the compact bounded-mask
MUSDB regime, nothing broader); 3 seeds bound, not eliminate, seed noise; the listening
pilot is small and raters are not audiologically screened; the silent-target guard is a
pre-registered asymmetry of the `sisdr` arm (defensible — without it the arm diverges —
but it means "SI-SDR loss" really means "guarded SI-SDR loss"); REDUCED-budget rankings
may not transfer (mitigated by FULL confirmations); MUSDB18's AAC compression caps
bandwidth at ~16 kHz, which slightly flatters spectrogram losses that never see the
missing band.

## 7. Conclusion

⟪Two paragraphs, written from the selected branches: (1) the H-01a/H-01b verdicts with
effect sizes and what they change for the SingNet project's remaining directions;
(2) the method point — what a pre-registered, seed-banded, paired-stats loss study at
compact scale adds over the folklore, independent of which way the results fell.⟫

## References

*(Full BibTeX in `theory/theory.tex`; verification status per entry in
`00-shared-research/VERIFICATION_LOG.md`.)*

1. Gusó, Pons, Pascual, Serrà. *On Loss Functions and Evaluation Metrics for Music
   Source Separation.* ICASSP 2022. arXiv:2202.07968.
2. Défossez, Usunier, Bottou, Bach. *Music Source Separation in the Waveform Domain.*
   arXiv:1911.13254.
3. Le Roux, Wisdom, Erdogan, Hershey. *SDR — Half-Baked or Well Done?* ICASSP 2019.
   arXiv:1811.02508.
4. Yamamoto, Song, Kim. *Parallel WaveGAN…* ICASSP 2020. arXiv:1910.11480.
5. Hennequin, Khlif, Voituret, Moussallam. *Spleeter…* JOSS 2020. 10.21105/joss.02154.
6. Stöter, Uhlich, Liutkus, Mitsufuji. *Open-Unmix…* JOSS 2019. 10.21105/joss.01667.
7. Jaffe, Burgoyne. *Musical Source Separation Bake-Off: Comparing Objective Metrics
   with Human Perception.* WASPAA 2025. arXiv:2507.06917.
8. Steinmetz, Reiss. *auraloss: Audio-focused loss functions in PyTorch.* DMRN+15 2020.
9. Rafii, Liutkus, Stöter, Mimilakis, Bittner. *MUSDB18.* Zenodo 1117372.
10. Stöter, Liutkus, Ito. *The 2018 Signal Separation Evaluation Campaign.*
    arXiv:1804.06267.
11. Kong, Cao, Liu, Choi, Wang. *Decoupling Magnitude and Phase Estimation…* ISMIR 2021.
    arXiv:2109.05418.

## Appendix A — Reproducibility

Exact commands: MASTER_PLAN.md §8 run book. Environment: `requirements.txt` (pinned;
torch 2.13.0). Seeds {0,1,2}; every run keyed by config hash in `results/registry.csv`
with GPU type and wall-clock. Test suite: `python -m pytest -q` (68 passed, 1 skipped —
museval-guarded — at scaffold time). Figures regenerate from committed CSVs via
notebook 03. GPU spend: ⟪actual total⟫ vs the ≈ 31–43 T4-h pre-registered ceiling.
Deviations from the pre-registered plan: see `results/DEVIATIONS.md` ⟪plus summary here⟫.

## Appendix B — Per-track tables

⟪Generated: per-track val/test SI-SDR for all systems; museval table.⟫

## Appendix C — Pre-registered alternative readings (unselected branches)

⟪Moved verbatim from §6 once branches are selected — kept as part of the scientific
record so the reader can verify no post-hoc rationalization occurred.⟫
