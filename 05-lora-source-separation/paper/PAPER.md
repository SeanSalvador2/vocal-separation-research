# LoRA for Source Separation: Parameter-Efficient Adaptation of a Pretrained Music Separator

**Draft scaffold — placeholders ⟪like this⟫ are filled only from committed result CSVs.
All interpretations were written before any training (pre-registered 2026-07-13); the
applicable branch is selected once results exist, the rest move to Appendix C.**

---

## Abstract

Low-rank adaptation (LoRA) is the default parameter-efficient fine-tuning technique for
transformers, yet there are no published PEFT results for music source separation — a
domain where per-user adaptation (low-bitrate libraries, genres, live recordings) is a
natural product need and full per-domain fine-tuning is wasteful. We present the first
careful small-scale study: adapting the pretrained Open-Unmix vocals separator
(`umxhq`, 8,893,348 parameters, recurrent BiLSTM core) to two constructed domain shifts
— 64 kbps codec-degraded audio, and ⟪genre subset / 12 dB live-noise⟫ — under five
recipes: zero-shot, head-only (23.73 % of parameters), LoRA rank 4 (1.27 %), LoRA rank
16 (4.85 %), and full fine-tuning, with per-recipe learning-rate probes for fairness and
three seeds on the headline cells. LoRA on the recurrent core wraps the gate-stacked
LSTM projections via weight parametrization; with B = 0 initialization the wrapped model
reproduces zero-shot output bit-exactly, giving every run a verified common origin. We
find ⟪H-05a verdict: LoRA-16 recovers ⟪⟫ % of the full-fine-tune gain (pre-registered
bar: ≥ 90 %)⟫ and ⟪H-05b verdict: source-domain forgetting of ⟪⟫ dB vs full FT's ⟪⟫
dB⟫. ⟪One sentence: the quality-vs-trainable-parameters curve's shape and the practical
recommendation.⟫

## 1. Introduction

Fine-tuning a separator per deployment domain is the practical answer to distribution
shift — but nobody wants to store and serve a full model copy per user library. In NLP
and vision this tension was resolved by parameter-efficient fine-tuning: LoRA freezes
the pretrained weights and learns low-rank additive updates, typically at ~1 % of the
parameter cost with near-parity quality. Whether that story transfers to source
separation is untested in the literature (verified gap, 2026-07-13): separation is a
regression task (mask estimation), the canonical compact separator is recurrent rather
than attention-based, and adaptation targets are audio-domain shifts (codecs, genres,
noise) rather than task switches.

We run the controlled study on the smallest credible host — Open-Unmix — with the
discipline of the surrounding SingNet project: pre-registered hypotheses with decision
rules, per-recipe LR fairness probes, seed-noise bands, a forgetting protocol anchored
to zero-shot, and honest branches for every outcome including "the shift was too mild
to need adaptation at all."

**Contributions.** (1) First LoRA-for-MSS measurements: the quality-vs-trainable-
parameters curve across zero-shot / head / LoRA-4 / LoRA-16 / full on two domain
shifts; (2) the recurrent-LoRA recipe: parametrization-based wrapping of gate-stacked
BiLSTM weights with a bit-exact B = 0 identity guarantee and exact merge-back (both
unit-tested); (3) a catastrophic-forgetting comparison (source-domain regression)
between LoRA and full fine-tuning on a regression model, connecting to recent theory
that low-rank updates resist noise/forgetting (arXiv 2602.00084); (4) a reproducible
harness where every claim about *what trains* is asserted by tests (exact trainable-
parameter-name sets per recipe).

## 2. Related Work

**PEFT.** LoRA (Hu et al., 2021): ΔW = (α/r)BA on frozen W₀, no inference overhead
after merging. PEFT for music foundation models reaches full-fine-tune parity at < 1 %
parameters on *tagging* tasks (Ding et al., arXiv 2411.19371) — classification, not
separation. LoRA-for-MSS: no published study (gap-check logged in the project
verification log); nearest neighbors are LoRA for beat tracking (2503.10086) and
full-fine-tune transfer of separators to dialogue separation (Strauss et al.,
2106.09093) — the latter sharpens our gap: transfer works, but nobody measured the
parameter-efficient version.

**Forgetting.** "Why LoRA Resists Label Noise" (arXiv 2602.00084) proves rank-limited
capacity bounds on fitting arbitrary noise — the theoretical cousin of our H-05b
(low-rank updates should also bound drift from the source solution). Continual/
personalized separation interest (arXiv 2512.02432) motivates the product framing.

**Host.** Open-Unmix (Stöter et al., JOSS 2019): per-target magnitude-domain BiLSTM
separator, MIT license, MUSDB-trained; architecture verified from source in the
project's Phase-0 review.

## 3. Method

**Host model.** `umxhq` vocals: input magnitudes cropped to 1487 bins (16 kHz), fc1
(2974→512, bias-free) → BN → tanh → 3-layer BiLSTM (256/dir) → skip-concat (1024) →
fc2 → BN → ReLU → fc3 (512→4098) → BN → scale/mean → ReLU mask over the full 2049 bins.
8,893,348 parameters (recomputed against the real checkpoint at gate G1; all shares
below are measured on the shape-faithful mock and asserted in unit tests).

**Recipes.**

| Recipe | Trains | Params | Share |
|---|---|---|---|
| zero-shot | nothing | 0 | 0 % |
| head | fc3 + bn3 + output scale/mean | 2,110,470 | 23.73 % |
| LoRA r=4 | A/B on fc1, fc2, fc3 + all 12 gate-stacked LSTM matrices; + input/output scale/mean | 113,184 | 1.2727 % |
| LoRA r=16 | as above | 431,520 | 4.8522 % |
| full | everything | 8,893,348 | 100 % |

LoRA: h = W₀x + (α/r)BAx, A ~ N(0, σ²), B = 0, α = 2r fixed. LSTM wrapping registers a
weight parametrization on `weight_ih/hh` for every layer and direction, with a forward
pre-hook refreshing the LSTM's cached flat weights (without it the update is silently
ignored — an implementation trap we document and test). B = 0 ⇒ the wrapped model
equals zero-shot bit-exactly (measured max |Δ| = 0.0); merge-back reproduces the
wrapped forward exactly (0.0). On GPU, parametrized weights take cuDNN's unfused path —
slower, numerically equivalent; wall-clock is reported per recipe.

**Domains.** T1 (primary): every stem re-encoded AAC 64 kbps, mixtures re-summed from
degraded stems (supervised pairs stay exactly additive — one-line proof in THEORY §5).
T2 (secondary, rule resolved at data-prep): largest genre cluster if official labels
materialize, else pink noise at exactly 12 dB SNR on mixtures. Resolved to: ⟪genre X /
noise⟫.

**Training.** 6,000 steps per trained recipe, batch 16 × 6-s stereo chunks, UMX-native
MSE-on-magnitude loss and augmentation recipe, Adam with per-recipe LR chosen by
500-step probes on T1 validation (grids: full {3e-5, 1e-4, 3e-4}; others {3e-4, 1e-3,
3e-3}); chosen LRs: ⟪table⟫. Seeds {0,1,2} on the two headline cells (LoRA-16, full ×
T1); 1 seed elsewhere.

## 4. Experimental Setup

- **Gain:** g_D(R) = SI-SDR on D's transformed 50-track test set − zero-shot on the
  same. **Forgetting:** f(R) = zero-shot − R, both on the *standard* test set.
- **H-05a:** g_T1(LoRA-16) ≥ 0.9 · g_T1(full), precondition g_T1(full) >
  max(2σ_seed, 0.3 dB); delta-method CI on the ratio.
- **H-05b:** f(LoRA-16) < f(full) − σ_seed.
- One consolidated test session (all recipes × {standard, T1, T2}), after all
  decisions froze on validation; museval SDR secondary table; Wiener post-filter off
  everywhere (one descriptive with-Wiener line at the end).

## 5. Results

> Generated from `results/registry.csv` / `results/test_matrix.csv`; figures
> regenerate from CSVs.

### 5.1 Preconditions and probes
Zero-shot standard-vs-T1 gap: ⟪⟫ dB (precondition ⟪met / not met — branch⟫). Chosen
LRs: ⟪table⟫. Checkpoint provenance: hub commit ⟪⟫, weight checksum ⟪⟫.

### 5.2 The headline curve (T1)

| Recipe | Trainable share | g_T1 (dB) | f (dB, standard) |
|---|---|---|---|
| zero-shot | 0 % | 0 (def.) | 0 (def.) |
| head | 23.73 % | ⟪⟫ | ⟪⟫ |
| LoRA r=4 | 1.27 % | ⟪⟫ | ⟪⟫ |
| LoRA r=16 | 4.85 % | ⟪ ± seed⟫ | ⟪ ± seed⟫ |
| full | 100 % | ⟪ ± seed⟫ | ⟪ ± seed⟫ |

Gain ratio LoRA-16/full = ⟪⟫ (CI ⟪⟫) → **H-05a: ⟪supported / refuted / mixed / not
evaluable⟫**. Forgetting: ⟪⟫ → **H-05b: ⟪…⟫**. ⟪Figure: gain vs log-trainable-params,
both domains.⟫ ⟪Figure: gain-vs-forgetting scatter.⟫

### 5.3 Secondary domain (T2 = ⟪…⟫)
⟪Same table, 1 seed, directional read.⟫

### 5.4 Practicalities
⟪Wall-clock and peak-VRAM per recipe (the PEFT sell); cuDNN unfused-path slowdown
factor; adapter file size at r=4/16 vs full checkpoint.⟫

## 6. Discussion — pre-registered interpretation branches

**⟪SELECT applicable; unselected → Appendix C.⟫**

### Branch S/S (H-05a and H-05b both supported)
The low-rank story transfers to recurrent separation models: ⟪⟫ % of the full-FT gain
at ⟪4.85⟫ % of the parameters, while drifting ⟪⟫ dB less on the source domain. The
practical reading is immediate: per-domain (even per-user-library) adapters are the
right deployment shape for separator personalization — ~⟪⟫ MB per adapter vs ⟪⟫ MB per
model copy, swappable at load time after merge. The forgetting result gives empirical
regression-task support to the rank-capacity argument of 2602.00084.

### Branch S/R (efficient but not forgetting-protective)
LoRA matches full FT's gain but forgets comparably (Δf within σ_seed). Adapters still
win on storage/compute; the "LoRA protects the source domain" intuition does not
survive contact with a regression model at this scale — worth stating against the
theory's classification-flavored assumptions.

### Branch R (H-05a refuted: ratio < 0.9)
Separation adaptation is *not* low-rank at rank ≤ 16 on this host: the missing ⟪⟫ % of
the gain is the finding. Two pre-declared follow-ups (proposed, not run): higher ranks
(the r-sweep curve suggests ⟪saturating / still climbing⟫) and per-gate (rather than
gate-stacked) LSTM wrapping. The negative is publishable precisely because the PEFT
literature contains no separation data point.

### Branch NE (precondition failed on both domains)
umxhq is robust zero-shot to 64 kbps degradation and ⟪T2⟫: full FT itself gains
< max(2σ_seed, 0.3 dB), so there is nothing for PEFT to recover ⟪numbers⟫. The
product answer is the useful one: **you don't need adaptation for these shifts** — and
the study design (precondition before verdict) prevented a vacuous "LoRA matches full
FT at 100 % of ≈nothing" headline. T2-as-primary fallback ⟪was / wasn't⟫ triggered;
stronger shifts (severe band-limiting, live noise at lower SNR) are the named future
work.

### Branch H (head ≥ LoRA at equal budget)
The adaptation this shift needs lives in the output remapping (fc3/BN statistics), not
the representation — consistent with codec shifts being largely a spectral-envelope
distortion. Per-layer LoRA placement (fc-only vs LSTM-only, the table THEORY §4
pre-computes) becomes the follow-up; the head recipe, at 23.7 % params, is the
pragmatic winner only if adapter size doesn't matter.

### Branch U (LSTM-LoRA trains poorly/unstably)
The engineering finding: recurrent LoRA is the hard part. We report the failure mode
(⟪divergence / no-progress / seed-instability⟫ from the curves), note the cuDNN
unfused-path interaction, and recommend fc-only LoRA (which trained ⟪…⟫) as the safe
recipe. Given the literature gap, even this outcome is a contribution.

### 6.x Threats to validity (written before results)
Constructed domains (real user libraries shift in correlated, messier ways); one host
architecture (recurrent — findings may not transfer to conv or attention separators);
6 k-step budget (adaptation, not re-training; longer budgets could change the
full-FT ceiling); BN-statistics adaptation is entangled with every trained recipe
(uniform policy, stated); 3 seeds only on headline cells; the T2 rule's fallback makes
cross-paper comparison of T2 conditional on the resolved domain; museval/SI-SDR
convention differences kept in separate tables.

## 7. Conclusion

⟪Two paragraphs from selected branches: (1) verdicts with effect sizes and the
parameter-efficiency curve's shape; (2) the deployment recommendation for StemCraft
(adapter-per-library or no-adaptation-needed) and what the recurrent-LoRA recipe
contributes to the PEFT literature gap.⟫

## References

1. Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen. *LoRA: Low-Rank Adaptation of
   Large Language Models.* ICLR 2022. arXiv:2106.09685.
2. Stöter, Uhlich, Liutkus, Mitsufuji. *Open-Unmix — A Reference Implementation for
   Music Source Separation.* JOSS 2019. 10.21105/joss.01667.
3. Ding et al. *Parameter-Efficient Transfer Learning for Music Foundation Models.*
   arXiv:2411.19371.
4. Strauss, Paulus, Torcoli, Edler. *(Transfer learning from MSS to dialogue
   separation.)* arXiv:2106.09093.
5. Steele. *Why LoRA Resists Label Noise: A Theoretical Framework for Noise-Robust
   Parameter-Efficient Fine-Tuning.* arXiv:2602.00084.
6. *(Continual/human-in-the-loop SVS adaptation.)* arXiv:2512.02432.
7. Le Roux, Wisdom, Erdogan, Hershey. *SDR — Half-Baked or Well Done?* ICASSP 2019.
   arXiv:1811.02508.
8. Rafii, Liutkus, Stöter, Mimilakis, Bittner. *MUSDB18.* Zenodo 1117372.
9. *(LoRA for beat tracking.)* arXiv:2503.10086.

## Appendix A — Reproducibility

Commands: MASTER_PLAN.md §6 run book. Trainable-set assertions:
`tests/test_umx_wrapper.py` (exact parameter-name sets per recipe; shares pinned).
B = 0 identity and merge round-trip: `tests/test_lora.py` (both 0.0 max diff at
scaffold time, independently re-verified in review). Suite at scaffold time: 218
passed, 1 skipped. Checkpoint provenance recorded at G1 (hub commit + checksum). GPU
spend: ⟪actual⟫ vs ≈18–24 T4-h ceiling. Deviations: `results/DEVIATIONS.md` (notably
the head-share correction 18 % → 23.73 % from the verified fc3 shape).

## Appendix B — Per-track tables and probe curves ⟪generated⟫

## Appendix C — Pre-registered alternative readings (unselected branches) ⟪moved here⟫
