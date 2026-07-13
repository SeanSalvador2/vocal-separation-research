# Does Band-Splitting Survive at 10 Million Parameters? A Param-Matched Study of the Core Idea Behind SOTA Music Source Separation

**Draft scaffold — placeholders ⟪like this⟫ are filled only from committed result CSVs.
All interpretations were written before any training (pre-registered 2026-07-13); the
applicable branch is selected once results exist, the rest move to Appendix C.**

---

## Abstract

Every current state-of-the-art music source separation family — BSRNN, BS-RoFormer,
Mel-RoFormer, SCNet, Moises-Light — is built on frequency band-splitting: partition the
spectrogram and give each band dedicated capacity. Published evidence for the idea is
exclusively large-scale and confounded (band schemes change alongside sequence models,
training recipes, and extra data). We isolate the idea itself: three models at matched
parameter count (baseline U-Net: 9,835,745; both band-split variants: 9,841,896,
+0.0625 %), differing only in the encoder front-end — full-spectrum, 3-band
equal-width, or 3-band mel-spaced — trained identically on MUSDB18 with 3 seeds each
under a pre-registered ordered hypothesis (mel > uniform > baseline, each link outside
the seed-noise band). Because the two variants share width and padded extent, they are
matched in **both** parameters and compute (~0.54× baseline encoder MACs), so only band
*placement* separates them. We find ⟪one-sentence verdict: full / partial (which
effect) / null / negative, with effect sizes vs σ_seed⟫. A per-band error breakdown
⟪supports / contradicts⟫ the dedicated-capacity mechanism. ⟪One sentence on the
budget-dependence check.⟫ We release the matching methodology, per-module accounting,
and all code; a within-noise null was pre-registered as a reportable finding, in line
with recent evidence that even full-scale band-split results resist replication
(arXiv 2603.09187).

## 1. Introduction

Band-splitting is the load-bearing idea of the modern source-separation stack, and it
has never been tested alone. BSRNN introduced hand-crafted subbands with per-band
modules and semi-supervised extras; Mel-RoFormer showed mel-spaced bands beat heuristic
ones — inside a full transformer at full scale; SCNet compresses bands unequally;
Moises-Light engineers the efficiency frontier of a band-split U-Net. In every case the
band scheme is entangled with sequence modeling, scale, data, and recipe. Meanwhile the
compact-model regime — where a hobbyist or product team actually trains from scratch —
has no evidence at all: maybe dedicated per-band capacity is exactly what a small model
needs (its capacity is scarce, so spending it where vocal energy lives should matter
*more*), or maybe the advantage only emerges with the sequence models and data scale
the SOTA family adds on top.

Both stories are plausible. That is what an experiment is for.

**Contributions.** (1) The first param-matched isolation of the band-split front-end at
compact scale (~10 M params, MUSDB18-only): full-spectrum vs equal-width vs mel-spaced
encoders, everything else bit-identical; (2) a two-effect decomposition — *does
splitting help?* and *does mel placement help beyond splitting?* — via the uniform-split
control; (3) a matching methodology worth copying: deterministic width search with a
committed per-module table (both variants land within +0.0625 % of the baseline count),
plus the capacity-vs-compute distinction made explicit (conv parameters are
spatial-extent-independent; the variants are ~2× cheaper in encoder MACs at matched
params); (4) a per-band mechanism analysis on a layout-neutral 6-band grid;
(5) pre-registered interpretations for all seven outcome cells, including the null.

## 2. Related Work

**The band-split lineage.** BSRNN (Luo & Yu 2022) splits the spectrogram into
hand-chosen subbands with per-band feature extraction and interleaved band/time
sequence modeling (~10 dB cSDR vocals on MUSDB18-HQ). BS-RoFormer (Lu et al. 2023) won
SDX'23 with hierarchical transformers over band tokens; Mel-RoFormer (Wang et al. 2023)
improved it by replacing ad-hoc bands with overlapping mel-scale bands — evidence that
band *placement* matters, at full scale. SCNet (2024) compresses low-information bands
more aggressively. Moises-Light (Hung et al., WASPAA 2025) gets competitive quality
from a band-split U-Net at ~13× fewer parameters than BS-RoFormer — the closest
published relative of our setting, but an efficiency result, not a controlled ablation.

**Replication context.** Magron, Douwes & Serizel (arXiv 2603.09187, 2026) attempted a
faithful BSRNN replication and could not reach the published numbers from the paper
alone, releasing corrected code. This calibrates expectations for compact-scale
transfer of band-split gains — and motivates our pre-registration of the null as a
finding, not a failure.

**This project.** The baseline arm, training stack, augmentation recipe, and evaluation
protocol are the SingNet project's frozen infrastructure (Directions 01–02); the
baseline cell is literally the same run (config-hash-asserted), so the comparison
inherits two directions' worth of tested machinery.

## 3. Method

**Baseline.** SingNet-C1: a 5-down/5-up magnitude-mask U-Net (5×5 convs, stride 2, BN,
sigmoid ratio mask, mixture-phase iSTFT), 2048 network frequency bins (STFT 4096/1024
Hann), 9,835,745 parameters, `l1mag` loss — the project's standard compact model.

**Band-split variants.** The 2048 bins are partitioned into 3 contiguous bands; each
band gets its own encoder tower (same block design, shared base width c = 23), with
internal padding to multiples of 32 and exact crop-back. Towers never mix information
before the bottleneck — that is the treatment. At each level, tower outputs concatenate
along frequency into full-spectrum skip tensors consumed by a baseline-style decoder
(width c, bottleneck width 382 — the single "+Δ" adjustment the deterministic width
search applies to land within tolerance). Two variants, identical in everything but the
edge list:
- **uniform**: bins {0, 683, 1365, 2048} → interior edges 7353.6 / 14696.4 Hz;
- **mel**: equal-mel partition (HTK m = 2595·log₁₀(1+f/700)) → bins
  {0, 142, 597, 2048} → interior edges 1533.9 / 6428.9 Hz — narrow low band, wide top
  band, concentrating capacity where vocal energy lives.

**Matching.** Baseline 9,835,745; both variants 9,841,896 (**+0.0625 %**; tolerance
was ±2 %). Because conv parameter count is independent of spatial extent, the two
variants match *each other exactly* in parameters and (since both pad to the same total
extent) in MACs (~0.54× baseline encoder MACs) — band placement is the only difference
between them. Per-module table: `results/param_match_table.md`, asserted in unit tests
and re-derived in THEORY.md §4.

**Training.** Identical for all arms (Direction-01 stack): full augmentation recipe,
AdamW + warmup/cosine, AMP, batch 16, REDUCED = 16 k steps (sweep) / FULL = 40 k
(confirmation), best-checkpoint by validation SI-SDR, seeds {0, 1, 2}.

## 4. Experimental Setup

- **Runs:** mel × 3 seeds + uniform × 3 seeds (new); baseline × 3 seeds shared with
  Directions 01/02 (hash-asserted); FULL confirmations: best variant + baseline;
  optional pre-registered contingency (loss flip) ⟪used / not used⟫.
- **Protocol:** decisions on the 14-track validation split; one test pass (50 tracks)
  for the two FULL checkpoints with paired bootstrap CI + Wilcoxon on the single
  pre-registered pair.
- **Pre-registered hypothesis (verbatim decision rules in MASTER_PLAN §2):**
  E1 = uniform − baseline > σ_seed ("splitting helps");
  E2 = mel − uniform > σ_seed ("placement helps beyond splitting");
  fully supported iff E1 ∧ E2; partial/null/negative cells as §13 of the plan.
  σ_seed = pooled 3-arm between-seed std.
- **Mechanism figure:** per-band deltas (variant − baseline) of band-limited SI-SDR
  and normalized band magnitude error on a fixed 6-band grid (union of both layouts'
  edges + a 100 Hz floor), so the figure favors neither layout.
- **Guard:** the FULL-budget pair must not reverse the REDUCED verdict's sign, else
  the verdict downgrades to budget-dependent (a finding about short-budget
  architecture comparisons).

## 5. Results

> Generated from `results/registry.csv` / `results/test_per_track.csv` /
> `results/banded.csv` by notebook 02; figures regenerate from CSVs.

### 5.1 Three-arm comparison (REDUCED, 3 seeds each)

| Arm | val SI-SDR mean ± seed-std (dB) |
|---|---|
| baseline | ⟪⟫ |
| split_uniform | ⟪⟫ |
| split_mel | ⟪⟫ |

σ_seed (pooled) = ⟪⟫ dB. **E1 = ⟪⟫ dB (⟪> / ≤⟫ σ_seed); E2 = ⟪⟫ dB (⟪> / ≤⟫ σ_seed)
→ H-03: ⟪fully supported / partial-E1 / partial-E2 / null / negative⟫.**
⟪Figure: 3-arm chart with band.⟫ ⟪Figure: training curves (optimization-difficulty
check).⟫

### 5.2 Budget-dependence and test pass

FULL (40 k): best variant ⟪⟫ dB vs baseline ⟪⟫ dB → sign ⟪consistent / REVERSED⟫.
Test (50 tracks, paired): Δ = ⟪⟫ dB, 95 % CI ⟪⟫, Wilcoxon p = ⟪⟫.
⟪Figure: per-track paired scatter.⟫

### 5.3 Per-band mechanism

⟪Figure: 6-band delta profile per variant.⟫ Gains concentrate in ⟪bands / nowhere⟫;
vocal-energy-dense bands (≈100 Hz–4 kHz per the EDA table) show ⟪⟫; band-boundary-
adjacent analysis bands show ⟪artifacts / nothing⟫; the ≈16 kHz-limited top band shows
⟪⟫. Mechanism verdict: ⟪supports / contradicts / orthogonal to⟫ dedicated-capacity.

### 5.4 Efficiency table (descriptive)

| Arm | Params | Encoder MACs (6-s chunk) | CPU RTF |
|---|---|---|---|
| baseline | 9,835,745 | ⟪measured⟫ | ⟪⟫ |
| split_uniform | 9,841,896 | ⟪≈0.54×⟫ | ⟪⟫ |
| split_mel | 9,841,896 | ⟪≈0.54×⟫ | ⟪⟫ |

## 6. Discussion — pre-registered interpretation branches

**⟪SELECT the applicable branch; unselected branches → Appendix C.⟫**

### Branch F (fully supported: mel > uniform > baseline, both links > σ_seed)
The SOTA family's core idea is scale-robust: even at 10 M params on 86 songs, isolating
bands buys ⟪E1⟫ dB and placing them on a mel scale buys a further ⟪E2⟫ dB — at equal
parameters and roughly half the encoder compute. The per-band profile ⟪localizes the
gain in vocal-dense bands, matching the dedicated-capacity mechanism / does not localize
as predicted — see the mechanism-mismatch note⟫. *Consequences:* the mel-split
front-end becomes a legitimate option for the shipped SingNet engine (quality *and*
efficiency); a band-count/overlap sweep is named as future work, not run.

### Branch P1 (E1 only: splitting helps, mel ≈ uniform)
Dedicated capacity is the active ingredient; *where* bands sit is second-order at
3-band granularity. This reads as: band-local statistics (per-band normalization
freedom, no cross-band interference in early layers) matter, but 3 bands are too coarse
for placement effects that Mel-RoFormer detects with many overlapping bands at full
scale. *Consequences:* prefer uniform split (simpler, no mel machinery); flag band
count as the follow-up variable.

### Branch P2 (E2 only: mel > uniform, but uniform ≤ baseline)
The intriguing cell: isolation alone does nothing (or hurts), yet mel placement rescues
it — capacity *concentration*, not band *independence*, is the mechanism. The per-band
figure adjudicates: expect uniform's mid-band (683–1365 bins ≈ 7.4–14.7 kHz) to be
wasted capacity on this AAC-band-limited data while mel's narrow low band wins where
vocals live. *Consequences:* placement-first reading; a follow-up would vary edges, not
band count; shipping decision defers to the FULL-budget pair.

### Branch N (null: all three within ±σ_seed)
**The pre-registered headline: the band-split advantage is not detectable at ~10 M
params / MUSDB-only / this budget.** Given the controls (param-matched to +0.06 %,
compute-matched variants, shared training stack, 3 seeds), this is the cleanest
available evidence that band-splitting's published gains live at larger scale, with
sequence modeling, more bands, or more data — consistent with the replication
difficulty documented for full-scale BSRNN (2603.09187). *Consequences:* baseline
stays; the project's capacity-vs-data narrative (with Direction 02's scaling verdict:
⟪cross-reference once known⟫) sharpens — if D02 found data-starvation, "more data"
beats "smarter front-end" at this scale, and Direction 10's premise strengthens.

### Branch NEG (negative: baseline beats both variants > σ_seed)
Band isolation actively costs at this scale. Leading candidate mechanism (pre-declared):
towers lose cross-band context that a small full-spectrum encoder exploits (harmonic
stacks span band edges — a vocal's fundamental and its harmonics land in different
towers), and zero-padded band boundaries add edge artifacts; the per-band figure should
show damage concentrated near band edges ⟪does / doesn't⟫. *Consequences:* documents
the failure mode the SOTA family engineered around (overlapping bands, band-embedding
MLPs, cross-band sequence layers); baseline stays; we explicitly warn against naive
band-splitting of small U-Nets.

### Branch BD (budget-dependent: FULL reverses REDUCED)
A methodological finding with teeth: 16 k-step architecture comparisons can invert at
40 k. Both readings reported; no architecture verdict claimed; future architecture
comparisons in this project run at FULL only (rule adopted into the project plan).

### Branch MM (mechanism mismatch: positive headline, wrong bands)
The effect is real but the dedicated-capacity story is not why. We say so plainly,
report where the gains actually live, and treat the mechanism as open — proposing (not
running) the follow-up that would separate normalization effects from capacity
allocation (e.g. per-band BN in the baseline).

### 6.x Threats to validity (written before results)
Three bands only (granularity untested); no overlapping bands (SOTA uses overlaps;
ours are disjoint by design for exact accounting); mono pipeline; MUSDB18's AAC
~16 kHz ceiling makes the top mel band partly dead spectrum (a property of mel-on-
this-data, stated in advance and visible in the EDA table); seeds control data order
but cannot match init across different-shaped arms (absorbed by 3-seed spread);
REDUCED-budget primary (guarded by the FULL pair); single dataset and architecture
family — claims are about compact magnitude-mask U-Nets on MUSDB, not about
band-splitting universally.

## 7. Conclusion

⟪Two paragraphs from the selected branch: (1) the verdict with effect sizes, the
mechanism reading, and the efficiency observation; (2) what it means for small-model
practitioners and for the SOTA family's narrative — plus what this project now does
differently (shipped front-end choice, or the strengthened data-first roadmap).⟫

## References

1. Luo, Yu. *Music Source Separation with Band-Split RNN.* arXiv:2209.15174.
2. Lu, Wang, Kong, Hung. *Music Source Separation with Band-Split RoPE Transformer.*
   arXiv:2309.02612.
3. Wang, Lu, Won. *Mel-Band RoFormer for Music Source Separation.* arXiv:2310.01809.
4. *(SCNet.)* arXiv:2401.13276, ICASSP 2024.
5. Hung, Pereira, Korzeniowski. *(Moises-Light.)* WASPAA 2025, arXiv:2510.06785.
6. Chen et al. *(DTTNet.)* arXiv:2309.08684, ICASSP 2024.
7. Magron, Douwes, Serizel. *The Costs of Reproducibility in Music Separation
   Research: a Replication of Band-Split RNN.* arXiv:2603.09187.
8. Watcharasupat et al. *A Generalized Bandsplit Neural Network for Cinematic Audio
   Source Separation.* arXiv:2309.02539.
9. Le Roux, Wisdom, Erdogan, Hershey. *SDR — Half-Baked or Well Done?* ICASSP 2019.
   arXiv:1811.02508.
10. Rafii, Liutkus, Stöter, Mimilakis, Bittner. *MUSDB18.* Zenodo 1117372.

## Appendix A — Reproducibility

Commands: MASTER_PLAN.md §7. Param accounting: `results/param_match_table.md`
(test-asserted; c = 23, bottleneck 382, +0.0625 %). Mel edges from the HTK closed form
(bins {0, 142, 597, 2048}), test-asserted. Environment: repo-root `requirements.txt`.
Suite at scaffold time: 158 passed, 1 skipped. GPU spend: ⟪actual⟫ vs ≈16–25 T4-h
ceiling. Deviations: `results/DEVIATIONS.md` (six build-time clarifications, incl. the
width-search tie-break rule and the full-5-level-tower reading of the architecture).

## Appendix B — Per-track, per-band, per-seed tables ⟪generated⟫

## Appendix C — Pre-registered alternative readings (unselected branches) ⟪moved here⟫
