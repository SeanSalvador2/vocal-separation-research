# Direction 02 — How far do 100 songs go? Augmentation factorization + data-scaling curves

Deep-dives: [`papers/wisdom2020-mixit.md`](papers/wisdom2020-mixit.md) (context),
[`papers/saijo2025-mixit-mss.md`](papers/saijo2025-mixit-mss.md) (CORE). The augmentation *recipes* live in
the shared Demucs/UMX notes (code-verified) and are tabled below.

## 1. Direction recap (from RESEARCH_DIRECTIONS #2; deepens PLAN H2)
- **Hypothesis:** (a) **source remixing accounts for the majority of the augmentation gain**; (b) with remixing on, **val SI-SDR vs #training-songs follows a log-like curve that has *not* saturated at 86 songs** — quantifying how data-starved MUSDB-only training is.
- **Minimal experiment:** leave-one-out augmentation ablation (**5 configs**: none / +gain+swap / +remix / +pitch-tempo, plus a full-recipe reference) + training on **{21, 43, 64, 86} songs** with full augmentation (**4 configs**), 1 seed each **+ 3 seeds at the two endpoints**; plot the scaling curve with extrapolation caveats.
- **Falsification / success test:** (a) supported iff removing **remix** costs more val SI-SDR than removing any other single transform, outside the seed band; (b) supported iff the {21,43,64,86} curve is **monotone increasing with a still-positive slope at 86** (a plateau *refutes* (b) — equally reportable). "The *shape* is the finding, whichever way it bends."
- Difficulty/Risk: **Low/Low**. Deepest synergy with the base plan (it *is* an expanded H2).

## 2. Paper map
| Paper | Role | Specific claim we rely on | Verified | Where |
|---|---|---|---|---|
| Demucs v1 (1911.13254) + `augment.py` | **infrastructure (recipe)** | exact transforms: Shift, FlipChannels, FlipSign, Remix (group_size=4), Scale (0.25–1.25) | CONFIRMED (code) | [shared](../../00-shared-research/papers/demucs2019-v1.md) |
| Open-Unmix `data.py` | **infrastructure (recipe)** | gain U(0.25,1.25), channelswap p=0.5, random_track_mix (remix), random chunk | CONFIRMED (code) | [shared](../../00-shared-research/papers/openunmix2019.md) |
| Saijo & Bando 2025 (2505.07631) | **motivation** | unlabeled-music pre-training helps SOTA MSS → data is the binding constraint | CONFIRMED | [papers/saijo2025-mixit-mss.md](papers/saijo2025-mixit-mss.md) |
| Wisdom et al. 2020 (2006.12701) | **context** | unlabeled data can substitute for labels (MixIT) | CONFIRMED | [papers/wisdom2020-mixit.md](papers/wisdom2020-mixit.md) |
| MUSDB18 | **infrastructure** | linear stem additivity → exact cross-song remix; 86-song ceiling | CONFIRMED | [shared](../../00-shared-research/papers/musdb18-dataset.md) |
| "SVS: a study on training data" (1906.02618) | **adjacent prior** | studies the effect of training-data amount/type for *singing-voice* separation | title/existence CONFIRMED (WS) | (gap §4) |

## 3. Replicate-vs-extend
- **Replicate:** the standard MSS augmentation recipe (Demucs/UMX), used as-is and *individually switchable* (PLAN Phase 2 builds it composable).
- **Extend (our delta):** nobody publishes a **factorized leave-one-out contribution per transform** *together with* a **data-scaling curve** for a **compact model on MUSDB-only**. Direction 02 produces both, with seed noise bands. The narrative: *"I measured the value of data vs each augmentation before asking for more of either"* — data-centric ML, hiring-manager-native.

## 4. The gap (with logged gap-check)
- **Gap-check query (2026-07-13, WebSearch):** *"factorized data augmentation ablation music source separation MUSDB remixing gain contribution controlled study."* **Finding:** augmentation transforms are ubiquitously *documented* (gain 0.25–1.25, channelswap, remix, shift; Demucs even discusses augmentation *impact*), and there is one **adjacent** prior — **"Singing voice separation: a study on training data" (arXiv 1906.02618, 2019)** which studies training-data amount/type for SVS. **No dedicated paper** reports a **per-transform leave-one-out contribution + a data-scaling curve for a compact MSS model on MUSDB-only** with seeds/CIs. Novelty of the *specific factorization+scaling combination at small scale* stands; 1906.02618 must be cited as the closest prior (it studies data quantity/type for SVS, not a factorized aug ablation).
- No mid-2026 paper closing this surfaced.

## 5. Direction-specific technical notes (for MASTER_PLAN)
**The code-verified augmentation recipe (what each config toggles):**
| Transform | Operation | Default | Source |
|---|---|---|---|
| Random gain | per-source amplitude $\times\,U(0.25,1.25)$ | on | UMX `data.py`, Demucs `Scale` |
| Channel swap | swap L/R with $p=0.5$ | on | UMX `data.py`, Demucs `FlipChannels` |
| **Source remixing** | draw each source from a different track, sum | on | UMX `random_track_mix`, Demucs `Remix(group_size=4)` |
| Sign flip | $s\to-s$ | Demucs-only | Demucs `FlipSign` |
| Random shift/crop | random start, fixed 6 s chunk | on | Demucs `Shift(8192)`, UMX chunking |
| Pitch/tempo | ±semitone / ±tempo% | heavy, optional | Demucs (separate path) |

**Leave-one-out design.** Because remixing changes the *mixture distribution* (a mixture is now `vocals_i + drums_j + bass_k + other_l`), it is categorically different from gain/swap/shift (which perturb a single real mixture). Expect remix to dominate — but the independence it assumes across stems is only *approximately* valid (real songs have key/tempo-consistent stems), a caveat to state (THEORY.md §6).

**Scaling-curve fitting methodology.** For $N\in\{21,43,64,86\}$ songs (subsample the 86-song train split by a fixed seed; **keep the 14-track valid split fixed**), fit val SI-SDR$(N)$ to a saturating form and report the fit + CIs:
- log fit: $\widehat{\text{SI-SDR}}(N)=a+b\log N$; or power-law residual: $\text{SI-SDR}_\infty - c\,N^{-\gamma}$.
- Report $b$ (or $\gamma$) with a bootstrap CI; the **sign and magnitude of the slope at $N{=}86$** is the finding. Draw the 3-seed endpoint noise band on the curve. **Extrapolation caveat:** cannot exceed 86 songs without new data; state that the fit is descriptive, not predictive past the range (Saijo & Bando suggests non-saturation but does not license extrapolation).

## 6. Risks this literature implies
- **Remix independence caveat:** cross-song remixing produces musically incoherent mixtures (mismatched key/tempo); this may *understate* real-world gains or introduce a distribution the test set doesn't share — report it, don't hide it.
- **Small-N subsampling noise:** the {21,43,64,86} curve is sensitive to *which* songs are dropped; fix the subsample seed and draw endpoint noise bands so readers see signal vs noise.
- **Confound with capacity:** at 21 songs a large model overfits; hold architecture/regularization fixed across N so the curve isolates data, not capacity (ties to Direction 03's param-matching discipline).
- **Adjacent-prior honesty:** cite 1906.02618 so the contribution is framed as "factorized + scaling at compact scale," not "first study of MSS training data."
