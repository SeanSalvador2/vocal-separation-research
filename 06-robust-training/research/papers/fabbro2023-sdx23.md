# The Sound Demixing Challenge 2023 – Music Demixing Track (robust MSS)

Giorgio Fabbro, Stefan Uhlich, + 24 co-authors. TISMIR 7(1):63–84 (2024) / arXiv **2308.06979** (14 Aug 2023, rev. Apr 2024). | Verified: WebSearch (arXiv abs + TISMIR + construction details) — not on HF paper_search. Verified 2026-07-13. **CORE** for Direction 06 — the paper that *invented* the robust-MSS task and whose corruption models parameterize the experiment.

## 1. Problem & context
Real training datasets for MSS are dirty: stems get mislabeled, and studio recordings have **bleeding** (one instrument audible in another's mic/stem). SDX'23's MDX track **formalizes "robust music source separation"**: *training MSS models in the presence of errors in the training data*. The organizers flag corrupted training data as an under-studied, industry-critical axis — the exact gap Direction 06 works in.

## 2. Method — the two corruption datasets (Direction 06's parameterization)
Both are built from **error-free recordings** (203 songs selected from **MoisesDB** — 240 tracks / 45 artists / 12 genres) that are then **artificially corrupted** by Sony:

- **SDXDB23_LabelNoise:** simulates **label noise** — *erratic instrument groupings* as would arise from **automatic, metadata-based stem generation** in production (e.g. a guitar routed into the wrong stem). The *content* is right but the *stem assignment* is wrong.
- **SDXDB23_Bleeding:** simulates **bleeding** — **each stem of a song also contains partial content from the other stems of the *same song*** (unintended cross-instrument overlap during recording). Bleeding components are processed to make the simulation realistic (per-song, variable).

**Relation to Direction 06's controlled ε-bleed model.** Direction 06 uses a **single-parameter simplification** of `SDXDB23_Bleeding`: corrupt the vocal *target* as
$$\tilde v = v + \varepsilon\,a,\qquad a = \text{drums}+\text{bass}+\text{other},\ \varepsilon\in\{0,0.05,0.15,0.30\},$$
leaving the **mixture $x=v+a$ unchanged** (only the *target* is corrupted). This is a clean, monotone knob (constant ε, accompaniment→vocal only) where SDX'23's construction is realistic-but-uncontrolled (per-song, all-directions, variable magnitude). Direction 06 trades SDX'23's realism for a **measurable degradation curve** and explicitly cites this as the simplification. (SDX'23's LabelNoise — wrong *grouping* — is a different corruption mode Direction 06 does not simulate; it focuses on bleed.)

## 3. Key results (exact)
- **Best-performing SDX'23 system: > +1.6 dB SDR over the MDX'21 winner.** CONFIRMED (WebSearch). Establishes that robust-training methods measurably help even at SOTA scale.
- Winning noise-robust approach (KUIELab, **TFC-TDF-UNet v3**, arXiv **2306.09382**): a **loss-masking** scheme for noise-robust training — a *mitigation precedent* directly relevant to Direction 06's loss-side defense.

## 4. Limitations & caveats
- The datasets are MoisesDB-derived and access-gated; Direction 06 does **not** need them — it constructs its ε-bleed corruption from MUSDB's *own* clean stems (no new data), which is the point (cheap, controlled, reproducible).
- The exact per-song bleeding parameters/SNRs of `SDXDB23_Bleeding` are not fully read here (PDF/TISMIR body not exhaustively fetched) — Direction 06's ε-model is inspired-by, not a replica.

## 5. Relevance to this project (Direction 06)
- **This is the paper that legitimizes Direction 06:** the challenge organizers themselves call corrupted training data an under-explored, important problem.
- **What we replicate:** the *bleeding* corruption concept (as the controlled ε-model) and the finding that noise-robust training helps.
- **What we extend:** nobody reports a **degradation curve (SI-SDR vs ε) + a cheap loss-side mitigation for a *compact* model** — Direction 06 produces both, on MUSDB-only, at notebook budget.
- **Mitigation lineage:** SDX'23's loss-masking winner + the noisy-label canon (`noisy-label-canon.md`) justify Direction 06's "discard top-k% highest-loss chunks" defense.

## 6. Verification notes
- Title/authors(Fabbro, Uhlich +24)/TISMIR 7(1):63–84/arXiv ID: CONFIRMED (WebSearch).
- Robust-MSS framing + **LabelNoise/Bleeding** names + construction (203 MoisesDB songs; bleeding = cross-stem partial content per song; label-noise = wrong groupings) + **>+1.6 dB over MDX'21 winner**: CONFIRMED (WebSearch). Matches RESEARCH_NOTES §6 / RESEARCH_DIRECTIONS §1.3.
- ε-bleed model $\tilde v=v+\varepsilon a$: **OUR construction** (a controlled simplification of SDXDB23_Bleeding), clearly marked as such.
</content>
