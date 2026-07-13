# Direction 03, explained — the mini band-split experiment

*Plain-language guide to this folder. Builds on Directions 01–02's guides
([`../01-loss-function-study/SEAN-README.md`](../01-loss-function-study/SEAN-README.md),
[`../02-augmentation-data-scaling/SEAN-README.md`](../02-augmentation-data-scaling/SEAN-README.md));
the model, metric, seeds, and "luck band" story is the same — read those first.*

---

## 1. The question in one paragraph

Every top-ranked vocal separator since ~2022 is built on one idea: **don't process the
whole spectrogram with one network — slice it into frequency bands and give each band
its own dedicated sub-network.** Bass frequencies get their own specialist, mids get
one, highs get one. The best systems (BSRNN, BS-RoFormer, Mel-RoFormer, SCNet) all do
this — but they're huge models, often trained with extra private data, and they change
five other things at the same time. So nobody actually knows the answer to a very
simple question: **does the band-splitting idea, by itself, help a small model like
ours?** We test exactly that: three networks with (almost exactly) the **same number of
parameters**, same everything else, differing only in the front end — no split, split
into 3 equal bands, or split into 3 "mel" bands.

## 2. What's "mel" and why a third arm?

A **mel scale** spaces bands the way human hearing does: fine resolution at low
frequencies, coarse at high. Our mel bands are ≈ 0–1.5 kHz, 1.5–6.4 kHz, and
6.4–22 kHz — the first two are where almost all vocal energy lives, so mel spacing
concentrates network capacity there. The equal-width arm (0–7.4, 7.4–14.7,
14.7–22 kHz) is the **control** that lets us tell two stories apart:

- If *equal-width splitting* already beats no-splitting → the win comes from bands
  having dedicated capacity at all.
- If *mel beats equal-width* on top → **where** you put the bands matters too.

Without the middle arm, a mel win would smush both effects together — which is
exactly the confusion the big papers leave behind.

## 3. The fairness trick this study lives or dies on

Comparing architectures is only meaningful if no arm secretly gets more capacity. A
fun, non-obvious fact makes this subtle: **a convolution layer's parameter count
doesn't depend on how big its input is** — chopping the spectrogram into 3 bands and
running 3 towers doesn't divide the parameters by 3, it *triples* them (3 towers!)
unless you shrink the towers. So we solved for the tower width that brings the
band-split models back to the baseline's size. Result (all committed and enforced by
unit tests, see [`results/param_match_table.md`](results/param_match_table.md)):

| Arm | Parameters | Difference |
|---|---|---|
| No split (baseline) | 9,835,745 | — |
| Equal-width split | 9,841,896 | **+0.06 %** |
| Mel split | 9,841,896 | **+0.06 %** |

A bonus discovery from the same math: at equal *parameters*, the band-split models do
about **half the computation** of the baseline (compute, unlike parameters, *does*
scale with input size). So even a tie in quality would be an efficiency win — we
report that honestly as a side result, not the headline.

## 4. Our bet, and the honest odds

**H-03 (pre-registered):** mel > equal-width > no-split, each gap bigger than the
luck band. Truthfully, the literature gives this bet real odds of failing at our
scale — a March 2026 study found that even professionals couldn't reproduce the
famous band-split model's published numbers. That's why the plan pre-registers **all
seven possible outcomes** (full win, two different partial wins, a tie, an outright
loss for splitting, a budget-dependent flip, and "wins for the wrong reason") each
with its interpretation written *before* training. A tie is a genuinely useful
headline: *"band-splitting is a big-model phenomenon."*

## 5. How we'll know **why** (not just whether)

The killer figure is the **per-band error breakdown**: we measure separation quality
inside six fixed frequency slices (chosen neutrally — the union of both layouts'
edges). If mel-split wins *because* of dedicated capacity, its advantage should sit in
the vocal-heavy bands (~100 Hz–4 kHz). If instead the gains show up in empty top-end
spectrum, or damage clusters at band boundaries (the seams where towers can't see
their neighbors), the pre-written interpretations say so. One more built-in honesty
check: our dataset's compressed audio contains almost nothing above ~16 kHz, so the
mel top band is partly "dead air" — that's a real property of mel-on-this-data, stated
up front rather than discovered in a reviewer comment.

## 6. What's in this folder

| Path | What it is |
|---|---|
| [`MASTER_PLAN.md`](MASTER_PLAN.md) | The frozen spec: hypothesis with decision rules, the 3-arm design, run matrix (only **8 new GPU runs** — the baseline arm is *the same runs* Directions 01/02 already schedule, proven by config-hash), budget ≈16–25 T4-h, gates. |
| [`THEORY.md`](THEORY.md) / [`theory/`](theory/) | The math: mel-scale derivation (the exact band edges), why conv parameters ignore input size (the whole fairness story), the closed-form width equation and its solution, boundary/receptive-field analysis, the per-band metrics, the statistics. |
| [`research/`](research/) | Verified literature: the band-split family, what each paper actually showed, and the replication cautionary tale. |
| [`notebooks/01_bandsplit_architecture.ipynb`](notebooks/) | See the idea: both band layouts drawn over a real spectrogram, where vocal energy lives, the architecture, the param-match table, the compute comparison. |
| [`notebooks/02_bandsplit_experiments.ipynb`](notebooks/) | Run and read the experiment: launch cells (RUN LATER), the 3-arm chart with the luck band, the per-band mechanism figure, verdict logic, pre-written interpretations. |
| [`configs/`](configs/) | The 6 variant runs + smokes + contingencies; FULL-budget confirm configs are generated after the sweep picks a winner. |
| `../singnet/models/bandsplit_unet.py` | The new model: 3 towers, padding/cropping, frequency-stitched skip connections. Tested (band routing round-trips exactly; parameter counts pinned). |
| `../singnet/eval/banded.py` | The per-band metrics (tested on synthetic signals). |
| [`paper/PAPER.md`](paper/PAPER.md) | The report scaffold with all seven interpretation branches pre-written. |

## 7. What you'll run later (~16–25 T4-hours)

1. **Sanity on paper (no GPU):** `python scripts/match_params.py --report` and
   `python scripts/run_sweep.py --direction 03 --dry-run`.
2. **Two smoke tests (~30 min GPU):** each variant must nail the single-chunk overfit
   test, same bar as the baseline passed.
3. **The 6 runs:** `python scripts/run_sweep.py --direction 03 --stage reduced`.
4. **Analysis freeze:** notebook 02 computes E1/E2 and names the winner.
5. **Two full-budget confirmations:** `... --stage full` (guards the "maybe it just
   needed longer" objection).
6. **One test pass + per-band + efficiency:** `python scripts/evaluate.py ... --banded
   --efficiency`.
7. **Fill the paper**, selecting the branch reality picked.

## 8. New glossary entries

**Band-splitting** — running separate sub-networks on different frequency ranges.
**Mel scale** — frequency spacing that mimics human hearing (fine low, coarse high).
**Param-matched** — all arms get the same parameter budget, so capacity can't explain
the outcome. **MACs / FLOPs** — how much arithmetic a model does; can differ wildly
even at identical parameter counts. **Tower** — one band's private encoder.
**Mechanism figure** — the plot that tests *why* something won, not just whether.
