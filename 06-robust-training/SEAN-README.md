# Direction 06, explained — training on dirty ground truth

*Plain-language guide to this folder. Same seeds/luck-band/pre-registration machinery
as before — see [`../01-loss-function-study/SEAN-README.md`](../01-loss-function-study/SEAN-README.md)
first.*

---

## 1. The question in one paragraph

Supervised separation training assumes the "isolated vocals" files are actually
isolated. In the real world they rarely are: drum microphones pick up the singer,
"vocal" stems carry instrument bleed, and big challenge organizers (SDX'23) consider
dirty training data important enough that they built entire corrupted datasets for it.
So: **if your training targets are contaminated, how much worse does your model
actually get — and can a one-line classical trick claw some of it back?** We contaminate
MUSDB18's training targets ourselves with a precise dial (ε = 0, 5, 15, 30% of the
accompaniment leaked into the vocal target), train the same model at each level, and
always grade on *clean* data.

## 2. The two clever bits

**1. A "perfect student" prediction line.** Here's the neat part: if the model learns
*exactly* what we feed it, we can compute — with pencil and paper, no training — what
its clean-test score must be. A perfectly-taught model reproduces the contaminated
target, so its error versus the true vocals is exactly the leaked ε·accompaniment, and
that converts to an SI-SDR number per song (it falls off like −20·log₁₀(ε)). That line
is a *null model*: after training we overlay the real curve on it.
- Real curve **on** the line → the model faithfully learns whatever you feed it; data
  cleanliness is a hard constraint, now priced in dB.
- Real curve **above** the line → the model partially *rejects* contamination
  (fascinating — likely thanks to the remixing augmentation scrambling the bleed).
- Real curve **below** → contamination also destabilizes training itself.

I verified the math in code during review: the implementation matches the closed form
to three decimals.

**2. Contamination that can't leak into grading.** The corruption code *structurally
refuses* to touch validation or test data — constructing it for those splits raises an
error (tested). The one bug class that would invalidate everything is made impossible
rather than merely avoided.

## 3. The defense we test

**Trimmed loss**: at every training step, compute each chunk's loss, and simply *skip
the worst 30%* (train on the best 12 of every 16). The folklore ("small-loss trick"):
networks learn clean patterns before memorizing garbage, so high-loss samples are
disproportionately the corrupted ones. Honest wrinkle we state up front: that theory
assumes *some* samples are clean, but our bleed contaminates **every** chunk equally.
The only way trimming can work here is subtler — chunks where the accompaniment happens
to be quiet are *effectively* cleaner, so trimming should act as
"curriculum-by-cleanliness." We don't just hope: the training loop logs the
accompaniment energy of kept vs dropped chunks, so the data will confirm or kill that
mechanism directly. We also run a control (trimming on *clean* data) to measure the
trick's price when nothing is wrong.

## 4. Our bets (pre-registered)

- **H-06a:** bleed hurts, monotonically and detectably (30% bleed costs more than the
  luck band and at least 0.5 dB) — with the *shape verdict* (on/above/below the
  prediction line) as the real scientific payload.
- **H-06b:** trimming at the worst level recovers a detectable fraction of the damage.
- Pre-registered surprises that would be findings, not failures: "the model shrugs off
  30% bleed" (implicit robustness), "trimming does nothing" (the classic trick has
  boundaries), "trimming hurts clean data" (the defense has a price).

## 5. What's in this folder

| Path | What it is |
|---|---|
| [`MASTER_PLAN.md`](MASTER_PLAN.md) | Frozen spec: the corruption model (mixture-preserving, exactly how real bleed redistributes), run matrix (only **10 new GPU runs ≈14–19 T4-h** — the clean cell is the shared baseline again), decision rules, gates. |
| [`THEORY.md`](THEORY.md) / [`theory/`](theory/) | The math: what the optimal model does under corrupted targets, the prediction-line derivation, why/when trimming can work under uniform contamination, honest comparison to the trimming theory's assumptions, the statistics. |
| [`research/`](research/) | Verified literature: SDX'23's corrupted datasets, the noisy-label classics (with real citations), recent MSS data-cleaning work. |
| [`notebooks/01_bleed_anatomy.ipynb`](notebooks/) | See ε-bleed: spectrograms at each level, the mixture-invariance property, the prediction line on toy stems, which chunks trimming drops. |
| [`notebooks/02_robustness_experiments.ipynb`](notebooks/) | The experiment: launch cells, then the headline figure (measured curve + luck band + prediction line), the recovery fraction, the mechanism telemetry, pre-written interpretations. |
| [`configs/`](configs/) | 13 configs (curve, defense, controls, contingencies). |
| `../singnet/data/corrupt.py`, `../singnet/losses/trimmed.py`, `../singnet/analysis/bleed.py` | The new machinery, all CPU-tested (suite now **292 green tests**). |
| [`paper/PAPER.md`](paper/PAPER.md) | Report scaffold with every outcome branch pre-written. |

## 6. What you'll run later (~14–19 T4-hours)

1. **Dry run (no GPU):** `python scripts/run_sweep.py --direction 06 --dry-run` —
   prints each run's ε and trim settings and confirms the eval-guard.
2. **The 10 runs:** `python scripts/run_sweep.py --direction 06 --stage reduced`.
3. **Prediction lines (CPU minutes):** `python -m singnet.analysis.bleed --split valid
   --epsilons 0.05 0.15 0.30`.
4. **One clean test pass:** `python scripts/evaluate.py --direction 06 --split test`.
5. **Fill the paper**, selecting the branches reality picked.

## 7. How to read the outcome

- **Curve on the prediction line** → data quality is destiny; the chart tells you how
  clean stems must be for a target quality. (Also directly prices the "teacher error"
  risk in Direction 10, where a big model's imperfect outputs become training targets.)
- **Curve above the line** → the standard training recipe has built-in decontamination;
  good news for everyone with slightly dirty data.
- **Trimming works (and telemetry shows the energy-ranking)** → a free robustness flag
  for the whole project, with its mechanism demonstrated, not assumed.
- **Trimming fails** → the small-loss trick needs clean samples to find; under uniform
  contamination use robust losses or data cleaning instead. Still a publishable-shaped
  boundary result.

## 8. New glossary entries

**Stem bleed** — audio leaking between supposedly-isolated recordings. **Label noise /
corrupted targets** — training answers that are partly wrong. **Dose–response curve** —
damage as a function of contamination level. **Null model / prediction line** — the
computable score of a model that learns the corruption perfectly; the baseline for
detecting anything interesting. **Trimmed loss / small-loss trick** — skip the
highest-loss samples each step, hoping they're the corrupted ones. **Telemetry** —
in-training logging that tests the *mechanism*, not just the outcome.
