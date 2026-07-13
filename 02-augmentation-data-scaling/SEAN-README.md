# Direction 02, explained — augmentation anatomy + the data-scaling curve

*A plain-language guide to this folder: the question, the design, the code, what you'll
run, and how to read whatever comes out. Builds on Direction 01's guide
([`../01-loss-function-study/SEAN-README.md`](../01-loss-function-study/SEAN-README.md)) —
read that first if you haven't; the model, metric, and seeds story is identical here.*

---

## 1. The question in one paragraph

We only have **86 training songs** (MUSDB18's usable train split). Everyone in this
field stretches them with the same three tricks, copied from repo to repo for years:
**(1) remixing** — build "Frankenstein songs" by taking the vocals from one track and
the instruments from another; **(2) random gain** — randomly turn each source up or
down; **(3) sign flip** — multiply the waveform by −1. Nobody has ever published how
much each trick is actually worth. And nobody has answered the sibling question: with
the tricks on, **do 86 songs even saturate a small model, or would more data keep
helping?** This direction measures both: knock out one trick at a time (a "leave-one-out"
study), and train on nested slices of 21 → 43 → 64 → 86 songs to draw the
**data-scaling curve**.

## 2. Why it's worth doing

- **It's the most industry-shaped question in the project.** "Should we buy more data,
  or tune augmentation, or get a bigger model?" is the daily triage of applied ML
  teams. This study answers it with controls for our regime.
- **It's pre-registered like the rest**: our bets and the exact win/lose rules were
  frozen before any run (see [`MASTER_PLAN.md §2`](MASTER_PLAN.md)).
- **It has a built-in lie detector.** The sign-flip trick provably cannot matter for
  our model type: flipping a waveform's sign leaves its spectrogram magnitudes — the
  only thing our network sees and is graded on — mathematically unchanged (the proof
  is two lines, [`THEORY.md`](THEORY.md) §3). So flip's measured effect *must* be ≈ 0.
  If it isn't, the experiment itself is broken and we stop and debug instead of
  publishing nonsense. A placebo arm, basically.

## 3. Our bets (pre-registered)

- **H-02a:** remixing is the single most valuable trick — removing it hurts more than
  removing any other, by more than the luck band.
- **H-02b:** the scaling curve is **still climbing** at 86 songs — i.e. our model is
  data-starved, not data-saturated. The concrete rule: going from 43 → 86 songs (one
  clean doubling) must gain more than the luck band.

Both bets can lose, and a loss is just as reportable: "augmentation barely matters" or
"86 songs is already plenty for a 10M-param model" would each be a genuinely useful,
slightly heretical finding.

## 4. How the design stays honest

- **One thing changes per arm.** Same network, same seeds, same data order, same number
  of steps — only the trick-switchboard or the song-count differs. Each trick even gets
  its own independent random stream, so turning one off can't secretly reshuffle the
  others (there's a unit test for exactly that).
- **The luck band.** Two cells (all-86-songs and only-21-songs) train 3× with different
  seeds; the spread defines σ_seed, and every claim must clear it.
- **Frugal by sharing.** The "everything on, 86 songs" cell is *bit-identical* to a
  cell Direction 01 already trains — the code hashes both configs and proves it — so we
  reuse those 3 runs instead of paying for them twice. Only **9 new GPU runs** total.
- **No re-rolls.** The 21/43/64-song subsets are drawn once, by a fixed seed, nested
  inside each other, and committed before training. If the 21-song draw happens to be
  weird, we *report* that; we don't quietly redraw until the curve looks nice.
- **One test-set touch.** Only the final 86-song cell ever sees the 50 test songs, once,
  to check the story generalizes.

## 5. What's in this folder

| Path | What it is |
|---|---|
| [`MASTER_PLAN.md`](MASTER_PLAN.md) | The frozen spec: bets, decision rules, run matrix, budget (≈13–25 T4-hours), gates. |
| [`THEORY.md`](THEORY.md) / [`theory/`](theory/) | The math: what remixing does to the training distribution (and the "instruments don't know about each other" approximation it makes), why flip is provably inert, what a 4-point curve can and can't tell you, how the luck band and the curve's confidence interval are computed. |
| [`research/`](research/) | Verified literature: the exact augmentation code in Open-Unmix/Demucs (with constants — fun fact: the real gain range is ×0.25–1.25, not the "±3–6 dB" folklore), and the closest prior study. |
| [`notebooks/01_augmentation_anatomy.ipynb`](notebooks/) | *See* the tricks: before/after waveforms and spectrograms, what a Frankenstein mixture looks like, the flip-inertness argument visually, subset balance tables. |
| [`notebooks/02_factorization_scaling_experiments.ipynb`](notebooks/) | The experiment: launch cells (RUN LATER), then the leave-one-out bar chart with the luck band, the scaling curve with its fitted "dB per doubling" slope, and pre-written interpretations to select from. |
| [`configs/`](configs/) | One YAML per run: 4 leave-one-out + 5 scaling + 2 contingency. `subsets/` fills at data-prep time. |
| `../singnet/analysis/scaling.py` | The tested math for the curve fit and the leave-one-out table — notebooks call it, never re-implement it. |
| [`paper/PAPER.md`](paper/PAPER.md) | The report scaffold; every result a ⟪placeholder⟫, every possible outcome pre-interpreted. |

## 6. What you'll run later (after Direction 01's data prep)

Everything resumable; ~13–25 T4-hours total.

1. **Make the subsets (no GPU, once):** `python scripts/make_subsets.py --manifest
   01-loss-function-study/configs/splits.csv --sizes 21 43 64 --seed 0 --out
   02-augmentation-data-scaling/configs/subsets/` — then commit the three tiny CSVs it
   writes (song names only).
2. **Dry run (no GPU):** `python scripts/run_sweep.py --direction 02 --dry-run` — prints
   each run's switchboard and song count so a mis-configured arm dies on paper, not on GPU.
3. **The 9 runs:** `python scripts/run_sweep.py --direction 02 --stage reduced`.
4. **Analysis:** open notebook 02; it reads the registry and draws everything.
5. **One test pass:** `python scripts/evaluate.py --checkpoints <3 full-86 ckpts> --split test`.
6. **Fill the paper**, selecting the pre-written branch that matches reality.

## 7. How to read the outcome

- **Remix dominates** → folklore confirmed, now with a number; remixing stays mandatory
  everywhere, and Direction 10 (make more data with a teacher model) gets a tailwind.
- **Remix doesn't dominate** → genuinely surprising; the plan pre-commits to checking
  whether it's a short-training artifact before believing it.
- **Nothing matters** → augmentation isn't the lever at this scale; one pre-registered
  full-budget re-check, then say it plainly.
- **Curve still rising** → we're data-starved: data-side ideas (more songs,
  pseudo-labels) outrank architecture tweaks. The slope literally prices a doubling of
  data in dB.
- **Curve flat** → the model, not the data, is the wall: boosts Direction 03
  (architecture), deflates Direction 10's premise — and we say so *before* Direction 10
  runs, which is exactly what pre-registration is for.
- **The flip "placebo" shows an effect** → we don't interpret anything; we debug.

## 8. New glossary entries

**Leave-one-out (LOO)** — measure a part's value by removing only it and seeing what
breaks. **Negative control / placebo arm** — a condition designed to show zero effect;
if it doesn't, your instrument is broken. **Nested subsets** — each smaller song set is
contained in the next, so the curve compares "less of the same," not "different data."
**dB per doubling** — the scaling curve's slope: how much quality one more doubling of
songs buys. **Interaction gap** — whether the tricks' individual values add up to the
value of all-at-once (synergy/redundancy check).
