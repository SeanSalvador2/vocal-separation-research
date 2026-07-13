# Direction 01, explained — the loss-function study

*A plain-language guide to everything in this folder: what we're asking, why it
matters, how the experiment works, what the code does, and exactly what you'll run
once you have Colab Pro. Written for you (Sean) — smart recent grad, no
source-separation background assumed.*

---

## 1. The question in one paragraph

When you train a neural network you must pick a **loss function** — the single number
the optimizer pushes downhill, the network's definition of "wrong." Every famous vocal
separator picked differently: Spleeter penalizes plain spectrogram differences (L1),
Open-Unmix penalizes squared differences (MSE), Demucs penalizes waveform differences.
Meanwhile, everyone *evaluates* with SI-SDR — a decibel score of separation quality.
So an obvious idea: **why not train directly on SI-SDR?** Make the exam the homework.
The literature hints this obvious idea doesn't help — but nobody has actually run the
clean head-to-head at the scale we care about (a small ~10M-parameter model, trained
only on the standard 100-song MUSDB18 dataset). That's this study. We train the *same*
network five times, changing *only* the loss, and see what actually matters.

## 2. Why this is worth doing

- **It settles our own default.** Six more research directions in this repo will train
  this same U-Net. Whatever wins here becomes the house loss, chosen by evidence.
- **It's a real gap.** We verified (July 2026) that no published paper runs this
  controlled comparison for compact models. Any outcome — "fancy loss wins," "fancy
  loss does nothing" — is a publishable-shaped finding.
- **It demonstrates the scientific process.** We wrote down our bets *before* running
  anything (pre-registration), fixed the rules for declaring a winner, and pre-wrote
  interpretations for every possible outcome so we can't fool ourselves after the fact.

## 3. Our bets (the pre-registered hypotheses)

- **H-01a:** training on SI-SDR will **not** beat the simple L1 loss, when judged on
  SI-SDR itself. (If it does beat it, great — that's a clean "refuted," equally
  reportable.)
- **H-01b:** adding a "multi-resolution STFT" term (borrowed from speech-synthesis
  research, where it removes robotic artifacts) won't move the SI-SDR score — but
  humans in a blind listening test will say it **sounds less glitchy**.

The exact numeric decision rules (what counts as "beat," what happens on a tie) are in
[`MASTER_PLAN.md §2`](MASTER_PLAN.md) — they were frozen before any training.

## 4. The five contestants, in plain words

Think of the network's job as painting a **mask** over a spectrogram (a picture of the
song, time → and frequency ↑): bright where the vocals are, dark where they aren't.
Multiply the mask by the mixture and you get the vocals. The five losses are five ways
of grading that painting:

1. **`l1mag`** — "How far off is each pixel, on average?" (Spleeter's choice.)
2. **`msemag`** — "How far off, but *squared* — big mistakes hurt extra." (Open-Unmix.)
3. **`logl1mag`** — same as #1 but on a log scale, so quiet details (breaths, reverb
   tails) count as much as loud ones.
4. **`sisdr`** — skip the picture-grading entirely: rebuild the audio, measure the
   actual SI-SDR score, push on that. ("Train on what you test.") One catch we handle:
   SI-SDR divides by the target's energy, so a chunk where the singer is *silent*
   breaks it (divide by ~zero). We skip those chunks for this loss only — and we
   *count* how often (that count feeds Direction 08, the silence study).
5. **`l1mrstft`** — loss #1 plus the artifact-smoothing term from speech synthesis,
   which compares the rebuilt audio at three different spectrogram resolutions at once.

**The golden rule of the experiment:** everything else is bit-for-bit identical across
the five — same network, same random initialization per seed, same shuffled data, same
optimizer, same number of steps. Only the loss differs. (This is enforced by design and
partly by unit tests.) So any difference in the outcome is *caused by the loss*.

## 5. How we judge — and why there are "seeds"

- **The score:** SI-SDR, in dB, on the vocals — think "how many times cleaner is the
  vocal than the leftover bleed," on a log scale. 0 dB ≈ doing nothing. Our classical
  baseline in StemCraft scores ~+4.4; Demucs ~+9.6. A tiny model like ours should land
  between them.
- **Seeds:** each arm trains 3 times with different random seeds (0, 1, 2). Training is
  noisy; the spread across seeds tells us the size of "luck." A loss only truly wins if
  it beats another by *more than the luck band*. This is the single most important
  honesty device in the study.
- **Validation vs test:** 14 songs are set aside for all decision-making; the 50 test
  songs are touched **exactly once**, at the very end. This prevents the classic sin of
  tuning on the exam.
- **Oracles:** we also score two "cheating" masks computed from the ground truth — the
  best any mask-based model could possibly do. That line on the chart tells us how much
  headroom our architecture even had.
- **Ears:** for H-01b, five short clips get a blind A/B test ("which has fewer
  glitches?") with a few friends as raters. Small and informal by design — we report it
  as supporting evidence, never as proof.

## 6. What's in this folder (and what each thing is for)

| Path | What it is |
|---|---|
| [`MASTER_PLAN.md`](MASTER_PLAN.md) | The frozen, self-contained experiment spec. If a file disagrees with it, the plan wins. |
| [`THEORY.md`](THEORY.md) / [`theory/theory.tex`](theory/theory.tex) | All the math, derived: STFT, masks, each loss, why SI-SDR is scale-proof, why "optimize X" ≠ "maximize Y", the parameter-count table (9,835,745 — verified by a test). |
| [`research/`](research/) | The literature: what each cited paper actually says (verified), and where our gap is. |
| [`notebooks/01_data_and_eda.ipynb`](notebooks/) | Get the dataset, look at it properly: how loud, how silent, where vocal energy lives. Motivates design choices. |
| [`notebooks/02_pipeline_and_model.ipynb`](notebooks/) | See the machinery: spectrograms round-trip, augmentation before/after, the mask the net must learn, the architecture, a tiny "can it learn at all?" overfit test. |
| [`notebooks/03_loss_study_experiments.ipynb`](notebooks/) | The experiment itself: launch the 15+3 runs, then all analysis — ranking chart with the luck band, curves, the one test pass, listening-test entry, and conclusions. |
| `../singnet/` (repo root) | The real code: data loading, augmentation, the U-Net, all five losses, training loop, evaluation. Notebooks only *orchestrate*; logic lives here, unit-tested. |
| `../tests/` | 69 CPU tests (no dataset, no GPU needed) proving the machinery is correct *before* we spend GPU money: spectrogram round-trips, loss math on hand-built cases, determinism, resume logic. |
| [`configs/`](configs/) | One YAML per run — the 15 sweep runs, the smoke test, exploratory extras. An experiment = a config file, never "I edited a constant." |
| [`paper/PAPER.md`](paper/PAPER.md) | The report, pre-scaffolded. Every result is a ⟪placeholder⟫; the Discussion already contains a written interpretation for *every* possible outcome — we just select the branch reality picks. |
| `results/` | Where run registries and score CSVs land (committed). Model weights stay on Drive — never in git. |

## 7. Exactly what you'll run later (when Colab Pro is ready)

Everything below is scripted; the notebooks contain the same steps with explanations.
Rough total: **~31–43 hours of T4 GPU time**, spread over ~3 weeks of casual sessions —
all resumable, so disconnects cost minutes, not runs.

1. **Install + data (once, ~1 h, no GPU):** run the setup cell in notebook 01; it
   downloads MUSDB18 (4.7 GB) to your Drive and decodes it (`scripts/prepare_data.py`).
2. **Smoke test (< 1 GPU-h):** `python -m singnet.train --config
   01-loss-function-study/configs/smoke_overfit.yaml` — the net must nail one chunk
   (> +20 dB) and beat "do nothing" on 5 songs. If this fails, we fix code, not spend.
3. **The sweep (~20–27 GPU-h):** `python scripts/run_sweep.py --stage reduced` — the 15
   runs. Interruptible; re-running resumes.
4. **Analysis freeze (no GPU):** run notebook 03's analysis cells; it names the top-2.
5. **Confirmations (~10–14 GPU-h):** `python scripts/run_sweep.py --stage full`.
6. **The one test pass + oracles (minutes):** `python scripts/evaluate.py … --split
   test --oracles --museval`.
7. **Listening kit (no GPU):** `python scripts/render_listening_kit.py …`, send the
   clips to ~5 friends, enter answers in notebook 03.
8. **Fill the paper:** notebook 03 prints every ⟪placeholder⟫ value; paste into
   `paper/PAPER.md`, select the matching pre-written interpretation branches, done.

## 8. How to read the outcome (no spin possible)

- **SI-SDR arm ≤ L1 within the luck band** → our bet held: at small scale, "train on
  what you test" is a myth; simple L1 stays our default. 
- **SI-SDR arm wins clearly** → our bet was wrong, which is fine — that's a sharper
  finding, and `sisdr` becomes the house loss for the other directions.
- **Too close to call even across seeds** → also a finding: papers claiming small
  loss-function wins from single runs at this scale are probably reporting luck.
- **MR-STFT: same score, better sound** → metrics have a blind spot we can name; ship
  the artifact-friendly loss in the karaoke product.
- Whatever happens, the skip-rate counter from the `sisdr` arm hands Direction 08 its
  opening measurement for free.

## 9. Glossary (30 seconds)

**Spectrogram** — picture of sound: time →, frequency ↑, brightness = energy.
**STFT/iSTFT** — the transform to/from that picture. **Mask** — per-pixel dimmer knob
the net predicts. **SI-SDR** — separation quality in dB, immune to volume tricks.
**Seed** — the dice-roll that sets random initialization/shuffling; same seed = same
run. **Pre-registration** — writing bets + rules before running, so results can't be
massaged. **Oracle** — the impossible-best mask computed from ground truth; our
ceiling. **Ablation/arm** — one configuration in a controlled comparison.
