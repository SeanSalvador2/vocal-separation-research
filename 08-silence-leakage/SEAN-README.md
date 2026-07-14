# Direction 08, explained — ghost vocals, and the metric that finally sees them

*Plain-language guide to this folder. Same machinery as before — see
[`../01-loss-function-study/SEAN-README.md`](../01-loss-function-study/SEAN-README.md)
first.*

---

## 1. The question in one paragraph

Play a karaoke track made by a separator and the most annoying failure isn't a slightly
muddy chorus — it's **ghost vocals**: faint singing haunting the instrumental breaks
where the vocal track should be dead silent. Here's the scandal: **the field's standard
metrics literally cannot see this failure.** The official scorer (museval) throws away
all frames where the reference vocal is silent (we verified this in its source code),
and SI-SDR divides by the reference's energy — zero in silence — so it's mathematically
undefined exactly where ghosts live. Meanwhile, a training-pipeline choice everyone
makes casually — *which* 6-second chunks to sample from songs that are mostly not
singing — plausibly controls this failure: a model that never trains on silent-vocal
passages never learns to output silence. This direction builds the missing measuring
stick, then runs the controlled sampling experiment with it.

## 2. Part one: the SLR metric (our main contribution)

**SLR — Silence-Leakage Ratio.** Find the passages where the true vocal is genuinely
silent (quieter than −60 dBFS for at least half a second). In just those passages, ask:
*how much energy did the separator's "vocal" output contain, compared to the full
mixture's energy there?* Express it in dB. That's it — about 80 lines of code, and it's
readable on sight:

- **0 dB** = the separator just passed the mixture through (worst case).
- **−10 dB** = it leaked 10% of the mixture's energy (audible ghosts).
- **Very negative** = properly silent (best).

I verified these anchors numerically during review: a do-nothing separator scores
*exactly* 0.0, a 10% leak scores *exactly* −10.0, a perfect one bottoms out at the
floor, and a track with no silent passages returns "not applicable" rather than a fake
number. The threshold choices are pre-registered and every conclusion gets re-checked
at three thresholds, so nobody can accuse us of tuning the ruler.

One honest limitation, stated up front: SLR measures *energy*, not *what* leaks — a
quiet hi-hat bleed and a quiet vocal ghost score the same. Human ears (the project's
listening checks) keep that part of the job.

## 3. Part two: the sampling experiment

Four ways to pick training chunks, everything else bit-for-bit identical:

1. **Uniform** — pick anywhere at random. (What Open-Unmix really does — code-verified.
   Also literally the same runs as our shared baseline: five directions now reuse that
   one 3-seed cell.)
2. **Energy-weighted** — prefer vocal-active regions, but keep a guaranteed 10% floor
   of random sampling so silence is *down-weighted, never excluded*.
3. **Drop-silent** — the tempting "don't waste compute" move: never sample chunks
   where the vocal is silent.
4. **Curriculum** — start uniform, gradually shift to energy-weighted.

**Our bets (pre-registered):** energy-weighting buys overall quality (H-08a), and
drop-silent poisons silence behavior — measurably worse SLR (H-08b). Together they'd
form a genuine **tradeoff**, drawn as a two-axis picture (quality on one axis, leakage
on the other, four policy dots with seed spreads, plus the "do nothing" and
"theoretically perfect" anchor points). We pre-committed *not* to mash the two axes
into one score — if there's a tradeoff, showing it honestly *is* the result.

Two anti-self-deception devices worth knowing about:
- **Checkpoint selection is blind to SLR** (enforced by a test): we always pick each
  run's best checkpoint by quality alone, so the leakage comparison can't be
  cherry-picked.
- **Exposure telemetry**: the training loop logs what fraction of chunks each policy
  *actually* served. If results are null, this tells us whether the policies even
  differed in practice — a null with a diagnosis instead of a shrug.

## 4. What's in this folder

| Path | What it is |
|---|---|
| [`MASTER_PLAN.md`](MASTER_PLAN.md) | Frozen spec: the metric definition, the four policies as exact formulas, decision rules, only **8 new GPU runs ≈11–15 T4-h**, gates. |
| [`THEORY.md`](THEORY.md) / [`theory/`](theory/) | The math: why museval/SI-SDR are structurally blind (from primary sources), SLR's properties and anchors, each policy's expected silent-chunk diet in closed form, the two-metric statistics. |
| [`research/`](research/) | Verified literature — including the honest novelty audit (an Aug-2025 paper attacks the same problem via architecture; the metric + tradeoff study are ours). |
| [`notebooks/01_silence_anatomy.ipynb`](notebooks/) | See the problem: how much silence real tracks contain, the region-finder walked through, the anchors demonstrated, each policy's sampling weights drawn over a song. |
| [`notebooks/02_sampling_experiments.ipynb`](notebooks/) | The experiment: launch cells, then the headline two-axis plane, threshold sensitivity, the exposure-vs-outcome mechanism plot, pre-written interpretations. |
| `../singnet/metrics/slr.py` | The metric itself — the artifact other projects could steal. |
| `../singnet/data/sampling.py` | The four policies (uniform reproduces the old behavior bit-for-bit — regression-tested, which is what keeps the shared baseline shareable). |
| [`paper/PAPER.md`](paper/PAPER.md) | Report scaffold, every branch pre-written. |

Suite is now at **358 green tests**.

## 5. What you'll run later (~11–15 T4-hours)

1. **Energy profiles (CPU, once):** `python scripts/prepare_data.py
   --write-energy-profiles --out $SHARD_ROOT`.
2. **Dry run:** `python scripts/run_sweep.py --direction 08 --dry-run`.
3. **The 8 runs:** `python scripts/run_sweep.py --direction 08 --stage reduced`.
4. **One test pass with SLR:** `python scripts/evaluate.py --direction 08 --split test
   --slr`.
5. **Fill the paper** from the pre-written branches.

## 6. How to read the outcome

- **The predicted tradeoff appears** → sampling is a real knob with a real price;
  quality products pick energy-weighting, karaoke products protect silence; SLR joins
  the standard scoreboard.
- **Free lunch (drop doesn't hurt silence)** → the mask architecture gives silence for
  free; focus sampling on vocals guilt-free. Genuinely useful heresy.
- **Pure downside (no quality gain, silence still poisoned)** → the strongest caution
  against lore: activity filtering is all risk. Ship uniform.
- **Nothing moves** → chunk sampling is a non-lever at this scale; the telemetry says
  whether that's about the policies or about the dataset. Community tuning effort saved.

## 7. New glossary entries

**Ghost vocals / leakage** — audible vocal residue where there should be silence.
**SLR** — our metric: leaked energy vs mixture energy in truly-silent vocal passages,
in dB. **dBFS** — decibels relative to digital full scale (0 = the loudest possible;
−60 = very quiet). **Sampling policy** — the rule for choosing training chunks.
**Exposure** — the fraction of training chunks whose vocal was silent; what the policy
actually did. **Selection blindness** — choosing checkpoints without looking at the
metric under dispute, so comparisons stay fair. **Tradeoff plane** — a two-axis plot
where refusing to average the axes is the honest move.
