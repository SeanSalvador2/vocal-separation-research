# Direction 05, explained — LoRA: teaching an old separator new tricks, cheaply

*Plain-language guide to this folder. The seeds/luck-band/pre-registration machinery is
the same as Directions 01–03 — see
[`../01-loss-function-study/SEAN-README.md`](../01-loss-function-study/SEAN-README.md)
first. This one has the strongest "nobody has published this" claim of the whole
project.*

---

## 1. The question in one paragraph

Suppose you have a good pretrained vocal separator, but *your* music library is
different from what it was trained on — low-bitrate rips, one dominant genre, live
recordings. You could **fine-tune the whole model** on your data (works, but now you
store a full model copy per situation, and the model may get *worse* on normal songs —
"catastrophic forgetting"). Or you could use **LoRA**: freeze the original model and
learn only a tiny low-rank "diff" on top — the technique that took over the LLM world
because it usually gets ~full quality at ~1% of the trainable weights. **Nobody has
ever published whether LoRA works for source separation.** We checked (July 2026,
logged searches): LoRA exists for music *generation*, *tagging*, even *beat tracking* —
not separation. This direction runs that missing experiment.

## 2. The cast

- **The host model:** Open-Unmix (`umxhq`) — the field's reference open separator
  (MIT-licensed, ~8.9M parameters). Its core is a **BiLSTM** (a recurrent network),
  which makes this spicier: LoRA's track record is on Transformers, and wrapping a
  recurrent layer is genuinely non-standard plumbing.
- **The five recipes** (what's allowed to change during adaptation):
  1. **Zero-shot** — change nothing (the "do you even need this?" baseline).
  2. **Head-only** — retrain just the output layer (23.7% of weights — the crude classic).
  3. **LoRA rank 4** — tiny diff: **1.27%** of weights.
  4. **LoRA rank 16** — small diff: **4.85%** of weights.
  5. **Full fine-tune** — everything (100%, the ceiling to match).
- **The two domains** (built deterministically from data we already planned to have —
  the original idea of using MUSDB18-HQ fell to its access restriction, so I redesigned
  around it):
  - **T1: 64 kbps codec-crushed audio** — the "karaoke from a bad rip" scenario. Each
    stem is re-encoded separately so the math of supervised training stays exact.
  - **T2:** a genre subset *if* official genre labels turn out to be available at
    data-prep time, *else* a pink-noise "live recording" domain — the rule (not a vibe)
    is pre-registered, because we honestly don't know yet whether MUSDB ships genres.

## 3. Our bets (pre-registered)

- **H-05a:** LoRA rank-16 recovers **≥90%** of whatever improvement full fine-tuning
  achieves, while training **<5%** of the weights. With a built-in sanity precondition:
  if full fine-tuning itself barely gains anything (the domain shift too mild), the
  hypothesis is declared *not evaluable* rather than trivially "supported" — no fake
  wins.
- **H-05b:** LoRA **forgets less**: after adapting, it loses less performance on
  normal (source-domain) songs than full fine-tuning does. There's even fresh theory
  (Feb 2026) arguing low-rank updates *can't* drift as far — we get to test it on a
  regression model.

## 4. The engineering trick (and why you can trust it)

Wrapping an LSTM with LoRA is the risky part — PyTorch hides the LSTM's four internal
gates inside big stacked weight matrices, and it *caches* them, so a naive wrapper
silently does nothing. Our implementation re-injects the LoRA-modified weights before
every forward pass, and the test suite proves the two properties that matter:

- **Start = zero-shot, exactly.** LoRA's B-matrix starts at zero, so before training
  the wrapped model must equal the original *bit for bit*. Measured difference: **0.0**
  (I re-verified this myself during review, independently of the builder's tests).
- **Merge-back is exact.** After training you can fold the diff into the weights
  (so inference costs nothing extra). Measured round-trip difference: **0.0**.

All of this runs on a **mock** Open-Unmix (same shapes, random weights) so the entire
test suite works with zero downloads — the real checkpoint is only fetched later, on
Colab, with its version and checksum recorded.

## 5. One honest correction along the way

My planning estimate said head-only ≈ 18% of weights. Building against the *verified*
architecture showed it's **23.73%** (the output layer maps to all 2049 frequency bins,
not the cropped 1487 the input uses). The numbers that the hypothesis depends on —
LoRA at 1.27%/4.85%, both under the 5% bar — were unaffected. The correction is logged
in `results/DEVIATIONS.md` and the literature note; the frozen plan stays frozen with
the deviation on record. That's the pre-registration system working as intended.

## 6. What's in this folder

| Path | What it is |
|---|---|
| [`MASTER_PLAN.md`](MASTER_PLAN.md) | The frozen spec: recipes, domains + the T2 rule, the LR-fairness protocol (each recipe gets its best learning rate from equal cheap probes — a single shared LR would rig the race), decision rules, budget ≈18–24 T4-h. |
| [`THEORY.md`](THEORY.md) / [`theory/`](theory/) | The math: LoRA end-to-end, what a low-rank update on gate-stacked LSTM weights means, the exact parameter accounting, why re-encoding stems separately keeps training math exact, the fairness and statistics arguments. |
| [`research/`](research/) | The verified literature + the gap check that makes this the novelty flagship. |
| [`notebooks/01_lora_anatomy.ipynb`](notebooks/) | Tour the host model, see where each recipe attaches, the params-per-recipe chart, what 64 kbps does to a spectrogram. |
| [`notebooks/02_lora_adaptation_experiments.ipynb`](notebooks/) | Probes → runs → the headline figure: quality gained vs weights trained, plus the forgetting table and pre-written interpretations. |
| [`configs/`](configs/) | 12 run configs + probe template (T2 configs marked "pending" until the rule resolves). |
| `../singnet/peft/` | The LoRA/wrapper/fine-tune code, all mock-tested (suite now at 218 green tests). |
| [`paper/PAPER.md`](paper/PAPER.md) | Report scaffold with six pre-written outcome branches. |

## 7. What you'll run later (~18–24 T4-hours)

1. **Build the domains (CPU):** `python scripts/make_domains.py --domain t1_aac64 ...`
   then `--resolve-t2` (prints whether T2 = genre or noise, and why).
2. **Checkpoint sanity (GPU minutes):** `python -m singnet.peft.finetune_umx --sanity`
   — downloads `umxhq`, recomputes the trainable-share table against the real weights,
   checks zero-shot sanity, and measures how big the T1 shift actually is.
3. **LR probes (~2h):** `python scripts/run_sweep.py --direction 05 --stage probes`.
4. **The 12 runs:** `... --stage main`.
5. **One consolidated test session:** `python scripts/evaluate.py --direction 05
   --test-matrix`.
6. **Fill the paper**, selecting the branch reality picked.

## 8. How to read the outcome

- **LoRA ≈ full FT, forgets less** → the LLM-world playbook works for separators;
  StemCraft could ship per-library "adapters" measured in single-digit megabytes.
- **LoRA ≈ full FT, forgets the same** → still a storage/compute win; the
  forgetting-protection folklore gets a data point against it.
- **LoRA clearly below 90%** → separation adaptation isn't low-rank at these ranks —
  a genuinely useful negative for the PEFT literature.
- **Full FT itself gains ~nothing** → the honest product answer: `umxhq` is already
  robust to these shifts; adaptation isn't the bottleneck. The precondition rule turns
  this from an embarrassment into the finding.
- **The LSTM part won't train** → the engineering result: recurrent LoRA is the hard
  bit; fc-only LoRA becomes the recommendation.

## 9. New glossary entries

**Fine-tuning** — continuing training of a pretrained model on new data. **PEFT /
LoRA** — freeze the model, learn a tiny low-rank correction; "rank" is the dial for
how expressive the correction is. **Catastrophic forgetting** — getting worse on old
data after adapting to new data. **Zero-shot** — using the pretrained model as-is.
**Head** — the model's final output layer(s). **Domain shift** — your data differing
systematically from the training data. **Merge-back** — folding the LoRA diff into
the weights so inference is unchanged.
