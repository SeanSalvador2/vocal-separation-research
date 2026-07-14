# Direction 10, explained — let the big model teach the small one

*Plain-language guide to the program's finale. Same machinery as before — see
[`../01-loss-function-study/SEAN-README.md`](../01-loss-function-study/SEAN-README.md)
first.*

---

## 1. The question in one paragraph

Our little from-scratch model tops out because it only has 86 labeled training songs.
Meanwhile, a big excellent separator (HT-Demucs — the one StemCraft already ships) is
sitting right there. The oldest trick in industrial ML: **have the big model label a
pile of unlabeled music, then train the small model on those "pseudo-labels."** The
teacher is imperfect, the free music is stylistically different from our benchmark, and
nobody has published whether this actually works at hobbyist scale for source
separation. Our pre-registered bet (**H-10**): mixing teacher-labeled data into
training closes **at least 25%** of the small model's gap to the teacher, measured on
the untouched MUSDB test set.

## 2. Where the free music comes from (and the licensing homework)

We use **FMA** — 8,000 free 30-second clips under Creative Commons licenses. Catch:
each artist chose their own license variant, and some variants (**"-ND", no
derivatives**) forbid exactly what a separator produces — a derivative work. So the
pipeline starts with a **license audit**: an allowlist filter that only accepts the
permissive CC variants, hard-rejects anything with ND, and rejects-by-default anything
unclear. The resulting track-ID + license manifest gets committed so anyone can audit
it; the audio itself never enters the repo. I probed the filter live during review:
`CC BY 4.0` → accepted, `CC BY-ND 2.0` → rejected, `All Rights Reserved` → rejected,
URL-form licenses parsed correctly, garbage rejected.

From the ~⟨filtered⟩ pool we take 1,200 clips (fixed seed), run the teacher over them
**once** (2–4 GPU-hours, with the exact demucs version recorded), keep the ~800 with
real vocal activity (reusing Direction 08's activity machinery on the teacher's own
output), and that's the pseudo-labeled dataset: ~6.7 hours, roughly the size of MUSDB's
train split, built from nothing but free audio and a weekend GPU.

## 3. The experiment (three data diets, everything else frozen)

1. **MUSDB-only** — the baseline. (Literally the same three runs every other direction
   shares — the config hash proves it. Sixth direction served by one 3-seed cell.)
2. **Mixed** — each training example flips a fair coin: real MUSDB chunk or
   pseudo-labeled FMA chunk. Frankenstein-remixing stays *within* each pool, so real
   stems never blend with teacher outputs (guaranteed by construction, not by hope).
3. **Distilled-only** — can 800 free pseudo-labeled clips replace 86 real songs
   entirely? (One seed; a provocative side-question.)

Plus two single-seed probes: a 75/25 mixing ratio, and a **trimmed-loss** arm that
reuses Direction 06's defense — because a teacher's mistakes are exactly "corrupted
training targets," which Direction 06 just spent a whole study pricing.

**Anti-leakage guarantees:** the MUSDB test set is never teacher-labeled, validation
stays pure MUSDB, and the pseudo-data loader *structurally refuses* to open anything
under a MUSDB path (I triggered that guard live in review — it raises before the
object can even be built).

## 4. How we score it

On the untouched 50-song test set, in one final session: all student variants, **the
teacher itself** (that's the ceiling that defines "the gap"), the do-nothing floor, and
the oracle. The headline number is **gap closure**: (mixed − baseline) / (teacher −
baseline). We also run Direction 08's **SLR** on everything — if the teacher's habit of
leaking sound into silent passages gets *inherited* by the student, that shows up here
and nowhere else.

This finale also ties the program together: Direction 02's scaling curve says whether
more data was ever the bottleneck; Direction 06's chart prices corrupted targets;
this study is where both cash out. The paper has pre-written branches for every
combination — including the honest nulls ("free shifted audio adds nothing") and the
cautionary tale ("teacher errors poison the student").

## 5. What's in this folder

| Path | What it is |
|---|---|
| [`MASTER_PLAN.md`](MASTER_PLAN.md) | Frozen spec: pipeline, license allowlist, arms, decision rules, guards, ≈12–17 T4-h budget (only 6 new training runs + one teacher pass). |
| [`THEORY.md`](THEORY.md) / [`theory/`](theory/) | The math: why classification-style distillation ("temperature") has no analogue here and what survives; teacher error as *structured* corruption (the formal bridge to Direction 06); the gap-closure statistic and its confidence interval; the ND-exclusion licensing argument. |
| [`research/`](research/) | Verified literature: the semi-supervised precedents, the teacher family, FMA's licensing fine print. |
| [`notebooks/01_teacher_and_data.ipynb`](notebooks/) | The pipeline end-to-end: license filter demo, teacher labeling cells (RUN LATER), what pseudo-stems look like next to real ones. |
| [`notebooks/02_distillation_experiments.ipynb`](notebooks/) | The experiment: launch cells, the gap-closure ladder, the SLR audit, the program-level synthesis cell, pre-written interpretations. |
| [`configs/`](configs/) | 8 configs (3 mixed seeds, distilled-only, ratio probe, trim probe, contingencies). |
| `../singnet/data/pseudo.py`, `../scripts/prepare_fma.py`, `../scripts/teacher_label.py` | The new machinery — license audit, teacher tooling, pool mixing — all fixture-tested (suite now **442 green tests**). |
| [`paper/PAPER.md`](paper/PAPER.md) | Report scaffold, every branch pre-written. |

## 6. What you'll run later (~12–17 T4-hours)

1. **FMA metadata + license manifest (CPU/network):** `python scripts/prepare_fma.py
   --metadata-dir ... --screen 1200 --seed 0 --out $PSEUDO_ROOT` — commit the manifest
   it writes.
2. **Teacher labeling (GPU, 2–4h, once):** `python scripts/teacher_label.py --manifest
   ... --keep 800 --activity-threshold 0.20 --out $PSEUDO_ROOT`.
3. **Dry run, then the 6 runs:** `python scripts/run_sweep.py --direction 10
   --dry-run` → `--stage reduced`.
4. **The final test session:** `python scripts/evaluate.py --direction 10
   --test-session --include-teacher --slr`.
5. **Fill the paper** — and its synthesis section, which pulls in Directions 02 and 06's
   verdicts.

## 7. How to read the outcome

- **≥25% of the gap closed** → the playbook works at your scale: more free audio =
  more quality; scaling the pseudo-pool becomes the obvious next move.
- **Real but smaller gain** → helpful, domain-shift-limited; the ratio and trim probes
  point at which fix matters.
- **Nothing** → composed with Direction 02: either the model (not data) was the wall,
  or remixed real stems already provide what shifted pseudo-audio can't. Either way
  the program answers the triage question with two independent studies.
- **The student gets worse** → teacher mistakes are contagious — the cautionary result
  the distillation hype never mentions, with Direction 06's chart to quantify it and
  the trim arm as the tested first aid.

## 8. New glossary entries

**Pseudo-labels / distillation / self-training** — training a model on another model's
outputs as if they were ground truth. **Teacher / student** — the big labeling model /
the small learning model. **Gap closure** — how much of the student-to-teacher distance
the trick recovered, as a fraction. **Derivative work** — a legal category that
separated stems fall into; why "-ND" licenses are excluded. **Provenance file** — the
recorded versions/settings that make a labeling run reproducible. **Structured
corruption** — errors that correlate with content (a teacher's), unlike the uniform
bleed of Direction 06.
