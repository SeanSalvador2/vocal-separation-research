# Vocal Separation Research — "SingNet"

A portfolio research project: seven independent, from-scratch research directions on
music/vocal source separation (karaoke-style vocal isolation), run through the full
data-science lifecycle with the rigor of a real research paper — pre-registered
hypotheses, a frozen evaluation protocol, reproducible code, and honest reporting of
whatever the results turn out to be.

**Status: all seven directions fully scaffolded and pre-registered — nothing trained
yet.** No datasets are downloaded and no models have been run; every experiment is
specified, coded, and unit-tested (442 CPU tests, no GPU required) and waits only on
compute. See [`PLAN.md`](PLAN.md) for the umbrella plan and each direction's
`MASTER_PLAN.md` for its frozen experiment spec and run book.

## Start here

- [`PLAN.md`](PLAN.md) — the optimized (v2) project plan: problem framing, hypotheses,
  evaluation protocol, reproducibility contract, and compute plan.
- [`RESEARCH_DIRECTIONS.md`](RESEARCH_DIRECTIONS.md) — literature review and the menu of
  candidate research directions, each cited against real published work.
- [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md) — deeper citation grounding for the
  architectures, losses, datasets, and feasibility claims used throughout.

## The seven research directions (all scaffolded)

Every direction folder contains the full artifact set: a frozen, self-contained
`MASTER_PLAN.md` (pre-registered hypotheses with numeric decision rules + a step-by-step
run book), `THEORY.md` + a compilable `theory/theory.tex`, lifecycle Jupyter notebooks
(scaffolded, deliberately **not executed** — every compute cell carries a "RUN THIS
LATER" banner with a runtime estimate), configs for every run, a `paper/PAPER.md`
scaffold with pre-written interpretations for **every** possible outcome, and a
plain-language `SEAN-README.md`.

1. [`01-loss-function-study/`](01-loss-function-study/SEAN-README.md) — "train on what you test?" — five losses, one U-Net, controlled
2. [`02-augmentation-data-scaling/`](02-augmentation-data-scaling/SEAN-README.md) — what each augmentation is worth + the data-scaling curve
3. [`03-mini-band-split/`](03-mini-band-split/SEAN-README.md) — the SOTA family's band-split idea, isolated at ~10 M params (param-matched to +0.06 %)
5. [`05-lora-source-separation/`](05-lora-source-separation/SEAN-README.md) — first careful LoRA-for-separation study (verified literature gap)
6. [`06-robust-training/`](06-robust-training/SEAN-README.md) — stem-bleed dose–response vs a closed-form null model + a trimmed-loss defense
8. [`08-silence-leakage/`](08-silence-leakage/SEAN-README.md) — the SLR "ghost vocals" metric + a chunk-sampling tradeoff study
10. [`10-demucs-distillation/`](10-demucs-distillation/SEAN-README.md) — teacher pseudo-labels on license-audited free audio; the program finale

Shared machinery lives in the tested [`singnet/`](singnet/) package (data pipeline,
augmentation switchboard, models, losses, metrics incl. SLR, training/eval, analysis)
with `tests/` runnable on CPU (`pip install -r requirements.txt && python -m pytest`).
One three-seed baseline cell is bit-identical across six directions (config-hash
`a97d5400e994`, test-asserted), saving ~15 redundant GPU-hours. Total pre-registered
GPU budget across all directions: ≈ 95–140 T4-hours on Colab Pro.

## Research library (Phase 0)

Verified source notes grounding every direction. Each paper deep-dive lives once (shared if
multiple directions need it) and is cross-linked; every citation is logged claim-by-claim.

- [`00-shared-research/`](00-shared-research/README.md) — shared library: dataset + metrics +
  Spleeter/UMX/Demucs/cIRM deep-dives, plus the
  [`VERIFICATION_LOG.md`](00-shared-research/VERIFICATION_LOG.md) (routes, verdicts, corrections,
  gap-checks; all verified 2026-07-13).

Per-direction literature guides (`research/LITERATURE.md` + `research/papers/`):

- 01 — [`01-loss-function-study/research/LITERATURE.md`](01-loss-function-study/research/LITERATURE.md)
- 02 — [`02-augmentation-data-scaling/research/LITERATURE.md`](02-augmentation-data-scaling/research/LITERATURE.md)
- 03 — [`03-mini-band-split/research/LITERATURE.md`](03-mini-band-split/research/LITERATURE.md)
- 05 — [`05-lora-source-separation/research/LITERATURE.md`](05-lora-source-separation/research/LITERATURE.md)
- 06 — [`06-robust-training/research/LITERATURE.md`](06-robust-training/research/LITERATURE.md)
- 08 — [`08-silence-leakage/research/LITERATURE.md`](08-silence-leakage/research/LITERATURE.md)
- 10 — [`10-demucs-distillation/research/LITERATURE.md`](10-demucs-distillation/research/LITERATURE.md)

Metric hygiene throughout: museval BSS-Eval SDR ≠ MDX cSDR/uSDR ≠ StemCraft window SI-SDR
(`RESEARCH_NOTES.md §0`). `[UNVERIFIED]` marks anything not groundable via a working route.

## Related project

The trained model(s) from this project are intended to ship into
[StemCraft](https://github.com/SeanSalvador2/stemcraft) as a new `SeparationEngine`,
alongside its existing classical-DSP and Demucs engines.

## Commit policy

No AI-attribution trailers or co-author lines in commits (e.g. no `Co-Authored-By: Claude`
or similar) — commit messages should read as the author's own.
