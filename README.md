# Vocal Separation Research — "SingNet"

A portfolio research project: seven independent, from-scratch research directions on
music/vocal source separation (karaoke-style vocal isolation), run through the full
data-science lifecycle with the rigor of a real research paper — pre-registered
hypotheses, a frozen evaluation protocol, reproducible code, and honest reporting of
whatever the results turn out to be.

**Status: planning complete, nothing trained yet.** No datasets are downloaded and no
models have been run. See [`PLAN.md`](PLAN.md) for the full project plan.

## Start here

- [`PLAN.md`](PLAN.md) — the optimized (v2) project plan: problem framing, hypotheses,
  evaluation protocol, reproducibility contract, and compute plan.
- [`RESEARCH_DIRECTIONS.md`](RESEARCH_DIRECTIONS.md) — literature review and the menu of
  candidate research directions, each cited against real published work.
- [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md) — deeper citation grounding for the
  architectures, losses, datasets, and feasibility claims used throughout.

## Approved research directions

Each will live in its own top-level folder once scaffolded:

1. `01-loss-function-study/` — SI-SDR vs. L1 loss for a compact separation U-Net
2. `02-augmentation-data-scaling/` — augmentation factorization + data-scaling curves
3. `03-mini-band-split/` — a mini band-split front-end, param-matched against a baseline
5. `05-lora-source-separation/` — parameter-efficient (LoRA) fine-tuning for source separation
6. `06-robust-training/` — robustness under stem-bleed / label noise + a cheap mitigation
8. `08-silence-leakage/` — the "silence problem": chunk-sampling ablation + a leakage metric
10. `10-demucs-distillation/` — Demucs-as-teacher distillation on pseudo-labeled data

Each direction folder will contain: a `MASTER_PLAN.md`, `THEORY.md` (+ LaTeX), one or more
Jupyter notebooks covering the full DS lifecycle (scaffolded, not yet executed), a
reproducible `singnet/`-style code package, a draft paper/report with placeholders for
results, and a `SEAN-README.md` explaining the direction in plain terms.

## Related project

The trained model(s) from this project are intended to ship into
[StemCraft](https://github.com/SeanSalvador2/stemcraft) as a new `SeparationEngine`,
alongside its existing classical-DSP and Demucs engines.

## Commit policy

No AI-attribution trailers or co-author lines in commits (e.g. no `Co-Authored-By: Claude`
or similar) — commit messages should read as the author's own.
