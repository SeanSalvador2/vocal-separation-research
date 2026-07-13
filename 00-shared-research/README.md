# 00 — Shared research library

Cross-cutting source notes every direction depends on. A paper's deep-dive lives **once**:
here if ≥2 directions need it, otherwise in the owning direction's `research/papers/`. Each
direction's `research/LITERATURE.md` links back here and adds its own "what WE take from it."

## Contents

| File | Source(s) | Role | Directions that rely on it |
|---|---|---|---|
| [`papers/musdb18-dataset.md`](papers/musdb18-dataset.md) | MUSDB18 (Zenodo 1117372/3338373), SiSEC | dataset + frozen 86/14/50 protocol + 14-track valid list | **all** |
| [`papers/leroux2019-si-sdr.md`](papers/leroux2019-si-sdr.md) | Le Roux 1811.02508 | primary metric (SI-SDR) + silence pathologies | all; **01, 08** |
| [`papers/bsseval-museval-sisec2018.md`](papers/bsseval-museval-sisec2018.md) | Vincent 2006 + SiSEC 1804.06267 + museval code | literature metric + **silent-frame NaN behavior** | all; **08** |
| [`papers/bakeoff2025-metrics-perception.md`](papers/bakeoff2025-metrics-perception.md) | Jaffe & Burgoyne 2507.06917 | metric↔perception (SDR best for vocals) | 01; metric policy |
| [`papers/spleeter2020.md`](papers/spleeter2020.md) | Spleeter JOSS 02154 | from-scratch U-Net + L1 template | 01, 02 |
| [`papers/openunmix2019.md`](papers/openunmix2019.md) | Open-Unmix JOSS 01667 + code | **exact BiLSTM shapes + augmentations** | **01, 02, 05** |
| [`papers/demucs2019-v1.md`](papers/demucs2019-v1.md) | Demucs v1 1911.13254 + `augment.py` | aug recipe + L1-loss provenance | **01, 02, 10** |
| [`papers/demucs-hybrid-family.md`](papers/demucs-hybrid-family.md) | Hybrid 2111.03600 + HT 2211.08553 | field ceiling + **distillation teacher** | 03 (ceiling), **10** |
| [`papers/kong2021-cirm-resunet.md`](papers/kong2021-cirm-resunet.md) | Kong 2109.05418 | magnitude-mask ceiling, cIRM math | 03; THEORY §2; H4 |

- [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) — full claim-by-claim log (all routes, verdicts, corrections, gap-checks).

## The seven directions

| Dir | Folder | One-line question |
|---|---|---|
| 01 | [`../01-loss-function-study`](../01-loss-function-study/research/LITERATURE.md) | Does training on SI-SDR beat L1-magnitude at small scale? |
| 02 | [`../02-augmentation-data-scaling`](../02-augmentation-data-scaling/research/LITERATURE.md) | How much of the aug gain is remixing, and how data-starved is 86 songs? |
| 03 | [`../03-mini-band-split`](../03-mini-band-split/research/LITERATURE.md) | Does band-splitting help at 5–10 M params, param-matched? |
| 05 | [`../05-lora-source-separation`](../05-lora-source-separation/research/LITERATURE.md) | Can LoRA recover full-fine-tune quality on UMX at <5% params? |
| 06 | [`../06-robust-training`](../06-robust-training/research/LITERATURE.md) | How much do label-noise/bleed hurt, and does a cheap defense recover it? |
| 08 | [`../08-silence-leakage`](../08-silence-leakage/research/LITERATURE.md) | Does activity-aware sampling trade off overall SDR vs silence leakage? |
| 10 | [`../10-demucs-distillation`](../10-demucs-distillation/research/LITERATURE.md) | Can Demucs-as-teacher distillation on unlabeled FMA close the gap? |

## How to read a deep-dive
Every `papers/*.md` follows: Problem & context → Method (with math) → Key results (exact
numbers + metric convention) → Limitations → Relevance to this project → Verification notes.
**Metric hygiene throughout:** museval BSS-Eval SDR ≠ MDX cSDR/uSDR ≠ StemCraft window SI-SDR
(RESEARCH_NOTES §0). Every number carries its provenance; `[UNVERIFIED]` marks anything not
groundable via a working route.
