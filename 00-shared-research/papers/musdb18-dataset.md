# MUSDB18 / MUSDB18-HQ — the evaluation substrate

Dataset (not a single paper). Primary sources verified 2026-07-13:
- Zenodo record **1117372** ("The MUSDB18 corpus for music separation") — route: WebFetch zenodo.org.
- Zenodo record **3338373** ("MUSDB18-HQ - an uncompressed version of MUSDB18") — route: WebFetch zenodo.org + WebSearch.
- Introduced by SiSEC 2018 (Stöter, Liutkus, Ito, arXiv **1804.06267**) — see `bsseval-museval-sisec2018.md`.
- `musdb` python parser / split lists — route: WebFetch raw.githubusercontent.com/sigsep/sigsep-mus-db.

This file is the shared reference for **every** direction: all seven train and evaluate on MUSDB18(-HQ) and all inherit the frozen 86/14/50 protocol below.

## 1. What it is
- **150 full-length stereo tracks**, 44.1 kHz, split **100 train / 50 test** (official). Roughly 10 hours of audio.
- Each track ships **5 stereo streams**: the `mixture` plus 4 stems — `vocals`, `drums`, `bass`, and `other` ("rest of accompaniment"). The mixture is the sum of the 4 stems (linear, sample-accurate), which is what makes source-remixing augmentation exact (see `02-augmentation-data-scaling`).
- For a **2-stem karaoke** target (this project's primitive), `accompaniment = drums + bass + other`, recovered exactly by subtraction; `vocals` is the isolated stem. Ground truth is therefore exact, no estimation.

## 2. Two distributions
| | MUSDB18 (record 1117372) | MUSDB18-HQ (record 3338373) |
|---|---|---|
| Format | Native Instruments **STEMS** `.mp4`, **AAC @256 kbps** | Uncompressed **WAV** |
| Effective bandwidth | ~16 kHz (AAC lossy cap) | full 22.05 kHz |
| Size | **4.7 GB** (Zenodo) | **22.7 GB** (Zenodo) |
| Access | open download | **restricted — "request access", manually approved (usually < 1 day)** |
| License | educational/non-commercial (identical text below) | same |

License text (verbatim, Zenodo): *"MUSDB18 is provided for educational purposes only and the material contained in them should not be used for any commercial purpose without the express permission of the copyright holders."* Source composition: 100 tracks from the *Mixing Secrets* Free Multitrack Library, 46 from MedleyDB (CC BY-NC-SA 4.0), 2 Native Instruments, 2 The Easton Ellises (CC BY-NC-SA 3.0).

> **Correction logged:** RESEARCH_NOTES §3 says "~4.4 GB". Zenodo states **4.7 GB** (StemCraft's downloader measured 4.68 GB, consistent). The size is corrected in-place. The **access-restricted** status of MUSDB18-HQ is CONFIRMED (an intermediate WebFetch summary erroneously said "open access"; a second route — SigSep website / WebSearch — confirms request-access-with-manual-approval, matching RESEARCH_NOTES).

## 3. The frozen 86 / 14 / 50 protocol (load-bearing)
`musdb` exposes the 100 "train" tracks as `train` (86) + `valid` (14). The **14-track validation split** (`musdb.configs/mus.yaml`, verified verbatim from sigsep-mus-db source) is:

1. Actions - One Minute Smile
2. Clara Berry And Wooldog - Waltz For My Victims
3. Johnny Lokke - Promises & Lies
4. Patrick Talbot - A Reason To Leave
5. Triviul - Angelsaint
6. Alexander Ross - Goodbye Bolero
7. Fergessen - Nos Palpitants
8. Leaf - Summerghost
9. Skelpolu - Human Mistakes
10. Young Griffo - Pennies
11. ANiMAL - Rockshow
12. James May - On The Line
13. Meaxic - Take A Step
14. Traffic Experiment - Sirens

**This is the exact list PLAN §1.5.1 references** and the same split Open-Unmix trains against. Every direction tunes on these 14 tracks only; the 50 test tracks are touched exactly twice (Phase 7 + report figures). The split manifest (86/14/50) is committed as CSV before any training.

## 4. Oracle references available on MUSDB (used by H4)
SiSEC 2018 shipped reference implementations of three oracle separators over MUSDB: **ideal binary mask (IBM)**, **ideal ratio mask (IRM)**, and **multichannel Wiener filter (MWF)** (verified via 1804.06267). These give the honest upper bound for the mask family (PLAN H4 oracle study) and anchor Directions 03/08 (band-split, silence) and the phase discussion (`kong2021-cirm-resunet.md`).

## 5. What each direction takes from it
- **All**: the 86/14/50 protocol, exact stems, non-redistribution policy (no audio committed).
- **02 (aug/scaling)**: linear stem additivity → exact cross-song remixing; the 86-song ceiling defines the data-scaling x-axis {21,43,64,86}.
- **06 (robust)**: clean stems are the substrate we deliberately corrupt (ε-bleed / label noise) — see `../../06-robust-training`.
- **08 (silence)**: GT `vocals` stem gives exact silent-region masks for the leakage metric.
- **10 (distillation)**: MUSDB test is the held-out evaluation; FMA (`../../10-demucs-distillation/research/papers/fma-dataset.md`) supplies license-safe *unlabeled* extra data.

## 6. Verification notes
- Track counts, splits, formats, sizes, license text, HQ access-restriction: CONFIRMED via Zenodo 1117372/3338373 and WebSearch (SigSep site).
- 14-track valid list: CONFIRMED verbatim from sigsep-mus-db `mus.yaml`.
- AAC "~16 kHz effective bandwidth": property of 256 kbps AAC; consistent with SiSEC/community reports — [not independently re-measured here].
</content>
</invoke>
