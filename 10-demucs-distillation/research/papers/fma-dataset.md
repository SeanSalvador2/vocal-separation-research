# FMA: A Dataset For Music Analysis — the license-safe unlabeled data

Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst, Xavier Bresson. ISMIR 2017 / arXiv **1612.01840**. Repo: `mdeff/fma`. | Verified: WebSearch (arXiv abs + subsets) + WebFetch `mdeff/fma` README (licensing verbatim), 2026-07-13. **CORE for Direction 10** — its "license-safe unlabeled data" claim rests entirely on the licensing details below.

## 1. Problem & context
Direction 10 needs **~100–200 unlabeled tracks** whose audio can be legally teacher-labeled and used without redistributing copyrighted material — MUSDB is research-only and *labeled*; a large public CC corpus is required. FMA is that corpus, but the licensing has nuance that matters.

## 2. What FMA is
- **106,574 tracks** from 16,341 artists / 14,854 albums, 161-genre hierarchy, 917 GiB / 343 days of audio, with track/album/artist metadata, tags, and biographies.
- **Four subsets** (verified from README):
  | Subset | Tracks | Clip | Genres | Size |
  |---|---|---|---|---|
  | `fma_small` | 8,000 | 30 s | 8 balanced | 7.2 GiB |
  | `fma_medium` | 25,000 | 30 s | 16 unbalanced | 22 GiB |
  | `fma_large` | 106,574 | 30 s | 161 unbalanced | 93 GiB |
  | `fma_full` | 106,574 | untrimmed | 161 unbalanced | 879 GiB |

For Direction 10, a **~100–200 track curated subset** (from `fma_small`) is ample and cheap to teacher-label.

## 3. Licensing — the load-bearing detail (verified verbatim from `mdeff/fma` README)
- **Metadata:** "released under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**."
- **Audio:** "**We do not hold the copyright on the audio and distribute it under the license chosen by the artist.**" → i.e. **audio is under *per-track, artist-chosen* Creative Commons licenses** (CC-BY, CC-BY-NC, CC-BY-SA, CC0, …) — **not a single uniform license.**
- **Use framing:** "**The dataset is meant for research purposes.**" No explicit blanket commercial-use grant.
- **Per-track license is recorded in the metadata** (`tracks.csv` carries a license field), so a maximally-safe subset (e.g. CC0 / CC-BY only) can be **filtered programmatically**.

**Implication for "license-safe" (state precisely in the report):** for a **non-commercial research/portfolio** project, FMA is license-safe — and safer still if the ~100–200 track subset is filtered to permissive licenses via the metadata. The phrase "FMA CC-licensed subset" (RESEARCH_DIRECTIONS #10) is **correct but must carry the nuance** that licenses vary per track and that only *audio-license-permitting* tracks should be used if any output is redistributed. No FMA audio is committed (same policy as MUSDB); only teacher-generated soft targets / trained weights are kept.

## 4. Limitations & caveats
- **Domain mismatch:** FMA skews toward independent/CC artists and genres that may differ from MUSDB's production style — the teacher's FMA outputs may not transfer cleanly to MUSDB test (Direction 10's central risk).
- **Quality variance:** FMA audio quality/loudness varies; teacher-labeling quality will vary with it.
- **30 s clips** (small/medium) are fine for chunked training but truncate song structure.

## 5. Relevance to this project (Direction 10)
- **Supplies the unlabeled data** the whole direction depends on; the licensing nuance above is what makes the "license-safe" claim honest.
- Same corpus that **MixIT-for-MSS (Saijo & Bando, 2505.07631)** pre-trains on — cross-validates FMA as the standard unlabeled-music source for MSS (`../../02-augmentation-data-scaling/research/papers/saijo2025-mixit-mss.md`).
- Practical recipe: filter `fma_small` metadata → permissive-license subset → teacher-label with `htdemucs` once → cache soft targets.

## 6. Verification notes
- arXiv ID + authors + 106,574 tracks + 4 subset sizes: CONFIRMED (WebSearch + README).
- **Licensing (metadata CC BY 4.0; audio per-artist-chosen license; "research purposes"): CONFIRMED verbatim from `mdeff/fma` README.** The per-track-license-varies nuance is the key correction to any "uniformly CC-BY" reading.
- Per-track license field in `tracks.csv`: `[README states audio license is artist-chosen and metadata includes license info; exact column name from repo docs]`.
</content>
