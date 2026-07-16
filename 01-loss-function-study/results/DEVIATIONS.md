# Deviations log — Direction 01

Per MASTER_PLAN footer: *any deviation during execution must be recorded here
with date + reason.*

## Experimental / execution deviations

- **2026-07-16 — Data-prep verify criterion revised (MASTER_PLAN §4.3).** The plan
  specified "mixture ≈ sum of stems (max abs error < 1e-3 for MUSDB18's AAC)".
  The first real decode (Windows, full 150-track MUSDB18) measured
  **max |mixture − sum(stems)| = 3.0** on healthy shards. Root cause: the plan's
  tolerance assumed near-exact additivity, but MUSDB18's five streams are
  **AAC-encoded independently**, and at loud transients the encoded mixture
  stream is additionally peak-limited relative to the raw float stem sum —
  pointwise max-abs is the wrong lens for codec noise. Impact on the science:
  **none** — every training/eval path constructs mixtures as stem sums
  (`singnet/data/musdb_dataset.py`, `singnet/eval/evaluate.py`) and never reads
  the decoded mixture stream. Revision: `--verify` now hard-gates decode-bug
  signatures (per-track relative RMS error ≤ 5 %, mixture↔sum correlation
  ≥ 0.99, shape/rate/finiteness checks) and reports the codec-noise statistics
  (median/max relative RMS, worst pointwise offenders) as information. New unit
  tests prove the gate still catches misalignment, gain errors, and truncation
  (`tests/test_prepare_data_verify.py`). This is the first execution deviation
  and the plan's 1e-3 figure should be read as superseded by this entry.
  **Measured on the real decode (150 tracks, 2026-07-16):** relative RMS error
  median **0.0293**, max healthy **0.047**; min healthy correlation 0.9989. One
  track tripped the gates — **"PR - Oh No" (rel 0.416, corr 0.941)** — and is
  precisely the track the **official SigSep errata** documents as *"sum of
  sources does not add up to the mix for the left channel"* (sigsep website,
  datasets/musdb.md, fetched 2026-07-16). Resolution: a cited
  `KNOWN_DATASET_ERRATA` allowlist in `scripts/prepare_data.py` downgrades
  gate failures on documented-errata tracks to WARN; undocumented failures
  still hard-fail (both behaviors unit-tested). The track **stays in the frozen
  50-track test protocol**: its stems are internally consistent and every
  pipeline constructs mixtures as stem sums, so evaluation is unaffected. Note
  for Direction 06: the same errata table documents real bleed in several
  train-split tracks (e.g. Chris Durban - Celebrate, Hop Along - Sister
  Cities) — real-world grounding for the stem-bleed premise.

Nothing has been trained; the frozen plan (MASTER_PLAN, 2026-07-13) is otherwise
intact.

## Build-time implementation clarifications

Not deviations from any explicit spec value — these record choices made where the
plan left an implementation detail open, logged here for full transparency.

- **2026-07-13 — Waveform-loss target is the STFT-consistent reconstruction.**
  For the `sisdr`/`l1mrstft` arms the loss compares the estimate
  `v̂ = iSTFT(M′⊙X)` against `v = iSTFT(crop(STFT(vocals)))` rather than the raw
  6 s chunk. Reason: reconstruction happens in the cropped 256-frame domain
  (MASTER_PLAN §5), so estimate and target must share that exact analysis grid
  and length; the ≤3 dropped edge frames (~3 ms of 6 s) are immaterial and are
  identical across arms, preserving the controlled comparison.
  (`singnet/audio/stft.py`, `singnet/train/loop.py::prepare_batch`.)
- **2026-07-13 — Evaluation overlap-add chunk = 261120 samples (exactly 256
  frames).** The training chunk stays 6.0 s / 264600 samples (§4.4); the *inference*
  OLA uses a chunk whose STFT is exactly 256 frames so the fixed-size model needs
  no frame crop/pad on the eval path. The first/last chunk keep a flat (un-tapered)
  window edge at true signal boundaries so identity-mask reconstruction is exact
  (< −60 dB). (`singnet/eval/overlap_add.py`.)
- **2026-07-13 — MR-STFT aggregation is a sum over resolutions.** Followed
  MASTER_PLAN §6 (`λ Σ_m (L_sc+L_mag)`) verbatim; note auraloss averages instead
  — the two differ only by a constant folded into λ. (`singnet/losses/mrstft.py`.)
