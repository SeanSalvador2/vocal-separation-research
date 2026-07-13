# Deviations log — Direction 01

Per MASTER_PLAN footer: *any deviation during execution must be recorded here
with date + reason.*

## Experimental / execution deviations

**None.** Nothing has been trained; the frozen plan (MASTER_PLAN, 2026-07-13) is
intact. This section will accumulate entries only if execution departs from the
pre-registered design (e.g. the §3.2 budget-halving rule fires, or gate G2
triggers the extra-seed escalation).

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
