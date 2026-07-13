# Open-Unmix (UMX) — a reference implementation for music source separation

Stöter, Uhlich, Liutkus, Mitsufuji. JOSS 4(41):1667, 2019. DOI **10.21105/joss.01667**. Repo: `sigsep/open-unmix-pytorch`. | Verified: WebFetch repo README + `openunmix/model.py` + `openunmix/data.py`, 2026-07-13. **This is the fine-tuning target for Track B and the LoRA host for Direction 05 — the code-level shapes below are load-bearing.**

## 1. Problem & context
UMX is the canonical *reproducible-on-MUSDB* baseline: MIT-licensed, trained on MUSDB18(-HQ) only, with published weights (`umx`, `umxhq`, `umxl`). It is the model PLAN Track B fine-tunes and Direction 05 wraps with LoRA. Its defining choice: a magnitude-spectrogram **BiLSTM** (not a conv U-Net), one model per target.

## 2. Method — exact architecture (from `model.py`, verified)
Per target, `OpenUnmix` is a per-frame MLP wrapped around a 3-layer BiLSTM. Defaults: `hidden_size = 512`, `nb_layers = 3`, `nb_bins` = number of (bandwidth-cropped) input frequency bins, `nb_channels = 2`. STFT front-end: `n_fft = 4096`, `hop = 1024`, Hann; magnitudes cropped to a `max_bin` bandwidth (UMX's convention PLAN Phase 2 adopts).

Forward pass (shapes per time frame, batch/time folded):
1. **Input crop + standardize:** magnitude cropped to `nb_bins`; subtract learnable `input_mean`, multiply `input_scale`.
2. **`fc1`:** `Linear(nb_bins*nb_channels → hidden_size=512, bias=False)` → **`bn1`** `BatchNorm1d(512)` → **`tanh`**. Compresses the (2·`nb_bins`)-dim spectral frame to 512.
3. **`lstm`:** `LSTM(input_size=512, hidden_size=hidden_size//2=256, num_layers=3, bidirectional=True, dropout=0.4)`. Bidirectional ⇒ output dim $256\times2 = 512$.
4. **Skip connection:** concatenate `fc1`-output (512) with `lstm`-output (512) → **1024**.
5. **`fc2`:** `Linear(1024 → 512, bias=False)` → **`bn2`** `BatchNorm1d(512)` → **`relu`**.
6. **`fc3`:** `Linear(512 → nb_output_bins*nb_channels, bias=False)` → **`bn3`** `BatchNorm1d(nb_output_bins*nb_channels)`.
7. **Output:** multiply learnable `output_scale`, add `output_mean`, **`relu`** → a **non-negative magnitude estimate** applied as a (possibly >1) mask on the input magnitude.

**Inference-time refinement:** the `Separator` wrapper runs a **multichannel generalized Wiener filter** (`norbert`/`wiener`) over the per-target magnitude estimates before iSTFT — worth ~0.3–0.5 dB and the reason UMX outputs are multichannel-consistent. Loss is **MSE on magnitude** (vs Spleeter's L1) — a data point for Direction 01.

## 3. Key results (repo README, verified)
Vocals, **museval SDR** (median-of-frames/median-of-tracks):
- **`umx` (MUSDB18, compressed): 6.32 dB vocals.**
- **`umxhq` (MUSDB18-HQ): 6.25 dB vocals.**
- `umxl` (private stems, larger): higher, weights CC-BY-NC-SA (not usable as an MIT baseline).
License: **MIT** (`umx`/`umxhq` code+weights). These are the exact anchors in RESEARCH_NOTES; CONFIRMED verbatim from the README table.

## 4. Augmentations (from `data.py`, verified) — code provenance for Direction 02
- `_augment_gain`: multiply a source by a random **gain ~ U(0.25, 1.25)** (≈ −12 dB … +1.9 dB amplitude).
- `_augment_channelswap`: **swap stereo channels with p = 0.5**.
- `_augment_force_stereo`: mono→stereo duplication / truncate to 2ch.
- **Random source remixing** ("random track mix"): `MUSDBDataset(random_track_mix=True)` draws each source via `random.choice(mus.tracks)` — i.e. `vocals` from track $i$, interferers from other tracks — assembling combinatorially many mixtures. Plus random fixed-length **chunk** cropping (`chunk_start = random.uniform(...)`).
- **Note:** UMX does **not** do sign-flip (that's Demucs) and its gain range is 0.25–1.25×, *not* the "±3–6 dB" loosely stated in RESEARCH_NOTES §3 — the exact code range is used in Direction 02.

## 5. Relevance to this project
- **Track B / H3:** `umxhq` is the pretrained model fine-tuned in Phase 6 (full vs head-only vs LR-warmup).
- **Direction 05 (CORE):** LoRA wraps UMX's linear/recurrent projections. Wrappable matrices and their shapes:
  - `fc1`: $512\times(2\cdot\texttt{nb\_bins})$ — the largest single dense matrix (≈1.5 M params for `nb_bins`≈1487).
  - `lstm` layers ×3: each has `weight_ih` ($4\cdot256\times \text{in}$) and `weight_hh` ($4\cdot256\times256$) **per direction** — the recurrent bulk; LoRA on an LSTM must wrap `weight_ih`/`weight_hh` (input and recurrent projections), the main plumbing risk.
  - `fc2`: $512\times1024$; `fc3`: $(2\cdot\texttt{nb\_output\_bins})\times512$.
  - Rank-$r$ accounting per wrapped $d_{out}\times d_{in}$ matrix: $r(d_{out}+d_{in})$ trainable params (see Direction 05 LITERATURE.md for the rank-4/16 budget table).
- **Direction 01:** UMX's **MSE-on-magnitude** vs Spleeter's **L1-on-magnitude** is the historical L1-vs-MSE split the loss study formalizes.
- **Direction 02:** UMX's documented aug recipe (gain/channelswap/random-track-mix) is one of the three code-verified recipes factorized.
- **Direction 08:** UMX training draws chunks uniformly (no vocal-activity weighting) — the baseline sampling policy Direction 08 ablates against.

## 6. Verification notes
- README SDR (6.32/6.25), MIT, MUSDB training, 3-layer BiLSTM, Wiener/norbert: CONFIRMED (README fetch).
- Exact layer shapes (fc1/bn1/tanh → LSTM 256×2×3 → skip 1024 → fc2/bn2/relu → fc3/bn3 → output): CONFIRMED from `model.py` (quoted defaults `hidden_size=512`, `hidden_size//2` when bidirectional).
- Augmentation functions and defaults (gain 0.25–1.25, channelswap p=0.5, random_track_mix): CONFIRMED from `data.py`.
- LoRA param-count formula $r(d_{out}+d_{in})$: standard (see `hu2021-lora.md`); the specific UMX matrix dims are the code defaults.
</content>
