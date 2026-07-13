# Parallel WaveGAN & the multi-resolution STFT loss

Ryuichi Yamamoto, Eunwoo Song, Jae-Min Kim. ICASSP 2020 / arXiv **1910.11480** (25 Oct 2019). Implementation reference: `csteinmetz1/auraloss` (`auraloss/freq.py`). | Verified: HF paper_search (title/authors/date/abstract) + WebFetch auraloss source, 2026-07-13.

The origin of the **multi-resolution STFT (MR-STFT) loss** — Direction 01's fifth loss and the auxiliary term hypothesized to cut audible artifacts.

## 1. Problem & context
Parallel WaveGAN is a non-autoregressive GAN vocoder (1.44 M params, 24 kHz, 28× real-time). Its non-adversarial supervision is a **multi-resolution STFT loss** that matches the generated waveform's time-frequency structure at several analysis resolutions simultaneously — capturing both fine spectral detail (small FFT) and coarse structure (large FFT). MSS borrows it when training *through* the waveform, where a single-resolution loss over-commits to one time-frequency trade-off.

## 2. Method — the math (exact, from `auraloss/freq.py`)
For one STFT resolution with magnitude spectrograms $|S|$ (target) and $|\hat S|$ (estimate), define two terms:

**Spectral convergence** (verified: `SpectralConvergenceLoss`):
$$\mathcal L_{\text{sc}}(|S|,|\hat S|) = \frac{\big\lVert\,|S|-|\hat S|\,\big\rVert_F}{\big\lVert\,|S|\,\big\rVert_F}$$
(Frobenius norm of the magnitude difference, normalized by the target magnitude norm — emphasizes large spectral peaks, scale-robust).

**Log-magnitude STFT** (verified: `STFTMagnitudeLoss`, defaults `log=True, distance="L1"`):
$$\mathcal L_{\text{mag}}(|S|,|\hat S|) = \big\lVert \log(|S|+\epsilon) - \log(|\hat S|+\epsilon)\big\rVert_1$$
(L1 on log-compressed magnitudes — emphasizes low-energy detail; $\epsilon$=`log_eps`).

A single-resolution STFT loss is their sum (auraloss `STFTLoss` defaults `fft_size=1024, hop=256, win=1024`, Hann):
$$\mathcal L_{\text{STFT}} = \mathcal L_{\text{sc}} + \mathcal L_{\text{mag}}.$$

**Multi-resolution** (verified: `MultiResolutionSTFTLoss`) averages $\mathcal L_{\text{STFT}}$ over $M$ resolutions:
$$\boxed{\;\mathcal L_{\text{MR-STFT}} = \frac{1}{M}\sum_{m=1}^{M}\Big(\mathcal L_{\text{sc}}^{(m)} + \mathcal L_{\text{mag}}^{(m)}\Big)\;}$$
auraloss/PWG defaults (verified): **`fft_sizes=[1024, 2048, 512]`, `hop_sizes=[120, 240, 50]`, `win_lengths=[600, 1200, 240]`** ($M=3$). PWG additionally adds an adversarial term; **for MSS we use only the MR-STFT term** (no GAN).

**As used in Direction 01:** the fifth loss is $\mathcal L = \mathcal L_{1,\text{mag}} + \lambda\,\mathcal L_{\text{MR-STFT}}$ (L1-magnitude backbone + MR-STFT auxiliary), which requires a differentiable STFT/iSTFT to reach the waveform for $\mathcal L_{\text{sc}}$/$\mathcal L_{\text{mag}}$ at multiple resolutions.

## 3. Key results
- PWG: **4.16 MOS** within a Transformer-TTS pipeline, comparable to distillation-based Parallel WaveNet; 1.44 M params. (Vocoder result — not an MSS number; cited for the *loss*, not the model.) CONFIRMED (abstract).
- MR-STFT has since become the standard auxiliary/artifact-reducing loss across vocoders and waveform-domain separators.

## 4. Limitations & caveats
- MR-STFT is a *magnitude* loss at multiple resolutions — it does not directly model phase; its benefit for separation is empirically artifact reduction, not necessarily SDR gain (consistent with Bake-Off: SDR is already the best vocal proxy, so expect MR-STFT to help *perceptually at equal SDR*).
- Sensitive to the resolution set and $\lambda$; the PWG defaults are a reasonable, verified starting point.

## 5. Relevance to this project (Direction 01)
- Provides the **exact, code-grounded definition** of Direction 01's fifth loss (`L1 + multi-res STFT`).
- The MR-STFT term is the concrete instantiation of Direction 01's sub-hypothesis: "a multi-resolution STFT auxiliary reduces audible artifacts" — measured by the 5-clip listening check, with SI-SDR/museval reported alongside to show it is *not* bought at an SDR cost.
- `auraloss` is a maintained, MIT-licensed dependency, so this loss ships tested (round-trip + gradient checks in the `singnet/` package).

## 6. Verification notes
- PWG title/authors/venue/arXiv ID + MOS/param claims: CONFIRMED (HF paper_search abstract).
- MR-STFT formulae + default resolutions: **CONFIRMED from `auraloss/freq.py` source** (spectral convergence = Frobenius ratio; log-mag L1; `fft_sizes=[1024,2048,512]`, `hop_sizes=[120,240,50]`, `win_lengths=[600,1200,240]`). These auraloss defaults follow the PWG paper's resolution set.
</content>
