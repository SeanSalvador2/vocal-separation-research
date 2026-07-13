# THEORY — Direction 01: the math behind the loss-function study

This note derives the mathematics the code implements. Every result is tied to a
concrete function path in `singnet/`, and the parameter count in §5 is the exact
number asserted by `tests/test_model.py`. It follows the MASTER_PLAN §13 outline.

Notation. A mono mixture waveform is $x\in\mathbb R^{L}$ (44.1 kHz). Its STFT is
$X=\mathrm{STFT}(x)\in\mathbb C^{F\times T}$ with magnitude $|X|$ and phase
$\angle X$. The target vocals are $v$ (STFT $S$), the accompaniment $a$ (STFT
$A$), with $x=v+a$ and, by linearity of the STFT, $X=S+A$. The network predicts a
soft mask $M\in[0,1]^{F\times T}$; the vocal estimate is
$\widehat V = M\odot|X|\,e^{i\angle X}$ and $\hat v=\mathrm{iSTFT}(\widehat V)$.

---

## 1. Signals, STFT/iSTFT, windowing, and the COLA condition

### 1.1 Definitions

With analysis window $w$ of length $N=n_\text{fft}$ and hop $H$,
$$
X[k,m]=\sum_{n=0}^{N-1} w[n]\,x[n+mH]\,e^{-i2\pi kn/N},
\qquad k=0,\dots,\tfrac N2,\; m=0,\dots,T-1 .
$$
The project pins $N=4096$, $H=1024=N/4$, a periodic **Hann** window, and
`center=True` (so frame $m$ is centred at sample $mH$ after reflect-padding).
A 6 s chunk ($L=264{,}600$) yields $F=N/2+1=2049$ frequency bins and $T=259$
frames.

Reconstruction is weighted overlap-add (WOLA):
$$
\hat x[n]=\frac{\sum_m w[n-mH]\,\big(\tfrac1N\sum_k X[k,m]e^{i2\pi k(n-mH)/N}\big)}
{\sum_m w^2[n-mH]} .
$$

> **Code.** `singnet/audio/stft.py::STFT.transform` / `STFT.inverse` wrap
> `torch.stft`/`torch.istft` with exactly these settings. The round-trip is
> unit-tested at $<-60$ dB (`tests/test_stft.py`; measured $\approx-137$ dB).

### 1.2 Perfect reconstruction (NOLA) for Hann at hop $N/4$

WOLA inverts the STFT exactly iff the squared window tiles to a positive constant,
the **NOLA** condition $\sum_m w^2[n-mH]=c>0\ \forall n$. For the Hann window
$w[n]=\tfrac12\big(1-\cos\tfrac{2\pi n}{N}\big)$,
$$
w^2[n]=\tfrac38-\tfrac12\cos\tfrac{2\pi n}{N}+\tfrac18\cos\tfrac{4\pi n}{N}.
$$
At hop $H=N/4$ the shifts $n,\,n-\tfrac N4,\,n-\tfrac N2,\,n-\tfrac{3N}4$ sample
each cosine at four points spaced by a quarter-period (for $\cos\tfrac{2\pi}{N}$)
or half-period (for $\cos\tfrac{4\pi}{N}$); both sets sum to zero. Hence
$$
\sum_{m}w^2[n-mH]=4\cdot\tfrac38=\tfrac32=\text{const}>0,
$$
so reconstruction is exact (up to edge frames handled by `center=True`). This is
why the fifth loss's differentiable iSTFT is numerically faithful and why
overlap-add inference with a raised-cosine crossfade reconstructs the identity
mask to $<-60$ dB (`tests/test_overlap_add.py`).

### 1.3 Why magnitude + mixture phase

Phase is high-entropy and hard to regress; the **mixture phase** is a strong
prior because, wherever a source dominates a time-frequency bin, its phase is
close to the mixture's. The mask family therefore keeps $\angle X$ fixed and
learns only the non-negative magnitude gain $M$. Multiplying the real mask by the
complex spectrogram scales magnitude and preserves phase:
$\widehat V=M\odot|X|e^{i\angle X}=M\odot X$.

> **Code.** `singnet/audio/stft.py::apply_mask` (real mask × complex spectrogram);
> `tests/test_stft.py::test_apply_mask_preserves_phase`.

### 1.4 Nyquist-row handling

The real FFT returns $F=2049$ bins; bin $2048$ is the **Nyquist** bin
($f=\tfrac{sr}{2}=22.05$ kHz), which is real-valued and carries negligible energy
for MUSDB (its AAC source is band-limited to $\approx16$ kHz). The network
consumes the first $2048=2^{11}$ bins — a power of two, so five stride-2 layers
reach a clean $64\times8$ bottleneck — and the Nyquist row is re-appended with
mask $1.0$ at reconstruction, i.e. passed through untouched.

> **Code.** `drop_nyquist` (keep bins $0\!:\!2048$), `append_nyquist` (re-append a
> row of $1.0$), verified in `tests/test_stft.py::test_nyquist_drop_and_reappend_roundtrip`.

---

## 2. The mask family, its oracle bounds, and where it caps out

### 2.1 Ratio mask and the sigmoid bound

The head applies a sigmoid, so $M=\sigma(z)\in(0,1)$ elementwise. The estimated
magnitude is $\widehat{|S|}=M\odot|X|\le|X|$: a bounded ratio mask can only
*attenuate* the mixture, never amplify a bin.

### 2.2 Oracle IRM and IBM (the §7.3 anchors)

Two oracles are scored on every figure as headroom lines:
$$
M_\text{IRM}=\frac{|S|}{|S|+|A|+\varepsilon}\in[0,1),\qquad
M_\text{IBM}=\mathbf 1\big[\,|S|>|A|\,\big]\in\{0,1\}.
$$
Both are representable by (the limit of) a $[0,1]$ mask, so they upper-bound what
*this* family can achieve at this STFT resolution. Applied with the mixture phase
they give the "best a bounded magnitude mask could do if it knew the sources".

> **Code.** `singnet/eval/evaluate.py::oracle_irm` / `oracle_ibm`;
> `tests/test_eval.py`.

### 2.3 Wiener / MMSE connection

Under a zero-mean Gaussian model of independent source STFT coefficients, the
MMSE estimate of $S$ given $X$ is the **Wiener** gain
$M_\text{W}=\dfrac{|S|^2}{|S|^2+|A|^2}$ applied to $X$. The magnitude IRM above is
its amplitude-domain cousin; both are monotone in the local SNR
$|S|^2/|A|^2$ and both lie in $[0,1]$. They formalise "the oracle ratio mask is
the right target for a magnitude-masking network".

### 2.4 Where a bounded mask is provably insufficient

Because $X=S+A$ and the accompaniment can *destructively* interfere with the
target, $|X|$ can be smaller than $|S|$, forcing the ideal gain $|S|/|X|>1$. Kong
et al. (2021, arXiv:2109.05418) measure this on **22 % of TF bins** for MUSDB18. A
sigmoid mask in $[0,1]$ cannot represent those bins, so the true ceiling of
"magnitude mask + mixture phase" sits *below* the complex-mask (cIRM) oracle that
decouples and corrects phase, $\widehat S=|M|\,|X|\,e^{i(\angle X+\angle M)}$.
This is exactly why the oracle IRM/IBM lines matter (they bound *our* family) and
why unbounded/complex masks are a separate future direction, not this study.

---

## 3. The five losses

All losses share one signature and run in float32 in an autocast-disabled region
(`singnet/losses/_base.py`). Write $\widehat{|S|}=M\odot|X|$ for the masked
magnitude; means are over $(F\times T)$ bins or over the valid samples.

### 3.1 `l1mag` — L1 on magnitude (Spleeter)

$$\mathcal L=\operatorname{mean}\big|\,M\odot|X|-|S|\,\big|,\qquad
\frac{\partial\mathcal L}{\partial M}=\frac{1}{FT}\,\operatorname{sign}\!\big(M\odot|X|-|S|\big)\odot|X|.$$
The gradient magnitude is independent of the error size (only its sign matters),
so `l1mag` is robust to loud outliers and treats all bins evenly in error units.
It ignores phase entirely. *Provenance:* Spleeter. *Code:* `losses/l1mag.py`;
analytic zero at $M|X|=|S|$ tested in `tests/test_losses.py`.

### 3.2 `msemag` — MSE on magnitude (Open-Unmix)

$$\mathcal L=\operatorname{mean}\big(M\odot|X|-|S|\big)^2,\qquad
\frac{\partial\mathcal L}{\partial M}=\frac{2}{FT}\,\big(M\odot|X|-|S|\big)\odot|X|.$$
The gradient scales with the error, so large-magnitude (loud) bins dominate: MSE
chases spectral peaks and tolerates small errors in quiet bins. *Provenance:*
Open-Unmix. *Code:* `losses/msemag.py`.

### 3.3 `logl1mag` — L1 on log-magnitude (Gusó `LOGL1freq`)

$$\mathcal L=\operatorname{mean}\big|\log(M\odot|X|+\varepsilon_{\log})-\log(|S|+\varepsilon_{\log})\big|,\quad \varepsilon_{\log}=10^{-5},$$
with gradient $\propto \operatorname{sign}(\cdot)\odot|X|/(M\odot|X|+\varepsilon_{\log})$.
The $1/(\cdot)$ factor up-weights **quiet** bins, compressing dynamic range toward
perceptual loudness. The $\varepsilon_{\log}$ floor keeps $\log$ finite where the
masked magnitude is zero (the reason the loss is pinned to fp32). *Code:*
`losses/logl1mag.py`.

### 3.4 `sisdr` — negative time-domain SI-SDR (Le Roux 2019)

The estimate is reconstructed by a differentiable iSTFT,
$\hat v=\mathrm{iSTFT}(M\odot X)$, and scored against $v$.

**Projection form.** $\alpha$ is the least-squares scalar minimising
$\|\hat v-\alpha v\|^2$:
$$\frac{d}{d\alpha}\|\hat v-\alpha v\|^2=-2\langle \hat v-\alpha v,\,v\rangle=0
\;\Longrightarrow\; \alpha=\frac{\langle \hat v,v\rangle}{\|v\|^2}.$$
Then $s_\text{target}=\alpha v$ is the orthogonal projection of $\hat v$ onto
$\operatorname{span}(v)$, and the residual $e=\alpha v-\hat v$ is orthogonal to
$v$ (indeed $\langle e,v\rangle=\alpha\|v\|^2-\langle\hat v,v\rangle=0$). The
metric is
$$\boxed{\ \text{SI-SDR}(\hat v,v)=10\log_{10}\frac{\|\alpha v\|^2}{\|\alpha v-\hat v\|^2}\ }\qquad
\mathcal L_\text{sisdr}=-\overline{\text{SI-SDR}}.$$

**Scale invariance (both arguments).** For any $c\neq0$:
*estimate* $\hat v\mapsto c\hat v$ gives $\alpha\mapsto c\alpha$, so numerator and
denominator both scale by $c^2$ — value unchanged. *reference* $v\mapsto cv$ gives
$\alpha\mapsto\alpha/c$ and $s_\text{target}=\alpha v\mapsto(\alpha/c)(cv)=\alpha v$
unchanged, and $e$ unchanged — value unchanged. Both invariances are tested
numerically with $\varepsilon=0$ on non-degenerate signals
(`tests/test_metrics.py`).

**Silence singularity and the guard.** If $v=0$ then $\|v\|^2=0$ and $\alpha=0/0$
is undefined; as $v\to0$ the ratio is ill-conditioned. The code guards two ways:
an $\varepsilon=10^{-8}$ in numerator and denominator (finite arithmetic), and a
**pre-registered chunk skip** — any chunk whose target RMS is below $-60$ dBFS
(amplitude $10^{-3}$) is excluded from the batch loss for the `sisdr` arm only,
with the skip *rate* logged as a finding (MASTER_PLAN §4.4, §12). Silent chunks
are excluded *before* the SI-SDR arithmetic so a $0\cdot\infty$ never enters the
mean. *Code:* `metrics/si_sdr.py` (shared math) and `losses/sisdr.py` (guard +
negation); gradient flow through the iSTFT to the mask is tested.

### 3.5 `l1mrstft` — L1-magnitude + multi-resolution STFT auxiliary

$$\mathcal L=\operatorname{mean}\big|M\odot|X|-|S|\big|+\lambda\sum_{m=1}^{3}\big(\mathcal L^{(m)}_\text{sc}+\mathcal L^{(m)}_\text{mag}\big),\qquad\lambda=0.5,$$
with, per resolution $m$ on the waveforms $(\hat v,v)$,
$$\mathcal L_\text{sc}=\frac{\big\|\,|S_m|-|\hat S_m|\,\big\|_F}{\big\|\,|S_m|\,\big\|_F},
\qquad
\mathcal L_\text{mag}=\operatorname{mean}\big|\log(|S_m|+\varepsilon_{\log})-\log(|\hat S_m|+\varepsilon_{\log})\big|,$$
resolutions $\text{fft}=[1024,2048,512]$, $\text{hop}=[120,240,50]$,
$\text{win}=[600,1200,240]$ (Hann). Spectral convergence normalises by the target
energy and so emphasises high-energy peaks; the log-magnitude term emphasises
low-energy detail; summing over resolutions captures both fine and coarse
time-frequency structure. *Provenance:* Yamamoto et al. (Parallel WaveGAN) /
auraloss. **Aggregation note:** MASTER_PLAN §6 pins a *sum* over resolutions;
auraloss averages — the difference is a constant absorbed into $\lambda$. *Code:*
`losses/mrstft.py`; $\mathcal L_\text{MR-STFT}(w,w)=0$ tested for identical
waveforms.

---

## 4. Why optimising loss $X$ need not optimise SI-SDR

### 4.1 The reachable set

Fix a mixture $X$. The model can only produce estimates in
$$\mathcal R(X)=\big\{\,\mathrm{iSTFT}(M\odot X)\ :\ M\in[0,1]^{F\times T}\,\big\},$$
a set that is **bounded** (mask $\le1$, so $|\widehat V|\le|X|$) and **phase-locked**
(every element carries the mixture phase $\angle X$). The true $v$ is in general
*not* in $\mathcal R(X)$: its magnitude may exceed $|X|$ on the 22 % of bins from
§2.4, and its phase differs from $\angle X$.

### 4.2 Different losses ⇒ different projections

Each loss defines a geometry on estimates and therefore a **projection** of the
unreachable $v$ onto $\mathcal R(X)$:
$$\hat v_{\mathcal L}=\arg\min_{u\in\mathcal R(X)}\ \mathrm{dist}_{\mathcal L}(u,v).$$
`l1mag`/`msemag`/`logl1mag` measure distance in magnitude space (phase-blind,
differently weighted); `sisdr` measures a scale-invariant time-domain ratio
(phase-sensitive). Two different distances generally have two different minimisers
on the *same* constrained set, and neither minimiser need coincide with the
SI-SDR-optimal element of $\mathcal R(X)$. Concretely, since the phase error is
common to every $M$ (all elements share $\angle X$), the arms differ only in how
they *allocate magnitude*; whether SI-SDR's particular allocation beats L1's at
this capacity is an empirical question — the content of pre-registered **H-01a**
(the bet: it does not, beyond seed noise). This is the honest, non-circular reason
"train on the metric you report" is not guaranteed to win.

### 4.3 What the Bake-Off implies for H-01b

Jaffe & Burgoyne (2025, arXiv:2507.06917) find that for **vocals**, BSS-Eval SDR
is already the *best* perceptual proxy among tested metrics. So we do **not**
expect the MR-STFT auxiliary to raise SI-SDR. Instead, two estimates in
$\mathcal R(X)$ can share (nearly) the same SI-SDR yet differ in the *structure* of
their residual — SI-SDR is a single scalar and is blind to this. MR-STFT shapes
the residual across resolutions, plausibly trading the same energy for fewer
musical-noise/gurgling artifacts. Hence **H-01b is pre-registered to be judged by
the blind listening check at equal SI-SDR**, never by SI-SDR itself.

---

## 5. U-Net parameter count and receptive field

SingNet-C1 (`singnet/models/unet.py`) is a five-down / five-up magnitude-mask
U-Net. Convolutions are $5\times5$; a `Conv2d(c_\text{in},c_\text{out})` has
$25\,c_\text{in}c_\text{out}+c_\text{out}$ parameters (bias included) and a
`ConvTranspose2d` the same; `BatchNorm2d(c)` adds $2c$ (running statistics are
buffers, not parameters); the head `Conv2d(32,1,1)` adds $33$.

| Block | Layer | shape out | params |
|---|---|---|---|
| enc1 | Conv$5^2$ $1\!\to\!32$ + BN | $32\times1024\times128$ | 896 |
| enc2 | Conv$5^2$ $32\!\to\!64$ + BN | $64\times512\times64$ | 51,392 |
| enc3 | Conv$5^2$ $64\!\to\!128$ + BN | $128\times256\times32$ | 205,184 |
| enc4 | Conv$5^2$ $128\!\to\!256$ + BN | $256\times128\times16$ | 819,968 |
| enc5 | Conv$5^2$ $256\!\to\!512$ + BN | $512\times64\times8$ | 3,278,336 |
| dec1 | ConvT$5^2$ $512\!\to\!256$ + BN | $256\times128\times16$ | 3,277,568 |
| dec2 | ConvT$5^2$ $512\!\to\!128$ + BN | $128\times256\times32$ | 1,638,784 |
| dec3 | ConvT$5^2$ $256\!\to\!64$ + BN | $64\times512\times64$ | 409,792 |
| dec4 | ConvT$5^2$ $128\!\to\!32$ + BN | $32\times1024\times128$ | 102,496 |
| dec5 | ConvT$5^2$ $64\!\to\!32$ + BN | $32\times2048\times256$ | 51,296 |
| head | Conv$1^2$ $32\!\to\!1$ | $1\times2048\times256$ | 33 |
| **total** | | | **9,835,745** |

Decoder inputs are doubled by skip concatenation (e.g. dec2 takes
$256+256=512$ channels: dec1's 256 output plus the enc4 skip). Dropout2d(0.5) sits
on dec1–dec3 (Spleeter convention) and has no parameters.

> **Verified.** `tests/test_model.py::test_exact_parameter_count` asserts
> `SingNetC1().num_parameters == 9_835_745`. **The code is authoritative:** if the
> architecture changes, this table and the test constant change together.

**Receptive field.** For a stack of $5\times5$ stride-2 convolutions the encoder
receptive field grows as $r_\ell=r_{\ell-1}+(k-1)\prod_{i<\ell}s_i$ with $k=5$,
$s=2$: $r=5,13,29,61,125$. So each bottleneck unit sees $\approx125$ frequency
bins and $\approx125$ frames of context (~2.9 s and ~2.7 kHz); the transposed-conv
decoder and skip connections then reintroduce local detail while retaining that
global context.

---

## 6. Statistics of the design

### 6.1 Seed noise and the pooled band

Each arm is trained with 3 seeds $\{0,1,2\}$, giving a between-seed standard
deviation $s_\text{arm}$ of validation SI-SDR. For the H-01a pair the pooled
$\sigma_\text{seed}=\sqrt{\tfrac12(s^2_\text{sisdr}+s^2_\text{l1})}$ is the noise
band drawn on every sweep figure. The decision statistic is
$\Delta=\overline{\text{SI-SDR}}_\text{sisdr}-\overline{\text{SI-SDR}}_\text{l1}$;
H-01a is *supported* when $\Delta\le+\sigma_\text{seed}$ (SI-SDR training buys
nothing beyond run-to-run noise) and the full-budget test pair does not reverse
it, *refuted* when $\Delta>+\sigma_\text{seed}$ and the test pair confirms with a
95 % CI excluding 0, *mixed* otherwise.

**What $\sigma_\text{seed}$ licenses:** a statement about *training stochasticity*
— whether an arm gap exceeds what re-seeding alone produces. **What it does not
license:** a claim about the population of songs (that needs track-level
resampling, below) or a $p$-value from a single seed.

### 6.2 Paired bootstrap over tracks

The single test pass (§7.4) computes per-track deltas $d_t=\text{SI-SDR}_A(t)-\text{SI-SDR}_B(t)$
over the 50 test tracks. Resampling tracks with replacement $10{,}000$ times gives
a 95 % CI for $\mathbb E[d]$. Pairing by track removes track-difficulty variance;
the CI quantifies sampling uncertainty over songs. With only 50 tracks the CI is
deliberately wide — honesty, not weakness.

### 6.3 Wilcoxon signed-rank

A nonparametric two-sided test on $\{d_t\}$ with $H_0$: median $=0$. It asks
whether the *sign* of the per-track advantage is consistent beyond chance without
assuming normality — a robust complement to the bootstrap CI on the single
headline pair only. No $p$-values are computed on the 5-track continuity windows
(too few points; MASTER_PLAN §3.3).

### 6.4 Why 3 seeds

Three is the minimum that yields an honest (if crude) std; the design pre-commits
(gate G2) to adding seeds 3–4 *only to the two closest arms* if $\sigma_\text{seed}$
swamps the between-arm spread, bounding extra cost to $\approx6$ T4-hours.

---

*Cross-reference index.* §1 → `audio/stft.py`; §2 → `eval/evaluate.py`
(`oracle_irm`/`oracle_ibm`); §3 → `losses/{l1mag,msemag,logl1mag,sisdr,mrstft}.py`
and `metrics/si_sdr.py`; §4 → `models/unet.py` + `losses/`; §5 →
`models/unet.py`, asserted by `tests/test_model.py`; §6 → the analysis cells of
`notebooks/03_loss_study_experiments.ipynb`.
