# THEORY — Direction 03: the math behind the mini band-split study

This note derives the mathematics the code implements for Direction 03. It follows
the MASTER_PLAN §10 outline and, like Directions 01/02, ties every result to a
concrete function path in `singnet/` + `scripts/`. The **crux** is §4 (parameter
and compute accounting): the whole experiment stands or falls on an honest,
exact parameter match, and the numbers below are the same ones
`scripts/match_params.py` writes to `results/param_match_table.md` and
`tests/test_match_params.py` asserts against the built `nn.Module`.

Notation. A mono mixture chunk has magnitude spectrogram $|X|\in\mathbb R_{\ge0}^{F\times T}$
on the $F=2048$ network bins and $T=256$ frames (Direction 01 THEORY §1). A
magnitude-mask U-Net predicts $M\in[0,1]^{F\times T}$; the vocal estimate is
$\widehat V=M\odot|X|e^{i\angle X}$, $\hat v=\mathrm{iSTFT}(\widehat V)$. Direction
03 changes **only the encoder front-end**: the baseline's single full-spectrum
encoder vs. $B=3$ per-band encoder towers, everything else (STFT, featurization,
bottleneck+decoder design, mask head, Nyquist handling, loss, optimizer,
protocol) held fixed.

> **Code.** The model is `singnet/models/bandsplit_unet.py::BandSplitUNet` (blocks
> reused from `singnet/models/unet.py`); the width search is
> `scripts/match_params.py`; the per-band metrics are `singnet/eval/banded.py`;
> the three-arm verdict is `singnet/analysis/bandsplit.py`.

---

## 1. The band-split idea, formalized

### 1.1 Weight-sharing scope

A 2-D convolution is **translation-equivariant along frequency**: one filter bank
$W$ slides over *all* $F$ bins, so the baseline encoder applies the **same**
weights to a 100 Hz region and a 15 kHz region. Formally the baseline learns a
single operator $W$ with
$$(\mathcal E_{\text{base}}x)[:,f,t]=\phi\big(W * x\big)[:,f,t]\quad\text{for every }f\in[0,F),$$
i.e. **full-spectrum weight sharing**.

The band-split encoder partitions $[0,F)$ into $B$ contiguous bands
$\mathcal B_b=[e_b,e_{b+1})$ and gives each its **own** operator $W_b$:
$$(\mathcal E_{\text{split}}x)[:,f,t]=\phi\big(W_b * x_{\mathcal B_b}\big)[:,f,t]\quad\text{for }f\in\mathcal B_b,$$
**band-local weight sharing**. Each band's spectral statistics — a low band is
dense with vocal fundamentals and formants; a high band is sparse overtones — get
**dedicated capacity** instead of sharing one filter bank with every other band.
This is the load-bearing idea of the SOTA family (BSRNN → BS/Mel-RoFormer → SCNet
→ Moises-Light), isolated here at compact scale.

### 1.2 What we keep and what we drop

BSRNN adds **per-band feature MLPs** *and* a cross-band sequence model (RNN);
Mel-RoFormer keeps the partition but swaps in a transformer and shows **mel**
bands beat heuristic bands. Every published demonstration therefore confounds the
partition with a powerful sequence model and with scale/extra data. We keep
exactly one ingredient — **the partition + dedicated per-band conv capacity** —
and **drop the sequence model**. Dropping it is what isolates the partition: with
a transformer in the loop, a win could be the attention, not the bands. The towers
are the *same* 5-level conv block design as the baseline (§4), so the **only**
change from baseline is the weight-sharing scope of §1.1. No cross-band mixing
happens until the bottleneck (the towers are independent); mixing returns only in
the shared decoder, whose convolutions span the frequency-concatenated features.

> **Code.** `BandSplitUNet.forward`: `split_and_pad` → independent `towers[b]` →
> frequency-axis `torch.cat` of matched-channel level outputs → baseline decoder.

---

## 2. The mel scale and the band edges

### 2.1 HTK mel and the equal-mel partition

The HTK mel scale is
$$m(f)=2595\,\log_{10}\!\Big(1+\frac{f}{700}\Big),\qquad
f(m)=700\,\big(10^{m/2595}-1\big).$$
An **equal-mel** partition of $[f_{\min},f_{\max}]$ into $B$ bands places interior
edges at $m_i=\tfrac{i}{B}\,m(f_{\max})$ (with $f_{\min}=0$), mapped back to Hz and
then to the nearest STFT bin $k(f)=\mathrm{round}\!\big(f\cdot n_{\text{fft}}/sr\big)$.

For the project pins $sr=44100,\ n_{\text{fft}}=4096$ ($f_{\max}=22050$ Hz,
$F=2048$ network bins), $m(f_{\max})=2595\log_{10}(32.5)=3923.34$, so
$$m_1=1307.78\Rightarrow f_1=1533.9\text{ Hz}\Rightarrow k_1=142,\qquad
m_2=2615.56\Rightarrow f_2=6428.9\text{ Hz}\Rightarrow k_2=597.$$
The mel edge list in bins is
$$\boxed{[\,0,\ 142,\ 597,\ 2048\,]}\quad(\text{interior}\approx 1.53,\ 6.43\text{ kHz}),$$
matching `mel_edges()` **exactly** (0-bin deviation from the plan's estimate; the
test recomputes the formula independently and asserts self-consistency + ±3-bin
proximity). The equal-**bin** (uniform) control is
$$[\,0,\ 683,\ 1365,\ 2048\,]\quad(\text{interior}\approx 7.35,\ 14.70\text{ kHz}).$$

> **Code.** `bandsplit_unet.py::mel_edges` / `uniform_edges`;
> `tests/test_bandsplit_model.py::test_mel_edges_formula_self_consistency`.

### 2.2 Where vocal energy lives, and the AAC caveat

Sung vowels put their energy in the fundamental and first formants, roughly
**100 Hz – 4 kHz**; sibilants and overtones extend higher but at lower energy. The
mel layout concentrates two of its three bands below 6.43 kHz (narrow low, wide
mid), **matching** the vocal-energy density — the pre-registered mechanism is that
gains, if any, concentrate in these bands (the per-band figure, §5). The uniform
layout instead spends two thirds of its bins above 7.35 kHz, where vocals are
sparse. This is precisely the difference the $E_2$ contrast (§6) tests.

**AAC ceiling.** MUSDB18's sources are AAC-encoded and band-limited to
$\approx16$ kHz. The mel top band $[6.43,22.05]$ kHz therefore contains a large
$[16,22]$ kHz sub-region that is **dead spectrum** — capacity the mel layout
spends where there is nothing to model. This is a *property of mel spacing on this
data*, stated in advance, part of the treatment (not a bug); the EDA
vocal-energy-by-band table quantifies it (RUN LATER), and the banded metrics
NaN-guard the dead region (§5).

---

## 3. Boundary analysis (the "refuted-negative" mechanism)

### 3.1 Padding at band edges

Each band of width $w_b$ is zero-padded on its **high-frequency side** to
$p_b=\lceil w_b/32\rceil\cdot32$ so five stride-2 layers halve exactly
($32=2^5$). For our layouts:
$$\text{mel}:\ (w_b)=(142,455,1451)\to(p_b)=(160,480,1472),\quad P=\textstyle\sum p_b=2112,$$
$$\text{uniform}:\ (w_b)=(683,682,683)\to(p_b)=(704,704,704),\quad P=2112.$$
The padding injects an artificial signal→zero discontinuity at each band's top;
the first conv layer sees it as an edge. The padded total $P=2112$ is carried
through the concatenated skips, bottleneck and decoder (all spatial dims stay
aligned because encoder and decoder both derive from $P$), and the final
$(B,1,2112,256)$ mask is **cropped back** to $(B,1,2048,256)$ by gathering each
band's real bins — a fixed index map that round-trips exactly (unit-tested).

### 3.2 Per-tower receptive field and severed correlations

A tower is the baseline 5-level $5\times5$/stride-2 encoder, so its receptive
field grows $r_\ell=r_{\ell-1}+(k-1)\prod_{i<\ell}s_i=5,13,29,61,125$ bins — the
**same** as the baseline. But it is **confined to its band**: a tower cannot see
across $e_{b+1}$. A spectral structure that straddles a band boundary (e.g. a
vocal harmonic sitting near the mel edge $k_1=142\approx1.53$ kHz) is split between
two towers that **cannot share it in the encoder**. The baseline's full-spectrum
encoder captures such cross-boundary correlations freely. This is the concrete
mechanism for a pre-registered **refuted (negative)** outcome: band isolation can
*sever* correlations worth more than the dedicated capacity buys. Cross-band
information returns only in the shared decoder — the encoder features are already
band-siloed. The per-band figure localizes any damage at **bands adjacent to
boundaries** vs. band interiors (§5, MASTER_PLAN §12).

---

## 4. Parameter and compute accounting (the crux)

### 4.1 Conv parameters are independent of spatial extent

A `Conv2d`/`ConvTranspose2d`$(c_{\text{in}},c_{\text{out}},k\times k)$ has
$k^2c_{\text{in}}c_{\text{out}}+c_{\text{out}}$ parameters (weight + bias) **for
any input spatial size**; its `BatchNorm2d`$(c_{\text{out}})$ adds $2c_{\text{out}}$
(running stats are buffers, not parameters). So one conv+BN block costs
$$b(c_{\text{in}},c_{\text{out}})=25\,c_{\text{in}}c_{\text{out}}+3\,c_{\text{out}}\quad(k=5).$$
A tower on a 160-bin band and a tower on a 1472-bin band with the same
$(c_{\text{in}},c_{\text{out}})$ have **identical** parameters. **Corollary
(load-bearing):** the mel and uniform variants — different band widths, same
widths $c$ — have the **identical** parameter count. The band edges change
*compute* (§4.5), not *parameters*.

### 4.2 Closed-form $P(c)$

With base width $c$ (channels $1\!\to\!c\!\to\!2c\!\to\!4c\!\to\!8c\!\to\!b_5$) the
encoder tower and the baseline-design decoder (which consumes the frequency
concatenations exactly as the baseline consumes its skips) cost
$$
\begin{aligned}
E(c,b_5)&=b(1,c)+b(c,2c)+b(2c,4c)+b(4c,8c)+b(8c,b_5)=1050c^2+70c+200c\,b_5+3b_5,\\
D(c,b_5)&=\underbrace{b(b_5,8c)}_{\text{dec1}}+\underbrace{b(16c,4c)}_{\text{dec2}}+\underbrace{b(8c,2c)}_{\text{dec3}}+\underbrace{b(4c,c)}_{\text{dec4}}+\underbrace{b(2c,c)}_{\text{dec5}}=2150c^2+48c+200c\,b_5,\\
\text{head}(c)&=c+1.
\end{aligned}
$$
The variant has **3 towers + one decoder + one head**:
$$\boxed{\,P(c,b_5)=3E(c,b_5)+D(c,b_5)+\text{head}(c)=5300c^2+259c+800c\,b_5+9b_5+1.\,}$$
At the "pure" width $b_5=16c$ this collapses to
$$P_{\text{pure}}(c)=18100\,c^2+403\,c+1.$$
Sanity: the baseline is one encoder + decoder + head at $c=32,\ b_5=512$, giving
$4250\cdot32^2+118\cdot32 \;+\; 5350\cdot32^2+48\cdot32 \;+\;33 = 9{,}835{,}745$
— the number `tests/test_model.py` pins. (Here $E_{\text{base}}(c)=4250c^2+118c$,
$D_{\text{base}}(c)=5350c^2+48c$.)

### 4.3 The width-selection equation and its solution

Matching $P_{\text{pure}}(c)=P_0=9{,}835{,}745$ gives
$18100c^2+403c+1=9{,}835{,}745\Rightarrow c\approx23.31$. Neither integer pure
width lands within the $\pm2\%$ band:
$$P_{\text{pure}}(23)=9{,}584{,}170\ (-2.558\%),\qquad P_{\text{pure}}(24)=10{,}435{,}273\ (+6.095\%).$$
The deterministic rule (`match_params.py`): (1) take the **largest base width
whose pure variant stays under budget** → $c=23$ (keeps $b_5\ge16c$ and matches the
plan's $c\approx23$ estimate); (2) close the gap with the **single positive
bottleneck bump** $\Delta$ (level-5 width $b_5=16c+\Delta$) that **minimizes**
$|P-P_0|$. Since $P$ is linear in $b_5$ with slope $800c+9=18409$ (at $c=23$), the
argmin over $\Delta\ge0$ of $|9{,}584{,}170+18409\,\Delta-9{,}835{,}745|$ is
$$\Delta=\Big\lfloor\tfrac{251{,}575}{18409}\Big\rceil=14\Rightarrow b_5=382,\qquad
P(23,382)=9{,}841{,}896\ \ (\boxed{+0.0625\%}).$$
Both variants have this exact count (§4.1). Pinned as `MATCHED_BASE_WIDTH=23`,
`MATCHED_BOTTLENECK_WIDTH=382`.

### 4.4 Per-module table (all three arms)

Encoder rows are the **total across towers** (baseline: 1 encoder at $c=32$;
variants: 3 towers at $c=23,\ b_5=382$). **These numbers equal
`scripts/match_params.py` output and the unit-test assertions.**

| Module | baseline ($c=32$) | split_mel / split_uniform ($c=23,\ b_5=382$) |
|---|---:|---:|
| enc1 | 896 | 1,932 |
| enc2 | 51,392 | 79,764 |
| enc3 | 205,184 | 318,228 |
| enc4 | 819,968 | 1,271,256 |
| enc5 (bottleneck) | 3,278,336 | 5,275,038 |
| dec1 | 3,277,568 | 1,757,752 |
| dec2 | 1,638,784 | 846,676 |
| dec3 | 409,792 | 211,738 |
| dec4 | 102,496 | 52,969 |
| dec5 | 51,296 | 26,519 |
| head | 33 | 24 |
| **encoder Σ** | **4,355,776** | **6,946,218** |
| **decoder Σ** | **5,479,936** | **2,895,654** |
| **total** | **9,835,745** | **9,841,896** |

The band-split arm spends **$1.59\times$** the baseline's parameters on the encoder
(three towers) and correspondingly **less** on the decoder (a narrower $c$), for a
total within $+0.06\%$ of the baseline. Both variants: `9,841,896`, `+0.0625%`,
$\le2\%$ ✓.

### 4.5 Compute: capacity $\ne$ compute

Unlike parameters, a conv layer's **MAC** cost scales with output spatial size:
$\text{MACs}_\ell\approx S_{\text{out},\ell}\cdot25\,c_{\text{in}}c_{\text{out}}$
(one $25c_{\text{in}}$-weight dot product per output pixel per output channel). The
baseline encoder runs on the full $F=2048$ frequency axis; the variant runs **3
towers each on $\approx F/3$**. The $3\times$ tower count and the $1/3$ spatial
extent **cancel**:
$$\sum_{b}\frac{p_b}{2^\ell}=\frac{P}{2^\ell}\approx\frac{2048}{2^\ell}\ (\text{padded }2112),$$
so the variant encoder MACs $\approx$ **a single encoder at width $c=23$** on the
padded spectrum:
$$\frac{\text{MACs}^{\text{enc}}_{\text{variant}}}{\text{MACs}^{\text{enc}}_{\text{base}}}
\approx\Big(\frac{c}{32}\Big)^2\cdot\frac{P}{2048}=\Big(\frac{23}{32}\Big)^2\cdot\frac{2112}{2048}\approx0.53,$$
and with the exact $b_5=382$ bottleneck the analytic value is **$0.541$** (both
mel and uniform — the MAC sum depends only on $P=2112$, so the two variants are
matched in **compute too**). Whole-model MACs are $\approx0.54\times$ baseline.

**What this licenses.** At matched *parameters* the band variants are
$\approx1.85\times$ cheaper in encoder compute — an **incidental efficiency
bonus**, reported honestly as a descriptive secondary, **never** as the claim. The
hypothesis (§6) is about *capacity allocation* and is decided at matched
parameters; the $0.54\times$ FLOPs figure is a separate result. **Capacity $\ne$
compute:** three towers hold $3\times$ the per-encoder parameters (matched to the
baseline by shrinking width), yet do $3\times\tfrac13=1\times$ the spatial work at
a smaller width — hence half the compute.

> **Code.** Closed form + search + cross-check: `scripts/match_params.py`
> (`variant_params`, `match_width`, `verify_against_model`, `per_module_rows`);
> asserted in `tests/test_match_params.py` and `tests/test_bandsplit_model.py`.

---

## 5. Per-band mechanism metrics

### 5.1 Band-limited SI-SDR

For an analysis band $b=[\underline f,\overline f)$, zero every STFT bin outside
$b$ in **both** the reference and the estimate (each via its **own** STFT), invert,
and score SI-SDR on the two band-limited waveforms:
$$\text{SI-SDR}_b(\hat v,v)=\text{SI-SDR}\big(\mathrm{iSTFT}(\mathbf 1_b\odot\widehat V_{\!\text{est}}),\ \mathrm{iSTFT}(\mathbf 1_b\odot V)\big).$$
It is **well-defined**: both signals are filtered by the *same* linear band mask,
so there is no mask/phase asymmetry (unlike masking only the estimate). It measures
reconstruction quality **restricted to $b$**; it does **not** measure cross-band
phase coherence, and — being a ratio — it is ill-conditioned when the reference
has little energy in $b$ (a near-empty band has a noisy SI-SDR). We therefore
**NaN-guard**: a band with no bins, or a silent reference band, returns NaN rather
than a $0/0$.

### 5.2 Band magnitude error

$$\text{MagErr}_b=\frac{\operatorname{mean}_{(k,t)\in b}\big|\,|R[k,t]|-|E[k,t]|\,\big|}
{\operatorname{mean}_{(k,t)\in b}|R[k,t]|},$$
a dimensionless **relative** L1 error, NaN when the normalizer (the reference band
level) is below $\varepsilon$ — the AAC-dead-top-band guard. It is energy-agnostic
in the numerator but energy-normalized, so a quiet band's small absolute errors are
not drowned by a loud band's.

### 5.3 The layout-neutral 6-band grid

Scoring each arm on **its own** band edges would be circular. The analysis grid is
the **union** of both variants' interior edges with a 100 Hz floor split and the
$[0,sr/2]$ outer edges:
$$\{0,\ 100,\ 1528.9,\ 6427.7,\ 7353.6,\ 14696.4,\ 22050\}\ \text{Hz}\ \Rightarrow\ 6\ \text{bands},$$
so neither the mel nor the uniform arm is measured on a grid that favors it. The
mechanism figure plots per-band (variant − baseline) deltas with per-track spread;
the pre-registered reading (§6, MASTER_PLAN §2): gains should concentrate in the
vocal-dense bands ($\sim100$ Hz–4 kHz) — else the dedicated-capacity story is
questioned even under a positive headline.

> **Code.** `singnet/eval/banded.py::{band_limited_sisdr, band_mag_error,
> analysis_grid_hz, banded_report}`; `tests/test_banded.py`.

---

## 6. Statistics of the design

### 6.1 Pooled three-arm noise band

Each arm has 3 seeds, giving between-seed sample variances
$s^2_{\text{base}},s^2_{\text{uni}},s^2_{\text{mel}}$. The design's noise band is
$$\sigma_{\text{seed}}=\sqrt{\tfrac13\big(s^2_{\text{base}}+s^2_{\text{uni}}+s^2_{\text{mel}}\big)}$$
(a $\sqrt{\text{mean of the three variances}}$, 6 dof). Init caveat (MASTER_PLAN
§3.3): matched seeds do **not** produce matched initial weights across arms (the
shapes differ), so seed matching here controls **data order**, not init; the
3-seed spread absorbs init variance. No claim rests on a single-seed pair.

### 6.2 The ordered chain, and why it is *not* a multiple comparison

H-03 is the **ordered chain** $\overline v(\text{mel})>\overline v(\text{uni})>\overline v(\text{base})$,
tested by two one-sided effects
$$E_1=\overline v(\text{uni})-\overline v(\text{base})\ (\text{splitting}),\qquad
E_2=\overline v(\text{mel})-\overline v(\text{uni})\ (\text{mel spacing}),$$
each against $\sigma_{\text{seed}}$. This is **one** jointly pre-registered claim,
not two independent hypotheses, so we do **not** apply a multiple-comparison
correction to the two links — there is no "report the largest of several deltas".
What keeps the *partial* outcomes honest is that each is a **distinct
pre-registered cell** enumerated in advance (§13): $E_1$-only (splitting helps,
mel $\approx$ uniform), $E_2$-only (mel rescues an unhelpful split), refuted-null
(all within $\pm\sigma$), refuted-negative (baseline beats both by $>\sigma$).
Because the cells are named before seeing data, selecting the one that occurs is
reporting, not $p$-hacking.

$$\text{verdict}=\begin{cases}
\text{fully supported} & E_1>\sigma\ \wedge\ E_2>\sigma,\\
\text{partial }E_1 & E_1>\sigma\ \wedge\ E_2\le\sigma,\\
\text{partial }E_2 & E_2>\sigma\ \wedge\ E_1\le\sigma,\\
\text{refuted (negative)} & \text{neither},\ \overline v(\text{base})-\overline v(\text{uni})>\sigma\ \wedge\ \overline v(\text{base})-\overline v(\text{mel})>\sigma,\\
\text{refuted (null)} & \text{neither, otherwise.}
\end{cases}$$

### 6.3 Budget-dependence guard

The primary verdict is on the REDUCED (16 k-step) sweep. The FULL (40 k) pair
(best variant + baseline, seed 0) must not **reverse the sign** of its comparison;
if it does, the verdict downgrades to **mixed (budget-dependent)** — itself a
pre-registered methodological finding about short-training architecture
comparisons. This guards the "the towers just needed longer" objection at
$2.5\times$ the budget.

> **Code.** `singnet/analysis/bandsplit.py::{three_arm_sigma, ordered_chain_verdict}`
> (reusing `analysis/scaling.py::pooled_seed_sigma`); `tests/test_bandsplit_stats.py`
> exercises every verdict cell.

---

## 7. Standalone LaTeX mirror

`theory/theory.tex` compiles the same content standalone (`article` +
`amsmath`/`amssymb`, no external figures). Compilation is untested in this
CPU/no-TeX environment (noted there), but the source is self-contained.

---

*Cross-reference index.* §1 → `models/bandsplit_unet.py::BandSplitUNet.forward`;
§2 → `bandsplit_unet.py::{mel_edges, uniform_edges}`; §3 → `BandSplitUNet.{split_and_pad,
crop_back}`; §4 → `scripts/match_params.py` (asserted by `tests/test_match_params.py`
and `tests/test_bandsplit_model.py`, mirrored from `results/param_match_table.md`);
§5 → `eval/banded.py`; §6 → `analysis/bandsplit.py` + the analysis cells of
`notebooks/02_bandsplit_experiments.ipynb`. Every number in §4 is machine-checked
against the built model.
