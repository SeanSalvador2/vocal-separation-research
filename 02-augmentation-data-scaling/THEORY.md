# THEORY — Direction 02: the math behind augmentation factorization and data scaling

This note derives the mathematics the code implements for Direction 02. It follows
the MASTER_PLAN §10 outline and, like Direction 01's `THEORY.md`, ties every result
to a concrete function path in `singnet/`. The two questions — *what is each
augmentation transform worth?* and *how far do 86 songs go?* — are both, at heart,
questions about the **training distribution** the model is fed, so the note starts
there.

Notation. A training example is a mono chunk pair $(v, a)$: target vocals $v$ and
accompaniment $a$, each in $\mathbb R^{L}$ ($L = 264{,}600$ samples, 6 s at 44.1 kHz).
The mixture is $x = v + a$ (additivity exact by construction, §5 of the plan). STFTs
are $V, A, X$ with $X = V + A$ by linearity; magnitudes $|V|, |A|, |X|$. The network
consumes a per-chunk-standardized $\log(1+|X|)$ and predicts a mask $M\in[0,1]^{F\times T}$;
the vocal estimate is $\widehat V = M\odot|X|e^{i\angle X}$, $\hat v = \mathrm{iSTFT}(\widehat V)$
(Direction 01 THEORY §1–2). The training loss for this direction is `l1mag`,
$\mathcal L = \operatorname{mean}\big||M|X| - |V|\big|$.

> **Code.** The transforms are `singnet/data/augment.py`; the mixture construction and
> per-transform RNG streams are `singnet/data/musdb_dataset.py::MusdbChunks`; the
> analysis functions are `singnet/analysis/scaling.py`; the config switchboard is
> `singnet/utils/config.py::augment_switches`.

---

## 1. Augmentation as distribution design

### 1.1 Remix as a product of marginals

A real song $k$ emits a **coupled** pair $(v_k, a_k)\sim p_{\text{song}}$: the vocal
and the backing track share key, tempo, downbeat grid, and production. Write the two
marginals of that joint as
$$p_V(v)=\!\int\! p_{\text{song}}(v,a)\,da,\qquad p_A(a)=\!\int\! p_{\text{song}}(v,a)\,dv .$$

**Cross-song remixing** draws the vocal from track $i$ and the accompaniment from an
**independently** chosen track $j$, so the remixed example is distributed as the
**product of the marginals**
$$\boxed{\,p_{\text{remix}}(v,a)=p_V(v)\,p_A(a)\,}\;\ne\;p_{\text{song}}(v,a)\quad\text{in general.}$$
The two distributions share their marginals by construction but differ in their
**dependence structure**: $p_{\text{song}}$ carries the vocal↔accompaniment coupling of
real music; $p_{\text{remix}}$ destroys it (independence). This is the exact sense in
which remix is not "adding noise" — it re-designs the joint law while holding each
source's own statistics fixed.

> **Code.** `MusdbChunks.__getitem__` draws the vocal track/window from the `sample`
> stream and, when `remix` is on, the accompaniment track/window from an **independent**
> `remix` stream — the two-marginal product above. With `remix` off it reuses the same
> track and window (the coupled $p_{\text{song}}$ draw, §1.3).

### 1.2 It moves the mixture manifold, not just the density

Let the set of reachable real mixtures be $\mathcal M=\{v+a:(v,a)\in\operatorname{supp}p_{\text{song}}\}$
and the remixed set be the **Minkowski sum** of the source supports
$$\mathcal M_{\otimes}=\operatorname{supp}p_V+\operatorname{supp}p_A=\{v+a:v\in\operatorname{supp}p_V,\;a\in\operatorname{supp}p_A\}.$$
Because every real pair is also an admissible independent pair, $\mathcal M\subseteq\mathcal M_{\otimes}$,
and the inclusion is generically strict and dimension-increasing: coupling confines real
mixtures to a lower-dimensional sheet of the product space (only key/tempo-compatible
combinations occur), while remix fills the surrounding volume. A separator trained on
$\mathcal M_{\otimes}$ must succeed on a **strictly larger** mixture set that still
contains the test mixtures ($\mathcal M\subseteq\mathcal M_{\otimes}$) — the mechanism by
which remix is expected to be the top small-data lever (pre-registered **H-02a**).

### 1.3 Where the independence approximation breaks (state it, don't hide it)

Two failures of the clean product-of-marginals picture, both reported rather than
idealized away:

1. **Key/tempo coherence.** $p_{\text{remix}}$ places mass on musically *incoherent*
   mixtures (a ballad vocal over a techno backing) that the test set — real songs — never
   contains. Formally $\mathcal M_{\otimes}\setminus\mathcal M\neq\varnothing$ and the
   training law puts weight there, a **covariate shift** between train and test. This can
   *understate* the benefit remix would give if the test distribution were as rich, or
   introduce a train-only regime the model wastes capacity on. It is the honest caveat on
   any remix result (MASTER_PLAN §12).
2. **Stem bleed.** MUSDB18 stems are not perfectly isolated: the vocal stem $v_k$ carries
   a little of its *own* song's accompaniment $a_k$ (mic bleed), so $v_k$ and $a_k$ retain
   a residual within-song correlation that the marginal $p_V$ inherits. Under remix the
   bleed in $v_i$ belongs to song $i$ while the accompaniment is song $j$ — a small
   inconsistency the idealized $p_V p_A$ ignores. It is second-order (bleed is quiet) but
   real.

A further pinned detail: the remix partner is an **independent** draw from the active pool
(UMX `random_track_mix` / Demucs `Remix`), so a source can rarely pair with its own track
($\Pr[i=j]=1/n$, ≈1.2 % at $n=86$); this is the code-verified behavior, documented in
`results/DEVIATIONS.md`, and does not bias the marginals.

---

## 2. Effective sample size — combinatorics with honest bounds

Let each of $N$ training songs yield $c=\lfloor T/L\rfloor$ non-overlapping 6 s chunks
($T\approx240$ s per MUSDB track, $L=6$ s, so $c\approx40$). Two regimes:

**Without remix** the model sees **coupled** chunk pairs; the pool has
$$P_{\text{no-remix}}\approx N\,c\quad(\approx 86\times40\approx 3.4\times10^{3}\text{ pairs}).$$
These are far from independent: adjacent chunks of one song share instrumentation, key and
timbre, so the *coherence* degrees of freedom are closer to the **song count** $N$ than to
$Nc$.

**With remix** any vocal chunk pairs with any accompaniment chunk:
$$P_{\text{remix}}\approx (Nc)_V\times(Nc)_A=(Nc)^2\quad(\approx (3.4\times10^{3})^2\approx1.2\times10^{7}\text{ mixtures}).$$
The quadratic blow-up is real as a count of **distinct mixtures**, but it is *not*
$(Nc)^2$ worth of independent information, for a concrete reason: those $10^7$ mixtures are
assembled from only $2Nc$ **source atoms** (the same $Nc$ vocal chunks and $Nc$
accompaniment chunks, reused). Remix manufactures **pairings**, not new source content — it
cannot invent a singer the $N$ songs do not contain.

**Honest bounds on the effective sample size $n_{\text{eff}}$.** Rather than a fabricated
precise number,
$$\underbrace{N}_{\text{song-level coherence dof}}\;\lesssim\;n_{\text{eff}}\;\lesssim\;\underbrace{(Nc)^2}_{\text{distinct mixtures}},$$
with the truth governed by the (unknown) correlation length across chunks and across songs
(many MUSDB songs share genre/era/production, so cross-song novelty is sub-linear). The
operational consequence for the scaling curve: doubling songs $N\to2N$ **doubles** the
source atoms (linear) and **quadruples** the pairings (quadratic), yet raises the
*musical* diversity — new singers, new arrangements — only linearly and with diminishing
novelty. This is exactly why "12× more songs ≠ 12× more effective data" and why the scaling
question is empirical, not a combinatorial identity.

> **Code.** The pool the sampler draws from is `MusdbChunks.tracks` (its size is
> `MusdbChunks.n_songs`, logged to the registry as `n_songs`); the remix product is the two
> independent stream draws in `__getitem__`.

---

## 3. Gain and flip as group actions on a magnitude model

### 3.1 Sign flip is exactly magnitude-invariant — the built-in negative control

Sign flip is the action of $\mathbb Z_2=\{+1,-1\}$ on a source, $s\mapsto -s$, applied
per source with $p=\tfrac12$ (Demucs `FlipSign`). The STFT is **linear**, so for any window
and hop
$$\mathrm{STFT}(-s)[k,m]=\sum_n w[n]\,(-s)[n+mH]\,e^{-i2\pi kn/N}=-\,\mathrm{STFT}(s)[k,m],$$
and therefore
$$\boxed{\ \big|\mathrm{STFT}(-s)\big|=\big|-\mathrm{STFT}(s)\big|=\big|\mathrm{STFT}(s)\big|.\ }$$
A magnitude-mask model's entire training signal is a function of $\big(|X|,|V|\big)$ only:
the input is the standardized $\log(1+|X|)$ and the `l1mag` target is $|V|$ (Direction 01
THEORY §3.1); phase is used *only* at reconstruction, and not at all inside the `l1mag`
loss. The target magnitude $|V|=|{-V}|$ is thus **exactly** invariant to flipping the
vocals.

The one subtlety, stated honestly: flip is applied **per source**, so it can flip the
vocal but not the accompaniment, giving a mixture $x=-v+a$ with $|X|=|A-V|\ne|V+A|$ in
general — seemingly a new input. It washes out at the distribution level because audio has
**no absolute polarity** (a waveform and its negation are perceptually identical and a
microphone's polarity is arbitrary), so each source marginal is negation-symmetric,
$p_S(s)=p_S(-s)$, hence $A\stackrel{d}{=}-A$ and
$$\big|V+A\big|\ \stackrel{d}{=}\ \big|V+(-A)\big|=\big|V-A\big|.$$
The two configurations flip produces are **equidistributed**; flip merely relabels within
one distribution. With remix on (sources already independent), independent per-source flips
are therefore a **measure-preserving** map of the training law onto itself, so the
population loss landscape — and its optimum — is unchanged:
$$\boxed{\ \Delta_{\text{flip}}=\overline{\mathrm{val}}(\text{FULL})-\mathrm{val}(\text{FULL}\setminus\text{flip})\approx 0\ \text{within}\ \sigma_{\text{seed}}.\ }$$
This is the design's **pre-registered placebo** (MASTER_PLAN §12): a magnitude model
*cannot* benefit from sign flip, so measuring $\Delta_{\text{flip}}>\sigma_{\text{seed}}$
signals a bug (RNG-stream leakage, a normalization fault, or an unintended phase-sensitive
term) and **halts interpretation** before any other Δ is read. The property is robust even
to the loss: SI-SDR (the §3.4 contingency loss) is itself sign-invariant — $\hat v\mapsto-\hat v$
sends $\alpha\mapsto-\alpha$ and leaves $\|s_{\text{target}}\|,\|e\|$ fixed (Direction 01
THEORY §3.4) — so flip is a negative control under both losses.

> **Code.** `augment.py::random_sign_flip` (the $\mathbb Z_2$ action, its own RNG stream);
> the invariance is exercised in `tests/test_augment_switchboard.py`
> (`test_disabling_flip_leaves_remix_and_gain_draws_unchanged`, which asserts the two
> sources' **magnitudes** are bit-identical with flip on vs off — only signs move).

### 3.2 Gain interacts with per-chunk standardization: the target scales, the input does not

Per-source gain multiplies each source by $g_v,g_a\sim U(0.25,1.25)$ (UMX `_augment_gain` /
Demucs `Scale`). Decompose it into a **common** scale and a **relative** balance,
$$g_v=g\,\rho_v,\qquad g_a=g\,\rho_a,\qquad g=\sqrt{g_vg_a},\quad \rho_v/\rho_a=g_v/g_a .$$

*Common scale $g$ — invisible on the input.* The model input is the per-chunk
**standardized** feature $\tilde u=\operatorname{std}\!\big(\log(1+|X|)\big)$ (subtract the
chunk's scalar mean, divide by its scalar std; Direction 01 THEORY §1.3, model §5). Under
$x\mapsto g\,x$ we have $|X|\mapsto g|X|$ and, for the bins that carry energy
($|X|\gg1$), $\log(1+g|X|)\approx \log g+\log|X|$ — a near-constant **additive shift**
$\log g$ that mean-subtraction removes, with the std-division absorbing the residual scale.
So per-chunk standardization renders the input **approximately invariant to a common gain**
(exactly so in the $|X|\gg1$ limit; the $\log(1+\cdot)$ floor leaves a small quiet-bin
residual). The model's *input statistics* do not encode the overall level.

*But the target scale genuinely changes.* The `l1mag` target is $|V|\mapsto g_v|V|$, which
does **not** cancel against a standardized input. Equivalently, in ratio form the ideal mask
is
$$M^\star=\frac{|V|}{|X|}\ \longmapsto\ \frac{g_v|V|}{|g_vV+g_aA|}=\frac{|V|}{\big|V+(g_a/g_v)A\big|},$$
which depends only on the **relative** gain $g_a/g_v$, i.e. the vocal-to-accompaniment
balance. Hence the two honest statements of the task: the standardized **input statistics
are (near-)invariant** to gain's common component, while the **target/mask scale changes**
with the relative component. Gain's *informative* content is therefore the augmentation of
the source-balance (SNR) distribution — teaching robustness to how loud the vocal sits in
the mix — not the overall level, which the front end has already normalized away. Its
$\Delta_{\text{gain}}$ is expected positive but small relative to remix, and is the design's
secondary (descriptive) contrast (§5.2).

> **Code.** `augment.py::random_gain` (per-source $U(0.25,1.25)$, its own stream); the
> standardization is `singnet/audio/stft.py` / the model front end used by
> `singnet/train/loop.py::prepare_batch`.

---

## 4. Scaling-curve models at four points

### 4.1 Three candidate forms and what four points can identify

| Form | Equation | Params | Saturates? |
|---|---|---|---|
| log-linear | $y=a+b\log_2 N$ | 2 | no |
| saturating exponential | $y=y_\infty-c\,e^{-\gamma N}$ | 3 | yes ($\to y_\infty$) |
| power law to a ceiling | $y=y_\infty-c\,N^{-\gamma}$ | 3 | yes ($\to y_\infty$) |

We measure four points $N\in\{21,43,64,86\}$. A **2-parameter** log-linear fit leaves 2
residual degrees of freedom, so its lack-of-fit is checkable. A **3-parameter** saturating
form leaves 1 dof, and — crucially — its asymptote $y_\infty$ is an **extrapolation to
$N\to\infty$** from data spanning only a $4\times$ range that has not visibly bent. That is
ill-posed: many $(y_\infty,c,\gamma)$ triples fit four un-bent points almost equally, so
$y_\infty$ is **not identifiable** here. Four points can identify the **local slope and its
sign**, **monotonicity**, and **whether the secants are decreasing** (concavity → bending
toward saturation); they *cannot* identify the asymptote, the true functional form, or
anything past $N=86$.

### 4.2 Why a secant decision rule, not a functional-form claim

Because the form is unidentifiable, **H-02b is pre-registered as a local, form-free
statement**: the last measured doubling still buys more than seed noise,
$\overline{\mathrm{val}}(86)-\mathrm{val}(43)>\sigma_{\text{seed}}$. The instrument is the
**secant slope** between $N_1<N_2$,
$$\mathrm{sec}(N_1,N_2)=\frac{y(N_2)-y(N_1)}{\log_2 N_2-\log_2 N_1}\quad[\text{dB per doubling}],$$
reported for the consecutive pairs (21→43, 43→64, 64→86); a decreasing sequence is
"bending", a flat one is "still log-linear", both **without** committing to a 2- or
3-parameter model. The log-linear fit is drawn only over $[21,86]$; any dashed segment to
200 songs carries the pre-registered caption *"descriptive extrapolation — no data beyond 86
songs"* (Saijo & Bando 2025 motivates non-saturation but does not license extrapolation).

### 4.3 The fit CI is a *parametric* bootstrap

With only four points there is nothing to resample, so the slope CI is a **parametric**
bootstrap: treat each mean $y_i$ as $\hat y_i$ plus Gaussian endpoint noise, draw
$y_i^{*}=y_i+\varepsilon_i^{*}$ with $\varepsilon^{*}\sim\mathcal N(0,\sigma_{\text{seed}})$
per point, refit $b^{*}$ by ordinary least squares on $x=\log_2 N$, repeat $B=10{,}000$
times, and take the central-95 % percentiles of $\{b^{*}\}$. The OLS slope has the closed
form $b=\sum_i(x_i-\bar x)(y_i-\bar y)\big/\sum_i(x_i-\bar x)^2$, vectorized across
replicates. The CI is honestly wide (4 points, 2 parameters); that width **is** the finding's
uncertainty, not a defect.

> **Code.** `singnet/analysis/scaling.py::fit_log2` (OLS + parametric bootstrap, per-point
> `sigma`), `::secant_slopes` (the per-doubling table). Known-slope recovery and
> CI-widens-with-$\sigma$ are in `tests/test_scaling_analysis.py`.

---

## 5. Statistics of the design

### 5.1 Pooled $\sigma_{\text{seed}}$ and why pooling the extremes is conservative

Two cells carry three training seeds — FULL-86 and n21 — giving between-seed sample
variances $s^2_{\text{FULL}}$ and $s^2_{\text{n21}}$. The design's noise band is the pooled
$$\sigma_{\text{seed}}=\sqrt{\tfrac12\big(s^2_{\text{FULL}}+s^2_{\text{n21}}\big)},$$
a $\sqrt{\text{mean of variances}}$ over the two cells (4 dof rather than 2, a steadier
estimate). Pooling the **two extremes** of the data axis is deliberately conservative: the
small-data cell (n21) is plausibly the *noisier* one — fewer songs, more sensitive to init
and to the single nested draw — so including it **inflates** $\sigma_{\text{seed}}$ relative
to a FULL-only estimate, which **raises** every decision threshold ($\Delta>\sigma_{\text{seed}}$,
last-doubling $>\sigma_{\text{seed}}$). Harder thresholds mean fewer false "it matters"
verdicts. The gate G2 escalation (add seeds to `no_remix` only) fires if this
$\sigma_{\text{seed}}$ is too large to resolve plausible deltas.

> **Code.** `singnet/analysis/scaling.py::pooled_seed_sigma` (skips cells with <2 seeds);
> tested against the closed form in `tests/test_scaling_analysis.py`.

### 5.2 One pre-registered contrast, not a multiple-comparison sweep

H-02a is a **single** pre-specified contrast — remix versus the best of the other two,
$\Delta_{\text{remix}}>\max(\Delta_{\text{gain}},\Delta_{\text{flip}})+\sigma_{\text{seed}}$.
Because exactly one comparison carries the falsifiable verdict, **no multiple-comparison
correction is applied** to it: there is no garden of forking paths, no "report the largest of
several Δ". The full Δ table (all three transforms) and the interaction gap are **descriptive**,
reported with CIs and *no* hypothesis test — so they cannot silently become extra comparisons
that inflate the error rate. Had we instead tested all three $\Delta_t$ as separate hypotheses,
a Bonferroni-style correction would be mandatory; the single-contrast design buys the verdict's
validity with pre-registration instead.

### 5.3 The interaction gap and what a non-zero value means

Each LOO delta $\Delta_t=\overline{\mathrm{val}}(\text{FULL})-\mathrm{val}(\text{FULL}\setminus t)$
is a **last-in** contribution: the value lost when $t$ is removed from the *full* recipe. If
the transforms contributed additively, the total recipe value would equal the sum of the
last-in deltas; the deviation is the **interaction gap**
$$G=\Delta_{\text{total}}-\sum_{t\in\{\text{remix,gain,flip}\}}\Delta_t,\qquad
\Delta_{\text{total}}=\overline{\mathrm{val}}(\text{FULL})-\mathrm{val}(\text{NONE}).$$
- $G>0$ (**synergy**): the transforms are complementary — removing one is partly compensated
  by those remaining, so the individual $\Delta_t$ *understate* each transform's worth and the
  recipe is worth more than the sum of its last-in parts.
- $G<0$ (**redundancy**): the transforms overlap — their effects double-count, so the last-in
  $\Delta_t$ *overstate* unique contributions (the expected sign here, since flip is redundant
  by §3.1 and remix/gain both perturb the mixture balance).
- $G\approx0$: approximately additive at this budget.

$G$ is reported descriptively with a CI; it is not a hypothesis. It contextualizes the LOO
table — e.g. a large negative $G$ warns that summing single-transform ablations would
overcount the recipe's value.

> **Code.** `singnet/analysis/scaling.py::loo_table` returns the per-transform $\Delta_t$,
> $\Delta_{\text{total}}$ and the `interaction_gap` row; verified on a synthetic registry in
> `tests/test_scaling_analysis.py::test_loo_table_deltas_and_interaction`.

---

*Cross-reference index.* §1 → `data/musdb_dataset.py::MusdbChunks.__getitem__`,
`data/augment.py` (remix streams); §2 → `MusdbChunks.tracks`/`n_songs` and the registry
`n_songs` column; §3 → `data/augment.py::{random_sign_flip,random_gain}` and the front-end
standardization in `train/loop.py::prepare_batch`; §4 → `analysis/scaling.py::{fit_log2,
secant_slopes}`; §5 → `analysis/scaling.py::{pooled_seed_sigma,loo_table}`. Every claim above
is exercised by `tests/test_augment_switchboard.py` and `tests/test_scaling_analysis.py`
(gate G0). The standalone LaTeX mirror is `theory/theory.tex`.
