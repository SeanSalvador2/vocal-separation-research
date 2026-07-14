# THEORY — Direction 06: what bleeding targets do to a mask model, and why the smallest losses are the cleanest

This note derives the mathematics the code implements for Direction 06. It follows the
MASTER_PLAN §9 outline and, like Directions 01–05, ties every result to a concrete
function path in `singnet/` + `scripts/`. The **crux** is the pair (§3, §5): §3 gives the
*falsifiable null model* — a closed-form clean SI-SDR for a model that perfectly learns
the corrupted target — and §5 gives the *mechanism* by which the cheap defense (trimmed
loss) can work at all under **uniform** corruption. Every number in §3 is machine-checked
against `singnet/analysis/bleed.py` on synthetic stems; the trim arithmetic in §5 is
pinned in `tests/test_trimmed_loss.py`.

Notation. Clean vocal waveform $v$, accompaniment $a=\text{drums}+\text{bass}+\text{other}$,
mixture $x=v+a$. Complex STFTs $V=\mathrm{STFT}(v)$, $A=\mathrm{STFT}(a)$, $X=V+A$. The
compact separator predicts a soft mask $M\in[0,1]^{F\times T}$ and estimates the vocal as
$\widehat V = M\odot|X|\,e^{i\angle X}$ (masked mixture magnitude, **mixture phase**,
Direction 01 §1); $\hat v = \mathrm{iSTFT}(\widehat V)$. The bleed level is
$\varepsilon\in\{0,0.05,0.15,0.30\}$.

> **Code.** Corruption: `singnet/data/corrupt.py` (`StemBleed`, `build_corruption`).
> Prediction line + recovery CI: `singnet/analysis/bleed.py`. Trimmed loss + per-chunk
> loss contract: `singnet/losses/trimmed.py` + `singnet/losses/_base.py`. Telemetry
> threading: `singnet/data/musdb_dataset.py` → `singnet/train/loop.py`. Every §3/§5/§6
> claim is unit-tested (gate G0): `tests/test_bleed_analysis.py`,
> `tests/test_trimmed_loss.py`, `tests/test_corrupt.py`.

---

## 1. Corruption as continuous-target label noise

### 1.1 The redistribution model

Real stems are dirty: microphone bleed leaves accompaniment content in the "vocals"
stem, and SDX'23 built two datasets (`SDXDB23_LabelNoise`, `SDXDB23_Bleeding`) to make
corrupted training data a first-class axis (`research/papers/fabbro2023-sdx23.md`).
Direction 06 uses a **single-parameter** version, applied at the **stem level, at load
time, before augmentation**, to **training targets only**:

$$
\boxed{\ \tilde v = v + \varepsilon\,a,\qquad \tilde a = (1-\varepsilon)\,a\ }
$$

This is a *redistribution*: a fraction $\varepsilon$ of the accompaniment is moved out of
$\tilde a$ and into the vocal target $\tilde v$. Two properties hold by construction.

**Mixture invariance.** $\tilde v + \tilde a = v + \varepsilon a + (1-\varepsilon)a = v+a
= x$. The corrupted stems still sum to the **true mixture**, so the network input $x$ is
*untouched* — only the *supervised target* is corrupted. This is the defining property of
real bleed (the mic captures the true acoustic sum; the mislabeling is in how it is split
into stems) and of `SDXDB23_Bleeding`'s construction. The model is trained on the pair
$(x,\ \tilde v)$, i.e. to learn $x\mapsto v+\varepsilon a$ — to **leave $\varepsilon a$ in
its vocal estimate**.

**Determinism.** $\tilde v,\tilde a$ are a pure affine map of the stem pair — **no RNG** —
so wiring corruption into the loader cannot perturb any augmentation stream
(`(seed,transform,step)`-keyed, Direction 02 §3.5). This is what keeps the corrupted arm
a *controlled* change: only the target's content differs; the data order, remix partner,
gain, and flip are bit-identical to the clean arm (unit-tested,
`tests/test_corrupt.py::test_kept_stream_equality_with_augmentation`).

### 1.2 Relation to `SDXDB23_Bleeding`, honestly

Ours is a controlled *simplification*: constant $\varepsilon$ (SDX'23 varies per song),
accompaniment→vocal direction only (SDX'23 is all-directions), and a single scalar knob
(SDX'23 is realistic-but-uncontrolled). We trade realism for a **monotone parameter** that
produces a *dose–response curve*, and we state this in every artifact. The
LabelNoise mode (wrong stem *grouping*) is a different corruption Direction 06 does not
simulate.

### 1.3 The loudness decision (pre-registered)

$\tilde v$'s energy grows with $\varepsilon$ ($\lVert\tilde v\rVert^2 = \lVert v\rVert^2 +
2\varepsilon\langle v,a\rangle + \varepsilon^2\lVert a\rVert^2$). Real bleed **adds
energy**, and we do **not** renormalize: (i) the per-chunk input standardization is
mixture-side and unaffected; (ii) the random-gain augmentation already randomizes source
levels; (iii) renormalizing would silently convert "bleed" into "bleed + attenuation,"
confounding $\varepsilon$ with a gain change. The prediction line (§3) uses the *same*
un-normalized construction, so the measured-vs-predicted comparison is internally
consistent. $\qquad\blacksquare$

> **Code.** `StemBleed.__call__` computes $\tilde v,\tilde a$ in float64 then casts, so the
> mixture-invariance residual is at float32 round-off, not accumulated
> (`tests/test_corrupt.py::test_mixture_invariance_exact`).

---

## 2. What the corrupted-optimal model does (L1-magnitude training)

The project default loss is `l1mag`: $\mathcal L = \operatorname{mean}\big|\,M\odot|X| -
|\tilde V|\,\big|$, with the **corrupted** target magnitude $|\tilde V| = |V+\varepsilon
A|$ and the untouched mixture magnitude $|X| = |V+A|$.

### 2.1 The per-bin optimum

The L1 objective is separable across time–frequency bins (the mask has one free scalar per
bin). At a bin with mixture magnitude $|X|>0$ and corrupted-target magnitude $t=|V+\varepsilon
A|$, the model minimizes $|M|X| - t|$ over the admissible $M\in[0,1]$. The unconstrained
minimizer is $M|X| = t$, i.e. $M = t/|X|$; the box constraint clips it:

$$
\boxed{\ M^\star = \operatorname{clip}\!\Big(\frac{|V+\varepsilon A|}{|V+A|},\,0,\,1\Big)\ }
\qquad\Longrightarrow\qquad
\big(M^\star|X|\big) = \min\big(|V+\varepsilon A|,\ |V+A|\big).
$$

Because $\varepsilon<1$, the corrupted target carries *less* accompaniment than the
mixture, so $|V+\varepsilon A|\le|V+A|$ on the vast majority of bins (the clip is inactive)
and the optimum **reproduces the corrupted-target magnitude exactly**:
$M^\star\odot|X| = |V+\varepsilon A|$. The clip binds only where the target magnitude would
exceed the mixture magnitude — the mask cannot *amplify* — a rare, small-energy event that
we note but do not lean on.

### 2.2 The perfectly-fit model leaks $\varepsilon a$ — with a phase caveat

Expanding for the first-order magnitude perturbation (the honest phase bookkeeping),

$$
|V+\varepsilon A| = |V| + \varepsilon\,\underbrace{\mathrm{Re}\!\big(A\,e^{-i\angle V}\big)}_{\displaystyle \Delta}
    + O(\varepsilon^2),
\qquad |\Delta|\le|A|,
$$

so the L1-optimal masked magnitude is $|V| + \varepsilon\Delta + O(\varepsilon^2)$: the
model is trained to output the clean vocal magnitude **plus a systematic $\varepsilon$-scaled
accompaniment term** $\Delta$ (the projection of $A$ onto $V$'s phase). It leaks
$\varepsilon a$ into the magnitude *by construction* — the residual against the clean
target $|V|$ is $\varepsilon\Delta \ne 0$, non-vanishing and monotone in $\varepsilon$.

**Phase caveat (stated, not swept under the rug).** The mask model wears the **mixture
phase** $\angle X=\angle(V+A)$, not the corrupted-target phase $\angle(V+\varepsilon A)$.
So even the magnitude-perfect estimate $\widehat V = |V+\varepsilon A|\,e^{i\angle(V+A)}$ is
**not** the complex $V+\varepsilon A$: its magnitude matches but its phase is the mixture's.
Two consequences: (i) the idealized time-domain estimator $\hat v = v+\varepsilon a$ used as
the null model in §3 is the *magnitude-and-phase* ideal, so the *magnitude-only* mask model
is upper-bounded by it (a real mask model also pays the mixture-phase penalty, which is
$\varepsilon$-independent and already present at $\varepsilon=0$); (ii) the clean-$\varepsilon=0$
baseline already sits below a phase-perfect oracle, so the **degradation curve**
$s(0)-s(\varepsilon)$ — a *difference* — cancels the shared mixture-phase term and isolates
the bleed cost. This is why the dose–response *curve* (not the absolute SI-SDR) is the clean
observable. $\qquad\blacksquare$

> **Code.** `singnet/losses/l1mag.py` (`M\odot|X| - |target|`); the masked-magnitude helper
> `singnet/losses/_base.py::masked_magnitude`. The mixture-phase reconstruction is
> `singnet/audio/stft.py::apply_mask`.

---

## 3. The prediction line $P(\varepsilon)$ (the falsifiable null)

The idealized model that perfectly learns the corrupted conditional outputs $\hat v =
\tilde v = v + \varepsilon a$. Its **clean** SI-SDR against the true $v$ is computable in
closed form — the score every un-defended bleed arm is compared to.

### 3.1 Exact projection form

SI-SDR (Le Roux et al., 2019; `singnet/metrics/si_sdr.py`) with estimate $\hat s = v +
\varepsilon a$ and reference $s = v$. The scale factor is

$$
\alpha = \frac{\langle \hat s, s\rangle}{\lVert s\rVert^2}
       = \frac{\langle v+\varepsilon a,\,v\rangle}{\lVert v\rVert^2}
       = 1 + \varepsilon\,\frac{\langle a,v\rangle}{\lVert v\rVert^2}
       = 1 + \varepsilon\,\rho\,\frac{\lVert a\rVert}{\lVert v\rVert},
$$

writing $\rho = \langle a,v\rangle/(\lVert a\rVert\lVert v\rVert)$ (the stems' cosine). The
target is $\alpha v$ and the noise is

$$
\alpha v - \hat s = (\alpha-1)v - \varepsilon a
   = \varepsilon\Big(\frac{\langle a,v\rangle}{\lVert v\rVert^2}v - a\Big)
   = -\varepsilon\,a_\perp,\qquad a_\perp = a - \frac{\langle a,v\rangle}{\lVert v\rVert^2}v,
$$

i.e. **only the component of $a$ orthogonal to $v$ survives as SI-SDR noise** — the
$v$-parallel bleed is absorbed by the scale-invariant $\alpha$. With
$\lVert a_\perp\rVert^2 = \lVert a\rVert^2(1-\rho^2)$,

$$
\boxed{\ P_{\text{track}}(\varepsilon) = 10\log_{10}
   \frac{\alpha^2\,\lVert v\rVert^2}{\varepsilon^2\,\lVert a\rVert^2\,(1-\rho^2)}\ }
\qquad(\text{exact; }v\not\parallel a).
$$

This is the `si_sdr_exact` column of `prediction_line`, computed numerically from the stems
(so no small-$\varepsilon$ approximation is made).

### 3.2 The orthogonality condition and the $-20\log_{10}\varepsilon$ slope

When $v\perp a$ ($\rho=0$): $\alpha=1$ and the box collapses to the closed form

$$
P_{\text{track}}(\varepsilon) = 10\log_{10}\frac{\lVert v\rVert^2}{\varepsilon^2\lVert a\rVert^2}
   = \underbrace{-20\log_{10}\varepsilon}_{\text{universal slope}}
   \;+\;\underbrace{10\log_{10}\frac{\lVert v\rVert^2}{\lVert a\rVert^2}}_{\text{per-track anchor}}.
$$

The prediction line is a **$-20\log_{10}\varepsilon$ line** — every halving of $\varepsilon$
buys $\approx 6.0$ dB — anchored per track by its vocal/accompaniment energy ratio. This is
the `si_sdr_orth` column. `prediction_line` returns **both** columns; their difference
`ortho_gap_db` $= P_{\text{exact}} - P_{\text{orth}}$ measures how far the stems depart from
orthogonality (positive when correlation makes the true score *more* forgiving than the
$\perp$ approximation, via the $\alpha^2/(1-\rho^2)$ factor).

**Machine-checked (`tests/test_bleed_analysis.py`).** Orthogonal $v=[1,1,1,1]$,
$a=[1,-1,1,-1]$ ($\rho=0$, equal energy): $P(0.1)=-20\log_{10}0.1=20.0$ dB exactly, and
`si_sdr_exact == si_sdr_orth`. Non-orthogonal $v=[1,0]$, $a=[1,1]$, $\varepsilon=0.5$:
$\alpha=1.5$, $\rho^2=\tfrac12$, so $P_{\text{exact}}=10\log_{10}\!\frac{2.25}{0.25}=10\log_{10}9=9.54$
dB while $P_{\text{orth}}=10\log_{10}2=3.01$ dB — a $+6.5$ dB gap the code reproduces to
$10^{-6}$.

### 3.3 What measured-above / below means (the scientific payload)

$P(\varepsilon)$ = the per-ε mean of $P_{\text{track}}$ over the 14 validation tracks
(`predicted_curve`). The **gap** between the *measured* dose–response curve $s(\varepsilon)$
and $P(\varepsilon)$ — not the bare slope — is the finding (`measured_vs_predicted`):

| Reading | Interpretation |
|---|---|
| $s(\varepsilon)\approx P(\varepsilon)$ (±$\sigma_{\text{seed}}$) | the model faithfully learns the corrupted conditional; **no implicit denoising** — data quality is a hard, quantified constraint |
| $s(\varepsilon) > P(\varepsilon)$ | **implicit robustness**: architecture/loss/augmentation partially reject the bleed (the model beats the "leak exactly $\varepsilon a$" estimator) |
| $s(\varepsilon) < P(\varepsilon)$ | corruption **additionally destabilizes optimization** — damage beyond the information limit (check grad norms; robust losses become the follow-up) |

If H-06a is *refuted* (curve flat), $P(\varepsilon)$ must sit **well below** the measured
curve, turning "flat" into a **positive, quantified robustness claim** rather than a null.
The prediction line is computed from **clean** stems only — independent of any run output or
corrupted/eval target — which is exactly what the §7 G3 audit needs.

> **Code.** `prediction_line`, `predicted_curve`, `measured_vs_predicted` in
> `singnet/analysis/bleed.py`; the exact-vs-orthogonal identity and the projection
> definition are pinned in `tests/test_bleed_analysis.py`.

---

## 4. Trimmed estimators, and an honest comparison to our setting

### 4.1 ITLM and the small-loss trick

**ITLM** (Shen & Sanghavi, arXiv 1810.11874; `research/papers/noisy-label-canon.md` §B):
alternate (i) **select the $\alpha$-fraction of samples with the lowest current loss** and
(ii) **retrain on only those**. With per-sample losses $\ell_i(\theta)$ and keep-fraction
$\alpha=1-q$, $S_t=\{\text{indices of the }\lceil\alpha N\rceil\text{ smallest }\ell_i\}$,
$\theta_{t+1}=\arg\min_\theta\sum_{i\in S_t}\ell_i$. It **provably** recovers the ground
truth (linear convergence) in generalized linear models **when a fraction of samples are
arbitrarily corrupted and the rest are clean**. The premise is *sample-level* cleanliness:
clean samples exist and are identifiable by low loss.

**Arpit** (arXiv 1706.05394): DNNs learn simple/general patterns **first** and memorize
noise **later**, so *early* in training high per-example loss $\approx$ likely-corrupted.
This is the assumption every small-loss method rests on — the signal is present before
memorization erases it.

### 4.2 Why our setting is a *stress test*, not the classic one

Under Direction 06's **uniform** bleed, **every** target is corrupted by the same
$\varepsilon$ — there are **no clean samples**. ITLM's "recover the clean subset" premise
does *not* hold. So a null result is entirely plausible and is pre-registered as an honest
boundary for the trick (MASTER_PLAN §11). What trimming *can* still do is select, among
uniformly-corrupted chunks, those whose corruption *happens to be small* — a **different**
selection principle ("less-corrupted vs more-corrupted," not "clean vs corrupted"). Whether
that principle has teeth depends on whether per-chunk loss correlates with per-chunk
corruption magnitude. §5 derives that it does, through accompaniment energy — and the
telemetry (§5.3) decides empirically. This reframing (uniform, not sample-level) is the
scientific honesty of the direction: we run the classic trick **outside its proven regime**
and report the mechanism, positive or null.

> **Code.** `singnet/losses/trimmed.py` implements the ITLM step as a per-batch, per-chunk
> loss wrapper; `keep_count = ⌈(1-q)B⌉` (MASTER_PLAN §3.2). `q=0` reduces to the base loss
> exactly (`tests/test_trimmed_loss.py::test_q0_equals_base_loss`).

---

## 5. The selection mechanism under uniform bleed (the crux for H-06b)

### 5.1 Per-chunk irreducible loss grows with accompaniment energy

Fix a converged, *general* model $f$ — one that has learned the clean vocal magnitude
structure $|V|$ (Arpit: the general pattern learned first) but not memorized chunk-specific
noise. Its per-chunk L1-mag loss on corrupted chunk $i$ (target $\tilde v_i=v_i+\varepsilon
a_i$) is, using the §2.2 expansion,

$$
\ell_i(f) = \operatorname*{mean}_{TF}\big|\,f(|X_i|) - |V_i+\varepsilon A_i|\,\big|
   \;\approx\; \operatorname*{mean}_{TF}\big|\,|V_i| - (|V_i| + \varepsilon\Delta_i)\,\big|
   = \varepsilon\,\operatorname*{mean}_{TF}|\Delta_i|,
$$

where $\Delta_i=\mathrm{Re}(A_i e^{-i\angle V_i})$ is the accompaniment component the
corrupted target injects. By Cauchy–Schwarz $\operatorname{mean}|\Delta_i|$ is bounded by,
and empirically tracks, the chunk's **accompaniment energy**
$\langle a\rangle_i \equiv \lVert a_i\rVert^2/N$. Hence

$$
\boxed{\ \ell_i(f) \;\approx\; \ell_{\text{clean}} + c\,\varepsilon\,g(\langle a\rangle_i),
   \qquad g\ \text{increasing}\ }
$$

the per-chunk **irreducible** loss is monotone in the chunk's accompaniment energy: a chunk
whose vocals sit over a loud accompaniment carries a louder $\varepsilon a$ bleed in its
target, which a clean-fitting model cannot reproduce without overfitting it.

### 5.2 Trimming = curriculum-by-cleanliness

Since $\ell_i$ increases with $\langle a\rangle_i$, keeping the $\lceil(1-q)B\rceil$
**lowest-loss** chunks preferentially **keeps low-$\langle a\rangle$ chunks and drops
high-$\langle a\rangle$ chunks**. Low-$\langle a\rangle$ chunks are exactly those whose
$\varepsilon a$ bleed is *quietest* — the **effectively cleanest targets**, even though no
target is fully clean. Trimming therefore acts as a **curriculum by cleanliness**: each step
trains on the least-contaminated slice of an entirely-contaminated dataset. This is the only
route by which the small-loss trick can transfer to uniform corruption — and it is a
genuine, testable prediction, not a hope.

### 5.3 The testable telemetry signature

The mechanism makes a falsifiable prediction about *which* chunks get trimmed:

$$
\boxed{\ \overline{\langle a\rangle}_{\text{kept}} \;<\; \overline{\langle a\rangle}_{\text{dropped}}\ }
$$

the kept chunks' mean accompaniment energy is **below** the dropped chunks'. The train loop
logs, every 500 steps, the accompaniment-energy distribution of kept vs dropped chunks. The
covariate is the **raw same-track accompaniment energy at the vocal window** —
$\langle a\rangle_i=\lVert a_{\text{voc},i}\rVert^2/N$ — computed from the *uncorrupted,
un-augmented* stem (it is the $\varepsilon$- and augmentation-invariant cleanliness covariate
that sets the bleed magnitude; DEVIATIONS 2026-07-14). If the telemetry shows **no**
energy separation ($\overline{\langle a\rangle}_{\text{kept}}\approx\overline{\langle
a\rangle}_{\text{dropped}}$), the selection signal never existed ⇒ H-06b is refuted *with a
mechanism* (the trick failed because uniform corruption offers no cleanliness gradient to
exploit). This is why the telemetry — not just the recovery $\rho$ — is a pre-registered
secondary.

> **Code.** The per-chunk energy is threaded dataset → `batch["acc_energy"]` → the training
> step → `TrimmedLoss(chunk_energy=…)` → `aux["kept_energy_mean"]`/`["dropped_energy_mean"]`
> → the every-500-steps `trim_energy_stats.csv` (`singnet/data/musdb_dataset.py`,
> `singnet/train/loop.py`). The kept-vs-dropped energy split with loss-correlated energy is
> pinned in `tests/test_trimmed_loss.py::test_energy_telemetry_split_by_kept_dropped`.

---

## 6. Statistics of the design

### 6.1 Pooled between-seed noise

$\sigma_{\text{seed}}$ is pooled over the three 3-seed cells (clean $\varepsilon=0$, bleed
$\varepsilon=0.30$, trim@$0.30$): $\sigma_{\text{seed}} = \sqrt{\tfrac13\sum_c
\hat\sigma_c^2}$ with $\hat\sigma_c^2$ the sample variance of cell $c$'s three seed scores.
It is the ruler for every decision threshold (H-06a: $s(0)-s(0.30)>\max(\sigma_{\text{seed}},
0.5\text{ dB})$; H-06b: $s_{\text{trim}}-s_{\text{bleed}}>\sigma_{\text{seed}}$).

> **Code.** `singnet/analysis/scaling.py::pooled_seed_sigma` (reused verbatim).

### 6.2 Recovery fraction $\rho$ and its delta-method CI

Under H-06a support, the recovered share of the degradation is

$$
\rho = \frac{s_{\text{trim}} - s_{\text{bleed}}}{s_{\text{clean}} - s_{\text{bleed}}}
     = \frac{u}{d}.
$$

Treating the three 3-seed cell means as independent with per-mean variances
$\hat\sigma_c^2/n$ ($n=3$), the **delta method** linearizes $\rho=u/d$ ($u=s_{\text{trim}}-
s_{\text{bleed}}$, $d=s_{\text{clean}}-s_{\text{bleed}}$):

$$
\widehat{\mathrm{Var}}(\rho)\approx
   \Big(\tfrac{1}{d}\Big)^2\tfrac{\hat\sigma_{\text{trim}}^2}{n}
 + \Big(\tfrac{u}{d^2}\Big)^2\tfrac{\hat\sigma_{\text{clean}}^2}{n}
 + \Big(\tfrac{u-d}{d^2}\Big)^2\tfrac{\hat\sigma_{\text{bleed}}^2}{n},
\qquad \mathrm{SE}(\rho)=\sqrt{\widehat{\mathrm{Var}}(\rho)},
$$

from $\partial\rho/\partial s_{\text{trim}}=1/d$, $\partial\rho/\partial s_{\text{clean}}=
-u/d^2$, $\partial\rho/\partial s_{\text{bleed}}=(u-d)/d^2$. The CI is $\rho\pm z\,\mathrm{SE}$
($z=1.96$). When $d\le0$ (the model was **not** hurt — H-06a unsupported) $\rho$ is **not
evaluable** and returned as `nan` (MASTER_PLAN §2), never fabricated.

> **Code.** `singnet/analysis/bleed.py::recovery_fraction` → `RecoveryResult`; the point
> estimate, the zero-variance CI collapse, and the not-evaluable branch are pinned in
> `tests/test_bleed_analysis.py`.

### 6.3 The two pre-registered paired tests, and multiple-comparison posture

The single test pass (§7) scores the three 3-seed cells' best checkpoints (`clean`,
`bleed30`, `trim30`) on the 50 clean test tracks. **Exactly two** pre-registered paired
contrasts are evaluated, per-track, with a bootstrap 95 % CI + Wilcoxon signed-rank:

1. $\textbf{bleed30} - \textbf{clean}$ — does bleed hurt (H-06a)?
2. $\textbf{trim30} - \textbf{bleed30}$ — does trimming recover (H-06b)?

Because only two primary pairs are pre-registered (no fishing over the single-seed curve/
control arms, which are validation-scoped and never tested), the multiple-comparison burden
is minimal: we report both CIs in full rather than applying a family-wise correction that a
two-hypothesis design does not warrant, and we state this posture up front.

---

## 7. Standalone LaTeX mirror

`theory/theory.tex` compiles the same content standalone (`article` + `amsmath`/`amssymb`,
no external figures). Compilation is untested in this CPU/no-TeX environment (noted there),
but the source is self-contained.

---

*Cross-reference index.* §1 → `data/corrupt.py::{StemBleed, build_corruption}` (+
`tests/test_corrupt.py`); §2 → `losses/l1mag.py` + `losses/_base.py::masked_magnitude`,
`audio/stft.py::apply_mask`; §3 → `analysis/bleed.py::{prediction_line, predicted_curve,
measured_vs_predicted}` (exact values machine-checked in `tests/test_bleed_analysis.py`);
§4 → `research/papers/noisy-label-canon.md` + `losses/trimmed.py`; §5 →
`losses/trimmed.py` + the `acc_energy` threading in `data/musdb_dataset.py` →
`train/loop.py` (telemetry pinned in `tests/test_trimmed_loss.py`,
`tests/test_train_d06.py`); §6 → `analysis/bleed.py::recovery_fraction` +
`analysis/scaling.py::pooled_seed_sigma`. Every §3 number is machine-checked against the
built functions.
