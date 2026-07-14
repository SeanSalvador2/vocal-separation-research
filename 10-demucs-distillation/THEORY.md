# THEORY — Direction 10: what a teacher's labels can and cannot buy a compact student

This note derives the mathematics Direction 10 rests on. It follows MASTER_PLAN §9 and, like
Directions 01–08, ties every result to a concrete function path in `singnet/` + `scripts/`.
The **crux** is the quartet (§1, §2, §4, §5): §1 derives *why the distillation temperature has
no regression analogue* and *why hard pseudo-labels with the unchanged loss are the honest way
to isolate the data effect*; §2 is the *formal bridge to Direction 06* — teacher error as
**structured** target corruption, and exactly what D06's dose–response chart does and does not
license here; §4 defines the **gap-closure estimand** Ĉ, its delta-method CI, and — as loudly —
what it does **not** claim; §5 is the **license formalism**, whose ND-exclusion is the
load-bearing legal call the whole pipeline turns on.

Notation. True (latent) vocal $v$, accompaniment $a$, mixture $x=v+a$. The compact **student**
(SingNet-C1) predicts a soft mask $M\in[0,1]^{F\times T}$ and estimates $\hat v_S =
\mathrm{iSTFT}(M\odot|X|\,e^{i\angle X})$ (Direction 01 §1). The **teacher** (`htdemucs`) is a
hybrid-waveform model outputting $\hat v_T$. On an unlabeled FMA clip the student's target is
the teacher's output $\hat v_T$, used as an **ordinary `l1mag` target**; the accompaniment
target is the exact 2-stem residual $\hat a_T := x - \hat v_T$ (§3.2). The **teacher residual**
is $r := \hat v_T - v$. On the MUSDB test set: $s_{\text{base}}$/$s_{\text{mix}}$ are the mean
vocal SI-SDR of `musdb_only`/`mixed` (3 seeds each), $s_T$ the teacher's own score; the **gap**
$G=s_T-s_{\text{base}}$ and **closure** $\hat C=(s_{\text{mix}}-s_{\text{base}})/G$. The FMA
pool probability is $p_{\text{FMA}}$.

> **Code.** License audit: `scripts/prepare_fma.py::{canonicalize_license, filter_licenses,
> screen_sample}`. Teacher post-processing: `scripts/teacher_label.py::{two_stem_consistency,
> consistency_residual_db, vocal_activity_ratio, screen_by_activity}`. Pools + guard:
> `singnet/data/pseudo.py::{PseudoLabeledShards, MixedPools, assert_not_under_musdb}`. The
> D06 bridge machinery: `singnet/analysis/bleed.py::{prediction_line, recovery_fraction}`,
> `singnet/losses/trimmed.py`. Every §1/§2/§5 code claim is unit-tested (gate G0):
> `tests/test_prepare_fma.py`, `tests/test_teacher_label.py`, `tests/test_pseudo.py`.

---

## 1. Distillation & pseudo-labeling, honestly taxonomized

### 1.1 Hinton KD and its temperature

Classification KD (Hinton et al. 2015, `research/papers/hinton2015-kd.md`) softens teacher
logits $z_i$ into a categorical target and trains the student's softened predictions against it:

$$
p_i(T)=\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)},\qquad
\mathcal L=(1-\lambda)\,\mathcal L_{\text{hard}}+\lambda\,T^2\,\mathcal L_{\text{soft}}\!\big(p^{S}(T),p^{T}(T)\big).
$$

Raising $T$ flattens the categorical distribution, exposing the teacher's **relative**
confidences among the non-target classes ("dark knowledge"). The $T^2$ factor rescales the
soft-loss gradient so it stays commensurate with the hard-loss gradient as $T$ varies.

### 1.2 Why the temperature has **no** regression analogue (derived, not asserted)

The temperature mechanism is intrinsically a property of the **softmax on a finite simplex**.
Two facts make it undefined for dense separation:

1. **There is no partition function to soften.** The teacher output $\hat v_T\in\mathbb R^{F\times
   T}$ (a magnitude/waveform) is not a normalized distribution over classes; there is no
   $\sum_j\exp(z_j/T)$ and no inter-class mass to redistribute. Temperature acts *before* a
   softmax; with no softmax, $T$ has nothing to act on.
2. **No monotone reparametrization does what $T$ does.** $T$ simultaneously (a) preserves the
   $\arg\max$ (the decision) and (b) moves probability mass between classes. In regression the
   "target" is the value $\hat v_T$ itself; any monotone map $\phi_T(\hat v_T)$ that changed the
   target's *scale* would change the regressed quantity — there is no mass to move while fixing a
   decision, because there is no decision and no mass.

Formally: KD's soft target is $p^T(T)=\mathrm{softmax}(z/T)$; as $T\!\to\!\infty$ it tends to the
uniform distribution and as $T\!\to\!0$ to the one-hot $\arg\max$. The regression target
$\hat v_T$ has **no such one-parameter family** with fixed endpoints — the only "endpoints" are
the teacher's raw output and (its own) $\arg\max$, which coincide. Hence:

$$
\boxed{\ \text{What survives KD in the regression setting is only the teacher's output as a
target: self-training / pseudo-labeling.}\ }
$$

The **closest cousins** to temperature (marked as **analogy, not identity**; LITERATURE §5) are
(i) the continuous teacher output itself, already "softer" than an absent GT on unlabeled audio;
(ii) spectral dynamic-range compression (a $\log$/power-law on magnitudes) that de-emphasizes
loud peaks — but this reshapes the *loss geometry*, not a class distribution, so it is **not**
Hinton's mechanism; (iii) the $\lambda$ soft/hard mix, which *does* transfer (it is just the
mixed-data weighting, realized here as the pool ratio $p_{\text{FMA}}$, §4.1).

### 1.3 The data-effect-isolation argument for hard pseudo-labels

We deliberately do **not** enter the richer distillation design space — feature matching,
mask-space KD, waveform-L1 KD (the LITERATURE §5 candidate table). The reason is an
identifiability argument, not timidity. The three arms differ in **exactly one** factor, the
training-data source:

$$
\underbrace{\text{arch}}_{\text{fixed}},\ \underbrace{\mathcal L_{\ell 1\text{mag}}}_{\text{fixed}},\
\underbrace{\text{aug recipe}}_{\text{fixed}},\ \underbrace{\text{budget}}_{\text{fixed}}\quad\Longrightarrow\quad
s_{\text{mix}}-s_{\text{base}}\ \text{is attributable to the DATA.}
$$

Pseudo-labels enter as **hard** targets: the teacher's *point* output $\hat v_T$ is used as the
ordinary `l1mag` regression target — the regression analogue of a hard label (a single value,
not an ensemble/distribution over sources). A distillation-specific objective (say, feature
matching) would add a *second* changed factor, and $s_{\text{mix}}-s_{\text{base}}$ would
confound "more data" with "new loss" — the effect would be **unidentifiable**. Isolating the
data effect first is the prerequisite for any later claim that a fancier objective *adds*
something. $\qquad\blacksquare$

> **Code.** Hard pseudo-labels are literally the `l1mag` target on the FMA pool: the pool's
> `MusdbChunks` yields `{mixture, vocals=\hat v_T, accompaniment=\hat a_T}` and the loop's loss
> call is byte-identical to a MUSDB chunk (`singnet/train/loop.py`; `MixedPools.__getitem__`).

---

## 2. Teacher error as structured corruption — the Direction-06 bridge

### 2.1 The residual is additive target corruption

On an FMA clip the student is trained on the pair $(x,\ \hat v_T)$ with $\hat v_T = v + r$,
$r=\hat v_T-v$. This is **exactly** the shape of Direction 06's corrupted target $\tilde v =
v+\varepsilon a$: *true source + additive corruption*. So the D06 result — an L1-mag model that
perfectly fits a corrupted target **reproduces the corruption** (D06 THEORY §2.2, the model
learns to leave $\varepsilon a$ in its estimate) — transfers in kind: **a student that fits
$\hat v_T$ learns to reproduce $r$, i.e. inherits the teacher's error** (its vocal↔other
bleeding, its over-suppression). This is the mechanism behind the pre-registered negative-transfer
branch.

### 2.2 But $r$ is *structured*, and differs from $\varepsilon$-bleed in three ways

D06's $\varepsilon a$ is a benign caricature; the teacher residual is not:

| Property | D06 $\varepsilon$-bleed $\tilde v = v+\varepsilon a$ | Teacher residual $r=\hat v_T-v$ |
|---|---|---|
| **Content-correlation** | fixed multiple of the *same-track* $a$ | the teacher's *error*, correlated with content (hard passages, dense mixes) — not a multiple of $a$ |
| **Level** | constant $\varepsilon$ across all chunks | varies with clip difficulty / teacher confidence |
| **Direction** | one-way: accompaniment **into** vocals only | two-way: leakage **in** ($+a$-like) *and* vocal removal **out** ($-v$-like, over-suppression) |

Decompose $r$ in the $\{v,a\}$-informed basis: $r = \underbrace{\alpha\,a}_{\text{leak-in}} +
\underbrace{\beta\,v}_{\text{suppress-out}} + r_\perp$ with $\alpha,\beta$ content-dependent and
signed, and $r_\perp$ the part explained by neither stem. D06's $\varepsilon a$ is the special
case $\alpha=\varepsilon,\ \beta=0,\ r_\perp=0$ with $\alpha$ **constant**. The extra structure
($\beta\neq0$, content-varying $\alpha$, $r_\perp\neq0$) is precisely what makes teacher error a
*harder* corruption than uniform bleed.

### 2.3 What D06's dose–response chart does — and does not — license

D06's **prediction line** $P(\varepsilon)$ (bleed THEORY §3) is the *closed-form clean SI-SDR* of a
model that perfectly learns $\tilde v = v+\varepsilon a$, valid because the corruption is a
**known** multiple of $a$. Teacher error is **not** a known multiple, so:

- **It does NOT** give a numerical prediction for $s_{\text{mix}}$. The $-20\log_{10}\varepsilon$
  slope rests on $\tilde v-v=\varepsilon a$ with the orthogonal component $a_\perp$ as the sole
  SI-SDR noise; with $r=\alpha a+\beta v+r_\perp$ (content-varying, two-way) that derivation's
  premises fail.
- **It DOES** license (i) the **qualitative** mechanism of §2.1 (fit-the-target ⇒ inherit-the-error),
  and (ii) a **coarse magnitude anchor**: matching the residual's orthogonal energy to an
  *effective* bleed, $\varepsilon_{\text{eff}}^2\lVert a_\perp\rVert^2 \approx \lVert r_\perp\rVert^2$,
  places the corrupted-target cost on D06's curve as an **order-of-magnitude** guide only — flagged
  as analogy, never a fitted prediction.

### 2.4 The mixed arm re-enters ITLM's proven regime — and what `mixed_trim`'s telemetry shows

The decisive difference from D06: under **uniform** bleed *every* target is corrupted, so ITLM's
"a clean subset exists" premise fails (D06 ran the small-loss trick *outside* its proven regime).
Under **mixed** training a fraction $1-p_{\text{FMA}}$ of each batch is **real MUSDB labels —
genuinely clean** — so the clean subset exists and is *pool-identifiable*. The trimmed loss
(reused verbatim from D06 via config, `mixed_trim`) keeps the lowest-loss $\lceil(1-q)B\rceil$
chunks; if teacher error behaves like bleed, the corrupted **FMA** chunks carry the higher
irreducible loss (their targets contain the un-fittable $r$), so trimming preferentially **drops
FMA chunks and keeps MUSDB chunks** — the small-loss trick **inside** its regime.

$$
\boxed{\ \text{If teacher error behaves like bleed: } \overline{\text{pool}}_{\text{dropped}}
\ \text{is FMA-enriched, and } \overline{\langle a\rangle}_{\text{kept}}<\overline{\langle a\rangle}_{\text{dropped}}.\ }
$$

The falsifiable telemetry signature: the D06 trim loop already logs kept/dropped indices +
per-chunk accompaniment energy (`trim_energy_stats.csv`); `MixedPools` threads the pool identity
(`pool_is_fma`) so the dropped set's **FMA fraction** is observable. If the dropped set is *not*
FMA-enriched (teacher error is small, the student fits $\hat v_T$ fine), trimming buys nothing
and `mixed_trim`$\approx$`mixed` — the honest null for the defense probe, with a mechanism.

> **Code.** The corrupted-target-⇒-inherit result: `singnet/analysis/bleed.py` (D06). The
> transferred defense: `singnet/losses/trimmed.py::TrimmedLoss` (unchanged); the pool tag it
> would rank against: `singnet/data/pseudo.py::MixedPools` (`pool_is_fma`).

---

## 3. Domain shift and why within-pool remix is mandatory

### 3.1 Two shifts, not one

FMA (CC indie, 30-s clips, diverse production) differs from MUSDB (full studio tracks) in **both**
directions of the supervised pair: a **covariate shift** on the mixtures $x$ (genre / loudness /
production) *and* a **label shift** via teacher error (§2), because a teacher that finds FMA
partly out-of-distribution produces noisier $\hat v_T$ there. The two compound: the student learns
$x_{\text{FMA}}\mapsto \hat v_T = v+r$ under both a shifted input **and** a shifted target, then is
tested on MUSDB — the pre-registered domain-mismatch risk (partial/null branches, §12).

### 3.2 Within-pool remix preserves each pool's mixture law

Remix forms a mixture from vocals of clip $i$ and accompaniment of clip $j$. **Cross-pool** remix
(MUSDB vocals $\times$ FMA accompaniment) would synthesize a **chimeric** mixture belonging to the
statistics of *neither* domain — a pure artifact with no test-time analogue. **Within-pool** remix
keeps $i,j\in P$ (one pool), so the remixed mixture $x' = g_v v_i + g_a a_j$ is a draw from pool
$P$'s own product-of-marginals (the biggest small-data lever, D02 §H-02a) **without domain
crossing**: the FMA pool's remixes stay FMA-like, MUSDB's stay MUSDB-like. This is why `MixedPools`
gives each pool its **own** `AugmentPipeline` over its **own** track list — cross-pool remix cannot
occur *by construction* (`tests/test_pseudo.py::test_within_pool_remix_by_construction`), not by a
runtime check.

### 3.3 The 30-s-clip boundary effect on 6-s chunking (negligible, quantified)

A $30$-s clip admits $30-6=24$ s of valid 6-s chunk starts ($80\%$ of the clip); only the final
$6$ s cannot begin a chunk, and no chunk crosses a clip boundary (clips are independent files). The
sole real effect is **less within-clip diversity**: a $\sim4$-min MUSDB track offers
$\approx234$ distinct 1-s-grid starts vs $\approx24$ for a 30-s clip ($\sim10\times$ fewer per
item). But the pool has $800$ FMA clips vs $86$ MUSDB tracks, so **total** distinct-chunk counts
are comparable — $800\times24\approx1.9\times10^4$ (FMA) vs $86\times234\approx2.0\times10^4$
(MUSDB): the same diversity delivered as *more clips $\times$ fewer chunks each*. This is counted as
part of the treatment ("cheap 30-s public audio"), reported in the data-stats table, and is not a
confound. $\qquad\blacksquare$

---

## 4. The gap-closure estimand $\hat C$

### 4.1 Definition and the right denominator

On the MUSDB **test** set (one pass, §5 protocol),

$$
\boxed{\ \hat C=\frac{s_{\text{mix}}-s_{\text{base}}}{s_T-s_{\text{base}}}=\frac{u}{d},\qquad
u=s_{\text{mix}}-s_{\text{base}},\ \ d=G=s_T-s_{\text{base}}.\ }
$$

The denominator is the **teacher's own test-set score**, not an abstract oracle, because
`htdemucs` is the engine StemCraft ships — the student→teacher gap is the *product-relevant* one.
**Stated honestly:** the teacher was trained on MUSDB *train* ($+$ private songs), so on MUSDB
*test* it is a strong-but-legitimate ceiling — an **optimistic** $s_T$ relative to a truly-OOD
teacher — while the student never sees MUSDB test through any path (the pseudo pool structurally
refuses MUSDB roots, §3.3; FMA↔MUSDB catalog overlap is implausible and noted, §3.3).

### 4.2 Seed + paired-track CI via the delta method

$s_{\text{base}},s_{\text{mix}}$ are 3-seed cell means; $s_T$ is a single deterministic teacher
score. Two variance sources enter: **between-seed** ($\sigma_{\text{seed}}^2/3$ for base/mix; the
teacher has none) and **paired-track** sampling on the 50 test tracks (all three scores are means
over the *same* tracks). The delta method linearizes $\hat C=u/d$:

$$
\frac{\partial\hat C}{\partial s_{\text{mix}}}=\frac1d,\qquad
\frac{\partial\hat C}{\partial s_T}=-\frac{u}{d^2},\qquad
\frac{\partial\hat C}{\partial s_{\text{base}}}=\frac{u-d}{d^2},
$$

$$
\widehat{\mathrm{Var}}(\hat C)\approx\Big(\tfrac1d\Big)^2\mathrm{Var}(s_{\text{mix}})
+\Big(\tfrac{u}{d^2}\Big)^2\mathrm{Var}(s_T)
+\Big(\tfrac{u-d}{d^2}\Big)^2\mathrm{Var}(s_{\text{base}}),\qquad
\hat C\pm z\,\mathrm{SE}(\hat C).
$$

This linearization is **identical** to Direction 06's recovery fraction with the map
(clean $\to s_T$, bleed $\to s_{\text{base}}$, trim $\to s_{\text{mix}}$): $\rho=(s_{\text{trim}}-
s_{\text{bleed}})/(s_{\text{clean}}-s_{\text{bleed}})$ has $\partial_\text{trim}=1/d$,
$\partial_\text{clean}=-u/d^2$, $\partial_\text{bleed}=(u-d)/d^2$ — so **$\hat C$ reuses
`recovery_fraction` verbatim** (`recovery_fraction(s_clean=s_T, s_bleed=s_base, s_trim=s_mix)`),
including its "not-evaluable when $d\le0$" guard — which is exactly the precondition gate (§6). The
per-cell $\mathrm{Var}(s_\cdot)$ each combine the seed term ($\sigma^2/3$; $0$ for $s_T$) with the
paired-track term. Because the three scores are **paired over tracks**, the *primary* reported CI
is a **paired bootstrap over the 50 tracks** (resample tracks, recompute $\hat C$) — it respects
the track covariance the independent-cell delta method drops; the delta-method CI is the
closed-form companion. Both are reported; the wider governs the verdict.

### 4.3 What $\hat C$ does **not** mean

- $\hat C\ge0.25$ does **not** claim the student approaches the teacher *off* MUSDB. $\hat C$ is a
  MUSDB-test quantity; the teacher's MUSDB-train exposure inflates $s_T$ there, so on FMA/other
  music both $G$ and $\hat C$ could differ. The claim is scoped to "closes a fraction of the
  **product-relevant MUSDB** gap."
- $\hat C$ is a fraction of a **specific** gap on a **frozen** protocol; if $G$ is small it is
  ill-conditioned (small $d$), which the $G>2$ dB gate forbids (§6).
- By the §1 isolation, $\hat C$ attributes the closure to **teacher-labeled data**, not to a
  distillation-specific mechanism — there is none in the loss.

> **Code.** `singnet/analysis/bleed.py::recovery_fraction` (delta-method core, unit-tested in
> `tests/test_bleed_analysis.py`) computes $\hat C$ under the §4.2 map; the paired-track bootstrap
> is the standard per-track resample over the test CSV (§6 protocol).

---

## 5. License formalism — the ND-exclusion is load-bearing

### 5.1 The FMA taxonomy

FMA's **metadata** is CC BY 4.0; its **audio** is under *per-track, artist-chosen* CC licenses
(the `tracks.csv` `license` field), **not** one uniform license (`research/papers/fma-dataset.md`
§3). A license-safe subset must therefore be filtered per track. The pinned canonical allowlist is

$$
\mathcal A=\{\text{CC0/PD},\ \text{CC-BY},\ \text{CC-BY-SA},\ \text{CC-BY-NC},\ \text{CC-BY-NC-SA}\}.
$$

### 5.2 The ND-exclusion argument (derivative works)

A separated stem is a **derivative work** of the source recording: source separation transforms
one recording into a new arrangement/version. CC licenses carrying the **ND (NoDerivatives)**
clause prohibit *distributing* derivative works, and — for legal safety — we do not even *create*
stems for a use an ND license forbids. Hence the decision rule is a predicate on the license, an
**allowlist with ND as a hard veto**:

$$
\boxed{\ \text{allow}(L)\ \Longleftrightarrow\ \big(L\ \text{is a recognized CC license}\big)\ \wedge\ \neg\,\mathrm{ND}(L)\ \wedge\ L\in\mathcal A.\ }
$$

The implementation checks **ND first** and denies outright, *before* any BY/NC/SA parsing, so an
otherwise-allowlisted BY-SA string with a NoDerivatives clause is still denied
(`canonicalize_license`; `tests/test_prepare_fma.py::test_nd_is_denied_even_with_attribution`).
**DENY-by-default:** anything not provably in $\mathcal A$ — empty, non-CC, `All Rights Reserved`,
unparsable — is denied (an allowlist, not a blocklist; §3.1, §11).

### 5.3 The NC-inclusion rationale (scope is the qualifier)

CC-BY-**NC** (NonCommercial) *is* included because the project is **non-commercial research /
portfolio** with **no redistribution of audio or derived stems** — only trained weights are kept.
NC forbids commercial use but **permits non-commercial derivatives** (NC $\neq$ ND), so NC-* tracks
are in-scope for research. The scope is the load-bearing qualifier and is stated precisely: *were
the project commercial or were it to redistribute audio, NC would be excluded.* The exclusion is
ND (derivative-prohibiting), not NC (commercial-prohibiting) — a distinction the taxonomy makes and
the filter encodes.

### 5.4 The audit trail

Two committed artifacts, **never audio**: the **manifest** (track IDs + raw + canonical licenses +
screen flags; `write_manifest`, with a `#` provenance header naming the allowlist and screen
seed), and the **provenance file** (teacher package version, model signature, inference settings,
and the recorded mean 4-stem consistency residual; `teacher_label.py::write_provenance`). Both are
diff-able and public-auditable; no FMA audio and no raw metadata CSV are ever committed. $\qquad\blacksquare$

> **Code.** `scripts/prepare_fma.py::{canonicalize_license (ND-first), license_audit,
> filter_licenses, write_manifest}`; `scripts/teacher_label.py::write_provenance`. The full
> allow/deny table is machine-checked in `tests/test_prepare_fma.py`.

---

## 6. Statistics of the design

### 6.1 One pre-registered pair; descriptive secondaries

The single test pass (§5 protocol) scores the two 3-seed cells' best checkpoints on the 50 MUSDB
test tracks. **Exactly one** pre-registered paired contrast is confirmatory —
$\mathbf{mixed}-\mathbf{musdb\_only}$ on vocals SI-SDR, per-track **paired bootstrap** 95 % CI +
**Wilcoxon** signed-rank. Everything else (`distill_only`, `mixed25`, `mixed_trim`, the **SLR**
battery at $\theta\in\{-50,-60,-70\}$) is **descriptive** with pre-written readings (§12), so the
multiple-comparison burden is a single pair; we report its CI in full rather than correcting a
one-hypothesis design, and state this posture up front.

### 6.2 Pooled between-seed noise

$\sigma_{\text{seed}}$ is pooled over the two 3-seed cells (`musdb_only`, `mixed`):
$\sigma_{\text{seed}}=\sqrt{\tfrac12(\hat\sigma_{\text{base}}^2+\hat\sigma_{\text{mix}}^2)}$ with
$\hat\sigma_c^2$ the sample variance of cell $c$'s three seed scores — the ruler for the H-10
thresholds ($s_{\text{mix}}-s_{\text{base}}>\sigma_{\text{seed}}$ for "real", and the
$\hat C\ge0.25$ magnitude bar). `singnet/analysis/scaling.py::pooled_seed_sigma` (reused verbatim).

### 6.3 The precondition gate $G>2$ dB and its rationale

$\hat C$ divides by $d=G$. If $G\le2$ dB on the frozen protocol, the **premise** — "a large gap
exists to close" — failed upstream (the student is unexpectedly strong, or the teacher weak on this
metric), and $\hat C$ is ill-conditioned (a ratio with a tiny, noisy denominator). The gate is a
**sanity precondition, not a hypothesis**: on failure the study reframes to documenting *why the
gap is small* (pre-registered), and `recovery_fraction`'s $d\le0$ / small-$d$ path returns $\hat C$
as not-evaluable rather than a fabricated large number. This mirrors D06's "not evaluable when the
model was not hurt" guard — the same numerical honesty. The H-10 decision rules (MASTER_PLAN §2)
sit on top: **supported** iff $s_{\text{mix}}-s_{\text{base}}>\sigma_{\text{seed}}$ **and**
$\hat C\ge0.25$; **partial** iff real but $\hat C<0.25$; **null** iff $|s_{\text{mix}}-
s_{\text{base}}|\le\sigma_{\text{seed}}$; **negative transfer** iff $s_{\text{mix}}<s_{\text{base}}
-\sigma_{\text{seed}}$ (the §2 bridge branch).

---

## 7. Standalone LaTeX mirror

`theory/theory.tex` compiles the same content standalone (`article` + `amsmath`/`amssymb`, no
external figures). Compilation is untested in this CPU/no-TeX environment (noted there), but the
source is self-contained.

---

*Cross-reference index.* §1 → `hinton2015-kd.md`, and the hard-pseudo-label target path
`singnet/data/pseudo.py::MixedPools` → `singnet/train/loop.py`; §2 → `singnet/analysis/bleed.py`
+ `singnet/losses/trimmed.py` (D06 machinery, reused) + `MixedPools.pool_is_fma`; §3 →
`singnet/data/pseudo.py::MixedPools` (within-pool remix, `tests/test_pseudo.py`); §4 →
`singnet/analysis/bleed.py::recovery_fraction` (the $\hat C$ delta method, machine-checked in
`tests/test_bleed_analysis.py`); §5 → `scripts/prepare_fma.py::canonicalize_license` +
`scripts/teacher_label.py::write_provenance` (`tests/test_prepare_fma.py`,
`tests/test_teacher_label.py`); §6 → `singnet/analysis/scaling.py::pooled_seed_sigma`. Every §5
allow/deny entry and every §2/§4 numeric claim is tied to a built, unit-tested function.
