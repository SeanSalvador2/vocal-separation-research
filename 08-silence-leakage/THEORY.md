# THEORY — Direction 08: measuring the silence the field throws away, and what four sampling policies do to it

This note derives the mathematics the code implements for Direction 08. It follows the
MASTER_PLAN §10 outline and, like Directions 01–06, ties every result to a concrete
function path in `singnet/` + `scripts/`. The **crux** is the pair (§2, §3): §2 shows —
from primary-source code — that *no* standard MSS metric can rank two systems by their
ghost-vocal behavior, and §3 derives the metric (SLR) that can, with its anchors,
invariances, and honest limits. §4 turns the four sampling policies into exposure
distributions with closed-form expected silent-exposure, the mechanism the experiment
tests. Every number in §3–§5 is machine-checked on synthetic signals (gate G0).

> **Code.** SLR + silent-region identification: `singnet/metrics/slr.py`. Sampling
> policies + the weighted draw: `singnet/data/sampling.py`. Energy profiles:
> `singnet/data/profiles.py`. Exposure telemetry + the SI-SDR-only selection guard:
> `singnet/train/loop.py` (`chunk_silent_fraction`, `is_new_best`). Eval SLR columns:
> `singnet/eval/evaluate.py`. Every §2–§6 claim is unit-tested: `tests/test_slr.py`,
> `tests/test_sampling.py`, `tests/test_profiles.py`, `tests/test_config_schema_d08.py`,
> `tests/test_train_d08.py`.

Notation. Clean vocal waveform $v$, accompaniment $a$, mixture $x = v + a$. A separator
predicts a soft magnitude mask and estimates the vocal $\hat v$ (masked mixture magnitude,
mixture phase — Direction 01 §1). For a track, $R_{\text{sil}}$ is the set of sample
indices in the **ground-truth-vocal-silent** regions (§2.1 identifier). $E(s)$ is the
windowed vocal RMS at chunk start $s$ (the energy profile, §4).

---

## 1. The silence problem, formalized

### 1.1 MUSDB vocals are silent a lot

A vocal stem is not vocal everywhere: verses have instrumental intros, breaks, and outros
where the target is *truly* zero while the mixture is loud accompaniment. Empirically
(reported per-track by notebook 01 on the real stems; synthetic demo now) a large fraction
of a typical MUSDB track's duration is vocal-silent for $\ge L_{\min} = 0.5$ s. Write
$\varphi(\text{track})$ = fraction of the $N$ candidate 6-s chunk starts whose window
vocal RMS is below the silence threshold $\theta$; the track-averaged silent fraction
$\bar\varphi = \mathbb{E}_{\text{track}}[\varphi]$ is the quantity every policy in §4
reweights.

### 1.2 Training exposure is a distribution the policy chooses

A training step draws a chunk start $s$ from a track under a policy $\pi$. This induces a
distribution over **(chunk, activity)**: with probability $\Pr_\pi[\text{silent}]$ the
drawn chunk's target is (near-)silent, and the model receives a gradient telling it to
**output silence** there. Uniform sampling — the UMX-verified baseline — makes
$\Pr_\pi[\text{silent}] = \bar\varphi$. Any other policy moves it (§4). The realized value
is logged every 500 steps (`chunk_silent_fraction`, §6) so the THEORY prediction is closed
against the actual run, not just asserted.

### 1.3 The karaoke failure mode: leakage in silence is catastrophic *and* invisible

For a karaoke product (StemCraft), the worst audible artifact is a **ghost vocal**: energy
in $\hat v$ during an instrumental break, where the listener expects silence. It is
perceptually catastrophic — there is no vocal to mask it, so even a 20-dB-down leak is
plainly audible — and, as §2 shows, it is **structurally unmeasured** by the field-standard
metrics. That double bind (most audible, least measured) is why Direction 08 leads with the
metric.

---

## 2. The measurement gap, from primary sources (crux)

### 2.1 Identifying silent regions (from GT stems)

Frame the GT vocal at 100 ms / 50 ms hop; a frame is *silent* iff its RMS is below
$\theta$ dBFS (linear amplitude $10^{\theta/20}$; primary $\theta = -60$, the **same**
constant as Direction 01's `sisdr` silent-target guard, `SILENCE_RMS_THRESHOLD` — one
project-wide notion of "silent"). Merge consecutive silent frames into runs; keep a run
iff it lasts $\ge L_{\min} = 0.5$ s (shorter gaps are breaths/rests, legitimately hard, not
instrumental passages). $R_{\text{sil}}$ is the union of kept runs
(`silent_regions(vocal_wave, sr, theta_db, frame_s, hop_s, min_run_s)`).

### 2.2 museval NaNs silent frames — verified from `metrics.py`

BSS-Eval SDR (the number every MUSDB paper quotes) is computed framewise on 1-s windows,
then median-over-frames, median-over-tracks. Read directly from `sigsep/sigsep-mus-eval`
`museval/metrics.py` (`00-shared-research/papers/bsseval-museval-sisec2018.md` §3), the
framewise guard is:

```python
if not _any_source_silent(ref_slice) and not _any_source_silent(est_slice):
    ...  # compute SDR, ISR, SIR, SAR for this window
else:
    a = np.empty((4, nsrc, nsrc)); a[:] = np.nan   # SDR, ISR, SIR, SAR
```

**Any window in which the reference (or estimate) source is silent is set to `NaN` for all
four metrics and dropped from the median.** So museval SDR is computed *only over frames
where the vocal is active*. Energy the model leaks into $\hat v$ during a truly silent
vocal region contributes **nothing** to the reported SDR — those frames are `NaN`ed away.

### 2.3 SI-SDR is singular on a silent target — one line

SI-SDR (Le Roux 2019; `00-shared-research/papers/leroux2019-si-sdr.md` §3) projects the
estimate onto the reference:

$$\alpha = \frac{\hat s^{\top} s}{\lVert s\rVert^{2}}, \qquad
  \text{SI-SDR} = 10\log_{10}\frac{\lVert \alpha s\rVert^{2}}{\lVert \alpha s - \hat s\rVert^{2}}.$$

On a silent target $s = 0$, $\alpha = \hat s^{\top} s / \lVert s\rVert^2$ **divides by
zero**: SI-SDR is undefined. Any nonzero leakage during a silent vocal region has *no*
well-defined SI-SDR — the analytic reason museval inserts `NaN` there.

### 2.4 Why SAR/SIR inherit the same blindness

SDR, ISR, SIR, SAR are one BSS-Eval family computed from the same projection decomposition
on the same frames. The museval guard NaNs **all four** together whenever the source is
silent in the window (§2.2). SIR (interference) and SAR (artifacts) would be exactly the
quantities one might hope quantify "how much accompaniment leaked", but they are discarded
on precisely the frames where the vocal is silent — the only frames where a *ghost-vocal*
leak can occur (leakage during active vocals is just ordinary separation error, already
scored). Hence: **no standard MSS metric can rank two systems by ghost-vocal behavior.**
That is the gap; SLR fills exactly it, and *only* it — SLR is computed **only** on
$R_{\text{sil}}$, the frames SDR throws away, so the two are complementary axes, never a
restatement (§6 plane).

---

## 3. SLR: definition, properties, limits (crux)

### 3.1 Definition

Over the samples of $R_{\text{sil}}$, with $\hat v$ the estimate and $x$ the mixture,

$$\boxed{\ \text{SLR} = 10\log_{10}
    \frac{\sum_{t\in R_{\text{sil}}}\hat v(t)^2 + \varepsilon}
         {\sum_{t\in R_{\text{sil}}} x(t)^2 + \varepsilon},
    \qquad \varepsilon = 10^{-8}\ }$$

leaked vocal-estimate energy **relative to the mixture energy present in the same silent
regions**. Lower is better. Referencing $x$ (not the silent $v = 0$) is the design choice
that dodges §2.3's singularity: the denominator is the mixture energy, which is *large* in
an instrumental break, not zero (`slr(est, mix, regions, eps)`).

### 3.2 Anchors (what makes it readable)

Let $E_{\hat v} = \sum_{R_{\text{sil}}}\hat v^2$, $E_x = \sum_{R_{\text{sil}}} x^2$.

- **Do-nothing** ($\hat v = x$): $E_{\hat v} = E_x$, so the ratio is exactly 1 and
  $\text{SLR} = 10\log_{10}1 = \mathbf{0}$ dB **exactly** (not approximately — the $\varepsilon$
  cancels since numerator and denominator are identical). A separator that does nothing
  passes the whole mixture and scores 0.
- **Perfect** ($\hat v = 0$ in silence): $E_{\hat v} = 0$, so
  $\text{SLR} = 10\log_{10}\frac{\varepsilon}{E_x + \varepsilon}$ — the **$\varepsilon$
  floor**, a large negative number set by how loud the break is. Not $-\infty$: $\varepsilon$
  floors the *log argument*.
- **$f$-fraction leak** ($E_{\hat v} = f\,E_x$): $\text{SLR} = 10\log_{10} f$ (for
  $E_x \gg \varepsilon$). A 10 % energy leak scores $10\log_{10}0.1 = \mathbf{-10}$ dB; 1 %
  scores $-20$ dB. This linear-in-dB readout is the interpretive payload.

All three are asserted to numerical precision in `tests/test_slr.py`
(`test_slr_do_nothing_anchor_is_exactly_zero`, `..._perfect_separator_at_eps_floor`,
`..._ten_percent_energy_leak_is_minus_ten_db`).

### 3.3 Joint-gain invariance

Scaling *both* $\hat v$ and $x$ by $c$ (a mastering gain on the whole track) multiplies
numerator and denominator by $c^2$; with $\varepsilon = 0$ the ratio — hence SLR — is
**unchanged**. SLR measures a *relative* leak, not an absolute level, so it is comparable
across tracks of different loudness (`test_slr_joint_gain_invariance`, exact at
$\varepsilon=0$). (Scaling only $\hat v$ *does* move SLR — as it must, since that is a real
change in how much leaks.)

### 3.4 Monotonicity in leaked energy

$\text{SLR} = 10\log_{10}\frac{E_{\hat v}+\varepsilon}{E_x+\varepsilon}$ is strictly
increasing in $E_{\hat v}$ for fixed $E_x$. More leaked energy ⇒ strictly higher (worse)
SLR, with no inversions — the metric is a faithful order on leakage
(`test_slr_monotone_in_leaked_energy`). Energy **outside** $R_{\text{sil}}$ never enters
the sum, so a loud, correct vocal elsewhere cannot mask a silent-region leak
(`test_slr_regions_restrict_the_support`).

### 3.5 $\varepsilon$ is a floor, not a projection guard

In SI-SDR, the analogous $\varepsilon$ patches a genuine singularity (division by
$\lVert s\rVert^2$). In SLR there is **no** projection and **no** singularity: the ratio is
of two nonnegative energies. $\varepsilon$ does one job — keep $\log$ finite when a region
is exactly silent in *both* signals (a perfect break in a perfect separation) — and is
otherwise negligible. This is why SLR is "well-defined for every estimator including
silence" (§2.3's failure mode simply cannot occur).

### 3.6 $\theta$ / $L_{\min}$ sensitivity is an explicit axis

$\theta$ sets which frames count as silent; $L_{\min}$ sets how long a run must be.
Lowering $\theta$ ($-60 \to -70$) makes the threshold **stricter** (only quieter frames
qualify), *shrinking* $R_{\text{sil}}$; raising it ($-60 \to -50$) *grows* it. Because
$R_{\text{sil}}$ moves, so can SLR (and even its sign of comparison between two systems).
Direction 08 therefore reports SLR at **$\theta \in \{-50, -60, -70\}$** on *every*
conclusion (`slr_report(..., thetas=(-50,-60,-70))`), rather than hiding a constant. The
monotone set behavior is unit-tested (`test_silent_regions_theta_sensitivity_monotone`);
$L_{\min}$'s run-merge/short-gap filtering in `test_silent_regions_min_run_filters_short_gaps`
and `..._merge_contiguous_frames_into_one_run`.

### 3.7 Estimator properties: valid-n and aggregation

A track whose vocal is never silent for $\ge L_{\min}$ has empty $R_{\text{sil}}$; SLR is
**`NaN`** and the track is excluded from the mean (mirroring museval's own convention,
§2.2), with the **valid-n reported** alongside every aggregate (`slr` returns `NaN`;
`slr_report` returns `n_regions`; the G2 gate requires $\ge 10/14$ val tracks valid). The
project aggregate is the mean over valid tracks; per-track distributions accompany it in
figures (never a bare mean).

### 3.8 Honest limitations (worded as such)

SLR is an **energy** claim, and only that.

1. **Timbre-blind.** A 12-dB-down hi-hat leak and a 12-dB-down vocal-ghost leak into
   $R_{\text{sil}}$ score **identically** — SLR sees energy, not *what* leaked. A karaoke
   listener cares which; SLR cannot say. (The umbrella plan's listening check carries the
   perceptual claim; SLR claims are worded as energy claims.)
2. **Phase-invisible / sample-domain.** SLR is computed on the waveform energy, so a leak
   and its phase-inverted copy score the same; it does not probe the STFT structure of the
   leak.
3. **Region-definition dependent.** §3.6 — the result lives on the $(\theta, L_{\min})$
   choice, mitigated by the mandatory three-$\theta$ report, not eliminated.

These are stated so no reader over-reads a $-14$ dB SLR as an audibility guarantee.

---

## 4. The four policies as exposure distributions (crux)

Fix a track with $N$ candidate 6-s starts on the 1-s grid, of which the silent set
$S = \{s : E(s) < 10^{\theta/20}\}$ has $|S| = k$, so the silent fraction is
$\varphi = k/N$. Let $\rho_E = \frac{\sum_{s\in S} E(s)}{\sum_{s} E(s)}$ be the share of
total windowed vocal energy sitting in silent windows ($\rho_E \approx 0$: silent windows
carry almost no vocal energy by definition). The chunk-start weight vectors are exactly
those in `ChunkSampler.weights` (unit-tested to the value).

### 4.1 The distributions and their expected silent exposure

| Policy | Weight on start $s$ | $\Pr_\pi[\text{silent}] = \sum_{s\in S} w(s)$ |
|---|---|---|
| `uniform` | $1/N$ | $\varphi$ (closed form) |
| `drop` | $\dfrac{\mathbb{1}[E(s)\ge\theta]}{N-k}$ | $\mathbf{0}$ (closed form — $S$ excluded) |
| `energy` | $(1-\lambda)\dfrac{E(s)}{\sum E} + \dfrac{\lambda}{N}$ | $(1-\lambda)\rho_E + \lambda\varphi$ |
| `curriculum` | `energy` with $\lambda = \lambda(t)$ | time-integral, §5 |

Two exact facts anchor the experiment:

- **`uniform` sees silence at the base rate $\varphi$**; **`drop` sees it with probability
  exactly 0** (its support is $S^{\complement}$; the degenerate all-silent track falls back
  to uniform — a documented guard, `test_drop_all_silent_falls_back_to_uniform`).
- **`energy` sees it at $(1-\lambda)\rho_E + \lambda\varphi \ge \lambda\varphi$.** The floor
  term $\lambda/N$ on each of the $k$ silent starts guarantees the lower bound
  $\lambda\varphi > 0$: **silence is down-weighted, never starved**
  (`test_energy_floor_lower_bound`). With $\lambda = 0.1$ and $\rho_E \approx 0$, `energy`
  sees silence roughly $\tfrac{1}{10}$ as often as `uniform` — reduced, but nonzero.

### 4.2 Mechanism hypotheses (why *leakage*, not noise, is the predicted failure)

The separator is a **sigmoid magnitude mask** $M\in[0,1]$ applied to the mixture. On a
silent-vocal region the input is pure accompaniment; the *correct* output is $M \approx 0$
(pass nothing). Whether the model learns that depends on exposure:

- **`drop` ⇒ zero exposure ⇒ the "output silence" mode is never fit.** The model never
  receives a gradient on a silent target, so on an instrumental break at test time it is
  **out of distribution**. It falls back to the mask it learned for spectrally similar
  content seen as the *complement* during active-vocal training — which is **nonzero** —
  and passes a fraction of the accompaniment through. The failure is therefore **structured
  leakage** (accompaniment-shaped energy in $\hat v$), not random noise. This is the H-08b
  prediction: `drop` leaks detectably more (higher SLR) than `uniform`.
- **`energy` ⇒ reduced-but-nonzero exposure ⇒ the silence mode is still fit**, while
  gradient mass concentrates on the information-rich active chunks (the H-08a quality
  prediction). The open question the plane answers: does `energy` buy quality *without*
  paying `drop`'s leakage price (the free lunch), or is there a real tradeoff?

The refutation of H-08b is itself interesting: if `drop` does **not** leak more, the mask
prior alone suffices to output silence "for free", a genuine negative about mask models at
this scale (pre-written, §13).

---

## 5. Curriculum: the schedule and its exposure integral (motivation)

`curriculum` is `energy` with $\lambda$ annealed **linearly $1.0 \to 0.1$ over the first
50 % of steps, then held at 0.1** (`ChunkSampler.lambda_at`; values
$[1.0, 0.55, 0.1, 0.1, 0.1]$ at $\{0,25,50,75,100\}\%$,
`test_curriculum_lambda_schedule_values`). At $t = 0$, $\lambda = 1$ ⇒ **pure uniform**
(sees silence at $\varphi$); by the midpoint and after, $\lambda = 0.1$ ⇒ **energy** (sees
it at $\approx 0.1\varphi$). Starts uniform, ends energy.

**Time-averaged exposure.** With $\Pr[\text{silent}\mid\lambda] = (1-\lambda)\rho_E +
\lambda\varphi$ linear in $\lambda$, the run-averaged exposure is
$\overline{\Pr}[\text{silent}] = (1-\bar\lambda)\rho_E + \bar\lambda\varphi$ where
$\bar\lambda = \frac1T\int_0^T \lambda(t)\,dt$. For the pinned schedule,

$$\bar\lambda = \underbrace{\tfrac12\cdot\tfrac{1.0+0.1}{2}}_{\text{first half, linear}}
             + \underbrace{\tfrac12\cdot 0.1}_{\text{second half, held}}
             = 0.275 + 0.05 = \mathbf{0.325}.$$

So over the run `curriculum` exposes the model to silence at $\approx 0.325\varphi$ —
**between** `energy` ($0.1\varphi$) and `uniform` ($\varphi$) — but front-loaded.

**Why "uniform early, energy late" is the motivated order (stated as motivation, not
fact).** Curriculum-learning intuition is easy-then-hard. Outputting silence on an
instrumental break is the *easy* mode (the target is literally zero); precise separation on
a dense active mix is the *hard* mode. Fitting the easy silence mode first (high $\lambda$,
uniform-like) may plant the "output 0" behavior before the optimizer specializes on the
hard active mode (low $\lambda$, energy-weighted) — capturing energy's quality gain without
drop's leakage cost. This is a **hypothesis** the plane tests (curriculum dominates /
sits between / dominates neither — three pre-written readings, §13); no primary claim rests
on it, and the schedule×LR-decay confound is noted (§12). The realized front-loading is
visible in the exposure telemetry (`sampling_exposure.csv`), closing §4's prediction
against the run.

---

## 6. Statistics: two metrics, no scalarization, and the selection guard

### 6.1 Two axes, two seed noises, two pre-registered pairs

Let $q(\pi)$ = mean best-checkpoint validation vocals SI-SDR and $\ell(\pi)$ = mean
validation SLR ($\theta=-60$, valid tracks). From the three 3-seed cells (uniform, energy,
drop) pool two between-seed standard deviations, $\sigma^q_{\text{seed}}$ (SI-SDR) and
$\sigma^\ell_{\text{seed}}$ (SLR), via `singnet.analysis.pooled_seed_sigma`. The two
pre-registered tests are:

$$\textbf{H-08a: } q(\text{energy}) - q(\text{uniform}) > \sigma^q_{\text{seed}},
  \qquad
  \textbf{H-08b: } \ell(\text{drop}) - \ell(\text{uniform}) > \sigma^\ell_{\text{seed}}.$$

The headline exhibit is the **(SI-SDR, SLR) plane**: the four policy points with per-seed
scatter, plus the do-nothing (0 dB SLR, §3.2) and oracle-IRM anchors. **We pre-commit to
NOT scalarizing** the two axes into a single score — if a quality-vs-leakage tradeoff
exists, *that geometry is the finding*, and collapsing it would hide it. The two hypotheses
are **jointly pre-registered on distinct metrics**, so no multiple-comparison correction is
applied across them (they are not a family of tests of one effect). Confirmatory inference
on the two pairs is paired bootstrap + Wilcoxon on the test set (§6, one pass).

### 6.2 The selection-bias guard (why $\ell$ is honest)

If checkpoint selection could see SLR, an arm's reported $\ell$ would be optimistically
biased (we would have *picked* the low-leakage checkpoint). So **selection is SI-SDR-only**:
`is_new_best(candidate_sisdr, best_sisdr)` takes **no SLR argument** by construction, and
`validation_report` computes SLR *alongside* SI-SDR but the loop compares only the SI-SDR
value. The SLR that H-08b judges is therefore the SLR *of the SI-SDR-best checkpoint*
(`best_val_slr`, descriptive) — an out-of-sample-for-leakage read. A test asserts the
selection helper has no `slr` parameter and never names SLR in its body
(`test_train_d08.py::test_checkpoint_selection_is_blind_to_slr`); the G3 gate audits the
registry to confirm selection never used SLR. Best-vs-final checkpoint SLR are both logged
(descriptive) in case selection-on-quality precedes a late silence regression (§12).

---

## 7. `theory/theory.tex`

A standalone LaTeX mirror of this note (amsmath/amssymb only; no external figures or
packages) lives at `theory/theory.tex`, written to compile with a bare `pdflatex`
(compile-untested in the build environment, as for Directions 01–06). It reproduces §2's
museval guard, §3's SLR definition + anchors, §4's exposure table with the closed forms,
and §5's $\bar\lambda = 0.325$ integral.
