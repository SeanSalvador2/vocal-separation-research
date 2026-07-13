# THEORY — Direction 05: the math behind LoRA for source separation

This note derives the mathematics the code implements for Direction 05. It follows the
MASTER_PLAN §9 outline and, like Directions 01–03, ties every result to a concrete
function path in `singnet/peft/` + `scripts/`. The **crux** is §3 (LoRA on the BiLSTM):
the whole engineering novelty is a rank-$r$ update on PyTorch's gate-stacked
$4h\times d$ recurrent weight, and the correctness of that wrap is what the
`B=0`-identity and merge round-trip tests pin. Every parameter number in §4 is the same
one `tests/test_umx_wrapper.py` asserts against the built `nn.Module`.

Notation. The host is the pretrained **Open-Unmix `umxhq` vocals** model
$\mathcal U_{\theta_0}$ (00-shared-research/papers/openunmix2019.md §2). Fine-tuning seeks
$\theta = \theta_0 + \Delta\theta$ adapting $\mathcal U$ to a shifted domain $D$; the
**recipe** fixes which coordinates of $\Delta\theta$ are free. A magnitude spectrogram of
a stereo mixture is $|X|\in\mathbb R_{\ge0}^{c\times F\times T}$ ($c=2$ channels,
$F=n_{\text{fft}}/2+1=2049$ output bins, $T$ frames); $\mathcal U$ predicts a
non-negative mask-like magnitude $\widehat V=\mathcal U(|X|)$ and the vocals waveform is
$\hat v=\mathrm{iSTFT}(\widehat V\,e^{i\angle X})$ (mixture phase, **no Wiener**, §1.3).

> **Code.** Adapters: `singnet/peft/lora.py`
> (`LoRALinear`, `wrap_lstm_lora`, `merge_lora`, `trainable_report`). Host + recipes:
> `singnet/peft/umx_wrapper.py` (`MockOpenUnmix`, `load_umxhq`, `apply_recipe`,
> `recipe_trainable_share`). Trainer: `singnet/peft/finetune_umx.py`. Domains:
> `scripts/make_domains.py`. Every §4 number is machine-checked in
> `tests/test_umx_wrapper.py`.

---

## 1. Transfer, PEFT, and what "adaptation" means for a mask model

### 1.1 Three points on the adaptation–cost curve

Given a pretrained $\theta_0$ and a target domain $D$, the classical choices are:

* **full fine-tune** — free all $|\theta_0|$ parameters ($\Delta\theta\in\mathbb R^{|\theta_0|}$).
  Maximal capacity, one fresh model per domain, and the most exposed to
  **catastrophic forgetting** (the source domain's function is overwritten).
* **head-only** — free only the output remapping ($\mathrm{fc3}$, $\mathrm{bn3}$, the
  output affine). Cheap, but it can only re-scale/re-mix the final magnitude — it cannot
  change the learned *representation*.
* **LoRA** — free a **low-rank additive update** to selected weight matrices
  (Hu et al., arXiv 2106.09685). The bet is that the *change* $\Delta W$ needed to adapt
  has **low intrinsic rank**, so $r\ll\min(d,k)$ suffices at $r(d+k)\ll dk$ parameters.

### 1.2 The low-intrinsic-rank hypothesis, stated for regression

PEFT's evidence base is **classification / generation** (LLMs, vision, music *tagging*
at $<1\%$ params, arXiv 2411.19371). Source separation is a **dense regression** —
$\mathcal U$ predicts a continuous $F\times T$ magnitude field, scored by SI-SDR, not a
label. There is no published PEFT-for-MSS number (gap-check verified 2026-07-13,
LITERATURE §4). Whether "the adaptation is low-rank" transfers from *what class is this*
to *what magnitude mask reconstructs this source under a shifted input distribution* is
exactly the open question. A rank-$r$ update constrains the adaptation to a rank-$r$
perturbation of each wrapped linear map; if the covariate shift (codec artefacts, added
noise) is absorbable by a low-rank re-tuning of the representation, LoRA wins on cost —
if it demands a full-rank re-tuning, LoRA under-fits, and that negative is itself a
reportable result about the geometry of separation adaptation (MASTER_PLAN §12).

### 1.3 What we deliberately hold fixed

One host, one loss (UMX-native MSE on magnitude, §5.4), one 6 k-step budget, one stereo
pipeline with the code-verified UMX augmentation recipe, Wiener **off**. The **only**
degree of freedom across recipes within a domain is *which parameters receive gradients*
(and the LoRA reparametrization) — the controlled-comparison invariant (§3.5 of the
plan). This is a *recipe* study, not a loss or architecture study (Directions 01/03 own
those).

---

## 2. LoRA, complete

### 2.1 The update, initialization, and merge

For a pretrained linear map $W_0\in\mathbb R^{d\times k}$ and input $x\in\mathbb R^k$,
LoRA reparametrizes the fine-tuned weight as

$$
W=W_0+\Delta W,\qquad \Delta W=\frac{\alpha}{r}\,BA,\quad
B\in\mathbb R^{d\times r},\ A\in\mathbb R^{r\times k},\ r\ll\min(d,k),
$$

so the adapted forward pass is

$$
\boxed{\,h=W_0x+\frac{\alpha}{r}\,B(Ax)\,}
$$

computed as **two small matmuls** ($z=Ax\in\mathbb R^r$, then $Bz\in\mathbb R^d$) — the
$d\times k$ product $BA$ is never formed at train time. **Initialization**
$A\sim\mathcal N(0,\sigma^2)$, $\boxed{B=0}$, so $\Delta W=0$ and the adapted map equals
$W_0$ *exactly* at step 0 — training begins at the pretrained function, no shock, and for
LoRA-wrapped $\mathcal U$ this start **is** zero-shot (the clean forgetting baseline, §6).
After training, **merge** $W\leftarrow W_0+\frac{\alpha}{r}BA$ folds the update into a
single matrix (zero added inference cost); swapping domains is swapping the merged delta.

> **Code.** `LoRALinear.forward` (the two-matmul form), `LoRAParametrization.forward`
> ($W\mapsto W+\frac{\alpha}{r}BA$), `LoRALinear.merge` / `merge_lora`. B=0 identity and
> merge round-trip: `tests/test_lora.py::{test_lora_linear_b0_is_exact_identity,
> test_lora_linear_merge_roundtrip}`.

### 2.2 Trainable count and the $\alpha=2r$ choice

Only $A,B$ train; $W_0$ is frozen. Per wrapped matrix,

$$
\#\text{LoRA}(d,k,r)=\underbrace{r\,k}_{A}+\underbrace{d\,r}_{B}=r\,(d+k)\quad\text{vs. full }dk .
$$

For $\mathrm{fc1}$ ($512\times2974$) at $r=16$: $16\cdot3486=55{,}776$ vs $1{,}522{,}688$ —
a $27\times$ reduction on that matrix. **Why $\alpha=2r$ fixed (no tuning).** The factor
$\alpha/r$ rescales $\Delta W$; writing $s=\alpha/r$, the update is $s\,BA$ and
$\partial\mathcal L/\partial B = s(\partial\mathcal L/\partial W)A^\top$. Changing $s$ is
equivalent to rescaling the effective learning rate on $(A,B)$ (Hu et al. §4.1): tuning
$\alpha$ *and* the LR is redundant. We therefore **fix** $\alpha=2r$ (so $s=2$, a mild
gain that keeps the update magnitude roughly $r$-independent as we sweep $r\in\{4,16\}$)
and put all step-size freedom into the per-recipe LR probe (§5.5) — one tuning axis, not
two.

> **Code.** $r(d+k)$ count: `lora_param_cost`, asserted in
> `tests/test_lora.py::test_lora_trainable_count_is_r_times_din_plus_dout`. $\alpha/r$
> scaling: `::test_lora_alpha_over_r_scaling`.

---

## 3. LoRA on a BiLSTM — the novel plumbing (crux)

LoRA's evidence base is Transformer attention projections. Open-Unmix's core is a **3-layer
bidirectional LSTM**, and wrapping its recurrent weights is Direction 05's #1 engineering
risk (MASTER_PLAN §11). This section derives *what* a rank-$r$ update on PyTorch's
gate-stacked weight means, *why* it is valid, and the *correctness condition* the tests
check.

### 3.1 The LSTM cell and the gate-stacked $4h\times d$ layout

One LSTM layer with input $x_t\in\mathbb R^{d}$ and previous hidden $h_{t-1}\in\mathbb R^{h}$
computes four gate pre-activations and combines them:

$$
\begin{aligned}
\begin{pmatrix} i_t\\ f_t\\ g_t\\ o_t \end{pmatrix}
&=\begin{pmatrix}\sigma\\\sigma\\\tanh\\\sigma\end{pmatrix}\!\!
\Big(\,\underbrace{W_{ih}}_{4h\times d}x_t+b_{ih}
        +\underbrace{W_{hh}}_{4h\times h}h_{t-1}+b_{hh}\Big),\\[2pt]
c_t&=f_t\odot c_{t-1}+i_t\odot g_t,\qquad h_t=o_t\odot\tanh(c_t).
\end{aligned}
$$

**PyTorch packs the four gates by rows** of a single matrix, in the order
$(i,f,g,o)$: rows $[0{:}h]$ of $W_{ih}$ produce the input gate, $[h{:}2h]$ the forget gate,
$[2h{:}3h]$ the cell candidate, $[3h{:}4h]$ the output gate (and likewise for $W_{hh}$,
$b_{ih}$, $b_{hh}$). So `weight_ih_l{k}` is the flat $4h\times d$ matrix
$W_{ih}=\big[\,W_{ih}^{(i)};W_{ih}^{(f)};W_{ih}^{(g)};W_{ih}^{(o)}\,\big]$ (a vertical
stack of four $h\times d$ blocks), and `weight_hh_l{k}` the $4h\times h$ stack.

### 3.2 What a rank-$r$ update on the *stacked* matrix means

Wrap the whole stacked $W_{ih}$ with one LoRA pair $A\in\mathbb R^{r\times d}$,
$B\in\mathbb R^{4h\times r}$: $\Delta W_{ih}=\frac{\alpha}{r}BA$. Partition $B$ by gate,
$B=\big[\,B^{(i)};B^{(f)};B^{(g)};B^{(o)}\,\big]$ with each $B^{(\bullet)}\in\mathbb R^{h\times r}$.
Then the update to each gate's pre-activation is

$$
\Delta\,\text{(gate }\bullet)=\frac{\alpha}{r}\,B^{(\bullet)}\underbrace{(A x_t)}_{z_t\in\mathbb R^r}
\qquad\bullet\in\{i,f,g,o\}.
$$

**Reading:** a rank-$r$ update on the stacked matrix computes **one** shared $r$-dimensional
projection $z_t=Ax_t$ of the input and distributes it to the four gates through
**gate-specific low-rank read-outs** $B^{(\bullet)}$. The four gates *share the same input
subspace* $\mathrm{row}(A)$ but read different directions out of it. This is a genuine
inductive bias — "the adaptation acts through one small shared feature of the input,
routed per gate" — and it is strictly cheaper than the alternative.

**Contrast with per-gate wrapping.** Wrapping each $h\times d$ gate block with its own
$(A^{(\bullet)},B^{(\bullet)})$ costs $4\,r(h+d)$ and gives the gates **independent** input
subspaces. Stacked costs $r(4h+d)$. For $d=h$: stacked $=5rh$ vs per-gate $=8rh$ — stacked
is $1.6\times$ cheaper *and* imposes the shared-subspace prior. We **pre-register stacked as
primary** (it is what PyTorch's flat weight exposes, and the cheaper, more constrained
hypothesis is the honest first bet); per-gate wrapping is named as future work, to be
proposed (not run) if LoRA under-fits (MASTER_PLAN §12, "propose higher ranks / per-gate
wrapping").

### 3.3 Why stacked-wrapping is valid (no cell re-derivation)

The gate nonlinearities and the recurrence in §3.1 act **downstream** of the matmul
$W_{ih}x_t$: the slicing into $(i,f,g,o)$ is applied to the *result* of the linear map.
Replacing $W_{ih}$ by $W_{ih}+\frac{\alpha}{r}BA$ therefore perturbs every gate
pre-activation additively and *exactly* as an ordinary weight edit — the cell equations
are unchanged, only the (shared) linear operator is reparametrized. Hence LoRA on the
flat $4h\times d$ matrix is well-defined without touching the LSTM recurrence, and the same
holds for the recurrent $W_{hh}$ (there $A\in\mathbb R^{r\times h}$ acts on $h_{t-1}$).

### 3.4 Bidirectionality and layer indexing

`umxhq`'s LSTM is `LSTM(512, 256, num_layers=3, bidirectional=True)`. Per layer $k$ and
per direction there are two stacked matrices; the backward direction carries a `_reverse`
suffix. Because the layer is bidirectional with hidden $256$, each layer's output is
$2\cdot256=512$, which **equals** the input width, so **every** `weight_ih_l{k}` is
$4\cdot256\times512=1024\times512$ and **every** `weight_hh_l{k}` is $1024\times256$
(uniform across all layers/directions). We wrap all $3\times2\times2=12$ recurrent matrices:

$$
\big\{\text{weight\_ih\_l}k,\ \text{weight\_hh\_l}k,\ \text{weight\_ih\_l}k\text{\_reverse},\
\text{weight\_hh\_l}k\text{\_reverse}\big\}_{k=0}^{2}.
$$

> **Code.** `wrap_lstm_lora` iterates `lstm._flat_weights_names` filtered to the
> `weight_ih_l`/`weight_hh_l` prefixes (`LSTM_LORA_PREFIXES`), covering both directions and
> all layers; `tests/test_lora.py::test_lstm_lora_wraps_all_directions_and_layers` asserts
> the $12\times2=24$ trainable A/B tensors.

### 3.5 The parametrization mechanism and its correctness condition

We register a `torch.nn.utils.parametrize` parametrization
$\pi:\;W_0\mapsto W_0+\frac{\alpha}{r}BA$ on each recurrent weight, with
`right_inverse` $=\mathrm{id}$ (so the stored `.original` is exactly $W_0$, frozen). The
**correctness condition** is the initialization:

$$
B=0\;\Longrightarrow\;\pi(W_0)=W_0\;\Longrightarrow\;\mathcal U_{\text{wrapped}}\equiv\mathcal U_{\theta_0}\ \text{(bit-exact)}.
$$

This is *testable*, and on CPU the wrapped LSTM reproduces the unwrapped output with **max
absolute difference $0.0$** (de-risked; `tests/test_lora.py::test_lstm_lora_b0_is_exact_identity`).
Gradients reach only $A,B$ (the `.original` has `requires_grad=False`); with $B\ne0$ the
output changes and both $A$ and $B$ receive gradient (`::test_lstm_lora_update_flows_and_merges`).

**One plumbing subtlety, resolved.** `nn.LSTM` caches its weights in a list
`_flat_weights` at construction and passes *that* list to the fused kernel; a
parametrization registered afterwards would leave the cache stale. We therefore attach a
**forward pre-hook** that re-pulls `getattr(lstm, name)` — i.e. the freshly recomputed
$W_0+\frac{\alpha}{r}BA$ — into `_flat_weights` before every forward, so the update flows
through and stays in the autograd graph. **cuDNN caveat** (MASTER_PLAN §11): the
parametrized weight is not a single contiguous buffer, so on GPU cuDNN's fused LSTM kernel
de-optimizes to the unfused path (slower, numerically equivalent) — acceptable at 6 k
steps; wall-clock is recorded per run. **Merge** removes the parametrizations with
`leave_parametrized=True` (baking $W_0+\frac{\alpha}{r}BA$ into a plain `Parameter`),
detaches the hook, and rebuilds `_flat_weights` — a plain `nn.LSTM` whose forward equals
the wrapped forward (round-trip diff $0.0$).

> **Code.** `LoRAParametrization`, `wrap_lstm_lora` (`_refresh_flat_weights` pre-hook),
> `merge_lora`; `tests/test_lora.py::test_lstm_lora_update_flows_and_merges`.

---

## 4. The Open-Unmix host, formalized

### 4.1 Architecture equations (verified shapes)

Per frame (batch/time folded where noted), with $n_b=1487$ bandwidth-cropped input bins,
$n_o=2049$ output bins, $c=2$ channels, hidden $H=512$:

$$
\begin{aligned}
\text{crop+standardize:}\quad & \tilde x = (\,|X|_{[:n_b]} + \mu_{\text{in}}\,)\odot s_{\text{in}},
   && \mu_{\text{in}},s_{\text{in}}\in\mathbb R^{n_b}\ \text{(learnable)}\\
\text{fc1/bn1/tanh:}\quad & u=\tanh\!\big(\mathrm{BN}_1(W_1\,\mathrm{vec}(\tilde x))\big),
   && W_1\in\mathbb R^{H\times c\,n_b}\ (512\times2974)\\
\text{BiLSTM:}\quad & \ell=\mathrm{LSTM}_{3,\leftrightarrow}(u),
   && \ell\in\mathbb R^{H}\ (256\times2\ \text{dir})\\
\text{skip+fc2/bn2/relu:}\quad & p=\mathrm{relu}\!\big(\mathrm{BN}_2(W_2[u;\ell])\big),
   && W_2\in\mathbb R^{H\times2H}\ (512\times1024)\\
\text{fc3/bn3:}\quad & q=\mathrm{BN}_3(W_3\,p),
   && W_3\in\mathbb R^{c\,n_o\times H}\ (4098\times512)\\
\text{output:}\quad & \widehat V=\mathrm{relu}\!\big(q\odot s_{\text{out}}+\mu_{\text{out}}\big)\odot|X|,
   && s_{\text{out}},\mu_{\text{out}}\in\mathbb R^{n_o}\ (2049).
\end{aligned}
$$

The load-bearing shape is $W_3$: the input is cropped to $n_b=1487$ but the mask is emitted
over the **full** $n_o=2049$-bin spectrum, so $\mathrm{fc3}$ outputs $2n_o=4098$, **not**
$2n_b$. This is why the head share below is $23.7\%$, not the plan's symmetric-spectrum
estimate of $18\%$ (§4.3, DEVIATIONS).

### 4.2 Where each recipe's trainable set sits

| Recipe | Free coordinates of $\Delta\theta$ | Graph location |
|---|---|---|
| `zeroshot` | $\varnothing$ | — (untuned host; = LoRA's $B{=}0$ start) |
| `head` | $W_3,\mathrm{bn}_3,s_{\text{out}},\mu_{\text{out}}$ | output remapping only |
| `lora4/16` | $A,B$ on $\{W_1,W_2,W_3\}\cup\{$12 LSTM matrices$\}$; plus $\mu_{\text{in}},s_{\text{in}},s_{\text{out}},\mu_{\text{out}}$ | low-rank edit of the representation + I/O affine |
| `full` | all $\theta_0$ | everything |

BatchNorm policy (pinned, §3.1/§11): for **every trained recipe** the BN layers run in
**train mode** (running stats adapt to $D$), uniformly — a shared confound, not a per-recipe
knob; zero-shot never updates BN. The trainer sets `model.train()`;
`apply_recipe` sets only `requires_grad`.

### 4.3 The trainable-share table (exact, mock-derived; checkpoint-recomputed at G1)

Base host $=8{,}893{,}348$ params. Shares are of the **host base** (the H-05a denominator
"$<5\%$ of the host's parameters"); the LoRA A/B additions are excluded from that
denominator. Per-matrix LoRA cost $r(d_{\text{out}}+d_{\text{in}})$ summed over
$\{W_1,W_2,W_3\}$ and the 12 recurrent matrices gives $A/B=r\cdot26{,}528$; the trained I/O
affine adds $2n_b+2n_o=7072$.

| Recipe | Trainable params | Share of host | Plan estimate |
|---|---:|---:|---:|
| `zeroshot` | 0 | 0.0000 % | 0 % |
| `head` | 2,110,470 | **23.7309 %** | ≈ 18 %† |
| `lora4` | 113,184 | **1.2727 %** | ≈ 1.2 % ✓ |
| `lora16` | 431,520 | **4.8522 %** | ≈ 4.9 % ✓ |
| `full` | 8,893,348 | 100 % | 100 % |

†The plan's $18\%$ used a symmetric-spectrum approximation ($\mathrm{fc3}$ out $=2n_b$,
total $\approx8.3$ M). The **verified** shapes give $\mathrm{fc3}$ out $=2n_o=4098$, so
$\text{head}=W_3(2{,}098{,}176)+\mathrm{bn}_3(8{,}196)+s_{\text{out}}(2049)+\mu_{\text{out}}(2049)
=2{,}110{,}470\Rightarrow23.73\%$. The **LoRA** shares — the ones H-05a rests on — are robust
to that approximation and land within $\pm0.3$ pp of the plan; `lora16` is comfortably under
$5\%$, so H-05a's budget clause is intact. See `results/DEVIATIONS.md`.

> **Code.** `recipe_trainable_share` (closed form) and `measured_trainable_share` (on the
> live wrapped model) agree to $10^{-9}$; the exact table is pinned in
> `tests/test_umx_wrapper.py::{test_trainable_share_table_pinned_and_measured,
> test_head_share_is_23_7_not_18}` and printed by `finetune_umx.sanity`.

---

## 5. Domain shift, additivity, and fairness

### 5.1 T1 (codec) as covariate shift on inputs *and* targets

Re-encoding at AAC 64 kbps applies a lossy, non-linear operator $\kappa$ (band-limiting +
psychoacoustic quantization) to each stem. Unlike additive noise, $\kappa$ shifts the
distribution of **both** the network input $|X|$ *and* the supervised target $|V|$ — the
model must learn to reconstruct the *degraded* vocals from the *degraded* mixture. `umxhq`
(HQ-trained) never saw such artefacts, so T1 is a genuine input+target covariate shift.

### 5.2 Additivity preservation (one line)

Materialize each stem's degraded version $\tilde s=\kappa(s)$ **independently**, then
**define** the degraded mixture as the sum $\tilde m:=\sum_s\tilde s$. Then
$\tilde m=\tilde v+\tilde a$ holds **by construction** (the mixture is *formed* as the sum
of the degraded stems, never itself re-encoded), so the supervised pair $(\tilde m,\tilde v)$
is exactly additive — no estimation error is introduced by the domain op. $\qquad\blacksquare$

### 5.3 T2 as input-only shift (fallback)

The noise fallback adds pink noise $\eta$ to the mixture only, at a fixed SNR: the target
$V$ stays clean, so $\hat m=m+\eta$ is an **input-only** shift and the task becomes
separate-and-denoise. The genre-subset alternative (if labels materialize) is instead a
*content* shift with no signal-level operator. The **rule** — genre if a $\ge14$-train/$\ge5$-test
cluster is materializable, else noise — is pre-registered; the choice is resolved at G0b.

### 5.4 The loss

Every recipe minimizes UMX-native **MSE on magnitude**,
$\mathcal L=\big\|\,\widehat V-|V|\,\big\|_2^2$ averaged over $(c,F,T)$ and the batch — held
fixed so the comparison is over recipes, not losses.

### 5.5 Per-recipe LR fairness (why one shared LR biases)

The LoRA gradient near $B=0$ is $\partial\mathcal L/\partial B=\frac{\alpha}{r}(\partial\mathcal L/\partial W)A^\top$
with $A$ small-random, so the *effective* step LoRA takes on $W$ is scaled very differently
from full-FT's direct step on $W$. Empirically PEFT methods want **larger** LRs; a single
shared LR would systematically hobble one side of the comparison. We therefore give each
trained recipe its **own** probe-chosen LR: a 500-step T1-val probe over a
recipe-appropriate grid (full-FT $\in\{3\mathrm e{-}5,1\mathrm e{-}4,3\mathrm e{-}4\}$;
head/LoRA $\in\{3\mathrm e{-}4,1\mathrm e{-}3,3\mathrm e{-}3\}$), LRs frozen at G2 before the
12 main runs. The schedule is otherwise identical (Adam, 200-step linear warmup then
**constant** LR).

> **Code.** T1 op `scripts/make_domains.py::aac64_commands` (pinned, deterministic); T2 op
> `::add_noise_at_snr` (exact SNR, `tests/test_make_domains.py::test_scale_noise_hits_exact_snr`);
> rule `::resolve_t2`. Schedule `finetune_umx.make_umx_lr_lambda`; probe grids
> `scripts/run_sweep.py::D05_PROBE_GRIDS`.

---

## 6. Statistics of the design

### 6.1 The gain-ratio delta-method CI

Let $g_D(R)=\overline{\mathrm{SISDR}}_R^{D\text{-test}}-\overline{\mathrm{SISDR}}_{\text{zeroshot}}^{D\text{-test}}$
be the adapted-domain gain (3-seed mean for the seeded cells). H-05a tests the ratio
$\rho=g_{T1}(\text{lora16})/g_{T1}(\text{full})$. Treating the two cells as independent with
seed variances $\hat\sigma^2_{L},\hat\sigma^2_{F}$ (each over 3 seeds), the **delta method**
linearizes $\rho$:

$$
\widehat{\mathrm{Var}}(\rho)\approx\frac{\hat\sigma_L^2/3}{g_F^2}
   +\frac{g_L^2\,(\hat\sigma_F^2/3)}{g_F^4},\qquad
\mathrm{SE}(\rho)=\sqrt{\widehat{\mathrm{Var}}(\rho)} .
$$

Decision (pre-registered): $\rho\ge0.9$ with CI excluding $<0.75$ $\Rightarrow$ **clean
support**; $\rho<0.9$ with CI excluding $\ge0.9$ $\Rightarrow$ **refuted**; else **mixed**.
The **precondition** $g_{T1}(\text{full})>\max(2\sigma_{\text{seed}},0.3\text{ dB})$ must hold
or H-05a is "not evaluable — shift too mild" and T2 promotes to primary.

### 6.2 Forgetting, anchored to zero-shot

Forgetting is $f(R)=\mathrm{SISDR}_{\text{zeroshot}}^{\text{std}}-\mathrm{SISDR}_R^{\text{std}}$
on the **standard** test set (positive $=$ regression). We anchor to **zero-shot**, not to
published UMX numbers, for two reasons: (i) zero-shot is the *same host under our exact eval
harness* (mixture-phase, no Wiener), so $f$ is a pure within-harness drift, free of
protocol mismatch; (ii) the LoRA $B=0$ start **is** zero-shot (§2.1), so $f(\text{lora})$
measures adaptation-induced drift from a bit-identical baseline. H-05b:
$f(\text{lora16})<f(\text{full})-\sigma_{\text{seed}}$ on T1.

### 6.3 Seed policy and what the test session licenses

Seeds fix data order/augmentation stream and LoRA-$A$ init (the $B=0$ start is seed-invariant).
$\sigma_{\text{seed}}$ is pooled over the two 3-seed cells (lora16, full on T1). The **one**
consolidated test session (15 evals) scores every checkpoint + zero-shot on (a) the domain
test set (gain) and (b) the standard test set (forgetting) in a single pass, **after** all
LRs and checkpoints freeze on validation (gate G3). It licenses the two pre-registered
comparisons (H-05a gain ratio, H-05b forgetting) plus the descriptive
quality-vs-trainable-params curve; it does **not** license any post-hoc recipe/LR selection
(none happens after it) — more test exposure than Directions 01–03, justified because
*adaptation quality on held-out tracks is the object of study*.

> **Code.** Consolidated session `singnet/peft/evaluate_umx.py::build_test_matrix`
> (RUN LATER; skeleton fails loud without checkpoints/data,
> `tests/test_finetune_umx.py::test_test_matrix_errors_cleanly_without_data`); registry
> metadata (`domain, recipe, rank, lr, trainable_params, trainable_share, peak_vram_gb`)
> `singnet/train/registry.py`.

---

## 7. Standalone LaTeX mirror

`theory/theory.tex` compiles the same content standalone (`article` + `amsmath`/`amssymb`,
no external figures). Compilation is untested in this CPU/no-TeX environment (noted there),
but the source is self-contained.

---

*Cross-reference index.* §2 → `peft/lora.py::{LoRALinear, LoRAParametrization, merge_lora}`;
§3 → `peft/lora.py::{wrap_lstm_lora, _refresh_flat_weights}` (asserted by
`tests/test_lora.py`); §4 → `peft/umx_wrapper.py::{MockOpenUnmix, apply_recipe,
recipe_trainable_share}` (numbers machine-checked in `tests/test_umx_wrapper.py`,
mirrored in `results/DEVIATIONS.md`); §5 → `scripts/make_domains.py` +
`peft/finetune_umx.py`; §6 → `peft/evaluate_umx.py` + the analysis cells of
`notebooks/02_lora_adaptation_experiments.ipynb`. Every number in §4 is machine-checked
against the built mock model.
