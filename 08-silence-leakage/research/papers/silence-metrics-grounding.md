# Silence in MSS — metric pathologies, training-code behavior, and a leakage metric

Grounding note for Direction 08. Assembles verified facts from primary sources into the case for a
**custom silence-leakage metric** and drafts that metric. Verified 2026-07-13.

## 1. Why standard metrics are blind to silence (verified from primary sources)
- **museval BSS-Eval SDR inserts `NaN` for silent frames and drops them from the median.** Read directly from `sigsep/sigsep-mus-eval` `metrics.py`: `_any_source_silent()` gates each 1 s window; if the reference (or estimate) source is silent in that window, all four metrics are set to `NaN` for that window and excluded from the median-over-frames. (Full quote: `../../00-shared-research/papers/bsseval-museval-sisec2018.md §3`.) **So museval SDR is computed only over frames where the vocal is active** — leakage into silent regions is invisible to it.
- **SI-SDR is singular on a silent target.** $\alpha=\hat s^\top s/\lVert s\rVert^2$ divides by $\lVert s\rVert^2=0$ → undefined; energy leaked when the vocal is silent has no defined SI-SDR (`../../00-shared-research/papers/leroux2019-si-sdr.md §3`).
- **Independent corroboration (gap-check):** empirical MSS studies note "increases in error primarily occur in silent segments, where the SDR metric is especially sensitive." The metrics either ignore silence (museval) or behave pathologically (SI-SDR) there.

**Consequence:** the karaoke-critical failure — **vocal energy leaking into instrumental breaks ("ghost vocals")** — is *structurally unmeasured* by the standard metrics. Direction 08 defines a metric that measures exactly it.

## 2. What training code actually does with silent chunks (verified from GitHub)
- **Open-Unmix (`data.py`, verified):** **all dataset classes sample chunks uniformly at random start positions with NO activity/energy/silence filtering.** Exact evidence: `track.chunk_start = random.uniform(0, track.duration - self.seq_duration)` (and identical `random.uniform(...)` in `SourceFolderDataset`/`FixedSourcesTrackFolderDataset`); there is **no VAD, RMS, or spectral-energy computation anywhere** in the dataset classes. So UMX's implicit policy = **uniform sampling**, silent-vocal chunks treated identically to active ones. This is Direction 08's **baseline policy**.
- **Demucs (gap-check):** applies a **chunk-drop with probability 0.1** to *simulate* silent sources (an augmentation that injects silence) — the opposite intervention from dropping silent chunks; noted as a design point.
- **Spleeter:** trains on random time-crops from (mix, stem) pairs; the repo does not expose an activity-weighted sampler (training pipeline is largely internal). Treat Spleeter's policy as **uniform/random-crop** [from repo structure + training knowledge; full training code not published].
- **BSRNN (verified, Direction 03):** the *only* MSS model that explicitly uses a **source-activity detector** — but for **pseudo-label mining on unlabeled data** (keep segments where the target is active), *not* as a labeled-data chunk-sampling policy. It is the closest precedent to "activity-aware sampling" and the cross-reference for Direction 08's energy-weighted policy. (`../../03-mini-band-split/research/papers/luo2022-bsrnn.md §3`.)

## 3. The silence-leakage metric (OUR construction — precise draft)
Goal: quantify vocal-estimate energy during **truly silent vocal regions**, which SI-SDR/museval cannot. Given GT vocal $v(t)$, mixture $x(t)=v+a$, estimate $\hat v(t)$, at sample rate with frame hop $H$:

**Step 1 — identify silent vocal regions from GT.** Compute windowed RMS of the GT vocal, $\text{RMS}_v[n]$, over windows of length $W$ (e.g. 50 ms). A frame is *vocal-silent* if
$$\text{RMS}_v[n] < \tau,\qquad \tau = \max_n \text{RMS}_v[n]\;/\;10^{\,60/20}\ \ (\text{i.e. }{-60}\text{ dB below the track's peak vocal RMS}),$$
with an **absolute floor** $\tau \ge \tau_{\min}$ (e.g. $-70$ dBFS) so fully-silent/instrumental tracks don't produce a degenerate threshold. Merge silent frames into segments and **keep only segments of length $\ge L_{\min}$** (e.g. 0.5 s) so inter-syllable gaps and breaths (legitimately hard, not "instrumental breaks") are excluded. Let $R_{\text{sil}}$ = union of kept silent frames.

**Step 2 — measure leakage on $R_{\text{sil}}$.** Define the **Silence Leakage Ratio (SLR, dB)**:
$$\boxed{\ \text{SLR} \;=\; 10\log_{10}\frac{\sum_{t\in R_{\text{sil}}}\hat v(t)^2 \;+\;\epsilon}{\sum_{t\in R_{\text{sil}}} x(t)^2 \;+\;\epsilon}\ }$$
— leaked vocal-estimate energy **relative to the mixture energy** present in the same silent regions. **More negative = less leakage = better.** Referencing $x$ (not $v\!=\!0$) avoids the SI-SDR zero-target singularity. A complementary **absolute** form reports $10\log_{10}\!\big(\overline{\hat v^2}\big|_{R_{\text{sil}}}\big)$ in dBFS.

**Edge cases (must handle):**
- **$\epsilon$** ($\sim 10^{-8}$) floors both sums → no $\log 0$.
- **Empty $R_{\text{sil}}$** (a track whose vocal is never silent for $\ge L_{\min}$): SLR **undefined → `NaN`, excluded from aggregation** (mirrors museval's own convention, deliberately).
- **Threshold sensitivity:** report SLR at $\{-50,-60,-70\}$ dB thresholds to show robustness; pre-register $-60$ dB + $L_{\min}=0.5$ s as primary.
- **Windowing:** align $W,H$ to the STFT hop for consistency with the model's resolution.
- **Aggregation:** median SLR over silent segments within a track, then median/mean over the 50 test tracks (report both; note $n$ of valid tracks).

**Why this is a real metric, not a restatement of SDR:** SI-SDR/museval discard $R_{\text{sil}}$; SLR is computed *only* on $R_{\text{sil}}$. The two are complementary axes — Direction 08's headline is the **(overall vocals SI-SDR) vs (SLR) tradeoff plot**.

## 4. Relevance & cross-refs
- **Direction 08** owns this metric and the 4-policy sampling study (uniform / energy-weighted / drop-silent / curriculum).
- **Direction 05 cross-ref:** the continual-SVS paper's human feedback is literally "mark false positives" = leakage (`../../05-lora-source-separation/research/papers/continual-svs-2025.md`) — SLR automates what that paper asks humans to flag.
- **Recent related architecture:** BSMamba2 ("Mamba2 Meets Silence," 2508.14556, Aug 2025) attacks the *same* sparse-vocal problem via long-range modeling (11.03 dB cSDR) — a **different lever** (architecture) than Direction 08's (sampling policy + metric). Must be cited (novelty note in `../LITERATURE.md §4`).

## 5. Verification notes
- museval `NaN`-on-silence: CONFIRMED from code. SI-SDR singularity: derived (CONFIRMED formula). UMX uniform sampling / no activity filter: CONFIRMED from `data.py`. Demucs chunk-drop p=0.1: gap-check (WebSearch). BSRNN activity detector: CONFIRMED (WebSearch/HF).
- **SLR metric: OUR construction** — clearly marked; thresholds/eps/$L_{\min}$ are proposed defaults to pre-register, not published values.
- Spleeter training-sampler detail: `[repo structure + training knowledge]` — full training pipeline not published.
</content>
