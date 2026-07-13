# Direction 05 — LoRA for source separation: parameter-efficient fine-tuning of a pretrained separator

Deep-dives: [`papers/hu2021-lora.md`](papers/hu2021-lora.md) (CORE),
[`papers/peft-music-2024.md`](papers/peft-music-2024.md),
[`papers/transfer-dialog-2021.md`](papers/transfer-dialog-2021.md),
[`papers/continual-svs-2025.md`](papers/continual-svs-2025.md).
Host model: [shared Open-Unmix note](../../00-shared-research/papers/openunmix2019.md).

## 1. Direction recap (from RESEARCH_DIRECTIONS #5)
- **Hypothesis:** **LoRA on Open-Unmix's BiLSTM/fc layers recovers ≥ 90% of the full-fine-tune SI-SDR gain at < 5% trainable parameters**, and is **less prone to catastrophic forgetting** when adapting to a shifted domain (MUSDB18 compressed-AAC vs HQ, or a genre subset).
- **Minimal experiment:** **4 recipes × 2 target domains** — zero-shot `umxhq`, head-only, **LoRA (rank ∈ {4,16})**, full fine-tune — measuring **adapted-domain gain** *and* **source-domain regression**; 2–3 seeds on the headline pair.
- **Falsification / success test:** a **quality-vs-trainable-params curve** + a **forgetting table**. Supported iff LoRA sits at ≥90% of full-FT gain with <5% params *and* regresses less on the source domain. "LoRA underperforms on LSTMs" is a useful, reportable negative.
- Difficulty/Risk: **Medium / Medium-Low** (compute cheap; main risk is LSTM-LoRA plumbing).

## 2. Paper map
| Paper | Role | Specific claim we rely on | Verified | Deep-dive |
|---|---|---|---|---|
| LoRA (2106.09685) | **replicate (method)** | $\Delta W=BA$, rank $r$, $r(d+k)$ trainable, zero merge latency | CONFIRMED | [papers/hu2021-lora.md](papers/hu2021-lora.md) |
| PETL for music (2411.19371) | **motivation** | adapters/LoRA reach ≥full-FT quality at <1% params on music *tagging* | CONFIRMED | [papers/peft-music-2024.md](papers/peft-music-2024.md) |
| Transfer→dialog (2106.09093) | **precedent** | MSS models (incl. UMX) fine-tune across domains (full FT) | CONFIRMED | [papers/transfer-dialog-2021.md](papers/transfer-dialog-2021.md) |
| Continual SVS (2512.02432) | **context** | adapting separators to specific content is an active problem | CONFIRMED (existence/scope) | [papers/continual-svs-2025.md](papers/continual-svs-2025.md) |
| Open-Unmix (JOSS + code) | **infrastructure (host)** | exact fc/LSTM shapes LoRA wraps; MIT; 6.25 dB vocals | CONFIRMED (code) | [shared](../../00-shared-research/papers/openunmix2019.md) |

## 3. Replicate-vs-extend
- **Replicate:** LoRA (Hu et al.) as a method; the "PEFT ≈ full-FT at <1% params" result (PETL-for-music) as the plausibility prior; the MSS→domain transfer precedent (Strauss et al.).
- **Extend (our delta):** **apply LoRA to an MSS model (UMX) — a setting with no published numbers** — and measure the quality/param trade-off **and** catastrophic forgetting on a real MSS domain shift. Honest novelty framing: *"first careful small-scale look at LoRA-for-MSS,"* not "first PEFT in music."

## 4. The gap (with logged gap-check — a key novelty result)
- **Gap-check query (2026-07-13, WebSearch):** *"LoRA parameter-efficient fine-tuning music source separation adapter low-rank demixing 2024 2025 2026."* **Finding: NULL for MSS.** LoRA/PEFT in music appears for **generation** (diffusion music, MDPI 2025), **singing-voice beat/downbeat tracking** (adapter tuning w/ SSL, arXiv 2503.10086), and speech deepfake detection — but **no LoRA/PEFT-for-music-source-separation study surfaced**. The search explicitly returned no MSS/demixing PEFT application.
- **Conclusion: Direction 05's central novelty claim ("essentially no published LoRA-for-MSS numbers") is VERIFIED as still true at mid-2026.** This is the direction's strongest novelty asset.
- **Cross-cut noted:** *"Why LoRA Resists Label Noise: A Theoretical Framework for Noise-Robust PEFT"* (arXiv **2602.00084**, Feb 2026) argues LoRA is *inherently* noise-robust — a theoretical bridge to Direction 06 (label noise) and an extra motivation for LoRA under the AAC/genre domain shift. Adjacent music-adapter prior: 2503.10086 (beat tracking, not separation).

## 5. Direction-specific technical notes — the exact UMX LoRA target & param budget (for MASTER_PLAN)
Host = `umxhq` vocals model. Per-matrix LoRA cost = $r(d_{out}+d_{in})$ (`hu2021-lora.md` §2). Shapes from `model.py` (`hidden_size=512`, LSTM hidden $256\times2$ dir $\times3$ layers); assume bandwidth-cropped **`nb_bins`≈1487** (16 kHz at n_fft=4096) — the exact value is a config constant, **verify programmatically in the build** (PLAN Phase 3 rule).

**Full UMX vocals model ≈ 8.3 M params** (computed from shapes): `fc1`≈1.52 M, LSTM≈4.73 M, `fc2`≈0.52 M, `fc3`≈1.52 M (+small BN/scale). Consistent with UMX's ~8.9 M/target.

**LoRA-wrappable matrices and cost:**
| Matrix | Shape $d_{out}\times d_{in}$ | LoRA cost | r=4 | r=16 |
|---|---|---|---|---|
| `fc1` | $512\times2974$ | $r\cdot3486$ | 13,944 | 55,776 |
| `fc2` | $512\times1024$ | $r\cdot1536$ | 6,144 | 24,576 |
| `fc3` | $2974\times512$ | $r\cdot3486$ | 13,944 | 55,776 |
| LSTM `weight_ih` ×(3 layers·2 dir) | $1024\times512$ each | $6\cdot r\cdot1536$ | 36,864 | 147,456 |
| LSTM `weight_hh` ×(3·2) | $1024\times256$ each | $6\cdot r\cdot1280$ | 30,720 | 122,880 |
| **Total (all wrapped)** | — | $r\cdot25404$ | **≈101.6 k (1.2%)** | **≈406 k (4.9%)** |
| LSTM-only variant | — | $r\cdot16896$ | 67.6 k (0.8%) | 270 k (3.3%) |

So **both rank-4 and rank-16 sit under the <5% budget** the hypothesis targets (rank-4 all-wrapped ≈1.2%; rank-16 ≈4.9%). "Head-only" (train `fc3` alone) ≈ **18%** of params — the inefficient baseline LoRA should beat. LoRA's **$B=0$ init** means the rank-4/16 models *start identical to zero-shot `umxhq`*, giving a clean forgetting baseline. **Forgetting metric:** source-domain (HQ) vocals SI-SDR *drop* after adapting to the target domain (AAC / genre subset), full-FT vs LoRA — the hypothesis predicts LoRA drops less.

**Plumbing note (the main risk):** PyTorch `nn.LSTM` packs the 4 gates into `weight_ih_l{k}` / `weight_hh_l{k}` (and `_reverse` for the backward direction). LoRA must wrap these gate-stacked matrices directly (or re-implement the LSTM cell to expose them). Unit-test that a rank-$r$-wrapped LSTM with $B=0$ reproduces the frozen model bit-for-bit before training.

## 6. Risks this literature implies
- **LSTM-LoRA is non-standard** — LoRA's evidence base is Transformers; recurrent low-rank adaptation may under/over-fit differently. Mitigate with the $B=0$ identity unit test and by reporting the honest negative if it underperforms.
- **Domain shift may be too small** — AAC-vs-HQ is a mild shift; if zero-shot `umxhq` already transfers, all recipes converge and the forgetting story is muted. Have the **genre-subset** shift as the stronger stress test.
- **"<5% params" depends on `nb_bins`** — the table assumes 16 kHz bandwidth; recompute for the actual `umxhq` config. The *conclusion* (rank 4/16 both <5%) is robust to reasonable `nb_bins`.
- **Novelty framing discipline** — cite PETL-for-music (tagging) and transfer-dialog (full-FT) so the claim is "first LoRA-for-MSS," accurately bounded.
