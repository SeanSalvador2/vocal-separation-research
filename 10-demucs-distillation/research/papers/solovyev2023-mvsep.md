# Benchmarks and leaderboards for sound demixing tasks (MVSep)

Roman Solovyev, Alexander Stempkovskiy, Tatiana Habruseva. arXiv **2305.07489** (12 May 2023). | Verified: HF paper_search (title/authors/date/abstract), 2026-07-13. **Context** for Direction 10 — the ensembling/benchmark reference and the public leaderboard.

## 1. Problem & context
MSS progress is driven by challenges (MDX/SDX) but scattered across papers with different protocols. This work introduces **two public benchmarks** and a **leaderboard (mvsep.com/quality_checker)** comparing popular demixing models and their **ensembles** — the reference for "how good is model X, and how much does ensembling add."

## 2. Method
- Two new benchmark datasets for source separation; standardized comparison of popular models.
- A **stem-specific ensembling** approach: combine the models best-suited to each stem — this achieved **top results in SDX'23 tracks**.

## 3. Key results
- The ensemble approach won top places in SDX'23 tracks; the leaderboard quantifies many models on a common footing. CONFIRMED (abstract). (Metrics are the challenge SDR conventions; do not conflate with SI-SDR/museval-median.)

## 4. Limitations & caveats
- A benchmark/ensembling paper, not a distillation method; ensembling *increases* inference cost (opposite of Direction 10's goal).

## 5. Relevance to this project (Direction 10)
- **Teacher-selection reference:** the leaderboard is where one confirms the chosen teacher (`htdemucs`) is a strong, appropriate teacher and sees where it sits vs alternatives — useful for justifying the teacher choice and for interpreting distillation ceilings.
- **Contrast with our goal:** MVSep's ensembling makes a *bigger, slower* system; Direction 10 does the opposite (distill a big teacher into a *small, fast* student). Cited to frame that contrast.
- **Not reproduced** (no ensembling); teacher is a single `htdemucs`.

## 6. Verification notes
- Title/authors/date/arXiv ID + scope (two benchmarks, mvsep leaderboard, ensembling, top SDX'23 results): CONFIRMED verbatim from HF paper_search abstract.
</content>
