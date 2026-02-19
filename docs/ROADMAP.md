# LLM Distillery - Roadmap

## Now (Current Sprint)

- **Retrain cultural-discovery with Gemma-3-1B** - Same approach as uplifting v6
- **belonging v1** - Assess existing work and develop
- **Deploy hybrid inference to NexusMind** - Probes trained and calibrated, need to sync to production
- **Fit calibration for other production filters** - Apply isotonic calibration to investment-risk v5, cultural-discovery v3, sustainability_technology v2

## Next (Coming Soon)
- **ai-engineering-practice v2** - Unblock by adding hardware engineering sources to FluxusSource
- **nature_recovery v1** - Develop harmonized prompt and prefilter
- **signs_of_wisdom v1** - Develop harmonized prompt and prefilter

## Later (Backlog)

- **uplifting v7** - HIGH-tier data collection
  - v6 has only 8 HIGH articles (0.08%) — model can't learn upper score range
  - Collect 50-100 HIGH articles from targeted sources (Better India, Upworthy, Reasons to be Cheerful, Solutions Journalism Network)
  - Apply crime content-type cap to oracle prompt (fix is designed, not yet in prompt-compressed.md)
- **Active Learning for HIGH-tier articles** - Continue using production filter to find high-scoring candidates
  - Method: Filter production output, screen predicted >= 5.5, oracle score, repeat
  - Target sources: positive_news_the_better_india, positive_news_upworthy, etc.
  - Goal: Collect 50+ HIGH-tier (7+) articles for v7
- **Prefilter strategy documented** - ADR-004: Commerce is only universal noise; filter-specific noise handled by trained model
  - Commerce prefilter v2 already deployed
  - Accept ~30-40% oracle waste during training (zeros = valuable negative examples)
  - No new universal prefilter needed
- **future-of-education filter** - Educational innovation (in filters/todo/)
- **seece filter** - Corporate excellence (in filters/todo/)
- **Batch processing pipeline** - High-volume scoring infrastructure
- **Production monitoring** - Accuracy drift detection
- **Qwen2.5-7B support** - Larger model option (lower priority given Gemma-3-1B results)

## Completed

### Filters
- [x] **uplifting v6** - Deployed on HuggingFace Hub (private) - 2026-02
  - Val MAE: 0.673 (v5 was 0.688), Gemma-3-1B, 12% faster inference
  - Data sculpting: active learning (495 MEDIUM enrichment) + label correction (57 crime articles capped)
  - See `filters/uplifting/v6/README.md` for full documentation
- [x] **cultural-discovery v3** - Production ready, deployed HuggingFace Hub - 2026-01
  - Val MAE: 0.77, 7,827 training articles (merged random+screened)
  - 39% improvement on medium-tier, 23% on high-tier vs v1
  - Key learning: screen+merge strategy for needle-in-haystack filters
- [x] **uplifting v5** - Production ready, deployed HuggingFace Hub (private) - 2024-11
  - Val MAE: 0.68, 10K training articles
- [x] **sustainability_technology v1** - Deployed HuggingFace Hub - 2024-11
  - Test MAE: 0.690
- [x] **sustainability_technology v2** - Complete (prefilter + model) - 2025-01
  - Val MAE: 0.71, 7,990 training samples
  - Prefilter: FP Block 88.2%, TP Pass 89.0%
- [x] **investment-risk v5** - Production ready - 2024-12
  - Test MAE: 0.484, 10K training articles

### Research
- [x] **Context length experiments** - 1024/2048/head+tail strategies - 2025-01
  - head+tail (256+256) deployed to production
  - See `docs/IDEAS.md` for full results
- [x] **Embedding vs fine-tuning** - Confirmed fine-tuning beats probes by ~18% MAE
  - Probes fast enough for Stage 1 screening in hybrid pipeline
- [x] **Stage 2 model comparison** - Gemma-3-1B adopted as default - 2026-02
  - Beats Qwen-1.5B on both uplifting (MAE 0.652 vs 0.660) and cultural-discovery (0.743 vs 0.755)
  - 8% faster inference, 38% faster training, fewer parameters (1B vs 1.5B)

### Hybrid Inference Pipeline
- [x] **Shared infrastructure** - embedding_stage.py, hybrid_scorer.py - 2026-02
- [x] **Uplifting v5 integration** - probe + inference_hybrid.py - 2026-02
- [x] **Threshold calibration** - 24K articles, threshold 4.5, 2.09x speedup - 2026-02
- [x] **Stage 2 model decision** - Gemma-3-1B adopted as default - 2026-02
- [x] **Generalize to all filters (Phase A)** - inference_hybrid.py + probe dirs + calibration fix for 3 filters - 2026-02
- [x] **Train probes + calibrate (Phase B)** - e5-small MLP probes for all 3 filters - 2026-02
  - sustainability_technology v2: MAE 0.707, threshold 1.25, 1.2% FN, 1.25x
  - investment-risk v5: MAE 0.497, threshold 1.50, 0.8% FN, 1.07x
  - cultural-discovery v3: MAE 0.609, threshold 1.25, 0.0% FN, 1.52x

### Score Calibration
- [x] **Isotonic regression calibration** (ADR-008) - 2026-02
  - Per-dimension isotonic regression corrects MSE score compression at inference time
  - Shared library: `filters/common/score_calibration.py` (fit, apply, save, load)
  - CLI tool: `scripts/calibration/fit_calibration.py` (works for any filter)
  - Applied to uplifting v6: val MAE 0.673 -> 0.653, tier distribution closer to oracle
  - Backwards compatible: scores pass through unchanged if `calibration.json` absent

### Filter Harmonization
- [x] **Harmonize filters: llm-distillery as single source of truth** - 2026-02
  - Fixed drift between llm-distillery and NexusMind (sadalsuud + gpu-server)
  - base_prefilter.py: thread-safe commerce detector loading (threading.Lock)
  - investment-risk v5: unified source-based + content-pattern prefilter, removed academic blocking
  - uplifting v5 + cultural-discovery v3: deployed (academic domain exclusion already removed)
  - All production prefilters verified identical across 3 locations

### Infrastructure
- [x] **Ground truth generation pipeline** - 2024-11
- [x] **Oracle output discipline** - Scores only, tier in postfilter - 2024-11
- [x] **Data preparation pipeline** - Stratified splits - 2024-11
- [x] **Training data validation** - Quality checks - 2024-11
- [x] **Training script** - Qwen2.5-1.5B + LoRA working - 2024-11
- [x] **Prefilter evaluation framework** - For sustainability_technology - 2025-01

---

*Last updated: 2026-02-19*
