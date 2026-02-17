# LLM Distillery - Roadmap

## Now (Current Sprint)

- **belonging v1** - Assess existing work and develop
- **Hybrid inference: train probes for remaining filters** - Phase A code complete, Phase B needs GPU
  - inference_hybrid.py + probe dirs created for sustainability_technology v2, investment-risk v5, cultural-discovery v3
  - Calibration script fixed for all config formats
  - **Next:** Generate e5-small embeddings, train MLP probes, calibrate thresholds (GPU required)

## Next (Coming Soon)
- **ai-engineering-practice v2** - Unblock by adding hardware engineering sources to FluxusSource
- **nature_recovery v1** - Develop harmonized prompt and prefilter
- **signs_of_wisdom v1** - Develop harmonized prompt and prefilter
- **Retrain deployed models with Gemma-3-1B** - Uplifting v5, cultural-discovery v3 (modest MAE gain, faster inference)

## Later (Backlog)

- **uplifting v6** - BLOCKED on HIGH-tier data collection
  - 10,495 articles ready but only 8 HIGH-tier (0.08%) — model can't learn upper range
  - **Before training:** Collect 50-100 HIGH articles from targeted sources (Better India, Upworthy, Reasons to be Cheerful, Solutions Journalism Network)
  - **Prompt fix queued:** Content-type cap for individual criminal cases
  - See `filters/uplifting/v6/PLAN.md` for full collection strategy
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

*Last updated: 2026-02-17*
