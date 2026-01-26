# LLM Distillery - Roadmap

## Now (Current Sprint)

- **Context Length Experiments** - Testing 1024/2048/head+tail token strategies
  - 1024tok: MAE 0.652 (complete)
  - 2048tok: MAE 0.627 (complete)
  - head+tail (256+256): Training in progress
  - See `docs/IDEAS.md` for full results

## Next (Coming Soon)

- **cultural-discovery v1** - Art, culture, history discoveries + cross-cultural connections
  - Target: ovr.news (Wisdom tab), Busara
  - Config created, needs prompt and training data
- **belonging v1** - Assess and develop
- **ai-engineering-practice v2** - Unblock by adding hardware engineering sources to FluxusSource
- **nature_recovery v1** - Develop harmonized prompt and prefilter
- **signs_of_wisdom v1** - Develop harmonized prompt and prefilter

## Later (Backlog)

- **Commerce Prefilter SLM** - Redo with proper multilingual embeddings and context size
- **future-of-education filter** - Educational innovation (in filters/todo/)
- **seece filter** - Corporate excellence (in filters/todo/)
- **Batch processing pipeline** - High-volume scoring infrastructure
- **Production monitoring** - Accuracy drift detection
- **Qwen2.5-7B support** - Larger model option

## Completed

### Filters
- [x] **uplifting v5** - Production ready, deployed HuggingFace Hub (private) - 2024-11
  - Val MAE: 0.68, 10K training articles
- [x] **sustainability_technology v1** - Deployed HuggingFace Hub - 2024-11
  - Test MAE: 0.690
- [x] **sustainability_technology v2** - Complete (prefilter + model) - 2025-01
  - Val MAE: 0.71, 7,990 training samples
  - Prefilter: FP Block 88.2%, TP Pass 89.0%
- [x] **investment-risk v5** - Production ready - 2024-12
  - Test MAE: 0.484, 10K training articles

### Infrastructure
- [x] **Ground truth generation pipeline** - 2024-11
- [x] **Oracle output discipline** - Scores only, tier in postfilter - 2024-11
- [x] **Data preparation pipeline** - Stratified splits - 2024-11
- [x] **Training data validation** - Quality checks - 2024-11
- [x] **Training script** - Qwen2.5-1.5B + LoRA working - 2024-11
- [x] **Prefilter evaluation framework** - For sustainability_technology - 2025-01

---

*Last updated: 2025-01-25*
