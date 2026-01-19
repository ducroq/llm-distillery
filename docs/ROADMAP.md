# LLM Distillery - Roadmap

## Now (Current Sprint)

- **Commerce Prefilter SLM** - Replace brittle regex with ML classifier
  - Risk: Low | Effort: Medium | Impact: HIGH (benefits all filters)
  - See `docs/COMMERCE_PREFILTER_SLM_DESIGN.md`
  - Training data: commerce (Electrek deals, affiliate) vs journalism (Reuters, Nature)

## Next (Coming Soon)

- **sustainability_technology v2 + Commerce SLM** - Integrate SLM into prefilter
- **belonging v1** - Assess and develop
- **ai-engineering-practice v2** - Unblock by adding hardware engineering sources to FluxusSource
- **nature_recovery v1** - Develop harmonized prompt and prefilter
- **signs_of_wisdom v1** - Develop harmonized prompt and prefilter

## Later (Backlog)

- **future-of-education filter** - Educational innovation (in filters/todo/)
- **seece filter** - Corporate excellence (in filters/todo/)
- **Batch processing pipeline** - High-volume scoring infrastructure
- **Production monitoring** - Accuracy drift detection
- **Qwen2.5-7B support** - Larger model option

## Completed

### Filters
- [x] **uplifting v5** - Production ready, deployed HuggingFace Hub (private) - 2025-11
  - Val MAE: 0.68, 10K training articles
- [x] **sustainability_technology v1** - Deployed HuggingFace Hub - 2025-11
  - Test MAE: 0.690
- [x] **sustainability_technology v2** - Complete (prefilter + model) - 2025-01
  - Val MAE: 0.71, 7,990 training samples
  - Prefilter: FP Block 88.2%, TP Pass 89.0%
- [x] **investment-risk v5** - Production ready - 2025-12
  - Test MAE: 0.484, 10K training articles

### Infrastructure
- [x] **Ground truth generation pipeline** - 2024-11
- [x] **Oracle output discipline** - Scores only, tier in postfilter - 2024-11
- [x] **Data preparation pipeline** - Stratified splits - 2024-11
- [x] **Training data validation** - Quality checks - 2024-11
- [x] **Training script** - Qwen2.5-1.5B + LoRA working - 2024-11
- [x] **Prefilter evaluation framework** - For sustainability_technology - 2025-01

---

*Last updated: 2025-01-16*
