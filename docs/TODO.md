# LLM Distillery - TODO

## Commerce Prefilter SLM - NEEDS REWORK

ML classifier for commerce/promotional content detection. Cross-cutting prefilter for all filters.

**Status:** v1 complete but needs redo - concerns about multilingual embeddings and context size.

- [x] **v1 Training data collection** - 2,847 examples (commerce + journalism)
- [x] **v1 Model training** - DistilBERT, MiniLM, XLM-RoBERTa compared
- [x] **v1 Backtesting** - 56,336 articles, threshold optimization
- [ ] **Redo with proper multilingual embeddings** - Current approach may not handle Dutch/multilingual well
- [ ] **Redo with proper context size** - May need longer context

See `filters/common/commerce_prefilter/docs/` for full documentation.

---

## Filters

### Production Ready
- [x] **uplifting v5** - Deployed on HuggingFace Hub (private)
  - Val MAE: 0.68, all dimensions < 0.80
  - 10,000 training articles
- [x] **sustainability_technology v1** - Deployed on HuggingFace Hub
  - Test MAE: 0.690
- [x] **sustainability_technology v2** - Complete (prefilter + model)
  - Val MAE: 0.71, 7,990 training samples
  - Prefilter: FP Block 88.2%, TP Pass 89.0%
- [x] **investment-risk v5** - Production ready
  - Test MAE: 0.484 (excellent)
  - 10,000 training articles

### In Active Development (priority order)
- [ ] **belonging v1** - Needs assessment
- [ ] **ai-engineering-practice v2** - BLOCKED ON DATA
  - Needs FluxusSource hardware engineering sources
  - Prompt calibration complete (~60% tier accuracy)
- [ ] **nature_recovery v1** - Early development
  - Concept and README complete, 8 dimensions defined
  - Next: harmonized prompt, prefilter.py, validation
- [ ] **signs_of_wisdom v1** - Early development
  - Concept and README complete
  - Next: harmonized prompt, prefilter.py
  - Challenge: wisdom is rare in news

### New Filters (recently created)
- [ ] **cultural-discovery v1** - Art/culture/history discoveries + cross-cultural connections
  - Config and README created
  - Target: ovr.news (Wisdom tab), Busara
  - Next: harmonized prompt, prefilter, training data

### Planned (filters/todo/)
- [ ] **future-of-education** - Educational innovation filter
- [ ] **seece** - Social, economic, environmental corporate excellence
- [ ] **sustainability_economic_viability** - Economic aspects of sustainability
- [ ] **sustainability_policy_effectiveness** - Policy impact and effectiveness

## Training Pipeline

- [x] **Data preparation pipeline** - Stratified splits working
- [x] **Training script** - Qwen2.5-1.5B + LoRA working
- [x] **Context length experiments** - 1024/2048/head+tail tested
  - 1024tok: MAE 0.652, 2048tok: MAE 0.627
  - head+tail (256+256): Training in progress
  - See `docs/IDEAS.md` for full results
- [ ] **Qwen2.5-7B support** - Larger model option for complex filters
- [ ] **Training monitoring improvements** - Better logging, early stopping

## Deployment

- [ ] **Inference server** - Unified prefilter + model + postfilter pipeline
- [ ] **Batch processing** - High-volume article scoring
- [ ] **Production monitoring** - Latency, accuracy drift detection

## Infrastructure

- [x] **Prefilter evaluation framework** - Complete for sustainability_technology
- [ ] **Generalize prefilter evaluation** - Apply to all filters
- [ ] **Dataset QA pipeline** - Automated quality checks
- [ ] **Cost tracking** - Monitor API usage for oracle scoring

## Documentation

- [ ] **Update filters/README.md** - Current status is outdated (Nov 2025)
- [ ] **Training guide** - Step-by-step for new filters
- [ ] **Deployment guide** - Production setup instructions

---

*Last updated: 2025-01-25*
