# Open Questions

## Critical (Blocking Progress)

*None currently*

## Important (Affects Design)

- [ ] **Can we achieve balanced tier distribution from current corpus?**
  - Context: Labeling 2,000 additional articles to test if high-tier examples emerge
  - What we need to learn: Natural distribution of deployed tech in news aggregator corpus
  - Decision point: Continue labeling vs synthetic augmentation vs accept imbalance

- [ ] **What oversampling strategy for training?**
  - Context: Even with aggressive labeling, may still have tier imbalance
  - Options: Simple duplication, SMOTE, focal loss, class weighting
  - Need to decide based on final distribution

## Nice to Know (Can Work Around)

- [ ] **Should we add more filter-specific prefilter patterns?**
  - Current: Generic sustainability keywords
  - Could add: Deployment-specific patterns (but risk blocking good examples)
  - Workaround: Current permissive approach works, oracle handles filtering

## Resolved

- [x] **Have we exhausted the corpus?** - 2025-11-09 - NO! Only 2.2% labeled, 9,839 unlabeled candidates available
  - **Key Learning**: Don't apply multiple layers of restrictive filtering and conclude scarcity
  - User correctly challenged premature "exhausted corpus" conclusion after tier-specific search
  - Tier-specific keyword search was TOO restrictive (double-filtering on top of prefilter)
  - **Strategy**: Random sampling from prefilter-passed articles is better than targeted keyword search
- [x] **Content truncation strategy?** - 2025-11-09 - ~800 words / 3000 tokens for oracle-student consistency (see ADR)
- [x] **Which oracle LLM to use?** - 2025-11-08 - Gemini Flash (better discrimination than Pro, cheaper)
- [x] **Which local model for distillation?** - 2025-11-08 - Qwen2.5-7B-Instruct (proven, multilingual, efficient)
