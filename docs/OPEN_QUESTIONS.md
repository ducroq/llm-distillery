# LLM Distillery - Open Questions

## Critical (Blocking Progress)

(None currently)

## Important (Affects Design)

- [ ] **Prefilter false negative tolerance** - Current target is <10% FN rate. Is this right for all filters? Some filters may need stricter prefilters.
- [ ] **Model versioning strategy** - How to handle model updates? Full retraining vs incremental? HuggingFace Hub versioning approach.
- [ ] **Multi-GPU training** - Current pipeline assumes single GPU. Worth supporting distributed training for larger models?

## Nice to Know (Can Work Around)

- [ ] **Qwen2.5-0.5B viability** - Would smaller model work for simpler filters? Could reduce inference cost further.
- [ ] **Oracle model comparison** - Is Gemini Flash always best? Worth testing Claude Haiku for some filters?
- [ ] **Batch size optimization** - Current default is 4. Worth tuning per-filter or per-hardware?

## Resolved

- [x] **Optimal training data size** - 2026-01 - 5K minimum, 7-10K ideal. Merged datasets (random + screened) work best for needle-in-haystack filters. See ADR-003 and cultural-discovery v3 results.
- [x] **Embedding vs fine-tuning** - 2026-01 - Fine-tuned Qwen2.5-1.5B significantly outperforms frozen embedding probes for nuanced dimensional scoring. See `research/embedding_vs_finetuning/`.
- [x] **Oracle output format** - 2024-11-13 - Scores only, no tier classification (see ADR)
- [x] **Base model selection** - 2024-11-10 - Qwen2.5-1.5B for speed/quality balance (see `docs/decisions/2025-11-10-model-selection-qwen-1.5b.md`)
- [x] **Content truncation** - 2024-11-09 - 4K token limit with smart truncation (see `docs/decisions/2025-11-09-content-truncation-strategy.md`)

---

*Last updated: 2026-02-14*
