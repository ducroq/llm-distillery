# Open Questions

## Critical (Blocking Progress)

*None currently - training in progress*

## Important (Affects Design)

- [ ] **Will trained model meet success criteria?**
  - Context: Training Qwen2.5-7B-Instruct on 4,328 examples (with oversampling)
  - Targets: Per-dimension MAE < 1.5, tier accuracy ≥70% per tier, deployed recall ≥60%
  - Decision point: If targets not met, need to adjust training (more epochs, different weighting)

- [ ] **How to handle model deployment and versioning?**
  - Context: First model to complete training → need deployment strategy
  - Questions:
    - Where to store trained model checkpoints? (local, HuggingFace Hub, S3?)
    - How to version models? (v1.0, with training date?)
    - How to handle model updates? (retrain, versioning strategy)
  - Impact: Sets pattern for 5 remaining filters

- [ ] **Should postfilter be integrated into main pipeline or standalone?**
  - Context: Creating postfilter for production inference on 99K corpus
  - Options:
    1. Standalone script (manual invocation)
    2. Integrate into existing filter pipeline (if applicable)
    3. API service (Flask/FastAPI)
  - Considerations: Ease of use, batch processing, future scaling

## Nice to Know (Can Work Around)

- [ ] **What's the optimal batch size for inference?**
  - Context: Will run inference on 99K articles
  - Current: Training uses batch size 8
  - Could optimize for inference (larger batch = faster, but memory constraints)
  - Workaround: Start with batch size 8, can adjust later

- [ ] **Should we fine-tune further if metrics are close but not meeting targets?**
  - Context: E.g., if deployed recall is 55% (target 60%)
  - Options: More epochs, adjust learning rate, different oversampling ratio
  - Decision: Depends on how close we get and diminishing returns

## Resolved

- [x] **Can we achieve balanced tier distribution from current corpus?** - 2025-11-09 - NO, but handled via oversampling + class weighting
  - Natural corpus distribution: 81.6% vaporware, 1.4% deployed
  - Mitigation: Training set oversampled to 48.4% vaporware, 12.4% deployed
  - Validation/test maintain natural distribution for realistic evaluation

- [x] **What oversampling strategy for training?** - 2025-11-09 - Simple duplication with 20% target ratio
  - Implemented in `prepare_training_data_tech_deployment.py`
  - Oversamples minority classes to 20% of majority class count
  - Applied only to training set, not validation/test

- [x] **Should model output include reasoning?** - 2025-11-09 - NO, score arrays only
  - Decision: Simplified format (8 dimension scores only)
  - Rationale: Faster inference, easier evaluation, avoids quality risk from 7B reasoning
  - See ADR: `docs/decisions/2025-11-09-model-output-format.md`

- [x] **How to handle class imbalance?** - 2025-11-09 - Stratified splits + oversampling + class weighting
  - Strategy: Accept natural corpus distribution, use ML techniques
  - Implementation: Stratified 80/10/10 split, 20% oversampling, class weights during training
  - See ADR: `docs/decisions/2025-11-09-class-imbalance-strategy.md`

- [x] **Have we exhausted the corpus?** - 2025-11-09 - NO! Only 2.2% labeled, but sufficient for training
  - Final dataset: 4,146 labels consolidated from 4 batches
  - Deduplication: 50.4% duplicates removed across batches
  - Conclusion: Corpus has more data, but current dataset sufficient for distillation

- [x] **Content truncation strategy?** - 2025-11-09 - ~800 words / 3000 tokens for oracle-student consistency
  - Rationale: Match oracle input to student training data
  - Implementation: `prepare_training_data_tech_deployment.py` truncates to 800 words
  - See ADR: `docs/decisions/2025-11-09-content-truncation-strategy.md`

- [x] **Which oracle LLM to use?** - 2025-11-08 - Gemini Flash
  - Reason: Better discrimination than Pro (Pro too conservative at 91% vaporware)
  - Cost: ~$0.001 per article
  - See calibration report: `reports/tech_deployment_oracle_calibration.md`

- [x] **Which local model for distillation?** - 2025-11-08 - Qwen2.5-7B-Instruct
  - Reasons: Proven track record, multilingual, efficient, good documentation
  - See ADR: `docs/decisions/2025-11-08-local-model-selection.md`

- [x] **Which training approach: regression or text generation?** - 2025-11-10 - Regression
  - Regression: Direct prediction of score arrays (matches data format)
  - Text generation: Incompatible with prepared data format (prompt/completion expected)
  - Decision: Use regression approach in `training/train.py`
  - Deprecated: `scripts/train_model.py` with explanation doc

- [x] **Should we add more filter-specific prefilter patterns?** - 2025-11-09 - NO, keep permissive
  - Current: Generic sustainability keywords (11% pass rate)
  - Decision: Oracle handles filtering, permissive prefilter ensures broad coverage
  - Observation: Targeted keyword searches were TOO restrictive (double-filtering)
