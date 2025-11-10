# LLM Distillery - Project Overview

## Vision
Train small, fast, specialized local models (7B) to replicate expensive cloud LLM behavior by distilling oracle-labeled ground truth into fine-tuned Qwen models. Enable free, fast semantic filtering at scale for multi-dimensional content analysis.

## Current Status
**Phase:** Training - Tech Deployment Filter Model Training
**Last Updated:** 2025-11-10

Training Qwen2.5-7B-Instruct on consolidated tech deployment dataset (4,146 labels). Dataset preparation complete with stratified splits and minority class oversampling. Training in progress on GPU machine (3 epochs, batch size 8). Next: Model evaluation and postfilter deployment.

## Active Focus Areas
- [x] Consolidate and deduplicate tech deployment labels (4,146 final)
- [x] Prepare training data with stratification + oversampling (4,328/413/417)
- [x] Start model training on GPU (Qwen2.5-7B-Instruct)
- [ ] Evaluate trained model on validation set (target: ≥70% per-tier accuracy)
- [ ] Create postfilter for production inference
- [ ] Deploy trained model for full corpus inference (99K articles)

## Recent Significant Decisions
- 2025-11-08: Selected Qwen2.5-7B-Instruct over Qwen3-7B and Kimi K2-14B → See `/docs/decisions/2025-11-08-local-model-selection.md`
- 2025-11-09: Model output format (score arrays only, no reasoning) → See `/docs/decisions/2025-11-09-model-output-format.md`
- 2025-11-09: Class imbalance strategy (stratified splits + 20% oversampling) → See `/docs/decisions/2025-11-09-class-imbalance-strategy.md`
- 2025-11-09: Content truncation strategy (~800 words for oracle-student consistency) → See `/docs/decisions/2025-11-09-content-truncation-strategy.md`
- 2025-11-10: Deprecated text generation training script in favor of regression approach → See `scripts/DEPRECATED_train_model.md`

## Quick Links
- [Architecture](ARCHITECTURE.md) - System design and data flow
- [Current Task](CURRENT_TASK.md) - What we're working on right now
- [Open Questions](OPEN_QUESTIONS.md) - Unresolved decisions
- [Filter Workflow](FILTER_WORKFLOW.md) - Standard process for new filters
- [Dataset README](../datasets/labeled/sustainability_tech_deployment/README.md) - Tech deployment dataset statistics
- [Getting Started](guides/getting-started.md) - Setup instructions
- [Qwen Fine-tuning Guide](guides/qwen-finetuning-guide.md) - Training documentation

## Key Metrics
- **Oracle Cost**: ~$0.001 per article (Gemini Flash)
- **Tech Deployment Labels**: 4,146 (consolidated, deduplicated)
- **Training Data**: 4,328 train / 413 val / 417 test (with oversampling)
- **Tier Distribution (original)**: 81.6% vaporware, 10.2% pilot, 6.8% early commercial, 1.4% deployed
- **Tier Distribution (train, after oversampling)**: 48.4% vaporware, 23.2% pilot, 15.9% early commercial, 12.4% deployed
- **Corpus Size**: 99,763 articles
- **Target Model**: Qwen2.5-7B-Instruct (regression-based training)
