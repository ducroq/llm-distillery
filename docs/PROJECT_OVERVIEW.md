# LLM Distillery - Project Overview

## Vision
Train small, fast, specialized local models (7B) to replicate expensive cloud LLM behavior by distilling oracle-labeled ground truth into fine-tuned Qwen models. Enable free, fast semantic filtering at scale for multi-dimensional content analysis.

## Current Status
**Phase:** Development - Tech Deployment Filter Dataset Salvage
**Last Updated:** 2025-11-09

Working to create balanced training dataset for sustainability tech deployment filter. Discovered corpus imbalance (85% vaporware, 1.5% deployed) and implementing aggressive labeling strategy to extract more high-tier examples from 9,839 unlabeled articles that passed prefilter.

## Active Focus Areas
- [x] Salvage tech deployment dataset from imbalanced corpus
- [ ] Label 2,000 additional articles from unlabeled prefilter-passed pool
- [ ] Analyze new distribution and determine if balanced dataset achievable
- [ ] Create train/val/test splits for tech deployment filter
- [ ] Fine-tune Qwen2.5-7B on balanced dataset

## Recent Significant Decisions
- 2025-11-08: Selected Qwen2.5-7B over Qwen3-7B and Kimi K2-14B → See `/docs/decisions/2025-11-08-local-model-selection.md`
- 2025-11-09: Content truncation strategy (~800 words for oracle-student consistency) → See `/docs/decisions/2025-11-09-content-truncation-strategy.md`
- 2025-11-09: Corpus not exhausted - only 2.2% labeled, 9,839 candidates available → Continue aggressive labeling

## Quick Links
- [Architecture](ARCHITECTURE.md) - System design and data flow
- [Current Task](CURRENT_TASK.md) - What we're working on right now
- [Open Questions](OPEN_QUESTIONS.md) - Unresolved decisions
- [Filter Workflow](FILTER_WORKFLOW.md) - Standard process for new filters
- [Getting Started](guides/getting-started.md) - Setup instructions
- [Qwen Fine-tuning Guide](guides/qwen-finetuning-guide.md) - Training documentation

## Key Metrics
- **Oracle Cost**: ~$0.001 per article (Gemini Flash)
- **Target Dataset Size**: 3,000-5,000 balanced examples per filter
- **Current Labels**: 2,186 (tech deployment) - severely imbalanced
- **Corpus Size**: 99,763 articles (2.2% labeled)
- **Available Candidates**: 9,839 unlabeled articles passed prefilter
- **Target Model**: Qwen2.5-7B-Instruct (LoRA fine-tuning via Unsloth)
