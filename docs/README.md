# LLM Distillery Documentation

Complete documentation for knowledge distillation from large language model oracles to specialized local models.

## 📚 Documentation Index

### Core Documentation

- **[Architecture](ARCHITECTURE.md)** - System design, oracle output discipline, harmonized structure
- **[System Overview](SYSTEM_OVERVIEW.md)** - Current state, datasets, filter status
- **[Repository Structure](REPOSITORY_STRUCTURE.md)** - Directory organization and conventions
- **[Decisions Log](DECISIONS.md)** - Strategic decisions and architectural rationale
- **[Article Metadata Schema](article-metadata-schema.md)** - Standard article format

### Development Agents

- **[Agents Overview](agents/README.md)** - Development workflow agents
- **[Filter Development Guide](agents/filter-development-guide.md)** - Complete 9-phase filter lifecycle
- **[Filter Harmonizer](agents/filter-harmonizer.md)** - Consistency checking and validation
- **[Dimensional Regression QA](agents/dimensional-regression-qa-agent.md)** - Training data validation
- **[Oracle Calibration](agents/oracle-calibration-agent.md)** - Oracle quality validation

### Filter Development

See **[Filter Development Guide](agents/filter-development-guide.md)** for complete workflow:

1. **Planning** - Define dimensions, tiers, gatekeepers, weights
2. **Architecture** - Harmonize prompt structure, inline filters
3. **Validation** - Oracle calibration on sample articles (50-100 articles)
4. **Prefilter** - Test false negative/positive rates (1K+ articles)
5. **Training Data** - Score 5K+ articles, validate quality
6. **Training** - Fine-tune Qwen2.5-7B student model
7. **Testing** - Benchmark vs oracle, integration tests
8. **Documentation** - Complete all reports and guides
9. **Deployment** - Production release with monitoring

### Current Status

**Phase 5 Complete**: Training Data Validated ✅

All three active filters have validated training datasets:
- **uplifting v4**: 6,705 examples (80/10/10 split)
- **sustainability_tech_innovation v2**: 4,968 examples (80/10/10 split)
- **investment-risk v4**: 4,880 examples (80/10/10 split)

**Next Phase**: Model training (Qwen2.5-7B fine-tuning)

### Key Concepts

**Oracle Output Discipline**:
- Oracle outputs dimensional scores ONLY (0-10 per dimension + reasoning)
- Tier classification happens in postfilters (flexible thresholds without retraining)
- Clean separation: oracle scores, postfilter classifies

**Harmonized Architecture**:
- Standard prompt structure across all filters
- Inline filters for every dimension (fast model compatibility)
- Consistent config.yaml format
- Gatekeepers enforce hard requirements

**Training Pipeline**:
- Stratified splitting (tier-based or score-bin based)
- Comprehensive quality validation (structural, distribution, content)
- Automatic deduplication (cross-split duplicate removal)
- Validation reports saved to filter directories

### Decision Records

Key architectural decisions documented in `decisions/`:
- **[Oracle Output Discipline](decisions/2025-11-13-remove-tier-classification-from-oracle.md)**
- **[Post-filter Architecture](decisions/2025-11-13-post-filter-architecture.md)**
- **[Inline Filters](decisions/2025-11-14-inline-filters-for-fast-models.md)**
- **[Local Model Selection](decisions/2025-11-08-local-model-selection.md)**

## 🎯 Quick Tasks

### Score Training Data for a Filter

```bash
# Score 5K articles with oracle
python -m ground_truth.batch_scorer \
  --filter filters/uplifting/v4 \
  --source datasets/raw/master_dataset.jsonl \
  --output-dir datasets/scored/uplifting_v4_training \
  --llm gemini-flash \
  --target-count 5000 \
  --batch-size 100
```

### Prepare and Validate Training Data

```bash
# Split into train/val/test with stratification
python training/prepare_data.py \
  --filter filters/uplifting/v4 \
  --data-source datasets/scored/uplifting_v4_training \
  --output-dir datasets/training/uplifting_v4

# Validate quality
python training/validate_training_data.py \
  --data-dir datasets/training/uplifting_v4 \
  --filter filters/uplifting/v4

# If duplicates found, deduplicate
python training/deduplicate_training_data.py datasets/training/uplifting_v4

# Generate validation report for filter documentation
python scripts/validation/generate_validation_summary.py \
  --data-dir datasets/training/uplifting_v4 \
  --filter-name uplifting \
  --version v4 \
  --output filters/uplifting/v4/TRAINING_DATA_VALIDATION.md
```

### Check Filter Harmonization

```bash
# Use filter harmonizer agent to check consistency
Task: "Check filter at filters/uplifting/v4 for harmonization.
Compare against reference patterns."
```

## 📖 Reading Order for New Contributors

1. **[System Overview](SYSTEM_OVERVIEW.md)** - Understand current state and datasets
2. **[Architecture](ARCHITECTURE.md)** - Learn system design and oracle discipline
3. **[Filter Development Guide](agents/filter-development-guide.md)** - Complete workflow
4. **[Repository Structure](REPOSITORY_STRUCTURE.md)** - Navigate the codebase
5. **[Decisions Log](DECISIONS.md)** - Understand strategic choices

## 🎓 Filter Development Learning Path

### Phase 1-2: Planning & Architecture (1-2 days)
- Define filter purpose and dimensions
- Create harmonized prompt structure
- Set up config.yaml
- Document tier scheme and gatekeepers

### Phase 3-4: Validation (1-2 days)
- Oracle calibration (50-100 articles)
- Prefilter validation (1K+ articles)
- Fix false negatives/positives
- Adjust thresholds

### Phase 5: Training Data (3-5 days)
- Score 5K+ articles ($5-10 cost)
- Validate data quality
- Deduplicate cross-split IDs
- Generate validation reports

### Phase 6: Model Training (1-2 days)
- Fine-tune Qwen2.5-7B
- Monitor training metrics
- Evaluate on test set
- Iterate if needed

### Phase 7-9: Testing & Deployment (2-3 days)
- Benchmark vs oracle
- Integration testing
- Complete documentation
- Production deployment

**Total Timeline**: 2-4 weeks from planning to production

## 💡 Best Practices

### Ground Truth Generation
✅ Start with oracle calibration (validate prompt on 50-100 articles)
✅ Test prefilter thoroughly (measure false negatives!)
✅ Use stratified sampling for balanced tier distribution
✅ Score 5K+ articles minimum (3K acceptable for simple filters)
✅ Validate training data before starting model training

### Training Data Quality
✅ Zero duplicate IDs across train/val/test splits
✅ Proper split ratios (80/10/10 ±5%)
✅ All scores in valid range [0-10]
✅ Sufficient variance (not all 0s or all 10s)
✅ Complete dimension coverage across score ranges

### Cost Management
✅ Use Gemini Flash for bulk scoring ($0.001/article)
✅ Implement effective prefilters (30-70% pass rate)
✅ Target 5K articles for training (~$5-10 per filter)
✅ Reuse oracle labels for multiple filter versions when possible

### Quality Assurance
✅ Manual review of 10-20 articles per tier
✅ Check gatekeeper enforcement
✅ Verify inline filters working correctly
✅ Monitor tier distribution (no tier >60%)
✅ Compare student model vs oracle (≥80% agreement)

## 🆘 Getting Help

**Filter Development Questions**
→ See [Filter Development Guide](agents/filter-development-guide.md)

**Training Data Issues**
→ See Phase 5 troubleshooting in Filter Development Guide

**Architecture Questions**
→ See [Architecture](ARCHITECTURE.md) and [Decisions Log](DECISIONS.md)

**Found a Bug**
→ Report at GitHub issues

## 📊 Current Filter Status

### Production Ready (Training Data Validated)

| Filter | Version | Examples | Status |
|--------|---------|----------|--------|
| uplifting | v4 | 6,705 | ✅ Ready for training |
| sustainability_tech_innovation | v2 | 4,968 | ✅ Ready for training |
| investment-risk | v4 | 4,880 | ✅ Ready for training |

All filters have:
- ✅ Harmonized architecture
- ✅ Oracle-validated prompts
- ✅ Prefilter validation complete
- ✅ Training data scored and validated
- ✅ Validation reports in filter directories
- ⏳ Model training pending

### Next Steps

**Phase 6: Model Training**
- Implement Qwen2.5-7B fine-tuning script
- Train all three filters
- Evaluate performance (target: MAE ≤1.5, tier accuracy ≥85%)

See [System Overview](SYSTEM_OVERVIEW.md) for detailed current state.
