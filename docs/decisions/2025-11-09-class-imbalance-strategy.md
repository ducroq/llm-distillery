# ADR: Class Imbalance Handling Strategy

**Date**: 2025-11-09
**Status**: Accepted
**Deciders**: User, Claude

## Context

The sustainability_tech_deployment filter exhibits severe class imbalance in the labeled dataset:
- **Vaporware (<4.0)**: 81.6% (3,382 labels)
- **Pilot (4.0-5.9)**: 10.2% (424 labels)
- **Early Commercial (6.0-7.9)**: 6.8% (282 labels)
- **Deployed (≥8.0)**: 1.4% (58 labels)

This imbalance reflects the natural distribution of technology news articles, which favor announcements and future promises over deployed solutions.

## Problem

Training a model on severely imbalanced data leads to:
1. Model bias toward majority class (vaporware)
2. Poor performance on minority classes (deployed, early commercial)
3. Reduced ability to detect high-value deployed technology articles

## Options Considered

### Option 1: Continue Random Labeling
**Approach**: Label more articles randomly from the unlabeled corpus

**Analysis**:
- Batch 1 (1,938 labels): 1.4% deployed, 81.2% vaporware
- Batch 2 (1,988 labels): 1.4% deployed, 82.2% vaporware
- **Result**: NO improvement in distribution

**Verdict**: REJECTED - Random sampling yields same imbalanced distribution

### Option 2: Targeted Keyword Search
**Approach**: Use deployment-specific keywords to find high-tier articles

**Analysis**:
- Tier 1 (deployed) candidates: 5 articles found
- Tier 2 (early commercial) candidates: 587 articles found
- Tier 3 (pilot) candidates: 587 articles found
- **Result**: Very low yield, double-filtering artifact

**Verdict**: REJECTED - Too restrictive, exhausts corpus quickly

### Option 3: Stratified Splitting + Oversampling (SELECTED)
**Approach**: Accept natural imbalance, use ML techniques to handle it

**Components**:
1. **Stratified Train/Val Split**: Maintain tier proportions in validation set
2. **Minority Class Oversampling**: Duplicate minority examples in training set only
3. **Class Weighting During Training**: Penalize misclassification of rare classes
4. **Validation on Natural Distribution**: Keep validation set imbalanced to reflect reality

**Rationale**:
- Respects natural corpus distribution (news aggregators favor announcements)
- ML-proven approach (used in medical diagnosis, fraud detection, etc.)
- Preserves all labeled data without synthetic generation
- Allows model to learn natural distribution while improving minority class performance

**Verdict**: ACCEPTED

## Decision

**We will use stratified splitting with minority class oversampling to handle class imbalance.**

### Implementation Details

**Stratified Splitting** (prepare_training_data.py):
```python
# Split train/val (90/10) while maintaining tier proportions
stratified_split(labels, val_ratio=0.1, seed=42)
```

**Oversampling** (training set only):
```python
# Oversample minority classes to 20% of majority class
oversample_minority_classes(train_set, target_ratio=0.2)
```

**Expected Results**:
- Original: 81.6% vaporware, 1.4% deployed
- After oversampling (train set): ~55% vaporware, ~15% deployed (estimated)
- Validation set: Maintains natural 81.6% vaporware, 1.4% deployed

**Class Weights** (during training):
```python
class_weights = {
    'vaporware': 1.0,
    'pilot': 8.0,
    'early_commercial': 12.0,
    'deployed': 60.0
}
```

## Consequences

### Positive
- ✅ Uses existing 4,146 labels without requiring more expensive oracle labeling
- ✅ Industry-standard ML approach with proven track record
- ✅ Validation set remains representative of real-world distribution
- ✅ Training set provides sufficient minority class examples
- ✅ Can proceed to model training immediately

### Negative
- ⚠️ Oversampling can lead to overfitting on minority class examples
  - Mitigation: Use data augmentation, dropout, regularization
- ⚠️ Model may still struggle with rare classes in production
  - Mitigation: Monitor precision/recall per tier, adjust thresholds
- ⚠️ Validation metrics may not reflect production performance
  - Mitigation: Track per-class metrics, not just overall accuracy

### Neutral
- Model will learn that vaporware is the most common class (which is true)
- Deployed technology articles will remain rare in inference (reflecting corpus reality)

## Alternatives Rejected

1. **Synthetic Data Generation**: Risk of introducing artifacts, losing oracle signal quality
2. **Different Data Source**: Would require new data pipeline, different domain characteristics
3. **Accept Imbalance Without Mitigation**: Model would completely ignore minority classes
4. **SMOTE/ADASYN**: Complex for text data, oversampling simpler and more interpretable

## References

- `scripts/prepare_training_data.py` - Implementation of stratification + oversampling
- `docs/CURRENT_TASK.md` - Batch 2 analysis showing no improvement from random labeling
- `scripts/consolidate_tech_deployment_labels.py` - Dataset consolidation with deduplication

## Success Criteria

Training success will be measured by:
1. **Per-tier accuracy** ≥70% on validation set for each tier
2. **Deployed tier recall** ≥60% (critical for finding high-value articles)
3. **Precision-recall balance** across all tiers
4. **Confusion matrix** showing reasonable classification across all tiers

If these criteria are not met, we may need to:
- Adjust oversampling ratios
- Apply synthetic augmentation for deployed tier
- Seek additional specialized data sources for deployed technology articles
