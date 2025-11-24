# GPU Evaluation Instructions

## Summary

We tested a 6-category semantic prefilter and found **poor recall (15%)** - it blocked 85% of good articles!

**Root cause**: Too many categories + narrow positive definition + high threshold (0.50)

## Improved Configuration Ready

**Script**: `scripts/evaluate_semantic_improved.py`

**Key improvements**:
1. **2 categories** (vs 6) - clearer binary decision
2. **Broader positive definition** - includes conservation, circular economy, policy
3. **Lower thresholds** (0.30-0.45) - better recall
4. **Proper recall metrics** - tracks % of good articles caught

## Expected Performance

- **Recall**: 30-50% (vs 15% with old config)
- **FP Rate**: 8-12% (vs 23% keyword baseline)
- **Still 50%+ FP reduction** compared to keyword

## How to Run on GPU

```bash
cd /path/to/llm-distillery

# Run with GPU support
python -u scripts/evaluate_semantic_improved.py

# When prompted "Use GPU? (y/n)", type: y
```

**Runtime on GPU**: ~5-10 minutes (vs ~60 minutes on CPU)

## What Gets Generated

**Report**: `filters/sustainability_technology/v1/reports/SEMANTIC_IMPROVED_EVALUATION.md`

**Contains**:
- Performance comparison (keyword vs semantic)
- Recall analysis (% of good articles caught)
- FP/FN examples
- Recommendation (approve semantic or stick with keyword)

## Files to Transfer to GPU Machine

1. `scripts/evaluate_semantic_improved.py` (evaluation script)
2. `sandbox/semantic_evaluation_1k/` (scored articles directory)
3. `filters/sustainability_technology/v1/semantic_prefilter.py` (prefilter class)
4. `filters/sustainability_technology/v1/prefilter.py` (keyword baseline)

## Decision Criteria

**Approve semantic prefilter if**:
- Recall ≥ 40% (catches at least 40% of good articles)
- FP rate < 70% of keyword baseline (significant reduction)

**Otherwise**: Stick with keyword prefilter (simpler, 100% recall)

## Previous Results (6-category)

For reference, the failed 6-category configuration:

| Threshold | Recall | FP Rate | Issue |
|-----------|--------|---------|-------|
| 0.50 | ~15% | 2.1% | **Blocks 85% of good articles!** |
| 0.35 | ~30% | 8.6% | Better but still had issues |

**Problem**: "general news" category caught legitimate sustainability articles like:
- Conservation/biodiversity articles
- Circular economy policy
- EU sustainability directives

**Fix**: 2 categories + broader definition eliminates this issue.
