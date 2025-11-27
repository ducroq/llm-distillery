# Semantic Prefilter Evaluation Report

**Date**: 2025-11-23
**Articles Evaluated**: 985
**Oracle**: gemini-flash

## Executive Summary

✅ **SEMANTIC PREFILTER RECOMMENDED**

**Best Configuration**: Semantic filter with confidence threshold = 0.50
- False Positive Rate: 2.1% (vs 23.2% keyword baseline)
- False Negative Rate: 13.7% (vs 0.0% keyword baseline)
- Precision: 93.7% (vs 64.4% keyword baseline)

---

## Performance Comparison

### Summary Table

| Filter | Pass Rate | FP Rate | FN Rate | Precision | Speed |
|--------|-----------|---------|---------|-----------|-------|
| Keyword (baseline) | 100.0% | 23.2% | 0.0% | 64.4% | 0.2s |
| Semantic (t=0.25) | 17.3% | 8.8% | 13.6% | 82.9% | 2042.5s |
| Semantic (t=0.30) | 15.4% | 7.9% | 13.8% | 83.6% | 1759.8s |
| Semantic (t=0.35) | 14.2% | 8.6% | 14.0% | 83.6% | 1810.2s |
| Semantic (t=0.40) | 12.7% | 8.0% | 14.0% | 87.2% | 2565.9s |
| Semantic (t=0.45) | 10.9% | 7.5% | 13.8% | 86.9% | 2552.1s |
| Semantic (t=0.50) | 9.6% | 2.1% | 13.7% | 93.7% | 2589.8s | **←**


### Metrics Explanation

- **Pass Rate**: % of articles that passed the prefilter
- **FP Rate**: % of passed articles with oracle scores ≤2.0 (false positives)
- **FN Rate**: % of blocked articles with oracle scores >5.0 (false negatives)
- **Precision**: % of passed articles with oracle scores >3.0 (actually relevant)
- **Speed**: Total processing time for 1000 articles

---

## Threshold Analysis

### False Positive vs False Negative Tradeoff

| Threshold | FP Rate | FN Rate | Combined Error | Recommendation |
|-----------|---------|---------|----------------|----------------|
| 0.25 | 8.8% | 13.6% | 22.4% | Too permissive |
| 0.30 | 7.9% | 13.8% | 21.7% | Too permissive |
| 0.35 | 8.6% | 14.0% | 22.5% | Balanced |
| 0.40 | 8.0% | 14.0% | 22.0% | Balanced |
| 0.45 | 7.5% | 13.8% | 21.3% | Too restrictive |
| 0.50 | 2.1% | 13.7% | 15.8% | Too restrictive **← BEST** |


**Recommendation**: Use threshold = **0.50**
- Lowest combined error rate (15.8%)
- Good balance between precision and recall
- Significantly better to keyword baseline

---

## False Positive Examples

Articles that passed the filter but scored ≤2.0 (should have been blocked):

### Keyword Prefilter

**1. [1.0/10]** Ireland Baldwin Calls Out ‘Poisonous’ And ‘Narcissistic’ Family Members In Explo
   - Reason passed: passed

**2. [1.5/10]** Weak Identification with Bounds in a Class of Minimum Distance Models
   - Reason passed: passed

**3. [1.8/10]** Orange boss thinks SFR carve
   - Reason passed: passed

### Semantic Prefilter (threshold=0.50)

**1. [1.0/10]** #Offline
   - Reason passed: passed_semantic

**2. [1.2/10]** TotalEnergies won't appeal ruling on 'misleading' climate claims
   - Reason passed: passed_semantic


---

## False Negative Examples

Articles that were blocked but scored >5.0 (should have passed):

### Keyword Prefilter

### Semantic Prefilter (threshold=0.50)

**1. [6.5/10]** ‘A Good Year for Species’: Conservationist Vivek Menon on His Journey From the W
   - Reason blocked: blocked_category:general

**2. [5.7/10]** Sustainability, Vol. 17, Pages 9408: From Policy to Practice: EU Circular Econom
   - Reason blocked: blocked_category:general

**3. [5.3/10]** Sustainability, Vol. 17, Pages 8762: Policy
   - Reason blocked: blocked_category:general


---

## Decision

✅ **APPROVE SEMANTIC PREFILTER**

**Rationale**:
1. False positive rate: 2.1% vs 23.2% (keyword)
2. Semantic understanding prevents 'oil in turmoil' type errors
3. Processing time acceptable for batch processing (2629.3ms per article)

**Recommended Implementation**:
- Use semantic prefilter for 10K training data generation
- Confidence threshold: 0.50
- Expected processing time: ~7.3 hours for 10K articles
- Expected cost savings: ~$3.17 (fewer false positives to oracle)


---

## Next Steps

1. Review false positive/negative examples above
2. If approved, integrate semantic prefilter into batch_scorer
3. Generate 10K training dataset using approved configuration
4. Train student model on filtered data

---

## References

- Semantic prefilter implementation: `filters/sustainability_technology/v1/semantic_prefilter.py`
- Integration guide: `filters/sustainability_technology/v1/SEMANTIC_INTEGRATION.md`
- Raw evaluation data: `sandbox/semantic_evaluation_1k/`
