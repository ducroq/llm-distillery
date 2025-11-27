# Improved Semantic Prefilter Evaluation

**Date**: 2025-11-24
**Articles Evaluated**: 985
**High-Scoring Articles**: 143 (threshold: >5.0)

## Configuration

**2-Category Setup (Improved)**:
1. **Positive**: sustainability, renewable energy, climate solutions, environmental conservation, circular economy, biodiversity, green technology, and sustainability policy
2. **Negative**: other topics including sports, entertainment, politics, and general news

**Key Improvements over 6-category**:
- Simpler binary decision (no ambiguous "general news" middle ground)
- Broader positive definition (includes conservation, circular economy, policy)
- Proper recall calculation

---

## Executive Summary

**Best by Recall**: Semantic-improved (t=0.30)
- **Recall**: 97.9% (catches 98% of good articles)
- **FP Rate**: 22.3% (vs 23.2% keyword)
- **Precision**: 65.9%

**Best by Combined Error**: Semantic-improved (t=0.30)
- **Recall**: 97.9%
- **FP Rate**: 22.3%
- **Combined Error**: 24.4%

---

## Performance Comparison

### Summary Table

| Filter | Pass Rate | Recall | Miss Rate | FP Rate | Precision | Speed |
|--------|-----------|--------|-----------|---------|-----------|-------|
| **Keyword** | 100.0% | 100.0% | 0.0% | 23.2% | 64.4% | 0.0s |
| Semantic (t=0.30) | 96.0% | 97.9% | 2.1% | 22.3% | 65.9% | 16.6s | **<- BEST RECALL**
| Semantic (t=0.35) | 96.0% | 97.9% | 2.1% | 22.3% | 65.9% | 16.4s |
| Semantic (t=0.40) | 96.0% | 97.9% | 2.1% | 22.3% | 65.9% | 16.4s |
| Semantic (t=0.45) | 96.0% | 97.9% | 2.1% | 22.3% | 65.9% | 16.4s |


### Metrics Explanation

- **Pass Rate**: % of all articles that passed the prefilter
- **Recall**: % of high-scoring articles (>5.0) that passed ← KEY METRIC
- **Miss Rate**: % of high-scoring articles that were blocked
- **FP Rate**: % of passed articles with oracle scores <=2.0 (false positives)
- **Precision**: % of passed articles with oracle scores >3.0 (actually relevant)
- **Speed**: Total processing time for 985 articles

---

## Comparison with Original 6-Category

**Original 6-category (threshold 0.50)**:
- Recall: ~15% (missed 85% of good articles!)
- FP Rate: 2.1%
- Problem: "general news" category caught legitimate sustainability articles

**Improved 2-category (threshold 0.30)**:
- Recall: 97.9% (misses only 2.1% of good articles)
- FP Rate: 22.3%
- Improvement: 82.9% better recall

---

## Threshold Analysis

| Threshold | Recall | Miss Rate | FP Rate | Combined Error | Recommendation |
|-----------|--------|-----------|---------|----------------|----------------|
| 0.30 | 97.9% | 2.1% | 22.3% | 24.4% | Good recall **<- RECOMMENDED** |
| 0.35 | 97.9% | 2.1% | 22.3% | 24.4% | Good recall |
| 0.40 | 97.9% | 2.1% | 22.3% | 24.4% | Good recall |
| 0.45 | 97.9% | 2.1% | 22.3% | 24.4% | Good recall |


**Recommendation**: Use threshold = **0.30**
- Best recall: 97.9%
- Still reduces FP significantly: 22.3% vs 23.2% (keyword)
- Good balance for training data generation

---

## False Positive Examples

Articles that passed but scored <=2.0:

### Keyword Prefilter

**1. [1.0/10]** Ireland Baldwin Calls Out ‘Poisonous’ And ‘Narcissistic’ Family Members In Explo

**2. [1.5/10]** Weak Identification with Bounds in a Class of Minimum Distance Models

**3. [1.8/10]** Orange boss thinks SFR carve

### Improved Semantic (threshold=0.30)

**1. [1.0/10]** Ireland Baldwin Calls Out ‘Poisonous’ And ‘Narcissistic’ Family Members In Explo

**2. [1.5/10]** Weak Identification with Bounds in a Class of Minimum Distance Models

**3. [1.8/10]** Orange boss thinks SFR carve


---

## False Negative Examples

Articles that were blocked but scored >5.0:

### Keyword Prefilter

### Improved Semantic (threshold=0.30)

**1. [7.0/10]** The Indian Innovations Bringing Big
   - Reason: blocked_category:other

**2. [5.7/10]** Hitmusical Moulin Rouge! sowieso tot eind mei te zien in Beatrix Theater
   - Reason: blocked_category:other

**3. [6.4/10]** Designing a self
   - Reason: blocked_category:other


---

## Decision

⚠️ **RECONSIDER - USE KEYWORD PREFILTER**

**Current Results**:
- Recall: 97.9% (acceptable)
- FP reduction: 4% (not significant)

**Recommendation**: Stick with keyword prefilter
- 100% recall (no missed articles)
- Simple and fast
- Accept 23.2% FP rate (oracle will score them low)
- Training data benefits from full spectrum of relevance


---

## Next Steps

1. Review false positive/negative examples above
2. If approved: Update semantic_prefilter.py with improved 2-category configuration
3. Generate 10K training dataset using chosen configuration
4. Train student model

---

## References

- Original evaluation: `SEMANTIC_EVALUATION_REPORT.md` (6-category, poor recall)
- Semantic prefilter: `filters/sustainability_technology/v1/semantic_prefilter.py`
- Integration guide: `filters/sustainability_technology/v1/SEMANTIC_INTEGRATION.md`
