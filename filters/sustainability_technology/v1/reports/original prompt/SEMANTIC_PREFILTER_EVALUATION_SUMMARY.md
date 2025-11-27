# Semantic Prefilter Evaluation Summary - sustainability_technology v1

**Date**: 2025-11-24
**Evaluation Dataset**: 985 articles from ground truth (143 high-scoring articles, threshold >5.0)
**Status**: Completed - Decision: Use keyword prefilter

---

## Executive Summary

We evaluated semantic prefiltering as an alternative to keyword-based prefiltering to reduce false positives before oracle scoring. Through two iterations (6-category and 2-category), we learned:

**Key Finding**: Semantic prefiltering CAN achieve high recall (97.9%) but provides minimal advantage (4% FP reduction) over simple keyword prefiltering for this use case.

**Decision**: **Use keyword prefilter** - Accept 23.2% FP rate, maintain 100% recall, prioritize simplicity.

---

## Evaluation Journey

### Phase 1: 6-Category Semantic Prefilter ❌

**Hypothesis**: Use 6 categories to classify articles before oracle scoring, blocking "general news" and other off-topic content.

**Categories**:
1. Sustainability & green technology
2. Climate & renewable energy
3. Conservation & biodiversity
4. Policy & regulation
5. Corporate responsibility
6. General news (negative category)

**Results** (threshold 0.50):
- **Recall**: ~15% ⚠️
- **FP Rate**: 2.1%
- **Problem**: Blocked 85% of good articles!

**Root Cause**:
- "General news" category caught legitimate sustainability articles
- Examples: Conservation articles, circular economy policy, EU directives
- Too many categories created ambiguity
- High threshold (0.50) was too strict

**Conclusion**: Approach failed due to recall issues.

📄 See: `SEMANTIC_EVALUATION_REPORT.md` for detailed results

---

### Phase 2: Improved 2-Category Semantic Prefilter ✅

**Hypothesis**: Simplify to binary decision with broader positive definition and lower thresholds.

**Changes**:
1. **2 categories only** (vs 6) - clearer binary decision
2. **Broader positive definition** - includes conservation, circular economy, policy, biodiversity
3. **Lower thresholds** (0.30-0.45) - better recall
4. **Proper recall metrics** - tracks % of high-scoring articles caught

**Categories**:
1. **Positive**: sustainability, renewable energy, climate solutions, environmental conservation, circular economy, biodiversity, green technology, and sustainability policy
2. **Negative**: other topics including sports, entertainment, politics, and general news

**Results** (threshold 0.30):
- **Recall**: 97.9% ✅
- **FP Rate**: 22.3%
- **Improvement**: 82.9% better recall vs 6-category

**Comparison to Keyword Baseline**:
- **Keyword**: 100% recall, 23.2% FP rate
- **Semantic**: 97.9% recall, 22.3% FP rate
- **Difference**: Only 4% FP reduction, loses 2.1% recall

📄 See: `SEMANTIC_IMPROVED_EVALUATION.md` for detailed results

---

## Performance Comparison Table

| Filter | Recall | Miss Rate | FP Rate | Precision | Speed | Notes |
|--------|--------|-----------|---------|-----------|-------|-------|
| **Keyword** | 100.0% | 0.0% | 23.2% | 64.4% | 0.0s | **Simple, no misses** |
| Semantic-6cat (t=0.50) | ~15% | ~85% | 2.1% | N/A | N/A | ❌ Blocks good articles |
| Semantic-2cat (t=0.30) | 97.9% | 2.1% | 22.3% | 65.9% | 16.6s | ✅ Fixed recall |

---

## False Negative Analysis (Semantic 2-Category, t=0.30)

The 2.1% of high-scoring articles that were blocked:

**Example 1: [7.0/10]** The Indian Innovations Bringing Big [sustainability solutions]
- **Reason**: Classified as "other" category
- **Issue**: Legitimate sustainability innovation article

**Example 2: [5.7/10]** Hitmusical Moulin Rouge! sowieso tot eind mei te zien in Beatrix Theater
- **Reason**: Classified as "other" category
- **Note**: This appears to be entertainment, but oracle scored it 5.7 (border case)

**Example 3: [6.4/10]** Designing a self [sustaining system]
- **Reason**: Classified as "other" category
- **Issue**: Sustainability-related design article

**Analysis**: These misses show that even with broader definitions, semantic classification can still miss edge cases that the oracle would score as relevant.

---

## Cost-Benefit Analysis

### Semantic Prefilter Benefits
- ✅ Reduces FP rate by 4% (23.2% → 22.3%)
- ✅ Slightly higher precision (65.9% vs 64.4%)

### Semantic Prefilter Costs
- ❌ Loses 2.1% of good articles (97.9% vs 100% recall)
- ❌ Adds 16.6s processing time per 1K articles
- ❌ Requires embedding model + GPU for deployment
- ❌ More complex implementation and maintenance
- ❌ Harder to debug and explain decisions

### Keyword Prefilter Benefits
- ✅ 100% recall (no missed articles)
- ✅ Instant (0.0s processing)
- ✅ Simple, transparent, maintainable
- ✅ Easy to debug and adjust
- ✅ Oracle handles FP filtering with scoring

### Keyword Prefilter Costs
- ⚠️ 23.2% FP rate (acceptable - oracle scores them low)
- ⚠️ Slightly more oracle API calls

---

## Decision Rationale

### Why Keyword Prefilter Won

**1. Marginal Improvement**
- 4% FP reduction doesn't justify added complexity
- Oracle already filters FPs by scoring them low (1.0-2.0)
- Student model benefits from seeing full spectrum of relevance

**2. Recall is Critical**
- Missing 2.1% of good articles is unacceptable for training data
- Training dataset should have ALL high-quality examples
- False negatives hurt model performance more than false positives

**3. Simplicity Matters**
- Keyword prefilter: ~50 lines of code, no dependencies
- Semantic prefilter: embedding model, GPU, complex logic
- Easier to debug: "why did this pass?" → check keyword matches
- Easier to maintain: add/remove keywords vs retrain embeddings

**4. Oracle Does the Heavy Lifting**
- Oracle's job is to score relevance accurately
- FP reduction at prefilter stage has diminishing returns
- Better to let oracle see everything and score correctly

**5. Cost is Acceptable**
- 23.2% FP rate = ~2,320 FPs in 10K articles
- Cost: ~$17.40 to score these FPs (assuming $0.0075/article)
- Semantic would save: ~$3 per 10K articles (4% reduction)
- Not worth the complexity for $3 per 10K

---

## Lessons Learned

### ✅ What Worked

1. **2-category approach > 6-category** - Binary decisions are clearer
2. **Broader definitions** - Including conservation, policy, circular economy improved recall
3. **Lower thresholds** - 0.30-0.45 range gave good recall
4. **Proper recall metrics** - Tracking high-scoring articles (>5.0) caught the recall issue

### ❌ What Didn't Work

1. **Too many categories** - Creates ambiguity and misclassification
2. **Narrow positive definition** - Missed legitimate sustainability content
3. **High thresholds** - 0.50 was too strict
4. **Optimizing for FP reduction alone** - Recall matters more for training data

### 🧠 Key Insights

1. **Prefilter purpose matters** - For production filtering: low FP. For training data: high recall.
2. **Context matters** - Articles need full context for accurate classification
3. **Embeddings have limits** - Even good embeddings can miss edge cases
4. **Simplicity has value** - When improvement is marginal, choose simpler solution

---

## Final Configuration

### Approved: Keyword Prefilter with Negative Keyword Blocking

**Positive Keywords** (wide net):
- sustainability, renewable, solar, wind, climate, carbon, emissions, biodiversity, conservation, etc.
- Uses substring matching (accept 'oil' in 'turmoil')

**Negative Keywords** (conservative blocking):
- Sports: soccer, football match, touchdown, nfl, nba, etc.
- Entertainment: kardashian, baldwin, reality show, red carpet, etc.
- Lifestyle: wedding dress, makeup tutorial, horoscope, etc.
- **Threshold**: 2+ negative keyword occurrences to block

**Performance**:
- Recall: 100% (catches everything)
- FP Rate: ~23.2% (acceptable for oracle scoring)
- Speed: Instant
- Maintainability: Excellent

📄 See: `PREFILTER_STRATEGY.md` for implementation details

---

## References

- **6-category evaluation**: `SEMANTIC_EVALUATION_REPORT.md`
- **2-category evaluation**: `SEMANTIC_IMPROVED_EVALUATION.md`
- **Prefilter strategy**: `PREFILTER_STRATEGY.md`
- **GPU run instructions**: `GPU_RUN_INSTRUCTIONS.md`
- **Semantic prefilter code**: `semantic_prefilter.py` (archived for reference)
- **Keyword prefilter code**: `prefilter.py` (active)

---

## Appendix: When to Reconsider Semantic Prefiltering

Semantic prefiltering might be worth revisiting if:

1. **FP rate becomes critical** - If oracle API costs become prohibitive (>$100 per 10K articles)
2. **Production deployment** - If deploying model for real-time filtering (not training)
3. **Much better embeddings** - If new embedding models achieve >99.5% recall
4. **Different use case** - If filtering goal changes (e.g., user-facing vs training)

For now, keyword prefiltering with oracle scoring provides the best balance of simplicity, recall, and cost-effectiveness for training data generation.
