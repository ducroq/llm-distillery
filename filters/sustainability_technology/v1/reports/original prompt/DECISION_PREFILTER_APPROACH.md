# Decision: Prefilter Approach for sustainability_technology v1

**Date**: 2025-11-24
**Decision Maker**: Project Team
**Status**: ✅ APPROVED

---

## Decision

**Use keyword-based prefilter with negative keyword blocking for sustainability_technology v1 training data generation.**

Do NOT use semantic prefiltering at this stage.

---

## Context

We evaluated three prefiltering approaches to reduce false positives before oracle scoring:

1. **6-category semantic prefilter** - Failed (15% recall)
2. **2-category semantic prefilter** - Works (97.9% recall) but marginal benefit
3. **Keyword prefilter with negative blocking** - Simple, 100% recall, acceptable FP rate

---

## Options Considered

### Option A: Keyword Prefilter ✅ SELECTED

**Description**:
- Positive keywords for sustainability content (wide net, substring matching)
- Negative keywords for obvious off-topic content (2+ occurrence threshold)
- No semantic/ML component

**Pros**:
- ✅ 100% recall - catches all good articles
- ✅ Simple, fast, maintainable
- ✅ Transparent and debuggable
- ✅ No GPU/embedding dependencies
- ✅ Easy to adjust (add/remove keywords)

**Cons**:
- ⚠️ 23.2% false positive rate
- ⚠️ More oracle API calls

**Metrics**:
- Recall: 100%
- FP Rate: 23.2%
- Precision: 64.4%
- Speed: Instant
- Complexity: Low

---

### Option B: 2-Category Semantic Prefilter ❌ REJECTED

**Description**:
- Binary classification using sentence embeddings
- Categories: sustainability-related vs other topics
- Threshold 0.30 for best recall

**Pros**:
- ✅ Slightly better FP rate (22.3% vs 23.2%)
- ✅ Slightly better precision (65.9% vs 64.4%)
- ✅ More sophisticated approach

**Cons**:
- ❌ Loses 2.1% of good articles (97.9% recall)
- ❌ Adds 16.6s processing per 1K articles
- ❌ Requires embedding model + GPU
- ❌ More complex to maintain and debug
- ❌ Harder to explain why articles were blocked

**Metrics**:
- Recall: 97.9%
- FP Rate: 22.3%
- Precision: 65.9%
- Speed: 16.6s per 1K articles
- Complexity: High

**Cost-Benefit**: 4% FP reduction (saves ~$3 per 10K articles) doesn't justify:
- Missing 2.1% of good training examples
- Added complexity and maintenance burden
- GPU deployment requirement
- Slower processing

---

### Option C: 6-Category Semantic Prefilter ❌ REJECTED

**Description**:
- Multi-class classification with 6 categories
- Block "general news" and other off-topic categories

**Pros**:
- ✅ Very low FP rate (2.1%)

**Cons**:
- ❌ **Critical failure**: Only 15% recall
- ❌ Blocks 85% of good articles
- ❌ Too many categories create ambiguity
- ❌ "General news" catches legitimate sustainability content

**Metrics**:
- Recall: ~15% ⚠️ UNACCEPTABLE
- FP Rate: 2.1%
- Complexity: Very High

**Conclusion**: Fundamentally flawed approach for this use case.

---

## Decision Criteria

### Priority 1: Recall (Weight: 50%)
- **Requirement**: ≥95% recall for training data quality
- **Winner**: Keyword (100%) > Semantic-2cat (97.9%) >> Semantic-6cat (15%)

### Priority 2: Simplicity (Weight: 25%)
- **Requirement**: Easy to maintain, debug, and adjust
- **Winner**: Keyword (simple) >> Semantic (complex)

### Priority 3: Cost-Effectiveness (Weight: 15%)
- **Requirement**: Balance API costs vs complexity
- **Winner**: Keyword (acceptable cost, no added complexity)

### Priority 4: FP Reduction (Weight: 10%)
- **Requirement**: Nice to have, but oracle handles scoring
- **Winner**: Semantic-2cat (22.3%) ≈ Keyword (23.2%)

---

## Rationale

### 1. Recall is Non-Negotiable

For training data generation, missing good articles (false negatives) is worse than including bad articles (false positives):

- **False Negatives**: Permanently lost training examples, hurt model performance
- **False Positives**: Oracle scores them low (1.0-2.0), model learns to reject them

The 2.1% recall loss in semantic prefiltering means missing ~300 good articles in a 10K dataset. This is unacceptable.

### 2. Marginal Improvement Doesn't Justify Complexity

**FP reduction**: 23.2% → 22.3% = 0.9 percentage points = 4% improvement

**Cost savings**: ~$3 per 10K articles (assuming $0.0075/article × 400 fewer FPs)

**Added complexity**:
- Embedding model deployment
- GPU infrastructure requirement
- Harder debugging ("why was this blocked?")
- Maintenance burden
- Dependency on embedding quality

**Verdict**: Not worth it for $3 per 10K articles.

### 3. Oracle is the Real Filter

The prefilter's job is to cast a wide net and reduce obvious garbage. The oracle's job is to score relevance accurately.

**Division of labor**:
- **Prefilter**: Remove obviously irrelevant (sports, celebrities, weddings)
- **Oracle**: Score sustainability relevance on 1-10 scale
- **Student model**: Learn from oracle's scores

Trying to do semantic filtering at prefilter stage duplicates oracle's work with worse accuracy.

### 4. Simplicity Enables Iteration

With keyword prefiltering:
- Add/remove keywords in 5 minutes
- Debug blocked articles by checking keyword matches
- No model retraining needed
- No GPU infrastructure required

With semantic prefiltering:
- Need to retrain/adjust embeddings
- Harder to understand why classifications fail
- GPU required for deployment
- More moving parts = more failure modes

### 5. Training Data Needs Full Spectrum

The student model benefits from seeing the full range of relevance:
- **10/10**: Perfect sustainability technology articles
- **7-9/10**: Strong sustainability relevance
- **4-6/10**: Moderate relevance or tangential
- **1-3/10**: Weak or irrelevant (learns to reject)

Aggressive prefiltering removes the 1-3 range, making the model less robust at rejecting false positives in production.

---

## Implementation

### Approved Configuration

**File**: `filters/sustainability_technology/v1/prefilter.py`

**Positive Keywords** (~50 terms):
- Core: sustainability, sustainable, renewable, solar, wind, climate, carbon, emissions, biodiversity, conservation
- Tech: electric vehicle, energy storage, green hydrogen, carbon capture
- Policy: net zero, paris agreement, eu taxonomy
- Uses substring matching

**Negative Keywords** (~53 terms, 3 categories):
- **Sports**: soccer, football match, touchdown, nfl, nba, premier league, etc.
- **Entertainment**: kardashian, baldwin, reality show, red carpet, grammy, etc.
- **Lifestyle**: wedding dress, makeup tutorial, horoscope, lottery, etc.

**Blocking Logic**:
1. Check positive keywords → If none found, BLOCK
2. Check negative keywords → If 2+ occurrences, BLOCK
3. Otherwise, PASS to oracle

**Expected Performance**:
- Recall: 100%
- FP Rate: ~23.2%
- Pass Rate: ~70-75%

📄 See: `PREFILTER_STRATEGY.md` for detailed implementation

---

## Success Metrics

### Training Data Quality (Post-Oracle)
- ✅ All high-relevance articles included (no false negatives from prefilter)
- ✅ Oracle scores distributed across full 1-10 range
- ✅ ~65% of passed articles score >3.0 (precision acceptable)

### Cost Efficiency
- ✅ Prefilter reduces corpus by ~25-30% (blocks obvious junk)
- ✅ Oracle API costs: ~$60-75 per 10K articles (acceptable budget)
- ✅ No additional GPU costs for prefiltering

### Maintainability
- ✅ Prefilter adjustments take <5 minutes
- ✅ Clear audit trail (keyword matches visible)
- ✅ No model retraining needed

---

## Next Steps

1. ✅ **Decision approved** - Use keyword prefilter
2. ⏳ **Generate 10K training dataset** - Run distillation with approved prefilter
3. ⏳ **Train student model** - Single-stage model on full relevance spectrum
4. ⏳ **Evaluate student** - Compare to oracle on held-out test set
5. ⏳ **Deploy to production** - If student achieves ≥0.90 oracle correlation

---

## Review & Reconsideration

### When to Reconsider Semantic Prefiltering

This decision should be revisited if:

1. **API costs become prohibitive** - If oracle costs exceed $200 per 10K articles
2. **Much better embeddings** - If new models achieve >99.5% recall with <10% FP rate
3. **Production deployment requirements** - If real-time filtering needs change
4. **Different use case** - If purpose shifts from training data to user-facing filtering

For now, keyword prefiltering is the right choice for training data generation.

### Review Schedule

- **After 10K dataset generation**: Verify FP rate is within expected range (20-25%)
- **After student training**: Assess if FP rate affected model quality
- **6 months from now**: Review if semantic embeddings have improved significantly

---

## Approval

**Decision**: Use keyword prefilter with negative keyword blocking

**Approved By**: Project Team
**Date**: 2025-11-24
**Status**: ✅ FINAL

---

## References

- **Evaluation summary**: `SEMANTIC_PREFILTER_EVALUATION_SUMMARY.md`
- **6-category results**: `SEMANTIC_EVALUATION_REPORT.md`
- **2-category results**: `SEMANTIC_IMPROVED_EVALUATION.md`
- **Prefilter strategy**: `PREFILTER_STRATEGY.md`
- **Implementation**: `prefilter.py`

---

## Appendix: Evaluation Data

### Comparison Table

| Metric | Keyword | Semantic-2cat | Semantic-6cat |
|--------|---------|---------------|---------------|
| Recall | 100% ✅ | 97.9% | ~15% ❌ |
| FP Rate | 23.2% | 22.3% | 2.1% |
| Precision | 64.4% | 65.9% | N/A |
| Speed | 0.0s ✅ | 16.6s | N/A |
| Complexity | Low ✅ | High | Very High |
| GPU Required | No ✅ | Yes | Yes |
| Maintainability | Excellent ✅ | Poor | Poor |

### Cost Analysis (per 10K articles)

**Scenario**: 10,000 raw articles → prefilter → oracle scoring

| Approach | Pass Rate | Oracle Calls | FPs | Oracle Cost | Infra Cost | Total |
|----------|-----------|--------------|-----|-------------|------------|-------|
| Keyword | 75% | 7,500 | ~1,740 | $56.25 | $0 | **$56.25** ✅ |
| Semantic | 72% | 7,200 | ~1,606 | $54.00 | ~$5 GPU | **$59.00** |
| Savings | -3% | -300 | -134 | **-$2.25** | | |

**Conclusion**: Semantic saves ~$2.25 per 10K but loses 2.1% recall and adds GPU costs. Not worth it.

---

## Document History

- **2025-11-24**: Initial decision - keyword prefilter approved
- **2025-11-23**: Semantic evaluation Phase 2 completed (2-category)
- **2025-11-22**: Semantic evaluation Phase 1 completed (6-category, failed)
- **2025-11-20**: Prefilter strategy with negative keywords approved
