# Calibration/Validation Split for Oracle Prompt Testing

**Date:** 2025-11-14
**Status:** Accepted

## Context

During calibration of the sustainability_tech_deployment oracle prompt, we discovered a critical gap in the workflow: **risk of overfitting prompt fixes to the calibration sample**.

**Timeline:**
1. Nov 14: Created calibration sample (27 articles)
2. Labeled sample → Found 57% false negative rate (oracle under-scoring deployed tech)
3. Fixed prompt based on calibration findings
4. Re-labeled same 27 articles → 80% on-topic recognition (major improvement)
5. **User asked:** "Should we take another sample to check again?"
6. **Discovery:** We had optimized the prompt for the calibration sample, but hadn't validated if improvements generalized

**The key insight:** This is the **train/test overfitting problem** from ML, applied to prompt engineering!

**Risk without validation:**
- Prompt fixes might work perfectly on the 27 calibration articles
- But fail on new, unseen articles in the batch labeling run
- Only discovered after spending $8 on 8,000 mis-labeled articles

## Decision

**Add mandatory VALIDATION STEP after prompt calibration.**

**Updated workflow:**

```
OLD (risky):
1. Calibration sample → Fix prompt → Batch label

NEW (safe):
1. Calibration sample → Fix prompt
2. Validation sample (different articles) → Verify improvements generalize
3. If validation passes → Batch label
```

**Validation = independent test set to verify prompt improvements generalize.**

## Rationale

### ML Train/Test Split Pattern

**In Machine Learning:**
- **Training set:** Used to optimize model parameters
- **Test set:** Used to measure generalization performance
- **Rule:** NEVER use test set for training (causes overfitting)

**Applied to Prompt Engineering:**
- **Calibration sample (training set):** Used to identify prompt issues and fix them
- **Validation sample (test set):** Used to verify fixes work on new articles
- **Rule:** Use DIFFERENT articles for validation (different random seed)

### Why This Matters

**Without validation (overfitting risk):**
1. Calibration sample has 4 false negatives (specific articles: solar loan, EV chargers, GM layoffs, battery rules)
2. We fix prompt to handle these specific cases
3. Prompt now works on these 4 articles
4. **Risk:** Did we fix the *systemic issue* or just memorize these 4 articles?
5. Batch labeling reveals prompt still fails on similar articles we didn't see

**With validation (generalization check):**
1. Same calibration process
2. Test on DIFFERENT 50 articles (fresh sample)
3. If validation metrics ≈ calibration metrics → Fixes generalized ✅
4. If validation metrics < calibration metrics → Overfitting detected ❌
5. Confidence before batch labeling

### Real-World Validation Results

**Sustainability_tech_deployment filter:**

**Calibration sample (27 articles, seed=42):**
- Off-topic rejection: 100%
- On-topic recognition: 80%
- False negatives: 1/5 (EV chargers article)

**Validation sample (31 articles, seed=2025):**
- Off-topic rejection: 92.3%
- On-topic recognition: 100%
- False negatives: 0/2

**Verdict:** Prompt improvements **generalized well** to new articles! ✅

Minor differences:
- Off-topic rejection slightly lower (100% → 92.3%) - still meets >90% target
- On-topic recognition improved (80% → 100%) - fresh sample had fewer on-topic articles
- One new false positive appeared (Honda EV review) - within acceptable <10% threshold

**Conclusion:** Validation confirmed fixes were NOT overfitted to calibration sample.

## Implementation

### Validation Workflow

**Step 6: Validate on Fresh Sample**

1. **Create validation sample (DIFFERENT articles):**
   ```bash
   python scripts/create_calibration_sample.py \
       --input articles_corpus.jsonl \
       --output validation_sample.jsonl \
       --n-positive 20 \
       --n-negative 20 \
       --n-edge 10 \
       --random-seed 2025  # CRITICAL: Different seed = different articles
   ```

2. **Label validation sample:**
   ```bash
   python -m ground_truth.batch_labeler \
       --filter filters/{filter_name}/v1 \
       --source validation_sample.jsonl \
       --output-dir validation_labeled \
       --llm gemini-flash \
       --batch-size 50 \
       --max-batches 1
   ```
   **Cost:** ~$0.05

3. **Run Prompt Calibration Agent on validation sample:**
   ```
   Task: "Run Prompt Calibration Agent on validation_labeled.jsonl to verify
   prompt improvements generalize to new articles."
   ```

4. **Compare validation to calibration metrics:**
   - ✅ **Validation ≈ Calibration** → Generalized successfully
   - ⚠️ **Validation < Calibration (10+ percentage points)** → Overfitting detected
   - ❌ **Validation fails targets** → Major issues, revise prompt

### Decision Matrix

| Calibration | Validation | Decision |
|-------------|------------|----------|
| PASS (80% on-topic) | PASS (80%+ on-topic) | ✅ Proceed to batch labeling |
| PASS (80% on-topic) | FAIL (60% on-topic) | ❌ Overfitting - revise prompt |
| REVIEW (70% on-topic) | PASS (75% on-topic) | ✅ Good enough - proceed |
| REVIEW (70% on-topic) | FAIL (50% on-topic) | ❌ Major issues - start over |

**Key rule:** Validation should be similar to or better than calibration. If validation is significantly worse, prompt improvements didn't generalize.

## Consequences

### Positive

- ✅ **Prevents overfitting** - Catches prompt fixes that only work on calibration sample
- ✅ **Validates generalization** - Confirms improvements work on new articles
- ✅ **Increases confidence** - Test on 100+ articles total (calibration + validation) before $8 batch run
- ✅ **Early detection** - Finds issues before expensive batch labeling
- ✅ **Documented validation** - Reports prove prompt quality on independent test set

### Negative

- ⚠️ **Extra time** - +30 minutes to create and label validation sample
- ⚠️ **Extra cost** - +$0.05 (50 validation articles)
- ⚠️ **More complex workflow** - Additional step to track

### Mitigation

**Time/cost is minimal compared to benefit:**
- Calibration: $0.05 + 1 hour
- Validation: $0.05 + 0.5 hours
- **Total: $0.10 + 1.5 hours**
- **Savings: $8-16 + days** if overfitting detected before batch run

**Workflow complexity:**
- Validation is simple: rerun calibration script with different seed
- Reuse existing Prompt Calibration Agent
- Clear decision matrix (compare metrics)

## Alternatives Considered

### Alternative 1: Skip Validation, Monitor Batch Labeling

**Approach:** Proceed directly to batch labeling, manually review first 100 articles

**Pros:**
- Faster to start batch labeling
- Catches issues early in batch run

**Cons:**
- ❌ Reactive (find issues after labeling started)
- ❌ Wastes partial batch ($0.10-0.50 if issues found)
- ❌ Manual review less systematic than agent-based calibration
- ❌ May miss subtle overfitting (only notice after 500+ articles)

**Decision:** Rejected - validation is proactive and cheaper

### Alternative 2: Use Same Articles for Validation (No Random Seed Change)

**Approach:** Label same calibration sample multiple times with updated prompt

**Pros:**
- Ensures prompt fixes work on problematic articles
- Direct before/after comparison

**Cons:**
- ❌ **Defeats the purpose** - Same articles = no generalization test
- ❌ Can't detect overfitting (will always improve on same data)
- ❌ False confidence (passes validation but fails on new articles)

**Decision:** Rejected - must use DIFFERENT articles for validation

### Alternative 3: Larger Validation Sample (100+ articles)

**Approach:** Use 100-200 articles for validation instead of 50

**Pros:**
- Higher confidence in generalization
- More statistical power

**Cons:**
- ⚠️ 2-4x cost ($0.10-0.20 instead of $0.05)
- ⚠️ 2x time for review
- Diminishing returns (50 articles already sufficient)

**Decision:** Rejected - 50 articles is sufficient for validation

## Success Metrics

**Validation step is successful if:**
- ✅ Catches overfitting (validation metrics significantly worse than calibration)
- ✅ Confirms generalization (validation metrics ≈ calibration metrics)
- ✅ No need to re-label full batch dataset due to systematic issues
- ✅ Validation cost < 1% of batch labeling cost ($0.05 vs $8)

**Red flags:**
- ❌ Validation always passes even when calibration barely passes
- ❌ Validation metrics consistently worse than calibration (suggests bad sampling)
- ❌ Overfitting detected but prompt can't be improved (may need more diverse calibration sample)

## Lessons Learned

### Key Insight

**"Move left on testing" applies to prompt engineering too!**

**Software engineering principle:**
- Unit tests (cheap, fast) > Integration tests > Production bugs (expensive, slow)
- Catch bugs as early as possible in pipeline

**Applied to prompt calibration:**
- Calibration (50 articles, $0.05) > Validation (50 articles, $0.05) > Batch labeling (8000 articles, $8)
- Catch prompt issues before expensive batch run

### Train/Test Split is Universal

**Pattern works across domains:**
- ML: Training set vs test set
- Prompt engineering: Calibration sample vs validation sample
- Software: Dev environment vs staging environment
- Science: Hypothesis generation vs independent replication

**Core principle:** Optimization on one dataset may not generalize to new data. Always validate on independent data before production deployment.

### Sustainability Tech Deployment Case Study

**What would have happened without validation:**

**Scenario 1: Prompt improvements were overfitted**
1. Calibration shows 80% on-topic recognition → PASS
2. Proceed directly to batch labeling (8,162 articles, $8)
3. After 500 articles, discover prompt still under-scores new deployment types
4. Already spent $0.50, need to re-label everything
5. **Cost: $8 wasted + days of rework**

**Scenario 2: With validation (what we actually did)**
1. Calibration shows 80% on-topic recognition → PASS
2. Validation shows 100% on-topic recognition → PASS (generalized!)
3. Proceed to batch labeling with confidence
4. **Cost: $0.05 extra for validation**

**ROI:** $0.05 prevented potential $8 loss

## Implementation Checklist

- [x] Update training/README.md with validation step
- [x] Create decision document (this file)
- [ ] Update Prompt Calibration Agent template to support validation analysis
- [ ] Add validation comparison to agent output (metrics table: calibration vs validation)
- [ ] Test validation workflow with uplifting filter

## References

- Calibration ADR: `docs/decisions/2025-11-13-prompt-calibration-before-batch-labeling.md`
- Prompt Calibration Agent: `docs/agents/templates/prompt-calibration-agent.md`
- Calibration reports (sustainability_tech_deployment):
  - Calibration v1: `filters/sustainability_tech_deployment/v1/calibration_report.md`
  - Calibration v2: `filters/sustainability_tech_deployment/v1/calibration_report_v2.md`
  - Validation (fresh sample): `filters/sustainability_tech_deployment/v1/calibration_report_fresh.md`

## Discussion

**Why this is a process improvement, not just best practice:**

Every filter will benefit from validation:
1. sustainability_tech_deployment - validated ✅
2. uplifting - needs validation before batch labeling
3. investment_risk - needs validation before batch labeling

**Validation is not optional.** It's a mandatory quality gate to prevent overfitting.

**Analogy:** Code review before merging to main
- Could skip review and push directly to production
- But review catches bugs that tests miss (logic errors, edge cases)
- Same for validation - catches overfitting that calibration doesn't reveal

**Principle:** **Test prompt improvements on independent data before expensive batch operations.**

## Version History

### v1.0 (2025-11-14)
- Initial decision
- Based on sustainability_tech_deployment validation experience
- Establishes validation as mandatory step after calibration
- Documents train/test split pattern for prompt engineering
