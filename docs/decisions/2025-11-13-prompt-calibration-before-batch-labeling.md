# Prompt Calibration Required Before Batch Scoring

**Date:** 2025-11-13
**Status:** Accepted

## Context

While generating example outputs for the sustainability_tech_deployment filter, we discovered a critical flaw: **the oracle severely mis-scored the dataset**.

**Symptoms:**
- Generic IT infrastructure (AWS IAM, Excel, Linux) scored 10.0 ("mass deployment")
- Actual climate tech (solar energy) scored only ~8.3
- Off-topic articles (toothbrushes, photo editing) scored 9.0+

**Root cause:** Oracle prompt lacked explicit scope definition. Oracle interpreted "technology deployment" too broadly, scoring ANY mature technology as climate tech.

**Cost of failure:**
- $8 wasted on 8,162 mis-labeled articles
- Days wasted training model on garbage data
- Filter examples show wrong content
- Need to re-label entire dataset

**This was discovered AFTER:**
- Batch labeling complete
- Model trained
- Infrastructure built
- Documentation written

**The question:** How do we prevent this in future filters?

## Decision

**Add mandatory PROMPT CALIBRATION step before batch scoring.**

**Workflow change:**

```
OLD (broken):
1. Write prompt → 2. Batch label 8k articles → 3. Train → 4. Discover prompt is broken

NEW (correct):
1. Write prompt → 2. Calibrate (50 articles + review) → 3. Batch label → 4. Train
```

**Calibration must pass before batch scoring starts.**

## Rationale

### Cost-Benefit Analysis

**Prompt calibration cost:**
- $0.05-0.10 (50-100 test articles × $0.001)
- 1-2 hours review time
- 2-3 prompt iterations
- **Total: ~$0.15 + 3 hours**

**Batch labeling without calibration:**
- $8 for 8,162 articles
- If prompt broken → **100% wasted**
- Re-labeling cost: **another $8**
- Time wasted: Days of processing + training

**ROI:** Spend $0.15 + 3 hours to save $8-16 + days of work

### What Calibration Would Have Caught

**Test sample (50 articles):**
- 20 obvious climate tech (solar, wind, EVs)
- 20 obvious off-topic (AWS, Excel, toothbrushes)
- 10 edge cases (oil news, carbon credits)

**Review would have found:**

```
❌ FAILURE: Off-topic articles scoring >5.0

Examples:
- AWS IAM: 10.0 (should be 0-2)
  Oracle: "Mature technology, mass deployed"
  → Prompt missing scope definition

- Excel: 9.4 (should be 0-2)
  Oracle: "Widely deployed data tool"
  → Prompt doesn't specify climate/sustainability only

- Toothbrush: 9.25 (should be 0-2)
  Oracle: "Mass-deployed hygiene technology"
  → Prompt completely off-topic
```

**Red flag threshold:** If >20% of off-topic articles score >5.0 → Prompt is broken

**Fix before batch scoring:**
```markdown
**SCOPE: Climate & Sustainability Technology ONLY**

**OUT OF SCOPE (score 0-2 on ALL dimensions):**
- Generic IT infrastructure
- Office productivity software
- Generic hardware
```

**Re-test → Pass → Proceed to batch scoring**

### Why This Step Is Critical

**Garbage in, garbage out:**
- Models learn from ground truth labels
- If oracle labels are wrong, model will be wrong
- No amount of training can fix bad labels

**Batch labeling is expensive:**
- Can't easily "fix" 8,162 labels after the fact
- Re-labeling doubles cost ($8 → $16)
- Iterative prompt fixing is impractical at scale

**Calibration is cheap insurance:**
- Small sample (50 articles) reveals systematic errors
- Quick to iterate (minutes to update prompt, re-test)
- Validates oracle before expensive batch run

## Prompt Calibration Workflow

### Step 1: Create Calibration Sample

**Stratified sampling (50-100 articles):**

```python
calibration_sample = {
    "positive_examples": 20,   # Obvious in-scope (solar, wind, EVs, batteries)
    "negative_examples": 20,   # Obvious out-of-scope (AWS, Excel, generic tech)
    "edge_cases": 10,          # Borderline (oil news, carbon credits, green hydrogen)
}
```

**Why stratified?**
- Random sample might be all vaporware or all off-topic
- Need coverage of score ranges and topics
- Edge cases reveal prompt ambiguities

**Source:** Sample from unlabeled article corpus before batch scoring

### Step 2: Label Calibration Sample

```bash
python scripts/label_batch.py \
    --filter filters/{filter_name}/v1 \
    --input calibration_sample.jsonl \
    --output calibration_labeled.jsonl \
    --oracle gemini-flash
```

**Cost:** 50 articles × $0.001 = $0.05

### Step 3: Review Calibration Results

**Two approaches:**

**Option A: Human Review (Manual)**
- Read labeled articles
- Check if scores align with expectations
- Identify systematic errors
- Suggest prompt improvements

**Option B: Agent Review (Semi-Automated) ⭐ RECOMMENDED**
- Use Prompt Calibration Agent (see template)
- Agent analyzes labels systematically
- Flags problematic patterns
- Generates calibration report
- Human reviews only flagged issues

**Calibration Agent advantages:**
- Faster than manual review
- Consistent criteria
- Structured output
- Scales to larger samples

### Step 4: Fix Prompt Based on Findings

**Common issues found in calibration:**

1. **Scope ambiguity** → Add explicit IN SCOPE / OUT OF SCOPE sections
2. **Wrong dimension emphasis** → Adjust dimension descriptions
3. **Edge case handling** → Add examples of borderline cases
4. **Scale miscalibration** → Clarify what 10.0 vs 7.0 means
5. **Prompt structure issue** → Restructure with inline filters (if oracle ignores top-level rules)
   - **Symptom:** False positives persist despite having correct OUT OF SCOPE rules
   - **Root cause:** Oracle skips top-level sections, jumps directly to dimensional scoring
   - **Solution:** Move critical filters inline with each dimension definition (see [2025-11-14-inline-filters-for-fast-models.md](2025-11-14-inline-filters-for-fast-models.md))
   - **Example:** Uplifting v3 → v4 reduced false positives from 87.5% to 0% by restructuring

### Step 5: Re-test Until Validated

**Validation criteria:**

```python
# On calibration sample:
off_topic_high_scores = count(off_topic articles with score > 5.0)
on_topic_low_scores = count(on_topic articles with score < 5.0)

# Pass threshold:
assert off_topic_high_scores < 10%  # <5 out of 50
assert on_topic_low_scores < 20%    # <10 out of 50
```

**If validation fails:**
1. Update prompt
2. Re-label same 50 articles (cost: $0.05)
3. Review again
4. Repeat until pass

**Typical iterations:** 2-3 rounds ($0.10-0.15 total)

### Step 6: Proceed to Batch Scoring

**Only after calibration passes:**
```bash
# Calibration validated ✓
# Now safe to batch label full dataset

python scripts/label_batch.py \
    --filter filters/{filter_name}/v1 \
    --input articles_to_label.jsonl \
    --output labeled_articles.jsonl
```

**Confidence:** High - prompt is validated on diverse sample

## Implementation

### Required Components

1. **Calibration sample creation script**
   - `sandbox/analysis_scripts/create_calibration_sample.py`
   - Stratified sampling from article corpus
   - Configurable positive/negative/edge case ratios

2. **Prompt Calibration Agent template**
   - `docs/agents/templates/prompt-calibration-agent.md`
   - Systematic review of labeled calibration sample
   - Flags off-topic high scores, on-topic low scores
   - Generates calibration report with pass/fail

3. **Updated workflow documentation**
   - Add calibration step to training/README.md
   - Create docs/workflows/prompt-calibration.md

4. **Calibration report template**
   - Documents validation results
   - Stored in filters/{name}/v1/calibration_report.md
   - Includes sample review, issues found, prompt changes

### Integration with Existing Workflow

**Current training/README.md section:**
```markdown
## Quick Start

### Prerequisites
Install dependencies...

### Step 1: Prepare Dataset  ← ADD CALIBRATION HERE
```

**Updated:**
```markdown
## Quick Start

### Prerequisites
Install dependencies...

### Step 1: Calibrate Oracle Prompt ⭐ CRITICAL
Before batch scoring, validate the oracle prompt on a small sample.

See: docs/workflows/prompt-calibration.md

### Step 2: Prepare Dataset (after calibration passes)
```

## Consequences

### Positive

- ✅ **Catches prompt errors early** - Before expensive batch scoring
- ✅ **Saves money** - $0.15 calibration vs $8-16 re-labeling
- ✅ **Saves time** - Hours of review vs days of re-processing
- ✅ **Higher quality ground truth** - Validated oracle before batch run
- ✅ **Iterative improvement** - Quick to fix prompt and re-test
- ✅ **Documented validation** - Calibration report proves prompt quality

### Negative

- ⚠️ **Extra upfront work** - 3 hours for calibration
- ⚠️ **Delays batch scoring** - Can't start until calibration passes
- ⚠️ **New workflow step** - Team must learn calibration process

### Mitigation

**Streamline calibration:**
- Use Prompt Calibration Agent (reduces manual review time)
- Pre-built calibration sample templates
- Clear pass/fail criteria (minimize subjectivity)

**Make it easy to adopt:**
- Document workflow clearly
- Provide agent template
- Show ROI ($0.15 saves $8-16)

## Alternatives Considered

### Alternative 1: Skip Calibration, Fix Prompt Reactively

**Approach:** Label full dataset, discover issues during training, re-label

**Pros:**
- No upfront calibration work
- Faster to start batch scoring

**Cons:**
- ❌ Wastes money (double labeling cost)
- ❌ Wastes time (days of processing wasted)
- ❌ Late discovery (after model training)
- ❌ Risk: Trained model before realizing data is garbage

**Decision:** Rejected - Penny wise, pound foolish

### Alternative 2: Human Expert Labels Subset, Compare to Oracle

**Approach:** Expert labels 100 articles, compare to oracle labels, measure agreement

**Pros:**
- Gold standard comparison
- Quantitative validation (inter-rater agreement)

**Cons:**
- Requires domain expert time (expensive)
- Expert may not catch systematic prompt issues
- Slower than agent review

**Decision:** Rejected as primary approach, but useful for final validation

### Alternative 3: Continuous Calibration During Batch Scoring

**Approach:** Label in batches of 500, review each batch, adjust prompt mid-run

**Pros:**
- Catches issues during labeling
- Can fix prompt and re-label remainder

**Cons:**
- ❌ Inconsistent labeling (earlier batches use worse prompt)
- ❌ Harder to track changes
- ❌ Still wastes partial labeling cost

**Decision:** Rejected - Better to validate upfront

## Success Metrics

**Prompt calibration is successful if:**
- ✅ Catches all major prompt issues before batch scoring
- ✅ Final batch-labeled dataset has <5% mis-labeled articles
- ✅ No need to re-label full dataset
- ✅ Model training succeeds on first attempt
- ✅ Example outputs show expected content

**Red flags:**
- ❌ Calibration passes but batch labels still wrong
- ❌ Calibration takes >1 day (too slow)
- ❌ Multiple re-labeling runs needed

**If red flags appear:** Improve calibration criteria or sample diversity

## Lessons Learned

### What Went Wrong (sustainability_tech_deployment)

**Timeline:**
1. Oct 28: Wrote prompt (without scope definition)
2. Nov 2-11: Batch labeled 8,162 articles ($8)
3. Nov 12-13: Trained model, built infrastructure
4. Nov 13: Generated examples → **DISCOVERED ORACLE IS BROKEN**

**Failure mode:**
- Assumed prompt was correct
- No validation before batch scoring
- Discovered error too late to easily fix

**Cost:**
- $8 wasted on bad labels
- Days of work wasted
- Need to re-label everything

### What We Should Do (uplifting filter)

**Timeline:**
1. Write prompt
2. **⭐ CALIBRATE (50 articles, agent review, iterate)** ← NEW STEP
3. Batch label (after calibration passes)
4. Train model
5. Evaluate

**Expected outcome:**
- Prompt validated before expensive labeling
- High-quality ground truth
- Model training succeeds
- Example outputs show correct content

### Key Insight

**"Move left on testing"** - Catch errors as early as possible

**Software engineering principle:**
- Unit tests > Integration tests > Production bugs
- Cheaper to find bugs earlier in pipeline

**Applied to ML labeling:**
- Prompt calibration (50 articles) > Batch labeling (8k articles) > Model training
- Cheaper to find prompt errors before batch run

## Implementation Checklist

- [ ] Create calibration sample script (`sandbox/analysis_scripts/create_calibration_sample.py`)
- [ ] Create Prompt Calibration Agent template (`docs/agents/templates/prompt-calibration-agent.md`)
- [ ] Update training/README.md with calibration step
- [ ] Create detailed calibration workflow doc (`docs/workflows/prompt-calibration.md`)
- [ ] Test calibration workflow with uplifting filter
- [ ] Create calibration report template

## References

- Mis-labeled dataset: `datasets/scored/sustainability_tech_deployment/labeled_articles.jsonl`
- Fixed prompt: `filters/sustainability_tech_deployment/v1/prompt-compressed.md` (updated 2025-11-13)
- Example outputs showing problem: `filters/sustainability_tech_deployment/v1/examples.md`
- Related: Inter-rater agreement in ML labeling best practices

## Discussion

**Why this is a process improvement, not just a one-time fix:**

Every new filter will face this risk:
1. uplifting filter - needs calibration before labeling
2. investment_risk filter - needs calibration before labeling
3. education filter - needs calibration before labeling

**Calibration is not optional.** It's a mandatory quality gate.

**Analogy:** Code review before merging to main
- Could skip review, push directly to production
- But code review catches bugs cheaply
- Same for prompt calibration - catches oracle bugs cheaply

**Principle:** **Validate oracle on small sample before expensive batch operations.**

## Version History

### v1.2 (2025-11-14)
- Added issue #5: Prompt structure (inline filters pattern)
- See: `docs/decisions/2025-11-14-inline-filters-for-fast-models.md`
- Addresses case where oracle ignores top-level OUT OF SCOPE rules
- Solution: Move critical filters inline with each dimension definition
- Validated on uplifting filter: 87.5% → 0% false positive rate after restructuring

### v1.1 (2025-11-14)
- Added validation step after calibration (train/test split pattern)
- See: `docs/decisions/2025-11-14-calibration-validation-split.md`
- Updated workflow: Calibration (train) → Validation (test) → Batch labeling
- Prevents overfitting prompt fixes to calibration sample
- Validated on sustainability_tech_deployment filter (calibration: 80% on-topic, validation: 100% on-topic)

### v1.0 (2025-11-13)
- Initial decision
- Based on sustainability_tech_deployment labeling failure
- Establishes calibration as mandatory step
- Proposes agent-assisted review
