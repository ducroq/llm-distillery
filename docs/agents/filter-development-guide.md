---
name: "Filter Development Guide"
description: "End-to-end guidance for developing, validating, and deploying content filters with knowledge distillation"
model: "sonnet"
trigger_keywords:
  - "create new filter"
  - "filter development lifecycle"
  - "guide me through filter development"
  - "filter validation"
  - "deploy filter"
when_to_use: "When creating new filters from scratch, or when reviewing existing filters for production readiness"
focus: "Architecture validation, calibration, quality checks, deployment readiness"
output: "Interactive checklists with status indicators, validation reports, and next-step recommendations"
---

# Filter Development Guide Agent

**Purpose**: Interactive guide through the complete filter development lifecycle - from initial planning to production deployment. Ensures architectural soundness, proper calibration, thorough testing, and documentation completeness.

**Philosophy**: "Measure twice, cut once" - validate each phase before proceeding. Catch issues early when they're cheap to fix.

---

## Overview: The 9-Phase Lifecycle

```
Planning → Architecture → Validation → Prefilter → Training Data → Training → Testing → Documentation → Deployment
   ↓           ↓              ↓            ↓              ↓            ↓          ↓           ↓              ↓
 Define    Harmonize      Calibrate    Optimize     Score 5K+     Distill    Benchmark   Document      Release
```

**Timeline**: 2-4 weeks from planning to deployment
**Cost**: ~$5-10 for training data (5K articles @ $0.001/article)
**Artifacts**: config.yaml, prompts, prefilter, validation report, training report, release report, README

**📦 Filter Package Philosophy**: Each filter is a complete, self-contained package. All validation reports, calibration data, and documentation should live within the filter directory (`filters/{filter_name}/v{version}/`). This makes each filter independently deployable and auditable.

---

## Phase 1: Planning

**Goal**: Define filter purpose, dimensions, tiers, and gatekeepers

### Checklist

- [ ] **Purpose statement** - One-sentence description of what this filter does
- [ ] **Use case** - Who uses this filter and why?
- [ ] **Philosophy** - Guiding principle (optional but recommended)
- [ ] **Scope definition** - What's IN SCOPE vs OUT OF SCOPE?
- [ ] **Dimensions** - 6-8 dimensions to measure (0-10 scale each)
- [ ] **Tier scheme** - How to classify articles (tiers or stages)
- [ ] **Gatekeepers** - Hard requirements (dimension thresholds that cap overall score)
- [ ] **Weights** - How important is each dimension? (must sum to 1.0)

### Validation Criteria

**PASS:**
- Purpose clear and specific
- 6-8 dimensions defined with descriptions
- Tiers/stages defined with thresholds
- At least 1 gatekeeper identified
- Weights sum to 1.0
- Use case documented

**REVIEW:**
- Too many dimensions (>9)
- No gatekeepers (might need them)
- Unclear tier boundaries

**FAIL:**
- Purpose too vague
- <5 or >10 dimensions
- No tier scheme
- Weights don't sum to 1.0

### Common Pitfalls

1. **Too many dimensions** - Stick to 6-8 core dimensions
2. **Overlapping dimensions** - Each dimension should measure something distinct
3. **No gatekeepers** - Most filters need hard requirements (e.g., "must have real-world data")
4. **Arbitrary weights** - Weights should reflect actual importance

### Output

**File**: `filters/{filter_name}/v1/config.yaml` (initial draft)

**Example**:
```yaml
filter:
  name: investment-risk
  version: "1.0"
  purpose: "Capital preservation for defense-first portfolio management"
  philosophy: "You can't predict crashes, but you can prepare for them."

scoring:
  dimensions:
    macro_risk_severity:
      weight: 0.25
      description: "Systemic economic/financial risk signals"
    # ... 7 more dimensions

  tier_thresholds:
    RED:
      threshold: 7.0
      condition: "macro_risk >= 7 OR credit_stress >= 7"
```

---

## Phase 2: Architecture

**Goal**: Create harmonized prompt structure following established patterns

### Checklist

- [ ] **Header complete** - Purpose, Version, Focus, Philosophy, Oracle Output statement
- [ ] **Scope section** - IN SCOPE / OUT OF SCOPE clearly defined
- [ ] **ARTICLE placement** - After scope/rules, before dimensions
- [ ] **Gatekeeper rules** - Documented and positioned correctly
- [ ] **Dimensions with inline filters** - Each dimension has ❌ CRITICAL FILTERS section
- [ ] **Examples section** - At least 3 examples (high, mid, low)
- [ ] **Output format** - JSON schema WITHOUT tier/stage classification
- [ ] **Post-processing section** - "NOT part of oracle output" - tier calculation explained
- [ ] **CHANGELOG** - Version history structure ready

### Validation Criteria

**PASS:**
- All sections present in correct order
- Oracle Output statement: "Dimensional scores only (0-10)"
- ARTICLE after scope/rules
- Every dimension has inline filters
- JSON output has NO tier/stage fields (only dimensional scores + metadata)
- Post-processing section explains tier calculation
- CHANGELOG present

**REVIEW:**
- Philosophy statement missing (optional but recommended)
- Inline filters present but could be more specific
- Examples could be more diverse

**FAIL:**
- Oracle outputs tier classification (violates architecture)
- ARTICLE before scope section
- Missing inline filters (fast models will skip top-level scope)
- No post-processing section
- No CHANGELOG

### Key Architectural Principles

#### 1. Oracle Output Discipline (CRITICAL)

**Oracle outputs:**
- Dimensional scores (0-10 per dimension)
- Reasoning for scores
- Metadata (content_type, primary_technology, etc.)

**Oracle does NOT output:**
- Tier classifications (impact/connection/not_uplifting)
- Stage classifications (deployment_stage, signal_tier)
- Overall scores or weighted calculations

**Why:** Tier classification is post-processing logic. Oracle focuses on accurate dimensional assessment. This allows changing tier thresholds without re-labeling data.

#### 2. Standard Prompt Structure

**Correct order:**
```
1. Header (Purpose, Version, Focus, Philosophy, Oracle Output)
2. Scope (IN SCOPE / OUT OF SCOPE)
3. Rules/Gatekeepers
4. ARTICLE ← Must be here, after scope
5. Dimensional scoring (with inline filters)
6. Examples
7. Output format (JSON schema)
8. Post-processing reference (NOT part of oracle output)
9. CHANGELOG
```

#### 3. Inline Filter Format (CRITICAL)

**Every dimension must have:**
```markdown
1. **Dimension Name**: Description

   **❌ CRITICAL FILTERS - If article is ANY of these, score 0-2:**
   - Filter criterion 1
   - Filter criterion 2
   - Filter criterion 3

   **If NONE of above filters match, score normally:**
   - 0-2: Description
   - 3-4: Description
   - 5-6: Description
   - 7-8: Description
   - 9-10: Description
```

**Why:** Fast models (Gemini Flash) skip top-level SCOPE sections. Inline filters force oracle to check criteria before scoring each dimension.

### Common Pitfalls

1. **Classification in oracle output** - Most common violation. Check JSON schema carefully.
2. **ARTICLE before scope** - Oracle sees article before understanding scope.
3. **Top-level filters only** - Fast models skip them. Must use inline filters.
4. **Weak gatekeepers** - "Should have X" vs "Must have X" - gatekeepers must be enforced.
5. **Missing harmonization** - Not following established patterns makes maintenance harder.

### Output

**Files**:
- `filters/{filter_name}/v1/prompt-compressed.md` - Oracle prompt
- `filters/{filter_name}/v1/config.yaml` - Complete configuration

### Tools to Use

Run harmonization check:
```bash
Task: "Check new filter at filters/{filter_name}/v1/prompt-compressed.md
for harmonization. Compare against filters/uplifting/v4/prompt-compressed.md
as reference."
```

---

## Phase 3: Validation (Oracle Calibration)

**Goal**: Test oracle on sample articles, verify output quality, calibrate thresholds

### Checklist

- [ ] **Sample size** - 50-100 articles from representative corpus
- [ ] **Oracle scoring** - Run batch_scorer with oracle (Gemini Flash or Pro)
- [ ] **Success rate** - >95% of articles scored successfully
- [ ] **Output format** - All outputs parse as valid JSON
- [ ] **No classification fields** - JSON does NOT include tier/stage
- [ ] **Score distribution** - Not all 0s or all 10s
- [ ] **Gatekeeper enforcement** - Rules actually working as intended
- [ ] **Tier distribution** - Examples from each tier
- [ ] **Manual review** - 10-20 articles reviewed by human
- [ ] **Threshold calibration** - Adjust if tier distribution skewed

### Validation Process

#### Step 1: Create Validation Sample

```bash
# Random sample from corpus
python scripts/sample_articles.py \
  --source datasets/raw/master_dataset.jsonl \
  --output validation_sample.jsonl \
  --count 100 \
  --seed 42
```

#### Step 2: Score with Oracle

```bash
python -m ground_truth.batch_scorer \
  --filter filters/{filter_name}/v1 \
  --source validation_sample.jsonl \
  --output-dir sandbox/{filter_name}_validation \
  --llm gemini-flash \
  --batch-size 50
```

**Cost:** ~$0.10 for 100 articles

#### Step 3: Analyze Results

Check success rate:
```bash
python scripts/analyze_scoring_results.py \
  --results sandbox/{filter_name}_validation/scores.jsonl \
  --output validation_analysis.md
```

**Look for:**
- Success rate >95%
- Parse errors (fix JSON schema)
- All scores 0 or 10 (scoring rubric too extreme)
- All scores 5 (oracle confused, rubric unclear)

#### Step 4: Check Output Format

**CRITICAL CHECK:**
```bash
# Ensure NO classification fields in oracle output
grep -r "\"tier\":" sandbox/{filter_name}_validation/scores.jsonl
grep -r "\"signal_tier\":" sandbox/{filter_name}_validation/scores.jsonl
grep -r "\"deployment_stage\":" sandbox/{filter_name}_validation/scores.jsonl
grep -r "\"overall_score\":" sandbox/{filter_name}_validation/scores.jsonl
```

**If any found:** Remove from prompt's JSON schema, add to post-processing section.

#### Step 5: Check Score Distribution

```bash
python scripts/plot_dimension_distributions.py \
  --scores sandbox/{filter_name}_validation/scores.jsonl \
  --output reports/{filter_name}_score_distributions.png
```

**Good distribution:**
- Each dimension has examples across 0-10 range
- Bell curves or bimodal (high/low) are fine
- All 0s or all 10s is BAD (dimension not working)

#### Step 6: Check Gatekeeper Enforcement

**Example:** If deployment_maturity < 5.0 should cap overall at 4.9:

```bash
python scripts/check_gatekeeper_rules.py \
  --scores sandbox/{filter_name}_validation/scores.jsonl \
  --config filters/{filter_name}/v1/config.yaml
```

**Look for:**
- Articles with deployment_maturity < 5.0 but overall > 4.9 (gatekeeper not working)
- Fix: Add explicit gatekeeper check to prompt or post-filter

#### Step 7: Examine Tier Distribution

```bash
python scripts/compute_tiers.py \
  --scores sandbox/{filter_name}_validation/scores.jsonl \
  --config filters/{filter_name}/v1/config.yaml \
  --output tier_distribution.txt
```

**Target distribution (adjust for filter type):**
- Tier 1 (high): 10-20%
- Tier 2: 20-30%
- Tier 3: 30-40%
- Tier 4 (low): 20-30%

**If skewed:**
- >60% in one tier: Adjust thresholds
- <5% in high tier: Lower threshold or check scope
- 0% in any tier: Major issue - review prompt

#### Step 8: Manual Review

**Select for review:**
- 3-5 high-scoring articles (overall ≥7.0)
- 3-5 edge cases (near threshold)
- 3-5 low-scoring articles (overall ≤3.0)

**Check:**
- Does score match your human judgment?
- Is reasoning specific to article (not generic)?
- Are inline filters working?
- Are examples being followed?

**Agreement rate:**
- ≥80%: Excellent
- 70-79%: Acceptable
- <70%: Review prompt, add examples, clarify rubrics

### Validation Criteria

**PASS:**
- Success rate ≥95%
- No classification fields in oracle output
- Score distribution reasonable (not all 0s/10s)
- Gatekeepers enforced correctly
- Tier distribution balanced (<60% in any tier)
- Manual review ≥70% agreement

**REVIEW:**
- Success rate 90-95% (investigate failures)
- Score distribution slightly skewed (minor calibration)
- Tier distribution 60-70% in one tier (adjust thresholds)
- Manual review 60-70% agreement (clarify prompt)

**FAIL:**
- Success rate <90% (prompt broken)
- Classification fields in oracle output (violates architecture)
- All scores 0-2 or 8-10 (rubric broken)
- Gatekeepers not enforced (prompt unclear)
- >70% in one tier (thresholds wrong)
- Manual review <60% agreement (prompt doesn't work)

### Common Issues and Fixes

#### Issue 1: Oracle outputs tier classification

**Symptom:** JSON includes "tier", "signal_tier", "deployment_stage", "overall_score"

**Fix:**
1. Remove field from JSON schema in prompt
2. Add to post-processing section
3. Document as "computed post-hoc, not by oracle"

#### Issue 2: All scores 0-2 or 8-10 (no middle ground)

**Symptom:** Dimension distributions are bimodal at extremes

**Fix:**
1. Clarify scoring rubric for 3-7 range
2. Add examples in middle range
3. Emphasize "score normally" after inline filters

#### Issue 3: Gatekeepers not enforced

**Symptom:** Articles with deployment_maturity < 5.0 score overall > 4.9

**Fix:**
1. Make gatekeeper more explicit in prompt
2. Add post-filter check to enforce cap
3. Test on edge cases

#### Issue 4: Oracle confused by scope

**Symptom:** Out-of-scope articles score high

**Fix:**
1. Add inline filters to catch out-of-scope
2. Add more out-of-scope examples
3. Emphasize scope in system message

#### Issue 5: Tier distribution heavily skewed

**Symptom:** 70%+ articles in one tier

**Fix:**
1. Adjust tier thresholds (if validation sample representative)
2. Check validation sample (might not be representative)
3. Review weights (might over-emphasize one dimension)

### Output

**File**: `filters/{filter_name}/v1/validation_report.md`

**Template**:
```markdown
# {Filter Name} v1 - Oracle Validation Report

**Date:** YYYY-MM-DD
**Oracle Model:** Gemini Flash 1.5
**Sample Size:** 100 articles
**Manual Review:** 15 articles (5 high, 5 edge, 5 low)

## Executive Summary

**Oracle Quality: ✅ PASS / ⚠️ ACCEPTABLE / ❌ FAIL**
**Manual Agreement: X%**

## Metrics

- Success rate: X% (target: ≥95%)
- Classification fields in output: ✅ None / ❌ Found
- Score distribution: ✅ Reasonable / ⚠️ Skewed / ❌ Broken
- Gatekeeper enforcement: ✅ Working / ❌ Not enforced
- Tier distribution: ✅ Balanced / ⚠️ Skewed / ❌ Broken

## Score Distributions

[Include plots or tables]

## Tier Distribution

- Tier 1: X% (target: 10-20%)
- Tier 2: X% (target: 20-30%)
- Tier 3: X% (target: 30-40%)
- Tier 4: X% (target: 20-30%)

## Manual Review

[Include 3-5 examples with oracle scores and manual assessment]

## Issues Found

[List issues with severity]

## Recommendations

- [ ] Issue 1: Fix
- [ ] Issue 2: Investigate
- [ ] Proceed to prefilter validation: ✅ / ❌
```

---

## Phase 4: Prefilter Validation

**Goal**: Test prefilter on large sample, measure pass rate, optimize for false negatives

### Checklist

- [ ] **Prefilter implemented** - Fast, deterministic, rule-based
- [ ] **Large sample** - 1K-5K articles from corpus
- [ ] **Pass rate measured** - What % passes prefilter?
- [ ] **Target pass rate** - 30-50% for most filters (adjust for filter type)
- [ ] **False negative check** - Are good articles blocked? (CRITICAL)
- [ ] **False positive check** - Are bad articles passing? (Less critical)
- [ ] **Speed test** - <10ms per article
- [ ] **Iteration** - Adjust rules if false negative rate >10%

### Why Prefilter Matters

**Purpose:** Fast, cheap noise reduction before expensive oracle/model scoring

**Benefits:**
- Saves cost (don't score obvious noise)
- Saves time (fast filtering)
- Improves data quality (cleaner training data)

**Design principle:** Err on side of false positives (let through borderline articles). Oracle catches them. False negatives (blocking good articles) are CRITICAL failures.

### Prefilter Types

#### Type 1: Keyword/Pattern Blocking (investment-risk)

**Approach:** Block obvious noise categories
- FOMO speculation (meme stocks, crypto pumping)
- Stock picking without macro context
- Affiliate marketing
- Clickbait

**Target pass rate:** 40-70% (conservative filtering)

#### Type 2: Requirement Checking (sustainability_tech_deployment)

**Approach:** Check for required signals
- Deployment indicators (MW, GW, "operational", "generating")
- Technology terms (solar, wind, battery, etc.)
- Evidence signals (data, performance, cost)

**Target pass rate:** 5-20% (aggressive filtering, very specific scope)

#### Type 3: Hybrid (uplifting)

**Approach:** Block known bad categories + check for positive signals
- Block: Product launches, corporate finance, generic business news
- Require: Progress indicators, collective benefit signals

**Target pass rate:** 30-50%

### Prefilter Validation Process

#### Step 1: Create Test Sample

```bash
# Sample 1K-5K articles
python scripts/sample_articles.py \
  --source datasets/raw/master_dataset.jsonl \
  --output prefilter_test_sample.jsonl \
  --count 1000 \
  --seed 2025
```

#### Step 2: Run Prefilter

```bash
python filters/{filter_name}/v1/prefilter.py \
  --input prefilter_test_sample.jsonl \
  --output prefilter_results.jsonl \
  --stats prefilter_stats.txt
```

#### Step 3: Measure Pass Rate

```bash
python scripts/analyze_prefilter.py \
  --results prefilter_results.jsonl \
  --output prefilter_analysis.md
```

**Check:**
- Pass rate within target range?
- Pass rate too low (<20%): Might miss good articles
- Pass rate too high (>80%): Not filtering enough noise

#### Step 4: Check False Negatives (CRITICAL)

**False negative:** Good article blocked by prefilter

**Process:**
1. Sample 100 blocked articles (random)
2. Score with oracle
3. Check how many score high (≥7.0)

```bash
# Sample blocked articles
python scripts/sample_blocked.py \
  --results prefilter_results.jsonl \
  --output blocked_sample.jsonl \
  --count 100

# Score with oracle
python -m ground_truth.batch_scorer \
  --filter filters/{filter_name}/v1 \
  --source blocked_sample.jsonl \
  --output-dir sandbox/prefilter_fn_check \
  --llm gemini-flash

# Analyze
python scripts/check_false_negatives.py \
  --scores sandbox/prefilter_fn_check/scores.jsonl \
  --threshold 7.0
```

**Acceptable false negative rate:** <10% (i.e., <10 articles out of 100 blocked would score ≥7.0)

**If >10% false negatives:**
- Prefilter too aggressive
- Review blocking rules
- Loosen criteria

#### Step 5: Check False Positives (Less Critical)

**False positive:** Bad article passes prefilter

**Process:**
1. Sample 100 passed articles (random)
2. Score with oracle
3. Check how many score low (≤3.0)

```bash
# Sample passed articles
python scripts/sample_passed.py \
  --results prefilter_results.jsonl \
  --output passed_sample.jsonl \
  --count 100

# Score with oracle
python -m ground_truth.batch_scorer \
  --filter filters/{filter_name}/v1 \
  --source passed_sample.jsonl \
  --output-dir sandbox/prefilter_fp_check \
  --llm gemini-flash

# Analyze
python scripts/check_false_positives.py \
  --scores sandbox/prefilter_fp_check/scores.jsonl \
  --threshold 3.0
```

**Acceptable false positive rate:** <50% (i.e., up to 50 out of 100 passed articles can score ≤3.0)

**Why high tolerance:** Oracle will catch them. Prefilter job is noise reduction, not perfect classification.

**If >60% false positives:**
- Prefilter too lenient
- Consider stricter rules
- But still prioritize avoiding false negatives

#### Step 6: Speed Test

```bash
time python filters/{filter_name}/v1/prefilter.py \
  --input prefilter_test_sample.jsonl \
  --output /dev/null
```

**Target:** <10ms per article (i.e., 1000 articles in <10 seconds)

**If slower:**
- Optimize regex patterns
- Reduce number of checks
- Use faster libraries (regex is usually fast enough)

### Validation Criteria

**PASS:**
- Pass rate within target range
- False negative rate <10% (good articles not blocked)
- False positive rate <50% (acceptable noise)
- Speed <10ms per article

**REVIEW:**
- Pass rate slightly outside target (±10%)
- False negative rate 10-15% (investigate)
- False positive rate 50-60% (consider tightening)
- Speed 10-20ms per article (acceptable)

**FAIL:**
- Pass rate way outside target (>30% difference)
- False negative rate >15% (blocking good articles - CRITICAL)
- Speed >20ms per article (too slow)

### Common Issues and Fixes

#### Issue 1: High false negative rate

**Symptom:** Good articles blocked by prefilter

**Fix:**
1. Review blocking rules - too strict?
2. Add exceptions for edge cases
3. Loosen requirements
4. Test on known good articles

**Example:** Sustainability_tech_deployment v3 blocked pilots (deployment < 5.0). Fixed in v1 by lowering to 3.0.

#### Issue 2: Pass rate too low

**Symptom:** <20% pass rate, corpus has more relevant articles

**Fix:**
1. Check if requirements too strict
2. Review blocked samples manually
3. Add more exception patterns
4. Consider hybrid approach (multiple ways to pass)

#### Issue 3: Pass rate too high

**Symptom:** >70% pass rate, most articles are noise

**Fix:**
1. Add more blocking rules
2. Check for common noise patterns in passed articles
3. Add requirement checks (must have X indicator)

#### Issue 4: Slow prefilter

**Symptom:** >20ms per article

**Fix:**
1. Optimize regex patterns (compile once, use repeatedly)
2. Reduce number of checks (prioritize fast checks first)
3. Avoid complex NLP (prefilter should be simple)

### Output

**File**: `filters/{filter_name}/v1/prefilter_validation_report.md`

**Template**:
```markdown
# {Filter Name} v1 - Prefilter Validation Report

**Date:** YYYY-MM-DD
**Test Sample:** 1000 articles
**Prefilter Version:** 1.0

## Executive Summary

**Prefilter Quality: ✅ PASS / ⚠️ REVIEW / ❌ FAIL**

## Metrics

- Pass rate: X% (target: Y-Z%)
- False negative rate: X% (target: <10%)
- False positive rate: X% (target: <50%)
- Speed: Xms per article (target: <10ms)

## False Negative Analysis

- Blocked sample size: 100 articles
- Scored high (≥7.0): X articles (X%)
- **Issue:** [If >10%, describe which articles blocked]
- **Fix:** [Proposed changes to prefilter]

## False Positive Analysis

- Passed sample size: 100 articles
- Scored low (≤3.0): X articles (X%)
- **Assessment:** ✅ Acceptable / ⚠️ High / ❌ Too high

## Examples

### False Negatives (Should Pass, Got Blocked)
[List 3-5 examples if found]

### False Positives (Should Block, Got Passed)
[List 3-5 examples if rate is high]

## Recommendations

- [ ] Adjust prefilter rules: [Specific changes]
- [ ] Re-run validation: ✅ / ❌
- [ ] Proceed to training data collection: ✅ / ❌
```

---

## Phase 5: Training Data Collection

**Goal**: Score 5K+ articles with oracle, validate dataset quality, prepare for training

### Checklist

- [ ] **Target size** - 5K+ articles (3K minimum for simple filters)
- [ ] **Sampling strategy** - Random OR stratified by tier/source
- [ ] **Oracle scoring** - Batch score all articles
- [ ] **Success rate** - >95% scored successfully
- [ ] **Tier distribution** - Not 99% one tier (balanced across tiers)
- [ ] **Dimension coverage** - All dimensions have examples across 0-10 range
- [ ] **No classification artifacts** - Oracle not outputting tiers
- [ ] **Gatekeeper enforcement** - Rules working correctly
- [ ] **Dataset stats documented** - Distribution, coverage, quality metrics

### Why Training Data Quality Matters

**Good data:** Model learns to score like oracle
**Bad data:** Model learns artifacts, biases, mistakes

**Common data quality issues:**
- Heavily skewed distribution (99% one tier)
- Missing examples for some dimensions (e.g., no high-scoring examples)
- Oracle mistakes (misclassified articles)
- Sampling bias (only tech news, missing other sources)

### Sampling Strategies

#### Strategy 1: Random Sampling (Default)

**When:** Filter scope is broad, corpus is balanced

**Approach:**
```bash
python -m ground_truth.batch_scorer \
  --filter filters/{filter_name}/v1 \
  --source datasets/raw/master_dataset.jsonl \
  --output-dir ground_truth/labeled/{filter_name}/v1 \
  --llm gemini-flash \
  --target-count 5000 \
  --random-sample \
  --batch-size 100
```

**Pros:** Simple, representative of production distribution
**Cons:** Might miss rare tiers if filter is very selective

#### Strategy 2: Stratified Sampling by Source

**When:** Different sources have different tier distributions

**Approach:** Sample from each source separately

```bash
# Sample 2000 from tech news
python -m ground_truth.batch_scorer \
  --filter filters/{filter_name}/v1 \
  --source datasets/raw/tech_news.jsonl \
  --output-dir ground_truth/labeled/{filter_name}/v1/tech_news \
  --llm gemini-flash \
  --target-count 2000 \
  --random-sample

# Sample 2000 from research publications
python -m ground_truth.batch_scorer \
  --filter filters/{filter_name}/v1 \
  --source datasets/raw/research_pubs.jsonl \
  --output-dir ground_truth/labeled/{filter_name}/v1/research \
  --llm gemini-flash \
  --target-count 2000 \
  --random-sample

# Sample 1000 from industry reports
python -m ground_truth.batch_scorer \
  --filter filters/{filter_name}/v1 \
  --source datasets/raw/industry_reports.jsonl \
  --output-dir ground_truth/labeled/{filter_name}/v1/reports \
  --llm gemini-flash \
  --target-count 1000 \
  --random-sample

# Combine
python scripts/combine_labeled.py \
  --inputs ground_truth/labeled/{filter_name}/v1/*/*.jsonl \
  --output ground_truth/labeled/{filter_name}/v1/combined.jsonl
```

**Pros:** Ensures coverage across sources, balanced tiers
**Cons:** More complex, need to know source distributions

#### Strategy 3: Targeted Tier Collection (If Needed)

**When:** Random sampling produces heavily skewed distribution (e.g., 95% low tier)

**Approach:** Sample more aggressively from sources likely to have rare tiers

**Example:** Sustainability_tech_deployment filter
- Mass deployment (rare): Sample from industry reports, case studies
- Pilots (uncommon): Sample from grant databases, pilot announcements
- Vaporware (common): Random sample from tech news

**Process:**
1. Run pilot scoring (500 articles random sample)
2. Analyze tier distribution
3. If >70% one tier, identify sources for rare tiers
4. Sample targeted sources for rare tiers
5. Combine with random sample for balanced dataset

### Data Quality Validation

#### Check 1: Tier Distribution

```bash
python scripts/analyze_tier_distribution.py \
  --scores ground_truth/labeled/{filter_name}/v1/combined.jsonl \
  --config filters/{filter_name}/v1/config.yaml
```

**Target distribution:**
- High tier: 10-25%
- Mid-high tier: 20-30%
- Mid-low tier: 25-35%
- Low tier: 20-35%

**Issues:**
- >60% in one tier: Skewed, consider resampling
- <5% in any tier: Missing examples, target that tier
- 0% in any tier: CRITICAL - must fix

#### Check 2: Dimension Coverage

```bash
python scripts/analyze_dimension_coverage.py \
  --scores ground_truth/labeled/{filter_name}/v1/combined.jsonl \
  --output reports/{filter_name}_dimension_coverage.md
```

**For each dimension, check:**
- Low range (0-3): At least 100 examples
- Mid range (4-6): At least 100 examples
- High range (7-10): At least 100 examples

**Issues:**
- Missing high-range examples: Model won't learn to score high
- Missing low-range examples: Model won't learn to score low
- All mid-range: Model won't learn extremes

#### Check 3: Oracle Artifacts

```bash
# Check for classification fields in training data
python scripts/check_oracle_artifacts.py \
  --scores ground_truth/labeled/{filter_name}/v1/combined.jsonl
```

**Look for:**
- "tier" in oracle output (should be computed post-hoc)
- "signal_tier" in oracle output
- "deployment_stage" in oracle output
- "overall_score" in oracle output

**If found:** Re-run scoring with corrected prompt

#### Check 4: Gatekeeper Enforcement

```bash
python scripts/validate_gatekeepers.py \
  --scores ground_truth/labeled/{filter_name}/v1/combined.jsonl \
  --config filters/{filter_name}/v1/config.yaml
```

**Check:**
- Articles with deployment_maturity < 5.0 should have overall < 4.9
- Gatekeeper rules applied consistently

**If violations found:** Check post-filter logic, might need to enforce in training data

#### Check 5: Reasonableness Sampling

**Manually review:**
- 10 high-scoring articles (overall ≥8.0)
- 10 mid-scoring articles (overall 4.0-6.0)
- 10 low-scoring articles (overall ≤2.0)

**Check:**
- Do scores make sense?
- Is reasoning specific?
- Are inline filters working?
- Any systematic errors?

### Validation Criteria

**PASS:**
- Size ≥5000 articles (or ≥3000 for simple filters)
- Tier distribution: No tier >60%
- Dimension coverage: All dimensions have ≥100 examples per range
- No classification artifacts
- Gatekeepers enforced
- Manual review: ≥80% reasonable
- Validation script passes with 0 critical issues
- No duplicate IDs across train/val/test splits
- All splits have correct proportions (80/10/10 ±5%)

**REVIEW:**
- Size 3000-5000 (acceptable but prefer more)
- Tier distribution: One tier 60-70% (consider resampling)
- Dimension coverage: Some ranges have 50-100 examples (monitor)
- Manual review: 70-80% reasonable
- Validation warnings present (review but acceptable)
- Split proportions slightly off (75-85/8-12/8-12)

**FAIL:**
- Size <3000 articles (too small)
- Tier distribution: One tier >70% (heavily skewed)
- Dimension coverage: Any range <50 examples (insufficient)
- Classification artifacts found (violates architecture)
- Gatekeepers not enforced (data quality issue)
- Manual review: <70% reasonable (oracle broken)
- Duplicate IDs found across splits (data leakage)
- Out-of-range scores (outside [0-10])
- Empty content or missing required fields

### Common Issues and Fixes

#### Issue 1: Heavily skewed tier distribution

**Symptom:** 70%+ articles in one tier

**Root causes:**
- Corpus not representative of target use case
- Tier thresholds miscalibrated
- Filter scope too narrow

**Fixes:**
1. **If filter scope is correct:** Use stratified sampling (target rare tiers)
2. **If thresholds wrong:** Recalibrate thresholds based on desired distribution
3. **If corpus wrong:** Find additional sources for rare tiers

#### Issue 2: Missing high-range examples

**Symptom:** No articles with dimension score ≥7 for some dimension

**Root causes:**
- Dimension definition too strict
- Corpus doesn't have high-quality examples
- Oracle not recognizing good examples

**Fixes:**
1. Review dimension definition (too strict?)
2. Sample from high-quality sources (case studies, industry reports)
3. Check oracle calibration (is it scoring too low?)

#### Issue 3: Training data too small

**Symptom:** <3000 articles after filtering

**Root causes:**
- Prefilter too aggressive
- Corpus too small
- Oracle failing too often

**Fixes:**
1. Loosen prefilter (more articles pass)
2. Collect more raw articles
3. Fix oracle prompt (reduce failures)

#### Issue 4: Duplicate IDs across splits

**Symptom:** Same article appears in multiple splits (train/val/test)

**Root causes:**
- Article scored multiple times in different batches
- Stratified split doesn't check for duplicates
- Same article in different source files

**Impact:** Data leakage - model "memorizes" validation/test articles during training

**Fixes:**
1. Run deduplication script:
   ```bash
   python training/deduplicate_training_data.py datasets/training/{filter_name}_v1
   ```
2. Re-validate to confirm all duplicates removed
3. Prevent in future: Check for duplicates in source data before scoring

### Final Validation Step

Before using the training data, run comprehensive quality validation:

```bash
# Validate training data quality
python training/validate_training_data.py \
  --data-dir datasets/training/{filter_name}_v1 \
  --filter filters/{filter_name}/v1
```

**Checks performed:**
- Structural integrity (required fields, ID uniqueness, label array length)
- Data distribution (train/val/test splits at 80/10/10)
- Label quality (score range [0-10], no NaN values, sufficient variance)
- Content quality (non-empty titles/content, reasonable lengths)
- Consistency (dimension names match across splits and config)
- Score distributions per dimension

**If duplicates found:**
```bash
# Remove duplicate IDs across splits (keeps in train, removes from val/test)
python training/deduplicate_training_data.py datasets/training/{filter_name}_v1

# Re-validate after deduplication
python training/validate_training_data.py \
  --data-dir datasets/training/{filter_name}_v1 \
  --filter filters/{filter_name}/v1
```

**Validation report:** Save summary to filter folder:
```bash
# Create validation summary for filter documentation
python scripts/save_validation_report.py \
  --data-dir datasets/training/{filter_name}_v1 \
  --filter filters/{filter_name}/v1 \
  --output filters/{filter_name}/v1/TRAINING_DATA_VALIDATION.md
```

### Output

**Files**:
- `ground_truth/labeled/{filter_name}/v1/combined.jsonl` - All labeled data
- `datasets/training/{filter_name}_v1/train.jsonl` - Training split (80%)
- `datasets/training/{filter_name}_v1/val.jsonl` - Validation split (10%)
- `datasets/training/{filter_name}_v1/test.jsonl` - Test split (10%)
- `reports/{filter_name}_training_data_report.md` - Quality analysis
- `filters/{filter_name}/v1/TRAINING_DATA_VALIDATION.md` - Validation summary

**Training Data Report Template**:
```markdown
# {Filter Name} v1 - Training Data Report

**Date:** YYYY-MM-DD
**Oracle Model:** Gemini Flash 1.5
**Total Articles:** X

## Dataset Summary

- Total scored: X articles
- Success rate: X%
- Training split: X articles (80%)
- Validation split: X articles (10%)
- Test split: X articles (10%)

## Tier Distribution

- Tier 1: X% (target: 10-25%)
- Tier 2: X% (target: 20-30%)
- Tier 3: X% (target: 25-35%)
- Tier 4: X% (target: 20-35%)

**Assessment:** ✅ Balanced / ⚠️ Skewed / ❌ Heavily skewed

## Dimension Coverage

[Table showing example counts per dimension per range]

**Assessment:** ✅ Complete / ⚠️ Some gaps / ❌ Major gaps

## Quality Checks

- Classification artifacts: ✅ None / ❌ Found
- Gatekeeper enforcement: ✅ Working / ❌ Violations
- Manual review (30 articles): X% reasonable

## Recommendations

- [ ] Proceed to training: ✅ / ❌
- [ ] Resample for balance: ✅ / ❌
- [ ] Collect more data: ✅ / ❌
```

---

## Phase 6: Model Training

**Goal**: Train student model to replicate oracle, evaluate performance, iterate if needed

### Checklist

- [ ] **Training data ready** - train.jsonl, val.jsonl, test.jsonl
- [ ] **Training mode selected** - Distillation OR instruction tuning
- [ ] **Base model selected** - Qwen 2.5 7B (recommended)
- [ ] **Training run complete** - Model checkpoints saved
- [ ] **Validation metrics** - MAE, accuracy, correlation tracked
- [ ] **Test set evaluation** - Final performance on held-out test set
- [ ] **Per-dimension analysis** - Which dimensions learned well?
- [ ] **Tier classification accuracy** - Model + postfilter accuracy
- [ ] **Training report** - Complete documentation of results

### Training Modes

#### Mode 1: Knowledge Distillation (Recommended)

**Approach:** Model learns to predict dimensional scores directly

**Input:** Article text
**Output:** 8 dimensional scores (0-10 each)

**Advantages:**
- Simpler task (regression only)
- Better performance (focused learning)
- Easier to debug (check per-dimension MAE)

**Training:**
```bash
python -m training.train \
  --mode distillation \
  --filter-name {filter_name} \
  --version v1 \
  --base-model unsloth/Qwen2.5-7B-Instruct \
  --epochs 3 \
  --learning-rate 2e-4 \
  --batch-size 4 \
  --gradient-accumulation 4 \
  --output-dir filters/{filter_name}/v1_distillation
```

#### Mode 2: Instruction Tuning

**Approach:** Model learns to generate full JSON output (like oracle)

**Input:** Article text + instruction
**Output:** Full JSON with scores and reasoning

**Advantages:**
- Can generate reasoning
- More flexible output format

**Disadvantages:**
- Harder to train (generation task)
- Slower inference
- More prone to hallucination

**Training:**
```bash
python -m training.train \
  --mode instruction \
  --filter-name {filter_name} \
  --version v1 \
  --base-model unsloth/Qwen2.5-7B-Instruct \
  --epochs 3 \
  --learning-rate 2e-4 \
  --batch-size 2 \
  --gradient-accumulation 8 \
  --output-dir filters/{filter_name}/v1_instruction
```

**Recommendation:** Start with distillation. Add instruction tuning later if reasoning needed.

### Training Process

#### Step 1: Prepare Training Data

```bash
python scripts/prepare_training_data.py \
  --input ground_truth/labeled/{filter_name}/v1/combined.jsonl \
  --config filters/{filter_name}/v1/config.yaml \
  --mode distillation \
  --output-dir training_data/{filter_name}/v1
```

**Output:**
- `train.jsonl` - Training examples
- `val.jsonl` - Validation examples
- `test.jsonl` - Test examples (never used during training)

#### Step 2: Run Training

```bash
python -m training.train \
  --mode distillation \
  --filter-name {filter_name} \
  --version v1 \
  --base-model unsloth/Qwen2.5-7B-Instruct \
  --epochs 3 \
  --learning-rate 2e-4 \
  --batch-size 4 \
  --gradient-accumulation 4 \
  --warmup-steps 100 \
  --output-dir filters/{filter_name}/v1_distillation \
  --wandb-project llm-distillery-{filter_name}
```

**Time:** 2-6 hours depending on data size and GPU
**Cost:** $0-5 if using cloud GPU (or free with local GPU)

#### Step 3: Monitor Training

**Watch:**
- Training loss (should decrease)
- Validation loss (should decrease, then plateau)
- Validation MAE (mean absolute error per dimension)
- Overfitting (validation loss increases while training loss decreases)

**Early stopping:**
- If validation loss stops improving for 2 epochs, stop
- If validation loss increases, revert to best checkpoint

#### Step 4: Evaluate on Test Set

```bash
python -m training.evaluate \
  --model filters/{filter_name}/v1_distillation/final \
  --test-file ground_truth/labeled/{filter_name}/v1/test.jsonl \
  --config filters/{filter_name}/v1/config.yaml \
  --output reports/{filter_name}_evaluation.md
```

**Metrics:**
- Overall MAE (mean absolute error across all dimensions)
- Per-dimension MAE (which dimensions learned well?)
- Tier classification accuracy (model + postfilter)
- Precision/recall per tier
- Confusion matrix

### Validation Criteria

**PASS:**
- Overall MAE ≤1.5 (excellent: ≤1.0)
- Per-dimension MAE ≤2.0 for all dimensions
- Tier classification accuracy ≥85%
- No dimension with MAE >3.0

**REVIEW:**
- Overall MAE 1.5-2.0 (acceptable)
- Some dimensions have MAE 2.0-3.0 (investigate)
- Tier classification accuracy 75-85% (usable)

**FAIL:**
- Overall MAE >2.0 (poor performance)
- Any dimension MAE >3.0 (model didn't learn)
- Tier classification accuracy <75% (not production-ready)

### Common Issues and Fixes

#### Issue 1: High MAE on specific dimension

**Symptom:** One dimension has MAE >2.5, others are fine

**Causes:**
- Insufficient training examples for that dimension
- Dimension definition unclear
- Highly subjective dimension

**Fixes:**
1. Check training data coverage for that dimension
2. Review dimension definition (too vague?)
3. Collect more examples highlighting that dimension
4. Consider adjusting dimension weight (if low importance)

#### Issue 2: Model overfitting

**Symptom:** Training loss decreases, validation loss increases

**Causes:**
- Training too many epochs
- Learning rate too high
- Training data too small

**Fixes:**
1. Use early stopping
2. Reduce learning rate
3. Add more training data
4. Reduce model size (try Qwen 1.5B)

#### Issue 3: Poor tier classification accuracy

**Symptom:** MAE is OK but tier accuracy is low

**Causes:**
- Tier thresholds don't match data distribution
- Postfilter logic wrong
- Model not learning tier boundaries

**Fixes:**
1. Check postfilter tier thresholds
2. Analyze confusion matrix (which tiers confused?)
3. Add tier-boundary examples to training data

### Output

**Files**:
- `filters/{filter_name}/v1_distillation/final/` - Trained model
- `filters/{filter_name}/v1_distillation/checkpoints/` - Training checkpoints
- `reports/{filter_name}_training_report.md` - Training results
- `reports/{filter_name}_evaluation.md` - Test set evaluation

**Training Report Template**:
```markdown
# {Filter Name} v1 - Training Report

**Date:** YYYY-MM-DD
**Training Mode:** Distillation
**Base Model:** Qwen 2.5 7B Instruct
**Training Data:** X articles
**Test Data:** X articles (held-out)

## Training Configuration

- Epochs: 3
- Learning rate: 2e-4
- Batch size: 4
- Gradient accumulation: 4
- Warmup steps: 100

## Training Results

- Final training loss: X
- Final validation loss: X
- Training time: X hours

## Test Set Performance

### Overall Metrics
- Overall MAE: X (target: ≤1.5)
- Tier classification accuracy: X% (target: ≥85%)

### Per-Dimension MAE
- dimension_1: X (target: ≤2.0)
- dimension_2: X (target: ≤2.0)
- ...

### Tier Classification
- Tier 1: Precision X%, Recall X%
- Tier 2: Precision X%, Recall X%
- Tier 3: Precision X%, Recall X%
- Tier 4: Precision X%, Recall X%

## Confusion Matrix

[Include tier confusion matrix]

## Analysis

[Which dimensions learned well? Which struggled? Why?]

## Recommendations

- [ ] Deploy to production: ✅ / ❌
- [ ] Retrain with more data: ✅ / ❌
- [ ] Tune hyperparameters: ✅ / ❌
- [ ] Improve specific dimension: [Which one?]
```

---

## Phase 7: Testing

**Goal**: Benchmark student model vs oracle, integration tests, edge cases, performance

### Checklist

- [ ] **Oracle benchmark** - Compare student vs oracle on sample
- [ ] **Agreement rate** - How often do student and oracle agree?
- [ ] **MAE vs oracle** - Mean absolute error per dimension
- [ ] **Edge case testing** - Challenging articles
- [ ] **Integration test** - Full pipeline (prefilter → model → postfilter)
- [ ] **Performance test** - Inference time <50ms per article
- [ ] **Throughput test** - Can process 1000 articles in <1 minute?
- [ ] **Failure mode analysis** - When does model fail?

### Testing Process

#### Test 1: Oracle Benchmark

**Goal:** Verify student matches oracle quality

```bash
# Sample 100 articles not in training data
python scripts/sample_articles.py \
  --source datasets/raw/master_dataset.jsonl \
  --output benchmark_sample.jsonl \
  --count 100 \
  --seed 9999 \
  --exclude ground_truth/labeled/{filter_name}/v1/combined.jsonl

# Score with oracle
python -m ground_truth.batch_scorer \
  --filter filters/{filter_name}/v1 \
  --source benchmark_sample.jsonl \
  --output-dir sandbox/{filter_name}_oracle_benchmark \
  --llm gemini-flash

# Score with student model
python -m inference.score \
  --model filters/{filter_name}/v1_distillation/final \
  --config filters/{filter_name}/v1/config.yaml \
  --input benchmark_sample.jsonl \
  --output sandbox/{filter_name}_student_benchmark.jsonl

# Compare
python scripts/compare_oracle_student.py \
  --oracle sandbox/{filter_name}_oracle_benchmark/scores.jsonl \
  --student sandbox/{filter_name}_student_benchmark.jsonl \
  --output reports/{filter_name}_benchmark.md
```

**Metrics:**
- Agreement rate (same tier): Target ≥80%
- MAE per dimension: Target ≤1.5
- Correlation per dimension: Target ≥0.8

#### Test 2: Edge Case Testing

**Goal:** Test model on challenging articles

**Edge cases:**
- Borderline tier classification (scores near thresholds)
- Mixed signals (high on some dimensions, low on others)
- Out-of-scope articles (should score low)
- Ambiguous articles (unclear tier)

**Process:**
```bash
# Manually curate 20-30 edge cases
# Score with both oracle and student
# Compare results
```

#### Test 3: Integration Test

**Goal:** Test full pipeline end-to-end

```bash
python scripts/test_full_pipeline.py \
  --filter filters/{filter_name}/v1 \
  --model filters/{filter_name}/v1_distillation/final \
  --input benchmark_sample.jsonl \
  --output integration_test_results.jsonl
```

**Checks:**
- Prefilter works correctly
- Model scores all articles
- Postfilter classifies tiers correctly
- Output format valid

#### Test 4: Performance Test

**Goal:** Measure inference speed

```bash
python scripts/benchmark_inference.py \
  --model filters/{filter_name}/v1_distillation/final \
  --config filters/{filter_name}/v1/config.yaml \
  --input benchmark_sample.jsonl \
  --iterations 10
```

**Metrics:**
- Single article inference: Target <50ms
- Batch inference (100 articles): Target <5 seconds
- Throughput: Target >1000 articles/minute

#### Test 5: Failure Mode Analysis

**Goal:** Understand when and why model fails

**Process:**
1. Find articles where student and oracle disagree significantly (MAE >3.0)
2. Analyze why (which dimension? what pattern?)
3. Document failure modes
4. Consider if failures are acceptable or require fixes

### Validation Criteria

**PASS:**
- Oracle agreement ≥80%
- MAE vs oracle ≤1.5
- Edge cases mostly correct (≥70%)
- Integration test passes
- Inference <50ms per article
- No critical failure modes

**REVIEW:**
- Oracle agreement 70-80% (investigate discrepancies)
- MAE vs oracle 1.5-2.0 (acceptable but not great)
- Edge cases 60-70% correct (some issues)
- Inference 50-100ms (usable but slow)

**FAIL:**
- Oracle agreement <70% (model not reliable)
- MAE vs oracle >2.0 (poor replication)
- Edge cases <60% correct (fails on important cases)
- Integration test fails (pipeline broken)
- Inference >100ms (too slow for production)
- Critical failure modes (blocks good articles, passes obvious noise)

### Common Issues and Fixes

#### Issue 1: Low oracle agreement but good MAE

**Symptom:** MAE is fine but tier classifications differ

**Cause:** Tier thresholds cause small MAE to result in tier changes

**Fix:**
1. Check tier boundaries (too close together?)
2. Consider wider tier ranges
3. Add tier-boundary examples to training data

#### Issue 2: Model too slow

**Symptom:** Inference >100ms per article

**Causes:**
- Model too large (7B might be overkill)
- Inefficient implementation
- GPU not utilized properly

**Fixes:**
1. Try smaller model (Qwen 1.5B)
2. Optimize inference code (batching, quantization)
3. Use GPU for inference

#### Issue 3: Systematic failures on specific type

**Symptom:** Model consistently fails on certain article types

**Causes:**
- Training data didn't include enough of that type
- Dimension unclear for that type

**Fixes:**
1. Add more examples of that type to training data
2. Review dimension definitions for that type
3. Consider separate model for that type

### Output

**File**: `reports/{filter_name}_testing_report.md`

**Template**:
```markdown
# {Filter Name} v1 - Testing Report

**Date:** YYYY-MM-DD
**Model:** Qwen 2.5 7B Distillation
**Test Sample:** 100 articles (unseen)

## Oracle Benchmark

- Agreement rate: X% (target: ≥80%)
- Overall MAE vs oracle: X (target: ≤1.5)
- Correlation: X (target: ≥0.8)

### Per-Dimension Comparison
[Table showing MAE and correlation per dimension]

## Edge Case Testing

- Total edge cases: 25
- Correct: X (X%)
- Acceptable: X (X%)
- Incorrect: X (X%)

[List 3-5 interesting edge cases]

## Integration Test

- Prefilter: ✅ Pass / ❌ Fail
- Model inference: ✅ Pass / ❌ Fail
- Postfilter: ✅ Pass / ❌ Fail
- Output format: ✅ Pass / ❌ Fail

## Performance Test

- Single article: Xms (target: <50ms)
- Batch (100 articles): Xs (target: <5s)
- Throughput: X articles/min (target: >1000)

## Failure Mode Analysis

[Describe 3-5 systematic failure patterns]

## Production Readiness

**Assessment:** ✅ READY / ⚠️ READY WITH CAVEATS / ❌ NOT READY

**Caveats:** [If any]

**Recommendations:**
- [ ] Deploy to production: ✅ / ❌
- [ ] Monitor specific failure modes: [Which?]
- [ ] Plan for improvements: [What?]
```

---

## Phase 8: Documentation

**Goal**: Complete all documentation for production use and future maintenance

### Checklist

- [ ] **README.md** - Purpose, usage, quick start
- [ ] **Validation report** - Oracle calibration results
- [ ] **Training data report** - Dataset statistics and quality
- [ ] **Training report** - Model performance and metrics
- [ ] **Testing report** - Benchmark and integration test results
- [ ] **Release report** - Production readiness assessment
- [ ] **config.yaml comments** - All fields documented
- [ ] **Example usage** - Code snippets for common tasks
- [ ] **Known limitations** - What filter can't do

### Documentation Files

#### File 1: README.md

**Purpose:** Entry point for anyone using this filter

**Location:** `filters/{filter_name}/v1/README.md`

**Structure:**
```markdown
# {Filter Name} - Version 1.0

**Purpose:** [One-sentence description]
**Status:** ✅ PRODUCTION READY / ⏳ IN DEVELOPMENT
**Use Case:** [Primary use case]

## Quick Start

[3-5 lines showing how to use filter]

## What This Filter Does

[2-3 paragraphs explaining filter's purpose and approach]

## Dimensions (Scoring)

[List all dimensions with brief descriptions]

## Tier System

[Table showing tiers, thresholds, descriptions]

## Example Scores

[3-5 examples with scores and explanations]

## Usage

### Prefilter
[How to run prefilter]

### Oracle Scoring
[How to score with oracle]

### Model Inference
[How to use trained model]

## Performance

- Prefilter pass rate: X%
- Model MAE: X
- Tier classification accuracy: X%
- Inference time: Xms per article

## Known Limitations

[List 3-5 limitations or failure modes]

## Development History

- **v1.0** (YYYY-MM-DD): Initial release

## Production Readiness

**Status:** [READY / NOT READY]
**Last validated:** YYYY-MM-DD
**Next review:** YYYY-MM-DD
```

#### File 2: Validation Report

**Purpose:** Document oracle calibration process and results

**Location:** `filters/{filter_name}/v1/validation_report.md`

**See Phase 3 for template**

#### File 3: Training Data Report

**Purpose:** Document training dataset quality and statistics

**Location:** `reports/{filter_name}_training_data_report.md`

**See Phase 5 for template**

#### File 4: Training Report

**Purpose:** Document model training process and results

**Location:** `reports/{filter_name}_training_report.md`

**See Phase 6 for template**

#### File 5: Testing Report

**Purpose:** Document benchmark and integration testing

**Location:** `reports/{filter_name}_testing_report.md`

**See Phase 7 for template**

#### File 6: Release Report

**Purpose:** Final production readiness assessment

**Location:** `reports/{filter_name}_v1_release_report.md`

**Template:**
```markdown
# {Filter Name} v1 - Release Report

**Date:** YYYY-MM-DD
**Status:** ✅ PRODUCTION READY / ⚠️ READY WITH CAVEATS / ❌ NOT READY

## Executive Summary

[2-3 paragraphs summarizing filter development and production readiness]

## Development Timeline

- Planning: YYYY-MM-DD
- Architecture: YYYY-MM-DD
- Validation: YYYY-MM-DD
- Prefilter: YYYY-MM-DD
- Training data: YYYY-MM-DD
- Model training: YYYY-MM-DD
- Testing: YYYY-MM-DD
- Documentation: YYYY-MM-DD
- **Total time:** X weeks

## Quality Metrics

### Oracle Quality
- Validation sample: 100 articles
- Manual agreement: X%
- Status: ✅ PASS

### Prefilter Quality
- Pass rate: X% (target: Y-Z%)
- False negative rate: X% (target: <10%)
- Status: ✅ PASS

### Training Data Quality
- Size: X articles
- Tier distribution: Balanced
- Dimension coverage: Complete
- Status: ✅ PASS

### Model Quality
- Overall MAE: X (target: ≤1.5)
- Tier accuracy: X% (target: ≥85%)
- Oracle agreement: X% (target: ≥80%)
- Status: ✅ PASS

### Performance
- Inference time: Xms (target: <50ms)
- Throughput: X articles/min (target: >1000)
- Status: ✅ PASS

## Production Readiness Checklist

- [ ] Oracle validated: ✅
- [ ] Prefilter validated: ✅
- [ ] Training data collected: ✅
- [ ] Model trained: ✅
- [ ] Testing complete: ✅
- [ ] Documentation complete: ✅
- [ ] Deployment plan ready: ✅
- [ ] Monitoring plan ready: ✅

## Known Limitations

[List 3-5 limitations]

## Deployment Plan

[Describe deployment approach]

## Monitoring Plan

[Describe metrics to monitor in production]

## Rollback Plan

[Describe how to rollback if issues found]

## Sign-off

**Prepared by:** [Name]
**Reviewed by:** [Name]
**Approved for production:** ✅ / ❌
**Date:** YYYY-MM-DD
```

### Validation Criteria

**PASS:**
- All required files present
- README complete with examples
- All reports complete
- Known limitations documented
- Release report shows all checks passed

**REVIEW:**
- Some optional sections missing
- Examples could be more detailed
- Some limitations unclear

**FAIL:**
- README missing or incomplete
- Validation report missing
- Training/testing reports missing
- Release report not complete

---

## Phase 9: Deployment

**Goal**: Deploy to production with monitoring and rollback plan

### Checklist

- [ ] **Deployment environment** - Production infrastructure ready
- [ ] **Model uploaded** - Trained model accessible
- [ ] **Integration complete** - Filter integrated into pipeline
- [ ] **Smoke test** - Process 100 test articles successfully
- [ ] **Monitoring configured** - Metrics, alerts, logging
- [ ] **Rollback plan** - Can revert to previous version
- [ ] **Documentation updated** - Production URLs, access info
- [ ] **Team notified** - Stakeholders aware of new filter

### Deployment Process

#### Step 1: Pre-Deployment Checklist

```markdown
## Pre-Deployment Checklist

- [ ] All Phase 8 documentation complete
- [ ] Release report approved
- [ ] Model artifacts ready (weights, config)
- [ ] Prefilter code reviewed and tested
- [ ] Postfilter config validated
- [ ] Integration tests passed
- [ ] Performance benchmarks met
- [ ] Rollback plan documented
```

#### Step 2: Deploy to Staging

```bash
# Deploy model to staging environment
python scripts/deploy.py \
  --model filters/{filter_name}/v1_distillation/final \
  --config filters/{filter_name}/v1/config.yaml \
  --environment staging

# Run smoke test
python scripts/smoke_test.py \
  --filter {filter_name} \
  --environment staging \
  --sample-size 100
```

**Verify:**
- Model loads correctly
- Prefilter works
- Postfilter classifies correctly
- Latency within target (<50ms)
- No errors or crashes

#### Step 3: Deploy to Production

```bash
# Deploy to production
python scripts/deploy.py \
  --model filters/{filter_name}/v1_distillation/final \
  --config filters/{filter_name}/v1/config.yaml \
  --environment production

# Run smoke test in production
python scripts/smoke_test.py \
  --filter {filter_name} \
  --environment production \
  --sample-size 100
```

#### Step 4: Configure Monitoring

**Metrics to monitor:**
- Throughput (articles/minute)
- Latency (ms per article)
- Error rate (% failed)
- Tier distribution (% per tier)
- Prefilter pass rate (%)
- Model score distribution (per dimension)

**Alerts:**
- Error rate >5%
- Latency >100ms (p95)
- Tier distribution shift >20% from validation
- Prefilter pass rate drops >50%

#### Step 5: Gradual Rollout (Optional)

**Approach:** Route small % of traffic to new filter, gradually increase

**Process:**
```
Day 1: 10% traffic → Monitor closely
Day 2: 25% traffic → Check metrics
Day 3: 50% traffic → Compare to baseline
Day 4: 75% traffic → Final checks
Day 5: 100% traffic → Full rollout
```

### Monitoring Dashboards

#### Dashboard 1: Health Metrics

**Metrics:**
- Requests per minute
- Success rate
- Error rate
- P50/P95/P99 latency

**Alerts:**
- Success rate <95%
- P95 latency >100ms
- Error rate >5%

#### Dashboard 2: Quality Metrics

**Metrics:**
- Tier distribution over time
- Per-dimension score distribution
- Prefilter pass rate
- Articles flagged for reasoning

**Alerts:**
- Tier distribution shifts >20% from baseline
- Prefilter pass rate drops >50%
- Score distributions diverge from validation

#### Dashboard 3: Cost/Performance

**Metrics:**
- Inference cost per article (if cloud)
- GPU utilization (if local)
- Throughput (articles/hour)
- Resource usage (CPU, memory, GPU)

### Rollback Plan

**Trigger:** Critical issue found in production

**Process:**
```bash
# Revert to previous version
python scripts/rollback.py \
  --filter {filter_name} \
  --to-version v0 \
  --environment production

# Verify rollback
python scripts/smoke_test.py \
  --filter {filter_name} \
  --environment production
```

**Post-rollback:**
1. Investigate issue
2. Fix in development
3. Re-test thoroughly
4. Attempt deployment again

### Validation Criteria

**PASS:**
- Smoke test passes in production
- Monitoring configured and working
- All metrics within expected ranges
- No critical alerts
- Rollback plan tested and ready

**REVIEW:**
- Some metrics slightly outside range (investigate)
- Monitoring incomplete (add missing metrics)
- Rollback plan not tested (test it)

**FAIL:**
- Smoke test fails
- Critical errors in production
- Monitoring not configured
- No rollback plan

### Post-Deployment

#### Week 1: Close Monitoring

- Check dashboards daily
- Review sample of classified articles
- Collect user feedback
- Document any issues

#### Month 1: Quality Review

- Analyze tier distribution vs validation
- Check for distribution shift
- Review edge cases and failures
- Plan improvements for v2

#### Quarterly: Revalidation

- Sample 100 articles
- Score with oracle
- Compare model vs oracle
- Check if model still accurate

---

## Common Pitfalls Across All Phases

### 1. Oracle Outputs Tier Classification

**Symptom:** JSON output includes "tier", "signal_tier", "deployment_stage"

**Impact:** Violates architecture, confuses oracle role, can't adjust thresholds without re-labeling

**Prevention:**
- Check JSON schema carefully in Phase 2
- Validate in Phase 3
- Verify in Phase 5 training data

**Fix:** Remove from prompt, add to post-processing section

### 2. Weak Gatekeepers

**Symptom:** Gatekeeper rules not actually enforced

**Impact:** Articles that should be capped score higher than intended

**Prevention:**
- Make gatekeeper rules explicit and strong
- Test enforcement in Phase 3
- Validate in training data (Phase 5)

**Fix:** Strengthen prompt language, add post-filter enforcement

### 3. Prefilter Too Aggressive

**Symptom:** High false negative rate (good articles blocked)

**Impact:** Filter misses valuable content, low yield

**Prevention:**
- Test on large sample in Phase 4
- Prioritize avoiding false negatives
- Iterate on rules

**Fix:** Loosen blocking rules, add exceptions

### 4. Skewed Training Data

**Symptom:** 70%+ articles in one tier

**Impact:** Model biased toward majority class, poor performance on rare tiers

**Prevention:**
- Plan sampling strategy in Phase 1
- Use stratified sampling in Phase 5
- Validate distribution before training

**Fix:** Resample to balance tiers, target rare tier sources

### 5. No Calibration

**Symptom:** Using arbitrary thresholds without validation

**Impact:** Tier distribution doesn't match use case, poor user experience

**Prevention:**
- Calibrate in Phase 3 on real data
- Adjust thresholds based on desired distribution
- Document rationale

**Fix:** Run calibration, analyze results, adjust thresholds

### 6. Insufficient Testing

**Symptom:** Deploy without benchmark or edge case testing

**Impact:** Production failures, poor quality, rollback required

**Prevention:**
- Complete Phase 7 thoroughly
- Test edge cases, integration, performance
- Benchmark vs oracle

**Fix:** Run full testing suite before deployment

### 7. Missing Documentation

**Symptom:** No README, validation report, or release report

**Impact:** Hard to maintain, unclear how to use, difficult to debug

**Prevention:**
- Complete Phase 8 before deployment
- Document as you go, not at the end
- Use templates

**Fix:** Write documentation retrospectively (harder but necessary)

### 8. No Monitoring

**Symptom:** Deploy to production without metrics or alerts

**Impact:** Issues go unnoticed, can't measure quality, blind deployment

**Prevention:**
- Plan monitoring in Phase 9
- Set up dashboards before deployment
- Define alerts

**Fix:** Add monitoring ASAP after deployment

---

## Success Metrics Summary

### Phase 1: Planning
- ✅ 6-8 dimensions defined
- ✅ Tier scheme documented
- ✅ Gatekeepers identified
- ✅ Weights sum to 1.0

### Phase 2: Architecture
- ✅ Harmonized structure
- ✅ Oracle outputs dimensional only
- ✅ Inline filters present
- ✅ Post-processing documented

### Phase 3: Validation
- ✅ 50+ articles scored
- ✅ Success rate ≥95%
- ✅ No classification in output
- ✅ Score distribution reasonable
- ✅ Gatekeepers working
- ✅ Manual agreement ≥70%

### Phase 4: Prefilter
- ✅ Pass rate 30-50% (adjust for filter)
- ✅ False negative rate <10%
- ✅ Speed <10ms per article

### Phase 5: Training Data
- ✅ 5K+ articles scored
- ✅ Tier distribution balanced
- ✅ Dimension coverage complete
- ✅ No classification artifacts
- ✅ Gatekeepers enforced

### Phase 6: Training
- ✅ Overall MAE ≤1.5
- ✅ Per-dimension MAE ≤2.0
- ✅ Tier accuracy ≥85%

### Phase 7: Testing
- ✅ Oracle agreement ≥80%
- ✅ Edge cases ≥70% correct
- ✅ Integration test passes
- ✅ Inference <50ms

### Phase 8: Documentation
- ✅ All reports complete
- ✅ README comprehensive
- ✅ Known limitations documented

### Phase 9: Deployment
- ✅ Smoke test passes
- ✅ Monitoring configured
- ✅ Rollback plan ready
- ✅ Production stable

---

## Agent Usage Examples

### Example 1: Starting New Filter

**User:** "I want to create a new filter for identifying breakthrough scientific research with real-world applications."

**Agent Response:**

```
Let's guide you through filter development! Starting with Phase 1: Planning.

## Phase 1: Planning Checklist

We need to define the filter's purpose, dimensions, and tier scheme.

### Questions to Answer:

1. **Purpose statement** (one sentence):
   "Identify breakthrough scientific research with real-world applications"

2. **Use case** - Who will use this and why?
   [Need your input]

3. **Scope definition**:
   - IN SCOPE: [What types of research?]
   - OUT OF SCOPE: [What to exclude?]

4. **Dimensions** - I suggest 6-8 dimensions. For scientific research, consider:
   - Scientific rigor (methodology, evidence quality)
   - Real-world application (practical value)
   - Breakthrough level (novelty, paradigm shift)
   - Validation status (replicated, validated)
   - Impact potential (how many people/systems affected)
   - Technical readiness (lab → field → commercial)

   Do these make sense? What would you add/change?

5. **Tier scheme**:
   - Breakthrough (8.0+): Paradigm-shifting, validated, high impact
   - Significant (6.0-7.9): Important advance, solid evidence
   - Promising (4.0-5.9): Good research, needs validation
   - Early stage (2.0-3.9): Interesting but preliminary
   - Weak (0-1.9): Poor quality or no real-world relevance

   Does this work for your use case?

6. **Gatekeepers** - Hard requirements. Suggestions:
   - Must have peer-reviewed publication (validation_status ≥4)
   - Must have real-world application (not pure theory)

   What gatekeepers make sense?

7. **Weights** - How important is each dimension? Draft:
   - Scientific rigor: 0.25 (critical)
   - Real-world application: 0.20
   - Breakthrough level: 0.20
   - Validation status: 0.15
   - Impact potential: 0.15
   - Technical readiness: 0.05

   Weights sum to 1.0 ✅

Please provide feedback on each section. Once we finalize planning, we'll move to Phase 2: Architecture.
```

### Example 2: Reviewing Existing Filter

**User:** "Review my sustainability filter for production readiness."

**Agent Response:**

```
I'll review your filter across all 9 phases and provide a readiness assessment.

## Filter Review: sustainability-filter v1

### Phase 1: Planning ✅ PASS
- Purpose: Clear and specific
- Dimensions: 8 defined (good)
- Tier scheme: Documented
- Gatekeepers: deployment_maturity ≥5.0 identified
- Weights: Sum to 1.0 ✅

### Phase 2: Architecture ⚠️ REVIEW
- Header: ✅ Complete
- Scope section: ✅ Present
- ARTICLE placement: ✅ After scope
- Inline filters: ⚠️ Present but could be more specific
- Output format: ❌ CRITICAL - Oracle outputs "deployment_stage" field
  **Fix required:** Remove deployment_stage from JSON schema, add to post-processing
- Post-processing: ✅ Present
- CHANGELOG: ✅ Present

**Issue:** Oracle outputs tier classification (violates architecture)

### Phase 3: Validation ❓ NOT FOUND
- No validation report found
- **Action required:** Run oracle calibration on 100 article sample

### Phase 4: Prefilter ❓ NOT TESTED
- Prefilter code found: sustainability_prefilter.py
- No validation report
- **Action required:** Test on 1K article sample, measure pass rate and false negatives

### Phase 5-9: NOT STARTED

## Production Readiness: ❌ NOT READY

### Blockers:
1. **CRITICAL:** Oracle outputs tier classification (Phase 2)
2. **REQUIRED:** No validation report (Phase 3)
3. **REQUIRED:** Prefilter not tested (Phase 4)
4. **REQUIRED:** No training data collected (Phase 5)
5. **REQUIRED:** No model trained (Phase 6)

### Recommended Next Steps:

**Step 1:** Fix architecture issue (30 minutes)
- Remove "deployment_stage" from JSON schema in prompt
- Add tier calculation to post-processing section
- Run harmonization check to verify

**Step 2:** Run oracle calibration (2 hours)
```bash
# Sample 100 articles
python scripts/sample_articles.py \
  --source datasets/raw/master_dataset.jsonl \
  --output validation_sample.jsonl \
  --count 100 \
  --seed 42

# Score with oracle
python -m ground_truth.batch_scorer \
  --filter filters/sustainability-filter/v1 \
  --source validation_sample.jsonl \
  --output-dir sandbox/sustainability_validation \
  --llm gemini-flash

# Analyze results
python scripts/analyze_scoring_results.py \
  --results sandbox/sustainability_validation/scores.jsonl \
  --output filters/sustainability-filter/v1/validation_report.md
```

**Step 3:** Test prefilter (3 hours)
[Commands provided]

Would you like me to generate a detailed roadmap for getting to production?
```

### Example 3: Debugging Validation Issues

**User:** "My oracle validation shows all scores are 8-10, nothing below 7. What's wrong?"

**Agent Response:**

```
This is a common calibration issue. Let's diagnose:

## Issue: Skewed Score Distribution (All High)

### Possible Causes:

1. **Validation sample not representative**
   - Check: Is validation sample random from full corpus?
   - Or: Did you manually select "good" articles?

   **Fix:** Use random sampling:
   ```bash
   python scripts/sample_articles.py \
     --source datasets/raw/full_corpus.jsonl \
     --output validation_sample.jsonl \
     --count 100 \
     --seed 42 \
     --random  # Important!
   ```

2. **Scoring rubric too lenient**
   - Check: Are 0-4 range descriptions specific enough?
   - Issue: If everything scores high, low-range rubrics might be too strict

   **Fix:** Review prompt, clarify when to score 0-6:
   - 0-2: [Make this very explicit - what MUST be absent/wrong]
   - 3-4: [Describe minimal/weak cases]
   - 5-6: [Describe moderate cases]

3. **Inline filters not working**
   - Check: Do dimensions have ❌ CRITICAL FILTERS sections?
   - Issue: Without filters, oracle doesn't check for low-scoring criteria

   **Fix:** Add inline filters to each dimension:
   ```markdown
   **❌ CRITICAL FILTERS - If article is ANY of these, score 0-2:**
   - [Specific disqualifying criteria]
   - [Examples of what should score low]
   ```

4. **Examples show only high scores**
   - Check: Does prompt have examples of low-scoring articles?
   - Issue: Oracle learns to score everything high

   **Fix:** Add 2-3 low-scoring examples (overall ≤3.0)

### Diagnosis Process:

**Step 1:** Check validation sample
```bash
# Manually review 10 articles from validation sample
# Should these ALL score 8-10? Or is sample biased?
```

**Step 2:** Review prompt rubrics
```bash
# For each dimension, check if 0-4 range is clearly defined
# Can you think of articles that SHOULD score 0-4?
```

**Step 3:** Test with known low-quality article
```bash
# Find article that should obviously score low
# Score it with oracle
# If scores high, prompt is broken
```

### Quick Fix:

1. Add 3 low-scoring examples to prompt
2. Strengthen inline filters (more specific criteria for 0-2)
3. Re-run validation on random sample (not curated)

Would you like me to review your prompt and suggest specific changes?
```

---

## Related Agents

- **filter-harmonizer**: Check structural consistency with established patterns
- **oracle-calibration-agent**: Specialized validation and calibration support
- **dimensional-regression-qa-agent**: Validate training data quality
- **model-evaluation-agent**: Evaluate trained model performance

---

## Related Documentation

- **FILTER_WORKFLOW.md**: Original filter development workflow
- **FILTER_CHECKLIST.md**: Quick checklist for filter creation
- **FILTER_HARMONIZATION_GUIDE.md**: Harmonization principles and patterns
- **Post-filter architecture**: `docs/decisions/2025-11-13-post-filter-architecture.md`
- **Oracle output discipline**: `docs/decisions/2025-11-13-remove-tier-classification-from-oracle.md`
- **Inline filters**: `docs/decisions/2025-11-14-inline-filters-for-fast-models.md`

---

## Version History

### v1.0 (2025-11-17)
- Initial comprehensive filter development guide
- 9-phase lifecycle with detailed checklists
- Validation criteria for each phase
- Common pitfalls and fixes
- Complete documentation templates
- Example agent interactions
