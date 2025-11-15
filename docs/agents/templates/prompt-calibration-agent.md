# Prompt Calibration Agent Template v1.0

**Purpose:** Validate oracle prompt quality before expensive batch scoring by systematically reviewing a calibration sample.

**When to use:** After writing initial oracle prompt, before batch scoring full dataset.

**Expected duration:** 5-10 minutes (reviewing 50 calibration labels)

---

## Agent Task Description

You are validating an oracle prompt to ensure it correctly scores articles before we invest in batch scoring thousands of articles. Use the criteria below to generate a comprehensive calibration report.

**Input artifacts:**
- Oracle prompt: `filters/{filter_name}/v1/prompt-compressed.md`
- Filter config: `filters/{filter_name}/v1/config.yaml`
- Calibration sample (oracle-labeled): `calibration_labeled.jsonl` (50-100 articles)
- Expected classifications: `calibration_sample_expected.json` (optional ground truth)

**Your responsibilities:**
1. Review oracle-labeled calibration articles
2. Identify off-topic articles that scored too high (false positives)
3. Identify on-topic articles that scored too low (false negatives)
4. Analyze dimensional score patterns
5. Check if oracle follows prompt instructions
6. Generate calibration report with PASS/FAIL decision
7. Recommend specific prompt improvements

---

## Calibration Criteria

### CRITICAL (Must Pass)

#### 1. Off-Topic Rejection Rate

**What to check:**
- Articles clearly OUT OF SCOPE should score low (overall < 3.0)
- Generic tech (AWS, Excel, programming) should NOT score as climate tech
- Unrelated domains (healthcare, finance, entertainment) should be rejected

**Review process:**
```python
# For each labeled article in calibration_labeled.jsonl:
# 1. Read title + content
# 2. Determine if it's in-scope for the filter
# 3. Check oracle's overall_score
# 4. Flag if off-topic article scored >= 5.0

off_topic_articles = []
for article in calibration_sample:
    if is_obviously_off_topic(article):
        score = article['{filter_name}_analysis']['overall_score']
        if score >= 5.0:
            off_topic_articles.append({
                'title': article['title'],
                'score': score,
                'why_off_topic': explain_why_off_topic(article),
                'oracle_reasoning': article['{filter_name}_analysis']['overall_assessment']
            })
```

**Pass criteria:**
- ✅ <10% of off-topic articles score >= 5.0 (< 5 out of 50)
- ✅ <5% of off-topic articles score >= 7.0 (< 2-3 out of 50)
- ❌ FAIL if >20% of off-topic articles score >= 5.0

**Example failure (from sustainability_tech_deployment):**

```
❌ OFF-TOPIC HIGH SCORES DETECTED

1. "Part 1: Understanding AWS IAM" → 10.0
   Why off-topic: Generic cloud infrastructure, not climate/sustainability tech
   Oracle reasoning: "IAM is foundational AWS service, mass deployed, enables cloud adoption"
   Issue: Oracle scored maturity of technology, not climate relevance

2. "Excel Still Relevant in Era of Python?" → 9.4
   Why off-topic: Office productivity software, unrelated to climate
   Oracle reasoning: "Excel is mature, widely deployed data analysis tool"
   Issue: Oracle doesn't recognize scope limitation

3. "Your Toothbrush is Bristling with Bacteria" → 9.25
   Why off-topic: Personal hygiene, not sustainability tech
   Oracle reasoning: "Toothbrushes are mass-deployed hygiene technology"
   Issue: Oracle completely missed the point of the filter
```

**Root cause analysis:**
- Prompt lacks explicit SCOPE definition
- Oracle interprets "deployment maturity" too broadly
- No IN SCOPE / OUT OF SCOPE examples

**Recommendation:**
```markdown
Add to prompt:

**SCOPE: Climate & Sustainability Technology ONLY**

**IN SCOPE (score normally):**
- Renewable energy (solar, wind, hydro, geothermal)
- Energy storage (batteries, pumped hydro)
- Electric vehicles and charging
- Heat pumps and efficient HVAC
- Green hydrogen and fuel cells
- Carbon capture and removal
- Sustainable agriculture tech
- Circular economy / recycling
- Grid modernization for renewables

**OUT OF SCOPE (score 0-2 on ALL dimensions):**
- Generic IT infrastructure (cloud, databases, APIs)
- Programming languages and frameworks
- Office productivity software
- Generic hardware (not climate-specific)
- Social media platforms
- Gaming and entertainment
- Healthcare tech (unless climate/sustainability related)
```

#### 2. On-Topic Recognition Rate

**What to check:**
- Articles clearly IN SCOPE should score reasonably (overall >= 5.0 if deployed)
- Climate tech deployment stories should score high if actually deployed
- Oracle should recognize positive examples

**Review process:**
```python
on_topic_low_scores = []
for article in calibration_sample:
    if is_obviously_on_topic(article):
        score = article['{filter_name}_analysis']['overall_score']
        # If it's deployed climate tech but scored low, that's a problem
        if article_describes_deployment(article) and score < 5.0:
            on_topic_low_scores.append({
                'title': article['title'],
                'score': score,
                'why_on_topic': explain_why_on_topic(article),
                'expected_score': estimate_expected_score(article)
            })
```

**Pass criteria:**
- ✅ <20% of on-topic deployed tech scores < 5.0 (< 10 out of 50)
- ✅ At least some articles score 7+ (shows oracle can recognize good examples)
- ❌ FAIL if >30% of on-topic articles score < 5.0

**Example issues:**

```
⚠️ ON-TOPIC LOW SCORES

1. "Solar Farm Powers 10,000 Homes in Texas" → 4.2
   Expected: 7-8 (deployed renewable energy at scale)
   Oracle reasoning: "Limited scale, only 10,000 homes"
   Issue: Oracle scale calibration wrong (10k homes is significant)

2. "EV Sales Hit 20% Market Share in Europe" → 5.8
   Expected: 8-9 (mass market adoption milestone)
   Oracle reasoning: "Still minority of market"
   Issue: Oracle doesn't recognize 20% as significant penetration
```

**Root cause:** Prompt scale examples don't match real-world benchmarks

**Recommendation:** Calibrate scale examples in prompt dimensions

#### 3. Dimensional Consistency

**What to check:**
- Oracle uses all 8 dimensions (not just scoring one high and rest low)
- Dimensional scores make sense individually
- Dimension weights are respected (via post-filter, not oracle's concern)

**Review process:**
```python
# Check if oracle is using dimension variance
for article in calibration_sample:
    dims = article['{filter_name}_analysis']['dimensions']
    variance = calculate_variance(dims.values())

    if variance < 0.5:
        # All dimensions nearly identical - oracle may not be differentiating
        flag_low_variance(article)
```

**Pass criteria:**
- ✅ Average dimensional variance > 1.0 (oracle differentiates dimensions)
- ✅ No single dimension always scores 10 or always scores 0
- ⚠️ REVIEW if >50% of articles have variance < 0.5

**Example issue:**

```
⚠️ LOW DIMENSIONAL VARIANCE

Multiple articles have all dimensions = 10.0:
- AWS IAM: [10, 10, 10, 10, 10, 10, 10, 10]
- Excel: [10, 8, 9, 10, 9, 10, 10, 7]

Issue: Oracle may be using "mature technology" heuristic for all dimensions
instead of evaluating each dimension independently.
```

### QUALITY (Report but Don't Block)

#### 4. Oracle Reasoning Quality

**What to check:**
- Oracle provides clear reasoning for scores
- Reasoning references specific evidence from article
- Reasoning explains why score assigned

**Review sample articles:**
- Read 5-10 oracle reasoning statements
- Check if reasoning makes sense
- Flag vague or circular reasoning

**Example good reasoning:**
```
"This solar farm generates 50 MW and has been operational for 3 years,
demonstrating deployment maturity (7/10). Cost has declined 40% since 2020,
showing strong cost trajectory (8/10). Market penetration is still limited
to utility-scale projects (5/10)."
```

**Example bad reasoning:**
```
"This is a mature technology that is widely deployed."
(Circular, doesn't reference article specifics)
```

#### 5. Edge Case Handling

**What to check:**
- Borderline articles (carbon credits, offset programs, green finance)
- Fossil fuel transition stories (oil companies investing in renewables)
- Negative news about climate tech (solar panel factory closes)

**Review edge cases:**
```python
edge_cases = [a for a in calibration_sample if a['category'] == 'edge_case']
for article in edge_cases:
    score = article['{filter_name}_analysis']['overall_score']
    # Check if score is reasonable given context
    validate_edge_case_handling(article, score)
```

**Example edge cases:**

```
✅ GOOD: "Oil Company Announces $5B Wind Farm Investment" → 6.5
   Reasoning: Wind farm deployment is in scope, despite being oil company.
   Score reflects early-stage commitment, not yet deployed.

⚠️ QUESTIONABLE: "Carbon Offset Program Plants Trees" → 7.0
   Reasoning: Tree planting is nature-based solution, but effectiveness disputed.
   May want to score lower (4-5) due to measurement challenges.

❌ BAD: "Solar Panel Factory Closes Due to Cheap Imports" → 8.0
   Reasoning: Factory closure is NEGATIVE news, shouldn't score high.
   Oracle may have scored the *topic* (solar) not the *news* (factory closing).
```

---

## Calibration Decision Matrix

### ✅ PASS (Prompt is Validated)

**Criteria:**
- Off-topic rejection: <10% false positives ✅
- On-topic recognition: <20% false negatives ✅
- Dimensional consistency: Variance > 1.0 ✅
- No systematic errors detected ✅

**Recommendation:**
```
DECISION: PASS - Prompt is Validated

Off-topic rejection rate: 95% (47/50 off-topic articles scored < 5.0)
On-topic recognition rate: 85% (17/20 on-topic articles scored >= 5.0)
Dimensional variance: 1.8 (good differentiation)

Minor issues:
- 2 edge cases need review (carbon offsets scored higher than expected)

Recommendation: PROCEED TO BATCH LABELING

The prompt correctly distinguishes in-scope from out-of-scope content
and produces reasonable dimensional scores. Minor edge case issues do not
warrant blocking batch scoring.

Next steps:
1. Review 2 edge case articles manually
2. Consider adding edge case examples to prompt
3. Proceed with batch scoring of full dataset
```

### ⚠️ REVIEW (Borderline - Fix Minor Issues)

**Criteria:**
- Off-topic rejection: 10-20% false positives
- OR on-topic recognition: 20-30% false negatives
- OR some dimensional scoring issues
- OR edge case handling inconsistent

**Recommendation:**
```
DECISION: REVIEW - Minor Prompt Issues Detected

Off-topic rejection rate: 85% (5 out of 50 off-topic articles scored >= 5.0)
Issue: Oracle scored some generic tech articles as "mature deployment"

On-topic recognition rate: 75% (acceptable)
Dimensional variance: 1.5 (good)

Recommended fixes:
1. Add explicit OUT OF SCOPE section listing generic tech domains
2. Add examples of what "climate tech" means
3. Re-test on same 50 articles after fix

Options:
1. Fix prompt now and re-calibrate (recommended)
2. Accept minor issues and proceed (not recommended - will label 400+ articles incorrectly)

Time to fix: 30 minutes (update prompt) + 5 minutes (re-label 50 articles)
Cost to fix: $0.05 (re-label same 50 articles)
Cost if not fixed: ~$1-2 (400-800 mis-labeled articles in batch run)

Recommendation: FIX PROMPT BEFORE BATCH LABELING
```

### ❌ FAIL (Prompt Needs Major Revision)

**Criteria:**
- Off-topic rejection: >20% false positives (10+ out of 50)
- OR on-topic recognition: >30% false negatives
- OR systematic misunderstanding of filter purpose
- OR oracle ignoring prompt instructions

**Recommendation:**
```
DECISION: FAIL - Prompt Requires Major Revision

Off-topic rejection rate: 40% (20 out of 50 off-topic articles scored >= 5.0)
Issue: Oracle does not understand filter scope - scoring ANY mature technology
as climate tech deployment, including AWS, Excel, toothbrushes.

Root cause analysis:
- Prompt lacks explicit SCOPE definition
- No IN SCOPE / OUT OF SCOPE examples
- Dimension descriptions emphasize "deployment maturity" without "climate tech" context

Critical failures:
1. "AWS IAM Tutorial" → 10.0 (should be 0-2)
2. "Excel vs Python for Data Analysis" → 9.4 (should be 0-2)
3. "Toothbrush Bacteria Study" → 9.25 (should be 0-2)

Oracle reasoning shows fundamental misunderstanding:
"IAM is mature, mass-deployed technology enabling cloud adoption"
→ Oracle scored IT maturity, not climate tech deployment

BLOCKING ISSUE: Batch labeling 8,000 articles with this prompt would
result in ~3,200 mis-labeled articles (40% false positive rate).

Required actions:
1. Add SCOPE section to prompt (CRITICAL)
2. Add IN SCOPE / OUT OF SCOPE lists with examples
3. Clarify each dimension description includes "climate/sustainability tech" context
4. Re-label calibration sample
5. Re-evaluate before proceeding

DO NOT PROCEED TO BATCH LABELING until this passes calibration.

Estimated fix time: 1-2 hours (major prompt revision)
Re-calibration cost: $0.05
Savings vs batch scoring broken prompt: $8-16 + days of rework
```

---

## Calibration Report Template

Use this structure for your calibration report:

```markdown
# Prompt Calibration Report: {filter_name}

**Date:** {date}
**Filter:** {filter_name}
**Oracle:** {oracle_model} (e.g., Gemini Flash 1.5)
**Calibrator:** Prompt Calibration Agent v1.0

---

## Executive Summary

**Decision:** ✅ PASS / ⚠️ REVIEW / ❌ FAIL

**Overall Assessment:** [One sentence summary]

**Recommendation:** [PROCEED / FIX MINOR ISSUES / MAJOR REVISION REQUIRED]

---

## Calibration Sample Overview

**Total articles reviewed:** {n}
- On-topic (expected high scores): {n_positive}
- Off-topic (expected low scores): {n_negative}
- Edge cases: {n_edge}

**Oracle used:** {oracle_model}
**Prompt version:** {prompt_file}

---

## CRITICAL METRICS

### 1. Off-Topic Rejection Rate

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Off-topic articles reviewed | {n} | N/A | ℹ️ |
| Scored < 5.0 (correctly rejected) | {n} ({pct}%) | >90% | ✅/⚠️/❌ |
| Scored >= 5.0 (false positives) | {n} ({pct}%) | <10% | ✅/⚠️/❌ |
| Scored >= 7.0 (severe false positives) | {n} ({pct}%) | <5% | ✅/⚠️/❌ |

**Status:** ✅ PASS / ⚠️ REVIEW / ❌ FAIL

#### False Positive Examples

[If any false positives found, list top 3-5:]

**1. "{article_title}" → {score}**
- **Why off-topic:** {explanation}
- **Oracle reasoning:** "{oracle_reasoning}"
- **Issue:** {what went wrong}

**2. ...**

#### Root Cause Analysis

[Explain systematic pattern in false positives]

**Example:**
```
Pattern: Oracle scored generic IT infrastructure (AWS, databases, programming)
as "mass-deployed technology" without checking climate relevance.

Root cause: Prompt lacks explicit SCOPE definition. Dimensions emphasize
"deployment maturity" without "climate/sustainability tech" qualifier.
```

### 2. On-Topic Recognition Rate

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| On-topic articles reviewed | {n} | N/A | ℹ️ |
| Scored >= 5.0 (correctly recognized) | {n} ({pct}%) | >80% | ✅/⚠️/❌ |
| Scored < 5.0 (false negatives) | {n} ({pct}%) | <20% | ✅/⚠️/❌ |
| At least one article >= 7.0 | Yes/No | Yes | ✅/❌ |

**Status:** ✅ PASS / ⚠️ REVIEW / ❌ FAIL

#### False Negative Examples

[If any false negatives found, list them:]

**1. "{article_title}" → {score}**
- **Why on-topic:** {explanation}
- **Expected score:** {expected_score}
- **Oracle reasoning:** "{oracle_reasoning}"
- **Issue:** {what went wrong}

### 3. Dimensional Consistency

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Average dimensional variance | {variance} | >1.0 | ✅/⚠️/❌ |
| Articles with variance < 0.5 | {n} ({pct}%) | <20% | ✅/⚠️/❌ |
| All dimensions used (not all 0 or all 10) | Yes/No | Yes | ✅/❌ |

**Status:** ✅ PASS / ⚠️ REVIEW

---

## QUALITY CHECKS

### 4. Oracle Reasoning Quality

**Sample reasoning review (5-10 articles):**

✅ **Good reasoning examples:**
- "{article_title}": Oracle cited specific evidence, explained scores clearly
- ...

⚠️ **Weak reasoning examples:**
- "{article_title}": Vague reasoning, no article specifics
- ...

**Assessment:** [Good / Acceptable / Needs improvement]

### 5. Edge Case Handling

**Edge cases reviewed:** {n}

| Article | Score | Expected | Assessment |
|---------|-------|----------|------------|
| "{title}" | {score} | {expected} | ✅ Reasonable / ⚠️ Questionable / ❌ Wrong |
| ... | ... | ... | ... |

**Assessment:** [Good / Needs guidance / Inconsistent]

---

## Recommendations

### Immediate Actions

[Choose one based on DECISION:]

**If PASS:**
1. ✅ PROCEED TO BATCH LABELING
2. Document calibration results
3. Monitor first batch (500 articles) for quality

**If REVIEW:**
1. ⚠️ FIX MINOR ISSUES BEFORE BATCH LABELING
2. Specific fixes: [list specific prompt changes needed]
3. Re-calibrate (cost: $0.05, time: 30 min)

**If FAIL:**
1. ❌ MAJOR PROMPT REVISION REQUIRED - DO NOT BATCH LABEL
2. Critical fixes: [list critical prompt changes]
3. Re-calibrate after major revision
4. Consider expert review of prompt

### Specific Prompt Improvements

[List concrete suggestions:]

**1. Add SCOPE section (CRITICAL if FAIL):**
```markdown
**SCOPE: Climate & Sustainability Technology ONLY**

**IN SCOPE:**
- [List domains]

**OUT OF SCOPE:**
- [List exclusions]
```

**2. Clarify dimension descriptions:**
- Add "climate/sustainability tech" qualifier to each dimension
- Update scale examples to match real-world benchmarks

**3. Add edge case handling:**
- Provide examples of borderline cases and expected handling

---

## Appendix

### Files Reviewed

- Prompt: `filters/{filter_name}/v1/prompt-compressed.md`
- Config: `filters/{filter_name}/v1/config.yaml`
- Calibration sample: `calibration_labeled.jsonl` ({n} articles)

### Calibration Command

```bash
python scripts/label_batch.py \
    --filter filters/{filter_name}/v1 \
    --input calibration_sample.jsonl \
    --output calibration_labeled.jsonl
```

### Scoring Distribution

**Overall scores:**
- 0-2: {n} articles
- 3-4: {n} articles
- 5-6: {n} articles
- 7-8: {n} articles
- 9-10: {n} articles

**By expected category:**
- On-topic: Mean={mean}, Median={median}, Range=[{min}-{max}]
- Off-topic: Mean={mean}, Median={median}, Range=[{min}-{max}]
```

---

## Example Invocation

**Usage:**

```
Task: "Calibrate the {filter_name} oracle prompt using the calibration sample
in calibration_labeled.jsonl. Follow the Prompt Calibration Agent criteria from
docs/agents/templates/prompt-calibration-agent.md.

Calibration sample location: calibration_labeled.jsonl (50 articles)
Filter path: filters/{filter_name}/v1

Generate calibration report and determine if prompt is ready for batch scoring."
```

**Expected Agent Workflow:**

1. ✅ Read filter config and oracle prompt
2. ✅ Read calibration sample (oracle-labeled articles)
3. ✅ Analyze off-topic rejection rate
4. ✅ Analyze on-topic recognition rate
5. ✅ Check dimensional consistency
6. ✅ Review oracle reasoning quality
7. ✅ Evaluate edge case handling
8. ✅ Make PASS/REVIEW/FAIL decision
9. ✅ Generate calibration report: `filters/{filter_name}/v1/calibration_report.md`
10. ✅ Provide specific prompt improvement recommendations

---

## Success Criteria for Agent

**Agent completes successfully if:**
- ✅ All CRITICAL metrics evaluated
- ✅ Clear PASS/REVIEW/FAIL decision made
- ✅ Specific examples of failures provided (if any)
- ✅ Root cause analysis performed
- ✅ Concrete prompt improvement recommendations
- ✅ Report saved to filter directory: `filters/{filter_name}/v1/calibration_report.md`

**Agent quality markers:**
- Uses actual data from calibration sample (not guessed)
- Identifies specific problematic articles
- Explains WHY articles scored incorrectly
- Provides actionable prompt fixes
- Saves report to filter directory for portability

---

## Version History

### v1.0 (2025-11-13)
- Initial template
- Based on sustainability_tech_deployment calibration failure
- Criteria: <10% false positives, <20% false negatives
- Decision matrix: PASS / REVIEW / FAIL
- Designed for semi-automated review (agent + human validation)
