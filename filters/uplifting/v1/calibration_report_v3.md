# Uplifting Filter - Calibration Report v3

**Date:** 2025-11-14
**Prompt Version:** v1 with professional knowledge sharing fix
**Sample Size:** 40 articles → 11 labeled (prefilter + API failures)
**Oracle:** Gemini Flash 1.5

---

## Executive Summary

**DECISION: ❌ FAIL - Iteration Required**

**Critical Issue:** False positive rate increased from 20% (v2) to 87.5% (v3), despite adding professional knowledge sharing to OUT OF SCOPE section.

**Metrics:**
- Off-topic Rejection: **12.5%** (target: >90%) ❌
- On-topic Recognition: **100%** (target: >80%) ✅
- False Positive Rate: **87.5%** (7/8 off-topic articles)

**Comparison:**
- v1: 80% false positive rate (baseline, no fixes)
- v2: 20% false positive rate (added SCOPE, Doom-framing, Outcomes)
- v3: **87.5%** false positive rate (added professional knowledge sharing)

**V3 IS WORSE THAN V1** - The prompt fixes are not working.

---

## Analysis

### Articles Breakdown

**Total:** 11 articles
- **Off-topic:** 8 articles
- **On-topic:** 2 articles
- **Edge cases:** 1 article

### False Positives (7/8 off-topic articles scored >= 5.0)

| Score | Article | Issue | Prompt Rule Violated |
|-------|---------|-------|---------------------|
| 6.6 | Productivity trap (burnout advice) | Professional knowledge (business advice) | OUT OF SCOPE line 34 |
| 6.6 | API Gateway (Rust/Go tutorial) | Professional knowledge (developer tutorial) | OUT OF SCOPE line 34 |
| 6.4 | SNAP cuts (benefits slashed) | Doom-framed (>50% harm content) | Doom-Framing line 50 |
| 6.3 | AI development proposal | Speculation ("could lead to") | Outcome Requirement line 81 |
| 6.0 | Budgeting app alternatives | Productivity advice | OUT OF SCOPE line 38 |
| 5.1 | Learning programming | Professional knowledge (coding skills) | OUT OF SCOPE line 34 |
| 5.0 | Gaming company ($1B funding) | Business success without broad benefit | OUT OF SCOPE line 36 |

### True Positives (2/2 on-topic articles scored >= 5.0) ✅

| Score | Article | Reason |
|-------|---------|--------|
| 8.5 | Water Warriors (India water crisis) | Community-led solutions, documented outcomes |
| 7.5 | Salmon comeback (Klamath River) | Environmental restoration after 100 years |

### True Negatives (1/8 off-topic articles scored < 5.0)

| Score | Article | Reason |
|-------|---------|--------|
| 4.5 | Astranis satellite (military comms) | Military/business, correctly rejected |

---

## Root Cause Analysis

### Why Did the Fixes Not Work?

**All three previous fixes failed:**

1. **Professional knowledge sharing (3 false positives)**
   - Added to OUT OF SCOPE in line 34
   - Oracle still scored API tutorials (6.6), programming learning (5.1), business advice (6.6)
   - **Fix didn't work**

2. **Doom-framing (1 false positive)**
   - Added Doom-Framing section (lines 45-62) with ">50% harm = max 3-4" rule
   - Oracle scored SNAP cuts article at 6.4
   - **Fix didn't work**

3. **Speculation (1 false positive)**
   - Added Outcome Requirement (lines 65-82) with "could/might/may = 2-3 max" rule
   - Oracle scored AI development proposal at 6.3
   - **Fix didn't work**

4. **Productivity advice (1 false positive)**
   - Added to OUT OF SCOPE in line 38
   - Oracle scored budgeting app at 6.0
   - **Fix didn't work**

5. **Business success (1 false positive)**
   - Should be capped by business_news + collective_benefit < 6 → max 4.0
   - Oracle scored gaming company at 5.0 (with CB=6, so no cap applied)
   - **Content cap logic may have worked, but CB scoring is too generous**

### Hypothesis: Why Prompt Instructions Are Being Ignored

**Possible causes:**

1. **Prompt structure issue**
   - Instructions may be buried or not prominent enough
   - OUT OF SCOPE section appears early (lines 31-40), but oracle may not apply it during dimensional scoring
   - Dimensional scoring instructions (lines 104-140) don't reference OUT OF SCOPE rules

2. **Model capability limits**
   - Gemini Flash may not reliably follow complex multi-step rules
   - Fast models may optimize for speed over instruction adherence

3. **Conflicting instructions**
   - Dimensional definitions may conflict with OUT OF SCOPE rules
   - Example: Agency dimension asks "People taking effective action toward wellbeing?"
   - Oracle may interpret "learning programming" as agency toward personal wellbeing

4. **Examples override rules**
   - Validation examples at end (lines 167-189) may bias oracle
   - None of the examples show professional knowledge sharing or productivity advice
   - Oracle may not have clear reference for rejecting these categories

---

## Specific Failures

### Example 1: API Gateway Tutorial (6.6 score)

**Article:** "Web Developer Travis McCracken on API Gateway Design with Rust and Go"

**Oracle reasoning:** "The article highlights a web developer's experience using Rust and Go to build scalable and high-performance API gateways..."

**Expected:** 0-2 (professional knowledge sharing - OUT OF SCOPE line 34)

**Actual scores:**
- Agency: 7 (oracle sees developer taking action)
- Collective_benefit: 7 (oracle sees knowledge sharing as beneficial)
- Overall: 6.57

**Issue:** Oracle interpreted professional knowledge sharing as having collective benefit, ignoring OUT OF SCOPE rule.

---

### Example 2: SNAP Cuts (6.4 score)

**Article:** "SNAP payments slashed to 50% as federal shutdown stretches into its seventh week"

**Oracle reasoning:** "Despite the reduction in SNAP benefits, there is agency demonstrated by federal judges ruling in favor of..."

**Expected:** Max 3-4 (doom-framed - >50% harm, Doom-Framing line 50)

**Actual scores:**
- Agency: 7 (oracle focused on silver lining: judge ruling)
- Progress: 6
- Overall: 6.4

**Issue:** Oracle scored the silver lining, not the main content (harm), violating "Score the MAIN CONTENT, not silver linings in doom stories" (line 47).

---

### Example 3: AI Development Proposal (6.3 score)

**Article:** "Economic survival pressure vs capability scaling" (AI development approach)

**Oracle reasoning:** "The article proposes a novel approach to AI development that **could lead to** beneficial outcomes..."

**Expected:** 2-3 (speculation - "could lead to", Outcome Requirement line 81)

**Actual scores:**
- Agency: 7
- Innovation: 6
- Overall: 6.27

**Issue:** Oracle scored speculation as if it were documented outcomes, violating "If article uses could/might/may/promises/aims to without showing results → score 2-3" (line 81).

---

## Pattern: Oracle Is Not Applying Filters

**Observation:** Oracle is scoring dimensional questions directly without applying OUT OF SCOPE / Doom-Framing / Outcome filters first.

**Prompt flow problem:**

```
Current prompt structure:
1. IN SCOPE / OUT OF SCOPE (lines 19-40)
2. Doom-Framing section (lines 45-62)
3. Outcome Requirement (lines 65-82)
4. Pre-classification (lines 91-103) ← Oracle may skip to here
5. Dimensional scoring (lines 104-140) ← Oracle focuses on this
```

**Hypothesis:** Oracle jumps directly to Step 2 (dimensional scoring) without properly applying Step 1 (pre-classification scope checks).

**Evidence:**
- All 7 false positives have dimensional scores that ignore scope rules
- Oracle reasoning shows dimensional thinking ("agency demonstrated", "novel approach") without scope filtering

---

## Recommendations

### Option A: Restructure Prompt (Fundamental Fix)

**Problem:** OUT OF SCOPE rules are separate from dimensional scoring, so oracle doesn't apply them consistently.

**Solution:** Integrate scope checks directly into dimensional definitions.

**Example - Rewrite Agency dimension:**

```markdown
1. **Agency**: People/communities taking effective action toward HUMAN WELLBEING or PLANETARY HEALTH?

   **CRITICAL FILTERS - Apply BEFORE scoring:**
   - ❌ Professional knowledge sharing (tutorials, business advice, coding) → Score 0-2
   - ❌ Speculation without outcomes ("could lead to", "aims to") → Score 0-2
   - ❌ Productivity advice (life hacks, self-help) → Score 0-2
   - ❌ Business success (funding, growth) → Score 0-2

   **If none of above, score normally:**
   - 0-2: None | 3-4: Limited | 5-6: Moderate | 7-8: Strong | 9-10: Transformative
```

**Repeat for all dimensions, especially Collective Benefit (gatekeeper).**

### Option B: Add Negative Examples (Simpler Fix)

**Problem:** Validation examples (lines 167-189) don't show professional knowledge sharing or productivity advice.

**Solution:** Add explicit negative examples showing what NOT to score high.

**Example:**

```markdown
LOW SCORE (1.5/10) - Professional Knowledge Sharing:
Article: "Senior developer shares API gateway design patterns using Rust and Go for high-performance microservices."
Scores: Agency=2, Progress=2, Collective=2, Connection=0, Innovation=2, Justice=0, Resilience=0, Wonder=0
Content Type: business_news (capped at 2)
Reasoning: "Professional knowledge sharing without addressing urgent human needs. Developer tutorial, not wellbeing progress."

LOW SCORE (1.8/10) - Productivity Advice:
Article: "5 best budgeting apps to replace Mint and optimize your personal finances."
Scores: Agency=2, Progress=2, Collective=2, Connection=0, Innovation=2, Justice=0, Resilience=0, Wonder=0
Reasoning: "Personal productivity advice, not systemic change or community benefit. Individual optimization, not wellbeing."
```

### Option C: Switch Oracle Model

**Problem:** Gemini Flash may not follow complex rules reliably.

**Options:**
1. **Gemini Pro** - Better instruction following, higher cost ($0.003/article vs $0.001)
2. **Claude Haiku** - May have better rule adherence

**Tradeoff:** 3x cost increase, but may eliminate need for multiple iterations.

### Option D: Two-Stage Filtering

**Problem:** Single-pass scoring tries to do too much at once.

**Solution:**
1. **Stage 1:** Scope filter (is this about human/planetary wellbeing? Yes/No)
2. **Stage 2:** Dimensional scoring (only if Stage 1 = Yes)

**Implementation:** Run prefilter twice, or add explicit scope-check prompt before dimensional scoring.

---

## Recommended Next Steps

**Immediate (Option B - Quickest):**
1. Add 3-5 negative examples to prompt showing:
   - Professional knowledge sharing (API tutorial, programming)
   - Productivity advice (budgeting apps, life hacks)
   - Doom-framed news (SNAP cuts with silver lining)
   - Speculation (AI proposal "could lead to")
2. Re-label same calibration sample (v3) with updated prompt
3. Check if false positive rate improves

**If Option B doesn't work (Option A - Fundamental):**
1. Restructure prompt to integrate scope filters into dimensional definitions
2. Make SCOPE section impossible to ignore by repeating it inline
3. Re-label with restructured prompt

**If both fail (Option C - Nuclear):**
1. Switch to Gemini Pro or Claude Haiku
2. Accept 3x cost increase
3. Re-label with better model

---

## Validation Status

**Validation sample status:** Created (40 articles, seed=4000), not yet labeled

**Decision:** DO NOT label validation sample yet. Fix prompt first based on calibration v3 results.

**Workflow:**
1. Fix prompt (Option B or A)
2. Re-label calibration v3 sample with fixed prompt
3. If metrics improve (false positive rate < 20%), THEN label validation sample
4. If validation passes, proceed to batch labeling

---

## Cost Analysis

**Calibration v3 cost:** 11 articles × $0.001 = $0.011

**Total calibration cost so far:**
- v1: 11 articles × $0.001 = $0.011
- v2: 9 articles × $0.001 = $0.009
- v3: 11 articles × $0.001 = $0.011
- **Total:** $0.031

**Still far below batch labeling cost:** $8 for full dataset

**Conclusion:** Keep iterating on calibration. Each iteration costs < $0.02, while a failed batch labeling costs $8-16.

---

## Appendix: All Articles

### Off-Topic Articles (Expected < 5.0)

1. ❌ **community_social_dev_to_c90592985034** - 6.6
   - Title: API Gateway Design with Rust and Go
   - Issue: Professional knowledge sharing

2. ❌ **community_social_reddit_singularity_3ff5694cf73d** - 6.3
   - Title: Economic survival pressure vs capability scaling
   - Issue: Speculation ("could lead to")

3. ❌ **community_social_programming_reddit_563947640886** - 5.1
   - Title: Learning programming
   - Issue: Professional knowledge sharing

4. ❌ **ai_engadget_601e3ecfd2e1** - 6.0
   - Title: Mint alternatives budgeting app
   - Issue: Productivity advice

5. ❌ **investor_signals_valuewalk_5efecfd84366** - 5.0
   - Title: Nexus gaming company ($1B)
   - Issue: Business success

6. ✅ **aerospace_defense_space_news_5d6a44d90d41** - 4.5
   - Title: Astranis satellite
   - Correctly rejected (military/business)

7. ❌ **industry_intelligence_fast_company_0102c17e27a2** - 6.4
   - Title: SNAP payments slashed
   - Issue: Doom-framed

8. ❌ **industry_intelligence_fast_company_e8c9ab8cfacd** - 6.6
   - Title: Productivity becomes a trap
   - Issue: Professional knowledge (business advice)

### On-Topic Articles (Expected >= 5.0)

1. ✅ **science_phys_org_7f3501f04b86** - 7.5
   - Title: Salmon comeback
   - Correctly recognized (environmental restoration)

2. ✅ **positive_news_the_better_india_87b647d277be** - 8.5
   - Title: Water Warriors
   - Correctly recognized (community solutions)

### Edge Cases

1. **industry_intelligence_fast_company_d08ef0625054** - 5.0
   - Title: Boeing trial
   - Justice in motion (families pursuing accountability)
   - Borderline score seems reasonable
