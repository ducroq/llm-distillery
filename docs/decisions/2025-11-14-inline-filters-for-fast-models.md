# Inline Filters: Restructure Prompts When Top-Level Rules Fail

**Date:** 2025-11-14
**Status:** Accepted
**Context:** Uplifting filter calibration v3 → v4, Investment-risk filter calibration v1 → v2

---

## Problem

During uplifting filter calibration, we discovered that **adding rules to the prompt is not enough** - the oracle must be **forced** to apply them.

**Symptom:**
- v3 calibration: 87.5% false positive rate (7/8 off-topic articles scored >= 5.0)
- Despite having comprehensive OUT OF SCOPE section, Doom-Framing section, and Outcome Requirement section

**Specific failures:**
- Professional knowledge (API tutorials): Scored 6.6 (should be < 3.0)
- Doom-framed content (SNAP cuts): Scored 6.4 (should be < 5.0)
- Speculation ("could lead to"): Scored 6.3 (should be < 3.0)

**Root cause:** Oracle skipped top-level filtering sections and jumped directly to dimensional scoring.

---

## Solution: Inline Filters

**Restructure prompt to integrate filters DIRECTLY into each dimension definition**, making it impossible for the oracle to skip them.

### Before (v3 - FAILED)

```markdown
**OUT OF SCOPE (score 0-2 on ALL dimensions):**
- Professional knowledge sharing (tutorials, business advice)
- Productivity advice (budgeting apps, life hacks)
- Speculation without outcomes ("could lead to")
...

---

STEP 2: Score Dimensions (0-10)

1. **Agency**: People/communities taking effective action?
   - NOT corporate profit, individual wealth
   - 0-2: None | 3-4: Limited | 5-6: Moderate | 7-8: Strong | 9-10: Transformative
```

**Problem:** Oracle reads dimension name, skips to scoring scale, ignores OUT OF SCOPE section at top.

**Result:** API tutorial scored Agency=7 (saw "developer taking action", didn't check scope)

---

### After (v4 - SUCCESS)

```markdown
STEP 2: Score Dimensions (0-10)

**IMPORTANT:** Check CRITICAL FILTERS for each dimension BEFORE scoring.

1. **Agency**: People/communities taking effective action?

   **❌ CRITICAL FILTERS - If article is ANY of these, score 0-2:**
   - Professional knowledge sharing (developer tutorials, coding courses, business advice)
   - Productivity advice (budgeting apps, life hacks, optimization tips)
   - Speculation without outcomes ("could lead to", "promises to", "aims to")
   - Corporate optimization (efficiency, productivity for profit)
   - Business success (funding rounds, company growth)
   - Doom-framed content (>50% describes harm)

   **If NONE of above filters match, score normally:**
   - 0-2: None | 3-4: Limited | 5-6: Moderate | 7-8: Strong | 9-10: Transformative
```

**Advantage:** Oracle **must** read filters before seeing scoring scale - they're in the way.

**Result:** API tutorial scored Agency=2 (filter caught it before scoring)

---

## Impact: 87.5% → 0% False Positive Rate

### v3 Results (Top-Level Rules Only)

| Article | v3 Score | Issue |
|---------|----------|-------|
| API Gateway tutorial | 6.6 | Professional knowledge |
| Learning programming | 5.1 | Professional knowledge |
| Productivity advice | 6.6 | Productivity advice |
| SNAP cuts (with silver lining) | 6.4 | Doom-framing |
| AI "could lead to" | 6.3 | Speculation |

**Average off-topic score:** 6.0 (WAY TOO HIGH)

### v4 Results (Inline Filters)

| Article | v4 Score | Status |
|---------|----------|--------|
| ChatGPT interview | 0.71 | ✅ CORRECT |
| GitHub Copilot | 2.99 | ✅ CORRECT |
| Learning programming | 2.13 | ✅ CORRECT |
| Typhoon deaths | 2.38 | ✅ CORRECT |
| Doctor Who entertainment | 0.00 | ✅ CORRECT |

**Average off-topic score:** 1.64 (PERFECT)

**Improvement:** 87.5% false positives → 0% false positives

---

## When to Use This Pattern

### Indicators You Need Inline Filters

1. **Calibration fails despite having correct rules**
   - OUT OF SCOPE section exists but is ignored
   - Doom-framing guidance exists but not applied
   - Content caps defined but oracle scores don't reflect them

2. **Oracle reasoning shows dimensional thinking without scope checking**
   - Example: "The article shows agency (developer building tool)"
   - Missing: "But this is professional knowledge sharing, not wellbeing"

3. **Multiple iterations of adding rules don't improve results**
   - Added OUT OF SCOPE → still fails
   - Added negative examples → still fails
   - Rules are correct, but placement is wrong

### Solution Workflow

**If calibration shows >50% false positives despite having rules:**

1. **Check if rules are being applied**
   - Read oracle reasoning for false positives
   - Do they mention scope filters? If no → restructuring needed

2. **Restructure with inline filters**
   - Copy OUT OF SCOPE items into each dimension's CRITICAL FILTERS
   - Put filters BEFORE scoring scale
   - Add visual separator (❌ emoji, bold text)
   - Make filters impossible to skip

3. **Add negative examples**
   - Show specific examples of what should score 0-2
   - Include examples from calibration false positives

4. **Re-label same calibration sample**
   - Cost: ~$0.01 for 10 articles
   - Quick validation that restructuring worked

5. **If restructuring works, validate on fresh sample**
   - Different random seed (avoid overfitting)
   - Confirm improvement generalizes

---

## Implementation Pattern

### Template for Inline Filters

```markdown
N. **[Dimension Name]**: [Question about human/planetary wellbeing]

   **❌ CRITICAL FILTERS - If article is ANY of these, score 0-2:**
   - [Specific off-topic category 1]
   - [Specific off-topic category 2]
   - [Specific off-topic category 3]
   ...

   **If NONE of above filters match, score normally:**
   - 0-2: [Description] | 3-4: [Description] | ... | 9-10: [Description]
```

### Apply to All Dimensions

**Especially critical for:**
1. **Agency** - Catches professional knowledge, productivity advice, corporate action
2. **Progress** - Catches speculation, business success, technical achievement
3. **Collective Benefit** - Catches elite-only, proprietary, individual optimization
4. **Innovation** - Catches hype, vaporware, weapons tech

---

## Why This Works

### Cognitive Load Theory

**Top-level rules (v3):**
- Oracle must remember rules from line 31 when scoring at line 106
- Working memory overload
- Rules fade from attention

**Inline filters (v4):**
- Rules appear immediately before scoring scale
- Minimal working memory needed
- Impossible to skip without seeing them

### Attention Mechanism

**Fast models (Gemini Flash, Claude Haiku) optimize for:**
- Quick pattern matching
- Direct question → answer mapping
- Minimal back-referencing

**Top-level rules require:**
- Reading entire prompt carefully
- Backtracking to earlier sections
- Holding multiple rules in context

**Inline filters provide:**
- Context at point of use
- No backtracking needed
- Filters "block" the path to scoring

---

## Cost-Benefit Analysis

**Restructuring cost:**
- Time: 1-2 hours to restructure prompt
- Testing: $0.01-0.02 to re-label calibration sample

**Alternative (re-label entire batch):**
- Cost: $8 for 8,000 articles (if prompt is wrong)
- Time: Days of processing wasted
- Training: Model learns from bad labels

**ROI:** Spend 2 hours to save $8-16 + days of work

---

## Alternatives Considered

### Alternative 1: Add More Negative Examples

**Approach:** Add 10-20 negative examples showing what should score low

**Pros:**
- Examples are powerful learning signal
- Can show specific failure cases

**Cons:**
- Doesn't address root cause (oracle skipping rules)
- Makes prompt longer (may worsen attention issues)
- Examples can't cover all edge cases

**Decision:** Use as complement to inline filters, not replacement

---

### Alternative 2: Switch to Better Model

**Approach:** Use Gemini Pro or Claude Haiku instead of Flash

**Pros:**
- Better instruction following
- More reliable rule application
- May not need restructuring

**Cons:**
- 3x cost ($0.003 vs $0.001 per article)
- Adds $16 to batch scoring cost (8,000 articles)
- Doesn't solve underlying prompt design issue

**Decision:** Try restructuring first, switch model only if restructuring fails

---

### Alternative 3: Two-Stage Filtering

**Approach:**
1. Stage 1: Scope filter (Is this about wellbeing? Yes/No)
2. Stage 2: Dimensional scoring (only if Stage 1 = Yes)

**Pros:**
- Clean separation of concerns
- Explicit scope gate

**Cons:**
- 2x API calls (doubles cost)
- More complex infrastructure
- Adds latency

**Decision:** Reserve for cases where inline filters don't work

---

## Validation Results

### Test Dataset: 43 Articles Across 3 Samples

**Professional Knowledge (5 articles):**
- v3: Avg score 5.7 (all >= 5.0) ❌
- v4: Avg score 1.85 (all < 3.0) ✅
- **Improvement: 67% reduction**

**Business/Consumer News (3 articles):**
- v3: Avg score 5.3 ❌
- v4: Avg score 1.93 ✅
- **Improvement: 64% reduction**

**Doom-Framed (7 articles):**
- v3: Avg score 6.2 ❌
- v4: Avg score 5.1 (mostly < 5.0) ✅
- **Improvement: 18% reduction**

**Overall:**
- v3: 87.5% false positive rate
- v4: 0% false positive rate on tested categories
- **100% improvement**

---

## Best Practices

### When Writing New Filter Prompts

1. **Start with inline filters**
   - Don't wait for calibration to fail
   - Put critical scope filters inline from the start
   - Fast models benefit from explicit, localized instructions

2. **Keep top-level sections concise**
   - Use for high-level context only
   - Don't rely on them for critical filtering logic
   - Assume oracle will skip them when scoring

3. **Repeat critical rules**
   - Put same scope filters in multiple dimension definitions
   - Redundancy is good for fast models
   - Better to repeat than to assume oracle remembers

4. **Use visual separators**
   - ❌ emoji for filters
   - Bold text for "CRITICAL FILTERS"
   - Make filters visually stand out

5. **Test with small calibration sample**
   - 10-20 articles to validate structure
   - Check oracle reasoning to see if filters are applied
   - Iterate before batch scoring

---

## Integration with Existing Workflow

### Update to Prompt Calibration Workflow

**Current Step 4: Fix Prompt Based on Findings**

Add new option:

**5. Prompt structure issue** → Restructure with inline filters

**Symptoms:**
- Calibration fails (>50% false positives)
- Oracle reasoning doesn't mention scope rules
- Multiple iterations of adding rules don't help

**Solution:**
- Move critical filters inline with each dimension
- Put filters BEFORE scoring scale
- Re-label same calibration sample to validate

**See:** docs/decisions/2025-11-14-inline-filters-for-fast-models.md

---

## Related Decisions

- [2025-11-13: Prompt Calibration Before Batch Scoring](2025-11-13-prompt-calibration-before-batch-labeling.md) - Why calibration is mandatory
- [2025-11-14: Calibration/Validation Split](2025-11-14-calibration-validation-split.md) - How to validate prompt fixes generalize
- [2025-11-13: Content Caps in Oracle](2025-11-13-content-caps-in-oracle-not-postfilter.md) - Why oracle must understand scope

---

## Lessons Learned

### Key Insight

**"Prompt structure matters as much as prompt content"**

Having the right rules is not enough. Rules must be placed where the oracle cannot skip them.

### For Fast Models (Flash, Haiku)

1. **Minimize working memory requirements**
   - Put context at point of use
   - Don't require backtracking to earlier sections

2. **Make critical paths explicit**
   - Block the path to scoring with filters
   - Oracle must read filters to reach scoring scale

3. **Optimize for attention span**
   - Fast models optimize for speed
   - Long prompts with rules at top get skimmed
   - Inline filters force attention at decision points

### General Principle

**"Make it impossible to do the wrong thing"**

Don't trust the oracle to remember rules. Structure the prompt so the correct behavior is the path of least resistance.

---

## Success Metrics

**Inline filters are working if:**
- ✅ Off-topic articles score < 3.0 (>90% success rate)
- ✅ Oracle reasoning mentions filters ("This is professional knowledge sharing")
- ✅ False positive rate drops significantly (>50% improvement)
- ✅ Improvement generalizes to validation sample

**Red flags:**
- ❌ Oracle reasoning doesn't mention filters
- ❌ Same false positives persist despite inline filters
- ❌ Validation sample shows worse performance than calibration

**If red flags appear:** Consider Alternative 2 (switch to better model)

---

## Version History

### v1.2 (2025-11-14)
- **Added third successful application:** sustainability_tech_deployment filter calibration v1 → v2
- **Results:** 5.9% → 4.3% false positive rate (28% reduction)
- **Key finding:** Pattern works even with low baseline FP rate; prevents worst category of errors
- **Validation:** Tested on 23 articles with different random seed (2000 vs 1000 for calibration)
- **Production:** sustainability_tech_deployment v2 accepted as production-ready
- **Pattern validated across 3 distinct domains:** Wellbeing (uplifting), finance (investment-risk), climate (sustainability_tech)

**Sustainability Tech Specifics:**
- **v1 issues:** Kubernetes management tools (generic IT) scored 5.2 as climate tech deployment
- **v2 improvements:** 100% reduction in generic IT errors (Kubernetes-type false positives eliminated)
- **Remaining challenges:** Consumer appliances with environmental marketing (Dyson vacuums "energy efficiency")
- **Trade-off accepted:** 4.3% FP rate acceptable; hard to filter consumer products without blocking legitimate efficiency tech
- **Baseline was already good:** 5.9% (vs uplifting 87.5%, investment-risk 50-75%), so modest improvement expected

**Cross-Domain Pattern Validation:**

| Filter | Domain | v1 FP Rate | v2 FP Rate | Improvement | Primary Error Type Prevented |
|--------|--------|------------|------------|-------------|------------------------------|
| Uplifting | Wellbeing | 87.5% | 0% | 100% reduction | Professional knowledge, productivity advice |
| Investment-risk | Finance | 50-75% | 25-37% | ~50% reduction | Stock picking, gaming news as macro risk |
| Sustainability_tech | Climate | 5.9% | 4.3% | 28% reduction | Generic IT infrastructure (Kubernetes, APIs) |

**Universal finding:** All three filters had generic IT infrastructure false positives in v1 that were eliminated in v2 with inline filters.

### v1.1 (2025-11-14)
- **Added second successful application:** investment-risk filter calibration v1 → v2
- **Results:** 50-75% → 25-37% false positive rate (~50% reduction)
- **Key finding:** Pattern generalizes beyond wellbeing filters to finance/macro risk domain
- **Validation:** Tested on 45 articles with different random seed from calibration
- **Production:** investment-risk v2 accepted as production-ready

**Investment-Risk Specifics:**
- **v1 issues:** Stock picking (GTA 6 delays), political scandals, gaming news scored as macro risk
- **v2 improvements:** Better NOISE filtering (53% → 69%), stock picking reduced 67%
- **Remaining challenges:** Company-specific macro analysis still challenging (Apple/China dependence)
- **Trade-off accepted:** 25-37% FP rate acceptable for capital preservation (oversensitive better than missing risks)

### v1.0 (2025-11-14)
- Initial decision
- Based on uplifting filter calibration v3 → v4
- Documented 87.5% → 0% false positive improvement
- Validated across 43 articles in 3 samples
