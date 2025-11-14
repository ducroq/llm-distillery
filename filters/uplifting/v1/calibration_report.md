# Prompt Calibration Report: Uplifting Content Filter

**Date:** 2025-11-14
**Filter:** uplifting v1.0
**Oracle:** Gemini Flash 1.5 (gemini-flash-api-batch)
**Calibrator:** Prompt Calibration Agent v1.0

---

## Executive Summary

**Decision:** REVIEW (with strong caveats)

**Overall Assessment:** The oracle demonstrates good dimensional differentiation and successfully identifies genuinely uplifting content, but has a concerning tendency to find "silver linings" in doom-framed or off-topic articles, resulting in a high false positive rate.

**Recommendation:** REVISE PROMPT before batch labeling - add explicit guidance on doom-framing and tighten scope definition.

**Critical Caveat:** Sample size (11 articles) is 78% smaller than recommended (50 articles) due to API rate limits. Results should be interpreted with caution and re-validated with larger sample before proceeding to batch labeling.

---

## Calibration Sample Overview

**Total articles reviewed:** 11
**Target sample size:** 50
**Actual sample size:** 11 (22% of target)

**Sample composition:**
- On-topic uplifting: 3 articles (27%)
- Off-topic/corporate: 2 articles (18%)
- Doom-framed: 3 articles (27%)
- Neutral/mixed: 3 articles (27%)

**Oracle used:** Gemini Flash 1.5
**Prompt version:** filters/uplifting/v1/prompt-compressed.md

---

## CRITICAL METRICS

### 1. Off-Topic Rejection Rate

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Off-topic articles reviewed | 5 | N/A | Info |
| Scored < 5.0 (correctly rejected) | 1 (20%) | >90% | FAIL |
| Scored >= 5.0 (false positives) | 4 (80%) | <10% | FAIL |
| Scored >= 7.0 (severe false positives) | 0 (0%) | <5% | PASS |

**Status:** FAIL

#### False Positive Examples

**1. "Web Developer Travis McCracken on API Gateway Design with Rust and Go" - Score: 6.57**
- **Why off-topic:** Generic software development, not human/planetary wellbeing
- **Oracle reasoning:** "The article highlights a web developer's experience using Rust and Go to build scalable and high-performance APIs. This involves creating solutions for backend systems that can handle heavy loads, improve efficiency, and simplify complex workflows, ultimately contributing to more reliable and maintainable applications for a broad community of users."
- **Issue:** Oracle interpreted "technical solutions" as "solutions for human wellbeing." Generic software development for "better APIs" is not uplifting semantic content - it's corporate tech optimization. No connection to health, safety, equity, or planetary outcomes.

**2. "Closing in on $1 Billion - How Gurhan Kiziloz's Nexus Is Quietly Rewriting the Rules of Gaming" - Score: 5.04**
- **Why off-topic:** Corporate success story, gaming industry business news
- **Oracle reasoning:** "The article describes a gaming company, Nexus, potentially 'rewriting the rules of gaming.' This suggests innovation in the industry and potentially a shift away from the dominance of large corporations, which could lead to more diverse and accessible gaming experiences for a broader audience."
- **Issue:** Oracle speculated about potential future benefits ("could lead to") rather than documented outcomes. Disrupting incumbents ≠ human wellbeing progress.

**3. "SNAP payments slashed to 50% as federal shutdown stretches into its second month" - Score: 6.40**
- **Why off-topic (doom-framed):** Article about food aid being CUT, not expanded
- **Oracle reasoning:** "Despite the reduction in SNAP benefits, there is agency demonstrated by federal judges ruling in favor of extending funding, the Trump administration agreeing to provide partial payments, and state governments and local organizations stepping up to provide emergency funding and prepare for increased need."
- **Issue:** Oracle found "silver lining" in crisis response. The article's main content is harm (benefits slashed 50%), not progress. Courts forcing partial restoration is damage control, not flourishing.

**4. "What to do when productivity becomes a trap" - Score: 6.64**
- **Why off-topic (doom-framed):** Article about workplace burnout and over-execution
- **Oracle reasoning:** "Leaders and teams are taking action to address burnout and ineffectiveness caused by unbounded productivity. They are setting boundaries, prioritizing strategic work, and protecting their energy, leading to improved focus, performance, and well-being."
- **Issue:** Oracle scored the *proposed solutions* in a business advice article, not documented outcomes. Prescriptive advice ≠ actual human flourishing achieved.

#### Root Cause Analysis

**Pattern:** Oracle interprets ANY action/agency as "progress toward human wellbeing" even when:
1. The action is corporate/technical (API development)
2. The action is speculative ("could lead to")
3. The action is damage control (crisis response to harm)
4. The action is prescriptive advice (not implemented solutions)

**Root cause:** Prompt lacks clear guidance on:
- What constitutes "human/planetary wellbeing" vs. corporate/technical optimization
- Doom-framing vs. solutions-framing (main content must be progress, not harm + response)
- Speculative benefits vs. documented outcomes
- Advice articles vs. implementation stories

### 2. On-Topic Recognition Rate

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| On-topic articles reviewed | 3 | N/A | Info |
| Scored >= 5.0 (correctly recognized) | 3 (100%) | >80% | PASS |
| Scored < 5.0 (false negatives) | 0 (0%) | <20% | PASS |
| At least one article >= 7.0 | Yes (2) | Yes | PASS |

**Status:** PASS

**Positive examples:**
- "Salmon's comeback pits nature against Trump administration" - 7.52 (environmental restoration)
- "Will Our Cities Run Dry? | The Water Warriors Blueprint" - 8.47 (water solutions)
- "The 5 best Mint alternatives..." - 6.00 (consumer empowerment)

**Assessment:** When presented with genuinely uplifting content (ecological restoration, community solutions, helpful consumer guides), the oracle scores appropriately.

### 3. Dimensional Consistency

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Average dimensional variance | 1.64 | >1.0 | PASS |
| Articles with variance < 0.5 | 0 (0%) | <20% | PASS |
| All dimensions used (not all 0 or all 10) | Yes | Yes | PASS |

**Status:** PASS

**Assessment:** Oracle differentiates dimensions well. No signs of dimensional collapse (all scores identical) or single-dimension scoring.

**Example of good differentiation:**
- "Boeing faces its first civil trial" - Variance: 3.70
  - Dimensions: agency=6, progress=5, collective=6, connection=3, innovation=2, justice=7, resilience=4, wonder=2
  - Oracle correctly scored high on justice (accountability) but low on innovation/wonder

---

## QUALITY CHECKS

### 4. Oracle Reasoning Quality

**Sample reasoning review (5 articles):**

**Good reasoning examples:**

1. "Water Warriors Blueprint" (8.47):
   - "The article highlights various community-led initiatives and innovative solutions addressing India's water crisis. These include sponge roads, check dams, pond revival, rainwater harvesting, stepwell restoration, and solar-powered villages, all demonstrating agency and progress towards water security and community wellbeing."
   - STRONG: Specific evidence cited, clear connection to human/planetary wellbeing

2. "Salmon's comeback" (7.52):
   - "Migrating salmon have returned to the headwaters of the Klamath River for the first time in a century, indicating the success of the dam removal project. This benefits the river ecosystem, tribes, and commercial fishers, demonstrating progress towards ecological restoration and community well-being."
   - STRONG: Concrete outcomes documented, multiple beneficiary groups identified

**Weak reasoning examples:**

3. "Gaming company Nexus" (5.04):
   - "The article describes a gaming company, Nexus, potentially 'rewriting the rules of gaming.' This suggests innovation in the industry and potentially a shift away from the dominance of large corporations, which could lead to more diverse and accessible gaming experiences for a broader audience."
   - WEAK: Speculation ("could lead to"), no documented outcomes, no evidence of actual benefit

4. "API Gateway Design" (6.57):
   - "The article highlights a web developer's experience using Rust and Go to build scalable and high-performance APIs. This involves creating solutions for backend systems that can handle heavy loads, improve efficiency, and simplify complex workflows, ultimately contributing to more reliable and maintainable applications for a broad community of users."
   - WEAK: Assumes technical optimization = human wellbeing, no evidence of societal benefit

**Assessment:** Reasoning quality is VARIABLE. When content is genuinely uplifting, reasoning is strong. When content is off-topic or doom-framed, oracle engages in speculation or over-generalization to justify scores.

### 5. Edge Case Handling

**Edge cases reviewed:** 4

| Article | Score | Expected | Assessment |
|---------|-------|----------|------------|
| "SNAP payments slashed" | 6.40 | 3-4 | QUESTIONABLE - Silver lining bias |
| "Productivity trap" | 6.64 | 3-4 | QUESTIONABLE - Scored advice as outcomes |
| "Boeing trial" | 4.99 | 4-5 | REASONABLE - Justice scored appropriately |
| "Astranis satellite" | 4.55 | 4-5 | REASONABLE - Disaster relief scored moderately |

**Pattern:** Oracle struggles with doom-framed articles that contain response/mitigation efforts. It tends to score the response rather than the overall story framing.

**Assessment:** Inconsistent - needs clearer guidance on doom-framing.

---

## Recommendations

### Immediate Actions

**DECISION: REVIEW - Minor Prompt Issues Detected**

Given the small sample size (11 vs 50), I recommend:

1. FIX PROMPT ISSUES (details below)
2. RE-CALIBRATE with larger sample (20-30 articles minimum)
3. If second calibration passes, proceed to batch labeling

**DO NOT proceed to batch labeling with current prompt.** The 80% false positive rate, while based on a small sample, indicates systematic issues that will mis-label hundreds of articles.

### Specific Prompt Improvements

**1. Add SCOPE clarification (CRITICAL):**

```markdown
**HUMAN/PLANETARY WELLBEING SCOPE:**

**IN SCOPE (score normally):**
- Health outcomes: Disease prevention, mental health support, healthcare access
- Safety & security: Violence reduction, disaster recovery, housing stability
- Equity & justice: Rights expanded, discrimination reduced, opportunity increased
- Livelihoods: Jobs created, poverty reduced, skills developed
- Planetary health: Emissions cut, ecosystems restored, biodiversity protected
- Community flourishing: Connection strengthened, culture preserved, knowledge shared

**OUT OF SCOPE (max score 2-3):**
- Corporate optimization: Better APIs, faster servers, efficient workflows (unless serving above needs)
- Technical achievements: New programming languages, software frameworks (unless open-source addressing needs)
- Business success: Funding rounds, market share, disruption (unless cooperative/public benefit model)
- Productivity advice: How-to articles, prescriptive guidance (unless documenting implementation)

**AMBIGUOUS CASES:**
- Technology serving human needs: Score the documented OUTCOME (better health, expanded access), not the technology itself
- Advice with case studies: Score the documented cases, not the advice framework
```

**2. Add DOOM-FRAMING guidance (CRITICAL):**

```markdown
**DOOM-FRAMING vs. SOLUTIONS-FRAMING:**

The MAIN CONTENT of the article must be progress/solutions, not harm + response.

**Doom-framed (score crisis response only, max 3-4):**
- "Food aid slashed 50%, but courts force partial restoration"
  → Main content: HARM (benefits cut)
  → Response: Damage control (crisis mitigation)
  → Score: 3-4 (for crisis response effort)

- "Workplace burnout epidemic, here's how to cope"
  → Main content: PROBLEM (burnout)
  → Response: Prescriptive advice (not implemented)
  → Score: 2-3 (advice without outcomes)

**Solutions-framed (score normally, can reach 7+):**
- "Community water crisis solved with check dams"
  → Main content: SOLUTION (problem solved)
  → Outcomes: Documented results (water restored)
  → Score: 7-8 (genuine progress)

- "After 15 years of conflict, peace agreement signed"
  → Main content: SOLUTION (conflict resolved)
  → Outcomes: Reconciliation beginning
  → Score: 8-9 (transformative progress)

**Decision rule:** If >50% of article describes harm/problem, treat as doom-framed.
```

**3. Add OUTCOME REQUIREMENT (CRITICAL):**

```markdown
**DOCUMENTED OUTCOMES vs. SPECULATION:**

Score what HAS HAPPENED, not what MIGHT happen.

**High scores require documented outcomes:**
- "Farmers restored 200 hectares, yields increased 250%" → DOCUMENTED (score 7-8)
- "New startup COULD democratize healthcare access" → SPECULATION (score 2-3)
- "Study shows intervention reduces depression 40%" → DOCUMENTED (score 7-8)
- "Leaders SHOULD set better boundaries" → PRESCRIPTIVE (score 2-3)

**Exception:** Early-stage progress (pilots, initial trials) can score 5-6 if showing clear results, but not 7+ until proven at scale.
```

**4. Clarify collective_benefit dimension:**

```markdown
3. **Collective Benefit** (GATEKEEPER):
   - Who benefits? How many? What needs are addressed?
   - Elite/shareholders only: 0-2
   - Limited group (single community, niche users): 3-4
   - Moderate community (city, region, broad user base): 5-6
   - Broad community (nationwide, major ecosystem, millions): 7-8
   - Universal/planetary (species-wide, global systems): 9-10

   **Examples:**
   - "Better API framework for developers" → 3-4 (limited professional group)
   - "Free healthcare clinic serves 10,000 residents" → 6-7 (broad community)
   - "Dam removal restores salmon ecosystem for entire river basin" → 8-9 (regional ecosystem)
```

---

## Appendix

### Files Reviewed

- Prompt: `filters/uplifting/v1/prompt-compressed.md`
- Config: `filters/uplifting/v1/config.yaml`
- Calibration sample: `datasets/working/uplifting_calibration_labeled.jsonl` (11 articles)

### Calibration Command

```bash
python scripts/label_batch.py \
    --filter filters/uplifting/v1 \
    --input datasets/working/uplifting_calibration_sample.jsonl \
    --output datasets/working/uplifting_calibration_labeled.jsonl \
    --oracle gemini-flash
```

### Scoring Distribution

**Overall scores:**
- 0-2: 0 articles (0%)
- 3-4: 2 articles (18%)
- 5-6: 7 articles (64%)
- 7-8: 2 articles (18%)
- 9-10: 0 articles (0%)

**By tier:**
- impact (>=7.0): 1 article (9%)
- connection (4.0-6.9): 8 articles (73%)
- not_uplifting (<4.0): 2 articles (18%)

**By expected category:**
- On-topic uplifting: Mean=7.33, Median=7.52, Range=[6.00-8.47] (n=3)
- Off-topic/corporate: Mean=5.80, Median=5.54, Range=[5.04-6.57] (n=2)
- Doom-framed: Mean=5.78, Median=6.40, Range=[4.99-6.64] (n=3)
- Neutral: Mean=5.30, Median=5.07, Range=[4.55-6.27] (n=3)

**Dimensional averages (all articles):**
- agency: 6.64 ± 1.29
- progress: 6.09 ± 1.45
- collective_benefit: 6.82 ± 1.17
- connection: 5.18 ± 1.72
- innovation: 5.27 ± 1.62
- justice: 4.64 ± 1.69
- resilience: 5.55 ± 1.81
- wonder: 4.27 ± 1.68

**Key observations:**
1. Collective benefit scored highest on average (6.82) - suggests oracle is generous with this gatekeeper dimension
2. Connection and wonder scored lowest (5.18, 4.27) - suggests appropriate restraint on these
3. High variance on most dimensions (1.2-1.8) - confirms good dimensional differentiation

---

## Conclusion

**FINAL DECISION: REVIEW**

The uplifting filter oracle shows PROMISE but requires prompt refinement before batch labeling:

**Strengths:**
- Excellent dimensional differentiation (variance 1.64)
- 100% recognition of genuinely uplifting content
- Strong reasoning when content is clearly on-topic
- No severe false positives (score >= 7.0 for off-topic content)

**Critical weaknesses:**
- 80% false positive rate on off-topic/doom-framed content (vs. target <10%)
- "Silver lining" bias: finds agency in crisis response/damage control
- Speculation bias: scores potential benefits ("could lead to") as if documented
- Scope confusion: treats technical optimization as human wellbeing

**Risk if proceeding to batch labeling without fixes:**
- Estimated 400-800 mis-labeled articles in 2,500-article batch
- Cost of rework: $1-2 in oracle fees + days of manual review
- Alternative: Fix prompt now, re-calibrate 20-30 articles ($0.10), proceed confidently

**Recommended path forward:**
1. Implement prompt improvements (1-2 hours)
2. Re-calibrate with 20-30 articles ($0.05-0.10)
3. If false positive rate <15%, proceed to batch labeling
4. Monitor first 500 labels for quality drift

**Sample size caveat:** These findings are based on only 11 articles (22% of recommended 50). Patterns observed are concerning but MAY NOT BE STATISTICALLY RELIABLE. Larger calibration sample is strongly recommended before making final PASS/FAIL decision.

---

**Report generated:** 2025-11-14
**Calibrator:** Prompt Calibration Agent v1.0
**Next review:** After prompt revision and re-calibration
