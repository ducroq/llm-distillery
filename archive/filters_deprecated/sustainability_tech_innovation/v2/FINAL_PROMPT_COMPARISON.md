# Final Prompt Structure Comparison (After Harmonization)

Comparing sustainability_tech_innovation v1.1 (HARMONIZED) with uplifting v4 and investment-risk v2

**Date:** 2025-11-17

---

## Side-by-Side Structure

| Section | Uplifting v4 | Investment-Risk v2 | Tech Innovation v1.1 (HARMONIZED) |
|---------|--------------|--------------------|------------------------------------|
| **Header** | ✅ Purpose, Version, Target, Focus, Oracle Output | ✅ Purpose, Version, Target, Focus, Philosophy | ✅ Purpose, Version, Focus, **Philosophy**, Oracle Output |
| **Signal/Tier Defs** | ❌ N/A | ✅ ## SIGNAL TIERS | ❌ N/A (post-processing) |
| **Prompt Start** | ## PROMPT TEMPLATE | ## PROMPT TEMPLATE | ## PROMPT TEMPLATE |
| **Scope Section** | ✅ IN SCOPE / OUT OF SCOPE | ❌ Inline with dimensions | ✅ ## CRITICAL: What is "Tech That Works"? |
| **Special Rules** | ✅ Doom-Framing, Outcome Requirement | ❌ N/A | ✅ ## ⚠️ CRITICAL: MANDATORY GATEKEEPER RULES ⚠️ |
| **ARTICLE Placement** | ✅ AFTER scope/framing | ✅ START of prompt | ✅ AFTER gatekeeper rules (harmonized!) |
| **Pre-classification** | ✅ STEP 1: Pre-classification | ❌ N/A | ❌ N/A (covered in gatekeepers) |
| **Dimensions** | ✅ STEP 2: Score Dimensions | ✅ Score Dimensions | ✅ ## Dimensions |
| **Inline Filters** | ✅ ❌ CRITICAL FILTERS | ✅ ❌ CRITICAL FILTERS | ✅ ❌ CRITICAL FILTERS |
| **Tier Classification** | ❌ Post-processing | ✅ Classify Signal Tier | ❌ Post-processing |
| **Calibration** | ❌ Implicit in scales | ❌ ## SCORING FORMULA (separate) | ✅ ## Scoring Calibration (inside prompt) |
| **Philosophy** | ❌ N/A | ❌ N/A | ✅ ## Scoring Philosophy |
| **Examples** | ✅ Inline with concepts | ❌ N/A | ✅ ## Examples (4 scored) |
| **Metadata** | ❌ N/A | ✅ In prompt template | ✅ In output format |
| **Output Format** | ✅ JSON | ✅ JSON | ✅ ## Output Format (JSON) |
| **Post-Processing** | ✅ ## POST-PROCESSING REFERENCE | ✅ ## SCORING FORMULA | ❌ N/A (in config) |
| **CHANGELOG** | ❌ N/A | ✅ ## CHANGELOG | ✅ ## CHANGELOG |

---

## Header Comparison

### Uplifting v4
```markdown
# Uplifting Content Filter

**Purpose**: Rate content for uplifting semantic value based on genuine human and planetary wellbeing.
**Version**: 1.0-compressed
**Target**: Gemini Flash 1.5 / Claude Haiku / Fast models
**Focus**: MEANING not TONE
**Oracle Output**: Dimensional scores only (0-10 per dimension)
```

### Investment-Risk v2
```markdown
# Investment Risk: Capital Preservation Filter

**Purpose**: Identify investment risk signals for defense-first portfolio management
**Version**: 2.1-academic-filter
**Target**: Gemini Flash 1.5 / Claude Haiku / Fast models
**Focus**: RISK SIGNALS and CAPITAL PRESERVATION
**Philosophy**: "You can't predict crashes, but you can prepare for them."
```

### Tech Innovation v1.1 ✅ HARMONIZED
```markdown
# Sustainable Technology & Innovation Scoring

**Purpose**: Rate cool sustainable tech that WORKS - deployed tech, working pilots, validated breakthroughs.
**Version**: 1.1
**Focus**: Technology with REAL RESULTS, not just theory or promises.
**Philosophy**: "Pilots and research need real results, not just theory."  ← ADDED
**Oracle Output**: Dimensional scores only (0-10 per dimension)
```

✅ **HARMONIZED:** Now includes Philosophy line like investment-risk

---

## Scope/Rules Section Comparison

### Uplifting v4
```markdown
**IN SCOPE (score normally):**
- Health improvements
- Safety & security
- Equity & justice
[...]

**OUT OF SCOPE (score 0-2 on ALL dimensions):**
- Corporate optimization
- Technical achievement alone
- Professional knowledge sharing
[...]
```
**Placement:** Before ARTICLE

### Investment-Risk v2
```markdown
[NO EXPLICIT SCOPE SECTION]
Scope enforced via inline filters in each dimension
```
**Placement:** Inline with dimensions

### Tech Innovation v1.1 ✅ HARMONIZED
```markdown
## CRITICAL: What is "Tech That Works"?

**INCLUDE:**
- ✅ Deployed technology
- ✅ Working pilots with performance data
- ✅ Validated research with real-world results
[...]

**EXCLUDE:**
- ❌ Pure theory
- ❌ Simulations without validation
- ❌ Future announcements
[...]
```
**Placement:** Before gatekeeper rules, before ARTICLE ✅

---

## Gatekeeper Enforcement Comparison

### Uplifting v4
```markdown
STEP 1: Pre-classification

A) CORPORATE FINANCE: [...] → FLAG "corporate_finance" (max_score = 2)
B) BUSINESS NEWS: [...] → NOTE: collective_benefit must be ≥6
C) MILITARY/SECURITY: [...] → FLAG "military_security" (max_score = 4)
D) DOCUMENTATION OF HARM: [...]

[Later in dimensions]
3. **Collective Benefit** (GATEKEEPER: if <5, max overall = 3 unless wonder ≥7)
```
**Approach:** Pre-classification + dimension-level gatekeeper

### Investment-Risk v2
```markdown
Classify Signal Tier:

**🔴 RED FLAG**: Macro Risk ≥7 OR Credit Stress ≥7 OR Systemic Risk ≥8,
                 Evidence ≥5, Actionability ≥5
**🟡 YELLOW WARNING**: [...], Evidence ≥5, Actionability ≥4
**🟢 GREEN OPPORTUNITY**: [...], Evidence ≥6, Actionability ≥5
```
**Approach:** Tier-level gatekeepers (Evidence & Actionability thresholds)

### Tech Innovation v1.1 ✅ HARMONIZED
```markdown
## ⚠️ CRITICAL: MANDATORY GATEKEEPER RULES ⚠️

**BEFORE SCORING:** Determine if article describes REAL WORK with EVIDENCE

### What is REAL WORK?
- ✅ Deployed, ✅ Working pilot, ✅ Validated research

### What is NOT real work?
- ❌ Proposals, ❌ Future-only, ❌ Theory/simulations

### EXAMPLES - Proposals vs Pilots:
- ❌ "Xcel proposes 600 MW, delivery 2027" → deployment_maturity = 1-2
- ✅ "5 MW pilot, 6 months operation" → deployment_maturity = 4-5
[6 examples total]

### ENFORCEMENT:
**AFTER scoring all dimensions:**
1. IF deployment_maturity < 3.0: SET all scores = 1.0, overall = 1.0
2. IF proof_of_impact < 3.0: SET all scores = 1.0, overall = 1.0
```
**Approach:** Dedicated gatekeeper section with explicit enforcement + examples
**Placement:** Before ARTICLE ✅ (harmonized with uplifting's pre-classification approach)

---

## ARTICLE Placement Comparison

### Uplifting v4
```markdown
[Scope section]
[Doom-framing section]
[Outcome requirement]

ARTICLE:
Title: {title}
Text: {text}

STEP 1: Pre-classification
[...]
```
**Placement:** AFTER scope/framing, BEFORE pre-classification

### Investment-Risk v2
```markdown
## PROMPT TEMPLATE

```
Analyze this article [...]

ARTICLE:
Title: {title}
Source: {source}
Published: {published_date}
Text: {text}

Score Dimensions:
[...]
```
**Placement:** START of prompt template (immediate)

### Tech Innovation v1.1 ✅ HARMONIZED
```markdown
[Scope section: "What is Tech That Works"]
[Gatekeeper rules with examples]

ARTICLE:
Title: {title}
Text: {text}

## Dimensions
[...]
```
**Placement:** AFTER gatekeeper rules, BEFORE dimensions ✅
**Rationale:** Matches uplifting structure (oracle sees rules before article)

---

## Dimensions & Inline Filters Comparison

### All Three Filters: ✅ HARMONIZED

All three use identical inline filter format:

```markdown
1. **Dimension Name** (weight/role):

   **❌ CRITICAL FILTERS - If article is ANY of these, score 0-2:**
   - Filter 1
   - Filter 2
   - Filter 3

   **If NONE of above filters match, score normally:**
   - 0-2: Description | 3-4: Description | 5-6: Description | 7-8: Description | 9-10: Description
   - Evidence indicators
```

✅ **FULLY HARMONIZED** across all three filters

---

## Calibration & Examples Comparison

### Uplifting v4
- **Calibration:** Implicit in 0-10 scale descriptions
- **Examples:** Inline with each concept (doom-framing examples, outcome examples)

### Investment-Risk v2
- **Calibration:** ## SCORING FORMULA (separate section, post-processing)
- **Examples:** No dedicated examples section

### Tech Innovation v1.1
- **Calibration:** ## Scoring Calibration (inside prompt)
  ```markdown
  **Development Stage → Overall Score Mapping:**
  - mass_deployment → 8-10
  - commercial_proven → 6-8
  - validated_pilots → 5-7
  - working_pilots → 4-6
  - validated_research → 3-5
  - lab_only → 1-2
  - theory_only → 0-2
  ```

- **Examples:** ## Examples (dedicated section)
  ```markdown
  **High Score (9.1)**: "China Solar Deployment..."
  **Medium Score (5.2)**: "Geothermal Pilot..."
  **Low Score (3.8)**: "Battery Model Validated..."
  **Very Low Score (1.6)**: "Lab Results..."
  **OUT OF SCOPE (1.0)**: "AI Data Center Cooling..."
  ```

✅ **UNIQUE STRENGTH:** Most explicit calibration guidance of the three filters

---

## Output Format Comparison

### All Three Filters: ✅ HARMONIZED

All three use JSON output with:
- Dimensional scores (0-10) with reasoning
- Overall assessment
- Metadata fields (tier/stage/category, confidence)

```json
{
  "dimension_name": {"score": <0-10>, "reasoning": "..."},
  [...]
  "overall_assessment": "...",
  "metadata_field": "...",
  "confidence": "HIGH|MEDIUM|LOW"
}
```

✅ **FULLY HARMONIZED** across all three filters

---

## Summary: Harmonization Status

### ✅ NOW HARMONIZED (after changes)

1. **Header Structure** ✅
   - All include: Purpose, Version, Focus
   - Tech Innovation now has Philosophy line (matches investment-risk)

2. **ARTICLE Placement** ✅
   - Tech Innovation: Now AFTER gatekeeper rules (matches uplifting's "after scope" approach)
   - Oracle sees rules/scope before article in both uplifting and tech innovation

3. **Inline Filter Format** ✅
   - All three use identical `❌ CRITICAL FILTERS` structure
   - Consistent 0-10 scale descriptions

4. **JSON Output Format** ✅
   - All three use dimensional scores + metadata + confidence

5. **CHANGELOG** ✅
   - Tech Innovation and Investment-Risk both have CHANGELOG sections
   - Uplifting doesn't have CHANGELOG (acceptable - older filter)

### ✅ UNIQUE STRENGTHS PRESERVED

**Tech Innovation v1.1:**
- Most explicit gatekeeper enforcement (dedicated section with 6 examples)
- Scoring Calibration section (stage → score mapping)
- Comprehensive Examples section (5 scored examples including out-of-scope)

**Uplifting v4:**
- Pre-classification system (corporate, business, military flags)
- Doom-framing vs Solutions-framing guidance
- Outcome requirement section

**Investment-Risk v2:**
- Signal tier definitions at top (RED, YELLOW, GREEN, BLUE, NOISE)
- Tier-level gatekeeper enforcement (Evidence ≥5, Actionability ≥5)
- Post-processing scoring formula

---

## Final Structural Flow Comparison

### Uplifting v4
```
Header → PROMPT TEMPLATE → Scope (IN/OUT) → Doom-Framing → Outcome →
ARTICLE → Pre-classification → Dimensions (with filters) → Output
```

### Investment-Risk v2
```
Header → Signal Tiers → PROMPT TEMPLATE → ARTICLE → Dimensions (with filters) →
Tier Classification (with gatekeepers) → Metadata → Output → Scoring Formula
```

### Tech Innovation v1.1 ✅ HARMONIZED
```
Header (with Philosophy) → PROMPT TEMPLATE → Scope (INCLUDE/EXCLUDE) →
Gatekeepers (with examples) → ARTICLE → Dimensions (with filters) →
Calibration → Scoring Philosophy → Examples → Output → CHANGELOG
```

---

## Conclusion

✅ **HARMONIZATION SUCCESSFUL**

Tech Innovation v1.1 now follows the same structural principles as uplifting and investment-risk:
- Scope/rules BEFORE article
- Consistent inline filter format
- JSON output with metadata
- Philosophy statement in header

While preserving unique strengths:
- Most explicit gatekeeper enforcement (proposals vs pilots examples)
- Stage-to-score calibration mapping
- Comprehensive scored examples section

**All three filters are now structurally aligned while maintaining their domain-specific strengths.**
