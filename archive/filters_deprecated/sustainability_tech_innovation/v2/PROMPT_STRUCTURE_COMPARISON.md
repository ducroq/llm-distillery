# Prompt Structure Comparison

Comparing sustainability_tech_innovation v1.1 with uplifting v4 and investment-risk v2

---

## Structure Overview

### Uplifting v4
```
1. Header (Purpose, Version, Target, Focus, Oracle Output)
2. ## PROMPT TEMPLATE
   ├── SCOPE (In Scope / Out of Scope)
   ├── ## Doom-Framing vs Solutions-Framing
   ├── ## Outcome Requirement
   ├── ARTICLE: Title/Text
   ├── STEP 1: Pre-classification (Corporate Finance, Business News, Military, Harm Documentation)
   ├── STEP 2: Score Dimensions (8 dimensions with inline ❌ CRITICAL FILTERS)
   └── Output Format (JSON)
3. ## POST-PROCESSING REFERENCE
```

### Investment-Risk v2
```
1. Header (Purpose, Version, Target, Focus, Philosophy)
2. ## SIGNAL TIERS (tier definitions)
3. ## PROMPT TEMPLATE
   ├── ARTICLE: Title/Source/Published/Text
   ├── Score Dimensions (8 dimensions with inline ❌ CRITICAL FILTERS)
   ├── Classify Signal Tier (with gatekeeper enforcement: Evidence ≥5, Actionability ≥5)
   ├── Metadata (risk indicators, asset classes, time horizon, actions)
   └── Output JSON
4. ## SCORING FORMULA (Applied post-labeling)
5. ## CHANGELOG
```

### Sustainability Tech Innovation v1.1 (CURRENT)
```
1. Header (Purpose, Version, Focus, Oracle Output)
2. ## PROMPT TEMPLATE
   ├── ## CRITICAL: What is "Tech That Works"?
   ├── ARTICLE: Title/Text
   ├── ## ⚠️ CRITICAL: MANDATORY GATEKEEPER RULES ⚠️ (NEW - placed AFTER article)
   ├── ## Dimensions (8 dimensions with inline ❌ CRITICAL FILTERS)
   ├── ## Scoring Calibration (Development Stage → Overall Score Mapping)
   ├── ## Scoring Philosophy (Evidence hierarchy)
   ├── ## ⚠️ REMINDER: Gatekeeper Enforcement ⚠️ (NEW - reminder section)
   ├── ## Examples (4 scored examples: High, Medium, Low, Out of Scope)
   └── ## Output Format (JSON)
3. ## CHANGELOG
```

---

## Key Structural Differences

### 1. **ARTICLE Placement**

| Filter | Placement | Rationale |
|--------|-----------|-----------|
| **Uplifting** | AFTER scope/framing sections | Oracle sees scope rules BEFORE article |
| **Investment-Risk** | START of prompt template | Oracle sees article immediately |
| **Tech Innovation** | AFTER "What is Tech That Works" | Oracle sees inclusion/exclusion criteria, THEN article |

**⚠️ ISSUE:** Tech Innovation places ARTICLE before GATEKEEPER RULES, but uplifting places ARTICLE after scope rules.

**RECOMMENDATION:** Move ARTICLE after gatekeeper rules for consistency with uplifting.

---

### 2. **Gatekeeper Enforcement**

| Filter | Approach | Location |
|--------|----------|----------|
| **Uplifting** | Pre-classification (STEP 1) + dimension-level gatekeepers | At start of scoring, inline with dimensions |
| **Investment-Risk** | Tier-level gatekeepers | In tier classification section |
| **Tech Innovation** | Dedicated gatekeeper section (NEW v1.1) | After article, before dimensions, with reminder later |

**✅ STRENGTH:** Tech Innovation has most explicit gatekeeper enforcement with examples.

**⚠️ DIFFERENCE:** Tech Innovation has TWO gatekeeper sections (rules + reminder), others have ONE.

---

### 3. **Examples**

| Filter | Examples | Placement |
|--------|----------|-----------|
| **Uplifting** | Inline examples for each concept | Throughout prompt (doom-framing, outcome requirement) |
| **Investment-Risk** | No dedicated examples section | N/A |
| **Tech Innovation** | 4 scored examples (9.1, 5.2, 3.8, 1.6, 1.0) | Dedicated ## Examples section after scoring philosophy |

**✅ STRENGTH:** Tech Innovation has most comprehensive examples section.

---

### 4. **Calibration Guidance**

| Filter | Calibration | Location |
|--------|-------------|----------|
| **Uplifting** | No explicit calibration section | Implied in 0-10 scale descriptions |
| **Investment-Risk** | Scoring formula (post-processing) | Separate section after prompt template |
| **Tech Innovation** | Scoring Calibration (Development Stage → Overall Score) | Inside prompt template |

**✅ STRENGTH:** Tech Innovation has explicit calibration mapping (mass_deployment → 8-10, pilots → 4-6, etc.)

---

### 5. **Metadata**

| Filter | Metadata | Included? |
|--------|----------|-----------|
| **Uplifting** | None | No |
| **Investment-Risk** | Risk indicators, asset classes, time horizon, actions | Yes - in prompt template |
| **Tech Innovation** | primary_technology, deployment_stage, confidence | Yes - in output format |

**✅ OK:** All filters include metadata in output format.

---

## Harmonization Recommendations

### HIGH PRIORITY

1. **Move ARTICLE placement** to match uplifting structure:
   ```
   ## PROMPT TEMPLATE

   ## CRITICAL: What is "Tech That Works"?
   [Include/Exclude criteria]

   ## ⚠️ CRITICAL: MANDATORY GATEKEEPER RULES ⚠️
   [Gatekeeper rules with examples]

   ARTICLE:
   Title: {title}
   Text: {text}

   ## Dimensions
   [Score dimensions]
   ```

2. **Remove duplicate gatekeeper section** - keep only ONE section (the main one after article), remove the reminder section later.

3. **Add Philosophy line** to header (like investment-risk):
   ```
   **Purpose**: Rate cool sustainable tech that WORKS - deployed tech, working pilots, validated breakthroughs.
   **Philosophy**: "Pilots and research need real results, not just theory."
   ```

### MEDIUM PRIORITY

4. **Consistent inline filter format** - All three filters use `❌ CRITICAL FILTERS` which is good ✅

5. **Scoring Calibration** - Keep this, it's a strength ✅

6. **Examples section** - Keep this, it's a strength ✅

### LOW PRIORITY

7. **CHANGELOG** - Already included ✅

---

## Summary

### ✅ HARMONIZED
- Inline `❌ CRITICAL FILTERS` for each dimension
- JSON output format with metadata
- CHANGELOG section
- Header structure (Purpose, Version, Target, Focus)

### ⚠️ NEEDS HARMONIZATION
1. **ARTICLE placement** - should be AFTER gatekeeper rules (like uplifting places AFTER scope)
2. **Duplicate gatekeeper sections** - remove reminder section, keep only one
3. **Philosophy line** - add to header

### ✅ UNIQUE STRENGTHS (keep these)
- Explicit gatekeeper enforcement with examples
- Scoring Calibration section (Development Stage → Score mapping)
- Comprehensive Examples section (4 scored examples)

---

**RECOMMENDATION:** Apply HIGH PRIORITY harmonizations to match uplifting/investment-risk structure while preserving unique strengths.
