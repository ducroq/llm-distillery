# Manual Validation Report - sustainability_technology v1

**Date**: 2025-11-23
**Validator**: Claude (Manual Review)
**Sample**: 100 articles from calibration run
**Status**: ❌ **VALIDATION FAILED - CRITICAL PREFILTER BUG**

## Executive Summary

Manual validation of the 100-article calibration revealed a **critical substring matching bug** in the prefilter causing ~30% false positive rate. The oracle scoring is working correctly, but the prefilter is passing irrelevant articles due to overly permissive keyword matching.

**Result**: **BLOCK progression to 5K training** until prefilter is fixed.

---

## Critical Finding: Prefilter Substring Matching Bug

### Root Cause

`filters/sustainability_technology/v1/prefilter.py:76`
```python
return any(kw in text for kw in keywords)  # BUG: Substring matching!
```

This causes false positives when keywords appear as substrings:
- `'oil'` matches **'turm**oil**'** (turmoil), 'b**oil**', 'f**oil**', 'c**oil**'
- `'gas'` could match 'Gas**par**', 'Ve**gas**'
- `'cop'` could match '**cop**e', '**cop**y', '**cop**per'
- `'ev '` could match 'l**ev**el', 'd**ev**elop'

### Evidence

**Ireland Baldwin celebrity article** (scored 1.0/10 avg - correctly low):
- Content: "Ireland Baldwin is opening up about family turm**oil**..."
- Keyword match: `'oil'` found at position 134 in "turm**oil**"
- Oracle correctly scored all dimensions 1.0 (not relevant)
- **Prefilter incorrectly passed** article due to substring match

---

## Score Distribution Analysis

### Overall Statistics (100 articles)

| Category | Count | % | Assessment |
|----------|-------|---|------------|
| **High (>6.0)** | 1 | 1% | ⚠️ Very low hit rate |
| **Med-High (5.0-6.0)** | 0 | 0% | ⚠️ No articles in this range |
| **Medium (3.0-5.0)** | 50 | 50% | ✓ Reasonable |
| **Low-Med (2.0-3.0)** | 19 | 19% | ~ Borderline |
| **Low (≤2.0)** | 30 | 30% | ❌ Prefilter failures |

**Interpretation**:
- **30% false positive rate** - Articles scoring ≤2.0 should have been blocked by prefilter
- **1% high scorer rate** - Very few articles are truly high-quality sustainability technology content
- **50% medium scorers** - Passing articles with marginal relevance

---

## False Positive Examples

### 1. Celebrity Gossip
**Article**: "Ireland Baldwin Calls Out 'Poisonous' And 'Narcissistic' Family Members"
**Score**: 1.0 average (all dimensions 1.0)
**Keyword match**: `'oil'` in "turm**oil**"
**Oracle assessment**: "The article discusses personal matters and does not provide any information about [dimensions]."
**Verdict**: ❌ Prefilter failure (substring bug)

### 2. Math/Statistics Paper
**Article**: "Weak Identification with Bounds in a Class of Minimum Distance Models"
**Score**: 1.5 average
**Content**: Statistical methods paper (arXiv)
**Oracle assessment**: Mostly 1-2 scores, "no mention of environmental impact"
**Verdict**: ❌ Prefilter failure (unclear which keyword triggered)

### 3. Telecom News
**Article**: "Orange boss thinks SFR carve [up]"
**Score**: 1.67 average
**Content**: Telecom company competition
**Likely keyword**: `'infrastructure'` or `'innovation'`
**Verdict**: ❌ Prefilter failure (overly broad keywords)

### 4. Dutch Soccer News
**Article**: "AZ blijft met invaller Troy Parrott jagen op bevrijdende treffer tegen Slowaken"
**Score**: 1.0 average (all dimensions 1.0)
**Content**: Soccer match report
**Likely keyword**: `'storm'` in metadata ("storm Benjamin")
**Verdict**: ❌ Prefilter failure

---

## Correct Low Scores (Not Prefilter Failures)

### "Green Energy Boom" Article
**Article**: "How One Country's Russian Gas Crisis Became a Green Energy Boom"
**Score**: 1.0 average (all dimensions 1.0)
**Content**: Only 70 words - truncated stub/teaser
**Oracle evidence**: "The article is a short summary. There is no information about..."
**Verdict**: ✓ Correct prefilter pass (has 'renewable', 'energy' keywords), ✓ Correct oracle scoring (insufficient content)

### "Kia EV4" Article
**Article**: "The Kia EV4 Shortlisted for 2026 Car of the Year"
**Score**: 4.8 average (Environment=3.0)
**Oracle evidence**: "Claims GHG reduction but ignores resource depletion, water use, or end-of-life recycling challenges"
**Verdict**: ✓ Correct scoring - product announcement lacks depth

---

## High Scorer Analysis

### Single High-Quality Article (7.0 average)
**Article**: "'A Good Year for Species': Conservationist Vivek Menon on His Journey"
**Scores**: TRL=9.0, Tech=7.0, Econ=5.0, Env=7.0, Social=7.0, Gov=7.0
**Content**: Wildlife conservation, species protection, IUCN leadership
**Oracle assessment**: Detailed evidence for each dimension
**Verdict**: ✓ Correct high scores - comprehensive sustainability content

**Only 1 out of 100 articles (1%) scored above 6.0** - indicates either:
1. Dataset has limited high-quality sustainability technology articles, OR
2. Oracle standards are appropriately rigorous, OR
3. Prefilter passing too many marginal articles (diluting the pool)

---

## Oracle Scoring Quality Assessment

✅ **Oracle scoring is working correctly**:
- Consistently scores irrelevant articles as 1.0 across all dimensions
- Provides clear evidence for each score
- Appropriately harsh on shallow content (e.g., product announcements)
- Correctly rewards comprehensive, detailed articles
- Enforces dimension independence (e.g., high TRL + low Economics for some articles)

No issues identified with prompt or oracle behavior.

---

## Recommended Fixes

### 1. Fix Substring Matching Bug (CRITICAL)

Replace line 76 in `prefilter.py`:

**Current (BROKEN)**:
```python
return any(kw in text for kw in keywords)
```

**Option A - Word Boundary Regex (RECOMMENDED)**:
```python
import re
# Compile patterns with word boundaries
patterns = [re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
            for kw in keywords]
return any(pattern.search(text) for pattern in patterns)
```

**Option B - Space-Padded Matching (SIMPLER)**:
```python
# Add spaces around text to catch word boundaries
padded_text = f" {text} "
# Add spaces around keywords that aren't already prefix patterns
safe_keywords = []
for kw in keywords:
    if kw.endswith('*') or kw.endswith('tion') or len(kw) > 8:
        safe_keywords.append(kw)  # Prefix match or long word (low false positive risk)
    else:
        safe_keywords.append(f" {kw} ")  # Force word boundaries
return any(kw in padded_text for kw in safe_keywords)
```

### 2. Tighten Overly Broad Keywords

Remove or scope these keywords:
- **Remove**: `'oil'`, `'gas'`, `'cop'` (too many false positives)
- **Replace with**: `'oil and gas'`, `'natural gas'`, `'cop27'`, `'cop28'`
- **Remove**: `'infrastructure'`, `'innovation'`, `'investments'`, `'subsidies'` (too generic)
- **Keep only in context**: Only match these if combined with sustainability terms

### 3. Re-run Calibration After Fix

After fixing prefilter:
1. Re-run 100-article calibration
2. Verify false positive rate drops to <5%
3. Check if high-scorer rate improves (should increase as junk is filtered)
4. **Only then** proceed to 5K training data generation

---

## Decision

❌ **DO NOT PROCEED TO 5K TRAINING**

**Rationale**:
1. 30% false positive rate is unacceptable - wastes oracle API costs
2. Substring matching bug is trivial to fix
3. May need to iterate on keyword list after fixing bug
4. Current dataset would produce poor student model (trained on 30% junk)

**Next Steps**:
1. Fix substring matching bug in prefilter.py
2. Re-run calibration (100 articles)
3. Verify false positive rate <5%
4. Create updated calibration report
5. **Then** proceed to 5K training

---

## References

- Calibration data: `sandbox/sustainability_technology_v1_calibration/`
- Prefilter bug: `filters/sustainability_technology/v1/prefilter.py:76`
- Correlation analysis: `dimension_analysis/CALIBRATION_REPORT.md` (approved, oracle working correctly)
- Filter Package Philosophy: `docs/agents/filter-development-guide.md`
