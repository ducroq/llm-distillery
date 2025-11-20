# Prefilter Harmonization Task

**Created:** 2025-11-20
**Priority:** Medium (Before Production Deployment)
**Blocks:** Phase 8-9 (Documentation & Deployment)
**Status:** TODO

---

## Problem Statement

Prefilters across the three active filters are **not harmonized**:

1. ❌ **Inconsistent interface** - Some have `should_label()`, some don't
2. ❌ **No standard usage** - Test functions instead of production code
3. ❌ **Not integrated** - Not used in batch scoring pipeline
4. ❌ **Cannot validate** - No systematic way to test prefilter performance

**Impact:** Cannot validate prefilter effectiveness, cannot integrate into production pipeline, inconsistent filter packages.

---

## Current State

### Base Class Exists ✅

**File:** `filters/base_prefilter.py`

**Provides:**
- Text sanitization (Unicode, HTML, invisible chars)
- Content length checking
- Standard interface: `should_label(article) -> (bool, str)`

### Filter Prefilter Status

| Filter | Class Exists | Implements `should_label()` | Integrated | Validated |
|--------|--------------|----------------------------|-----------|-----------|
| **uplifting v4** | ✅ `UpliftingPreFilterV1` | ❌ No | ❌ No | ❌ Deferred |
| **sustainability_tech_innovation v2** | ✅ `SustainabilityTechInnovationPreFilterV2` | ⚠️ Partial | ⚠️ Partial | ✅ Yes |
| **investment-risk v4** | ✅ `InvestmentRiskPreFilterV1` | ❌ No | ❌ No | ❌ Deferred |

### Current Prefilter Code Patterns

**uplifting v4:**
```python
class UpliftingPreFilterV1(BasePreFilter):
    VERSION = "1.0"
    # ... pattern definitions ...

def test_prefilter():  # ❌ Test function, not production code
    # Test cases
```

**Problem:** No `should_label()` implementation, only test function.

---

## Target State

### Harmonized Prefilter Interface

All prefilters should follow this structure:

```python
"""
{Filter Name} Pre-Filter v{version}

Blocks obvious low-value content before LLM labeling.

Philosophy: {filter philosophy}
"""

from typing import Dict, Tuple
from filters.base_prefilter import BasePreFilter


class {FilterName}PreFilter(BasePreFilter):
    """Fast rule-based pre-filter for {filter_name} content"""

    VERSION = "{version}"

    # Pattern definitions
    BLOCK_PATTERNS = [
        # List of patterns that should block
    ]

    PASS_PATTERNS = [
        # Exceptions that should pass despite block patterns
    ]

    def should_label(self, article: Dict) -> Tuple[bool, str]:
        """
        Determine if article should be sent to oracle for labeling.

        Args:
            article: Dict with 'title' and 'content' keys

        Returns:
            (should_send, reason):
            - (True, "passed"): Send to oracle
            - (False, "reason_code"): Block with specific reason
        """
        # 1. Check content length (framework leakage prevention)
        length_ok, reason = self.check_content_length(article)
        if not length_ok:
            return (False, reason)

        # 2. Extract and clean text
        title = article.get('title', '')
        content = article.get('content', article.get('text', ''))
        text = f"{title} {content}".lower()

        # 3. Apply block patterns
        for pattern in self.BLOCK_PATTERNS:
            if pattern.search(text):
                # Check exceptions
                has_exception = any(exc.search(text) for exc in self.PASS_PATTERNS)
                if not has_exception:
                    return (False, f"blocked_{pattern.pattern[:30]}")

        return (True, "passed")


# Factory function for easy import
def create_prefilter() -> {FilterName}PreFilter:
    """Factory function to create prefilter instance."""
    return {FilterName}PreFilter()
```

### Standard Usage in Batch Scorer

```python
# In ground_truth/batch_scorer.py
from filters.{filter_name}.{version}.prefilter import create_prefilter

prefilter = create_prefilter()

for article in articles:
    should_send, reason = prefilter.should_label(article)
    if should_send:
        # Send to oracle
        result = oracle.score(article)
    else:
        # Log blocked article
        stats['blocked'][reason] += 1
```

---

## Implementation Plan

### Phase 1: Harmonize Prefilter Implementations (2-3 hours)

**For each filter (uplifting v4, investment-risk v4):**

1. **Implement `should_label()` method**
   - Move pattern matching logic into method
   - Use `BasePreFilter.check_content_length()`
   - Return `(bool, str)` tuple

2. **Add factory function**
   - `create_prefilter()` for easy imports

3. **Keep test function**
   - Rename to `test_prefilter_patterns()`
   - Update to use `should_label()` interface

4. **Update imports**
   - Ensure proper inheritance from `BasePreFilter`

### Phase 2: Integration Testing (1-2 hours)

**For each filter:**

1. **Create test script**
   ```python
   # filters/{filter_name}/{version}/test_prefilter.py
   from prefilter import create_prefilter

   def test_prefilter_on_samples():
       prefilter = create_prefilter()

       # Test cases
       test_cases = [
           {"title": "...", "content": "...", "expected": (True, "passed")},
           {"title": "...", "content": "...", "expected": (False, "blocked_...")},
       ]

       for case in test_cases:
           result = prefilter.should_label(case)
           assert result == case['expected']
   ```

2. **Run on 100 sample articles**
   - Verify pass rate is reasonable
   - Check blocked articles are actually low-value

### Phase 3: Validation on Real Data (2-3 hours)

**For each filter:**

1. **Run on 1K+ raw articles**
   ```bash
   python scripts/validate_prefilter.py \
     --filter filters/{filter_name}/{version} \
     --source datasets/raw/master_dataset.jsonl \
     --sample-size 1000 \
     --output filters/{filter_name}/{version}/prefilter_validation_report.md
   ```

2. **Generate validation report**
   - Pass rate (% articles sent to oracle)
   - Block reason distribution
   - False negative rate (manually review 20 blocked articles)
   - False positive rate (manually review 20 passed articles)

3. **Update prefilter_validation_report.md**
   - Replace "DEFERRED" status with real metrics
   - Document any pattern adjustments needed

### Phase 4: Integrate into Batch Scorer (1 hour)

1. **Update `ground_truth/batch_scorer.py`**
   - Add prefilter integration
   - Track prefilter statistics (passed/blocked/reasons)

2. **Test end-to-end**
   - Score 100 articles with prefilter enabled
   - Verify statistics are tracked correctly

### Phase 5: Documentation (1 hour)

1. **Update Filter Development Guide**
   - Add prefilter interface requirements to Phase 2 (Architecture)
   - Add prefilter validation requirements to Phase 4 (Prefilter)
   - Add example harmonized prefilter code

2. **Update filter README files**
   - Document prefilter pass rate
   - Show example usage

---

## Validation Criteria

### PASS ✅

All three filters must have:
- ✅ Class inheriting from `BasePreFilter`
- ✅ `should_label(article) -> (bool, str)` method implemented
- ✅ Factory function `create_prefilter()`
- ✅ Integration tests passing
- ✅ Validation report with real metrics (1K+ articles)
- ✅ Integrated into batch scorer
- ✅ Pass rate documented in README

### Metrics Targets

| Filter | Target Pass Rate | Target False Negative Rate |
|--------|------------------|----------------------------|
| uplifting v4 | 85-95% | <5% |
| sustainability_tech_innovation v2 | 60-70% | <10% |
| investment-risk v4 | 40-60% | <10% |

**Note:** Pass rate varies by filter philosophy:
- uplifting: Permissive (minimal blocking)
- sustainability: Moderate (climate/energy focus)
- investment-risk: Stricter (heavy noise filtering)

---

## Timeline

**Total Effort:** 7-11 hours (1-2 days)

**Recommended Schedule:**
- Day 1: Phase 1-2 (Harmonize implementations + integration tests)
- Day 2: Phase 3-5 (Validate on real data + integrate + document)

**When to Execute:**
- ⏳ After Phase 6 (Model Training) complete
- ⏳ Before Phase 8 (Documentation) complete
- ⏳ Before Phase 9 (Deployment)

---

## Success Criteria

**Before marking this task complete:**

1. ✅ All three filters have harmonized prefilter interface
2. ✅ All integration tests passing
3. ✅ All prefilter validation reports updated with real metrics
4. ✅ Prefilters integrated into batch scorer
5. ✅ Filter Development Guide updated
6. ✅ All filter READMEs document prefilter performance

**Validation:**
```bash
# Check harmonization
python scripts/check_prefilter_harmonization.py

# Expected output:
# ✅ uplifting v4: Harmonized
# ✅ sustainability_tech_innovation v2: Harmonized
# ✅ investment-risk v4: Harmonized
```

---

## Related Documents

- **Base Prefilter:** `filters/base_prefilter.py`
- **Filter Development Guide:** `docs/agents/filter-development-guide.md`
- **Current Prefilter Validation Reports:**
  - `filters/uplifting/v4/prefilter_validation_report.md` (DEFERRED)
  - `filters/sustainability_tech_innovation/v2/prefilter_options_validation_report.md` (DONE)
  - `filters/investment-risk/v4/prefilter_validation_report.md` (DEFERRED)

---

## Notes

- This is **not blocking** Phase 6 (Model Training) - proceed with training
- Prefilters mainly optimize cost/speed in production
- Oracle inline filters provide primary quality control
- Can be completed in parallel with or after model training

---

**Last Updated:** 2025-11-20
