# ADR-010: Oracle Consistency Over Data Volume

**Date**: 2026-03-05
**Status**: Accepted
**Decision**: Invest in oracle prompt precision and anti-contamination design before scaling data volume. Oracle consistency is the strongest predictor of student model MAE.

## Context

After deploying five production filters, we observed a ~2x MAE spread (0.47-0.74) across filters using the same architecture, training pipeline, and comparable data volumes. The question: what explains why some filters learn much better than others?

## Evidence

| Filter | Version | Raw MAE | Cal. MAE | Training Data |
|--------|---------|---------|----------|---------------|
| investment-risk | v6 | 0.497 | 0.465 | 10.4K |
| belonging | v1 | 0.534 | 0.489 | 7.4K |
| uplifting | v6 | 0.673 | -- | 10.5K |
| sustainability_technology | v3 | 0.734 | -- | 10.6K |
| cultural-discovery | v4 | 0.795 | -- | 8.0K |

Belonging v1 achieves the second-best MAE with the *least* training data. Data volume is not the differentiator.

## Analysis: What the best filters share

### 1. Concept precision (primary factor)

The best-performing filters have sharp, observable concept boundaries. "Does this article describe investment risks?" and "Does this article show intergenerational bonds?" are questions with relatively objective answers. "Is this article uplifting?" and "Is this culturally interesting?" are inherently more subjective.

Subjective concepts produce inconsistent oracle scores. Inconsistent labels are noisy targets that a 1B-parameter student cannot learn from cleanly.

### 2. Anti-contamination design

Belonging v1 has the most aggressive anti-contamination mechanisms of any filter:

- **6 content type caps** that box out lookalike content (wellness, networking, tourism, self-help, corporate, online-only)
- **Critical filters per dimension** -- explicit "score 0-2 if..." rules placed *before* the scale table in the oracle prompt
- **Gatekeeper dimension** -- community_fabric < 3 caps the weighted average at 3.42

These mechanisms reduce oracle scoring variance by making edge cases explicit. When the oracle doesn't need to exercise judgment on ambiguous articles, it produces more consistent scores.

### 3. Clean noise floor

Belonging's distribution (89% LOW / 10% MEDIUM / 1% HIGH) creates an extremely clean noise floor. Most articles score near-zero on all dimensions, giving the model a strong baseline signal. The few genuine positives stand out clearly against this background.

### 4. Hard negatives via screen+merge

The scope probe (ADR-003) identified near-miss articles -- content that *resembled* belonging but wasn't. The model learns discriminative boundaries, not just "tech articles aren't belonging."

### 5. Calibration

Isotonic calibration (ADR-008) provided +8.3% improvement for belonging. Earlier filters were deployed without calibration, suggesting their reported MAEs could also improve.

## The mechanism

```
Precise concept definition
  -> Clear oracle prompt (less ambiguity)
    -> Consistent oracle scores (low label noise)
      -> Clean training signal
        -> Lower student MAE
```

The student model is a distillation target. Its ceiling is the oracle's consistency, not the data volume. Doubling data with noisy labels helps less than halving label noise with the same data.

## Decision

For new filters (nature_recovery, signs_of_wisdom, future-of-education):

1. **Invest heavily in Phase 1-2** -- concept grounding, critical filters, content type caps, gatekeepers. Use belonging v1 as the template for prompt structure.
2. **Validate oracle consistency in Phase 3** -- score 50-100 articles, check inter-article variance on similar content. If edge cases produce wildly different scores, fix the prompt before scaling.
3. **Add "score 0-2 if..." critical filters** to every dimension. These are the highest-ROI prompt improvement.
4. **Always fit isotonic calibration** (ADR-008). Free improvement, no reason to skip it.
5. **Data volume 5K-7K is sufficient** when oracle consistency is high. Don't chase 10K+ as a default.

## Consequences

**Positive:**
- New filter development spends more time on prompt quality, less on data volume
- Expected MAE for new filters: <0.55 (calibrated) if concept is well-defined
- Faster development cycle (5K articles instead of 10K)

**Negative:**
- More upfront work in Phase 1-2 (concept grounding, edge case enumeration)
- Some concepts (e.g., "uplifting") may be inherently fuzzy -- not all filters can achieve belonging-level MAE regardless of prompt quality
- May need to revisit existing filters (uplifting, cultural-discovery) with better prompts rather than more data

## Checklist for new filters

Before proceeding to Phase 4 (batch labeling), verify:

- [ ] Can you state in one sentence what scores HIGH vs. what does NOT? Is the boundary crisp?
- [ ] Does every dimension have "score 0-2 if..." critical filters?
- [ ] Are there content type caps for the 3-5 most common lookalike categories?
- [ ] Is there at least one gatekeeper dimension?
- [ ] Phase 3 validation (50-100 articles): do similar articles get similar scores?

## References

- `filters/belonging/v1/config.yaml` -- template for anti-contamination design
- `filters/belonging/v1/DEEP_ROOTS.md` -- concept grounding example
- `filters/belonging/v1/STATUS.md` -- full development history with metrics
- ADR-003: Screen+merge for needle-in-haystack filters
- ADR-008: Isotonic score calibration
