# Prompt Calibration Report: Fresh Sample Validation

**Date:** 2025-11-14
**Filter:** sustainability_tech_deployment
**Oracle:** Gemini Flash 1.5 (gemini-flash-api-batch)
**Calibrator:** Prompt Calibration Agent v1.0
**Prompt Version:** v2 (with fixed SCOPE section)
**Sample:** FRESH - completely different from calibration articles

---

## Executive Summary

**Decision:** ⚠️ REVIEW

**Overall Assessment:** ⚠️ The prompt shows mixed generalization. Fresh sample: 92.3% rejection, 100.0% recognition. Changes from v2: -7.7% off-topic, +20.0% on-topic.

**Recommendation:** GOOD GENERALIZATION - MINOR CALIBRATION ADJUSTMENT

---

## Key Question: Does the Prompt Generalize?

**The fresh sample contains 31 completely different articles that were NOT used during calibration.**

**Answer:** ⚠️ PARTIALLY. Some degradation on fresh articles, but still functional.

---

## Fresh Sample Overview

**Total articles reviewed:** 31
- On-topic (expected high scores): 2
- Off-topic (expected low scores): 13
- Edge cases: 16

**Oracle used:** gemini-flash-api-batch (Gemini Flash 1.5)
**Prompt version:** filters/sustainability_tech_deployment/v1/prompt-compressed.md (v2)

---

## CRITICAL METRICS

### 1. Off-Topic Rejection Rate

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Off-topic articles reviewed | 13 | N/A | ℹ️ |
| Scored < 5.0 (correctly rejected) | 12 (92.3%) | >90% | ✅ |
| Scored >= 5.0 (false positives) | 1 (7.7%) | <10% | ✅ |
| Scored >= 7.0 (severe false positives) | 0 (0.0%) | <5% | ✅ |

**Status:** ✅ PASS

#### False Positive Examples

**1. "Honda’s Tiny EV Hot Hatch Pretends It Has A Gas Engine" → 5.1**
- **Why off-topic:** no climate tech indicators
- **Oracle reasoning:** "The article describes a commercially available EV with a focus on the driving experience. While it doesn't provide specific performance or impact data, the fact that it's a deployed EV warrants a scor..."
- **Issue:** Should have been rejected

---

### 2. On-Topic Recognition Rate

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| On-topic articles reviewed | 2 | N/A | ℹ️ |
| Scored >= 5.0 (correctly recognized) | 2 (100.0%) | >80% | ✅ |
| Scored < 5.0 (false negatives) | 0 (0.0%) | <20% | ✅ |
| At least one article >= 7.0 | No (5.60 max) | Yes | ❌ |

**Status:** ⚠️ REVIEW

#### False Negative Examples

**None detected!** All on-topic articles were correctly recognized.

---

### 3. Dimensional Consistency

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Average dimensional variance | 0.43 | >1.0 | ⚠️ |
| Median dimensional variance | 0.12 | N/A | ℹ️ |
| Articles with variance < 0.5 | 18 (58.1%) | <20% | ❌ |
| All dimensions used (not all 0 or all 10) | Yes | Yes | ✅ |

**Status:** ⚠️ REVIEW

---

## Score Distribution

**Overall scores:**
- 0-2: 17 articles (54.8%)
- 3-4: 8 articles (25.8%)
- 5-6: 6 articles (19.4%)
- 7-8: 0 articles (0.0%)
- 9-10: 0 articles (0.0%)

**On-topic articles:** Mean=5.38, Median=5.38, Range=[5.15-5.60]
**Off-topic articles:** Mean=2.51, Median=2.00, Range=[1.00-5.10]

---

## Comparison: Fresh Sample vs. v2 Calibration (27 articles)

| Metric | v2 (27 calibration articles) | Fresh (31 NEW articles) | Difference | Generalization |
|--------|------------------------------|-------------------------|------------|----------------|
| Off-topic rejection rate | 100.0% | 92.3% | -7.7% | ✅ Good |
| On-topic recognition rate | 80.0% | 100.0% | +20.0% | ⚠️ Degraded |
| False positive rate | 0.0% | 7.7% | +7.7% | ⚠️ |
| False negative rate | 20.0% | 0.0% | -20.0% | ✅ |
| Dimensional variance | 0.53 | 0.43 | -0.10 | ✅ |

**Generalization Assessment:**
✅ **GOOD GENERALIZATION** - Some performance change on fresh articles, but still meets minimum quality standards.

---

## Final Recommendation

**Decision:** ⚠️ **REVIEW - ACCEPTABLE WITH MONITORING**

The prompt shows acceptable generalization with some performance variation. Consider proceeding with enhanced monitoring.

**Next steps:**
1. Review specific false negatives/positives from fresh sample
2. Consider minor prompt refinements if issues are systematic
3. Proceed to batch labeling with enhanced quality monitoring
4. Validate first 100 labeled articles manually

---

## Appendix

### Files Reviewed

- Prompt: `filters/sustainability_tech_deployment/v1/prompt-compressed.md` (v2)
- Fresh calibration sample: `datasets/working/sustainability_tech_calibration_labeled_fresh.jsonl` (31 articles)
- Comparison baseline: v2 calibration report (27 articles)
