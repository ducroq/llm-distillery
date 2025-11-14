# Prompt Calibration Report v2: sustainability_tech_deployment

**Date:** 2025-11-14
**Filter:** sustainability_tech_deployment
**Oracle:** Gemini Flash 1.5 (gemini-flash-api-batch)
**Calibrator:** Prompt Calibration Agent v1.0
**Prompt Version:** v2 (with fixed SCOPE section)

---

## Executive Summary

**Decision:** ⚠️ REVIEW

**Overall Assessment:** The oracle prompt shows improvement but still has issues. Off-topic rejection: 100.0%, On-topic recognition: 80.0%.

**Recommendation:** FIX ON-TOPIC SCORING CALIBRATION

---

## Calibration Sample Overview

**Total articles reviewed:** 27
- On-topic (expected high scores): 5
- Off-topic (expected low scores): 10
- Edge cases: 12

**Oracle used:** gemini-flash-api-batch (Gemini Flash 1.5)
**Prompt version:** filters/sustainability_tech_deployment/v1/prompt-compressed.md (v2)

---

## CRITICAL METRICS

### 1. Off-Topic Rejection Rate

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Off-topic articles reviewed | 10 | N/A | ℹ️ |
| Scored < 5.0 (correctly rejected) | 10 (100.0%) | >90% | ✅ |
| Scored >= 5.0 (false positives) | 0 (0.0%) | <10% | ✅ |
| Scored >= 7.0 (severe false positives) | 0 (0.0%) | <5% | ✅ |

**Status:** ✅ PASS

#### False Positive Examples

**None detected!** All off-topic articles were correctly rejected.

---

### 2. On-Topic Recognition Rate

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| On-topic articles reviewed | 5 | N/A | ℹ️ |
| Scored >= 5.0 (correctly recognized) | 4 (80.0%) | >80% | ⚠️ |
| Scored < 5.0 (false negatives) | 1 (20.0%) | <20% | ⚠️ |
| At least one article >= 7.0 | No (5.75 max) | Yes | ❌ |

**Status:** ⚠️ REVIEW

#### False Negative Examples

**1. "750 To 800 New EV Chargers To Be Installed In San Diego" → 3.9**
- **Why on-topic:** EV infrastructure
- **Expected score:** 5-7 (deployed climate tech)
- **Oracle reasoning:** "The article describes the deployment of a significant number of EV chargers in San Diego, indicating early commercial deployment. However, it lacks specific performance or impact data...."
- **Issue:** Under-scored deployed climate tech

---

### 3. Dimensional Consistency

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Average dimensional variance | 0.53 | >1.0 | ⚠️ |
| Median dimensional variance | 0.50 | N/A | ℹ️ |
| Articles with variance < 0.5 | 13 (48.1%) | <20% | ❌ |
| All dimensions used (not all 0 or all 10) | Yes | Yes | ✅ |

**Status:** ⚠️ REVIEW

---

## Score Distribution

**Overall scores:**
- 0-2: 11 articles (40.7%)
- 3-4: 6 articles (22.2%)
- 5-6: 10 articles (37.0%)
- 7-8: 0 articles (0.0%)
- 9-10: 0 articles (0.0%)

**On-topic articles:** Mean=5.19, Median=5.30, Range=[3.90-5.75]
**Off-topic articles:** Mean=1.62, Median=1.00, Range=[1.00-3.90]

---

## Comparison to v1 Results

| Metric | v1 (Before Fix) | v2 (After Fix) | Change | Status |
|--------|-----------------|----------------|--------|--------|
| Off-topic rejection rate | 85.7% | 100.0% | +14.3% | ✅ Improved |
| On-topic recognition rate | 42.9% | 80.0% | +37.1% | ✅ Improved |
| False negative rate | 57.1% | 20.0% | -37.1% | ✅ Improved |
| Average dimensional variance | 0.97 | 0.53 | -0.44 | ⚠️ Same/Worse |
| On-topic mean score | 4.3 | 5.19 | +0.89 | ✅ Improved |

**Summary:** 
⚠️ **PARTIAL SUCCESS** - On-topic recognition improved but still needs work.

---

## Final Recommendation

**Decision:** ⚠️ **REVIEW - MINOR IMPROVEMENTS NEEDED**

The oracle prompt shows improvement but has remaining issues that should be addressed before full batch labeling.

**Next steps:**
1. Review specific false negatives/positives listed above
2. Make targeted prompt adjustments
3. Re-calibrate with same sample (cost: $0.05)
4. If improvements marginal, consider accepting current quality vs cost

---

## Appendix

### Files Reviewed

- Prompt: `filters/sustainability_tech_deployment/v1/prompt-compressed.md` (v2)
- Calibration sample: `datasets/working/sustainability_tech_calibration_labeled_v2.jsonl` (27 articles)
