# Filters Placed On Hold - 2025-11-15

## Summary

Three sustainability pillar filters have been placed on hold during the fresh revalidation process:

1. **sustainability_tech_deployment v2/v3** → Pivoting to **sustainability_tech_innovation**
2. **sustainability_economic_viability v1**
3. **sustainability_policy_effectiveness v2**

---

## sustainability_tech_deployment v2/v3

**Status:** ON HOLD → Pivoting to **sustainability_tech_innovation**
**Reason:** "Deployed-only" constraint too restrictive for random corpus, pivot attempt failed

### Issues Discovered

**v2 Performance Degradation:**
- Original calibration (2025-11-14): 83.3% deployed tech detection (5/6 articles)
- Fresh validation (seed=20000): 9.1% deployed tech detection (1/11 articles)
- False positive: Train protest article scored as "deployed" because trains mentioned

**v3 Attempt (Tightened Prefilter + Strengthened Prompt):**
- Blocked research papers, social media, infrastructure disruption
- Required explicit deployment keywords
- Extended validation (300 articles, seeds 22000/24000/25000): **100% FP rate**
- False positives: Future announcements ("delivery 2027"), proposals ("proposes to build")

**Pivot Attempt: sustainability_tech_innovation v1 (Cool Tech, Not Just Deployed):**
- Lowered gatekeeper: deployment_maturity threshold 5.0 → 3.0 (allow pilots)
- Loosened prefilter: Added "pilot", "demonstration", "validated" keywords
- Result: **85.7% FP rate** on 300 articles (worse than v3!)
- Critical issues:
  - Gatekeeper rules not enforced in batch_scorer.py
  - Oracle misinterprets proposals as pilots
  - Sustainability scope too broad (medicinal plants passed)
  - Prefilter still too strict (2.3% pass rate vs 5-20% target)

### Strategic Decision: Pivot to sustainability_tech_innovation

**Status:** STRATEGIC PIVOT (Permanent)
**Date:** 2025-11-15
**Decision Maker:** User

**Rationale:**
- "Deployed-only" climate tech is too rare in random corpus (~0.1% hit rate)
- Multiple attempts failed (v2: 9.1% detection, v3: 100% FP, v1 pivot: 85.7% FP)
- User strategic direction: "perhaps we should change the purpose of the filter? Just cool sustainable tech?"
- Innovation filter (pilots + deployed + validated research) serves broader discovery use case
- Deployed-only filter better suited for curated climate news sources, not random corpus

**Required Actions for sustainability_tech_innovation v2:**
1. **Fix gatekeeper enforcement** in batch_scorer.py code
2. **Rewrite prefilter** with clearer categories:
   - BLOCK: Proposals, future announcements, pure theory, IT infrastructure
   - PASS: Pilots, commercial deployments, validated research with results
3. **Strengthen oracle prompt** to distinguish:
   - Proposal ("plans to deploy") ❌ vs Pilot ("deployed prototype") ✅
   - Future announcement ("2027 delivery") ❌ vs Current status ("operational since 2024") ✅
   - Generic IT ("Kubernetes") ❌ vs Sustainable tech ("solar + storage") ✅
4. **Narrow sustainability scope** to climate/energy/transport (exclude medicinal/agricultural unless climate-related)
5. **Validate thoroughly** - 300+ articles before production

---

## sustainability_economic_viability v1

**Status:** ON HOLD
**Reason:** Incomplete filter package - missing `post_classifier.py`

### Issues Discovered

**Critical:**
- ❌ Missing `post_classifier.py` - cannot calculate overall scores or tier assignments
- ❌ Cannot validate gatekeeper rules (cost_competitiveness < 5.0 cap)
- ❌ No proper post-processing of oracle dimensional scores

**Validation Results (100 articles, seed=30000):**
- Prefilter block rate: 91.0% (91/100 blocked)
- Articles scored: 9/100
- Oracle IS working correctly (dimensional scores generated)
- Off-topic articles: 6/9 (66.7%) - correctly scored low by oracle
- Barely relevant: 2/9 (22.2%)
- Potentially relevant: 1/9 (11.1%)

### Required Actions Before Production

1. **Create post_classifier.py** based on config.yaml weights:
   - Implement weighted average calculation (8 dimensions)
   - Implement gatekeeper rule: cost_competitiveness < 5.0 → cap at 4.9
   - Implement tier assignment logic

2. **Review prefilter** - 91% block rate may be too high or corpus has few matching articles

3. **Re-run validation** with complete filter package

4. **Compare to existing release_report.md** (if any) to understand filter history

---

## sustainability_policy_effectiveness v2

**Status:** ON HOLD
**Reason:** Not yet validated (dependency on economic_viability completion)

### Context

This filter was scheduled for validation after sustainability_economic_viability v1, but was placed on hold to avoid accumulating more incomplete filter validations.

### Required Actions Before Validation

1. Check if `post_classifier.py` exists for this filter
2. Review filter package completeness
3. If complete, proceed with 100-article fresh validation
4. If incomplete, fix before validating

---

## Filters Successfully Validated

The following filters passed fresh revalidation with the new 402K article corpus:

1. ✅ **investment-risk v2** - 0% FP rate, 100% academic blocking (PRODUCTION READY)
2. ✅ **uplifting v4** - 0% FP rate, 82.2% block rate, 58.3% uplifting detection (PRODUCTION READY)

---

## Next Steps

**Focus: sustainability_tech_innovation v2**

The immediate priority is building sustainability_tech_innovation v2 based on lessons learned from v1 failure:

1. **Fix gatekeeper enforcement** in batch_scorer.py
   - Implement deployment_maturity < 3.0 → cap at 2.9
   - Ensure post_classifier applies rules correctly

2. **Rewrite prefilter.py** with explicit categories:
   - Block: Proposals, future-only announcements, pure theory, generic IT
   - Pass: Pilots with results, commercial deployments, validated research

3. **Strengthen prompt-compressed.md**:
   - Clear examples distinguishing proposals vs pilots vs deployed
   - Explicit OUT OF SCOPE for IT infrastructure and non-climate tech
   - Narrow sustainability to climate/energy/transport

4. **Validate thoroughly**:
   - Initial 100-article test
   - Extended 300-article validation if FP rate < 20%
   - Target: <10% FP rate, 5-20% prefilter pass rate

**Later:**
- Fix sustainability_economic_viability v1 (create post_classifier.py)
- Validate sustainability_policy_effectiveness v2 completeness
- Batch score production datasets only with validated filters

---

**Note created:** 2025-11-15
**Context:** Fresh revalidation with historical_dataset_19690101_20251115.jsonl (402,818 articles)
