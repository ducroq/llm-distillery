# Strategic Decision: Pivot from Tech Deployment to Tech Innovation Filter

**Date:** 2025-11-15
**Status:** ACCEPTED
**Decision Maker:** User
**Context:** Fresh filter revalidation with 402K article corpus

---

## Decision

Replace the **sustainability_tech_deployment** filter (focused on "deployed-only" climate tech) with **sustainability_tech_innovation** filter (focused on "cool sustainable tech" including pilots, deployed systems, and validated research).

---

## Context

During fresh revalidation of all filters with the new 402,818-article corpus, the sustainability_tech_deployment filter showed severe degradation and multiple failed fix attempts.

### Performance Timeline

**v2 (Original - 2025-11-14):**
- Calibration: 83.3% deployed tech detection (5/6 articles)
- Fresh validation: 9.1% deployed tech detection (1/11 articles)
- False positive: Train protest article scored as "deployed tech"

**v3 (Tightened Prefilter + Strengthened Prompt):**
- Extended validation: 100% FP rate across 300 articles
- False positives: Future announcements ("2027 delivery"), proposals ("proposes to build")
- Root cause: "Deployed-only" constraint too strict for random corpus

**sustainability_tech_innovation v1 (Pivot Attempt):**
- Lowered gatekeeper: deployment_maturity 5.0 → 3.0 (allow pilots)
- Result: 85.7% FP rate on 300 articles (WORSE than v3)
- Critical issues:
  - Gatekeeper rules not enforced in batch_scorer.py
  - Oracle misinterprets proposals as pilots
  - Sustainability scope too broad
  - Prefilter still too strict (2.3% pass rate)

---

## Decision Rationale

### Why Pivot (Not Fix)?

1. **Fundamental constraint issue**: "Deployed-only" climate tech represents ~0.1% of random corpus
   - Would require processing 40,000-50,000 articles to score 2,500 (per v2 release report)
   - Random sampling inefficient for such rare targets

2. **User strategic direction**: "perhaps we should change the purpose of the filter? Just cool sustainable tech?"
   - Innovation filter serves broader discovery use case
   - Better alignment with content curation needs

3. **Multiple validation failures**: Three attempts (v2, v3, v1) all failed with high FP rates
   - Indicates fundamental mismatch, not fixable with incremental improvements

4. **Better use case separation**:
   - Innovation filter → Continuous discovery from mixed content
   - Deployed-only filter → Quarterly reports from curated climate news sources

---

## Alternatives Considered

### Option A: Keep Trying with Tech Deployment Filter
- **Pros**: Original vision preserved, very selective filter
- **Cons**: Three failures suggest fundamental issue, inefficient for random corpus
- **Decision**: REJECTED - Evidence shows this won't work for random content

### Option B: Use Curated Corpus for Deployed-Only Filter
- **Pros**: Deployed-only filter still valuable for specific use cases
- **Cons**: Requires building curated climate news source list first
- **Decision**: DEFERRED - Can revisit later for specialized reports

### Option C: Pivot to Innovation Filter (SELECTED)
- **Pros**:
  - Broader target (pilots + deployed + validated research)
  - Better fit for mixed content discovery
  - User-requested strategic direction
- **Cons**: Requires building new filter from scratch
- **Decision**: ACCEPTED

---

## Implementation Plan

### Phase 1: Build sustainability_tech_innovation v2

Based on v1 failure analysis, v2 must address:

1. **Fix gatekeeper enforcement** in batch_scorer.py
   - Implement deployment_maturity < 3.0 → cap at 2.9
   - Ensure post_classifier applies rules correctly
   - Add unit tests for gatekeeper logic

2. **Rewrite prefilter.py** with explicit categories:
   - BLOCK: Proposals, future-only announcements, pure theory, generic IT infrastructure
   - PASS: Pilots with results, commercial deployments, validated research with data
   - Target: 5-20% pass rate (vs v1's 2.3%)

3. **Strengthen prompt-compressed.md**:
   - Clear examples distinguishing:
     - Proposal ("plans to deploy 2027") ❌ vs Pilot ("operational prototype") ✅
     - Future ("will build") ❌ vs Current ("deployed since 2024") ✅
     - Generic IT ("Kubernetes") ❌ vs Sustainable tech ("solar+storage") ✅
   - Narrow sustainability scope to climate/energy/transport
   - Exclude medicinal/agricultural unless climate-related

4. **Validate thoroughly**:
   - Initial: 100 articles
   - Extended: 300 articles if FP < 20%
   - Target: <10% FP rate, 5-20% prefilter pass rate

### Phase 2: Archive Tech Deployment Filters

1. Move to `filters/sustainability_tech_deployment/archived/`:
   - v1/ (never production-ready)
   - v2/ (degraded on fresh validation)
   - v3/ (100% FP rate)

2. Document future use case:
   - Curated corpus: Climate/energy/sustainability news feeds
   - Periodic reports: Quarterly deployed tech analysis
   - Targeted validation: Feed known deployment announcements

---

## Success Criteria

sustainability_tech_innovation v2 is production-ready when:

1. ✅ Prefilter pass rate: 5-20% (not <3% like v1)
2. ✅ False positive rate: <10% on 300-article validation
3. ✅ Gatekeeper rules enforced correctly (unit tests pass)
4. ✅ Oracle distinguishes proposals from pilots (no "2027 delivery" false positives)
5. ✅ Sustainability scope clear (no medicinal plants unless climate-related)
6. ✅ Complete filter package (config, prompt, prefilter, post_classifier, README)

---

## Risks and Mitigation

**Risk 1: v2 also fails with high FP rate**
- Mitigation: Thorough calibration on diverse sample before extended validation
- Fallback: Reassess if innovation filter is viable, consider different approach

**Risk 2: Gatekeeper enforcement still broken**
- Mitigation: Fix batch_scorer.py first, add unit tests before filter development
- Validation: Test gatekeeper rules explicitly in validation

**Risk 3: Scope creep (too broad "innovation")**
- Mitigation: Clear OUT OF SCOPE boundaries in prompt (climate/energy/transport only)
- Validation: Monitor for off-topic articles in validation samples

---

## Related Decisions

- [2025-11-15 Dimensional Scoring Terminology](./2025-11-15-dimensional-scoring-terminology.md)

---

## References

- Original calibration: `filters/sustainability_tech_deployment/v2/release_report.md`
- v3 validation: `sandbox/sustainability_tech_deployment_v3_validation/`
- v1 pivot failure: `sandbox/sustainability_tech_innovation_v1_validation_*/`
- Filters on hold: `FILTERS_ON_HOLD.md`

---

**Decision documented by:** Claude Code
**Approved by:** User
**Next review:** After sustainability_tech_innovation v2 validation
