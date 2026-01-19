# Resilience Filter v1

**Status:** Design complete, not yet trained
**Created:** 2026-01-17

## Philosophy

> Bad things happen. What matters is how systems respond.

Most news covers the problem. This filter finds the response and recovery evidence.

This is **not** toxic positivity - it requires a real adversity baseline. And it's **not** doom - it specifically looks for evidence of response and learning.

## Why This Filter?

Constructive journalism principle: complete coverage means problem + response + outcome.

| Filter | Focus |
|--------|-------|
| investment_risk | "Should I move my money?" |
| uplifting | "What good is happening?" |
| nature_recovery | "Are ecosystems bouncing back?" |
| **resilience** | "Are systems learning and adapting?" |

## Dimensions

| Dimension | Weight | Question |
|-----------|--------|----------|
| adversity_severity | 0.15 | HOW bad was the original problem? |
| response_initiated | 0.20 | HAS action been taken? |
| response_effectiveness | 0.20 | IS it actually working? |
| institutional_learning | 0.15 | ARE lessons being captured? |
| replication_potential | 0.10 | CAN others copy this? |
| evidence_quality | 0.20 | HOW documented? (GATEKEEPER) |

## Gatekeeper

If `evidence_quality < 3.5`, overall score capped at 3.0.

This prevents:
- PR spin without verification
- Premature celebration
- Wishful thinking

## Tier Classification

| Tier | Description | Condition |
|------|-------------|-----------|
| **high** | Verified resilience | Effective response + learning + evidence |
| **medium** | Emerging resilience | Response underway, early signs |
| **low** | Weak signal | Problem acknowledged, response unclear |
| **not_resilience** | Not applicable | No adversity baseline or no response |

## Example Articles

**High tier:**
- "How Japan rebuilt Tohoku: 10 years after the tsunami"
- "Rwanda's healthcare transformation: from genocide to model system"

**Medium tier:**
- "California's new wildfire protocols show early promise"

**Low tier:**
- "Community begins cleanup after flood" (no effectiveness data yet)

**Not resilience:**
- "Hurricane threatens coastal communities" (pure doom, no response)
- "Company launches new product" (no adversity baseline)

## Relationship to Other Filters

```
                    ┌─────────────────┐
                    │ adverse event   │
                    │   (detected)    │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
      ┌───────────┐  ┌───────────┐  ┌───────────┐
      │investment │  │ resilience│  │  nature   │
      │   _risk   │  │  (this)   │  │ _recovery │
      └───────────┘  └───────────┘  └───────────┘
           │              │              │
           ▼              ▼              ▼
      "act on your   "systems are   "ecosystems
       portfolio"     learning"      heal"
```

## Use Cases

1. **ovr.news** - Balance problem reporting with recovery evidence
2. **Policy research** - Find what actually works
3. **Civic education** - Counter doom-scrolling with documented resilience
4. **Disaster preparedness** - Learn from others' responses

## Next Steps

1. [ ] Build prefilter keywords list
2. [ ] Source stratified sample (official reports, journalism, academic)
3. [ ] Run oracle calibration (target: 500 articles)
4. [ ] Analyze dimension correlations
5. [ ] Train model if correlations acceptable
6. [ ] Deploy to NexusMind

## Design Notes

**Origin:** Evolved from discussion about generalizing investment_risk to "adverse_event". Realized the interesting dimension wasn't detecting bad things (already covered), but detecting response and recovery.

**Key insight:** The `response_status` dimension transforms doom into constructive coverage. It's not about ignoring problems - it's about complete coverage.
