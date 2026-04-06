# ADR-015: Lenses as Perspectives, Not Partitions

**Date:** 2026-04-06
**Status:** Accepted

## Decision

Lenses (ovr.news tabs) are **overlapping perspectives** on the same article pool, not mutually exclusive partitions. Oracle prompts must not contain exclusion clauses like "this is NOT [other lens], score 0-2." The same article may score high on multiple lenses — that is correct behavior, not a bug.

## Context

### The problem

Thriving v1 (intended replacement for uplifting v7) failed with MAE 0.94 — far worse than uplifting v7's 0.67 on the same Thriving tab. Root cause: bimodal oracle score distribution with a sparse 2-5 "dead zone" the student model couldn't learn.

### What changed between uplifting v7 and thriving v1

The thriving v1 prompt was designed to be **orthogonal** to the Belonging lens:

- Removed `social_cohesion_impact` dimension (weight 0.20) because it overlapped with Belonging's `community_fabric`
- Added explicit exclusion: *"This filter does NOT measure community bonds, solidarity, or social connection — those belong to the Belonging lens"*
- Noise checklist included: *"Community event primarily about togetherness/solidarity → BELONGS TO BELONGING LENS (score 0-2)"*
- Redistributed weight to `justice_rights_impact` (0.15 → 0.25) and `human_wellbeing_impact` (0.30 → 0.40)

### Why orthogonality broke the model

Analysis of 5,568 thriving v1 training articles:

| Score zone | Articles | Pattern |
|-----------|----------|---------|
| Noise (WA 0-2) | 2,699 (48%) | 96% have all dimensions < 3. Coherent — model learns fine. |
| Dead zone (WA 2-5) | 1,261 (23%) | **59% are mixed-signal** — some dims >= 4, some < 3. High inter-dimension variance. |
| Signal (WA 5+) | 1,608 (29%) | Mostly coherent high scores. Model learns fine. |

The dead zone is full of articles that have *partial* thriving signals — a policy win (justice_rights high) but no wellbeing outcome, or a health finding (wellbeing moderate) but no rights dimension. The exclusion of social_cohesion removed the "glue" dimension that gave these borderline articles gradual partial credit in uplifting v7.

In uplifting v7, a community garden story scores social_cohesion 5-6 and pulls the weighted average into a learnable range. In thriving v1, the same story is told to score 0-2, creating a cliff between "not Belonging enough to be Belonging" and "not Thriving enough to be Thriving."

### Evidence from successful filters

The filters with the best MAE don't enforce orthogonality:

| Filter | MAE | Enforces exclusions? |
|--------|-----|---------------------|
| belonging v1 | 0.49 | No — doesn't exclude uplifting content |
| investment-risk v6 | 0.47 | No — standalone concept |
| nature_recovery v1 | 0.54 | No — doesn't exclude sustainability_technology |
| uplifting v7 | 0.67 | No — scores everything including Belonging-adjacent content |
| foresight v1 | 0.75 | No — any topic can show foresight |
| thriving v1 | 0.94 | **Yes** — excludes Belonging content |

### What overlap means in practice

A story about a community cooperative that trains women in solar panel installation could legitimately score:
- **Thriving**: high (wellbeing, economic dignity)
- **Belonging**: high (community bonds, mutual support)
- **Solutions**: medium (technology deployment)
- **Opportunity** (future): high (skills, livelihoods)

This is not double-counting. Each lens highlights a different facet. A reader browsing the Belonging tab sees the community angle; on Thriving they see the wellbeing outcome. The same article, different insight.

## Consequences

### For oracle prompt design
- **Never** include "this is NOT [other lens]" exclusions in prompts
- **Never** instruct the oracle to score 0-2 because content "belongs to" another lens
- Each prompt defines what it measures; silence on other lenses' dimensions is sufficient
- Scope checks should exclude genuinely out-of-scope content (corporate PR, military buildup) — not adjacent lenses

### For thriving v2 (if resumed)
- Restore a soft connecting dimension or restore social_cohesion
- Remove all Belonging exclusion clauses from the prompt
- Accept that some articles will score high on both Thriving and Belonging

### For new lenses (Opportunity, Breakthroughs)
- Design dimensions for what the lens measures, not against what other lenses measure
- Overlap with existing lenses is expected and acceptable

### For ovr.news
- The same article appearing in multiple tabs is correct behavior
- ADR-014 (percentile normalization) already handles cross-tab ranking — overlap doesn't create scoring problems
- Tab assignment on HOME uses the tab where the *normalized* score is highest, so articles appear where they're *most* relevant, not exclusively relevant
