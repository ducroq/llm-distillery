# Nature Recovery Filter

**Version**: 1.0
**Status**: Phase 1 complete, Phase 2 next (prompt architecture)
**Philosophy**: "Nature recovers when we let it, and faster than we expect"
**Purpose**: Detect documented ecosystem recovery — hope grounded in data, not aspiration
**Target**: ovr.news Recovery tab

## Concept

Finds articles with **measured evidence** that ecosystems recover when human pressure is removed or restoration is applied. Grounded in restoration ecology, rewilding science, and proven recoveries (ozone layer, Yellowstone wolves, Thames fish, bald eagles).

Deliberately excludes: climate doom, climate tech (→ sustainability_technology), greenwashing, fundraising appeals, policy announcements without outcomes, symbolic gestures.

See `DEEP_ROOTS.md` for full scientific and philosophical grounding.

## Dimensions (6)

| # | Dimension | Weight | Role |
|---|-----------|--------|------|
| 1 | recovery_evidence | 25% | GATEKEEPER — is nature actually recovering? |
| 2 | measurable_outcomes | 20% | Quantified data: before/after, populations, areas |
| 3 | ecological_significance | 20% | Keystone species, critical habitats, trophic cascades |
| 4 | restoration_scale | 15% | Geographic scope and temporal duration |
| 5 | human_agency | 10% | Recovery caused by deliberate action or policy? |
| 6 | protection_durability | 10% | Will this recovery last? |

## Anti-Contamination (ADR-010)

| Content Type | Max Score | Example |
|---|---|---|
| climate_doom | 2.0 | "Extinction crisis accelerates" |
| climate_tech | 3.0 | "New solar panel efficiency record" |
| greenwashing | 2.0 | "Company pledges net zero by 2050" |
| conservation_appeal | 2.0 | "Donate to save the rainforest" |
| policy_announcement | 3.0 | "Government to protect 30% of ocean" |
| symbolic_gesture | 3.0 | "1,000 trees planted on Earth Day" |

## Examples

- **Ozone layer (9.5)**: Montreal Protocol → measurable healing, 40-year trend, planetary scale
- **Yellowstone wolves (8.5)**: Reintroduction → trophic cascade, rivers changed course, 25+ years
- **Beijing air quality (7.8)**: Policy-driven PM2.5 reduction of 50%, decade of data
- **Bald eagles (8.0)**: DDT ban → 417 to 9,789 breeding pairs, delisted
- **"Amazon is burning" (1.5)**: Decline only, no recovery evidence
- **"Company plants 1,000 trees" (2.5)**: Symbolic gesture, no ecological data

## Development Progress

See `STATUS.md` for detailed phase tracking.

## Key Design Decisions

- **Consolidated from 8 to 6 dimensions**: Bottom 3 (5%, 4%, 3% weight) too low for student model to learn. See STATUS.md for rationale.
- **recovery_evidence as gatekeeper** (not ecosystem_health): More precise — "is nature bouncing back?" not "is this ecosystem important?"
- **Boundary with sustainability_technology**: This filter scores *ecological outcomes*; sustainability_tech scores *technologies* via LCSA.
- **ADR-010 applied**: Critical filters per dimension, content type caps, gatekeeper — the belonging v1 pattern that achieves MAE < 0.55.
