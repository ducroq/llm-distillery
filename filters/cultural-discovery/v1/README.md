# Cultural Discovery Filter v1

## Concept

Surfaces articles about:
1. **Discoveries** - New findings about art, culture, and history
2. **Connections** - Bridges between different peoples and civilizations

Target publications: **ovr.news** (Wisdom tab), **Busara**

## Dimensions

| Dimension | Weight | Category | Focus |
|-----------|--------|----------|-------|
| discovery_novelty | 0.25 | Discovery | What's new? What didn't we know? |
| heritage_significance | 0.20 | Discovery | Cultural/historical importance |
| cross_cultural_connection | 0.25 | Connection | Bridges between peoples |
| human_resonance | 0.15 | Connection | Lived experience, not dry facts |
| evidence_quality | 0.15 | Assessment | Well-researched? (gatekeeper) |

## High-Scoring Examples

- Archaeological discovery reveals shared ancestry of distant cultures
- Art restoration uncovers hidden layers showing cultural exchange
- Historical research connects modern tradition to ancient roots
- Cross-cultural music collaboration revives endangered heritage

## Filtered Out

- Political/conflict framing of cultural differences
- Tourism listicles ("10 must-see temples")
- Celebrity art auction news
- Cultural appropriation debates (polarizing, not connecting)

## Status

**Phase 1: Planning** - Initial config created

### Next Steps

1. [ ] Draft prompt-compressed.md (oracle instructions)
2. [ ] Write prefilter rules
3. [ ] Calibrate on sample articles
4. [ ] Collect training data (5K articles)
5. [ ] Train student model
