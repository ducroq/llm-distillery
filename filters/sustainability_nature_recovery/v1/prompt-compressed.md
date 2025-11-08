# Nature Recovery Scoring

Rate on **8 dimensions** (0-10). Focus: Evidence nature RECOVERS when we stop harming it.

## Dimensions
1. **ECOSYSTEM_HEALTH (30%)** - Species recovery, biodiversity
   - 9-10: Thriving (species rebounding, biodiversity increasing)
   - 7-8: Clear recovery (populations growing 20%+)
   - 5-6: Stable, no longer declining
   - 1-2: Declining

2. **POLLUTION_REDUCTION (25%)** - Air/water quality
   - 9-10: 50%+ improvement (Beijing air quality)
   - 7-8: 20-50% improvement
   - 5-6: 10-20% improvement
   - 1-2: Worsening

3. **RESTORATION_SCALE (15%)** - Hectares restored
   - 9-10: >1M hectares
   - 7-8: 100k-1M ha
   - 5-6: 10k-100k ha
   - 1-2: <1k ha

4. **RECOVERY_SPEED (10%)** - How fast?
   - 9-10: Rapid (<5 years)
   - 7-8: Moderate (5-20 years)
   - 5-6: Slow (20+ years)
   - 1-2: No recovery

5. **PERMANENCE (8%)** - Protected status?
   - 9-10: Permanent (law/treaty)
   - 7-8: Strong protection
   - 5-6: Some protection
   - 1-2: Temporary

6. **CONNECTIVITY (5%)** - Ecosystem links
7. **HISTORICAL_VALIDATION (4%)** - Done before? (ozone, bald eagles)
8. **ATTRIBUTION (3%)** - Policy-driven vs natural?

## Examples
- Ozone layer healing (9.5)
- Beijing air quality improved 50% (8.2)
- Thames River fish return (7.8)

## Output JSON
```json
{
  "ecosystem_health": {"score": X, "reasoning": "..."},
  ...
  "ecosystem_type": "ocean | forest | wetland | air_quality | species | other"
}
```
