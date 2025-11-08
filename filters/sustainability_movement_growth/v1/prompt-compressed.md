# Climate Movement Growth Scoring

Rate on **8 dimensions** (0-10). Focus: Social momentum, behavior change, institutional action.

## Dimensions
1. **PARTICIPATION_GROWTH (25%)** - More people engaging?
   - 9-10: Explosive (>100% YoY, millions mobilized)
   - 7-8: Strong growth (30-100% YoY)
   - 5-6: Modest growth (10-30% YoY)
   - 1-2: Declining

2. **INSTITUTIONAL_ACTION (20%)** - Divestment, corporate commitments
   - 9-10: Widespread ($1T+ divested, systemic change)
   - 7-8: Major institutions ($10B+, Fortune 500)
   - 5-6: Some commitments
   - 1-2: Resisting

3. **CONSUMER_BEHAVIOR (20%)** - Market shifts (EVs, plant-based, solar)
   - 9-10: New normal (>30% market)
   - 7-8: Mainstream (15-30%)
   - 5-6: Growing (5-15%)
   - 1-2: No change

4. **YOUTH_ENGAGEMENT (12%)** - Gen Z mobilization
   - 9-10: Youth-led transformation
   - 7-8: Mass youth mobilization (millions)
   - 5-6: Growing movement
   - 1-2: Disengaged

5. **GEOGRAPHIC_SPREAD (10%)** - Local → Global
   - 9-10: Global (100+ countries)
   - 7-8: Continental (50+)
   - 5-6: National (multiple countries)
   - 1-2: Single location

6. **MEDIA_COVERAGE (6%)** - Frequency, tone
7. **POLITICAL_SALIENCE (5%)** - Electoral issue?
8. **ACCELERATION (2%)** - Momentum increasing?

## Examples
- Climate strikes: 4M people globally (9.2)
- Divestment: $40T committed (8.7)
- EV sales: 35% China market (8.5)

## Output JSON
```json
{
  "participation_growth": {"score": X, "reasoning": "..."},
  ...
  "movement_type": "protest | divestment | consumer | electoral | institutional"
}
```
