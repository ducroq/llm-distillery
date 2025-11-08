# Climate Policy Effectiveness Scoring

Rate policy on **8 dimensions** (0-10). Focus: OUTCOMES and REPLICABILITY, not announcements.

## Article
{title} | {content}

## Dimensions

1. **POLICY_OUTCOMES (30%)** - Did emissions/deployment measurably change?
   - 9-10: Transformative (Norway EVs 5%→90%)
   - 7-8: Clear impact (emissions fell 20%+)
   - 5-6: Modest outcomes
   - 3-4: Announced, no impact yet
   - 1-2: No data

2. **REPLICABILITY (25%)** - Can others copy this?
   - 9-10: Global blueprint (50+ copying)
   - 7-8: Highly replicable (10+ adopting)
   - 5-6: Some transferability
   - 3-4: Very context-dependent
   - 1-2: Impossible to replicate

3. **POLITICAL_DURABILITY (15%)** - Survives government change?
   - 9-10: 30+ years, bipartisan
   - 7-8: Institutionalized, hard to reverse
   - 5-6: Bipartisan support
   - 3-4: Single-party, vulnerable
   - 1-2: Executive order, easily reversed

4. **SPEED_OF_IMPACT (10%)** - How fast did it work?
   - 9-10: <1 year
   - 7-8: 1-3 years
   - 5-6: 3-5 years
   - 3-4: 5-10 years
   - 1-2: >10 years or no results

5. **EQUITY_IMPACT (8%)** - Who benefits? Just transition?
   - 9-10: Reparative, helps vulnerable
   - 7-8: Equity-centered
   - 5-6: Some equity provisions
   - 3-4: Neutral
   - 1-2: Regressive, harms poor

6. **ENFORCEMENT (7%)** - Are there teeth?
   - 9-10: Automatic market enforcement
   - 7-8: Strong penalties, monitoring
   - 5-6: Moderate enforcement
   - 3-4: Weak enforcement
   - 1-2: Voluntary, no penalties

7. **UNINTENDED_CONSEQUENCES (3%)** - Downsides minimized?
   - 9-10: Net positive all dimensions
   - 7-8: Minimal downsides
   - 5-6: Minor trade-offs
   - 3-4: Some significant downsides
   - 1-2: Major negative consequences

8. **POLICY_SPREADING (2%)** - Are others adopting?
   - 9-10: >15 jurisdictions adopted
   - 7-8: 5-15 adopted
   - 5-6: 2-5 considering
   - 3-4: Limited interest
   - 1-2: No one copying

## Gatekeeper
If POLICY_OUTCOMES < 5.0: cap at 4.9

## Output JSON
```json
{
  "policy_outcomes": {"score": X, "reasoning": "..."},
  ...
  "overall_assessment": "...",
  "policy_type": "carbon_tax | feed_in_tariff | ev_incentives | mandate | other",
  "stage": "proven | effective | promising | announced | ineffective"
}
```
