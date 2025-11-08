# Sustainability Pillar 2: Economics Favor Action Filter

**Version**: 1.0
**Purpose**: Prove that green solutions are economically rational, not expensive sacrifice
**Pillar**: "Economics Favor Climate Action (it's not sacrifice anymore)"

## Overview

This filter evaluates the economic competitiveness of climate solutions vs fossil fuels. It supports the narrative that climate action makes economic sense - renewables are cheaper, green companies are profitable, and fossil assets are being stranded.

### What Makes "Economics Favor" Stories

**Tier 1 (8+): Economically Superior**
- ✅ Solar LCOE $20/MWh vs coal $60/MWh - IEA confirms
- ✅ EVs reach price parity with ICE, no subsidies needed
- ✅ Coal plants close early as uneconomical vs renewables
- ✅ Renewable energy investment $500B/year, surpassing fossil

**Tier 2 (6.5-7.9): Competitive**
- ✅ Onshore wind matches natural gas LCOE in most US markets
- ✅ Heat pump companies profitable, sales doubled YoY
- ✅ Green jobs growing faster than fossil jobs declining

**Filtered Out (<5.0): Subsidy-Dependent**
- ❌ "Technology costs 3x fossil baseline, needs heavy subsidies"
- ❌ "Industry unprofitable, requires government support"
- ❌ Pure advocacy without cost data

## Eight Dimensions

See `prompt-compressed.md` for full scoring rubrics. Summary:

1. **Cost Competitiveness (25%)** - LCOE vs fossil, gatekeeper dimension
2. **Profitability (20%)** - Are green companies making money?
3. **Job Creation (15%)** - Net jobs vs fossil displacement
4. **Stranded Assets (15%)** - Fossil infrastructure uneconomical
5. **Investment Flows (10%)** - Capital moving to green
6. **Payback Period (8%)** - ROI timeline
7. **Subsidy Dependence (4%)** - Lower = better
8. **Economic Multiplier (3%)** - Broader benefits (health, security)

## Example Scores

**Solar Cost Decline (9.2)**: Perfect economics story
- Cost: 10, Profitability: 9, Jobs: 7, Stranded: 8, Investment: 9, Payback: 6, Subsidies: 8, Multiplier: 7

**Green Hydrogen Early Stage (3.2)**: Too subsidy-dependent
- Cost: 3, Profitability: 2, Jobs: 5, Stranded: 3, Investment: 6, Payback: 3, Subsidies: 2, Multiplier: 5

## Related Filters

Part of 5-pillar sustainability framework:
1. Tech Works (`sustainability_tech_deployment`)
2. **Economics Favor** (this filter)
3. Policy Works (`sustainability_policy_effectiveness`)
4. Nature Recovers (`sustainability_nature_recovery`)
5. Movement Growing (`sustainability_movement_growth`)
