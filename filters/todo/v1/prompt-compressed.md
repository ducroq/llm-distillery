# Sustainability Filter

**Purpose**: Rate content for sustainability relevance, impact potential, and credibility based on DEPLOYED TECHNOLOGY and MEASURED OUTCOMES.

**Version**: 1.0-compressed
**Target**: Gemini Flash 1.5 / Claude Haiku / Fast models

**Focus**: Distinguishes announcements from deployments, detects greenwashing/vaporware/fossil fuel delay tactics.

---

## PROMPT

```
Analyze this article for sustainability impact based on CONCRETE ACTIONS and MEASURABLE OUTCOMES, not aspirational statements.

ARTICLE:
Title: {title}
Source: {source}
Published: {published_date}
Text: {text}

STEP 1: Pre-classification Filters

A) GREENWASHING: Corporate sustainability reports, net-zero pledges, carbon offsets, ESG ratings, awards, green marketing?
   - If YES and NO third-party verification/specific emissions data/deployed technology/regulatory compliance → FLAG "greenwashing_risk" (max_credibility = 3)

B) VAPORWARE: Product announcements, prototypes, pilots, early-stage concepts?
   - If YES and NO deployed units/customer contracts/operational data/peer-reviewed validation → FLAG "vaporware" (max_investment_readiness = 4)

C) FOSSIL FUEL TRANSITION: "Clean coal", "natural gas bridge", CCS for oil recovery, fossil hydrogen without lifecycle accounting?
   - If YES → FLAG "fossil_transition" (max_impact = 4)
   - Exception: Genuine renewable H2, direct air capture for storage, fossil asset retirement CAN score 7+

D) ANNOUNCEMENT vs DEPLOYMENT:
   - Announcements/pledges → Score LOWER
   - Deployed technology/measured outcomes → Score HIGHER

STEP 2: Score Dimensions (0-10)

1. **Climate Impact Potential**: Demonstrable GHG reduction/carbon sequestration/adaptation?
   - 0-2: Minimal/unproven | 3-4: Modest/theoretical | 5-6: Significant/pilot-proven | 7-8: Transformative/deployed | 9-10: Breakthrough/scaling
   - Must have direct climate mechanism with quantified impact (tons CO2e, % reduction, MW capacity)
   - Requires lifecycle accounting (EVs only count if grid is clean)

2. **Technical Credibility** (GATEKEEPER: if <5, max overall = 4):
   - 0-2: Unverified claims | 3-4: Industry sources | 5-6: Some independent data | 7-8: Peer-reviewed | 9-10: Multiple independent confirmations
   - Scientific evidence, realistic vs. thermodynamic limits, specific metrics, independent verification

3. **Economic Viability**: Path to cost-competitiveness with fossil alternatives?
   - 0-2: No path | 3-4: Needs major subsidies | 5-6: Approaching parity | 7-8: Competitive now | 9-10: Cheaper than fossil
   - Unit economics (LCOE, $/ton CO2, payback period), demonstrated demand

4. **Deployment Readiness**: What stage? Research → pilot → commercial → scaling?
   - 0-2: Concept/lab | 3-4: Pilot | 5-6: First commercial | 7-8: Proven at scale | 9-10: Mass deployment
   - TRL 1-3: basic research | TRL 4-5: applied research | TRL 6-7: pilot | TRL 8: commercial | TRL 9: scaling

5. **Systemic Impact**: Enables broader decarbonization? Addresses bottlenecks? Gigaton-scale potential?
   - 0-2: Niche | 3-4: Sectoral | 5-6: Cross-sectoral | 7-8: Economy-wide | 9-10: Global infrastructure
   - Grid storage enables renewables, green steel decarbonizes construction, etc.

6. **Justice & Equity**: Avoids harm to frontline/indigenous communities? Equitable access?
   - 0: Actively harmful | 3: Neutral | 5: Some equity considerations | 7: Equity-centered | 10: Reparative

7. **Innovation Quality**: Genuine breakthrough or incremental? Solves previously unsolved problem?
   - 0-2: Hype | 3-4: Incremental | 5-6: Significant | 7-8: Breakthrough | 9-10: Paradigm shift
   - NOT innovation theater - must have technical substance

8. **Evidence Strength**: Quality and independence of sources?
   - 0-2: Unverified claims | 3-4: Industry sources | 5-6: Some independent data | 7-8: Peer-reviewed | 9-10: Multiple confirmations
   - Peer-reviewed > government data > third-party audits > industry reports > press releases

STEP 3: Metadata

**Content Type**: breakthrough_research | technology_deployment | policy_action | market_signal | impact_measurement | greenwashing | transition_delay

**Innovation Stage**: basic_research | applied_research | pilot | commercial | scaling

**Investment Signals** (true/false): has_funding, has_patents, has_customers, has_metrics, has_peer_review, has_deployment

**Verification Indicators** (true/false): owid_indicator, ipcc_alignment, iea_data, third_party_verified, regulatory_approved

STEP 4: Output JSON

{{
  "content_type": "<type>",
  "innovation_stage": "<stage>",
  "climate_impact_potential": <0-10>,
  "technical_credibility": <0-10>,
  "economic_viability": <0-10>,
  "deployment_readiness": <0-10>,
  "systemic_impact": <0-10>,
  "justice_equity": <0-10>,
  "innovation_quality": <0-10>,
  "evidence_strength": <0-10>,
  "investment_signals": {{
    "has_funding": <bool>,
    "has_patents": <bool>,
    "has_customers": <bool>,
    "has_metrics": <bool>,
    "has_peer_review": <bool>,
    "has_deployment": <bool>
  }},
  "verification_indicators": {{
    "owid_indicator": <bool>,
    "ipcc_alignment": <bool>,
    "iea_data": <bool>,
    "third_party_verified": <bool>,
    "regulatory_approved": <bool>
  }},
  "flags": {{
    "greenwashing_risk": <bool>,
    "vaporware_risk": <bool>,
    "fossil_transition": <bool>
  }},
  "reasoning": "<2-3 sentences: what concrete action, what evidence, what stage>",
  "key_impact_metrics": ["<metric1 with number>", "<metric2>"],
  "technology_tags": ["<tech1>", "<tech2>"],
  "sdg_alignment": [<SDG numbers 1-17>]
}}

CRITICAL REMINDERS:
- Focus on DEPLOYED TECHNOLOGY and MEASURED OUTCOMES, not promises
- Greenwashing = commitments without action (score LOW)
- Vaporware = announcements without deployments (score LOW)
- Fossil "bridge" = delay tactics (score LOW)
- Technical credibility gatekeeps everything (if <5, cap at 4)
- Quantified metrics > vague claims
- Independent verification > corporate press releases

VALIDATION EXAMPLES:

HIGH SCORE (8.0/10):
Article: "Tesla Megapack battery storage facility in Texas reaches 1 GWh capacity, storing enough renewable energy to power 250,000 homes during peak demand. Facility achieved 90% round-trip efficiency in first year, exceeding design specs."
Scores: Climate=8, Technical=8, Economic=7, Deployment=9, Systemic=8, Justice=5, Innovation=6, Evidence=7
Reasoning: "Operational grid-scale storage enabling renewable integration with verified performance data. Technology at TRL 9 (mass deployment) with proven economics and measured efficiency."

LOW SCORE (1.8/10):
Article: "Oil major commits to net-zero by 2050 through carbon offset investments and future technology breakthroughs. Company maintains current production while pledging 5% of capex to renewable energy over next decade."
Scores: Climate=2, Technical=2, Economic=3, Deployment=1, Systemic=2, Justice=2, Innovation=1, Evidence=2
Flags: greenwashing_risk, vaporware_risk, fossil_transition
Reasoning: "Corporate commitment without concrete action. Offsets don't cancel new extraction, no specific technology deployed, relies on undefined 'future breakthroughs.' Greenwashing penalty applied."
```

---

## SCORING FORMULA (Applied post-labeling)

```python
sustainability_score = (
    climate_impact_potential * 0.30 +
    technical_credibility * 0.20 +
    systemic_impact * 0.20 +
    evidence_strength * 0.15 +
    justice_equity * 0.10 +
    innovation_quality * 0.05
)

# Apply gatekeeper
if technical_credibility < 5:
    sustainability_score = min(sustainability_score, 4.0)

# Apply flags
if greenwashing_risk: sustainability_score *= 0.6
if vaporware_risk: sustainability_score *= 0.7
if fossil_transition: sustainability_score = min(sustainability_score, 4.0)
```

```python
investment_readiness_score = (
    deployment_readiness * 0.30 +
    economic_viability * 0.25 +
    technical_credibility * 0.20 +
    climate_impact_potential * 0.15 +
    innovation_quality * 0.10
)

# Bonus for investment signals (up to +2.0)
bonus = sum([
    0.4 if has_funding else 0,
    0.3 if has_patents else 0,
    0.5 if has_customers else 0,
    0.4 if has_metrics else 0,
    0.2 if has_peer_review else 0,
    0.2 if has_deployment else 0
])
investment_readiness_score = min(10, investment_readiness_score + bonus)

if vaporware_risk: investment_readiness_score = min(investment_readiness_score, 4.0)
```

---
