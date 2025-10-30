# SEECE Energy Tech Intelligence Filter

**Purpose**: Actionable energy technology intelligence for SEECE (HAN University applied research center).

**Version**: 1.0-compressed
**Target**: Gemini Flash 1.5 / Claude Haiku / Fast models
**Extends**: sustainability.md with SEECE-specific dimensions

**SEECE Focus**: TRL 4-7 (applied research, pilots, early deployment), Dutch/EU priority, industry collaboration opportunities.

---

## PROMPT

```
Analyze this article for energy technology intelligence based on CONCRETE DEPLOYMENTS, MEASURED OUTCOMES, and ACTIONABLE INSIGHTS for applied research.

ARTICLE:
Title: {title}
Source: {source}
Published: {published_date}
Text: {text}

STEP 1: Apply Sustainability Pre-filters

A) GREENWASHING: Corporate reports, pledges, offsets, ESG ratings without verification/data/deployment?
   → FLAG "greenwashing_risk"

B) VAPORWARE: Announcements without deployed units/contracts/operational data?
   → FLAG "vaporware_risk"

C) FOSSIL TRANSITION: "Clean coal", "bridge fuel", CCS for EOR, fossil H2 without lifecycle data?
   → FLAG "fossil_transition"

STEP 2: Score Sustainability Dimensions (0-10)

1. **Climate Impact Potential**: Quantified GHG reduction? Lifecycle accounting?
   0-2: Minimal | 3-4: Modest | 5-6: Significant | 7-8: Transformative | 9-10: Breakthrough

2. **Technical Credibility** (GATEKEEPER): Independent validation? Realistic metrics?
   0-2: Unverified | 3-4: Industry | 5-6: Some independent | 7-8: Peer-reviewed | 9-10: Multiple confirmations

3. **Economic Viability**: Path to cost-competitiveness?
   0-2: No path | 3-4: Needs subsidies | 5-6: Approaching parity | 7-8: Competitive | 9-10: Cheaper than fossil

4. **Deployment Readiness**: TRL? Operational units?
   0-2: Concept/lab | 3-4: Pilot | 5-6: First commercial | 7-8: Proven at scale | 9-10: Mass deployment

5. **Systemic Impact**: Enables broader decarbonization? Gigaton potential?
   0-2: Niche | 3-4: Sectoral | 5-6: Cross-sectoral | 7-8: Economy-wide | 9-10: Global infrastructure

6. **Justice & Equity**: Avoids harm to frontline communities? Equitable access?
   0: Harmful | 3: Neutral | 5: Some consideration | 7: Equity-centered | 10: Reparative

7. **Innovation Quality**: Breakthrough or incremental? Solves unsolved problem?
   0-2: Hype | 3-4: Incremental | 5-6: Significant | 7-8: Breakthrough | 9-10: Paradigm shift

8. **Evidence Strength**: Quality of sources?
   0-2: Unverified | 3-4: Industry | 5-6: Some independent | 7-8: Peer-reviewed | 9-10: Multiple confirmations

STEP 3: SEECE-Specific Dimensions (0-10)

1. **Dutch/EU Policy Relevance**: Relates to Dutch Climate Agreement, EU Green Deal, REPowerEU, Fit for 55? Dutch/EU funding (Horizon, NWO, RVO)? Dutch energy targets?
   0: No relevance | 3: General EU | 5: Specific policy | 7: Dutch policy impact | 10: Urgent Dutch/EU priority

2. **Industry Collaboration Potential**: Companies SEECE could partner with? Pilot projects to join? Technology at stage where HAN could add value? Regional clusters (Arnhem-Nijmegen, Rotterdam, Groningen)?
   0: No opportunities | 3: General industry | 5: Named companies | 7: Regional players | 10: Explicit collaboration call

3. **Educational Value**: Integrate into HAN curriculum? Hands-on learning? Student thesis topics? Internship possibilities?
   0: Purely commercial | 3: General knowledge | 5: Teaching examples | 7: Thesis topics | 10: Structured learning program

4. **Applied Research Fit**: Right TRL for SEECE (4-7: prototype → pilot → early deployment)? Can HAN contribute with testing/optimization/deployment studies? Clear path from research to implementation?
   0: Too basic/mature | 3: Adjacent fit | 5: Good fit | 7: Ideal TRL | 10: Perfect opportunity

5. **Regional Impact Potential**: Deploy in Gelderland/Netherlands? Addresses Dutch grid challenges? Relevant to Dutch industry? Create regional jobs?
   0: Not applicable | 3: General relevance | 5: Dutch context | 7: Regional deployment likely | 10: Urgent Dutch need

STEP 4: SEECE Priority Topics (mark true/false)

**Priority**: hydrogen_energy, grid_integration, mobility_electrification, building_efficiency, industrial_decarbonization, renewable_integration

**Cross-cutting**: power_electronics, sector_coupling, energy_data, circular_economy

STEP 5: Partnership Intelligence

**Organizations**: Companies (name, country, role), Research institutions (name, country, type), Government programs (name, country, funding), Deployment sites (location, capacity, status)

**Student Opportunities**: Thesis topics, internship potential, hands-on opportunities

STEP 6: Output JSON

{{
  "content_type": "breakthrough_research|technology_deployment|policy_action|market_signal|greenwashing",
  "innovation_stage": "basic_research|applied_research|pilot|commercial|scaling",

  "climate_impact_potential": <0-10>,
  "technical_credibility": <0-10>,
  "economic_viability": <0-10>,
  "deployment_readiness": <0-10>,
  "systemic_impact": <0-10>,
  "justice_equity": <0-10>,
  "innovation_quality": <0-10>,
  "evidence_strength": <0-10>,

  "investment_signals": {{
    "has_funding": <bool>, "has_patents": <bool>, "has_customers": <bool>,
    "has_metrics": <bool>, "has_peer_review": <bool>, "has_deployment": <bool>
  }},

  "verification_indicators": {{
    "owid_indicator": <bool>, "ipcc_alignment": <bool>, "iea_data": <bool>,
    "third_party_verified": <bool>, "regulatory_approved": <bool>
  }},

  "flags": {{
    "greenwashing_risk": <bool>, "vaporware_risk": <bool>, "fossil_transition": <bool>
  }},

  "seece_dimensions": {{
    "dutch_eu_policy_relevance": <0-10>,
    "industry_collaboration_potential": <0-10>,
    "educational_value": <0-10>,
    "applied_research_fit": <0-10>,
    "regional_impact_potential": <0-10>
  }},

  "priority_topics": {{
    "hydrogen_energy": <bool>, "grid_integration": <bool>,
    "mobility_electrification": <bool>, "building_efficiency": <bool>,
    "industrial_decarbonization": <bool>, "renewable_integration": <bool>
  }},

  "cross_cutting_topics": {{
    "power_electronics": <bool>, "sector_coupling": <bool>,
    "energy_data": <bool>, "circular_economy": <bool>
  }},

  "geographic_context": {{
    "dutch_context": <bool>, "eu_context": <bool>,
    "regional_locations": ["<city/region>"],
    "transferability_to_nl": "high|medium|low|none"
  }},

  "partnership_intelligence": {{
    "companies": [{{"name": "<company>", "country": "<country>", "role": "<role>"}}],
    "research_institutions": [{{"name": "<institution>", "country": "<country>", "collaboration_type": "<type>"}}],
    "government_programs": [{{"name": "<program>", "country": "<country>", "funding_type": "<type>"}}],
    "deployment_sites": [{{"location": "<city/region>", "capacity": "<MW/units>", "status": "<operational|planned>"}}]
  }},

  "educational_opportunities": {{
    "thesis_topics": ["<topic1>", "<topic2>"],
    "internship_potential": <bool>,
    "curriculum_integration": "<description or null>",
    "hands_on_opportunities": "<description or null>"
  }},

  "seece_intelligence_summary": "<3-4 sentence executive summary: core technology, SEECE relevance, opportunity, maturity>",
  "reasoning": "<2-3 sentences: technology, SEECE fit, maturity>",
  "key_impact_metrics": ["<metric1 with number>", "<metric2>"],
  "technology_tags": ["<tech1>", "<tech2>"],
  "sdg_alignment": [<SDG numbers>]
}}

CRITICAL REMINDERS:
- Prioritize TRL 4-7 (applied research, pilots, early deployment)
- Focus on ACTIONABLE intelligence (collaboration, research, education opportunities)
- Flag Dutch/EU context explicitly - high priority
- Identify SPECIFIC organizations SEECE could contact
- Distinguish DEPLOYED tech from announcements
- Greenwashing detection critical
- Regional relevance matters - can this work in Netherlands?
- Think APPLIED RESEARCH not pure science

VALIDATION EXAMPLES:

HIGH SEECE RELEVANCE (9.2/10):
Article: "Dutch startup Battolyser Systems (Schiedam) demonstrates combined battery-electrolyser at 1 MW scale in Rotterdam port. System stores renewable electricity as both battery power and green hydrogen, achieving 85% round-trip efficiency. Funded by Horizon Europe, now seeking partners for 10 MW pilot. TU Delft validation shows 1,000-cycle durability."

Sustainability: Climate=8, Technical=8, Economic=6, Deployment=6, Systemic=9, Evidence=8
SEECE: Dutch/EU=10, Applied Research=9, Collaboration=9, Regional=10, Educational=8
Priority Topics: hydrogen_energy, grid_integration, sector_coupling
Geographic: Dutch (Schiedam, Rotterdam), EU (Horizon Europe)
Partnership: Battolyser Systems (NL), TU Delft (NL), Rotterdam port (1MW operational, 10MW planned)
Summary: "Battolyser demonstrates 1 MW combined battery-electrolyser in Rotterdam with TU Delft validation. At TRL 6-7 seeking 10 MW pilot partners. Ideal SEECE opportunity: Dutch company, Horizon funded, combines hydrogen + grid storage, clear collaboration pathway."

MEDIUM SEECE RELEVANCE (3.4/10):
Article: "Tesla announces new 'Megapack 2' battery storage with 40% cost reduction. Plans installations across US and China. No technical specifications disclosed."

Sustainability: Climate=6, Technical=3, Economic=7, Deployment=8, Systemic=7, Evidence=2
SEECE: Dutch/EU=2, Applied Research=3, Collaboration=3, Regional=5, Educational=4
Flags: vaporware_risk (announcement without data)
Geographic: US, China (not EU)
Summary: "Tesla announcement with claimed cost reduction but no technical validation or EU deployment plans. Low SEECE priority: insufficient data for research, no regional relevance, TRL 9 (already mature). Monitor for future EU installations with disclosed performance."
```

---

## SCORING FORMULAS (Applied post-labeling)

```python
# Sustainability base score (from sustainability.md)
sustainability_score = (
    climate_impact_potential * 0.30 +
    technical_credibility * 0.20 +
    systemic_impact * 0.20 +
    evidence_strength * 0.15 +
    justice_equity * 0.10 +
    innovation_quality * 0.05
)
if technical_credibility < 5: sustainability_score = min(sustainability_score, 4.0)
if greenwashing_risk: sustainability_score *= 0.6
if fossil_transition: sustainability_score = min(sustainability_score, 4.0)

# SEECE relevance score
seece_relevance_score = (
    dutch_eu_policy_relevance * 0.25 +
    applied_research_fit * 0.25 +
    industry_collaboration_potential * 0.20 +
    regional_impact_potential * 0.15 +
    educational_value * 0.15
)

# Priority topic boost
priority_count = sum([hydrogen_energy, grid_integration, mobility_electrification, ...])
if priority_count >= 1: seece_relevance_score += 1.0
if priority_count >= 2: seece_relevance_score += 0.5
seece_relevance_score = min(10, seece_relevance_score)

# Combined intelligence score
intelligence_score = sustainability_score * 0.60 + seece_relevance_score * 0.40
if sustainability_score < 5.0 and seece_relevance_score < 6.0:
    intelligence_score = min(intelligence_score, 4.0)
if dutch_context and any_priority_topic:
    intelligence_score = min(10, intelligence_score + 1.0)
```

