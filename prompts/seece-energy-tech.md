# SEECE Energy Tech Intelligence Filter

**Purpose**: Identify actionable energy technology intelligence for SEECE (Sustainable Energy & Environmental Change Engineering) research center at HAN University of Applied Sciences.

**Version**: 1.0
**Target LLM**: Claude 3.5 Sonnet / Gemini 1.5 Pro
**Use Case**: Generate ground truth labels for fine-tuning local models
**Parent Filter**: Extends `sustainability.md` with SEECE-specific dimensions

---

## SEECE CONTEXT

**What is SEECE?**
- Research center at HAN University of Applied Sciences (Hogeschool van Arnhem en Nijmegen)
- Focus: **Applied research** in sustainable energy and environmental technology
- Key domains: Hydrogen energy, grid integration, mobility electrification, building efficiency
- Geographic focus: Netherlands, EU, with global technology monitoring
- Sweet spot: **TRL 4-7** (applied research, pilots, early deployment)

**SEECE's Needs:**
- Track technology maturation (lab → pilot → deployment)
- Identify industry collaboration opportunities
- Monitor Dutch/EU policy and funding
- Discover student thesis topics and educational opportunities
- Filter out greenwashing and vaporware
- Focus on technologies approaching market readiness

---

## PRE-FILTER (BEFORE LLM LABELING)

Only analyze articles where:

**Source Categories:**
- `energy_utilities`, `climate_solutions`, `dutch_energy`, `automotive_transport`, `eu_policy`, `eu_digital_policy`, `science`, `biotech_pharma` (for hydrogen biology), `industry_intelligence`, `semiconductor_hardware` (for power electronics)

**AND/OR Keywords Present:**
- **Hydrogen**: hydrogen, fuel cell, electrolysis, green hydrogen, hydrogen storage, H2, PEMFC, SOFC, hydrogen infrastructure
- **Grid/Storage**: battery storage, grid storage, energy storage, smart grid, grid integration, demand response, V2G, vehicle-to-grid, microgrid, grid flexibility, BESS
- **Mobility**: electric vehicle, EV, charging infrastructure, fast charging, battery electric, PHEV, e-mobility, charging station, battery management
- **Building**: heat pump, building efficiency, insulation, HVAC, smart building, energy performance, building automation, district heating
- **Industrial**: industrial heat, process electrification, industrial decarbonization, high-temperature heat, electric furnace, industrial energy
- **Renewable Integration**: solar integration, wind integration, renewable curtailment, power-to-X, sector coupling
- **Power Electronics**: inverter, converter, power electronics, SiC, GaN, wide bandgap

**AND Innovation Stage:**
- Prefer: `applied_research`, `pilot`, `commercial` (TRL 4-8)
- Include but lower priority: `basic_research` (TRL 1-3) IF breakthrough
- Include but flag: `scaling` (TRL 9) IF novel deployment model

**Geographic Relevance:**
- **High priority**: Netherlands, Dutch, Holland, EU, European Union, Germany (neighbor), Belgium (neighbor)
- **Medium priority**: Denmark (wind leader), Norway (hydro/hydrogen), UK, France
- **Include if breakthrough**: Global (USA, China, etc.) but note "international-benchmark" flag

---

## PROMPT TEMPLATE

```
You are an energy technology analyst for SEECE, a Dutch applied research center focused on sustainable energy. Analyze this article for energy technology intelligence based on CONCRETE DEPLOYMENTS, MEASURED OUTCOMES, and ACTIONABLE INSIGHTS for applied research.

ARTICLE:
Title: {title}
Source: {source}
Published: {published_date}
Text: {text}

STEP 1: Apply Standard Sustainability Analysis

Follow the complete analysis from `sustainability.md`:
- Pre-classification Filters (Greenwashing, Vaporware, Fossil Fuel Transition, Announcement vs. Deployment)
- Evaluate 8 Sustainability Dimensions (0-10 each)
- Determine content_type and innovation_stage
- Identify investment_signals and verification_indicators
- Apply flags (greenwashing_risk, vaporware_risk, fossil_transition)

(See parent prompt for full details)

STEP 2: SEECE-Specific Dimensions (score 0-10 for each)

1. **Dutch/EU Policy Relevance**:
   - Does this relate to Dutch Climate Agreement, EU Green Deal, REPowerEU, Fit for 55?
   - Are there Dutch/EU funding opportunities (Horizon Europe, NWO, RVO subsidies)?
   - Does it affect Dutch energy targets (70% renewable by 2030, hydrogen economy goals)?
   - Is there regulatory alignment (EU taxonomy, CBAM, energy performance standards)?
   - (0: no relevance, 3: general EU trend, 5: specific policy mention, 7: Dutch policy impact, 10: urgent Dutch/EU priority)

2. **Industry Collaboration Potential**:
   - Are companies mentioned that SEECE could partner with?
   - Are there pilot projects SEECE could join or contribute to?
   - Is technology at a stage where HAN researchers/students could add value?
   - Are there regional clusters mentioned (Arnhem-Nijmegen, Rotterdam port, Groningen hydrogen)?
   - (0: no opportunities, 3: general industry, 5: named companies, 7: regional players, 10: explicit collaboration call)

3. **Educational Value**:
   - Could this be integrated into HAN energy engineering curriculum?
   - Are there hands-on learning opportunities (lab setups, field visits)?
   - Student thesis topics (bachelor/master applied research)?
   - Internship possibilities at mentioned organizations?
   - (0: purely commercial, 3: general knowledge, 5: teaching examples, 7: thesis topics, 10: structured learning program)

4. **Applied Research Fit**:
   - Is this at the right TRL for SEECE (4-7: prototype → pilot → early deployment)?
   - Can HAN contribute with applied research (testing, optimization, deployment studies)?
   - Are there measurement/validation opportunities?
   - Is there a clear path from research to implementation?
   - (0: too basic or too mature, 3: adjacent fit, 5: good fit, 7: ideal TRL, 10: perfect applied research opportunity)

5. **Regional Impact Potential**:
   - Could this be deployed in Gelderland/Netherlands?
   - Does it address Dutch grid challenges (congestion, renewable integration)?
   - Relevant to Dutch industry (logistics, agriculture, manufacturing)?
   - Could it create regional jobs/economic activity?
   - (0: not applicable to NL, 3: general relevance, 5: Dutch context mentioned, 7: regional deployment likely, 10: addresses urgent Dutch need)

STEP 3: SEECE Priority Topic Detection

Identify if article is about SEECE's priority domains (mark true/false for each):

**Priority Topics:**
- **hydrogen_energy**: Green hydrogen production, fuel cells, H2 storage, hydrogen infrastructure
- **grid_integration**: Battery storage, smart grids, V2G, demand response, grid flexibility
- **mobility_electrification**: EV charging, battery technology, electric transport
- **building_efficiency**: Heat pumps, building automation, insulation, HVAC
- **industrial_decarbonization**: Industrial heat, process electrification, industrial energy
- **renewable_integration**: Solar/wind integration, curtailment solutions, power-to-X

**Cross-Cutting Topics (bonus points):**
- **power_electronics**: Inverters, converters, SiC/GaN semiconductors
- **sector_coupling**: Integrated energy systems, multi-carrier networks
- **energy_data**: Smart meters, energy management systems, grid data analytics
- **circular_economy**: Battery recycling, material recovery, lifecycle optimization

STEP 4: Partnership Intelligence

Extract actionable partnership information:

**Organizations Mentioned** (list each with type):
- Companies (name, country, what they do)
- Research institutions (name, country, collaboration type)
- Government bodies (name, country, program/funding)
- Industry consortia (name, focus area)

**Deployment Locations:**
- Specific cities/regions mentioned (especially in Netherlands/EU)
- Pilot site details (capacity, duration, partners)
- Planned expansions or replication opportunities

**Student/Educational Opportunities:**
- Internship possibilities at mentioned organizations
- Potential thesis topics based on article content
- Hands-on learning opportunities (lab equipment, field visits)
- Industry challenges that students could address

STEP 5: SEECE-Adjusted Scoring

Apply SEECE priority boosts to sustainability dimensions:

**Boost Climate Impact if article is about:**
- Hydrogen energy systems: +2 to climate_impact_potential
- Grid storage/integration: +2 to systemic_impact
- Mobility electrification: +1.5 to climate_impact_potential
- Building efficiency: +1 to climate_impact_potential
- Industrial heat/process: +1.5 to innovation_quality

**Boost Technical Credibility if:**
- Independent testing data from recognized labs (TNO, ECN, Fraunhofer, NREL)
- Field deployment data from Dutch/EU projects
- Peer-reviewed validation from TU Delft, TU Eindhoven, other Dutch universities

**Boost Deployment Readiness if:**
- TRL 4-7 (applied research sweet spot): +1
- Operational pilots in Netherlands/EU: +1.5
- Dutch/EU regulatory approval: +1

**Penalize if:**
- TRL 1-3 without clear path to application: -1 to applied_research_fit
- TRL 9 (mass deployment) without novel aspects: -1 to innovation_quality
- Non-EU context without transferability: -1 to regional_impact_potential

STEP 6: Calculate SEECE Relevance Score

Composite score for SEECE-specific relevance (0-10):

```
seece_relevance_score = (
    dutch_eu_policy_relevance * 0.25 +
    applied_research_fit * 0.25 +
    industry_collaboration_potential * 0.20 +
    regional_impact_potential * 0.15 +
    educational_value * 0.15
)

# Boost for priority topics
if any priority_topic == true:
    seece_relevance_score += 1.0

# Bonus for multiple priority topics
priority_count = sum([hydrogen_energy, grid_integration, mobility_electrification, ...])
if priority_count >= 2:
    seece_relevance_score += 0.5

# Cap at 10
seece_relevance_score = min(10, seece_relevance_score)
```

STEP 7: Generate SEECE Intelligence Summary

Create a 3-4 sentence executive summary specifically for SEECE researchers:
- What is the core technology/development?
- Why is it relevant to SEECE's research domains?
- What is the opportunity (collaboration, research, education)?
- What is the maturity/timeline?

Example: "Hydrogen fuel cell manufacturer Nedstack (Netherlands) reports 30% cost reduction in maritime fuel cell stacks through automated manufacturing. Technology now at TRL 7 with 2MW pilot operating in Rotterdam port. Potential SEECE collaboration on stack testing and optimization, plus student thesis opportunities in manufacturing process improvement. Addresses Dutch maritime decarbonization goals."

STEP 8: Provide Complete Response

Respond with ONLY valid JSON in this exact format:

{{
  // ===== STANDARD SUSTAINABILITY DIMENSIONS (from parent prompt) =====
  "content_type": "breakthrough_research|technology_deployment|policy_action|market_signal|impact_measurement|greenwashing|transition_delay",
  "innovation_stage": "basic_research|applied_research|pilot|commercial|scaling",

  "climate_impact_potential": <score 0-10>,
  "technical_credibility": <score 0-10>,
  "economic_viability": <score 0-10>,
  "deployment_readiness": <score 0-10>,
  "systemic_impact": <score 0-10>,
  "justice_equity": <score 0-10>,
  "innovation_quality": <score 0-10>,
  "evidence_strength": <score 0-10>,

  "investment_signals": {{
    "has_funding": <true|false>,
    "has_patents": <true|false>,
    "has_customers": <true|false>,
    "has_metrics": <true|false>,
    "has_peer_review": <true|false>,
    "has_deployment": <true|false>
  }},

  "verification_indicators": {{
    "owid_indicator": <true|false>,
    "ipcc_alignment": <true|false>,
    "iea_data": <true|false>,
    "third_party_verified": <true|false>,
    "regulatory_approved": <true|false>
  }},

  "flags": {{
    "greenwashing_risk": <true|false>,
    "vaporware_risk": <true|false>,
    "fossil_transition": <true|false>
  }},

  // ===== SEECE-SPECIFIC DIMENSIONS =====
  "seece_dimensions": {{
    "dutch_eu_policy_relevance": <score 0-10>,
    "industry_collaboration_potential": <score 0-10>,
    "educational_value": <score 0-10>,
    "applied_research_fit": <score 0-10>,
    "regional_impact_potential": <score 0-10>
  }},

  "seece_relevance_score": <composite score 0-10>,

  "priority_topics": {{
    "hydrogen_energy": <true|false>,
    "grid_integration": <true|false>,
    "mobility_electrification": <true|false>,
    "building_efficiency": <true|false>,
    "industrial_decarbonization": <true|false>,
    "renewable_integration": <true|false>
  }},

  "cross_cutting_topics": {{
    "power_electronics": <true|false>,
    "sector_coupling": <true|false>,
    "energy_data": <true|false>,
    "circular_economy": <true|false>
  }},

  "geographic_context": {{
    "dutch_context": <true|false>,
    "eu_context": <true|false>,
    "regional_locations": ["<city/region>", ...],
    "transferability_to_nl": "high|medium|low|none"
  }},

  "partnership_intelligence": {{
    "companies": [
      {{"name": "<company>", "country": "<country>", "role": "<what they do>"}},
      ...
    ],
    "research_institutions": [
      {{"name": "<institution>", "country": "<country>", "collaboration_type": "<type>"}},
      ...
    ],
    "government_programs": [
      {{"name": "<program>", "country": "<country>", "funding_type": "<type>"}},
      ...
    ],
    "deployment_sites": [
      {{"location": "<city/region>", "capacity": "<MW/units>", "status": "<operational|planned>"}},
      ...
    ]
  }},

  "educational_opportunities": {{
    "thesis_topics": ["<topic 1>", "<topic 2>", ...],
    "internship_potential": <true|false>,
    "curriculum_integration": "<brief description or null>",
    "hands_on_opportunities": "<description or null>"
  }},

  "seece_intelligence_summary": "<3-4 sentence executive summary for SEECE researchers>",

  "reasoning": "<2-3 sentences explaining: core technology, SEECE relevance, maturity stage>",
  "key_impact_metrics": ["<metric1 with number>", "<metric2 with number>"],
  "technology_tags": ["<specific_tech1>", "<specific_tech2>"],
  "sdg_alignment": [<SDG numbers 1-17 that apply>]
}}

CRITICAL REMINDERS FOR SEECE:
- Prioritize **TRL 4-7** (applied research, pilots, early deployment)
- Focus on **actionable intelligence** (collaboration, research, education opportunities)
- Flag **Dutch/EU context** explicitly - this is high priority
- Identify **specific organizations** SEECE could contact
- Distinguish **deployed technology** from **announcements** - SEECE needs proven tech
- **Greenwashing detection** is critical - don't waste researchers' time on hype
- **Regional relevance** matters - can this work in Netherlands?
- Think **applied research** not pure science - SEECE bridges research and industry

DO NOT include any text outside the JSON object.
```

---

## SEECE SCORING WEIGHTS

### SEECE Relevance Score (0-10)
Primary filter for "Is this worth SEECE's attention?"

```python
seece_relevance_score = (
    dutch_eu_policy_relevance * 0.25 +      # Critical: Dutch/EU funding and policy
    applied_research_fit * 0.25 +            # Critical: Right TRL for SEECE
    industry_collaboration_potential * 0.20 + # Important: Real-world partnerships
    regional_impact_potential * 0.15 +       # Important: Can deploy in NL?
    educational_value * 0.15                 # Important: Student opportunities
)

# Priority topic boost
priority_count = sum([
    hydrogen_energy,
    grid_integration,
    mobility_electrification,
    building_efficiency,
    industrial_decarbonization,
    renewable_integration
])

if priority_count >= 1:
    seece_relevance_score += 1.0
if priority_count >= 2:
    seece_relevance_score += 0.5

# Cap at 10
seece_relevance_score = min(10, seece_relevance_score)
```

### Combined Intelligence Score (0-10)
Combines sustainability quality with SEECE relevance:

```python
# Get sustainability_score from parent prompt
sustainability_score = calculate_sustainability_score()  # From sustainability.md

# Combine with SEECE relevance
intelligence_score = (
    sustainability_score * 0.60 +      # Quality of the technology/development
    seece_relevance_score * 0.40       # SEECE-specific actionability
)

# Must pass both bars
if sustainability_score < 5.0 and seece_relevance_score < 6.0:
    intelligence_score = min(intelligence_score, 4.0)  # Low confidence

# Priority topic in Dutch context gets bonus
if (dutch_context and any_priority_topic):
    intelligence_score = min(10, intelligence_score + 1.0)
```

---

## EXPECTED SCORE DISTRIBUTIONS

### SEECE Dimension Distributions
- **dutch_eu_policy_relevance**: Bimodal (0-2 for non-EU, 6-9 for EU policy)
- **applied_research_fit**: Normal, mean ~5.5 (sweet spot TRL 4-7)
- **industry_collaboration_potential**: Right-skewed, mean ~4.5 (few clear opportunities)
- **regional_impact_potential**: Left-skewed, mean ~6.0 (most energy tech applicable to NL)
- **educational_value**: Uniform, mean ~5.0 (varies widely)

### Priority Topic Distribution (expected in energy content)
- **hydrogen_energy**: 15-25%
- **grid_integration**: 20-30%
- **mobility_electrification**: 15-20%
- **building_efficiency**: 10-15%
- **industrial_decarbonization**: 5-10%
- **renewable_integration**: 25-35%

### SEECE Relevance Score Distribution
- **High relevance (7-10)**: 20-30% of energy content
- **Medium relevance (4-6)**: 40-50%
- **Low relevance (0-3)**: 20-30%
- **Target**: Surface top 20-30% for researcher review

---

## VALIDATION EXAMPLES

### Example 1: High SEECE Relevance (9.2/10)

**Article**: "Dutch startup Battolyser Systems (Schiedam) demonstrates combined battery-electrolyser at 1 MW scale in Rotterdam port. System stores renewable electricity as both battery power and green hydrogen, achieving 85% round-trip efficiency. Funded by Horizon Europe, now seeking partners for 10 MW pilot. TU Delft validation shows 1,000-cycle durability."

**Sustainability Scores**:
- Climate Impact: 8 (enables renewable integration + green H2)
- Technical Credibility: 8 (TU Delft validated, operational data)
- Economic Viability: 6 (pilot stage, cost trajectory promising)
- Deployment Readiness: 6 (TRL 6-7, first MW-scale demo)
- Systemic Impact: 9 (combines grid storage + hydrogen production)
- Evidence Strength: 8 (independent validation, operational data)
→ **Sustainability Score: 7.8**

**SEECE Dimensions**:
- Dutch/EU Policy Relevance: 10 (Dutch company, Horizon funded, Rotterdam port)
- Applied Research Fit: 9 (TRL 6-7, perfect for SEECE)
- Industry Collaboration: 9 (explicit partnership call, regional)
- Regional Impact: 10 (Netherlands, port energy transition)
- Educational Value: 8 (thesis topics, internships possible)
→ **SEECE Relevance Score: 9.2**

**Priority Topics**: hydrogen_energy, grid_integration, sector_coupling
**Geographic**: Dutch (Schiedam, Rotterdam), EU (Horizon Europe)
**Partnership Intelligence**:
- Company: Battolyser Systems, Netherlands, combined battery-electrolyser
- Institution: TU Delft, Netherlands, technical validation
- Program: Horizon Europe, EU, research funding
- Site: Rotterdam port, 1 MW operational, 10 MW planned

**SEECE Intelligence Summary**: "Battolyser Systems demonstrates 1 MW combined battery-electrolyser in Rotterdam port with TU Delft validation, achieving 85% efficiency. At TRL 6-7 seeking 10 MW pilot partners. Ideal SEECE opportunity: Dutch company, Horizon funded, combines two SEECE priority topics (hydrogen + grid storage), clear collaboration pathway. Student thesis potential in system optimization and grid integration strategies."

**Intelligence Score: 8.4** (0.6 × 7.8 + 0.4 × 9.2 + 1.0 Dutch priority bonus)

---

### Example 2: Medium SEECE Relevance (5.1/10)

**Article**: "Tesla announces new 'Megapack 2' battery storage system with 40% cost reduction. Company plans installations across US and China. No technical specifications disclosed beyond marketing claims of 'industry-leading efficiency.'"

**Sustainability Scores**:
- Climate Impact: 6 (grid storage enables renewables)
- Technical Credibility: 3 (no independent verification, marketing claims)
- Economic Viability: 7 (Tesla has deployment track record)
- Deployment Readiness: 8 (Tesla already deploys at scale)
- Systemic Impact: 7 (grid storage is important)
- Evidence Strength: 2 (press release only, no data)
→ **Sustainability Score: 4.8** (capped by low credibility)

**SEECE Dimensions**:
- Dutch/EU Policy Relevance: 2 (no EU mention, US/China focus)
- Applied Research Fit: 3 (TRL 9, too mature for applied research)
- Industry Collaboration: 3 (no partnership opportunities mentioned)
- Regional Impact: 5 (grid storage generally applicable, but no EU plans)
- Educational Value: 4 (generic case study, no hands-on access)
→ **SEECE Relevance Score: 3.4**

**Priority Topics**: grid_integration
**Geographic**: US, China (not EU)
**Flags**: vaporware_risk (announcement without technical data)

**SEECE Intelligence Summary**: "Tesla announces Megapack 2 with claimed 40% cost reduction but no technical validation or EU deployment plans. Low SEECE priority: insufficient technical data for research, no regional relevance, technology already mature (TRL 9). Monitor for future EU installations with disclosed performance data."

**Intelligence Score: 4.2** (0.6 × 4.8 + 0.4 × 3.4)

---

### Example 3: Low SEECE Relevance Despite High Sustainability (3.8/10)

**Article**: "IPCC report highlights critical role of energy efficiency in limiting warming to 1.5°C. Meta-analysis of 500 studies shows building retrofits could reduce global emissions by 8-10% by 2030. Calls for urgent policy action and increased funding."

**Sustainability Scores**:
- Climate Impact: 9 (enormous potential, well-documented)
- Technical Credibility: 10 (IPCC, 500 studies)
- Systemic Impact: 9 (economy-wide)
- Evidence Strength: 10 (IPCC = gold standard)
→ **Sustainability Score: 9.2** (very high)

**SEECE Dimensions**:
- Dutch/EU Policy Relevance: 5 (general policy, not Dutch-specific)
- Applied Research Fit: 2 (TRL varies, not specific technology)
- Industry Collaboration: 1 (no specific opportunities)
- Regional Impact: 6 (building efficiency relevant to NL)
- Educational Value: 7 (good teaching material, case studies)
→ **SEECE Relevance Score: 4.2**

**Priority Topics**: building_efficiency
**Content Type**: impact_measurement (IPCC meta-analysis)

**SEECE Intelligence Summary**: "IPCC meta-analysis confirms 8-10% emission reduction potential from building retrofits globally. High-quality evidence but low SEECE actionability: no specific technologies, no collaboration opportunities, no applied research angles. Better suited for policy researchers than SEECE's applied technology focus. Use as teaching material for building efficiency courses."

**Intelligence Score: 6.5** (0.6 × 9.2 + 0.4 × 4.2)
**Note**: High sustainability score but moderate intelligence score due to lack of actionable research opportunities.

---

## SEECE PRIORITY USE CASES

### 1. Weekly Intelligence Digest
**Filter**: `seece_relevance_score >= 7.0` AND `intelligence_score >= 6.5`
**Output**: Top 10-15 articles per week with full SEECE analysis
**Audience**: SEECE research leads, lectoraat coordinators

### 2. Hydrogen Technology Radar
**Filter**: `hydrogen_energy == true` AND `seece_relevance_score >= 6.0`
**Output**: Focused hydrogen intelligence, partnership opportunities
**Audience**: Hydrogen researchers, industry liaison

### 3. Partnership Opportunity Tracker
**Filter**: `industry_collaboration_potential >= 7.0` AND `dutch_context == true`
**Output**: Companies, projects, funding opportunities in Netherlands
**Audience**: Business development, project acquisition

### 4. Student Thesis Topic Generator
**Filter**: `educational_value >= 7.0` AND `applied_research_fit >= 6.0`
**Output**: Curated thesis topics with industry context
**Audience**: Student supervisors, thesis coordinators

### 5. Policy & Funding Monitor
**Filter**: `dutch_eu_policy_relevance >= 8.0` OR `government_programs.length > 0`
**Output**: Funding calls, policy changes, regulatory updates
**Audience**: Grant writers, lectoraat management

---

## INTEGRATION WITH CONTENT AGGREGATOR

### Pre-Filter Configuration

Add to `config/app.yaml`:

```yaml
content_filters:
  seece_energy_tech:
    enabled: true

    source_categories:
      - energy_utilities
      - climate_solutions
      - dutch_energy
      - automotive_transport
      - eu_policy
      - science
      - industry_intelligence
      - semiconductor_hardware

    required_keywords:
      hydrogen: ["hydrogen", "fuel cell", "electrolysis", "green hydrogen", "H2"]
      grid: ["battery storage", "grid storage", "smart grid", "V2G", "microgrid"]
      mobility: ["electric vehicle", "EV", "charging", "e-mobility"]
      building: ["heat pump", "building efficiency", "HVAC"]
      industrial: ["industrial heat", "process electrification"]
      renewables: ["solar integration", "wind integration", "power-to-X"]
      power_electronics: ["inverter", "converter", "SiC", "GaN"]

    geographic_priority:
      high: ["Netherlands", "Dutch", "Holland", "Nederland"]
      medium: ["EU", "Europe", "Germany", "Belgium", "Denmark"]
      low: ["global", "international"]

    trl_preference:
      optimal: [4, 5, 6, 7]  # Applied research, pilots, early commercial
      acceptable: [3, 8]      # Late basic research, commercial
      flag: [1, 2, 9]         # Too early or too mature

    output:
      format: json
      destination: data/seece_filtered/
      batch_size: 100
      max_per_run: 500
```

### Batch Processing Workflow

```python
# Pseudo-code for SEECE intelligence pipeline

def process_seece_intelligence(content_items):
    """
    Process content items through SEECE filter
    """
    # Step 1: Pre-filter
    seece_candidates = apply_seece_prefilter(content_items)
    # Expect ~15-25% of total items to pass pre-filter

    # Step 2: LLM Labeling (expensive step)
    labeled_items = []
    for item in seece_candidates:
        # Use sustainability.md FIRST (comprehensive analysis)
        sustainability_scores = llm_label(item, prompt="sustainability.md")

        # Then add SEECE dimensions (extends sustainability)
        seece_scores = llm_label(item, prompt="seece-energy-tech.md",
                                  base_scores=sustainability_scores)

        labeled_items.append({
            **item,
            **sustainability_scores,
            **seece_scores
        })

    # Step 3: Rank by intelligence score
    ranked = sorted(labeled_items,
                    key=lambda x: x['intelligence_score'],
                    reverse=True)

    # Step 4: Generate intelligence products
    weekly_digest = generate_digest(
        top_n(ranked, n=15, min_score=7.0)
    )

    hydrogen_radar = generate_radar(
        filter_by(ranked, priority_topics={'hydrogen_energy': True})
    )

    partnership_opportunities = generate_opportunities(
        filter_by(ranked,
                  industry_collaboration_potential__gte=7.0,
                  dutch_context=True)
    )

    return {
        'weekly_digest': weekly_digest,
        'hydrogen_radar': hydrogen_radar,
        'partnership_opportunities': partnership_opportunities,
        'all_labeled': labeled_items
    }
```

---

## ETHICAL CONSIDERATIONS

### SEECE-Specific Biases to Monitor

1. **Dutch/EU Bias**: Don't ignore global breakthroughs just because they're not European
   - Include international benchmarks
   - Note transferability to Dutch context
   - Learn from China, US, Japan innovations

2. **TRL Sweet Spot Bias**: Don't completely ignore TRL 1-3 or TRL 9
   - Breakthrough basic research (TRL 1-3) can inform future SEECE projects
   - Mature deployments (TRL 9) show what works at scale
   - Just flag and prioritize differently

3. **Academic vs. Industry Balance**: SEECE bridges both worlds
   - Don't over-weight academic papers (many aren't actionable)
   - Don't over-weight industry (often greenwashing)
   - Sweet spot: academic validation + industry deployment

4. **Technology Neutrality**: Don't favor SEECE's current expertise
   - If building efficiency breakthroughs emerge, include them
   - If new domains gain importance (e.g., green materials), adapt
   - Let evidence guide priorities, not institutional inertia

### Quality Assurance

**Validation Checks**:
- If `dutch_context == true`, verify Netherlands actually mentioned in text
- If `applied_research_fit > 7`, confirm TRL 4-7 explicitly stated or clearly implied
- If `industry_collaboration_potential > 7`, confirm specific companies/organizations named
- If `educational_value > 7`, confirm concrete educational opportunities described

**Consistency Checks**:
- If `seece_relevance_score > 8` but `sustainability_score < 5`, flag for review (possible false positive)
- If `hydrogen_energy == true` but text doesn't mention hydrogen/H2/fuel cell, flag error
- If `dutch_context == true` but `dutch_eu_policy_relevance < 5`, investigate discrepancy
- If `partnership_intelligence.companies` is empty but `industry_collaboration_potential > 6`, flag inconsistency

---

## SUCCESS METRICS

### How to Measure SEECE Filter Performance

**Precision (Relevance)**:
- % of high-scored articles (intelligence_score > 7) that SEECE researchers find valuable
- Target: >80% precision on top 20 articles per week

**Recall (Coverage)**:
- % of important energy tech developments captured by filter
- Measure via: researcher feedback "Did we miss anything important this week?"
- Target: >90% recall on SEECE priority topics

**Actionability**:
- % of surfaced articles that lead to:
  - New industry contacts (target: 2-3/month)
  - Student thesis topics (target: 5-10/semester)
  - Research proposals (target: 1-2/quarter)
  - Curriculum updates (target: 2-3/year)

**Time Savings**:
- Researcher time saved vs. manual scanning
- Baseline: ~5-10 hours/week per researcher
- Target: 80% reduction → 1-2 hours/week

**Early Warning**:
- Days ahead of competitors in identifying breakthroughs
- Target: 7-14 days earlier awareness of relevant developments

---

## FUTURE ENHANCEMENTS

### Phase 2 (After Initial Validation)

1. **Automated Partner Matching**
   - Match SEECE capabilities to company needs mentioned in articles
   - Generate "warm introduction" templates
   - Track partnership success rate

2. **Technology Maturation Tracking**
   - Track same technology over time (TRL progression)
   - Alert when technologies reach SEECE sweet spot
   - Identify acceleration/stagnation patterns

3. **Competitive Intelligence**
   - Track what other applied research centers are doing
   - Identify white spaces (underserved research areas)
   - Monitor HAN competitors (Saxion, Windesheim, Fontys)

4. **Student Matching**
   - Match student interests/skills to thesis opportunities
   - Auto-generate thesis proposal templates
   - Track student-company placements

5. **Grant Alignment**
   - Auto-match developments to funding calls
   - Generate proposal section drafts
   - Track funding success rate

---

**This SEECE-specific filter transforms your content aggregator from a general sustainability feed into actionable energy technology intelligence for applied research.**
