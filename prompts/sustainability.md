# Sustainability Semantic Filter - Ground Truth Generation Prompt

**Purpose**: Rate content for sustainability relevance, impact potential, and credibility for climate tech investment intelligence and progress tracking.

**Version**: 1.0
**Target LLM**: Claude 3.5 Sonnet / Gemini 1.5 Pro
**Use Case**: Generate ground truth labels for fine-tuning local models

**Semantic Framework**: Focuses on DEPLOYED TECHNOLOGY and MEASURED OUTCOMES
- Distinguishes announcements from actual deployments
- Detects greenwashing (commitments without action)
- Identifies vaporware (announcements without deployments)
- Recognizes fossil fuel delay tactics

---

## PROMPT TEMPLATE

```
Analyze this article for sustainability impact based on CONCRETE ACTIONS and MEASURABLE OUTCOMES, not aspirational statements or commitments.

ARTICLE:
Title: {title}
Source: {source}
Published: {published_date}
Text: {text}

STEP 1: Pre-classification Filters

A) GREENWASHING FILTER
Is this primarily about: corporate sustainability reports, net-zero pledges, carbon offset purchases, ESG ratings, sustainability awards, or green marketing campaigns?
- If YES and article does NOT mention: third-party verification, specific emissions data, deployed technology, regulatory compliance data, supply chain changes
  → FLAG as "greenwashing_risk" (max_credibility = 3)

B) VAPORWARE FILTER
Is this about: product announcements, prototypes, pilot programs, or early-stage concepts?
- If YES and NO mention of: deployed units, customer contracts, operational data, peer-reviewed validation
  → FLAG as "vaporware" (max_investment_readiness = 4)

C) FOSSIL FUEL TRANSITION FILTER
Is this about: "clean coal", "natural gas bridge fuel", carbon capture for enhanced oil recovery, hydrogen from fossil fuels without full lifecycle accounting?
- If YES → FLAG as "fossil_transition" (max_impact_potential = 4)
- EXCEPTION: Genuine renewable hydrogen, direct air capture for permanent storage, fossil asset retirement CAN score 7+

D) ANNOUNCEMENT vs. DEPLOYMENT
Does article describe:
- Announcements/pledges/goals? → Score LOWER (aspirational, not achieved)
- Deployed technology/measured outcomes? → Score HIGHER (proven impact)
- Both? → Score the deployment, note the aspiration separately

STEP 2: Evaluate Sustainability Dimensions (score 0-10 for each)

1. **Climate Impact Potential**:
   - Will this demonstrably reduce GHG emissions, sequester carbon, or adapt to climate change?
   - Is the impact quantified (tons CO2e, % reduction, MW capacity)?
   - NOT just "supports sustainability" - must have direct climate mechanism
   - Lifecycle accounting required (e.g., EVs only count if grid is clean)
   - (0-2: minimal/unproven, 3-4: modest/theoretical, 5-6: significant/pilot-proven, 7-8: transformative/deployed, 9-10: breakthrough/scaling)

2. **Technical Credibility**:
   - Is there scientific evidence, peer-reviewed research, or independent validation?
   - Are efficiency claims realistic vs. thermodynamic limits?
   - Does it cite specific metrics (efficiency %, energy density, cost per kWh)?
   - Are claims verified by independent parties (NOT just company press releases)?
   - (GATEKEEPER: if <5, max overall sustainability_score = 4)

3. **Economic Viability**:
   - Is there a path to cost-competitiveness with fossil alternatives?
   - Are unit economics disclosed (LCOE, $/ton CO2, payback period)?
   - Is there demonstrated demand (customer contracts, deployment commitments)?
   - For early-stage: is there a credible cost reduction roadmap?
   - (0-2: no path to viability, 3-4: needs major subsidies, 5-6: approaching parity, 7-8: competitive now, 9-10: cheaper than fossil)

4. **Deployment Readiness**:
   - What stage: research → pilot → commercial → scaling?
   - Are there operational units generating real-world data?
   - Supply chain established? Manufacturing capacity?
   - Regulatory approvals in place?
   - (0-2: concept/lab, 3-4: pilot, 5-6: first commercial, 7-8: proven at scale, 9-10: mass deployment)

5. **Systemic Impact**:
   - Does this enable broader decarbonization (e.g., grid storage enables renewables)?
   - Does it address a bottleneck (e.g., battery recycling, green steel)?
   - Can it scale to gigaton-level impact (1% of global emissions = 0.5 Gt CO2)?
   - Does it create co-benefits (jobs, health, energy access)?
   - (0-2: niche, 3-4: sectoral, 5-6: cross-sectoral, 7-8: economy-wide, 9-10: global infrastructure)

6. **Justice & Equity**:
   - Does this avoid harm to frontline/indigenous communities?
   - Is access equitable (not just for wealthy nations/individuals)?
   - Does it address historical climate injustice?
   - Are workers/communities part of decision-making?
   - Examples: just transition programs, community solar, climate reparations
   - (0: actively harmful, 3: neutral, 5: some equity considerations, 7: equity-centered, 10: reparative)

7. **Innovation Quality**:
   - Is this a genuine breakthrough or incremental improvement?
   - Does it solve a previously unsolved problem?
   - Is the technology open/accessible or proprietary?
   - For incremental: is the improvement significant (>20% efficiency gain)?
   - NOT "innovation theater" - must have technical substance
   - (0-2: hype, 3-4: incremental, 5-6: significant, 7-8: breakthrough, 9-10: paradigm shift)

8. **Evidence Strength**:
   - Quality of sources cited (peer-reviewed > industry report > press release)
   - Transparency of data (open data > disclosed metrics > vague claims)
   - Independent verification (third-party audits, government data, academic studies)
   - Track record of source (known for accuracy vs. known for hype)
   - (0-2: unverified claims, 3-4: industry sources, 5-6: some independent data, 7-8: peer-reviewed, 9-10: multiple independent confirmations)

STEP 3: Investment Intelligence Metadata

Classify the article's content type:
- **breakthrough_research**: Novel scientific findings with climate relevance
- **technology_deployment**: Actual installed/operating climate tech
- **policy_action**: Enacted regulations/incentives (NOT proposals)
- **market_signal**: Investment, corporate commitment with binding contracts
- **impact_measurement**: OWID-style progress indicators, emissions data
- **greenwashing**: Primarily marketing/pledges without concrete action
- **transition_delay**: Fossil fuel industry delay tactics

Identify innovation stage:
- **basic_research**: Lab/academic research (TRL 1-3)
- **applied_research**: Proof of concept, prototypes (TRL 4-5)
- **pilot**: Demonstration projects, first deployments (TRL 6-7)
- **commercial**: Market-ready, early adoption (TRL 8)
- **scaling**: Mass deployment, cost-competitive (TRL 9)

Flag investment signals (true/false for each):
- **has_funding**: Recent investment round disclosed
- **has_patents**: IP protection mentioned
- **has_customers**: Named customers/contracts
- **has_metrics**: Quantified performance/impact data
- **has_peer_review**: Academic validation
- **has_deployment**: Operational units in field

STEP 4: Cross-Reference Indicators

Does the article mention (mark true/false):
- **owid_indicator**: References Our World in Data metrics
- **ipcc_alignment**: Aligns with IPCC pathways/reports
- **iea_data**: Cites International Energy Agency data
- **third_party_verified**: Independent audit/certification mentioned
- **regulatory_approved**: Government approval/compliance noted

STEP 5: Provide Scores

DO NOT calculate composite scores yourself - just provide dimension scores and metadata. The system will calculate weighted sustainability_score and investment_readiness_score.

Respond with ONLY valid JSON in this exact format:
{{
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

  "reasoning": "<2-3 sentences explaining key factors: what concrete action is happening, what evidence supports claims, what stage of deployment>",

  "key_impact_metrics": ["<metric1 with number>", "<metric2 with number>"],
  "technology_tags": ["<specific_tech1>", "<specific_tech2>"],
  "sdg_alignment": [<SDG numbers 1-17 that apply>]
}}

CRITICAL REMINDERS:
- Focus on DEPLOYED TECHNOLOGY and MEASURED OUTCOMES, not promises
- Greenwashing = commitments without action (score LOW on credibility)
- Vaporware = announcements without deployments (score LOW on readiness)
- Fossil fuel "bridge" strategies = delay tactics (score LOW on impact)
- Technical credibility gatekeeps everything (if <5, cap sustainability_score at 4)
- Quantified metrics > vague claims
- Independent verification > corporate press releases
- Climate justice matters - extractive solutions score lower

DO NOT include any text outside the JSON object.
```

---

## SCORING WEIGHTS (for downstream processing)

### Sustainability Score (0-10)
Used for Progress Tracker application:

```python
sustainability_score = (
    climate_impact_potential * 0.30 +
    technical_credibility * 0.20 +  # Gatekeeper dimension
    systemic_impact * 0.20 +
    evidence_strength * 0.15 +
    justice_equity * 0.10 +
    innovation_quality * 0.05
)

# Apply gatekeeper
if technical_credibility < 5:
    sustainability_score = min(sustainability_score, 4.0)

# Apply flags
if flags['greenwashing_risk']:
    sustainability_score *= 0.6
if flags['vaporware_risk']:
    sustainability_score *= 0.7
if flags['fossil_transition']:
    sustainability_score = min(sustainability_score, 4.0)
```

### Investment Readiness Score (0-10)
Used for Climate Tech Investment Intelligence application:

```python
investment_readiness_score = (
    deployment_readiness * 0.30 +
    economic_viability * 0.25 +
    technical_credibility * 0.20 +
    climate_impact_potential * 0.15 +
    innovation_quality * 0.10
)

# Bonus for investment signals (up to +2.0)
investment_signals_bonus = sum([
    0.4 if has_funding else 0,
    0.3 if has_patents else 0,
    0.5 if has_customers else 0,
    0.4 if has_metrics else 0,
    0.2 if has_peer_review else 0,
    0.2 if has_deployment else 0
])

investment_readiness_score = min(10, investment_readiness_score + investment_signals_bonus)

# Apply vaporware penalty
if flags['vaporware_risk']:
    investment_readiness_score = min(investment_readiness_score, 4.0)
```

---

## EXPECTED SCORE DISTRIBUTIONS

Based on sustainability content characteristics:

### Dimension Score Distributions
- **climate_impact_potential**: Normal distribution, mean ~5.5
- **technical_credibility**: Right-skewed (filter out low-credibility), mean ~6.5
- **deployment_readiness**: Left-skewed (more research than deployment), mean ~4.5
- **investment_readiness_score**: Bimodal (clear winners/losers), peaks at ~3 and ~7

### Content Type Distribution
- **technology_deployment**: 25-35%
- **breakthrough_research**: 15-25%
- **greenwashing**: 20-30%
- **market_signal**: 10-20%
- **policy_action**: 5-10%
- **vaporware**: 10-15%
- **transition_delay**: 5-10%

---

## PRE-FILTER RECOMMENDATION

To reduce labeling costs while maintaining coverage:

**Only analyze articles where**:
- Source category in: `climate_solutions`, `energy_utilities`, `renewable_energy`, `automotive_transport`, `science`, `economics`
- OR article contains keywords: `climate`, `carbon`, `renewable`, `solar`, `wind`, `battery`, `ev`, `electric`, `sustainability`, `green`, `emission`

This filter is implemented in `batch_labeler.py` as `sustainability_pre_filter()`.

---

## SEMANTIC FRAMEWORK DETAILS

### Core Dimensions Explained

#### 1. Climate Impact Potential (Weight: 30%)
**What it means**: Demonstrable reduction in GHG emissions, carbon sequestration, or climate adaptation

**CRITICAL**: Must have direct climate mechanism with quantified impact
- NOT vague "supports sustainability"
- Requires lifecycle accounting (EVs only count if grid is clean)
- Must cite specific metrics (tons CO2e, % reduction, MW capacity)

**Scoring Guide**:
- **0-2 (Minimal/Unproven)**: Vague claims, no quantification, indirect mechanisms
- **3-4 (Modest/Theoretical)**: Small impact, theoretical models only
- **5-6 (Significant/Pilot-proven)**: Material impact demonstrated in pilots
- **7-8 (Transformative/Deployed)**: Large-scale deployment with verified impact
- **9-10 (Breakthrough/Scaling)**: Gigaton-scale potential, rapidly scaling

**Examples**:
- ✅ "Grid-scale battery stores 1 GWh, enables 500 MW of solar integration, avoiding 200,000 tons CO2/year"
- ✅ "Direct air capture removes 36,000 tons CO2 annually, permanently stored in basalt"
- ❌ "Company commits to carbon neutrality" (no concrete mechanism)
- ❌ "Green hydrogen from natural gas with CCS" (lifecycle emissions unclear)

#### 2. Technical Credibility (Weight: 20% - GATEKEEPER)
**What it means**: Scientific evidence, peer review, independent validation

**GATEKEEPER RULE**: If this scores <5, maximum sustainability_score = 4
- Prevents hype and vaporware from scoring high
- Requires independent verification, not just company claims

**Scoring Guide**:
- **0-2 (Unverified)**: Company press releases only, no independent validation
- **3-4 (Industry sources)**: Industry reports, trade publications
- **5-6 (Some independent data)**: Government data, third-party audits
- **7-8 (Peer-reviewed)**: Academic publications, multiple independent sources
- **9-10 (Robust validation)**: Multiple peer-reviewed studies, government/IPCC data

**Examples**:
- ✅ "Nature paper shows 27.3% efficiency, validated by NREL, 1,000-hour stability test"
- ✅ "IEA report confirms 90% round-trip efficiency across 50 installations"
- ❌ "Startup claims breakthrough efficiency" (company press release only)
- ❌ "Technology surpasses thermodynamic limits" (physically impossible claim)

#### 3. Economic Viability (Weight: 25% for Investment Readiness)
**What it means**: Path to cost-competitiveness with fossil alternatives

**Critical for investment**: Must show unit economics and demonstrated demand
- LCOE (levelized cost of energy)
- $/ton CO2 removed
- Payback period
- Customer contracts

**Scoring Guide**:
- **0-2**: No path to viability, requires permanent subsidies
- **3-4**: Needs major subsidies, unclear cost trajectory
- **5-6**: Approaching parity, credible cost reduction roadmap
- **7-8**: Cost-competitive now in some markets
- **9-10**: Cheaper than fossil alternatives globally

**Examples**:
- ✅ "Solar+storage at $40/MWh vs. natural gas at $45/MWh in Texas market"
- ✅ "Green steel production cost drops 30% annually, reaching fossil parity by 2027"
- ❌ "Revolutionary technology" (no cost data disclosed)
- ❌ "Will be cost-competitive after government support" (no clear path)

#### 4. Deployment Readiness (Weight: 30% for Investment Readiness)
**What it means**: Technology Readiness Level (TRL) and operational status

**Innovation Stages**:
- **TRL 1-3 (Basic research)**: Lab/academic research, theoretical concepts
- **TRL 4-5 (Applied research)**: Prototypes, proof of concept
- **TRL 6-7 (Pilot)**: Demonstration projects, first deployments
- **TRL 8 (Commercial)**: Market-ready, early commercial adoption
- **TRL 9 (Scaling)**: Mass deployment, proven at scale

**Scoring Guide**:
- **0-2**: Concept/lab only, no physical prototype
- **3-4**: Pilot projects, limited operational data
- **5-6**: First commercial installations, early customers
- **7-8**: Proven at scale, established supply chain
- **9-10**: Mass deployment, mature industry

**Examples**:
- ✅ "10 GW of offshore wind capacity installed across 50 sites"
- ✅ "Battery recycling facility processes 10,000 tons annually, 4-year track record"
- ❌ "Prototype successfully tested in lab conditions" (TRL 4, not deployed)
- ❌ "Plans to build first commercial unit next year" (vaporware)

#### 5. Systemic Impact (Weight: 20%)
**What it means**: Enables broader decarbonization, addresses bottlenecks, scales to gigatons

**Critical**: Must enable other solutions or address structural barriers
- Grid storage → enables variable renewables
- Green steel → decarbonizes construction, manufacturing
- Battery recycling → enables circular EV economy

**Scoring Guide**:
- **0-2 (Niche)**: Limited scope, doesn't enable other solutions
- **3-4 (Sectoral)**: Important for one sector only
- **5-6 (Cross-sectoral)**: Enables multiple sectors
- **7-8 (Economy-wide)**: Transforms major economic systems
- **9-10 (Global infrastructure)**: Gigaton-scale potential (1% of global emissions = 0.5 Gt CO2)

**Examples**:
- ✅ "Long-duration energy storage enables 100% renewable grids"
- ✅ "Green ammonia enables zero-carbon shipping AND fertilizer production"
- ❌ "Energy-efficient dishwasher" (niche impact)
- ❌ "Better insulation for luxury homes" (limited accessibility)

#### 6. Justice & Equity (Weight: 10%)
**What it means**: Avoids harm, promotes access, addresses climate injustice

**Critical considerations**:
- Frontline/indigenous communities not harmed
- Equitable access (not just for wealthy)
- Just transition for workers
- Addresses historical climate debt

**Scoring Guide**:
- **0**: Actively harmful to frontline communities
- **3**: Neutral, no explicit equity considerations
- **5**: Some equity considerations included
- **7**: Equity-centered design, community voice
- **10**: Reparative, addresses historical injustice

**Examples**:
- ✅ "Community solar co-op prioritizes low-income households, 40% discount"
- ✅ "Coal plant closure includes 5-year retraining program for all workers"
- ❌ "EV charging in wealthy suburbs only" (reinforces inequality)
- ❌ "Carbon offset displaces indigenous communities" (actively harmful)

#### 7. Innovation Quality (Weight: 5%)
**What it means**: Genuine breakthrough vs. incremental vs. hype

**NOT innovation theater**: Must have technical substance
- Solves previously unsolved problem
- Significant improvement (>20% efficiency gain)
- Open/accessible vs. proprietary

**Scoring Guide**:
- **0-2 (Hype)**: Marketing buzzwords, no technical substance
- **3-4 (Incremental)**: Minor improvement on existing tech
- **5-6 (Significant)**: Major improvement, new approach
- **7-8 (Breakthrough)**: Solves previously unsolved problem
- **9-10 (Paradigm shift)**: Fundamentally changes the field

**Examples**:
- ✅ "Perovskite achieves 1,000-hour stability (vs. previous 100 hours)"
- ✅ "Direct lithium extraction reduces water use 95% vs. evaporation ponds"
- ❌ "AI-powered blockchain for carbon credits" (buzzword bingo)
- ❌ "5% efficiency improvement in mature technology" (incremental)

#### 8. Evidence Strength (Weight: 15%)
**What it means**: Quality and independence of sources

**Evidence Hierarchy**:
1. **Multiple peer-reviewed studies** (strongest)
2. **Single peer-reviewed study**
3. **Government data (IEA, IPCC, OWID)**
4. **Third-party audits/certifications**
5. **Industry reports**
6. **Company disclosures**
7. **Press releases** (weakest)

**Scoring Guide**:
- **0-2**: Unverified claims, press releases only
- **3-4**: Industry sources, company data
- **5-6**: Some independent data, third-party audits
- **7-8**: Peer-reviewed, government data
- **9-10**: Multiple independent confirmations, IPCC/IEA data

**Examples**:
- ✅ "IPCC AR6 cites three studies confirming 90% emission reduction"
- ✅ "Independent audit by DNV-GL confirms methane reduction claims"
- ❌ "Company announces record-breaking results" (unverified)
- ❌ "Influencer promotes climate solution" (not credible source)

---

## VALIDATION EXAMPLES

### Example 1: High Score (Deployed Technology - 8.2/10)
**Article**: "Tesla Megapack battery storage facility in Texas reaches 1 GWh capacity, storing enough renewable energy to power 250,000 homes during peak demand. Facility achieved 90% round-trip efficiency in first year of operation, exceeding design specs."

**Scores**:
- Climate Impact Potential: 8 (enables renewable integration at scale)
- Technical Credibility: 8 (operational data from deployed system)
- Economic Viability: 7 (proven market competitiveness)
- Deployment Readiness: 9 (mass deployment, TRL 9)
- Systemic Impact: 8 (enables grid-scale renewables)
- Justice & Equity: 5 (neutral, no explicit equity focus)
- Innovation Quality: 6 (proven technology, incremental scale-up)
- Evidence Strength: 7 (operational data, industry reports)

**Content Type**: technology_deployment
**Innovation Stage**: scaling
**Sustainability Score**: 8.0
**Investment Readiness**: 8.5 (has deployment, has metrics, has customers)

### Example 2: Low Score (Greenwashing - 1.8/10)
**Article**: "Oil major commits to net-zero by 2050 through carbon offset investments and future technology breakthroughs. Company maintains current production levels while pledging to invest 5% of capex in renewable energy projects over next decade."

**Scores**:
- Climate Impact Potential: 2 (offsets don't cancel new extraction)
- Technical Credibility: 2 (no concrete technology specified)
- Economic Viability: 3 (vague commitments)
- Deployment Readiness: 1 (no deployed technology)
- Systemic Impact: 2 (maintains fossil fuel system)
- Justice & Equity: 2 (no mention of impacted communities)
- Innovation Quality: 1 (relies on undefined "future technology")
- Evidence Strength: 2 (company press release only)

**Flags**: greenwashing_risk, vaporware_risk, fossil_transition
**Content Type**: greenwashing
**Sustainability Score**: 1.8 (greenwashing penalty: 3.0 × 0.6)
**Investment Readiness**: 1.5

### Example 3: Research Breakthrough (7.5/10)
**Article**: "New perovskite solar cell achieves 27.3% efficiency, peer-reviewed in Nature. Researchers demonstrate 1,000-hour stability under standard test conditions, addressing key commercialization barrier. Manufacturing process uses Earth-abundant materials."

**Scores**:
- Climate Impact Potential: 7 (high potential, not yet deployed)
- Technical Credibility: 9 (peer-reviewed in Nature, validated by NREL)
- Economic Viability: 5 (potential for low cost, not proven at scale)
- Deployment Readiness: 4 (applied research, TRL 5)
- Systemic Impact: 8 (solar is key to decarbonization)
- Justice & Equity: 6 (Earth-abundant materials = more equitable access)
- Innovation Quality: 8 (breakthrough stability achievement)
- Evidence Strength: 9 (peer-reviewed, multiple independent validations)

**Content Type**: breakthrough_research
**Innovation Stage**: applied_research
**Sustainability Score**: 7.5
**Investment Readiness**: 5.8 (has peer review, has metrics, has patents)

---

## ETHICAL CONSIDERATIONS

### What This Filter EXCLUDES
- **Greenwashing**: Commitments without concrete action
- **Vaporware**: Announcements without deployments
- **Fossil fuel delay tactics**: "Bridge fuels", CCS for EOR, blue hydrogen without lifecycle data
- **Extractive solutions**: Harm to frontline/indigenous communities
- **Proprietary hype**: Locked-down technology without accessibility

### Known Biases to Monitor
- **Techno-optimism**: Don't over-weight novel technology vs. proven solutions
- **Global North bias**: Ensure solutions from Global South get fair evaluation
- **Market bias**: Don't conflate market success with climate impact
- **Deployment bias**: Early-stage breakthrough research matters too

### Consistency Checks
- If `deployment_readiness > 7`, then `has_deployment` should be `true`
- If `technical_credibility < 5`, then `sustainability_score` capped at 4
- If `content_type == "greenwashing"`, then `climate_impact_potential < 4`
- If `innovation_stage == "scaling"`, then `economic_viability > 6`

---

**This filter distinguishes genuine climate solutions from greenwashing, vaporware, and fossil fuel delay tactics.**
