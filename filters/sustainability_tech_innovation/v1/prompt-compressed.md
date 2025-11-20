# Sustainable Technology & Innovation Scoring

**Purpose**: Rate cool sustainable tech that WORKS - deployed tech, working pilots, validated breakthroughs.

**Version**: 1.1

**Focus**: Technology with REAL RESULTS, not just theory or promises.

**Philosophy**: "Pilots and research need real results, not just theory."

**Oracle Output**: Dimensional scores only (0-10 per dimension). Tier classification (if needed) is applied post-processing, not by the oracle.

---

## PROMPT TEMPLATE

```
Rate this article on 8 dimensions (0-10 scale). Focus: SUSTAINABLE TECH THAT WORKS (deployed, pilots, validated research).

**IMPORTANT:** Check CRITICAL FILTERS for each dimension BEFORE scoring.

## CRITICAL: What is "Tech That Works"?

**INCLUDE:**
- ✅ **Deployed technology** (mass market, commercial, operational installations)
- ✅ **Working pilots** with performance data (MW generated, emissions reduced, efficiency achieved)
- ✅ **Validated research** with real-world results (field tests, actual measurements, not simulations)
- ✅ **Technology demonstrations** that work (proven in field, not just lab)
- ✅ **Breakthroughs** with exceptional validated performance (real data, not claims)

**EXCLUDE:**
- ❌ Pure theory (no real-world validation)
- ❌ Simulations/models without deployment or field validation
- ❌ Future announcements ("coming in 2027", "plans to deploy")
- ❌ Proposals without operational data or pilot results
- ❌ Vaporware (claims without evidence)
- ❌ Infrastructure disruption (protests blocking trains)

**Examples:**
- ✅ "Pilot geothermal plant generates 5 MW for 6 months" → INCLUDE (working pilot with data)
- ✅ "Battery degradation model achieves 95% accuracy on real EV data" → INCLUDE (validated on real data)
- ✅ "Solar farm produces 100 GWh annually" → INCLUDE (deployed)
- ✅ "Heat pump field trial shows 3.5 COP in Nordic climate" → INCLUDE (field validation)
- ❌ "Scientists propose new battery chemistry" → EXCLUDE (theory only)
- ❌ "Startup announces 2027 hydrogen plant" → EXCLUDE (future promise)
- ❌ "Simulation predicts 40% efficiency improvement" → EXCLUDE (no real-world validation)

---

## ⚠️ CRITICAL: MANDATORY GATEKEEPER RULES ⚠️

**BEFORE SCORING:** Determine if article describes REAL WORK with EVIDENCE:

### What is REAL WORK?
- ✅ **Deployed** (operational, generating power, in use)
- ✅ **Working pilot** (pilot with performance data: "5 MW for 6 months", "COP 3.5 achieved", "95% uptime")
- ✅ **Validated research** (field tests, real-world measurements, not simulations)

### What is NOT real work?
- ❌ **Proposals** ("plans to deploy", "proposes 600 MW", "announces 2027 launch")
- ❌ **Future-only** ("will deploy", "coming in 2027", "expected to generate")
- ❌ **Theory/simulations** (models without field validation, predictions without real data)

### EXAMPLES - Proposals vs Pilots:
- ❌ **PROPOSAL:** "Xcel proposes 600 MW solar farm, delivery 2027" → deployment_maturity = 1-2 (future only)
- ✅ **PILOT:** "5 MW geothermal pilot generates power for 6 months" → deployment_maturity = 4-5 (working pilot with data)
- ❌ **ANNOUNCEMENT:** "Company announces breakthrough battery, production 2026" → deployment_maturity = 1-2 (vaporware)
- ✅ **DEPLOYMENT:** "50 MW battery operates since 2023, 90% uptime" → deployment_maturity = 6-7 (deployed, proven)
- ❌ **SIMULATION:** "Model predicts 40% efficiency gain" → deployment_maturity = 1-2 (theory only)
- ✅ **FIELD TEST:** "Heat pump achieves COP 3.5 in Nordic field trial" → deployment_maturity = 3-4 (validated)

### ENFORCEMENT:

**AFTER scoring all dimensions:**

1. IF **deployment_maturity < 3.0** (no real work, theory/proposals only):
   - **IMMEDIATELY SET all dimensional scores = 1.0**
   - **SET overall score = 1.0**
   - **SET deployment_stage = "theory_only" OR "out_of_scope"**
   - **REASONING:** No real-world work = Not in scope for this filter

2. IF **proof_of_impact < 3.0** (no real data, no measurements):
   - **IMMEDIATELY SET all dimensional scores = 1.0**
   - **SET overall score = 1.0**
   - **SET deployment_stage = "theory_only"**
   - **REASONING:** No real impact data = Not in scope

**NOTE:** Proposals about future deployments MUST score deployment_maturity = 1-2, triggering gatekeeper enforcement.

---

ARTICLE:
Title: {title}
Text: {text}

---

## Dimensions

### 1. DEPLOYMENT_MATURITY (20%) - GATEKEEPER

Lab → Pilot → Commercial → Mass Deployment

   **❌ CRITICAL FILTERS - If article is ANY of these, score 0-2:**
   - **Pure theory** (no pilot, no deployment, no field validation)
   - **Simulations only** (models/predictions without real-world testing)
   - **Future-only announcements** (plans, proposals, "coming in 2027", no current work)
   - **Infrastructure disruption** (protests, strikes, service outages)
   - Generic IT infrastructure (cloud, Kubernetes, databases - not climate-specific)
   - Programming languages and frameworks (Python, JavaScript, React)
   - Office productivity software (Microsoft Office, Google Workspace)
   - Gaming and entertainment tech
   - AI/data center energy (unless directly climate/decarbonization focused)
   - Generic software tools without climate application

   **If NONE of above filters match AND article is about sustainable tech with real results, score normally:**
   - **9-10**: Mass deployment (GW-scale, millions of units, years operational)
   - **7-8**: Proven at scale (multi-site commercial, years of operation, reliable)
   - **5-6**: First commercial deployment OR validated pilots with strong performance data
   - **3-4**: Working pilot/demo with real performance data OR validated research with field results
   - **1-2**: Lab/concept only, no real-world validation

   **Evidence**: "X MW/GW operational", "pilot generated Y MWh over Z months", "field test achieved W% efficiency", "validated on real-world data"

   **IMPORTANT:** Pilots and validated research CAN score 3-6 if they have real performance data. Don't require mass deployment.

### 2. TECHNOLOGY_PERFORMANCE (15%)

Real-world performance vs promises

   **❌ CRITICAL FILTERS - If article is ANY of these, score 0-2:**
   - Generic IT/software performance (faster databases, better APIs)
   - Programming language benchmarks
   - Non-climate tech performance
   - Gaming/entertainment performance
   - Simulations without real-world validation

   **If NONE of above filters match AND tech is sustainable tech with real results, score normally:**
   - **9-10**: Outperforms fossil, proven reliability, exceeds expectations
   - **7-8**: Meets expectations, real-world data validates projections
   - **5-6**: Real-world validated OR pilot results show promise, minor limitations
   - **3-4**: Lab/pilot conditions, some real data but limited validation
   - **1-2**: No performance data, claims only, simulations only

   **Evidence**: Capacity factor, efficiency (COP, round-trip), uptime %, degradation rates, pilot performance data, field test results

   **IMPORTANT:** Pilot performance data (even limited) is valid evidence. Don't require full-scale deployment.

### 3. COST_TRAJECTORY (15%)

Is cost declining? Path to competitiveness?

   **❌ CRITICAL FILTERS - If article is ANY of these, score 0-2:**
   - Generic IT/software pricing
   - Non-climate tech costs
   - Consumer electronics pricing
   - No cost information at all

   **If NONE of above filters match AND tech is sustainable tech, score normally:**
   - **9-10**: Cheaper than fossil, costs fell >50% in 5 years
   - **7-8**: Cost-competitive (LCOE parity), no subsidies needed
   - **5-6**: Approaching parity, costs declining steadily, OR pilot shows cost potential
   - **3-4**: 2-5x fossil costs, slow decline OR early stage with cost estimates
   - **1-2**: No cost data, increasing costs, >5x fossil

   **Evidence**: LCOE or $/unit, historical cost curve, vs fossil baseline, pilot cost analysis, cost projections with rationale

   **IMPORTANT:** For pilots, projected costs based on scale-up are acceptable if well-reasoned.

### 4. SCALE_OF_DEPLOYMENT (15%)

How much deployed? MW/GW, units, installations, OR pilot scale

   **❌ CRITICAL FILTERS - If article is ANY of these, score 0-2:**
   - IT infrastructure scale (servers, cloud instances, API calls)
   - Software adoption metrics (users, downloads)
   - Non-climate tech deployments
   - Zero scale (theory only)

   **If NONE of above filters match AND tech is sustainable tech with real work, score normally:**

   **Context-aware** (adjust for entity type and stage):
   - **Countries**: 9-10 = >10 GW, 7-8 = 1-10 GW, 5-6 = 100 MW-1 GW, 3-4 = 10-100 MW
   - **Companies**: 9-10 = >100k units, 7-8 = 10k-100k, 5-6 = 1k-10k, 3-4 = 100-1k units
   - **Pilots**: 9-10 = N/A, 7-8 = N/A, 5-6 = multi-site pilots 10+ MW, 3-4 = single pilot 1-10 MW
   - **Research**: 9-10 = N/A, 7-8 = N/A, 5-6 = multi-site validation, 3-4 = single-site validation

   **Evidence**: "X MW/GW installed", "Y units sold", "Z installations", "pilot: W MW for V months", "field test: Q kW system"

   **IMPORTANT:** Pilots at 1-10 MW scale are significant (score 3-5). Don't penalize for not being GW-scale.

### 5. MARKET_PENETRATION (15%)

% of total market captured OR potential market application

   **❌ CRITICAL FILTERS - If article is ANY of these, score 0-2:**
   - IT market share (cloud providers, software platforms)
   - Non-climate tech market share
   - Gaming/entertainment market share
   - No market application at all

   **If NONE of above filters match AND tech is sustainable tech, score normally:**
   - **9-10**: >20% market share (mainstream/dominant)
   - **7-8**: 10-20% (widespread adoption)
   - **5-6**: 5-10% (growing niche) OR clear path to significant market
   - **3-4**: 1-5% (early adopters) OR pilot demonstrates market fit
   - **1-2**: <1% OR no clear market application

   **Evidence**: "X% of new car sales", "Y% of electricity", market share data, addressable market analysis

   **IMPORTANT:** For pilots/research, clear market application is sufficient (don't require current market share).

### 6. TECHNOLOGY_READINESS (10%)

Technical risks resolved? Proven technology?

   **❌ CRITICAL FILTERS - If article is ANY of these, score 0-2:**
   - Generic IT/software readiness
   - Non-climate tech maturity
   - Pure theory with no implementation

   **If NONE of above filters match AND tech is sustainable tech with real work, score normally:**
   - **9-10**: Mature (decades deployed, well-understood, no barriers)
   - **7-8**: Proven (multiple deployments, reliability demonstrated)
   - **5-6**: Early commercial OR validated pilots working reliably
   - **3-4**: Pilot stage with some issues being resolved OR validated research
   - **1-2**: Experimental (major hurdles, uncertain feasibility)

   **IMPORTANT:** Successful pilots demonstrate technical feasibility (can score 4-6 even if not commercial).

### 7. SUPPLY_CHAIN_MATURITY (5%)

Can we manufacture millions? OR is manufacturing feasible?

   **❌ CRITICAL FILTERS - If article is ANY of these, score 0-2:**
   - IT/software supply chain (cloud providers, SaaS)
   - Non-climate hardware supply chain
   - No manufacturing considerations

   **If NONE of above filters match AND tech is sustainable tech, score normally:**
   - **9-10**: Global supply chain, mass production capacity
   - **7-8**: Robust supply chain, multiple manufacturers
   - **5-6**: Growing supply chain, some constraints
   - **3-4**: Limited suppliers OR path to manufacturing at scale
   - **1-2**: Single supplier OR no manufacturing plan

   **Evidence**: Manufacturing capacity (units/year), # of suppliers, raw material availability, manufacturing feasibility

   **IMPORTANT:** For pilots, evidence of manufacturing feasibility is sufficient.

### 8. PROOF_OF_IMPACT (5%)

Verified emissions avoided / energy generated (OR pilot impact data)

   **❌ CRITICAL FILTERS - If article is ANY of these, score 0-2:**
   - IT efficiency claims without climate focus
   - Non-climate environmental claims
   - Pure theory with no impact data

   **If NONE of above filters match AND tech has sustainable tech impact, score normally:**
   - **9-10**: Government-certified, multiple confirmations
   - **7-8**: Third-party audited (DNV GL, EPA, IEA)
   - **5-6**: Credible self-report with methodology (GHG Protocol) OR pilot impact data
   - **3-4**: Pilot results OR research validation with real data
   - **1-2**: No quantified impact data, theoretical only

   **Evidence**: "X tons CO2 avoided/year", "Y MWh generated", lifecycle analysis, "pilot reduced emissions by Z%", "field test generated W MWh"

   **IMPORTANT:** Pilot impact data counts as real impact (don't require GW-scale to have impact).

---

## Scoring Calibration

**Score dimensional evidence honestly. Postfilter will classify stage based on scores.**

**Development Stage → Expected Score Range (for postfilter reference):**

- **mass_deployment** (GW-scale, years operational, proven) → Scores typically 8-10
- **commercial_proven** (multi-site, revenue-generating, scaling) → Scores typically 6-8
- **validated_pilots** (working pilots with strong performance data) → Scores typically 5-7
- **working_pilots** (pilots with some performance data) → Scores typically 4-6
- **validated_research** (field validation, real-world data) → Scores typically 3-5
- **lab_only** (no real-world validation) → Scores typically 1-2
- **theory_only** (no implementation at all) → Scores typically 0-2

**Important:**
- Score based on EVIDENCE, not stage label
- Working pilots with performance data should score 4-6 (not 1-3)
- Validated research with field data should score 3-5 (not 1-2)
- Deployed commercial tech should score 5-8+ (as before)

---

## Scoring Philosophy

**Evidence hierarchy** (from strongest to weakest):
1. **Deployment evidence** (X MW deployed, Y units installed, operational since Z) → Score 5-10
2. **Pilot evidence** (W MW pilot, V months runtime, performance data) → Score 3-6
3. **Validation evidence** (field tests, real-world data, measured results) → Score 3-5
4. **Performance data** (efficiency, costs, impact measured in real conditions) → Boost scores
5. **Detailed metrics** (LCOE, degradation rates, supply chain) → Only for scores 7+

**Rule:** If article provides evidence of real work (#1, #2, or #3) for sustainable tech, score normally. Do NOT penalize for "not being GW-scale" if pilot/validation data exists.

**Negative News Handling:**
Score the TECHNOLOGY MATURITY, not the news sentiment.

Examples:
- "Solar company goes bankrupt" → Score the deployed solar tech (7-8), not the business failure
- "Pilot project faces delays" → Score the pilot tech (4-5), note challenges in readiness dimension
- "Battery recall announced" → Score deployment scale, note issues in performance dimension

Negative news often indicates REAL tech facing real-world challenges, which confirms it's past theory stage.

---

## Examples

**High Score (9.1)**: "China Solar Deployment Hits 200 GW in 2024, Costs Fall 20% - IEA"
- Deployment: 10 (mass deployment, operational)
- Performance: 9 (proven reliability)
- Cost: 10 (fell 90% over decade, cheapest electricity)
- Scale: 10 (200 GW country-level)
- Market: 8 (18% of China electricity)
- Readiness: 10 (mature tech)
- Supply: 10 (500 GW global capacity)
- Impact: 9 (government-verified)

**Medium Score (5.2)**: "Geothermal Pilot Generates 5 MW for 6 Months, Costs $8M"
- Deployment: 5 (pilot with 6 months operation, proven concept)
- Performance: 6 (real-world data, reliable performance)
- Cost: 4 (early stage, costs high but path to improvement)
- Scale: 4 (5 MW pilot is significant proof)
- Market: 5 (clear application for baseload renewable)
- Readiness: 5 (pilot proves technical feasibility)
- Supply: 3 (limited suppliers, but standard equipment)
- Impact: 5 (measured CO2 avoidance from pilot)

**Low Score (3.8)**: "Battery Degradation Model Validated on 50 Real EVs Over 2 Years"
- Deployment: 3 (research validated on real data, not deployed)
- Performance: 5 (95% accuracy on real EV data)
- Cost: 3 (cost implications analyzed, not proven)
- Scale: 3 (validated on 50 vehicles, real but limited)
- Market: 5 (clear application for EV industry)
- Readiness: 4 (validated but not commercially deployed)
- Supply: 2 (software tool, not hardware)
- Impact: 4 (helps optimize battery life, measured improvement)

**Very Low Score (1.6)**: "Startup Unveils Revolutionary Solar Panel with 60% Efficiency (Lab Results)"
- Deployment: 1 (lab only, not deployed, no pilot)
- Performance: 3 (lab conditions, not real-world)
- Cost: 2 (no cost data)
- Scale: 1 (zero deployed)
- Market: 1 (0% - doesn't exist)
- Readiness: 2 (experimental)
- Supply: 1 (no manufacturing)
- Impact: 1 (zero impact, not deployed)

**OUT OF SCOPE (1.0)**: "AI Platform Optimizes Data Center Cooling, Reduces Energy 30%"
- Deployment: 1 (IT infrastructure - OUT OF SCOPE unless climate-focused)
- Performance: 1 (IT performance - OUT OF SCOPE)
- Cost: 1 (IT costs - OUT OF SCOPE)
- Scale: 1 (IT scale - OUT OF SCOPE)
- Market: 1 (IT market - OUT OF SCOPE)
- Readiness: 1 (IT readiness - OUT OF SCOPE)
- Supply: 1 (IT supply chain - OUT OF SCOPE)
- Impact: 1 (generic energy reduction, not climate-focused - OUT OF SCOPE)
- **Reasoning**: Generic IT infrastructure optimization. Unless article explicitly frames this as climate/decarbonization initiative with climate impact goals, score 0-2 on ALL dimensions.

---

## Output Format (JSON)

**ORACLE OUTPUTS DIMENSIONAL SCORES ONLY. Postfilter will classify tier/stage.**

{{
  "deployment_maturity": {{"score": <0-10>, "reasoning": "Brief justification"}},
  "technology_performance": {{"score": <0-10>, "reasoning": "..."}},
  "cost_trajectory": {{"score": <0-10>, "reasoning": "..."}},
  "scale_of_deployment": {{"score": <0-10>, "reasoning": "..."}},
  "market_penetration": {{"score": <0-10>, "reasoning": "..."}},
  "technology_readiness": {{"score": <0-10>, "reasoning": "..."}},
  "supply_chain_maturity": {{"score": <0-10>, "reasoning": "..."}},
  "proof_of_impact": {{"score": <0-10>, "reasoning": "..."}},
  "overall_assessment": "<1-2 sentence summary>",
  "primary_technology": "solar|wind|batteries|EVs|heat_pumps|hydrogen|geothermal|other",
  "confidence": "HIGH|MEDIUM|LOW"
}}

**Be balanced**:
- Real deployments with scale → 7-10
- Commercial deployments (early stage) → 5-7
- Working pilots with data → 4-6
- Validated research with field results → 3-5
- Lab only or theory → 1-2
- Pure vaporware → 0-2

**CRITICAL REMINDER:**
1. Check inline filters BEFORE scoring each dimension
2. Pilots and validated research ARE in scope (score 3-6, not 0-2)
3. Generic IT/software without climate focus is OUT OF SCOPE (score 0-2)
4. Don't penalize pilots for "not being GW-scale" - judge by stage-appropriate metrics

DO NOT include any text outside the JSON object.
```

---

## CHANGELOG

**v1.1 (2025-11-17):**
- **CRITICAL FIX: MANDATORY GATEKEEPER ENFORCEMENT**
  - Moved gatekeeper rules to TOP of prompt (before article, before dimensions)
  - Added ALL-CAPS enforcement instructions with explicit SET commands
  - Added 6 examples distinguishing proposals from pilots
  - Problem: v1.0 had 85.7% false positive rate (proposals scored as pilots)
  - Solution: "IF deployment_maturity < 3.0 → SET all scores = 1.0" (MANDATORY)

- **PREFILTER: Option D (Minimal Filtering)**
  - Strategy: Trust oracle, only block obvious out-of-scope
  - Results: 68% pass rate on climate articles (vs 16% for v1.0)
  - 62% improvement in false negative rate (84 → 32 blocked articles)

- **HARMONIZATION with uplifting/investment-risk:**
  - Added Philosophy line to header
  - Moved ARTICLE placement to after gatekeeper rules (matches uplifting structure)
  - Removed duplicate gatekeeper reminder section (consolidated to one enforcement point)
  - Structure now: Scope → Gatekeepers → Article → Dimensions → Examples → Output

**v1.0 (2025-11-15):**
- **PIVOT from tech_deployment v3:** Broadened scope from "deployed only" to "tech that works"
- **NEW INCLUSIONS:** Working pilots with performance data, validated research with field results
- **LOWERED GATEKEEPERS:** deployment_maturity 5.0→3.0, proof_of_impact 4.0→3.0
- **NEW TIER THRESHOLDS:** breakthrough 8.0, validated 6.0, promising 4.0, early_stage 2.0
- **SCORING GUIDANCE:** Added pilot/research scoring examples, stage-appropriate metrics
- **Expected impact:** Increase pass rate from 2-5% to 5-20%, reduce false negatives for innovative tech

---

**Token estimate**: ~2,900 tokens (v1.0 was ~2,400 tokens, added gatekeeper enforcement section)
