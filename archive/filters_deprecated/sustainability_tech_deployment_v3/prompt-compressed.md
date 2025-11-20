# Climate Technology Deployment Scoring

**Purpose**: Rate deployed climate technology for maturity, performance, and scale.

**Version**: 3.0-revalidation-fixes

**Focus**: DEPLOYED tech with measurable impact, not lab prototypes or vaporware.

**Oracle Output**: Dimensional scores only (0-10 per dimension). Tier classification (if needed) is applied post-processing, not by the oracle.

---

## PROMPT TEMPLATE

```
Rate this article on 8 dimensions (0-10 scale). Focus: DEPLOYED climate/sustainability tech, not lab prototypes or vaporware.

**IMPORTANT:** Check CRITICAL FILTERS for each dimension BEFORE scoring.

## CRITICAL: What is "Deployed Climate Tech"?

**DEPLOYED means:**
- A REAL INSTALLATION that is CURRENTLY OPERATING
- Generating energy, reducing emissions, or providing climate solutions NOW
- Not a prototype, not a plan, not research

**NOT deployed:**
- Research papers (arXiv, bioRxiv, journals)
- Simulations, models, theoretical studies
- Prototypes, pilots, demonstrations
- Plans, announcements, proposals
- Infrastructure disruption (protests blocking trains)
- General infrastructure mentions (trains exist ≠ climate tech deployment)

**Example:**
- ✅ "50 MW solar farm operational in Arizona, generating 100 GWh/year"
- ❌ "Study models optimal solar placement using ML"
- ❌ "Protesters block train service to demand climate action"
- ❌ "Company announces plans to build hydrogen plant"

ARTICLE:
Title: {title}
Text: {text}

---

## Dimensions

### 1. DEPLOYMENT_MATURITY (20%)
Lab → Pilot → Commercial → Mass Deployment

   **❌ CRITICAL FILTERS - If article is ANY of these, score 0-2:**
   - **Research papers** (arXiv, bioRxiv, medRxiv, DOI references, "study finds", "researchers discover")
   - **Infrastructure disruption** (protests blocking trains/roads, strikes, service outages, political demonstrations)
   - **Simulations and models** (ML models, theoretical frameworks, predictive models - NOT real deployments)
   - Generic IT infrastructure (cloud, Kubernetes, databases, APIs, DevOps tools)
   - Programming languages and frameworks (Python, JavaScript, React, etc.)
   - Operating system features (Windows, Linux, macOS updates)
   - Office productivity software (Microsoft Office, Google Workspace)
   - Generic hardware (not climate-specific - laptops, servers, networking)
   - Social media platforms (Facebook, Twitter, TikTok, Instagram)
   - Gaming and entertainment tech (video games, streaming, VR/AR for entertainment)
   - General healthcare tech (unless directly climate/sustainability related)
   - AI/data center energy (even if renewable) - focus must be on climate/decarbonization goals, not just "powering tech"
   - Generic software tools (project management, collaboration, analytics without climate focus)

   **If NONE of above filters match AND article mentions climate/energy/sustainability/emissions, score normally:**
   - **9-10**: Mass deployment (GW-scale, millions of units, operational for years)
   - **7-8**: Proven at scale (multi-site, years of operation, reliable)
   - **5-6**: First commercial (early deployments, revenue-generating, limited scale)
   - **3-4**: Pilot/demo (test installations, not yet commercial)
   - **1-2**: Lab/concept (research, prototype, no deployment)

   **Evidence**: "X MW/GW operational", "Y units deployed", "generating since [date]"

### 2. TECHNOLOGY_PERFORMANCE (15%)
Real-world performance vs promises

   **❌ CRITICAL FILTERS - If article is ANY of these, score 0-2:**
   - Generic IT/software performance (faster databases, better APIs, CPU optimizations)
   - Programming language benchmarks
   - Office productivity features
   - Non-climate hardware improvements
   - Gaming/entertainment performance
   - AI performance without climate application

   **If NONE of above filters match AND tech is climate/sustainability focused, score normally:**
   - **9-10**: Outperforms fossil, proven reliability, exceeds expectations
   - **7-8**: Meets expectations, real-world data validates projections
   - **5-6**: Real-world validated, minor limitations
   - **3-4**: Lab conditions only, no field validation
   - **1-2**: No performance data, claims only

   **Evidence**: Capacity factor, efficiency (COP, round-trip), uptime %, degradation rates

### 3. COST_TRAJECTORY (15%)
Is cost declining? Path to competitiveness?

   **❌ CRITICAL FILTERS - If article is ANY of these, score 0-2:**
   - Generic IT/software pricing
   - Non-climate tech costs
   - Consumer electronics pricing
   - Entertainment service subscriptions

   **If NONE of above filters match AND tech is climate/sustainability focused, score normally:**
   - **9-10**: Cheaper than fossil, costs fell >50% in 5 years
   - **7-8**: Cost-competitive (LCOE parity), no subsidies needed
   - **5-6**: Approaching parity, costs declining steadily
   - **3-4**: 2-5x fossil costs, slow decline
   - **1-2**: No cost data, increasing costs, >5x fossil

   **Evidence**: LCOE or $/unit, historical cost curve, vs fossil baseline

### 4. SCALE_OF_DEPLOYMENT (15%)
How much deployed? MW/GW, units, installations

   **❌ CRITICAL FILTERS - If article is ANY of these, score 0-2:**
   - IT infrastructure scale (number of servers, cloud instances, API calls)
   - Software adoption metrics (users, downloads, deployments)
   - Gaming/entertainment reach
   - Non-climate hardware shipments

   **If NONE of above filters match AND tech is climate/sustainability focused, score normally:**

   **Context-aware** (adjust for entity type):
   - **Countries**: 9-10 = >10 GW, 7-8 = 1-10 GW, 5-6 = 100 MW-1 GW
   - **Companies**: 9-10 = >100k units, 7-8 = 10k-100k, 5-6 = 1k-10k
   - **Startups**: +1.5x multiplier (1 MW is significant proof-of-concept)

   **Evidence**: "X MW/GW installed", "Y units sold", "Z installations"

### 5. MARKET_PENETRATION (15%)
% of total market captured

   **❌ CRITICAL FILTERS - If article is ANY of these, score 0-2:**
   - IT market share (cloud providers, software platforms)
   - Non-climate tech market share
   - Gaming/entertainment market share

   **If NONE of above filters match AND tech is climate/sustainability focused, score normally:**
   - **9-10**: >20% market share (mainstream/dominant)
   - **7-8**: 10-20% (widespread adoption)
   - **5-6**: 5-10% (growing niche)
   - **3-4**: 1-5% (early adopters)
   - **1-2**: <1% (experimental)

   **Evidence**: "X% of new car sales", "Y% of electricity", market share data

### 6. TECHNOLOGY_READINESS (10%)
Technical risks resolved? Proven technology?

   **❌ CRITICAL FILTERS - If article is ANY of these, score 0-2:**
   - Generic IT/software readiness
   - Non-climate tech maturity
   - Gaming/entertainment tech readiness

   **If NONE of above filters match AND tech is climate/sustainability focused, score normally:**
   - **9-10**: Mature (decades deployed, well-understood, no barriers)
   - **7-8**: Proven (multiple deployments, reliability demonstrated)
   - **5-6**: Early commercial (working but some issues)
   - **3-4**: Pilot (challenges being addressed)
   - **1-2**: Experimental (major hurdles, uncertain)

### 7. SUPPLY_CHAIN_MATURITY (5%)
Can we manufacture millions?

   **❌ CRITICAL FILTERS - If article is ANY of these, score 0-2:**
   - IT/software supply chain (cloud providers, SaaS vendors)
   - Non-climate hardware supply chain
   - Gaming/entertainment production

   **If NONE of above filters match AND tech is climate/sustainability focused, score normally:**
   - **9-10**: Global supply chain, mass production capacity
   - **7-8**: Robust supply chain, multiple manufacturers
   - **5-6**: Growing supply chain, some constraints
   - **3-4**: Limited suppliers, bottlenecks exist
   - **1-2**: Single supplier, custom fabrication

   **Evidence**: Manufacturing capacity (units/year), # of suppliers, raw material availability

### 8. PROOF_OF_IMPACT (5%)
Verified emissions avoided / energy generated

   **❌ CRITICAL FILTERS - If article is ANY of these, score 0-2:**
   - IT efficiency claims without climate focus
   - Non-climate environmental claims
   - Greenwashing (renewable energy to power data centers without climate goal)

   **If NONE of above filters match AND tech has climate/sustainability impact, score normally:**
   - **9-10**: Government-certified, multiple confirmations
   - **7-8**: Third-party audited (DNV GL, EPA, IEA)
   - **5-6**: Credible self-report with methodology (GHG Protocol)
   - **3-4**: Unverified self-report
   - **1-2**: No quantified impact data

   **Evidence**: "X tons CO2 avoided/year", "Y MWh generated", lifecycle analysis

---

## Scoring Calibration

**Deployment Stage → Overall Score Mapping:**

- **mass_deployment** (GW-scale, years operational) → 7-10
- **commercial_proven** (multi-site, revenue-generating) → 6-8
- **early_commercial** (first deployments, limited scale) → 5-6 ← CRITICAL
- **pilot** (test installations, not revenue) → 3-4
- **lab** (prototype, no deployment) → 1-2

**Important:** If article describes REAL DEPLOYMENT (not just plans) of climate tech, minimum score is 5.0, even if specific performance data is missing.

---

## Scoring Philosophy

**Evidence hierarchy** (from strongest to weakest):
1. **Deployment evidence** (X MW deployed, Y units installed, $Z investment) → Score based on this
2. **Performance data** (efficiency, capacity factor, cost trends) → Nice to have, but not required for basic scoring
3. **Detailed metrics** (LCOE, degradation rates, supply chain) → Only for scores 8+

**Rule:** If article provides deployment evidence (#1) for climate tech, score normally even if #2 and #3 are missing. Do NOT penalize for "lack of specific data" if deployment scale is clear.

**Negative News Handling:**
Score the TECHNOLOGY MATURITY, not the news sentiment.

Examples:
- "Solar company goes bankrupt" → Score the deployed solar tech (7-8), not the business failure
- "EV factory layoffs" → Score the EV technology maturity (6-8), not the workforce reduction
- "Battery recall announced" → Score the deployment scale, note issues in performance dimension

Negative news often indicates DEPLOYED tech facing real-world challenges, which confirms it's past the lab stage.

---

## Gatekeeper Rules

1. **If DEPLOYMENT_MATURITY < 5.0**: Cap overall score at 4.9 (lab/pilot can't support "tech works")
2. **If PROOF_OF_IMPACT < 4.0**: Cap overall score at 3.9 (must have some verified impact)

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

**Low Score (1.6)**: "Startup Unveils Revolutionary Solar Panel with 60% Efficiency (Lab Results)"
- Deployment: 1 (lab only, not deployed)
- Performance: 3 (lab conditions, not real-world)
- Cost: 2 (no cost data)
- Scale: 1 (zero deployed)
- Market: 1 (0% - doesn't exist)
- Readiness: 2 (experimental)
- Supply: 1 (no manufacturing)
- Impact: 1 (zero impact, not deployed)

**OUT OF SCOPE (1.0)**: "The 10 Best Kubernetes Management Tools using AI for 2026"
- Deployment: 1 (generic IT infrastructure - OUT OF SCOPE, not climate tech)
- Performance: 1 (IT performance - OUT OF SCOPE)
- Cost: 1 (IT costs - OUT OF SCOPE)
- Scale: 1 (IT scale - OUT OF SCOPE)
- Market: 1 (IT market - OUT OF SCOPE)
- Readiness: 1 (IT readiness - OUT OF SCOPE)
- Supply: 1 (IT supply chain - OUT OF SCOPE)
- Impact: 1 (no climate impact - OUT OF SCOPE)
- **Reasoning**: Generic IT infrastructure (Kubernetes) has no climate/sustainability focus. Should score 0-2 on ALL dimensions per CRITICAL FILTERS.

---

## Output Format (JSON)

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
  "primary_technology": "solar|wind|batteries|EVs|heat_pumps|hydrogen|other|out_of_scope",
  "deployment_stage": "mass_deployment|commercial_proven|early_commercial|pilot|lab|out_of_scope",
  "confidence": "HIGH|MEDIUM|LOW"
}}

**Be balanced**: Most tech news is vaporware (score 1-4), but real deployments exist. If article shows deployment evidence (MW deployed, units installed, investment amount) FOR CLIMATE TECH, score 5+. Don't penalize for missing detailed metrics if deployment scale is clear.

**CRITICAL REMINDER:** Check inline filters BEFORE scoring each dimension. Generic IT infrastructure, software tools, gaming, and non-climate tech should score 0-2 on ALL dimensions.

DO NOT include any text outside the JSON object.
```

---

## CHANGELOG

**v2.0 (2025-11-14):**
- **BREAKING CHANGE:** Restructured with inline filters (following uplifting v3→v4, investment-risk v1→v2 pattern)
- Removed top-level "SCOPE" section (lines 18-45 in v1)
- Moved OUT OF SCOPE filters INLINE within each dimension definition
- Added explicit Kubernetes example to OUT OF SCOPE examples
- Expected impact: Reduce false positives from 8.3% to <2% (based on uplifting + investment-risk results)
- **Rationale:** Fast models (Gemini Flash) skip top-level instructions. Inline filters force oracle to check scope before scoring.

**v1.0 (2024):**
- Initial compressed prompt with top-level SCOPE section
- Known issue: 7.7-8.3% false positive rate (fast models skip top-level SCOPE filters)

---

**Token estimate**: ~1,800 tokens (v1 was ~1,400 tokens)
