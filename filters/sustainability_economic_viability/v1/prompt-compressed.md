# Climate Economics Scoring

**Purpose**: Evaluate economic competitiveness of climate solutions vs fossil fuels
**Version**: 1.0
**Target**: Gemini Flash 1.5 / Fast models

**Focus**: Cost competitiveness, profitability, job creation, and economic viability of green technologies

---

## PROMPT TEMPLATE

```
Rate this article on **8 dimensions** (0-10 scale). Focus: Economic competitiveness of green vs fossil fuels.

## Article

**Title**: {title}

**Content**: {text}

---

## Dimensions

### 1. COST_COMPETITIVENESS (25%)
LCOE/unit cost vs fossil alternatives

- **9-10**: Cheaper than fossil (LCOE <$30/MWh vs coal $60), disrupting incumbents
- **7-8**: Cost parity, subsidy-independent, competitive standalone
- **5-6**: Approaching parity, costs declining steadily, modest subsidies (<30%)
- **3-4**: 2-5x fossil costs, heavy subsidies needed
- **1-2**: >5x fossil costs, no clear path to competitiveness

**Evidence**: LCOE ($/MWh), $/kg, $/unit, vs fossil baseline, subsidy levels

### 2. PROFITABILITY (20%)
Are green companies making money?

- **9-10**: Highly profitable (margin >15%), outperforming fossil competitors
- **7-8**: Profitable, positive margins, sustainable business model
- **5-6**: Path to profitability, positive unit economics, revenue growing
- **3-4**: Unprofitable, cash burn, dependent on funding rounds
- **1-2**: Massive losses, bankruptcy risk, no revenue

**Evidence**: Operating margin %, quarterly profits, revenue growth %, cash flow

### 3. JOB_CREATION (15%)
Green jobs vs fossil jobs displaced

- **9-10**: Massive job creation (100k+), quality jobs, wage premium over fossil
- **7-8**: Net positive jobs (more green than fossil displaced)
- **5-6**: Net neutral (green jobs ≈ fossil jobs lost)
- **3-4**: Some green jobs but fewer than fossil displaced
- **1-2**: Net job losses, automation without replacement

**Evidence**: # jobs created, wage comparison, geographic distribution

### 4. STRANDED_ASSETS (15%)
Fossil infrastructure becoming uneconomical

- **9-10**: Widespread stranding (coal/gas plants closing, oil writedowns across industry)
- **7-8**: Significant stranding (coal plants retired early due to renewables)
- **5-6**: Some early retirements, writedowns beginning
- **3-4**: Fossil stable, no stranding signals yet
- **1-2**: Fossil assets increasing in value

**Evidence**: Early closures, asset writedowns, decommissioning announcements

### 5. INVESTMENT_FLOWS (10%)
Capital moving to green tech

- **9-10**: Investment dominates (green > fossil in new capital, $100B+ sector-wide)
- **7-8**: Major investment ($10-100B annually in sector)
- **5-6**: Growing investment, corporate R&D increasing
- **3-4**: Limited investment, niche venture funding
- **1-2**: Divestment from green, capital fleeing

**Evidence**: $B invested, # deals, YoY growth, VC/corporate funding

### 6. PAYBACK_PERIOD (8%)
ROI timeline for green investments

- **9-10**: <2 years payback (e.g. LED bulbs, efficient motors)
- **7-8**: 2-5 years (e.g. heat pumps, rooftop solar in sunny regions)
- **5-6**: 5-10 years (e.g. commercial solar, EVs with fuel savings)
- **3-4**: 10-20 years (e.g. offshore wind, large infrastructure)
- **1-2**: >20 years or never pays back

**Evidence**: Payback calculation, ROI %, NPV data

### 7. SUBSIDY_DEPENDENCE (4%)
Does it need subsidies? (Lower dependence = higher score)

- **9-10**: Zero subsidies, profitable standalone
- **7-8**: Minimal subsidies (<10% of economics)
- **5-6**: Modest subsidies (~20-30%), approaching independence
- **3-4**: Heavy subsidies (>50% of economics)
- **1-2**: 100% subsidy-dependent, unviable without

**Evidence**: Subsidy levels, phase-out plans, competitiveness without support

### 8. ECONOMIC_MULTIPLIER (3%)
Broader economic benefits beyond direct costs

- **9-10**: Transformative benefits (energy independence + health + resilience)
- **7-8**: Significant multiplier (e.g. healthcare savings from air quality)
- **5-6**: Some co-benefits (e.g. reduced oil imports, grid reliability)
- **3-4**: Neutral, no broader benefits identified
- **1-2**: Negative externalities, economic drag

**Evidence**: Energy security, healthcare cost reductions, economic resilience

---

## Gatekeeper Rule

**If COST_COMPETITIVENESS < 5.0**: Cap overall score at 4.9 (must be approaching parity)

---

## Examples

**High Score (8.5)**: "Solar Now Cheapest Electricity in History at $20/MWh - IEA"
- Cost: 10 (cheaper than coal $60/MWh, gas $40/MWh)
- Profitability: 9 (solar companies highly profitable)
- Jobs: 7 (4M solar jobs globally, growing)
- Stranded: 8 (coal plants closing early)
- Investment: 9 ($500B/year renewable investment)
- Payback: 6 (5-8 years for commercial solar)
- Subsidies: 8 (minimal subsidies needed)
- Multiplier: 7 (energy security, health benefits)

**Low Score (3.2)**: "Green Hydrogen Costs $6/kg vs $2/kg Fossil, Needs Subsidies"
- Cost: 3 (3x fossil costs)
- Profitability: 2 (industry unprofitable)
- Jobs: 5 (some jobs created)
- Stranded: 3 (fossil H2 stable)
- Investment: 6 (growing but small)
- Payback: 3 (>15 years)
- Subsidies: 2 (heavily subsidy-dependent)
- Multiplier: 5 (some decarbonization benefit)

---

## Output Format (JSON)

```json
{{
  "cost_competitiveness": {{"score": X, "reasoning": "Brief justification with data"}},
  "profitability": {{"score": X, "reasoning": "..."}},
  "job_creation": {{"score": X, "reasoning": "..."}},
  "stranded_assets": {{"score": X, "reasoning": "..."}},
  "investment_flows": {{"score": X, "reasoning": "..."}},
  "payback_period": {{"score": X, "reasoning": "..."}},
  "subsidy_dependence": {{"score": X, "reasoning": "..."}},
  "economic_multiplier": {{"score": X, "reasoning": "..."}},
  "overall_assessment": "1-2 sentence economic summary",
  "primary_sector": "solar | wind | EVs | batteries | hydrogen | heat_pumps | other",
  "economic_stage": "superior | competitive | approaching_parity | subsidy_dependent | unviable",
  "confidence": "HIGH | MEDIUM | LOW"
}}
```

**Be data-driven**: Require specific cost/financial data, not just claims or advocacy.

DO NOT include any text outside the JSON object.
```

---
