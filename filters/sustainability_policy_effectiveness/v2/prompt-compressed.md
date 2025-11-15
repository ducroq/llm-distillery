# Climate Policy Effectiveness Scoring

**Purpose**: Evaluate climate policies for proven outcomes and replicability
**Version**: 2.0 (with inline filters)
**Target**: Gemini Flash 1.5 / Fast models

**Focus**: ACTUAL GOVERNMENT POLICIES with MEASURED OUTCOMES, not academic proposals

---

## PROMPT TEMPLATE

```
Rate this article on **8 dimensions** (0-10 scale). Focus: ACTUAL POLICY IMPLEMENTATION with MEASURED OUTCOMES.

## Article

**Title**: {title}

**Content**: {text}

---

## Dimensions

### 1. POLICY_OUTCOMES (30%)

**OUT OF SCOPE** if:
- Academic research paper (not actual policy)
- Technical/engineering content (not policy implementation)
- Corporate strategy or voluntary commitments (not government policy)
- Policy proposals without implementation
- Opinion pieces or advocacy

**IF IN SCOPE**: Did emissions/deployment measurably change after policy was implemented?

- **9-10**: Transformative (Norway EVs 5%→90%, emissions fell 50%+)
- **7-8**: Clear impact (emissions fell 20%+, deployment grew significantly)
- **5-6**: Modest outcomes, some positive change
- **3-4**: Policy implemented but no measurable impact yet
- **1-2**: No data, or announced but not implemented

**Evidence Required**: Before/after data, time series, attribution to government policy

### 2. REPLICABILITY (25%)

**OUT OF SCOPE** if:
- Not a government policy
- Unique to one research project or company

**IF IN SCOPE**: Can other jurisdictions copy this policy blueprint?

- **9-10**: Global blueprint (50+ countries copying)
- **7-8**: Highly replicable (10+ jurisdictions adopting)
- **5-6**: Some transferability, requires adaptation
- **3-4**: Very context-dependent, hard to transfer
- **1-2**: Impossible to replicate, unique circumstances

**Evidence**: Countries adopting, clear policy mechanism, works across political systems

### 3. POLITICAL_DURABILITY (15%)

**OUT OF SCOPE** if:
- Not a government policy
- Research methodology or technical framework

**IF IN SCOPE**: Will this survive government change? Institutionalized?

- **9-10**: 30+ years track record, bipartisan, constitutional
- **7-8**: Institutionalized, bipartisan, hard to reverse
- **5-6**: Bipartisan support, some institutional backing
- **3-4**: Single-party legislation, vulnerable
- **1-2**: Executive order, easily reversed, partisan

**Evidence**: Time in effect, cross-party support, legal framework

### 4. SPEED_OF_IMPACT (10%)

**OUT OF SCOPE** if not a government policy

**IF IN SCOPE**: How fast did policy deliver measurable results?

- **9-10**: <1 year (immediate behavior change)
- **7-8**: 1-3 years (rapid deployment response)
- **5-6**: 3-5 years
- **3-4**: 5-10 years to see impact
- **1-2**: >10 years or no results yet

**Evidence**: Time from policy enactment to measurable outcomes

### 5. EQUITY_IMPACT (8%)

**OUT OF SCOPE** if not a government policy

**IF IN SCOPE**: Who benefits? Who pays? Just transition?

- **9-10**: Reparative, addresses historical injustice, helps vulnerable
- **7-8**: Equity-centered, benefits low-income disproportionately
- **5-6**: Some equity provisions (rebates, worker support)
- **3-4**: Neutral, no equity considerations
- **1-2**: Regressive, harms vulnerable populations

**Evidence**: Revenue recycling, just transition provisions, distributional impact

### 6. ENFORCEMENT (7%)

**OUT OF SCOPE** if not a government policy

**IF IN SCOPE**: Are there teeth? Penalties? Monitoring?

- **9-10**: Automatic market enforcement (price signal)
- **7-8**: Strong enforcement, automatic penalties, robust monitoring
- **5-6**: Moderate enforcement, some accountability
- **3-4**: Weak enforcement, rarely penalized
- **1-2**: Voluntary, no penalties, no monitoring

**Evidence**: Penalty structure, compliance rates, monitoring systems

### 7. UNINTENDED_CONSEQUENCES (3%)

**OUT OF SCOPE** if not a government policy

**IF IN SCOPE**: Negative side effects minimized?

- **9-10**: Net positive across all dimensions, no downsides
- **7-8**: Minimal downsides, well-designed
- **5-6**: Minor trade-offs, manageable
- **3-4**: Some significant downsides
- **1-2**: Major negative consequences (economic damage, unfairness)

**Evidence**: Economic impacts, social effects, unintended outcomes

### 8. POLICY_SPREADING (2%)

**OUT OF SCOPE** if not a government policy

**IF IN SCOPE**: Are other governments adopting? Network effect?

- **9-10**: >15 jurisdictions adopted, global standard emerging
- **7-8**: 5-15 jurisdictions adopted
- **5-6**: 2-5 jurisdictions considering/adopting
- **3-4**: Limited interest, no adoption yet
- **1-2**: No one copying, failed experiment

**Evidence**: Government adoption count, diffusion rate, international interest

---

## Gatekeeper Rule

**If POLICY_OUTCOMES < 5.0**: Cap overall score at 4.9 (must have measurable outcomes)

---

## Examples

**HIGH SCORE (9.1)**: "Norway EV Policy: 30 Years Later, 90% Market Share"
- Outcomes: 10 (5% → 90% EV share, transformative government policy)
- Replicability: 9 (15+ countries copying tax structure)
- Durability: 9 (30+ years, all parties support)
- Speed: 7 (took 10+ years but steady growth)
- Equity: 6 (some regressive aspects of tax breaks)
- Enforcement: 8 (automatic via tax system)
- Consequences: 7 (electricity grid challenges manageable)
- Spreading: 9 (15+ countries adopting)

**LOW SCORE (0.0 - OUT OF SCOPE)**: "Machine Learning Framework for Log Anomaly Detection"
- NOT A GOVERNMENT POLICY
- Research paper about technical systems
- Mark all dimensions OUT OF SCOPE
- Overall score: 0.0

**LOW SCORE (2.8)**: "President Announces Net-Zero by 2050 Pledge"
- Outcomes: 1 (no implementation, no data)
- Replicability: 3 (common pledge, no mechanism)
- Durability: 2 (executive announcement, easily reversed)
- Speed: 1 (no action yet)
- Equity: 3 (no provisions)
- Enforcement: 1 (voluntary, no penalties)
- Consequences: 5 (neutral, nothing happened)
- Spreading: 4 (many making similar pledges)

---

## Output Format (JSON)

```json
{{
  "policy_outcomes": {{"score": X, "reasoning": "Brief justification with data OR 'OUT OF SCOPE: not a government policy'"}},
  "replicability": {{"score": X, "reasoning": "..."}},
  "political_durability": {{"score": X, "reasoning": "..."}},
  "speed_of_impact": {{"score": X, "reasoning": "..."}},
  "equity_impact": {{"score": X, "reasoning": "..."}},
  "enforcement_mechanisms": {{"score": X, "reasoning": "..."}},
  "unintended_consequences": {{"score": X, "reasoning": "..."}},
  "policy_spreading": {{"score": X, "reasoning": "..."}},
  "overall_assessment": "1-2 sentence summary OR 'OUT OF SCOPE: not a government policy'",
  "policy_type": "carbon_tax | feed_in_tariff | ev_incentives | mandate | subsidy | regulation | out_of_scope | other",
  "stage": "proven | effective | promising | announced | ineffective | out_of_scope",
  "confidence": "HIGH | MEDIUM | LOW"
}}
```

**CRITICAL**: If this is NOT a government climate policy (e.g., academic paper, technical research, corporate strategy), score ALL dimensions as 0 and mark stage as "out_of_scope".

**Be data-driven**: Require specific before/after data from actual government policy implementation, not just proposals or academic discussions.

DO NOT include any text outside the JSON object.
```

---
