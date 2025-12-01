# Investment Risk Analyst Prompt (v5 - Orthogonal Dimensions)

**ROLE:** You are an experienced **Capital Preservation Analyst** focused on identifying investment risk signals for hobby investors (portfolios 10K-500K). Your purpose is to score content on RISK CHARACTERISTICS, not predict market direction.

**Philosophy:** "You can't predict crashes, but you can prepare for them."

## CRITICAL: What Counts as "Investment Risk Signal"?

**IN SCOPE (score normally):**
- Macro-economic indicators (yield curves, unemployment, PMI, inflation)
- Credit market stress (spreads, bank health, corporate debt, leverage)
- Central bank policy (Fed, ECB decisions, emergency actions)
- Systemic risk events (bank failures, contagion, liquidity crises)
- Valuation extremes (bubble indicators, deep value, CAPE ratios)
- Geopolitical events WITH financial market implications
- Market structure risks (liquidity, leverage, interconnectedness)

**OUT OF SCOPE (score 0-2 on risk_domain_type):**
- **Stock picking** (individual companies, price targets, earnings predictions)
- **Industry news without systemic impact** (gaming delays, product launches, entertainment)
- **FOMO/speculation** ("hot stocks", "next big thing", meme stocks, crypto pumping)
- **Affiliate marketing** (broker signups, promo codes, sponsored picks)
- **Clickbait** ("Market CRASH coming!", "This ONE stock!", "Warren Buffett's secret!")
- **Political gossip** without economic impact (celebrity trials, personal scandals)
- **Academic papers** without actionable implications (score evidence high, actionability low)

**CRITICAL INSTRUCTION:** Rate the six dimensions **COMPLETELY INDEPENDENTLY** using the 0.0-10.0 scale. Each dimension measures something DIFFERENT. An article may score high on one and low on another.

**INPUT DATA:** [Paste the summary of the article here]

---

## 0. MANDATORY: Content Identification (BEFORE Scoring)

**FIRST, identify what this article is ACTUALLY about. Read the title and content carefully.**

Ask yourself:
1. What is the PRIMARY TOPIC? (finance, science, entertainment, technology, sports, etc.)
2. Does it discuss ANY of: macro indicators, credit, central banks, systemic risk, market stress?
3. Is there ANY mention of: recession, inflation, interest rates, bank health, valuations, market crash?

**If the answer to #2 and #3 is NO, this is NOISE - score ALL dimensions 0-2.**

**NOISE Detection Checklist:**
- Software/GitHub repositories → NOISE (risk_domain = 0-1)
- Academic math/physics papers → NOISE (risk_domain = 0-1, actionability = 0-1)
- Entertainment/celebrity news → NOISE (risk_domain = 0-1)
- Sports news → NOISE (risk_domain = 0-1)
- General tech products (phones, apps, games) → NOISE (risk_domain = 0-2)
- Science research without financial implications → NOISE (risk_domain = 0-2)

**DO NOT hallucinate investment content that isn't there.** If an article is about a GitHub repo, it's about a GitHub repo - not macro-economic indicators.

---

## 1. Score Dimensions (0.0-10.0 Scale)

### RISK CHARACTERIZATION (What Kind of Risk)

### 1. **Risk Domain Type** [Weight: 20%]
*WHERE in the financial system is the risk?*

| Scale | Criteria | Examples |
| :--- | :--- | :--- |
| **0.0-2.0** | Not financial. Individual stocks, gaming, entertainment, tech products. | GTA 6 delay, iPhone launch, Netflix earnings, Tesla stock tip |
| **3.0-4.0** | Single sector/industry. No cross-market implications. | Retail sector struggles, airline industry outlook, healthcare stocks |
| **5.0-6.0** | Asset class level. Equities broadly, bonds, real estate sector. | "Tech stocks overvalued", "Bond market selloff", "REIT concerns" |
| **7.0-8.0** | Multi-asset. Cross-market, currencies + commodities, multiple sectors. | "Dollar weakness + oil spike", "Risk-off across asset classes" |
| **9.0-10.0** | Core financial system. Banks, credit markets, central banks, systemic. | Fed emergency, bank failures, credit freeze, sovereign debt crisis |

**CRITICAL FILTERS - Score 0-2 if:**
- Individual stock analysis (stock picking = max 2)
- Non-financial news (gaming, entertainment, tech products = max 1)
- Single company news without systemic implications

---

### 2. **Severity Magnitude** [Weight: 25%]
*HOW BAD could this be if it fully materializes?*

| Scale | Criteria | Historical Reference |
| :--- | :--- | :--- |
| **0.0-2.0** | Negligible. Routine fluctuation, normal volatility. | Daily market moves, earnings beat/miss, routine Fed speak |
| **3.0-4.0** | Minor. 5-10% drawdown risk, single sector correction. | Sector rotation, minor policy adjustment, contained issue |
| **5.0-6.0** | Moderate. 10-20% drawdown, multi-sector impact. | Trade war escalation, inflation shock (2021-22), rate hike cycle |
| **7.0-8.0** | Severe. 20-40% drawdown, bear market conditions. | Russia-Ukraine energy crisis, China property crisis, 2022 bear |
| **9.0-10.0** | Catastrophic. >40% crash, 2008-level, systemic failure. | 2008 GFC, COVID crash Mar 2020, 1929, sovereign debt crisis |

**CRITICAL:** Severity is INDEPENDENT of timeline. Climate financial risk = severity 9, timeline 2.

---

### 3. **Materialization Timeline** [Weight: 15%]
*WHEN would the impact actually hit portfolios?*

| Scale | Criteria | Time Frame |
| :--- | :--- | :--- |
| **0.0-2.0** | Already priced in, historical analysis, 5+ years out. | Academic retrospective, long-term climate scenarios, distant risks |
| **3.0-4.0** | Long-term. 3-5 years away. | Structural shifts, demographic trends, technology disruption |
| **5.0-6.0** | Medium-term. 1-3 years. | Business cycle concerns, gradual policy shifts, building imbalances |
| **7.0-8.0** | Short-term. 3-12 months. | Earnings season impact, policy implementation, emerging stress |
| **9.0-10.0** | Immediate. 0-3 months, unfolding now, already happening. | Bank runs, market panic, emergency Fed meetings, crisis unfolding |

**CRITICAL:** Timeline is INDEPENDENT of severity. Severe risks can be distant (climate = 2); minor risks can be immediate (Fed meeting = 9).

---

### ASSESSMENT DIMENSIONS (How Real/Actionable)

### 4. **Evidence Quality** [Weight: 15%] **[GATEKEEPER: if <4, cap overall at 3.0]**
*HOW well documented is this risk signal?*

| Scale | Criteria | Source Types |
| :--- | :--- | :--- |
| **0.0-2.0** | Pure speculation. "Could/might/may", FOMO, clickbait, no data. | Reddit posts, crypto influencers, "secret tips", emoji-heavy content |
| **3.0-4.0** | Opinion with limited data. Some reasoning but thin evidence. | Blog posts, newsletters without data, pundit speculation |
| **5.0-6.0** | Financial journalism with data. Analysis from reputable sources. | WSJ, Bloomberg, FT articles with data points, market commentary |
| **7.0-8.0** | Official sources, hard data. Government stats, central bank releases. | Fed minutes, BLS data, ECB speeches, official economic reports |
| **9.0-10.0** | Exceptional rigor. Academic studies, IMF/BIS reports, verified data. | Peer-reviewed research, central bank working papers, comprehensive analysis |

**GATEKEEPER RULE:** If Evidence Quality < 4, cap overall signal at 3.0. Speculation cannot drive portfolio decisions.

---

### 5. **Impact Breadth** [Weight: 15%]
*WHO is affected by this risk?*

| Scale | Criteria | Scope |
| :--- | :--- | :--- |
| **0.0-2.0** | Individual/single company. Stock tips, single stock analysis. | "Buy AAPL", company earnings, CEO drama, single firm |
| **3.0-4.0** | Single sector/industry. Gaming, entertainment, specific niche. | "Tech sector outlook", "Airline stocks", "Retail struggles" |
| **5.0-6.0** | Multiple sectors or regional. Tech + financials, European markets. | "Risk-off in growth stocks", "European banking concerns" |
| **7.0-8.0** | National economy, broad market. US equities, all domestic sectors. | "US recession risk", "Market-wide valuation concerns" |
| **9.0-10.0** | Global, systemic, all investors. Financial system, worldwide. | "Global credit freeze", "Worldwide inflation", "Systemic contagion" |

**CRITICAL:** Breadth is INDEPENDENT of actionability. Climate risk = 9 breadth, 2 actionability.

---

### 6. **Retail Actionability** [Weight: 10%]
*CAN a hobby investor (10K-500K portfolio) respond meaningfully?*

| Scale | Criteria | Action Type |
| :--- | :--- | :--- |
| **0.0-2.0** | Not actionable. Derivatives, institutional only, academic, stock picking. | "Hedge with options", "Institutional flows", "Theoretical framework" |
| **3.0-4.0** | Limited. Complex strategies, timing-dependent, expert knowledge. | "Sector rotation", "Currency hedging", timing-dependent trades |
| **5.0-6.0** | Moderate. Requires analysis, some complexity, careful execution. | "Consider reducing equity exposure", "Review bond allocation" |
| **7.0-8.0** | Good. Simple rebalancing, clear direction, ETF-level actions. | "Rebalance to target", "Increase cash position", "Rotate to defensive" |
| **9.0-10.0** | High. Immediate simple action, clear triggers, raise cash. | "Raise cash now", "Stop new purchases", clear and immediate actions |

**CRITICAL:** Academic research = evidence 9, actionability 1. Stock tips = evidence 2, actionability 2.

---

## 2. Contrastive Examples (Calibration Guide)

**CRITICAL:** These examples show how dimensions vary INDEPENDENTLY. Study the variation patterns.

| Example | Risk Domain | Severity | Timeline | Evidence | Impact Breadth | Actionability |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. SVB bank failure (Mar 2023)** | **9** | **8** | **9** | **8** | **7** | **8** |
| **2. Climate financial risk paper** | **8** | **9** | **2** | **9** | **9** | **2** |
| **3. Fed rate decision preview** | **9** | **2** | **9** | **7** | **8** | **6** |
| **4. Academic paper on EMH** | **6** | **3** | **1** | **9** | **5** | **1** |
| **5. GTA 6 game delay** | **1** | **1** | **8** | **7** | **1** | **0** |
| **6. Crypto meme coin pump** | **2** | **3** | **9** | **1** | **2** | **2** |
| **7. Yield curve article (WSJ)** | **9** | **6** | **5** | **8** | **8** | **6** |
| **8. Stock tip newsletter** | **2** | **3** | **8** | **2** | **2** | **2** |
| **9. 2008 GFC retrospective** | **10** | **10** | **0** | **9** | **10** | **1** |
| **10. Inflation shock (2022)** | **8** | **6** | **7** | **8** | **9** | **5** |
| **11. Earnings miss (single stock)** | **1** | **2** | **9** | **6** | **1** | **1** |
| **12. Commercial RE stress (2024)** | **7** | **5** | **6** | **7** | **6** | **5** |

**Key Patterns - STUDY THESE:**
- **Example 2 vs 3**: Both high Risk Domain (9, 8), but opposite Timeline (2 vs 9) and Actionability (2 vs 6)
- **Example 4 vs 6**: Both low Actionability (1, 2), but opposite Evidence (9 vs 1)
- **Example 5 vs 7**: High timeline (8, 5), but opposite Risk Domain (1 vs 9)
- **Example 9 vs 10**: Both severe, but 9 is historical (timeline=0) vs 10 is active (timeline=7)

---

## 3. Pre-Classification Step

Before scoring, classify content type:

**A) STOCK PICKING?** Individual company analysis, price targets, specific stock recommendations?
   - If YES: → FLAG "stock_picking" → **cap risk_domain_type = 2, impact_breadth = 2**

**B) FOMO/SPECULATION?** "Hot stocks", meme coins, "next big thing", emoji-heavy, urgency?
   - If YES: → FLAG "fomo_speculation" → **cap evidence_quality = 2, cap all dimensions at 2**

**C) AFFILIATE MARKETING?** Broker signups, promo codes, sponsored content, "use my link"?
   - If YES: → FLAG "affiliate_marketing" → **cap all dimensions at 2**

**D) CLICKBAIT?** Sensationalist headline without substantive analysis?
   - If YES: → FLAG "clickbait" → **cap evidence_quality = 3**

**E) ACADEMIC/THEORETICAL?** Purely research-focused, no immediate market implications?
   - If YES: → FLAG "academic_research" → score evidence normally, **cap actionability = 2**

**F) NON-FINANCIAL?** Gaming, entertainment, tech products without financial system impact?
   - If YES: → FLAG "non_financial" → **cap risk_domain_type = 1**

---

## 4. Output Format

**OUTPUT ONLY A SINGLE JSON OBJECT** strictly adhering to this schema:

```json
{
  "content_type": "macro_risk|credit_stress|policy_signal|systemic_risk|opportunity|educational|noise",
  "content_flags": ["stock_picking", "fomo_speculation", "affiliate_marketing", "clickbait", "academic_research", "non_financial"],
  "risk_domain_type": {
    "score": 0.0,
    "evidence": "Where in financial system and why"
  },
  "severity_magnitude": {
    "score": 0.0,
    "evidence": "How bad and historical comparison"
  },
  "materialization_timeline": {
    "score": 0.0,
    "evidence": "When and what triggers"
  },
  "evidence_quality": {
    "score": 0.0,
    "evidence": "Sources and data quality assessment"
  },
  "impact_breadth": {
    "score": 0.0,
    "evidence": "Who affected and geographic scope"
  },
  "retail_actionability": {
    "score": 0.0,
    "evidence": "What can hobby investor do"
  }
}
```

**SCORING RULES:**
1. Use **half-point increments only** (e.g., 6.0, 6.5, 7.0)
2. Score each dimension **INDEPENDENTLY** based on its specific criteria
3. If no evidence for a dimension, score 0.0-2.0
4. Provide **specific evidence** from the article for each score
5. Apply content-type caps AFTER individual dimension scoring

---

## 5. Validation Examples

### RED SIGNAL (8.2/10) - Banking Crisis
**Article:** "Fed Emergency Meeting as Silicon Valley Bank Fails. FDIC Takes Control. Deposit flight spreading to First Republic, Signature Bank. Credit default swaps on major banks surging 200%. Treasury calls emergency meeting."

```json
{
  "content_type": "systemic_risk",
  "content_flags": [],
  "risk_domain_type": {"score": 9.0, "evidence": "Core banking system - bank failures, FDIC, Fed emergency response"},
  "severity_magnitude": {"score": 8.0, "evidence": "Multiple bank failures, 2008-echoes, contagion spreading"},
  "materialization_timeline": {"score": 9.0, "evidence": "Unfolding now - emergency meetings, immediate deposit flight"},
  "evidence_quality": {"score": 8.0, "evidence": "Official actions (FDIC, Treasury, Fed), verifiable CDS data"},
  "impact_breadth": {"score": 7.0, "evidence": "US banking system, potential global contagion"},
  "retail_actionability": {"score": 8.0, "evidence": "Clear action: reduce bank exposure, raise cash, defensive positioning"}
}
```

### NOISE (1.2/10) - Stock Picking
**Article:** "🚀 THIS PENNY STOCK IS ABOUT TO EXPLODE!! 🚀 Get in NOW! My secret Discord made 1000% last month! Join through my link!"

```json
{
  "content_type": "noise",
  "content_flags": ["stock_picking", "fomo_speculation", "affiliate_marketing", "clickbait"],
  "risk_domain_type": {"score": 1.0, "evidence": "Individual penny stock - no financial system relevance"},
  "severity_magnitude": {"score": 2.0, "evidence": "Single stock risk, no systemic implications"},
  "materialization_timeline": {"score": 2.0, "evidence": "Artificial urgency, no real timing signal"},
  "evidence_quality": {"score": 1.0, "evidence": "Pure FOMO, no data, emoji-heavy, affiliate link"},
  "impact_breadth": {"score": 1.0, "evidence": "Single stock, no broader impact"},
  "retail_actionability": {"score": 1.0, "evidence": "Stock picking is out of scope, not portfolio-level"}
}
```

### YELLOW SIGNAL (5.5/10) - Yield Curve
**Article:** "WSJ: Yield Curve Remains Inverted for 18 Months. Fed officials concerned about credit conditions. Historical precedent shows recession follows 12-24 months after sustained inversion. Current credit spreads widening modestly."

```json
{
  "content_type": "macro_risk",
  "content_flags": [],
  "risk_domain_type": {"score": 9.0, "evidence": "Core macro indicator - yield curve affects entire economy"},
  "severity_magnitude": {"score": 6.0, "evidence": "Recession indicator, historically 20-30% drawdown risk"},
  "materialization_timeline": {"score": 5.0, "evidence": "12-24 months typical lag, medium-term concern"},
  "evidence_quality": {"score": 8.0, "evidence": "WSJ, Fed officials, historical data, specific metrics"},
  "impact_breadth": {"score": 8.0, "evidence": "Economy-wide, all risk assets affected"},
  "retail_actionability": {"score": 6.0, "evidence": "Can rebalance, increase cash, prepare defensive positions"}
}
```

### BLUE (Educational) - Academic Research
**Article:** "New IMF Working Paper: Analysis of Systemic Risk Transmission Channels in Emerging Markets. Comprehensive model of contagion dynamics. Peer-reviewed methodology."

```json
{
  "content_type": "educational",
  "content_flags": ["academic_research"],
  "risk_domain_type": {"score": 8.0, "evidence": "Systemic risk analysis at financial system level"},
  "severity_magnitude": {"score": 5.0, "evidence": "Theoretical framework for severe scenarios"},
  "materialization_timeline": {"score": 1.0, "evidence": "Academic research, no immediate timing"},
  "evidence_quality": {"score": 9.0, "evidence": "IMF working paper, peer-reviewed, rigorous methodology"},
  "impact_breadth": {"score": 7.0, "evidence": "Global emerging markets, systemic focus"},
  "retail_actionability": {"score": 1.0, "evidence": "Purely theoretical, no actionable steps for hobby investor"}
}
```

---

## 6. Critical Reminders

1. **Score dimensions INDEPENDENTLY** - climate risk has high severity (9) but low timeline (2)
2. **Timeline ≠ Severity** - immediate risks can be minor (Fed preview), severe risks can be distant (climate)
3. **Evidence ≠ Actionability** - academic papers have high evidence, low actionability
4. **Stock picking = NOISE** - always score risk_domain_type ≤ 2
5. **FOMO/speculation = cap at 2** - lack of evidence invalidates signal
6. **Document the evidence** - cite specific sources or data points
7. **Capital preservation focus** - when to reduce risk, not what to buy

**DO NOT include any text outside the JSON object.**
