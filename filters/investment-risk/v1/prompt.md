# Investment Risk: Capital Preservation Filter

**Purpose**: Identify investment risk signals for defense-first portfolio management focused on capital preservation, not speculation.

**Version**: 1.0
**Created**: 2025-10-29
**Target LLM**: Claude 3.5 Sonnet / Gemini 1.5 Pro
**Use Case**: Generate ground truth labels for fine-tuning local models

**Semantic Framework**: Focuses on RISK SIGNALS and CAPITAL PRESERVATION
- Distinguishes genuine risk from noise and media hype
- Detects FOMO marketing, pump-and-dump schemes, speculation narratives
- Prioritizes macro risk over individual stock picks
- Values evidence-based analysis over hot tips

**Core Philosophy**: "You can't predict crashes, but you can prepare for them."

---

## THE CAPITAL PRESERVATION FRAMEWORK

### Defense-First Investing
Traditional investor tools focus on **offense** (what to buy, stock picks, "10 stocks now").
This filter focuses on **defense** (when to reduce risk, portfolio protection, capital preservation).

**The Math**: Losing 50% requires 100% gain to recover. Avoiding that loss is worth more than finding the next 10-bagger.

### Signal Classification: RED / YELLOW / GREEN / BLUE

**🔴 RED FLAGS** (Act Now - Reduce Risk Immediately):
- Yield curve inversion + recession indicators converging
- Banking crisis signals (credit defaults, liquidity stress)
- Policy errors (unexpected rate hikes, currency crisis)
- Systemic failures (Lehman moment, contagion risk)
- **Action**: Reduce equity exposure, raise cash, defensive positioning

**🟡 YELLOW WARNINGS** (Monitor Closely - Prepare for Defense):
- Rising unemployment with credit stress
- Extreme valuations without catalysts for correction
- Policy uncertainty (election risk, regulatory changes)
- Credit market deterioration (widening spreads, covenant-lite deals)
- **Action**: Review portfolio, prepare rebalancing, reduce speculative positions

**🟢 GREEN OPPORTUNITIES** (Consider Buying - Value Emerging):
- Extreme fear (VIX >40, capitulation selling)
- Valuations at historical lows with improving fundamentals
- Policy support kicking in (QE, fiscal stimulus)
- Oversold quality assets (baby with bathwater)
- **Action**: Dollar-cost average in, buy quality at discount, rebalance to target allocation

**🔵 BLUE CONTEXT** (Understand - No Immediate Action):
- Long-term trends and structural changes
- Academic research on investing/economics
- Historical pattern analysis and lessons
- Market mechanics and behavioral finance insights
- **Action**: Educate yourself, refine framework, no portfolio changes

**⚫ NOISE** (Ignore - No Signal):
- Stock tips and "hot picks"
- Individual stock movements without macro relevance
- Celebrity investor opinions without data
- FOMO marketing and pump-and-dump schemes
- **Action**: Ignore completely

---

## PROMPT TEMPLATE

```
Analyze this article for investment risk signals based on CAPITAL PRESERVATION and MACRO RISK EVIDENCE, not stock picks or speculation.

ARTICLE:
Title: {title}
Source: {source}
Published: {published_date}
Text: {text}

STEP 1: Pre-classification Filters

A) FOMO / SPECULATION FILTER
Is this primarily about: "hot stocks", "stocks to buy now", "next big thing", meme stocks, crypto pumping, "don't miss out"?
- If YES and NO macro risk analysis/capital preservation focus → FLAG as "speculation_noise" (signal_tier = NOISE)

B) STOCK PICKING FILTER
Is this about: individual stock analysis, specific buy recommendations, earnings predictions, technical patterns for single stocks?
- If YES and NOT about systemic risk/sector-wide implications → FLAG as "stock_picking" (signal_tier = NOISE)

C) AFFILIATE / CONFLICT FILTER
Does this contain: "sign up with this broker", "buy through my link", promotional codes, sponsored stock picks?
- If YES → FLAG as "affiliate_conflict" (signal_tier = NOISE, credibility_penalty)

D) CLICKBAIT / SENSATIONALISM FILTER
Headlines like: "Market CRASH coming!", "This ONE stock!", "Warren Buffett's secret!", "Hidden gem!"?
- If YES and NO substantive analysis → FLAG as "clickbait" (max_evidence_quality = 3)

STEP 2: Evaluate Risk Signal Dimensions (score 0-10 for each)

1. **Macro Risk Severity**: Is there evidence of systemic economic/financial risk?
   - Economic indicators: Recession signals (yield curve, unemployment, PMI, LEI)
   - Credit markets: Bank stress, credit defaults, liquidity crises
   - Policy errors: Central bank mistakes, fiscal crises, currency instability
   - Geopolitical: War, trade conflict, political instability
   - (0-2: No macro risk | 3-4: Minor concerns | 5-6: Moderate risk developing | 7-8: Serious risk | 9-10: Crisis unfolding)

2. **Credit Market Stress**: Are there signs of credit market deterioration?
   - Credit spreads: Widening high-yield spreads, investment grade stress
   - Bank health: Capital adequacy, deposit flight, interbank lending freeze
   - Corporate debt: Covenant-lite concerns, refinancing risk, default rates
   - Leverage: Excessive borrowing, margin debt, financial sector leverage
   - (0-2: Healthy credit markets | 3-4: Some stress | 5-6: Moderate deterioration | 7-8: Serious stress | 9-10: Credit crisis)

3. **Market Sentiment Extremes**: Is sentiment at dangerous extremes?
   - Fear: VIX levels, put/call ratios, capitulation selling
   - Greed: FOMO, retail speculation, "can't lose" mentality
   - Complacency: Low volatility, ignored risks, "this time is different"
   - Positioning: Fund flows, retail vs institutional, leverage levels
   - (0-2: Balanced sentiment | 3-4: Slight tilt | 5-6: Moderate extreme | 7-8: Dangerous extreme | 9-10: Panic or euphoria)

4. **Valuation Risk**: Are assets priced for perfection or extreme pessimism?
   - Equity valuations: P/E ratios, CAPE, price-to-sales, ERP
   - Bond valuations: Real yields, yield curve shape, credit spreads
   - Asset class comparison: Stocks vs bonds vs cash, relative attractiveness
   - Historical context: Current vs historical norms, mean reversion potential
   - (0-2: Attractive valuations | 3-4: Fair value | 5-6: Somewhat expensive | 7-8: Very expensive | 9-10: Bubble territory)
   - (GREEN signal: 0-3 | BLUE: 4-6 | YELLOW: 7-8 | RED: 9-10)

5. **Policy/Regulatory Risk**: Are policy changes creating risk or opportunity?
   - Monetary policy: Rate hikes/cuts, QE/QT, central bank errors
   - Fiscal policy: Stimulus or austerity, deficit concerns, tax changes
   - Regulatory changes: New rules affecting markets/sectors
   - Political risk: Elections, regime change, policy uncertainty
   - (0-2: Supportive policy | 3-4: Neutral | 5-6: Some uncertainty | 7-8: Concerning changes | 9-10: Policy crisis)

6. **Systemic Risk**: Is there potential for contagion/cascading failures?
   - Interconnectedness: Bank linkages, counterparty risk, derivatives
   - Leverage in system: Margin debt, repo markets, shadow banking
   - Liquidity: Market depth, bid-ask spreads, flash crash risk
   - Tail risk: Black swan potential, correlation breakdown
   - (0-2: Resilient system | 3-4: Normal risks | 5-6: Some fragility | 7-8: Significant fragility | 9-10: Lehman-moment risk)

7. **Evidence Quality**: How credible is the analysis?
   - Data: Hard economic data vs opinion, official stats vs blog posts
   - Sources: Central banks, academics, respected analysts vs random commentators
   - Logic: Sound reasoning vs conspiracy theories, cause-effect clear
   - Track record: Historical accuracy vs perma-bears/perma-bulls
   - (0-2: Poor evidence | 3-4: Weak | 5-6: Moderate | 7-8: Strong evidence | 9-10: Exceptional evidence)
   - (GATEKEEPER: If <5, cannot be RED signal)

8. **Actionability for Hobby Investors**: Can a hobby investor act on this?
   - Time horizon: Days (not actionable) vs weeks/months (actionable)
   - Portfolio level: Asset allocation vs individual stock (prefer allocation)
   - Complexity: Simple rebalancing vs complex derivatives (prefer simple)
   - Cost: Low-cost actions vs expensive trades vs frequent trading
   - (0-2: Not actionable | 3-4: Limited | 5-6: Moderately actionable | 7-8: Very actionable | 9-10: Clear simple action)

STEP 3: Classify Signal Tier

Based on dimension scores and content analysis, classify into ONE tier:

**🔴 RED FLAG** (Reduce Risk Immediately):
- Macro Risk ≥7 OR Credit Stress ≥7 OR Systemic Risk ≥8
- Evidence Quality ≥5 (gatekeeper)
- Multiple serious risks converging
- Actionability ≥5 (must be actionable)

**🟡 YELLOW WARNING** (Monitor Closely, Prepare):
- Macro Risk 5-6 OR Credit Stress 5-6 OR Valuation Risk 7-8
- Evidence Quality ≥5
- Risk developing but not imminent
- Actionability ≥4

**🟢 GREEN OPPORTUNITY** (Consider Buying):
- Sentiment Extremes ≥7 (extreme fear) AND Valuation Risk 0-3 (cheap)
- OR Market recovering from RED/YELLOW with improving fundamentals
- Evidence Quality ≥6
- Actionability ≥5

**🔵 BLUE CONTEXT** (Understand, No Action):
- Educational content, historical analysis, behavioral finance
- Long-term trends without immediate portfolio implications
- Improves decision-making framework but no trades needed

**⚫ NOISE** (Ignore):
- Flagged by pre-filters (speculation, stock picking, affiliate, clickbait)
- Individual stock tips without macro relevance
- Low evidence quality (<4) on risk claims

STEP 4: Investment Intelligence Metadata

**Risk Indicators** (mark true/false):
- yield_curve_inversion
- recession_indicators_converging
- credit_spread_widening
- bank_stress_signals
- policy_error_risk
- extreme_sentiment (fear or greed)
- valuation_extreme (expensive or cheap)
- systemic_fragility

**Asset Classes Affected**:
- equities (stocks)
- fixed_income (bonds)
- credit (corporate bonds, high yield)
- currencies
- commodities
- real_estate
- cash_equivalents

**Time Horizon**:
- immediate (0-3 months)
- short_term (3-12 months)
- medium_term (1-3 years)
- long_term (3+ years)

**Geographic Scope**:
- us (United States)
- europe (European Union)
- global (worldwide)
- emerging_markets
- specific_country

**Recommended Actions** (for actionability):
- increase_cash (reduce equity exposure)
- rebalance_to_target (back to allocation)
- reduce_risk_assets (defensive positioning)
- consider_buying_quality (value emerging)
- monitor_closely (no action yet)
- no_action_needed (context only)

STEP 5: Output JSON

Respond with ONLY valid JSON in this exact format:
{{
  "signal_tier": "RED|YELLOW|GREEN|BLUE|NOISE",
  "signal_strength": <0-10>,

  "macro_risk_severity": <0-10>,
  "credit_market_stress": <0-10>,
  "market_sentiment_extremes": <0-10>,
  "valuation_risk": <0-10>,
  "policy_regulatory_risk": <0-10>,
  "systemic_risk": <0-10>,
  "evidence_quality": <0-10>,
  "actionability": <0-10>,

  "risk_indicators": {{
    "yield_curve_inversion": <true|false>,
    "recession_indicators_converging": <true|false>,
    "credit_spread_widening": <true|false>,
    "bank_stress_signals": <true|false>,
    "policy_error_risk": <true|false>,
    "extreme_sentiment": <true|false>,
    "valuation_extreme": <true|false>,
    "systemic_fragility": <true|false>
  }},

  "asset_classes_affected": {{
    "equities": <true|false>,
    "fixed_income": <true|false>,
    "credit": <true|false>,
    "currencies": <true|false>,
    "commodities": <true|false>,
    "real_estate": <true|false>,
    "cash_equivalents": <true|false>
  }},

  "time_horizon": "immediate|short_term|medium_term|long_term",
  "geographic_scope": ["us", "europe", "global", "emerging_markets"],

  "recommended_actions": ["increase_cash", "rebalance_to_target", "reduce_risk_assets", "consider_buying_quality", "monitor_closely", "no_action_needed"],

  "flags": {{
    "speculation_noise": <true|false>,
    "stock_picking": <true|false>,
    "affiliate_conflict": <true|false>,
    "clickbait": <true|false>
  }},

  "reasoning": "<2-3 sentences: what risk signal, what evidence, what action for hobby investors>",

  "key_risk_metrics": ["<metric1 with data>", "<metric2>"],
  "similar_historical_periods": ["<period1>", "<period2>"],
  "expert_sources_cited": ["<source1>", "<source2>"]
}}

CRITICAL REMINDERS:
- Focus on CAPITAL PRESERVATION not stock picking
- Defense-first: when to reduce risk, not what to buy
- For hobby investors (€10K-€500K portfolios)
- Macro-focused, not individual stocks
- Evidence-based, not FOMO or hot tips
- Time-appropriate (weeks/months, not day-trading)
- Quality sources (Fed, ECB, academics, respected analysts)
- Actionable (simple rebalancing, not complex derivatives)

DO NOT include any text outside the JSON object.
```

---

## SCORING WEIGHTS (for downstream processing)

### Signal Strength Score (0-10)
Combined metric for signal importance:

```python
# For RED signals
if signal_tier == "RED":
    signal_strength = (
        macro_risk_severity * 0.30 +
        credit_market_stress * 0.25 +
        systemic_risk * 0.25 +
        evidence_quality * 0.20
    )
    # Must have evidence quality >= 5 to be RED
    if evidence_quality < 5:
        signal_tier = "YELLOW"  # Downgrade

# For YELLOW signals
elif signal_tier == "YELLOW":
    signal_strength = (
        macro_risk_severity * 0.25 +
        credit_market_stress * 0.20 +
        valuation_risk * 0.25 +
        policy_regulatory_risk * 0.20 +
        evidence_quality * 0.10
    )

# For GREEN signals
elif signal_tier == "GREEN":
    signal_strength = (
        (10 - valuation_risk) * 0.35 +  # Lower is better
        market_sentiment_extremes * 0.35 +  # Higher fear = higher score
        evidence_quality * 0.20 +
        actionability * 0.10
    )

# For BLUE signals
elif signal_tier == "BLUE":
    signal_strength = evidence_quality * 0.50 + actionability * 0.50

# Risk count bonus
risk_count = sum([
    risk_indicators['yield_curve_inversion'],
    risk_indicators['recession_indicators_converging'],
    risk_indicators['credit_spread_widening'],
    risk_indicators['bank_stress_signals'],
    risk_indicators['policy_error_risk'],
    risk_indicators['systemic_fragility']
])

if risk_count >= 3:
    signal_strength = min(10, signal_strength + 1.0)
```

### Portfolio Action Priority (0-10)
For hobby investors, how urgent is action?

```python
action_priority = (
    (signal_strength if signal_tier in ["RED", "YELLOW"] else 0) * 0.50 +
    actionability * 0.30 +
    (10 - evidence_quality if signal_tier in ["RED", "YELLOW"] else evidence_quality) * 0.20
)

# Time horizon adjustment
if time_horizon == "immediate":
    action_priority = min(10, action_priority + 2.0)
elif time_horizon == "short_term":
    action_priority = min(10, action_priority + 1.0)
```

---

## EXPECTED SCORE DISTRIBUTIONS

### Dimension Score Distributions
- **macro_risk_severity**: Left-skewed in normal times (mean ~3), right-skewed in crises (mean ~7)
- **credit_market_stress**: Bimodal (normal ~2-3, crisis ~8-9)
- **market_sentiment_extremes**: Normal distribution, mean ~5, extremes at 0-2 (fear) or 8-10 (greed)
- **valuation_risk**: Right-skewed in bull markets (mean ~6-7), normal in bear markets (mean ~4-5)
- **evidence_quality**: Right-skewed (filter out low-quality), mean ~6.5

### Signal Tier Distribution (expected in financial news)
- **NOISE**: 60-70% (most content is stock tips, individual analysis, clickbait)
- **BLUE Context**: 15-20% (educational, historical analysis)
- **YELLOW Warning**: 8-12% (monitoring situations)
- **GREEN Opportunity**: 3-5% (rare, only in extreme fear)
- **RED Flag**: 2-4% (rare, serious risks only)

**Target**: Surface top 1-2% RED signals immediately, 5-10% YELLOW for monitoring

---

## VALIDATION EXAMPLES

### Example 1: RED FLAG (9.2/10) - Act Now

**Article**: "Fed Emergency Meeting as Silicon Valley Bank Fails, FDIC Takes Control. Deposit flight spreading to First Republic, Signature Bank. Credit default swaps on major banks surging 200%. Treasury Secretary calls emergency meeting with banking regulators."

**Scores**:
- Macro Risk Severity: 8 (banking crisis, systemic implications)
- Credit Market Stress: 10 (bank failures, contagion spreading)
- Market Sentiment: 7 (fear building but not panic yet)
- Valuation Risk: 5 (irrelevant in crisis)
- Policy/Regulatory Risk: 7 (emergency response underway)
- Systemic Risk: 9 (contagion risk, interconnected banks)
- Evidence Quality: 9 (FDIC official action, CDS data)
- Actionability: 9 (clear action: reduce bank exposure, raise cash)

**Signal Tier**: 🔴 RED FLAG
**Signal Strength**: 9.2
**Time Horizon**: immediate
**Recommended Actions**: increase_cash, reduce_risk_assets

**Risk Indicators**:
- bank_stress_signals: true
- credit_spread_widening: true
- systemic_fragility: true
- recession_indicators_converging: true

**Reasoning**: "Banking crisis unfolding with contagion spreading (SVB → First Republic → Signature). Emergency Fed/Treasury response indicates systemic risk. Hobby investors should reduce bank exposure, raise cash reserves, defensive positioning immediately."

**Similar Historical Periods**: ["2008 Lehman Crisis", "2011 European Debt Crisis"]

---

### Example 2: YELLOW WARNING (6.8/10) - Monitor Closely

**Article**: "Yield Curve Inverts for 8th Consecutive Month as 10-Year Falls Below 2-Year. Historical recession signal with 80% accuracy. However, unemployment still at 3.6%, PMI above 50. Fed signals pause on rate hikes. Economists divided on timing of potential recession."

**Scores**:
- Macro Risk Severity: 6 (recession signal but not confirmed)
- Credit Market Stress: 4 (some stress but manageable)
- Market Sentiment: 5 (neutral, not extreme)
- Valuation Risk: 6 (moderately expensive)
- Policy/Regulatory Risk: 5 (Fed pausing, uncertainty)
- Systemic Risk: 4 (no systemic fragility yet)
- Evidence Quality: 8 (hard data: yield curve, employment, PMI)
- Actionability: 7 (clear monitoring plan, prepare rebalancing)

**Signal Tier**: 🟡 YELLOW WARNING
**Signal Strength**: 6.8
**Time Horizon**: short_term (3-12 months)
**Recommended Actions**: monitor_closely, rebalance_to_target (if overweight equities)

**Risk Indicators**:
- yield_curve_inversion: true
- recession_indicators_converging: false (mixed signals)

**Reasoning**: "Yield curve inversion is reliable recession predictor (80% historical accuracy) but timing uncertain. Other indicators (unemployment, PMI) still healthy. Hobby investors should monitor closely, ensure portfolio not overweight risk assets, but no panic selling. Review allocation quarterly."

**Similar Historical Periods**: ["2006 Pre-Crisis", "2000 Dot-com Bubble Peak"]

---

### Example 3: GREEN OPPORTUNITY (7.5/10) - Consider Buying

**Article**: "VIX Hits 45 as Market Sells Off 12% in Week, Highest Fear Since March 2020. Quality Large-Caps Trading at 15x Earnings (vs 20x Historical Average). Warren Buffett Quote Resurfaces: 'Be Greedy When Others Are Fearful.' Fund Flows Show $50B Retail Exodus from Equities."

**Scores**:
- Macro Risk Severity: 4 (sell-off but no systemic crisis)
- Credit Market Stress: 3 (normal)
- Market Sentiment: 9 (extreme fear, VIX 45, panic selling)
- Valuation Risk: 2 (attractive valuations, below historical average)
- Policy/Regulatory Risk: 3 (supportive, no policy errors)
- Systemic Risk: 2 (no systemic fragility)
- Evidence Quality: 8 (VIX data, P/E ratios, fund flow data)
- Actionability: 8 (clear opportunity to rebalance, DCA)

**Signal Tier**: 🟢 GREEN OPPORTUNITY
**Signal Strength**: 7.5
**Time Horizon**: short_term to medium_term
**Recommended Actions**: consider_buying_quality, rebalance_to_target

**Risk Indicators**:
- extreme_sentiment: true (fear extreme)
- valuation_extreme: true (cheap)

**Reasoning**: "Extreme fear (VIX 45) combined with attractive valuations (15x P/E vs 20x historical) creates buying opportunity. No systemic crisis, just sentiment-driven selloff. Hobby investors with cash reserves should dollar-cost average into quality large-caps, rebalance to target allocation. Don't time the bottom, but add gradually."

**Similar Historical Periods**: ["March 2020 COVID Selloff", "December 2018 Taper Tantrum"]

---

### Example 4: BLUE CONTEXT (7.0/10) - Understand

**Article**: "Academic Study: 60/40 Stock/Bond Portfolio Historical Returns 1926-2024. Analysis shows losing 50% requires 100% gain to recover. Behavioral Finance insights on why investors sell at bottoms. Recommended reading: 'A Random Walk Down Wall Street' by Burton Malkiel."

**Scores**:
- Macro Risk Severity: 0 (not discussing current risks)
- Credit Market Stress: 0
- Market Sentiment: 0
- Valuation Risk: 0
- Policy/Regulatory Risk: 0
- Systemic Risk: 0
- Evidence Quality: 9 (academic research, historical data)
- Actionability: 5 (improves framework, no immediate trades)

**Signal Tier**: 🔵 BLUE CONTEXT
**Signal Strength**: 7.0 (based on evidence quality + educational value)
**Time Horizon**: long_term
**Recommended Actions**: no_action_needed

**Reasoning**: "Educational content on portfolio construction and behavioral finance. Improves decision-making framework but no immediate portfolio actions needed. Valuable for understanding why capital preservation matters (losing 50% requires 100% to recover). Add to reading list."

---

### Example 5: NOISE (1.0/10) - Ignore

**Article**: "🚀 THIS PENNY STOCK IS ABOUT TO EXPLODE!! 🚀 Get in NOW before it's too late! My secret Discord group made 1000% last month! Join through my link for exclusive picks! Not financial advice 😉"

**Scores**:
- Evidence Quality: 1 (no evidence, pure hype)
- Actionability: 0 (dangerous advice)

**Signal Tier**: ⚫ NOISE
**Signal Strength**: 1.0
**Flags**: speculation_noise, clickbait, affiliate_conflict

**Reasoning**: "Pure speculation and FOMO marketing with affiliate links. No macro analysis, no evidence, no capital preservation focus. Hobby investors should ignore completely. Red flags: rocket emojis, 'explode', urgency tactics, 'secret group', affiliate link."

---

## PRE-FILTER RECOMMENDATION

To reduce labeling costs while maintaining coverage:

**Only analyze articles where**:
- Source category in: `economics`, `finance`, `central_banks`, `credit_markets`, `macro_research`
- OR article contains keywords: `recession`, `fed`, `interest rates`, `credit`, `bank`, `crisis`, `yield curve`, `unemployment`, `inflation`, `valuation`, `bear market`, `correction`
- AND NOT contains: `stock pick`, `buy now`, `hot tip`, specific ticker symbols as main focus

**Expected filter pass rate**: 20-30% of financial content

This pre-filter is implemented as `investment_risk_pre_filter()` in batch processing.

---

## ETHICAL CONSIDERATIONS

### What This Filter EXCLUDES
- **Stock picking services**: Individual stock tips, "buy this now"
- **FOMO marketing**: Pump-and-dump schemes, urgency tactics, affiliate conflicts
- **Speculation narratives**: Meme stocks, crypto pumping, gambling mindset
- **Day trading advice**: Minute-by-minute calls, technical patterns for quick trades
- **Perma-bears/bulls**: Those who always predict crash/boom regardless of data

### Known Biases to Monitor
1. **Recency bias**: Don't over-weight recent events
   - 2008 doesn't mean every dip is Lehman
   - Balance historical context with current data

2. **Confirmation bias**: Don't seek only risk signals
   - GREEN opportunities matter too
   - BLUE context prevents emotional mistakes

3. **Sophistication bias**: Don't favor complex over simple
   - Hobby investors need actionable, not academic complexity
   - Simple rebalancing > complex derivatives

4. **US-centric bias**: Include European and global perspectives
   - ECB matters as much as Fed for EU investors
   - Emerging market risks affect global portfolios

5. **Short-term bias**: Don't confuse volatility with risk
   - Weekly dips ≠ portfolio risk
   - Focus on months/years, not days

### Consistency Checks
- If `signal_tier == "RED"`, then `evidence_quality >= 5`
- If `signal_tier == "GREEN"`, then `valuation_risk <= 3` AND `market_sentiment_extremes >= 7`
- If `macro_risk_severity >= 7`, then `signal_tier` should be "RED" or "YELLOW"
- If `flags.speculation_noise == true`, then `signal_tier == "NOISE"`
- If `actionability < 4`, then cannot be "RED" or "GREEN" (not actionable enough)

---

## USE CASES

### 1. Capital Preservation Intelligence Dashboard
**Filter**: `signal_tier == "RED"` OR `signal_tier == "YELLOW"`
**Output**: Weekly risk signal digest
**Audience**: Hobby investors (€10K-€500K portfolios)
**Revenue**: €149/year subscription

### 2. Portfolio Protection System
**Filter**: `signal_tier == "RED"` OR (`signal_tier == "YELLOW"` AND `signal_strength >= 6`)
**Output**: Real-time risk alerts + recommended actions
**Audience**: Fund managers, financial advisors
**Revenue**: €300-500/month B2B

### 3. Value Opportunity Scanner
**Filter**: `signal_tier == "GREEN"` AND `signal_strength >= 6`
**Output**: Buy-the-dip opportunities with rationale
**Audience**: Value investors, contrarians
**Revenue**: €99-199/year

### 4. Investor Education Platform
**Filter**: `signal_tier == "BLUE"` AND `evidence_quality >= 7`
**Output**: Curated educational content
**Audience**: Learning investors, FIRE community
**Revenue**: Courses (€79-149), eBooks (€19-39)

### 5. Advisor White-Label Service
**Filter**: All signals with actionability >= 6
**Output**: Client-ready risk briefs, talking points
**Audience**: Fee-only RIAs (20-200 clients each)
**Revenue**: €200-500/month per advisor

---

## SUCCESS METRICS

### How to Measure Filter Performance

**Signal Accuracy** (Most Critical):
- % of RED signals followed by market decline (target: >60% within 6 months)
- % of GREEN signals followed by recovery (target: >70% within 12 months)
- False alarm rate for RED signals (target: <30%)

**Actionability**:
- % of signals that lead to portfolio actions by users
- Target: >50% of RED/YELLOW signals acted on
- Target: >30% of GREEN signals acted on

**Time Value**:
- Hobby investor time saved vs DIY research
- Baseline: 5-8 hours/week scanning financial news
- Target: 80% reduction → 1 hour/week reviewing digest

**Capital Preservation**:
- Portfolio value protected vs buy-and-hold during drawdowns
- Target: Reduce max drawdown by 15-30% vs unmanaged portfolio
- Example: 2008 unmanaged -55%, managed -35-40%

**User Satisfaction**:
- NPS score (target: >50)
- Retention rate (target: >75% annual renewal)
- Referral rate (target: >20% refer a friend)

---

## INTEGRATION WITH CONTENT AGGREGATOR

### Add Investment Risk Data Sources

Recommend adding to `config/sources/rss_investment_risk.yaml`:

**45 sources already configured** in `investor-signals-application.md`:

**Macro Risk & Economics** (Priority 8-9):
- Calculated Risk (called 2008 crisis)
- Fed official statements
- ECB policy analysis
- IMF reports

**Credit Markets** (Priority 8):
- Credit markets data/analysis
- High-yield spread tracking
- Banking sector health

**Valuation & Value Investing** (Priority 7-8):
- GMO (Jeremy Grantham)
- Research Affiliates
- Vanguard research

**Behavioral Finance** (Priority 7):
- Barry Ritholtz (behavioral focus)
- Daniel Kahneman insights
- Academic behavioral research

**Quality Analysis** (Priority 6-7):
- Portfolio managers focused on capital preservation
- Long-term value investors
- Risk-first strategists

---

## FUTURE ENHANCEMENTS

### Phase 2 (After Initial Validation)

1. **Historical Signal Tracking**
   - Track all RED/YELLOW/GREEN signals
   - Measure accuracy over time
   - Publish transparent track record

2. **Portfolio Stress Test**
   - Input user portfolio
   - Model impact of current risks
   - Suggest adjustments

3. **Risk Scenario Library**
   - Historical crisis patterns
   - Early warning indicators
   - Response playbooks

4. **Correlation Analysis**
   - Cross-reference with sustainability filter (climate risks)
   - Cross-reference with education filter (EdTech investment trends)
   - Multi-filter risk intelligence

5. **Advisor Tools**
   - Client communication templates
   - Risk explanation frameworks
   - Rebalancing calculators

---

**This filter transforms your content aggregator into a Capital Preservation Intelligence system for defense-first investors.**

**Last Updated**: 2025-10-29
**Version**: 1.0
**Status**: Ready for testing
**Target Audience**: Hobby investors (€10K-€500K portfolios)
**Core Philosophy**: "You can't predict crashes, but you can prepare for them."
