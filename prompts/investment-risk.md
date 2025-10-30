# Investment Risk: Capital Preservation Filter

**Purpose**: Identify investment risk signals for defense-first portfolio management focused on capital preservation, not speculation.

**Version**: 1.0-compressed
**Target**: Gemini Flash 1.5 / Claude Haiku / Fast models

**Focus**: RISK SIGNALS and CAPITAL PRESERVATION, not stock picks or speculation.

**Philosophy**: "You can't predict crashes, but you can prepare for them."

---

## SIGNAL TIERS

**🔴 RED**: Act now - reduce risk immediately (yield curve + recession, bank crisis, systemic failure)
**🟡 YELLOW**: Monitor closely - prepare for defense (rising unemployment + credit stress, extreme valuations)
**🟢 GREEN**: Consider buying - value emerging (extreme fear + cheap valuations, quality at discount)
**🔵 BLUE**: Understand - no action (education, historical analysis, framework improvement)
**⚫ NOISE**: Ignore (stock tips, FOMO, pump-and-dump, clickbait)

---

## PROMPT

```
Analyze this article for investment risk signals based on CAPITAL PRESERVATION and MACRO RISK EVIDENCE.

ARTICLE:
Title: {title}
Source: {source}
Published: {published_date}
Text: {text}

STEP 1: Pre-classification Filters

A) FOMO/SPECULATION: "Hot stocks", "buy now", "next big thing", meme stocks, crypto pumping?
   - If YES and NO macro risk analysis → FLAG "speculation_noise" (signal_tier = NOISE)

B) STOCK PICKING: Individual stock analysis, specific buy recommendations, earnings predictions?
   - If YES and NOT systemic risk → FLAG "stock_picking" (signal_tier = NOISE)

C) AFFILIATE/CONFLICT: "Sign up with this broker", promotional codes, sponsored picks?
   - If YES → FLAG "affiliate_conflict" (signal_tier = NOISE)

D) CLICKBAIT: "Market CRASH coming!", "This ONE stock!", "Warren Buffett's secret!"?
   - If YES and NO substantive analysis → FLAG "clickbait" (max_evidence_quality = 3)

STEP 2: Score Dimensions (0-10)

1. **Macro Risk Severity**: Systemic economic/financial risk?
   - 0-2: None | 3-4: Minor | 5-6: Moderate | 7-8: Serious | 9-10: Crisis unfolding
   - Indicators: Recession signals (yield curve, unemployment, PMI), geopolitical, policy errors

2. **Credit Market Stress**: Credit market deterioration?
   - 0-2: Healthy | 3-4: Some stress | 5-6: Moderate deterioration | 7-8: Serious stress | 9-10: Credit crisis
   - Indicators: Spreads widening, bank health, corporate debt, leverage

3. **Market Sentiment Extremes**: Dangerous extremes?
   - 0-2: Balanced | 3-4: Slight tilt | 5-6: Moderate extreme | 7-8: Dangerous extreme | 9-10: Panic or euphoria
   - Indicators: VIX, put/call ratios, FOMO/fear, positioning

4. **Valuation Risk**: Priced for perfection or extreme pessimism?
   - 0-2: Attractive | 3-4: Fair | 5-6: Somewhat expensive | 7-8: Very expensive | 9-10: Bubble territory
   - Indicators: P/E ratios, CAPE, yield curves, historical context
   - GREEN signal: 0-3 | BLUE: 4-6 | YELLOW: 7-8 | RED: 9-10

5. **Policy/Regulatory Risk**: Policy changes creating risk/opportunity?
   - 0-2: Supportive | 3-4: Neutral | 5-6: Some uncertainty | 7-8: Concerning changes | 9-10: Policy crisis
   - Indicators: Monetary policy (Fed, ECB), fiscal policy, regulatory changes, political risk

6. **Systemic Risk**: Potential for contagion/cascading failures?
   - 0-2: Resilient | 3-4: Normal risks | 5-6: Some fragility | 7-8: Significant fragility | 9-10: Lehman-moment risk
   - Indicators: Interconnectedness, leverage, liquidity, tail risk

7. **Evidence Quality** (GATEKEEPER for RED: must be ≥5):
   - 0-2: Poor | 3-4: Weak | 5-6: Moderate | 7-8: Strong | 9-10: Exceptional
   - Hard data vs opinion, official stats vs blogs, sound reasoning, track record

8. **Actionability** (for hobby investors €10K-€500K):
   - 0-2: Not actionable | 3-4: Limited | 5-6: Moderate | 7-8: Very actionable | 9-10: Clear simple action
   - Time horizon (weeks/months not days), portfolio-level (not individual stocks), low-cost, simple

STEP 3: Classify Signal Tier

**🔴 RED FLAG**: Macro Risk ≥7 OR Credit Stress ≥7 OR Systemic Risk ≥8, Evidence ≥5, Actionability ≥5
**🟡 YELLOW WARNING**: Macro Risk 5-6 OR Credit Stress 5-6 OR Valuation Risk 7-8, Evidence ≥5, Actionability ≥4
**🟢 GREEN OPPORTUNITY**: Sentiment ≥7 (fear) AND Valuation ≤3 (cheap), Evidence ≥6, Actionability ≥5
**🔵 BLUE CONTEXT**: Educational, historical analysis, long-term trends (no immediate action)
**⚫ NOISE**: Flagged by pre-filters OR individual stock tips OR evidence <4

STEP 4: Metadata

**Risk Indicators** (true/false): yield_curve_inversion, recession_indicators_converging, credit_spread_widening, bank_stress_signals, policy_error_risk, extreme_sentiment, valuation_extreme, systemic_fragility

**Asset Classes**: equities, fixed_income, credit, currencies, commodities, real_estate, cash_equivalents

**Time Horizon**: immediate (0-3mo) | short_term (3-12mo) | medium_term (1-3yr) | long_term (3+yr)

**Geographic**: us, europe, global, emerging_markets, specific_country

**Actions**: increase_cash, rebalance_to_target, reduce_risk_assets, consider_buying_quality, monitor_closely, no_action_needed

STEP 5: Output JSON

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
    "yield_curve_inversion": <bool>,
    "recession_indicators_converging": <bool>,
    "credit_spread_widening": <bool>,
    "bank_stress_signals": <bool>,
    "policy_error_risk": <bool>,
    "extreme_sentiment": <bool>,
    "valuation_extreme": <bool>,
    "systemic_fragility": <bool>
  }},

  "asset_classes_affected": {{
    "equities": <bool>, "fixed_income": <bool>, "credit": <bool>,
    "currencies": <bool>, "commodities": <bool>, "real_estate": <bool>, "cash_equivalents": <bool>
  }},

  "time_horizon": "immediate|short_term|medium_term|long_term",
  "geographic_scope": ["us", "europe", "global"],

  "recommended_actions": ["<action1>", "<action2>"],

  "flags": {{
    "speculation_noise": <bool>, "stock_picking": <bool>,
    "affiliate_conflict": <bool>, "clickbait": <bool>
  }},

  "reasoning": "<2-3 sentences: risk signal, evidence, action for hobby investors>",

  "key_risk_metrics": ["<metric1 with data>", "<metric2>"],
  "similar_historical_periods": ["<period1>", "<period2>"],
  "expert_sources_cited": ["<source1>", "<source2>"]
}}

CRITICAL REMINDERS:
- Capital preservation, not stock picking
- Defense-first: when to reduce risk, not what to buy
- For hobby investors (€10K-€500K portfolios)
- Macro-focused, not individual stocks
- Evidence-based, not FOMO
- Time-appropriate (weeks/months, not day-trading)
- Quality sources (Fed, ECB, academics)
- Actionable (simple rebalancing, not derivatives)

VALIDATION EXAMPLES:

RED FLAG (9.2/10):
Article: "Fed Emergency Meeting as Silicon Valley Bank Fails, FDIC Takes Control. Deposit flight spreading to First Republic, Signature Bank. Credit default swaps on major banks surging 200%. Treasury calls emergency meeting."
Scores: Macro=8, Credit=10, Sentiment=7, Valuation=5, Policy=7, Systemic=9, Evidence=9, Actionability=9
Signal: 🔴 RED FLAG | Time: immediate | Actions: increase_cash, reduce_risk_assets
Risk Indicators: bank_stress_signals, credit_spread_widening, systemic_fragility, recession_indicators_converging
Reasoning: "Banking crisis unfolding with contagion (SVB → First Republic → Signature). Emergency Fed/Treasury response indicates systemic risk. Reduce bank exposure, raise cash, defensive positioning immediately."

NOISE (1.0/10):
Article: "🚀 THIS PENNY STOCK IS ABOUT TO EXPLODE!! 🚀 Get in NOW! My secret Discord made 1000% last month! Join through my link for exclusive picks! Not financial advice 😉"
Signal: ⚫ NOISE | Flags: speculation_noise, clickbait, affiliate_conflict
Reasoning: "Pure speculation and FOMO marketing with affiliate links. No macro analysis, no evidence. Ignore completely. Red flags: rocket emojis, urgency tactics, 'secret group', affiliate link."
```

---

## SCORING FORMULA (Applied post-labeling)

```python
# RED signals
if signal_tier == "RED":
    signal_strength = (
        macro_risk_severity * 0.30 +
        credit_market_stress * 0.25 +
        systemic_risk * 0.25 +
        evidence_quality * 0.20
    )
    if evidence_quality < 5: signal_tier = "YELLOW"  # Downgrade

# YELLOW signals
elif signal_tier == "YELLOW":
    signal_strength = (
        macro_risk_severity * 0.25 +
        credit_market_stress * 0.20 +
        valuation_risk * 0.25 +
        policy_regulatory_risk * 0.20 +
        evidence_quality * 0.10
    )

# GREEN signals
elif signal_tier == "GREEN":
    signal_strength = (
        (10 - valuation_risk) * 0.35 +  # Lower is better
        market_sentiment_extremes * 0.35 +  # Higher fear = better
        evidence_quality * 0.20 +
        actionability * 0.10
    )

# Risk count bonus
risk_count = sum([risk_indicators[key] for key in risk_indicators])
if risk_count >= 3: signal_strength = min(10, signal_strength + 1.0)

# Action priority
action_priority = (
    (signal_strength if signal_tier in ["RED", "YELLOW"] else 0) * 0.50 +
    actionability * 0.30 +
    evidence_quality * 0.20
)
if time_horizon == "immediate": action_priority = min(10, action_priority + 2.0)
```

---

**Compressed from 640 → 305 lines (52% reduction)**
**Token estimate**: ~1,200 tokens (was ~2,400)
