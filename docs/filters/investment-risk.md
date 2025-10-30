# Investment Risk Filter - Delivery Summary

**Date**: 2025-10-29
**Status**: ✅ Complete and ready for testing

---

## 🎯 **NEW: 5th Semantic Filter**

### Investment Risk: Capital Preservation Filter

**Purpose**: Defense-first investing focused on capital preservation, not speculation
**Target**: Hobby investors (€10K-€500K portfolios)
**Philosophy**: "You can't predict crashes, but you can prepare for them."

---

## 📦 What Was Created

### 1. **Full Semantic Filter**
`prompts/investment-risk.md` (640 lines)

**Signal Classification**: RED / YELLOW / GREEN / BLUE / NOISE

- **🔴 RED FLAGS** (Act Now): Banking crisis, systemic risk, yield curve + recession converging
- **🟡 YELLOW WARNINGS** (Monitor): Rising unemployment + credit stress, extreme valuations
- **🟢 GREEN OPPORTUNITIES** (Consider Buying): Extreme fear + cheap valuations
- **🔵 BLUE CONTEXT** (Understand): Education, historical analysis, framework improvement
- **⚫ NOISE** (Ignore): Stock tips, FOMO, pump-and-dump, clickbait

### 2. **Compressed Filter**
`prompts/compressed/investment-risk.md` (305 lines)

**Compression**: 52% reduction (640 → 305 lines)
**Token savings**: ~50% (2,400 → 1,200 tokens)

---

## 📊 **Filter Capabilities**

### 8 Scoring Dimensions (0-10 each)

1. **Macro Risk Severity** - Recession signals, geopolitical, policy errors
2. **Credit Market Stress** - Bank health, spreads, corporate debt, leverage
3. **Market Sentiment Extremes** - VIX, fear/greed, FOMO, positioning
4. **Valuation Risk** - P/E ratios, CAPE, historical context
5. **Policy/Regulatory Risk** - Fed, ECB, fiscal policy, political risk
6. **Systemic Risk** - Contagion potential, interconnectedness, liquidity
7. **Evidence Quality** (GATEKEEPER) - Data quality, source credibility
8. **Actionability** - Can hobby investors act? (portfolio-level, low-cost, simple)

### Pre-Classification Filters

**Filters out NOISE**:
- ❌ FOMO/speculation ("hot stocks", "buy now", meme stocks)
- ❌ Stock picking (individual stock tips without macro relevance)
- ❌ Affiliate conflicts ("sign up through my link")
- ❌ Clickbait ("Market CRASH coming!", "This ONE stock!")

### Rich Metadata

**Risk Indicators**:
- yield_curve_inversion
- recession_indicators_converging
- credit_spread_widening
- bank_stress_signals
- policy_error_risk
- extreme_sentiment
- valuation_extreme
- systemic_fragility

**Asset Classes**: equities, fixed_income, credit, currencies, commodities, real_estate, cash_equivalents

**Recommended Actions**: increase_cash, rebalance_to_target, reduce_risk_assets, consider_buying_quality, monitor_closely, no_action_needed

---

## 🎯 **Unique Value Proposition**

### What Competitors Lack

**Standard Investor Tools** (Seeking Alpha, Motley Fool, Morningstar):
- Focus on OFFENSE (what to buy, stock picks)
- FOMO-driven ("Don't miss out!")
- Conflicted (affiliate commissions, stock pumping)
- No systematic risk framework

**Our Approach**:
- Focus on DEFENSE (when to reduce risk)
- Fear-management through understanding
- No affiliate conflicts (pure subscription)
- Systematic framework (RED/YELLOW/GREEN/BLUE)

**Target Audience**: The Anxious Accumulator
- Age: 30-55
- Portfolio: €20K-€200K
- Primary fear: Losing years of savings in a crash
- Goal: Retire comfortably without becoming a day trader
- Pain point: Overwhelmed by contradictory financial media

---

## 💰 **Revenue Potential**

### Capital Preservation Intelligence (Substack/Web App)

**Individual Subscribers**:
- €149/year subscription
- Weekly risk signal digests
- Target: 500 subs Year 1 = €74,500 ARR

**Financial Advisors** (B2B White-Label):
- €200-500/month per advisor
- Client-ready risk briefs
- Target: 20 advisors Year 1 = €48K-120K ARR

**Total Year 1**: €122K-195K ARR

### With Educational Products

**Product Ladder**:
- eBooks: €19-39 (1,500 sales = €37K)
- Courses: €79-149 (400 sales = €42K)
- Workshops: €349 (100 attendees = €35K)
- Podcast sponsors: €20K-40K

**Total with Education**: €250K-€420K Year 2

---

## 📁 **Files Created**

```
✅ prompts/investment-risk.md (640 lines - full version)
✅ prompts/compressed/investment-risk.md (305 lines - compressed)
✅ INVESTMENT_RISK_FILTER_SUMMARY.md (this file)
```

---

## 🔗 **Integration with Existing Filters**

### Cross-Filter Intelligence

**Investment Risk ↔ Sustainability**:
- Climate tech investment opportunities
- ESG risk signals
- Energy transition portfolio impacts

**Investment Risk ↔ SEECE**:
- Energy sector risk/opportunity analysis
- Dutch/EU energy policy investment implications
- Hydrogen economy investment signals

**Investment Risk ↔ Education**:
- EdTech investment trends
- Structural shifts in education sector
- AI transformation investment opportunities

**Investment Risk ↔ Uplifting**:
- ESG/impact investing signals
- Community-focused investment opportunities
- Social good + returns

---

## 📊 **Expected Performance**

### Signal Tier Distribution (in financial news)

- **NOISE**: 60-70% (most content is stock tips, clickbait)
- **BLUE Context**: 15-20% (educational, historical)
- **YELLOW Warning**: 8-12% (monitoring situations)
- **GREEN Opportunity**: 3-5% (rare, extreme fear only)
- **RED Flag**: 2-4% (rare, serious risks only)

**Target**: Surface top 1-2% RED signals immediately

### Quality Targets

**Signal Accuracy**:
- RED signals followed by decline: >60% within 6 months
- GREEN signals followed by recovery: >70% within 12 months
- False alarm rate: <30%

**Capital Preservation**:
- Reduce max drawdown by 15-30% vs unmanaged
- Example: 2008 unmanaged -55%, managed -35-40%

---

## 🚀 **Use Cases**

### 1. Capital Preservation Intelligence Dashboard
**Filter**: signal_tier == "RED" OR "YELLOW"
**Output**: Weekly risk signal digest
**Revenue**: €149/year × 500 subs = €74,500

### 2. Portfolio Protection System
**Filter**: signal_tier == "RED" OR (YELLOW AND signal_strength ≥ 6)
**Output**: Real-time risk alerts + actions
**Revenue**: €300-500/month B2B

### 3. Value Opportunity Scanner
**Filter**: signal_tier == "GREEN" AND signal_strength ≥ 6
**Output**: Buy-the-dip opportunities
**Revenue**: €99-199/year

### 4. Investor Education Platform
**Filter**: signal_tier == "BLUE" AND evidence_quality ≥ 7
**Output**: Curated educational content
**Revenue**: Courses (€79-149), eBooks (€19-39)

### 5. Advisor White-Label Service
**Filter**: All signals with actionability ≥ 6
**Output**: Client-ready risk briefs
**Revenue**: €200-500/month per advisor

---

## 📚 **Data Sources**

**45 RSS sources already configured** in `investor-signals-application.md`:

### Macro Risk & Economics (Priority 8-9)
- Calculated Risk (called 2008 crisis)
- Fed official statements
- ECB policy analysis
- IMF reports

### Credit Markets (Priority 8)
- Credit markets data/analysis
- High-yield spread tracking
- Banking sector health

### Valuation & Value Investing (Priority 7-8)
- GMO (Jeremy Grantham)
- Research Affiliates
- Vanguard research

### Behavioral Finance (Priority 7)
- Barry Ritholtz
- Daniel Kahneman insights
- Academic behavioral research

**Expected volume**: 30-50 articles/day after pre-filter

---

## 🧪 **Next Steps - Testing**

### Phase 1: Validate Filter (Week 1)

**Step 1: Collect sample articles** (10 min)
```bash
# Use existing investor signals sources
python run_aggregator.py --sources rss_investor_signals --days-back 7
```

**Step 2: Test filter** (30 min)
```bash
python scripts/test_compressed_quality.py \
    --filter investment-risk \
    --articles 20 \
    --original-model sonnet \
    --compressed-model flash
```

**Step 3: Validate signal classification** (30 min)
- Do RED signals indicate genuine risk?
- Are GREEN signals true opportunities?
- Is NOISE properly filtered?

**Decision criteria**:
- ✅ Signal tier accuracy >85%: Proceed
- ✅ NOISE correctly identified >90%: Proceed
- ❌ Fails: Refine filters or use hybrid approach

---

### Phase 2: Build Application (Weeks 2-8)

**Option A: Substack Newsletter** (Fastest)
- Weekly risk signal digests
- FREE tier: Monthly + framework education
- PAID tier: €149/year for weekly digests
- Target: 100 subs (50 paying) = €7,500 in Month 2

**Option B: Web App + Dashboard** (2-3 months)
- Signal dashboard (current RED/YELLOW/GREEN)
- Historical signal archive
- Portfolio stress test tool
- API for integrations
- Target: €15/mo or €150/yr

**Option C: B2B White-Label First** (4-6 weeks)
- Client-ready risk briefs for advisors
- Co-branded with advisor logo
- Target: 5 advisors × €300/mo = €18K ARR

---

### Phase 3: Educational Products (Months 3-6)

**eBook**: "The 4-Signal Framework" (€24)
- Complete RED/YELLOW/GREEN/BLUE explanation
- 30+ historical examples
- Decision trees for portfolio actions
- Target: 1,000 sales = €24,000

**Course**: "Recession-Proof Your Portfolio" (€79)
- 16 video lessons (4 hours)
- Portfolio stress test calculator
- Rebalancing during volatility
- Target: 200 students = €15,800

**Workshop**: In-person intensive (€349)
- Full-day (8 hours)
- Portfolio review sessions
- Amsterdam, Rotterdam, Brussels
- Target: 100 attendees/year = €34,900

---

## 💡 **Competitive Positioning**

### vs. Seeking Alpha (€239/year)
- **Them**: Stock-picking focused, conflicting opinions
- **Us**: Risk-focused, systematic framework

### vs. Morningstar Premium (€199-299/year)
- **Them**: Expensive, stock/fund selection focus
- **Us**: Affordable, macro/risk focused, passive-friendly

### vs. Motley Fool (€199/year)
- **Them**: Pure stock-picking, FOMO marketing
- **Us**: No stock picking, defense-first, no affiliates

### vs. Bloomberg Terminal (€24K/year)
- **Them**: Institutional pricing, overwhelming data
- **Us**: Right-sized for retail (€149/year), actionable insights

### vs. Financial Advisors (1% AUM = €1K-5K/year)
- **Them**: Personalized but expensive
- **Us**: Self-service at 5-10% the cost

### vs. Free Sources (Reddit, Bogleheads)
- **Them**: Time-intensive to curate, noise >> signal
- **Us**: Pre-curated from 45+ quality sources, systematic

**Our Quadrant**: High specialization (risk/defense focus) + Mid-tier pricing (€100-300/year)

---

## ✅ **Success Criteria**

### Filter Quality
- ✅ Signal tier accuracy: >85%
- ✅ RED signal followed by decline: >60%
- ✅ GREEN signal followed by recovery: >70%
- ✅ NOISE correctly filtered: >90%

### Business Metrics
- ✅ Month 6: 250 paid subs = €37K ARR
- ✅ Month 12: 500 paid subs = €74K ARR
- ✅ Year 2: 1,000 paid subs = €149K ARR

### User Satisfaction
- ✅ NPS score: >50
- ✅ Annual retention: >75%
- ✅ Referral rate: >20%

---

## 🎓 **Key Differentiators**

### 1. **Defense-First** (Not Offense)
Most tools tell you WHAT to buy. We tell you WHEN to get defensive.

### 2. **Capital Preservation** (Not Speculation)
Focus on not losing vs. chasing gains. Losing 50% requires 100% to recover.

### 3. **Macro-Focused** (Not Stock Picking)
Portfolio-level risk, not individual stock tips.

### 4. **Evidence-Based** (Not FOMO)
Hard data (Fed, ECB, credit spreads) not hot tips.

### 5. **No Conflicts** (Pure Subscription)
No affiliate commissions, no stock pumping, aligned incentives.

### 6. **Actionable for Hobby Investors**
Simple rebalancing (weeks/months), not day-trading (minutes).

---

## 📞 **Documentation**

- **Full filter**: `prompts/investment-risk.md`
- **Compressed filter**: `prompts/compressed/investment-risk.md`
- **Business case**: `docs/investor-signals-application.md`
- **Portfolio protection**: `docs/separate-projects/financial-intelligence-suite.md`
- **This summary**: `INVESTMENT_RISK_FILTER_SUMMARY.md`

---

## 🎉 **Ready to Test!**

The investment-risk filter is production-ready for testing. Start with quality validation to ensure signal classification accuracy.

**Next step**:
```bash
python scripts/test_compressed_quality.py --filter investment-risk --articles 20
```

---

## 📊 **All 5 Semantic Filters**

| Filter | Lines | Focus | Target Audience |
|--------|-------|-------|-----------------|
| **Education** | 664 (291 compressed) | AI paradox in education | Universities, faculty, educators |
| **Sustainability** | 564 (274 compressed) | Climate tech deployments | Investors, researchers |
| **SEECE** | 760 (346 compressed) | Dutch energy tech | HAN University, applied research |
| **Uplifting** | 347 (201 compressed) | Human/planetary wellbeing | Progress tracking, news curation |
| **Investment Risk** | 640 (305 compressed) | Capital preservation | Hobby investors, advisors |

**Total**: 2,975 lines (1,417 compressed)
**Overall compression**: 52%

---

**Last Updated**: 2025-10-29
**Status**: Ready for testing
**Recommendation**: Start with Substack for fastest validation (€149/year, target 100 subs in 3 months)
