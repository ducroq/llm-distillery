# Commerce Content Detection

## Purpose
Classify whether an article is **commerce/promotional content** vs **journalism/editorial content**.

## Definition

**Commerce/Promotional Content** includes:
- Product deals, discounts, sales announcements
- Affiliate marketing content (links to buy products)
- Shopping guides focused on prices and where to buy
- Black Friday, Cyber Monday, Prime Day deals coverage
- PR/press releases announcing product launches
- "Best X to buy" listicles focused on purchasing
- Content with promo codes, discount codes, coupon codes
- Price comparison content
- Sponsored product reviews

**Journalism/Editorial Content** includes:
- News reporting on events, policies, research
- Investigative journalism
- Opinion/analysis pieces
- Scientific papers and research
- Government reports and announcements
- Feature stories about people, places, trends
- Reviews focused on quality/experience (not purchasing)
- Educational/explainer content

## Scoring Guide (0-10)

| Score | Classification | Description |
|-------|---------------|-------------|
| 0-2 | **Journalism** | Pure editorial/news content, no commercial intent |
| 3-4 | **Mostly Journalism** | Editorial with minor product mentions, no purchase push |
| 5-6 | **Mixed/Ambiguous** | Contains both editorial value and commercial elements |
| 7-8 | **Mostly Commerce** | Primary purpose is driving purchases, some editorial wrapper |
| 9-10 | **Commerce** | Pure promotional/deals content, affiliate, shopping guides |

## Key Signals

**Strong Commerce Signals (push score higher):**
- Prices mentioned with savings/discounts ("$500 off", "40% discount")
- Urgency language ("deal ends tonight", "limited time")
- Promo/coupon codes
- Affiliate disclaimers
- "Where to buy" sections
- Multiple product links
- Price comparisons across retailers

**Strong Journalism Signals (push score lower):**
- Quotes from experts/officials
- Data and statistics from studies
- Policy implications discussed
- Historical context provided
- Multiple perspectives presented
- No purchase calls-to-action

## Examples

### Example 1: Pure Commerce (Score: 9)
**Title:** "Green Deals: Save $500 on Jackery Solar Generator"
**Content:** "Today's Green Deals are headlined by an exclusive discount on the Jackery Explorer 1000 Plus solar generator kit. Originally priced at $1,999, you can now get it for just $1,499..."
**Reasoning:** Price-focused, discount language, urgency ("today's deals"), no editorial analysis.

### Example 2: Pure Journalism (Score: 1)
**Title:** "EPA Announces New Clean Energy Regulations"
**Content:** "The Environmental Protection Agency announced new regulations requiring power plants to reduce carbon emissions by 50% by 2030. Industry groups have expressed concerns about implementation timelines..."
**Reasoning:** Policy reporting, quotes stakeholders, discusses implications, no commercial intent.

### Example 3: Mixed Content (Score: 5)
**Title:** "Tesla Model 3 Review: Is It Worth the Price?"
**Content:** "After two weeks with the Tesla Model 3, here's our comprehensive review. The driving experience is excellent, with instant torque and smooth handling. At $42,990, it's competitive with..."
**Reasoning:** Editorial review with genuine analysis, but includes pricing and implicit purchase consideration.

---

**INPUT DATA:** [Paste the article here]

---

## Response Format

Respond with JSON:
```json
{
  "commerce_score": 7.5,
  "reasoning": "Brief explanation of why this score...",
  "key_signals": ["signal1", "signal2"]
}
```
