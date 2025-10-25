# Ground Truth Generation - Best Practices

Proven strategies for creating high-quality labeled datasets with LLM oracles.

---

## 🎯 Core Principles

### 1. **Start Small, Scale Gradually**

❌ **Don't**: Generate 50K labels immediately
✅ **Do**: Test → Validate → Scale

**Recommended progression**:
1. **10 articles** - Verify prompt works, API keys work, output format correct
2. **100 articles** - Check tier distribution, review edge cases
3. **1,000 articles** - Validate quality matches expectations
4. **5,000 articles** - First production dataset for fine-tuning
5. **50,000 articles** - Full-scale ground truth (if needed)

**Cost at each stage**:
- 10 articles: $0.02 (Gemini) or $0.09 (Claude)
- 100 articles: $0.18 (Gemini) or $0.90 (Claude)
- 1,000 articles: $1.80 (Gemini) or $9.00 (Claude)
- 5,000 articles: $9.00 (Gemini) or $45.00 (Claude)
- 50,000 articles: $90.00 (Gemini) or $450.00 (Claude)

### 2. **Use Stratified Sampling**

**Why it matters**: Random sampling can miss important edge cases

**Stratified sampling ensures**:
- Representation across source categories
- Diverse content types (short/long articles)
- Edge cases (greenwashing, vaporware, military content)
- Temporal balance (not all from one time period)

**Example for sustainability filter**:
```python
# 70% stratified by source category
climate_solutions: 1,000 articles
renewable_energy: 800 articles
automotive_transport: 600 articles
energy_utilities: 500 articles
# etc.

# 20% edge cases
greenwashing_likely: 400 articles
vaporware_likely: 300 articles
fossil_transition: 200 articles
breakthrough_research: 100 articles

# 10% random
random_sample: 500 articles
```

### 3. **Implement Pre-Filters**

**Pre-filtering reduces costs by 40-60% with minimal false negatives**

**Uplifting filter pre-filter**:
```python
def uplifting_pre_filter(article):
    # Only analyze articles with positive sentiment OR high joy
    sentiment = article['metadata']['sentiment_score']
    joy = article['metadata']['raw_emotions']['joy']
    return sentiment >= 5.0 or joy >= 0.25
```

**Cost savings**:
- Without filter: 10,000 articles × $0.009 = $90
- With filter (50% pass): 5,000 articles × $0.009 = $45
- **Savings: $45 (50%)**

**Sustainability filter pre-filter**:
```python
def sustainability_pre_filter(article):
    # Check category OR keywords
    category = article['metadata']['source_category']
    text = article['title'] + ' ' + article['content']

    sustainability_categories = [
        'climate_solutions', 'energy_utilities',
        'renewable_energy', 'automotive_transport'
    ]

    keywords = [
        'climate', 'carbon', 'renewable', 'solar',
        'wind', 'battery', 'ev', 'sustainability'
    ]

    return (category in sustainability_categories or
            any(kw in text.lower() for kw in keywords))
```

### 4. **Monitor Quality Early**

**Check after first 100 articles**:

✅ **Tier distribution** matches expectations
```
Uplifting filter:
- Impact (>= 7.0): 5-15%
- Connection (4.0-6.9): 20-40%
- Not uplifting (< 4.0): 50-70%

Sustainability filter:
- High (>= 7.0): 10-20%
- Medium (4.0-6.9): 30-50%
- Low (< 4.0): 40-60%
```

✅ **Edge cases handled correctly**
- Corporate finance: Capped at score 2
- Military buildups: Capped at score 4
- Greenwashing: Flagged and scored low
- Peace processes: NOT capped (can score 7+)

✅ **JSON parsing success rate > 95%**
- If lower: Improve prompt clarity
- Check for markdown formatting issues

---

## 📊 Quality Assurance

### Validation Checklist

**After every 1,000 articles**:

1. **Review sample articles** (10 random from each tier)
   - Read article
   - Check LLM scores
   - Verify reasoning makes sense

2. **Check tier distribution**
   ```python
   import json
   from collections import Counter

   tiers = []
   for batch_file in Path('datasets/uplifting').glob('labeled_batch_*.jsonl'):
       with open(batch_file) as f:
           for line in f:
               article = json.loads(line)
               tier = article['uplifting_analysis']['tier']
               tiers.append(tier)

   print(Counter(tiers))
   # Expected: {'not_uplifting': 600, 'connection': 300, 'impact': 100}
   ```

3. **Spot-check edge cases**
   - Find corporate finance articles → should be capped at 2
   - Find military articles → should be capped at 4 (unless peace process)
   - Find greenwashing → should have greenwashing_risk flag

4. **Compare to known examples**
   - Run 5-10 articles you've manually reviewed
   - Check LLM scores match your expectations

### Calibration Techniques

**If scores seem off**:

1. **Too many high scores** (> 20% in Impact tier)
   - Review prompt: Is it too lenient?
   - Check gatekeeper dimension (collective_benefit or technical_credibility)
   - Add stricter pre-classification filters

2. **Too many low scores** (< 5% in Impact tier)
   - Review prompt: Is it too strict?
   - Check if pre-filter is too aggressive
   - Verify you're sampling uplifting/sustainability content

3. **Inconsistent scoring**
   - Use temperature=0 (deterministic mode)
   - Provide more examples in prompt
   - Clarify dimension definitions

---

## 💰 Cost Optimization

### Choose the Right LLM

**For quality validation** (100-1,000 articles):
- **Use Claude 3.5 Sonnet**
- Higher quality, more consistent
- Cost: $0.009/article
- Use this to validate your prompt

**For bulk labeling** (5,000-50,000 articles):
- **Use Gemini 1.5 Pro Tier 1**
- 50x cheaper: $0.00018/article
- Still high quality (validated against Claude)
- **MUST enable Cloud Billing for Tier 1 (150 RPM)**

**Hybrid approach**:
1. Label 100 articles with Claude ($9)
2. Label same 100 with Gemini ($0.18)
3. Compare quality (should be 90%+ agreement)
4. If good → use Gemini for remaining 4,900 ($8.82)
5. **Total: $18 instead of $45 (60% savings)**

### Batch Processing

**Optimize request patterns**:

```python
# ❌ DON'T: One article at a time with long delays
for article in articles:
    label(article)
    time.sleep(5)  # Too slow!

# ✅ DO: Batch processing with appropriate delays
for batch in chunks(articles, size=50):
    for article in batch:
        label(article)
        time.sleep(0.5)  # Gemini Tier 1: 150 RPM
    save_batch()  # Save progress every 50
```

**Rate limits**:
- Claude Sonnet: 50 RPM → use 1.5s delay (~40 req/min)
- Gemini Tier 1: 150 RPM → use 0.5s delay (~120 req/min)
- Gemini Free: 2-5 RPM → **NOT viable for batch labeling**

### Resume Capability

**Always implement resume**:

```python
# ✅ State tracking prevents re-labeling
state = {
    'processed': ['article_id_1', 'article_id_2', ...],
    'total_labeled': 2543,
    'batches_completed': 51,
    'last_updated': '2025-10-25T14:30:00'
}

# Skip already-processed articles
if article_id in state['processed']:
    continue
```

**Benefits**:
- Survive API errors
- Survive rate limits
- Survive network issues
- Can stop/start anytime

---

## 🎨 Prompt Engineering Tips

### 1. Be Specific About What to Score

❌ **Vague**:
```
Rate this article for sustainability
```

✅ **Specific**:
```
Score based on DEPLOYED TECHNOLOGY and MEASURED OUTCOMES
NOT aspirational statements or commitments
```

### 2. Use Pre-Classification Filters

❌ **Without filters**:
```
Score all articles 0-10 on climate impact
```

✅ **With filters**:
```
STEP 1: Pre-classification
- If greenwashing (no concrete action) → max_credibility = 3
- If vaporware (no deployment) → max_readiness = 4
- If fossil transition → max_impact = 4

STEP 2: Score dimensions
```

### 3. Provide Concrete Examples

❌ **Abstract**:
```
Score innovation quality
```

✅ **With examples**:
```
Innovation Quality:
- 9-10: "Perovskite achieves 1,000-hour stability (vs. previous 100 hours)"
- 3-4: "5% efficiency improvement in mature technology"
- 0-2: "AI-powered blockchain for carbon credits" (buzzword bingo)
```

### 4. Use Gatekeeper Dimensions

**Uplifting filter**:
```python
# Collective benefit < 5 → cap overall score at 3
# Prevents individual/corporate wins from scoring high
if collective_benefit < 5:
    overall_score = min(overall_score, 3.0)
```

**Sustainability filter**:
```python
# Technical credibility < 5 → cap overall score at 4
# Prevents hype and vaporware from scoring high
if technical_credibility < 5:
    sustainability_score = min(sustainability_score, 4.0)
```

### 5. Request JSON Only

```
Respond with ONLY valid JSON in this exact format:
{
  "dimension1": <score>,
  "dimension2": <score>,
  ...
}

DO NOT include any text outside the JSON object.
```

**Why**: Easier parsing, no markdown formatting issues

---

## 🔄 Continuous Improvement

### Active Learning

**After initial 5K dataset**:

1. **Train first model**
2. **Find uncertain predictions**
   - Model predicts score = 5.2 with low confidence
   - These are most valuable to label!
3. **Label 1,000 uncertain cases with LLM**
4. **Retrain model**
5. **Repeat**

**Result**: Better model with fewer labeled examples

### Drift Detection

**Every 3-6 months**:

1. Sample 1,000 recent articles
2. Re-label with LLM
3. Compare to old labels
4. If distributions shifted → retrain model

---

## 📝 Documentation

### Document Your Decisions

**Create a labeling log**:

```markdown
# Labeling Log - Sustainability Filter v1.0

## Prompt Version
- Version: 1.0
- Date: 2025-10-25
- LLM: Gemini 1.5 Pro Tier 1

## Sampling Strategy
- Total articles: 5,000
- Stratified: 70% by source category
- Edge cases: 20% (greenwashing, vaporware, etc.)
- Random: 10%

## Pre-Filter
- Source categories: climate_solutions, renewable_energy, etc.
- Keywords: climate, carbon, renewable, solar, etc.
- Pass rate: 45%

## Quality Metrics
- Impact tier (>= 7.0): 12%
- Connection tier (4.0-6.9): 38%
- Low tier (< 4.0): 50%
- JSON parse success: 97%

## Edge Cases
- Greenwashing articles: 23% of dataset
- Fossil transition: 8%
- Vaporware: 15%

## Cost
- Total articles labeled: 5,000
- Cost: $9.00 (Gemini Tier 1)
- Time: ~11 hours (with resume from interruptions)

## Issues Encountered
- API rate limit: 2 times (fixed with longer delay)
- JSON parsing errors: 3% (reformatted prompt)

## Next Steps
- Generate additional 5K for uplifting filter
- Train DeBERTa-v3-small models
- Validate quality vs. Claude on 100-article test set
```

---

## ✅ Checklist: Before Large-Scale Labeling

Before spending $50+ on ground truth generation:

- [ ] Tested prompt on 10 articles
- [ ] Tier distribution looks correct (100 articles)
- [ ] Edge cases handled properly
- [ ] JSON parsing > 95% success
- [ ] Pre-filter implemented (if applicable)
- [ ] Resume capability tested
- [ ] Cost estimated and approved
- [ ] Spending limits set in API dashboard
- [ ] Documentation started (labeling log)

---

## 🎯 Success Metrics

**Good ground truth dataset has**:

✅ **Appropriate tier distribution**
- Not all in one tier
- Matches semantic expectations

✅ **Diverse examples**
- Across source categories
- Across content types
- Includes edge cases

✅ **High consistency**
- Similar articles get similar scores
- Edge cases handled uniformly

✅ **Strong signal**
- Clear difference between tiers
- Gatekeeper rules enforced

✅ **Complete metadata**
- All required fields populated
- Reasoning provided
- Flags set correctly

---

**Follow these best practices to generate high-quality ground truth efficiently and cost-effectively!**
