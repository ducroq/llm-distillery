# Education Intelligence - Quick Start Guide

Get started with the Future of Education semantic filter in 4 steps.

---

## Step 1: Test RSS Feeds (5 minutes)

Verify that the 22 enabled education RSS feeds are working:

```bash
cd C:\local_dev\content-aggregator

# Install feedparser if needed
pip install feedparser pyyaml

# Test all education feeds
python scripts/test_education_feeds.py
```

**Expected output**:
- ✅ 18-22 working feeds
- 📄 200-400 total entries available
- ⚠️ 0-2 problematic feeds (acceptable)

**If you see errors**:
- Feeds marked "disabled" are OK (we have 22 enabled sources)
- If >3 feeds show errors, check URLs or disable them in `config/sources/rss_education.yaml`

---

## Step 2: Collect Sample Content (10 minutes)

Collect 1 week of education articles:

```bash
# Collect from education sources only, last 7 days
python run_aggregator.py --sources rss_education --days-back 7

# Check results
ls data/aggregated/ | grep education
```

**Expected results**:
- 150-250 articles collected
- Stored in `data/aggregated/education_YYYYMMDD.jsonl`
- Mix of higher ed, K-12, EdTech, research

**Content breakdown**:
- ~60-80 articles from top sources (IHE, EdSurge, Chronicle, EDUCAUSE)
- ~30-50 AI-related articles (mention AI, automation, ChatGPT, etc.)
- ~5-10 high-quality transformation stories

---

## Step 3: Test the Education Filter (30 minutes)

Label a sample of articles using Claude 3.5 Sonnet:

### Option A: Use Claude API directly

Create `scripts/test_education_filter.py`:

```python
import anthropic
import json
from pathlib import Path

# Load prompt
with open('prompts/future-of-education.md', 'r', encoding='utf-8') as f:
    prompt_text = f.read()
    # Extract the prompt template between the ``` markers
    prompt_template = prompt_text.split('```')[1]

# Load sample articles
articles = []
with open('data/aggregated/education_YYYYMMDD.jsonl', 'r') as f:
    for line in f:
        articles.append(json.loads(line))

# Take 10 articles with AI keywords
ai_articles = [
    a for a in articles
    if any(kw in a.get('title', '').lower() + a.get('content', '').lower()
           for kw in ['ai', 'artificial intelligence', 'chatgpt', 'automation'])
][:10]

# Label with Claude
client = anthropic.Anthropic(api_key="YOUR_API_KEY")

for article in ai_articles:
    filled_prompt = prompt_template.format(
        title=article['title'],
        source=article.get('source', 'Unknown'),
        published_date=article.get('published', 'Unknown'),
        text=article.get('content', '')[:4000]  # Truncate long articles
    )

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2000,
        messages=[{"role": "user", "content": filled_prompt}]
    )

    # Parse JSON response
    result = json.loads(response.content[0].text)

    # Print summary
    print(f"\n{'='*80}")
    print(f"Article: {article['title']}")
    print(f"Source: {article.get('source', 'Unknown')}")
    print(f"Content Type: {result['content_type']}")
    print(f"Transformation Stage: {result['transformation_stage']}")
    print(f"Paradox Engagement: {result['paradox_engagement']}/10")
    print(f"Assessment Transformation: {result['assessment_transformation']}/10")
    print(f"Reasoning: {result['reasoning']}")
    print(f"{'='*80}\n")

    # Save result
    article['education_labels'] = result
    with open('data/labeled/education_sample.jsonl', 'a') as out:
        out.write(json.dumps(article) + '\n')
```

Run:
```bash
python scripts/test_education_filter.py
```

### Option B: Manual Testing (No API Key Needed)

1. Pick 3-5 articles from collected data
2. Go to https://claude.ai
3. Copy prompt from `prompts/future-of-education.md` (between the ``` markers)
4. Fill in article details manually
5. Paste into Claude and review JSON output

**Check for**:
- Scores make sense (EdTech products capped at 3, transformation stories 7+)
- Content type classification correct
- Reasoning explains scores well
- Key insights captured

---

## Step 4: Generate Intelligence Digest (30 minutes)

Create a sample weekly digest:

### Create `scripts/generate_education_digest.py`:

```python
import json
from datetime import datetime, timedelta
from collections import defaultdict

# Load labeled articles
labeled = []
with open('data/labeled/education_sample.jsonl', 'r') as f:
    for line in f:
        labeled.append(json.loads(line))

# Calculate education transformation scores
for article in labeled:
    labels = article['education_labels']
    score = (
        labels['paradox_engagement'] * 0.30 +
        labels['assessment_transformation'] * 0.20 +
        labels['curricular_innovation'] * 0.15 +
        labels['pedagogical_depth'] * 0.15 +
        labels['evidence_implementation'] * 0.10 +
        labels['cross_disciplinary_relevance'] * 0.05 +
        labels['discipline_specific_adaptation'] * 0.05
    )

    # Apply caps
    if labels['flags']['edtech_product']:
        score = min(score, 3.0)
    if labels['flags']['surface_ai_literacy']:
        score = min(score, 4.0)

    article['education_transformation_score'] = score

# Sort by score
labeled.sort(key=lambda x: x['education_transformation_score'], reverse=True)

# Generate digest
print("# Education Transformation Intelligence - Weekly Digest")
print(f"**Week of {datetime.now().strftime('%Y-%m-%d')}**\n")
print("---\n")

# High-impact stories (score >= 7.0)
high_impact = [a for a in labeled if a['education_transformation_score'] >= 7.0]
if high_impact:
    print("## 🌟 High-Impact Transformations\n")
    for i, article in enumerate(high_impact, 1):
        labels = article['education_labels']
        print(f"### {i}. {article['title']}")
        print(f"**Source**: {article.get('source', 'Unknown')} | **Score**: {article['education_transformation_score']:.1f}/10")
        print(f"**Type**: {labels['content_type'].replace('_', ' ').title()} | **Stage**: {labels['transformation_stage'].title()}")
        print(f"\n**Key Insight**: {labels['reasoning']}")

        if labels.get('transformation_examples'):
            print(f"\n**Examples**:")
            for ex in labels['transformation_examples'][:2]:
                print(f"- {ex}")

        print(f"\n[Read more]({article.get('url', '#')})")
        print("\n---\n")

# Assessment innovations (assessment_transformation >= 7)
assessment_innovations = [
    a for a in labeled
    if a['education_labels']['assessment_transformation'] >= 7.0
]
if assessment_innovations:
    print("## 📝 Assessment Innovations\n")
    for article in assessment_innovations[:3]:
        print(f"- **{article['title']}** ({article.get('source', 'Unknown')})")
        print(f"  {article['education_labels']['reasoning']}")
        print()

# Paradox deep dives (paradox_engagement >= 8)
paradox_articles = [
    a for a in labeled
    if a['education_labels']['paradox_engagement'] >= 8.0
]
if paradox_articles:
    print("## 🎯 Paradox Deep Dives\n")
    print("Articles deeply engaging with the execution vs. understanding paradox:\n")
    for article in paradox_articles[:3]:
        print(f"- **{article['title']}**")
        if article['education_labels'].get('notable_quotes'):
            print(f"  > {article['education_labels']['notable_quotes'][0]}")
        print(f"  [Read more]({article.get('url', '#')})")
        print()

# Discipline breakdown
disciplines = defaultdict(int)
for article in labeled:
    for discipline, covered in article['education_labels']['disciplines_covered'].items():
        if covered:
            disciplines[discipline] += 1

if disciplines:
    print("## 📊 Transformation by Discipline\n")
    sorted_disciplines = sorted(disciplines.items(), key=lambda x: x[1], reverse=True)
    for discipline, count in sorted_disciplines[:5]:
        print(f"- **{discipline.title()}**: {count} articles")
    print()

print("---")
print("\n*Generated with Future of Education semantic filter*")
print("*Powered by Claude 3.5 Sonnet*")
```

Run:
```bash
python scripts/generate_education_digest.py > reports/education_digest_sample.md
```

**Review the digest**:
- High-impact transformations (score 7+)
- Assessment innovations
- Paradox deep dives
- Discipline breakdown

---

## Next Steps

### If Filter Works Well (80%+ precision on top articles):

**Week 2-3: Scale Up**
```bash
# Collect 30 days of content
python run_aggregator.py --sources rss_education --days-back 30

# Label 100-200 articles
# Target: 500 labeled articles for training data
```

**Week 4: Build Downstream App**
- Create Education Intelligence Dashboard
- Weekly digest generator
- Assessment innovation tracker
- Paradox insights feed

**Week 5-8: Pilot with HAN University**
- Share digests with teacher training program
- Get feedback from 5-10 educators
- Refine based on feedback
- Target: €200-500/month pilot deal

### If Filter Needs Refinement:

**Calibration Issues**:
- Scores too high/low overall → Adjust dimension weights
- Missing important articles → Improve pre-filter keywords
- False positives → Tighten pre-classification filters

**Prompt Improvements**:
- Add more validation examples
- Refine scoring rubrics
- Clarify edge cases

**Re-test** on 20-30 articles after changes

---

## Cost Tracking

### Current Phase (Ground Truth Generation)

**Per Article**:
- Claude 3.5 Sonnet: ~$0.015 (1K tokens in, 500 out)
- Gemini 1.5 Pro: ~$0.010 (alternative)

**Milestones**:
- 50 articles (testing): $0.75
- 200 articles (validation): $3.00
- 500 articles (training dataset): $7.50
- 1,000 articles (robust dataset): $15.00

**Total to production-ready filter**: $10-20

### Future Phase (Fine-tuned Local Model)

**Per Article**:
- Llama 3 8B or DistilBERT: ~$0.001
- **100x cheaper** than frontier LLM

**Monthly at Scale**:
- 2,000 articles/month: $2 (vs. $200 with Claude)
- 10,000 articles/month: $10 (vs. $1,000 with Claude)

**ROI**: 2-5 months to break even on fine-tuning investment

---

## Troubleshooting

### RSS Feed Issues
**Problem**: Feeds returning errors
**Solution**: Check `config/sources/rss_education.yaml`, disable broken feeds

### No AI-Related Content
**Problem**: <10 AI articles in 150+ collected
**Solution**: Add more EdTech/tech-focused sources or lower pre-filter threshold

### Scores Too High/Low
**Problem**: All articles scoring 7+ or all scoring <4
**Solution**: Review validation examples, adjust dimension weights

### API Rate Limits
**Problem**: Claude API rate limit hit
**Solution**: Add delays between requests or switch to Gemini 1.5 Pro

### Out of Memory
**Problem**: Large articles causing memory issues
**Solution**: Truncate content to 4,000 characters before labeling

---

## Files Created

```
content-aggregator/
├── config/sources/
│   ├── rss_education.yaml ✅ (24 sources, 22 enabled)
│   └── README_EDUCATION.md ✅ (source breakdown)
├── prompts/
│   ├── future-of-education.md ✅ (664 lines, production-ready)
│   ├── sustainability.md ✅
│   ├── seece-energy-tech.md ✅
│   ├── uplifting.md ✅
│   └── README.md ✅ (integration guide)
├── scripts/
│   ├── test_education_feeds.py ✅ (RSS feed tester)
│   ├── test_education_filter.py (to create in Step 3)
│   └── generate_education_digest.py (to create in Step 4)
├── data/
│   ├── aggregated/ (collected articles)
│   ├── labeled/ (LLM-labeled articles)
│   └── training/ (fine-tuning datasets)
└── EDUCATION_QUICKSTART.md ✅ (this file)
```

---

## Success Criteria

### After Step 1 (RSS Testing):
- ✅ 18+ feeds working
- ✅ 200+ entries available

### After Step 2 (Collection):
- ✅ 150+ articles collected
- ✅ 30+ AI-related articles

### After Step 3 (Filter Testing):
- ✅ 80%+ precision on top 10 articles
- ✅ Scores align with expectations
- ✅ Content types classified correctly

### After Step 4 (Digest Generation):
- ✅ 3-5 high-impact stories per week
- ✅ Clear insights in reasoning
- ✅ Actionable for educators

---

## Support & Resources

**Documentation**:
- Filter design: `prompts/future-of-education.md`
- Source details: `config/sources/README_EDUCATION.md`
- Integration guide: `prompts/README.md`

**Downstream Applications**:
- Ideas & plans: `docs/separate-projects/`
- MVP roadmap: `docs/separate-projects/mvp-recommendation-action-plan.md`

**Questions**:
- Filter calibration issues → Review validation examples in prompt
- Integration questions → See `prompts/README.md`
- Downstream app design → See `docs/separate-projects/`

---

**Last Updated**: 2025-10-29
**Status**: Ready to test
**Next**: Run `python scripts/test_education_feeds.py`
