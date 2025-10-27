# Dataset Profile Report
**Dataset:** `datasets/raw/master_dataset.jsonl`
**Analyzed:** 2025-10-27
**Total Articles:** 51,869

## Executive Summary

### Key Statistics
- **Total articles**: 51,869
- **Articles with entities**: 46,910 (90.4%) ✅
- **Average quality score**: 0.96 (excellent) ✅
- **Median word count**: 56 words
- **Date range**: Primarily October 2025 (99.96%)
- **Languages**: 83% English, 9% Dutch, 5% Spanish

### Critical Issues
1. **⚠️ 47.8% articles have <50 words** - Too short for meaningful distillation
2. **⚠️ 269 different sources** - Highly fragmented
3. **❌ Sentiment data missing** - All articles show "unknown" sentiment
4. **❌ Pre-filtering will be difficult** without sentiment scores

## Detailed Breakdown

### Source Distribution

**Top 10 Sources** (60% of dataset):

| Rank | Source | Count | % | Avg Words | Quality |
|------|--------|-------|---|-----------|---------|
| 1 | science_arxiv_cs | 10,361 | 20.0% | 181 | 0.96 |
| 2 | newsapi_general | 4,095 | 7.9% | 24 | 0.95 |
| 3 | science_arxiv_math | 2,917 | 5.6% | 123 | 0.97 |
| 4 | arxiv | 2,148 | 4.1% | 176 | 0.97 |
| 5 | global_news_el_pais | 1,576 | 3.0% | 25 | 0.97 |
| 6 | dutch_news_ad_algemeen | 1,416 | 2.7% | 43 | 0.98 |
| 7 | global_news_reuters | 1,356 | 2.6% | 12 | 1.00 |
| 8 | global_news_spiegel | 1,246 | 2.4% | 28 | 0.98 |
| 9 | science_arxiv_physics | 1,242 | 2.4% | 176 | 0.96 |
| 10 | science_biorxiv | 1,131 | 2.2% | 203 | 0.96 |

### Content Categories

**By Source Type:**
- **Academic papers** (~35%): arXiv CS, Math, Physics, BioRxiv
- **News aggregators** (~15%): NewsAPI, Reuters, BBC, El Pais
- **Community/Social** (~10%): Reddit, HackerNews, Dev.to, GitHub
- **Specialized news** (~15%): AI, Healthcare, Energy, Climate
- **Dutch news** (~10%): NOS, NRC, AD, RTL Nieuws
- **Positive news** (<1%): Upworthy, Good News Network, Positive News UK

### Language Distribution

| Language | Count | Percentage |
|----------|-------|------------|
| English | 43,076 | 83.0% |
| Dutch | 4,654 | 9.0% |
| Spanish | 2,479 | 4.8% |
| German | 1,257 | 2.4% |
| French | 277 | 0.5% |
| Other | 126 | 0.2% |

### Content Length Distribution

| Word Count | Articles | Percentage | Analysis |
|------------|----------|------------|----------|
| 0-50 | 24,787 | 47.8% | ❌ Too short |
| 50-100 | 4,797 | 9.2% | ⚠️ Minimal |
| 100-200 | 12,627 | 24.3% | ✅ Good |
| 200-500 | 8,300 | 16.0% | ✅ Good |
| 500-1000 | 868 | 1.7% | ✅ Excellent |
| 1000-5000 | 479 | 0.9% | ✅ Excellent |
| 5000+ | 11 | 0.0% | ✅ Long-form |

**Key stats:**
- Average: 126 words
- Median: 56 words
- Min: 10 words
- Max: 34,115 words

### Time Distribution

**Articles by month:**
- **October 2025**: 51,846 (99.96%)
- **August 2025**: 6
- **September 2025**: 6
- **Other dates**: 11 (data errors - includes 1969, 1991, 9999)

**Analysis:** Dataset is essentially a snapshot from October 2025.

### Quality Metrics

- **Average quality score**: 0.96
- **Median quality score**: 1.00
- **Articles <0.5 quality**: 104 (0.2%)

**Conclusion:** Quality is exceptionally high across all sources.

## Critical Finding: Missing Sentiment Data

**Problem:** All 51,869 articles show `sentiment_category: "unknown"`

### Impact on Pre-Filtering

The current `uplifting_pre_filter` function relies on:
```python
def uplifting_pre_filter(article: Dict) -> bool:
    sentiment_score = article.get('metadata', {}).get('sentiment_score', 0)
    joy = article.get('metadata', {}).get('raw_emotions', {}).get('joy', 0)
    return sentiment_score >= 5.0 or joy >= 0.25
```

**This will FAIL** because:
1. `sentiment_score` appears to be 0 or missing in all articles
2. `raw_emotions` data may also be missing

### Investigation Needed

Check actual article structure:
```bash
head -1 datasets/raw/master_dataset.jsonl | jq '.sentiment_score, .metadata.sentiment_score, .raw_emotions'
```

## Pre-Filtering Recommendations

### Option 1: Fix Sentiment Data First

**Best approach** if you need sentiment-based filtering:

1. **Run sentiment analysis** on the dataset
2. **Add sentiment fields** to articles
3. **Then** apply uplifting pre-filter

**Time required:** ~5-10 hours for VADER sentiment on 51K articles

### Option 2: Alternative Pre-Filters (No Sentiment Needed)

#### A. Word Count Filter (Simplest)
**Effect:** Removes 47.8% of articles (24,787)
**Remaining:** 27,082 articles
**Savings:** $3.72, 101 hours

```python
def word_count_pre_filter(article: Dict) -> bool:
    word_count = article.get('metadata', {}).get('word_count', 0)
    return word_count >= 50
```

#### B. Source-Based Filter (Targeted)
**Effect:** Focus on sources likely to contain uplifting content
**Example sources:**
- Positive news: upworthy, good_news_network, positive_news_uk, the_better_india
- Innovation: fast_company, tech_eu, innovation_news
- Science breakthroughs: science_arxiv_*, nature_news, sciencedaily
- Community: reddit_uplifting_news, reddit_made_me_smile

```python
def source_based_pre_filter(article: Dict) -> bool:
    uplifting_sources = [
        'positive_news_upworthy',
        'positive_news_good_news_network',
        'positive_news_the_better_india',
        'positive_news_positive_news_uk',
        'industry_intelligence_fast_company',
        'community_social_reddit_uplifting_news',
        'community_social_reddit_made_me_smile',
        # Add more...
    ]
    return article.get('source') in uplifting_sources
```

#### C. Keyword-Based Filter (Content)
**Effect:** Filter by uplifting keywords in title/content

```python
def keyword_pre_filter(article: Dict) -> bool:
    text = (article.get('title', '') + ' ' + article.get('content', '')).lower()

    uplifting_keywords = [
        'breakthrough', 'innovation', 'solution', 'success', 'achievement',
        'hope', 'progress', 'inspiring', 'positive', 'impact', 'breakthrough',
        'transforms', 'improves', 'saves', 'helps', 'benefits', 'advance'
    ]

    negative_keywords = [
        'war', 'death', 'disaster', 'crisis', 'catastrophe', 'attack',
        'violence', 'corruption', 'scandal', 'crisis'
    ]

    has_uplifting = any(kw in text for kw in uplifting_keywords)
    has_negative = any(kw in text for kw in negative_keywords)

    return has_uplifting and not has_negative
```

#### D. Combined Filter (Best)
**Effect:** Multiple criteria for better results

```python
def smart_pre_filter(article: Dict) -> bool:
    # 1. Minimum length
    word_count = article.get('metadata', {}).get('word_count', 0)
    if word_count < 100:  # More strict than 50
        return False

    # 2. Exclude very short news and GitHub
    source = article.get('source', '')
    if source in ['github', 'newsapi_general']:
        return False

    # 3. Language filter (English only for uplifting)
    if article.get('language') != 'en':
        return False

    # 4. Quality threshold
    quality = article.get('metadata', {}).get('quality_score', 1.0)
    if quality < 0.8:
        return False

    # 5. Keyword check (optional)
    text = (article.get('title', '') + ' ' + article.get('content', '')).lower()
    uplifting_keywords = ['breakthrough', 'innovation', 'solution', 'success']
    has_uplifting = any(kw in text for kw in uplifting_keywords)

    return has_uplifting
```

### Estimated Results by Filter

| Filter Strategy | Articles Remaining | Est. Cost | Est. Time | Savings |
|-----------------|-------------------|-----------|-----------|---------|
| **No filter** | 51,869 | $7.78 | 216h (9d) | - |
| **Word count ≥50** | 27,082 | $4.06 | 113h (4.7d) | $3.72, 103h |
| **Word count ≥100** | 22,285 | $3.34 | 93h (3.9d) | $4.44, 123h |
| **English + ≥100 words** | 18,497 | $2.77 | 77h (3.2d) | $5.01, 139h |
| **Smart combined** | ~10,000 | $1.50 | 42h (1.8d) | $6.28, 174h |

## Cost-Benefit Analysis

### Full Dataset (No Filtering)
- **Articles**: 51,869
- **Time**: 216 hours (~9 days)
- **Cost**: ~$7.78 (Gemini @ $0.00015/request)
- **Issues**: ~25K articles too short, many irrelevant

### With Smart Pre-Filtering (Recommended)
- **Articles**: ~10,000 (80% reduction)
- **Time**: 42 hours (~1.8 days)
- **Cost**: ~$1.50
- **Savings**: $6.28 + 174 hours (7.3 days)
- **Quality**: Much higher relevance

### ROI Calculation
- **Time savings**: 174 hours = 4.3 work weeks
- **Cost savings**: $6.28
- **Quality improvement**: ~3-5x fewer irrelevant results
- **Review time saved**: ~80% less manual filtering needed

## Recommendations

### Immediate Actions

1. **Investigate sentiment data**:
   ```bash
   python -c "
   import json
   with open('datasets/raw/master_dataset.jsonl', 'r') as f:
       article = json.loads(f.readline())
       print('sentiment_score:', article.get('sentiment_score'))
       print('sentiment_category:', article.get('sentiment_category'))
       print('raw_emotions:', article.get('metadata', {}).get('raw_emotions'))
   "
   ```

2. **Implement combined pre-filter** for best results

3. **Test on small batch** (100 articles):
   ```bash
   python -m ground_truth.batch_labeler \
       --prompt prompts/uplifting.md \
       --source datasets/raw/master_dataset.jsonl \
       --llm gemini \
       --batch-size 100 \
       --max-batches 1 \
       --pre-filter uplifting \
       --output-dir datasets/test
   ```

4. **Review test results** before full run

### For Uplifting Content Specifically

**Best sources to target**:
- `positive_news_*` (548 articles) - Explicitly positive news
- `industry_intelligence_fast_company` (449 articles) - Innovation stories
- `community_social_reddit_uplifting_news` (8 articles) - Curated uplifting
- `community_social_reddit_made_me_smile` (61 articles) - Positive content
- Science breakthroughs from arxiv_* and nature_news

**Sources to exclude**:
- `newsapi_general` (4,095) - Generic short news snippets
- `github` (987) - Code repositories, not articles
- War/conflict sources - Various military/defense
- Financial news - Often neutral/negative

### For Sustainability Content

**Best sources**:
- `science_mdpi_sustainability` (579)
- `energy_utilities_*` (several hundred)
- `climate_solutions_*` (several hundred)
- `automotive_transport_electrek` (201) - EV news
- Science papers from relevant arXiv categories

## Next Steps

1. ✅ **Profile completed** - This document
2. 🔄 **Check sentiment data** - Verify article structure
3. ⏳ **Build improved pre-filters** - Based on findings
4. ⏳ **Test locally** - Small batch with metrics
5. ⏳ **Design pipeline** - Configurable pre-filtering system
6. ⏳ **Full distillation** - Run on server with best filter

## Files Generated

- **This report**: `docs/analysis/dataset-profile-2025-10-27.md`
- **JSON summary**: `datasets/raw/master_dataset_profile.json`
- **Raw output**: `datasets/raw/dataset_profile_report.txt`

---

**Analysis complete.** Next: Investigate sentiment data and build application-specific pre-filters.
