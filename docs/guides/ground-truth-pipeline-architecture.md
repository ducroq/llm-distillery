# Ground Truth Generation Pipeline Architecture

This document describes the architecture for generating ground truth data with application-specific pre-filtering strategies.

## Overview

The distillation pipeline follows this flow:

```
Raw Data → Profile → Design Filter → Pre-Filter → LLM Distillation → Ground Truth
```

Each application (uplifting, sustainability, etc.) needs its own pre-filtering strategy based on the raw data characteristics.

## Pipeline Stages

### Stage 1: Data Profiling

**Purpose:** Understand your raw data before distillation

**Tool:** `ground_truth/data_profiler.py`

```bash
python -m ground_truth.data_profiler datasets/raw/master_dataset.jsonl
```

**Outputs:**
- `datasets/raw/master_dataset_profile.json` - Machine-readable stats
- `docs/analysis/dataset-profile-YYYY-MM-DD.md` - Human-readable report

**What to analyze:**
- Source distribution
- Content length distribution
- Language breakdown
- Quality scores
- Sentiment/emotion availability
- Time distribution

**Key questions:**
1. What percentage of articles are too short (<100 words)?
2. Which sources are most relevant for your use case?
3. Is sentiment data available?
4. What languages are present?
5. What's the quality distribution?

### Stage 2: Pre-Filter Design

**Purpose:** Create application-specific filter to reduce costs

**Location:** `ground_truth/batch_labeler.py`

**Pre-filter template:**

```python
def your_application_pre_filter(article: Dict) -> bool:
    """
    Pre-filter for [YOUR APPLICATION].

    Criteria:
    1. [Criterion 1]
    2. [Criterion 2]
    ...

    Expected reduction: [X]% of articles filtered out
    """
    # 1. Minimum quality checks
    word_count = article.get('metadata', {}).get('word_count', 0)
    if word_count < 100:  # Adjust threshold
        return False

    quality = article.get('metadata', {}).get('quality_score', 1.0)
    if quality < 0.7:
        return False

    # 2. Application-specific logic
    # ... add your criteria here ...

    return True
```

**Common filter patterns:**

#### Pattern A: Emotion-Based (Uplifting)
```python
emotions = article.get('metadata', {}).get('raw_emotions', {})
joy = emotions.get('joy', 0)
negative_emotion = emotions.get('sadness', 0) + emotions.get('fear', 0)

# High joy OR low negative
return joy >= 0.15 or negative_emotion < 0.05
```

#### Pattern B: Keyword-Based (Sustainability)
```python
text = (article.get('title', '') + ' ' + article.get('content', '')).lower()

sustainability_keywords = [
    'climate', 'carbon', 'renewable', 'solar', 'wind',
    'battery', 'ev', 'electric', 'sustainability', 'emission'
]

return any(kw in text for kw in sustainability_keywords)
```

#### Pattern C: Source-Based (Tech Innovation)
```python
relevant_sources = [
    'science_arxiv_cs',
    'ai_techcrunch',
    'ai_the_verge',
    'industry_intelligence_fast_company'
]

return article.get('source') in relevant_sources
```

#### Pattern D: Combined Multi-Criteria
```python
# Combine multiple patterns
meets_length = word_count >= 100
meets_language = language == 'en'
meets_source = source in relevant_sources
meets_keywords = has_relevant_keywords(text)

return meets_length and meets_language and (meets_source or meets_keywords)
```

### Stage 3: Pre-Filter Testing

**Purpose:** Validate filter effectiveness before full run

**Tool:** Create custom test script (template below)

```python
"""Test your pre-filter."""
import json
from collections import Counter
from ground_truth.batch_labeler import your_application_pre_filter

def test_prefilter(dataset_path: str, sample_size: int = 10000):
    total = 0
    passed = 0

    with open(dataset_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break

            article = json.loads(line.strip())
            total += 1

            if your_application_pre_filter(article):
                passed += 1

    pass_rate = passed / total
    estimated_full = int(51869 * pass_rate)

    print(f"Pass rate: {pass_rate*100:.1f}%")
    print(f"Estimated articles: {estimated_full:,}")
    print(f"Estimated cost: ${estimated_full*0.00015:.2f}")
    print(f"Estimated time: {estimated_full*15/3600:.1f} hours")

if __name__ == '__main__':
    test_prefilter('datasets/raw/master_dataset.jsonl')
```

**What to check:**
- Pass rate (target: 10-30% for good filtering)
- Examples of passed articles (relevant?)
- Examples of failed articles (correctly rejected?)
- Cost/time estimates

### Stage 4: LLM Distillation

**Purpose:** Generate ground truth labels with LLM

**Tool:** `ground_truth/batch_labeler.py`

```bash
python -m ground_truth.batch_labeler \
    --prompt prompts/your_application.md \
    --source datasets/raw/master_dataset.jsonl \
    --llm gemini \
    --batch-size 50 \
    --pre-filter your_application \
    --output-dir datasets
```

**Parameters:**
- `--prompt`: Path to prompt markdown file
- `--source`: Input JSONL file
- `--llm`: Provider (claude, gemini, gpt4)
- `--batch-size`: Articles per batch
- `--pre-filter`: Pre-filter function name
- `--max-batches`: Limit for testing

**Monitoring:**
```bash
# Real-time log monitoring
tail -f datasets/your_application/distillation.log

# Check metrics
cat datasets/your_application/metrics.jsonl | jq '.success' | sort | uniq -c
```

### Stage 5: Quality Validation

**Purpose:** Validate distillation results

**Checks:**
1. **Success rate** (target: >95%)
2. **JSON repair usage** (target: <10%)
3. **Retry rate** (target: <20%)
4. **Error distribution** (identify patterns)

**Tools:**
```python
import json
import pandas as pd

# Load metrics
metrics = []
with open('datasets/your_application/metrics.jsonl', 'r') as f:
    for line in f:
        metrics.append(json.loads(line))

df = pd.DataFrame(metrics)

# Analysis
print(f"Success rate: {df['success'].mean()*100:.1f}%")
print(f"Avg time: {df['time_taken_seconds'].mean():.1f}s")
print(f"Retry rate: {(df['attempts_made'] > 1).mean()*100:.1f}%")
print(f"\nError distribution:")
print(df[~df['success']]['error_type'].value_counts())
```

## Application-Specific Configurations

### Example 1: Uplifting Content

**Profile insights** (from dataset-profile-2025-10-27.md):
- 47.8% articles <50 words (too short)
- Sentiment data missing (use emotions instead)
- Positive news sources: <1% of dataset

**Filter design:**
```python
def uplifting_pre_filter(article: Dict) -> bool:
    # Multi-criteria: length + language + emotions + keywords
    word_count = article.get('metadata', {}).get('word_count', 0)
    if word_count < 100:
        return False

    if article.get('language') != 'en':
        return False

    emotions = article.get('metadata', {}).get('raw_emotions', {})
    joy = emotions.get('joy', 0)
    negative = emotions.get('sadness', 0) + emotions.get('fear', 0)

    has_positive_emotion = joy >= 0.15 or negative < 0.05

    text = article.get('title', '').lower()
    uplifting_kw = ['breakthrough', 'innovation', 'solution', 'success']
    has_uplifting_kw = any(kw in text for kw in uplifting_kw)

    return has_positive_emotion or has_uplifting_kw
```

**Expected results:**
- Pass rate: ~18-20%
- Articles: ~9,600
- Cost: ~$1.44
- Time: ~40 hours (1.7 days)

### Example 2: Sustainability Content

**Filter design:**
```python
def sustainability_pre_filter(article: Dict) -> bool:
    # Source + keyword based
    source = article.get('source', '')

    # Relevant sources
    sustainability_sources = [
        'science_mdpi_sustainability',
        'energy_utilities_clean_technica',
        'climate_solutions_inside_climate_news',
        'energy_utilities_pv_magazine',
        'automotive_transport_electrek'
    ]

    if source in sustainability_sources:
        return True

    # Keyword fallback
    text = (article.get('title', '') + ' ' + article.get('content', ''))[:500].lower()
    keywords = [
        'climate', 'carbon', 'renewable', 'solar', 'wind', 'battery',
        'ev', 'electric', 'sustainability', 'green', 'emission',
        'net zero', 'clean energy', 'climate change'
    ]

    return any(kw in text for kw in keywords)
```

### Example 3: AI/ML Innovation

**Filter design:**
```python
def ai_innovation_pre_filter(article: Dict) -> bool:
    # Source + keyword + quality
    source = article.get('source', '')

    ai_sources = [
        'science_arxiv_cs',
        'ai_techcrunch',
        'ai_the_verge',
        'ai_mit_tech_review',
        'community_social_hacker_news'
    ]

    # Length requirement
    word_count = article.get('metadata', {}).get('word_count', 0)
    if word_count < 150:  # Longer for AI content
        return False

    # Source filter
    if source in ai_sources:
        return True

    # AI keywords
    text = (article.get('title', '') + ' ' + article.get('content', ''))[:500].lower()
    ai_keywords = [
        'artificial intelligence', 'machine learning', 'deep learning',
        'neural network', 'transformer', 'llm', 'gpt', 'ai model',
        'computer vision', 'nlp', 'reinforcement learning'
    ]

    return sum(1 for kw in ai_keywords if kw in text) >= 2  # At least 2 keywords
```

## Best Practices

### 1. Iterative Development

**Process:**
1. Profile data (1 hour)
2. Design initial filter (30 min)
3. Test on 1,000 articles (5 min)
4. Refine filter (30 min)
5. Test on 10,000 articles (30 min)
6. Run full distillation (1-3 days)

**Don't:** Jump straight to full distillation without testing

### 2. Filter Tuning

**Too strict (pass rate <5%):**
- Loosen criteria (lower thresholds)
- Add alternative paths (OR logic)
- Include more sources

**Too loose (pass rate >40%):**
- Add more criteria (AND logic)
- Raise thresholds
- Add negative keyword filters

**Target:** 10-30% pass rate for good ROI

### 3. Cost vs. Quality Trade-off

| Pass Rate | Articles | Cost | Quality | Use Case |
|-----------|----------|------|---------|----------|
| 5-10% | ~5K | $0.75 | Highest | Very specific needs |
| 10-20% | ~10K | $1.50 | High | **Recommended** |
| 20-30% | ~15K | $2.25 | Good | Broader coverage |
| 30-50% | ~20K | $3.00 | Medium | Initial exploration |
| >50% | >25K | >$4.00 | Lower | Minimal filtering |

### 4. Monitoring & Optimization

**During run:**
```bash
# Watch logs
tail -f datasets/your_app/distillation.log

# Count successes
grep "SUCCESS" datasets/your_app/distillation.log | wc -l

# Check errors
grep "FAILED" datasets/your_app/distillation.log | tail -20
```

**After completion:**
```bash
# Session summary
cat datasets/your_app/session_summary.json

# Error analysis
cat datasets/your_app/metrics.jsonl | jq -r 'select(.success==false) | .error_type' | sort | uniq -c
```

### 5. Documentation

For each application, document:
- **Filter logic**: Why each criterion?
- **Expected pass rate**: From testing
- **Test results**: Examples of pass/fail
- **Actual results**: From full run
- **Lessons learned**: What worked/didn't work

## Common Pitfalls

### ❌ Don't:
1. Skip profiling - leads to wasted time/money
2. Use sentiment_score without checking if it exists
3. Set thresholds too high (filters everything)
4. Forget language filtering (if English-only LLM)
5. Ignore source quality differences
6. Run full dataset without testing filter first

### ✅ Do:
1. Always profile data first
2. Test filters on 1K and 10K samples
3. Review passed/failed examples manually
4. Monitor logs during distillation
5. Analyze session metrics after completion
6. Document your filter logic and results

## File Structure

```
llm-distillery/
├── datasets/
│   ├── raw/
│   │   ├── master_dataset.jsonl          # Raw input data
│   │   ├── master_dataset_profile.json   # Profiling results
│   │   └── dataset_profile_report.txt    # Human-readable profile
│   │
│   └── <application>/                     # Per-application outputs
│       ├── labeled_batch_001.jsonl       # Distilled data
│       ├── labeled_batch_002.jsonl
│       ├── distillation.log              # Processing log
│       ├── metrics.jsonl                 # Per-article metrics
│       ├── session_summary.json          # Run statistics
│       ├── .labeled_ids.json             # Resume state
│       └── error_logs/                   # Failed responses
│           └── article_id_attempt1_timestamp.txt
│
├── ground_truth/
│   ├── batch_labeler.py                  # Main distillation engine
│   ├── data_profiler.py                  # Profiling tool
│   └── ...
│
├── prompts/
│   ├── uplifting.md                      # Application prompts
│   ├── sustainability.md
│   └── ...
│
└── docs/
    ├── analysis/
    │   └── dataset-profile-YYYY-MM-DD.md # Profile reports
    └── guides/
        └── this file
```

## Quick Reference

### Commands

```bash
# 1. Profile data
python -m ground_truth.data_profiler datasets/raw/master_dataset.jsonl

# 2. Test filter
python test_prefilter.py

# 3. Small test run
python -m ground_truth.batch_labeler \
    --prompt prompts/uplifting.md \
    --source datasets/raw/master_dataset.jsonl \
    --llm gemini \
    --batch-size 10 \
    --max-batches 1 \
    --pre-filter uplifting \
    --output-dir datasets/test

# 4. Full production run
python -m ground_truth.batch_labeler \
    --prompt prompts/uplifting.md \
    --source datasets/raw/master_dataset.jsonl \
    --llm gemini \
    --batch-size 50 \
    --pre-filter uplifting \
    --output-dir datasets
```

### Files to Check

- **Before:** `docs/analysis/dataset-profile-*.md`
- **During:** `datasets/<app>/distillation.log`
- **After:** `datasets/<app>/session_summary.json`
- **Errors:** `datasets/<app>/error_logs/*.txt`
- **Metrics:** `datasets/<app>/metrics.jsonl`

---

**Ready to build your pipeline!** Start with profiling, design your filter, test it, then run full distillation.
