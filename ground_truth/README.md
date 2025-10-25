# Ground Truth Generation

Tools for generating high-quality labeled datasets using LLM oracles (Claude, Gemini, GPT-4).

## 📋 Workflow Overview

```
1. Calibrate Models   →  2. Generate Ground Truth  →  3. Fine-tune Models
   (100 articles)         (5,000-50,000 articles)      (DeBERTa, DistilBERT)
```

---

## 1️⃣ Model Calibration

**Purpose**: Compare Claude vs Gemini on your prompt to decide which LLM to use for large-scale labeling.

**Why**: Gemini is 50x cheaper ($0.00018/article vs $0.009/article), but you need to verify it gives similar quality to Claude for your specific prompt.

### Usage

```bash
python -m ground_truth.calibrate_models \
    --prompt prompts/sustainability.md \
    --source ../content-aggregator/data/collected/articles.jsonl \
    --sample-size 100 \
    --output reports/sustainability_calibration.md
```

### What It Does

1. Selects 100 random articles from your source
2. Labels them with **both Claude and Gemini**
3. Compares tier distributions, score statistics, quality adherence
4. Identifies articles with largest score disagreements
5. Generates detailed markdown report with recommendation

### Sample Output

```
CALIBRATION COMPARISON: Claude vs Gemini
========================================

TIER DISTRIBUTION:
  Tier                 Claude          Gemini      Difference
  -------------------- --------------- --------------- ---------------
  impact                  12.0%           10.0%           -2.0%
  connection              34.0%           38.0%           +4.0%
  not_uplifting           54.0%           52.0%           -2.0%

SCORE STATISTICS:
  Average:  3.45 vs 3.52 (diff: +0.07)
  Median:   3.20 vs 3.30 (diff: +0.10)

COST COMPARISON (for 5,000 articles):
  Claude:  $45.00
  Gemini:  $0.90 (50x cheaper)
  Savings: $44.10 by using Gemini

RECOMMENDATION:
  Distributions are very similar (tier diff: 8.0%, score diff: 0.07)
  RECOMMENDED: Use Gemini for large-scale labeling (50x cheaper, similar quality)
```

### Decision Matrix

| Tier Difference | Score Difference | Recommendation |
|----------------|------------------|----------------|
| < 10% | < 0.5 | **Use Gemini** (50x cheaper, very similar) |
| 10-20% | 0.5-1.0 | **Use Gemini** but spot-check 100 with Claude |
| > 20% | > 1.0 | **Use Claude** or refine prompt for better Gemini consistency |

---

## 2️⃣ Batch Labeling

**Purpose**: Generate ground truth labels for thousands of articles using your chosen LLM.

### Usage

```bash
python -m ground_truth.batch_labeler \
    --prompt prompts/sustainability.md \
    --source ../content-aggregator/data/collected/articles.jsonl \
    --output datasets/sustainability_5k.jsonl \
    --llm gemini \
    --batch-size 50 \
    --max-batches 100
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--prompt` | Path to prompt markdown file | *required* |
| `--source` | Source JSONL file with articles | *required* |
| `--output` | Output directory for labeled data | `datasets/` |
| `--llm` | LLM provider: `claude`, `gemini`, or `gpt4` | `claude` |
| `--batch-size` | Articles per batch (resume checkpoint) | `50` |
| `--max-batches` | Maximum batches to process | `None` (all) |
| `--pre-filter` | Optional pre-filter function | `None` |

### Features

- **Resume capability**: Tracks processed articles, can stop/restart anytime
- **Rate limiting**: Respects API rate limits (Claude: 40 RPM, Gemini: 120 RPM)
- **Batch checkpointing**: Saves every 50 articles (configurable)
- **Progress tracking**: Real-time progress and ETA
- **Error handling**: Retries failed API calls, logs errors

### Output Structure

Labeled articles are saved to `datasets/<filter_name>/labeled_batch_NNN.jsonl`:

```json
{
  "id": "article_123",
  "title": "...",
  "content": "...",
  "sustainability_analysis": {
    "tier": "impact",
    "sustainability_score": 8.2,
    "dimensions": {
      "climate_impact": 9,
      "technical_credibility": 8,
      "deployment_readiness": 7,
      "scale_potential": 8,
      "innovation_quality": 6
    },
    "reasoning": "...",
    "content_type": "breakthrough_research",
    "greenwashing_risk": false
  }
}
```

### Cost Estimates

| Sample Size | Claude 3.5 Sonnet | Gemini 1.5 Pro (Tier 1) |
|-------------|-------------------|-------------------------|
| 100 articles | $0.90 | $0.02 |
| 1,000 articles | $9.00 | $0.18 |
| 5,000 articles | $45.00 | $0.90 |
| 10,000 articles | $90.00 | $1.80 |
| 50,000 articles | $450.00 | $9.00 |

**Note**: Gemini pricing requires Cloud Billing enabled (Tier 1: 150 RPM).

---

## 3️⃣ Pre-Filters (Optional)

**Purpose**: Reduce labeling costs by 40-60% by filtering out irrelevant articles before sending to LLM.

### Example: Sustainability Pre-Filter

```python
def sustainability_pre_filter(article):
    """Only analyze articles likely to be sustainability-related."""
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

### Usage

Pass pre-filter function to batch labeler:

```python
from ground_truth.batch_labeler import GenericBatchLabeler

labeler = GenericBatchLabeler(
    prompt_path='prompts/sustainability.md',
    llm_provider='gemini',
    pre_filter=sustainability_pre_filter
)
```

### Cost Savings

- **Without pre-filter**: 10,000 articles × $0.00018 = $1.80
- **With 50% pre-filter**: 5,000 articles × $0.00018 = $0.90
- **Savings**: $0.90 (50%)

---

## 📂 Directory Structure

```
ground_truth/
├── README.md                    # This file
├── batch_labeler.py             # Generic batch labeling engine
├── calibrate_models.py          # Claude vs Gemini comparison
├── secrets_manager.py           # API key management
└── __init__.py

datasets/
├── sustainability/
│   ├── labeled_batch_001.jsonl
│   ├── labeled_batch_002.jsonl
│   └── .labeled_ids.json        # Resume state
└── uplifting/
    ├── labeled_batch_001.jsonl
    └── .labeled_ids.json

prompts/
├── sustainability.md            # Sustainability filter prompt
├── uplifting.md                 # Uplifting filter prompt
└── <your_filter>.md             # Your custom prompts

reports/
├── sustainability_calibration.md
└── uplifting_calibration.md
```

---

## 🔧 Advanced Usage

### Create Custom Filter Prompt

1. Copy an existing prompt: `cp prompts/uplifting.md prompts/my_filter.md`
2. Edit the prompt with your scoring dimensions
3. Test with calibration: `python -m ground_truth.calibrate_models --prompt prompts/my_filter.md ...`
4. Generate ground truth: `python -m ground_truth.batch_labeler --prompt prompts/my_filter.md ...`

### Prompt Format Requirements

Your prompt markdown must include:

````markdown
## Prompt

```
Analyze this article and score it on the following dimensions:

<Your dimensions and scoring criteria here>

Respond with ONLY valid JSON in this exact format:
{
  "dimension1": <score>,
  "dimension2": <score>,
  ...
}

DO NOT include any text outside the JSON object.
```
````

The batch labeler extracts the prompt between:
- Start marker: ` ```\nAnalyze this article`
- End marker: `DO NOT include any text outside the JSON object.\n``` `

---

## 🎯 Best Practices

### 1. Start Small, Scale Gradually

- 10 articles: Verify prompt works, API keys work
- 100 articles: Calibrate models, check tier distribution
- 1,000 articles: Validate quality, review edge cases
- 5,000 articles: First production dataset
- 50,000 articles: Full-scale ground truth (if needed)

### 2. Always Calibrate First

Don't skip model calibration! 10 minutes testing 100 articles can save you $40+ on a 5,000-article batch.

### 3. Use Stratified Sampling

Don't rely on pure random sampling:
- 70% stratified by source category
- 20% edge cases (greenwashing, vaporware, etc.)
- 10% random

### 4. Implement Pre-Filters

Pre-filtering reduces costs by 40-60% with minimal false negatives.

### 5. Monitor Quality Early

After first 100 articles, check:
- ✅ Tier distribution matches expectations
- ✅ Edge cases handled correctly
- ✅ JSON parsing success rate > 95%

---

## 📚 Related Documentation

- [API Keys & Secrets Management](../docs/secrets_management.md)
- [Ground Truth Best Practices](../docs/ground_truth_best_practices.md)
- [Sustainability Filter Framework](../prompts/sustainability.md)
- [Uplifting Filter Framework](../prompts/uplifting.md)

---

## 🆘 Troubleshooting

### Error: "API key not found"

**Solution**: Create `config/credentials/secrets.ini` or set environment variables. See [secrets_management.md](../docs/secrets_management.md).

### Error: "RateLimitError: Too many requests"

**Solution**:
- For Gemini: Enable Cloud Billing for Tier 1 (150 RPM)
- For Claude: Script already uses 1.5s delay (should not happen)

### Tier distributions look wrong

**Solution**:
- Run calibration first to validate prompt
- Check prompt clarity - add more examples
- Review edge cases in calibration report

### JSON parsing errors > 5%

**Solution**:
- Simplify prompt output format
- Add clearer instructions: "Respond with ONLY valid JSON"
- Remove any markdown formatting from prompt

---

**Ready to generate ground truth? Start with model calibration!**

```bash
python -m ground_truth.calibrate_models \
    --prompt prompts/sustainability.md \
    --source ../content-aggregator/data/collected/articles.jsonl \
    --sample-size 100
```
