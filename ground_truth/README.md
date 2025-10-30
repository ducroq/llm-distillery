# Ground Truth Generation

Tools for generating high-quality labeled datasets using LLM oracles (Claude, Gemini, GPT-4).

## 📋 Workflow Overview

```
1a. Pre-filter Calibration  →  1b. Oracle Calibration  →  2. Generate Ground Truth  →  3. Fine-tune Models
    (500 articles)              (100 articles)              (2,500-5,000 articles)        (Qwen 2.5-7B)
```

**Key Concept**: Pre-filters block obvious noise BEFORE expensive oracle labeling, saving 5-30% of API costs.

---

## 1️⃣ Calibration Phase (Two Steps)

### 1a. Pre-filter Calibration

**Purpose**: Test your pre-filter's blocking patterns and measure pass rate.

**Why**: Pre-filters save money by blocking irrelevant articles before sending to expensive LLMs. You need to calibrate to ensure:
- Pass rate is reasonable (70-95%)
- Blocking reasons make sense
- No false negatives on important content

**Usage:**

```bash
python -m ground_truth.calibrate_prefilter \
    --filter filters/uplifting/v1 \
    --source "datasets/raw/master_dataset_*.jsonl" \
    --sample-size 500
```

**What It Does:**
1. Loads filter package (prefilter.py + config.yaml)
2. Tests pre-filter on 500 random articles
3. Shows pass rate and blocking reasons
4. Identifies potential issues (false negatives)

**Sample Output:**

```
Pre-filter Calibration: UpliftingPreFilterV1
============================================

Total articles tested:  500
Passed pre-filter:      474 (94.8%)
Blocked by pre-filter:  26 (5.2%)

Block reasons:
  - Rage/outrage/negativity: 15 (57.7%)
  - Decline/crisis: 11 (42.3%)

Recommendation: ✓ Pass rate looks good (94.8%)
```

**Decision Matrix:**

| Pass Rate | Recommendation |
|-----------|----------------|
| 95-100% | Pre-filter may be too lenient - consider stricter patterns |
| 70-95% | ✅ **GOOD** - Blocking noise while keeping signal |
| 50-70% | Pre-filter may be too strict - review blocked articles |
| < 50% | Pre-filter is blocking too much - needs adjustment |

---

### 1b. Oracle Calibration

**Purpose**: Compare LLM models (Flash vs Pro vs Sonnet) on articles that PASS your pre-filter.

**Why**: Gemini Flash is 50x cheaper than Claude ($0.00018/article vs $0.009/article), but you need to verify it gives similar quality to Gemini Pro or Claude Sonnet for your specific filter.

**Usage:**

```bash
# Default: Compare Gemini Flash vs Gemini Pro (cheap + cheaper)
python -m ground_truth.calibrate_oracle \
    --filter filters/uplifting/v1 \
    --source "datasets/raw/master_dataset_*.jsonl" \
    --sample-size 100 \
    --models gemini-flash,gemini-pro

# Optional: Compare against Claude Sonnet (expensive but high quality)
python -m ground_truth.calibrate_oracle \
    --filter filters/uplifting/v1 \
    --source "datasets/raw/master_dataset_*.jsonl" \
    --sample-size 100 \
    --models gemini-flash,claude-sonnet
```

**What It Does:**

1. Loads filter package (prefilter.py + prompt-compressed.md + config.yaml)
2. Applies **pre-filter first** to 100 sampled articles
3. Labels articles that **pass pre-filter** with both models
4. Compares tier distributions, score statistics, quality adherence
5. Identifies articles with largest score disagreements
6. Generates detailed markdown report with recommendation

**Sample Output:**

```
======================================================================
APPLYING PRE-FILTER: UpliftingPreFilterV1
======================================================================

Pre-filter Results:
  Total sampled:  100
  Passed:         95 (95.0%)
  Blocked:        5 (5.0%)

  Block reasons:
    - Rage/outrage/negativity: 3 (60.0%)
    - Decline/crisis: 2 (40.0%)

Proceeding with 95 passed articles for oracle labeling...

======================================================================
CALIBRATION COMPARISON: gemini-flash vs gemini-pro
======================================================================

TIER DISTRIBUTION:
  Tier                 Flash           Pro         Difference
  -------------------- --------------- ----------- ---------------
  impact                  12.6%          11.6%         -1.0%
  connection              34.7%          36.8%         +2.1%
  not_uplifting           52.7%          51.6%         -1.1%

SCORE STATISTICS:
  Average:  5.45 vs 5.52 (diff: +0.07)
  Median:   5.20 vs 5.30 (diff: +0.10)

COST COMPARISON (for 5,000 articles):
  Gemini Flash:  $0.90
  Gemini Pro:    $0.90
  Claude Sonnet: $45.00 (50x more expensive)

RECOMMENDATION:
  Distributions are very similar (tier diff: 4.2%, score diff: 0.07)
  RECOMMENDED: Use Gemini Flash for large-scale labeling (cheapest, similar quality)
```

**Decision Matrix:**

| Tier Difference | Score Difference | Recommendation |
|----------------|------------------|----------------|
| < 10% | < 0.5 | **Use Gemini Flash** (cheapest, very similar quality) |
| 10-20% | 0.5-1.0 | **Use Gemini Pro** (slightly better, same cost tier) |
| > 20% | > 1.0 | **Use Claude Sonnet** or refine prompt for better Gemini consistency |

---

## 2️⃣ Ground Truth Generation (Batch Labeling)

**Purpose**: Generate ground truth labels for thousands of articles using your chosen LLM and pre-filter.

**When to Run**: After calibration shows your pre-filter and oracle are working well.

### Usage

```bash
python -m ground_truth.batch_labeler \
    --filter filters/uplifting/v1 \
    --source "datasets/raw/master_dataset_*.jsonl" \
    --output-dir datasets/uplifting \
    --llm gemini-flash \
    --batch-size 50 \
    --max-batches 50
```

**This will:**
1. Load filter package (prefilter.py + prompt-compressed.md)
2. Apply pre-filter to block obvious noise (saves 5-30% API calls)
3. Label articles that pass pre-filter with Gemini Flash
4. Save labeled articles to `datasets/uplifting/labeled_batch_*.jsonl`

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--filter` | Path to filter package directory | *required* |
| `--source` | Source JSONL file(s), supports globs | *required* |
| `--output-dir` | Output directory for labeled data | *required* |
| `--llm` | LLM provider: `gemini`, `gemini-pro`, `claude` | `gemini` |
| `--batch-size` | Articles per batch (resume checkpoint) | `50` |
| `--max-batches` | Maximum batches to process | `None` (all) |

**Legacy Support**: `--prompt` parameter still works but doesn't load pre-filter (not recommended).

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

## 3️⃣ Filter Packages (Integrated Pre-filters)

**Purpose**: Filter packages bundle pre-filters, prompts, and configuration into versioned directories.

### Filter Package Structure

```
filters/<filter-name>/v<version>/
├── prefilter.py              # Rule-based pre-filter (blocks noise)
├── prompt-compressed.md      # Compressed prompt for batch labeling
├── prompt-extended.md        # Full prompt with examples (optional)
├── config.yaml              # Weights, thresholds, tier boundaries
└── README.md                # Filter documentation
```

### Example: Uplifting Pre-Filter

See `filters/uplifting/v1/prefilter.py`:

```python
class UpliftingPreFilterV1:
    """Fast rule-based pre-filter for uplifting content."""
    VERSION = "1.0"

    RAGE_PATTERNS = [
        r'\b(outrage|fury|anger|rage)\b',
        r'\b(slam|blast|rip into)\b',
        # ... more patterns
    ]

    DECLINE_PATTERNS = [
        r'\b(decline|crisis|collapse|disaster)\b',
        # ... more patterns
    ]

    def should_label(self, article: Dict) -> Tuple[bool, str]:
        """
        Returns:
            (True, "passed") if article should be labeled
            (False, reason) if article should be blocked
        """
        text = (article.get('title', '') + ' ' +
                article.get('content', '')).lower()

        if any(re.search(p, text) for p in self.RAGE_PATTERNS):
            return False, "Rage/outrage/negativity"

        if any(re.search(p, text) for p in self.DECLINE_PATTERNS):
            return False, "Decline/crisis"

        return True, "passed"
```

### Cost Savings with Pre-filters

**Example:** Uplifting filter with 5.2% blocking rate

- **Without pre-filter**: 5,000 articles × $0.00018 = $0.90
- **With pre-filter**: 4,740 articles × $0.00018 = $0.85
- **Savings**: $0.05 (5.2%)

**Note**: Even small savings add up! For 50,000 articles = $2.50 saved. Plus faster processing time.

---

## 📂 Directory Structure

```
ground_truth/
├── README.md                       # This file
├── batch_labeler.py                # Generic batch labeling engine
├── calibrate_oracle.py             # Oracle model comparison
├── calibrate_prefilter.py          # Pre-filter calibration
├── secrets_manager.py              # API key management
└── __init__.py

filters/                            # Filter packages (single source of truth)
├── uplifting/v1/
│   ├── prefilter.py
│   ├── prompt-compressed.md
│   ├── prompt-extended.md
│   ├── config.yaml
│   └── README.md
├── sustainability/v1/
│   └── ...
└── investment-risk/v1/
    └── ...

datasets/
├── raw/
│   ├── master_dataset_20250929_20251008.jsonl
│   └── master_dataset_*.jsonl
├── uplifting/
│   ├── labeled_batch_001.jsonl
│   ├── labeled_batch_002.jsonl
│   ├── distillation.log
│   ├── metrics.jsonl
│   └── .labeled_ids.json           # Resume state
└── sustainability/
    └── ...

calibrations/                       # Pre-filter calibration results
├── uplifting/
│   ├── prefilter_test_sample.jsonl
│   ├── prefilter_blocked.jsonl
│   └── prefilter_report.txt
└── sustainability/
    └── ...

reports/                            # Oracle calibration reports
├── uplifting_calibration.md
├── sustainability_calibration.md
└── investment_risk_calibration.md
```

---

## 🔧 Advanced Usage

### Create Custom Filter Package

1. **Copy existing filter package:**
   ```bash
   cp -r filters/uplifting/v1 filters/my-filter/v1
   ```

2. **Update prefilter.py** with your blocking patterns:
   ```python
   class MyFilterPreFilterV1:
       VERSION = "1.0"

       def should_label(self, article: Dict) -> Tuple[bool, str]:
           # Your logic here
           return True, "passed"
   ```

3. **Edit prompt-compressed.md** with your scoring dimensions

4. **Update config.yaml** with weights and thresholds

5. **Test the filter package:**
   ```bash
   # Pre-filter calibration
   python -m ground_truth.calibrate_prefilter \
       --filter filters/my-filter/v1 \
       --source "datasets/raw/master_dataset_*.jsonl" \
       --sample-size 500

   # Oracle calibration
   python -m ground_truth.calibrate_oracle \
       --filter filters/my-filter/v1 \
       --source "datasets/raw/master_dataset_*.jsonl" \
       --sample-size 100
   ```

6. **Generate ground truth:**
   ```bash
   python -m ground_truth.batch_labeler \
       --filter filters/my-filter/v1 \
       --source "datasets/raw/master_dataset_*.jsonl" \
       --output-dir datasets/my-filter \
       --llm gemini-flash
   ```

### Prompt Format Requirements

Your `prompt-compressed.md` must follow this structure:

````markdown
Analyze this article and score it on the following dimensions:

**Dimension 1**: [description]
- Score 1-3: [criteria]
- Score 4-6: [criteria]
- Score 7-10: [criteria]

**Dimension 2**: [description]
...

**Output Format**: Return ONLY a JSON object with this structure:
```json
{
  "dimension1": <1-10>,
  "dimension2": <1-10>,
  "overall_score": <float>,
  "tier": "impact|connection|not_relevant",
  "reasoning": "<brief explanation>"
}
```
````

---

## 🎯 Best Practices

### 1. Always Run Two-Phase Calibration

**Phase 1a - Pre-filter Calibration (500 articles):**
- Verify pre-filter pass rate is 70-95%
- Check blocking reasons make sense
- Look for false negatives

**Phase 1b - Oracle Calibration (100 articles):**
- Compare Gemini Flash vs Gemini Pro (or Claude)
- Ensure tier distributions are similar (< 10% difference)
- Review articles with largest disagreements

**Time investment**: 30 minutes saves $10-40+ on large batches

### 2. Start Small, Scale Gradually

- 500 articles: Pre-filter calibration
- 100 articles: Oracle calibration, check tier distribution
- 2,500 articles: First production dataset (recommended)
- 5,000 articles: Larger dataset if needed
- 10,000+ articles: Only if Qwen fine-tuning requires more data

### 3. Use Random Sampling (Simple is Good!)

For RSS-fed datasets, random sampling already reflects production reality:
- Your sources are already curated (ArXiv, Nature, etc.)
- Pre-filter handles noise removal
- No need for complex stratification

### 4. Pre-filters Are Not Optional

Pre-filtering provides:
- 5-30% cost savings (even small savings add up!)
- Faster processing time
- Better signal-to-noise ratio

Every filter package should have a prefilter.py with blocking patterns.

### 5. Monitor Quality During Generation

After first 100 labeled articles, check:
- ✅ Tier distribution matches calibration expectations
- ✅ Pre-filter blocking rate is consistent
- ✅ JSON parsing success rate > 95%
- ✅ No unexpected errors in distillation.log

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

## 🚀 Quick Start

**Ready to generate ground truth? Follow the two-phase calibration workflow:**

### Step 1: Pre-filter Calibration (500 articles)

```bash
python -m ground_truth.calibrate_prefilter \
    --filter filters/uplifting/v1 \
    --source "datasets/raw/master_dataset_*.jsonl" \
    --sample-size 500
```

**Goal**: Verify pre-filter pass rate is 70-95% and blocking patterns are correct.

### Step 2: Oracle Calibration (100 articles)

```bash
python -m ground_truth.calibrate_oracle \
    --filter filters/uplifting/v1 \
    --source "datasets/raw/master_dataset_*.jsonl" \
    --sample-size 100 \
    --models gemini-flash,gemini-pro
```

**Goal**: Compare models on articles that passed pre-filter. Choose cheapest model with similar quality.

### Step 3: Generate Ground Truth (2,500 articles)

```bash
python -m ground_truth.batch_labeler \
    --filter filters/uplifting/v1 \
    --source "datasets/raw/master_dataset_*.jsonl" \
    --output-dir datasets/uplifting \
    --llm gemini-flash \
    --batch-size 50 \
    --max-batches 50
```

**Time**: 3-4 hours | **Cost**: ~$0.85-0.90 with pre-filter

---

**See also:**
- [Filter Development Guide](../filters/README.md) - Create new filter packages
- [Qwen Fine-tuning Guide](../docs/guides/qwen-finetuning-guide.md) - Next step after ground truth generation
