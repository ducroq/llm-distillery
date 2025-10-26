# Model Calibration Guide

## Overview

Model calibration helps you choose the best LLM provider (Claude vs Gemini vs GPT-4) for generating ground truth datasets. This guide shows you how to run calibration and interpret results.

## Why Calibrate?

**Problem**: Different LLMs produce different ratings for the same semantic filter.

**Solution**: Run both Claude and Gemini on a sample (100 articles), compare their outputs, and decide:
- **Use Claude**: Higher quality, better consistency ($0.003/article)
- **Use Gemini**: 96% cheaper, good for cost savings ($0.00018/article with billing)

**When to calibrate**:
- Before generating large (50K+) ground truth datasets
- When creating a new semantic filter
- Quarterly, to detect model drift
- When switching LLM providers

## Quick Start

### 1. Set Up API Keys

Create `config/credentials/secrets.ini`:

```ini
[api_keys]
anthropic_api_key = sk-ant-your_key_here
gemini_api_key = AIza_your_key_here
gemini_billing_api_key = AIza_your_billing_key_here  # Optional: 150 RPM vs 2 RPM
```

**Important**: Add `config/credentials/` to `.gitignore` (already done).

### 2. Prepare Content

You need content items in JSONL format from Content Aggregator:

```bash
ls ../content-aggregator/data/content_items_*.jsonl
```

### 3. Run Calibration

```bash
cd C:\local_dev\llm-distillery

python -m ground_truth.calibrate_models \
    --prompt prompts/uplifting.md \
    --source ../content-aggregator/data/content_items_20251022_145619.jsonl \
    --sample-size 100 \
    --output reports/uplifting_calibration.md \
    --seed 42
```

**Parameters**:
- `--prompt`: Semantic filter prompt (e.g., `prompts/uplifting.md`)
- `--source`: JSONL file with content items
- `--sample-size`: Number of articles to compare (default: 100)
- `--output`: Where to save calibration report
- `--seed`: Random seed for reproducibility (optional)

**Expected Duration**: 5-10 minutes for 100 articles

**Expected Cost**:
- Claude: 100 × $0.003 = $0.30
- Gemini: 100 × $0.00018 = $0.02
- **Total**: ~$0.32

### 4. Review Report

The calibration generates:

1. **Report**: `reports/uplifting_calibration.md`
2. **Cached Labels**:
   - `calibrations/uplifting/claude_labels.jsonl`
   - `calibrations/uplifting/gemini_labels.jsonl`

## Interpreting Results

### Executive Summary

```markdown
## Executive Summary

- **Tier Distribution Difference**: 86.0%
- **Average Score Difference**: 1.64
- **Claude Average Score**: 5.85
- **Gemini Average Score**: 4.21

**Recommendation**: Models differ significantly. **Use Claude** or refine prompt for better Gemini consistency.
```

**What this means**:
- **Tier Distribution Difference**: Percentage of articles where models assigned different tiers
  - < 20%: Models agree well → **Use Gemini** for cost savings
  - 20-50%: Moderate disagreement → Review samples, consider prompt refinement
  - > 50%: Significant disagreement → **Use Claude** for ground truth

- **Average Score Difference**: Mean difference in overall scores
  - < 0.5: Very similar → **Use Gemini**
  - 0.5-1.5: Somewhat similar → Review disagreements
  - > 1.5: Very different → **Use Claude**

### Tier Distribution

```markdown
| Tier | Claude | Gemini | Difference |
|------|--------|--------|------------|
| Connection | 24.0% | 53.0% | +29.0% |
| Impact | 51.0% | 8.0% | -43.0% |
| Not Uplifting | 25.0% | 39.0% | +14.0% |
```

**Analysis**:
- Claude rates 51% as "Impact" (high-impact)
- Gemini rates only 8% as "Impact"
- **Gemini is more conservative** → rates fewer articles as high-impact

**Decision**:
- If you want **strict filtering** (only truly impactful content) → Use Gemini
- If you want **comprehensive coverage** (include borderline cases) → Use Claude
- If distributions differ >40% → **Use Claude for ground truth**

### Sample Article Comparisons

The report shows articles with largest disagreements:

```markdown
### Sample 1 - Disagreement: 5.53

**Title**: Your Remote Team Could Be Putting Your Company Data at Risk...

| Model | Score | Tier | Reasoning |
|-------|-------|------|-----------|
| Claude | 5.55 | connection | Organizations are taking concrete steps... |
| Gemini | 0.02 | not_uplifting | The article describes actions for corporate risk management... |
```

**How to interpret**:
1. **Large disagreements** (>4 points): Models fundamentally differ on what's uplifting
2. **Reasoning differences**: Claude sees "taking action", Gemini sees "corporate risk"
3. **Pattern analysis**: Look for systematic biases

**Red flags**:
- Models analyzing **different content** (title same, reasoning diverges) → BUG
- One model consistently **missing context**
- **Opposite conclusions** from same evidence

### Cost Analysis

```markdown
| Model | Cost per Article | Total Cost | Savings |
|-------|------------------|------------|---------|
| Claude 3.5 Sonnet | $0.009 | $45.00 | - |
| Gemini 1.5 Pro | $0.00018 | $0.90 | $44.10 (96%) |
```

**For 5,000 articles**:
- Claude: $45
- Gemini: $0.90 (96% savings!)

**Decision factors**:
- **Quality critical?** → Claude (e.g., investment decisions, clinical evidence)
- **Cost critical?** → Gemini (e.g., content curation, news aggregation)
- **Large dataset (50K+)?** → Consider Gemini to save $2,000+

## Advanced Usage

### Custom Sample Size

```bash
# Small test (10 articles)
python -m ground_truth.calibrate_models \
    --sample-size 10 \
    --output reports/uplifting_calibration_test.md

# Large calibration (500 articles) - more confidence
python -m ground_truth.calibrate_models \
    --sample-size 500 \
    --output reports/uplifting_calibration_large.md
```

**Recommendations**:
- **10 articles**: Quick test (~1 min, $0.03)
- **100 articles**: Standard calibration (~10 min, $0.32)
- **500 articles**: High confidence (~50 min, $1.60)

### Re-analyze Cached Results

The calibration caches labeled results to `calibrations/<filter>/`:

```bash
# Labels are cached - you can re-analyze without re-labeling
ls calibrations/uplifting/
# Output:
# claude_labels.jsonl
# gemini_labels.jsonl
```

This enables:
- Different comparison metrics
- Subset analysis
- Manual review of labels

### Compare Across Filters

```bash
# Calibrate uplifting filter
python -m ground_truth.calibrate_models \
    --prompt prompts/uplifting.md \
    --output reports/uplifting_calibration.md

# Calibrate sustainability filter
python -m ground_truth.calibrate_models \
    --prompt prompts/sustainability.md \
    --output reports/sustainability_calibration.md
```

**Patterns**:
- Some filters have **consistent agreement** → Use Gemini
- Others have **high disagreement** → Use Claude
- Domain-specific: Scientific content may need Claude, news may work with Gemini

## Troubleshooting

### Gemini Hangs or Times Out

**Symptom**: Gemini processing stalls at specific article

**Solution**: Timeout protection (60s) automatically handles this:

```python
# Already implemented in batch_labeler.py
def analyze_article(self, article, timeout_seconds=60):
    # Wraps LLM call in thread with timeout
    thread.join(timeout=timeout_seconds)
    if thread.is_alive():
        return None  # Skip this article
```

**What happens**:
1. Article times out after 60s
2. Returns `None` (skipped)
3. Continues with next article
4. Logged as "Timeout after 60s for article {id}"

### Rate Limiting

**Gemini Free Tier** (2 RPM):
```
Error: 429 Resource Exhausted
```

**Solution**:
1. Add billing to Google Cloud account → 150 RPM
2. Add `gemini_billing_api_key` to `secrets.ini`
3. SecretsManager auto-prioritizes billing key

**Claude Rate Limits**:
- Tier 1: 50 RPM (default for most users)
- Tier 2: 1000 RPM (after spending $100)

### Article Matching Bug

**Symptom**: Calibration report shows models analyzing different articles

**Example**:
```
Title: "White House announces that Trump..."
Claude: "Ukraine ceasefire negotiations..."
Gemini: "Braille AI assistance for blind users..."
```

**Cause**: Position-based matching (old bug, now fixed!)

**Fix**: We now use ID-based matching:

```python
# OLD (BUGGY):
for claude_art, gemini_art in zip(claude_articles, gemini_articles):

# NEW (FIXED):
claude_by_id = {art['id']: art for art in claude_articles}
gemini_by_id = {art['id']: art for art in gemini_articles}
common_ids = set(claude_by_id.keys()) & set(gemini_by_id.keys())

for article_id in common_ids:
    claude_art = claude_by_id[article_id]
    gemini_art = gemini_by_id[article_id]
```

**Verification**: Check report line:
```
**Matched articles**: 100 (analyzed successfully by both models)
```

## Best Practices

### 1. Run Calibration First

**ALWAYS** calibrate before generating large ground truth datasets:

```bash
# Step 1: Calibrate (100 articles, $0.32)
python -m ground_truth.calibrate_models \
    --prompt prompts/sustainability.md \
    --sample-size 100

# Step 2: Review report → Decide on Claude or Gemini

# Step 3: Generate ground truth (50K articles with chosen model)
python -m ground_truth.generate \
    --prompt prompts/sustainability.md \
    --llm claude  # or gemini
    --num-samples 50000
```

### 2. Use Reproducible Seeds

```bash
# Same seed → same random sample
python -m ground_truth.calibrate_models --seed 42

# Different seed → different sample (test stability)
python -m ground_truth.calibrate_models --seed 123
```

### 3. Document Your Decision

Create a decision log:

```markdown
# Calibration Decision Log

## Uplifting Filter
- **Date**: 2025-10-26
- **Sample**: 100 articles, seed=42
- **Tier Difference**: 86%
- **Score Difference**: 1.64
- **Decision**: **Use Claude** for ground truth
- **Reason**: Significant disagreement, quality critical for user experience
- **Report**: `reports/uplifting_calibration.md`
```

### 4. Re-calibrate Periodically

LLM models change over time:

```bash
# Q1 2025
python -m ground_truth.calibrate_models --output reports/uplifting_calibration_2025Q1.md

# Q2 2025 (detect drift)
python -m ground_truth.calibrate_models --output reports/uplifting_calibration_2025Q2.md
```

## Next Steps

After calibration:

1. **Review report** → Make quality vs. cost decision
2. **Generate ground truth** → `python -m ground_truth.generate` (when implemented)
3. **Train local model** → `python -m training.train`
4. **Deploy** → Local inference at $0 cost

See [docs/guides/getting-started.md](getting-started.md) for full workflow.

## Examples

### Example 1: High Agreement → Use Gemini

```markdown
## Executive Summary
- Tier Distribution Difference: 12%
- Average Score Difference: 0.4
- **Recommendation**: Models agree well. **Use Gemini** for 96% cost savings.
```

**Decision**: Generate 50K ground truth with Gemini, save $44/5K articles.

### Example 2: High Disagreement → Use Claude

```markdown
## Executive Summary
- Tier Distribution Difference: 86%
- Average Score Difference: 1.64
- **Recommendation**: Models differ significantly. **Use Claude**.
```

**Decision**: Use Claude for ground truth, prioritize quality over cost.

### Example 3: Mixed Results → Refine Prompt

```markdown
## Executive Summary
- Tier Distribution Difference: 45%
- Average Score Difference: 1.2
- **Recommendation**: Moderate disagreement. Refine prompt for clarity.
```

**Next Steps**:
1. Review sample disagreements
2. Identify ambiguous criteria in prompt
3. Add clarifying examples
4. Re-run calibration
5. If still high disagreement → Use Claude

## References

- [Architecture Overview](../architecture/overview.md#model-calibration)
- [API Reference](../api/batch_labeler.md)
- [Prompts Documentation](../prompts/README.md)
