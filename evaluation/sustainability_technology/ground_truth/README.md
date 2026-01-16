# Ground Truth Data for Sustainability Technology Filter

## Source
Manually reviewed corrections from NexusMind production use (Jan 2025).

## Files

| File | Count | Description |
|------|-------|-------------|
| `batch_corrections_142_500.json` | 97 | False positives from articles 142-500 |
| `batch_corrections_501_1000.json` | 145 | False positives from articles 501-1000 |
| `additional_corrections.json` | 29 | Additional corrections |
| **Total** | **271** | All false positives (off-topic) |

## Data Format

```json
{
  "num": 142,
  "article_id": "ai_arxiv_cs_lg_12ac0b578be0",
  "title": "Communication-Efficient Federated Learning...",
  "source": "ai_arxiv_cs_lg",
  "predicted_tier": "medium",
  "correct_tier": "low",
  "reason": "off_topic",
  "category": "ai_ml_infrastructure"
}
```

## Usage

These 271 known false positives can be used to:
1. Evaluate prefilter improvements (v1 vs v2)
2. Test if new versions correctly reject these articles
3. Calculate false positive reduction rate

## Categories of False Positives

Common off-topic categories found:
- `ai_ml_infrastructure` - Generic AI/ML papers
- `developer_tutorial` - Programming tutorials
- Hardware reviews, proxy/VPN articles, etc.
