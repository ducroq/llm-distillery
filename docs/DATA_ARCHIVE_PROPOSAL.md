# Data Archive Structure Proposal

**Date:** 2026-01-30
**Total Data Value:** ~€105 in Oracle API costs + training compute
**Total Size:** ~5.6 GB

## Why Archive?

Our scored datasets represent significant investment:
- **95,696 Oracle-scored articles** (Gemini API cost: ~€100+)
- **53,755 training-ready articles** with verified splits
- **11 trained LoRA models** (compute + iteration time)
- **Calibration datasets** for quality assurance

## Proposed Archive Structure

```
llm-distillery-archive-2026-01-30/
│
├── MANIFEST.json                    # Complete inventory with checksums
├── README.md                        # This documentation
│
├── 1-production/                    # PRIORITY: Production-ready assets
│   │
│   ├── scored/                      # Oracle-scored datasets (most valuable)
│   │   ├── uplifting-v5/           # 10,000 articles, €~10 API cost
│   │   │   ├── all_scored.jsonl
│   │   │   ├── metrics.jsonl
│   │   │   └── README.md
│   │   ├── investment-risk-v5/     # 10,248 articles
│   │   ├── sustainability_technology-v1/  # 10,000 articles
│   │   ├── sustainability_technology-v2/  # 8,000 articles
│   │   ├── cultural-discovery-v1/  # 10,000 articles
│   │   └── cultural-discovery-v2/  # 2,919 articles (screening filter)
│   │
│   ├── training/                    # Ready-to-train splits
│   │   ├── uplifting_v5/
│   │   │   ├── train.jsonl         # 7,999 examples
│   │   │   ├── val.jsonl           # 1,000 examples
│   │   │   ├── test.jsonl          # 1,001 examples
│   │   │   └── metadata.json
│   │   ├── investment_risk_v5/
│   │   ├── sustainability_technology_v1/
│   │   ├── sustainability_technology_v2/
│   │   ├── cultural-discovery-v1/
│   │   └── cultural-discovery_v2/
│   │
│   ├── models/                      # Trained LoRA adapters
│   │   ├── uplifting-v5/
│   │   │   ├── adapter_model.safetensors
│   │   │   ├── adapter_config.json
│   │   │   ├── training_metadata.json
│   │   │   ├── training_history.json
│   │   │   └── tokenizer/
│   │   ├── investment-risk-v5/
│   │   ├── sustainability_technology-v1/
│   │   ├── sustainability_technology-v2/
│   │   ├── cultural-discovery-v1/
│   │   └── cultural-discovery-v2/
│   │
│   └── calibration/                 # Quality assurance data
│       ├── uplifting_v5/
│       ├── sustainability_technology_v1/
│       └── commerce_prefilter_v1/
│
├── 2-raw-sources/                   # Source data (can regenerate scored from this)
│   ├── master_dataset_20251009_20251124.jsonl  # 178K articles, 937 MB
│   ├── fluxus_20260113.jsonl                   # 277K articles, 433 MB
│   ├── fluxus_20251130_20251219.jsonl          # 103K articles, 172 MB
│   └── master_dataset_20250929_20251008.jsonl  # 37K articles, 97 MB
│
├── 3-historical/                    # Previous versions (for reference)
│   ├── scored/
│   │   ├── uplifting-v4/
│   │   ├── investment-risk-v3/
│   │   ├── investment-risk-v4/
│   │   └── sustainability_tech_innovation-v1/
│   ├── training/
│   │   ├── uplifting_v4/
│   │   └── investment_risk_v4/
│   └── models/
│       ├── uplifting-v4/
│       └── investment-risk-v4/
│
├── 4-research/                      # Experimental data
│   ├── embedding_vs_finetuning/
│   │   ├── embeddings/
│   │   ├── models/
│   │   └── results/
│   └── context_length_experiments/
│
└── 5-legacy/                        # Old archives (low priority)
    └── archive_contents/
```

## File Manifest Format

`MANIFEST.json` structure:

```json
{
  "archive_date": "2026-01-30",
  "archive_version": "1.0",
  "total_size_bytes": 5600000000,
  "total_files": 1247,
  "sections": {
    "1-production": {
      "description": "Production-ready assets - highest priority for backup",
      "size_bytes": 1800000000,
      "contents": {
        "scored": {
          "uplifting-v5": {
            "articles": 10000,
            "oracle_cost_eur": 10.50,
            "files": [
              {"path": "all_scored.jsonl", "size": 74000000, "sha256": "abc123..."},
              {"path": "metrics.jsonl", "size": 1200000, "sha256": "def456..."}
            ]
          }
        }
      }
    }
  },
  "filters_summary": {
    "uplifting": {"latest": "v5", "versions": ["v4", "v5"]},
    "investment-risk": {"latest": "v5", "versions": ["v2", "v3", "v4", "v5"]},
    "sustainability_technology": {"latest": "v2", "versions": ["v1", "v2"]},
    "cultural-discovery": {"latest": "v2", "versions": ["v1", "v2"]}
  }
}
```

## Data Value Summary

| Category | Articles | Est. API Cost | Priority |
|----------|----------|---------------|----------|
| Production Scored | 51,167 | €55 | CRITICAL |
| Production Training | 42,882 | (derived) | CRITICAL |
| Production Models | 6 | €50 compute | CRITICAL |
| Historical Scored | 26,653 | €30 | HIGH |
| Raw Sources | 595,351 | €0 | MEDIUM |
| Research | varies | €10 | LOW |

## Backup Recommendations

### Tier 1: Critical (Daily backup)
- `1-production/scored/` - Cannot regenerate without API cost
- `1-production/models/` - Training time + compute cost

### Tier 2: Important (Weekly backup)
- `1-production/training/` - Can regenerate from scored
- `1-production/calibration/` - Quality assurance data

### Tier 3: Archival (Monthly backup)
- `2-raw-sources/` - Large files, source of truth
- `3-historical/` - Previous versions for reference

### Tier 4: Optional
- `4-research/` - Experimental, reproducible
- `5-legacy/` - Old data, low priority

## Checksums

Generate with:
```bash
find . -type f -exec sha256sum {} \; > SHA256SUMS.txt
```

Verify with:
```bash
sha256sum -c SHA256SUMS.txt
```

## Restoration Priority

If restoring from backup, order of importance:

1. **Scored datasets** - Irreplaceable without API cost
2. **Trained models** - Hours of compute time
3. **Training splits** - Can regenerate from scored
4. **Raw sources** - Can re-collect from Fluxus
5. **Historical** - Reference only

## Storage Recommendations

| Storage Type | Use For | Estimated Cost |
|--------------|---------|----------------|
| Local SSD | Active development | Free |
| Cloud (S3/GCS) | Tier 1-2 backup | ~€2/month |
| Cold storage | Full archive | ~€0.50/month |
| External HDD | Physical backup | One-time €50-100 |

## Next Steps

1. [ ] Create archive directory structure
2. [ ] Generate MANIFEST.json with checksums
3. [ ] Copy production data to archive
4. [ ] Verify checksums
5. [ ] Upload to cloud backup
6. [ ] Document restoration procedure
