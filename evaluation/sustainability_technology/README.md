# Sustainability Technology Filter - Evaluation

Cross-version evaluation and ground truth data for the sustainability_technology filter.

## Contents

| Item | Description |
|------|-------------|
| `V1_VS_V2_COMPARISON.md` | Comparison report: v2 reduces FPs by 32.1% |
| `chart_*.png` | Visualization charts |
| `compare_v1_v2.py` | Comparison script |
| `generate_charts.py` | Chart generation script |
| `ground_truth/` | 271 manually reviewed false positives |

## Key Results

- **v1 false positive rate:** 100% (on known FPs)
- **v2 false positive rate:** 67.9%
- **Improvement:** -32.1% (87 articles now correctly blocked)
- **Regressions:** 0

## Related Documentation

For filter-specific reports (training, calibration, validation), see:

**[`filters/sustainability_technology/v2/reports/`](../../filters/sustainability_technology/v2/reports/)**

## Running the Comparison

```bash
cd llm-distillery
python evaluation/sustainability_technology/compare_v1_v2.py
python evaluation/sustainability_technology/generate_charts.py
```

Requires access to filtered data in `I:/Mijn Drive/NexusMind/filtered/sustainability_technology/`.
