# Sustainability Technology v2 - Reports

Documentation of v2 filter development, training, and validation.

## Reports in this folder

| Report | Description |
|--------|-------------|
| `ORACLE_CALIBRATION_REPORT.md` | Oracle (LLM) calibration results |
| `PREFILTER_V2_VALIDATION.md` | Prefilter validation on test set |
| `TRAINING_DATA_PREPARATION.md` | Training data generation process |
| `TRAINING_REPORT.md` | Model training results and metrics |

## Cross-version Evaluation

For v1 vs v2 comparison with production false positives, see:

**[`evaluation/sustainability_technology/`](../../../../evaluation/sustainability_technology/)**

Contains:
- `V1_VS_V2_COMPARISON.md` - Full comparison report with charts
- `ground_truth/` - 271 manually reviewed false positives
- Comparison scripts and generated charts

**Key result:** v2 reduces false positives by 32.1% with zero regressions.
