# Sustainability Technology v2 - Reports

Documentation of v2 filter development, training, and validation.

## Reports in this folder

| Report | Description |
|--------|-------------|
| `ORACLE_CALIBRATION_REPORT.md` | Oracle (LLM) calibration results |
| `PREFILTER_V2_VALIDATION.md` | Prefilter validation on test set |
| `TRAINING_DATA_PREPARATION.md` | Training data generation process |
| `TRAINING_REPORT.md` | Model training results and metrics |

## Prefilter Performance (v2.1)

| Metric | Value |
|--------|-------|
| FP Block Rate | 88.2% |
| TP Pass Rate | 89.0% |

The prefilter effectively blocks most off-topic content while passing legitimate sustainability articles.

## Evaluation & Ground Truth

For comprehensive evaluation with production data, see:

**[`evaluation/sustainability_technology/`](../../../../evaluation/sustainability_technology/)**

Contains:
- `PREFILTER_EVALUATION_REPORT.md` - Full prefilter evaluation (v2.1 vs v2.2 experiment)
- `V1_VS_V2_COMPARISON.md` - Historical v1 vs v2 comparison
- `compare_prefilters.py` - A/B test script for prefilter versions
- `ground_truth/` - 271 manually reviewed false positives
- `true_positives/` - 300 frozen true positives for testing
