# ADR 008: Post-Hoc Score Calibration with Isotonic Regression

**Date**: 2026-02-19
**Status**: Accepted
**Context**: Student models trained with MSE loss compress the oracle's score range. Isotonic regression can learn a monotonic correction per dimension.

## Problem

MSE training loss causes student models (Gemma-3-1B) to regress toward the mean, compressing the oracle's 0-10 score range. This manifests as:
- High scores under-predicted (oracle 7+ mapped to student ~5-6)
- Low scores slightly over-predicted
- Tier boundaries (especially HIGH at 7.0) become harder to reach
- Score distributions are narrower than oracle's, reducing discriminative power

For uplifting v6: oracle weighted averages peak at 7.35, but student predictions rarely exceed 6.5.

## Decision

Apply **per-dimension isotonic regression** as a post-hoc calibration step at inference time:

1. **Fit**: On the validation set, fit `IsotonicRegression(out_of_bounds='clip')` per dimension from `student_predicted -> oracle_actual`
2. **Store**: Save breakpoint pairs (x, y) as JSON in `calibration.json` within the filter directory
3. **Apply**: At inference, use `numpy.interp(score, x, y)` per dimension — zero sklearn dependency at scoring time
4. **Position in pipeline**: raw model output -> **calibrate** -> clamp 0-10 -> weighted avg -> gatekeeper -> tier

### Key design choices

- **JSON format** (not pickle) — human-readable, inspectable, version-controllable
- **Per-dimension** calibration, not on the weighted average — each dimension has its own compression pattern
- **`numpy.interp`** at inference — stateless, thread-safe, no sklearn needed in production
- **`out_of_bounds='clip'`** — scores beyond the observed range clamp to the nearest breakpoint, never extrapolate
- **Backwards compatible** — if `calibration.json` doesn't exist, `self.calibration` is `None` and scores pass through unchanged
- **Fitted on val set** — accepted tradeoff: slight overfitting to val distribution vs unbiased test set (which is single-use)

### What calibration does NOT do

- Does not expand the output range to the full 0-10 scale. It maps to the oracle's actual distribution in the val set (typically 0-8 for uplifting)
- Does not affect Stage 1 probe scores in hybrid inference — only the fine-tuned model's output is calibrated
- Does not change tier thresholds — it changes the scores that are compared against those thresholds

## Consequences

- Every new filter version should have `calibration.json` fitted after training
- `fit_calibration.py` is the shared CLI tool — works for any filter with a scorer and val.jsonl
- The calibration is monotonic by construction (isotonic regression guarantee)
- Val MAE improves ~3% (fitted set); test MAE is roughly flat (expected for post-hoc correction)
- Tier distribution moves closer to oracle on both val and test sets

## Files

- `filters/common/score_calibration.py` — shared library (fit, apply, save, load)
- `scripts/calibration/fit_calibration.py` — CLI fitting tool
- `filters/<name>/<version>/calibration.json` — per-filter calibration data
- `filters/<name>/<version>/base_scorer.py` — loads and applies calibration

## Results (uplifting v6)

| Set | MAE Before | MAE After | Change |
|-----|-----------|-----------|--------|
| Val (1,049) | 0.673 | 0.653 | +3.1% |
| Test (1,050) | 0.680 | 0.685 | -0.8% |

Tier distribution (test): Oracle 377 MEDIUM -> Student 356 -> Calibrated 366 (closer to oracle).
