---
name: uplifting-v7-training
description: Uplifting v7 training status — evolved into thriving v1 (ADR-012). V7 not deployed, thriving v1 filter created.
type: project
---

# Uplifting v7 → Thriving v1

## Status: v7 NOT deployed. Evolved into thriving v1 (ADR-012 lens-aligned naming, social_cohesion removed).

## Training complete

- **Best epoch**: 6/6, val MAE **0.787** (clamp-to-1.0 targets)
- **Calibration**: fitted but hurts on test (raw 0.811 → calibrated 0.841)
- **HIGH tier detection broken**: oracle finds 11 HIGH on test, calibrated model finds only 2
- Model files at `filters/uplifting/v7/model/` (local)
- `calibration.json` saved but calibration is overfitting val set

## Training history (all on gpu-server)

| Run | MAE | Notes |
|-----|-----|-------|
| Unclamped, 3 epochs | 0.96 | Bimodal zero-inflated distribution |
| Clamp 0→1.0, 6 epochs | 0.78 | Best so far |
| Round to integers, 6 epochs | 0.91 | Coarser targets hurt |
| Clamp 0→1.0, 6 epochs (final) | **0.787** | Completed |

## Root cause of high MAE vs v6 (0.67)

- V7 prompt (ADR-010) produces bimodal scores: 30-43% zeros per dimension (v6 had 0-9%)
- V7 has only 15-17 discrete score values per dimension (v6 had 250+ continuous floats from multi-run averaging)
- V7 has less training data: 5,271 vs 8,396
- Clamping zeros to 1.0 helped (0.96 → 0.78) but not enough

## Evolution to thriving v1 (2026-03-18)

Rather than patching v7, created thriving v1:
- Renamed from uplifting → thriving (ADR-012)
- Removed social_cohesion_impact (overlaps Belonging's community_fabric)
- 5 dimensions: human_wellbeing (0.40), justice_rights (0.25), evidence (0.10), distribution (0.10), durability (0.15)
- Filter dir: `filters/thriving/v1/`
- Oracle averaging script: `scripts/oracle/average_oracle_runs.py`
- Next: 3-run oracle scoring with 5-dim prompt → average → train → deploy
- Can't reuse v7 oracle data (dimensions changed)

## Previous plan: multi-run oracle averaging (now applied via thriving v1)

**Why:** V6's 250+ continuous float values came from averaging multiple oracle runs. V7's 15-17 discrete integers are much harder for the student to learn. This is the biggest lever.

**How to apply (via thriving v1):**
1. Score all articles 3x with thriving v1 prompt (~$18)
2. Average 3 runs → continuous targets, fewer zeros
3. Train on smoothed data
4. Calibrate and deploy

## Data locations

- Oracle scored: `datasets/scored/uplifting/` + `datasets/scored/uplifting_enrichment/uplifting/`
- Combined: `datasets/scored/uplifting_v7_combined/all_scored.jsonl`
- Splits: `datasets/training/uplifting_v7/` (5,271 train / 659 val / 660 test)
- Training data: `train.jsonl` / `val.jsonl` / `test.jsonl` (clamp-to-1.0), `*_original.jsonl` (unclamped backups)
- GPU server log: `~/llm-distillery/training_uplifting_v7_final.log`

## Code changes made

1. `filters/uplifting/v7/base_scorer.py` — new, v7-specific constants (updated weights from ADR-010)
2. `filters/uplifting/v7/inference.py` — new, local inference pipeline
3. `filters/uplifting/v7/calibration.json` — fitted (but overfitting, may discard)
4. `ground_truth/batch_scorer.py` — added `filter_version` + `prompt_hash` to scoring metadata
5. `filters/common/filter_base_scorer.py` — added `scoring_metadata()` method + `_compute_prompt_hash()`
6. GitHub issues created (previous session): NexusMind #103, ovr.news #115
