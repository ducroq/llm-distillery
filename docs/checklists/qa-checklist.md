# QA Checklist

Definition-of-done for quality assurance review. Use before approving a merge or declaring work complete.

## Git Reality Check

- [ ] Run `git diff --stat` and compare against claimed changes
- [ ] Flag files changed but not mentioned (forgotten documentation or accidental edits)
- [ ] Flag files mentioned but not changed (incomplete work)
- [ ] Verify each acceptance criterion has corresponding code changes

## Minimum Findings Requirement

- [ ] Review surfaced at least 3 observations (issues, risks, or improvement notes)
- [ ] If fewer than 3: explicitly documented what was verified and why no issues exist
- [ ] Each finding classified: CRITICAL (blocks merge) / HIGH (should fix) / MEDIUM (consider) / LOW (optional)
- [ ] No "looks good" without listing specific checks performed

## Model & Data Quality

- [ ] MAE is within expected range for this filter type
- [ ] No dimension has MAE > 1.3 (flag for investigation, cf. #23)
- [ ] Score distribution is unimodal or expected shape (flag bimodal, cf. thriving v1)
- [ ] Calibration and normalization files are consistent with training data

## Deployment Readiness

- [ ] Filter loads from HF Hub without errors (test `inference_hub.py`)
- [ ] NexusMind integration tested — filter appears in pipeline, scores articles correctly
- [ ] Normalization LUT covers the score range the model actually produces
- [ ] Rollback plan exists — previous filter version still on Hub and can be reverted

## Documentation

- [ ] RUNBOOK updated if operational procedures changed
- [ ] Gotcha log entry added if a non-obvious problem was encountered
- [ ] ADR created if a significant design choice was made
- [ ] CLAUDE.md updated if constraints, architecture, or production filters changed
