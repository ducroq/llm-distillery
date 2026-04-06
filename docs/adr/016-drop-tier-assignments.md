# ADR-016: Drop Tier Assignments from Filter Output

**Date:** 2026-04-06
**Status:** Accepted

## Decision

Filters produce only two outputs: **pass/block** (prefilter) and a **continuous score** (weighted average, 0-10). Tier assignments ("high", "medium", "low") are removed from the scoring pipeline. Any downstream logic that needs thresholds (e.g., "only translate articles above score X") uses the continuous score directly.

## Context

### What tiers do today

Every filter defines `TIER_THRESHOLDS` in `base_scorer.py` and `tiers:` in `config.yaml`. After scoring, `_assign_tier()` maps the weighted average to a tier name. The tier is stored in the output JSON alongside the score.

Example from cultural-discovery v4:
```yaml
tiers:
  high:
    threshold: 7.0
  medium:
    threshold: 4.0
  low:
    threshold: 0.0
```

### Why tiers add no value

1. **They carry no information beyond the score.** `tier: "medium"` means `4.0 <= weighted_average < 7.0`. The score already says that.

2. **They create false precision.** An article at 3.99 is "low" and at 4.01 is "medium." The model's MAE is 0.74 — that boundary is noise.

3. **They complicate storage.** NexusMind stores filtered output in tier subdirectories on some filters, flat files on others. This caused a data counting bug during active learning planning (312 nature_recovery articles missed). See NexusMind#144.

4. **They will change.** ADR-014 introduced percentile normalization. ovr.news uses normalized scores, not tiers. Threshold changes require updating config.yaml in every filter — but the scores don't change.

5. **They leak into too many places.** `should_translate`, active learning sampling, probe training — all reference tiers when they should reference score thresholds. Each is a maintenance burden.

### What consumers actually need

- **ovr.news**: normalized continuous score for ranking. No tiers.
- **NexusMind pipeline**: pass/block + score. Translation decision can be a score threshold in pipeline config.
- **Active learning**: score ranges for candidate selection. No tiers.
- **Probe training**: continuous scores. No tiers.

## Implementation (gradual, consumer-first)

### Phase 1: NexusMind / ovr.news (first)
- Stop using tier for storage decisions (NexusMind#144)
- Replace any `tier == "high"` checks with score threshold checks
- Flat file storage only

### Phase 2: llm-distillery scoring pipeline
- Remove `_assign_tier()` from `FilterBaseScorer`
- Remove `TIER_THRESHOLDS` from subclass constants
- Remove `tier` and `tier_description` from result dict
- Keep `tiers:` in config.yaml as documentation-only (editorial reference for what score ranges mean), not consumed by code

### Phase 3: Cleanup
- Remove `translate:` flags from tier definitions in config.yaml
- Update batch scorer output format
- Update merge/prepare scripts that reference tiers

## Consequences

- **Simpler filter output**: `{passed_prefilter, scores, weighted_average, gatekeeper_applied}` — no tier fields
- **Flexible thresholds**: NexusMind can change what "worth translating" means without touching filters
- **No more tier-based storage bugs**
- **Breaking change for NexusMind**: needs to stop reading `tier` field. Gradual rollout (Phase 1 first) mitigates this.
