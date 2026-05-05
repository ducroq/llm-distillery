# Sustainability Technology v4 — DRAFT

**Status: DRAFT.** Do not score the full corpus against this config until
the 300-article calibration batch (`config.yaml :: calibration_batch`)
confirms the dimension distributions are not bimodal/dead-zone.

## What this is

v4 is the broadened Solutions lens design. It keeps ST v3's LCSA spine
but covers governance and community solutions in addition to clean tech.
Foresight v1's top governance articles are the gap this version is meant
to capture — see [llm-distillery#43](https://github.com/ducroq/llm-distillery/issues/43).

## What changed from v3

| Dimension change | v3 | v4 |
|---|---|---|
| **Gatekeeper shape** | `technology_readiness_level` (TRL) — tech-only | `solution_concreteness` — universal across tech/governance/community |
| **NEW** governance dim | n/a | `governance_intervention_strength` (0.15) — scores 0 for pure tech |
| **NEW** community dim | n/a | `community_practice_strength` (0.10) — scores 0 for pure tech/policy |
| **Renamed/broadened** | `life_cycle_environmental_impact` (0.30) | `systemic_impact` (0.20) — covers tech LC + governance reach + community replicability |
| **Slimmed** | `economic_competitiveness` (0.20) | `economic_viability` (0.10) — kept for investment-DD use case |
| **Added pre-step** | (implicit) | Step-1 scope check (`is this an article about a solution?`) before per-dim scoring |
| **Added pre-step** | (implicit) | Step-2 type tag (tech / governance / community / hybrid) |

Total weight: 1.00. Six scored dimensions plus the type tag.

## Decisions inherited from #43 sign-off

- **Fork 1 = C** — broaden v3 in place rather than redesign from scratch
- **Fork 2** — combine ST v3 (10.6K) + foresight v1 (3.5K) corpora and
  re-score with v4 prompt after the calibration batch
- **Fork 3** — foresight v1 stays parked through v4 calibration; retire
  on v4 production deploy

## Next step

Run the 300-article calibration batch with the v4 prompt (prompt itself
not yet written; will live at `prompt-compressed.md` and `prompt-full.md`
once drafted). Decision criteria are listed in `config.yaml ::
calibration_batch.decision_criteria`. Cost ~$0.30.

If criteria pass: re-score the combined ST v3 + foresight v1 corpora
with v4 prompt (~$15) and proceed to training.

If criteria fail: iterate on dimension scales / prompt / weights and
re-run the calibration batch before any further spend.

## Files in this directory at draft stage

- `config.yaml` — dimension architecture, weights, gatekeeper, calibration batch spec
- `README.md` — this file
- (no model, no prompt, no calibration.json yet — those land after the
  calibration batch decides direction)
