# Belonging Filter - Development Status

**Last Updated:** 2026-03-01
**Status:** Phase 3 Oracle Validation Complete — Ready for Batch Labeling

---

## Completed

- [x] Filter concept developed (Blue Zones inspiration, non-commercial dimensions)
- [x] Philosophical grounding documented (DEEP_ROOTS.md)
- [x] 6 dimensions defined with weights (config.yaml)
- [x] Oracle prompt created with examples (prompt-compressed.md)
- [x] Prefilter implemented with multilingual patterns (prefilter.py)
- [x] Prefilter tested: 10/10 unit tests passing
- [x] Prefilter run on corpus: 31% pass rate on 37,643 articles
- [x] 72 candidate articles extracted for oracle testing
- [x] RSS sources created in FluxusSource (rss_belonging.yaml, 15/24 working)
- [x] **Phase 1-2 reiteration** (2026-03-01):
  - [x] Prompt restructured with Step 1 scope check (matches production pattern)
  - [x] Inline critical filters added per dimension (before scale tables)
  - [x] Anti-hallucination rule: evidence must be EXACT QUOTE from article
  - [x] `community_fabric` added as gatekeeper dimension (threshold 3.0, cap ~3.42)
  - [x] Config updated: model → Gemma-3-1B, head_tail 256+256, target_samples → 10K
  - [x] `hybrid_inference` section added (placeholder, threshold TBD after training)
  - [x] `gatekeepers` section added to config
- [x] **Phase 3 oracle validation** (2026-03-01):
  - [x] 152 articles scored with Gemini Flash (72 candidates + 80 master dataset)
  - [x] Fixed missing article placeholder — root cause of hallucinated evidence
  - [x] Added `out_of_scope` content type for noise articles
  - [x] Added anti-hallucination warning after validation examples
  - [x] **Results:** 3 HIGH, 16 MEDIUM, 133 LOW — correct tier distribution
  - [x] **Hallucinations:** 0 (after fix)
  - [x] **Gatekeeper:** triggered on 88% of articles, correctly blocks noise
  - [x] **Evidence quality:** exact quotes from articles, "No evidence in article" for absent dimensions
  - [x] **Dimension independence confirmed:** slow_presence consistently lowest (max 6.0), purpose_beyond_self often highest

---

## Oracle Validation Results (152 articles)

| Metric | Value |
|--------|-------|
| Articles scored | 152 |
| HIGH (≥7.0) | 3 (2%) |
| MEDIUM (4.0-7.0) | 16 (11%) |
| LOW (<4.0) | 133 (88%) |
| Gatekeeper triggered | 133 (88%) |
| Hallucinated evidence | 0 |
| Content type "out_of_scope" | 127 |

**MEDIUM+ dimension stats:**

| Dimension | Min | Max | Mean |
|-----------|-----|-----|------|
| intergenerational_bonds | 2.0 | 8.5 | 5.7 |
| community_fabric | 3.0 | 8.0 | 6.4 |
| reciprocal_care | 2.0 | 8.0 | 5.4 |
| rootedness | 2.0 | 8.0 | 6.3 |
| purpose_beyond_self | 4.0 | 9.0 | 6.7 |
| slow_presence | 3.0 | 6.0 | 4.2 |

**Note:** `slow_presence` caps at 6.0 in news corpus — unhurried rituals need literary/longform content to score higher. Not a prompt issue.

---

## Next Steps

1. **Phase 4: Batch Labeling** (next)
   ```bash
   python -m ground_truth.batch_scorer \
       --filter filters/belonging/v1 \
       --source "datasets/raw/master_dataset_*.jsonl" \
       --llm gemini-flash \
       --target-scored 10000 \
       --random-sample --seed 42
   ```

2. **Phase 5: Training** (after labeling)
   - Prepare training splits (80/10/10)
   - Train Gemma-3-1B + LoRA on gpu-server
   - Fit isotonic calibration on val set

---

## Key Files

| File | Purpose |
|------|---------|
| `config.yaml` | Dimensions, weights, content type caps, gatekeeper |
| `prompt-compressed.md` | Oracle prompt for LLM scoring |
| `prefilter.py` | Rule-based blocker (saves API costs) |
| `DEEP_ROOTS.md` | Philosophical grounding (Weil, Tönnies, etc.) |
| `calibrations/candidates/belonging_candidates.jsonl` | 72 candidate articles |

---

## Dimensions (weights)

1. **intergenerational_bonds** (25%) — Multi-generation connection
2. **community_fabric** (25%) — Local social infrastructure **[GATEKEEPER]**
3. **rootedness** (15%) — Place attachment, staying put
4. **purpose_beyond_self** (15%) — Transcendent meaning
5. **slow_presence** (10%) — Unhurried togetherness
6. **reciprocal_care** (10%) — Mutual support networks

---

## Notes

- Prefilter pass rate (31%) is lower than typical filters because corpus is mostly tech/news
- RSS sources in FluxusSource will improve future data quality for this filter
- Gatekeeper on community_fabric prevents wellness/longevity articles from scoring HIGH without actual community evidence
- Phase 1-2 reiteration aligned prompt and config with uplifting v6 / cultural-discovery v4 production standards
- Master dataset prefilter blocks 52% of articles (91,992/178,462) — saves API costs during batch labeling
