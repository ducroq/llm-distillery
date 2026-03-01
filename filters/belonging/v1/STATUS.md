# Belonging Filter - Development Status

**Last Updated:** 2026-03-01
**Status:** Phase 1-2 Reiteration Complete — Ready for Oracle Validation

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

---

## Next Steps

1. **Phase 3: Oracle Validation** (next)
   - Pick 20 articles from `calibrations/candidates/belonging_candidates.jsonl`
   - Test with rewritten prompt against Gemini Flash
   - Validate: scope check catches out-of-scope, gatekeeper caps wellness leakage
   - Validate: exact-quote evidence works (no hallucinated paraphrases)
   - Check dimension independence and scoring distribution

2. **Automated Oracle Calibration**
   ```bash
   python -m ground_truth.batch_scorer \
       --filter filters/belonging/v1 \
       --source "filters/belonging/v1/calibrations/candidates/belonging_candidates.jsonl" \
       --target-count 100
   ```

3. **Batch Labeling** (after calibration)
   - Generate ground truth dataset (target: 10,000 articles)
   - Use for fine-tuning Gemma-3-1B + LoRA

---

## Key Files

| File | Purpose |
|------|---------|
| `config.yaml` | Dimensions, weights, content type caps, gatekeeper |
| `prompt-compressed.md` | Oracle prompt for LLM scoring |
| `prefilter.py` | Rule-based blocker (saves API costs) |
| `DEEP_ROOTS.md` | Philosophical grounding (Weil, Tönnies, etc.) |
| `calibrations/candidates/belonging_candidates.jsonl` | 72 articles for testing |

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
