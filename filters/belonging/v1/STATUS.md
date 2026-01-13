# Belonging Filter - Development Status

**Last Updated:** 2026-01-13
**Status:** Ready for Oracle Calibration

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
- [x] All changes committed to git

---

## Next Steps

1. **Manual Oracle Testing** (recommended first)
   - Pick 10-20 articles from `calibrations/candidates/belonging_candidates.jsonl`
   - Test with prompt-compressed.md against an LLM (Claude/Gemini)
   - Validate dimension scoring makes sense
   - Adjust prompt if needed

2. **Automated Oracle Calibration**
   ```bash
   python -m ground_truth.calibrate_oracle \
       --filter filters/belonging/v1 \
       --source "filters/belonging/v1/calibrations/candidates/belonging_candidates.jsonl" \
       --models gemini-flash,gemini-pro \
       --sample-size 50
   ```

3. **Batch Labeling** (after calibration)
   - Generate ground truth dataset
   - Use for fine-tuning smaller models

---

## Key Files

| File | Purpose |
|------|---------|
| `config.yaml` | Dimensions, weights, content type caps |
| `prompt-compressed.md` | Oracle prompt for LLM scoring |
| `prefilter.py` | Rule-based blocker (saves API costs) |
| `DEEP_ROOTS.md` | Philosophical grounding (Weil, Tönnies, etc.) |
| `calibrations/candidates/belonging_candidates.jsonl` | 72 articles for testing |

---

## Dimensions (weights)

1. **intergenerational_bonds** (25%) - Multi-generation connection
2. **community_fabric** (25%) - Local social infrastructure
3. **rootedness** (15%) - Place attachment, staying put
4. **purpose_beyond_self** (15%) - Transcendent meaning
5. **slow_presence** (10%) - Unhurried togetherness
6. **reciprocal_care** (10%) - Mutual support networks

---

## Notes

- Prefilter pass rate (31%) is lower than typical filters because corpus is mostly tech/news
- RSS sources in FluxusSource will improve future data quality for this filter
- Consider pairing with future `equanimity` filter (see docs/FUTURE_FILTER_IDEAS.md)
