# Uplifting Pre-Filter: Final Configuration

**Date:** 2025-10-27
**Dataset:** 51,869 articles
**Configuration:** Source-based thresholds + Smart compression + Multilingual

## Executive Summary

The final uplifting pre-filter uses:
1. **Source-aware word count thresholds** - Different minimums for different content types
2. **Smart content compression** - Preserves key information for long articles
3. **Multilingual support** - English, Dutch, Spanish keywords
4. **Emotion + keyword filtering** - Balanced detection approach

**Expected Results:**
- **Pass rate:** 20.6% (~10,664 articles)
- **Cost:** ~$1.60 (Gemini @ $0.00015/request)
- **Time:** ~44 hours (~1.9 days)
- **Savings:** $6.18 and 172 hours vs no filtering

## Key Design Decisions

### 1. Source-Based Word Count Thresholds

**Why:** Different content types need different minimum lengths

| Source Type | Minimum Words | Rationale |
|-------------|---------------|-----------|
| **News aggregators** (reuters, bbc, newsapi) | 20 words | RSS excerpts designed to convey story essence |
| **Long-form** (new_yorker, atlantic, fast_company) | 200 words | Need substantial content for quality |
| **Positive news** (upworthy, good_news) | 100 words | Medium-length articles typical |
| **Academic** (arxiv, nature, science) | 150 words | Need context for research papers |
| **GitHub** | Excluded | Repos, not articles |
| **Default** | 50 words | General threshold |

**Impact:**
- Allows meaningful news excerpts (20-50 words)
- Filters GitHub repos (379 excluded)
- Maintains quality for long-form sources

### 2. Smart Compression for Long Articles

**Why:** Prepare for future small models + reduce token costs

**Strategy:**
- Articles ≤800 words: Keep full content
- Articles >800 words: Keep 70% beginning + 30% ending
- Beginning: Provides context, setup, who/what/where
- Ending: Often contains conclusions, impact statements, calls to action

**Example:**
```
Original: 1,004 words (5,989 chars)
Compressed: 802 words (4,758 chars)
Reduction: 20.1%
```

**Benefits:**
- Preserves uplifting detection signals
- Stays within reasonable context windows
- Compatible with future smaller models
- Clear marker "[...content compressed...]" indicates sampling

### 3. Multilingual Support

**Supported Languages:**
- **English:** 83% of dataset
- **Dutch:** 9% of dataset (4,654 articles)
- **Spanish:** 5% of dataset (2,479 articles)

**Keywords per language:**

**English:**
- Uplifting: breakthrough, innovation, solution, success, achievement, hope, progress, inspiring, positive, transforms, improves, saves, helps, benefits, advance, discovered, cure, solved, revolutionary, pioneer
- Negative: war, death, killed, disaster, catastrophe, attack, violence, shooting, bomb, crisis, collapse, scandal, conflict, terror

**Dutch:**
- Uplifting: doorbraak, innovatie, oplossing, succes, prestatie, hoop, vooruitgang, inspirerend, positief, transformeert, verbetert, helpt, voordelen, ontdekt
- Negative: oorlog, dood, gedood, ramp, catastrofe, aanval, geweld, schietpartij, bom, crisis, instorting, schandaal

**Spanish:**
- Uplifting: avance, innovación, solución, éxito, logro, esperanza, progreso, inspirador, positivo, transforma, mejora, ayuda, beneficios, descubierto
- Negative: guerra, muerte, muerto, desastre, catástrofe, ataque, violencia, tiroteo, bomba, crisis, colapso, escándalo

### 4. Emotion-Based Filtering

**Uses available `raw_emotions` data:**
- joy (target: ≥0.15)
- sadness, fear, anger (combined negative target: <0.05)

**Logic:**
- High joy (≥0.15) → Pass
- Low negative emotions (<0.05 combined) → Pass
- Uplifting keywords present → Pass
- Negative keywords present → Fail (hard block)

## Filter Performance

### Test Results (10,000 sample)

| Metric | Value |
|--------|-------|
| **Pass rate** | 20.6% |
| **Fail rate** | 79.4% |
| **GitHub excluded** | 379 (3.8%) |
| **Too short** | 5,380 (53.8%) |
| **No positive signals** | 1,536 (15.4%) |
| **Negative keywords** | 643 (6.4%) |
| **Low quality** | 6 (0.1%) |

### Projected Full Dataset (51,869 articles)

| Metric | Estimate |
|--------|----------|
| **Articles passing** | 10,664 |
| **Articles filtered** | 41,205 (79.4%) |
| **Processing time** | 44.4 hours (~1.9 days) |
| **Cost** | ~$1.60 (Gemini) |
| **Savings vs no filter** | $6.18 + 172 hours |

### Sources Passing Filter

**Examples from test:**
- newsapi_general (news excerpts)
- dutch_news_correspondent
- automotive_transport_electrek (EV innovation)
- climate_solutions_inside_climate_news
- positive_news_good_news_network
- positive_news_reasons_to_be_cheerful
- industry_intelligence_fast_company
- healthcare_medrxiv

## Implementation Details

### Pre-Filter Function

Location: `ground_truth/batch_labeler.py:866-971`

**Key features:**
1. Source-based word count checks
2. Quality threshold (≥0.7)
3. Emotion analysis (joy vs negative)
4. Multilingual keyword matching
5. Negative keyword blocking

### Compression Function

Location: `ground_truth/batch_labeler.py:367-401`

**Parameters:**
- `max_words`: 800 (default, ≈3000 tokens)
- Split: 70% beginning, 30% ending
- Marker: `[...content compressed...]`

**Usage:** Automatic in `build_prompt()` method

## Usage

### Run Distillation

```bash
# On server in tmux
cd /path/to/llm-distillery
git pull
source venv/bin/activate

python -m ground_truth.batch_labeler \
    --prompt prompts/uplifting.md \
    --source datasets/raw/master_dataset.jsonl \
    --llm gemini \
    --batch-size 50 \
    --pre-filter uplifting \
    --output-dir datasets
```

### Monitor Progress

```bash
# Real-time log
tail -f datasets/uplifting/distillation.log

# Check metrics
cat datasets/uplifting/metrics.jsonl | jq '.success' | sort | uniq -c

# Session summary (after completion)
cat datasets/uplifting/session_summary.json
```

## Comparison: Evolution of Filter

| Version | Word Count | Language | Pass Rate | Articles | Cost |
|---------|------------|----------|-----------|----------|------|
| **v1.0** (Original) | 100 words | English only | 18.6% | 9,663 | $1.45 |
| **v2.0** (Multilingual) | 50 words | All languages | 23.6% | 12,225 | $1.83 |
| **v3.0** (Source-based) ✅ | Source-aware | All languages | 20.6% | 10,664 | $1.60 |

**v3.0 Advantages:**
- ✅ Smart about source types (accepts 20-word news excerpts, rejects 100-word GitHub repos)
- ✅ Higher quality (filters inappropriate sources)
- ✅ Better compression (800-word limit preserves key info)
- ✅ Multilingual (Dutch + Spanish support)
- ✅ Future-proof (compatible with small models)

## Quality Expectations

### LLM Distillation Success Rate

With error handling improvements:
- **Expected success:** >95%
- **JSON repair usage:** <10%
- **Retry attempts:** <20%
- **Complete failures:** <5%

### Content Quality by Source

**High confidence (full articles):**
- Long-form sources (200+ words)
- Positive news sites (100+ words)
- Academic papers (150+ words)

**Medium confidence (excerpts):**
- News aggregators (20-50 words)
- Can rate most dimensions
- May have uncertainty on connection/resilience/justice

**Filtered out:**
- GitHub repos
- Very short snippets (<20 words)
- Low quality sources
- Articles with negative keywords

## Future Optimizations

### Possible Improvements:

1. **Dynamic compression ratio**
   - Scientific papers: 80% beginning, 20% end (context-heavy)
   - News: 50% beginning, 50% end (inverted pyramid)
   - Stories: 60% beginning, 40% end (narrative arc)

2. **Extractive summarization**
   - Use sentence scoring (keyword density, position)
   - Keep most important sentences
   - More sophisticated than simple head/tail

3. **Source-specific keywords**
   - Science: "breakthrough", "discovery", "advance"
   - Community: "helps", "support", "connects"
   - Climate: "solution", "renewable", "sustainable"

4. **Confidence scoring**
   - Track which articles had sufficient context
   - Use for quality control
   - Adjust thresholds based on actual LLM performance

## Files Modified/Created

### Modified:
- `ground_truth/batch_labeler.py`
  - Updated `uplifting_pre_filter()` (lines 866-971)
  - Added `_smart_compress_content()` (lines 367-401)
  - Updated `build_prompt()` (lines 403-422)

### Created:
- `docs/analysis/dataset-profile-2025-10-27.md` - Full dataset analysis
- `docs/guides/ground-truth-pipeline-architecture.md` - Pipeline guide
- `ground_truth/data_profiler.py` - Profiling tool
- `test_prefilter.py` - Filter testing
- `test_compression.py` - Compression testing
- This document

## References

- Dataset profile: `docs/analysis/dataset-profile-2025-10-27.md`
- Pipeline guide: `docs/guides/ground-truth-pipeline-architecture.md`
- Error handling: `docs/guides/json-error-handling-improvements.md`
- Logging system: `docs/guides/distillation-logging-system.md`

---

**Ready for production!** The filter is optimized for quality, cost-effectiveness, and future compatibility with smaller models.
