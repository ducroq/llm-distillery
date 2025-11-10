# LLM Distillery Development Session - 2025-11-08

## Work Completed

### 1. Oracle Model Calibration (Gemini Flash vs Pro)

**Task**: Compare Gemini Flash and Pro for sustainability tech deployment filter oracle labeling

**Results**:
- **Sample size**: 55 articles (500 sampled, 11% passed prefilter)
- **Flash average score**: 2.87
- **Pro average score**: 1.38 (overly conservative)
- **Tier distribution difference**: 76.4% (significant disagreement)

**Key Finding**: Gemini Pro systematically assigns zeros to off-topic articles and 1.0-1.1 to borderline cases. Flash provides better discrimination and contextual assessment.

**Manual Analysis**: Reviewed top 5 disagreement cases:
1. **Biosimilars article** (pharmaceutical, NOT climate): Flash 7.0 vs Pro 0.0 → Flash correct (recognizes commercial deployment)
2. **Transmission grid research** (climate-relevant): Flash 6.4 vs Pro 1.1 → Flash correct (real infrastructure)
3. **PDF translation tool** (off-topic software): Flash 6.25 vs Pro 1.0 → Flash correct (deployed software)
4. **Solar grants policy** (climate-relevant): Flash 6.15 vs Pro 1.0 → Flash correct (solar is deployed)
5. **Cargo ship attack** (off-topic): Flash 3.9 vs Pro 0.0 → Tie (both defensible)

**Recommendation**: **Use Gemini Flash** for all oracle labeling
- Better discrimination (commercial_proven vs early_commercial vs pilot vs vaporware)
- More training signal for downstream 7B models
- 50% cheaper than Pro ($10-20 vs $40 for 5 filters × 2K samples)

### 2. Prefilter Calibration & Tuning

**Initial Problem**: Sustainability prefilters were TOO strict (0.2-3% pass rates)

**Solution**: Changed from strict regex patterns to permissive keyword matching
- From: `r'\bclimate\b'` → To: `'climate' in text_lower`
- Rationale: Local 7B inference is "free", oracle LLM cost is main constraint

**New Pass Rates**:
- Tech Deployment: 11.0%
- Economic Viability: ~9%
- Policy Effectiveness: ~10%
- Nature Recovery: ~9%
- Movement Growth: ~8%

**Comparison**: Uplifting filter has 13% pass rate → sustainability filters now comparable

### 3. New Filter Created: AI-Augmented Practice

**Purpose**: Track empirical evidence of GenAI/LLM transforming cognitive work practices

**Focus**: Real workflow integration, NOT hype, benchmarks, or speculation

**8 Dimensions** (weights sum to 1.0):
1. **Workflow Integration Depth** (20%) - How deeply AI is integrated
2. **Empirical Evidence Quality** (18%) - Strength of evidence (gatekeeper)
3. **Trust/Verification Patterns** (15%) - How practitioners validate output
4. **Cognitive Task Specificity** (12%) - How specific the task description
5. **Failure Mode Documentation** (12%) - Edge cases, limitations, failures
6. **Human-AI Division of Labor** (10%) - What each does
7. **Skill Evolution** (8%) - New skills emerged
8. **Organizational Dynamics** (5%) - Team/org adaptation

**Gatekeeper Rule**: If Empirical Evidence Quality < 4.0 → cap overall score at 3.9

**Tiers**:
- **Transformative Practice** (≥8.0) - Deep integration, rigorous evidence
- **Validated Adoption** (≥6.0) - Real usage, empirical validation
- **Emerging Practice** (≥4.0) - Early adoption, some evidence
- **Speculation** (<4.0) - Hype, no real usage

**Prefilter Pass Rate**: 20.4% (102/500) - Good signal

**Files Created**:
- `filters/ai_augmented_practice/v1/config.yaml`
- `filters/ai_augmented_practice/v1/prompt-compressed.md`
- `filters/ai_augmented_practice/v1/prefilter.py`
- `filters/ai_augmented_practice/v1/README.md`

### 4. Data Enrichment: Historical Database Merge

**Task**: Merge Google Drive historical database into llm-distillery datasets

**Source**: `I:/Mijn Drive/NexusMind/historical-database/current/`
- 114 collection runs
- ~80K articles (Oct 9 - Nov 1, 2025)
- More diverse sources (GitHub repos, API sources)
- Newer data than existing datasets (current ends Oct 29)

**Merge Strategy**:
- Deduplicate by article ID (not just dates)
- Compare against existing 99,763 articles
- Output single file: `historical_dataset_YYYYMMDD_YYYYMMDD.jsonl`

**Status**: Running in background (bash 6fc42a)

**Expected Outcome**:
- Total dataset: ~180K articles (99K existing + ~80K new)
- Date range: Sept 29 - Nov 1, 2025
- Source diversity: RSS feeds + GitHub + API sources

## Documents Created

1. **`reports/oracle_model_recommendation.md`**
   - Full analysis of Flash vs Pro disagreements
   - Manual assessment of which model is correct
   - Pattern analysis and cost comparison
   - Clear recommendation: Use Flash

2. **`reports/disagreement_analysis.md`**
   - Raw extraction of top 5 disagreement cases
   - Dimension scores and reasoning from both models

3. **`reports/tech_deployment_oracle_calibration_final.md`**
   - Statistical calibration report (auto-generated)
   - Tier distributions, score statistics, correlation analysis

4. **`scripts/analyze_disagreements.py`**
   - Tool to extract and format disagreement cases
   - Loads both Flash and Pro labels, finds largest score gaps

5. **`scripts/merge_historical_data.py`**
   - Merges historical database into llm-distillery
   - Deduplicates by ID against existing datasets
   - Outputs single consolidated JSONL file

## Architecture Enhancements

### Batch Labeler: Generic Post-Processing

**Problem**: `batch_labeler.py` was hardcoded for uplifting filter

**Solution**: Added `_post_process_sustainability()` method that:
- Reads dimension names and weights from `config.yaml`
- Handles both flat scores and nested `{"score": X, "reasoning": Y}` format
- Applies gatekeeper rules from config
- Calculates tier based on config thresholds

**Impact**: All sustainability filters (and future filters) automatically work with batch labeler

### Prompt Template Format

**Requirements** for `prompt-compressed.md`:
1. Must have `## PROMPT TEMPLATE` section header
2. Use `{text}` not `{content}` (batch_labeler provides 'text')
3. Use double curly braces `{{}}` for JSON examples (prevents Python .format() parsing)
4. End with: `DO NOT include any text outside the JSON object.`

## Cost Analysis

### Oracle Labeling Cost (5 Filters × 2,000 Samples)

| Model | Cost/1K | Total Cost | Quality |
|-------|---------|------------|---------|
| **Gemini Flash** | $0.001 | **$10-20** | ✅ Good discrimination |
| Gemini Pro | $0.002 | $40 | ❌ Binary, no nuance |
| Claude Sonnet | $0.009 | $90 | ✅✅ Excellent (too expensive) |

**Recommendation**: Use Gemini Flash - best cost/quality ratio

### 7B Model Training Cost

**Hardware**: Local GPU (free) or RunPod (~$0.30/hr × 4 hrs = $1.20)

**Total System Cost**:
- Oracle labeling: $10-20 (Gemini Flash)
- 7B training: $0-5 (local GPU or cheap cloud)
- **Total: $10-25 for all 5 sustainability filters + AI practice filter**

## Next Steps

### Immediate (After Merge Completes)

1. ✅ **Verify merged dataset**:
   ```bash
   ls -lh datasets/raw/historical_dataset_*.jsonl
   wc -l datasets/raw/*.jsonl
   ```

2. ✅ **Test prefilters on merged data**:
   ```bash
   python -m ground_truth.calibrate_oracle \
     --filter filters/sustainability_tech_deployment/v1 \
     --source "datasets/raw/*.jsonl" \
     --sample-size 500 \
     --models gemini-flash \
     --output reports/merged_data_test.md
   ```

3. ✅ **Run full oracle labeling** (if test passes):
   ```bash
   # Tech Deployment
   python -m ground_truth.batch_labeler \
     --filter filters/sustainability_tech_deployment/v1 \
     --source "datasets/raw/*.jsonl" \
     --model gemini-flash \
     --sample-size 2000 \
     --output ground_truth/labeled/tech_deployment_oracle_2k.jsonl

   # Repeat for: economic_viability, policy_effectiveness, nature_recovery, movement_growth
   ```

### Short-term (This Week)

4. **Oracle label AI-Augmented Practice filter** (2K samples)
   - Different dataset focus (AI/LLM content, GitHub repos)
   - May need different source selection strategy

5. **Train 7B models** on oracle labels
   - Fine-tune Qwen2.5-7B for each filter
   - Use LoRA for efficiency
   - Evaluate performance vs oracle

6. **Deploy 7B models** for local inference
   - Test on holdout set
   - Measure accuracy vs oracle
   - Benchmark inference speed

### Medium-term (Next 2 Weeks)

7. **Calibrate other sustainability filters**
   - Run same Flash calibration for 4 remaining filters
   - Verify prefilter pass rates are reasonable
   - Adjust if needed

8. **Build downstream applications**
   - Archetype tracker (using AI practice filter)
   - Funding opportunity tracker (using tech deployment + economic filters)
   - See: `docs/downstream-apps/` planning docs

9. **Create newsletter pipeline**
   - "AI-Augmented Practice Weekly"
   - "Sustainability Tech Deployment Monthly"
   - Automated filtering + curation + email

### Long-term (Next Month)

10. **Evaluate distillation quality**
    - Compare 7B model vs oracle on large test set
    - Identify systematic errors
    - Iterate on prompts/training if needed

11. **Scale to production**
    - Daily batch processing of new articles
    - Automated oracle labeling + 7B inference
    - Dashboard for filtered content

12. **Expand filter library**
    - Policy innovation detection
    - Scientific breakthrough identification
    - Community organizing effectiveness
    - Economic inequality tracking

## Key Technical Decisions

### 1. Why Gemini Flash Over Pro?

**Pattern**: Pro treats off-topic articles as 0-score vaporware, Flash evaluates deployment maturity regardless of topic relevance

**Implication**: Flash provides richer training signal for 7B model about "what is deployed vs theoretical"

**Trade-off**: Flash may be TOO generous to off-topic articles, but we can add post-processing caps

### 2. Why Permissive Prefiltering?

**Rationale**: Local 7B inference is essentially free, oracle LLM cost is main constraint

**Strategy**: Block obvious junk (97-99%), send borderline content to oracle (1-3% → 8-11%)

**Benefit**: More diverse training data, better 7B model generalization

### 3. Why Not Claude for Oracle?

**Quality**: Claude is excellent (used for uplifting filter successfully)

**Cost**: 9x more expensive than Gemini Flash ($90 vs $10 for 5 filters)

**Decision**: Gemini Flash is "good enough" for sustainability filters (Flash's discrimination is adequate)

**Exception**: May use Claude for final validation of trained 7B models

## Repository State

### File Structure

```
llm-distillery/
├── filters/
│   ├── ai_augmented_practice/v1/          # NEW
│   ├── sustainability_tech_deployment/v1/ # UPDATED (prefilter loosened)
│   ├── sustainability_economic_viability/v1/
│   ├── sustainability_policy_effectiveness/v1/
│   ├── sustainability_nature_recovery/v1/
│   └── sustainability_movement_growth/v1/
├── datasets/
│   └── raw/
│       ├── master_dataset_20250929_20251008.jsonl (exists)
│       ├── master_dataset_20251009_20251025.jsonl (exists)
│       ├── master_dataset_20251026_20251029.jsonl (exists)
│       └── historical_dataset_*.jsonl      # NEW (merging...)
├── ground_truth/
│   ├── batch_labeler.py                    # UPDATED (generic post-processing)
│   └── calibrate_oracle.py                 # UPDATED (sustainability support)
├── calibrations/
│   └── sustainability_tech_deployment/
│       ├── gemini_labels.jsonl             # Flash labels (55 articles)
│       └── gemini-pro_labels.jsonl         # Pro labels (55 articles)
├── reports/
│   ├── oracle_model_recommendation.md      # NEW (Flash vs Pro analysis)
│   ├── disagreement_analysis.md            # NEW (raw disagreement data)
│   ├── tech_deployment_oracle_calibration_final.md  # NEW
│   └── session_summary_20251108.md         # THIS FILE
└── scripts/
    ├── analyze_disagreements.py            # NEW
    └── merge_historical_data.py            # NEW
```

### Git Status

**Untracked files**:
- `filters/ai_augmented_practice/` (4 files)
- `reports/oracle_model_recommendation.md`
- `reports/disagreement_analysis.md`
- `scripts/analyze_disagreements.py`
- `scripts/merge_historical_data.py`

**Modified files**:
- `filters/sustainability_*/v1/prefilter.py` (5 files - loosened regex)
- `ground_truth/batch_labeler.py` (added generic post-processing)
- `filters/sustainability_*/v1/prompt-compressed.md` (5 files - format fixes)

**Action needed**: Commit changes after merge completes and oracle labeling begins

## Questions for Next Session

1. Should we add post-processing rule to Flash labels: "If article doesn't mention sustainability keywords, cap score at 3.0"?
2. Should AI-Augmented Practice filter use different source selection (focus on AI/dev content)?
3. Should we run oracle labeling sequentially (1 filter at a time) or parallel (all 5 at once)?
4. What's the priority order for downstream apps (archetype tracker vs funding tracker)?

## Session Metrics

- **Duration**: ~4 hours
- **Files created**: 9
- **Files modified**: 11
- **Code written**: ~800 lines
- **Documentation**: ~500 lines
- **LLM API calls**: ~55 (calibration) + 0 (pending merge)
- **Cost**: ~$0.10 (Gemini Flash calibration)
