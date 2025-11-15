# Model Calibration Report: Sustainability Tech Deployment

**Generated**: 2025-11-08 16:52:03
**Prompt**: `filters\sustainability_tech_deployment\v1\prompt-compressed.md`
**Sample Size**: 55 articles (labeled by both models)
**Models**: Claude 3.5 Sonnet vs Gemini 1.5 Pro

---

## Pre-filter Statistics

- **Total Articles Sampled**: 500
- **Passed Pre-filter**: 55 (11.0%)
- **Blocked by Pre-filter**: 445 (89.0%)

**Block Reasons**:

- not_sustainability_topic: 439 (98.7%)
- vaporware_announcement: 5 (1.1%)
- token_scale_negligible: 1 (0.2%)

**Cost Savings**: Pre-filter blocked 445 articles, saving expensive LLM API calls.
For ground truth generation (5,000 articles), this would save ~4450 API calls.

---

## Executive Summary

- **Tier Distribution Difference**: 76.4%
- **Average Score Difference**: 1.50
- **Claude Average Score**: 2.87
- **Gemini Average Score**: 1.38

**Recommendation**: Models differ significantly. **Use Claude** or refine prompt for better Gemini consistency.

---

## Tier Distribution Comparison

| Tier | Claude | Gemini | Difference |
|------|--------|--------|------------|
| Commercial Proven | 3.6% | 3.6% | +0.0% |
| Early Commercial | 10.9% | 1.8% | -9.1% |
| Pilot Stage | 32.7% | 3.6% | -29.1% |
| Vaporware | 52.7% | 90.9% | +38.2% |

---

## Score Statistics

| Metric | Claude | Gemini | Difference |
|--------|--------|--------|------------|
| Average | 2.87 | 1.38 | -1.50 |
| Median | 2.60 | 1.00 | -1.60 |
| Min | 1.00 | 0.00 | -1.00 |
| Max | 7.40 | 7.90 | +0.50 |

---

## Cost Analysis (5,000 articles)

| Model | Cost per Article | Total Cost | Savings |
|-------|------------------|------------|---------|
| Claude 3.5 Sonnet | $0.009 | $45.00 | - |
| Gemini 1.5 Pro | $0.00018 | $0.90 | $44.10 (96%) |

*Gemini pricing assumes Cloud Billing enabled (Tier 1: 150 RPM)*

---

## Sample Article Comparisons

Comparing 55 Claude articles vs 55 Gemini articles

**Matched articles**: 55 (analyzed successfully by both models)

Showing 5 articles with largest score disagreement:

### Sample 1 - Disagreement: 7.00

**Title**: Opinion: Onshoring could threaten the resilient supply chain for biosimilars and generics

**Excerpt**: Supply chain disruptions with biosimilars will threaten the entire biologics ecosystem, writes a Samsung Bioepis VP and head of regulatory strategy and policy....

| Model | Score | Tier | Reasoning |
|-------|-------|------|-----------|
| Claude | 7.00 | commercial_proven | ... |
| Gemini | 0.00 | vaporware | ... |

### Sample 2 - Disagreement: 5.30

**Title**: Optimal transmission expansion modestly reduces decarbonization costs of U.S. electricity

**Excerpt**: arXiv:2402.14189v3 Announce Type: replace-cross Abstract: Expanding interregional transmission is widely viewed as essential for integrating clean energy into decarbonized power systems. Using the ope...

| Model | Score | Tier | Reasoning |
|-------|-------|------|-----------|
| Claude | 6.40 | early_commercial | ... |
| Gemini | 1.10 | vaporware | ... |

### Sample 3 - Disagreement: 5.25

**Title**: Byaidu/PDFMathTranslate

**Excerpt**: PDF scientific paper translation with preserved formats - 基于 AI 完整保留排版的 PDF 文档全文双语翻译，支持 Google/DeepL/Ollama/OpenAI 等服务，提供 CLI/GUI/MCP/Docker/Zotero English | 简体中文 | 繁體中文 | 日本語 | 한국어 PDFMathTranslate 1...

| Model | Score | Tier | Reasoning |
|-------|-------|------|-----------|
| Claude | 6.25 | early_commercial | ... |
| Gemini | 1.00 | vaporware | ... |

### Sample 4 - Disagreement: 5.15

**Title**: States sue to stop Trump cancellation of $7 billion solar grant program

**Excerpt**: States sue to stop Trump cancellation of $7 billion solar grant program Reuters...

| Model | Score | Tier | Reasoning |
|-------|-------|------|-----------|
| Claude | 6.15 | early_commercial | ... |
| Gemini | 1.00 | vaporware | ... |

### Sample 5 - Disagreement: 3.90

**Title**: Aangevallen Nederlands vrachtschip nog in brand op zee bij Jemen

**Excerpt**: Het Nederlandse vrachtschip dat gisteren in de Golf van Aden werd aangevallen, staat nog steeds in brand. De MV Minervagracht drijft nog stuurloos op zee. Het schip was gisteren onderweg vanuit Djibou...

| Model | Score | Tier | Reasoning |
|-------|-------|------|-----------|
| Claude | 3.90 | pilot_stage | ... |
| Gemini | 0.00 | vaporware | ... |

---

## Next Steps

1. Review the tier distribution and score statistics
2. Examine sample articles with large disagreements
3. If distributions are similar, proceed with Gemini for cost savings
4. If distributions differ, consider:
   - Refining the prompt for better clarity
   - Adding more examples to the prompt
   - Using Claude for ground truth (higher cost but higher quality)
