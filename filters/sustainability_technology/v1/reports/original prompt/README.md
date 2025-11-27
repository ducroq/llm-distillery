# Reports - sustainability_technology v1

This directory contains all evaluation reports, decisions, and documentation for the sustainability_technology v1 filter development.

---

## 📋 Quick Navigation

### 🎯 Start Here

1. **[DECISION_PREFILTER_APPROACH.md](DECISION_PREFILTER_APPROACH.md)** - **Final decision document**
   - ✅ Decision: Use keyword prefilter
   - Comparison of all approaches
   - Rationale and approval

2. **[SEMANTIC_PREFILTER_EVALUATION_SUMMARY.md](SEMANTIC_PREFILTER_EVALUATION_SUMMARY.md)** - **Complete evaluation summary**
   - Journey from 6-category → 2-category → final decision
   - Lessons learned
   - When to reconsider semantic approach

---

## 📊 Evaluation Reports

### Semantic Prefilter Evaluations

#### Phase 1: 6-Category (Failed)
**[SEMANTIC_EVALUATION_REPORT.md](SEMANTIC_EVALUATION_REPORT.md)**
- ❌ Result: 15% recall (blocked 85% of good articles)
- Problem: Too many categories, narrow definitions
- Conclusion: Approach fundamentally flawed

#### Phase 2: 2-Category (Improved)
**[SEMANTIC_IMPROVED_EVALUATION.md](SEMANTIC_IMPROVED_EVALUATION.md)**
- ✅ Result: 97.9% recall (fixed the recall problem!)
- Comparison: Only 4% FP improvement over keyword
- Conclusion: Works but marginal benefit

#### GPU Instructions
**[GPU_RUN_INSTRUCTIONS.md](GPU_RUN_INSTRUCTIONS.md)**
- How to run semantic evaluation on GPU machine
- Expected runtime and outputs
- File transfer requirements

---

## 🔧 Strategy & Implementation

**[PREFILTER_STRATEGY.md](PREFILTER_STRATEGY.md)**
- ✅ Approved approach: Keyword + negative blocking
- Positive keywords (wide net)
- Negative keywords (conservative blocking)
- Implementation details

---

## ✅ Validation & Calibration

**[MANUAL_VALIDATION_REPORT.md](MANUAL_VALIDATION_REPORT.md)**
- Manual review of oracle outputs
- Quality assessment
- Edge case analysis

**[CALIBRATION_REPORT.md](CALIBRATION_REPORT.md)**
- Oracle calibration results
- Scoring consistency
- Performance benchmarks

---

## 📚 Document Relationships

```
DECISION_PREFILTER_APPROACH.md (START HERE)
    ├─ References: SEMANTIC_PREFILTER_EVALUATION_SUMMARY.md
    │   ├─ Phase 1: SEMANTIC_EVALUATION_REPORT.md
    │   ├─ Phase 2: SEMANTIC_IMPROVED_EVALUATION.md
    │   └─ GPU runs: GPU_RUN_INSTRUCTIONS.md
    │
    └─ Implementation: PREFILTER_STRATEGY.md
        ├─ Validation: MANUAL_VALIDATION_REPORT.md
        └─ Calibration: CALIBRATION_REPORT.md
```

---

## 🎯 Key Takeaways

### Decision: Keyword Prefilter ✅
- **100% recall** - catches all good articles
- **23.2% FP rate** - acceptable for training data
- **Simple & maintainable** - easy to adjust and debug

### Why Not Semantic?
- Only **4% FP improvement** vs keyword
- Loses **2.1% recall** (misses good articles)
- **16.6s** processing time vs instant
- Requires **GPU infrastructure**
- **More complex** to maintain

### Lessons Learned
1. ✅ 2-category >> 6-category for semantic classification
2. ✅ Broader definitions improve recall
3. ✅ Lower thresholds (0.30) better than high (0.50)
4. ⚠️ For training data: Recall > FP reduction
5. ⚠️ Marginal gains don't justify complexity

---

## 📈 Performance Summary

| Approach | Recall | FP Rate | Precision | Speed | Complexity |
|----------|--------|---------|-----------|-------|------------|
| **Keyword (approved)** | 100% ✅ | 23.2% | 64.4% | 0.0s ✅ | Low ✅ |
| Semantic-2cat | 97.9% | 22.3% | 65.9% | 16.6s | High |
| Semantic-6cat | ~15% ❌ | 2.1% | N/A | N/A | Very High |

---

## 🔄 Next Steps After This Decision

1. ⏳ **Generate 10K training dataset** - Using approved keyword prefilter
2. ⏳ **Train student model** - Single-stage on full relevance spectrum
3. ⏳ **Evaluate student** - Compare to oracle baseline
4. ⏳ **Production deployment** - If student ≥0.90 correlation with oracle

---

## 📅 Timeline

- **2025-11-20**: Keyword prefilter strategy approved
- **2025-11-22**: Semantic 6-category evaluation (failed)
- **2025-11-23**: Semantic 2-category evaluation (improved but marginal)
- **2025-11-24**: **Final decision: Use keyword prefilter** ✅

---

## 📝 Document Status

| Document | Status | Last Updated |
|----------|--------|--------------|
| DECISION_PREFILTER_APPROACH.md | ✅ Final | 2025-11-24 |
| SEMANTIC_PREFILTER_EVALUATION_SUMMARY.md | ✅ Complete | 2025-11-24 |
| SEMANTIC_IMPROVED_EVALUATION.md | ✅ Complete | 2025-11-24 |
| GPU_RUN_INSTRUCTIONS.md | ✅ Complete | 2025-11-23 |
| SEMANTIC_EVALUATION_REPORT.md | ✅ Complete | 2025-11-22 |
| PREFILTER_STRATEGY.md | ✅ Approved | 2025-11-23 |
| MANUAL_VALIDATION_REPORT.md | ✅ Complete | Earlier |
| CALIBRATION_REPORT.md | ✅ Complete | Earlier |

---

## 🔍 Finding Specific Information

### "How did we decide on keyword prefilter?"
→ [DECISION_PREFILTER_APPROACH.md](DECISION_PREFILTER_APPROACH.md)

### "What went wrong with semantic prefiltering?"
→ [SEMANTIC_PREFILTER_EVALUATION_SUMMARY.md](SEMANTIC_PREFILTER_EVALUATION_SUMMARY.md)

### "What are the detailed results?"
- 6-category: [SEMANTIC_EVALUATION_REPORT.md](SEMANTIC_EVALUATION_REPORT.md)
- 2-category: [SEMANTIC_IMPROVED_EVALUATION.md](SEMANTIC_IMPROVED_EVALUATION.md)

### "How do I implement the prefilter?"
→ [PREFILTER_STRATEGY.md](PREFILTER_STRATEGY.md)

### "How was the oracle validated?"
→ [MANUAL_VALIDATION_REPORT.md](MANUAL_VALIDATION_REPORT.md)
→ [CALIBRATION_REPORT.md](CALIBRATION_REPORT.md)

---

## 📧 Questions?

If you have questions about:
- **Decision rationale**: See DECISION_PREFILTER_APPROACH.md
- **Implementation**: See PREFILTER_STRATEGY.md and `../prefilter.py`
- **Evaluation methodology**: See SEMANTIC_PREFILTER_EVALUATION_SUMMARY.md
- **Specific results**: See individual evaluation reports

---

*Last updated: 2025-11-24*
