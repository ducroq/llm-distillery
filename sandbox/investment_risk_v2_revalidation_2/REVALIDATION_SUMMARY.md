# Investment-Risk v2.1 Revalidation #2

**Date**: 2025-11-15
**Sample**: 30 fresh articles (seed=2025) from historical_dataset_19690101_20251108.jsonl
**Filter Version**: v2.1-academic-filter

---

## Results Summary

### ✅ Academic Paper False Positives: **ELIMINATED**

- Academic papers in sample: 8/30
- Marked as NOISE: 7/8 (87.5%)
- Marked as BLUE: 1/8 (12.5%)
- **Marked as YELLOW/RED: 0/8 (0.0%)**

**False positive rate (academic papers): 0%**
**Target: <3% ✓ PASS**

### Overall Distribution

| Tier | Count | Percentage |
|------|-------|------------|
| RED | 0 | 0.0% |
| YELLOW | 12 | 40.0% |
| GREEN | 0 | 0.0% |
| BLUE | 3 | 10.0% |
| NOISE | 15 | 50.0% |

### YELLOW Articles Breakdown (12/30)

**Legitimate macro/policy risk signals (9):**
- Fed interest rate policy / bank supervision cuts
- China-US trade (soybean purchases)
- Europe energy sovereignty concerns
- Geopolitical: UN condemns US strikes, Lithuania-Belarus border, Hong Kong instability
- China 5-year digital technology plan
- Spanish political/fraud case

**Borderline tech news (3):**
- OpenAI Atlas browser security flaw (cybersecurity concern)
- WhatsApp/Luzia chatbot restriction (regulatory policy)
- Trier city website cyberattack (infrastructure security)

**Assessment**: Borderline cases have policy/security angles. Investment-risk filter
is designed for capital preservation and accepts 25-37% FP rate. These are within
acceptable range (not as clear-cut false positives as academic papers).

---

## Conclusion

**PRIMARY FIX VALIDATED ✓**

The academic paper false positive issue has been completely eliminated on fresh data:
- Revalidation #1 (seed=42): 0/12 academic papers were false positives
- Revalidation #2 (seed=2025): 0/8 academic papers were false positives

The fix (v2.1-academic-filter) generalizes successfully to new, unseen data.

**Borderline tech news**: Within filter's design tolerance for conservative capital
preservation (intentionally casts wide net for risk signals).
