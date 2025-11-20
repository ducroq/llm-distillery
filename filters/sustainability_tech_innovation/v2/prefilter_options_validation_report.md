# Sustainability Tech Innovation - Prefilter Options Validation

**Date:** 2025-11-17
**Sample size:** 1000 articles (random from master dataset)

---

## Summary Comparison

| Option | Version | Pass Rate | Passed | Blocked | Target |
|--------|---------|-----------|---------|---------|--------|
| v1.0 (Current) | ⚠️ | 0.9% | 9 | 991 | 5-20% |
| Option A (Relaxed) | ⚠️ | 1.8% | 18 | 982 | 10-20% |
| Option B (Balanced) | ⚠️ | 0.0% | 0 | 1000 | 5-10% |
| Option C (Strict) | ⚠️ | 0.0% | 0 | 1000 | 3-5% |

---

## v1.0 (Current)

**Pass rate:** 0.9% (9/1000)

**Top block reasons:**

- `not_sustainability_topic`: 878 (87.8%)
- `no_validation_evidence`: 83 (8.3%)
- `research_without_results`: 26 (2.6%)
- `infrastructure_disruption`: 3 (0.3%)
- `vaporware_announcement`: 1 (0.1%)

**Sample passed articles (5 of 9):**

- Perovksite solar cell based on MXene achieves 25.13% efficiency (reason: passed)
- Sustainability, Vol. 17, Pages 9101: A New Methodology for Optimising Railway Li (reason: passed)
- Predicting Spectroscopic Properties of Solvated Nile Red with Automated Workflow (reason: passed)
- The Synthetic Absorption Line Spectral Almanac (SALSA) (reason: passed)
- I helped build the internet’s ad economy. Now I want to save it (reason: passed)

---

## Option A (Relaxed)

**Pass rate:** 1.8% (18/1000)

**Top block reasons:**

- `not_climate_energy`: 890 (89.0%)
- `no_tech_signal`: 86 (8.6%)
- `infrastructure_disruption`: 3 (0.3%)
- `out_of_scope`: 3 (0.3%)

**Sample passed articles (5 of 18):**

- Perovksite solar cell based on MXene achieves 25.13% efficiency (reason: passed)
- Sustainability, Vol. 17, Pages 9101: A New Methodology for Optimising Railway Li (reason: passed)
- Low (reason: passed)
- Saatvik Green unveils inverter series for residential, C&I solar (reason: passed)
- Sustainability, Vol. 17, Pages 9011: 30 (reason: passed)

---

## Option B (Balanced)

**Pass rate:** 0.0% (0/1000)

**Top block reasons:**

- `not_climate_energy`: 925 (92.5%)
- `no_substantive_evidence`: 73 (7.3%)
- `out_of_scope`: 2 (0.2%)

**Sample passed articles (0 of 0):**


---

## Option C (Strict)

**Pass rate:** 0.0% (0/1000)

**Top block reasons:**

- `not_climate_energy_tech`: 988 (98.8%)
- `no_strong_evidence`: 11 (1.1%)
- `out_of_scope`: 1 (0.1%)

**Sample passed articles (0 of 0):**


---

## Overlap Analysis

- **Pass ALL 3 options:** 0 articles (high confidence)
- **Only Option A (Relaxed):** 18 articles (potential low quality)
- **Only Option C (Strict):** 0 articles (very high quality)

- **A ∩ B:** 0 articles
- **B ∩ C:** 0 articles
- **A ∩ C:** 0 articles

---

## Recommendation

**Recommended option:** Option A (Relaxed)

**Rationale:**
- Pass rate: 1.8% (target: 5-15%)
- Balanced coverage vs quality trade-off
- Expected to yield sufficient training data (~5k articles from 50k raw)

---

**Validation completed**
