# Calibration Report - sustainability_technology v1

**Date**: 2025-11-23  
**Oracle**: gemini-flash  
**Samples**: 100 articles (randomly sampled from 109,232 eligible)

## Summary

✅ **APPROVED FOR TRAINING**
- PC1 variance: 66.5% (< 70% threshold) ✓
- High correlations (>0.85): 0 ✓
- Moderate correlations reflect real domain relationships
- 100% oracle success rate

## Dimension Statistics

| Dimension | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| technology_readiness_level | 4.39 | 2.68 | 0.0 | 9.0 |
| technical_performance | 4.49 | 2.23 | 0.0 | 7.5 |
| economic_competitiveness | 2.79 | 1.97 | 0.0 | 7.0 |
| life_cycle_environmental_impact | 2.51 | 1.37 | 0.0 | 7.0 |
| social_equity_impact | 2.68 | 1.66 | 0.0 | 7.5 |
| governance_systemic_impact | 3.71 | 2.11 | 0.0 | 7.0 |

Good score distribution across all dimensions - no clustering at extremes.

## Correlation Analysis

### High Correlations (r > 0.85)
**None found** ✓ - Dimensions are independent

### Moderate Correlations (0.70-0.85)
- **TRL ↔ Economics**: r=0.763 (mature tech tends to be cheaper)
- **Social ↔ Governance**: r=0.768 (governance affects social equity)
- **Environment ↔ Governance**: r=0.708 (governance influences environment)

**Assessment**: These correlations reflect **realistic domain relationships**, not oracle failure. Individual articles show variation from these trends (e.g., high TRL + low Economics for nuclear power).

## PCA Analysis

### Variance Explained
- PC1: 66.5% (multi-dimensional problem ✓)
- PC2: 12.8%
- PC3: 8.6%
- PC4: 5.6%
- PC5: 3.8%
- PC6: 2.7%

### Intrinsic Dimensionality
- **90% variance**: 4/6 dimensions needed
- **95% variance**: 5/6 dimensions needed
- **Redundancy score**: 60% (moderate, acceptable)

**Comparison to old filters**:
- Old: PC1 = 89.1% (essentially one-dimensional)
- New: PC1 = 66.5% (clearly multi-dimensional) ✓

## Decision

**PROCEED TO FULL TRAINING (5,000 samples)**

Moderate correlations are acceptable because:
1. They reflect real-world causal relationships
2. PC1 < 70% indicates multi-dimensional problem
3. Zero high correlations (r > 0.85)
4. Dimensions provide distinct filtering value
5. Much better than old filters (89% → 67% PC1)

## References

- ADR-001: Moderate Dimension Correlations Are Acceptable
- Visualizations: `correlation_heatmap.png`, `pca_analysis.png`
- Raw data: `analysis_summary.json`
- Calibration data: `sandbox/sustainability_technology_v1_calibration/`
