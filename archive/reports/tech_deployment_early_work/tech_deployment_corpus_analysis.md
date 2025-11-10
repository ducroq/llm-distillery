# Tech Deployment Corpus Analysis - Reality Check

**Date**: 2025-11-09
**Analysis**: Searched 147K article corpus for tier-specific content

---

## Key Finding: Corpus is Fundamentally Imbalanced

**Targeted Search Results**:

| Tier | Signal Strength | Candidates Found | Need | Gap | Feasibility |
|------|----------------|------------------|------|-----|-------------|
| **Tier 1** (deployed ≥8.0) | Very strong (score ≥5) | **5** | 600 | -595 | ❌ **IMPOSSIBLE** |
| **Tier 2** (early comm 6.0-7.9) | Strong (score ≥3) | **37** | 600 | -563 | ❌ **IMPOSSIBLE** |
| **Tier 3** (pilot 4.0-5.9) | Moderate (score ≥2) | **587** | 600 | -13 | ✅ **FEASIBLE** |
| **Tier 4** (vaporware <4.0) | Random sample | ~145,000 | 600 | +144,400 | ✅ Have plenty |

**Conclusion**: The corpus is a **news aggregator**, not a deployment database. It contains announcements, concepts, and pilot projects - NOT operational systems.

---

## What This Means

### Current State
- Have 2,080 labels: 1,688 vaporware, 224 pilot, 139 early comm, 29 deployed
- Can add ~500 more pilot examples from targeted candidates
- **Cannot** meaningfully increase deployed or early commercial from this corpus

### Realistic Options

**Option A: Accept Imbalance + Heavy Mitigation**
- Final dataset: ~2,600 labels (2,080 current + 500 pilot candidates)
- Distribution: ~65% vaporware, ~25% pilot, ~8% early commercial, ~2% deployed
- Training: 20x oversampling for deployed, 10x for early commercial, 3x for pilot
- Class weighting + focal loss
- **Pros**: Uses real data, cheap ($0.50 for 500 pilot labels)
- **Cons**: Model will struggle with high tiers, may never predict ≥8 scores

**Option B: Synthetic Augmentation for High Tiers**
- Take 29 deployed examples → generate 20 variations each → 580 synthetic deployed
- Take 139 early commercial → generate 4 variations each → 556 synthetic early comm
- Final: ~4,000 labels with synthetic balance
- **Pros**: Balanced dataset achieved
- **Cons**: Synthetic data quality lower, potential overfitting, cost ~$1-2

**Option C: Restart with Better Corpus** (following FILTER_WORKFLOW.md properly)
- Acknowledge we violated Step 2 ("Plan data sourcing FIRST")
- Find sources with actual deployment content:
  - IEA/IRENA operational reports (not in current corpus)
  - Trade publication archives (only partial coverage)
  - Company earnings calls / operational updates (not in corpus)
  - Government statistics (grid data, not in corpus)
- Restart data collection phase
- **Pros**: Proper ground truth, high-quality dataset
- **Cons**: Requires new data sources, significantly more work

**Option D: Hybrid Approach** (RECOMMENDED)
- Use what we have: 2,080 current labels
- Add 500 pilot candidates from corpus (label them): ~$0.50
- Synthetically augment deployed tier only (29 → 300): ~$0.30
- Accept early commercial underrepresentation
- Final: ~2,900 labels with partial synthetic help
- **Pros**: Pragmatic, leverages corpus strengths, low cost
- **Cons**: Still imbalanced, model may struggle with Tier 2

---

## Hybrid Approach Details (RECOMMENDED)

### Step 1: Label Tier 3 Candidates
- Have 587 pilot candidates
- Label all 587 (cost: ~$0.59)
- Expected actual pilot yield: ~300-400 (rest may be vaporware despite signals)
- **New pilot total**: 224 + 350 = ~574 examples

### Step 2: Synthetic Augmentation for Tier 1 Only
- Take 29 deployed examples
- Generate 10 paraphrased variations each using Gemini Flash
- Label variations (~$0.29)
- **New deployed total**: 29 + 290 = ~319 examples

### Step 3: Accept Tier 2 Underrepresentation
- Keep 139 early commercial examples
- Don't augment (hardest tier to fake realistically)
- Rely on oversampling (5x) during training

### Step 4: Downsample Vaporware
- Have 1,688, need ~600
- Random sample 600 best examples (variety of sources)

### Final Dataset
- **Vaporware**: 600 (20.5%)
- **Pilot**: 574 (19.6%)
- **Early Commercial**: 139 (4.7%) ← Still underrepresented
- **Deployed**: 319 (10.9%) ← Boosted with synthetic
- **Total**: 2,932 examples

### Training Mitigation
- Oversample early commercial 5x (139 → 695)
- Oversample deployed 2x (319 → 638)
- Final training set: ~4,500 examples with balance
- Class weighting for remaining imbalance
- Focal loss for hard examples

### Cost
- Label 587 pilot: $0.59
- Generate + label 290 deployed synthetic: $0.59
- **Total**: ~$1.20

### Time
- ~2 hours for labeling
- ~30 min for synthetic generation

---

## Comparison to "Proper" FILTER_WORKFLOW.md Approach

**What We Should Have Done** (Step 2 of workflow):

1. **Analyzed natural tier distribution** BEFORE random sampling
2. **Identified tier-specific sources**:
   - Tier 1: IEA reports, case studies, grid statistics
   - Tier 2: Trade publications, earnings calls
   - Tier 3: Grant announcements, pilot databases
   - Tier 4: General tech news (what we have)
3. **Collected from each source separately**
4. **Curated balanced dataset** from diverse sources

**What We Actually Did**:
- Random sampled from general tech news aggregator
- Got expected vaporware-heavy distribution
- Attempted post-hoc stratified sampling (too late)

**Lesson**: "Don't randomly sample from imbalanced sources and hope for the best" - this is exactly what happened

---

## Recommendation

**Use Hybrid Approach (Option D)**

**Why**:
1. **Pragmatic**: Works with corpus we have
2. **Cost-effective**: ~$1.20 vs $10+ for full re-labeling
3. **Partial quality**: Real data for 3 tiers, synthetic only for hardest tier
4. **Time-efficient**: 2-3 hours vs days of new corpus collection
5. **Proves pipeline**: Can still demonstrate distillation works

**Success Criteria** (adjusted for reality):
- Vaporware detection: ≥85% (majority class)
- Pilot detection: ≥70% (good coverage)
- Early commercial: ≥50% (underrepresented, lower bar)
- Deployed: ≥40% (synthetic augmented, lower bar)
- MAE per dimension: ≤1.5

**Next Steps**:
1. Label 587 pilot candidates
2. Generate + label 290 deployed synthetic examples
3. Merge all labels
4. Sample balanced dataset
5. Create train/val/test splits
6. Train model
7. Evaluate

**For Future Filters**: Follow FILTER_WORKFLOW.md Step 2 properly - source tier-specific content BEFORE labeling

---

**Analyst**: Claude (AI Assistant)
**Date**: 2025-11-09
**Reality**: Corpus lacks deployed tech content - synthetic augmentation needed
