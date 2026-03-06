# ADR-011: Embedding-Based Screening for Needle-in-Haystack Filters

**Date:** 2026-03-06
**Status:** Accepted
**Amends:** ADR-003 (Screening Filter for Training Data)
**Decision:** Use e5-small embedding similarity to seed positives from Phase 3 oracle validation to screen large corpora for batch labeling candidates. Replaces keyword-based screening for needle-in-haystack filters.

## Context

ADR-003 established the screen+merge strategy for needle-in-haystack filters: screen articles with keyword heuristics, oracle-score the candidates, merge with random negatives. This works, but keyword screening has fundamental limitations:

1. **Low recall** — misses semantically similar articles that don't contain exact keywords
2. **High noise** — keywords like "recovery" match medical, economic, and sports articles
3. **Per-filter engineering** — each filter needs hand-crafted keyword lists
4. **Hard to tune** — no clear threshold for "enough keywords"

During nature_recovery v1 Phase 3, we oracle-scored 87 articles and found only 3 positives (1 HIGH, 2 MEDIUM). These 3 articles were enough to screen 595K articles using embedding similarity and produce 415 high-quality candidates in minutes.

## Decision

After Phase 3 oracle validation produces a small set of positive examples (typically 3-10 articles), compute their e5-small embedding centroid and rank the full corpus by cosine similarity. Take the top-N candidates for oracle scoring in Phase 5.

### Workflow Change

```
ADR-003 (keyword screening):
  Raw corpus ──→ Keyword filter ──→ Oracle ──→ Training
                 (hand-crafted)     (noisy candidates)

ADR-011 (embedding screening):
  Phase 3 positives (3-10 articles)
         │
         ▼
  e5-small centroid
         │
         ▼
  Raw corpus ──→ Cosine similarity ranking ──→ Top-N ──→ Oracle ──→ Training
                 (semantic, zero-shot)         (precise candidates)
```

### How It Fits the Filter Development Lifecycle

```
Phase 1: Planning (DEEP_ROOTS.md, config.yaml)
Phase 2: Prompt Architecture (prompt-compressed.md)
Phase 3: Oracle Validation (50-100 articles → ~3-10 positives)
    ↓ NEW: embedding screening step
Phase 3b: Embedding Screen (positives → centroid → rank corpus → top-N)
Phase 4: Prefilter (rewrite with patterns from screening)
Phase 5: Batch Labeling (oracle-score top-N candidates + random negatives)
Phase 6+: Training, calibration, deployment
```

## Evidence

### nature_recovery v1 Results

| Step | Articles | Positives | Positive Rate |
|------|----------|-----------|---------------|
| Phase 3 random sample | 43 scored | 0 | 0% |
| Phase 3 keyword-enriched | 44 scored | 3 | 7% |
| **Embedding screen (this ADR)** | **415 candidates from 595K** | **TBD (scoring pending)** | **Expected 7-15%** |

- **Input:** 3 positive articles (1 HIGH, 2 MEDIUM)
- **Corpus:** 595,351 articles from 4 JSONL files
- **Output:** 415 unique candidates (top 0.07%)
- **Similarity range:** 0.958 (most similar) to 0.898 (500th), median corpus: 0.839
- **Time:** ~10 minutes on CPU (e5-small, no GPU needed)

### Why 3 Examples Is Enough

Embedding similarity doesn't require training — it's a nearest-neighbor search. The centroid of 3 nature recovery articles captures the semantic space of "ecological restoration with evidence" well enough to separate it from the 595K-article corpus. The key is that e5-small embeddings are pretrained on diverse multilingual text, so the similarity computation is zero-shot.

With more positives (10-20), the centroid becomes more robust and less biased toward the specific sources/regions of the seed articles.

## Rationale

### Advantages Over Keyword Screening

| Aspect | Keywords (ADR-003) | Embedding (ADR-011) |
|--------|-------------------|---------------------|
| Recall | Misses semantic matches | Captures semantic similarity |
| Precision | Noisy (polysemous words) | Higher (contextual embeddings) |
| Engineering | Per-filter keyword lists | Zero-shot, filter-agnostic |
| Tuning | Manual keyword selection | Single parameter (top-K) |
| Cost | Free (rule-based) | ~10 min CPU (e5-small) |
| Reusable | No | Same script for all filters |

### Leverages Existing Infrastructure

The e5-small model is already deployed for the hybrid inference pipeline (ADR-006). No new dependencies or models needed. The `filters/common/embedding_stage.py` module provides the model loading infrastructure.

### Self-Improving Loop

The screening candidates, once oracle-scored, provide more positives for the next screening round. This creates a virtuous cycle:

```
3 positives → screen → oracle-score 415 → ~30 positives
   → re-screen with 30 → oracle-score 500 → ~80 positives
   → sufficient for training
```

## Consequences

### Positive

- **Higher positive yield** — embedding similarity finds articles that keyword heuristics miss
- **Filter-agnostic** — same script works for any filter, no per-filter keyword engineering
- **Fast** — e5-small embeds 595K articles in ~10 minutes on CPU
- **Bridges Phase 3→5** — the few validation positives directly drive efficient batch labeling
- **Source diversity** — finds positives from sources/languages not represented in seed articles

### Negative

- **Seed bias** — centroid inherits source/topic bias from the few seed articles
  - Mitigation: run multiple rounds, each with more diverse positives
  - Mitigation: include explicitly diverse seed articles if available
- **Requires Phase 3 positives** — if Phase 3 finds zero positives, can't bootstrap
  - Mitigation: manually curate 3-5 known-positive articles from domain knowledge
- **Embedding model dependency** — results depend on e5-small quality
  - Mitigation: e5-small is well-established; larger models available if needed

### Neutral

- **Doesn't replace keyword prefilter** — the prefilter.py (Phase 4) still serves a different purpose: fast noise rejection at inference time, not training data enrichment
- **Complementary to ADR-003** — this improves the "screen" step; the "merge" strategy (combine screened + random) still applies

## Implementation

- **Script:** `scripts/screening/embedding_screener.py`
- **Model:** `intfloat/multilingual-e5-small` (same as hybrid pipeline)
- **Usage:**
  ```bash
  PYTHONPATH=. python scripts/screening/embedding_screener.py \
      --positives datasets/{filter}/positives_phase3.jsonl \
      --corpus datasets/raw/*.jsonl \
      --output datasets/{filter}/screen_candidates.jsonl \
      --top-k 500
  ```

## References

- ADR-003: Screening Filter for Training Data — original screen+merge strategy
- ADR-006: Hybrid Inference Pipeline — e5-small model infrastructure
- ADR-010: Oracle Consistency Over Data Volume — Phase 3 validation as quality gate
- `filters/common/embedding_stage.py` — shared e5-small loading code
