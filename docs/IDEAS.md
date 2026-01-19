# LLM Distillery - Ideas

A parking lot for project ideas worth exploring later.

---

## Architecture & Infrastructure

### Multi-Model Ensemble
**Status:** Idea
**Origin:** Potential accuracy improvement

Could train multiple smaller models and ensemble their predictions. Trade-off: complexity vs marginal accuracy gain.

### Active Learning Loop
**Status:** Idea
**Origin:** Training efficiency

Identify uncertain predictions and route back to oracle for labeling. Could improve training data quality with less oracle cost.

### Confidence Calibration
**Status:** Idea
**Origin:** Production reliability

Output calibrated confidence scores alongside dimensional predictions. Useful for flagging uncertain results for human review.

---

## Filter Ideas

See `docs/FUTURE_FILTER_IDEAS.md` for filter-specific ideas (equanimity, etc.)

---

## Tooling & Developer Experience

### Filter Development CLI
**Status:** Idea
**Origin:** DX improvement

```bash
distillery new-filter --name belonging --dimensions 6
distillery score --filter belonging/v1 --articles 1000
distillery train --filter belonging/v1
distillery evaluate --filter belonging/v1
```

### Dashboard for Filter Performance
**Status:** Idea
**Origin:** Monitoring need

Web dashboard showing:
- Per-filter accuracy metrics over time
- Score distributions
- Prefilter false positive/negative rates
- API cost tracking

---

## Research Experiments

### Smaller Base Models
**Status:** Idea
**Origin:** Cost/speed optimization

Test Qwen2.5-0.5B or Phi-2 as base model. Potential for even faster inference on weaker hardware.

### Cross-Filter Transfer Learning
**Status:** Idea
**Origin:** Training efficiency

Train shared encoder across filters, with filter-specific heads. Could reduce training time for new filters.

### Synthetic Training Data
**Status:** Idea
**Origin:** Data augmentation

Use LLM to generate synthetic articles with known scores. Could help with rare edge cases.

---

## Template

**Status:** Idea | Exploring | Parked | Rejected
**Origin:** Context of how this idea emerged

### Concept
Brief description.

### Benefits
- Benefit 1
- Benefit 2

### Challenges
- Challenge 1
- Challenge 2

### Notes
Any other context.

---

*Last updated: 2025-01-16*
