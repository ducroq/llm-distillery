# Semantic Prefilter Integration Guide

## How Candidate Labels Are Chosen

### The Model's Perspective

The model (bart-large-mnli) converts each label into a **hypothesis**:

```
Label: "sustainability technology and renewable energy"
↓
Hypothesis: "This text is about sustainability technology and renewable energy"
↓
Model asks: "Does the article support this hypothesis?"
↓
Score: 0.0 (no support) to 1.0 (strong support)
```

### Design Principles

**1. Be Specific** - More context = better accuracy
- ❌ Vague: `"sustainability"`
- ✅ Specific: `"sustainability technology and renewable energy solutions"`

**2. Use Natural Language** - The model understands phrases
- ✅ "articles about climate solutions and environmental technology"
- ✅ "professional sports games and athletic competitions"
- ❌ "green stuff"

**3. Include Negatives** - Help the model discriminate
```python
[
    "sustainability technology",  # What we WANT
    "sports",                      # What we DON'T want
    "entertainment",               # What we DON'T want
]
```

**4. Test and Iterate**
- Start with defaults
- Run on 100 sample articles
- Check false positives/negatives
- Refine label wording
- Re-test

### Example: Refining Labels

**Round 1** - Too vague:
```python
["sustainability", "other topics"]
```
Result: 40% false positive rate (too many "other topics" with incidental sustainability mentions)

**Round 2** - More specific:
```python
["sustainability technology and renewable energy", "sports", "entertainment", "general news"]
```
Result: 15% false positive rate (better, but some military articles pass)

**Round 3** - Add missing category:
```python
[
    "sustainability technology renewable energy and climate solutions",
    "sports and athletics",
    "entertainment and celebrities",
    "military conflict and warfare",  # ← Added
    "general news"
]
```
Result: 8% false positive rate ✓

---

## Integration Options

### Option A: Semantic-Only (High Accuracy)

Replace keyword prefilter entirely:

```python
# In prefilter.py
from filters.sustainability_technology.v1.semantic_prefilter import SemanticPreFilter

class SustainabilityTechnologyPreFilterV1(BasePreFilter):
    def __init__(self):
        super().__init__()
        self.semantic_filter = SemanticPreFilter(
            confidence_threshold=0.35,
            device=-1  # CPU
        )

    def apply_filter(self, article: Dict) -> Tuple[bool, str]:
        # Only use semantic classification
        return self.semantic_filter.apply_filter(article)
```

**Pros**: Most accurate
**Cons**: Slow (~0.5-1 sec per article on CPU)

---

### Option B: Keyword First, Semantic Second (RECOMMENDED)

Fast keyword prefilter, then semantic for edge cases:

```python
class SustainabilityTechnologyPreFilterV1(BasePreFilter):
    def __init__(self, use_semantic=True):
        super().__init__()
        self.use_semantic = use_semantic
        if use_semantic:
            self.semantic_filter = SemanticPreFilter(
                confidence_threshold=0.35,
                device=-1
            )

    def apply_filter(self, article: Dict) -> Tuple[bool, str]:
        text = self._get_combined_clean_text(article)

        # Stage 1: Fast keyword check (99% of cases)
        has_positive = self._is_sustainability_related(text)
        has_negative = self._is_obvious_off_topic(text)

        # Clear positive: sustainability keywords + no negative keywords
        if has_positive and not has_negative:
            return (True, "passed_keywords")

        # Clear negative: no sustainability keywords
        if not has_positive:
            return (False, "not_sustainability_topic")

        # Ambiguous: has both positive AND negative keywords
        # Use semantic classifier to decide (the Ireland Baldwin case)
        if self.use_semantic:
            return self.semantic_filter.apply_filter(article)
        else:
            # Fallback: Block if negative keywords present
            return (False, "obvious_off_topic")
```

**Workflow**:
1. Check keywords (fast, ~0.001 sec)
2. If ambiguous (sustainability + negative keywords), use semantic (~0.5 sec)
3. Only ~5-10% of articles need semantic classification

**Pros**:
- Fast for 90-95% of articles
- Accurate for ambiguous cases
- Best of both worlds

---

### Option C: Semantic with Keyword Fallback

Try semantic first, fall back to keywords on error:

```python
def apply_filter(self, article: Dict) -> Tuple[bool, str]:
    try:
        # Try semantic classification
        return self.semantic_filter.apply_filter(article)
    except Exception as e:
        # Fallback to keywords if semantic fails
        print(f"Semantic filter failed, using keywords: {e}")
        text = self._get_combined_clean_text(article)

        if not self._is_sustainability_related(text):
            return (False, "not_sustainability_topic")
        if self._is_obvious_off_topic(text):
            return (False, "obvious_off_topic")
        return (True, "passed_keywords")
```

**Use case**: Production environments where robustness > accuracy

---

## Tuning the Confidence Threshold

Test on your calibration data:

```python
from filters.sustainability_technology.v1.semantic_prefilter import SemanticPreFilter
import json

# Load calibration articles
articles = []
for line in open('sandbox/sustainability_technology_v1_calibration/sustainability_technology/scored_batch_001.jsonl'):
    articles.append(json.loads(line))

# Test different thresholds
thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

for threshold in thresholds:
    filter = SemanticPreFilter(confidence_threshold=threshold)

    passed = 0
    false_positives = 0  # Articles that scored ≤2.0 but passed filter

    for article in articles:
        # Calculate oracle score
        analysis = article['sustainability_technology_analysis']
        avg_score = sum(analysis[d]['score'] for d in analysis if d != 'analyzed_at') / 6

        # Check if semantic filter passes
        result, reason = filter.apply_filter(article)

        if result:
            passed += 1
            if avg_score <= 2.0:
                false_positives += 1

    fp_rate = (false_positives / passed * 100) if passed > 0 else 0
    print(f"Threshold {threshold:.2f}: {passed}/100 passed, FP rate: {fp_rate:.1f}%")
```

Expected output:
```
Threshold 0.25: 75/100 passed, FP rate: 20.0%
Threshold 0.30: 68/100 passed, FP rate: 15.2%
Threshold 0.35: 62/100 passed, FP rate: 9.7%  ← Recommended
Threshold 0.40: 55/100 passed, FP rate: 5.5%
Threshold 0.45: 48/100 passed, FP rate: 2.1%
Threshold 0.50: 42/100 passed, FP rate: 0.0%  ← Very restrictive
```

---

## Summary

**Candidate Labels** = The categories you want the model to distinguish

**How to choose them**:
1. Start with recommended defaults (sustainability + common negatives)
2. Run on 100 sample articles
3. Check where false positives land
4. Add/refine categories
5. Adjust confidence threshold
6. Re-test

**Recommended setup**: Option B (Keyword + Semantic hybrid)
- Fast for clear cases
- Accurate for ambiguous cases
- Best overall performance
