# Prefilter Strategy - sustainability_technology v1

**Date**: 2025-11-23
**Status**: Approved with negative keyword blocking

## Strategy Overview

**Two-stage filtering**:
1. **Positive keywords** - Wide net for sustainability content (keeps 'oil', 'gas', substring matching)
2. **Negative keywords** - Block obvious off-topic content (sports, celebrity, lifestyle)

## Rationale

### Why NOT tighten positive keywords

**Against word boundary matching**:
- ❌ Would break prefix matching: 'sustainab' → sustainable, sustainability, unsustainability
- ❌ Would break compound words: 'oil' → biofuel, turmoil (turmoil is false positive, but biofuel is valid)
- ❌ Risks false negatives on legitimate articles

**Current substring matching is acceptable**:
- 'oil' in 'turmoil' = false positive, BUT 'oil' in 'biofuel' = valid
- 'gas' in 'biogas' = valid
- Tradeoff: Accept false positives, let oracle filter them out

### Why ADD negative keywords

**Problem**: Some false positives are obvious and waste oracle API costs
- Celebrity gossip: Ireland Baldwin (matched 'oil' in 'turmoil')
- Sports news: Soccer matches (matched 'gas' or 'cop')
- Lifestyle articles: Wedding/fashion (matched generic terms)

**Solution**: Conservative negative keyword blocking
- Only block if 2+ negative keyword matches (or 1 keyword appears 2+ times)
- Very specific keywords: 'kardashian', 'soccer', 'touchdown', 'wedding dress'
- Avoids false negatives (e.g., doesn't block 'sport' since 'transport' contains it)

## Implementation

### Negative Keyword Categories

**Sports** (25 keywords):
- Team sports: 'soccer', 'football match', 'basketball', 'baseball', 'hockey'
- Leagues: 'premier league', 'champions league', 'fifa', 'nfl', 'nba', 'nhl', 'mlb'
- Terms: 'goal scorer', 'touchdown', 'home run', 'penalty kick', 'hat trick'

**Entertainment & Celebrity** (16 keywords):
- Celebrities: 'kardashian', 'baldwin', 'bieber', 'swift', 'beyonce'
- Shows: 'real housewives', 'bachelor', 'reality show', 'sitcom'
- Events: 'box office', 'red carpet', 'grammy', 'oscar', 'emmy'

**Personal Lifestyle** (12 keywords):
- Weddings: 'wedding dress', 'bridal', 'engagement ring'
- Beauty: 'makeup tutorial', 'beauty tips', 'hairstyle'
- Other: 'dating advice', 'horoscope', 'lottery'

### Blocking Threshold

**Requirement**: 2+ total negative keyword occurrences

Examples:
- 'baldwin' appears 2 times → BLOCKED ✓
- 'soccer' appears 1 time → PASSED (single mention might be incidental)
- 'soccer' + 'goal scorer' → BLOCKED ✓ (2 unique keywords)
- 'kardashian' + 'reality show' → BLOCKED ✓

## Expected Impact

### Before negative keywords:
- False positive rate: ~30% (30 articles scoring ≤2.0 out of 100)
- Estimated cost waste: ~$22.50 per 10K articles

### After negative keywords:
- Expected false positive rate: ~15-20% (blocking obvious cases)
- Estimated cost savings: ~$10-12 per 10K articles
- Low risk of false negatives (very specific keywords)

## Next Steps

1. ✅ Add `_is_obvious_off_topic()` method to prefilter.py
2. ✅ Update `apply_filter()` to call negative keyword check
3. ⏳ Re-run 100-article calibration to verify impact
4. ⏳ Generate 10K training dataset (accept remaining 15-20% false positives)
5. ⏳ Train student model (single-stage approach)

## Tradeoff Acceptance

**Accepted tradeoffs**:
- ✅ Keep substring matching (accept 'oil' in 'turmoil' false positives)
- ✅ Accept 15-20% false positive rate after negative keyword blocking
- ✅ Oracle correctly scores false positives as 1.0 (training signal works)
- ✅ Single-stage approach (model learns full relevance spectrum)

**Why 15-20% FP rate is acceptable**:
1. Student model needs negative examples (learns to recognize irrelevant content)
2. Oracle filtering is working correctly
3. Cost is reasonable (~$60 for 10K articles including ~1.5K false positives)
4. Alternative (word boundaries) risks false negatives on legitimate content
5. Remaining false positives are edge cases (hard to filter without context)

## Function Location

Implementation: `filters/sustainability_technology/v1/negative_keywords_function.py`

Integration: Add to `prefilter.py` between sustainability check and final pass
