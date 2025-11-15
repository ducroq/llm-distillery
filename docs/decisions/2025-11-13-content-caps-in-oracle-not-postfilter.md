# Enforce Content Type Caps in Oracle, Not Post-Filter

**Date:** 2025-11-13
**Status:** Accepted

## Context

Article metadata includes tags, entities, sentiment, and other signals that could be used to detect content types (corporate finance, military/security, business news). The question is: **Where should content type caps be enforced?**

**Options:**
1. In oracle prompt (semantic understanding)
2. In post-filter using metadata (tag/entity matching)
3. Both (redundant enforcement)

## Decision

**Enforce content type caps in the oracle prompt, NOT in the post-filter.**

**Post-filter stays simple:** Pure arithmetic (weighted average, tier assignment, reasoning flagging). No content type detection logic.

**Metadata pre-filter is optional:** Can add quality gating (clickbait, duplicates) before oracle calls to save API costs, but this is NOT required.

## Rationale

### 1. Oracle Understands Context Better

**Example: Worker cooperative vs VC-funded startup**

Article: "Worker cooperative raises $2M to expand employee-owned bakery network"

**Metadata signals:**
- Tags: `["business", "finance"]`
- Entities: `["$2M", "funding"]`
- → Tag-based filter would cap at 2.0 (corporate finance)

**Oracle semantic understanding:**
- Worker cooperative → exception to corporate finance cap
- Employee ownership → high collective_benefit
- → Oracle correctly scores collective_benefit = 7, overall = 6.5

**Post-filter with metadata would mis-classify this as corporate finance and cap it incorrectly.**

### 2. Avoid False Positives from Noisy Tags

**Problem:** Automated tagging can be incorrect or incomplete.

**Example: Peace process article**

Article: "Military commanders from both sides sign historic demilitarization agreement"

**Metadata signals:**
- Tags: `["military", "security", "defense"]`
- Entities: `["military commanders"]`
- → Tag-based filter would cap at 4.0 (military/security)

**Oracle semantic understanding:**
- Demilitarization → exception to military cap
- Peace process → high progress score
- → Oracle correctly scores without cap

**We don't want to miss good content due to bad tags.**

### 3. Single Source of Truth

**If we enforce caps in both oracle AND post-filter:**
- Duplicate logic (config → oracle prompt + config → post-filter code)
- Risk of inconsistency (oracle updated, post-filter not updated)
- Harder to maintain

**With caps only in oracle:**
- Config → Oracle prompt (single source of truth)
- Post-filter just does math on oracle output
- Easy to update caps (change config → update oracle prompt)

### 4. Keep Components Focused

**Clean architecture:**

```
Metadata → [Optional: Pre-filter quality gate]
              ↓
           Oracle (semantic understanding + content caps)
              ↓
           Dimensional scores (already encode content type)
              ↓
           Post-filter (arithmetic: weighted average → tier)
              ↓
           Result
```

**Each component has ONE responsibility:**
- **Pre-filter (optional):** Quality gate using cheap metadata checks
- **Oracle:** Semantic understanding, dimensional scoring, content cap enforcement
- **Post-filter:** Pure math (no semantic logic, no content detection)

### 5. Dimensional Scores Already Encode Content Type

**The uplifting filter already handles corporate finance:**

```yaml
collective_benefit:
  weight: 0.38  # Heaviest weight - acts as gatekeeper
  scale: |
    0-2: Elite only (investors/founders benefit)
    3-4: Limited group
    5-6: Moderate community
    7-8: Broad community
    9-10: Universal benefit
```

**Corporate finance without exceptions:**
- Oracle scores collective_benefit = 2
- With weight 0.38, overall score naturally capped ~3.5
- Tier assignment = "not_uplifting"

**Worker cooperative:**
- Oracle scores collective_benefit = 7
- Overall score = 6+ (depends on other dimensions)
- Tier assignment = "connection" or "impact"

**Content type caps are already working through dimensional scores.** Post-filter doesn't need to re-detect content types.

## Implementation

### 1. Update Oracle Prompts ✅ Required

Add content_type_caps from config to oracle prompts.

**Example for uplifting filter prompt:**

```markdown
## Content Type Score Caps

Apply maximum score caps for certain content types:

### Corporate Finance (Max score: 2.0)
**Triggers:** Stock prices, earnings reports, funding rounds, valuations, M&A, IPO announcements

**Exceptions (do NOT cap):**
- Worker cooperatives
- Public benefit corporations
- Open source projects with broad access
- Community ownership models
- Affordable access initiatives

**Reasoning:** Corporate finance primarily benefits elite groups (investors, executives) unless structured for public/worker benefit.

### Military/Security (Max score: 4.0)
**Triggers:** Military buildup, defense spending increases, weapons development, NATO expansion, armed forces deployment

**Exceptions (do NOT cap):**
- Demilitarization agreements
- Peace processes and negotiations
- Conflict resolution initiatives
- Reconciliation programs
- Arms reduction treaties

**Reasoning:** Military/security expansion is not uplifting unless moving toward peace.

### Business News (Max score: 4.0 IF collective_benefit < 6)
**Triggers:** Product launches, business expansion, company announcements, market growth

**Condition:** Only cap if collective_benefit < 6 (limited group benefit)

**Do NOT cap if:** Broad public benefit (collective_benefit >= 6)

**Reasoning:** Business news is only uplifting if there's significant collective benefit.
```

### 2. Post-Filter Stays Simple ✅ Complete

**Current implementation:**
```python
class PostFilter:
    def classify(self, scores, flag_reasoning_threshold=None):
        # 1. Calculate weighted average
        overall_score = self.calculate_overall_score(scores)

        # 2. Apply gatekeeper rules (sustainability filters)
        #    Example: if deployment_maturity < 5.0 → cap at 4.9
        if self.gatekeeper_rules:
            overall_score, rules = self.apply_gatekeeper_rules(scores, overall_score)

        # 3. Assign tier
        tier = self.assign_tier(overall_score)

        # 4. Flag for reasoning (optional)
        needs_reasoning = overall_score >= flag_reasoning_threshold if flag_reasoning_threshold else False

        return {
            "tier": tier,
            "overall_score": overall_score,
            "needs_reasoning": needs_reasoning,
            ...
        }
```

**NO content type detection logic.** No article metadata required.

**Note:** `apply_gatekeeper_rules()` for sustainability filters is different from content caps:
- Gatekeeper rules are mathematical: `if dimension_score < threshold → cap overall`
- Content caps are semantic: `if article_is_corporate_finance → cap overall`
- Gatekeeper rules can be post-filter (simple math)
- Content caps must be oracle (semantic understanding)

### 3. Metadata Pre-Filter (Optional)

**Optional:** Add quality gate before oracle calls to save API costs.

**Checks (all cheap metadata operations):**
- Duplicate detection (check hashes)
- Too short (word_count < 50)
- Clickbait/sensationalism (extreme surprise + fear emotions)
- Parse failures (quality_score < 0.3, robust_parsing_used = false)

**Benefits:**
- Saves oracle API costs by filtering junk early
- Fast (metadata checks are cheap)
- Low false positive risk (only extreme cases)

**Not required because:**
- Oracle is already cheap (~$0.001/article)
- Volume is low (1000 articles/day)
- Quality issues are rare in curated feeds

**Decision:** Pre-filter is optional. Can add later if quality/cost becomes an issue.

## Config Structure

**Keep content_type_caps in config.yaml** (they're valuable filter rules):

```yaml
scoring:
  content_type_caps:
    corporate_finance:
      max_score: 2.0
      triggers:
        - "Stock prices, earnings, funding rounds, valuations, M&A, IPO"
      exceptions:
        - "worker cooperative"
        - "public benefit"
        - "open source"
        - "affordable access"
        - "community ownership"

    military_security:
      max_score: 4.0
      triggers:
        - "Military buildup, defense spending, weapons, NATO expansion"
      exceptions:
        - "demilitarization"
        - "peace processes"
        - "conflict resolution"
        - "reconciliation"

    business_news:
      max_score: 4.0
      condition: "collective_benefit < 6"
      triggers:
        - "Product launches, business expansion, company announcements"
```

**How it's used:**
- Config → Oracle prompt (enforced semantically)
- Config → Filter documentation (explains expected behavior)
- NOT used in post-filter code

## Alternatives Considered

### Alternative 1: Enforce Caps in Post-Filter Using Metadata

**Approach:** Post-filter scans tags/entities and applies caps

**Example:**
```python
if "business" in tags and len(entities.monetary) > 0:
    overall_score = min(overall_score, 2.0)  # Corporate finance cap
```

**Pros:**
- Could catch edge cases oracle missed
- Deterministic (same tags always trigger cap)

**Cons:**
- ❌ False positives (worker cooperative tagged as "business")
- ❌ Duplicate logic (oracle already scored collective_benefit low)
- ❌ Post-filter "overriding" oracle intelligence (bad pattern)
- ❌ Needs article metadata (more complexity)
- ❌ Hard to maintain (keyword lists drift from config)

**Decision:** Rejected - Risk of false positives too high

### Alternative 2: Enforce Caps in Both Oracle and Post-Filter

**Approach:** Defense in depth - oracle applies caps, post-filter double-checks

**Pros:**
- Catches oracle errors/hallucinations
- Redundancy

**Cons:**
- ❌ If oracle works, post-filter check is redundant
- ❌ If oracle fails, post-filter using noisy tags also likely wrong
- ❌ Duplicate logic to maintain
- ❌ Still has false positive risk

**Decision:** Rejected - Oracle quality is high enough, redundancy not worth complexity

### Alternative 3: Remove content_type_caps from Config

**Approach:** Rely entirely on dimensional scores, no explicit caps

**Reasoning:** collective_benefit dimension already encodes corporate finance detection

**Pros:**
- Simplest approach
- Dimensional scores handle it naturally

**Cons:**
- ❌ Loses explicit documentation of filter rules
- ❌ Harder to audit ("why was this capped?")
- ❌ Oracle might not be consistent without explicit instruction

**Decision:** Rejected - Explicit caps in oracle prompt ensure consistency

## Consequences

### Positive

- ✅ **No false positives:** Oracle understands context/exceptions
- ✅ **Simple post-filter:** Pure math, easy to test
- ✅ **Single source of truth:** Config → Oracle prompt
- ✅ **Maintainable:** Change config → update oracle prompt only
- ✅ **Semantic correctness:** Oracle makes intelligent decisions
- ✅ **Auditable:** Dimensional scores show why article scored low

### Negative

- ⚠️ **Oracle dependency:** If oracle hallucinates/errors, no backup enforcement
- ⚠️ **Consistency risk:** Different oracle models might interpret caps differently

### Neutral

- Post-filter doesn't use article metadata (simpler interface)
- Pre-filter is optional (can add later if needed)

### Mitigation

**Oracle consistency:**
- Use standardized prompts across all oracle models (Gemini Flash, Claude)
- Include content caps explicitly in prompt
- Test oracle output against labeled dataset
- If consistency issues arise, could add post-filter checks as backup

**Oracle errors:**
- Rare (oracle is good at semantic understanding)
- If systematic errors found, fix oracle prompt
- Post-filter is not the right place to "fix" oracle mistakes

## Success Metrics

**This decision is successful if:**
- ✅ No complaints about "good articles filtered incorrectly"
- ✅ Content type caps work as expected (corporate finance → low tier)
- ✅ Exceptions handled correctly (worker cooperatives → high tier)
- ✅ Post-filter tests pass without needing article metadata
- ✅ Oracle prompts consistent across models

**Red flags:**
- ❌ Frequent mis-classification (worker coops capped incorrectly)
- ❌ Oracle ignoring content caps from prompt
- ❌ Inconsistent behavior across oracle models

**If red flags appear:** Add metadata-based post-filter checks as backup enforcement

## Implementation Checklist

- [x] Decision documented (this ADR)
- [x] Post-filter stays simple (already implemented in filters/{filter_name}/v1/postfilter.py)
- [ ] Update uplifting oracle prompt with content_type_caps
- [ ] Update sustainability_tech_deployment oracle prompt (if has content caps)
- [ ] Test oracle enforcement with labeled dataset
- [ ] Document pre-filter as optional (not required)

## References

- Post-filter implementation: `filters/{filter_name}/v1/postfilter.py`
- Post-filter architecture: `docs/decisions/2025-11-13-post-filter-architecture.md`
- Article metadata schema: `docs/article-metadata-schema.md`
- Oracle prompts: `filters/*/v1/prompt-compressed.md`
- Filter configs: `filters/*/v1/config.yaml`

## Related Decisions

- **2025-11-13-remove-tier-classification-from-oracle.md** - Oracle outputs dimensional scores only
- **2025-11-13-post-filter-architecture.md** - Post-filter does tier classification
- This decision: **Content caps enforced in oracle, not post-filter**

**Principle:** Oracle does semantic understanding. Post-filter does arithmetic. Keep concerns separated.

## Version History

### v1.0 (2025-11-13)
- Initial decision
- Post-filter stays simple (arithmetic only)
- Content caps enforced in oracle prompt
- Pre-filter (metadata quality gate) is optional
- Rationale: Avoid false positives from noisy tags
