# Prefilter v2 Validation Report

**Date**: 2026-01-14
**Version**: 2.0 (multi-lingual update)
**Sample Size**: 500 articles
**Dataset**: fluxus_20260113.jsonl (276,822 articles)

## Summary

| Metric | Value |
|--------|-------|
| Pass rate | 12.8% |
| Blocked | 87.2% |
| Blocked by exclusion patterns | 3.0% |
| Blocked as not_sustainability_topic | 97.0% |

## Changes in v2.0

1. **Exclusion patterns** (English-only): Block AI/ML infrastructure, consumer electronics, programming, military, travel
2. **Multi-lingual inclusion keywords** (21 languages): EN, NL, DE, FR, ES, PT, IT, ZH, SV, DA, NO, FI, PL, CS, RU, UK, EL, HU, RO, TR, AR

## Exclusion Patterns Performance

| Category | Blocked | Sample Titles |
|----------|---------|---------------|
| ai_ml_infrastructure | 7 | "Detecting LLM-Generated Text...", "7 novidades de IA..." |
| military | 4 | "NTT Data JV for submarine cable", "Greenland's PM..." |
| consumer_electronics | 2 | "Non-Convex Portfolio..." (false positive), "Ghost of Yōtei..." |
| programming | 0 | - |
| travel | 0 | - |

### Issues Identified

1. **Submarine cable misclassified as military**: "NTT Data forms JV for intra-Asia submarine cable" blocked by `\b(submarine)\b` pattern. This is telecom infrastructure, not military.

2. **Consumer electronics false positive**: "Non-Convex Portfolio Optimization via Energy-Based Models" blocked because content mentions "GPU" for computation, not consumer hardware.

## Passed Articles Analysis (sample of 15)

| # | Source | Title | Sustainability Keyword | Assessment |
|---|--------|-------|----------------------|------------|
| 1 | swiss_nzz_wirtschaft | Nachts an der Tankstelle... | 'gas' (Tankstelle) | ⚠️ False positive - gas station work conditions |
| 2 | ai_engadget | reMarkable E Ink tablet... | 'battery' | ⚠️ False positive - tablet has battery |
| 3 | austrian_kurier | Auch im Iran geht es ums Öl | 'oil' | ✅ Valid - oil/geopolitics relevant to energy |
| 4 | global_news_spiegel | Ifo-Umfrage... | 'energy', 'klima' | ⚠️ Borderline - economic survey |
| 5 | austrian_kurier | Vorwürfe gegen Julio Iglesias | 'klima' | ❌ False positive - celebrity news |
| 6 | positive_news_better_india | Bird rescue team... | 'biodiversity' | ✅ Valid - environmental |
| 7 | portuguese_canaltech | Inspeção veicular... | 'emission' | ✅ Valid - vehicle emissions inspection |
| 8 | positive_news_better_india | Village without LPG... | 'energy', 'gas' | ✅ Valid - sustainable cooking |
| 9 | hungarian_telex | Iran news... | 'oil' | ⚠️ Borderline - geopolitics |
| 10 | positive_news_good_news | Google CEO telescope... | 'solar', 'energy' | ⚠️ Borderline - space tech |
| 11 | danish_borsen | Nina Smith... | German 'klima' | ❌ False positive - business news |
| 12 | romanian_adevarul | Coalition scandal... | 'climate' | ⚠️ Check - possibly political |
| 13 | german_handelsblatt | Klimaneutralität... | 'Klimaneutralität' | ✅ Valid - climate neutrality |
| 14 | dutch_rtl_nieuws | Barcelona De Jong... | Dutch keywords? | ❌ False positive - soccer |
| 15 | portuguese_canaltech | Galaxy Buds Core... | 'battery' | ❌ False positive - earbuds |

### Pass Assessment

| Category | Count | % of Passed |
|----------|-------|-------------|
| ✅ Valid sustainability | 5 | 33% |
| ⚠️ Borderline/check | 5 | 33% |
| ❌ Clear false positive | 5 | 33% |

## Known Limitations (Accepted Tradeoffs)

### 1. "Battery" keyword too broad
Articles about consumer electronics with batteries (tablets, earbuds) pass the sustainability check.

**Decision**: Accept. The oracle will correctly score these low. Tightening the keyword would risk missing battery storage and EV articles.

### 2. "Gas"/"Oil" in non-energy context
Gas stations, turmoil (contains 'oil'), Vegas (contains 'gas') can trigger matches.

**Decision**: Accept. Key energy transition keywords must remain. False positives scored low by oracle.

### 3. "Klima" in metadata/unrelated context
German articles may contain 'Klima' in boilerplate text, navigation, or tangentially.

**Decision**: Accept. Multi-lingual keywords are essential for FluxusSource's global coverage.

### 4. Submarine exclusion too aggressive
Telecom submarine cables blocked as military.

**Recommendation**: Add exception for "submarine cable" or "undersea cable".

## Recommendations

### For v2.1 (future)

1. **Add consumer electronics patterns**:
   - `r'\b(tablet (bundle|deal|sale)|e.?ink tablet)\b'`
   - `r'\b(earbuds|headphones) (deal|sale|discount)\b'`

2. **Refine submarine exclusion**:
   - Add negative lookahead: `r'\b(submarine)(?! cable)\b'`

3. **Add source-based filtering**:
   - Sources like `ai_engadget` are predominantly gadget reviews
   - Could pre-filter by known non-sustainability sources

### Not Recommended

1. **Requiring 2+ sustainability keywords** - Would cause false negatives on legitimate short articles
2. **Word-boundary matching** - Would break prefix matching (sustainab-, electrif-, recycl-)
3. **Removing common keywords** - "battery", "energy", "gas" are core to the domain

## False Negative Analysis

Initial check found 2 potential false negatives (0.2% rate):
- Tesla Cybertruck article - said "electric pickup" not "electric vehicle"
- Tesla water motor article - brand name not in keywords

**Root Cause**: Missing EV-related keywords:
- `'electric car'`, `'electric truck'`, `'electric pickup'`
- Brand names: `'tesla'`, `'rivian'`, `'lucid'`

**Fix Applied**: Added missing keywords to inclusion list.

**After Fix**: 0 false negatives detected (0.00% rate)

## Decision

✅ **APPROVED FOR BATCH SCORING**

Rationale:
1. 12.8% pass rate is appropriately selective
2. ~67% of passed articles are valid or borderline (worth oracle scoring)
3. ~33% false positives will be correctly scored low by oracle
4. Multi-lingual coverage working (German, Dutch, Portuguese articles passing)
5. Exclusion patterns catching obvious off-topic content (AI/ML papers, military)
6. Strategy aligns with "wide net prefilter, precision oracle" philosophy

## Next Steps

1. ✅ Sync prefilter to llm-distiller server
2. ⏳ Run batch scoring with updated oracle prompt
3. ⏳ Review oracle scores for false positive articles
4. ⏳ Iterate on prefilter if needed based on oracle results
