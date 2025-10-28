"""
Test SEECE energy tech pre-filter effectiveness on dataset.

Shows how many articles pass/fail for SEECE applied research focus (TRL 4-7).
"""

import json
from collections import Counter
from ground_truth.batch_labeler import seece_pre_filter

def test_prefilter(dataset_path: str, sample_size: int = 10000):
    """Test SEECE pre-filter on sample of dataset."""

    print(f"Testing seece_pre_filter on {sample_size} articles")
    print("="*80)

    # Counters for analysis
    total = 0
    passed = 0
    failed_reasons = Counter()

    # Track detailed stats
    stats = {
        'too_short': 0,
        'low_quality': 0,
        'excluded_source': 0,
        'no_seece_signals': 0,
        'passed': 0
    }

    passed_articles = []
    failed_articles = []
    passed_sources = Counter()
    passed_keywords = Counter()
    dutch_eu_count = 0

    with open(dataset_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break

            try:
                article = json.loads(line.strip())
                total += 1

                # Get article metadata
                word_count = article.get('metadata', {}).get('word_count', 0)
                quality = article.get('metadata', {}).get('quality_score', 1.0)
                source = article.get('source', '')

                # Check why it might fail
                if source == 'github':
                    stats['excluded_source'] += 1
                    failed_reasons['github_excluded'] += 1
                    failed_articles.append((article.get('id'), 'github_excluded'))
                    continue

                # Check source-based word count thresholds
                failed_word_check = False
                if any(src in source for src in ['newsapi', 'reuters', 'bbc', 'npr', 'ap_news']):
                    if word_count < 20:
                        failed_word_check = True
                elif any(src in source for src in ['longform', 'new_yorker', 'atlantic', 'fast_company']):
                    if word_count < 200:
                        failed_word_check = True
                elif any(src in source for src in ['arxiv', 'nature', 'science', 'plos', 'frontiers']):
                    if word_count < 150:
                        failed_word_check = True
                else:
                    if word_count < 50:
                        failed_word_check = True

                if failed_word_check:
                    stats['too_short'] += 1
                    failed_reasons['word_count'] += 1
                    failed_articles.append((article.get('id'), f'word_count={word_count}'))
                    continue

                if quality < 0.7:
                    stats['low_quality'] += 1
                    failed_reasons['quality'] += 1
                    failed_articles.append((article.get('id'), f'quality={quality:.2f}'))
                    continue

                # Check if it passes the filter
                if seece_pre_filter(article):
                    stats['passed'] += 1
                    passed += 1
                    passed_articles.append(article.get('id'))
                    passed_sources[source] += 1

                    # Track which keywords matched
                    text = (article.get('title', '') + ' ' + article.get('content', ''))[:500].lower()
                    seece_keywords = [
                        'hydrogen', 'fuel cell', 'electrolysis', 'battery storage',
                        'grid storage', 'smart grid', 'v2g', 'electric vehicle', 'ev',
                        'charging', 'heat pump', 'building efficiency', 'hvac',
                        'industrial heat', 'solar integration', 'wind integration',
                        'inverter', 'converter', 'power electronics'
                    ]
                    for kw in seece_keywords:
                        if kw in text:
                            passed_keywords[kw] += 1

                    # Track Dutch/EU context
                    text_extended = (article.get('title', '') + ' ' + article.get('content', ''))[:1000].lower()
                    dutch_eu_markers = [
                        'netherlands', 'dutch', 'holland', 'nederland', 'amsterdam', 'rotterdam',
                        'eu ', 'european union', 'europe', 'germany', 'belgium', 'denmark'
                    ]
                    if any(marker in text_extended for marker in dutch_eu_markers):
                        dutch_eu_count += 1
                else:
                    stats['no_seece_signals'] += 1
                    failed_reasons['no_seece_signals'] += 1
                    failed_articles.append((article.get('id'), 'no_seece_signals'))

            except Exception as e:
                print(f"Error processing line {i}: {e}")
                continue

    # Print results
    print(f"\nRESULTS")
    print("="*80)
    print(f"Total articles tested: {total:,}")
    print(f"Passed filter: {passed:,} ({passed/total*100:.1f}%)")
    print(f"Failed filter: {total-passed:,} ({(total-passed)/total*100:.1f}%)")
    print(f"Dutch/EU context: {dutch_eu_count:,} ({dutch_eu_count/passed*100:.1f}% of passed)")

    print(f"\nFAILURE BREAKDOWN")
    print("-"*80)
    for reason, count in failed_reasons.most_common():
        pct = count / total * 100
        print(f"  {reason:30s}: {count:6,} ({pct:5.1f}%)")

    print(f"\nTOP SOURCES PASSING FILTER (Top 10)")
    print("-"*80)
    for source, count in passed_sources.most_common(10):
        pct = count / passed * 100
        print(f"  {source:40s}: {count:4} ({pct:5.1f}%)")

    print(f"\nTOP SEECE KEYWORDS MATCHED (Top 15)")
    print("-"*80)
    for keyword, count in passed_keywords.most_common(15):
        pct = count / passed * 100
        print(f"  {keyword:20s}: {count:4} ({pct:5.1f}%)")

    print(f"\nESTIMATED FULL DATASET (51,869 articles)")
    print("-"*80)
    pass_rate = passed / total
    estimated_pass = int(51869 * pass_rate)
    estimated_dutch_eu = int(estimated_pass * (dutch_eu_count / passed))
    print(f"Expected to pass: {estimated_pass:,} articles ({pass_rate*100:.1f}%)")
    print(f"Expected Dutch/EU: {estimated_dutch_eu:,} articles")
    print(f"Estimated cost (Gemini): ${estimated_pass*0.00015:.2f}")
    print(f"Estimated time: {estimated_pass*15/3600:.1f} hours (~{estimated_pass*15/3600/24:.1f} days)")

    print(f"\nEXAMPLE PASSED ARTICLES (first 10):")
    for article_id in passed_articles[:10]:
        print(f"  - {article_id}")

    print(f"\nEXAMPLE FAILED ARTICLES (first 10 with reasons):")
    for article_id, reason in failed_articles[:10]:
        print(f"  - {article_id}: {reason}")

    print("\n" + "="*80)
    print("Test complete!")
    print("\nNext steps:")
    print("1. Review pass rates - target 15-25% for SEECE")
    print("2. Check top sources - are they SEECE-relevant?")
    print("3. Review keywords - do they match SEECE priorities?")
    print("4. Test LLM on 5-10 passed articles to validate prompt quality")

if __name__ == '__main__':
    test_prefilter('datasets/raw/master_dataset.jsonl', sample_size=10000)
