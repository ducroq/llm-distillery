"""
Test pre-filter effectiveness on dataset.

Shows how many articles pass/fail each criterion.
"""

import json
from collections import Counter
from ground_truth.batch_labeler import uplifting_pre_filter

def test_prefilter(dataset_path: str, sample_size: int = 10000):
    """Test pre-filter on sample of dataset."""

    print(f"Testing uplifting_pre_filter on {sample_size} articles")
    print("="*80)

    # Counters for analysis
    total = 0
    passed = 0
    failed_reasons = Counter()

    # Track detailed stats
    stats = {
        'too_short': 0,
        'wrong_language': 0,
        'low_quality': 0,
        'excluded_source': 0,
        'too_negative': 0,
        'no_positive_signals': 0,
        'passed': 0
    }

    passed_articles = []
    failed_articles = []

    with open(dataset_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break

            try:
                article = json.loads(line.strip())
                total += 1

                # Manual step-through to understand failures
                word_count = article.get('metadata', {}).get('word_count', 0)
                language = article.get('language', '')
                quality = article.get('metadata', {}).get('quality_score', 1.0)
                source = article.get('source', '')
                emotions = article.get('metadata', {}).get('raw_emotions', {})

                # Track failure reasons (now source-based)
                # GitHub always excluded
                if source == 'github':
                    stats['excluded_source'] += 1
                    failed_reasons['excluded_source'] += 1
                    failed_articles.append((article.get('id'), f'source=github'))
                    continue

                # Check source-based word count thresholds
                failed_word_check = False
                if any(src in source for src in ['newsapi', 'reuters', 'bbc', 'npr', 'ap_news']):
                    if word_count < 20:
                        failed_word_check = True
                elif any(src in source for src in ['longform', 'new_yorker', 'atlantic', 'fast_company']):
                    if word_count < 200:
                        failed_word_check = True
                elif any(src in source for src in ['positive_news', 'good_news', 'upworthy', 'optimist']):
                    if word_count < 100:
                        failed_word_check = True
                elif any(src in source for src in ['arxiv', 'nature', 'science', 'plos', 'frontiers']):
                    if word_count < 150:
                        failed_word_check = True
                else:  # Default
                    if word_count < 50:
                        failed_word_check = True

                if failed_word_check:
                    stats['too_short'] += 1
                    failed_reasons['word_count'] += 1
                    failed_articles.append((article.get('id'), f'word_count={word_count}'))
                    continue

                # NO language filter anymore!
                # if language != 'en':
                #     stats['wrong_language'] += 1
                #     failed_reasons['language'] += 1
                #     failed_articles.append((article.get('id'), f'language={language}'))
                #     continue

                if quality < 0.7:
                    stats['low_quality'] += 1
                    failed_reasons['quality'] += 1
                    failed_articles.append((article.get('id'), f'quality={quality:.2f}'))
                    continue

                # GitHub check already done above

                # Check emotions
                joy = emotions.get('joy', 0)
                sadness = emotions.get('sadness', 0)
                fear = emotions.get('fear', 0)
                anger = emotions.get('anger', 0)
                negative_emotion = sadness + fear + anger

                has_positive_emotion = joy >= 0.15
                has_low_negative = negative_emotion < 0.05

                # Check keywords (multilingual)
                text = (article.get('title', '') + ' ' + article.get('content', ''))[:500].lower()

                uplifting_keywords = [
                    # English
                    'breakthrough', 'innovation', 'solution', 'success', 'achievement',
                    'hope', 'progress', 'inspiring', 'positive', 'transforms', 'improves',
                    'saves', 'helps', 'benefits', 'advance', 'discovered', 'cure', 'solved',
                    'revolutionary', 'pioneer',
                    # Dutch
                    'doorbraak', 'innovatie', 'oplossing', 'succes', 'prestatie',
                    'hoop', 'vooruitgang', 'inspirerend', 'positief', 'transformeert',
                    'verbetert', 'helpt', 'voordelen', 'ontdekt',
                    # Spanish
                    'avance', 'innovación', 'solución', 'éxito', 'logro',
                    'esperanza', 'progreso', 'inspirador', 'positivo', 'transforma',
                    'mejora', 'ayuda', 'beneficios', 'descubierto'
                ]

                negative_keywords = [
                    # English
                    'war', 'death', 'killed', 'disaster', 'catastrophe', 'attack',
                    'violence', 'shooting', 'bomb', 'crisis', 'collapse', 'scandal',
                    'conflict', 'terror',
                    # Dutch
                    'oorlog', 'dood', 'gedood', 'ramp', 'catastrofe', 'aanval',
                    'geweld', 'schietpartij', 'bom', 'crisis', 'instorting', 'schandaal',
                    # Spanish
                    'guerra', 'muerte', 'muerto', 'desastre', 'catástrofe', 'ataque',
                    'violencia', 'tiroteo', 'bomba', 'crisis', 'colapso', 'escándalo'
                ]

                has_uplifting_keywords = any(kw in text for kw in uplifting_keywords)
                has_negative_keywords = any(kw in text for kw in negative_keywords)

                if has_negative_keywords:
                    stats['too_negative'] += 1
                    failed_reasons['negative_keywords'] += 1
                    failed_articles.append((article.get('id'), 'negative_keywords'))
                    continue

                if not ((has_positive_emotion or has_low_negative) or has_uplifting_keywords):
                    stats['no_positive_signals'] += 1
                    failed_reasons['no_positive_signals'] += 1
                    failed_articles.append((article.get('id'), 'no_positive_signals'))
                    continue

                # Passed all filters!
                stats['passed'] += 1
                passed += 1
                passed_articles.append(article.get('id'))

            except Exception as e:
                print(f"Error processing line {i}: {e}")
                continue

    # Print results
    print(f"\nRESULTS")
    print("="*80)
    print(f"Total articles tested: {total:,}")
    print(f"Passed filter: {passed:,} ({passed/total*100:.1f}%)")
    print(f"Failed filter: {total-passed:,} ({(total-passed)/total*100:.1f}%)")

    print(f"\nFAILURE BREAKDOWN")
    print("-"*80)
    for reason, count in failed_reasons.most_common():
        pct = count / total * 100
        print(f"  {reason:30s}: {count:6,} ({pct:5.1f}%)")

    print(f"\nESTIMATED FULL DATASET (51,869 articles)")
    print("-"*80)
    pass_rate = passed / total
    estimated_pass = int(51869 * pass_rate)
    print(f"Expected to pass: {estimated_pass:,} articles ({pass_rate*100:.1f}%)")
    print(f"Estimated savings: ${(51869-estimated_pass)*0.00015:.2f}")
    print(f"Estimated time: {estimated_pass*15/3600:.1f} hours (~{estimated_pass*15/3600/24:.1f} days)")

    print(f"\nEXAMPLE PASSED ARTICLES (first 10):")
    for article_id in passed_articles[:10]:
        print(f"  - {article_id}")

    print(f"\nEXAMPLE FAILED ARTICLES (first 10 with reasons):")
    for article_id, reason in failed_articles[:10]:
        print(f"  - {article_id}: {reason}")

    print("\n" + "="*80)
    print("Test complete!")

if __name__ == '__main__':
    test_prefilter('datasets/raw/master_dataset.jsonl', sample_size=10000)
