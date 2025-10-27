"""Test smart compression and source-based filtering."""

import json
from ground_truth.batch_labeler import GenericBatchLabeler, uplifting_pre_filter

# Test compression
def test_compression():
    print("="*80)
    print("TESTING SMART COMPRESSION")
    print("="*80)

    labeler = GenericBatchLabeler(
        prompt_path="prompts/uplifting.md",
        llm_provider="gemini",
        output_dir="datasets/test"
    )

    # Find a long article
    with open('datasets/raw/master_dataset.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            article = json.loads(line.strip())
            word_count = article.get('metadata', {}).get('word_count', 0)

            if word_count > 1000:
                print(f"\nFound long article:")
                print(f"  ID: {article.get('id')}")
                print(f"  Source: {article.get('source')}")
                print(f"  Word count: {word_count}")
                print(f"  Original length: {len(article.get('content', ''))} chars")

                # Test compression
                compressed = labeler._smart_compress_content(article.get('content', ''), max_words=800)
                compressed_words = len(compressed.split())

                print(f"  Compressed to: {compressed_words} words")
                print(f"  Compressed length: {len(compressed)} chars")
                print(f"  Reduction: {(1 - compressed_words/word_count)*100:.1f}%")

                # Show preview
                print(f"\n  Compressed preview (first 300 chars):")
                print(f"  {compressed[:300]}...")

                print(f"\n  Compressed end (last 200 chars):")
                print(f"  ...{compressed[-200:]}")
                break

# Test source-based filtering
def test_source_filtering():
    print("\n" + "="*80)
    print("TESTING SOURCE-BASED WORD COUNT THRESHOLDS")
    print("="*80)

    test_cases = [
        {'source': 'newsapi_general', 'word_count': 15, 'expected': False, 'reason': 'too short'},
        {'source': 'newsapi_general', 'word_count': 25, 'expected': True, 'reason': 'acceptable excerpt'},
        {'source': 'github', 'word_count': 100, 'expected': False, 'reason': 'github excluded'},
        {'source': 'longform_new_yorker_science', 'word_count': 150, 'expected': False, 'reason': 'longform needs 200+'},
        {'source': 'longform_new_yorker_science', 'word_count': 300, 'expected': True, 'reason': 'sufficient'},
        {'source': 'positive_news_upworthy', 'word_count': 80, 'expected': False, 'reason': 'needs 100+'},
        {'source': 'positive_news_upworthy', 'word_count': 150, 'expected': True, 'reason': 'sufficient'},
        {'source': 'science_arxiv_cs', 'word_count': 100, 'expected': False, 'reason': 'needs 150+'},
        {'source': 'science_arxiv_cs', 'word_count': 200, 'expected': True, 'reason': 'sufficient'},
        {'source': 'dutch_news_ad_algemeen', 'word_count': 40, 'expected': False, 'reason': 'below default 50'},
        {'source': 'dutch_news_ad_algemeen', 'word_count': 60, 'expected': True, 'reason': 'above default 50'},
    ]

    print(f"\n{'Source':<35} {'Words':<8} {'Pass?':<8} {'Reason'}")
    print("-"*80)

    for case in test_cases:
        # Create mock article
        mock_article = {
            'source': case['source'],
            'metadata': {
                'word_count': case['word_count'],
                'quality_score': 0.9,
                'raw_emotions': {
                    'joy': 0.2,
                    'sadness': 0.01,
                    'fear': 0.01,
                    'anger': 0.01
                }
            },
            'title': 'Test Article',
            'content': 'innovation breakthrough positive success ' * 20,
            'language': 'en'
        }

        result = uplifting_pre_filter(mock_article)
        status = "✓ PASS" if result == case['expected'] else "✗ FAIL"

        print(f"{case['source']:<35} {case['word_count']:<8} {str(result):<8} {case['reason']}")

        if result != case['expected']:
            print(f"  ERROR: Expected {case['expected']}, got {result}")

if __name__ == '__main__':
    test_compression()
    test_source_filtering()
    print("\n" + "="*80)
    print("Tests complete!")
