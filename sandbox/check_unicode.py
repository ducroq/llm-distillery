import json
import sys

def check_unicode_issues(filepath, max_articles=5):
    """Check for unicode issues in JSONL file."""
    issues_found = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_articles:
                break

            try:
                article = json.loads(line)
                article_id = article.get('id', f'article_{i}')

                for field in ['title', 'content', 'text']:
                    if field not in article:
                        continue

                    text = article[field]
                    if not isinstance(text, str):
                        continue

                    # Check for problematic characters
                    problematic = []

                    # Check for surrogates (U+D800 to U+DFFF)
                    for char in text:
                        code = ord(char)
                        if 0xD800 <= code <= 0xDFFF:
                            problematic.append(f'surrogate U+{code:04X}')

                    # Check for zero-width characters
                    zero_width = ['\u200b', '\u200c', '\u200d', '\ufeff', '\u2060', '\u180e']
                    for zw in zero_width:
                        if zw in text:
                            problematic.append(f'zero-width U+{ord(zw):04X}')

                    # Check for bidi marks
                    bidi = ['\u200e', '\u200f', '\u202a', '\u202b', '\u202c', '\u202d',
                            '\u202e', '\u2066', '\u2067', '\u2068', '\u2069']
                    for b in bidi:
                        if b in text:
                            problematic.append(f'bidi U+{ord(b):04X}')

                    # Check for HTML entities
                    if '&' in text and (';' in text):
                        import re
                        entities = re.findall(r'&[a-zA-Z]+;|&#\d+;', text)
                        if entities:
                            problematic.append(f'HTML entities: {entities[:3]}')

                    if problematic:
                        issues_found.append({
                            'article': article_id,
                            'field': field,
                            'issues': problematic,
                            'preview': text[:100]
                        })

            except Exception as e:
                print(f"Error processing line {i}: {e}")

    return issues_found

if __name__ == "__main__":
    files = [
        "C:/local_dev/llm-distillery/datasets/labeled/sustainability_tech_deployment/labeled_articles.jsonl",
        "C:/local_dev/llm-distillery/datasets/labeled/sustainability_tech_deployment/labeled_batch_001.jsonl",
        "C:/local_dev/llm-distillery/datasets/labeled/sustainability_tech_deployment/all_labels_after_another_round_of_labeling.jsonl"
    ]

    for filepath in files:
        print(f"\n{'='*80}")
        print(f"Checking: {filepath.split('/')[-1]}")
        print(f"{'='*80}")

        try:
            issues = check_unicode_issues(filepath, max_articles=10)

            if issues:
                print(f"\nFound {len(issues)} issues:")
                for issue in issues:
                    print(f"\n  Article: {issue['article']}")
                    print(f"  Field: {issue['field']}")
                    print(f"  Issues: {', '.join(issue['issues'])}")
                    print(f"  Preview: {issue['preview'][:80]}...")
            else:
                print("\nNo unicode issues found in first 10 articles.")
        except FileNotFoundError:
            print(f"File not found: {filepath}")
        except Exception as e:
            print(f"Error: {e}")
