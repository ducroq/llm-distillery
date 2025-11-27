"""Consolidate calibration data from different formats."""
import json
from pathlib import Path
import sys

def convert_to_oracle_format(article):
    """Convert batch_scorer format to oracle format."""
    if 'sustainability_technology_analysis' in article:
        analysis = article['sustainability_technology_analysis']
        article['oracle_scores'] = {
            dim: analysis[dim]['score']
            for dim in analysis
            if isinstance(analysis[dim], dict) and 'score' in analysis[dim]
        }
        article['oracle_reasoning'] = {
            dim: analysis[dim]['evidence']
            for dim in analysis
            if isinstance(analysis[dim], dict) and 'evidence' in analysis[dim]
        }
        # Keep timestamp if available
        if 'analyzed_at' in analysis:
            article['oracle_timestamp'] = analysis['analyzed_at']
    return article

def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')

    print("Consolidating calibration data...")

    scored_by_url = {}

    # Load original merged articles
    if (folder / 'articles_scored.jsonl').exists():
        with open(folder / 'articles_scored.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    article = json.loads(line)
                    article = convert_to_oracle_format(article)
                    if article.get('url') and 'oracle_scores' in article:
                        scored_by_url[article['url']] = article
        print(f"  Loaded {len(scored_by_url)} from articles_scored.jsonl")

    # Load batch scorer output
    if (folder / 'new_scored.jsonl').exists():
        with open(folder / 'new_scored.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    article = json.loads(line)
                    article = convert_to_oracle_format(article)
                    if article.get('url') and 'oracle_scores' in article:
                        scored_by_url[article['url']] = article
        print(f"  Total unique after batch scorer: {len(scored_by_url)}")

    # Save consolidated
    output_path = folder / 'articles_scored.jsonl'
    with open(output_path, 'w', encoding='utf-8') as f:
        for article in scored_by_url.values():
            f.write(json.dumps(article, ensure_ascii=False) + '\n')

    print(f"\nConsolidated {len(scored_by_url)} unique scored articles")
    print(f"Saved to: {output_path}")

    # Cleanup
    if (folder / 'new_scored.jsonl').exists():
        (folder / 'new_scored.jsonl').unlink()
        print("Cleaned up temporary files")

if __name__ == '__main__':
    main()
