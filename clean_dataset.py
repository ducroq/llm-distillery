"""
Clean dataset by removing articles with invalid/outlier publication dates.

Strategy:
- Keep articles published between 2023-01-01 and 2026-12-31
- Remove NASA images with historical dates (1969-2019)
- Remove placeholder dates (9999)
- Remove the single 2022 outlier

Impact: ~9 articles removed (0.02% of dataset)
"""

import json
from datetime import datetime
from pathlib import Path

def clean_dataset(
    input_path: str,
    output_path: str,
    min_year: int = 2023,
    max_year: int = 2026
):
    """
    Clean dataset by filtering publication dates.

    Args:
        input_path: Source JSONL file
        output_path: Cleaned JSONL output
        min_year: Minimum year to keep (inclusive)
        max_year: Maximum year to keep (inclusive)
    """

    kept = 0
    removed = 0
    removed_articles = []

    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Cleaning dataset: {input_path}")
    print(f"Date range: {min_year}-01-01 to {max_year}-12-31")
    print("=" * 80)

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for i, line in enumerate(infile):
            try:
                article = json.loads(line.strip())

                # Check publication date
                keep = True
                if 'published_date' in article and article['published_date']:
                    try:
                        dt = datetime.fromisoformat(article['published_date'].replace('Z', '+00:00'))

                        # Filter by year range
                        if not (min_year <= dt.year <= max_year):
                            keep = False
                            removed_articles.append({
                                'id': article.get('id'),
                                'date': article['published_date'],
                                'year': dt.year,
                                'source': article.get('source'),
                                'title': article.get('title', '')[:80]
                            })
                    except:
                        # Invalid date format - keep article but flag
                        print(f"Warning: Invalid date format at line {i+1}: {article['published_date']}")

                if keep:
                    outfile.write(line)
                    kept += 1
                else:
                    removed += 1

                # Progress indicator
                if (i + 1) % 10000 == 0:
                    print(f"Processed {i+1:,} articles... (kept: {kept:,}, removed: {removed:,})")

            except Exception as e:
                print(f"Error processing line {i+1}: {e}")
                continue

    print("\n" + "=" * 80)
    print("CLEANING COMPLETE")
    print("=" * 80)
    print(f"Total articles processed: {kept + removed:,}")
    print(f"Articles kept: {kept:,} ({kept/(kept+removed)*100:.2f}%)")
    print(f"Articles removed: {removed:,} ({removed/(kept+removed)*100:.2f}%)")
    print(f"\nOutput saved to: {output_path}")

    if removed_articles:
        print("\n" + "=" * 80)
        print("REMOVED ARTICLES")
        print("=" * 80)
        for article in removed_articles:
            print(f"Year: {article['year']} | {article['id'][:40]}")
            print(f"  Date: {article['date'][:10]}")
            print(f"  Source: {article['source']}")
            print(f"  Title: {article['title']}")
            print()

    return kept, removed, removed_articles


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Clean dataset by filtering publication dates'
    )
    parser.add_argument(
        '--input',
        default='datasets/raw/master_dataset.jsonl',
        help='Input JSONL file (default: datasets/raw/master_dataset.jsonl)'
    )
    parser.add_argument(
        '--output',
        default='datasets/raw/master_dataset_cleaned.jsonl',
        help='Output JSONL file (default: datasets/raw/master_dataset_cleaned.jsonl)'
    )
    parser.add_argument(
        '--min-year',
        type=int,
        default=2023,
        help='Minimum year to keep (default: 2023)'
    )
    parser.add_argument(
        '--max-year',
        type=int,
        default=2026,
        help='Maximum year to keep (default: 2026)'
    )
    parser.add_argument(
        '--replace',
        action='store_true',
        help='Replace input file with cleaned version (DANGEROUS)'
    )

    args = parser.parse_args()

    # Safety check for --replace
    if args.replace:
        confirm = input(
            f"\nWARNING: This will REPLACE {args.input} with the cleaned version.\n"
            f"Type 'yes' to confirm: "
        )
        if confirm.lower() != 'yes':
            print("Aborted.")
            exit(1)

        # Clean to temp file, then replace
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
        temp_path = temp_file.name
        temp_file.close()

        kept, removed, _ = clean_dataset(args.input, temp_path, args.min_year, args.max_year)

        if removed > 0:
            import shutil
            shutil.move(temp_path, args.input)
            print(f"\nSUCCESS: Replaced {args.input} with cleaned version")
        else:
            print(f"\nSUCCESS: No articles removed - original file unchanged")
    else:
        kept, removed, _ = clean_dataset(args.input, args.output, args.min_year, args.max_year)
