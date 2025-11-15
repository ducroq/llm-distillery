"""
Analyze all JSONL files to find articles that can be added to the corpus.
"""
import json
from pathlib import Path
from collections import Counter, defaultdict

data_dir = Path("C:/local_dev/llm-distillery/datasets/labeled/sustainability_tech_deployment")

print("="*80)
print("ANALYZING ALL JSONL FILES")
print("="*80)

# Step 1: Load current corpus IDs
print("\n1. Loading current corpus IDs from labeled_articles.jsonl...")
corpus_ids = set()
with open(data_dir / "labeled_articles.jsonl", 'r', encoding='utf-8') as f:
    for line in f:
        try:
            article = json.loads(line)
            if 'id' in article:
                corpus_ids.add(article['id'])
        except:
            pass

print(f"   Current corpus has {len(corpus_ids)} articles")

# Step 2: Scan all JSONL files
print("\n2. Scanning all JSONL files...")
all_files = sorted(data_dir.glob("*.jsonl"))

file_stats = {}
all_ids_found = defaultdict(list)  # id -> list of files containing it
new_articles_by_file = {}

for filepath in all_files:
    filename = filepath.name

    # Skip the main corpus file and metrics
    if filename in ['labeled_articles.jsonl', 'metrics.jsonl']:
        continue

    article_count = 0
    new_count = 0
    duplicate_count = 0
    error_count = 0
    ids_in_file = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    article = json.loads(line)

                    # Check if it's an article (has id field)
                    if 'id' not in article:
                        continue

                    article_id = article['id']
                    ids_in_file.append(article_id)
                    all_ids_found[article_id].append(filename)
                    article_count += 1

                    # Check if new
                    if article_id not in corpus_ids:
                        new_count += 1

                except json.JSONDecodeError:
                    error_count += 1

        # Count duplicates within this file
        id_counts = Counter(ids_in_file)
        duplicate_count = sum(1 for count in id_counts.values() if count > 1)

        file_stats[filename] = {
            'total': article_count,
            'new': new_count,
            'duplicates_within': duplicate_count,
            'errors': error_count
        }

    except Exception as e:
        print(f"   ERROR reading {filename}: {e}")

# Step 3: Analyze results
print("\n3. Analysis Results:")
print("-" * 80)

# Group files by type
batch_files = [f for f in file_stats.keys() if f.startswith('labeled_batch_')]
other_files = [f for f in file_stats.keys() if not f.startswith('labeled_batch_')]

# Summary
total_new = sum(stats['new'] for stats in file_stats.values())
total_articles_scanned = sum(stats['total'] for stats in file_stats.values())

print(f"\nFiles scanned: {len(file_stats)}")
print(f"  - Batch files: {len(batch_files)}")
print(f"  - Other files: {len(other_files)}")
print(f"\nTotal articles scanned: {total_articles_scanned}")
print(f"New articles found: {total_new}")
print(f"Already in corpus: {total_articles_scanned - total_new}")

# Check for articles in multiple files
print("\n4. Duplicate Detection Across Files:")
print("-" * 80)
multi_file_ids = {id: files for id, files in all_ids_found.items() if len(files) > 1}

if multi_file_ids:
    print(f"\nFound {len(multi_file_ids)} IDs appearing in multiple files")

    # Sample a few
    sample = list(multi_file_ids.items())[:5]
    for article_id, files in sample:
        print(f"\n  ID: {article_id}")
        print(f"  Appears in {len(files)} files: {files[:3]}{'...' if len(files) > 3 else ''}")
else:
    print("\nNo duplicates found across files")

# Show files with new articles
print("\n5. Files With New Articles:")
print("-" * 80)

files_with_new = [(f, stats) for f, stats in file_stats.items() if stats['new'] > 0]
files_with_new.sort(key=lambda x: x[1]['new'], reverse=True)

if files_with_new:
    print(f"\n{len(files_with_new)} files contain new articles:\n")

    # Show top files with most new articles
    for filename, stats in files_with_new[:10]:
        print(f"  {filename:50s} {stats['new']:5d} new articles (of {stats['total']:5d} total)")

    if len(files_with_new) > 10:
        remaining_new = sum(stats['new'] for _, stats in files_with_new[10:])
        print(f"  ... and {len(files_with_new) - 10} more files with {remaining_new} new articles")
else:
    print("\nNo new articles found in any file")

# Show other files (non-batch)
if other_files:
    print("\n6. Non-Batch Files:")
    print("-" * 80)
    for filename in other_files:
        stats = file_stats[filename]
        print(f"  {filename:50s} {stats['total']:5d} articles, {stats['new']:5d} new")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

if total_new > 0:
    print(f"\nYou can add {total_new} new articles to your corpus.")
    print("\nStrategy:")
    print("1. Merge all new articles from batch files")
    print("2. Use latest version if same ID appears in multiple files")
    print(f"3. Final corpus will have: {len(corpus_ids) + total_new} articles")
else:
    print("\nAll articles are already in your corpus - no action needed!")
