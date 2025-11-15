"""
Merge labeled_articles.jsonl with all_labels_after_another_round_of_labeling.jsonl
"""
import json
from pathlib import Path
from datetime import datetime
import shutil

data_dir = Path("C:/local_dev/llm-distillery/datasets/labeled/sustainability_tech_deployment")

# File paths
labeled_articles = data_dir / "labeled_articles.jsonl"
all_labels = data_dir / "all_labels_after_another_round_of_labeling.jsonl"
backup_file = data_dir / f"labeled_articles.jsonl.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
output_file = data_dir / "labeled_articles_merged.jsonl"

print("="*80)
print("MERGING DATA FILES")
print("="*80)

# Step 1: Create backup
print(f"\n1. Creating backup...")
shutil.copy2(labeled_articles, backup_file)
print(f"   Backup created: {backup_file.name}")

# Step 2: Load and filter articles from both files
print(f"\n2. Loading articles from both files...")

all_articles = {}  # Using dict to deduplicate by ID

# Load from labeled_articles.jsonl
print(f"   Loading from labeled_articles.jsonl...")
loaded_from_labeled = 0
skipped_lines = 0

with open(labeled_articles, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        try:
            article = json.loads(line)

            # Skip if not an article (e.g., metrics lines)
            if 'id' not in article:
                skipped_lines += 1
                continue

            # Skip if missing required fields
            if 'title' not in article and 'content' not in article:
                skipped_lines += 1
                continue

            article_id = article['id']
            all_articles[article_id] = article
            loaded_from_labeled += 1

        except json.JSONDecodeError as e:
            print(f"   WARNING: Line {i+1} in labeled_articles.jsonl - invalid JSON")
            skipped_lines += 1

print(f"   Loaded {loaded_from_labeled} articles from labeled_articles.jsonl")
print(f"   Skipped {skipped_lines} non-article lines")

# Load from all_labels_after_another_round_of_labeling.jsonl
print(f"   Loading from all_labels_after_another_round_of_labeling.jsonl...")
loaded_from_all_labels = 0
duplicates = 0
skipped_lines_2 = 0

with open(all_labels, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        try:
            article = json.loads(line)

            # Skip if not an article
            if 'id' not in article:
                skipped_lines_2 += 1
                continue

            # Skip if missing required fields
            if 'title' not in article and 'content' not in article:
                skipped_lines_2 += 1
                continue

            article_id = article['id']

            # Check for duplicates (should be none based on analysis)
            if article_id in all_articles:
                duplicates += 1
                # Use the newer version (from all_labels)
                all_articles[article_id] = article
            else:
                all_articles[article_id] = article
                loaded_from_all_labels += 1

        except json.JSONDecodeError as e:
            print(f"   WARNING: Line {i+1} in all_labels file - invalid JSON")
            skipped_lines_2 += 1

print(f"   Loaded {loaded_from_all_labels} new articles from all_labels file")
if skipped_lines_2 > 0:
    print(f"   Skipped {skipped_lines_2} non-article lines")
if duplicates > 0:
    print(f"   Found {duplicates} duplicates (used newer version)")

# Step 3: Write merged file
print(f"\n3. Writing merged file...")
total_articles = len(all_articles)

with open(output_file, 'w', encoding='utf-8') as f:
    for article_id in sorted(all_articles.keys()):  # Sort by ID for consistency
        article = all_articles[article_id]
        f.write(json.dumps(article, ensure_ascii=False, separators=(',', ':')) + '\n')

print(f"   Written {total_articles} articles to {output_file.name}")

# Step 4: Replace original with merged version
print(f"\n4. Replacing original file...")
shutil.move(str(output_file), str(labeled_articles))
print(f"   Replaced labeled_articles.jsonl with merged version")

print("\n" + "="*80)
print("MERGE COMPLETE")
print("="*80)
print(f"\nOriginal backup: {backup_file.name}")
print(f"Articles in merged file: {total_articles}")
print(f"  - From labeled_articles.jsonl: {loaded_from_labeled}")
print(f"  - From all_labels file: {loaded_from_all_labels}")
if duplicates > 0:
    print(f"  - Duplicates resolved: {duplicates}")
