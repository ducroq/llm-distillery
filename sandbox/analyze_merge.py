"""
Analyze and prepare to merge labeled data files.
"""
import json
from pathlib import Path
from collections import Counter

data_dir = Path("C:/local_dev/llm-distillery/datasets/labeled/sustainability_tech_deployment")

# File paths
labeled_articles = data_dir / "labeled_articles.jsonl"
all_labels = data_dir / "all_labels_after_another_round_of_labeling.jsonl"

print("="*80)
print("ANALYZING DATA FILES")
print("="*80)

# Analyze labeled_articles.jsonl
print("\n1. labeled_articles.jsonl")
print("-" * 40)
labeled_ids = []
labeled_articles_data = []

with open(labeled_articles, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        try:
            article = json.loads(line)
            if 'id' not in article:
                print(f"  WARNING: Line {i+1} missing 'id' field")
                continue
            labeled_ids.append(article['id'])
            labeled_articles_data.append(article)
        except json.JSONDecodeError as e:
            print(f"  WARNING: Line {i+1} invalid JSON: {e}")

print(f"Total articles: {len(labeled_ids)}")
print(f"Unique IDs: {len(set(labeled_ids))}")
if len(labeled_ids) != len(set(labeled_ids)):
    duplicates = [id for id, count in Counter(labeled_ids).items() if count > 1]
    print(f"WARNING: WARNING: {len(duplicates)} duplicate IDs found in labeled_articles.jsonl")
    print(f"  First few: {duplicates[:5]}")

# Check if analyzed field exists
has_analysis = sum(1 for a in labeled_articles_data if 'sustainability_tech_deployment_analysis' in a)
print(f"Articles with analysis: {has_analysis}/{len(labeled_articles_data)}")

# Analyze all_labels_after_another_round_of_labeling.jsonl
print("\n2. all_labels_after_another_round_of_labeling.jsonl")
print("-" * 40)
all_labels_ids = []
all_labels_data = []

with open(all_labels, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        try:
            article = json.loads(line)
            if 'id' not in article:
                print(f"  WARNING: Warning: Line {i+1} missing 'id' field")
                continue
            all_labels_ids.append(article['id'])
            all_labels_data.append(article)
        except json.JSONDecodeError as e:
            print(f"  WARNING: Warning: Line {i+1} invalid JSON: {e}")

print(f"Total articles: {len(all_labels_ids)}")
print(f"Unique IDs: {len(set(all_labels_ids))}")
if len(all_labels_ids) != len(set(all_labels_ids)):
    duplicates = [id for id, count in Counter(all_labels_ids).items() if count > 1]
    print(f"WARNING: WARNING: {len(duplicates)} duplicate IDs found in all_labels file")
    print(f"  First few: {duplicates[:5]}")

has_analysis = sum(1 for a in all_labels_data if 'sustainability_tech_deployment_analysis' in a)
print(f"Articles with analysis: {has_analysis}/{len(all_labels_data)}")

# Check overlap
print("\n3. OVERLAP ANALYSIS")
print("-" * 40)
labeled_set = set(labeled_ids)
all_labels_set = set(all_labels_ids)

overlap = labeled_set & all_labels_set
only_in_labeled = labeled_set - all_labels_set
only_in_all_labels = all_labels_set - labeled_set

print(f"IDs in both files: {len(overlap)}")
print(f"IDs only in labeled_articles.jsonl: {len(only_in_labeled)}")
print(f"IDs only in all_labels_after_another_round_of_labeling.jsonl: {len(only_in_all_labels)}")
print(f"\nTotal unique IDs across both files: {len(labeled_set | all_labels_set)}")

# Compare timestamps if available
if overlap:
    print("\n4. TIMESTAMP COMPARISON (for overlapping IDs)")
    print("-" * 40)

    # Build lookup dicts
    labeled_dict = {a['id']: a for a in labeled_articles_data}
    all_labels_dict = {a['id']: a for a in all_labels_data}

    # Sample a few overlapping IDs
    sample_overlap = list(overlap)[:3]
    for article_id in sample_overlap:
        labeled_ts = labeled_dict[article_id].get('sustainability_tech_deployment_analysis', {}).get('analyzed_at', 'N/A')
        all_labels_ts = all_labels_dict[article_id].get('sustainability_tech_deployment_analysis', {}).get('analyzed_at', 'N/A')

        print(f"\nID: {article_id}")
        print(f"  labeled_articles.jsonl: {labeled_ts}")
        print(f"  all_labels...jsonl:    {all_labels_ts}")

        # Compare scores
        labeled_score = labeled_dict[article_id].get('sustainability_tech_deployment_analysis', {}).get('overall_score', 'N/A')
        all_labels_score = all_labels_dict[article_id].get('sustainability_tech_deployment_analysis', {}).get('overall_score', 'N/A')

        if labeled_score != all_labels_score:
            print(f"  WARNING: Scores differ: {labeled_score} vs {all_labels_score}")

print("\n" + "="*80)
print("MERGE STRATEGY RECOMMENDATION")
print("="*80)

if len(overlap) > 0:
    print("\nWARNING: Files have overlapping IDs!")
    print("\nRecommended approach:")
    print("1. Use 'all_labels_after_another_round_of_labeling.jsonl' version for duplicates")
    print("   (assumes it's newer/more recent)")
    print("2. Keep articles that only exist in labeled_articles.jsonl")
    print("3. Add new articles from all_labels_after_another_round_of_labeling.jsonl")
    print(f"\nThis will result in: {len(labeled_set | all_labels_set)} total articles")
else:
    print("\nOK: No overlapping IDs - simple concatenation possible")
    print(f"\nMerged file will have: {len(labeled_ids) + len(all_labels_ids)} articles")
