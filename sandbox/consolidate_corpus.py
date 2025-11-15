"""
Consolidate the corpus by archiving duplicate/redundant files.
Keeps only the main corpus file.
"""
import shutil
from pathlib import Path
from datetime import datetime

data_dir = Path("C:/local_dev/llm-distillery/datasets/labeled/sustainability_tech_deployment")
archive_dir = data_dir / "archive"
batches_archive = archive_dir / "batches"

print("="*80)
print("CONSOLIDATING CORPUS - REMOVING DUPLICATES")
print("="*80)

# Step 1: Create archive directories
print("\n1. Creating archive directories...")
archive_dir.mkdir(exist_ok=True)
batches_archive.mkdir(exist_ok=True)
print(f"   Created: {archive_dir.relative_to(data_dir)}")
print(f"   Created: {batches_archive.relative_to(data_dir)}")

# Step 2: List files to keep vs archive
print("\n2. Identifying files...")

files_to_keep = [
    'labeled_articles.jsonl',  # Main corpus
    'metrics.jsonl',  # Metrics data (different type)
    '.labeled_ids.json',  # Tracking file
    'distillation.log',  # Log file
    'session_summary.json',  # Session tracking
    'README.md',  # Documentation
]

# Add backup files to keep list
backup_files = list(data_dir.glob("labeled_articles.jsonl.backup_*"))
for backup in backup_files:
    files_to_keep.append(backup.name)

print(f"\n   Files to KEEP:")
for filename in sorted(files_to_keep):
    if (data_dir / filename).exists():
        print(f"     - {filename}")

# Step 3: Move batch files
print("\n3. Moving batch files to archive...")
batch_files = sorted(data_dir.glob("labeled_batch_*.jsonl"))
print(f"   Found {len(batch_files)} batch files")

moved_batches = 0
for batch_file in batch_files:
    dest = batches_archive / batch_file.name
    try:
        shutil.move(str(batch_file), str(dest))
        moved_batches += 1
    except Exception as e:
        print(f"   ERROR moving {batch_file.name}: {e}")

print(f"   Moved {moved_batches} batch files to archive/batches/")

# Step 4: Move all_labels file
print("\n4. Moving all_labels file to archive...")
all_labels_file = data_dir / "all_labels_after_another_round_of_labeling.jsonl"

if all_labels_file.exists():
    dest = archive_dir / all_labels_file.name
    try:
        shutil.move(str(all_labels_file), str(dest))
        print(f"   Moved {all_labels_file.name} to archive/")
    except Exception as e:
        print(f"   ERROR: {e}")
else:
    print(f"   File not found (may already be moved)")

# Step 5: List remaining files
print("\n5. Remaining files in corpus directory:")
print("-" * 80)

remaining_jsonl = sorted(data_dir.glob("*.jsonl"))
remaining_other = sorted([f for f in data_dir.glob("*") if f.is_file() and f.suffix != '.jsonl'])

print(f"\n   JSONL files ({len(remaining_jsonl)}):")
for f in remaining_jsonl:
    size = f.stat().st_size / (1024*1024)  # MB
    print(f"     - {f.name:50s} {size:8.1f} MB")

print(f"\n   Other files ({len(remaining_other)}):")
for f in remaining_other[:10]:  # Show first 10
    print(f"     - {f.name}")

if len(remaining_other) > 10:
    print(f"     ... and {len(remaining_other) - 10} more files")

# Step 6: Verify corpus
print("\n6. Verifying main corpus...")
import json

article_count = 0
with open(data_dir / "labeled_articles.jsonl", 'r', encoding='utf-8') as f:
    for line in f:
        try:
            json.loads(line)
            article_count += 1
        except:
            pass

print(f"   labeled_articles.jsonl: {article_count} articles")

print("\n" + "="*80)
print("CONSOLIDATION COMPLETE")
print("="*80)

print(f"\nSummary:")
print(f"  - Main corpus: labeled_articles.jsonl ({article_count} articles)")
print(f"  - Archived: {moved_batches} batch files")
print(f"  - Archive location: archive/batches/")
print(f"  - Backup preserved: {len(backup_files)} backup file(s)")

print("\nYour corpus is now consolidated!")
print("All redundant files have been moved to archive/ directory.")
