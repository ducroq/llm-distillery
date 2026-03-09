"""Step 3: Sample random negatives from raw NexusMind articles for v7 training.

Joins raw articles with enrichment cache to get full text.
Excludes articles already in positives.jsonl.
"""
import json
import glob
import random
import os
import sys
import hashlib

random.seed(42)

RAW_DIR = "/home/jeroen/local_dev/NexusMind/data/raw"
CACHE_DIR = "/home/jeroen/local_dev/NexusMind/data/enrichment_cache"
OUTPUT_DIR = "/home/jeroen/local_dev/NexusMind/data/training_data/uplifting_v7"
POSITIVES_PATH = os.path.join(OUTPUT_DIR, "positives.jsonl")
TARGET_NEGATIVES = 3500
MIN_CONTENT_LENGTH = 200  # skip very short articles

# Phase 1: Load positive content_hashes to exclude
positive_hashes = set()
with open(POSITIVES_PATH) as f:
    for line in f:
        art = json.loads(line)
        positive_hashes.add(art.get("content_hash", ""))

print("Positive hashes to exclude:", len(positive_hashes))

# Phase 2: Build enrichment cache index (URL hash -> cache file path)
print("Building enrichment cache index...")
cache_files = os.listdir(CACHE_DIR)
print("Cache entries:", len(cache_files))

# Phase 3: Scan raw articles, join with cache, exclude positives
print("Scanning raw articles...")
raw_files = sorted(glob.glob(os.path.join(RAW_DIR, "content_items_*.jsonl")))

candidates = []
no_cache = 0
too_short = 0
is_positive = 0
total_scanned = 0

for f in raw_files:
    with open(f) as fh:
        for line in fh:
            total_scanned += 1
            try:
                art = json.loads(line)
            except:
                continue

            # Skip if already in positives
            ch = art.get("content_hash", "")
            if ch in positive_hashes:
                is_positive += 1
                continue

            # Try to get full text from enrichment cache
            url = art.get("url", "")
            if not url:
                no_cache += 1
                continue

            cache_key = hashlib.sha256(url.encode()).hexdigest()[:16]
            cache_path = os.path.join(CACHE_DIR, cache_key + ".json")

            if not os.path.exists(cache_path):
                no_cache += 1
                continue

            # Read cached content
            try:
                with open(cache_path) as cf:
                    cached = json.load(cf)
                full_text = cached.get("content", "")
            except:
                no_cache += 1
                continue

            if len(full_text) < MIN_CONTENT_LENGTH:
                too_short += 1
                continue

            # Good candidate
            candidates.append({
                "id": art.get("id", ""),
                "title": art.get("title", ""),
                "content": full_text,
                "source": art.get("source", ""),
                "language": art.get("language", ""),
                "url": url,
                "published_date": art.get("published_date", ""),
                "content_hash": ch,
                "v6_tier": "none",  # not in filtered output
                "v6_weighted_average": 0,
            })

    # Progress
    if total_scanned % 50000 == 0:
        print("  scanned {} articles, {} candidates so far".format(total_scanned, len(candidates)))

print("\n=== Scan results ===")
print("Total scanned:", total_scanned)
print("Already in positives:", is_positive)
print("No cache / no URL:", no_cache)
print("Too short (<{} chars):".format(MIN_CONTENT_LENGTH), too_short)
print("Valid candidates:", len(candidates))

# Phase 4: Language distribution of candidates
langs = {}
for a in candidates:
    lang = a.get("language", "unknown")
    langs[lang] = langs.get(lang, 0) + 1
top = sorted(langs.items(), key=lambda x: -x[1])[:10]
print("Candidate languages (top 10):", top)

# Phase 5: Content length distribution
lengths = [len(a["content"]) for a in candidates]
lengths.sort()
if lengths:
    print("Content length: min={}, median={}, max={}".format(
        lengths[0], lengths[len(lengths)//2], lengths[-1]))

if "--sample" not in sys.argv:
    print("\nDry run. Pass --sample to write output.")
    sys.exit(0)

# Phase 6: Sample with language-proportional stratification
# Sample proportional to language distribution in candidates
sample = random.sample(candidates, min(TARGET_NEGATIVES, len(candidates)))

print("\nSampled:", len(sample))

# Language distribution of sample
sample_langs = {}
for a in sample:
    lang = a.get("language", "unknown")
    sample_langs[lang] = sample_langs.get(lang, 0) + 1
top = sorted(sample_langs.items(), key=lambda x: -x[1])[:10]
print("Sample languages (top 10):", top)

# Phase 7: Write output
output_path = os.path.join(OUTPUT_DIR, "negatives.jsonl")
with open(output_path, "w") as f:
    for art in sample:
        f.write(json.dumps(art) + "\n")

print("Written to:", output_path)
