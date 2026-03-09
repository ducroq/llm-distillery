"""Sample additional v6-HIGH articles for v7 training enrichment.

Takes 2000 v6-HIGH articles not already in positives.jsonl, writes to enrichment_input.jsonl.
"""
import json
import glob
import random
import os

random.seed(123)

FILTERED_DIR = "/home/jeroen/local_dev/NexusMind/data/filtered/uplifting"
OUTPUT_DIR = "/home/jeroen/local_dev/NexusMind/data/training_data/uplifting_v7"
POSITIVES_PATH = os.path.join(OUTPUT_DIR, "positives.jsonl")
TARGET = 2000

# Load already-sampled content hashes
used_hashes = set()
with open(POSITIVES_PATH) as f:
    for line in f:
        art = json.loads(line)
        used_hashes.add(art.get("content_hash", ""))

# Also load negatives hashes
neg_path = os.path.join(OUTPUT_DIR, "negatives.jsonl")
if os.path.exists(neg_path):
    with open(neg_path) as f:
        for line in f:
            art = json.loads(line)
            used_hashes.add(art.get("content_hash", ""))

print("Already used hashes:", len(used_hashes))

# Collect all HIGH articles not already used
candidates = []
seen = set()
files = sorted(glob.glob(os.path.join(FILTERED_DIR, "high", "filtered_*.jsonl")))
for f in files:
    with open(f) as fh:
        for line in fh:
            try:
                art = json.loads(line)
                ch = art.get("content_hash", "")
                if ch and ch not in used_hashes and ch not in seen:
                    seen.add(ch)
                    candidates.append(art)
            except:
                pass

print("Available v6-HIGH candidates (not yet used):", len(candidates))

# Sample
sample = random.sample(candidates, min(TARGET, len(candidates)))
print("Sampled:", len(sample))

# Language distribution
langs = {}
for a in sample:
    lang = a.get("language", "unknown")
    langs[lang] = langs.get(lang, 0) + 1
top = sorted(langs.items(), key=lambda x: -x[1])[:8]
print("Languages:", top)

# Write slim output
output_path = os.path.join(OUTPUT_DIR, "enrichment_input.jsonl")
with open(output_path, "w") as f:
    for art in sample:
        ua = art.get("nexus_mind_attributes", {}).get("uplifting_analysis", {})
        slim = {
            "id": art.get("id", ""),
            "title": art.get("title", ""),
            "content": art.get("content", ""),
            "source": art.get("source", ""),
            "language": art.get("language", ""),
            "url": art.get("url", ""),
            "published_date": art.get("published_date", ""),
            "content_hash": art.get("content_hash", ""),
            "v6_tier": "high",
            "v6_weighted_average": ua.get("weighted_average", 0),
        }
        f.write(json.dumps(slim) + "\n")

print("Written to:", output_path)
