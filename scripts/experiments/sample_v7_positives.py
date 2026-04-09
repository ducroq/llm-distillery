"""Step 2: Sample MEDIUM/HIGH articles from NexusMind filtered output for v7 training."""
import json
import glob
import random
import os
import sys

random.seed(42)

FILTERED_DIR = "/home/jeroen/local_dev/NexusMind/data/filtered/uplifting"
OUTPUT_DIR = "/home/jeroen/local_dev/NexusMind/data/training_data/uplifting_v7"

# Phase 1: Collect and deduplicate
articles = {"high": [], "medium": []}
seen_hashes = set()

# Read flat JSONL files (NexusMind#144 removed tier subdirectories)
files = sorted(glob.glob(os.path.join(FILTERED_DIR, "filtered_*.jsonl")))
for f in files:
    with open(f) as fh:
        for line in fh:
            try:
                art = json.loads(line)
                ua = art.get("nexus_mind_attributes", {}).get("uplifting_analysis", {})
                tier = ua.get("tier", "low")
                if tier not in ("high", "medium"):
                    continue
                ch = art.get("content_hash", "")
                if ch and ch not in seen_hashes:
                    seen_hashes.add(ch)
                    articles[tier].append(art)
            except:
                pass

print("=== Deduplicated counts ===")
print("HIGH:", len(articles["high"]))
print("MEDIUM:", len(articles["medium"]))

# Phase 2: Content length distribution
for tier in ["high", "medium"]:
    lengths = [len(a.get("content", "")) for a in articles[tier]]
    lengths.sort()
    if lengths:
        print("{} content length: min={}, median={}, max={}".format(
            tier, lengths[0], lengths[len(lengths)//2], lengths[-1]))

# Phase 3: Language distribution
for tier in ["high", "medium"]:
    langs = {}
    for a in articles[tier]:
        lang = a.get("language", "unknown")
        langs[lang] = langs.get(lang, 0) + 1
    top = sorted(langs.items(), key=lambda x: -x[1])[:10]
    print("{} languages (top 10): {}".format(tier, top))

# Phase 4: v6 score distribution
for tier in ["high", "medium"]:
    scores = []
    for a in articles[tier]:
        ua = a.get("nexus_mind_attributes", {}).get("uplifting_analysis", {})
        wa = ua.get("weighted_average", 0)
        scores.append(wa)
    scores.sort()
    if scores:
        print("{} v6 scores: min={:.2f}, p25={:.2f}, median={:.2f}, p75={:.2f}, max={:.2f}".format(
            tier,
            scores[0],
            scores[len(scores)//4],
            scores[len(scores)//2],
            scores[3*len(scores)//4],
            scores[-1]))

if "--sample" not in sys.argv:
    print("\nDry run. Pass --sample to write output.")
    sys.exit(0)

# Phase 5: Sample
# Sample 500 HIGH (stratified across score range)
# Sample 1000 MEDIUM (stratified by v6 score quintiles)

# Stratify HIGH by v6 score
high_with_scores = []
for a in articles["high"]:
    ua = a.get("nexus_mind_attributes", {}).get("uplifting_analysis", {})
    wa = ua.get("weighted_average", 0)
    high_with_scores.append((wa, a))

high_with_scores.sort(key=lambda x: x[0])
n_high = len(high_with_scores)
high_quintiles = [
    high_with_scores[:n_high//5],
    high_with_scores[n_high//5:2*n_high//5],
    high_with_scores[2*n_high//5:3*n_high//5],
    high_with_scores[3*n_high//5:4*n_high//5],
    high_with_scores[4*n_high//5:],
]

per_high_q = 500 // 5  # 100 each
sample_high = []
for i, q in enumerate(high_quintiles):
    chosen = random.sample(q, min(per_high_q, len(q)))
    sample_high.extend([a for _, a in chosen])
    print("HIGH quintile {}: {} available, sampled {}".format(i+1, len(q), len(chosen)))

# Stratify MEDIUM by v6 score
medium_with_scores = []
for a in articles["medium"]:
    ua = a.get("nexus_mind_attributes", {}).get("uplifting_analysis", {})
    wa = ua.get("weighted_average", 0)
    medium_with_scores.append((wa, a))

medium_with_scores.sort(key=lambda x: x[0])
n = len(medium_with_scores)
quintiles = [
    medium_with_scores[:n//5],
    medium_with_scores[n//5:2*n//5],
    medium_with_scores[2*n//5:3*n//5],
    medium_with_scores[3*n//5:4*n//5],
    medium_with_scores[4*n//5:],
]

per_quintile = 1000 // 5  # 200 each
sample_medium = []
for i, q in enumerate(quintiles):
    chosen = random.sample(q, min(per_quintile, len(q)))
    sample_medium.extend([a for _, a in chosen])
    print("MEDIUM quintile {}: {} available, sampled {}".format(i+1, len(q), len(chosen)))

print("\nTotal sample: {} HIGH + {} MEDIUM = {}".format(
    len(sample_high), len(sample_medium), len(sample_high) + len(sample_medium)))

# Phase 6: Write output
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, "positives.jsonl")

# Slim down to oracle-relevant fields
def slim(art):
    return {
        "id": art.get("id", ""),
        "title": art.get("title", ""),
        "content": art.get("content", ""),
        "source": art.get("source", ""),
        "language": art.get("language", ""),
        "url": art.get("url", ""),
        "published_date": art.get("published_date", ""),
        "content_hash": art.get("content_hash", ""),
        "v6_tier": "high" if art in sample_high else "medium",
        "v6_weighted_average": art.get("nexus_mind_attributes", {}).get("uplifting_analysis", {}).get("weighted_average", 0),
    }

with open(output_path, "w") as f:
    for art in sample_high:
        f.write(json.dumps(slim(art)) + "\n")
    for art in sample_medium:
        f.write(json.dumps(slim(art)) + "\n")

print("Written to:", output_path)
