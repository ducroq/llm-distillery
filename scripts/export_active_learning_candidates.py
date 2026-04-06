"""Export active learning candidates from NexusMind production data on sadalsuud.

Reads scored articles from both tier subdirectories and flat files,
deduplicates against existing training data, and samples candidates
for oracle re-scoring.

Usage:
    # Run on sadalsuud (where production data lives)
    python3 scripts/export_active_learning_candidates.py \
        --filter cultural-discovery \
        --analysis-key cultural_discovery_analysis \
        --nexusmind-dir /home/jeroen/local_dev/NexusMind/data/filtered/cultural-discovery \
        --existing-training /home/jeroen/local_dev/llm-distillery/datasets/training/cultural-discovery_v4 \
        --sample 500 --seed 42 \
        --output /tmp/cd_v5_candidates.jsonl

    # Then scp back:
    # scp sadalsuud:/tmp/cd_v5_candidates.jsonl datasets/raw/cd_v5_active_learning_candidates.jsonl
"""

import argparse
import json
import os
import random
from pathlib import Path


def load_existing_ids(training_dir: str) -> set:
    """Load article IDs from existing training splits to avoid duplicates."""
    ids = set()
    for split in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        path = os.path.join(training_dir, split)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        a = json.loads(line)
                        ids.add(a.get("id", ""))
                    except json.JSONDecodeError:
                        pass
    return ids


def load_from_tier_dirs(base_dir: str, analysis_key: str, min_tier: str = "medium") -> list:
    """Load articles from tier subdirectories (old storage pattern)."""
    tiers = ["medium", "high"] if min_tier == "medium" else ["high"]
    articles = []
    for tier in tiers:
        tier_dir = os.path.join(base_dir, tier)
        if not os.path.isdir(tier_dir):
            continue
        for fname in os.listdir(tier_dir):
            if not fname.endswith(".jsonl"):
                continue
            with open(os.path.join(tier_dir, fname), encoding="utf-8") as f:
                for line in f:
                    try:
                        articles.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return articles


def load_from_flat_files(base_dir: str, analysis_key: str, min_wa: float = 4.0) -> list:
    """Load MEDIUM+ articles from flat files (new storage pattern)."""
    articles = []
    for fname in os.listdir(base_dir):
        fpath = os.path.join(base_dir, fname)
        if not fname.endswith(".jsonl") or not os.path.isfile(fpath):
            continue
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                try:
                    a = json.loads(line)
                    nma = a.get("nexus_mind_attributes", {})
                    analysis = nma.get(analysis_key, {})
                    wa = analysis.get("weighted_average", 0)
                    tier = analysis.get("tier", "low")
                    if tier in ("medium", "high") or wa >= min_wa:
                        articles.append(a)
                except json.JSONDecodeError:
                    pass
    return articles


def to_candidate_format(article: dict, analysis_key: str) -> dict:
    """Convert production article to batch scorer input format."""
    nma = article.get("nexus_mind_attributes", {})
    analysis = nma.get(analysis_key, {})
    return {
        "id": article.get("id", ""),
        "title": article.get("title", ""),
        "content": article.get("content", ""),
        "url": article.get("url", ""),
        "source": article.get("source", ""),
        "language": article.get("language", ""),
        "published_date": article.get("published_date", ""),
        "_student_weighted_avg": analysis.get("weighted_average", 0),
        "_student_tier": analysis.get("tier", ""),
    }


def main():
    parser = argparse.ArgumentParser(description="Export active learning candidates from production data")
    parser.add_argument("--filter", required=True, help="Filter name (e.g., cultural-discovery)")
    parser.add_argument("--analysis-key", required=True, help="NexusMind analysis key (e.g., cultural_discovery_analysis)")
    parser.add_argument("--nexusmind-dir", required=True, help="Path to NexusMind filtered data dir")
    parser.add_argument("--existing-training", required=True, help="Path to existing training data dir")
    parser.add_argument("--sample", type=int, default=0, help="Sample N candidates (0 = take all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    args = parser.parse_args()

    # Load existing training IDs
    existing_ids = load_existing_ids(args.existing_training)
    print(f"Existing training IDs: {len(existing_ids)}")

    # Load from both storage patterns
    tier_articles = load_from_tier_dirs(args.nexusmind_dir, args.analysis_key)
    flat_articles = load_from_flat_files(args.nexusmind_dir, args.analysis_key)
    print(f"Tier dir articles: {len(tier_articles)}")
    print(f"Flat file MEDIUM+ articles: {len(flat_articles)}")

    # Merge and deduplicate
    seen_ids = set()
    candidates = []
    for a in tier_articles + flat_articles:
        aid = a.get("id", "")
        if aid and aid not in existing_ids and aid not in seen_ids:
            seen_ids.add(aid)
            candidates.append(a)
    print(f"After dedup (excl. existing training): {len(candidates)}")

    # Sample if requested
    if args.sample > 0 and len(candidates) > args.sample:
        random.seed(args.seed)
        candidates = random.sample(candidates, args.sample)
        print(f"Sampled: {len(candidates)}")

    # Convert and write
    with open(args.output, "w", encoding="utf-8") as f:
        for a in candidates:
            slim = to_candidate_format(a, args.analysis_key)
            f.write(json.dumps(slim, ensure_ascii=False) + "\n")

    print(f"Written: {len(candidates)} candidates to {args.output}")


if __name__ == "__main__":
    main()
