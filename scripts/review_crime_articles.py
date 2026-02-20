"""
Review tool for identifying individual crime articles in uplifting training data.

Finds MEDIUM+ articles matching crime/sentencing keywords and outputs a TSV
for manual review. Mark column "cap" as Y for articles that should be capped to 3.0.

Usage:
    python scripts/review_crime_articles.py

Output: datasets/training/uplifting_v6/crime_review.tsv
"""

import json
from pathlib import Path

WEIGHTS = [0.25, 0.15, 0.10, 0.20, 0.20, 0.10]
DIM_NAMES = [
    "human_wellbeing_impact", "social_cohesion_impact", "justice_rights_impact",
    "evidence_level", "benefit_distribution", "change_durability",
]

CRIME_KEYWORDS = [
    # English
    'convicted', 'sentenced', 'sentencing', 'prison sentence', 'guilty verdict',
    'arrested', 'murder', 'manslaughter', 'criminal case', 'court ruled',
    'years in prison', 'jail time', 'conviction', 'indicted', 'felony',
    'life sentence', 'death penalty', 'guilty plea',
    # Dutch
    'veroordeeld', 'gevangenisstraf', 'gevangenis', 'arrestatie',
    'cel en tbs', 'levenslang', 'schuldig',
]

# Keywords that suggest systemic/reform (likely legitimate, not individual crime)
REFORM_KEYWORDS = [
    'reform', 'rehabilitation', 'restorative justice', 'policy change',
    'landmark ruling', 'class action', 'systemic', 'precedent',
    'legislation', 'prison reform', 'decriminalization',
    'hervorming', 'rehabilitatie', 'wetgeving',
    'hostage', 'gijzelaar', 'vrijgelaten', 'released', 'freed',
    'peace', 'vrede', 'ceasefire',
]


def weighted_avg(labels):
    return sum(s * w for s, w in zip(labels, WEIGHTS))


def find_keywords(text, keywords):
    text_lower = text.lower()
    return [kw for kw in keywords if kw in text_lower]


def main():
    data_dir = Path("datasets/training/uplifting_v6")
    output_path = data_dir / "crime_review.tsv"

    candidates = []

    for split in ["train", "val", "test"]:
        filepath = data_dir / f"{split}.jsonl"
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                w_avg = weighted_avg(ex["labels"])

                # Only MEDIUM+ articles (>= 4.0)
                if w_avg < 4.0:
                    continue

                title = ex.get("title", "")
                content = ex.get("content", "")
                text = f"{title} {content}"

                crime_matches = find_keywords(text, CRIME_KEYWORDS)
                if not crime_matches:
                    continue

                reform_matches = find_keywords(text, REFORM_KEYWORDS)

                candidates.append({
                    "split": split,
                    "id": ex.get("id", ""),
                    "title": title[:120],
                    "snippet": content[:200].replace("\n", " ").replace("\t", " "),
                    "weighted_avg": round(w_avg, 2),
                    "justice_score": ex["labels"][2],
                    "crime_keywords": ", ".join(crime_matches[:3]),
                    "reform_keywords": ", ".join(reform_matches[:3]) if reform_matches else "",
                    "likely_type": "REFORM" if reform_matches else "REVIEW",
                })

    # Sort: REVIEW first (needs attention), then by score descending
    candidates.sort(key=lambda x: (x["likely_type"] == "REFORM", -x["weighted_avg"]))

    # Write TSV
    headers = ["cap", "likely_type", "split", "weighted_avg", "justice_score",
               "title", "snippet", "crime_keywords", "reform_keywords", "id"]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\t".join(headers) + "\n")
        for c in candidates:
            row = [
                "",  # cap column - user fills in Y or N
                c["likely_type"],
                c["split"],
                str(c["weighted_avg"]),
                str(round(c["justice_score"], 1)),
                c["title"],
                c["snippet"],
                c["crime_keywords"],
                c["reform_keywords"],
                c["id"],
            ]
            f.write("\t".join(row) + "\n")

    print(f"Written {len(candidates)} candidates to {output_path}")
    print(f"  REVIEW (needs attention): {sum(1 for c in candidates if c['likely_type'] == 'REVIEW')}")
    print(f"  REFORM (likely legitimate): {sum(1 for c in candidates if c['likely_type'] == 'REFORM')}")
    print()
    print("Instructions:")
    print("  1. Open crime_review.tsv in Excel/Sheets")
    print("  2. For each row, put Y in 'cap' column if it's an individual crime case")
    print("  3. Leave blank or put N for legitimate articles (reform, systemic, etc.)")
    print("  4. Save and run: python scripts/apply_crime_caps.py")


if __name__ == "__main__":
    main()
