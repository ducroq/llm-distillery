"""
Cross-lens overlap audit for ovr.news editorial lenses.

Computes pairwise Pearson correlations and MEDIUM+ article overlap between
all deployed filters. Used to validate lens distinctiveness and identify
dimension redundancy.

Key metrics:
- Pairwise Pearson r: how correlated are weighted averages between lenses
- MEDIUM+ overlap %: what fraction of one lens's articles also appear in another
- Exclusive %: articles that appear in exactly one lens

Usage:
    # Using training data (available offline)
    python scripts/analysis/cross_lens_audit.py --mode training

    # Using production NexusMind output (richer, requires scored data)
    python scripts/analysis/cross_lens_audit.py --mode production \
        --production-data datasets/production/nexusmind_output.jsonl

    # Include thriving v1 (after oracle scoring)
    python scripts/analysis/cross_lens_audit.py --mode training --include-thriving
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ground_truth import analysis_field_name

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

THRESHOLD_MEDIUM = 4.0
THRESHOLD_HIGH = 7.0

# Lens → filter mapping (ADR-012)
LENS_FILTER_MAP = {
    "Thriving": ("uplifting", "v6", "datasets/training/uplifting_v6"),
    "Belonging": ("belonging", "v1", "datasets/belonging/belonging_all_scored.jsonl"),
    "Recovery": ("nature_recovery", "v1", "datasets/training/nature_recovery_v1"),
    "Solutions": ("sustainability_technology", "v3", "datasets/training/sustainability_technology_v3"),
    "Discovery": ("cultural-discovery", "v4", "datasets/training/cultural-discovery_v4"),
}

# Dimension weights per filter (for weighted average computation)
FILTER_WEIGHTS = {
    "uplifting": {
        "human_wellbeing_impact": 0.25, "social_cohesion_impact": 0.15,
        "justice_rights_impact": 0.10, "evidence_level": 0.20,
        "benefit_distribution": 0.20, "change_durability": 0.10,
    },
    "thriving": {
        "human_wellbeing_impact": 0.40, "justice_rights_impact": 0.25,
        "evidence_level": 0.10, "benefit_distribution": 0.10,
        "change_durability": 0.15,
    },
    "belonging": {
        "intergenerational_bonds": 0.25, "community_fabric": 0.25,
        "reciprocal_care": 0.10, "rootedness": 0.15,
        "purpose_beyond_self": 0.15, "slow_presence": 0.10,
    },
    "nature_recovery": {
        "recovery_evidence": 0.25, "measurable_outcomes": 0.20,
        "ecological_significance": 0.20, "restoration_scale": 0.15,
        "human_agency": 0.10, "protection_durability": 0.10,
    },
    "sustainability_technology": {
        "technology_readiness_level": 0.15, "technical_performance": 0.15,
        "economic_competitiveness": 0.20, "life_cycle_environmental_impact": 0.30,
        "social_equity_impact": 0.10, "governance_systemic_impact": 0.10,
    },
    "cultural-discovery": {
        "discovery_novelty": 0.25, "heritage_significance": 0.20,
        "cross_cultural_connection": 0.25, "human_resonance": 0.15,
        "evidence_quality": 0.15,
    },
}

# Known dimension overlaps between lenses (for documentation)
KNOWN_OVERLAPS = [
    ("Thriving.human_wellbeing_impact", "Recovery.measurable_outcomes",
     "Both measure positive outcomes, but Recovery requires ecological context"),
    ("Thriving.social_cohesion_impact", "Belonging.community_fabric",
     "REMOVED in thriving v1 -- this was the primary overlap"),
    ("Thriving.change_durability", "Recovery.protection_durability",
     "Both measure lasting change, but in different domains (human vs ecological)"),
    ("Solutions.social_equity_impact", "Belonging.reciprocal_care",
     "Weak overlap: equity in tech vs organic mutual care"),
    ("Discovery.cross_cultural_connection", "Belonging.rootedness",
     "Opposing forces: cross-cultural bridging vs local rootedness"),
    ("Recovery.human_agency", "Solutions.technology_readiness_level",
     "Both involve human intervention, but recovery = stepping back, solutions = engineering forward"),
]


def load_training_scores(data_dir: str, filter_name: str) -> Dict[str, float]:
    """Load weighted average scores from training splits, keyed by URL."""
    weights = FILTER_WEIGHTS.get(filter_name, {})
    dim_names = list(weights.keys())
    weight_vals = [weights[d] for d in dim_names]

    scores = {}
    data_path = Path(data_dir)

    for split in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        split_path = data_path / split
        if not split_path.exists():
            continue
        with open(split_path, encoding="utf-8") as f:
            for line in f:
                a = json.loads(line)
                url = a.get("url", "")
                if not url:
                    continue
                labels = a.get("labels", [])
                if labels:
                    scores[url] = float(np.mean(labels))

    return scores


def load_belonging_scores(data_path: str) -> Dict[str, float]:
    """Load belonging scores from the oracle-scored JSONL."""
    weights = FILTER_WEIGHTS["belonging"]
    dim_names = list(weights.keys())
    weight_vals = [weights[d] for d in dim_names]

    scores = {}
    path = Path(data_path)

    if path.is_file():
        files = [path]
    else:
        files = list(path.glob("*.jsonl"))

    for f_path in files:
        with open(f_path, encoding="utf-8") as f:
            for line in f:
                a = json.loads(line)
                url = a.get("url", a.get("link", ""))
                if not url:
                    continue
                ba = a.get("belonging_analysis", {})
                if not ba:
                    continue
                vals = []
                for d in dim_names:
                    entry = ba.get(d, {})
                    if isinstance(entry, dict):
                        vals.append(float(entry.get("score", 0)))
                    elif isinstance(entry, (int, float)):
                        vals.append(float(entry))
                    else:
                        vals.append(0.0)
                scores[url] = float(np.dot(vals, weight_vals))

    return scores


def load_production_scores(data_path: str) -> Dict[str, Dict[str, float]]:
    """Load multi-filter scores from NexusMind production output."""
    all_scores = defaultdict(dict)  # url -> {lens_name: score}

    with open(data_path, encoding="utf-8") as f:
        for line in f:
            a = json.loads(line)
            url = a.get("url", "")
            if not url:
                continue

            attrs = a.get("nexus_mind_attributes", a)
            for lens, (filter_name, _, _) in LENS_FILTER_MAP.items():
                field = analysis_field_name(filter_name)
                if field in attrs:
                    analysis = attrs[field]
                    if isinstance(analysis, dict):
                        wa = analysis.get("weighted_average")
                        if wa is not None:
                            all_scores[url][lens] = float(wa)

    return all_scores


def pearson_r(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient."""
    if len(x) < 3:
        return float("nan")
    x_arr = np.array(x)
    y_arr = np.array(y)
    if np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def compute_pairwise_correlations(
    lens_scores: Dict[str, Dict[str, float]],
) -> Dict[Tuple[str, str], Tuple[float, int]]:
    """Compute pairwise Pearson r between all lens pairs.

    Returns: {(lens_a, lens_b): (r, n_common)} for all pairs.
    """
    lenses = sorted(lens_scores.keys())
    results = {}

    for i, a in enumerate(lenses):
        for b in lenses[i + 1 :]:
            common_urls = set(lens_scores[a].keys()) & set(lens_scores[b].keys())
            if len(common_urls) < 10:
                results[(a, b)] = (float("nan"), len(common_urls))
                continue

            x = [lens_scores[a][url] for url in common_urls]
            y = [lens_scores[b][url] for url in common_urls]
            r = pearson_r(x, y)
            results[(a, b)] = (r, len(common_urls))

    return results


def compute_overlap_matrix(
    lens_scores: Dict[str, Dict[str, float]],
    threshold: float = THRESHOLD_MEDIUM,
) -> Dict[Tuple[str, str], Tuple[int, float, float]]:
    """Compute MEDIUM+ overlap between all lens pairs.

    Returns: {(lens_a, lens_b): (n_overlap, pct_of_a, pct_of_b)}
    """
    lenses = sorted(lens_scores.keys())
    results = {}

    for i, a in enumerate(lenses):
        a_passing = {url for url, s in lens_scores[a].items() if s >= threshold}
        for b in lenses[i + 1 :]:
            b_passing = {url for url, s in lens_scores[b].items() if s >= threshold}
            overlap = a_passing & b_passing
            n = len(overlap)
            pct_a = 100 * n / len(a_passing) if a_passing else 0
            pct_b = 100 * n / len(b_passing) if b_passing else 0
            results[(a, b)] = (n, pct_a, pct_b)

    return results


def print_report(
    lens_scores: Dict[str, Dict[str, float]],
    correlations: Dict[Tuple[str, str], Tuple[float, int]],
    overlaps: Dict[Tuple[str, str], Tuple[int, float, float]],
):
    """Print the full cross-lens audit report."""
    lenses = sorted(lens_scores.keys())

    # 1. Per-lens summary
    print("=" * 80)
    print("LENS SUMMARY")
    print("=" * 80)
    print()
    fmt = "{:<12s}  {:>6s}  {:>6s}  {:>6s}  {:>8s}  {:>8s}"
    print(fmt.format("Lens", "Total", "MED+", "HIGH", "MED+ %", "HIGH %"))
    print("-" * 60)
    for lens in lenses:
        total = len(lens_scores[lens])
        med = sum(1 for s in lens_scores[lens].values() if s >= THRESHOLD_MEDIUM)
        high = sum(1 for s in lens_scores[lens].values() if s >= THRESHOLD_HIGH)
        med_pct = 100 * med / total if total else 0
        high_pct = 100 * high / total if total else 0
        print(f"{lens:<12s}  {total:>6d}  {med:>6d}  {high:>6d}  {med_pct:>7.1f}%  {high_pct:>7.1f}%")

    # 2. Pairwise correlations
    print()
    print("=" * 80)
    print("PAIRWISE PEARSON CORRELATIONS (weighted averages)")
    print("=" * 80)
    print()
    sorted_corr = sorted(correlations.items(), key=lambda x: -abs(x[1][0]) if not np.isnan(x[1][0]) else 0)
    fmt = "{:<30s}  {:>8s}  {:>8s}  {:>6s}"
    print(fmt.format("Lens Pair", "r", "|r|", "N"))
    print("-" * 60)
    for (a, b), (r, n) in sorted_corr:
        flag = " *** HIGH" if abs(r) > 0.5 else " ** moderate" if abs(r) > 0.3 else ""
        r_str = f"{r:>8.3f}" if not np.isnan(r) else "     N/A"
        abs_str = f"{abs(r):>8.3f}" if not np.isnan(r) else "     N/A"
        print(f"{a + ' vs ' + b:<30s}  {r_str}  {abs_str}  {n:>6d}{flag}")

    # 3. MEDIUM+ overlap
    print()
    print("=" * 80)
    print(f"MEDIUM+ OVERLAP (threshold >= {THRESHOLD_MEDIUM})")
    print("=" * 80)
    print()
    sorted_overlap = sorted(overlaps.items(), key=lambda x: -x[1][0])
    fmt = "{:<30s}  {:>6s}  {:>8s}  {:>8s}  {:>6s}"
    print(fmt.format("Lens Pair", "N", "% of A", "% of B", "Flag"))
    print("-" * 70)
    for (a, b), (n, pct_a, pct_b) in sorted_overlap:
        flag = "WARN" if max(pct_a, pct_b) > 50 else "note" if max(pct_a, pct_b) > 25 else ""
        print(f"{a + ' vs ' + b:<30s}  {n:>6d}  {pct_a:>7.1f}%  {pct_b:>7.1f}%  {flag}")

    # 4. Exclusivity
    print()
    print("=" * 80)
    print("EXCLUSIVITY (articles in exactly one lens at MEDIUM+)")
    print("=" * 80)
    print()

    all_urls = set()
    for scores in lens_scores.values():
        all_urls |= set(scores.keys())

    for lens in lenses:
        passing = {url for url, s in lens_scores[lens].items() if s >= THRESHOLD_MEDIUM}
        exclusive = set()
        for url in passing:
            in_others = False
            for other_lens in lenses:
                if other_lens == lens:
                    continue
                if url in lens_scores[other_lens] and lens_scores[other_lens][url] >= THRESHOLD_MEDIUM:
                    in_others = True
                    break
            if not in_others:
                exclusive.add(url)
        pct = 100 * len(exclusive) / len(passing) if passing else 0
        print(f"  {lens:<12s}: {len(exclusive):>5d}/{len(passing):>5d} exclusive ({pct:.1f}%)")

    # 5. Known dimension overlaps
    print()
    print("=" * 80)
    print("KNOWN DIMENSION OVERLAPS (from lens design)")
    print("=" * 80)
    print()
    for dim_a, dim_b, note in KNOWN_OVERLAPS:
        print(f"  {dim_a}")
        print(f"    vs {dim_b}")
        print(f"    -> {note}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Cross-lens overlap audit for ovr.news editorial lenses"
    )
    parser.add_argument(
        "--mode", choices=["training", "production"], default="training",
        help="Data source: 'training' uses training splits, 'production' uses NexusMind output"
    )
    parser.add_argument(
        "--production-data", type=Path,
        help="Path to NexusMind production JSONL (required for --mode production)"
    )
    parser.add_argument(
        "--include-thriving", action="store_true",
        help="Include thriving v1 (requires training data at datasets/training/thriving_v1/)"
    )
    parser.add_argument(
        "--threshold", type=float, default=THRESHOLD_MEDIUM,
        help=f"Overlap threshold (default: {THRESHOLD_MEDIUM})"
    )

    args = parser.parse_args()

    if args.mode == "production":
        if not args.production_data:
            parser.error("--production-data required for --mode production")
        logger.info(f"Loading production data from {args.production_data}")
        multi_scores = load_production_scores(str(args.production_data))
        lens_scores = defaultdict(dict)
        for url, scores in multi_scores.items():
            for lens, score in scores.items():
                lens_scores[lens][url] = score
    else:
        lens_scores = {}
        for lens, (filter_name, version, data_path) in LENS_FILTER_MAP.items():
            if lens == "Belonging":
                scores = load_belonging_scores(data_path)
            else:
                scores = load_training_scores(data_path, filter_name)
            if scores:
                lens_scores[lens] = scores
                logger.info(f"Loaded {lens}: {len(scores)} articles")
            else:
                logger.warning(f"No data found for {lens} at {data_path}")

        if args.include_thriving:
            thriving_scores = load_training_scores(
                "datasets/training/thriving_v1", "thriving"
            )
            if thriving_scores:
                lens_scores["Thriving_v1"] = thriving_scores
                logger.info(f"Loaded Thriving v1: {len(thriving_scores)} articles")
            else:
                logger.warning("No thriving v1 training data found")

    if len(lens_scores) < 2:
        logger.error("Need at least 2 lenses with data")
        sys.exit(1)

    correlations = compute_pairwise_correlations(lens_scores)
    overlaps = compute_overlap_matrix(lens_scores, threshold=args.threshold)
    print_report(lens_scores, correlations, overlaps)


if __name__ == "__main__":
    main()
