"""Phase 3 (Oracle Calibration) inline analyses for cultural_discovery v5.

Runs the gap-filling analyses that the prescribed `scripts/analyze_*` tooling would
do if it existed:
  - 3-way comparison on v3 data (mechanism + vendor)
  - Per-oracle score distribution (mean/std/modal-bin per dim)
  - Gatekeeper enforcement check (evidence_quality < 3.0 → overall capped at 3.0)
  - Tier distribution vs target (very_low/low/medium/high/very_high)
  - Dimension redundancy (pairwise Pearson + PCA on each oracle output)
  - Flag-firing rates per oracle (A-E max_score + F-K penalty)

Output: prints structured JSON-ish report to stdout for inclusion in the formal
calibration_report.md artifact.
"""

import json
import glob
import math
from collections import Counter

DIMS = ["discovery_novelty", "heritage_significance", "cross_cultural_connection",
        "human_resonance", "evidence_quality"]
WEIGHTS = [0.25, 0.20, 0.25, 0.15, 0.15]
PENALTY_FLAGS = {"historical_harm_reckoning", "commemoration_memorial",
                 "perpetrator_biography", "decline_loss", "launch_announcement"}
MAX_SCORE_FLAGS = {"political_conflict", "tourism_fluff", "celebrity_art",
                   "appropriation_debate", "speculation"}


def extract(val):
    if isinstance(val, dict):
        return val.get("score")
    return val


def wavg(dims):
    vals = [dims.get(d) for d in DIMS]
    if any(v is None for v in vals):
        return None
    return sum(float(v) * w for v, w in zip(vals, WEIGHTS))


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = (sum((x - mx) ** 2 for x in xs)) ** 0.5
    dy = (sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / (dx * dy) if dx and dy else None


def load_gemini_batched(glob_pattern):
    out = {}
    for fp in sorted(glob.glob(glob_pattern)):
        with open(fp, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                af = r.get("cultural_discovery_analysis", {})
                if not af:
                    continue
                dims = {d: extract(af.get(d)) for d in DIMS}
                if any(v is None for v in dims.values()):
                    continue
                out[r["id"]] = {
                    "dims": {d: float(dims[d]) for d in DIMS},
                    "ct": af.get("content_type", "unknown"),
                }
    return out


def load_deepseek(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "error" in r:
                continue
            dims = {d: extract(r["deepseek"].get(d)) for d in DIMS}
            if any(v is None for v in dims.values()):
                continue
            out[r["id"]] = {
                "dims": {d: float(dims[d]) for d in DIMS},
                "ct": r.get("deepseek_content_type", "unknown"),
            }
    return out


def stats_per_dim(records):
    s = {}
    n = len(records)
    if n == 0:
        return s
    for d in DIMS:
        vals = [r["dims"][d] for r in records.values()]
        m = sum(vals) / n
        sd = (sum((v - m) ** 2 for v in vals) / n) ** 0.5
        # modal bin (round to nearest 0.5)
        bin_counter = Counter(round(v * 2) / 2 for v in vals)
        modal = max(bin_counter.items(), key=lambda x: x[1])
        s[d] = {
            "mean": round(m, 2),
            "std": round(sd, 2),
            "min": min(vals),
            "max": max(vals),
            "modal_bin": modal[0],
            "modal_pct": round(100 * modal[1] / n, 1),
        }
    return s


def tier_distribution(records):
    """Score-bin stratification (prepare_data.py default since no tiers in v5 config)."""
    bins = Counter()
    for r in records.values():
        w = wavg(r["dims"])
        if w is None:
            continue
        if w >= 8.0:
            bins["very_high"] += 1
        elif w >= 6.0:
            bins["high"] += 1
        elif w >= 4.0:
            bins["medium"] += 1
        elif w >= 2.0:
            bins["low"] += 1
        else:
            bins["very_low"] += 1
    n = sum(bins.values())
    return {b: {"n": bins[b], "pct": round(100 * bins[b] / n, 1)} for b in
            ["very_low", "low", "medium", "high", "very_high"]}


def gatekeeper_check(records):
    """Evidence Quality gatekeeper: if eq < 3.0, overall (weighted_avg) should be ≤ 3.0.
    Counts violations: articles with eq < 3.0 AND wavg > 3.0."""
    violations = []
    eq_under_3 = 0
    for id_, r in records.items():
        if r["dims"]["evidence_quality"] < 3.0:
            eq_under_3 += 1
            w = wavg(r["dims"])
            if w > 3.0:
                violations.append({"id": id_, "eq": r["dims"]["evidence_quality"], "wavg": round(w, 2)})
    return {
        "articles_with_eq_under_3": eq_under_3,
        "violations": len(violations),
        "violation_rate_pct": round(100 * len(violations) / max(eq_under_3, 1), 1),
        "violation_sample": violations[:5],
    }


def dim_redundancy(records):
    """Pairwise Pearson + crude PCA-ish: redundancy ratio = fraction of dim pairs with |r| > 0.7."""
    n = len(records)
    if n < 2:
        return {}
    pairs = {}
    high_corr_pairs = []
    for i, d1 in enumerate(DIMS):
        for d2 in DIMS[i + 1:]:
            x = [r["dims"][d1] for r in records.values()]
            y = [r["dims"][d2] for r in records.values()]
            r = pearson(x, y)
            pairs[f"{d1} ↔ {d2}"] = round(r, 3) if r is not None else None
            if r is not None and abs(r) > 0.7:
                high_corr_pairs.append((d1, d2, round(r, 3)))
    # Crude variance explained: largest pairwise r^2 as upper bound on PC1 dominance
    rs = [p for p in pairs.values() if p is not None]
    max_abs_r = max(abs(p) for p in rs) if rs else 0
    return {
        "pairwise_pearson": pairs,
        "n_pairs_above_0.7": len(high_corr_pairs),
        "high_corr_pairs": high_corr_pairs,
        "max_abs_pairwise_r": round(max_abs_r, 3),
        "redundancy_ratio_pct": round(100 * len(high_corr_pairs) / 10, 1),  # 10 pairs total
    }


def flag_rates(records):
    counts = Counter(r["ct"] for r in records.values())
    fk = sum(counts[f] for f in PENALTY_FLAGS)
    ae = sum(counts[f] for f in MAX_SCORE_FLAGS)
    n = len(records)
    return {
        "F_K_rate_pct": round(100 * fk / n, 1),
        "A_E_rate_pct": round(100 * ae / n, 1),
        "by_flag": {f: counts.get(f, 0) for f in (PENALTY_FLAGS | MAX_SCORE_FLAGS) if counts.get(f, 0)},
        "cultural_discovery_count": counts.get("cultural_discovery", 0),
        "general_count": counts.get("general", 0),
        "n": n,
    }


def print_section(title, data):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def main():
    print("# Phase 3 Calibration Analysis — cultural_discovery v5")
    print("**Generated:** 2026-05-31  |  **Sample:** 522 articles\n")

    # Load all four runs
    gem_hard = load_gemini_batched("datasets/scored/cd_v5_hard_negatives/cultural_discovery/scored_batch_*.jsonl")
    gem_hard_al = load_gemini_batched("datasets/scored/active_learning_cd_v5_rescored/cultural_discovery/scored_batch_*.jsonl")
    gem_hard.update(gem_hard_al)
    gem_soft_v1 = load_gemini_batched("datasets/scored/cd_v5_softpenalty_rescored/cultural_discovery/scored_batch_*.jsonl")
    gem_soft_v3 = load_gemini_batched("datasets/scored/cd_v5_softpenalty_rescored_v3/cultural_discovery/scored_batch_*.jsonl")
    ds_soft_v1 = load_deepseek("datasets/scored/cd_v5_softpenalty_deepseek/results.jsonl")
    ds_soft_v3 = load_deepseek("datasets/scored/cd_v5_softpenalty_deepseek_v3/results.jsonl")

    print(f"## Records loaded")
    print(f"- gem_hard (v1): {len(gem_hard)}")
    print(f"- gem_soft_v1: {len(gem_soft_v1)}")
    print(f"- gem_soft_v3 (after A-E + F-K tightening): {len(gem_soft_v3)}")
    print(f"- ds_soft_v1: {len(ds_soft_v1)}")
    print(f"- ds_soft_v3: {len(ds_soft_v3)}")

    # 3-way on v3 data
    print_section("1. Flag-firing rates per oracle (v3 = tightened prompt)", {
        "gem_hard": flag_rates(gem_hard),
        "gem_soft_v1": flag_rates(gem_soft_v1),
        "gem_soft_v3": flag_rates(gem_soft_v3),
        "ds_soft_v1": flag_rates(ds_soft_v1),
        "ds_soft_v3": flag_rates(ds_soft_v3),
    })

    print_section("2. Per-dim distribution (gem_soft_v3)", stats_per_dim(gem_soft_v3))
    print_section("2b. Per-dim distribution (ds_soft_v3)", stats_per_dim(ds_soft_v3))

    print_section("3. Tier distribution vs target", {
        "target_per_workflow_doc": {"high": "10-20%", "medium-low": "60-70%", "very_high": ">=2%"},
        "gem_soft_v3": tier_distribution(gem_soft_v3),
        "ds_soft_v3": tier_distribution(ds_soft_v3),
    })

    print_section("4. Gatekeeper enforcement (Evidence Quality < 3.0 → overall ≤ 3.0)", {
        "gem_soft_v3": gatekeeper_check(gem_soft_v3),
        "ds_soft_v3": gatekeeper_check(ds_soft_v3),
    })

    print_section("5. Dimension redundancy (gem_soft_v3)", dim_redundancy(gem_soft_v3))
    print_section("5b. Dimension redundancy (ds_soft_v3)", dim_redundancy(ds_soft_v3))

    # v1 → v3 mechanism+tightening shift for Gemini (did the prompt edits work?)
    if len(gem_soft_v1) and len(gem_soft_v3):
        v1_fk = sum(1 for r in gem_soft_v1.values() if r["ct"] in PENALTY_FLAGS)
        v3_fk = sum(1 for r in gem_soft_v3.values() if r["ct"] in PENALTY_FLAGS)
        v1_ae = sum(1 for r in gem_soft_v1.values() if r["ct"] in MAX_SCORE_FLAGS)
        v3_ae = sum(1 for r in gem_soft_v3.values() if r["ct"] in MAX_SCORE_FLAGS)
        print_section("6. Did Gemini v3 prompt tightening work?", {
            "F_K_firing_rate": f"v1 {100*v1_fk/len(gem_soft_v1):.1f}% → v3 {100*v3_fk/len(gem_soft_v3):.1f}%",
            "A_E_firing_rate": f"v1 {100*v1_ae/len(gem_soft_v1):.1f}% → v3 {100*v3_ae/len(gem_soft_v3):.1f}%",
            "DS_v3_reference": f"F-K {100*sum(1 for r in ds_soft_v3.values() if r['ct'] in PENALTY_FLAGS)/len(ds_soft_v3):.1f}%",
        })


if __name__ == "__main__":
    main()
