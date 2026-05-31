"""Compute 4-oracle consensus + per-oracle alignment metrics for cultural_discovery v5.

Inputs (joined by article ID from canonical 522-article list):
  - Gemini Flash 2.5 v3 (soft-penalty, A-E + F-K tightened):
      datasets/scored/cd_v5_softpenalty_rescored_v3/cultural_discovery/scored_batch_*.jsonl
  - DeepSeek V4 Flash v3:
      datasets/scored/cd_v5_softpenalty_deepseek_v3/results.jsonl
  - Qwen3:14b (Ollama, gpu-server):
      datasets/scored/cd_v5_ollama_qwen3_14b/results.jsonl
  - Phi4:14b (Ollama, gpu-server):
      datasets/scored/cd_v5_ollama_phi4_14b/results.jsonl

Outputs:
  - datasets/scored/cd_v5_consensus.jsonl  — per-article consensus + dissenters
  - datasets/scored/cd_v5_hard_cases.jsonl — articles where <3 of 4 oracles agree on content_type
  - datasets/scored/cd_v5_consensus_summary.json — per-oracle alignment metrics

Per-oracle alignment metric: rank correlation per dim vs consensus median, plus
content_type agreement rate (excluding hard cases). Used to pick the production oracle.

This is independent of the agent-judge layer (which runs on hard_cases.jsonl separately).
"""

import json
import glob
from collections import Counter, defaultdict
from pathlib import Path

DIMS = ["discovery_novelty", "heritage_significance", "cross_cultural_connection",
        "human_resonance", "evidence_quality"]
WEIGHTS = [0.25, 0.20, 0.25, 0.15, 0.15]


def extract(val):
    if val is None:
        return None
    if isinstance(val, dict):
        s = val.get("score")
        return float(s) if s is not None else None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            return None
    return None


def wavg(dims):
    vals = [dims.get(d) for d in DIMS]
    if any(v is None for v in vals):
        return None
    return sum(v * w for v, w in zip(vals, WEIGHTS))


def median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return None
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


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


def spearman(xs, ys):
    n = len(xs)
    if n < 2:
        return None

    def ranks(v):
        idx = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[idx[j + 1]] == v[idx[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[idx[k]] = avg
            i = j + 1
        return r

    return pearson(ranks(xs), ranks(ys))


def load_gemini():
    out = {}
    for fp in sorted(glob.glob("datasets/scored/cd_v5_softpenalty_rescored_v3/cultural_discovery/scored_batch_*.jsonl")):
        with open(fp, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                af = r.get("cultural_discovery_analysis", {})
                if not af:
                    continue
                dims = {d: extract(af.get(d)) for d in DIMS}
                if any(v is None for v in dims.values()):
                    continue
                out[r["id"]] = {"dims": dims, "ct": af.get("content_type", "unknown")}
    return out


def load_deepseek():
    out = {}
    with open("datasets/scored/cd_v5_softpenalty_deepseek_v3/results.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "error" in r:
                continue
            dims = {d: extract(r["deepseek"].get(d)) for d in DIMS}
            if any(v is None for v in dims.values()):
                continue
            out[r["id"]] = {"dims": dims, "ct": r.get("deepseek_content_type", "unknown")}
    return out


def load_ollama(path):
    out = {}
    p = Path(path)
    if not p.exists():
        print(f"WARN: {path} not found — skipping this oracle")
        return out
    with open(p, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "error" in r:
                continue
            dims = {d: extract(r.get("dims", {}).get(d)) for d in DIMS}
            if any(v is None for v in dims.values()):
                continue
            out[r["id"]] = {"dims": dims, "ct": r.get("content_type", "unknown")}
    return out


def main():
    oracles = {
        "gemini_v3": load_gemini(),
        "deepseek_v3": load_deepseek(),
        "qwen3_14b": load_ollama("datasets/scored/cd_v5_ollama_qwen3_14b/results.jsonl"),
        "phi4_14b": load_ollama("datasets/scored/cd_v5_ollama_phi4_14b/results.jsonl"),
    }
    counts = {k: len(v) for k, v in oracles.items()}
    print(f"Loaded oracles: {counts}")

    # Find articles present in ALL oracles (canonical set)
    common_ids = set.intersection(*[set(v.keys()) for v in oracles.values() if v])
    print(f"Common to all oracles: {len(common_ids)}")
    if not common_ids:
        print("ERROR: No articles common to all oracles — check upstream runs")
        return

    consensus_path = Path("datasets/scored/cd_v5_consensus.jsonl")
    hard_path = Path("datasets/scored/cd_v5_hard_cases.jsonl")
    summary_path = Path("datasets/scored/cd_v5_consensus_summary.json")
    consensus_path.parent.mkdir(parents=True, exist_ok=True)

    consensus_records = []
    hard_cases = []
    per_oracle_dim_vals = {k: {d: [] for d in DIMS} for k in oracles}
    consensus_dim_vals = {d: [] for d in DIMS}
    ct_agreement = {k: {"agree": 0, "total": 0} for k in oracles}

    for id_ in common_ids:
        per_oracle = {k: oracles[k][id_] for k in oracles}

        # content_type majority
        ct_counter = Counter(per_oracle[k]["ct"] for k in oracles)
        top_ct, top_count = ct_counter.most_common(1)[0]
        ct_consensus = top_ct if top_count >= 3 else None
        is_hard = ct_consensus is None

        # per-dim median
        dim_consensus = {}
        for d in DIMS:
            vals = [per_oracle[k]["dims"][d] for k in oracles]
            dim_consensus[d] = median(vals)
            consensus_dim_vals[d].append(dim_consensus[d])
            for k in oracles:
                per_oracle_dim_vals[k][d].append(per_oracle[k]["dims"][d])

        # ct agreement bookkeeping (only on consensus-clear articles)
        if ct_consensus is not None:
            for k in oracles:
                ct_agreement[k]["total"] += 1
                if per_oracle[k]["ct"] == ct_consensus:
                    ct_agreement[k]["agree"] += 1

        record = {
            "id": id_,
            "consensus_ct": ct_consensus,
            "ct_split": dict(ct_counter),
            "consensus_dims": dim_consensus,
            "consensus_wavg": wavg(dim_consensus),
            "per_oracle": {k: {"ct": per_oracle[k]["ct"], "dims": per_oracle[k]["dims"]} for k in oracles},
            "is_hard_case": is_hard,
        }
        consensus_records.append(record)
        if is_hard:
            hard_cases.append(record)

    # Per-oracle alignment metrics
    summary = {
        "total_articles": len(common_ids),
        "hard_cases": len(hard_cases),
        "consensus_clear": len(common_ids) - len(hard_cases),
        "ct_distribution_in_consensus": dict(Counter(r["consensus_ct"] for r in consensus_records if r["consensus_ct"])),
        "per_oracle_alignment": {},
    }

    for k in oracles:
        per_dim_corr = {}
        for d in DIMS:
            r_p = pearson(per_oracle_dim_vals[k][d], consensus_dim_vals[d])
            r_s = spearman(per_oracle_dim_vals[k][d], consensus_dim_vals[d])
            mae = sum(abs(a - b) for a, b in zip(per_oracle_dim_vals[k][d], consensus_dim_vals[d])) / len(per_oracle_dim_vals[k][d])
            per_dim_corr[d] = {
                "pearson": round(r_p, 3) if r_p is not None else None,
                "spearman": round(r_s, 3) if r_s is not None else None,
                "mae": round(mae, 2),
            }
        valid_pearsons = [per_dim_corr[d]["pearson"] for d in DIMS if per_dim_corr[d]["pearson"] is not None]
        avg_pearson = sum(valid_pearsons) / len(valid_pearsons) if valid_pearsons else None
        ct_agree = ct_agreement[k]
        ct_agree_rate = round(100 * ct_agree["agree"] / max(ct_agree["total"], 1), 1)
        summary["per_oracle_alignment"][k] = {
            "per_dim": per_dim_corr,
            "avg_pearson_with_consensus": round(avg_pearson, 3) if avg_pearson is not None else None,
            "ct_agreement_rate_pct": ct_agree_rate,
            "ct_agreement_count": f"{ct_agree['agree']}/{ct_agree['total']}",
        }

    # Write outputs
    with open(consensus_path, "w", encoding="utf-8") as f:
        for r in consensus_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(hard_path, "w", encoding="utf-8") as f:
        for r in hard_cases:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nWrote:")
    print(f"  {consensus_path}  ({len(consensus_records)} records)")
    print(f"  {hard_path}       ({len(hard_cases)} hard cases)")
    print(f"  {summary_path}    (per-oracle alignment)")
    print()
    print("=" * 72)
    print("Per-oracle alignment with 4-oracle consensus:")
    print("=" * 72)
    print(f"{'oracle':<16} {'avg_Pearson':>12} {'ct_agreement':>15}")
    for k, v in summary["per_oracle_alignment"].items():
        print(f"{k:<16} {v['avg_pearson_with_consensus']:>12.3f} {v['ct_agreement_count']:>15}  ({v['ct_agreement_rate_pct']}%)")


if __name__ == "__main__":
    main()
