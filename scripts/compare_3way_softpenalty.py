"""Three-way comparison: Gemini-hard-cap vs Gemini-soft-penalty vs DeepSeek-soft-penalty
on the same 522 cultural_discovery v5 articles.

Isolates oracle-mechanism effect (Gemini hard vs soft) from oracle-vendor effect
(Gemini soft vs DeepSeek soft) without using either as ground truth.
"""

import json
import glob
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
    if None in vals:
        return None
    return sum(v * w for v, w in zip(vals, WEIGHTS))


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


def main():
    ds_data = {}
    with open("datasets/scored/cd_v5_softpenalty_deepseek/results.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "error" in r:
                continue
            ds_data[r["id"]] = {
                "title": r.get("title", ""),
                "gem_hard": r["gemini"],
                "gem_hard_ct": r["gemini_content_type"],
                "ds_soft": r["deepseek"],
                "ds_soft_ct": r["deepseek_content_type"],
            }

    gs_data = {}
    for fp in sorted(glob.glob("datasets/scored/cd_v5_softpenalty_rescored/cultural_discovery/scored_batch_*.jsonl")):
        with open(fp, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                af = r.get("cultural_discovery_analysis", {})
                gs_data[r["id"]] = {
                    "dims": {d: extract(af.get(d)) for d in DIMS},
                    "ct": af.get("content_type"),
                }

    joined = []
    for id_, ds in ds_data.items():
        if id_ in gs_data:
            joined.append({
                "id": id_,
                "title": ds["title"],
                "gem_hard": {d: float(ds["gem_hard"].get(d)) if ds["gem_hard"].get(d) is not None else None for d in DIMS},
                "gem_hard_ct": ds["gem_hard_ct"],
                "gem_soft": {d: float(gs_data[id_]["dims"].get(d)) if gs_data[id_]["dims"].get(d) is not None else None for d in DIMS},
                "gem_soft_ct": gs_data[id_]["ct"],
                "ds_soft": {d: float(ds["ds_soft"].get(d)) if ds["ds_soft"].get(d) is not None else None for d in DIMS},
                "ds_soft_ct": ds["ds_soft_ct"],
            })

    print(f"Joined: {len(joined)} articles\n")

    # MECHANISM EFFECT
    print("=" * 78)
    print("1. MECHANISM: Gemini hard-cap vs Gemini soft-penalty (same oracle, same prompt-text)")
    print("=" * 78)
    print(f"{'Dimension':<32} {'hard_mean':>10} {'soft_mean':>10} {'diff':>8}")
    print("-" * 78)
    for d in DIMS:
        h = [p["gem_hard"][d] for p in joined if p["gem_hard"][d] is not None]
        s = [p["gem_soft"][d] for p in joined if p["gem_soft"][d] is not None]
        if h and s:
            print(f"{d:<32} {sum(h)/len(h):>10.2f} {sum(s)/len(s):>10.2f} {(sum(s)/len(s) - sum(h)/len(h)):>+8.2f}")

    wh = [wavg(p["gem_hard"]) for p in joined if wavg(p["gem_hard"]) is not None]
    ws = [wavg(p["gem_soft"]) for p in joined if wavg(p["gem_soft"]) is not None]
    print(f"{'WEIGHTED AVG':<32} {sum(wh)/len(wh):>10.2f} {sum(ws)/len(ws):>10.2f} {(sum(ws)/len(ws) - sum(wh)/len(wh)):>+8.2f}")

    ct_agree = sum(1 for p in joined if p["gem_hard_ct"] == p["gem_soft_ct"])
    print(f"\ncontent_type agreement (hard vs soft): {ct_agree}/{len(joined)} ({100*ct_agree/len(joined):.0f}%)")

    flagged = [p for p in joined if p["gem_soft_ct"] in PENALTY_FLAGS]
    print(f"\nGemini-soft penalty-flagged articles: {len(flagged)}")
    hard_spreads, soft_spreads = [], []
    for p in flagged:
        if all(v is not None for v in p["gem_hard"].values()):
            hard_spreads.append(max(p["gem_hard"].values()) - min(p["gem_hard"].values()))
        if all(v is not None for v in p["gem_soft"].values()):
            soft_spreads.append(max(p["gem_soft"].values()) - min(p["gem_soft"].values()))
    if hard_spreads and soft_spreads:
        print(f"  hard-cap mean dim-spread:    {sum(hard_spreads)/len(hard_spreads):.2f}")
        print(f"  soft-penalty mean dim-spread: {sum(soft_spreads)/len(soft_spreads):.2f}")
        print(f"  diff (positive = gradient preserved): {sum(soft_spreads)/len(soft_spreads) - sum(hard_spreads)/len(hard_spreads):+.2f}")

    # VENDOR EFFECT
    print("\n" + "=" * 78)
    print("2. VENDOR (same soft-penalty prompt): Gemini-soft vs DeepSeek-soft")
    print("=" * 78)
    print(f"{'Dimension':<32} {'gem_mean':>10} {'ds_mean':>10} {'Pearson':>9} {'Spearman':>9} {'MAE':>7}")
    print("-" * 78)
    for d in DIMS:
        g = [p["gem_soft"][d] for p in joined if p["gem_soft"][d] is not None and p["ds_soft"][d] is not None]
        s = [p["ds_soft"][d] for p in joined if p["gem_soft"][d] is not None and p["ds_soft"][d] is not None]
        if len(g) >= 2:
            r_p = pearson(g, s)
            r_s = spearman(g, s)
            mae = sum(abs(a - b) for a, b in zip(g, s)) / len(g)
            print(f"{d:<32} {sum(g)/len(g):>10.2f} {sum(s)/len(s):>10.2f} {r_p:>9.3f} {r_s:>9.3f} {mae:>7.2f}")

    ct_agree_v = sum(1 for p in joined if p["gem_soft_ct"] == p["ds_soft_ct"])
    print(f"\ncontent_type agreement (Gemini-soft vs DeepSeek-soft): {ct_agree_v}/{len(joined)} ({100*ct_agree_v/len(joined):.0f}%)")

    gs_pen = sum(1 for p in joined if p["gem_soft_ct"] in PENALTY_FLAGS)
    ds_pen = sum(1 for p in joined if p["ds_soft_ct"] in PENALTY_FLAGS)
    gs_max = sum(1 for p in joined if p["gem_soft_ct"] in MAX_SCORE_FLAGS)
    ds_max = sum(1 for p in joined if p["ds_soft_ct"] in MAX_SCORE_FLAGS)
    print(f"\nFlag-firing rates (out of {len(joined)} articles):")
    print(f"  Gemini-soft  F-K penalty flags:  {gs_pen} ({100*gs_pen/len(joined):.1f}%)")
    print(f"  DeepSeek-soft F-K penalty flags: {ds_pen} ({100*ds_pen/len(joined):.1f}%)")
    print(f"  Gemini-soft  A-E max_score flags:  {gs_max} ({100*gs_max/len(joined):.1f}%)")
    print(f"  DeepSeek-soft A-E max_score flags: {ds_max} ({100*ds_max/len(joined):.1f}%)")

    # Top divergences
    print("\n" + "=" * 78)
    print("3. TRIPLE DIVERGENCE: 5 articles where all 3 oracles spread most on weighted_avg")
    print("=" * 78)

    def triple_spread(p):
        if any(p[k][d] is None for k in ("gem_hard", "gem_soft", "ds_soft") for d in DIMS):
            return -1
        wh = wavg(p["gem_hard"])
        ws = wavg(p["gem_soft"])
        wd = wavg(p["ds_soft"])
        return max(wh, ws, wd) - min(wh, ws, wd)

    candidates = [p for p in joined if triple_spread(p) > 0]
    for p in sorted(candidates, key=triple_spread, reverse=True)[:5]:
        print(f"\n  {p['title'][:90]!r}")
        print(f"  ct: hard={p['gem_hard_ct']!r}  soft={p['gem_soft_ct']!r}  ds={p['ds_soft_ct']!r}")
        wh = wavg(p["gem_hard"])
        ws = wavg(p["gem_soft"])
        wd = wavg(p["ds_soft"])
        print(f"  weighted_avg: hard={wh:.2f}  soft={ws:.2f}  deepseek={wd:.2f}")
        for d in DIMS:
            print(f"    {d:<30}: hard {p['gem_hard'][d]:>4.1f}  soft {p['gem_soft'][d]:>4.1f}  ds {p['ds_soft'][d]:>4.1f}")

    # Vendor disagreements (specifically Gemini-soft vs DeepSeek-soft)
    print("\n" + "=" * 78)
    print("4. VENDOR DISAGREEMENT: 5 articles where Gemini-soft and DeepSeek-soft diverge most")
    print("=" * 78)

    def vendor_spread(p):
        if any(p["gem_soft"][d] is None or p["ds_soft"][d] is None for d in DIMS):
            return -1
        return sum(abs(p["gem_soft"][d] - p["ds_soft"][d]) for d in DIMS) / 5

    candidates = [p for p in joined if vendor_spread(p) > 0]
    for p in sorted(candidates, key=vendor_spread, reverse=True)[:5]:
        print(f"\n  {p['title'][:90]!r}")
        print(f"  ct: gem-soft={p['gem_soft_ct']!r}  ds-soft={p['ds_soft_ct']!r}")
        for d in DIMS:
            print(f"    {d:<30}: gem {p['gem_soft'][d]:>4.1f}  ds {p['ds_soft'][d]:>4.1f}  diff {p['ds_soft'][d] - p['gem_soft'][d]:+.1f}")


if __name__ == "__main__":
    main()
