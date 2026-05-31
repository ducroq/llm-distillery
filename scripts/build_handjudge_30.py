"""Build a markdown file for hand-judging 30 articles where Gemini fired an F-K penalty flag
but DeepSeek did not (or vice versa). The user reads each, judges who's right per the v5 prompt,
and we use the result to decide whether the prompt needs tightening or Gemini's prior is correct.

Input:
    datasets/scored/cd_v5_softpenalty_deepseek/results.jsonl  (DeepSeek + Gemini-hard pairs)
    datasets/scored/cd_v5_softpenalty_rescored/cultural_discovery/scored_batch_*.jsonl  (Gemini-soft)
    datasets/scored/cd_v5_522_for_softpenalty_rescore.jsonl  (article content for context)

Output:
    datasets/scored/cd_v5_handjudge_30.md  (markdown with 30 articles + judgment slots)
"""

import json
import glob
import random
from collections import Counter

DIMS = ["discovery_novelty", "heritage_significance", "cross_cultural_connection",
        "human_resonance", "evidence_quality"]
PENALTY_FLAGS = {"historical_harm_reckoning", "commemoration_memorial",
                 "perpetrator_biography", "decline_loss", "launch_announcement"}


def extract(val):
    if isinstance(val, dict):
        return val.get("score")
    return val


def main():
    # Load DeepSeek results
    ds_data = {}
    with open("datasets/scored/cd_v5_softpenalty_deepseek/results.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "error" in r:
                continue
            ds_data[r["id"]] = {
                "title": r.get("title", ""),
                "ds_soft_dims": {d: extract(r["deepseek"].get(d)) for d in DIMS},
                "ds_soft_ct": r["deepseek_content_type"],
            }

    # Load Gemini-soft results
    gs_data = {}
    for fp in sorted(glob.glob("datasets/scored/cd_v5_softpenalty_rescored/cultural_discovery/scored_batch_*.jsonl")):
        with open(fp, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                af = r.get("cultural_discovery_analysis", {})
                gs_data[r["id"]] = {
                    "gem_soft_dims": {d: extract(af.get(d)) for d in DIMS},
                    "gem_soft_ct": af.get("content_type"),
                }

    # Load article content for context
    article_content = {}
    with open("datasets/scored/cd_v5_522_for_softpenalty_rescore.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            article_content[r["id"]] = {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "source": r.get("source", ""),
                "published_date": r.get("published_date", ""),
                "content": r.get("content", "")[:1500],  # first 1500 chars for context
            }

    # Find disagreements: Gemini fired F-K, DeepSeek did not (or vice versa)
    disagreements_gem_fires = []  # Gem F-K, DS doesn't
    disagreements_ds_fires = []   # DS F-K, Gem doesn't
    for id_ in ds_data:
        if id_ not in gs_data:
            continue
        g_ct = gs_data[id_]["gem_soft_ct"]
        d_ct = ds_data[id_]["ds_soft_ct"]
        if g_ct in PENALTY_FLAGS and d_ct not in PENALTY_FLAGS:
            disagreements_gem_fires.append(id_)
        elif d_ct in PENALTY_FLAGS and g_ct not in PENALTY_FLAGS:
            disagreements_ds_fires.append(id_)

    print(f"Gem-fires-F-K-but-DS-doesn't: {len(disagreements_gem_fires)}")
    print(f"DS-fires-F-K-but-Gem-doesn't: {len(disagreements_ds_fires)}")

    # Sample 20 from gem_fires + 10 from ds_fires (proportional-ish to gap)
    random.seed(42)
    random.shuffle(disagreements_gem_fires)
    random.shuffle(disagreements_ds_fires)
    sample_gem = disagreements_gem_fires[:20]
    sample_ds = disagreements_ds_fires[:10]

    # Stratify gem_fires sample by Gemini's flag (so we cover F/G/H/I/K)
    gem_by_flag = Counter(gs_data[id_]["gem_soft_ct"] for id_ in sample_gem)
    ds_by_flag = Counter(ds_data[id_]["ds_soft_ct"] for id_ in sample_ds)
    print(f"\nSample Gem-fires distribution: {dict(gem_by_flag)}")
    print(f"Sample DS-fires distribution: {dict(ds_by_flag)}")

    # Build markdown
    out = []
    out.append("# Cultural Discovery v5 — Hand-Judgment Task (30 F-K disagreements)\n")
    out.append("**Date generated:** 2026-05-31\n")
    out.append("**Purpose:** When Gemini-soft and DeepSeek-soft disagree on whether to fire an F-K penalty flag, which one is correct per the v5 prompt? The result decides whether the v5 prompt needs tightening (if Gemini over-fires) or whether DeepSeek is under-applying (if Gemini is right).\n")
    out.append("\n## How to use\n")
    out.append("For each article, read the title + content snippet. Then judge:\n")
    out.append("1. Should the F-K flag fire per the v5 prompt's flag definitions (F=historical_harm_reckoning, G=commemoration_memorial, H=perpetrator_biography, I=decline_loss, K=launch_announcement)?\n")
    out.append("2. Mark `[ ] Gemini correct`, `[ ] DeepSeek correct`, or `[ ] Both reasonable / unclear`\n")
    out.append("3. Optional: 1-line note on why\n")
    out.append("\nAggregate at the end: if Gemini-correct count > 18/30, the prompt is fine; if DS-correct > 18/30, the prompt needs tightening; in between is ambiguous.\n")
    out.append("\n---\n")

    out.append("\n## Part 1 — Gemini fired F-K, DeepSeek didn't (20 articles)\n")
    out.append("*Hypothesis to test: is Gemini over-applying the F-K flags here?*\n")
    for i, id_ in enumerate(sample_gem, 1):
        art = article_content.get(id_, {})
        out.append(f"\n### {i}. {art.get('title','(no title)')[:120]}\n")
        out.append(f"**ID:** `{id_}`  \n")
        out.append(f"**Source:** {art.get('source','?')} | **Date:** {art.get('published_date','?')}  \n")
        if art.get("url"):
            out.append(f"**URL:** <{art['url']}>  \n")
        out.append(f"\n**Content snippet (first 1500 chars):**\n")
        out.append(f"> {art.get('content','(no content)').replace(chr(10), ' ')[:1500]}\n")
        out.append(f"\n**Gemini-soft:** content_type=`{gs_data[id_]['gem_soft_ct']}` (F-K flag fired)  \n")
        out.append("  dims: " + ", ".join(f"{d}={gs_data[id_]['gem_soft_dims'][d]}" for d in DIMS) + "  \n")
        out.append(f"\n**DeepSeek-soft:** content_type=`{ds_data[id_]['ds_soft_ct']}` (no F-K flag)  \n")
        out.append("  dims: " + ", ".join(f"{d}={ds_data[id_]['ds_soft_dims'][d]}" for d in DIMS) + "  \n")
        out.append(f"\n**Verdict:** `[ ]` Gemini correct  `[ ]` DeepSeek correct  `[ ]` Both reasonable / unclear  \n")
        out.append(f"**Note:** _______________________________________________\n")
        out.append(f"\n---\n")

    out.append("\n## Part 2 — DeepSeek fired F-K, Gemini didn't (10 articles)\n")
    out.append("*Hypothesis to test: is DeepSeek over-applying the F-K flags here?*\n")
    for i, id_ in enumerate(sample_ds, 1):
        art = article_content.get(id_, {})
        out.append(f"\n### {i}. {art.get('title','(no title)')[:120]}\n")
        out.append(f"**ID:** `{id_}`  \n")
        out.append(f"**Source:** {art.get('source','?')} | **Date:** {art.get('published_date','?')}  \n")
        if art.get("url"):
            out.append(f"**URL:** <{art['url']}>  \n")
        out.append(f"\n**Content snippet (first 1500 chars):**\n")
        out.append(f"> {art.get('content','(no content)').replace(chr(10), ' ')[:1500]}\n")
        out.append(f"\n**Gemini-soft:** content_type=`{gs_data[id_]['gem_soft_ct']}` (no F-K flag)  \n")
        out.append("  dims: " + ", ".join(f"{d}={gs_data[id_]['gem_soft_dims'][d]}" for d in DIMS) + "  \n")
        out.append(f"\n**DeepSeek-soft:** content_type=`{ds_data[id_]['ds_soft_ct']}` (F-K flag fired)  \n")
        out.append("  dims: " + ", ".join(f"{d}={ds_data[id_]['ds_soft_dims'][d]}" for d in DIMS) + "  \n")
        out.append(f"\n**Verdict:** `[ ]` Gemini correct  `[ ]` DeepSeek correct  `[ ]` Both reasonable / unclear  \n")
        out.append(f"**Note:** _______________________________________________\n")
        out.append(f"\n---\n")

    out.append("\n## Final Tally\n")
    out.append("After judging all 30:\n")
    out.append("- Gemini correct: ___ / 30\n")
    out.append("- DeepSeek correct: ___ / 30\n")
    out.append("- Both reasonable / unclear: ___ / 30\n")
    out.append("\n**Interpretation:**\n")
    out.append("- If Gemini-correct ≥ 18: the v5 prompt is well-tuned to Gemini; DeepSeek under-applies. Stick with Gemini for retrain.\n")
    out.append("- If DeepSeek-correct ≥ 18: the v5 prompt is too aggressive with F-K; Gemini over-applies. Tighten F-K trigger language before retrain.\n")
    out.append("- If neither ≥ 18: the prompt is genuinely ambiguous on these cases — consider adding more contrastive examples or hand-tuning per flag.\n")

    output_path = "datasets/scored/cd_v5_handjudge_30.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(out))

    print(f"\nWrote {output_path}")
    print(f"  20 articles where Gemini fired F-K, DeepSeek didn't")
    print(f"  10 articles where DeepSeek fired F-K, Gemini didn't")


if __name__ == "__main__":
    main()
