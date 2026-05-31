"""Extract 30 articles where Gemini v3 and DeepSeek v3 disagree on content_type.
Stratified by which oracle fired which flag, so Opus's judgments cover all
disagreement patterns. Used to build a per-oracle accuracy truth set
(Bayesian update on agent-judged hard cases, per data-analyzer reviewer).

Output: datasets/scored/cd_v5_2oracle_disagreement_30.md
"""

import json
import glob
import random
from collections import Counter, defaultdict
from pathlib import Path

DIMS = ["discovery_novelty", "heritage_significance", "cross_cultural_connection",
        "human_resonance", "evidence_quality"]
PENALTY_FLAGS = {"historical_harm_reckoning", "commemoration_memorial",
                 "perpetrator_biography", "decline_loss", "launch_announcement"}


def extract(val):
    if val is None: return None
    if isinstance(val, dict): return val.get("score")
    return val


def main():
    # Load Gemini v3
    gem = {}
    for fp in sorted(glob.glob("datasets/scored/cd_v5_softpenalty_rescored_v3/cultural_discovery/scored_batch_*.jsonl")):
        with open(fp, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                af = r.get("cultural_discovery_analysis", {})
                if not af: continue
                gem[r["id"]] = {
                    "ct": af.get("content_type", "unknown"),
                    "dims": {d: extract(af.get(d)) for d in DIMS},
                }

    # Load DeepSeek v3
    ds = {}
    with open("datasets/scored/cd_v5_softpenalty_deepseek_v3/results.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "error" in r: continue
            ds[r["id"]] = {
                "ct": r["deepseek_content_type"],
                "dims": {d: extract(r["deepseek"].get(d)) for d in DIMS},
            }

    # Load article content
    article_content = {}
    with open("datasets/scored/cd_v5_522_for_softpenalty_rescore.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            article_content[r["id"]] = {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "source": r.get("source", ""),
                "published_date": r.get("published_date", ""),
                "content": r.get("content", "")[:2000],
            }

    # Find content_type disagreements
    common = set(gem) & set(ds)
    disagreements = [id_ for id_ in common if gem[id_]["ct"] != ds[id_]["ct"]]
    print(f"Total common articles: {len(common)}")
    print(f"content_type disagreements: {len(disagreements)} ({100*len(disagreements)/len(common):.0f}%)")

    # Stratify by disagreement pattern (gem_ct -> ds_ct)
    by_pattern = defaultdict(list)
    for id_ in disagreements:
        key = (gem[id_]["ct"], ds[id_]["ct"])
        by_pattern[key].append(id_)

    patterns_sorted = sorted(by_pattern.items(), key=lambda x: -len(x[1]))
    print(f"\nTop disagreement patterns:")
    for (g, d), ids in patterns_sorted[:15]:
        print(f"  {len(ids):>3} : gem={g!r:>30} ds={d!r}")

    # Sample 30 stratified: take ~3 from top 10 patterns
    random.seed(42)
    sample_ids = []
    for (g, d), ids in patterns_sorted[:10]:
        random.shuffle(ids)
        sample_ids.extend(ids[:3])
    # If we have fewer than 30, fill from remaining patterns
    if len(sample_ids) < 30:
        remaining = [id_ for (g, d), ids in patterns_sorted[10:] for id_ in ids]
        random.shuffle(remaining)
        sample_ids.extend(remaining[:30 - len(sample_ids)])
    sample_ids = sample_ids[:30]

    print(f"\nSampled {len(sample_ids)} articles for agent judgment")

    out = []
    out.append("# cd v5 — 2-Oracle Disagreement Sample for Agent Judgment\n")
    out.append("**Date:** 2026-05-31\n")
    out.append(f"**Sample:** {len(sample_ids)} articles where Gemini-v3 (tightened) and DeepSeek-v3 (tightened) classified the same article differently on content_type.\n")
    out.append("\n## Purpose\n\n")
    out.append("Per data-analyzer reviewer's recommendation: use agent-judged hard cases as a 'small but real truth set' to compute per-oracle accuracy. This sample includes 30 articles where Gemini-v3 and DeepSeek-v3 disagree on classification. Your verdicts will be used to compute:\n")
    out.append("- Gemini accuracy on disagreement set = (your 'gemini-correct' count) / (non-unclear total)\n")
    out.append("- DeepSeek accuracy on disagreement set = (your 'deepseek-correct' count) / (non-unclear total)\n")
    out.append("These accuracies feed the Bayesian update on which oracle to use for production retrain.\n")
    out.append("\n## Reference\n")
    out.append("Authoritative prompt: `C:\\local_dev\\llm-distillery\\filters\\cultural_discovery\\v5\\prompt-compressed.md` (latest version with F-K tightenings + A-E refinements + G/I/K specific anti-triggers).\n")
    out.append("\n## How to judge\n")
    out.append("For each article: read title + content snippet. Apply the v5 prompt's content_type definitions (including all flag triggers, carve-outs, and the K anti-trigger list). Decide which oracle's classification is correct PER THE PROMPT AS WRITTEN.\n")
    out.append("\nMark verdict as one of:\n")
    out.append("- `[X] Gemini correct` — Gemini's content_type matches the prompt's intent\n")
    out.append("- `[X] DeepSeek correct` — DeepSeek's content_type matches the prompt's intent\n")
    out.append("- `[X] Both reasonable / unclear` — the prompt is genuinely ambiguous on this article\n")
    out.append("- `[X] Both wrong` — neither classification is correct (specify the correct content_type)\n")
    out.append("\nAdd 1-sentence reasoning citing the specific flag/carve-out.\n")
    out.append("\n---\n")

    for i, id_ in enumerate(sample_ids, 1):
        art = article_content.get(id_, {})
        out.append(f"\n### {i}. {art.get('title','(no title)')[:120]}\n")
        out.append(f"**ID:** `{id_}`  \n")
        out.append(f"**Source:** {art.get('source','?')} | **Date:** {art.get('published_date','?')}  \n")
        if art.get("url"):
            out.append(f"**URL:** <{art['url']}>  \n")
        out.append(f"\n**Content (first 2000 chars):**\n")
        out.append(f"> {art.get('content','(no content)').replace(chr(10), ' ')[:2000]}\n")
        out.append(f"\n**Gemini-v3:** content_type=`{gem[id_]['ct']}`\n")
        out.append(f"  dims: " + " | ".join(f"{d}={gem[id_]['dims'][d]}" for d in DIMS) + "\n")
        out.append(f"\n**DeepSeek-v3:** content_type=`{ds[id_]['ct']}`\n")
        out.append(f"  dims: " + " | ".join(f"{d}={ds[id_]['dims'][d]}" for d in DIMS) + "\n")
        out.append(f"\n**Verdict:** `[ ]` Gemini correct  `[ ]` DeepSeek correct  `[ ]` Both reasonable / unclear  `[ ]` Both wrong (correct: `_______`)\n")
        out.append(f"**Reasoning:** _________________________________________________________\n")
        out.append(f"\n---\n")

    out.append("\n## Final Tally\n")
    out.append("After judging all 30:\n")
    out.append("- Gemini correct: ___ / 30\n")
    out.append("- DeepSeek correct: ___ / 30\n")
    out.append("- Both reasonable / unclear: ___ / 30\n")
    out.append("- Both wrong: ___ / 30\n")
    out.append("\n**Per-oracle accuracy on disagreement set:**\n")
    out.append("- Gemini accuracy = gem-correct / (gem-correct + ds-correct) = ___\n")
    out.append("- DeepSeek accuracy = ds-correct / (gem-correct + ds-correct) = ___\n")
    out.append("\nThese feed the Bayesian-update oracle-selection rule.\n")

    output_path = Path("datasets/scored/cd_v5_2oracle_disagreement_30.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(out))
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
