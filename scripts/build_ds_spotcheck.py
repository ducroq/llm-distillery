"""Build two spot-check markdowns for Claude agents to judge DeepSeek v3 scoring quality.

Pool A (false-negative check): 15 articles DeepSeek called cultural_discovery with high
weighted_avg. Should any have been F-K flagged?

Pool B (false-positive check): 15 articles DeepSeek fired F-K on. Is the flag correct?

The two pools answer:
  A: Is DeepSeek missing real F-K cases? (leakage problem unfixed)
  B: When DeepSeek fires F-K, is it right? (already validated on disagreements; this is the
     stronger version — fires DeepSeek made independently of Gemini)
"""

import json
import glob
import random

DIMS = ["discovery_novelty", "heritage_significance", "cross_cultural_connection",
        "human_resonance", "evidence_quality"]
WEIGHTS = [0.25, 0.20, 0.25, 0.15, 0.15]
PENALTY_FLAGS = {"historical_harm_reckoning", "commemoration_memorial",
                 "perpetrator_biography", "decline_loss", "launch_announcement"}


def extract(val):
    if isinstance(val, dict):
        return val.get("score")
    return val


def wavg(dims):
    vals = [dims.get(d) for d in DIMS]
    if None in vals:
        return None
    return sum(float(v) * w for v, w in zip(vals, WEIGHTS))


def main():
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

    # Load DeepSeek v3 results
    ds_records = []
    with open("datasets/scored/cd_v5_softpenalty_deepseek_v3/results.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "error" in r:
                continue
            ds_dims = {d: extract(r["deepseek"].get(d)) for d in DIMS}
            if any(v is None for v in ds_dims.values()):
                continue
            ds_records.append({
                "id": r["id"],
                "title": r.get("title", ""),
                "ds_ct": r["deepseek_content_type"],
                "ds_dims": ds_dims,
                "ds_wavg": wavg(ds_dims),
            })

    print(f"Loaded {len(ds_records)} DeepSeek records")

    random.seed(42)

    # POOL A: DS=cultural_discovery, high score
    pool_a = [r for r in ds_records if r["ds_ct"] == "cultural_discovery" and r["ds_wavg"] >= 4.5]
    random.shuffle(pool_a)
    sample_a = pool_a[:15]
    print(f"Pool A (DS=cd, wavg>=4.5): {len(pool_a)} candidates -> 15 sampled")

    # POOL B: DS fired F-K
    pool_b_by_flag = {f: [] for f in PENALTY_FLAGS}
    for r in ds_records:
        if r["ds_ct"] in PENALTY_FLAGS:
            pool_b_by_flag[r["ds_ct"]].append(r)
    print(f"Pool B (DS F-K fires) by flag:")
    sample_b = []
    for flag, items in pool_b_by_flag.items():
        random.shuffle(items)
        # Take ~3 per flag if available, more from larger pools
        take = min(len(items), 4 if len(items) > 10 else 2)
        sample_b.extend(items[:take])
        print(f"  {flag}: {len(items)} candidates -> {take} sampled")
    sample_b = sample_b[:15]

    def write_pool(filename, sample, pool_name, framing):
        out = []
        out.append(f"# DeepSeek v3 Spot-Check — {pool_name}\n")
        out.append("**Date:** 2026-05-31\n")
        out.append(f"**Sample size:** {len(sample)} articles\n")
        out.append(f"\n## Purpose\n\n{framing}\n")
        out.append("\n## Reference\n")
        out.append("Authoritative v5 prompt: `C:\\local_dev\\llm-distillery\\filters\\cultural_discovery\\v5\\prompt-compressed.md`\n")
        out.append("Read the dimension scale definitions (Section 1) and the F-K flag definitions (Section 3) before judging.\n")
        out.append("\n## How to judge\n\n")
        out.append("For each article below: read title + content snippet. Then judge the listed verdict question.\n")
        out.append("Mark verdict as one of: `[ ] CORRECT` (DeepSeek's call is right), `[ ] WRONG` (DeepSeek's call is wrong; explain what it should have been), `[ ] UNCLEAR` (genuine ambiguity).\n")
        out.append("\n---\n")

        for i, r in enumerate(sample, 1):
            art = article_content.get(r["id"], {})
            out.append(f"\n### {i}. {art.get('title','(no title)')[:120]}\n")
            out.append(f"**ID:** `{r['id']}`  \n")
            out.append(f"**Source:** {art.get('source','?')} | **Date:** {art.get('published_date','?')}  \n")
            if art.get("url"):
                out.append(f"**URL:** <{art['url']}>  \n")
            out.append(f"\n**Content (first 2000 chars):**\n")
            out.append(f"> {art.get('content','(no content)').replace(chr(10), ' ')[:2000]}\n")
            out.append(f"\n**DeepSeek v3 call:**  \n")
            out.append(f"  content_type = `{r['ds_ct']}`  \n")
            out.append(f"  dims = " + " | ".join(f"{d}={r['ds_dims'][d]}" for d in DIMS) + "  \n")
            out.append(f"  weighted_avg = {r['ds_wavg']:.2f}  \n")
            out.append(f"\n**Verdict:** `[ ]` CORRECT  `[ ]` WRONG (what should it be?)  `[ ]` UNCLEAR  \n")
            out.append(f"**Note:** _______________________________________________\n")
            out.append(f"\n---\n")

        out.append("\n## Final Tally\n")
        out.append(f"After judging all {len(sample)}:\n")
        out.append("- CORRECT: ___ / " + str(len(sample)) + "\n")
        out.append("- WRONG: ___ / " + str(len(sample)) + "\n")
        out.append("- UNCLEAR: ___ / " + str(len(sample)) + "\n")
        out.append("\n**Interpretation:**\n")
        if "false-negative" in pool_name.lower():
            out.append("- If WRONG ≥ 3/15 (20%): DeepSeek is leaking real F-K cases as cultural_discovery — the very leakage problem #62 was created to fix. Don't use DeepSeek for retrain.\n")
            out.append("- If WRONG ≤ 1/15: DeepSeek's restraint is genuine prompt-alignment, not laziness. Safe for retrain.\n")
        else:
            out.append("- If WRONG ≥ 3/15 (20%): DeepSeek's F-K fires include too many false positives. Not as good as we thought.\n")
            out.append("- If WRONG ≤ 1/15: DeepSeek's F-K fires are reliable. Safe for retrain.\n")

        with open(filename, "w", encoding="utf-8") as f:
            f.write("".join(out))
        print(f"Wrote {filename}")

    write_pool(
        "datasets/scored/cd_v5_ds_spotcheck_A_falseneg.md",
        sample_a,
        "POOL A — FALSE NEGATIVE CHECK",
        "DeepSeek v3 classified these 15 articles as `cultural_discovery` (in-scope, no F-K penalty) "
        "with weighted_avg >= 4.5 (above ovr.news's display threshold).\n\n"
        "**Question to judge per article:** Should any of these have triggered an F-K penalty flag "
        "(historical_harm_reckoning, commemoration_memorial, perpetrator_biography, decline_loss, "
        "launch_announcement) per the v5 prompt's flag definitions?\n\n"
        "**Failure mode being tested:** DeepSeek under-applies F-K (we already saw this in the "
        "DeepSeek-vs-Gemini comparison). If DeepSeek is missing real F-K cases, those articles will "
        "leak into Discovery at the same rate as v4 — the original #62 leakage problem stays unfixed."
    )

    write_pool(
        "datasets/scored/cd_v5_ds_spotcheck_B_falsepos.md",
        sample_b,
        "POOL B — FALSE POSITIVE CHECK",
        "DeepSeek v3 fired an F-K penalty flag on these 15 articles, stratified across F/G/H/I/K.\n\n"
        "**Question to judge per article:** Is DeepSeek's flag correctly applied per the v5 prompt's "
        "flag definitions and carve-outs?\n\n"
        "**Failure mode being tested:** DeepSeek over-applies F-K (sanity check on our prior finding "
        "that DeepSeek 'restraint = good'). If DeepSeek's fires include many false positives, then "
        "even though DeepSeek fires LESS than Gemini, it may not actually be MORE correct — just "
        "differently wrong."
    )


if __name__ == "__main__":
    main()
