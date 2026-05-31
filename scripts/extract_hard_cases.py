"""Build agent-readable markdown task files from cd_v5_hard_cases.jsonl.

Reads the consensus output and produces a markdown file that Opus + Haiku
agents can read to judge each hard case independently.

Output: datasets/scored/cd_v5_hard_cases_for_agents.md
"""

import json
from pathlib import Path

DIMS = ["discovery_novelty", "heritage_significance", "cross_cultural_connection",
        "human_resonance", "evidence_quality"]


def main():
    hard_path = Path("datasets/scored/cd_v5_hard_cases.jsonl")
    canonical_path = Path("datasets/scored/cd_v5_522_for_softpenalty_rescore.jsonl")
    output_path = Path("datasets/scored/cd_v5_hard_cases_for_agents.md")

    if not hard_path.exists():
        print(f"ERROR: {hard_path} not found. Run multi_oracle_consensus.py first.")
        return

    # Load article content
    article_content = {}
    with open(canonical_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            article_content[r["id"]] = {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "source": r.get("source", ""),
                "published_date": r.get("published_date", ""),
                "content": r.get("content", "")[:2000],
            }

    # Load hard cases
    hard_cases = []
    with open(hard_path, "r", encoding="utf-8") as f:
        for line in f:
            hard_cases.append(json.loads(line))

    # Cap at 40 for agent feasibility
    if len(hard_cases) > 40:
        print(f"Capping hard cases from {len(hard_cases)} → 40 for agent feasibility")
        hard_cases = hard_cases[:40]

    out = []
    out.append("# Cultural Discovery v5 — Hard Cases for Agent Judgment\n")
    out.append("**Date:** 2026-05-31\n")
    out.append(f"**Total hard cases:** {len(hard_cases)} (where <3 of 4 batch oracles agree on content_type)\n")
    out.append("\n## Purpose\n")
    out.append("Multi-oracle batch calibration (Gemini Flash 2.5 + DeepSeek V4 Flash + Qwen3:14b + Phi4:14b) produced clear consensus on most articles. These are the hard cases where consensus broke down — different oracles classified the article in different content_type categories. Agent judgment establishes the small-but-real truth set used to score each oracle's accuracy on the ambiguous edge.\n")
    out.append("\n## Reference (read first)\n")
    out.append("Authoritative prompt: `C:\\local_dev\\llm-distillery\\filters\\cultural_discovery\\v5\\prompt-compressed.md` — Section 3 (Pre-Classification Step) for flag definitions.\n")
    out.append("\n## How to judge each article\n")
    out.append("1. Read title + content snippet carefully\n")
    out.append("2. Apply the v5 prompt's flag definitions + carve-outs to determine the CORRECT content_type per the prompt as written\n")
    out.append("3. Choose ONE of the following (or none if completely ambiguous):\n")
    out.append("   - `[X] correct_ct=` followed by the correct content_type per the prompt\n")
    out.append("   - `[X] genuinely ambiguous` if you can't decide from the snippet\n")
    out.append("4. Optional: 1-sentence reasoning citing the specific flag definition\n")
    out.append("\n## Hard cases\n")
    out.append("\n---\n")

    for i, r in enumerate(hard_cases, 1):
        art = article_content.get(r["id"], {})
        out.append(f"\n### {i}. {art.get('title','(no title)')[:120]}\n")
        out.append(f"**ID:** `{r['id']}`  \n")
        out.append(f"**Source:** {art.get('source','?')} | **Date:** {art.get('published_date','?')}  \n")
        if art.get("url"):
            out.append(f"**URL:** <{art['url']}>  \n")
        out.append(f"\n**Content (first 2000 chars):**\n")
        out.append(f"> {art.get('content','(no content)').replace(chr(10), ' ')[:2000]}\n")
        out.append(f"\n**Oracle calls (where they split):**\n")
        for oracle_name, oracle_data in r["per_oracle"].items():
            dim_str = " | ".join(f"{d}={oracle_data['dims'][d]}" for d in DIMS)
            out.append(f"- **{oracle_name}**: content_type=`{oracle_data['ct']}`, dims: {dim_str}\n")
        out.append(f"\n**ct_split:** {r['ct_split']}\n")
        out.append(f"\n**Your verdict:**  \n")
        out.append(f"`[ ]` correct_ct = `_______________`  \n")
        out.append(f"`[ ]` genuinely ambiguous  \n")
        out.append(f"**Reasoning:** _________________________________________________________\n")
        out.append(f"\n---\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(out))

    print(f"Wrote {output_path}")
    print(f"  {len(hard_cases)} hard cases ready for agent judgment")


if __name__ == "__main__":
    main()
