"""One-shot relicensing of HF Hub model-card YAML frontmatter (mit -> eupl-1.2).

Triggered by llm-distillery#65: the upload template was hardcoding `license: mit`
in every push since the repo was relicensed to EUPL-1.2 (commit d04c9a2,
2026-05-18). Source-side fix landed in commit fb67d05 (2026-05-22). This
script fixes the 14 Hub repos already deployed under the misconfigured template.

Idempotent: skips repos already carrying `license: eupl-1.2` and repos with
no `license: mit` in their frontmatter. Operates only on the YAML frontmatter
(between the first two `---` lines), never the model-card body.

Manual-action artifact, not part of the standard deployment pipeline. After
running once, this script can be archived.
"""

import os
import re
import sys
import tempfile

from huggingface_hub import HfApi, hf_hub_download

REPOS = [
    "jeergrvgreg/uplifting-filter-v1",
    "jeergrvgreg/uplifting-filter-v5",
    "jeergrvgreg/investment-risk-filter-v5",
    "jeergrvgreg/sustainability-technology-v2",
    "jeergrvgreg/investment-risk-v5",
    "jeergrvgreg/cultural-discovery-v3",
    "jeergrvgreg/uplifting-filter-v6",
    "jeergrvgreg/cultural-discovery-v4",
    "jeergrvgreg/sustainability-technology-v3",
    "jeergrvgreg/investment-risk-filter-v6",
    "jeergrvgreg/belonging-filter-v1",
    "jeergrvgreg/nature-recovery-filter-v1",
    "jeergrvgreg/foresight-filter-v1",
    "jeergrvgreg/nature-recovery-filter-v2",
]

COMMIT_MESSAGE = (
    "license: model-card YAML frontmatter MIT -> EUPL-1.2 "
    "(source repo relicensed 2026-05-18; ducroq/llm-distillery#65)"
)


def relicense_one(api: HfApi, repo_id: str) -> str:
    """Returns one of: 'OK', 'SKIP_ALREADY', 'SKIP_NO_MATCH', 'SKIP_NO_FM', 'FAIL: <reason>'."""
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename="README.md",
            repo_type="model",
            force_download=True,
        )
    except Exception as e:
        return f"FAIL: download README ({e})"

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return f"FAIL: read README ({e})"

    if not content.startswith("---\n"):
        return "SKIP_NO_FM"
    end = content.find("\n---\n", 4)
    if end == -1:
        return "SKIP_NO_FM"

    fm = content[4:end]
    rest_index = end + len("\n---\n")
    rest = content[rest_index:]

    if re.search(r"^license:\s*eupl-1\.2\b", fm, flags=re.MULTILINE):
        return "SKIP_ALREADY"

    new_fm, n = re.subn(
        r"^license:\s*mit\b",
        "license: eupl-1.2",
        fm,
        count=1,
        flags=re.MULTILINE,
    )
    if n == 0:
        return "SKIP_NO_MATCH"

    new_content = "---\n" + new_fm + "\n---\n" + rest

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".md", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(new_content)
            tmp_path = tmp.name
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message=COMMIT_MESSAGE,
        )
    except Exception as e:
        return f"FAIL: upload ({e})"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return "OK"


def main():
    api = HfApi()
    user = api.whoami()
    print(f"Authenticated as: {user.get('name')}")
    print(f"Operating on {len(REPOS)} repos under jeergrvgreg/")
    print()

    results = {}
    for repo_id in REPOS:
        status = relicense_one(api, repo_id)
        results[repo_id] = status
        print(f"{status:25s} {repo_id}")

    print()
    print("--- Summary ---")
    ok = sum(1 for s in results.values() if s == "OK")
    skip_already = sum(1 for s in results.values() if s == "SKIP_ALREADY")
    skip_no_match = sum(1 for s in results.values() if s == "SKIP_NO_MATCH")
    skip_no_fm = sum(1 for s in results.values() if s == "SKIP_NO_FM")
    fail = sum(1 for s in results.values() if s.startswith("FAIL"))
    print(f"  Updated:           {ok}")
    print(f"  Already eupl-1.2:  {skip_already}")
    print(f"  No mit in fm:      {skip_no_match}")
    print(f"  No frontmatter:    {skip_no_fm}")
    print(f"  Failed:            {fail}")

    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
