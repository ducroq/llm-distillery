"""
Schema conformance gate for filter config.yaml files.

Purpose: lock in a canonical shape for filter configuration so new filters
don't introduce fresh drift. Existing known drift is allowed via the
EXEMPTIONS set — each exemption is a concrete (filter, version, issue_code)
tuple that documents a specific deviation.

Removing an exemption while the underlying drift still exists fails the
test (the exemption must match a real violation). Fixing the drift while
leaving the exemption in place also fails (exemption must stop matching).
This forces the cleanup and the allow-list to stay in lockstep.

See also: docs/adr/ (add ADR when the canonical schema is ratified).
"""

import sys
from pathlib import Path

import pytest
import yaml

# Add project root to path so `filters.*` imports resolve during test collection.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

REPO_ROOT = Path(__file__).parent.parent.parent

# --- Active filters ------------------------------------------------------
# The versions NexusMind currently deploys. Legacy versions (older than these)
# are archived and intentionally not validated. When a filter is promoted to
# a new version, update this list AND the NexusMind CLAUDE.md table.

ACTIVE_FILTERS = [
    ("sustainability_technology", "v3"),
    ("uplifting", "v7"),
    ("cultural-discovery", "v4"),
    ("investment-risk", "v6"),  # hyphen in llm-distillery (investment_risk in NexusMind)
    ("belonging", "v1"),
    ("nature_recovery", "v2"),
    ("foresight", "v1"),
]

# --- Canonical schema ---------------------------------------------------

REQUIRED_TOP_LEVEL = {
    "filter",             # filter metadata (name, version, id)
    "prefilter",          # keyword prefilter config
    "oracle",             # training-data oracle (NOT "ground_truth")
    "preprocessing",      # text preprocessing (head+tail, max_tokens, etc.)
    "scoring",            # dimensions, gatekeepers, tiers, scale factor
    "training",           # LoRA / training hyperparams
    "hybrid_inference",   # e5-small probe + threshold
    "deployment",         # HuggingFace Hub target repo
}

REQUIRED_SCORING_KEYS = {
    "dimensions",         # dict of dimension_name -> weight / description
    "gatekeepers",        # MUST be dict (not list)
    "tiers",              # dict of tier_name -> threshold/description
    "score_scale_factor", # linear fallback when percentile normalization is off
}

# --- Known-drift exemptions ---------------------------------------------
# Each entry = (filter, version, issue_code). Issue codes follow the
# convention used in _violations(). Tuples here are ACCEPTED; they don't
# cause the test to fail. But every tuple MUST still correspond to a real
# violation — removing one while drift persists is a bug.
#
# When fixing a drift during the B migration, remove the exemption(s) in
# the same commit as the config fix.

EXEMPTIONS = {
    # sustainability_technology v3 — the oldest filter, missing newer sections
    ("sustainability_technology", "v3", "missing_top_level:prefilter"),
    ("sustainability_technology", "v3", "missing_top_level:deployment"),
    ("sustainability_technology", "v3", "scoring_type:gatekeepers_is_list_not_dict"),

    # cultural-discovery v4 — no gatekeepers at all
    ("cultural-discovery", "v4", "scoring_missing:gatekeepers"),

    # investment-risk v6 — drift-y naming
    ("investment-risk", "v6", "missing_top_level:oracle"),
    ("investment-risk", "v6", "unexpected_top_level:ground_truth"),
    ("investment-risk", "v6", "missing_top_level:preprocessing"),
}


def _violations(cfg: dict) -> set[str]:
    """
    Return the set of issue codes this config triggers against the canonical
    schema. Empty set = fully conformant.
    """
    issues: set[str] = set()

    # Top-level section presence
    present = set(cfg.keys())
    for missing in REQUIRED_TOP_LEVEL - present:
        issues.add(f"missing_top_level:{missing}")
    # Flag well-known aliases explicitly — easier to read than a generic
    # "unexpected key" for keys we intentionally want to discourage.
    for alias in ("ground_truth",):
        if alias in present:
            issues.add(f"unexpected_top_level:{alias}")

    # Scoring section
    scoring = cfg.get("scoring")
    if scoring is None:
        issues.add("missing_top_level:scoring")
    elif isinstance(scoring, dict):
        scoring_keys = set(scoring.keys())
        for missing in REQUIRED_SCORING_KEYS - scoring_keys:
            issues.add(f"scoring_missing:{missing}")

        # gatekeepers must be a dict — list form is legacy
        gk = scoring.get("gatekeepers")
        if isinstance(gk, list):
            issues.add("scoring_type:gatekeepers_is_list_not_dict")
    else:
        issues.add("scoring_type:not_a_dict")

    return issues


def _filter_dir(filter_name: str, version: str) -> Path:
    return REPO_ROOT / "filters" / filter_name / version


def _config_path(filter_name: str, version: str) -> Path:
    return _filter_dir(filter_name, version) / "config.yaml"


def _load_config(filter_name: str, version: str) -> dict:
    with open(_config_path(filter_name, version), encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize("filter_name,version", ACTIVE_FILTERS,
                         ids=lambda x: x if isinstance(x, str) else f"v{x}")
def test_active_filter_config_conforms_or_is_exempt(filter_name, version):
    """
    Each active filter must conform to the canonical config schema, OR every
    deviation must be in EXEMPTIONS. New drift → test fails. Fixed drift with
    stale exemption → test fails (see separate test_no_stale_exemptions).
    """
    assert _config_path(filter_name, version).exists(), (
        f"{filter_name}/{version}: config.yaml missing — is ACTIVE_FILTERS stale?"
    )

    cfg = _load_config(filter_name, version)
    issues = _violations(cfg)

    filter_exemptions = {
        code for (f, v, code) in EXEMPTIONS
        if f == filter_name and v == version
    }
    unexpected = issues - filter_exemptions
    assert not unexpected, (
        f"{filter_name}/{version}: new config drift detected.\n"
        f"  Unexpected violations: {sorted(unexpected)}\n"
        f"  Fix the config, or add to EXEMPTIONS with justification."
    )


def test_no_stale_exemptions():
    """
    Every EXEMPTIONS entry must correspond to a real current violation. If
    someone fixes the drift but forgets to remove the exemption, this test
    tells them. Prevents the allow-list from silently growing obsolete.
    """
    stale = []
    for filter_name, version, code in EXEMPTIONS:
        if (filter_name, version) not in ACTIVE_FILTERS:
            stale.append((filter_name, version, code, "not in ACTIVE_FILTERS"))
            continue
        cfg = _load_config(filter_name, version)
        if code not in _violations(cfg):
            stale.append((filter_name, version, code, "no matching violation"))

    assert not stale, (
        "Stale EXEMPTIONS found — remove these entries:\n"
        + "\n".join(f"  {f}/{v}: {c}  ({reason})" for f, v, c, reason in stale)
    )
