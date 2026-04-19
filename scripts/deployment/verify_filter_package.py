"""
Verify a filter package is internally consistent and (optionally) matches what's on Hub.

Purpose: catch the failure mode in issue #44 — a new version directory (v2, v3, …)
copied from the previous version but with imports / repo_id / config still pointing
at the old version, so production runs v_new config × v_old weights.

Static checks (always run):
  1. inference_hub.py default repo_id ends with -v{N} matching the directory version.
  2. inference*.py files import from filters.{name}.v{N}.* matching the directory version.
  3. config.yaml filter.version matches the directory version.
  4. base_scorer.py FILTER_VERSION matches the directory version.

Hub check (run when --check-hub is passed):
  5. HfApi().repo_info(repo_id) succeeds.
  6. Hub last_modified is after local training_history.json mtime — catches the case where
     the repo exists but the new weights were never uploaded (stale Hub state).

Exit code is non-zero if any check fails. Intended to run as the first step of any
deploy action, and as a pre-commit gate for commits that claim "deploy".
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


class CheckFailure(Exception):
    pass


def parse_version_from_dir(filter_dir: Path) -> str:
    """filters/nature_recovery/v2 -> '2'. Raises if path doesn't match expected shape."""
    m = re.fullmatch(r"v(\d+)", filter_dir.name)
    if not m:
        raise CheckFailure(
            f"filter dir name must be v{{N}}, got: {filter_dir.name}"
        )
    return m.group(1)


def parse_filter_name_from_dir(filter_dir: Path) -> str:
    """filters/nature_recovery/v2 -> 'nature_recovery'."""
    return filter_dir.parent.name


def check_inference_hub_repo_id(filter_dir: Path, version: str) -> tuple[bool, str, str | None]:
    """Check inference_hub.py default repo_id ends with -v{version}.

    Returns (passed, message, extracted_repo_id).
    """
    path = filter_dir / "inference_hub.py"
    if not path.exists():
        return True, f"skip: {path.name} not present", None

    text = path.read_text(encoding="utf-8")
    # Match repo_id: str = "..."
    m = re.search(r'repo_id\s*:\s*str\s*=\s*"([^"]+)"', text)
    if not m:
        return False, f"{path.name}: no default repo_id found in signature", None

    repo_id = m.group(1)
    if re.search(rf"-v{version}(/|$)", repo_id):
        return True, f"{path.name}: repo_id ok ({repo_id})", repo_id
    return False, (
        f"{path.name}: repo_id {repo_id!r} does not end in -v{version} "
        f"(directory is v{version})"
    ), repo_id


def check_imports(filter_dir: Path, filter_name: str, version: str) -> list[tuple[bool, str]]:
    """Check inference*.py files import only from the matching vN subpackage."""
    results: list[tuple[bool, str]] = []
    pattern = re.compile(
        rf"filters\.{re.escape(filter_name)}\.v(\d+)\."
    )

    for name in ("inference.py", "inference_hub.py", "inference_hybrid.py"):
        path = filter_dir / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        bad: list[tuple[int, str]] = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            # Skip comments and docstrings heuristically - only check import statements
            stripped = line.lstrip()
            if not (stripped.startswith("from ") or stripped.startswith("import ")):
                continue
            for m in pattern.finditer(line):
                if m.group(1) != version:
                    bad.append((lineno, line.strip()))
        if bad:
            detail = "; ".join(f"L{ln}: {txt}" for ln, txt in bad)
            results.append((False, f"{name}: cross-version imports — {detail}"))
        else:
            results.append((True, f"{name}: imports ok"))
    return results


def check_config_yaml(filter_dir: Path, version: str) -> tuple[bool, str]:
    path = filter_dir / "config.yaml"
    if not path.exists():
        return True, "skip: config.yaml not present"

    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    declared = str(cfg.get("filter", {}).get("version", "")).split(".")[0]
    if declared == version:
        return True, f"config.yaml: filter.version={cfg['filter']['version']} ok"
    return False, (
        f"config.yaml: filter.version={cfg['filter']['version']!r} "
        f"does not match directory v{version}"
    )


def check_base_scorer_version(filter_dir: Path, version: str) -> tuple[bool, str]:
    path = filter_dir / "base_scorer.py"
    if not path.exists():
        return True, "skip: base_scorer.py not present"

    text = path.read_text(encoding="utf-8")
    m = re.search(r'FILTER_VERSION\s*=\s*"([^"]+)"', text)
    if not m:
        return True, "base_scorer.py: no FILTER_VERSION declared"
    declared = m.group(1).split(".")[0]
    if declared == version:
        return True, f"base_scorer.py: FILTER_VERSION={m.group(1)!r} ok"
    return False, (
        f"base_scorer.py: FILTER_VERSION={m.group(1)!r} "
        f"does not match directory v{version}"
    )


def check_hub(filter_dir: Path, repo_id: str | None, token: str | None) -> list[tuple[bool, str]]:
    """Verify repo exists on Hub and was updated after local training_history.json."""
    results: list[tuple[bool, str]] = []
    if not repo_id:
        results.append((False, "hub: cannot check — no repo_id extracted from inference_hub.py"))
        return results

    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import RepositoryNotFoundError
    except ImportError:
        results.append((False, "hub: huggingface_hub not installed"))
        return results

    api = HfApi(token=token)
    try:
        info = api.repo_info(repo_id=repo_id, repo_type="model")
    except RepositoryNotFoundError:
        results.append((False, f"hub: repo {repo_id!r} not found (or token lacks access)"))
        return results
    except Exception as e:
        results.append((False, f"hub: repo_info({repo_id!r}) failed — {type(e).__name__}: {e}"))
        return results

    results.append((True, f"hub: {repo_id} exists"))

    history_path = filter_dir / "training_history.json"
    if not history_path.exists():
        results.append((True, "hub: skip freshness check (no training_history.json)"))
        return results

    local_mtime = datetime.fromtimestamp(history_path.stat().st_mtime, tz=timezone.utc)
    hub_mtime = info.last_modified
    if hub_mtime is None:
        results.append((False, f"hub: repo has no last_modified timestamp — can't verify freshness"))
        return results

    # Hub last_modified must be AFTER local training finished.
    # If the local training artefact is newer than Hub, the upload hasn't happened.
    if hub_mtime >= local_mtime:
        results.append((
            True,
            f"hub: last_modified {hub_mtime.isoformat()} >= local training_history "
            f"{local_mtime.isoformat()}",
        ))
    else:
        results.append((
            False,
            f"hub: last_modified {hub_mtime.isoformat()} is OLDER than local "
            f"training_history {local_mtime.isoformat()} — weights likely not uploaded",
        ))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument(
        "--filter",
        type=Path,
        required=True,
        help="Path to filter version directory, e.g., filters/nature_recovery/v2",
    )
    parser.add_argument(
        "--check-hub",
        action="store_true",
        help="Also verify the Hub repo exists and is fresh (requires HF_TOKEN or hf auth login)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (or set HF_TOKEN env var)",
    )
    args = parser.parse_args()

    filter_dir: Path = args.filter.resolve()
    if not filter_dir.is_dir():
        print(f"[FAIL] filter dir does not exist: {filter_dir}")
        return 2

    try:
        version = parse_version_from_dir(filter_dir)
    except CheckFailure as e:
        print(f"[FAIL] {e}")
        return 2
    filter_name = parse_filter_name_from_dir(filter_dir)

    print(f"Verifying {filter_name} v{version} at {filter_dir}")
    print("-" * 60)

    all_checks: list[tuple[bool, str]] = []

    passed, msg, repo_id = check_inference_hub_repo_id(filter_dir, version)
    all_checks.append((passed, msg))
    all_checks.extend(check_imports(filter_dir, filter_name, version))
    all_checks.append(check_config_yaml(filter_dir, version))
    all_checks.append(check_base_scorer_version(filter_dir, version))

    if args.check_hub:
        token = args.token or os.environ.get("HF_TOKEN")
        all_checks.extend(check_hub(filter_dir, repo_id, token))

    for passed, msg in all_checks:
        prefix = "[OK]  " if passed else "[FAIL]"
        print(f"{prefix} {msg}")

    print("-" * 60)
    failed = sum(1 for p, _ in all_checks if not p)
    total = len(all_checks)
    if failed == 0:
        print(f"All {total} checks passed.")
        return 0
    print(f"{failed}/{total} checks failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
