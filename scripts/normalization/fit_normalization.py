"""
Fit cross-filter percentile normalization from production MEDIUM+ data (ADR-014).

Reads weighted average scores from NexusMind filtered output on sadalsuud,
fits a percentile CDF, and saves normalization.json to the filter directory.

Usage:
    # From local JSONL files (e.g., after scp from sadalsuud)
    PYTHONPATH=. python scripts/normalization/fit_normalization.py \
        --filter filters/nature_recovery/v1 \
        --data-dir /path/to/filtered/nature_recovery

    # From sadalsuud directly via SSH
    PYTHONPATH=. python scripts/normalization/fit_normalization.py \
        --filter filters/nature_recovery/v1 \
        --ssh sadalsuud \
        --remote-dir /home/jeroen/local_dev/NexusMind/data/filtered/nature_recovery
"""

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from filters.common.score_normalization import fit_normalization, save_normalization

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_weighted_averages_local(data_dir: Path, filter_name: str, all_tiers: bool = False) -> list:
    """Load weighted averages from local filtered JSONL files."""
    was = []

    if all_tiers:
        # Read from root-level filtered_*.jsonl files (all scored articles)
        jsonl_files = sorted(data_dir.glob("filtered_*.jsonl"))
    else:
        # Read from high/ and medium/ subdirs only
        jsonl_files = []
        for tier_dir in ["high", "medium"]:
            tier_path = data_dir / tier_dir
            if tier_path.is_dir():
                jsonl_files.extend(sorted(tier_path.glob("filtered_*.jsonl")))

    for jsonl_file in jsonl_files:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    article = json.loads(line)
                    attrs = article.get("nexus_mind_attributes", {})
                    for key, analysis in attrs.items():
                        if isinstance(analysis, dict) and "weighted_average" in analysis:
                            was.append(analysis["weighted_average"])
                except (json.JSONDecodeError, KeyError):
                    continue

    return was


def load_weighted_averages_ssh(ssh_host: str, remote_dir: str, all_tiers: bool = False) -> list:
    """Load weighted averages from a remote host via SSH."""
    # Write extraction script to temp file, scp to remote, execute, retrieve results
    script_content = """import json, glob, os, sys
remote_dir = sys.argv[1]
all_tiers = sys.argv[2] == "1" if len(sys.argv) > 2 else False
was = []
if all_tiers:
    files = sorted(glob.glob(os.path.join(remote_dir, "filtered_*.jsonl")))
else:
    files = []
    for tier in ["high", "medium"]:
        tier_dir = os.path.join(remote_dir, tier)
        if not os.path.isdir(tier_dir):
            continue
        files.extend(sorted(glob.glob(os.path.join(tier_dir, "filtered_*.jsonl"))))
for fp in files:
    with open(fp) as f:
        for line in f:
            try:
                d = json.loads(line)
                attrs = d.get("nexus_mind_attributes", {})
                for k, v in attrs.items():
                    if isinstance(v, dict) and "weighted_average" in v:
                        was.append(v["weighted_average"])
            except:
                pass
for w in was:
    print(w)
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script_content)
        local_script = f.name

    remote_script = "/tmp/_extract_wa.py"
    try:
        subprocess.run(["scp", local_script, f"{ssh_host}:{remote_script}"],
                       capture_output=True, timeout=30, check=True)
        all_tiers_flag = "1" if all_tiers else "0"
        result = subprocess.run(
            ["ssh", ssh_host, "python3", remote_script, remote_dir, all_tiers_flag],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            logger.error(f"SSH command failed: {result.stderr}")
            sys.exit(1)
    finally:
        Path(local_script).unlink(missing_ok=True)
        subprocess.run(["ssh", ssh_host, "rm", "-f", remote_script],
                       capture_output=True, timeout=15)

    was = []
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            was.append(float(line))
    return was


def main():
    parser = argparse.ArgumentParser(
        description="Fit cross-filter percentile normalization from production data (ADR-014)"
    )
    parser.add_argument(
        "--filter", type=Path, required=True,
        help="Path to filter directory (e.g., filters/nature_recovery/v1)",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Local directory with high/ and medium/ subdirs of filtered JSONL files",
    )
    parser.add_argument(
        "--ssh", type=str, default=None,
        help="SSH host to read production data from (e.g., sadalsuud)",
    )
    parser.add_argument(
        "--remote-dir", type=str, default=None,
        help="Remote directory on SSH host (e.g., /home/jeroen/local_dev/NexusMind/data/filtered/nature_recovery)",
    )
    parser.add_argument(
        "--n-bins", type=int, default=200,
        help="Number of breakpoints in the lookup table (default: 200)",
    )
    parser.add_argument(
        "--all-tiers", action="store_true",
        help="Include all scored articles (not just MEDIUM+). "
             "Reads root-level filtered_*.jsonl instead of high/medium subdirs.",
    )
    args = parser.parse_args()

    # Validate inputs
    if not args.filter.is_dir():
        logger.error(f"Filter directory not found: {args.filter}")
        sys.exit(1)

    if args.ssh and not args.remote_dir:
        logger.error("--remote-dir is required when using --ssh")
        sys.exit(1)

    if not args.ssh and not args.data_dir:
        logger.error("Either --data-dir or --ssh + --remote-dir is required")
        sys.exit(1)

    # Load config for filter name/version
    config_path = args.filter / "config.yaml"
    filter_name = args.filter.parent.name
    filter_version = args.filter.name

    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        filter_info = config.get("filter", {})
        filter_name = filter_info.get("name", filter_name)
        filter_version = str(filter_info.get("version", filter_version))

    logger.info(f"Filter: {filter_name} v{filter_version}")

    # Load production weighted averages
    tier_label = "ALL tiers" if args.all_tiers else "MEDIUM+"
    if args.ssh:
        logger.info(f"Loading production data ({tier_label}) from {args.ssh}:{args.remote_dir}")
        source_desc = f"production {tier_label} from {args.ssh}:{args.remote_dir}"
        was = load_weighted_averages_ssh(args.ssh, args.remote_dir, all_tiers=args.all_tiers)
    else:
        logger.info(f"Loading production data ({tier_label}) from {args.data_dir}")
        source_desc = f"production {tier_label} from {args.data_dir}"
        was = load_weighted_averages_local(args.data_dir, filter_name, all_tiers=args.all_tiers)

    if len(was) < 10:
        logger.error(f"Only {len(was)} weighted averages found — need at least 10")
        sys.exit(1)

    logger.info(f"Loaded {len(was)} weighted averages")

    # Fit normalization
    wa_array = np.array(was, dtype=np.float64)
    norm_data = fit_normalization(
        wa_array,
        filter_name=filter_name,
        filter_version=filter_version,
        source_description=source_desc,
        n_bins=args.n_bins,
    )

    # Report
    stats = norm_data["stats"]
    pcts = stats["percentiles"]
    logger.info(f"\nNormalization fitted on {norm_data['n_articles']} articles")
    logger.info(f"  Raw WA range: {stats['raw_min']:.2f} - {stats['raw_max']:.2f}")
    logger.info(f"  Raw WA mean:  {stats['raw_mean']:.2f} (std {stats['raw_std']:.2f})")
    logger.info(f"  Percentiles (raw):  p25={pcts['p25']:.2f}  p50={pcts['p50']:.2f}  "
                f"p75={pcts['p75']:.2f}  p90={pcts['p90']:.2f}  p95={pcts['p95']:.2f}")

    # Show what key raw scores map to after normalization
    logger.info(f"\n  Sample mappings (raw -> normalized):")
    for raw in [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0]:
        norm = float(np.interp(raw, norm_data["x"], norm_data["y"]))
        logger.info(f"    {raw:.1f} -> {norm:.2f}")

    # Save
    output_path = args.filter / "normalization.json"
    save_normalization(norm_data, str(output_path))
    logger.info(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
