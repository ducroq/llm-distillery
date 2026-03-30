"""
Cross-filter percentile normalization (ADR-014).

Maps weighted average scores to a common 0-10 scale using percentile rank,
so scores are comparable across filters. Fitted from production MEDIUM+ data.

Supersedes score_scale_factor (linear stretch). This is a non-linear monotonic
mapping — same mathematical pattern as isotonic calibration (ADR-008) but
applied on the weighted average across filters, not per-dimension within a filter.

Fitting requires numpy; inference uses only numpy.interp (same as calibration).
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def fit_normalization(
    weighted_averages: np.ndarray,
    filter_name: str = "",
    filter_version: str = "",
    source_description: str = "production MEDIUM+ data",
    n_bins: int = 200,
) -> Dict:
    """
    Fit percentile normalization from a set of weighted average scores.

    Builds an empirical CDF: for each score, what fraction of articles
    score at or below that value? Maps to 0-10 via percentile_rank * 10.

    Args:
        weighted_averages: 1D array of weighted average scores (MEDIUM+ articles)
        filter_name: Filter name for metadata
        filter_version: Filter version for metadata
        source_description: Description of data source
        n_bins: Number of evenly-spaced breakpoints for the lookup table.
                Higher = smoother interpolation, but 200 is more than enough.

    Returns:
        JSON-serializable normalization dict with lookup table and stats
    """
    wa = np.array(weighted_averages, dtype=np.float64)
    wa = wa[np.isfinite(wa)]

    if len(wa) < 10:
        raise ValueError(f"Need at least 10 scores to fit normalization, got {len(wa)}")

    wa_sorted = np.sort(wa)
    n = len(wa_sorted)

    # Build lookup table: evenly-spaced x values across the observed range,
    # with y = percentile rank at each x, scaled to 0-10.
    x_min = float(wa_sorted[0])
    x_max = float(wa_sorted[-1])
    x_points = np.linspace(x_min, x_max, n_bins)

    y_points = np.array([
        np.searchsorted(wa_sorted, x, side="right") / n * 10.0
        for x in x_points
    ])

    # Compute stats
    percentiles = {
        "p10": float(np.percentile(wa, 10)),
        "p25": float(np.percentile(wa, 25)),
        "p50": float(np.percentile(wa, 50)),
        "p75": float(np.percentile(wa, 75)),
        "p90": float(np.percentile(wa, 90)),
        "p95": float(np.percentile(wa, 95)),
        "p99": float(np.percentile(wa, 99)),
    }

    return {
        "method": "percentile",
        "filter_name": filter_name,
        "filter_version": filter_version,
        "fitted_from": source_description,
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "n_articles": n,
        "x": [round(float(v), 6) for v in x_points],
        "y": [round(float(v), 6) for v in y_points],
        "stats": {
            "raw_min": round(x_min, 4),
            "raw_max": round(x_max, 4),
            "raw_mean": round(float(np.mean(wa)), 4),
            "raw_std": round(float(np.std(wa)), 4),
            "percentiles": {k: round(v, 4) for k, v in percentiles.items()},
        },
    }


def apply_normalization(
    weighted_average: float,
    normalization_data: Dict,
) -> float:
    """
    Apply percentile normalization to a single weighted average score.

    Uses numpy.interp for fast linear interpolation between breakpoints.
    Scores below the observed range map to 0.0, above to 10.0.

    Args:
        weighted_average: Raw (calibrated) weighted average score
        normalization_data: Normalization dict from fit_normalization/load_normalization

    Returns:
        Normalized score on 0-10 scale (percentile rank * 10)
    """
    if not np.isfinite(weighted_average):
        return 0.0

    x = normalization_data["x"]
    y = normalization_data["y"]

    # np.interp clips to boundary values by default (out-of-bounds handling)
    return float(np.interp(weighted_average, x, y))


def apply_normalization_batch(
    weighted_averages: np.ndarray,
    normalization_data: Dict,
) -> np.ndarray:
    """
    Apply percentile normalization to an array of weighted average scores.

    Args:
        weighted_averages: 1D array of raw weighted average scores
        normalization_data: Normalization dict

    Returns:
        1D array of normalized 0-10 scores
    """
    x = normalization_data["x"]
    y = normalization_data["y"]
    return np.interp(weighted_averages, x, y)


def save_normalization(data: Dict, path: str) -> None:
    """Save normalization data to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Normalization saved to {path}")


def load_normalization(path: str) -> Optional[Dict]:
    """
    Load normalization data from JSON file.

    Returns None if file doesn't exist or is malformed.
    Scores pass through unchanged when normalization is unavailable.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Malformed normalization file at {path}: {e}. Normalization disabled.")
        return None

    if "x" not in data or "y" not in data:
        logger.error(f"Normalization file missing x/y arrays: {path}. Normalization disabled.")
        return None

    return data
