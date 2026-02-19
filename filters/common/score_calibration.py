"""
Post-hoc score calibration using isotonic regression.

Student models trained with MSE loss compress the oracle's score range.
Per-dimension isotonic regression learns a monotonic mapping from
student_predicted -> oracle_actual on the validation set, decompressing
scores at inference time.

Fitting requires sklearn; inference uses only numpy.interp (zero sklearn
dependency at scoring time).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def fit_calibration(
    oracle_scores: np.ndarray,
    student_scores: np.ndarray,
    dimension_names: List[str],
    filter_name: str = "",
    filter_version: str = "",
    split_name: str = "val",
) -> Dict:
    """
    Fit per-dimension isotonic regression calibration.

    Args:
        oracle_scores: (n_samples, n_dims) array of oracle labels
        student_scores: (n_samples, n_dims) array of student predictions
        dimension_names: List of dimension names matching column order
        filter_name: Filter name for metadata
        filter_version: Filter version for metadata
        split_name: Name of data split used for fitting

    Returns:
        JSON-serializable calibration dict with breakpoints and stats
    """
    from sklearn.isotonic import IsotonicRegression

    n_samples, n_dims = oracle_scores.shape
    if student_scores.shape != oracle_scores.shape:
        raise ValueError(
            f"Shape mismatch: oracle {oracle_scores.shape} vs student {student_scores.shape}"
        )
    if len(dimension_names) != n_dims:
        raise ValueError(
            f"Got {len(dimension_names)} dimension names but {n_dims} columns"
        )

    dimensions = {}
    per_dim_stats = {}

    total_mae_before = 0.0
    total_mae_after = 0.0

    for i, dim_name in enumerate(dimension_names):
        oracle_col = oracle_scores[:, i]
        student_col = student_scores[:, i]

        # Fit isotonic regression: student -> oracle
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(student_col, oracle_col)

        # Extract breakpoints
        x_bp = iso.X_thresholds_.tolist()
        y_bp = iso.y_thresholds_.tolist()

        dimensions[dim_name] = {"x": x_bp, "y": y_bp}

        # Compute stats
        mae_before = float(np.mean(np.abs(oracle_col - student_col)))
        calibrated = iso.predict(student_col)
        mae_after = float(np.mean(np.abs(oracle_col - calibrated)))

        per_dim_stats[dim_name] = {
            "mae_before": round(mae_before, 4),
            "mae_after": round(mae_after, 4),
            "student_min": round(float(student_col.min()), 4),
            "student_max": round(float(student_col.max()), 4),
            "student_std": round(float(student_col.std()), 4),
            "calibrated_min": round(float(calibrated.min()), 4),
            "calibrated_max": round(float(calibrated.max()), 4),
            "calibrated_std": round(float(calibrated.std()), 4),
            "oracle_min": round(float(oracle_col.min()), 4),
            "oracle_max": round(float(oracle_col.max()), 4),
            "oracle_std": round(float(oracle_col.std()), 4),
            "n_breakpoints": len(x_bp),
        }

        total_mae_before += mae_before
        total_mae_after += mae_after

    avg_mae_before = total_mae_before / n_dims
    avg_mae_after = total_mae_after / n_dims

    return {
        "method": "isotonic",
        "fitted_on": split_name,
        "n_samples": n_samples,
        "filter_name": filter_name,
        "filter_version": filter_version,
        "dimensions": dimensions,
        "stats": {
            "mae_before": round(avg_mae_before, 4),
            "mae_after": round(avg_mae_after, 4),
            "per_dimension": per_dim_stats,
        },
    }


def apply_calibration(
    raw_scores: np.ndarray,
    calibration_data: Dict,
    dimension_names: List[str],
) -> np.ndarray:
    """
    Apply calibration to raw model scores using numpy.interp.

    Thread-safe: stateless, operates on passed-in data only.
    Zero sklearn dependency — uses precomputed breakpoints.

    Args:
        raw_scores: 1D array of raw scores (one per dimension)
        calibration_data: Calibration dict from fit_calibration/load_calibration
        dimension_names: Dimension names matching score order

    Returns:
        Calibrated scores as 1D numpy array (same shape as input)
    """
    calibrated = raw_scores.copy()
    dims = calibration_data.get("dimensions", {})

    for i, dim_name in enumerate(dimension_names):
        if dim_name not in dims:
            continue
        val = float(raw_scores[i])
        if not np.isfinite(val):
            logger.warning(f"Non-finite raw score for dimension '{dim_name}': {val}")
            continue
        bp = dims[dim_name]
        calibrated[i] = np.interp(val, bp["x"], bp["y"])

    return calibrated


def save_calibration(data: Dict, path: str) -> None:
    """Save calibration data to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Calibration saved to {path}")


def load_calibration(path: str) -> Optional[Dict]:
    """
    Load calibration data from JSON file.

    Returns None if file doesn't exist or is malformed.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Malformed calibration file at {path}: {e}. Calibration disabled.")
        return None
    return data
