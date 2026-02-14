"""
Fit isotonic regression calibration for cultural-discovery v3.

Uses validation set to learn a monotonic mapping from predicted scores to oracle scores.
This corrects for systematic under-prediction of high scores.

Usage:
    python filters/cultural-discovery/v3/fit_calibration.py
"""

import json
import pickle
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt

# Paths
FILTER_DIR = Path(__file__).parent
VAL_DATA = Path("datasets/training/cultural-discovery_v3/val.jsonl")
TEST_DATA = Path("datasets/training/cultural-discovery_v3/test.jsonl")
CALIBRATION_FILE = FILTER_DIR / "calibration.pkl"

# Dimension weights (from config.yaml)
WEIGHTS = [0.25, 0.20, 0.25, 0.15, 0.15]  # discovery, heritage, cross_cultural, human, evidence


def load_data(path):
    """Load JSONL data."""
    articles = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            articles.append(json.loads(line))
    return articles


def compute_weighted_avg(scores, weights=WEIGHTS):
    """Compute weighted average score."""
    return sum(s * w for s, w in zip(scores, weights))


def get_predictions(articles, scorer):
    """Get model predictions for articles."""
    predictions = []
    for article in articles:
        result = scorer.score_article({
            "title": article["title"],
            "content": article["content"],
        })
        if result["passed_prefilter"] and result["scores"]:
            pred_scores = [result["scores"][dim] for dim in [
                "discovery_novelty", "heritage_significance", "cross_cultural_connection",
                "human_resonance", "evidence_quality"
            ]]
            pred_avg = compute_weighted_avg(pred_scores)
            oracle_avg = compute_weighted_avg(article["labels"])
            predictions.append({
                "id": article["id"],
                "oracle_avg": oracle_avg,
                "pred_avg": pred_avg,
                "oracle_scores": article["labels"],
                "pred_scores": pred_scores,
            })
    return predictions


def fit_calibration(predictions):
    """Fit isotonic regression on predictions."""
    pred_avgs = np.array([p["pred_avg"] for p in predictions])
    oracle_avgs = np.array([p["oracle_avg"] for p in predictions])

    # Fit isotonic regression (monotonically increasing)
    iso_reg = IsotonicRegression(out_of_bounds="clip")
    iso_reg.fit(pred_avgs, oracle_avgs)

    return iso_reg


def evaluate_calibration(predictions, iso_reg):
    """Evaluate calibration improvement."""
    pred_avgs = np.array([p["pred_avg"] for p in predictions])
    oracle_avgs = np.array([p["oracle_avg"] for p in predictions])
    calibrated = iso_reg.predict(pred_avgs)

    # Overall metrics
    mae_before = np.mean(np.abs(pred_avgs - oracle_avgs))
    mae_after = np.mean(np.abs(calibrated - oracle_avgs))

    # Tier-level metrics
    tiers = {"LOW": (0, 4), "MEDIUM": (4, 7), "HIGH": (7, 11)}
    tier_metrics = {}

    for tier, (low, high) in tiers.items():
        mask = (oracle_avgs >= low) & (oracle_avgs < high)
        if mask.sum() > 0:
            tier_mae_before = np.mean(np.abs(pred_avgs[mask] - oracle_avgs[mask]))
            tier_mae_after = np.mean(np.abs(calibrated[mask] - oracle_avgs[mask]))
            tier_bias_before = np.mean(pred_avgs[mask] - oracle_avgs[mask])
            tier_bias_after = np.mean(calibrated[mask] - oracle_avgs[mask])
            tier_metrics[tier] = {
                "count": int(mask.sum()),
                "mae_before": tier_mae_before,
                "mae_after": tier_mae_after,
                "bias_before": tier_bias_before,
                "bias_after": tier_bias_after,
            }

    return {
        "overall_mae_before": mae_before,
        "overall_mae_after": mae_after,
        "improvement": (mae_before - mae_after) / mae_before * 100,
        "tiers": tier_metrics,
    }


def plot_calibration(predictions, iso_reg, output_path):
    """Plot calibration curve and scatter."""
    pred_avgs = np.array([p["pred_avg"] for p in predictions])
    oracle_avgs = np.array([p["oracle_avg"] for p in predictions])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Scatter plot
    ax = axes[0]
    ax.scatter(pred_avgs, oracle_avgs, alpha=0.3, s=10, label="Data points")

    # Plot calibration curve
    x_line = np.linspace(0, 10, 100)
    y_line = iso_reg.predict(x_line)
    ax.plot(x_line, y_line, "r-", linewidth=2, label="Isotonic calibration")
    ax.plot([0, 10], [0, 10], "k--", alpha=0.5, label="Perfect calibration")

    ax.set_xlabel("Predicted Score")
    ax.set_ylabel("Oracle Score")
    ax.set_title("Calibration: Predicted vs Oracle")
    ax.legend()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3)

    # Right: Before/After comparison
    ax = axes[1]
    calibrated = iso_reg.predict(pred_avgs)

    ax.scatter(oracle_avgs, pred_avgs, alpha=0.3, s=10, c="blue", label="Before calibration")
    ax.scatter(oracle_avgs, calibrated, alpha=0.3, s=10, c="red", label="After calibration")
    ax.plot([0, 10], [0, 10], "k--", alpha=0.5, label="Perfect")

    ax.set_xlabel("Oracle Score")
    ax.set_ylabel("Model Score")
    ax.set_title("Before vs After Calibration")
    ax.legend()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    print("=" * 70)
    print("Cultural Discovery v3 - Isotonic Calibration")
    print("=" * 70)

    # Load scorer
    print("\nLoading model...")
    from inference_hub import CulturalDiscoveryScorerHub

    # Get token
    import configparser
    config = configparser.ConfigParser()
    config.read("config/credentials/secrets.ini")
    token = config.get("api_keys", "huggingface_token", fallback=None)

    scorer = CulturalDiscoveryScorerHub(
        repo_id="jeergrvgreg/cultural-discovery-v3",
        token=token,
        use_prefilter=False,  # Don't filter for calibration
    )

    # Load validation data
    print(f"\nLoading validation data from {VAL_DATA}...")
    val_articles = load_data(VAL_DATA)
    print(f"Loaded {len(val_articles)} validation articles")

    # Get predictions
    print("\nRunning model on validation set...")
    val_predictions = get_predictions(val_articles, scorer)
    print(f"Got predictions for {len(val_predictions)} articles")

    # Fit calibration
    print("\nFitting isotonic regression...")
    iso_reg = fit_calibration(val_predictions)

    # Evaluate on validation set
    print("\n" + "-" * 70)
    print("Validation Set Results")
    print("-" * 70)
    val_metrics = evaluate_calibration(val_predictions, iso_reg)

    print(f"\nOverall MAE: {val_metrics['overall_mae_before']:.3f} -> {val_metrics['overall_mae_after']:.3f} ({val_metrics['improvement']:+.1f}%)")

    print("\nPer-Tier Results:")
    print(f"{'Tier':<8} {'Count':<8} {'MAE Before':<12} {'MAE After':<12} {'Bias Before':<12} {'Bias After':<12}")
    print("-" * 70)
    for tier in ["LOW", "MEDIUM", "HIGH"]:
        if tier in val_metrics["tiers"]:
            m = val_metrics["tiers"][tier]
            print(f"{tier:<8} {m['count']:<8} {m['mae_before']:<12.3f} {m['mae_after']:<12.3f} {m['bias_before']:<+12.3f} {m['bias_after']:<+12.3f}")

    # Load and evaluate on test set
    print(f"\nLoading test data from {TEST_DATA}...")
    test_articles = load_data(TEST_DATA)
    print(f"Loaded {len(test_articles)} test articles")

    print("\nRunning model on test set...")
    test_predictions = get_predictions(test_articles, scorer)
    print(f"Got predictions for {len(test_predictions)} articles")

    print("\n" + "-" * 70)
    print("Test Set Results (held-out)")
    print("-" * 70)
    test_metrics = evaluate_calibration(test_predictions, iso_reg)

    print(f"\nOverall MAE: {test_metrics['overall_mae_before']:.3f} -> {test_metrics['overall_mae_after']:.3f} ({test_metrics['improvement']:+.1f}%)")

    print("\nPer-Tier Results:")
    print(f"{'Tier':<8} {'Count':<8} {'MAE Before':<12} {'MAE After':<12} {'Bias Before':<12} {'Bias After':<12}")
    print("-" * 70)
    for tier in ["LOW", "MEDIUM", "HIGH"]:
        if tier in test_metrics["tiers"]:
            m = test_metrics["tiers"][tier]
            print(f"{tier:<8} {m['count']:<8} {m['mae_before']:<12.3f} {m['mae_after']:<12.3f} {m['bias_before']:<+12.3f} {m['bias_after']:<+12.3f}")

    # Save calibration model
    print(f"\nSaving calibration model to {CALIBRATION_FILE}...")
    with open(CALIBRATION_FILE, "wb") as f:
        pickle.dump(iso_reg, f)
    print("Done!")

    # Plot
    plot_path = FILTER_DIR / "calibration_plot.png"
    print(f"\nGenerating calibration plot...")
    plot_calibration(val_predictions + test_predictions, iso_reg, plot_path)

    # Print calibration lookup table for reference
    print("\n" + "-" * 70)
    print("Calibration Lookup Table")
    print("-" * 70)
    print("Predicted -> Calibrated")
    for x in range(0, 11):
        y = iso_reg.predict([x])[0]
        print(f"  {x:.1f} -> {y:.2f}")

    print("\n" + "=" * 70)
    print("DONE - Calibration fitted and saved")
    print("=" * 70)


if __name__ == "__main__":
    main()
