#!/usr/bin/env python3
"""
Detailed error analysis for multilingual-e5-large to understand where it can be trusted.
"""

import json
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class MLPProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def score_to_tier(scores: np.ndarray) -> np.ndarray:
    """Convert dimension scores to tier (0-3 scale based on average)."""
    avg_scores = np.mean(scores, axis=1)
    tiers = np.zeros_like(avg_scores, dtype=int)
    tiers[avg_scores >= 7] = 3  # Tier 1 (best)
    tiers[(avg_scores >= 5) & (avg_scores < 7)] = 2  # Tier 2
    tiers[(avg_scores >= 3) & (avg_scores < 5)] = 1  # Tier 3
    # tiers < 3 stay at 0 (Tier 4 / rejected)
    return tiers


def main():
    results_dir = Path('research/embedding_vs_finetuning/results')
    embeddings_dir = Path('research/embedding_vs_finetuning/embeddings')

    # Load all embeddings
    train_data = np.load(embeddings_dir / 'uplifting_v5_multilingual-e5-large_train.npz')
    X_train = train_data['embeddings']
    y_train = train_data['labels']

    val_data = np.load(embeddings_dir / 'uplifting_v5_multilingual-e5-large_val.npz')
    X_val = val_data['embeddings']
    y_val = val_data['labels']

    test_data = np.load(embeddings_dir / 'uplifting_v5_multilingual-e5-large_test.npz')
    X_test = test_data['embeddings']
    y_test = test_data['labels']

    # Train a quick MLP probe locally (faster than loading saved model)
    print("Training MLP probe on embeddings...")
    model = MLPProbe(X_train.shape[1], y_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.MSELoss()

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)

    best_val_mae = float('inf')
    best_state = None
    patience = 0

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t).numpy()
            val_mae = mean_absolute_error(y_val, val_pred)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 15:
                break

    model.load_state_dict(best_state)
    model.eval()
    print(f"Trained MLP: val_mae = {best_val_mae:.4f}")

    # Generate predictions
    with torch.no_grad():
        X_t = torch.FloatTensor(X_test)
        predictions = model(X_t).numpy()

    # Dimension names
    dim_names = [
        'human_wellbeing_impact',
        'social_cohesion_impact',
        'justice_rights_impact',
        'evidence_level',
        'benefit_distribution',
        'change_durability'
    ]

    print("=" * 80)
    print("MULTILINGUAL-E5-LARGE: DETAILED ERROR ANALYSIS")
    print("=" * 80)

    # Overall stats
    overall_mae = mean_absolute_error(y_test, predictions)
    print(f"\nOverall MAE: {overall_mae:.4f}")
    print(f"Test samples: {len(y_test)}")

    # Per-article error distribution
    article_errors = np.mean(np.abs(y_test - predictions), axis=1)

    print("\n" + "-" * 60)
    print("ERROR DISTRIBUTION (per article)")
    print("-" * 60)
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(article_errors, p)
        print(f"  {p:3d}th percentile: {val:.3f}")

    # Percentage within thresholds
    print("\n" + "-" * 60)
    print("PREDICTIONS WITHIN THRESHOLD (per dimension)")
    print("-" * 60)
    thresholds = [0.5, 1.0, 1.5, 2.0]
    dim_errors = np.abs(y_test - predictions)

    for thresh in thresholds:
        within = (dim_errors <= thresh).mean() * 100
        print(f"  Within ±{thresh}: {within:.1f}%")

    # Per-dimension analysis
    print("\n" + "-" * 60)
    print("PER-DIMENSION BREAKDOWN")
    print("-" * 60)

    for i, dim in enumerate(dim_names):
        dim_mae = mean_absolute_error(y_test[:, i], predictions[:, i])
        within_1 = (np.abs(y_test[:, i] - predictions[:, i]) <= 1.0).mean() * 100
        print(f"  {dim:30s}: MAE={dim_mae:.3f}, within ±1.0: {within_1:.1f}%")

    # Error by actual score range
    print("\n" + "-" * 60)
    print("ERROR BY ACTUAL SCORE RANGE (all dimensions pooled)")
    print("-" * 60)

    ranges = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]

    for low, high in ranges:
        mask = (y_test >= low) & (y_test < high)
        if mask.sum() > 0:
            range_mae = np.abs(y_test[mask] - predictions[mask]).mean()
            count = mask.sum()
            print(f"  [{low:2d}-{high:2d}): MAE={range_mae:.3f} (n={count:5d})")

    # Tier classification analysis
    print("\n" + "-" * 60)
    print("TIER CLASSIFICATION ACCURACY")
    print("-" * 60)
    print("Tiers: 3=best (avg>=7), 2=good (5-7), 1=fair (3-5), 0=poor (<3)")

    actual_tiers = score_to_tier(y_test)
    pred_tiers = score_to_tier(predictions)

    tier_accuracy = (actual_tiers == pred_tiers).mean() * 100
    print(f"\n  Exact tier match: {tier_accuracy:.1f}%")

    # Within 1 tier
    within_1_tier = (np.abs(actual_tiers - pred_tiers) <= 1).mean() * 100
    print(f"  Within ±1 tier:   {within_1_tier:.1f}%")

    # Tier confusion matrix
    print("\n  Confusion Matrix (actual vs predicted):")
    cm = confusion_matrix(actual_tiers, pred_tiers, labels=[0, 1, 2, 3])
    print("              Predicted")
    print("              T4   T3   T2   T1")
    tier_labels = ['T4 (0)', 'T3 (1)', 'T2 (2)', 'T1 (3)']
    for i, label in enumerate(tier_labels):
        row = cm[i]
        print(f"  Actual {label}: {row[0]:4d} {row[1]:4d} {row[2]:4d} {row[3]:4d}")

    # Tier distribution
    print("\n  Tier Distribution:")
    for tier in [3, 2, 1, 0]:
        actual_count = (actual_tiers == tier).sum()
        pred_count = (pred_tiers == tier).sum()
        print(f"    Tier {tier}: Actual={actual_count:4d}, Predicted={pred_count:4d}")

    # High-confidence predictions analysis
    print("\n" + "-" * 60)
    print("FILTERING SCENARIOS")
    print("-" * 60)

    # Scenario 1: Using as prefilter (reject low-scoring)
    print("\n1. PREFILTER: Reject articles with predicted avg < 3")
    pred_avg = predictions.mean(axis=1)
    actual_avg = y_test.mean(axis=1)

    reject_mask = pred_avg < 3
    keep_mask = pred_avg >= 3

    # What would be rejected?
    rejected_actual_good = ((actual_avg >= 5) & reject_mask).sum()
    rejected_total = reject_mask.sum()
    kept_total = keep_mask.sum()
    kept_actually_bad = ((actual_avg < 3) & keep_mask).sum()

    print(f"   Would reject: {rejected_total} articles ({rejected_total/len(y_test)*100:.1f}%)")
    print(f"   False negatives (good articles rejected): {rejected_actual_good}")
    print(f"   Would keep: {kept_total} articles")
    print(f"   False positives (bad articles kept): {kept_actually_bad}")

    # Scenario 2: Top-tier selection
    print("\n2. TOP-TIER SELECTION: Select articles with predicted avg >= 7")
    select_mask = pred_avg >= 7

    selected_total = select_mask.sum()
    selected_actually_top = ((actual_avg >= 7) & select_mask).sum()
    selected_actually_good = ((actual_avg >= 5) & select_mask).sum()
    missed_top = ((actual_avg >= 7) & ~select_mask).sum()

    if selected_total > 0:
        precision_top = selected_actually_top / selected_total * 100
        precision_good = selected_actually_good / selected_total * 100
        print(f"   Selected: {selected_total} articles")
        print(f"   Precision (actual avg>=7): {precision_top:.1f}%")
        print(f"   Precision (actual avg>=5): {precision_good:.1f}%")
        print(f"   Recall (missed top-tier): {missed_top} articles")
    else:
        print("   No articles selected at this threshold")

    # Scenario 3: Medium threshold
    print("\n3. QUALITY FILTER: Select articles with predicted avg >= 5")
    select_mask = pred_avg >= 5

    selected_total = select_mask.sum()
    selected_actually_good = ((actual_avg >= 5) & select_mask).sum()
    selected_actually_fair = ((actual_avg >= 3) & select_mask).sum()
    missed_good = ((actual_avg >= 5) & ~select_mask).sum()

    if selected_total > 0:
        precision_good = selected_actually_good / selected_total * 100
        precision_fair = selected_actually_fair / selected_total * 100
        recall_good = selected_actually_good / (actual_avg >= 5).sum() * 100
        print(f"   Selected: {selected_total} articles ({selected_total/len(y_test)*100:.1f}%)")
        print(f"   Precision (actual avg>=5): {precision_good:.1f}%")
        print(f"   Precision (actual avg>=3): {precision_fair:.1f}%")
        print(f"   Recall (of actually good): {recall_good:.1f}%")

    # Regression to mean analysis
    print("\n" + "-" * 60)
    print("REGRESSION TO MEAN ANALYSIS")
    print("-" * 60)

    # Check prediction range compression
    print(f"   Actual score range:    [{y_test.min():.2f}, {y_test.max():.2f}]")
    print(f"   Predicted score range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    print(f"   Actual std:     {y_test.std():.3f}")
    print(f"   Predicted std:  {predictions.std():.3f}")
    print(f"   Compression:    {predictions.std() / y_test.std():.2f}x")

    # Bias analysis
    print("\n" + "-" * 60)
    print("BIAS ANALYSIS (predicted - actual)")
    print("-" * 60)

    bias = predictions - y_test

    for i, dim in enumerate(dim_names):
        dim_bias = bias[:, i].mean()
        print(f"  {dim:30s}: bias={dim_bias:+.3f}")

    overall_bias = bias.mean()
    print(f"\n  Overall bias: {overall_bias:+.3f}")

    # Summary for use case decision
    print("\n" + "=" * 80)
    print("SUMMARY: CAN E5-LARGE BE USED?")
    print("=" * 80)

    print("""
For PREFILTERING (reject obvious non-uplifting):
  - Rejecting pred_avg < 3 loses few good articles ({} false negatives)
  - Can safely reduce workload by ~{:.0f}%
  - VERDICT: SUITABLE for coarse prefiltering

For RANKING/SCORING:
  - Within ±1.0 of true score {:.1f}% of the time
  - Tier accuracy only {:.1f}%
  - Strong regression to mean (compresses range)
  - VERDICT: NOT SUITABLE for precise scoring

For TOP-TIER SELECTION:
  - Precision varies by threshold
  - Better for "definitely good" than "best of best"
  - VERDICT: MARGINAL - use with conservative thresholds
""".format(
        rejected_actual_good,
        rejected_total/len(y_test)*100,
        (dim_errors <= 1.0).mean() * 100,
        tier_accuracy
    ))


if __name__ == '__main__':
    main()
