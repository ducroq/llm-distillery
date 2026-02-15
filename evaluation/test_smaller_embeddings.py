"""Test smaller embedding models for Stage 1 filtering.

Compares embedding models on:
1. Embedding speed (ms/article)
2. Probe accuracy (MAE, weighted avg MAE)
3. Binary classification at threshold 4.5 (what we actually care about)

Uses cached labels from calibration data.
"""
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

DIMENSION_NAMES = [
    "human_wellbeing_impact", "social_cohesion_impact", "justice_rights_impact",
    "evidence_level", "benefit_distribution", "change_durability",
]
DIMENSION_WEIGHTS = {
    "human_wellbeing_impact": 0.25, "social_cohesion_impact": 0.15,
    "justice_rights_impact": 0.10, "evidence_level": 0.20,
    "benefit_distribution": 0.20, "change_durability": 0.10,
}
WEIGHTS_ARRAY = np.array([DIMENSION_WEIGHTS[d] for d in DIMENSION_NAMES])

MODELS_TO_TEST = [
    "intfloat/multilingual-e5-small",   # ~118M params, 384d
    "intfloat/multilingual-e5-base",    # ~278M params, 768d
    "intfloat/multilingual-e5-large",   # ~560M params, 1024d (current)
]


class MLPProbe(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=[256, 128], dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_probe(train_emb, train_labels, val_emb, val_labels, device="cpu"):
    """Train MLP probe and return best model + metrics."""
    scaler = StandardScaler()
    train_emb_s = scaler.fit_transform(train_emb)
    val_emb_s = scaler.transform(val_emb)

    train_X = torch.FloatTensor(train_emb_s)
    train_y = torch.FloatTensor(train_labels)
    val_X = torch.FloatTensor(val_emb_s).to(device)
    val_y = torch.FloatTensor(val_labels).to(device)

    model = MLPProbe(
        input_dim=train_emb.shape[1],
        output_dim=train_labels.shape[1],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.L1Loss()
    loader = DataLoader(TensorDataset(train_X, train_y), batch_size=64, shuffle=True)

    best_val_mae = float("inf")
    best_state = None
    patience = 0

    for epoch in range(200):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_X)
            val_mae = torch.mean(torch.abs(val_pred - val_y)).item()

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if patience >= 20:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, scaler, best_val_mae


def evaluate_model(model, scaler, val_emb, val_labels, tiers_val, device="cpu"):
    """Evaluate probe on binary classification at threshold 4.5."""
    val_emb_s = scaler.transform(val_emb)

    with torch.no_grad():
        preds = model(torch.FloatTensor(val_emb_s).to(device)).cpu().numpy()

    # Compute weighted averages
    probe_wavgs = []
    gt_wavgs = []
    for i in range(len(val_labels)):
        raw = preds[i]
        scores = np.clip(raw, 0, 10)
        probe_wavgs.append(np.sum(scores * WEIGHTS_ARRAY))
        gt_wavgs.append(np.sum(val_labels[i] * WEIGHTS_ARRAY))

    probe_wavgs = np.array(probe_wavgs)
    gt_wavgs = np.array(gt_wavgs)

    # Weighted avg metrics
    wavg_mae = np.mean(np.abs(probe_wavgs - gt_wavgs))
    wavg_bias = np.mean(probe_wavgs) - np.mean(gt_wavgs)

    # Binary classification at various thresholds
    results = {"wavg_mae": wavg_mae, "wavg_bias": wavg_bias, "thresholds": {}}

    for threshold in [4.0, 4.5, 5.0]:
        gt_positive = gt_wavgs >= threshold  # "interesting" articles
        probe_positive = probe_wavgs >= threshold

        # What matters: how many gt_positive does the probe catch?
        tp = np.sum(gt_positive & probe_positive)
        fn = np.sum(gt_positive & ~probe_positive)  # missed good articles
        fp = np.sum(~gt_positive & probe_positive)   # unnecessary Stage 2
        tn = np.sum(~gt_positive & ~probe_positive)  # correctly skipped

        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        skip_rate = (tn + fn) / len(gt_wavgs)  # articles that skip Stage 2

        results["thresholds"][threshold] = {
            "recall": recall,
            "fn": int(fn),
            "tp": int(tp),
            "skip_rate": skip_rate,
            "gt_positive": int(np.sum(gt_positive)),
        }

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--n", type=int, default=None, help="Limit articles (None=all)")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load articles
    print(f"Loading articles from {args.data}...")
    articles = []
    labels_list = []
    tiers_list = []
    with open(args.data) as f:
        for i, line in enumerate(f):
            if args.n and i >= args.n:
                break
            d = json.loads(line)
            articles.append(d)
            labels_list.append(d["labels"])
            tiers_list.append(d.get("production_tier", "unknown"))

    labels = np.array(labels_list)
    print(f"Loaded {len(articles)} articles")

    texts = [f"{a['title']}\n\n{a['content']}" for a in articles]

    # Split
    np.random.seed(42)
    indices = np.random.permutation(len(articles))
    split = int(0.8 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    tiers_val = [tiers_list[i] for i in val_idx]
    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    # Test each model
    from sentence_transformers import SentenceTransformer

    all_results = {}

    for model_name in MODELS_TO_TEST:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")

        # Load and time embedding
        t0 = time.time()
        embedder = SentenceTransformer(model_name, device=device)
        load_time = time.time() - t0
        print(f"  Model loaded in {load_time:.1f}s")

        # Embedding dimension
        test_emb = embedder.encode(["test"], convert_to_numpy=True)
        emb_dim = test_emb.shape[1]
        print(f"  Embedding dim: {emb_dim}")

        # Time embedding on 1000 articles
        sample_texts = texts[:1000]
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        _ = embedder.encode(sample_texts, batch_size=args.batch_size,
                           show_progress_bar=False, convert_to_numpy=True)
        if device == "cuda":
            torch.cuda.synchronize()
        embed_time = time.time() - t0
        ms_per_article = embed_time / len(sample_texts) * 1000
        print(f"  Embedding speed: {ms_per_article:.1f}ms/article (1000 articles)")

        # Generate all embeddings
        print(f"  Generating embeddings for all {len(texts)} articles...")
        t0 = time.time()
        all_embeddings = embedder.encode(texts, batch_size=args.batch_size,
                                         show_progress_bar=False, convert_to_numpy=True)
        total_embed_time = time.time() - t0
        print(f"  Done in {total_embed_time:.1f}s")

        train_emb = all_embeddings[train_idx]
        val_emb = all_embeddings[val_idx]

        # Train probe
        print(f"  Training MLP probe...")
        t0 = time.time()
        probe, scaler, best_mae = train_probe(train_emb, train_labels, val_emb, val_labels, device)
        train_time = time.time() - t0
        print(f"  Probe trained in {train_time:.1f}s, best val MAE: {best_mae:.4f}")

        # Evaluate
        eval_results = evaluate_model(probe, scaler, val_emb, val_labels, tiers_val, device)
        print(f"  Weighted avg MAE: {eval_results['wavg_mae']:.3f}, bias: {eval_results['wavg_bias']:+.3f}")

        print(f"\n  Binary classification (does probe catch the good articles?):")
        print(f"  {'Threshold':>10} {'Recall':>8} {'Missed':>8} {'Skip rate':>10}")
        for t, r in eval_results["thresholds"].items():
            print(f"  {t:>10.1f} {r['recall']:>7.1%} {r['fn']:>8} {r['skip_rate']:>9.1%}")

        all_results[model_name] = {
            "emb_dim": emb_dim,
            "ms_per_article": ms_per_article,
            "val_mae": best_mae,
            "wavg_mae": eval_results["wavg_mae"],
            "wavg_bias": eval_results["wavg_bias"],
            "thresholds": eval_results["thresholds"],
        }

        # Free GPU memory
        del embedder, all_embeddings
        if device == "cuda":
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':>40} {'Dim':>5} {'ms/art':>7} {'MAE':>6} {'Recall@4.5':>11} {'Skip@4.5':>9}")
    for model_name, r in all_results.items():
        short_name = model_name.split("/")[-1]
        t45 = r["thresholds"][4.5]
        print(f"{short_name:>40} {r['emb_dim']:>5} {r['ms_per_article']:>6.1f} {r['wavg_mae']:>6.3f} {t45['recall']:>10.1%} {t45['skip_rate']:>8.1%}")


if __name__ == "__main__":
    main()
