"""
Quick experiment: Can embedding + classifier match/beat fine-tuned DistilBERT for commerce detection?

Baseline: Fine-tuned DistilBERT = 97.8% F1

Key differences from uplifting experiment:
- Binary classification (not regression)
- No regression-to-mean problem expected
- Simpler task: "is this promotional content?"
"""

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sentence_transformers import SentenceTransformer
import time


def load_commerce_data(split_path: Path):
    """Load commerce prefilter data."""
    articles = []
    with open(split_path, 'r', encoding='utf-8') as f:
        for line in f:
            article = json.loads(line)
            # Combine title and content
            text = f"{article.get('title', '')} {article.get('content', '')}"
            label = article.get('label', 0)
            articles.append({
                'id': article.get('id', ''),
                'text': text,
                'label': label
            })
    return articles


def embed_articles(articles: list, model_name: str, device: str = 'cpu'):
    """Generate embeddings for articles."""
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name, device=device)

    texts = [a['text'] for a in articles]
    labels = np.array([a['label'] for a in articles])

    print(f"Generating embeddings for {len(texts)} articles...")
    start = time.time()
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    elapsed = time.time() - start
    print(f"Embedding time: {elapsed:.1f}s ({len(texts)/elapsed:.1f} articles/sec)")

    return embeddings, labels


def evaluate_classifier(clf, X_test, y_test, name: str):
    """Evaluate a classifier and return metrics."""
    start = time.time()
    y_pred = clf.predict(X_test)
    inference_time = (time.time() - start) / len(y_test) * 1000  # ms per sample

    metrics = {
        'name': name,
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'inference_time_ms': inference_time
    }
    return metrics


def main():
    # Paths
    data_dir = Path('datasets/training/commerce_prefilter_v1/splits')

    # Load data
    print("Loading commerce prefilter data...")
    train_data = load_commerce_data(data_dir / 'train.jsonl')
    val_data = load_commerce_data(data_dir / 'val.jsonl')
    test_data = load_commerce_data(data_dir / 'test.jsonl')

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Count labels
    train_labels = [a['label'] for a in train_data]
    print(f"Train distribution: {sum(train_labels)} commerce, {len(train_labels) - sum(train_labels)} journalism")

    # Models to test
    embedding_models = [
        'all-MiniLM-L6-v2',      # Fast baseline (384 dims)
        'all-mpnet-base-v2',      # Quality baseline (768 dims)
        # 'bge-large-en-v1.5',      # Skip for speed
        'BAAI/bge-small-en-v1.5', # Small but good (384 dims)
    ]

    # Classifiers to test
    classifiers = {
        'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'SVM_RBF': SVC(kernel='rbf', class_weight='balanced', probability=True),
        'MLP': MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, early_stopping=True),
    }

    results = []

    for model_name in embedding_models:
        print(f"\n{'='*60}")
        print(f"Testing: {model_name}")
        print('='*60)

        # Generate embeddings
        train_emb, train_labels = embed_articles(train_data, model_name)
        val_emb, val_labels = embed_articles(val_data, model_name)
        test_emb, test_labels = embed_articles(test_data, model_name)

        # Combine train + val for final training
        X_train = np.vstack([train_emb, val_emb])
        y_train = np.concatenate([train_labels, val_labels])
        X_test = test_emb
        y_test = test_labels

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"\nTraining classifiers on {X_train.shape[0]} samples, testing on {X_test.shape[0]}...")

        for clf_name, clf in classifiers.items():
            print(f"\n  Training {clf_name}...")
            start = time.time()
            clf.fit(X_train_scaled, y_train)
            train_time = time.time() - start

            metrics = evaluate_classifier(clf, X_test_scaled, y_test, f"{model_name}_{clf_name}")
            metrics['embedding_model'] = model_name
            metrics['classifier'] = clf_name
            metrics['train_time_seconds'] = train_time
            metrics['embedding_dim'] = X_train.shape[1]

            results.append(metrics)

            print(f"    F1: {metrics['f1']*100:.1f}%, Precision: {metrics['precision']*100:.1f}%, Recall: {metrics['recall']*100:.1f}%")
            print(f"    Train time: {train_time:.1f}s, Inference: {metrics['inference_time_ms']:.3f}ms/sample")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY: Embedding + Classifier vs Fine-tuned DistilBERT")
    print("="*80)
    print(f"\nBaseline (Fine-tuned DistilBERT): F1 = 97.8%, Precision = 96.7%, Recall = 98.9%")
    print("\nEmbedding + Classifier Results:")
    print("-"*80)
    print(f"{'Model':<45} {'F1':>8} {'Prec':>8} {'Recall':>8} {'Inf(ms)':>10}")
    print("-"*80)

    # Sort by F1
    results_sorted = sorted(results, key=lambda x: x['f1'], reverse=True)

    for r in results_sorted:
        name = f"{r['embedding_model'].split('/')[-1]}_{r['classifier']}"
        print(f"{name:<45} {r['f1']*100:>7.1f}% {r['precision']*100:>7.1f}% {r['recall']*100:>7.1f}% {r['inference_time_ms']:>9.3f}")

    # Best result
    best = results_sorted[0]
    print("\n" + "-"*80)
    print(f"BEST: {best['embedding_model']} + {best['classifier']}")
    print(f"  F1: {best['f1']*100:.1f}% (vs 97.8% baseline = {(best['f1'] - 0.978)*100:+.1f}%)")
    print(f"  Precision: {best['precision']*100:.1f}%")
    print(f"  Recall: {best['recall']*100:.1f}%")
    print(f"  Confusion matrix: {best['confusion_matrix']}")

    # Save results
    output_path = Path('research/embedding_vs_finetuning/results/commerce_embedding_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
