"""
Test multilingual embedder for commerce detection.
Need to verify paraphrase-multilingual-mpnet-base-v2 achieves >= 97% F1.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sentence_transformers import SentenceTransformer
import pickle
import time


def load_commerce_data(split_path: Path):
    """Load commerce prefilter data."""
    articles = []
    with open(split_path, 'r', encoding='utf-8') as f:
        for line in f:
            article = json.loads(line)
            text = f"{article.get('title', '')} {article.get('content', '')}"
            label = article.get('label', 0)
            articles.append({'id': article.get('id', ''), 'text': text, 'label': label})
    return articles


def main():
    data_dir = Path('datasets/training/commerce_prefilter_v1/splits')
    output_dir = Path('filters/common/commerce_prefilter/v2/models')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    train_data = load_commerce_data(data_dir / 'train.jsonl')
    val_data = load_commerce_data(data_dir / 'val.jsonl')
    test_data = load_commerce_data(data_dir / 'test.jsonl')

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Test multilingual model
    model_name = 'paraphrase-multilingual-mpnet-base-v2'
    print(f"\nTesting: {model_name}")

    print("Loading embedder...")
    embedder = SentenceTransformer(model_name, device='cpu')

    # Generate embeddings
    print("Generating train embeddings...")
    train_texts = [a['text'] for a in train_data]
    train_labels = np.array([a['label'] for a in train_data])
    train_emb = embedder.encode(train_texts, show_progress_bar=True, batch_size=16)

    print("Generating val embeddings...")
    val_texts = [a['text'] for a in val_data]
    val_labels = np.array([a['label'] for a in val_data])
    val_emb = embedder.encode(val_texts, show_progress_bar=True, batch_size=16)

    print("Generating test embeddings...")
    test_texts = [a['text'] for a in test_data]
    test_labels = np.array([a['label'] for a in test_data])
    test_emb = embedder.encode(test_texts, show_progress_bar=True, batch_size=16)

    # Combine train + val
    X_train = np.vstack([train_emb, val_emb])
    y_train = np.concatenate([train_labels, val_labels])
    X_test = test_emb
    y_test = test_labels

    # Scale
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train MLP
    print("Training MLP classifier...")
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )
    start = time.time()
    clf.fit(X_train_scaled, y_train)
    train_time = time.time() - start
    print(f"Training time: {train_time:.1f}s")

    # Evaluate
    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]

    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "="*60)
    print(f"RESULTS: {model_name} + MLP")
    print("="*60)
    print(f"F1 Score:    {f1*100:.1f}%")
    print(f"Precision:   {precision*100:.1f}%")
    print(f"Recall:      {recall*100:.1f}%")
    print(f"Confusion Matrix:")
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")

    # Check if meets threshold
    TARGET_F1 = 0.97
    if f1 >= TARGET_F1:
        print(f"\n[OK] F1 >= {TARGET_F1*100}% - APPROVED for v2")

        # Save models for v2
        print("\nSaving models for v2...")

        # Save scaler
        with open(output_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        print(f"  Saved: {output_dir / 'scaler.pkl'}")

        # Save MLP
        with open(output_dir / 'mlp_classifier.pkl', 'wb') as f:
            pickle.dump(clf, f)
        print(f"  Saved: {output_dir / 'mlp_classifier.pkl'}")

        # Save config
        config = {
            'embedder_model': model_name,
            'embedding_dim': X_train.shape[1],
            'classifier_type': 'MLPClassifier',
            'hidden_layers': [256, 128],
            'test_f1': float(f1),
            'test_precision': float(precision),
            'test_recall': float(recall),
            'confusion_matrix': cm.tolist(),
            'train_samples': len(y_train),
            'test_samples': len(y_test),
            'created': '2026-01-23'
        }
        with open(output_dir / 'training_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print(f"  Saved: {output_dir / 'training_config.json'}")

        print("\n[OK] v2 models saved successfully!")
        return True
    else:
        print(f"\n[FAIL] F1 < {TARGET_F1*100}% - Does not meet threshold")
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
