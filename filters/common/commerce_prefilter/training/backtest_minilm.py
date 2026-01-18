"""
Backtest MiniLM on sustainability_technology high/medium tier articles.
Compare with DistilBERT results.
"""

import json
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODELS_DIR = Path(__file__).parent.parent / "v1" / "models"
BACKTEST_RESULTS = Path(__file__).parent / "backtest_results.json"


def load_model(model_path: Path):
    """Load an encoder model."""
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()
    return model, tokenizer


def predict(model, tokenizer, text: str) -> float:
    """Get commerce score."""
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        score = probs[0, 1].item()

    return score


def main():
    print("=" * 70)
    print("MiniLM Backtest on sustainability_technology")
    print("=" * 70)

    # Load backtest results (has article metadata)
    with open(BACKTEST_RESULTS, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Get sustainability_technology high and medium tier flagged articles
    flagged = [a for a in data['flagged_articles']
               if a['filter'] == 'sustainability_technology'
               and a['tier'] in ['high', 'medium']]

    print(f"\nArticles to test: {len(flagged)}")
    print(f"  - High tier: {len([a for a in flagged if a['tier'] == 'high'])}")
    print(f"  - Medium tier: {len([a for a in flagged if a['tier'] == 'medium'])}")

    # Load MiniLM
    print("\nLoading MiniLM...", flush=True)
    model, tokenizer = load_model(MODELS_DIR / "minilm")
    print("Done.", flush=True)

    # Run predictions
    print("\nRunning predictions...", flush=True)
    results = []

    for i, article in enumerate(flagged):
        if i % 100 == 0:
            print(f"  {i}/{len(flagged)}...", flush=True)

        # Use title only (we don't have full content in backtest results)
        text = f"[TITLE] {article['title']} [CONTENT] "
        minilm_score = predict(model, tokenizer, text)

        results.append({
            'title': article['title'],
            'source': article['source'],
            'tier': article['tier'],
            'distilbert_score': article['score'],
            'minilm_score': minilm_score,
        })

    print(f"  {len(flagged)}/{len(flagged)} done.", flush=True)

    # Analyze results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Threshold analysis
    for thresh in [0.85, 0.90, 0.95]:
        print(f"\n--- Threshold {thresh} ---")

        for tier in ['high', 'medium']:
            tier_results = [r for r in results if r['tier'] == tier]

            distilbert_blocked = sum(1 for r in tier_results if r['distilbert_score'] >= thresh)
            minilm_blocked = sum(1 for r in tier_results if r['minilm_score'] >= thresh)

            print(f"{tier:8s}: DistilBERT blocks {distilbert_blocked:3d}/{len(tier_results)}, MiniLM blocks {minilm_blocked:3d}/{len(tier_results)}")

    # High tier detailed
    print("\n" + "=" * 70)
    print("HIGH TIER DETAILS")
    print("=" * 70)

    high_tier = [r for r in results if r['tier'] == 'high']
    for r in high_tier:
        title_safe = r['title'][:60].encode('ascii', 'replace').decode('ascii')
        print(f"\n{title_safe}...")
        print(f"  DistilBERT: {r['distilbert_score']:.3f}")
        print(f"  MiniLM:     {r['minilm_score']:.3f}")

    # Score distribution comparison
    print("\n" + "=" * 70)
    print("SCORE DISTRIBUTION (Medium tier)")
    print("=" * 70)

    medium_results = [r for r in results if r['tier'] == 'medium']

    for model_name, key in [('DistilBERT', 'distilbert_score'), ('MiniLM', 'minilm_score')]:
        scores = [r[key] for r in medium_results]
        high = sum(1 for s in scores if s >= 0.95)
        med = sum(1 for s in scores if 0.85 <= s < 0.95)
        low = sum(1 for s in scores if s < 0.85)

        print(f"\n{model_name}:")
        print(f"  >= 0.95 (block):  {high:3d} ({high/len(scores)*100:5.1f}%)")
        print(f"  0.85-0.95 (gray): {med:3d} ({med/len(scores)*100:5.1f}%)")
        print(f"  < 0.85 (pass):    {low:3d} ({low/len(scores)*100:5.1f}%)")

    # Save results
    output_path = Path(__file__).parent / "backtest_minilm_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path.name}")


if __name__ == "__main__":
    main()
