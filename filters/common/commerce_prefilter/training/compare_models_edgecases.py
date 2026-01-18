"""
Compare all trained models on edge cases.

Edge cases:
- 3 high-tier articles (disputable)
- 71 gray-zone articles (0.85-0.95 score range from DistilBERT)
"""

import json
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Paths
MODELS_DIR = Path(__file__).parent.parent / "v1" / "models"
BACKTEST_RESULTS = Path(__file__).parent / "backtest_results.json"


def load_encoder_model(model_path: Path):
    """Load an encoder model (DistilBERT, MiniLM, XLM-RoBERTa)."""
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()
    return model, tokenizer


def predict_encoder(model, tokenizer, text: str, device: str = "cpu") -> float:
    """Get commerce score from encoder model."""
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        score = probs[0, 1].item()

    return score


def prepare_text(article: dict) -> str:
    """Prepare input text from article."""
    title = article.get('title', '')
    content = article.get('content', article.get('text', ''))
    content = content[:4000] if content else ''
    return f"[TITLE] {title} [CONTENT] {content}"


def main():
    print("=" * 70)
    print("Model Comparison on Edge Cases")
    print("=" * 70)

    # Load backtest results
    with open(BACKTEST_RESULTS, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Get sustainability_technology high and medium tier flagged articles
    flagged = [a for a in data['flagged_articles']
               if a['filter'] == 'sustainability_technology'
               and a['tier'] in ['high', 'medium']]

    # Split into categories
    high_tier = [a for a in flagged if a['tier'] == 'high']
    gray_zone = [a for a in flagged if a['tier'] == 'medium' and 0.85 <= a['score'] < 0.95]

    print(f"\nEdge cases to test:")
    print(f"  - High tier (disputable): {len(high_tier)}")
    print(f"  - Gray zone (0.85-0.95): {len(gray_zone)}")

    # Load models
    models = {}

    print("\nLoading models...")

    # DistilBERT
    print("  - DistilBERT...", end=" ", flush=True)
    models['distilbert'] = load_encoder_model(MODELS_DIR / "distilbert")
    print("done")

    # MiniLM
    print("  - MiniLM...", end=" ", flush=True)
    models['minilm'] = load_encoder_model(MODELS_DIR / "minilm")
    print("done")

    # XLM-RoBERTa
    print("  - XLM-RoBERTa...", end=" ", flush=True)
    models['xlm-roberta'] = load_encoder_model(MODELS_DIR / "xlm-roberta")
    print("done")

    # Skip Qwen for now - needs different loading
    print("  - Qwen LoRA: skipped (requires base model download)")

    # Test high-tier articles
    print("\n" + "=" * 70)
    print("HIGH TIER ARTICLES (should NOT be blocked)")
    print("=" * 70)

    for article in high_tier:
        print(f"\n{article['title'][:70]}...")
        print(f"  Source: {article['source']}")

        # We don't have full content in backtest results, use title only
        text = f"[TITLE] {article['title']} [CONTENT] "

        for name, (model, tokenizer) in models.items():
            score = predict_encoder(model, tokenizer, text)
            flag = "BLOCK" if score >= 0.95 else "pass" if score < 0.85 else "gray"
            print(f"  {name:15s}: {score:.3f} [{flag}]")

    # Test gray zone articles
    print("\n" + "=" * 70)
    print("GRAY ZONE ARTICLES (0.85-0.95 from DistilBERT)")
    print("=" * 70)

    # Categorize by model agreement
    results = []

    for article in gray_zone[:30]:  # Test first 30
        text = f"[TITLE] {article['title']} [CONTENT] "

        scores = {}
        for name, (model, tokenizer) in models.items():
            scores[name] = predict_encoder(model, tokenizer, text)

        results.append({
            'title': article['title'],
            'source': article['source'],
            'distilbert_original': article['score'],
            **scores
        })

    # Print results sorted by disagreement
    print("\nModel scores on gray zone (sorted by max disagreement):")
    print("-" * 90)
    print(f"{'Title'[:40]:40s} | {'DistilB':>7s} | {'MiniLM':>7s} | {'XLM-R':>7s} | {'Spread':>6s}")
    print("-" * 90)

    for r in sorted(results, key=lambda x: max(x['distilbert'], x['minilm'], x['xlm-roberta']) - min(x['distilbert'], x['minilm'], x['xlm-roberta']), reverse=True):
        scores = [r['distilbert'], r['minilm'], r['xlm-roberta']]
        spread = max(scores) - min(scores)
        # Handle Unicode by encoding to ascii with replacement
        title_safe = r['title'][:40].encode('ascii', 'replace').decode('ascii')
        print(f"{title_safe:40s} | {r['distilbert']:7.3f} | {r['minilm']:7.3f} | {r['xlm-roberta']:7.3f} | {spread:6.3f}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY: How many would each model block at threshold 0.95?")
    print("=" * 70)

    for name in ['distilbert', 'minilm', 'xlm-roberta']:
        high_blocked = sum(1 for r in results[:3] if r.get(name, 0) >= 0.95)  # Won't work, need to recompute

    # Recompute for high tier
    print("\nHigh tier (3 articles):")
    for name, (model, tokenizer) in models.items():
        blocked = 0
        for article in high_tier:
            text = f"[TITLE] {article['title']} [CONTENT] "
            score = predict_encoder(model, tokenizer, text)
            if score >= 0.95:
                blocked += 1
        print(f"  {name:15s}: {blocked}/3 would be blocked")

    print("\nGray zone (30 articles tested):")
    for name in ['distilbert', 'minilm', 'xlm-roberta']:
        blocked = sum(1 for r in results if r.get(name, 0) >= 0.95)
        passed = sum(1 for r in results if r.get(name, 0) < 0.85)
        gray = len(results) - blocked - passed
        print(f"  {name:15s}: {blocked} blocked, {gray} gray, {passed} passed")


if __name__ == "__main__":
    main()
