"""
Quick test of the batch labeler to verify it works with both prompts.

This creates a small test dataset to verify the labeler works before
running expensive full labeling jobs.
"""

import json
from pathlib import Path

# Create test data
test_articles = [
    {
        "id": "test_001",
        "title": "Community Solar Project Brings Clean Energy to Low-Income Neighborhood",
        "content": "Residents of the Eastside neighborhood launched a community-owned solar cooperative that will provide affordable renewable energy to 200 low-income households. The project was funded through a combination of grants and community investment, with local workers receiving training in solar installation.",
        "source": "test_source",
        "published_date": "2025-01-15",
        "metadata": {
            "sentiment_score": 7.5,
            "raw_emotions": {"joy": 0.65},
            "source_category": "climate_solutions"
        }
    },
    {
        "id": "test_002",
        "title": "Tesla Stock Hits New High on Q4 Earnings",
        "content": "Tesla shares surged 15% after reporting record quarterly profits driven by strong Model 3 sales. CEO discussed plans for new factory expansion.",
        "source": "test_source",
        "published_date": "2025-01-15",
        "metadata": {
            "sentiment_score": 6.0,
            "raw_emotions": {"joy": 0.4},
            "source_category": "business"
        }
    },
    {
        "id": "test_003",
        "title": "New Battery Recycling Facility Achieves 98% Material Recovery Rate",
        "content": "A startup's innovative battery recycling process demonstrates 98% recovery of lithium, cobalt, and nickel from used EV batteries. The facility will process 10,000 tons annually, with plans to expand. The company open-sourced key parts of their process to accelerate industry adoption.",
        "source": "test_source",
        "published_date": "2025-01-15",
        "metadata": {
            "sentiment_score": 7.0,
            "raw_emotions": {"joy": 0.5},
            "source_category": "renewable_energy"
        }
    }
]

# Create test file
test_dir = Path("datasets/test")
test_dir.mkdir(parents=True, exist_ok=True)
test_file = test_dir / "test_articles.jsonl"

with open(test_file, 'w', encoding='utf-8') as f:
    for article in test_articles:
        f.write(json.dumps(article) + '\n')

print(f"[OK] Created test file: {test_file}")
print(f"   Articles: {len(test_articles)}")
print()
print("To test the batch labeler:")
print()
print("1. Set your API key:")
print("   export ANTHROPIC_API_KEY=sk-ant-...")
print()
print("2. Test with uplifting prompt:")
print(f"   python -m ground_truth.batch_labeler \\")
print(f"       --prompt prompts/uplifting.md \\")
print(f"       --source {test_file} \\")
print(f"       --llm claude \\")
print(f"       --batch-size 3 \\")
print(f"       --max-batches 1 \\")
print(f"       --pre-filter uplifting")
print()
print("3. Test with sustainability prompt:")
print(f"   python -m ground_truth.batch_labeler \\")
print(f"       --prompt prompts/sustainability.md \\")
print(f"       --source {test_file} \\")
print(f"       --llm claude \\")
print(f"       --batch-size 3 \\")
print(f"       --max-batches 1 \\")
print(f"       --pre-filter sustainability")
print()
print("Expected cost: ~$0.03 (3 articles × $0.01 each)")
print()
print("Check output in:")
print("   datasets/uplifting/labeled_batch_001.jsonl")
print("   datasets/sustainability/labeled_batch_001.jsonl")
