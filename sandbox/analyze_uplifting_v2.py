import json

# Read v2 calibration data
with open(r'C:\local_dev\llm-distillery\datasets\working\uplifting_calibration_labeled_v2.jsonl', encoding='utf-8') as f:
    articles = [json.loads(line) for line in f]

print(f"Total articles: {len(articles)}\n")

# Extract labels and key info
for i, article in enumerate(articles, 1):
    title = article.get('title', 'N/A')
    ground_truth = article.get('ground_truth_label', 'N/A')
    uplift = article.get('uplifting_analysis', {})
    tier = uplift.get('tier', 'N/A')
    score = uplift.get('overall_uplift_score', 'N/A')
    content_type = uplift.get('content_type', 'N/A')

    print(f"{i}. {title[:80]}")
    print(f"   Ground Truth: {ground_truth}")
    print(f"   Oracle: tier={tier}, score={score}, type={content_type}")
    print()
