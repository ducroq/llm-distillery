import json
import sys
from pathlib import Path

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

# Read v2 calibration data
v2_path = Path(r'C:\local_dev\llm-distillery\datasets\working\uplifting_calibration_labeled_v2.jsonl')
articles = []
with open(v2_path, encoding='utf-8') as f:
    for line in f:
        articles.append(json.loads(line))

print("=" * 80)
print("UPLIFTING FILTER - PROMPT CALIBRATION v2 ANALYSIS")
print("=" * 80)
print(f"\nTotal articles: {len(articles)}\n")

# Manual ground truth classification based on content analysis
# Based on actual article IDs from v2 calibration sample
ground_truth = {
    # OFF-TOPIC (corporate/technical optimization, business news, entertainment)
    "community_social_dev_to_c90592985034": {  # API Gateway Design with Rust and Go
        "category": "off_topic",
        "reason": "Generic software development tutorial, no human wellbeing outcomes",
        "expected_score_range": (0, 3)
    },
    "community_social_dev_to_9dbe3463c71f": {  # Gantt chart time management
        "category": "off_topic",
        "reason": "Business productivity tool tutorial, no wellbeing outcomes",
        "expected_score_range": (0, 3)
    },
    "community_social_hacker_news_0018a0483a2f": {  # Ask HN: Modern Emacs
        "category": "off_topic",
        "reason": "Software tool advice, no human wellbeing focus",
        "expected_score_range": (0, 3)
    },
    "portuguese_canaltech_8b2948fc5416": {  # Logitech steering wheel
        "category": "off_topic",
        "reason": "Gaming hardware product news, no wellbeing outcomes",
        "expected_score_range": (0, 3)
    },
    "portuguese_canaltech_052b45e83655": {  # Free games weekend
        "category": "off_topic",
        "reason": "Gaming entertainment news, no wellbeing outcomes",
        "expected_score_range": (0, 3)
    },

    # ON-TOPIC (genuine wellbeing progress)
    "positive_news_upworthy_170743b81286": {  # Anti-homeless spikes art protest
        "category": "on_topic",
        "reason": "Community advocacy for homeless rights, justice focus",
        "expected_score_range": (6, 8)
    },
    "positive_news_upworthy_7c56f1d81c50": {  # Women over 60 aging wisdom
        "category": "on_topic",
        "reason": "Knowledge sharing for wellbeing, community connection",
        "expected_score_range": (6, 8)
    },
    "industry_intelligence_fast_company_789837e00ebc": {  # Amazon disaster relief Jamaica
        "category": "on_topic",
        "reason": "Disaster response, humanitarian aid delivery",
        "expected_score_range": (6, 7)
    },
    "positive_news_the_better_india_12d3056c17b8": {  # Farming tips for kharif crops
        "category": "on_topic",
        "reason": "Agricultural knowledge for food security, farmer livelihoods",
        "expected_score_range": (5, 7)
    },
}

# Analyze each article
false_positives = []
false_negatives = []
correct_rejections = []
correct_recognitions = []

print("ARTICLE ANALYSIS:")
print("-" * 80)

for article in articles:
    article_id = article['id']
    title = article['title']
    analysis = article.get('uplifting_analysis', {})
    score = analysis.get('overall_uplift_score', 0)
    tier = analysis.get('tier', 'unknown')
    content_type = analysis.get('content_type', 'unknown')
    reasoning = analysis.get('reasoning', '')

    # Get ground truth
    gt = ground_truth.get(article_id)
    if not gt:
        print(f"\n⚠️ NO GROUND TRUTH for {article_id}")
        print(f"   Title: {title[:70]}")
        continue

    category = gt['category']
    expected_range = gt['expected_score_range']

    print(f"\n{len(false_positives) + len(false_negatives) + len(correct_rejections) + len(correct_recognitions) + 1}. {title[:70]}")
    print(f"   Ground Truth: {category.upper()}")
    print(f"   Oracle Score: {score:.2f} (tier: {tier})")
    print(f"   Expected: {expected_range[0]}-{expected_range[1]}")

    # Classify result
    if category == "off_topic":
        if score >= 5.0:
            print(f"   ❌ FALSE POSITIVE (scored too high)")
            false_positives.append({
                'title': title,
                'score': score,
                'expected': expected_range,
                'reason': gt['reason'],
                'oracle_reasoning': reasoning
            })
        else:
            print(f"   ✅ CORRECT REJECTION")
            correct_rejections.append({
                'title': title,
                'score': score
            })

    elif category == "on_topic":
        if score < 5.0:
            print(f"   ❌ FALSE NEGATIVE (scored too low)")
            false_negatives.append({
                'title': title,
                'score': score,
                'expected': expected_range,
                'reason': gt['reason'],
                'oracle_reasoning': reasoning
            })
        else:
            print(f"   ✅ CORRECT RECOGNITION")
            correct_recognitions.append({
                'title': title,
                'score': score
            })

# Calculate metrics
total_off_topic = sum(1 for gt in ground_truth.values() if gt['category'] == 'off_topic')
total_on_topic = sum(1 for gt in ground_truth.values() if gt['category'] == 'on_topic')

off_topic_rejection_rate = len(correct_rejections) / total_off_topic * 100 if total_off_topic > 0 else 0
on_topic_recognition_rate = len(correct_recognitions) / total_on_topic * 100 if total_on_topic > 0 else 0
false_positive_rate = len(false_positives) / total_off_topic * 100 if total_off_topic > 0 else 0

print("\n" + "=" * 80)
print("CALIBRATION METRICS")
print("=" * 80)

print(f"\n1. OFF-TOPIC REJECTION RATE")
print(f"   Off-topic articles: {total_off_topic}")
print(f"   Correctly rejected (score < 5.0): {len(correct_rejections)} ({off_topic_rejection_rate:.1f}%)")
print(f"   False positives (score >= 5.0): {len(false_positives)} ({false_positive_rate:.1f}%)")
print(f"   Target: >90% rejection, <10% false positives")
if false_positive_rate <= 10:
    print(f"   ✅ PASS")
elif false_positive_rate <= 20:
    print(f"   ⚠️ REVIEW")
else:
    print(f"   ❌ FAIL")

print(f"\n2. ON-TOPIC RECOGNITION RATE")
print(f"   On-topic articles: {total_on_topic}")
print(f"   Correctly recognized (score >= 5.0): {len(correct_recognitions)} ({on_topic_recognition_rate:.1f}%)")
print(f"   False negatives (score < 5.0): {len(false_negatives)} ({100 - on_topic_recognition_rate:.1f}%)")
print(f"   Target: >80% recognition")
if on_topic_recognition_rate >= 80:
    print(f"   ✅ PASS")
else:
    print(f"   ❌ FAIL")

print("\n" + "=" * 80)
print("V1 vs V2 COMPARISON")
print("=" * 80)

print("\n                              V1          V2        Change")
print("-" * 80)
print(f"Off-topic rejection rate:     20%         {off_topic_rejection_rate:.0f}%       {'+' if off_topic_rejection_rate > 20 else ''}{off_topic_rejection_rate - 20:.0f} pp")
print(f"On-topic recognition rate:    100%        {on_topic_recognition_rate:.0f}%       {'+' if on_topic_recognition_rate > 100 else ''}{on_topic_recognition_rate - 100:.0f} pp")
print(f"False positive rate:          80%         {false_positive_rate:.0f}%        {'+' if false_positive_rate > 80 else ''}{false_positive_rate - 80:.0f} pp")

print("\n" + "=" * 80)
print("FINAL DECISION")
print("=" * 80)

# Decision logic
if false_positive_rate <= 10 and on_topic_recognition_rate >= 80:
    decision = "✅ PASS"
    recommendation = "PROCEED TO BATCH LABELING"
elif false_positive_rate <= 20 and on_topic_recognition_rate >= 70:
    decision = "⚠️ REVIEW"
    recommendation = "MINOR PROMPT ADJUSTMENTS RECOMMENDED"
else:
    decision = "❌ FAIL"
    recommendation = "MAJOR PROMPT REVISION REQUIRED"

print(f"\nDecision: {decision}")
print(f"Recommendation: {recommendation}")

if len(false_positives) > 0:
    print(f"\n⚠️ FALSE POSITIVES DETECTED ({len(false_positives)} articles):")
    for fp in false_positives:
        print(f"\n- \"{fp['title'][:70]}\" → {fp['score']:.2f}")
        print(f"  Reason off-topic: {fp['reason']}")
        print(f"  Oracle reasoning: {fp['oracle_reasoning'][:150]}...")

if len(false_negatives) > 0:
    print(f"\n⚠️ FALSE NEGATIVES DETECTED ({len(false_negatives)} articles):")
    for fn in false_negatives:
        print(f"\n- \"{fn['title'][:70]}\" → {fn['score']:.2f}")
        print(f"  Reason on-topic: {fn['reason']}")
        print(f"  Oracle reasoning: {fn['oracle_reasoning'][:150]}...")

print("\n" + "=" * 80)
