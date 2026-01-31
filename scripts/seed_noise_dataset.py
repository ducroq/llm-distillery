"""Seed the universal noise prefilter dataset with existing labeled data."""

import json
import os
from collections import Counter

os.chdir(r'C:\local_dev\llm-distillery')

noise_samples = []
signal_samples = []

DIMS = ['technology_readiness_level', 'technical_performance', 'economic_competitiveness',
        'life_cycle_environmental_impact', 'social_equity_impact', 'governance_systemic_impact']

# 1. Oracle-rejected sustainability_tech articles (NOISE)
print('Processing oracle-rejected articles...')
for batch in ['scored_batch_001.jsonl', 'scored_batch_002.jsonl']:
    path = f'datasets/scored/sustainability_tech_active_learning/sustainability_technology/{batch}'
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            for line in f:
                article = json.loads(line)
                analysis = article.get('sustainability_technology_analysis', {})
                scores = [analysis.get(d, {}).get('score', 0) for d in DIMS]
                reason = analysis.get('technology_readiness_level', {}).get('evidence', '')

                # Categorize based on rejection reason
                if sum(scores) == 0:
                    if 'AI/ML infrastructure' in reason or 'Programming' in reason:
                        label = 'software'
                    elif 'farming' in reason.lower() or 'traditional' in reason.lower() or 'practices' in reason.lower():
                        label = 'farming_practices'
                    elif 'policy' in reason.lower() or 'economic' in reason.lower():
                        label = 'policy_economic'
                    elif 'Consumer' in reason or 'product review' in reason:
                        label = 'consumer'
                    elif 'business news' in reason.lower() or 'market' in reason.lower():
                        label = 'business'
                    elif 'Healthcare' in reason or 'medical' in reason.lower():
                        label = 'medical'
                    else:
                        label = 'other_noise'

                    noise_samples.append({
                        'id': article['id'],
                        'title': article['title'],
                        'content': article['content'][:2000],
                        'source': article.get('source', 'unknown'),
                        'label': label,
                        'is_noise': True,
                        'labeled_by': 'oracle',
                        'labeled_at': '2026-01-31',
                        'notes': reason[:200] if reason else ''
                    })
                else:
                    # In-scope = signal
                    signal_samples.append({
                        'id': article['id'],
                        'title': article['title'],
                        'content': article['content'][:2000],
                        'source': article.get('source', 'unknown'),
                        'label': 'applied_research',
                        'is_noise': False,
                        'labeled_by': 'oracle',
                        'labeled_at': '2026-01-31',
                        'notes': f'avg_score={sum(scores)/len(scores):.2f}'
                    })

print(f'  Noise from oracle: {len(noise_samples)}')
print(f'  Signal from oracle: {len(signal_samples)}')

# 2. Commerce prefilter test data
print('Processing commerce test data...')
test_path = 'filters/common/commerce_prefilter/training/splits/test.jsonl'
if os.path.exists(test_path):
    with open(test_path, encoding='utf-8') as f:
        for line in f:
            article = json.loads(line)
            if article.get('label') == 1:  # commerce
                noise_samples.append({
                    'id': article.get('id', f'commerce_test_{len(noise_samples)}'),
                    'title': article.get('title', ''),
                    'content': article.get('content', '')[:2000],
                    'source': article.get('source', 'unknown'),
                    'label': 'commerce',
                    'is_noise': True,
                    'labeled_by': 'manual',
                    'labeled_at': '2026-01-31',
                    'notes': 'from commerce prefilter test set'
                })
            else:  # journalism
                signal_samples.append({
                    'id': article.get('id', f'journalism_test_{len(signal_samples)}'),
                    'title': article.get('title', ''),
                    'content': article.get('content', '')[:2000],
                    'source': article.get('source', 'unknown'),
                    'label': 'journalism',
                    'is_noise': False,
                    'labeled_by': 'manual',
                    'labeled_at': '2026-01-31',
                    'notes': 'from commerce prefilter test set'
                })

print(f'  Total noise: {len(noise_samples)}')
print(f'  Total signal: {len(signal_samples)}')

# 3. New commerce articles from curation
print('Processing new commerce curation...')
commerce_dir = 'datasets/curation/commerce_prefilter_training'
if os.path.exists(commerce_dir):
    for fname in os.listdir(commerce_dir):
        if fname.endswith('.jsonl'):
            with open(f'{commerce_dir}/{fname}', encoding='utf-8') as f:
                for line in f:
                    article = json.loads(line)
                    noise_samples.append({
                        'id': article.get('id', f'commerce_new_{len(noise_samples)}'),
                        'title': article.get('title', ''),
                        'content': article.get('content', '')[:2000],
                        'source': article.get('source', 'unknown'),
                        'label': 'commerce',
                        'is_noise': True,
                        'labeled_by': 'manual',
                        'labeled_at': '2026-01-31',
                        'notes': f'from {fname}'
                    })

print(f'  Final noise: {len(noise_samples)}')
print(f'  Final signal: {len(signal_samples)}')

# Save
out_dir = 'datasets/training/universal_noise_prefilter'
with open(f'{out_dir}/noise.jsonl', 'w', encoding='utf-8') as f:
    for sample in noise_samples:
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')

with open(f'{out_dir}/signal.jsonl', 'w', encoding='utf-8') as f:
    for sample in signal_samples:
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')

print(f'Saved to {out_dir}/')

# Category breakdown
noise_cats = Counter(s['label'] for s in noise_samples)
signal_cats = Counter(s['label'] for s in signal_samples)

print()
print('=== NOISE CATEGORIES ===')
for cat, count in noise_cats.most_common():
    print(f'  {cat}: {count}')

print()
print('=== SIGNAL CATEGORIES ===')
for cat, count in signal_cats.most_common():
    print(f'  {cat}: {count}')
