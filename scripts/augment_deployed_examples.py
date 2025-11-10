"""
Generate synthetic variations of deployed tech examples via paraphrasing.

This script takes high-scoring deployed examples and generates paraphrased
variations to expand the training set for the underrepresented deployed tier.
"""

import json
import time
from pathlib import Path
from typing import Dict, List
import argparse
from datetime import datetime

# Import LLM client
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.llm.llm_client import create_llm_client


PARAPHRASE_PROMPT = """You are a paraphrasing assistant. Your task is to rewrite the following article while preserving all factual content, technical details, and deployment information.

**Guidelines**:
- Keep all numbers, dates, company names, and technical specifications EXACTLY the same
- Preserve the core message and facts
- Change sentence structure, vocabulary, and phrasing
- Maintain the same level of detail
- Do NOT add new information or speculation
- Do NOT remove important deployment details (scale, timeline, impact)

**Original Article**:
Title: {title}
Content: {content}

**Task**: Rewrite this article with different wording but identical facts. Output ONLY the rewritten content (no title needed).
"""


def paraphrase_article(article: Dict, llm_client, variation_num: int) -> Dict:
    """Generate a paraphrased variation of an article."""

    # Build prompt
    prompt = PARAPHRASE_PROMPT.format(
        title=article['title'],
        content=article['content'][:3000]  # Limit to avoid token limits
    )

    # Generate paraphrase
    try:
        paraphrased_content = llm_client.generate(prompt)

        # Create variation with modified content
        variation = article.copy()
        variation['id'] = f"{article['id']}_synthetic_v{variation_num}"
        variation['content'] = paraphrased_content
        variation['title'] = f"[Synthetic Variation {variation_num}] {article['title']}"
        variation['metadata'] = variation.get('metadata', {})
        variation['metadata']['synthetic'] = True
        variation['metadata']['source_article'] = article['id']
        variation['metadata']['variation_number'] = variation_num

        return variation

    except Exception as e:
        print(f"  ERROR paraphrasing: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic deployed examples')
    parser.add_argument('--source', required=True, help='Source JSONL with labeled deployed examples')
    parser.add_argument('--min-score', type=float, default=8.0, help='Minimum overall score for source examples')
    parser.add_argument('--variations', type=int, default=10, help='Number of variations per example')
    parser.add_argument('--output', required=True, help='Output JSONL for synthetic examples')
    parser.add_argument('--llm', default='gemini-flash', help='LLM to use for paraphrasing')

    args = parser.parse_args()

    print('='*60)
    print('DEPLOYED EXAMPLES SYNTHETIC AUGMENTATION')
    print('='*60)

    # Load high-scoring deployed examples
    print(f'\nLoading source examples from: {args.source}')

    source_examples = []
    source_files = list(Path().glob(args.source))

    for source_file in source_files:
        with open(source_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue

                article = json.loads(line)

                # Check if has analysis
                analysis = article.get('sustainability_tech_deployment_analysis', {})
                if not analysis:
                    continue

                score = analysis.get('overall_score', 0)

                # Only use deployed tier (≥8.0)
                if score >= args.min_score:
                    source_examples.append(article)

    print(f'Found {len(source_examples)} deployed examples (score ≥{args.min_score})')

    if len(source_examples) == 0:
        print('ERROR: No deployed examples found!')
        return

    # Initialize LLM
    print(f'\nInitializing LLM: {args.llm}')
    llm_client = create_llm_client(args.llm)

    # Generate variations
    print(f'\nGenerating {args.variations} variations per example...')
    print(f'Total synthetic examples to create: {len(source_examples) * args.variations}')

    synthetic_examples = []

    for i, source in enumerate(source_examples, 1):
        print(f'\n[{i}/{len(source_examples)}] Source: {source["id"]}')
        print(f'  Original score: {source["sustainability_tech_deployment_analysis"]["overall_score"]:.1f}')

        for var_num in range(1, args.variations + 1):
            print(f'  Generating variation {var_num}/{args.variations}...', end=' ')

            variation = paraphrase_article(source, llm_client, var_num)

            if variation:
                synthetic_examples.append(variation)
                print('SUCCESS')
            else:
                print('FAILED')

            # Rate limiting
            time.sleep(0.5)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for example in synthetic_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    print(f'\n{"="*60}')
    print('AUGMENTATION COMPLETE')
    print(f'{"="*60}')
    print(f'\nGenerated {len(synthetic_examples)} synthetic examples')
    print(f'Written to: {output_path}')
    print(f'\nNext step: Label these synthetic examples with oracle')
    print(f'  Cost: ~${len(synthetic_examples) * 0.001:.2f}')


if __name__ == '__main__':
    main()
