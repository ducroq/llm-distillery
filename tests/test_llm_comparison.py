"""
Test LLM comparison: Gemini vs Claude on Sustainability and SEECE prompts.

Selects sample articles passing each filter, runs them through both LLMs,
and generates a detailed comparison report.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from ground_truth.batch_labeler import (
    GenericBatchLabeler,
    sustainability_pre_filter,
    seece_pre_filter
)

def test_llm_comparison(
    dataset_path: str,
    num_samples_per_filter: int = 5,
    output_dir: str = 'reports'
):
    """
    Compare Gemini vs Claude on sustainability and SEECE prompts.

    Args:
        dataset_path: Path to master dataset
        num_samples_per_filter: Number of articles to test per filter (default 5)
        output_dir: Where to save the comparison report
    """

    print("="*80)
    print("LLM COMPARISON TEST: Gemini vs Claude")
    print("="*80)
    print(f"Dataset: {dataset_path}")
    print(f"Samples per filter: {num_samples_per_filter}")
    print(f"Testing filters: sustainability, seece")
    print(f"Testing LLMs: gemini, claude")
    print("="*80 + "\n")

    # Step 1: Find sample articles for each filter
    print("STEP 1: Finding sample articles...")
    print("-"*80)

    sustainability_samples = []
    seece_samples = []

    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(sustainability_samples) >= num_samples_per_filter and \
               len(seece_samples) >= num_samples_per_filter:
                break

            article = json.loads(line.strip())

            # Try sustainability filter
            if len(sustainability_samples) < num_samples_per_filter:
                if sustainability_pre_filter(article):
                    sustainability_samples.append(article)
                    print(f"  Sustainability sample {len(sustainability_samples)}: {article.get('id')} ({article.get('source')})")

            # Try SEECE filter
            if len(seece_samples) < num_samples_per_filter:
                if seece_pre_filter(article):
                    seece_samples.append(article)
                    print(f"  SEECE sample {len(seece_samples)}: {article.get('id')} ({article.get('source')})")

    print(f"\nFound {len(sustainability_samples)} sustainability samples")
    print(f"Found {len(seece_samples)} SEECE samples\n")

    # Step 2: Test each combination
    results = {
        'sustainability': {'gemini': [], 'claude': []},
        'seece': {'gemini': [], 'claude': []}
    }

    # Test Sustainability prompt
    print("STEP 2: Testing Sustainability prompt...")
    print("-"*80)

    for llm in ['gemini', 'claude']:
        print(f"\nTesting with {llm.upper()}...")
        labeler = GenericBatchLabeler(
            prompt_path='prompts/sustainability.md',
            llm_provider=llm,
            output_dir='datasets/test_comparison'
        )

        for i, article in enumerate(sustainability_samples, 1):
            print(f"  Article {i}/{len(sustainability_samples)}: {article.get('id')[:20]}...", end=' ')
            start_time = time.time()

            try:
                analysis = labeler.analyze_article(article, timeout_seconds=90)
                elapsed = time.time() - start_time

                results['sustainability'][llm].append({
                    'article_id': article.get('id'),
                    'article_title': article.get('title', '')[:100],
                    'article_source': article.get('source'),
                    'success': True,
                    'time_seconds': round(elapsed, 2),
                    'analysis': analysis
                })
                print(f"OK ({elapsed:.1f}s)")

            except Exception as e:
                elapsed = time.time() - start_time
                results['sustainability'][llm].append({
                    'article_id': article.get('id'),
                    'article_title': article.get('title', '')[:100],
                    'article_source': article.get('source'),
                    'success': False,
                    'time_seconds': round(elapsed, 2),
                    'error': str(e)
                })
                print(f"FAIL ({elapsed:.1f}s) - {str(e)[:50]}")

            # Small delay between requests
            time.sleep(1)

    # Test SEECE prompt
    print(f"\n\nSTEP 3: Testing SEECE prompt...")
    print("-"*80)

    for llm in ['gemini', 'claude']:
        print(f"\nTesting with {llm.upper()}...")
        labeler = GenericBatchLabeler(
            prompt_path='prompts/seece-energy-tech.md',
            llm_provider=llm,
            output_dir='datasets/test_comparison'
        )

        for i, article in enumerate(seece_samples, 1):
            print(f"  Article {i}/{len(seece_samples)}: {article.get('id')[:20]}...", end=' ')
            start_time = time.time()

            try:
                analysis = labeler.analyze_article(article, timeout_seconds=90)
                elapsed = time.time() - start_time

                results['seece'][llm].append({
                    'article_id': article.get('id'),
                    'article_title': article.get('title', '')[:100],
                    'article_source': article.get('source'),
                    'success': True,
                    'time_seconds': round(elapsed, 2),
                    'analysis': analysis
                })
                print(f"OK ({elapsed:.1f}s)")

            except Exception as e:
                elapsed = time.time() - start_time
                results['seece'][llm].append({
                    'article_id': article.get('id'),
                    'article_title': article.get('title', '')[:100],
                    'article_source': article.get('source'),
                    'success': False,
                    'time_seconds': round(elapsed, 2),
                    'error': str(e)
                })
                print(f"FAIL ({elapsed:.1f}s) - {str(e)[:50]}")

            # Small delay between requests
            time.sleep(1)

    # Step 3: Generate report
    print(f"\n\nSTEP 4: Generating comparison report...")
    print("-"*80)

    report_path = Path(output_dir) / f'llm-comparison-{datetime.now().strftime("%Y-%m-%d-%H%M%S")}.md'
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# LLM Comparison Report: Gemini vs Claude\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Prompts Tested:** Sustainability, SEECE Energy Tech\n")
        f.write(f"**LLMs Tested:** Gemini 1.5 Pro, Claude 3.5 Sonnet\n")
        f.write(f"**Samples per filter:** {num_samples_per_filter}\n\n")

        f.write("---\n\n")

        # Summary statistics
        f.write("## Summary Statistics\n\n")

        for filter_name in ['sustainability', 'seece']:
            f.write(f"### {filter_name.title()} Prompt\n\n")
            f.write("| LLM | Success Rate | Avg Time | Total Time |\n")
            f.write("|-----|--------------|----------|------------|\n")

            for llm in ['gemini', 'claude']:
                test_results = results[filter_name][llm]
                successes = sum(1 for r in test_results if r['success'])
                total = len(test_results)
                success_rate = (successes / total * 100) if total > 0 else 0
                avg_time = sum(r['time_seconds'] for r in test_results) / total if total > 0 else 0
                total_time = sum(r['time_seconds'] for r in test_results)

                f.write(f"| **{llm.capitalize()}** | {successes}/{total} ({success_rate:.0f}%) | {avg_time:.1f}s | {total_time:.1f}s |\n")

            f.write("\n")

        # Detailed results
        f.write("---\n\n")
        f.write("## Detailed Results\n\n")

        for filter_name in ['sustainability', 'seece']:
            f.write(f"### {filter_name.title()} Prompt\n\n")

            # Show article-by-article comparison
            gemini_results = results[filter_name]['gemini']
            claude_results = results[filter_name]['claude']

            for i in range(len(gemini_results)):
                gemini_r = gemini_results[i]
                claude_r = claude_results[i]

                f.write(f"#### Article {i+1}: {gemini_r['article_title']}\n\n")
                f.write(f"**Source:** `{gemini_r['article_source']}`  \n")
                f.write(f"**ID:** `{gemini_r['article_id']}`\n\n")

                # Gemini results
                f.write("**Gemini Results:**\n")
                if gemini_r['success'] and gemini_r.get('analysis'):
                    f.write(f"- Time: {gemini_r['time_seconds']}s\n")
                    analysis = gemini_r['analysis']

                    # Show key scores for sustainability
                    if filter_name == 'sustainability':
                        f.write(f"- Climate Impact: {analysis.get('climate_impact_potential', 'N/A')}/10\n")
                        f.write(f"- Technical Credibility: {analysis.get('technical_credibility', 'N/A')}/10\n")
                        f.write(f"- Deployment Readiness: {analysis.get('deployment_readiness', 'N/A')}/10\n")
                        f.write(f"- Content Type: {analysis.get('content_type', 'N/A')}\n")
                        f.write(f"- Innovation Stage: {analysis.get('innovation_stage', 'N/A')}\n")
                    else:  # SEECE
                        f.write(f"- SEECE Relevance: {analysis.get('seece_relevance_score', 'N/A')}/10\n")
                        f.write(f"- Dutch/EU Policy: {analysis.get('seece_dimensions', {}).get('dutch_eu_policy_relevance', 'N/A')}/10\n")
                        f.write(f"- Applied Research Fit: {analysis.get('seece_dimensions', {}).get('applied_research_fit', 'N/A')}/10\n")
                        priority_topics = analysis.get('priority_topics', {})
                        active_topics = [k for k, v in priority_topics.items() if v]
                        f.write(f"- Priority Topics: {', '.join(active_topics) if active_topics else 'None'}\n")

                    f.write(f"- Reasoning: {analysis.get('reasoning', 'N/A')[:200]}...\n")
                elif gemini_r['success']:
                    f.write(f"- Time: {gemini_r['time_seconds']}s\n")
                    f.write(f"- **WARNING:** Success but no analysis data (possible parsing issue)\n")
                else:
                    f.write(f"- **FAILED:** {gemini_r.get('error', 'Unknown error')}\n")

                f.write("\n**Claude Results:**\n")
                if claude_r['success'] and claude_r.get('analysis'):
                    f.write(f"- Time: {claude_r['time_seconds']}s\n")
                    analysis = claude_r['analysis']

                    # Show key scores
                    if filter_name == 'sustainability':
                        f.write(f"- Climate Impact: {analysis.get('climate_impact_potential', 'N/A')}/10\n")
                        f.write(f"- Technical Credibility: {analysis.get('technical_credibility', 'N/A')}/10\n")
                        f.write(f"- Deployment Readiness: {analysis.get('deployment_readiness', 'N/A')}/10\n")
                        f.write(f"- Content Type: {analysis.get('content_type', 'N/A')}\n")
                        f.write(f"- Innovation Stage: {analysis.get('innovation_stage', 'N/A')}\n")
                    else:  # SEECE
                        f.write(f"- SEECE Relevance: {analysis.get('seece_relevance_score', 'N/A')}/10\n")
                        f.write(f"- Dutch/EU Policy: {analysis.get('seece_dimensions', {}).get('dutch_eu_policy_relevance', 'N/A')}/10\n")
                        f.write(f"- Applied Research Fit: {analysis.get('seece_dimensions', {}).get('applied_research_fit', 'N/A')}/10\n")
                        priority_topics = analysis.get('priority_topics', {})
                        active_topics = [k for k, v in priority_topics.items() if v]
                        f.write(f"- Priority Topics: {', '.join(active_topics) if active_topics else 'None'}\n")

                    f.write(f"- Reasoning: {analysis.get('reasoning', 'N/A')[:200]}...\n")
                elif claude_r['success']:
                    f.write(f"- Time: {claude_r['time_seconds']}s\n")
                    f.write(f"- **WARNING:** Success but no analysis data (possible parsing issue)\n")
                else:
                    f.write(f"- **FAILED:** {claude_r.get('error', 'Unknown error')}\n")

                f.write("\n**Comparison:**\n")
                if gemini_r['success'] and claude_r['success'] and gemini_r.get('analysis') and claude_r.get('analysis'):
                    if filter_name == 'sustainability':
                        gemini_climate = gemini_r['analysis'].get('climate_impact_potential', 0)
                        claude_climate = claude_r['analysis'].get('climate_impact_potential', 0)
                        diff = abs(gemini_climate - claude_climate)
                        f.write(f"- Climate Impact difference: {diff} points\n")

                        gemini_cred = gemini_r['analysis'].get('technical_credibility', 0)
                        claude_cred = claude_r['analysis'].get('technical_credibility', 0)
                        diff_cred = abs(gemini_cred - claude_cred)
                        f.write(f"- Technical Credibility difference: {diff_cred} points\n")
                    else:  # SEECE
                        gemini_seece = gemini_r['analysis'].get('seece_relevance_score', 0)
                        claude_seece = claude_r['analysis'].get('seece_relevance_score', 0)
                        diff = abs(gemini_seece - claude_seece)
                        f.write(f"- SEECE Relevance difference: {diff:.1f} points\n")

                    f.write(f"- Time difference: {abs(gemini_r['time_seconds'] - claude_r['time_seconds']):.1f}s\n")
                else:
                    f.write("- Cannot compare (one or both failed or missing analysis)\n")

                f.write("\n---\n\n")

        # Conclusions
        f.write("## Conclusions\n\n")
        f.write("### Success Rates\n\n")
        for filter_name in ['sustainability', 'seece']:
            gemini_success = sum(1 for r in results[filter_name]['gemini'] if r['success'])
            claude_success = sum(1 for r in results[filter_name]['claude'] if r['success'])
            total = len(results[filter_name]['gemini'])

            f.write(f"**{filter_name.title()}:**\n")
            f.write(f"- Gemini: {gemini_success}/{total} ({gemini_success/total*100:.0f}%)\n")
            f.write(f"- Claude: {claude_success}/{total} ({claude_success/total*100:.0f}%)\n\n")

        f.write("### Performance\n\n")
        for filter_name in ['sustainability', 'seece']:
            gemini_times = [r['time_seconds'] for r in results[filter_name]['gemini'] if r['success']]
            claude_times = [r['time_seconds'] for r in results[filter_name]['claude'] if r['success']]

            gemini_avg = sum(gemini_times) / len(gemini_times) if gemini_times else 0
            claude_avg = sum(claude_times) / len(claude_times) if claude_times else 0

            f.write(f"**{filter_name.title()}:**\n")
            f.write(f"- Gemini average: {gemini_avg:.1f}s\n")
            f.write(f"- Claude average: {claude_avg:.1f}s\n")
            if gemini_avg > 0 and claude_avg > 0:
                faster = "Gemini" if gemini_avg < claude_avg else "Claude"
                speedup = max(gemini_avg, claude_avg) / min(gemini_avg, claude_avg)
                f.write(f"- **{faster} is {speedup:.1f}x faster**\n")
            f.write("\n")

        f.write("### Recommendations\n\n")
        f.write("Based on the test results:\n\n")
        f.write("1. **Success Rate:** Review which LLM had fewer failures\n")
        f.write("2. **Performance:** Consider speed vs cost tradeoff\n")
        f.write("3. **Scoring Consistency:** Check if both LLMs agree on article quality\n")
        f.write("4. **Filter Quality:** Verify that passed articles are truly relevant\n\n")

        f.write("### Next Steps\n\n")
        f.write("1. Review article-by-article comparisons above\n")
        f.write("2. Check if filter pass rates need adjustment (sustainability: 2.9%, SEECE: 5.3%)\n")
        f.write("3. Decide on LLM provider for production runs\n")
        f.write("4. Consider running larger samples (50-100 articles) for statistical significance\n\n")

    print(f"\nReport saved to: {report_path}")
    print("\n" + "="*80)
    print("Test complete!")
    print("="*80)

    return results, report_path


if __name__ == '__main__':
    results, report_path = test_llm_comparison(
        dataset_path='datasets/raw/master_dataset.jsonl',
        num_samples_per_filter=5,  # Test 5 articles per filter
        output_dir='reports'
    )

    print(f"\nQuick summary:")
    print(f"  Sustainability - Gemini: {sum(1 for r in results['sustainability']['gemini'] if r['success'])}/5 success")
    print(f"  Sustainability - Claude: {sum(1 for r in results['sustainability']['claude'] if r['success'])}/5 success")
    print(f"  SEECE - Gemini: {sum(1 for r in results['seece']['gemini'] if r['success'])}/5 success")
    print(f"  SEECE - Claude: {sum(1 for r in results['seece']['claude'] if r['success'])}/5 success")
    print(f"\nFull report: {report_path}")
