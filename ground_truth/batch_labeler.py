"""
Generic batch labeling using Claude/Gemini API for ground truth generation.

This is a refactored, prompt-agnostic version that works with ANY semantic filter.
Migrated from NexusMind-Filter/scripts/batch_label_with_claude.py

Usage:
    python -m ground_truth.batch_labeler \
        --prompt prompts/sustainability.md \
        --source ../content-aggregator/data/collected/articles.jsonl \
        --output datasets/sustainability_5k.jsonl \
        --llm claude \
        --batch-size 50 \
        --max-batches 100
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional
import anthropic
import google.generativeai as genai
from datetime import datetime

# Import secrets manager
from .secrets_manager import get_secrets_manager


class GenericBatchLabeler:
    """
    Generic batch labeler that works with any semantic filter prompt.

    Key improvements over NexusMind version:
    - Prompt-agnostic (loads from markdown files)
    - Supports multiple LLM providers (Claude, Gemini, GPT-4)
    - Configurable output structure
    - No hardcoded filter logic
    """

    def __init__(
        self,
        prompt_path: str,
        llm_provider: str = "claude",
        api_key: Optional[str] = None,
        output_dir: str = "datasets",
        filter_name: Optional[str] = None
    ):
        """
        Initialize batch labeler.

        Args:
            prompt_path: Path to prompt markdown file (e.g., prompts/sustainability.md)
            llm_provider: "claude", "gemini", or "gpt4"
            api_key: API key (or None to use environment variables)
            output_dir: Directory to save labeled data
            filter_name: Name of filter (auto-detected from prompt_path if None)
        """
        self.prompt_path = Path(prompt_path)
        self.llm_provider = llm_provider.lower()

        # Auto-detect filter name from prompt filename
        if filter_name is None:
            self.filter_name = self.prompt_path.stem  # e.g., "sustainability" from "sustainability.md"
        else:
            self.filter_name = filter_name

        # Load prompt template
        self.prompt_template = self._load_prompt_template()

        # Initialize LLM client
        self.llm_client = self._init_llm_client(api_key)

        # Setup output directory
        self.output_dir = Path(output_dir) / self.filter_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load state
        self.state_file = self.output_dir / '.labeled_ids.json'
        self.state = self._load_state()

    def _load_prompt_template(self) -> str:
        """Load prompt template from markdown file."""
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_path}")

        with open(self.prompt_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract prompt from markdown code blocks
        start_marker = "```\nAnalyze this article"
        end_marker = "DO NOT include any text outside the JSON object.\n```"

        start_idx = content.find(start_marker)
        end_idx = content.find(end_marker, start_idx)

        if start_idx == -1 or end_idx == -1:
            raise ValueError(
                f"Could not find prompt template in {self.prompt_path}\n"
                f"Expected format:\n"
                f"```\n"
                f"Analyze this article...\n"
                f"DO NOT include any text outside the JSON object.\n"
                f"```"
            )

        prompt = content[start_idx + 4:end_idx + len(end_marker) - 4]
        return prompt.strip()

    def _init_llm_client(self, api_key: Optional[str]):
        """Initialize LLM client based on provider."""
        # Use SecretsManager if no API key provided
        if api_key is None:
            secrets = get_secrets_manager()
            api_key = secrets.get_llm_key(self.llm_provider)

        if self.llm_provider == "claude":
            if not api_key:
                raise ValueError(
                    "Claude API key not found. Set ANTHROPIC_API_KEY in environment or secrets.ini"
                )
            return anthropic.Anthropic(api_key=api_key)

        elif self.llm_provider == "gemini":
            if not api_key:
                raise ValueError(
                    "Gemini API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY in environment or secrets.ini"
                )
            genai.configure(api_key=api_key)
            return genai.GenerativeModel('gemini-2.5-pro')

        elif self.llm_provider == "gpt4":
            import openai
            if not api_key:
                raise ValueError(
                    "OpenAI API key not found. Set OPENAI_API_KEY in environment or secrets.ini"
                )
            return openai.OpenAI(api_key=api_key)

        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def _load_state(self) -> Dict:
        """Load processing state for resume capability."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            'processed': [],
            'total_labeled': 0,
            'batches_completed': 0,
            'last_updated': None,
            'filter_name': self.filter_name,
            'llm_provider': self.llm_provider
        }

    def _save_state(self):
        """Save processing state."""
        self.state['last_updated'] = datetime.utcnow().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def _post_process_uplifting(self, analysis: Dict) -> Dict:
        """
        Post-process uplifting filter analysis to calculate tier and overall_uplift_score.
        This matches the NexusMind-Filter implementation.
        """
        # Dimension weights
        weights = {
            'agency': 0.14,
            'progress': 0.19,
            'collective_benefit': 0.38,
            'connection': 0.10,
            'innovation': 0.08,
            'justice': 0.04,
            'resilience': 0.02,
            'wonder': 0.05
        }

        # Extract dimensions
        dimensions = {k: analysis.get(k, 0) for k in weights.keys()}

        # Calculate base score
        base_score = sum(dimensions[k] * weights[k] for k in dimensions)

        # Apply content-type caps
        content_type = analysis.get('content_type', '')
        max_score = 10.0

        if content_type == "corporate_finance":
            max_score = 2.0
        elif content_type == "military_security":
            max_score = 4.0
        elif content_type == "business_news" and dimensions['collective_benefit'] < 6:
            max_score = 4.0

        # Apply gatekeeper rule
        if dimensions['collective_benefit'] < 5:
            # Wonder exemption
            if dimensions['wonder'] >= 7 and dimensions['collective_benefit'] >= 3:
                pass  # No cap
            else:
                max_score = min(max_score, 3.0)

        # Apply cap
        final_score = min(base_score, max_score)

        # Determine tier
        if final_score >= 7.0:
            tier = "impact"
        elif final_score >= 4.0:
            tier = "connection"
        else:
            tier = "not_uplifting"

        # Add calculated fields to analysis
        analysis['dimensions'] = dimensions
        analysis['overall_uplift_score'] = round(final_score, 2)
        analysis['tier'] = tier

        return analysis

    def build_prompt(self, article: Dict) -> str:
        """Build prompt by filling in article data."""
        return self.prompt_template.format(
            title=article.get('title', 'N/A'),
            source=article.get('source', 'N/A'),
            published_date=article.get('published_date', 'N/A'),
            text=article.get('content', '')[:4000]  # Truncate to fit context window
        )

    def analyze_article(self, article: Dict, timeout_seconds: int = 60) -> Optional[Dict]:
        """Analyze a single article using LLM with timeout protection."""
        prompt = self.build_prompt(article)

        try:
            # Use threading.Timer for cross-platform timeout
            import threading

            result = [None]  # Mutable container for thread result
            exception = [None]  # Mutable container for exceptions

            def call_llm():
                try:
                    if self.llm_provider == "claude":
                        message = self.llm_client.messages.create(
                            model="claude-3-5-sonnet-20241022",
                            max_tokens=2048,
                            temperature=0.3,  # Lower for more consistent ground truth
                            system="You are an expert analyst. You respond only with valid JSON following the exact format specified.",
                            messages=[{"role": "user", "content": prompt}]
                        )
                        result[0] = message.content[0].text.strip()

                    elif self.llm_provider == "gemini":
                        response = self.llm_client.generate_content(
                            prompt,
                            generation_config=genai.types.GenerationConfig(
                                temperature=0.3,
                                max_output_tokens=2048,
                            )
                        )
                        result[0] = response.text.strip()

                    elif self.llm_provider == "gpt4":
                        response = self.llm_client.chat.completions.create(
                            model="gpt-4-turbo-preview",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.3,
                            max_tokens=2048,
                        )
                        result[0] = response.choices[0].message.content.strip()
                except Exception as e:
                    exception[0] = e

            # Run LLM call in thread with timeout
            thread = threading.Thread(target=call_llm)
            thread.daemon = True
            thread.start()
            thread.join(timeout=timeout_seconds)

            if thread.is_alive():
                print(f"  ⏱️  Timeout after {timeout_seconds}s for article {article.get('id')}")
                return None

            if exception[0]:
                raise exception[0]

            response_text = result[0]
            if not response_text:
                print(f"  ⚠️  No response for article {article.get('id')}")
                return None

            # Remove markdown formatting if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            # Parse JSON
            analysis = json.loads(response_text.strip())

            # Filter-specific post-processing
            if self.filter_name == 'uplifting':
                analysis = self._post_process_uplifting(analysis)

            # Add metadata
            analysis['analyzed_at'] = datetime.utcnow().isoformat() + 'Z'
            analysis['analyzed_by'] = f'{self.llm_provider}-api-batch'
            analysis['filter_name'] = self.filter_name

            return analysis

        except json.JSONDecodeError as e:
            print(f"  ⚠️  JSON decode error for article {article.get('id')}: {e}")
            print(f"     Response: {response_text[:200]}...")
            return None
        except Exception as e:
            print(f"  ⚠️  Error analyzing article {article.get('id')}: {e}")
            return None

    def process_batch(self, articles: List[Dict], batch_num: int) -> Dict:
        """Process a batch of articles."""
        results = []
        processed_ids = []

        print(f"\n{'='*60}")
        print(f"Processing batch {batch_num} ({len(articles)} articles)")
        print(f"{'='*60}")

        for i, article in enumerate(articles, 1):
            article_id = article.get('id')

            # Skip if already processed
            if article_id in self.state['processed']:
                print(f"  [{i}/{len(articles)}] ⏭️  Skipping {article_id} (already processed)")
                continue

            print(f"  [{i}/{len(articles)}] 🔄 Analyzing {article_id}...")

            analysis = self.analyze_article(article)

            if analysis:
                # Add analysis to article
                article[f'{self.filter_name}_analysis'] = analysis

                results.append(article)
                processed_ids.append(article_id)

                # Rate limiting based on provider
                if self.llm_provider == "claude":
                    time.sleep(1.5)  # 50 RPM limit → ~40 req/min to be safe
                elif self.llm_provider == "gemini":
                    time.sleep(0.5)  # 150 RPM limit (Tier 1) → ~120 req/min
                elif self.llm_provider == "gpt4":
                    time.sleep(1.0)  # Vary based on tier

                print(f"     ✅ Success")
            else:
                print(f"     ❌ Failed to analyze")

        # Save batch results
        if results:
            output_file = self.output_dir / f'labeled_batch_{batch_num:03d}.jsonl'
            with open(output_file, 'w', encoding='utf-8') as f:
                for article in results:
                    f.write(json.dumps(article, ensure_ascii=False, separators=(',', ':')) + '\n')

            print(f"\n💾 Saved {len(results)} labeled articles to {output_file.name}")

        # Update state
        self.state['processed'].extend(processed_ids)
        self.state['total_labeled'] += len(processed_ids)
        self.state['batches_completed'] += 1
        self._save_state()

        return {
            'batch_num': batch_num,
            'articles_processed': len(results),
            'articles_failed': len(articles) - len(results),
        }

    def load_unlabeled_articles(
        self,
        source_file: str,
        batch_size: int = 50,
        pre_filter: Optional[callable] = None
    ) -> List[Dict]:
        """
        Load unlabeled articles from source file.

        Args:
            source_file: Path to JSONL file with articles
            batch_size: Number of articles to load
            pre_filter: Optional function to pre-filter articles
                       Should return True to include article, False to skip
        """
        articles = []
        processed_ids = set(self.state['processed'])

        with open(source_file, 'r', encoding='utf-8') as f:
            for line in f:
                if len(articles) >= batch_size:
                    break

                try:
                    article = json.loads(line.strip())
                    article_id = article.get('id')

                    # Skip if already processed
                    if article_id in processed_ids:
                        continue

                    # Apply pre-filter if provided
                    if pre_filter and not pre_filter(article):
                        continue

                    articles.append(article)
                except:
                    continue

        return articles

    def run(
        self,
        source_file: str,
        max_batches: Optional[int] = None,
        batch_size: int = 50,
        pre_filter: Optional[callable] = None
    ):
        """
        Run batch labeling process.

        Args:
            source_file: Path to JSONL file with articles
            max_batches: Maximum number of batches to process (None = unlimited)
            batch_size: Articles per batch
            pre_filter: Optional function to pre-filter articles before labeling
        """
        print(f"\n🥃 LLM Distillery - Batch Labeling")
        print(f"{'='*60}")
        print(f"Filter: {self.filter_name}")
        print(f"Prompt: {self.prompt_path}")
        print(f"LLM: {self.llm_provider}")
        print(f"Source: {source_file}")
        print(f"Output: {self.output_dir}")
        print(f"Batch size: {batch_size}")
        print(f"Max batches: {max_batches or 'unlimited'}")
        print(f"Already processed: {len(self.state['processed'])} articles")
        print(f"{'='*60}\n")

        batch_num = self.state['batches_completed'] + 1
        total_processed = 0
        total_failed = 0

        while True:
            # Check if we've hit max batches
            if max_batches and batch_num > self.state['batches_completed'] + max_batches:
                print(f"\n✋ Reached max batches ({max_batches})")
                break

            # Load next batch
            articles = self.load_unlabeled_articles(source_file, batch_size, pre_filter)

            if not articles:
                print(f"\n🏁 No more unlabeled articles found")
                break

            # Process batch
            result = self.process_batch(articles, batch_num)
            total_processed += result['articles_processed']
            total_failed += result['articles_failed']

            print(f"\n📊 Batch {batch_num} Summary:")
            print(f"   ✅ Processed: {result['articles_processed']}")
            print(f"   ❌ Failed: {result['articles_failed']}")

            batch_num += 1

        # Final summary
        print(f"\n{'='*60}")
        print(f"🎉 Batch Labeling Complete!")
        print(f"{'='*60}")
        print(f"Articles labeled this run: {total_processed}")
        print(f"Articles failed this run: {total_failed}")
        print(f"Total articles labeled: {self.state['total_labeled']}")
        print(f"Total batches completed: {self.state['batches_completed']}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")


def uplifting_pre_filter(article: Dict) -> bool:
    """
    Pre-filter for uplifting content (migrated from NexusMind).
    Only analyze articles with VADER >= 5.0 OR joy >= 0.25.
    This reduces labeling costs by ~50% with minimal false negatives.
    """
    sentiment_score = article.get('metadata', {}).get('sentiment_score', 0)
    joy = article.get('metadata', {}).get('raw_emotions', {}).get('joy', 0)
    return sentiment_score >= 5.0 or joy >= 0.25


def sustainability_pre_filter(article: Dict) -> bool:
    """
    Pre-filter for sustainability content.
    Only analyze articles likely related to climate/environment/energy.
    """
    # Check if source category is sustainability-related
    category = article.get('metadata', {}).get('source_category', '')
    sustainability_categories = [
        'climate_solutions', 'energy_utilities', 'renewable_energy',
        'automotive_transport', 'science', 'economics'
    ]

    if category in sustainability_categories:
        return True

    # Or check for sustainability keywords in title/content
    text = (article.get('title', '') + ' ' + article.get('content', '')).lower()
    keywords = [
        'climate', 'carbon', 'renewable', 'solar', 'wind', 'battery',
        'ev', 'electric', 'sustainability', 'green', 'emission'
    ]

    return any(kw in text for kw in keywords)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Generic batch labeling for ground truth generation'
    )
    parser.add_argument(
        '--prompt',
        required=True,
        help='Path to prompt markdown file (e.g., prompts/sustainability.md)'
    )
    parser.add_argument(
        '--source',
        required=True,
        help='Source JSONL file with articles'
    )
    parser.add_argument(
        '--output-dir',
        default='datasets',
        help='Output directory for labeled data (default: datasets/)'
    )
    parser.add_argument(
        '--llm',
        default='claude',
        choices=['claude', 'gemini', 'gpt4'],
        help='LLM provider (default: claude)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='Articles per batch (default: 50)'
    )
    parser.add_argument(
        '--max-batches',
        type=int,
        help='Maximum number of batches to process (default: unlimited)'
    )
    parser.add_argument(
        '--pre-filter',
        choices=['uplifting', 'sustainability', 'none'],
        default='none',
        help='Pre-filter to apply before labeling'
    )
    parser.add_argument(
        '--api-key',
        help='API key (or set via environment variable)'
    )

    args = parser.parse_args()

    # Select pre-filter
    pre_filter_func = None
    if args.pre_filter == 'uplifting':
        pre_filter_func = uplifting_pre_filter
    elif args.pre_filter == 'sustainability':
        pre_filter_func = sustainability_pre_filter

    # Create labeler
    labeler = GenericBatchLabeler(
        prompt_path=args.prompt,
        llm_provider=args.llm,
        api_key=args.api_key,
        output_dir=args.output_dir
    )

    # Run labeling
    labeler.run(
        source_file=args.source,
        max_batches=args.max_batches,
        batch_size=args.batch_size,
        pre_filter=pre_filter_func
    )
