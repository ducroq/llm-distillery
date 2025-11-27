"""
Oracle Calibration - 1,000 Article Sample

This script performs an expanded calibration of the oracle (Gemini + prompt) on 1,000 articles
to validate scoring quality, consistency, and dimension independence before generating the
full 10K training dataset.

Usage:
    python scripts/calibrate_oracle_1k.py

Output:
    - datasets/calibration/calibration_1k_YYYYMMDD_HHMMSS/
        - articles_sampled.jsonl (1,020 articles: 1,000 unique + 20 duplicates)
        - articles_scored.jsonl (1,020 scored articles)
        - calibration_stats.json (statistics and metadata)
        - calibration_log.txt (execution log)
"""

import json
import random
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from filters.sustainability_technology.v1.prefilter import SustainabilityTechnologyPreFilterV1
from filters.sustainability_technology.v1.oracle import SustainabilityTechnologyOracleV1


class CalibrationRunner:
    """Run oracle calibration on 1,000 article sample."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.log_file = output_dir / "calibration_log.txt"
        self.log_file.write_text(f"Calibration started at {datetime.now().isoformat()}\n")

        # Initialize filter and oracle
        self.prefilter = SustainabilityTechnologyPreFilterV1()
        self.oracle = SustainabilityTechnologyOracleV1()

        # Stats
        self.stats = defaultdict(int)

    def log(self, message: str):
        """Log message to console and file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        print(log_line)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_line + '\n')

    def load_master_dataset(self, dataset_path: Path) -> List[Dict]:
        """Load all articles from master dataset."""
        self.log(f"Loading master dataset from {dataset_path}...")

        articles = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        article = json.loads(line)
                        articles.append(article)
                    except json.JSONDecodeError:
                        self.stats['load_errors'] += 1

        self.log(f"Loaded {len(articles):,} articles")
        return articles

    def apply_prefilter(self, articles: List[Dict]) -> List[Dict]:
        """Apply prefilter to get candidate articles."""
        self.log("Applying prefilter to master dataset...")

        passed_articles = []
        for article in articles:
            should_score, reason = self.prefilter.apply_filter(article)

            if should_score:
                passed_articles.append(article)
                self.stats['prefilter_passed'] += 1
            else:
                self.stats['prefilter_blocked'] += 1

        pass_rate = (len(passed_articles) / len(articles)) * 100 if articles else 0
        self.log(f"Prefilter: {len(passed_articles):,} passed ({pass_rate:.1f}%), {self.stats['prefilter_blocked']:,} blocked")

        return passed_articles

    def sample_articles(self, articles: List[Dict], n: int = 1000) -> Tuple[List[Dict], List[int]]:
        """
        Sample n articles randomly + 20 duplicates for consistency testing.

        Returns:
            - List of sampled articles (1,020 total)
            - List of duplicate indices (which articles are duplicates)
        """
        self.log(f"Sampling {n} articles + 20 duplicates for consistency testing...")

        if len(articles) < n:
            self.log(f"WARNING: Only {len(articles)} articles available, sampling all")
            n = len(articles)

        # Sample main set
        sampled = random.sample(articles, n)

        # Add 20 duplicates for consistency testing
        duplicate_indices = random.sample(range(n), 20)
        duplicates = [sampled[i] for i in duplicate_indices]

        # Mark duplicates with metadata
        for i, dup_article in enumerate(duplicates):
            dup_article['_duplicate_of_index'] = duplicate_indices[i]
            dup_article['_duplicate_id'] = i

        # Combine and shuffle
        all_sampled = sampled + duplicates
        random.shuffle(all_sampled)

        self.log(f"Sampled {len(all_sampled)} articles total (1,000 unique + 20 duplicates)")

        return all_sampled, duplicate_indices

    def score_articles(self, articles: List[Dict], output_path: Path) -> List[Dict]:
        """Score all articles using oracle, saving incrementally."""

        # Check for existing progress
        already_scored_urls = set()
        if output_path.exists():
            self.log(f"Found existing progress in {output_path.name}, loading...")
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        article = json.loads(line)
                        if 'oracle_scores' in article or 'oracle_error' in article:
                            already_scored_urls.add(article.get('url'))
            self.log(f"Already scored: {len(already_scored_urls)} articles")
            self.log(f"Remaining: {len(articles) - len(already_scored_urls)} articles")

        self.log(f"Scoring {len(articles)} articles with oracle...")
        self.log("This will take approximately 15-20 minutes...")
        self.log(f"Progress will be saved incrementally to {output_path.name}")

        scored_articles = []

        # Open output file in append mode for incremental saving
        with open(output_path, 'a', encoding='utf-8') as f:
            for i, article in enumerate(articles, 1):
                # Skip if already scored
                if article.get('url') in already_scored_urls:
                    scored_articles.append(article)
                    self.stats['scored_success'] += 1
                    continue

                if i % 50 == 0:
                    self.log(f"  Progress: {i}/{len(articles)} ({i/len(articles)*100:.1f}%)")

                try:
                    # Score article
                    scores, reasoning = self.oracle.score_article(article)

                    # Add scores and reasoning to article
                    article['oracle_scores'] = scores
                    article['oracle_reasoning'] = reasoning
                    article['oracle_timestamp'] = datetime.now().isoformat()

                    scored_articles.append(article)
                    self.stats['scored_success'] += 1

                except Exception as e:
                    self.log(f"  ERROR scoring article {i}: {e}")
                    self.stats['scoring_errors'] += 1

                    # Still save article with error info
                    article['oracle_error'] = str(e)
                    scored_articles.append(article)

                # Save immediately (incremental save)
                f.write(json.dumps(article, ensure_ascii=False) + '\n')
                f.flush()  # Ensure it's written to disk

        success_rate = (self.stats['scored_success'] / len(articles)) * 100 if articles else 0
        self.log(f"Scoring complete: {self.stats['scored_success']} success ({success_rate:.1f}%), {self.stats['scoring_errors']} errors")

        return scored_articles

    def save_sampled_articles(self, articles: List[Dict], output_path: Path):
        """Save sampled articles (before scoring)."""
        self.log(f"Saving sampled articles to {output_path.name}...")

        with open(output_path, 'w', encoding='utf-8') as f:
            for article in articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')

        self.log(f"Saved {len(articles)} sampled articles")

    def save_scored_articles(self, articles: List[Dict], output_path: Path):
        """Save scored articles."""
        self.log(f"Saving scored articles to {output_path.name}...")

        with open(output_path, 'w', encoding='utf-8') as f:
            for article in articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')

        self.log(f"Saved {len(articles)} scored articles")

    def generate_stats(self, scored_articles: List[Dict]) -> Dict:
        """Generate calibration statistics."""
        self.log("Generating statistics...")

        stats = {
            'timestamp': datetime.now().isoformat(),
            'total_sampled': len(scored_articles),
            'successfully_scored': self.stats['scored_success'],
            'scoring_errors': self.stats['scoring_errors'],
            'prefilter_stats': {
                'passed': self.stats['prefilter_passed'],
                'blocked': self.stats['prefilter_blocked']
            },
            'dimensions': [
                'technology_readiness_level',
                'technical_performance',
                'economic_competitiveness',
                'life_cycle_environmental_impact',
                'social_equity_impact',
                'governance_systemic_impact'
            ]
        }

        # Calculate score distributions
        dimension_scores = defaultdict(list)
        for article in scored_articles:
            if 'oracle_scores' in article:
                scores = article['oracle_scores']
                for dim in stats['dimensions']:
                    if dim in scores and scores[dim] is not None:
                        dimension_scores[dim].append(scores[dim])

        # Calculate statistics per dimension
        dimension_stats = {}
        for dim, scores in dimension_scores.items():
            if scores:
                dimension_stats[dim] = {
                    'mean': sum(scores) / len(scores),
                    'min': min(scores),
                    'max': max(scores),
                    'count': len(scores)
                }

        stats['dimension_stats'] = dimension_stats

        return stats

    def save_stats(self, stats: Dict, output_path: Path):
        """Save calibration statistics."""
        self.log(f"Saving statistics to {output_path.name}...")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        self.log("Statistics saved")


def main():
    """Main execution function."""
    print("="*70)
    print("Oracle Calibration - 1,000 Article Sample")
    print("="*70)
    print()

    # Configuration
    master_dataset_path = Path("datasets/raw/master_dataset_20251009_20251124.jsonl")

    # Check for --resume flag or provide directory to resume
    import sys
    resume_dir = None
    if len(sys.argv) > 1:
        if sys.argv[1] == "--resume" and len(sys.argv) > 2:
            resume_dir = Path(sys.argv[2])
        elif Path(sys.argv[1]).exists():
            resume_dir = Path(sys.argv[1])

    if resume_dir:
        output_dir = resume_dir
        print(f"RESUMING calibration from: {output_dir}")
        print()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"datasets/calibration/calibration_1k_{timestamp}")

    # Check if master dataset exists
    if not master_dataset_path.exists():
        print(f"ERROR: Master dataset not found at {master_dataset_path}")
        print("Please ensure the dataset exists or update the path in the script.")
        return

    # Initialize runner
    runner = CalibrationRunner(output_dir)

    try:
        # Check if resuming
        sampled_path = output_dir / "articles_sampled.jsonl"
        if sampled_path.exists():
            # Resume: Load existing sampled articles
            runner.log("Resuming from existing calibration...")
            runner.log(f"Loading sampled articles from {sampled_path.name}")
            sampled_articles = []
            with open(sampled_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        sampled_articles.append(json.loads(line))
            runner.log(f"Loaded {len(sampled_articles)} sampled articles")
        else:
            # New run: Sample articles
            # Step 1: Load master dataset
            all_articles = runner.load_master_dataset(master_dataset_path)

            if not all_articles:
                runner.log("ERROR: No articles loaded from master dataset")
                return

            # Step 2: Apply prefilter
            passed_articles = runner.apply_prefilter(all_articles)

            if not passed_articles:
                runner.log("ERROR: No articles passed prefilter")
                return

            if len(passed_articles) < 1000:
                runner.log(f"WARNING: Only {len(passed_articles)} articles passed prefilter (target: 1000)")

            # Step 3: Sample articles (1,000 + 20 duplicates)
            sampled_articles, duplicate_indices = runner.sample_articles(passed_articles, n=1000)

            # Step 4: Save sampled articles (before scoring)
            runner.save_sampled_articles(sampled_articles, output_dir / "articles_sampled.jsonl")

        # Step 5: Score articles (saves incrementally to articles_scored.jsonl)
        scored_output_path = output_dir / "articles_scored.jsonl"
        scored_articles = runner.score_articles(sampled_articles, scored_output_path)

        # Step 7: Generate and save statistics
        stats = runner.generate_stats(scored_articles)
        runner.save_stats(stats, output_dir / "calibration_stats.json")

        # Final summary
        print()
        print("="*70)
        print("CALIBRATION COMPLETE")
        print("="*70)
        print(f"Output directory: {output_dir}")
        print(f"Articles sampled:  {len(sampled_articles)}")
        print(f"Articles scored:   {runner.stats['scored_success']}")
        print(f"Errors:            {runner.stats['scoring_errors']}")
        print()
        print("Next steps:")
        print("  1. Review calibration_stats.json for score distributions")
        print("  2. Run analysis script: python scripts/analyze_calibration_1k.py")
        print("  3. Manual validation of sample articles")
        print("="*70)

    except Exception as e:
        runner.log(f"FATAL ERROR: {e}")
        import traceback
        runner.log(traceback.format_exc())
        raise


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    main()
