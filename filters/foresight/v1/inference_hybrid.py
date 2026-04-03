"""
Foresight Filter v1 - Hybrid Inference Pipeline

Two-stage scorer that uses embedding + MLP probe for fast screening (Stage 1)
and the trained Gemma-3-1B model for precise scoring (Stage 2).

Stage 1 (~1.3ms): Embedding probe estimates scores. Articles with weighted_avg
below threshold are classified as LOW without running the expensive model.

Stage 2 (~20ms): Full fine-tuned model scoring for articles that pass Stage 1.

Usage:
    from filters.foresight.v1.inference_hybrid import ForesightHybridScorer

    scorer = ForesightHybridScorer()
    result = scorer.score_article(article)
    # result["stage_used"] -> "stage1_low" or "stage2"

    # CLI
    python filters/foresight/v1/inference_hybrid.py --input articles.jsonl --output results.jsonl
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

from filters.common.hybrid_scorer import HybridScorer
from filters.foresight.v1.inference import ForesightScorer

logger = logging.getLogger(__name__)

# Default Stage 1 threshold: articles below this skip Stage 2
# TODO: Calibrate on production data
DEFAULT_THRESHOLD = 2.25


class ForesightHybridScorer(HybridScorer):
    """
    Two-stage hybrid scorer for foresight filter v1.

    Combines:
    - Stage 1: multilingual-e5-small embeddings + MLP probe
    - Stage 2: Existing ForesightScorer (Gemma-3-1B fine-tuned)
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        probe_path: Optional[Path] = None,
        threshold: float = DEFAULT_THRESHOLD,
        device: Optional[str] = None,
        use_prefilter: bool = True,
    ):
        self._model_path = model_path
        self._probe_path = probe_path or (
            Path(__file__).parent / "probe" / "embedding_probe_e5small.pkl"
        )
        self._threshold = threshold

        super().__init__(device=device, use_prefilter=use_prefilter)

    def _create_stage2_scorer(self):
        """Create the existing ForesightScorer as Stage 2.

        Prefilter is disabled: HybridScorer handles prefiltering itself,
        so Stage 2 doesn't need to load or run the prefilter again.
        """
        return ForesightScorer(
            model_path=self._model_path,
            device=self.device_str,
            use_prefilter=False,
        )

    def _get_embedding_stage_config(self) -> Dict:
        """Return EmbeddingStage configuration for foresight v1."""
        return {
            "embedding_model_name": "intfloat/multilingual-e5-small",
            "probe_path": str(self._probe_path),
            "threshold": self._threshold,
        }


def main():
    """CLI interface for hybrid batch scoring."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Score articles with foresight hybrid scorer (two-stage pipeline)"
    )
    parser.add_argument(
        "--input", "-i", type=Path, help="Input JSONL file with articles"
    )
    parser.add_argument(
        "--output", "-o", type=Path, help="Output JSONL file for results"
    )
    parser.add_argument(
        "--no-prefilter", action="store_true", help="Skip prefilter"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for inference"
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Stage 1 threshold (default: {DEFAULT_THRESHOLD})"
    )

    args = parser.parse_args()

    print("Initializing hybrid scorer...")
    scorer = ForesightHybridScorer(
        threshold=args.threshold,
        use_prefilter=not args.no_prefilter,
    )

    if args.input:
        print(f"Loading articles from {args.input}")
        articles = []
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    articles.append(json.loads(line))

        print(f"Scoring {len(articles)} articles with hybrid pipeline...")
        import time
        start = time.time()
        results = scorer.score_batch(articles, batch_size=args.batch_size)
        hybrid_time = time.time() - start

        stage1_low = sum(1 for r in results if r.get("stage_used") == "stage1_low")
        stage2 = sum(1 for r in results if r.get("stage_used") == "stage2")
        prefilter_blocked = sum(1 for r in results if not r.get("passed_prefilter", True))

        print(f"\nHybrid results ({hybrid_time:.2f}s):")
        print(f"  Prefilter blocked: {prefilter_blocked}")
        print(f"  Stage 1 LOW (skipped model): {stage1_low}")
        print(f"  Stage 2 (full model): {stage2}")
        print(f"  Avg time per article: {hybrid_time/len(articles)*1000:.1f}ms")

        tiers = {}
        for r in results:
            tier = r.get("tier")
            if tier:
                tiers[tier] = tiers.get(tier, 0) + 1
        print(f"\nTier distribution:")
        for tier, count in sorted(tiers.items()):
            print(f"  {tier}: {count}")

        if args.output:
            print(f"\nWriting results to {args.output}")
            with open(args.output, "w", encoding="utf-8") as f:
                for article, result in zip(articles, results):
                    article_id = article.get("id") or article.get("article_id", "")
                    output = {"article_id": article_id, **result}
                    f.write(json.dumps(output) + "\n")
    else:
        print("\n--- Hybrid Scorer Demo ---")
        demo_article = {
            "title": "New Zealand replaces GDP with wellbeing metrics in historic budget reform",
            "content": (
                "New Zealand became the first country to require all government spending to be "
                "evaluated against wellbeing metrics rather than GDP growth. Finance Minister "
                "Grant Robertson acknowledged that 'GDP never captured what matters' and committed "
                "to measuring success across 12 domains including mental health, child poverty, "
                "and environmental sustainability over a 30-year horizon."
            ),
        }

        print(f"\nDemo article: {demo_article['title']}")
        result = scorer.score_article(demo_article)

        print(f"\nResults:")
        print(f"  Stage used: {result.get('stage_used')}")
        print(f"  Stage 1 estimate: {result.get('stage1_estimate', 'N/A')}")
        if result["scores"]:
            print(f"  Scores:")
            for dim, score in result["scores"].items():
                print(f"    {dim}: {score:.2f}")
            print(f"  Weighted average: {result['weighted_average']:.2f}")
            print(f"  Tier: {result['tier']}")


if __name__ == "__main__":
    main()
