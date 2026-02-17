"""
Investment Risk Filter v5 - Hybrid Inference Pipeline

Two-stage scorer that uses embedding + MLP probe for fast screening (Stage 1)
and the trained Qwen2.5-1.5B model for precise scoring (Stage 2).

Stage 1 (~1.3ms): Embedding probe estimates scores. Articles with weighted_avg < threshold
are classified as NOISE/low-signal without running the expensive model.

Stage 2 (~25ms): Full fine-tuned model scoring for articles that pass Stage 1.

Usage:
    from filters.investment_risk.v5.inference_hybrid import InvestmentRiskHybridScorer

    scorer = InvestmentRiskHybridScorer()
    result = scorer.score_article(article)
    # result["stage_used"] -> "stage1_low" or "stage2"

    # CLI
    python filters/investment-risk/v5/inference_hybrid.py --input articles.jsonl --output results.jsonl
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

from filters.common.hybrid_scorer import HybridScorer

logger = logging.getLogger(__name__)

# Default Stage 1 threshold: articles below this skip Stage 2
# Investment-risk uses GREEN min_score=4.0 as lowest actionable tier
# To be calibrated after probe training (Phase B)
DEFAULT_THRESHOLD = 4.0


class InvestmentRiskHybridScorer(HybridScorer):
    """
    Two-stage hybrid scorer for investment-risk filter v5.

    Combines:
    - Stage 1: multilingual-e5-small embeddings + MLP probe
    - Stage 2: Existing InvestmentRiskScorer (Qwen2.5-1.5B fine-tuned)
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

        # Deferred import to avoid circular imports with hyphenated package name
        from importlib import import_module
        self._scorer_module = import_module("filters.investment-risk.v5.inference")

        super().__init__(device=device, use_prefilter=use_prefilter)

    def _create_stage2_scorer(self):
        """Create the existing InvestmentRiskScorer as Stage 2."""
        InvestmentRiskScorer = self._scorer_module.InvestmentRiskScorer
        return InvestmentRiskScorer(
            model_path=self._model_path,
            device=self.device_str,
            use_prefilter=self.use_prefilter,
        )

    def _get_embedding_stage_config(self) -> Dict:
        """Return EmbeddingStage configuration for investment-risk v5."""
        return {
            "embedding_model_name": "intfloat/multilingual-e5-small",
            "probe_path": str(self._probe_path),
            "threshold": self._threshold,
        }


def main():
    """CLI interface for hybrid batch scoring."""
    import argparse
    from importlib import import_module

    parser = argparse.ArgumentParser(
        description="Score articles with investment-risk hybrid scorer (two-stage pipeline)"
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
    parser.add_argument(
        "--compare", action="store_true",
        help="Also run standard scorer and compare results"
    )

    args = parser.parse_args()

    # Initialize hybrid scorer
    print("Initializing hybrid scorer...")
    scorer = InvestmentRiskHybridScorer(
        threshold=args.threshold,
        use_prefilter=not args.no_prefilter,
    )

    if args.input:
        # Score from file
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

        # Stats
        stage1_low = sum(1 for r in results if r.get("stage_used") == "stage1_low")
        stage2 = sum(1 for r in results if r.get("stage_used") == "stage2")
        prefilter_blocked = sum(1 for r in results if not r.get("passed_prefilter", True))

        print(f"\nHybrid results ({hybrid_time:.2f}s):")
        print(f"  Prefilter blocked: {prefilter_blocked}")
        print(f"  Stage 1 LOW (skipped model): {stage1_low}")
        print(f"  Stage 2 (full model): {stage2}")
        print(f"  Avg time per article: {hybrid_time/len(articles)*1000:.1f}ms")

        # Tier distribution
        tiers = {}
        for r in results:
            tier = r.get("tier")
            if tier:
                tiers[tier] = tiers.get(tier, 0) + 1
        print(f"\nTier distribution:")
        for tier, count in sorted(tiers.items()):
            print(f"  {tier}: {count}")

        # Optional comparison with standard scorer
        if args.compare:
            print(f"\nRunning standard scorer for comparison...")
            ir_module = import_module("filters.investment-risk.v5.inference")
            standard_scorer = ir_module.InvestmentRiskScorer(
                use_prefilter=not args.no_prefilter,
            )
            start = time.time()
            standard_results = standard_scorer.score_batch(
                articles, batch_size=args.batch_size
            )
            standard_time = time.time() - start

            print(f"Standard scorer: {standard_time:.2f}s ({standard_time/len(articles)*1000:.1f}ms/article)")
            print(f"Speedup: {standard_time/hybrid_time:.2f}x")

            # Check agreement — investment-risk uses RED/YELLOW/GREEN as actionable tiers
            actionable_tiers = ("RED", "YELLOW", "GREEN")
            disagreements = 0
            for i, (h, s) in enumerate(zip(results, standard_results)):
                if s.get("tier") in actionable_tiers and h.get("stage_used") == "stage1_low":
                    disagreements += 1
                    print(
                        f"  FALSE NEGATIVE #{disagreements}: article {i} "
                        f"(standard={s.get('tier')}, stage1_est={h.get('stage1_estimate', 0):.2f})"
                    )

            print(f"\nFalse negatives (actionable classified as LOW by Stage 1): {disagreements}")

        # Output
        if args.output:
            print(f"\nWriting results to {args.output}")
            with open(args.output, "w", encoding="utf-8") as f:
                for article, result in zip(articles, results):
                    article_id = article.get("id") or article.get("article_id", "")
                    output = {"article_id": article_id, **result}
                    f.write(json.dumps(output) + "\n")
    else:
        # Interactive demo
        print("\n--- Hybrid Scorer Demo ---")
        demo_article = {
            "title": "Federal Reserve Signals Emergency Rate Discussion Amid Banking Stress",
            "content": (
                "The Federal Reserve has called an unscheduled board meeting to discuss "
                "potential emergency measures as regional bank stress indicators hit levels "
                "not seen since March 2023. Three mid-size banks reported deposit outflows "
                "exceeding 5% in the past week, while credit default swap spreads on "
                "bank debt have widened significantly."
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
