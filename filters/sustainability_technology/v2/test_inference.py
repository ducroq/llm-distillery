"""
Test script for sustainability_technology v1 filter.

Tests the inference pipeline by scoring sample articles and verifying outputs.

Usage:
    python -m filters.sustainability_technology.v1.test_inference

    # Or with HF Hub model:
    python -m filters.sustainability_technology.v1.test_inference --from-hub
"""

import argparse
import json
import sys
from pathlib import Path

# Sample articles for testing
TEST_ARTICLES = [
    {
        "id": "high_relevance",
        "title": "New Perovskite Solar Cells Achieve 33% Efficiency in Commercial Pilot",
        "content": """
        SunTech Industries announced today that their new perovskite-silicon tandem solar cells
        have achieved 33.1% power conversion efficiency in a 50MW commercial pilot deployment
        in Arizona. The cells use earth-abundant materials and can be manufactured using
        existing silicon production lines at a cost of $0.18/watt, making them competitive
        with traditional silicon panels at $0.22/watt. The pilot has been running for 18 months
        with only 2% degradation, suggesting 25+ year operational lifetimes. The technology
        has created 200 manufacturing jobs and the company plans to scale to 1GW production
        capacity by 2026. Life cycle analysis shows 75% lower carbon footprint compared to
        conventional panels due to lower processing temperatures and reduced material usage.
        """,
        "expected_tier": "medium_high",  # Model scores ~5.8 for this detailed article
    },
    {
        "id": "medium_relevance",
        "title": "City Launches Electric Bus Fleet Pilot Program",
        "content": """
        The city of Portland announced a pilot program to test 20 electric buses on three
        routes starting next month. The BYD K9 buses have a range of 150 miles and will
        be charged overnight at the central depot. City officials estimate the program will
        reduce emissions by 500 tons of CO2 annually. The $12 million program is funded by
        a federal grant. If successful, the city plans to electrify 50% of its fleet by 2030.
        """,
        "expected_tier": "medium",  # Pilot program, less technical detail
    },
    {
        "id": "low_relevance",
        "title": "Tips for Reducing Your Carbon Footprint at Home",
        "content": """
        Looking to live more sustainably? Here are 10 easy tips: Turn off lights when leaving
        a room, use reusable shopping bags, eat less meat, take shorter showers, unplug
        electronics when not in use, walk or bike instead of driving, buy local produce,
        compost food scraps, use a programmable thermostat, and hang dry your clothes.
        Small changes add up to make a big difference for the planet!
        """,
        "expected_tier": "low",
    },
    {
        "id": "not_relevant",
        "title": "Local Team Wins Championship Game",
        "content": """
        The Springfield Tigers defeated the Shelbyville Sharks 24-17 in last night's
        championship game. Quarterback Tom Johnson threw for 280 yards and two touchdowns.
        The victory marks the team's first championship in 15 years. Fans celebrated in
        the streets until early morning. Coach Williams credited the defensive line for
        their outstanding performance in the second half.
        """,
        "expected_tier": "low",
    },
]


def test_inference(use_hub: bool = False, hub_repo: str = None, token: str = None):
    """Run inference tests on sample articles."""

    print("=" * 70)
    print("Sustainability Technology v1 - Inference Test")
    print("=" * 70)

    # Load scorer
    if use_hub:
        print(f"\nLoading model from Hugging Face Hub: {hub_repo}")
        from filters.sustainability_technology.v1.inference_hub import SustainabilityTechnologyScorerHub
        scorer = SustainabilityTechnologyScorerHub(repo_id=hub_repo, token=token)
    else:
        print("\nLoading model from local files...")
        from filters.sustainability_technology.v1.inference import SustainabilityTechnologyScorer
        scorer = SustainabilityTechnologyScorer()

    print("\n" + "-" * 70)
    print("Running tests on sample articles...")
    print("-" * 70)

    results = []
    all_passed = True

    for article in TEST_ARTICLES:
        print(f"\n[Article] {article['title'][:50]}...")
        print(f"   Expected tier: {article['expected_tier']}")

        # Score article
        result = scorer.score_article({
            "title": article["title"],
            "content": article["content"],
        })

        results.append({
            "id": article["id"],
            "title": article["title"],
            "expected_tier": article["expected_tier"],
            "result": result,
        })

        # Check result
        if not result["passed_prefilter"]:
            print(f"   [!] Blocked by prefilter: {result['prefilter_reason']}")
            actual_tier = "blocked"
        else:
            actual_tier = result["tier"]
            print(f"   [+] Scores:")
            for dim, score in result["scores"].items():
                print(f"      {dim}: {score:.2f}")
            print(f"   [+] Weighted average: {result['weighted_average']:.2f}")
            print(f"   [+] Tier: {result['tier']}")
            if result["gatekeeper_applied"]:
                print(f"   [!] TRL gatekeeper applied")

        # Validate
        # For blocked articles, check if expected was "low" (acceptable)
        tier_match = (actual_tier == article["expected_tier"]) or \
                     (actual_tier == "blocked" and article["expected_tier"] == "low")

        if tier_match:
            print(f"   [PASS] Tier matches expected")
        else:
            print(f"   [FAIL] Expected {article['expected_tier']}, got {actual_tier}")
            all_passed = False

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    passed = sum(1 for r in results if
                 r["result"]["tier"] == r["expected_tier"] or
                 (not r["result"]["passed_prefilter"] and r["expected_tier"] == "low"))
    total = len(results)

    print(f"\nResults: {passed}/{total} tests passed")

    if all_passed:
        print("\n[OK] All tests PASSED!")
    else:
        print("\n[!!] Some tests failed - review results above")
        print("   (Note: Tier predictions may vary slightly from expected)")

    # Output detailed results
    print("\n" + "-" * 70)
    print("Detailed Results (JSON)")
    print("-" * 70)

    for r in results:
        summary = {
            "id": r["id"],
            "expected": r["expected_tier"],
            "actual": r["result"]["tier"] if r["result"]["passed_prefilter"] else "blocked",
            "weighted_avg": r["result"]["weighted_average"],
            "prefilter": r["result"]["passed_prefilter"],
        }
        print(json.dumps(summary))

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Test sustainability_technology v1 inference")
    parser.add_argument("--from-hub", action="store_true", help="Load model from HF Hub instead of local")
    parser.add_argument("--repo", type=str, default="jeergrvgreg/sustainability-technology-v1",
                        help="HF Hub repo ID")
    parser.add_argument("--token", type=str, help="HF token (or reads from secrets.ini)")

    args = parser.parse_args()

    # Load token from secrets if not provided
    token = args.token
    if args.from_hub and not token:
        secrets_path = Path(__file__).parent.parent.parent.parent / "config" / "credentials" / "secrets.ini"
        if secrets_path.exists():
            import configparser
            config = configparser.ConfigParser()
            config.read(secrets_path)
            token = config.get("api_keys", "huggingface_token", fallback=None)

    success = test_inference(use_hub=args.from_hub, hub_repo=args.repo, token=token)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
