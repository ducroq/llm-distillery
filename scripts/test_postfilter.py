"""
Test script for post-filter tier classification.

Tests both sustainability_tech_deployment and uplifting filters
with various dimensional score scenarios.
"""

from postfilter import PostFilter
import json


def test_sustainability_tech_deployment():
    """Test post-filter with sustainability_tech_deployment filter."""
    print("="*60)
    print("Testing: sustainability_tech_deployment")
    print("="*60)

    pf = PostFilter("filters/sustainability_tech_deployment/v1")

    # Test case 1: High scores → mass_deployment tier
    print("\nTest 1: High scores (should be mass_deployment)")
    scores = {
        "deployment_maturity": 9.0,
        "technology_performance": 8.5,
        "cost_trajectory": 8.0,
        "scale_of_deployment": 9.5,
        "market_penetration": 8.5,
        "technology_readiness": 9.0,
        "supply_chain_maturity": 8.0,
        "proof_of_impact": 8.5
    }
    result = pf.classify(scores, flag_reasoning_threshold=7.0)
    print(json.dumps(result, indent=2))
    assert result["tier"] == "mass_deployment", f"Expected mass_deployment, got {result['tier']}"
    assert result["needs_reasoning"] == True, "Should flag for reasoning"

    # Test case 2: Medium scores → commercial_proven
    print("\nTest 2: Medium scores (should be commercial_proven)")
    scores = {
        "deployment_maturity": 7.0,
        "technology_performance": 6.5,
        "cost_trajectory": 7.0,
        "scale_of_deployment": 6.5,
        "market_penetration": 7.0,
        "technology_readiness": 6.5,
        "supply_chain_maturity": 6.0,
        "proof_of_impact": 6.5
    }
    result = pf.classify(scores, flag_reasoning_threshold=7.0)
    print(json.dumps(result, indent=2))
    assert result["tier"] == "commercial_proven", f"Expected commercial_proven, got {result['tier']}"

    # Test case 3: Gatekeeper rule triggered (deployment_maturity < 5.0)
    print("\nTest 3: Gatekeeper rule (deployment_maturity < 5.0)")
    scores = {
        "deployment_maturity": 4.0,  # Below 5.0 threshold
        "technology_performance": 8.0,
        "cost_trajectory": 8.0,
        "scale_of_deployment": 7.0,
        "market_penetration": 7.5,
        "technology_readiness": 7.0,
        "supply_chain_maturity": 6.5,
        "proof_of_impact": 7.0
    }
    result = pf.classify(scores, flag_reasoning_threshold=7.0)
    print(json.dumps(result, indent=2))
    assert result["overall_score"] <= 4.9, "Should be capped at 4.9"
    assert "Gatekeeper" in result["applied_rules"][0], "Should mention gatekeeper rule"

    # Test case 4: Gatekeeper rule triggered (proof_of_impact < 4.0)
    print("\nTest 4: Gatekeeper rule (proof_of_impact < 4.0)")
    scores = {
        "deployment_maturity": 7.0,
        "technology_performance": 7.0,
        "cost_trajectory": 7.0,
        "scale_of_deployment": 7.0,
        "market_penetration": 7.0,
        "technology_readiness": 7.0,
        "supply_chain_maturity": 7.0,
        "proof_of_impact": 3.5  # Below 4.0 threshold
    }
    result = pf.classify(scores, flag_reasoning_threshold=7.0)
    print(json.dumps(result, indent=2))
    assert result["overall_score"] <= 3.9, "Should be capped at 3.9"

    print("\n[PASS] All sustainability_tech_deployment tests passed!")


def test_uplifting():
    """Test post-filter with uplifting filter."""
    print("\n" + "="*60)
    print("Testing: uplifting")
    print("="*60)

    pf = PostFilter("filters/uplifting/v1")

    # Test case 1: High scores -> impact tier
    print("\nTest 1: High scores (should be impact)")
    scores = {
        "agency": 8.0,
        "progress": 8.5,
        "collective_benefit": 8.0,
        "connection": 7.5,
        "innovation": 7.5,
        "justice": 7.0,
        "resilience": 7.0,
        "wonder": 7.5
    }
    result = pf.classify(scores, flag_reasoning_threshold=7.0)
    print(json.dumps(result, indent=2))
    assert result["tier"] == "impact", f"Expected impact, got {result['tier']}"
    assert result["needs_reasoning"] == True, "Should flag for reasoning"

    # Test case 2: Medium scores -> connection
    print("\nTest 2: Medium scores (should be connection)")
    scores = {
        "agency": 5.0,
        "progress": 5.5,
        "collective_benefit": 5.0,
        "connection": 4.5,
        "innovation": 4.5,
        "justice": 4.0,
        "resilience": 4.5,
        "wonder": 5.0
    }
    result = pf.classify(scores, flag_reasoning_threshold=7.0)
    print(json.dumps(result, indent=2))
    assert result["tier"] == "connection", f"Expected connection, got {result['tier']}"
    assert result["needs_reasoning"] == False, "Should not flag for reasoning"

    # Test case 3: Low scores -> not_uplifting
    print("\nTest 3: Low scores (should be not_uplifting)")
    scores = {
        "agency": 2.0,
        "progress": 2.5,
        "collective_benefit": 2.0,
        "connection": 1.5,
        "innovation": 1.5,
        "justice": 1.0,
        "resilience": 1.5,
        "wonder": 2.0
    }
    result = pf.classify(scores, flag_reasoning_threshold=7.0)
    print(json.dumps(result, indent=2))
    assert result["tier"] == "not_uplifting", f"Expected not_uplifting, got {result['tier']}"

    print("\n[PASS] All uplifting tests passed!")


def test_weighted_average():
    """Test that weighted average calculation is correct."""
    print("\n" + "="*60)
    print("Testing: Weighted average calculation")
    print("="*60)

    pf = PostFilter("filters/sustainability_tech_deployment/v1")

    # All dimensions = 5.0 -> overall should be 5.0
    scores = {dim: 5.0 for dim in pf.dimension_names}
    overall = pf.calculate_overall_score(scores)
    print(f"\nAll dimensions = 5.0 -> Overall = {overall}")
    assert abs(overall - 5.0) < 0.01, f"Expected 5.0, got {overall}"

    # Test weighted calculation
    # deployment_maturity (weight 0.20) = 10.0, rest = 0.0 -> overall should be 2.0
    scores = {dim: 0.0 for dim in pf.dimension_names}
    scores["deployment_maturity"] = 10.0
    overall = pf.calculate_overall_score(scores)
    print(f"deployment_maturity=10, rest=0 -> Overall = {overall}")
    assert abs(overall - 2.0) < 0.01, f"Expected 2.0 (10 * 0.20), got {overall}"

    print("\n[PASS] Weighted average tests passed!")


if __name__ == "__main__":
    test_weighted_average()
    test_sustainability_tech_deployment()
    test_uplifting()
    print("\n" + "="*60)
    print("[PASS] ALL TESTS PASSED!")
    print("="*60)
