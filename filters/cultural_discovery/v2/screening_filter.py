"""
Screening Filter: Cultural Discovery v2

Purpose: Enrich training data distribution before oracle scoring.
NOT for inference - use prefilter.py for that.

This filter is MORE AGGRESSIVE than the prefilter because:
- Prefilter: Conservative, goal is avoiding false negatives at inference
- Screening: Aggressive, goal is enriching signal in training data

Target: ~20% of screened articles should pass (vs ~15% for prefilter)
Of those, ~30-40% should score >= 4.0 (vs ~6% in random sample)
"""

import re
import json
import argparse
from typing import Dict, Any, Tuple, List


# =============================================================================
# CONFIGURATION
# =============================================================================

# SIGNAL PATTERNS - Articles must match at least 2 of these to pass
# These are indicators of cultural discovery content
SIGNAL_PATTERNS = [
    # Archaeological/Discovery language
    (r'\b(archaeolog|excavat|unearth|discover|found|reveal)\w*\b', re.IGNORECASE, "Discovery language"),
    (r'\b(artifact|artefact|relic|ancient|prehistoric|centuries-old)\b', re.IGNORECASE, "Artifact mentions"),
    (r'\b(tomb|temple|ruins|remains|site|monument)\b', re.IGNORECASE, "Archaeological sites"),

    # Heritage/Cultural significance
    (r'\b(unesco|world heritage|patrimonio|welterbe)\b', re.IGNORECASE, "UNESCO/heritage"),
    (r'\b(heritage|cultural heritage|patrimoine|erfgoed)\b', re.IGNORECASE, "Heritage language"),
    (r'\b(preserv|conserv|restor|protect)\w*\b', re.IGNORECASE, "Preservation language"),
    (r'\b(museum|gallery|archive|collection)\b', re.IGNORECASE, "Cultural institutions"),

    # Cross-cultural language
    (r'\b(cross-cultural|intercultural|multicultural|transcultural)\b', re.IGNORECASE, "Cross-cultural explicit"),
    (r'\b(cultural (exchange|bridge|dialogue|connection))\b', re.IGNORECASE, "Cultural connection"),
    (r'\b(civilizat|tradition|indigenous|ancestral)\b', re.IGNORECASE, "Civilization/tradition"),
    (r'\b(shared (heritage|history|ancestry|roots))\b', re.IGNORECASE, "Shared heritage"),

    # Academic/Research indicators
    (r'\b(researcher|scholar|historian|archaeologist|anthropologist)\b', re.IGNORECASE, "Academic roles"),
    (r'\b(study (shows|reveals|finds|suggests))\b', re.IGNORECASE, "Research findings"),
    (r'\b(university|institute|foundation|academy)\b', re.IGNORECASE, "Academic institutions"),
    (r'\b(peer-reviewed|published in|journal)\b', re.IGNORECASE, "Academic publication"),

    # Art/Culture specific
    (r'\b(art\s*(history|restoration|movement)|masterpiece|artwork)\b', re.IGNORECASE, "Art specific"),
    (r'\b(repatria|return\w* (to|artifact|relic|treasure))\b', re.IGNORECASE, "Repatriation"),
    (r'\b(looted|stolen|illicit|provenance)\b', re.IGNORECASE, "Provenance/looting"),
]

# BOOST PATTERNS - Increase confidence
BOOST_PATTERNS = [
    (r'\b(breakthrough|groundbreaking|unprecedented|first-ever)\b', re.IGNORECASE, "Impact language"),
    (r'\b(reveal|unveil|shed light|new evidence)\b', re.IGNORECASE, "Revelation language"),
    (r'\b(ancient|medieval|prehistoric|3000|2000|1000)\s*(year|century)\b', re.IGNORECASE, "Historical depth"),
    (r'\b(DNA|carbon dating|radiocarbon|isotope)\b', re.IGNORECASE, "Scientific methods"),
    (r'\b(maya|aztec|inca|egyptian|roman|greek|viking|celtic)\b', re.IGNORECASE, "Major civilizations"),
]

# PENALTY PATTERNS - Reduce confidence (but don't auto-reject)
PENALTY_PATTERNS = [
    (r'\b(top \d+|best \d+|\d+ must-see|bucket list)\b', re.IGNORECASE, "Listicle patterns"),
    (r'\b(tourism|tourist|travel|visit|trip|vacation)\b', re.IGNORECASE, "Tourism language"),
    (r'\b(instagram|selfie|photo op|breathtaking view)\b', re.IGNORECASE, "Tourism marketing"),
    (r'\b(auction|sold for|fetched|hammer price|million dollar)\b', re.IGNORECASE, "Art market"),
    (r'\b(celebrity|billionaire|collector|investment)\b', re.IGNORECASE, "Celebrity/wealth"),
    (r'\b(controversy|outrage|backlash|slammed|blasted)\b', re.IGNORECASE, "Conflict framing"),
    (r'\b(appropriation|cancel culture|woke|identity politics)\b', re.IGNORECASE, "Culture war"),
]

# BLOCK PATTERNS - Auto-reject (obvious non-cultural content)
BLOCK_PATTERNS = [
    (r'\b(stock|shares|earnings|quarterly|ipo|nasdaq|nyse)\b', re.IGNORECASE, "Financial news"),
    (r'\b(sports|football|basketball|soccer|nfl|nba|match|game score)\b', re.IGNORECASE, "Sports"),
    (r'\b(weather|forecast|temperature|rain|snow|climate alert)\b', re.IGNORECASE, "Weather"),
    (r'\b(recipe|cooking|chef|restaurant review|food critic)\b', re.IGNORECASE, "Food/cooking"),
    (r'\b(crypto|bitcoin|ethereum|blockchain|nft)\b', re.IGNORECASE, "Cryptocurrency"),
    (r'\b(election|campaign|candidate|poll|vote|ballot)\b', re.IGNORECASE, "Political campaigns"),
]

# Minimum requirements
MIN_WORD_COUNT = 200
MAX_WORD_COUNT = 15000
MIN_TITLE_LENGTH = 10
SIGNAL_THRESHOLD = 2  # Must match at least 2 signal patterns


# =============================================================================
# SCREENING LOGIC
# =============================================================================

def screen_article(article: Dict[str, Any]) -> Tuple[bool, str, float]:
    """
    Screen an article for training data enrichment.

    Args:
        article: Dict with 'title', 'content'/'text', 'source' (optional)

    Returns:
        Tuple of (passes: bool, reason: str, confidence: float)
        - passes: Whether article passes screening
        - reason: Human-readable explanation
        - confidence: 0.0-1.0 confidence that article has signal
    """
    title = article.get('title', '') or ''
    content = article.get('content', article.get('text', '')) or ''
    source = article.get('source', article.get('url', '')) or ''

    full_text = f"{title} {content}"
    word_count = len(full_text.split())

    # Basic filters
    if word_count < MIN_WORD_COUNT:
        return False, f"Too short ({word_count} words < {MIN_WORD_COUNT})", 0.0

    if word_count > MAX_WORD_COUNT:
        return False, f"Too long ({word_count} words > {MAX_WORD_COUNT})", 0.0

    if len(title) < MIN_TITLE_LENGTH:
        return False, f"Title too short ({len(title)} chars)", 0.0

    # Check block patterns first (auto-reject)
    for pattern, flags, description in BLOCK_PATTERNS:
        if re.search(pattern, full_text, flags):
            return False, f"Blocked: {description}", 0.0

    # Count signal patterns
    signal_count = 0
    matched_signals = []

    for pattern, flags, description in SIGNAL_PATTERNS:
        if re.search(pattern, full_text, flags):
            signal_count += 1
            matched_signals.append(description)

    # Check minimum signal threshold
    if signal_count < SIGNAL_THRESHOLD:
        return False, f"Insufficient signal ({signal_count}/{SIGNAL_THRESHOLD}): {matched_signals}", 0.1

    # Calculate confidence
    confidence = 0.4 + (signal_count * 0.05)  # Base from signals

    # Apply boosts
    boost_count = 0
    for pattern, flags, description in BOOST_PATTERNS:
        if re.search(pattern, full_text, flags):
            confidence += 0.08
            boost_count += 1
            matched_signals.append(f"+{description}")

    # Apply penalties (but don't auto-reject)
    penalty_count = 0
    for pattern, flags, description in PENALTY_PATTERNS:
        if re.search(pattern, full_text, flags):
            confidence -= 0.10
            penalty_count += 1
            matched_signals.append(f"-{description}")

    # Clamp confidence
    confidence = max(0.1, min(1.0, confidence))

    # Pass if sufficient signal and confidence
    if confidence >= 0.35:
        reason = f"Pass (signal={signal_count}, boost={boost_count}, penalty={penalty_count}): {', '.join(matched_signals[:5])}"
        return True, reason, confidence
    else:
        reason = f"Low confidence ({confidence:.2f}): {', '.join(matched_signals[:5])}"
        return False, reason, confidence


def screen_batch(articles: list, target_count: int = None, verbose: bool = False) -> list:
    """
    Screen a batch of articles, optionally limiting to target count.

    Args:
        articles: List of article dicts
        target_count: Optional max number to return (highest confidence first)
        verbose: Print progress

    Returns:
        List of (article, reason, confidence) tuples that pass screening
    """
    results = []
    rejected = 0

    for i, article in enumerate(articles):
        if verbose and i % 1000 == 0:
            print(f"  Screening {i}/{len(articles)}... ({len(results)} passed, {rejected} rejected)")

        passes, reason, confidence = screen_article(article)
        if passes:
            results.append((article, reason, confidence))
        else:
            rejected += 1

    # Sort by confidence (highest first)
    results.sort(key=lambda x: x[2], reverse=True)

    if target_count and len(results) > target_count:
        results = results[:target_count]

    if verbose:
        print(f"  Screening complete: {len(results)} passed, {rejected} rejected")
        print(f"  Pass rate: {100 * len(results) / len(articles):.1f}%")

    return results


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Screen articles for cultural discovery training data")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--target", type=int, help="Target number of articles to output")
    parser.add_argument("--max-input", type=int, help="Maximum articles to read from input")
    parser.add_argument("--stats", help="Output stats file (JSON)")
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    args = parser.parse_args()

    # Load articles
    print(f"Loading articles from {args.input}...")
    articles = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if args.max_input and i >= args.max_input:
                break
            try:
                articles.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(articles)} articles")

    # Screen
    print("Screening articles...")
    passed = screen_batch(articles, args.target, verbose=args.verbose)

    # Write output
    print(f"Writing {len(passed)} screened articles to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        for article, reason, confidence in passed:
            article['_screening_reason'] = reason
            article['_screening_confidence'] = confidence
            f.write(json.dumps(article, ensure_ascii=False) + '\n')

    # Summary
    pass_rate = 100 * len(passed) / len(articles) if articles else 0
    avg_confidence = sum(c for _, _, c in passed) / len(passed) if passed else 0

    print(f"\nScreening Summary:")
    print(f"  Input articles: {len(articles)}")
    print(f"  Passed screening: {len(passed)}")
    print(f"  Pass rate: {pass_rate:.1f}%")
    print(f"  Average confidence: {avg_confidence:.2f}")

    # Stats file
    if args.stats:
        stats = {
            "total_input": len(articles),
            "total_passed": len(passed),
            "pass_rate": pass_rate / 100,
            "avg_confidence": avg_confidence,
            "target_count": args.target,
            "config": {
                "signal_threshold": SIGNAL_THRESHOLD,
                "min_word_count": MIN_WORD_COUNT,
                "signal_patterns": len(SIGNAL_PATTERNS),
                "boost_patterns": len(BOOST_PATTERNS),
                "penalty_patterns": len(PENALTY_PATTERNS),
                "block_patterns": len(BLOCK_PATTERNS),
            }
        }
        with open(args.stats, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Stats written to {args.stats}")


if __name__ == '__main__':
    main()
