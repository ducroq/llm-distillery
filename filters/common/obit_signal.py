"""
Loose obituary signal — regex-only.

Provides a measurement-grade probe for obituary / memorial / funeral
content. Originally lived as belonging v1's `obituary_funeral` exclusion
category (#45 hold-the-line); hoisted here so any consumer can read the
signal from a single source of truth instead of copy-pasting patterns.

Two consumers as of 2026-05-05:
  - belonging v1 prefilter — uses `OBIT_PATTERNS` as one of its
    `EXCLUSION_PATTERNS` categories, plus the case-sensitive RIP
    title check via `uppercase_rip_in_title`.
  - NexusMind#199 — production logging of `_obit_pattern_count` per
    article for cross-lens leak measurement (gates llm-distillery#51).

This module is INTENTIONALLY a regex probe, not a precision detector.
False-positive rate is acceptable for measurement; recall doesn't need
to be perfect because the consumer questions are order-of-magnitude
("does cultural_discovery have an obit problem at all?"), not per-
article precision.

If llm-distillery#51 escalates and a trained universal detector ships,
that detector replaces this probe in NexusMind's production path. This
module stays — belonging v1 still uses it as one component of its
prefilter logic, where the trained detector's binary block isn't the
right shape.
"""

import re
from typing import Dict, List, Pattern

# Body patterns. Compiled with re.IGNORECASE.
#
# Cross-references:
#   `dies aged|died aged|dies at \d+|died at \d+` — \d+ (not \d) needed
#   for two-digit ages like "Dies at 99" (#45).
#   `(dies|died) (after|following|in|while)` — verb-form variants not
#   anchored by aged/at-N (#45 cases: "Dies After Decades of Protest",
#   "Dies Following Long Illness", "Dies in Crash", "Dies While Hiking").
#   `procession|candlelight vigil|memorial vigil` — strong death-context
#   signal, low FP risk (#45).
#   `rest in peace` — included; standalone uppercase RIP handled below.
#   `(killed|murdered|assassinated) in \d{4}` — historical-tragedy
#   commemoration framing ("Family Killed in 1976 Bombing Remembered").
#   Anchored to 4-digit year so we don't over-match current conflict
#   reporting (#45 item 8).
OBIT_PATTERNS: List[str] = [
    r'\b(obituary|obituaries|in memoriam)\b',
    r'\b(funeral|funeral mass|funeral service|memorial service)\b',
    r'\b(passed away|laid to rest|death notice)\b',
    r'\b(dies aged|died aged|dies at \d+|died at \d+)\b',
    r'\b(dies|died) (after|following|in|while)\b',
    r'\b(survived by|in loving memory|paying tribute|pays tribute)\b',
    r'\b(mourners?|mourning|condolences)\b',
    r'\b(procession|candlelight vigil|memorial vigil)\b',
    r'\b(rest in peace)\b',
    r'\b(killed|murdered|assassinated) in \d{4}\b',
]

_COMPILED_OBIT: List[Pattern] = [re.compile(p, re.IGNORECASE) for p in OBIT_PATTERNS]

# Case-sensitive `\bRIP\b` on the raw title only.
#
# Why this lives outside OBIT_PATTERNS:
# Standard prefilter pipelines lowercase input before pattern matching
# (so re.IGNORECASE on a list of patterns can pick up obituary / Obituary
# / OBITUARY equally). For RIP specifically we need the OPPOSITE — match
# only the uppercase token, not "rip current" / "rip the page". An inline
# `(?-i:RIP)` flag in the pattern can't help because the input string was
# already lowercased before the regex engine saw it. The fix is to skip
# the lowercasing for this one signal: read the raw title and search
# case-sensitively. Title-only (not body) keeps FP risk minimal — body
# text occasionally goes all-caps for emphasis, but obit titles use
# "RIP" as a recognised acronym deliberately.
#
# See `memory/feedback-regex-ignorecase-trap.md` for the original
# 2026-04-28 production gotcha.
_UPPERCASE_RIP_RE = re.compile(r'\bRIP\b')  # NO re.IGNORECASE — see above


def loose_obit_signal(article: Dict) -> int:
    """Count obituary-flavored regex matches in the article.

    Returns 0 (no signal) or a positive integer (obit-flavored, with the
    integer being the total match count across body patterns plus 1 if
    the raw title contains the uppercase RIP token).

    Use for cross-lens leak-rate logging (NexusMind#199). Do NOT use as
    a production block on its own — for blocking, consume via a
    per-filter prefilter that wraps these patterns with appropriate
    exception/positive-signal logic (e.g. belonging v1's per-category
    threshold rules).

    Args:
      article: dict with 'title' and 'content'/'text' keys.

    Returns:
      int. 0 if no signal, otherwise the match count.
    """
    title_raw = article.get('title') or ''
    content = article.get('content') or article.get('text') or ''
    combined = f"{title_raw} {content[:2000]}".lower()

    count = sum(len(p.findall(combined)) for p in _COMPILED_OBIT)
    if _UPPERCASE_RIP_RE.search(title_raw):
        count += 1
    return count


def has_obit_signal(article: Dict) -> bool:
    """Boolean shortcut: True if any obit pattern matches."""
    return loose_obit_signal(article) >= 1


def uppercase_rip_in_title(title: str) -> bool:
    """Case-sensitive `\\bRIP\\b` check on the raw (un-lowercased) title.

    Exposed as a public helper so consumers can call this on the raw
    title alongside their own pipeline rather than relying on the
    lowercased combined text path. See module docstring for the
    case-sensitivity rationale.
    """
    return bool(_UPPERCASE_RIP_RE.search(title or ''))
