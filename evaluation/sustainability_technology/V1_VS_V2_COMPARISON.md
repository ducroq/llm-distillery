# Sustainability Technology Filter: v1 vs v2 Comparison

**Generated:** 2026-01-16 11:49

## Executive Summary

Testing both filter versions on **271 known false positives** (articles that were incorrectly classified as "medium" tier but should be "low/off-topic").

| Metric | v1 | v2 | Change |
|--------|----|----|--------|
| **False Positive Rate** | 100.0% | 63.5% | **+36.5%** |
| Articles passing prefilter | 271 | 172 | -99 |
| Articles blocked | 0 | 99 | +99 |

## Detailed Breakdown

| Category | Count | % of Total |
|----------|-------|------------|
| Both pass (still FP) | 172 | 63.5% |
| Both block (already caught) | 0 | 0.0% |
| **v1 pass → v2 block (improvement)** | 99 | 36.5% |
| v1 block → v2 pass (regression) | 0 | 0.0% |

## Interpretation

**v2 is better:** Reduces false positives by 36.5 percentage points.

The v2 prefilter correctly blocks 99 articles that v1 incorrectly passed.

## Sample Improvements (v1 pass → v2 block)

Articles that v2 correctly rejects:

- **Samsung‚Äôs Galaxy Book 6 series launches at CES with Intel‚Äôs newest chips and**
  - v2 reason: excluded_consumer_electronics
- **TCL unveils its X11L SQD-Mini LED TVs at CES 2026**
  - v2 reason: excluded_consumer_electronics
- **Sketch to Adapt: Fine-Tunable Sketches for Efficient LLM Adaptation**
  - v2 reason: excluded_ai_ml_infrastructure
- **Narwal's first mattress vacuum heats, taps, UV-blasts and sucks up all the ick l**
  - v2 reason: excluded_consumer_electronics
- **Samsung HW-QS90H soundbar hands-on: Impressive bass performance without a subwoo**
  - v2 reason: excluded_consumer_electronics
- **Optimized Hybrid Feature Engineering for Resource-Efficient Arrhythmia Detection**
  - v2 reason: excluded_ai_ml_infrastructure
- **How I Translated 277 Strings in 5 Minutes (Real-World Case Study)**
  - v2 reason: excluded_ai_ml_infrastructure

## Remaining False Positives (both pass)

Articles that still slip through v2:

- **MSched: GPU Multitasking via Proactive Memory Scheduling**
- **Detection of Malaria Infection from parasite-free blood smears**
- **OBS-Diff: Accurate Pruning For Diffusion Models in One-Shot**
- **Adapting Natural Language Processing Models Across Jurisdictions: A pilot Study **
- **Scheduling for TWDM-EPON-Based Fronthaul Without a Dedicated Registration Wavele**
- **At CES, Belkin launches a new charging case for the Switch 2 with a screen for c**
- **GE Appliances' new Smart Refrigerator automates grocery shopping with a barcode **
- **The HP Omnibook Ultra 14 at CES 2026: Super sleek and surprisingly durable**
- **Xreal updates its entry-level personal cinema glasses at CES**
- **Can a UK Residential Proxy Reliably Replace VPNs for Geo Access Without Triggeri**
