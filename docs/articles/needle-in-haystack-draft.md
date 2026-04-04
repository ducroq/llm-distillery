# The Needle-in-Haystack Problem: Why Filtering for Constructive News Breaks Standard ML

*Draft for llm-distillery#30*

---

## The promise and the trap

"Just filter the news for what's actually working." It sounds like a weekend project. Grab an LLM, score some articles, train a classifier. Ship it.

We tried. It took us a year, seven filters, and a few humbling failures before we understood why this is fundamentally harder than it looks. Not because the technology is immature, but because the problem has a structural property that defeats standard approaches: **the thing you're looking for is rare, and the reason it's rare is the same bias you're trying to correct.**

ovr.news filters global news through seven constructive lenses — Thriving, Belonging, Recovery, Solutions, Discovery, Breakthroughs, and Foresight. Each lens counters a specific cognitive bias: negativity bias, atomization, eco-anxiety, learned helplessness, declinism, and short-termism. Together they surface evidence that the world is more functional than the news makes it appear.

The engineering challenge: constructive news is a needle in a haystack, and the haystack is *designed* to hide the needle.

## Why keywords and sentiment fail

The first instinct is keyword matching. Surely "community" finds belonging, "long-term" finds foresight, "recovery" finds nature recovery?

It doesn't. What we're looking for isn't a topic — it's a *judgment*.

Our belonging filter scores articles on six dimensions: intergenerational bonds, community fabric, reciprocal care, rootedness, purpose beyond self, and slow presence. A LinkedIn article about "building community at work" contains all the right keywords. It scores 1.3 out of 10. A story about a 94-year-old making pasta with her granddaughter using her mother's recipe scores 8.5. No keyword distinguishes them. The difference is *what kind of community* — commodified versus organic, optimized versus lived.

Sentiment analysis fails in the opposite direction. Constructive news isn't positive news. A country admitting its drug war failed and shifting to treatment — that reads as negative. Our foresight filter scores it 6.1 because the *decision-making process* shows evidence-based course correction, systems awareness, and institutional durability. Meanwhile, a cheerful wellness listicle about living longer scores 1.3 on belonging because it commodifies community as a longevity hack.

The judgment we need — "does this article demonstrate genuine foresight / belonging / recovery?" — requires understanding intent, process quality, and evidence, not just topic or tone.

## Dimensional scoring: breaking judgment into measurable sub-factors

Our approach: decompose each lens into 6 weighted dimensions that can be scored independently on a 0-10 scale.

For foresight (counters short-termism):

| Dimension | Weight | What it measures |
|-----------|--------|------------------|
| Time Horizon | 25% | How far ahead does the decision look? |
| Systems Awareness | 20% | Are trade-offs and second-order effects acknowledged? |
| Course Correction | 20% | Is there willingness to admit error and change? |
| Intergenerational Investment | 15% | Are future generations explicitly considered? |
| Institutional Durability | 10% | Will the decision survive a change of leadership? |
| Evidence Foundation | 10% | Is the decision grounded in evidence? |

An oracle (Gemini Flash) scores articles on these dimensions. Each score has explicit rubrics with calibration examples, critical filters, and anti-hallucination rules (evidence must be exact quotes from the article). The oracle outputs only scores — tier classification (high/medium/low) happens in post-processing, so thresholds can be adjusted without re-labeling.

This is expensive: ~$0.001 per article, 1-2 seconds per call. We can't run this on every article forever. But we can use it to *teach* a smaller model.

## Knowledge distillation: invest energy once, infer forever

Knowledge distillation is the core of the pipeline. The oracle (Gemini Flash, cloud GPU, ~0.5s/article) scores thousands of articles. A student model (Gemma-3-1B, local CPU, 20ms/article) learns to replicate those scores.

The student is a Gemma-3-1B language model with a LoRA adapter — 13 million trainable parameters on top of a 1 billion parameter base. It takes article text in, outputs 6 continuous scores. After training, it runs locally, on CPU, at 20ms per article. No cloud, no API, no per-article cost.

At ovr.news scale (2,000+ articles, 7 filters, multiple runs per day), this is the difference between needing a cloud GPU budget and running on a mini PC. The energy investment happens once during training. After that, inference is essentially free.

But this only works if the student has good training data to learn from. And that's where the needle problem appears.

## The needle problem

When we trained the foresight filter, we scored 300 random articles from our news corpus. The distribution:

| Score range | % of articles |
|-------------|---------------|
| 0-2 (noise) | 90% |
| 2-5 (some foresight) | 9% |
| 5+ (genuine foresight) | 1% |

Ninety percent of articles pile up at the noise floor. The 2-5 range — where the model needs to learn the *gradient* from "a bit of foresight" to "strong foresight" — is almost empty. And the high-scoring articles that define what foresight looks like? Three articles out of 300.

This is not a labeling error. This is the negativity bias, measured. News selects for immediacy: this week's crisis, this quarter's earnings, this election's polls. Genuine foresight — decisions made for generations ahead — is not what newsrooms cover. It happens in governance documents, policy journals, institutional reforms. It's real, but it's rare in the daily news cycle.

A student model trained on this distribution learns exactly one thing: predict low scores for everything. That minimizes average error when 90% of your training data is noise. The resulting model has a technically acceptable loss but is useless — it can't distinguish a New Zealand wellbeing budget reform from a celebrity interview.

This same pattern appeared across our filters, and the correlation is clear:

| Filter | Concept rarity | Training MAE |
|--------|---------------|-------------|
| Investment-risk | Common (risk is everywhere) | 0.47 |
| Belonging | Common (community themes) | 0.49 |
| Nature recovery | Rare | 0.54 |
| Sustainability tech | Medium | 0.72 |
| Cultural discovery | Medium-low | 0.74 |
| Foresight | Very rare | 0.94 (before fix) |
| Thriving | Rare | 0.94 (unsolved) |

The rarer the concept, the worse the bimodal distribution, the harder the student model's job. This is a general property of semantic filtering for constructive concepts, not a bug in any particular filter.

## Two-stage screening: solving the needle problem

The solution separates two questions that the oracle was trying to answer simultaneously:

**Stage 1: "Is this article relevant?"** — handled by an embedding screener before oracle scoring.

We write 10-15 synthetic article summaries representing canonical examples of the concept (for foresight: New Zealand's wellbeing budget, Costa Rica's 30-year reforestation, Wales's Future Generations Commissioner). A small embedding model (e5-small, 33M parameters) computes cosine similarity between these seeds and every article in the corpus. The top candidates — articles that *look like* foresight — get sent to the oracle.

**Stage 2: "How much foresight does it contain?"** — handled by the oracle, scoring only relevant articles.

With pre-screened articles, the oracle can focus on gradients instead of binary classification. Content-type caps are softened (4.0-5.0 instead of 2.0-3.0) so that false positives from the screener land in the useful mid-range instead of being hard-capped at the noise floor.

The result:

| Score range | Before screening | After screening |
|-------------|-----------------|-----------------|
| 0-2 (noise) | 90% | 23% |
| 2-5 (some foresight) | 9% | 55% |
| 5+ (genuine foresight) | 1% | 20% |

The dead zone disappeared. The student model now has examples across the full score range. Foresight's MAE dropped from 0.94 to 0.75 — from unusable to on par with our mid-tier production filters.

The cost of this pre-screening step? Embedding 178,000 articles takes 15 minutes on a laptop CPU. The seeds take an hour to write. Compared to the oracle scoring cost (€4 for 3,500 articles), the screening is essentially free.

This pattern generalizes. Nature recovery used it first. Foresight proved it works for an even rarer concept. Thriving — currently paused at MAE 0.94 — is the next candidate.

## Distillation as energy investment

The standard framing of knowledge distillation is cost reduction: replace an expensive API with a cheap local model. That's true but insufficient.

The deeper framing is energy. An oracle scoring run is a one-time energy investment. It runs cloud GPUs for a few hours, scores a few thousand articles, and produces training data. After that, the student model runs on CPU — 20ms per article, no data center, no network round-trip. At 2,000 articles per day across 7 filters, the daily energy cost of inference is negligible compared to a single oracle run.

This is cathedral thinking applied to ML infrastructure: invest upfront to build something that runs efficiently for a long time.

The numbers for foresight: €4 in oracle scoring produced a model that will score millions of articles over its lifetime at essentially zero marginal energy cost. Even accounting for the GPU training time (~30 minutes on an RTX 4080), the energy payback period is measured in days, not months.

## What we don't solve

Honesty requires listing what's hard and what's still broken.

**Dimension correlation.** Our foresight filter's Time Horizon and Institutional Durability dimensions correlate at r=0.857. The oracle conflates them despite explicit instructions not to. The student model inherits this confusion. Cross-dimension exclusion notes in the prompt help but don't fully solve it. For now, we accept correlated dimensions as an imperfect approximation of concepts that are genuinely related.

**The fuzzy middle.** Systems Awareness (MAE 0.86) and Course Correction (MAE 0.79) are our worst-performing dimensions. They require subtle judgment — distinguishing "token caveat" from "genuine nuance" — that a 1-billion-parameter model struggles with. More training data helps (doubling from 1,374 to 2,761 examples improved every dimension) but there may be a floor below which small models simply can't go.

**Calibration on small datasets.** Isotonic regression calibration improved our validation MAE by 7.5% but was neutral on the held-out test set. With only 346 validation examples, the calibration overfits the validation distribution. This will improve as production data accumulates, but for now, our calibrated test MAE of 0.75 is the honest number.

**The 95% we throw away.** Pre-screening selects 2-3% of the corpus for oracle scoring. That means 97% of articles are never evaluated by the oracle. If a foresighted decision is described in unusual language that doesn't resemble our seed articles, the embedding screener won't find it. We accept this false-negative rate as the price of tractability.

## The pattern

The negativity bias in news isn't just an editorial problem. It's an engineering problem. It creates data distributions that defeat standard ML pipelines. When you filter for what's working in a world that selects for what's broken, you will hit the needle-in-haystack problem.

The solution is not a single technique but a pipeline:

1. **Dimensional scoring** — decompose judgment into measurable sub-factors
2. **Embedding pre-screening** — find the needles before you score them
3. **Soft scope gating** — let the oracle grade on a gradient, not a binary
4. **Knowledge distillation** — invest energy once, infer sustainably forever

Each step addresses a specific failure mode. Skip the dimensional scoring and you're back to sentiment analysis. Skip the pre-screening and your training data is 90% noise. Skip the soft gating and you create dead zones in your score distribution. Skip the distillation and you're paying cloud API costs on every article forever.

We didn't design this pipeline in advance. We discovered it by failing — thriving v1's bimodal distribution, foresight's first 300-article calibration batch, the seeds that got contaminated by corpus composition. Each failure taught us one piece.

The pipeline is now documented and reusable. The next filter we build will start from this template. If you're building semantic filters for rare concepts — in news or anywhere else — the pattern applies. The negativity bias is not unique to news. Any domain where the target signal is rare and the noise is systematically produced will have this property.

The needles are there. You just need a better way to find them.

---

*This article describes work on the [LLM Distillery](https://github.com/ducroq/llm-distillery) project, which powers the constructive news filters at [ovr.news](https://ovr.news).*
