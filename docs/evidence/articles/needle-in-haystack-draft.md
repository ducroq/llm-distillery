# The Needle-in-Haystack Problem: Why Filtering for Constructive News Breaks Standard ML

_Draft for llm-distillery#30_

---

## The promise and the trap

"Just filter the news for what's actually working." It sounds like a weekend project. Grab an LLM, score some articles, train a classifier. Ship it.

We tried. It took us a year, seven filters, and a few humbling failures before we understood why this is fundamentally harder than it looks. Not because the technology is immature, but because the problem has a structural property that standard approaches don't handle well: **the thing you're looking for is rare, and the reason it's rare is the same bias you're trying to correct.**

ovr.news filters global news through five constructive lenses — from community bonds to ecosystem recovery to rediscovered historical knowledge. Each lens counters one or more cognitive biases that shape what the daily news cycle covers. Together they surface evidence that the world is more functional than the news makes it appear.

The engineering challenge: constructive news is a needle in a haystack, and the haystack is _designed_ to hide the needle.

## Why keywords and sentiment fail

The first instinct is keyword matching. Surely "community" finds belonging, "long-term" finds foresight, "recovery" finds nature recovery?

It doesn't. What we're looking for isn't a topic — it's a _judgment_.

Our belonging filter scores articles on six dimensions (defined in [llm-distillery/filters/belonging](https://github.com/ducroq/llm-distillery)): intergenerational bonds, community fabric, reciprocal care, rootedness, purpose beyond self, and slow presence. A LinkedIn article about "building community at work" contains all the right keywords. It scores 1.3 out of 10. A story about a 94-year-old making pasta with her granddaughter using her mother's recipe scores 8.5. No keyword distinguishes them. The difference is _what kind of community_ — commodified versus organic, optimized versus lived.

Sentiment analysis fails in the opposite direction. Constructive news isn't positive news. A country admitting its drug war failed and shifting to treatment — that reads as negative. Our foresight filter scores it 6.1 because the _decision-making process_ shows evidence-based course correction, systems awareness, and institutional durability. Meanwhile, a cheerful wellness listicle about living longer scores 1.3 on belonging because it commodifies community as a longevity hack.

The judgment we need — "does this article demonstrate genuine foresight / belonging / recovery?" — requires understanding intent, process quality, and evidence, not just topic or tone.

## Dimensional scoring: breaking judgment into measurable sub-factors

Our approach: decompose each lens into 6 weighted dimensions that can be scored independently on a 0-10 scale.

For foresight (counters short-termism):

| Dimension                    | Weight | What it measures                                      |
| ---------------------------- | ------ | ----------------------------------------------------- |
| Time Horizon                 | 25%    | How far ahead does the decision look?                 |
| Systems Awareness            | 20%    | Are trade-offs and second-order effects acknowledged? |
| Course Correction            | 20%    | Is there willingness to admit error and change?       |
| Intergenerational Investment | 15%    | Are future generations explicitly considered?         |
| Institutional Durability     | 10%    | Will the decision survive a change of leadership?     |
| Evidence Foundation          | 10%    | Is the decision grounded in evidence?                 |

A large language model (the oracle) scores articles on these dimensions. It's good at this. But it's a cloud API call for every article, every lens, every run. What if we could capture what it knows and run it locally?

## Knowledge distillation: invest energy once, infer forever

We use the oracle to score thousands of articles, then train a small language model (SLM) to replicate those judgments. This is loosely called knowledge distillation — the large model teaches, the small model learns. Training takes about 30 minutes on a consumer GPU. After that, the SLM runs locally — no cloud, no external dependency, no data leaving the machine.

But this only works if the student has good training data to learn from. And that's where the needle problem appears.

## The needle problem

When we trained the foresight filter, we scored 300 random articles from our news corpus. The distribution:

| Score range             | % of articles |
| ----------------------- | ------------- |
| 0–2 (outside this lens) | 90%           |
| 2–5 (some foresight)    | 9%            |
| 5+ (genuine foresight)  | 1%            |

A low score does not mean bad journalism. It means the article covers territory outside what this particular lens looks for. Most of the 90% is competent, well-reported work on topics that simply aren't about long-term institutional decision-making. But from a training-data perspective, ninety percent of articles cluster at the bottom of the scale. The 2-5 range — where the model needs to learn the _gradient_ from "a bit of foresight" to "strong foresight" — is almost empty. And the high-scoring articles that define what foresight looks like? Three articles out of 300.

This is not a labeling error. It reflects what the daily news cycle covers. News selects for immediacy: this week's crisis, this quarter's earnings, this election's polls. Our news corpus amplifies this: it is composed mostly of general news outlets, not policy journals or governance publications. Genuine foresight — decisions made for generations ahead — is not what newsrooms cover. It happens in governance documents, policy journals, institutional reforms. It's real, but it's rare in the daily news cycle.

A small model trained on this distribution learns exactly one thing: predict low scores for everything. That minimizes average error when 90% of your training data is low-relevance articles. The resulting model has a technically acceptable loss but is useless — it can't distinguish a New Zealand wellbeing budget reform from a celebrity interview.

This same pattern appeared across all our filters: the rarer the concept, the worse the bimodal distribution, the harder the SLM's job. Belonging (community themes are common in news) trains easily. Foresight (genuine long-term thinking is rare) was nearly impossible — until we found a fix.

## Two-stage screening: solving the needle problem

The solution separates two questions that the oracle was trying to answer simultaneously:

**Stage 1: "Is this article relevant?"** — handled by an embedding screener before oracle scoring.

We write 10-15 descriptions of what the concept looks like in practice (for foresight: New Zealand's wellbeing budget, Costa Rica's 30-year reforestation, Wales's Future Generations Commissioner). A small, fast model finds articles that resemble these examples. The top candidates get sent to the oracle.

**Stage 2: "How much foresight does it contain?"** — handled by the oracle, now scoring only relevant articles instead of everything.

The result:

| Score range            | Before screening | After screening |
| ---------------------- | ---------------- | --------------- |
| 0–2 (outside lens)     | 90%              | 23%             |
| 2–5 (some foresight)   | 9%               | 55%             |
| 5+ (genuine foresight) | 1%               | 20%             |

The dead zone disappeared. The SLM now has examples across the full score range, and foresight went from unusable to a working production filter. The pre-screening step itself is lightweight — it runs in minutes on a laptop.

## Distillation as energy investment

The deeper framing of distillation is energy. An oracle scoring run is a one-time energy investment. It calls a cloud API for a few hours, scores a few thousand articles, and produces training data. After that, the SLM runs on a local GPU — no data center, no network round-trip. The energy investment happens once. After that, the system is self-sufficient.

## What we don't solve yet

- **Dimensions bleed into each other.** Some concepts we try to score separately turn out to be genuinely related. The models conflate them, and we haven't fully solved that.
- **Subtle judgment is hard for small models.** Distinguishing "token caveat" from "genuine nuance" requires a kind of reading comprehension that improves with more training data, but may have a floor.
- **Calibration misleads on small datasets.** We can tune accuracy on a validation set, but with only a few hundred examples, the tuning overfits. The number looks better without actually being better. More production data will fix this, but for now we report the uncalibrated number as the honest one.
- **We miss things.** Pre-screening finds articles that resemble our seed examples. If a foresighted decision is described in unusual language, the screener won't find it. The seeds themselves encode our editorial judgment — different seeds would find different needles.

If you're a publisher and believe your work fits one of our lenses, [get in touch](/contact).

## The pattern

The scarcity of constructive news in standard news corpora isn't just an editorial observation. It's an engineering problem. It creates data distributions that standard ML pipelines don't handle well. When you filter for what's working in a world that selects for what's broken, you will hit the needle-in-haystack problem.

The solution is not a single technique but a pipeline:

1. **Dimensional scoring** — decompose judgment into measurable sub-factors
2. **Embedding pre-screening** — find the needles before you score them
3. **Soft scope gating** — let the oracle grade on a gradient, not a binary
4. **Knowledge distillation** — invest energy once, infer sustainably forever

We didn't design this pipeline in advance. We discovered it by failing. Each failure taught us one piece. The pattern applies beyond news — any domain where the target signal is rare and the noise is systematically produced will have this property.

The needles are there. You just need a better way to find them.

---

_This article describes work on the [LLM Distillery](https://github.com/ducroq/llm-distillery) project, which powers the constructive news filters at [ovr.news](https://ovr.news). Every article on ovr.news links to its original source and shows its score. The filter dimension definitions are in the [llm-distillery repository](https://github.com/ducroq/llm-distillery)._
