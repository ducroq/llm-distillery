"""Score cultural_discovery v5 calibration sample with an Ollama-hosted oracle on gpu-server.

Adds Ollama batch oracles to the multi-oracle calibration mix. Uses the same prompt
template + content compression + sanitization as Gemini batch_scorer.py and
validate_deepseek_oracle.py (byte-for-byte parity at the prompt level).

Usage:
    PYTHONPATH=. python scripts/score_ollama_oracle.py --model qwen3:14b
    PYTHONPATH=. python scripts/score_ollama_oracle.py --model phi4:14b

Reads the canonical 522-article ID list from
`datasets/scored/cd_v5_522_for_softpenalty_rescore.jsonl` (frozen per ADR-020-draft
operational safety: same article set across all oracles).

Output: `datasets/scored/cd_v5_ollama_{model_slug}/results.jsonl`
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ground_truth.text_cleaning import (
    clean_article as clean_article_comprehensive,
    sanitize_text_comprehensive,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V5_PROMPT_PATH = PROJECT_ROOT / "filters" / "cultural_discovery" / "v5" / "prompt-compressed.md"
CANONICAL_INPUT = PROJECT_ROOT / "datasets" / "scored" / "cd_v5_522_for_softpenalty_rescore.jsonl"
OLLAMA_HOST = "http://gpu-server:11434"
PROMPT_PLACEHOLDER = "[Paste the summary of the article here]"
DIMENSIONS = [
    "discovery_novelty",
    "heritage_significance",
    "cross_cultural_connection",
    "human_resonance",
    "evidence_quality",
]


def smart_compress(content: str, max_words: int = 800) -> str:
    """Mirrors GenericBatchScorer._smart_compress_content exactly."""
    words = content.split()
    if len(words) <= max_words:
        return content
    start_words = int(max_words * 0.7)
    end_words = int(max_words * 0.3)
    beginning = " ".join(words[:start_words])
    ending = " ".join(words[-end_words:])
    return f"{beginning}\n\n[...content compressed...]\n\n{ending}"


def build_prompt(prompt_template: str, article: dict) -> str:
    article = clean_article_comprehensive(article)
    content = article.get("content", "")
    compressed = smart_compress(content, max_words=800)
    title = sanitize_text_comprehensive(article.get("title", "N/A"))
    source = sanitize_text_comprehensive(article.get("source", "N/A"))
    published = sanitize_text_comprehensive(article.get("published_date", "N/A"))
    text = sanitize_text_comprehensive(compressed)
    article_summary = (
        f"Title: {title}\nSource: {source}\nPublished: {published}\n\n{text}"
    )
    return prompt_template.replace(PROMPT_PLACEHOLDER, article_summary)


def load_canonical_articles() -> list[dict]:
    """Load the frozen 522 articles. This is the single source of truth across all oracle runs."""
    articles = []
    with open(CANONICAL_INPUT, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            articles.append(json.loads(line))
    return articles


def call_ollama(model: str, prompt: str, max_retries: int = 2):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.3,
            "num_predict": 4096,
            "num_ctx": 16384,  # enough for 8.4K prompt + 1.2K article + headroom
        },
    }
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json=body,
                timeout=600,  # 27B CPU-offload would need much longer; 14B should be <60s
            )
            if resp.status_code == 200:
                return resp.json()
            last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
            time.sleep(2 ** attempt)
        except requests.exceptions.RequestException as e:
            last_err = str(e)
            time.sleep(2 ** attempt)
    return {"error": f"Max retries exceeded: {last_err}"}


def extract_dim_score(value):
    """Normalize dim value to flat float regardless of input shape.

    Handles:
      - {"score": 7.0, "evidence": "..."}  (Gemini/v5-prompt nested)
      - 7.0                                 (flat)
      - "7.0"                               (string, sometimes from Ollama)
      - None                                (missing)
    """
    if value is None:
        return None
    if isinstance(value, dict):
        s = value.get("score")
        if s is None:
            return None
        try:
            return float(s)
        except (TypeError, ValueError):
            return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def parse_response(resp: dict):
    if "error" in resp:
        return {"error": resp["error"]}
    try:
        text = resp["message"]["content"]
        parsed = json.loads(text)
        out = {dim: extract_dim_score(parsed.get(dim)) for dim in DIMENSIONS}
        if any(v is None for v in out.values()):
            missing = [d for d, v in out.items() if v is None]
            return {
                "error": f"Missing/invalid dims: {missing}",
                "raw_text": text[:500],
                "parsed_keys": list(parsed.keys()),
            }
        out["content_type"] = parsed.get("content_type", "unknown")
        out["_eval_count"] = resp.get("eval_count", 0)
        out["_prompt_eval_count"] = resp.get("prompt_eval_count", 0)
        out["_total_duration_s"] = resp.get("total_duration", 0) / 1e9
        return out
    except (json.JSONDecodeError, KeyError) as e:
        raw = ""
        try:
            raw = resp.get("message", {}).get("content", "")[:500]
        except Exception:
            pass
        return {"error": f"Parse failed: {e}", "raw_text": raw}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Ollama model tag (e.g., qwen3:14b)")
    parser.add_argument("--concurrency", type=int, default=2,
                        help="Concurrent requests (default 2; Ollama serves sequentially per model)")
    parser.add_argument("--output-dir", default=None,
                        help="Default: datasets/scored/cd_v5_ollama_{slug}/")
    args = parser.parse_args()

    model_slug = args.model.replace(":", "_").replace("/", "_")
    output_dir = Path(args.output_dir) if args.output_dir else (PROJECT_ROOT / f"datasets/scored/cd_v5_ollama_{model_slug}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.jsonl"

    prompt_template = V5_PROMPT_PATH.read_text(encoding="utf-8")
    articles = load_canonical_articles()
    print(f"Loaded {len(articles)} canonical articles from {CANONICAL_INPUT.name}")
    print(f"Output: {output_path}")
    print(f"Model: {args.model}")
    print(f"Concurrency: {args.concurrency}")
    print()

    # Verify model exists on remote
    try:
        tags = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10).json()
        if not any(m["name"] == args.model for m in tags.get("models", [])):
            print(f"ERROR: Model {args.model} not found on {OLLAMA_HOST}")
            print(f"Available: {[m['name'] for m in tags.get('models', [])]}")
            sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Cannot reach Ollama at {OLLAMA_HOST}: {e}")
        sys.exit(1)

    def _process(article):
        prompt = build_prompt(prompt_template, article)
        resp = call_ollama(args.model, prompt)
        parsed = parse_response(resp)
        return article, parsed

    successes = 0
    errors = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_duration_s = 0

    start = time.time()
    with open(output_path, "w", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {executor.submit(_process, art): art for art in articles}
            completed = 0
            for future in as_completed(futures):
                completed += 1
                article = futures[future]
                article_id = article.get("id", "unknown")[:60]
                try:
                    _, parsed = future.result()
                except Exception as e:
                    parsed = {"error": f"Exception: {e}"}

                if "error" in parsed:
                    errors += 1
                    print(f"  [{completed:3d}/{len(articles)}] {article_id:60s} ERROR: {parsed['error'][:80]}")
                    record = {
                        "id": article["id"],
                        "title": article.get("title", "")[:120],
                        "model": args.model,
                        "error": parsed["error"],
                        "raw_text": parsed.get("raw_text", "")[:300],
                    }
                else:
                    successes += 1
                    total_input_tokens += parsed.get("_prompt_eval_count", 0)
                    total_output_tokens += parsed.get("_eval_count", 0)
                    total_duration_s += parsed.get("_total_duration_s", 0)
                    record = {
                        "id": article["id"],
                        "title": article.get("title", "")[:120],
                        "model": args.model,
                        "content_type": parsed["content_type"],
                        "dims": {d: parsed[d] for d in DIMENSIONS},
                        "_prompt_eval_count": parsed["_prompt_eval_count"],
                        "_eval_count": parsed["_eval_count"],
                        "_total_duration_s": round(parsed["_total_duration_s"], 2),
                    }
                    if completed % 25 == 0 or completed == len(articles):
                        print(f"  [{completed:3d}/{len(articles)}] {article_id:60s} OK")

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

    wall = time.time() - start
    print()
    print(f"=" * 72)
    print(f"RESULTS for {args.model}")
    print(f"=" * 72)
    print(f"Successful: {successes}/{len(articles)}  Errors: {errors}")
    print(f"Wall clock: {wall/60:.1f} min  Sum inference: {total_duration_s/60:.1f} min")
    print(f"Tokens: input {total_input_tokens:,}  output {total_output_tokens:,}")
    if successes:
        print(f"Avg input tokens/article: {total_input_tokens/successes:.0f}")
        print(f"Avg duration/article: {total_duration_s/successes:.1f}s")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
